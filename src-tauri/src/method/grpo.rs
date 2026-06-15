use crate::error::Result;
use crate::llamafactory;
use crate::runs::{LoraConfig, Run};

use super::common::sh_quote;
use super::{CommandKind, LlamaFactoryYamlOptions, MethodOptions};

pub const KEY: &str = "grpo";

pub fn yaml() -> LlamaFactoryYamlOptions {
    LlamaFactoryYamlOptions::lora_like()
}

pub fn options() -> MethodOptions {
    MethodOptions {
        command_kind: CommandKind::Grpo,
        yaml: yaml(),
        ..MethodOptions::lora_like(KEY)
    }
}
/// The reward functions in those notebooks are task-specific (Sudoku, 2048).
/// For a general-purpose GRPO run on the user's dataset we use a length+stop-
/// token reward as a sensible baseline. Users who need bespoke rewards should
/// switch to the `custom` method and paste their own reward script.
pub fn build_train_cmd(run: &Run, lora: &LoraConfig, hf_export: &str) -> Result<String> {
    let base_model = llamafactory::resolve_trainable_repo(&run.student_model);
    let lower = base_model.to_lowercase();
    // gpt-oss requires BF16 load (no 4-bit), per the notebook.
    let load_in_4bit = !(lower.contains("gpt-oss") || lower.contains("gpt_oss"));
    let train_yaml = format!("{}/train.yaml", run.remote_dir);
    let data_dir = format!("{}/data", run.remote_dir);
    let output_dir = format!("{}/lora", run.remote_dir);
    let script_path = format!("{}/grpo_train.py", run.remote_dir);

    // The Python script is written to disk via a heredoc so we don't have to
    // worry about shell-escaping every quote. Variables that need to be
    // interpolated from Rust use {placeholders} BEFORE we feed the result
    // into the heredoc, and we use a literal-marker heredoc tag ('PYEOF') so
    // bash itself doesn't expand anything.
    let py = format!(
        r#"import os, json, glob
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
import torch
from trl import GRPOConfig, GRPOTrainer

BASE_MODEL = "{base_model}"
DATA_DIR = "{data_dir}"
OUTPUT_DIR = "{output_dir}"
MAX_SEQ = {cutoff_len}
LORA_R = {lora_r}
LORA_ALPHA = {lora_alpha}
LR = {learning_rate}
EPOCHS = {epochs}
PER_DEVICE_BS = {batch_size}
GRAD_ACCUM = {gradient_accumulation}
SAVE_STEPS = {save_steps}
LOAD_IN_4BIT = {load_in_4bit}

print(f"[grpo] loading base model: {{BASE_MODEL}} (4bit={{LOAD_IN_4BIT}})", flush=True)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ,
    load_in_4bit=LOAD_IN_4BIT,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_ALPHA,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Build a prompt-only dataset from the run's local JSONL training file. Each
# row must have a "prompt" field shaped like trl expects: a list of chat msgs.
candidates = sorted(glob.glob(os.path.join(DATA_DIR, "*.jsonl")))
if not candidates:
    raise SystemExit(f"[grpo] no .jsonl files found in {{DATA_DIR}}")
rows = []
for path in candidates:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Accept either a bare "prompt" string or LLaMA-Factory's
            # {{"instruction": ..., "input": ..., "output": ...}} schema.
            prompt = obj.get("prompt")
            if not prompt and "instruction" in obj:
                instr = obj.get("instruction", "")
                inp = obj.get("input", "")
                prompt = f"{{instr}}\n\n{{inp}}" if inp else instr
            if not prompt:
                continue
            rows.append({{"prompt": [{{"role": "user", "content": str(prompt)}}]}})
if not rows:
    raise SystemExit("[grpo] no usable rows extracted from dataset")
print(f"[grpo] loaded {{len(rows)}} prompts", flush=True)
dataset = Dataset.from_list(rows)

# Baseline length+EOS reward. Users who want task-specific rewards should
# switch to the `custom` method and paste the notebook's reward functions.
EOS_TOKENS = {{tokenizer.eos_token}} if getattr(tokenizer, "eos_token", None) else set()
def length_reward(completions, **kwargs):
    scores = []
    for c in completions:
        text = c[0]["content"] if isinstance(c, list) else str(c)
        n_chars = len(text.strip())
        if n_chars == 0:
            scores.append(-1.0)
        elif n_chars < 20:
            scores.append(-0.5)
        elif any(text.rstrip().endswith(t) for t in EOS_TOKENS):
            scores.append(1.0)
        else:
            scores.append(0.5)
    return scores

args = GRPOConfig(
    temperature=1.0,
    learning_rate=LR,
    weight_decay=0.001,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    optim="adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=PER_DEVICE_BS,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    max_grad_norm=0.3,
    output_dir=OUTPUT_DIR,
    save_steps=SAVE_STEPS,
    report_to="none",
)
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[length_reward],
    args=args,
    train_dataset=dataset,
)
trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("[grpo] training complete; LoRA saved to", OUTPUT_DIR, flush=True)
"#,
        base_model = base_model,
        data_dir = data_dir,
        output_dir = output_dir,
        cutoff_len = lora.cutoff_len,
        lora_r = lora.r,
        lora_alpha = lora.alpha,
        learning_rate = lora.learning_rate,
        epochs = lora.epochs,
        batch_size = lora.batch_size,
        gradient_accumulation = lora.gradient_accumulation,
        save_steps = lora.save_steps,
        load_in_4bit = if load_in_4bit { "True" } else { "False" },
    );
    let py = py.replace("\r", "");

    // Write the Python script via a literal-marker heredoc so bash does no
    // expansion (single-quoted 'PYEOF'). The shell-side script then runs it.
    let heredoc = format!(
        "cat > {script} <<'PYEOF'\n{py}\nPYEOF\n",
        script = sh_quote(&script_path),
        py = py
    );

    // GRPO doesn't read train.yaml — the surrounding pipeline still writes it
    // for resume/inspect purposes, but the trainer ignores it.
    let _ = train_yaml;

    Ok(format!(
        "set -o pipefail; \
         {hf_export} cd {dir} && \
         export PYTHONUNBUFFERED=1 && \
         mkdir -p {output_dir} && \
         : > {dir}/log.txt && : > {dir}/errorlog.txt && : > {dir}/train.log && \
         {heredoc}\
         {{ echo {start_msg} && python3 {script} ; }} \
           > >(tee -a {dir}/log.txt {dir}/train.log) \
           2> >(tee -a {dir}/errorlog.txt {dir}/train.log >&2)",
        hf_export = hf_export,
        dir = sh_quote(&run.remote_dir),
        output_dir = sh_quote(&output_dir),
        heredoc = heredoc,
        start_msg = sh_quote("[grpo] starting unsloth GRPO trainer"),
        script = sh_quote(&script_path),
    ))
}
