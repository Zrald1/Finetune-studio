use crate::error::Result;
use crate::llamafactory;
use crate::runs::{LoraConfig, Run};

use super::common::sh_quote;
use super::{CommandKind, LlamaFactoryYamlOptions, MethodOptions};

pub const KEY: &str = "zrald";

pub fn yaml() -> LlamaFactoryYamlOptions {
    LlamaFactoryYamlOptions::lora_like()
}

pub fn options() -> MethodOptions {
    MethodOptions {
        command_kind: CommandKind::Zrald,
        yaml: yaml(),
        ..MethodOptions::lora_like(KEY)
    }
}
pub fn build_train_cmd(run: &Run, lora: &LoraConfig, hf_export: &str) -> Result<String> {
    let global_prompt_template = run.prompt_template.clone().unwrap_or_default();
    let mut topic_prompts_map = std::collections::HashMap::new();
    for t in &run.topics {
        if let Some(ref p) = t.prompt_template {
            topic_prompts_map.insert(t.topic.clone(), p.clone());
        }
    }
    let topic_prompts_json =
        serde_json::to_string(&topic_prompts_map).unwrap_or_else(|_| "{}".to_string());

    let base_model = llamafactory::resolve_trainable_repo(&run.student_model);
    let lower = base_model.to_lowercase();
    let load_in_4bit = !(lower.contains("gpt-oss") || lower.contains("gpt_oss"));
    let data_dir = format!("{}/data", run.remote_dir);
    let output_dir = format!("{}/lora", run.remote_dir);
    let script_path = format!("{}/zrald_train.py", run.remote_dir);
    let reward_endpoint = if lora.zrald_reward_endpoint.trim().is_empty() {
        format!("http://127.0.0.1:{}", run.teacher_cfg.vllm_port)
    } else {
        lora.zrald_reward_endpoint.trim().to_string()
    };
    let reward_model = lora.zrald_reward_model.trim().to_string();
    let train_questions = lora.zrald_train_questions.max(1);
    let benchmark_questions = lora.zrald_benchmark_questions.min(train_questions).max(1);
    let num_generations = lora.zrald_num_generations.clamp(2, 8);
    let max_completion_tokens = lora
        .zrald_max_completion_tokens
        .clamp(64, lora.cutoff_len.max(64));
    let hf_dataset_repos = if run.hub_dataset.enabled {
        llamafactory::hub_dataset_repos(run)
    } else {
        Vec::new()
    };

    let mut py = r#"import hashlib
import inspect
import json
import os
import random
import re
import shutil
import statistics
import time
from pathlib import Path

import requests
import torch
from datasets import Dataset, load_dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

BASE_MODEL = __BASE_MODEL__
DATA_DIR = Path(__DATA_DIR__)
RUN_DIR = Path(__RUN_DIR__)
OUTPUT_DIR = Path(__OUTPUT_DIR__)
REWARD_ENDPOINT = __REWARD_ENDPOINT__.rstrip("/")
REWARD_MODEL = __REWARD_MODEL__
MAX_SEQ = __MAX_SEQ__
LORA_R = __LORA_R__
LORA_ALPHA = __LORA_ALPHA__
LR = __LR__
EPOCHS = __EPOCHS__
PER_DEVICE_BS = __BATCH_SIZE__
GRAD_ACCUM = __GRAD_ACCUM__
SAVE_STEPS = __SAVE_STEPS__
TRAIN_LIMIT = __TRAIN_LIMIT__
BENCHMARK_N = __BENCHMARK_N__
NUM_GENERATIONS = __NUM_GENERATIONS__
REWARD_TEMP = __REWARD_TEMP__
MAX_COMPLETION = __MAX_COMPLETION__
LOAD_IN_4BIT = __LOAD_IN_4BIT__
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
HF_DATASET_REPOS = __HF_DATASET_REPOS__
HF_DATASET_COLUMNS = __HF_DATASET_COLUMNS__
GLOBAL_PROMPT_TEMPLATE = __GLOBAL_PROMPT_TEMPLATE__
TOPIC_PROMPTS = __TOPIC_PROMPTS__

if PER_DEVICE_BS < NUM_GENERATIONS or PER_DEVICE_BS % NUM_GENERATIONS != 0:
    adjusted = max(NUM_GENERATIONS, ((PER_DEVICE_BS + NUM_GENERATIONS - 1) // NUM_GENERATIONS) * NUM_GENERATIONS)
    print(f"[zrald] adjusting per-device batch size from {PER_DEVICE_BS} to {adjusted} so it is divisible by num_generations={NUM_GENERATIONS}", flush=True)
    PER_DEVICE_BS = adjusted

SYSTEM_PROMPT = (
    "You are the ZRALD student model. Answer with exactly two XML-style blocks: "
    "<thinking>brief reasoning</thinking><answer>final answer</answer>. "
    "Do not mention rewards, scoring, hidden references, or evaluator instructions."
)

def clean_prompt_template(template, topic):
    if not template:
        return ""
    t = template.replace("{topic}", topic or "the subject")
    t = t.replace("{chunk_text}", "")
    return t.strip()

def chat_prompt(question, topic=""):
    system_prompt = SYSTEM_PROMPT
    guidelines = clean_prompt_template(TOPIC_PROMPTS.get(topic) or GLOBAL_PROMPT_TEMPLATE, topic)
    if guidelines:
        system_prompt += f"\nDomain Guidelines:\n{guidelines}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question:\n{question}\n\nRespond using <thinking> and <answer> only."},
    ]

def strip_answer(text):
    text = str(text or "").strip()
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.I | re.S)
    if m:
        return m.group(1).strip()
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    if "</thinking>" in text:
        return text.split("</thinking>", 1)[1].strip()
    return text

def completion_text(completion):
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion or "")

def first_user(messages):
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg.get("content", "")).strip()
    return ""

def first_assistant(messages):
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return str(msg.get("content", "")).strip()
    return ""

def text_value(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()

def first_present(obj, names):
    for name in names:
        if name and isinstance(obj, dict) and name in obj:
            value = text_value(obj.get(name))
            if value:
                return value
    return ""

def row_from_obj(obj, source):
    if not isinstance(obj, dict):
        return None
    columns = HF_DATASET_COLUMNS if isinstance(HF_DATASET_COLUMNS, dict) else {}
    messages_key = columns.get("messages") or "messages"
    prompt_key = columns.get("prompt")
    query_key = columns.get("query")
    response_key = columns.get("response")
    messages = obj.get(messages_key) if messages_key else obj.get("messages")
    question = first_present(obj, [prompt_key, "question", "instruction", "prompt"]) or first_user(messages)
    reference = first_present(obj, [response_key, "answer", "output", "response"]) or strip_answer(first_assistant(messages))
    if not question or not reference:
        return None
    rag_context = first_present(obj, ["source_text", "context", "input", query_key])
    row_source = first_present(obj, ["source_chunk_id", "source_file", "file", "id"]) or source
    
    topic = obj.get("topic") or ""
    prompt_template = TOPIC_PROMPTS.get(topic) or GLOBAL_PROMPT_TEMPLATE or ""
    rubric = (
        "Score from -1.0 to 1.0. 1.0 means the final answer is precise, complete, and supported. "
        "0.0 means partially useful but needs correction. -1.0 means incorrect, contradictory, hallucinated, or empty. "
        "Correctness and RAG faithfulness dominate style.\n"
    )
    if topic:
        rubric += f"The question focus topic is: '{topic}'.\n"
    guidelines = clean_prompt_template(prompt_template, topic)
    if guidelines:
        rubric += f"The question was generated under the following prompt template guidelines:\n{guidelines}\n"
        rubric += "Verify that the student's completion adheres to these guidelines and focus topic rules."

    return {
        "prompt": chat_prompt(question, topic),
        "question": question,
        "reference_answer": reference,
        "rag_context": rag_context,
        "rubric": rubric,
        "source": row_source,
    }

def load_local_pool():
    rows = []
    candidates = [DATA_DIR / "qa_dataset.jsonl", RUN_DIR / "qa_dataset.jsonl", DATA_DIR / "train.jsonl", DATA_DIR / "val.jsonl"]
    for path in candidates:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                row = row_from_obj(obj, path.name)
                if row:
                    rows.append(row)
        if rows:
            break
    return rows

def load_dataset_with_auth(repo):
    kwargs = {}
    if HF_TOKEN:
        kwargs["token"] = HF_TOKEN
    try:
        return load_dataset(repo, **kwargs)
    except TypeError:
        if HF_TOKEN:
            kwargs.pop("token", None)
            kwargs["use_auth_token"] = HF_TOKEN
            return load_dataset(repo, **kwargs)
        raise

def load_hf_pool():
    rows = []
    for repo in HF_DATASET_REPOS or []:
        repo = str(repo or "").strip()
        if not repo:
            continue
        print(f"[zrald] loading HF dataset {repo}", flush=True)
        dataset = load_dataset_with_auth(repo)
        if hasattr(dataset, "keys"):
            names = list(dataset.keys())
            ordered = [name for name in ["train", "validation", "val", "test"] if name in dataset]
            ordered.extend([name for name in names if name not in ordered])
            splits = [(name, dataset[name]) for name in ordered]
        else:
            splits = [("train", dataset)]
        for split_name, split in splits:
            for obj in split:
                row = row_from_obj(obj, f"{repo}:{split_name}")
                if row:
                    rows.append(row)
    return rows

def load_pool():
    rows = load_local_pool()
    if not rows and HF_DATASET_REPOS:
        rows = load_hf_pool()
    dedup = {}
    for row in rows:
        key = hashlib.sha256((row["question"] + "\n" + row["reference_answer"]).encode("utf-8")).hexdigest()
        dedup.setdefault(key, row)
    rows = list(dedup.values())
    random.Random(3407).shuffle(rows)
    if not rows:
        raise SystemExit("[zrald] no usable question rows found in local qa_dataset/train/val JSONL or selected HF datasets")
    return rows

def dump_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def reward_url():
    if REWARD_ENDPOINT.endswith("/v1"):
        return REWARD_ENDPOINT + "/chat/completions"
    return REWARD_ENDPOINT + "/v1/chat/completions"

def reward_models_url():
    if REWARD_ENDPOINT.endswith("/v1"):
        return REWARD_ENDPOINT + "/models"
    return REWARD_ENDPOINT + "/v1/models"

def detect_reward_model():
    configured = str(REWARD_MODEL or "").strip()
    if configured:
        return configured
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    try:
        res = requests.get(reward_models_url(), headers=headers, timeout=15)
        if res.status_code >= 400:
            raise RuntimeError(f"http {res.status_code}: {res.text[:300]}")
        payload = res.json()
        for item in payload.get("data", []):
            model_id = str(item.get("id") or "").strip()
            if model_id:
                print(f"[zrald] auto-detected reward teacher model: {model_id}", flush=True)
                return model_id
    except Exception as exc:
        raise SystemExit(f"[zrald] reward teacher model is blank and auto-detect failed at {reward_models_url()}: {exc}")
    raise SystemExit(f"[zrald] reward teacher model is blank and {reward_models_url()} returned no models")

REWARD_MODEL = detect_reward_model()
reward_cache = {}
reward_log = RUN_DIR / "zrald_rewards_train.jsonl"

def clamp_score(value):
    try:
        return max(-1.0, min(1.0, float(value)))
    except Exception:
        return -0.25

def parse_jsonish(text):
    text = str(text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}
    return {}

def heuristic_adjustment(completion):
    text = completion.strip()
    penalty = 0.0
    if not text:
        return -1.0
    if not re.search(r"<thinking>.*?</thinking>", text, flags=re.I | re.S):
        penalty -= 0.15
    if not re.search(r"<answer>.*?</answer>", text, flags=re.I | re.S):
        penalty -= 0.20
    if len(strip_answer(text)) < 8:
        penalty -= 0.25
    return penalty

def judge_score(question, reference_answer, rag_context, rubric, completion, phase="train"):
    key = hashlib.sha256(json.dumps([question, reference_answer, rag_context, completion], ensure_ascii=False).encode("utf-8")).hexdigest()
    if key in reward_cache:
        return reward_cache[key]
    prompt = {
        "question": question,
        "rag_context": rag_context,
        "reference_answer": reference_answer,
        "student_completion": completion,
        "rubric": rubric,
        "required_output": {"score": "number from -1.0 to 1.0", "verdict": "short label", "reason": "short private note"},
    }
    headers = {"Content-Type": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    body = {
        "model": REWARD_MODEL,
        "temperature": REWARD_TEMP,
        "max_tokens": 256,
        "messages": [
            {"role": "system", "content": "You are the ZRALD reward teacher. Return strict JSON only. Grade factual correctness against the reference and RAG context. Never reward unsupported claims."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
    }
    score = -0.25
    verdict = "judge_error"
    reason = ""
    for attempt in range(2):
        try:
            res = requests.post(reward_url(), headers=headers, json=body, timeout=120)
            if res.status_code >= 400:
                reason = f"http {res.status_code}: {res.text[:400]}"
                time.sleep(1.0)
                continue
            payload = res.json()
            content = payload["choices"][0]["message"]["content"]
            judged = parse_jsonish(content)
            score = clamp_score(judged.get("score", -0.25))
            verdict = str(judged.get("verdict", "scored"))
            reason = str(judged.get("reason", ""))[:500]
            break
        except Exception as exc:
            reason = repr(exc)
            time.sleep(1.0)
    score = clamp_score(score + heuristic_adjustment(completion))
    reward_cache[key] = score
    with reward_log.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "phase": phase,
            "question": question,
            "score": score,
            "verdict": verdict,
            "reason": reason,
            "completion": completion,
        }, ensure_ascii=False) + "\n")
    return score

def pick(values, idx, default=""):
    if isinstance(values, list) and values:
        return values[idx % len(values)]
    return default

def zrald_reward(completions, **kwargs):
    scores = []
    for i, completion in enumerate(completions):
        q = pick(kwargs.get("question"), i)
        ref = pick(kwargs.get("reference_answer"), i)
        rag = pick(kwargs.get("rag_context"), i)
        rubric = pick(kwargs.get("rubric"), i)
        scores.append(judge_score(q, ref, rag, rubric, completion_text(completion), "train"))
    return scores

def prompt_to_text(messages, tokenizer):
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n".join(f"{m.get('role', 'user').upper()}: {m.get('content', '')}" for m in messages) + "\nASSISTANT:"

def generate_one(model, tokenizer, row):
    model.eval()
    prompt = prompt_to_text(row["prompt"], tokenizer)
    encoded = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        out = model.generate(
            **encoded,
            max_new_tokens=MAX_COMPLETION,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][encoded["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def summarize(scores):
    if not scores:
        return {"count": 0, "mean": 0.0, "median": 0.0, "passRate": 0.0, "failRate": 0.0}
    return {
        "count": len(scores),
        "mean": statistics.fmean(scores),
        "median": statistics.median(scores),
        "passRate": sum(1 for s in scores if s >= 0.8) / len(scores),
        "failRate": sum(1 for s in scores if s < 0.0) / len(scores),
    }

def run_benchmark(model, tokenizer, rows, phase):
    path = RUN_DIR / f"zrald_benchmark_{phase}.jsonl"
    scores = []
    with path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows, 1):
            completion = generate_one(model, tokenizer, row)
            score = judge_score(row["question"], row["reference_answer"], row["rag_context"], row["rubric"], completion, f"benchmark_{phase}")
            scores.append(score)
            f.write(json.dumps({
                "idx": idx,
                "question": row["question"],
                "score": score,
                "completion": completion,
                "reference_answer": row["reference_answer"],
                "source": row.get("source", ""),
            }, ensure_ascii=False) + "\n")
            if idx % 10 == 0 or idx == len(rows):
                print(f"[zrald] benchmark {phase}: {idx}/{len(rows)} mean={statistics.fmean(scores):.3f}", flush=True)
    return scores

rows = load_pool()
train_rows = rows[:min(TRAIN_LIMIT, len(rows))]
benchmark_rows = rows[:min(BENCHMARK_N, len(rows))]
dump_jsonl(RUN_DIR / "zrald_train_prompts.jsonl", train_rows)
dump_jsonl(RUN_DIR / "zrald_benchmark_prompts.jsonl", benchmark_rows)
print(f"[zrald] question pool: train={len(train_rows)} benchmark={len(benchmark_rows)} reward_model={REWARD_MODEL} endpoint={REWARD_ENDPOINT}", flush=True)
print(f"[zrald] loading student: {BASE_MODEL} (4bit={LOAD_IN_4BIT})", flush=True)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ,
    load_in_4bit=LOAD_IN_4BIT,
)
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_ALPHA,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

before_scores = run_benchmark(model, tokenizer, benchmark_rows, "before")
model.train()
dataset = Dataset.from_list(train_rows)

grpo_kwargs = {
    "temperature": 0.9,
    "learning_rate": LR,
    "weight_decay": 0.001,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "linear",
    "optim": "adamw_torch",
    "logging_steps": 1,
    "per_device_train_batch_size": PER_DEVICE_BS,
    "gradient_accumulation_steps": GRAD_ACCUM,
    "num_train_epochs": EPOCHS,
    "max_grad_norm": 0.3,
    "output_dir": str(OUTPUT_DIR),
    "save_steps": SAVE_STEPS,
    "report_to": "none",
    "num_generations": NUM_GENERATIONS,
    "max_completion_length": MAX_COMPLETION,
    "max_prompt_length": max(128, MAX_SEQ - MAX_COMPLETION),
}
accepted_args = inspect.signature(GRPOConfig).parameters
args = GRPOConfig(**{k: v for k, v in grpo_kwargs.items() if k in accepted_args})

trainer_kwargs = {
    "model": model,
    "reward_funcs": [zrald_reward],
    "args": args,
    "train_dataset": dataset,
}
trainer_sig = inspect.signature(GRPOTrainer.__init__).parameters
if "processing_class" in trainer_sig:
    trainer_kwargs["processing_class"] = tokenizer
elif "tokenizer" in trainer_sig:
    trainer_kwargs["tokenizer"] = tokenizer

print(f"[zrald] starting GRPO: generations={NUM_GENERATIONS} train_prompts={len(train_rows)}", flush=True)
trainer = GRPOTrainer(**trainer_kwargs)
trainer.train()
model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))
print("[zrald] training complete; running after benchmark", flush=True)

after_scores = run_benchmark(model, tokenizer, benchmark_rows, "after")
report = {
    "method": "ZRALD",
    "meaning": "Zero-shot Retrieval-Augmented Learning with Dynamic rewards",
    "rewardModel": REWARD_MODEL,
    "rewardEndpoint": REWARD_ENDPOINT,
    "numGenerations": NUM_GENERATIONS,
    "trainQuestions": len(train_rows),
    "benchmarkQuestions": len(benchmark_rows),
    "before": summarize(before_scores),
    "after": summarize(after_scores),
}
report["deltaMean"] = report["after"]["mean"] - report["before"]["mean"]
paired = [a - b for a, b in zip(after_scores, before_scores)]
report["paired"] = {
    "meanDelta": statistics.fmean(paired) if paired else 0.0,
    "wins": sum(1 for d in paired if d > 0.05),
    "losses": sum(1 for d in paired if d < -0.05),
    "ties": sum(1 for d in paired if -0.05 <= d <= 0.05),
}
benchmark_summary = {
    "before": report["before"],
    "after": report["after"],
    "deltaMean": report["deltaMean"],
    "paired": report["paired"],
}
artifacts_dir = OUTPUT_DIR / "zrald_artifacts"
artifacts_dir.mkdir(parents=True, exist_ok=True)
copied_artifacts = []

def copy_artifact(src, name=None):
    src = Path(src)
    if not src.exists():
        return
    dest_name = name or src.name
    shutil.copy2(src, artifacts_dir / dest_name)
    copied_artifacts.append(dest_name)

for artifact_name in [
    "zrald_train_prompts.jsonl",
    "zrald_benchmark_prompts.jsonl",
    "zrald_rewards_train.jsonl",
    "zrald_benchmark_before.jsonl",
    "zrald_benchmark_after.jsonl",
]:
    copy_artifact(RUN_DIR / artifact_name)

for source_dataset in [DATA_DIR / "qa_dataset.jsonl", RUN_DIR / "qa_dataset.jsonl"]:
    if source_dataset.exists():
        copy_artifact(source_dataset, "qa_dataset.jsonl")
        break

copied_artifacts.extend(["zrald_report.json", "README.md", "manifest.json"])
report["benchmark"] = benchmark_summary
report["artifactDir"] = str(artifacts_dir)
report["artifacts"] = copied_artifacts
(RUN_DIR / "zrald_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
(artifacts_dir / "zrald_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
(artifacts_dir / "README.md").write_text(
    '# ZRALD artifacts\n\n'
    'This folder is saved with the trained adapter and copied into merged model uploads.\n\n'
    '- zrald_train_prompts.jsonl: training prompt/reference pool used for reward learning.\n'
    '- zrald_benchmark_prompts.jsonl: held-out benchmark prompt/reference pool.\n'
    '- zrald_rewards_train.jsonl: reward teacher scores and verdicts during GRPO training.\n'
    '- zrald_benchmark_before.jsonl: student benchmark before ZRALD training.\n'
    '- zrald_benchmark_after.jsonl: student benchmark after ZRALD training.\n'
    '- zrald_report.json: benchmark summary, reward source, and paired win/loss counts.\n\n'
    f'Benchmark mean: {benchmark_summary["before"]["mean"]:.4f} to {benchmark_summary["after"]["mean"]:.4f}; '
    f'delta {benchmark_summary["deltaMean"]:.4f}.\n',
    encoding="utf-8",
)
(artifacts_dir / "manifest.json").write_text(json.dumps({
    "method": "ZRALD",
    "baseModel": BASE_MODEL,
    "rewardModel": REWARD_MODEL,
    "rewardEndpoint": REWARD_ENDPOINT,
    "trainQuestions": len(train_rows),
    "benchmarkQuestions": len(benchmark_rows),
    "numGenerations": NUM_GENERATIONS,
    "benchmark": benchmark_summary,
    "files": copied_artifacts,
}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print("[zrald] report", json.dumps(report, ensure_ascii=False), flush=True)
print(f"[zrald] artifacts saved to {artifacts_dir}: {', '.join(copied_artifacts)}", flush=True)
print("[zrald] LoRA saved to", OUTPUT_DIR, flush=True)
"#.to_string();

    let replacements = [
        (
            "__BASE_MODEL__",
            serde_json::to_string(&base_model).unwrap_or_else(|_| "\"\"".to_string()),
        ),
        (
            "__DATA_DIR__",
            serde_json::to_string(&data_dir).unwrap_or_else(|_| "\"\"".to_string()),
        ),
        (
            "__RUN_DIR__",
            serde_json::to_string(&run.remote_dir).unwrap_or_else(|_| "\"\"".to_string()),
        ),
        (
            "__OUTPUT_DIR__",
            serde_json::to_string(&output_dir).unwrap_or_else(|_| "\"\"".to_string()),
        ),
        (
            "__REWARD_ENDPOINT__",
            serde_json::to_string(&reward_endpoint).unwrap_or_else(|_| "\"\"".to_string()),
        ),
        (
            "__REWARD_MODEL__",
            serde_json::to_string(&reward_model).unwrap_or_else(|_| "\"\"".to_string()),
        ),
        ("__MAX_SEQ__", lora.cutoff_len.to_string()),
        ("__LORA_R__", lora.r.to_string()),
        ("__LORA_ALPHA__", lora.alpha.to_string()),
        ("__LR__", lora.learning_rate.to_string()),
        ("__EPOCHS__", lora.epochs.to_string()),
        ("__BATCH_SIZE__", lora.batch_size.max(1).to_string()),
        (
            "__GRAD_ACCUM__",
            lora.gradient_accumulation.max(1).to_string(),
        ),
        ("__SAVE_STEPS__", lora.save_steps.max(1).to_string()),
        ("__TRAIN_LIMIT__", train_questions.to_string()),
        ("__BENCHMARK_N__", benchmark_questions.to_string()),
        ("__NUM_GENERATIONS__", num_generations.to_string()),
        ("__REWARD_TEMP__", lora.zrald_reward_temperature.to_string()),
        ("__MAX_COMPLETION__", max_completion_tokens.to_string()),
        (
            "__LOAD_IN_4BIT__",
            if load_in_4bit {
                "True".to_string()
            } else {
                "False".to_string()
            },
        ),
        (
            "__HF_DATASET_REPOS__",
            serde_json::to_string(&hf_dataset_repos).unwrap_or_else(|_| "[]".to_string()),
        ),
        (
            "__HF_DATASET_COLUMNS__",
            serde_json::to_string(&run.hub_dataset.dataset_columns)
                .unwrap_or_else(|_| "{}".to_string()),
        ),
        (
            "__GLOBAL_PROMPT_TEMPLATE__",
            serde_json::to_string(&global_prompt_template).unwrap_or_else(|_| "\"\"".to_string()),
        ),
        ("__TOPIC_PROMPTS__", topic_prompts_json),
    ];
    for (needle, value) in replacements {
        py = py.replace(needle, &value);
    }
    let py = py.replace("\r", "");

    let heredoc = format!(
        "cat > {script} <<'PYEOF'\n{py}\nPYEOF\n",
        script = sh_quote(&script_path),
        py = py
    );
    let bnb_amd_wheel = "https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_main/bitsandbytes-1.33.7.preview-py3-none-manylinux_2_24_x86_64.whl";
    // Like build_zrald_offline_train_cmd: training runs via `docker exec` in the
    // shared rocm-vllm container, so a `--force-reinstall torch` here would
    // overwrite the torch that the resident vLLM teacher's prebuilt C extensions
    // need (undefined symbol / triton constexpr_function crashes). Build the
    // unsloth+TRL stack in a fully isolated venv that owns its own ROCm torch.
    let venv_dir = format!("{}/.zrald_venv", run.remote_dir);
    let zrald_probe =
        "python3 -c 'import torch,sys; sys.exit(0 if getattr(torch.version,\"hip\",None) else 1)' \
             >/dev/null 2>&1 && \
         python3 -c 'import requests, datasets, unsloth; from trl import GRPOConfig, GRPOTrainer' >/dev/null 2>&1";
    let torch_hip_probe = "python3 -c 'import torch,sys; sys.exit(0 if getattr(torch.version,\"hip\",None) else 1)' >/dev/null 2>&1";
    let rocm_torch_install = "pip install --no-cache-dir \
             --index-url https://download.pytorch.org/whl/rocm7.0 \
             'torch>=2.4,<2.11.0' 'torchvision<0.26.0' 'torchaudio<2.11.0'";
    let torch_hip_guard = format!(
        " && ({probe} || {rocm_torch})",
        probe = torch_hip_probe,
        rocm_torch = rocm_torch_install,
    );

    // See build_zrald_offline_train_cmd: a heredoc cannot follow a `&& \`
    // Rust line-continuation, which collapses it onto one logical shell line.
    // Terminate the install prefix with a real newline (`&&\n`) and emit the
    // heredoc + final exec block as genuine multi-line shell. The student/GRPO
    // script runs with the venv python ($PYBIN); the resident teacher keeps the
    // container's system Python.
    let install_prefix = format!(
        "set -o pipefail; \
         {hf_export} cd {dir} && \
         export UNSLOTH_IS_ROCM=1 PYTORCH_ROCM_ARCH=${{PYTORCH_ROCM_ARCH:-gfx950}} \
                PYTHONUNBUFFERED=1 && \
         (test -d {venv}/bin || python3 -m venv {venv}) && \
         . {venv}/bin/activate && \
         ({probe} || \
             (python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
              ({torch_probe} || {rocm_torch}) && \
              pip install --no-cache-dir 'unsloth[amd]' 'unsloth_zoo' 'trl>=0.19.0' \
                 'datasets>=2.16.0' 'requests>=2.31.0' 'peft>=0.19,<0.20' \
                 'accelerate>=0.34.0' 'sentencepiece>=0.2.0' 'protobuf' 'hf_transfer' 'psutil'{torch_hip_guard} && \
              (pip install --force-reinstall --no-cache-dir --no-deps '{bnb_wheel}' || \
               pip install --force-reinstall --no-cache-dir --no-deps 'bitsandbytes>=0.49.1'))) && \
         mkdir -p {output_dir} && \
         : > {dir}/log.txt && : > {dir}/errorlog.txt && : > {dir}/train.log",
        hf_export = hf_export,
        dir = sh_quote(&run.remote_dir),
        venv = sh_quote(&venv_dir),
        torch_probe = torch_hip_probe,
        rocm_torch = rocm_torch_install,
        probe = zrald_probe,
        bnb_wheel = bnb_amd_wheel,
        output_dir = sh_quote(&output_dir),
        torch_hip_guard = torch_hip_guard,
    );

    Ok(format!(
        "{install_prefix} &&\n\
         {heredoc}\
         {{ echo {start_msg} && {venv}/bin/python3 {script}; }} \
           > >(tee -a {dir}/log.txt {dir}/train.log) \
           2> >(tee -a {dir}/errorlog.txt {dir}/train.log >&2)\n",
        install_prefix = install_prefix,
        heredoc = heredoc,
        venv = sh_quote(&venv_dir),
        dir = sh_quote(&run.remote_dir),
        start_msg = sh_quote("[zrald] starting ZRALD RAG reward GRPO trainer"),
        script = sh_quote(&script_path),
    ))
}
