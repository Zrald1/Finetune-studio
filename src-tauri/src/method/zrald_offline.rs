use base64::{engine::general_purpose::STANDARD as B64, Engine as _};

use crate::error::Result;
use crate::llamafactory;
use crate::runs::{LoraConfig, Run};

use super::common::sh_quote;
use super::{CommandKind, LlamaFactoryYamlOptions, MethodOptions};

pub const KEY: &str = "zrald_offline";

pub fn yaml() -> LlamaFactoryYamlOptions {
    LlamaFactoryYamlOptions::lora_like()
}

pub fn options() -> MethodOptions {
    MethodOptions {
        command_kind: CommandKind::ZraldOffline,
        yaml: yaml(),
        ..MethodOptions::lora_like(KEY)
    }
}

fn safe_write_cmd(dest_path: &str, content: &str) -> String {
    let encoded = B64.encode(content.as_bytes());
    let chunks = encoded
        .as_bytes()
        .chunks(76)
        .map(|chunk| std::str::from_utf8(chunk).unwrap_or(""))
        .map(sh_quote)
        .collect::<Vec<_>>()
        .join(" ");
    let dest = sh_quote(dest_path);
    let tmp = sh_quote(&format!("{dest_path}.b64.$$"));

    format!(
        "printf '%s\\n' {chunks} > {tmp} && base64 -d {tmp} > {dest} && rm -f {tmp}\n",
        chunks = chunks,
        tmp = tmp,
        dest = dest,
    )
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
    let script_path = format!("{}/zrald_offline.py", run.remote_dir);
    let runner_path = format!("{}/zrald_offline_run.sh", run.remote_dir);
    let reward_endpoint = if lora.zrald_reward_endpoint.trim().is_empty() {
        format!("http://127.0.0.1:{}", run.teacher_cfg.vllm_port)
    } else {
        lora.zrald_reward_endpoint.trim().to_string()
    };
    let local_reward_teacher = lora.zrald_reward_endpoint.trim().is_empty();
    let reward_model = lora.zrald_reward_model.trim().to_string();
    let train_questions = lora.zrald_train_questions.max(1);
    let benchmark_questions = lora.zrald_benchmark_questions.min(train_questions).max(1);
    assert!(
        benchmark_questions <= train_questions,
        "benchmark_questions ({}) must be <= train_questions ({})",
        benchmark_questions,
        train_questions
    );
    if benchmark_questions as f32 / train_questions as f32 > 0.5 {
        tracing::warn!(
            benchmark_questions,
            train_questions,
            "benchmark_questions is greater than 50% of train_questions; consider using a 10-20% benchmark split"
        );
    }
    let num_generations = lora.zrald_num_generations.clamp(2, 8);
    let max_completion_tokens = lora
        .zrald_max_completion_tokens
        .clamp(64, lora.cutoff_len.max(64));
    let hf_dataset_repos = if run.hub_dataset.enabled {
        llamafactory::hub_dataset_repos(run)
    } else {
        Vec::new()
    };

    let teacher_port = run.teacher_cfg.vllm_port;
    let teacher_log = format!("{}/zrald_offline_teacher.log", run.remote_dir);
    let teacher_pid = format!("{}/zrald_offline_teacher.pid", run.remote_dir);
    let teacher_cmd = if let Some(custom_cmd) = run
        .teacher_cfg
        .custom_serve_cmd
        .as_ref()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
    {
        format!("{hf_export}cd /app && {custom_cmd}")
    } else {
        let mut tokenizer_arg = String::new();
        let repo_id_lower = run.teacher_cfg.repo_id.to_lowercase();
        if repo_id_lower.contains("gguf") {
            let parts: Vec<&str> = run.teacher_cfg.repo_id.split('/').collect();
            let base_repo = if parts.len() >= 2 {
                format!(
                    "{}/{}",
                    parts[0],
                    parts[1].split(':').next().unwrap_or(parts[1])
                )
            } else {
                run.teacher_cfg
                    .repo_id
                    .split(':')
                    .next()
                    .unwrap_or(&run.teacher_cfg.repo_id)
                    .to_string()
            };
            let base_model = base_repo
                .replace("-GGUF", "")
                .replace("-gguf", "")
                .replace(".GGUF", "")
                .replace(".gguf", "");
            tokenizer_arg = format!("--tokenizer {}", sh_quote(&base_model));
        }
        let vllm_env = "export PYTHONUNBUFFERED=1; \
             export MASTER_ADDR=127.0.0.1; \
             export GLOO_SOCKET_IFNAME=lo; \
             export NCCL_SOCKET_IFNAME=lo; \
             export VLLM_HOST_IP=127.0.0.1; \
             export VLLM_SLEEP_WHEN_IDLE=1; \
             export VLLM_USE_DEEP_GEMM=0; \
             export VLLM_USE_FLASHINFER_MOE_FP16=1; \
             export VLLM_USE_FLASHINFER_SAMPLER=0; \
             export VLLM_ROCM_USE_AITER=1; \
             export VLLM_ROCM_USE_AITER_FP4BMM=0; \
             export HIP_FORCE_DEV_KERNARG=1; \
             export OMP_NUM_THREADS=4; ";
        format!(
            "{hf_export}{vllm_env}{runtime_prepare}cd /app && vllm serve {model} --port {port} --host 0.0.0.0 \
             --max-model-len {max_model_len} --dtype {dtype} --download-dir /root/hf-cache \
             --tensor-parallel-size {tensor_parallel} --gpu-memory-utilization {gpu_memory_utilization} {tokenizer_arg} {extra_args}",
            runtime_prepare = run.teacher_cfg.vllm_runtime_prepare_cmd(),
            model = sh_quote(&run.teacher_cfg.repo_id),
            port = teacher_port,
            max_model_len = run.teacher_cfg.max_model_len,
            dtype = sh_quote(&run.teacher_cfg.dtype),
            tensor_parallel = run.teacher_cfg.tensor_parallel,
            gpu_memory_utilization = run.teacher_cfg.gpu_memory_utilization,
            tokenizer_arg = tokenizer_arg,
            extra_args = run.teacher_cfg.vllm_extra_args(),
        )
    };

    let mut py = r##"import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import statistics
import sys
import time
from collections import OrderedDict
from pathlib import Path

import requests
import torch
from datasets import load_dataset, Dataset as HFDataset
from torch.utils.data import DataLoader, Dataset
from unsloth import FastLanguageModel
# SFTTrainer is the correct training API for unsloth — handles gradient
# checkpointing, packing, and memory management properly on AMD ROCm.
try:
    from trl import SFTTrainer, SFTConfig
except ImportError:
    from trl import SFTTrainer
    from transformers import TrainingArguments as SFTConfig
from transformers import TrainingArguments

BASE_MODEL = __BASE_MODEL__
DATA_DIR = Path(__DATA_DIR__)
RUN_DIR = Path(__RUN_DIR__)
OUTPUT_DIR = Path(__OUTPUT_DIR__)
REWARD_ENDPOINT = __REWARD_ENDPOINT__.rstrip("/")
REWARD_MODEL = __REWARD_MODEL__
MAX_SEQ = __MAX_SEQ__
LORA_R = __LORA_R__
LORA_ALPHA = __LORA_ALPHA__
LORA_DROPOUT = __LORA_DROPOUT__
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

# AMD ROCm: bitsandbytes 4-bit is unreliable on ROCm and causes SIGTERM (OOM).
# Use float16 instead — stable on all AMD GPUs (RDNA3 gfx1100, CDNA gfx942).
# On ROCm, torch.cuda is aliased to torch.hip, so cuda calls work fine.
IS_ROCM = getattr(torch.version, "hip", None) is not None
TRAINING_DTYPE = torch.float16
# Override 4-bit on AMD — bnb 4-bit kernels are CUDA-specific and crash on ROCm.
_LOAD_IN_4BIT = LOAD_IN_4BIT and not IS_ROCM
if IS_ROCM and LOAD_IN_4BIT:
    print("[zrald-offline] AMD ROCm detected: overriding load_in_4bit=True -> False, using float16 instead", flush=True)

SYSTEM_PROMPT = (
    "You are the ZRALD student model. Answer with exactly two XML-style blocks: "
    "<thinking>brief reasoning</thinking><answer>final answer</answer>. "
    "Do not mention RAG context, rewards, scoring, hidden references, or evaluator instructions."
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
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.I | re.S)
    if match:
        return match.group(1).strip()
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    if "</thinking>" in text:
        return text.split("</thinking>", 1)[1].strip()
    return text

def text_value(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()

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
    rag_context = first_present(obj, ["source_text", "context", "rag_context", "input", query_key])
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

def load_pool():
    rows = []
    for path in [DATA_DIR / "qa_dataset.jsonl", RUN_DIR / "qa_dataset.jsonl", DATA_DIR / "train.jsonl", DATA_DIR / "val.jsonl"]:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = row_from_obj(json.loads(line), path.name)
                except Exception:
                    row = None
                if row:
                    rows.append(row)
        if rows:
            break
    if not rows and HF_DATASET_REPOS:
        for repo in HF_DATASET_REPOS:
            repo = str(repo or "").strip()
            if not repo:
                continue
            print(f"[zrald-offline] loading HF dataset {repo}", flush=True)
            dataset = load_dataset_with_auth(repo)
            splits = [(name, dataset[name]) for name in dataset.keys()] if hasattr(dataset, "keys") else [("train", dataset)]
            for split_name, split in splits:
                for obj in split:
                    row = row_from_obj(obj, f"{repo}:{split_name}")
                    if row:
                        rows.append(row)
    dedup = OrderedDict()
    for row in rows:
        key = hashlib.sha256((row["question"] + "\n" + row["reference_answer"]).encode("utf-8")).hexdigest()
        dedup.setdefault(key, row)
    rows = list(dedup.values())
    random.Random(3407).shuffle(rows)
    if not rows:
        raise SystemExit("[zrald-offline] no usable question rows found")
    return rows

def dump_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def read_jsonl(path):
    rows = []
    path = Path(path)
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def vram_flush():
    """Aggressively free GPU memory — call before every model load on AMD."""
    gc.collect()
    try:
        # On ROCm, torch.cuda is aliased to torch.hip — both work.
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    gc.collect()
    if IS_ROCM:
        try:
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"[zrald-offline] VRAM reserved after flush: {reserved:.2f} GB", flush=True)
        except Exception:
            pass

def cleanup_model(model=None, tokenizer=None):
    """Unload model from GPU and flush VRAM — critical on AMD to prevent OOM."""
    if model is not None:
        try:
            model.cpu()
        except Exception:
            pass
    del model
    del tokenizer
    vram_flush()

def prompt_to_text(messages, tokenizer):
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except Exception:
            messages = [{"role": "user", "content": messages}]
    if not isinstance(messages, list) or not messages:
        messages = [{"role": "user", "content": str(messages)}]
    messages = [
        message if isinstance(message, dict) else {"role": "user", "content": str(message)}
        for message in messages
    ]
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as exc:
            print(f"[zrald-offline] apply_chat_template failed: {exc}; falling back", flush=True)
    return "\n".join(f"{m.get('role', 'user').upper()}: {m.get('content', '')}" for m in messages) + "\nASSISTANT:"

def load_student(train_adapter=False):
    vram_flush()  # flush before every model load
    print(f"[zrald-offline] loading student: {BASE_MODEL} train_adapter={train_adapter} 4bit={_LOAD_IN_4BIT} dtype={'float16' if IS_ROCM else 'auto'}", flush=True)
    kwargs = dict(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ,
        load_in_4bit=_LOAD_IN_4BIT,
    )
    if IS_ROCM:
        # On AMD: always use float16 — avoids bitsandbytes 4-bit kernel crashes
        kwargs["dtype"] = TRAINING_DTYPE
    model, tokenizer = FastLanguageModel.from_pretrained(**kwargs)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if train_adapter:
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_R,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            use_gradient_checkpointing="unsloth",  # unsloth's smart gradient checkpointing
            random_state=3407,
        )
    return model, tokenizer

def generate_one(model, tokenizer, row, sample=False, seed=3407):
    model.eval()
    if hasattr(FastLanguageModel, "for_inference"):
        FastLanguageModel.for_inference(model)
    prompt = prompt_to_text(row["prompt"], tokenizer)
    encoded = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    encoded = {k: v.to(device) for k, v in encoded.items()}
    if sample:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    kwargs = {
        "max_new_tokens": MAX_COMPLETION,
        "do_sample": sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if sample:
        kwargs.update({"temperature": 0.9, "top_p": 0.95})
    with torch.no_grad():
        out = model.generate(**encoded, **kwargs)
    new_tokens = out[0][encoded["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def prepare():
    rows = load_pool()
    train_rows = rows[:min(TRAIN_LIMIT, len(rows))]
    benchmark_rows = rows[:min(BENCHMARK_N, len(rows))]
    dump_jsonl(RUN_DIR / "zrald_offline_train_prompts.jsonl", train_rows)
    dump_jsonl(RUN_DIR / "zrald_offline_benchmark_prompts.jsonl", benchmark_rows)
    print(f"[zrald-offline] question pool: train={len(train_rows)} benchmark={len(benchmark_rows)} candidates_per_question={NUM_GENERATIONS}", flush=True)
    model, tokenizer = load_student(train_adapter=False)
    before = []
    for idx, row in enumerate(benchmark_rows, 1):
        completion = generate_one(model, tokenizer, row, sample=False)
        before.append({**row, "idx": idx, "candidate_index": 0, "completion": completion})
        if idx % 10 == 0 or idx == len(benchmark_rows):
            print(f"[zrald-offline] benchmark-before answers: {idx}/{len(benchmark_rows)}", flush=True)
    dump_jsonl(RUN_DIR / "zrald_offline_benchmark_before_candidates.jsonl", before)
    candidates = []
    total = len(train_rows) * NUM_GENERATIONS
    done = 0
    for idx, row in enumerate(train_rows, 1):
        for cand_idx in range(NUM_GENERATIONS):
            seed = 3407 + idx * 97 + cand_idx
            completion = generate_one(model, tokenizer, row, sample=True, seed=seed)
            candidates.append({**row, "idx": idx, "candidate_index": cand_idx, "seed": seed, "completion": completion})
            done += 1
            if done % 20 == 0 or done == total:
                print(f"[zrald-offline] student candidates: {done}/{total}", flush=True)
    dump_jsonl(RUN_DIR / "zrald_offline_student_candidates.jsonl", candidates)
    cleanup_model(model, tokenizer)
    print("[zrald-offline] student unloaded; VRAM freed for teacher scoring", flush=True)

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
    res = requests.get(reward_models_url(), headers=headers, timeout=30)
    if res.status_code >= 400:
        raise SystemExit(f"[zrald-offline] reward model auto-detect failed: http {res.status_code}: {res.text[:300]}")
    for item in res.json().get("data", []):
        model_id = str(item.get("id") or "").strip()
        if model_id:
            print(f"[zrald-offline] auto-detected reward teacher model: {model_id}", flush=True)
            return model_id
    raise SystemExit("[zrald-offline] reward model auto-detect returned no models")

def parse_jsonish(text):
    text = str(text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}
    return {}

def clamp_score(value):
    try:
        return max(-1.0, min(1.0, float(value)))
    except Exception:
        return -0.25

def heuristic_adjustment(completion):
    text = str(completion or "").strip()
    if not text:
        return -0.60
    penalty = 0.0
    has_thinking = bool(
        re.search(r"<thinking>.*?</thinking>", text, flags=re.I | re.S)
        or re.search(r"<think>.*?</think>", text, flags=re.I | re.S)
    )
    has_answer = bool(re.search(r"<answer>.*?</answer>", text, flags=re.I | re.S))
    answer_text = strip_answer(text)
    if not has_thinking:
        penalty -= 0.10
    if not has_answer:
        penalty -= 0.20
    if len(answer_text) < 8:
        penalty -= 0.20
    elif len(answer_text) < 30:
        penalty -= 0.10
    if has_thinking and has_answer and len(answer_text) >= 30:
        penalty += 0.05
    return max(-0.60, penalty)

def judge_row(row, reward_model, phase):
    prompt = {
        "question": row.get("question", ""),
        "rag_context": row.get("rag_context", ""),
        "reference_answer": row.get("reference_answer", ""),
        "student_completion": row.get("completion", ""),
        "rubric": row.get("rubric", ""),
        "required_output": {"score": "number from -1.0 to 1.0", "verdict": "short label", "reason": "short private note"},
    }
    headers = {"Content-Type": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    body = {
        "model": reward_model,
        "temperature": REWARD_TEMP,
        "max_tokens": 256,
        "messages": [
            {"role": "system", "content": "You are the ZRALD offline reward teacher. Return strict JSON only. Grade factual correctness against the reference answer and stored RAG context. Never reward unsupported claims."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
    }
    score = -0.25
    verdict = "judge_error"
    reason = ""
    for _attempt in range(2):
        try:
            res = requests.post(reward_url(), headers=headers, json=body, timeout=120)
            if res.status_code >= 400:
                reason = f"http {res.status_code}: {res.text[:400]}"
                time.sleep(1.0)
                continue
            content = res.json()["choices"][0]["message"]["content"]
            judged = parse_jsonish(content)
            score = clamp_score(judged.get("score", -0.25))
            verdict = str(judged.get("verdict", "scored"))
            reason = str(judged.get("reason", ""))[:500]
            break
        except Exception as exc:
            reason = repr(exc)
            time.sleep(1.0)
    score = clamp_score(score + heuristic_adjustment(row.get("completion", "")))
    return {**row, "phase": phase, "score": score, "verdict": verdict, "reason": reason, "reward_model": reward_model}

def score_file(input_name, output_name, phase):
    reward_model = detect_reward_model()
    rows = read_jsonl(RUN_DIR / input_name)
    scored = []
    for idx, row in enumerate(rows, 1):
        scored.append(judge_row(row, reward_model, phase))
        if idx % 20 == 0 or idx == len(rows):
            recent = [r["score"] for r in scored[-20:]]
            print(f"[zrald-offline] scored {phase}: {idx}/{len(rows)} recent_mean={statistics.fmean(recent):.3f}", flush=True)
    dump_jsonl(RUN_DIR / output_name, scored)
    return scored

def score_train_before():
    score_file("zrald_offline_student_candidates.jsonl", "zrald_offline_rewards_train.jsonl", "train")
    score_file("zrald_offline_benchmark_before_candidates.jsonl", "zrald_offline_benchmark_before.jsonl", "benchmark_before")

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

def best_training_examples():
    scored = read_jsonl(RUN_DIR / "zrald_offline_rewards_train.jsonl")
    groups = OrderedDict()
    for row in scored:
        key = hashlib.sha256((row["question"] + "\n" + row["reference_answer"]).encode("utf-8")).hexdigest()
        groups.setdefault(key, []).append(row)
    examples = []
    for rows in groups.values():
        best = max(rows, key=lambda r: float(r.get("score", -1.0)))
        answer = best.get("completion", "")
        answer_source = "student_candidate"
        if float(best.get("score", -1.0)) < 0.0:
            ref = best.get("reference_answer", "").strip()
            answer = (
                "<thinking>\n"
                "The student model did not produce a satisfactory answer. "
                "Falling back to the reference answer derived from the source material.\n"
                "</thinking>\n\n"
                f"<answer>{ref}</answer>"
            )
            answer_source = "teacher_reference_fallback"
        examples.append({
            "prompt": best["prompt"],
            "question": best["question"],
            "answer": answer,
            "score": best.get("score", 0.0),
            "answer_source": answer_source,
            "reference_answer": best.get("reference_answer", ""),
            "source": best.get("source", ""),
        })
    dump_jsonl(RUN_DIR / "zrald_offline_sft_train.jsonl", examples)
    return examples


def train_after():
    examples = best_training_examples()
    if not examples:
        raise SystemExit("[zrald-offline] no scored examples available for training")
    vram_flush()  # ensure VRAM is free before loading student for training
    model, tokenizer = load_student(train_adapter=True)
    if hasattr(FastLanguageModel, "for_training"):
        FastLanguageModel.for_training(model)

    # Build HuggingFace Dataset for SFTTrainer — the correct API for unsloth.
    # SFTTrainer handles gradient checkpointing, packing, and memory management
    # properly on AMD ROCm, unlike a manual training loop.
    def format_example(ex):
        prompt = prompt_to_text(ex["prompt"], tokenizer)
        answer = str(ex["answer"]).strip()
        eos = tokenizer.eos_token or ""
        return {"text": prompt + answer + eos}

    formatted = [format_example(ex) for ex in examples]
    hf_ds = HFDataset.from_list(formatted)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_epochs = max(1, math.ceil(EPOCHS))
    print(f"[zrald-offline] training student adapter via SFTTrainer: examples={len(hf_ds)} epochs={total_epochs} bs={max(1,PER_DEVICE_BS)} grad_accum={max(1,GRAD_ACCUM)}", flush=True)

    # SFTConfig / TrainingArguments — AMD-safe settings
    try:
        train_args = SFTConfig(
            output_dir=str(OUTPUT_DIR),
            num_train_epochs=total_epochs,
            per_device_train_batch_size=max(1, PER_DEVICE_BS),
            gradient_accumulation_steps=max(1, GRAD_ACCUM),
            learning_rate=LR,
            fp16=IS_ROCM,          # float16 on AMD
            bf16=not IS_ROCM,      # bf16 on CUDA (if supported)
            logging_steps=5,
            save_strategy="no",    # we save manually below
            optim="adamw_torch",   # pure-PyTorch AdamW — no fused CUDA kernels needed
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            report_to="none",
            dataloader_pin_memory=False,  # AMD: pinned memory can cause hangs
            max_seq_length=MAX_SEQ,
            dataset_text_field="text",
            packing=False,
        )
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=hf_ds,
            args=train_args,
        )
    except TypeError:
        # Older trl that doesn't have SFTConfig — fall back to TrainingArguments
        train_args = TrainingArguments(
            output_dir=str(OUTPUT_DIR),
            num_train_epochs=total_epochs,
            per_device_train_batch_size=max(1, PER_DEVICE_BS),
            gradient_accumulation_steps=max(1, GRAD_ACCUM),
            learning_rate=LR,
            fp16=IS_ROCM,
            bf16=not IS_ROCM,
            logging_steps=5,
            save_strategy="no",
            optim="adamw_torch",
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            report_to="none",
            dataloader_pin_memory=False,
        )
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=hf_ds,
            args=train_args,
            dataset_text_field="text",
            max_seq_length=MAX_SEQ,
        )

    train_result = trainer.train()
    losses = [x["loss"] for x in trainer.state.log_history if "loss" in x]
    (RUN_DIR / "zrald_offline_train_loss.json").write_text(
        json.dumps({"losses": losses, "train_runtime": train_result.metrics.get("train_runtime", 0)}, indent=2) + "\n",
        encoding="utf-8"
    )
    print(f"[zrald-offline] training complete; saving LoRA adapter to {OUTPUT_DIR}", flush=True)

    # Flush activations before save — prevents OOM during model.save_pretrained on AMD
    vram_flush()
    # Use unsloth's memory-safe save (limits GPU memory used during serialisation)
    try:
        model.save_pretrained(str(OUTPUT_DIR), maximum_memory_usage=0.5)
    except TypeError:
        # Older unsloth without maximum_memory_usage parameter
        model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("[zrald-offline] LoRA adapter saved; generating after-benchmark answers", flush=True)

    # Switch to inference mode and generate benchmark completions
    if hasattr(FastLanguageModel, "for_inference"):
        FastLanguageModel.for_inference(model)
    benchmark_rows = read_jsonl(RUN_DIR / "zrald_offline_benchmark_prompts.jsonl")
    after = []
    for idx, row in enumerate(benchmark_rows, 1):
        completion = generate_one(model, tokenizer, row, sample=False)
        after.append({**row, "idx": idx, "candidate_index": 0, "completion": completion})
        if idx % 10 == 0 or idx == len(benchmark_rows):
            print(f"[zrald-offline] benchmark-after answers: {idx}/{len(benchmark_rows)}", flush=True)
    dump_jsonl(RUN_DIR / "zrald_offline_benchmark_after_candidates.jsonl", after)
    cleanup_model(model, tokenizer)
    print("[zrald-offline] student unloaded after training and after-benchmark generation", flush=True)


def score_after_report():
    after = score_file("zrald_offline_benchmark_after_candidates.jsonl", "zrald_offline_benchmark_after.jsonl", "benchmark_after")
    before = read_jsonl(RUN_DIR / "zrald_offline_benchmark_before.jsonl")
    train_rewards = read_jsonl(RUN_DIR / "zrald_offline_rewards_train.jsonl")
    before_scores = [float(r.get("score", 0.0)) for r in before]
    after_scores = [float(r.get("score", 0.0)) for r in after]
    train_scores = [float(r.get("score", 0.0)) for r in train_rewards]
    paired = [a - b for a, b in zip(after_scores, before_scores)]
    report = {
        "method": "ZRALD Offline",
        "meaning": "Low-VRAM offline ZRALD preference distillation",
        "rewardEndpoint": REWARD_ENDPOINT,
        "rewardModel": after[0].get("reward_model", REWARD_MODEL) if after else REWARD_MODEL,
        "numGenerations": NUM_GENERATIONS,
        "trainQuestions": len(read_jsonl(RUN_DIR / "zrald_offline_train_prompts.jsonl")),
        "benchmarkQuestions": len(before_scores),
        "trainRewards": summarize(train_scores),
        "before": summarize(before_scores),
        "after": summarize(after_scores),
        "deltaMean": (statistics.fmean(after_scores) if after_scores else 0.0) - (statistics.fmean(before_scores) if before_scores else 0.0),
        "paired": {
            "meanDelta": statistics.fmean(paired) if paired else 0.0,
            "wins": sum(1 for d in paired if d > 0.05),
            "losses": sum(1 for d in paired if d < -0.05),
            "ties": sum(1 for d in paired if -0.05 <= d <= 0.05),
        },
    }
    artifacts_dir = OUTPUT_DIR / "zrald_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    def copy_artifact(name):
        src = RUN_DIR / name
        if src.exists():
            shutil.copy2(src, artifacts_dir / name)
            copied.append(name)
    for name in [
        "zrald_offline_train_prompts.jsonl",
        "zrald_offline_benchmark_prompts.jsonl",
        "zrald_offline_student_candidates.jsonl",
        "zrald_offline_rewards_train.jsonl",
        "zrald_offline_sft_train.jsonl",
        "zrald_offline_benchmark_before_candidates.jsonl",
        "zrald_offline_benchmark_before.jsonl",
        "zrald_offline_benchmark_after_candidates.jsonl",
        "zrald_offline_benchmark_after.jsonl",
        "zrald_offline_train_loss.json",
    ]:
        copy_artifact(name)
    report["artifacts"] = copied + ["zrald_report.json", "README.md", "manifest.json"]
    (RUN_DIR / "zrald_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (artifacts_dir / "zrald_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (artifacts_dir / "README.md").write_text(
        "# ZRALD Offline artifacts\n\n"
        "Teacher and student are staged separately to reduce VRAM pressure.\n\n"
        "- zrald_offline_student_candidates.jsonl: four saved student answers per training question.\n"
        "- zrald_offline_rewards_train.jsonl: teacher scores for saved student answers.\n"
        "- zrald_offline_sft_train.jsonl: selected/fallback answers used for adapter training.\n"
        "- zrald_offline_benchmark_before.jsonl and zrald_offline_benchmark_after.jsonl: teacher-scored benchmark comparisons.\n"
        "- zrald_report.json: aggregate before/after and reward summaries.\n",
        encoding="utf-8",
    )
    (artifacts_dir / "manifest.json").write_text(json.dumps({
        "method": "ZRALD Offline",
        "baseModel": BASE_MODEL,
        "rewardEndpoint": REWARD_ENDPOINT,
        "numGenerations": NUM_GENERATIONS,
        "benchmark": {"before": report["before"], "after": report["after"], "deltaMean": report["deltaMean"], "paired": report["paired"]},
        "files": report["artifacts"],
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("[zrald-offline] report", json.dumps(report, ensure_ascii=False), flush=True)
    print(f"[zrald-offline] artifacts saved to {artifacts_dir}: {', '.join(report['artifacts'])}", flush=True)

COMMANDS = {
    "prepare": prepare,
    "score_train_before": score_train_before,
    "train_after": train_after,
    "score_after_report": score_after_report,
}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd not in COMMANDS:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} [{'|'.join(COMMANDS)}]")
    COMMANDS[cmd]()
"##.to_string();

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
        ("__LORA_DROPOUT__", lora.dropout.to_string()),
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

    // Isolated unsloth training venv (see install_prefix below). Defined here
    // because the runner script needs its python path for the student stages.
    let venv_dir = format!("{}/.zrald_venv", run.remote_dir);

    let runner = format!(
        r#"#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
# CRITICAL: these must be set at runtime (not just install time) so unsloth
# loads in ROCm mode. Without UNSLOTH_IS_ROCM=1, unsloth falls back to CUDA/CPU,
# fails to find the AMD GPU, and the process is killed with SIGTERM (exit 143).
export UNSLOTH_IS_ROCM=1
export PYTORCH_ROCM_ARCH="${{PYTORCH_ROCM_ARCH:-gfx1100}}"
LOCAL_REWARD_TEACHER={local_reward_teacher}
TEACHER_PORT={teacher_port}
TEACHER_LOG={teacher_log}
TEACHER_PID={teacher_pid}
TEACHER_CMD={teacher_cmd}
# Student stages run in the isolated unsloth venv; the teacher (vLLM) must run
# with the container's system Python — never the venv. PYBIN selects the venv
# python for the ZRALD stages; boot_teacher strips the venv from its env so
# `vllm serve` uses the container's prebuilt ROCm vLLM.
PYBIN={venv_dir}/bin/python3
teacher_live=0

# Workaround: libdrm looks for amdgpu.ids to enumerate GPU IDs. If the file is
# missing it prints a warning and may cause GPU detection to fail. Create a
# minimal stub so libdrm finds the file.
if [ ! -f /opt/amdgpu/share/libdrm/amdgpu.ids ]; then
  mkdir -p /opt/amdgpu/share/libdrm 2>/dev/null || true
  touch /opt/amdgpu/share/libdrm/amdgpu.ids 2>/dev/null || true
fi

stop_teacher() {{
  if [ "$LOCAL_REWARD_TEACHER" != "1" ]; then return 0; fi
  echo "[zrald-offline] stopping reward teacher"
  if [ -f "$TEACHER_PID" ]; then kill "$(cat "$TEACHER_PID")" 2>/dev/null || true; fi
  pkill -f '[v]llm.entrypoints' 2>/dev/null || true
  pkill -f '[v]llm serve' 2>/dev/null || true
  pkill -f 'multiproc.*vllm' 2>/dev/null || true
  pkill -f 'python.*vllm' 2>/dev/null || true
  pkill -f 'sglang.launch_server' 2>/dev/null || true
  pkill -f 'python.*sglang' 2>/dev/null || true
  (command -v fuser >/dev/null 2>&1 && fuser -k "$TEACHER_PORT"/tcp 2>/dev/null) || true
  rm -f "$TEACHER_PID" 2>/dev/null || true
  teacher_live=0
  # Wait for GPU VRAM to be released before loading student model
  echo "[zrald-offline] waiting 15s for VRAM to free after stopping teacher..."
  sleep 15
}}

boot_teacher() {{
  if [ "$LOCAL_REWARD_TEACHER" != "1" ]; then return 0; fi
  stop_teacher || true
  echo "[zrald-offline] booting reward teacher on port $TEACHER_PORT"
  mkdir -p /root/hf-cache "$(dirname "$TEACHER_LOG")"
  : > "$TEACHER_LOG"
  nohup env -u VIRTUAL_ENV -u PYTHONHOME bash -lc "$TEACHER_CMD" > "$TEACHER_LOG" 2>&1 &
  echo $! > "$TEACHER_PID"
  teacher_live=1
  local boot_start=$SECONDS
  for i in $(seq 1 240); do
    code="$(curl -s -o /dev/null -w '%{{http_code}}' "http://127.0.0.1:$TEACHER_PORT/v1/models" 2>/dev/null || echo 000)"
    if [ "$code" = "200" ]; then
      echo "[zrald-offline] reward teacher ready in $((SECONDS - boot_start))s"
      return 0
    fi
    if [ $i -eq 12 ] || [ $i -eq 24 ] || [ $((i % 40)) -eq 0 ]; then
      echo "[zrald-offline] still waiting for teacher (${{i}}x5s elapsed)..."
      tail -n 20 "$TEACHER_LOG" 2>/dev/null || true
      if [ -f "$TEACHER_PID" ] && ! kill -0 "$(cat "$TEACHER_PID")" 2>/dev/null; then
        echo "[zrald-offline] ERROR: teacher process died" >&2
        tail -n 80 "$TEACHER_LOG" 2>/dev/null || true
        exit 1
      fi
    fi
    sleep 5
  done
  tail -n 160 "$TEACHER_LOG" 2>/dev/null || true
  echo "[zrald-offline] reward teacher boot timeout" >&2
  exit 1
}}

cleanup() {{
  local ec=$?
  if [ "$teacher_live" = "1" ]; then stop_teacher || true; fi
  if [ $ec -ne 0 ]; then
    echo "[zrald-offline] runner exiting with code $ec" >&2
    if command -v rocm-smi >/dev/null 2>&1; then rocm-smi 2>/dev/null || true; fi
  fi
}}
trap cleanup EXIT

# SIGTERM handler — print diagnostics so the user knows what was killed
handle_sigterm() {{
  echo "[zrald-offline] SIGTERM received — process was killed (likely OOM or container stop)" >&2
  echo "[zrald-offline] Check GPU memory with: rocm-smi" >&2
  echo "[zrald-offline] If OOM: reduce batch_size, lower epochs, or increase GPU memory utilization budget" >&2
  exit 143
}}
trap handle_sigterm TERM

# Run one ZRALD stage and, on failure, surface WHY with the real exit code.
# A stage killed by the OOM killer dies with 137 (SIGKILL) or 143 (SIGTERM);
# bash's `|| ` branch would otherwise mask that as a bare "exit 1" and the user
# never learns it was OOM. Echo the actual code and an OOM hint when it matches.
run_stage() {{
  local stage="$1"
  local max_retries="${{2:-1}}"
  local attempt=0
  local ec=0
  while [ "$attempt" -lt "$max_retries" ]; do
    if "$PYBIN" {script} "$stage"; then
      return 0
    else
      ec=$?
    fi
    attempt=$((attempt + 1))
    echo "[zrald-offline] $stage stage failed (exit $ec)" >&2
    if [ "$ec" -eq 137 ] || [ "$ec" -eq 143 ]; then
      echo "[zrald-offline] exit $ec means the $stage process was KILLED (OOM killer or container stop), not a Python error." >&2
      echo "[zrald-offline] Free VRAM/RAM: lower zrald_train_questions, zrald_num_generations, batch size, or cutoff_len; check rocm-smi." >&2
      if command -v rocm-smi >/dev/null 2>&1; then rocm-smi 2>/dev/null || true; fi
      exit "$ec"
    fi
    if [ "$attempt" -lt "$max_retries" ]; then
      echo "[zrald-offline] $stage failed (exit $ec), retry $attempt/$max_retries in 10s..." >&2
      sleep 10
    fi
  done
  echo "[zrald-offline] $stage failed after $max_retries attempt(s) (exit $ec)" >&2
  exit "$ec"
}}

stop_teacher || true
echo "[zrald-offline] stage: prepare (generating student candidates)"
run_stage prepare 1
echo "[zrald-offline] stage: boot_teacher for scoring"
boot_teacher
echo "[zrald-offline] stage: score_train_before"
run_stage score_train_before 2
stop_teacher
echo "[zrald-offline] stage: train_after (training student adapter)"
run_stage train_after 1
echo "[zrald-offline] stage: boot_teacher for after-scoring"
boot_teacher
echo "[zrald-offline] stage: score_after_report"
run_stage score_after_report 2
stop_teacher
trap - EXIT
"#,
        local_reward_teacher = if local_reward_teacher { "1" } else { "0" },
        venv_dir = sh_quote(&venv_dir),
        teacher_port = teacher_port,
        teacher_log = sh_quote(&teacher_log),
        teacher_pid = sh_quote(&teacher_pid),
        teacher_cmd = sh_quote(&teacher_cmd),
        script = sh_quote(&script_path),
    );
    let py = py.replace("\r", "");
    let runner = runner.replace("\r", "");

    let py_write = safe_write_cmd(&script_path, &py);
    let runner_write = format!(
        "{}chmod +x {}\n",
        safe_write_cmd(&runner_path, &runner),
        sh_quote(&runner_path)
    );
    let bnb_amd_wheel = "https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_main/bitsandbytes-1.33.7.preview-py3-none-manylinux_2_24_x86_64.whl";
    // CRITICAL: training runs via `docker exec` INSIDE the shared `rocm-vllm`
    // container — the SAME interpreter that hosts the vLLM teacher + embedder.
    // A bare `pip install --force-reinstall torch ...` there overwrites the
    // container's ROCm torch out from under vLLM's prebuilt C extensions
    // (`vllm._C`, `flash_attn_2_cuda`), producing on the next teacher boot:
    //   undefined symbol: _ZN3c103hip28c10_hip_check_implementation...
    //   Skipping import of cpp extensions due to incompatible torch version
    // and a triton skew (`triton.language has no attribute constexpr_function`).
    //
    // Fix: build the unsloth training stack in a FULLY ISOLATED venv that owns
    // its OWN ROCm torch (no --system-site-packages). vLLM's prebuilt C
    // extensions are tightly coupled to the container's exact torch ABI, while
    // unsloth needs a torch it controls — sharing one torch between the two is
    // the original sin. The venv decouples them completely: training can never
    // again overwrite the torch that the teacher's vLLM depends on. `venv_dir`
    // is defined above (the runner script also needs it).
    //
    // Probe runs INSIDE the venv (after activation) — torch must expose HIP and
    // unsloth/peft must import.
    let offline_probe =
        "python3 -c 'import torch,sys; sys.exit(0 if getattr(torch.version,\"hip\",None) else 1)' \
             >/dev/null 2>&1 && \
         python3 -c 'import requests, datasets, unsloth; import peft' >/dev/null 2>&1";
    let torch_hip_probe = "python3 -c 'import torch,sys; sys.exit(0 if getattr(torch.version,\"hip\",None) else 1)' >/dev/null 2>&1";
    // The venv's own ROCm torch. Pinned <2.11 because torch 2.11+ only has ROCm
    // 7.2 wheels; index rocm7.0 matches the container's ROCm runtime. Per memory
    // note + https://unsloth.ai/docs/get-started/install/amd .
    let rocm_torch_install = "pip install --no-cache-dir \
             --index-url https://download.pytorch.org/whl/rocm7.0 \
             'torch>=2.4,<2.11.0' 'torchvision<0.26.0' 'torchaudio<2.11.0'";
    // unsloth's transitive deps can pull a CUDA torch from the default PyPI
    // index, replacing the venv's ROCm torch. Re-verify HIP after install and
    // repair inside the venv only.
    let torch_hip_guard = " && (".to_string() + torch_hip_probe + " || " + rocm_torch_install + ")";

    // IMPORTANT: script writes and launch steps must remain genuine shell
    // statements. The `&&`-chained install prefix below uses Rust `\`
    // line-continuations, which collapse following lines into one logical shell
    // line. We therefore terminate the install prefix with a real newline
    // (`&&\n`) and emit script writes + final exec block as multi-line shell.
    //
    // Install steps:
    //   1. Create/activate an isolated venv (no system site-packages) and
    //      upgrade its pip.
    //   2. Install the venv's own ROCm torch (HIP), pinned <2.11.
    //   3. unsloth[amd] WITH deps (the `[amd]` marker is itself a dep; --no-deps
    //      would skip it). torch_hip_guard repairs torch if deps clobber it.
    //   4. AMD bitsandbytes wheel (--no-deps; non-standard version string).
    // The whole block is gated by `offline_probe` so a warm venv from a prior
    // run skips reinstall and starts training immediately.
    let install_prefix = format!(
        "set -eo pipefail; \
         {hf_export} cd {dir} && \
         export UNSLOTH_IS_ROCM=1 PYTORCH_ROCM_ARCH=${{PYTORCH_ROCM_ARCH:-gfx950}} \
                PYTHONUNBUFFERED=1 && \
         (test -d {venv}/bin || python3 -m venv {venv}) && \
         . {venv}/bin/activate && \
         ({probe} || \
             (python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
              ({torch_probe} || {rocm_torch}) && \
              pip install --no-cache-dir 'unsloth[amd]' 'unsloth_zoo' \
                 'datasets>=2.16.0' 'requests>=2.31.0' 'peft>=0.19,<0.20' \
                 'accelerate>=0.34.0' 'sentencepiece>=0.2.0' 'protobuf' 'hf_transfer' 'psutil' \
                 'trl>=0.8.0' 'transformers>=4.41.2,<4.58'{torch_hip_guard} && \
              (pip install --force-reinstall --no-cache-dir --no-deps '{bnb_wheel}' || \
               pip install --force-reinstall --no-cache-dir --no-deps 'bitsandbytes>=0.49.1'))) && \
         mkdir -p {output_dir} && \
         : > {dir}/log.txt && : > {dir}/errorlog.txt && : > {dir}/train.log",
        hf_export = hf_export,
        dir = sh_quote(&run.remote_dir),
        venv = sh_quote(&venv_dir),
        torch_probe = torch_hip_probe,
        rocm_torch = rocm_torch_install,
        probe = offline_probe,
        bnb_wheel = bnb_amd_wheel,
        output_dir = sh_quote(&output_dir),
        torch_hip_guard = torch_hip_guard,
    );

    // ── Launch the runner in a setsid-detached background process ───────────
    // ZRALD Offline training can run for many hours. A direct `bash runner`
    // in the exec_stream channel is vulnerable to SIGTERM if the SSH connection
    // drops (server reboot, NAT timeout, network blip). The fix:
    //
    //   1. Launch the runner via `setsid nohup bash runner &` — this detaches
    //      it from the SSH session's process group so SSH hangup (HUP) and
    //      terminal close don't propagate SIGTERM to the training process.
    //   2. Write the PID to a sentinel file so we can re-attach on reconnect.
    //   3. Write the exit code to a sentinel file when the runner finishes so
    //      the SSH waiter can surface the real exit code.
    //   4. The SSH exec_stream channel is held by a `tail -f train.log` +
    //      a PID-polling loop. This gives us live log streaming while the
    //      background process runs, and a clean exit with the runner's exit code
    //      when it finishes. If SSH drops, the tail/waiter dies but the nohup
    //      background process keeps running; the remote-tail poller in the Rust
    //      pipeline picks up new bytes from train.log on the next tick.
    let pid_file = format!("{}/zrald_offline_runner.pid", run.remote_dir);
    let exit_file = format!("{}/zrald_offline_runner.exit", run.remote_dir);
    let dir_q = sh_quote(&run.remote_dir);
    let runner_q = sh_quote(&runner_path);
    let pid_file_q = sh_quote(&pid_file);
    let exit_file_q = sh_quote(&exit_file);
    let start_msg_q = sh_quote("[zrald-offline] starting low-VRAM staged ZRALD");
    let poll_and_exit = format!(
        r#"RUNNER_PID=$(cat {pid_file_q} 2>/dev/null | tr -d '[:space:]') || true
if [ -z "$RUNNER_PID" ]; then
  echo "[zrald-offline] ERROR: could not read runner PID" >&2
  exit 143
fi
echo "[zrald-offline] runner detached as PID $RUNNER_PID - surviving SSH reconnects"
tail -f {dir_q}/train.log &
TAIL_PID=$!
_ec=143
for _i in $(seq 1 7200); do
  if [ -f {exit_file_q} ]; then
    _ec=$(cat {exit_file_q} 2>/dev/null | tr -d '[:space:]')
    break
  fi
  if ! kill -0 "$RUNNER_PID" 2>/dev/null; then
    sleep 3
    if [ -f {exit_file_q} ]; then
      _ec=$(cat {exit_file_q} 2>/dev/null | tr -d '[:space:]')
    fi
    break
  fi
  sleep 5
done
kill $TAIL_PID 2>/dev/null || true
wait $TAIL_PID 2>/dev/null || true
exit ${{_ec:-143}}
"#,
        pid_file_q = pid_file_q,
        exit_file_q = exit_file_q,
        dir_q = dir_q,
    );

    Ok(format!(
        "{install_prefix} &&\n\
         {py_write}\
         {runner_write}\
         rm -f {exit_file_q} {pid_file_q};\n\
         setsid nohup bash -c 'set -o pipefail; \
           {{ echo {start_msg_q} && bash {runner_q}; }} \
             > >(tee -a {dir_q}/log.txt {dir_q}/train.log) \
             2> >(tee -a {dir_q}/errorlog.txt {dir_q}/train.log >&2); \
           _ec=$?; echo $_ec > {exit_file_q}; exit $_ec' &\n\
         echo $! > {pid_file_q};\n\
         {poll_and_exit}",
        install_prefix = install_prefix,
        py_write = py_write,
        runner_write = runner_write,
        exit_file_q = exit_file_q,
        pid_file_q = pid_file_q,
        start_msg_q = start_msg_q,
        runner_q = runner_q,
        dir_q = dir_q,
        poll_and_exit = poll_and_exit,
    ))
}
