use crate::error::Result;
use crate::generator::GeneratedPair;
use crate::runs::{HubConfig, LoraConfig, Run};
use regex::Regex;
use serde::Serialize;
use serde_json::json;

/// Convert generated pairs → ShareGPT-style messages JSONL for LLaMA-Factory.
/// 95/5 train/val split, deterministic by source_chunk_id hash.
/// Convert generated pairs → format-specific JSONL for LLaMA-Factory.
/// 95/5 train/val split, deterministic by source_chunk_id hash.
pub fn build_jsonl(pairs: &[GeneratedPair]) -> (String, String) {
    let mut train = String::new();
    let mut val = String::new();
    for (i, p) in pairs.iter().enumerate() {
        let line = crate::generator::export_to_jsonl(p);
        let row = line.to_string();
        // Deterministic 1-in-20 → val.
        if i % 20 == 0 {
            val.push_str(&row);
            val.push('\n');
        } else {
            train.push_str(&row);
            train.push('\n');
        }
    }
    if val.is_empty() && !train.is_empty() {
        // Make sure val always has at least one row.
        let cut = train.rfind('\n').unwrap_or(0);
        val = train.split_off(cut).trim_start_matches('\n').to_string();
    }
    (train, val)
}

/// dataset_info.json that LLaMA-Factory expects in --dataset_dir.
pub fn dataset_info(run_name: &str, format: Option<crate::generator::DatasetFormat>) -> String {
    let fmt = format.unwrap_or(crate::generator::DatasetFormat::MultipleChoice);

    let (formatting, columns) = if fmt == crate::generator::DatasetFormat::InstructionIo {
        let mut cols = serde_json::Map::new();
        cols.insert("prompt".to_string(), json!("instruction"));
        cols.insert("query".to_string(), json!("input"));
        cols.insert("response".to_string(), json!("output"));
        ("alpaca".to_string(), cols)
    } else {
        let mut cols = serde_json::Map::new();
        cols.insert("messages".to_string(), json!("messages"));
        ("sharegpt".to_string(), cols)
    };

    let mut entry = json!({
        "file_name": "train.jsonl",
        "formatting": formatting,
        "columns": columns,
    });

    if formatting == "sharegpt" {
        entry["tags"] = json!({
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant"
        });
    }

    let mut entry_val = entry.clone();
    entry_val["file_name"] = json!("val.jsonl");

    let mut map = serde_json::Map::new();
    map.insert(run_name.to_string(), entry);
    map.insert(format!("{}_val", run_name), entry_val);

    serde_json::to_string_pretty(&serde_json::Value::Object(map)).unwrap()
}

/// Trimmed, deduplicated list of HF dataset repo IDs configured for this run.
/// Combines `repo_id` (primary) and `repo_ids` (extras), preserving order and
/// dropping blanks/duplicates. Used by Train-Only mode to fan out a single
/// LLaMA-Factory job over multiple HF datasets.
pub fn hub_dataset_repos(run: &Run) -> Vec<String> {
    let ds = &run.hub_dataset;
    let mut out: Vec<String> = Vec::new();
    let mut push = |s: &str| {
        let t = s.trim();
        if !t.is_empty() && !out.iter().any(|x| x == t) {
            out.push(t.to_string());
        }
    };
    push(&ds.repo_id);
    for r in &ds.repo_ids {
        push(r);
    }
    out
}

/// LLaMA-Factory dataset entry name(s) for a run. Returns a comma-separated
/// list when multiple HF datasets are configured (Train-Only multi-dataset),
/// matching the keys produced by `dataset_info_hf`.
pub fn hub_dataset_names(run: &Run) -> String {
    let base = format!("ft_{}", &run.id[..8]);
    let repos = hub_dataset_repos(run);
    if repos.len() <= 1 {
        return base;
    }
    repos
        .iter()
        .enumerate()
        .map(|(i, _)| format!("{base}_d{i}"))
        .collect::<Vec<_>>()
        .join(",")
}

/// dataset_info.json pointing to one or more Hugging Face Hub datasets.
/// Single-dataset runs keep the legacy entry name `ft_<id8>`; multi-dataset
/// runs emit `ft_<id8>_d0`, `ft_<id8>_d1`, ... so LLaMA-Factory can interleave
/// them via a comma-separated `dataset:` field.
pub fn dataset_info_hf(run: &Run) -> String {
    let base = format!("ft_{}", &run.id[..8]);
    let ds = &run.hub_dataset;
    let mut columns = ds.dataset_columns.clone();

    let format_str = if ds.dataset_format.is_empty() {
        let fmt = run
            .dataset_format
            .unwrap_or(crate::generator::DatasetFormat::MultipleChoice);
        if fmt == crate::generator::DatasetFormat::InstructionIo {
            "alpaca".to_string()
        } else {
            "sharegpt".to_string()
        }
    } else {
        ds.dataset_format.clone()
    };

    // Default columns if none provided
    if columns.is_empty() {
        if format_str == "sharegpt" {
            columns.insert("messages".to_string(), "messages".to_string());
        } else {
            let fmt = run
                .dataset_format
                .unwrap_or(crate::generator::DatasetFormat::MultipleChoice);
            if fmt == crate::generator::DatasetFormat::InstructionIo {
                columns.insert("prompt".to_string(), "instruction".to_string());
                columns.insert("query".to_string(), "input".to_string());
                columns.insert("response".to_string(), "output".to_string());
            } else {
                columns.insert("prompt".to_string(), "question".to_string());
                columns.insert("query".to_string(), "reasoning".to_string());
                columns.insert("response".to_string(), "answer".to_string());
            }
        }
    }

    let make_entry = |repo: &str| -> serde_json::Value {
        let mut entry = json!({
            "hf_hub_url": repo,
            "formatting": format_str,
            "columns": columns,
        });
        if format_str == "sharegpt" {
            entry["tags"] = json!({
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant"
            });
        }
        entry
    };

    let repos = hub_dataset_repos(run);
    let mut map = serde_json::Map::new();
    if repos.len() <= 1 {
        // Single dataset path — keep backward-compatible key (`ft_<id8>`).
        let repo = repos
            .into_iter()
            .next()
            .unwrap_or_else(|| ds.repo_id.clone());
        map.insert(base, make_entry(&repo));
    } else {
        for (i, repo) in repos.iter().enumerate() {
            map.insert(format!("{base}_d{i}"), make_entry(repo));
        }
    }
    serde_json::to_string_pretty(&serde_json::Value::Object(map)).unwrap()
}

/// Emit a self-contained Python script that cleans every dataset referenced by
/// `dataset_info.json` *in place* before training, then repoints the config at
/// the cleaned local files. Covers both data paths with one step:
///   - generated runs (entries with `file_name`) → cleaned local jsonl
///   - train-only runs (entries with `hf_hub_url`) → materialized from the Hub,
///     cleaned, and rewritten to a local `file_name` so the trainer reads the
///     cleaned copy instead of streaming the raw Hub data.
///
/// Cleaning is format-aware (sharegpt `messages` and alpaca `prompt/query/
/// response`), removing exact-duplicate rows and/or rows whose assistant/answer
/// text is shorter than `min_chars`. It prints `[clean] …` progress lines and a
/// final per-dataset summary so the user sees exactly what was dropped.
///
/// `data_dir` is the absolute path that holds `dataset_info.json` (the trainer's
/// `--dataset_dir`). Returns the script text to run with `python3 -c`.
pub fn dataset_clean_script(
    data_dir: &str,
    remove_duplicates: bool,
    remove_short: bool,
    min_chars: u32,
) -> String {
    const TEMPLATE: &str = r#"
import json, os, sys

DATA_DIR = "__DATA_DIR__"
REMOVE_DUP = __REMOVE_DUP__
REMOVE_SHORT = __REMOVE_SHORT__
MIN_CHARS = __MIN_CHARS__

info_path = os.path.join(DATA_DIR, "dataset_info.json")
with open(info_path, "r", encoding="utf-8") as f:
    info = json.load(f)

def answer_text(row, fmt):
    # Extract the assistant/answer text used for the short-paragraph filter.
    if fmt == "sharegpt":
        msgs = row.get("messages") or []
        parts = [m.get("content", "") for m in msgs if m.get("role") == "assistant"]
        if not parts and msgs:
            parts = [msgs[-1].get("content", "")]
        return "\n".join(p for p in parts if p)
    # alpaca-style
    return str(row.get("output") or row.get("response") or "")

def row_key(row, fmt):
    # Stable identity for de-duplication.
    return json.dumps(row, sort_keys=True, ensure_ascii=False)

def load_rows(entry):
    url = entry.get("hf_hub_url")
    if url:
        from datasets import load_dataset
        ds = load_dataset(url, split="train")
        return [dict(r) for r in ds]
    fn = entry.get("file_name")
    rows = []
    path = os.path.join(DATA_DIR, fn)
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

changed = False
for key, entry in list(info.items()):
    fmt = entry.get("formatting", "sharegpt")
    try:
        rows = load_rows(entry)
    except Exception as e:
        print("[clean] %s: skipped (load failed: %s)" % (key, e), flush=True)
        continue
    total = len(rows)
    kept = []
    seen = set()
    dropped_dup = 0
    dropped_short = 0
    for row in rows:
        if REMOVE_SHORT and len(answer_text(row, fmt).strip()) < MIN_CHARS:
            dropped_short += 1
            continue
        if REMOVE_DUP:
            k = row_key(row, fmt)
            if k in seen:
                dropped_dup += 1
                continue
            seen.add(k)
        kept.append(row)
    out_name = key + "_clean.jsonl"
    out_path = os.path.join(DATA_DIR, out_name)
    with open(out_path, "w", encoding="utf-8") as fh:
        for row in kept:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    # Repoint the dataset entry at the cleaned local file.
    entry.pop("hf_hub_url", None)
    entry["file_name"] = out_name
    changed = True
    print("[clean] %s: %d -> %d rows (removed %d duplicate, %d short)" % (
        key, total, len(kept), dropped_dup, dropped_short), flush=True)

if changed:
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print("[clean] dataset_info.json updated to use cleaned files", flush=True)
print("[clean] done", flush=True)
"#;

    TEMPLATE
        .replace("__DATA_DIR__", data_dir)
        .replace(
            "__REMOVE_DUP__",
            if remove_duplicates { "True" } else { "False" },
        )
        .replace(
            "__REMOVE_SHORT__",
            if remove_short { "True" } else { "False" },
        )
        .replace("__MIN_CHARS__", &min_chars.to_string())
}

/// LLaMA-Factory / Transformers cannot fine-tune directly from a GGUF repo —
/// those snapshots only contain the quantised weights and no tokenizer files.
/// If the user picked a `*-GGUF` repo as the student, fall back to the matching
/// safetensors repo (same logic the teacher vLLM path uses to resolve a
/// compatible tokenizer). Also strips a trailing `:Q4_K_M`-style quant tag.
pub fn resolve_trainable_repo(student_model: &str) -> String {
    let lower = student_model.to_lowercase();
    if !lower.contains("gguf") {
        return student_model.to_string();
    }
    let parts: Vec<&str> = student_model.split('/').collect();
    let base_repo = if parts.len() >= 2 {
        format!(
            "{}/{}",
            parts[0],
            parts[1].split(':').next().unwrap_or(parts[1])
        )
    } else {
        student_model
            .split(':')
            .next()
            .unwrap_or(student_model)
            .to_string()
    };
    base_repo
        .replace("-GGUF", "")
        .replace("-gguf", "")
        .replace(".GGUF", "")
        .replace(".gguf", "")
}

/// Pick a reasonable LLaMA-Factory template based on the student model repo id.
/// Falls back to "default" for unknown architectures (LLaMA-Factory accepts it).
pub fn pick_template(student_model: &str) -> &'static str {
    let lower = student_model.to_lowercase();
    if lower.contains("qwen3") {
        "qwen3"
    } else if lower.contains("qwen2.5") || lower.contains("qwen2") || lower.contains("qwen") {
        "qwen"
    } else if lower.contains("llama-3") || lower.contains("llama3") {
        "llama3"
    } else if lower.contains("llama-2") || lower.contains("llama2") || lower.contains("llama") {
        "llama2"
    } else if lower.contains("mistral") {
        "mistral"
    } else if lower.contains("gemma") {
        "gemma"
    } else if lower.contains("deepseek") {
        "deepseek"
    } else if lower.contains("phi") {
        "phi"
    } else {
        "default"
    }
}

/// train.yaml for `llamafactory-cli train`.
///
/// `resume`: when true, sets `resume_from_checkpoint: true` so LLaMA-Factory (HF
/// Trainer under the hood) picks up the latest checkpoint-*/ directory in
/// `output_dir` and continues. Also keeps `overwrite_output_dir: false` so the
/// existing checkpoints aren't wiped.
pub fn train_yaml(
    run: &Run,
    lora: &LoraConfig,
    _hub: &HubConfig,
    resume: bool,
    is_rocm: bool,
) -> Result<String> {
    #[derive(Serialize)]
    struct Y<'a> {
        model_name_or_path: &'a str,
        stage: &'a str,
        do_train: bool,
        finetuning_type: &'a str,
        #[serde(skip_serializing_if = "Option::is_none")]
        lora_target: Option<&'a str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        lora_rank: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        lora_alpha: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        lora_dropout: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        quantization_bit: Option<u8>,
        #[serde(skip_serializing_if = "Option::is_none")]
        quantization_method: Option<&'a str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        use_unsloth: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        use_dora: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        loraplus_lr_ratio: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pissa_init: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pissa_iter: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pissa_convert: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        freeze_trainable_layers: Option<i32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        use_galore: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        galore_layerwise: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        galore_target: Option<&'a str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        galore_rank: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        galore_update_interval: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        galore_scale: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        use_badam: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        badam_mode: Option<&'a str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        badam_switch_mode: Option<&'a str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        badam_switch_interval: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        badam_verbose: Option<u8>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pure_bf16: Option<bool>,
        dataset: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        eval_dataset: Option<String>,
        dataset_dir: String,
        template: &'a str,
        cutoff_len: u32,
        max_samples: u32,
        overwrite_cache: bool,
        preprocessing_num_workers: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        dataloader_pin_memory: Option<bool>,
        output_dir: String,
        logging_steps: u32,
        save_steps: u32,
        save_total_limit: u32,
        plot_loss: bool,
        overwrite_output_dir: bool,
        resume_from_checkpoint: bool,
        per_device_train_batch_size: u32,
        gradient_accumulation_steps: u32,
        learning_rate: f32,
        num_train_epochs: f32,
        lr_scheduler_type: &'a str,
        warmup_ratio: f32,
        #[serde(skip_serializing_if = "Option::is_none")]
        bf16: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        fp16: Option<bool>,
        ddp_timeout: u32,
        val_size: f32,
        per_device_eval_batch_size: u32,
        eval_strategy: &'a str,
        eval_steps: u32,
        report_to: &'a str,
        // HF Hub
        #[serde(skip_serializing_if = "Option::is_none")]
        push_to_hub: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        hub_model_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        hub_strategy: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        hub_private_repo: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        hub_always_push: Option<bool>,
    }

    let run_name = format!("ft_{}", &run.id[..8]);
    let trainable_model = resolve_trainable_repo(&run.student_model);
    let template = pick_template(&trainable_model);
    let method = lora.method.trim().to_lowercase();
    let method_yaml = crate::method::yaml(&method);
    // Vision-language models (Qwen-VL, Llava, etc.) emit collated tensors with
    // broadcast/expanded storage (e.g. 3D rope position_ids, expanded attention
    // masks). torch's `pin_memory` rejects tensors whose strides overlap memory,
    // crashing the first training step with "more than one element of the
    // written-to tensor refers to a single memory location". Disabling pinned
    // memory for VL training avoids the crash; the throughput cost is small
    // relative to GPU compute on these models.
    let model_lower = trainable_model.to_lowercase();
    let is_vl_model = model_lower.contains("-vl")
        || model_lower.contains("_vl")
        || model_lower.contains("llava")
        || model_lower.contains("vision")
        || model_lower.contains("internvl")
        || model_lower.contains("minicpm-v");
    let dataloader_pin_memory = if is_vl_model { Some(false) } else { None };
    // `loftq` is no longer surfaced in the UI (the offline scripts/loftq_init.py
    // workflow that LLaMA-Factory requires isn't wired up here); old saved runs
    // with that value fall through to QLoRA. `peft` is likewise an alias for
    // plain LoRA — both names accepted for resume compatibility only.

    // The app uploads adapters itself after training so it can repair PEFT
    // metadata before the Hub validates README.md. Trainer-managed pushes can
    // generate `base_model: ""` for VL LoRA runs, which makes an otherwise
    // successful training job exit as failed during checkpoint upload.
    let (push_to_hub, hub_model_id, hub_strategy, hub_private_repo, hub_always_push) =
        (None, None, None, None, None);

    let use_hf_dataset = run.hub_dataset.enabled && !hub_dataset_repos(run).is_empty();
    let eval_dataset = if use_hf_dataset {
        None
    } else {
        Some(format!("{}_val", run_name))
    };
    // `dataset` is comma-separated when training from multiple HF datasets,
    // otherwise the single legacy entry name `ft_<id8>`.
    let dataset_field = if use_hf_dataset {
        hub_dataset_names(run)
    } else {
        run_name.clone()
    };

    let y = Y {
        model_name_or_path: &trainable_model,
        stage: "sft",
        do_train: true,
        finetuning_type: method_yaml.finetuning_type,
        lora_target: if method_yaml.is_lora_family {
            Some("all")
        } else {
            None
        },
        lora_rank: if method_yaml.is_lora_family {
            Some(lora.r)
        } else {
            None
        },
        lora_alpha: if method_yaml.is_lora_family {
            Some(lora.alpha)
        } else {
            None
        },
        lora_dropout: if method_yaml.is_lora_family {
            Some(lora.dropout)
        } else {
            None
        },
        quantization_bit: method_yaml.quantization_bit,
        // LLaMA-Factory v0.9.4 QuantizationMethod enum: "bnb"|"gptq"|"awq"|"aqlm"|"quanto"|"eetq"|"hqq"|"mxfp4"|"fp8".
        // We use bitsandbytes via the "bnb" alias — "bitsandbytes" is rejected at arg-parse time.
        quantization_method: method_yaml.quantization_method,
        use_unsloth: if method_yaml.use_unsloth {
            Some(true)
        } else {
            None
        },
        use_dora: if method_yaml.use_dora {
            Some(true)
        } else {
            None
        },
        loraplus_lr_ratio: method_yaml.loraplus_lr_ratio,
        pissa_init: if method_yaml.pissa_init {
            Some(true)
        } else {
            None
        },
        // Number of SVD iterations and convert-on-save: without `pissa_convert`,
        // LF leaves the original base weights untouched and the residual is
        // never folded into the saved adapter — measurably hurts accuracy.
        pissa_iter: method_yaml.pissa_iter,
        pissa_convert: if method_yaml.pissa_convert {
            Some(true)
        } else {
            None
        },
        // Positive = last N transformer blocks are trainable; negative = first N.
        // Default of 2 mirrors LLaMA-Factory's freeze example.
        freeze_trainable_layers: method_yaml.freeze_trainable_layers,
        use_galore: if method_yaml.use_galore {
            Some(true)
        } else {
            None
        },
        galore_layerwise: if method_yaml.galore_layerwise {
            Some(true)
        } else {
            None
        },
        galore_target: method_yaml.galore_target,
        galore_rank: method_yaml.galore_rank,
        galore_update_interval: method_yaml.galore_update_interval,
        galore_scale: method_yaml.galore_scale,
        use_badam: if method_yaml.use_badam {
            Some(true)
        } else {
            None
        },
        badam_mode: method_yaml.badam_mode,
        badam_switch_mode: method_yaml.badam_switch_mode,
        badam_switch_interval: method_yaml.badam_switch_interval,
        badam_verbose: method_yaml.badam_verbose,
        // GaLore-layerwise + BAdam all benefit from / require pure bf16.
        // Layerwise optimizers update parameter groups one layer at a time so
        // the usual mixed-precision master copy doesn't help.
        pure_bf16: if method_yaml.pure_bf16 {
            Some(true)
        } else {
            None
        },
        dataset: dataset_field,
        eval_dataset,
        dataset_dir: format!("{}/data", run.remote_dir),
        template,
        cutoff_len: lora.cutoff_len,
        max_samples: 1_000_000,
        overwrite_cache: true,
        preprocessing_num_workers: 4,
        dataloader_pin_memory,
        output_dir: format!("{}/lora", run.remote_dir),
        logging_steps: 5,
        save_steps: lora.save_steps.max(10),
        save_total_limit: 3,
        plot_loss: true,
        // When resuming we MUST NOT overwrite — we'd lose the checkpoint we just resumed from.
        overwrite_output_dir: !resume,
        resume_from_checkpoint: resume,
        per_device_train_batch_size: lora.batch_size,
        gradient_accumulation_steps: lora.gradient_accumulation,
        learning_rate: lora.learning_rate,
        num_train_epochs: lora.epochs,
        lr_scheduler_type: "cosine",
        warmup_ratio: 0.05,
        bf16: if is_rocm { None } else { Some(true) },
        fp16: if is_rocm { Some(true) } else { None },
        ddp_timeout: 180_000_000,
        val_size: 0.05,
        per_device_eval_batch_size: lora.batch_size,
        eval_strategy: "steps",
        eval_steps: lora.save_steps.max(10),
        report_to: "none",
        push_to_hub,
        hub_model_id,
        hub_strategy,
        hub_private_repo,
        hub_always_push,
    };
    Ok(serde_yaml::to_string(&y)?)
}

/// Parse a single LLaMA-Factory stdout line for loss / epoch / step metrics.
/// Returns None if the line isn't a metric line.
#[derive(Debug, Clone, Serialize)]
pub struct ParsedMetric {
    pub step: u32,
    pub loss: f32,
    pub epoch: f32,
}

pub fn parse_metric(line: &str) -> Option<ParsedMetric> {
    // Matches HF Transformers Trainer log: {'loss': 1.234, 'learning_rate': ..., 'epoch': 0.12}
    // also matches: "loss=1.234, epoch=0.12, step=15"
    let r1 = Regex::new(r#"'loss'\s*:\s*([0-9.eE+-]+).*?'epoch'\s*:\s*([0-9.eE+-]+)"#).ok()?;
    if let Some(c) = r1.captures(line) {
        // also try to find a step
        let step_re = Regex::new(r#"global_step\s*=\s*(\d+)"#).ok()?;
        let step = step_re
            .captures(line)
            .and_then(|c| c.get(1).and_then(|m| m.as_str().parse().ok()))
            .unwrap_or(0);
        return Some(ParsedMetric {
            step,
            loss: c.get(1)?.as_str().parse().ok()?,
            epoch: c.get(2)?.as_str().parse().ok()?,
        });
    }

    let r2 = Regex::new(r"loss\s*=\s*([0-9.eE+-]+).*?epoch\s*=\s*([0-9.eE+-]+).*?step\s*=\s*(\d+)")
        .ok()?;
    if let Some(c) = r2.captures(line) {
        return Some(ParsedMetric {
            step: c.get(3)?.as_str().parse().ok()?,
            loss: c.get(1)?.as_str().parse().ok()?,
            epoch: c.get(2)?.as_str().parse().ok()?,
        });
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TeacherConfig;
    use crate::runs::{HubConfig, HubDatasetConfig};

    fn run_with_method(method: &str) -> (Run, LoraConfig, HubConfig) {
        let mut lora = LoraConfig::defaults();
        lora.method = method.to_string();
        let run = Run::new(
            "test".to_string(),
            "teacher/model".to_string(),
            "Qwen/Qwen2.5-7B-Instruct".to_string(),
            TeacherConfig::default(),
            lora.clone(),
            HubConfig::default(),
            HubDatasetConfig::default(),
        );
        (run, lora, HubConfig::default())
    }

    #[test]
    fn lora_yaml_has_no_quantization() {
        let (run, lora, hub) = run_with_method("lora");
        let yaml = train_yaml(&run, &lora, &hub, false, false).unwrap();
        assert!(yaml.contains("finetuning_type: lora"));
        assert!(!yaml.contains("quantization_bit"));
        assert!(!yaml.contains("quantization_method"));
    }

    #[test]
    fn legacy_peft_method_falls_through_to_lora() {
        // `peft` was removed from the UI but old saved runs may still carry it.
        // Resume must keep producing valid LoRA YAML.
        let (run, lora, hub) = run_with_method("peft");
        let yaml = train_yaml(&run, &lora, &hub, false, false).unwrap();
        assert!(yaml.contains("finetuning_type: lora"));
        assert!(!yaml.contains("quantization_bit"));
    }

    #[test]
    fn legacy_loftq_method_falls_through_to_qlora() {
        // `loftq` was removed from the UI (true LoftQ needs the offline init
        // script which we don't run). Old runs fall through to QLoRA YAML.
        let (run, lora, hub) = run_with_method("loftq");
        let yaml = train_yaml(&run, &lora, &hub, false, false).unwrap();
        assert!(yaml.contains("finetuning_type: lora"));
        assert!(yaml.contains("quantization_bit: 4"));
    }

    #[test]
    fn qlora_yaml_enables_four_bit_quantization() {
        let (run, lora, hub) = run_with_method("qlora");
        let yaml = train_yaml(&run, &lora, &hub, false, false).unwrap();
        assert!(yaml.contains("finetuning_type: lora"));
        assert!(yaml.contains("quantization_bit: 4"));
        assert!(yaml.contains("quantization_method: bnb"));
    }

    #[test]
    fn unsloth_yaml_enables_unsloth_acceleration() {
        let (run, lora, hub) = run_with_method("unsloth");
        let yaml = train_yaml(&run, &lora, &hub, false, false).unwrap();
        assert!(yaml.contains("finetuning_type: lora"));
        assert!(yaml.contains("use_unsloth: true"));
        assert!(!yaml.contains("quantization_bit"));
        assert!(!yaml.contains("quantization_method"));
    }

    #[test]
    fn full_yaml_uses_full_finetuning_without_lora_fields() {
        let (run, lora, hub) = run_with_method("full");
        let yaml = train_yaml(&run, &lora, &hub, false, false).unwrap();
        assert!(yaml.contains("finetuning_type: full"));
        assert!(!yaml.contains("lora_rank"));
        assert!(!yaml.contains("lora_target"));
    }

    #[test]
    fn freeze_yaml_uses_freeze_finetuning() {
        let (run, lora, hub) = run_with_method("freeze");
        let yaml = train_yaml(&run, &lora, &hub, false, false).unwrap();
        assert!(yaml.contains("finetuning_type: freeze"));
        assert!(yaml.contains("freeze_trainable_layers: 2"));
    }

    #[test]
    fn lora_variant_yaml_flags_are_emitted() {
        for (method, expected) in [
            ("dora", "use_dora: true"),
            ("loraplus", "loraplus_lr_ratio: 16.0"),
            ("pissa", "pissa_init: true"),
        ] {
            let (run, lora, hub) = run_with_method(method);
            let yaml = train_yaml(&run, &lora, &hub, false, false).unwrap();
            assert!(yaml.contains("finetuning_type: lora"));
            assert!(
                yaml.contains(expected),
                "{method} yaml missing {expected}:\n{yaml}"
            );
        }
    }

    #[test]
    fn pissa_yaml_emits_iter_and_convert() {
        let (run, lora, hub) = run_with_method("pissa");
        let yaml = train_yaml(&run, &lora, &hub, false, false).unwrap();
        assert!(
            yaml.contains("pissa_iter: 16"),
            "missing pissa_iter:\n{yaml}"
        );
        assert!(
            yaml.contains("pissa_convert: true"),
            "missing pissa_convert:\n{yaml}"
        );
    }

    #[test]
    fn galore_yaml_emits_required_optimizer_args() {
        let (run, lora, hub) = run_with_method("galore");
        let yaml = train_yaml(&run, &lora, &hub, false, false).unwrap();
        assert!(yaml.contains("finetuning_type: full"));
        assert!(yaml.contains("use_galore: true"));
        assert!(yaml.contains("galore_layerwise: true"));
        assert!(yaml.contains("galore_target: all"));
        assert!(yaml.contains("galore_rank: 128"));
        assert!(yaml.contains("galore_update_interval: 200"));
        assert!(yaml.contains("galore_scale: 2"));
        assert!(yaml.contains("pure_bf16: true"));
    }

    #[test]
    fn badam_yaml_emits_required_optimizer_args() {
        let (run, lora, hub) = run_with_method("badam");
        let yaml = train_yaml(&run, &lora, &hub, false, false).unwrap();
        assert!(yaml.contains("finetuning_type: full"));
        assert!(yaml.contains("use_badam: true"));
        assert!(yaml.contains("badam_mode: layer"));
        assert!(yaml.contains("badam_switch_mode: ascending"));
        assert!(yaml.contains("badam_switch_interval: 50"));
        assert!(yaml.contains("badam_verbose: 2"));
        assert!(yaml.contains("pure_bf16: true"));
    }

    #[test]
    fn train_only_multiple_hf_datasets_are_emitted_for_training() {
        let (mut run, lora, hub) = run_with_method("lora");
        run.hub_dataset.enabled = true;
        run.hub_dataset.train_only = true;
        run.hub_dataset.repo_id = "Zrald/dataset-one".to_string();
        run.hub_dataset.repo_ids = vec![
            "Zrald/dataset-one".to_string(),
            "Zrald/dataset-two".to_string(),
            "other/dataset-three".to_string(),
        ];

        let yaml = train_yaml(&run, &lora, &hub, false, false).unwrap();
        let base = format!("ft_{}", &run.id[..8]);
        assert!(
            yaml.contains(&format!("dataset: {base}_d0,{base}_d1,{base}_d2")),
            "multi-dataset train yaml did not use all dataset entries:\n{yaml}"
        );
        assert!(
            !yaml.contains("eval_dataset"),
            "HF train-only datasets should not reference local validation files:\n{yaml}"
        );

        let info = dataset_info_hf(&run);
        let parsed: serde_json::Value = serde_json::from_str(&info).unwrap();
        assert_eq!(
            parsed[format!("{base}_d0")]["hf_hub_url"],
            "Zrald/dataset-one"
        );
        assert_eq!(
            parsed[format!("{base}_d1")]["hf_hub_url"],
            "Zrald/dataset-two"
        );
        assert_eq!(
            parsed[format!("{base}_d2")]["hf_hub_url"],
            "other/dataset-three"
        );
    }
}
