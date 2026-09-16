pub mod badam;
pub mod common;
pub mod custom;
pub mod dora;
pub mod freeze;
pub mod full;
pub mod galore;
pub mod grpo;
pub mod lora;
pub mod loraplus;
pub mod pissa;
pub mod qlora;
pub mod unsloth;
pub mod zrald;
pub mod zrald_offline;

use crate::error::Result;
use crate::runs::{LoraConfig, Run};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandKind {
    LlamaFactory,
    Custom,
    Grpo,
    Zrald,
    ZraldOffline,
}

#[derive(Debug, Clone, Copy)]
pub struct LlamaFactoryYamlOptions {
    pub finetuning_type: &'static str,
    pub is_lora_family: bool,
    pub quantization_bit: Option<u8>,
    pub quantization_method: Option<&'static str>,
    pub use_unsloth: bool,
    pub use_dora: bool,
    pub loraplus_lr_ratio: Option<f32>,
    pub pissa_init: bool,
    pub pissa_iter: Option<u32>,
    pub pissa_convert: bool,
    pub freeze_trainable_layers: Option<i32>,
    pub use_galore: bool,
    pub galore_layerwise: bool,
    pub galore_target: Option<&'static str>,
    pub galore_rank: Option<u32>,
    pub galore_update_interval: Option<u32>,
    pub galore_scale: Option<f32>,
    pub use_badam: bool,
    pub badam_mode: Option<&'static str>,
    pub badam_switch_mode: Option<&'static str>,
    pub badam_switch_interval: Option<u32>,
    pub badam_verbose: Option<u8>,
    pub pure_bf16: bool,
}

impl LlamaFactoryYamlOptions {
    pub fn lora_like() -> Self {
        Self {
            finetuning_type: "lora",
            is_lora_family: true,
            quantization_bit: None,
            quantization_method: None,
            use_unsloth: false,
            use_dora: false,
            loraplus_lr_ratio: None,
            pissa_init: false,
            pissa_iter: None,
            pissa_convert: false,
            freeze_trainable_layers: None,
            use_galore: false,
            galore_layerwise: false,
            galore_target: None,
            galore_rank: None,
            galore_update_interval: None,
            galore_scale: None,
            use_badam: false,
            badam_mode: None,
            badam_switch_mode: None,
            badam_switch_interval: None,
            badam_verbose: None,
            pure_bf16: false,
        }
    }

    pub fn full_like() -> Self {
        Self {
            finetuning_type: "full",
            is_lora_family: false,
            ..Self::lora_like()
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MethodOptions {
    pub command_kind: CommandKind,
    pub yaml: LlamaFactoryYamlOptions,
    pub needs_bitsandbytes: bool,
    pub extra_optimizer_install: &'static str,
    pub needs_gpu_preflight: bool,
}

impl MethodOptions {
    pub fn lora_like(_key: &'static str) -> Self {
        Self {
            command_kind: CommandKind::LlamaFactory,
            yaml: LlamaFactoryYamlOptions::lora_like(),
            needs_bitsandbytes: false,
            extra_optimizer_install: "",
            needs_gpu_preflight: false,
        }
    }

    pub fn full_like(key: &'static str) -> Self {
        Self {
            yaml: LlamaFactoryYamlOptions::full_like(),
            ..Self::lora_like(key)
        }
    }
}

pub fn options(method: &str) -> MethodOptions {
    match method.trim().to_ascii_lowercase().as_str() {
        lora::KEY | "peft" => lora::options(),
        qlora::KEY | "loftq" => qlora::options(),
        unsloth::KEY => unsloth::options(),
        full::KEY => full::options(),
        freeze::KEY => freeze::options(),
        dora::KEY => dora::options(),
        loraplus::KEY => loraplus::options(),
        pissa::KEY => pissa::options(),
        galore::KEY => galore::options(),
        badam::KEY => badam::options(),
        grpo::KEY => grpo::options(),
        zrald::KEY => zrald::options(),
        zrald_offline::KEY => zrald_offline::options(),
        custom::KEY => custom::options(),
        _ => lora::options(),
    }
}

pub fn command_kind(method: &str) -> CommandKind {
    options(method).command_kind
}

pub fn yaml(method: &str) -> LlamaFactoryYamlOptions {
    options(method).yaml
}

pub fn is_zrald_method(method: &str) -> bool {
    matches!(
        command_kind(method),
        CommandKind::Zrald | CommandKind::ZraldOffline
    )
}

/// True when the method trains and saves a *complete* model (no PEFT adapter):
/// `full` and `freeze`. These run through LLaMA-Factory like LoRA, but the
/// output dir holds `model.safetensors[.index.json]` + `config.json` instead of
/// `adapter_model.safetensors` + `adapter_config.json`, so the post-training
/// existence check, Hub upload, and "merge" steps must be handled differently
/// (the model is already merged — there is nothing to merge into a base).
pub fn is_full_model_method(method: &str) -> bool {
    let opts = options(method);
    opts.command_kind == CommandKind::LlamaFactory && !opts.yaml.is_lora_family
}

pub fn build_train_cmd(
    method: &str,
    run: &Run,
    lora: &LoraConfig,
    hf_export: &str,
) -> Result<String> {
    match method.trim().to_ascii_lowercase().as_str() {
        lora::KEY | "peft" => lora::build_train_cmd(run, lora, hf_export),
        qlora::KEY | "loftq" => qlora::build_train_cmd(run, lora, hf_export),
        unsloth::KEY => unsloth::build_train_cmd(run, lora, hf_export),
        full::KEY => full::build_train_cmd(run, lora, hf_export),
        freeze::KEY => freeze::build_train_cmd(run, lora, hf_export),
        dora::KEY => dora::build_train_cmd(run, lora, hf_export),
        loraplus::KEY => loraplus::build_train_cmd(run, lora, hf_export),
        pissa::KEY => pissa::build_train_cmd(run, lora, hf_export),
        galore::KEY => galore::build_train_cmd(run, lora, hf_export),
        badam::KEY => badam::build_train_cmd(run, lora, hf_export),
        grpo::KEY => grpo::build_train_cmd(run, lora, hf_export),
        zrald::KEY => zrald::build_train_cmd(run, lora, hf_export),
        zrald_offline::KEY => zrald_offline::build_train_cmd(run, lora, hf_export),
        custom::KEY => custom::build_train_cmd(run, lora, hf_export),
        _ => lora::build_train_cmd(run, lora, hf_export),
    }
}

// ── ZRALD simulation ─────────────────────────────────────────────────────────
//
// The ZRALD methods emit a large Python program inside a shell heredoc. Both
// halves can fail in ways that only surface on a GPU host mid-run:
//
//   * a `__PLACEHOLDER__` the replacement table does not cover becomes a
//     NameError inside the trainer;
//   * a delimiter collision truncates the heredoc and the trainer runs half a
//     script;
//   * unbalanced shell quoting aborts the launch before Python starts.
//
// These tests generate the real command for a matrix of configurations and
// check all three, then hand the Python to a real interpreter and the shell to
// a real parser. Run with `cargo test`.
#[cfg(test)]
mod zrald_simulation {
    use super::*;
    use crate::runs::{LoraConfig, Run, RunStatus};
    use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
    use regex::Regex;
    use std::process::Command;

    fn base_lora() -> LoraConfig {
        LoraConfig {
            method: zrald::KEY.to_string(),
            r: 16,
            alpha: 32,
            dropout: 0.05,
            learning_rate: 2e-4,
            epochs: 1.0,
            batch_size: 1,
            gradient_accumulation: 4,
            cutoff_len: 2048,
            save_steps: 50,
            ..Default::default()
        }
    }

    fn run_with(method: &str, lora: LoraConfig) -> Run {
        let value = serde_json::json!({
            "id": "run-zrald-sim",
            "name": "zrald simulation",
            "createdAt": "2026-01-01T00:00:00Z",
            "updatedAt": "2026-01-01T00:00:00Z",
            "teacherModel": "Qwen/Qwen3.8-27B",
            "studentModel": "Qwen/Qwen2.5-7B-Instruct",
            "status": "pending",
            "qaTotal": 0,
            "qaKept": 0,
            "qaRejected": 0,
            "error": null,
            "logTail": "",
            "remoteDir": "/root/fine-tune/runs/run-zrald-sim",
            "localDir": "/tmp/runs/run-zrald-sim",
            "lora": lora,
            "teacherCfg": { "repoId": "Qwen/Qwen3.8-27B", "vllmPort": 44319 },
            "promptTemplate": "FOCUS TOPIC: {topic}\n\nSource:\n\"\"\"\n{chunk_text}\n\"\"\"",
            "topics": [
                { "topic": "biology", "promptTemplate": "BIO {topic} {chunk_text}" },
                { "topic": "chemistry" }
            ]
        });
        let mut run: Run = serde_json::from_value(value).expect("fixture must deserialize");
        run.lora.method = method.to_string();
        run
    }

    fn build(method: &str, lora: &LoraConfig) -> String {
        let run = run_with(method, lora.clone());
        build_train_cmd(method, &run, lora, "export HF_TOKEN=secret; ")
            .unwrap_or_else(|e| panic!("{method} build_train_cmd failed: {e}"))
    }

    /// True for this app's `__PLACEHOLDER__` convention specifically. Python
    /// dunders (`__init__`, `__name__`) are lowercase and must not be flagged.
    fn is_placeholder(token: &str) -> bool {
        token.len() > 4
            && token.starts_with("__")
            && token.ends_with("__")
            && token[2..token.len() - 2]
                .chars()
                .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
    }

    fn leftovers(text: &str) -> Vec<String> {
        text.split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
            .filter(|tok| is_placeholder(tok))
            .map(str::to_string)
            .collect()
    }

    /// Files the generated command materialises on the host, as (path, body).
    ///
    /// Two different mechanisms are in play, so both are decoded here:
    ///   * `zrald` emits `cat > … <<'PYEOF' … PYEOF` (a heredoc);
    ///   * `zrald_offline` emits base64 through `printf … | base64 -d`, because
    ///     its payload contains heredoc-hostile content.
    fn materialised_files(cmd: &str) -> Vec<(String, String)> {
        let mut files = Vec::new();

        if let Some(marker) = cmd.find("<<'PYEOF'\n") {
            let start = marker + "<<'PYEOF'\n".len();
            let rest = &cmd[start..];
            if let Some(end) = rest.find("\nPYEOF") {
                files.push(("<heredoc>".to_string(), rest[..end].to_string()));
            }
        }

        let re = Regex::new(
            r"printf '%s\\n' ((?:'[^']*'\s*)+)> '[^']*' && base64 -d '[^']*' > '([^']*)'",
        )
        .expect("valid regex");

        for caps in re.captures_iter(cmd) {
            let chunk_blob = &caps[1];
            let dest = caps[2].to_string();
            let mut encoded = String::new();
            for chunk in chunk_blob.split('\'').skip(1).step_by(2) {
                encoded.push_str(chunk);
            }
            match B64.decode(encoded.as_bytes()) {
                Ok(bytes) => files.push((dest, String::from_utf8_lossy(&bytes).to_string())),
                Err(e) => panic!("base64 payload for {dest} did not decode: {e}"),
            }
        }

        files
    }

    /// A working Python 3 launcher, or None when the host has none.
    fn python() -> Option<&'static str> {
        for candidate in ["python3", "python"] {
            let ok = Command::new(candidate)
                .arg("--version")
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false);
            if ok {
                return Some(candidate);
            }
        }
        None
    }

    /// A working bash, or None (Windows without Git Bash).
    fn bash() -> Option<&'static str> {
        for candidate in ["bash", "sh"] {
            let ok = Command::new(candidate)
                .arg("-c")
                .arg("true")
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false);
            if ok {
                return Some(candidate);
            }
        }
        None
    }

    /// Configurations that exercise every branch of the placeholder table.
    fn matrix() -> Vec<(&'static str, LoraConfig)> {
        let mut cases: Vec<(&'static str, LoraConfig)> = Vec::new();

        for method in [zrald::KEY, zrald_offline::KEY] {
            let mut defaults = base_lora();
            defaults.method = method.to_string();
            cases.push((method, defaults));

            // Reward endpoint pinned by the user instead of the teacher port.
            let mut pinned = base_lora();
            pinned.method = method.to_string();
            pinned.zrald_reward_endpoint = "http://10.0.0.5:9000/".to_string();
            pinned.zrald_reward_model = "my-reward-model".to_string();
            cases.push((method, pinned));

            // Degenerate numbers the UI can produce via a stray 0.
            let mut extremes = base_lora();
            extremes.method = method.to_string();
            extremes.zrald_train_questions = 0;
            extremes.zrald_benchmark_questions = 0;
            extremes.zrald_num_generations = 0;
            extremes.zrald_max_completion_tokens = 0;
            extremes.batch_size = 0;
            extremes.gradient_accumulation = 0;
            extremes.save_steps = 0;
            cases.push((method, extremes));

            let mut oversized = base_lora();
            oversized.method = method.to_string();
            oversized.zrald_train_questions = 100_000;
            oversized.zrald_benchmark_questions = 50_000;
            oversized.zrald_num_generations = 999;
            oversized.zrald_max_completion_tokens = 1_000_000;
            oversized.cutoff_len = 1024;
            cases.push((method, oversized));

            // Temperature and dataset source configured.
            let mut tuned = base_lora();
            tuned.method = method.to_string();
            tuned.zrald_reward_temperature = 0.7;
            tuned.zrald_dataset_source = "huggingface".to_string();
            cases.push((method, tuned));
        }

        cases
    }

    // ── Payload extraction sanity ───────────────────────────────────────────

    #[test]
    fn both_methods_materialise_their_payloads() {
        // Guards the extractor itself: if a method changes how it writes files,
        // every check below would silently pass on an empty set.
        let z = materialised_files(&build(zrald::KEY, &base_lora()));
        assert!(!z.is_empty(), "zrald should emit a heredoc payload");
        assert!(
            z.iter().any(|(_, body)| body.contains("GRPOTrainer")),
            "zrald payload should be the GRPO trainer"
        );

        let o = materialised_files(&build(zrald_offline::KEY, &base_lora()));
        assert!(
            o.len() >= 2,
            "zrald_offline should emit both a trainer and a runner, got {}",
            o.len()
        );
        assert!(
            o.iter().any(|(p, _)| p.ends_with("zrald_offline.py")),
            "expected the offline trainer file"
        );
        assert!(
            o.iter().any(|(p, _)| p.ends_with("zrald_offline_run.sh")),
            "expected the offline runner file"
        );
    }

    // ── Placeholder coverage ────────────────────────────────────────────────

    #[test]
    fn every_placeholder_is_substituted() {
        // A leftover `__FOO__` reaches Python as a bare name and raises
        // NameError mid-run, so this is the highest-value check here.
        for (method, lora) in matrix() {
            let cmd = build(method, &lora);
            for (path, body) in materialised_files(&cmd) {
                let found = leftovers(&body);
                assert!(
                    found.is_empty(),
                    "[{method}] {path} has unsubstituted placeholders {found:?}"
                );
            }
        }
    }

    #[test]
    fn no_placeholder_survives_anywhere_in_the_command() {
        for (method, lora) in matrix() {
            let cmd = build(method, &lora);
            let found = leftovers(&cmd);
            assert!(
                found.is_empty(),
                "[{method}] placeholders leaked into the shell command: {found:?}"
            );
        }
    }

    // ── Heredoc integrity (zrald) ───────────────────────────────────────────

    #[test]
    fn heredoc_delimiter_is_well_formed() {
        let cmd = build(zrald::KEY, &base_lora());
        let body = &materialised_files(&cmd)[0].1;
        assert!(!body.contains("PYEOF"), "payload contains the heredoc delimiter");
        assert_eq!(
            cmd.matches("PYEOF").count(),
            2,
            "expected exactly one open and one close delimiter"
        );
    }

    #[test]
    fn heredoc_is_quoted_so_the_body_is_not_expanded() {
        // `<<'PYEOF'` (quoted) keeps `$` and backticks literal. An unquoted
        // delimiter would let the shell expand them and corrupt the script.
        let cmd = build(zrald::KEY, &base_lora());
        assert!(cmd.contains("<<'PYEOF'"), "heredoc delimiter must stay quoted");
    }

    // ── Functional checks against real interpreters ─────────────────────────

    #[test]
    fn generated_python_compiles() {
        let Some(py_bin) = python() else {
            eprintln!("skipping: no python3/python on PATH");
            return;
        };
        let dir = std::env::temp_dir().join("zrald_sim_py");
        std::fs::create_dir_all(&dir).expect("temp dir");

        let mut checked = 0usize;
        for (idx, (method, lora)) in matrix().into_iter().enumerate() {
            let cmd = build(method, &lora);
            for (path, body) in materialised_files(&cmd) {
                if !path.ends_with(".py") {
                    continue;
                }
                let file = dir.join(format!("{idx}_{method}_{}", checked));
                std::fs::write(&file, &body).expect("write script");
                checked += 1;

                let out = Command::new(py_bin)
                    .arg("-m")
                    .arg("py_compile")
                    .arg(&file)
                    .output()
                    .expect("run py_compile");
                assert!(
                    out.status.success(),
                    "[{method}] {path} does not compile:\n{}\n--- script ---\n{}",
                    String::from_utf8_lossy(&out.stderr),
                    body
                );
            }
        }
        assert!(checked > 0, "no Python payloads were checked");
    }

    #[test]
    fn generated_runner_scripts_parse() {
        let Some(sh) = bash() else {
            eprintln!("skipping: no bash/sh on PATH");
            return;
        };
        let dir = std::env::temp_dir().join("zrald_sim_sh");
        std::fs::create_dir_all(&dir).expect("temp dir");

        let mut checked = 0usize;
        for (idx, (method, lora)) in matrix().into_iter().enumerate() {
            let cmd = build(method, &lora);
            for (path, body) in materialised_files(&cmd) {
                if !path.ends_with(".sh") {
                    continue;
                }
                let file = dir.join(format!("{idx}_{method}_{checked}.sh"));
                std::fs::write(&file, &body).expect("write runner");
                checked += 1;

                let out = Command::new(sh)
                    .arg("-n")
                    .arg(&file)
                    .output()
                    .expect("run bash -n");
                assert!(
                    out.status.success(),
                    "[{method}] {path} does not parse:\n{}",
                    String::from_utf8_lossy(&out.stderr)
                );
            }
        }
        assert!(checked > 0, "no shell payloads were checked");
    }

    #[test]
    fn generated_command_is_shell_parseable() {
        let Some(sh) = bash() else {
            eprintln!("skipping: no bash/sh on PATH");
            return;
        };
        let dir = std::env::temp_dir().join("zrald_sim_cmd");
        std::fs::create_dir_all(&dir).expect("temp dir");

        for (idx, (method, lora)) in matrix().into_iter().enumerate() {
            let cmd = build(method, &lora);
            let path = dir.join(format!("{idx}_{method}.sh"));
            std::fs::write(&path, &cmd).expect("write command");

            // `-n` parses without executing — catches unbalanced quotes and
            // truncated heredocs without needing a GPU host.
            let out = Command::new(sh)
                .arg("-n")
                .arg(&path)
                .output()
                .expect("run bash -n");
            assert!(
                out.status.success(),
                "[{method}] generated shell command does not parse:\n{}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
    }

    // ── Clamping and defaults ───────────────────────────────────────────────

    #[test]
    fn generation_count_is_clamped_to_a_usable_range() {
        // GRPO needs at least 2 candidates to compute a relative advantage, and
        // an unbounded value would OOM the sampler.
        let mut low = base_lora();
        low.zrald_num_generations = 0;
        assert!(
            payload_contains(zrald::KEY, &low, "NUM_GENERATIONS = 2"),
            "0 must clamp up to 2"
        );

        let mut high = base_lora();
        high.zrald_num_generations = 999;
        assert!(
            payload_contains(zrald::KEY, &high, "NUM_GENERATIONS = 8"),
            "999 must clamp down to 8"
        );

        let mut ok = base_lora();
        ok.zrald_num_generations = 4;
        assert!(
            payload_contains(zrald::KEY, &ok, "NUM_GENERATIONS = 4"),
            "valid value must pass through"
        );
    }

    #[test]
    fn completion_budget_is_clamped_to_the_sequence_length() {
        let mut zero = base_lora();
        zero.zrald_max_completion_tokens = 0;
        assert!(
            payload_contains(zrald::KEY, &zero, "MAX_COMPLETION = 64"),
            "0 must clamp up to 64"
        );

        let mut over = base_lora();
        over.zrald_max_completion_tokens = 1_000_000;
        over.cutoff_len = 2048;
        assert!(
            payload_contains(zrald::KEY, &over, "MAX_COMPLETION = 2048"),
            "completion budget must not exceed the sequence length"
        );
    }

    #[test]
    fn benchmark_never_exceeds_the_training_set() {
        let mut lora = base_lora();
        lora.zrald_train_questions = 10;
        lora.zrald_benchmark_questions = 500;
        let cmd = build(zrald::KEY, &lora);
        let body = &materialised_files(&cmd)[0].1;
        assert!(body.contains("TRAIN_LIMIT = 10"));
        assert!(
            body.contains("BENCHMARK_N = 10"),
            "held-out benchmark set cannot be larger than the training set"
        );
    }

    #[test]
    fn zero_sized_batches_are_raised_to_one() {
        let mut lora = base_lora();
        lora.batch_size = 0;
        lora.gradient_accumulation = 0;
        lora.save_steps = 0;
        let cmd = build(zrald::KEY, &lora);
        let body = &materialised_files(&cmd)[0].1;
        assert!(
            body.contains("PER_DEVICE_BS = 1"),
            "batch size 0 would divide by zero"
        );
        assert!(body.contains("GRAD_ACCUM = 1"));
        assert!(body.contains("SAVE_STEPS = 1"));
    }

    #[test]
    fn reward_endpoint_falls_back_to_the_teacher_port() {
        let lora = base_lora();
        assert!(lora.zrald_reward_endpoint.trim().is_empty());
        assert!(
            payload_contains(zrald::KEY, &lora, r#"REWARD_ENDPOINT = "http://127.0.0.1:44319""#),
            "empty endpoint must default to the deployed teacher port"
        );
    }

    #[test]
    fn explicit_reward_endpoint_is_honoured() {
        let mut lora = base_lora();
        lora.zrald_reward_endpoint = "http://10.0.0.5:9000/".to_string();
        assert!(
            payload_contains(zrald::KEY, &lora, r#"REWARD_ENDPOINT = "http://10.0.0.5:9000/""#),
            "user-supplied reward endpoint must win over the teacher port"
        );
    }

    #[test]
    fn offline_runner_only_boots_a_local_teacher_when_needed() {
        // With an external reward endpoint the runner must NOT spin up its own
        // vLLM teacher — that would fight the deployed one for VRAM.
        let local = materialised_files(&build(zrald_offline::KEY, &base_lora()));
        let runner = local
            .iter()
            .find(|(p, _)| p.ends_with(".sh"))
            .map(|(_, b)| b.clone())
            .expect("runner");
        assert!(
            runner.contains("LOCAL_REWARD_TEACHER=1"),
            "no endpoint configured should boot a local teacher"
        );

        let mut pinned = base_lora();
        pinned.zrald_reward_endpoint = "http://10.0.0.5:9000/".to_string();
        let external = materialised_files(&build(zrald_offline::KEY, &pinned));
        let runner = external
            .iter()
            .find(|(p, _)| p.ends_with(".sh"))
            .map(|(_, b)| b.clone())
            .expect("runner");
        assert!(
            runner.contains("LOCAL_REWARD_TEACHER=0"),
            "an external endpoint must suppress the local teacher"
        );
    }

    #[test]
    fn gpt_oss_students_skip_4bit_loading() {
        // gpt-oss ships MXFP4 weights that cannot be re-quantized to 4-bit.
        let lora = base_lora();

        let mut oss = run_with(zrald::KEY, lora.clone());
        oss.student_model = "openai/gpt-oss-20b".to_string();
        let cmd = build_train_cmd(zrald::KEY, &oss, &lora, "").expect("build");
        assert!(
            materialised_files(&cmd)[0].1.contains("LOAD_IN_4BIT = False"),
            "gpt-oss must not load in 4-bit"
        );

        let mut dense = run_with(zrald::KEY, lora.clone());
        dense.student_model = "Qwen/Qwen2.5-7B-Instruct".to_string();
        let cmd = build_train_cmd(zrald::KEY, &dense, &lora, "").expect("build");
        assert!(
            materialised_files(&cmd)[0].1.contains("LOAD_IN_4BIT = True"),
            "dense students load in 4-bit"
        );
    }

    // ── Environment isolation ───────────────────────────────────────────────

    #[test]
    fn trainer_uses_an_isolated_venv() {
        // A `--force-reinstall torch` in the container's system python breaks
        // the resident vLLM teacher's prebuilt C extensions, so the GRPO stack
        // must live in its own venv.
        for (method, lora) in matrix() {
            let cmd = build(method, &lora);
            assert!(cmd.contains(".zrald_venv"), "[{method}] expected an isolated venv");
            assert!(
                cmd.contains("UNSLOTH_IS_ROCM=1"),
                "[{method}] ROCm unsloth flag must be exported"
            );
            for (_, body) in materialised_files(&cmd) {
                if body.contains("unsloth") {
                    assert!(
                        body.contains(".zrald_venv") || cmd.contains(".zrald_venv"),
                        "[{method}] unsloth must run from the venv"
                    );
                }
            }
        }
    }

    #[test]
    fn shell_quoting_is_balanced() {
        // Only the shell-visible portion is counted. The heredoc body is inside
        // `<<'PYEOF'`, so apostrophes there are literal data and must not be
        // treated as quote delimiters.
        for (method, lora) in matrix() {
            let cmd = build(method, &lora);
            let shell_visible = match cmd.find("<<'PYEOF'\n") {
                Some(start) => match cmd[start..].find("\nPYEOF") {
                    Some(end) => format!("{}{}", &cmd[..start], &cmd[start + end..]),
                    None => cmd.clone(),
                },
                None => cmd.clone(),
            };
            assert_eq!(
                shell_visible.matches('\'').count() % 2,
                0,
                "[{method}] odd number of single quotes — a path is unquoted"
            );
        }
    }

    #[test]
    fn remote_paths_are_quoted() {
        for (method, lora) in matrix() {
            let cmd = build(method, &lora);
            assert!(
                cmd.contains("'/root/fine-tune/runs/run-zrald-sim'"),
                "[{method}] remote dir must be shell-quoted"
            );
        }
    }

    /// True when any materialised payload contains `needle`.
    fn payload_contains(method: &str, lora: &LoraConfig, needle: &str) -> bool {
        let cmd = build(method, lora);
        materialised_files(&cmd)
            .iter()
            .any(|(_, body)| body.contains(needle))
    }

    // ── Method classification ───────────────────────────────────────────────

    #[test]
    fn zrald_methods_are_classified_together() {
        assert!(is_zrald_method("zrald"));
        assert!(is_zrald_method("zrald_offline"));
        assert!(is_zrald_method("ZRALD"));
        assert!(is_zrald_method("  Zrald_Offline  "));
        assert!(!is_zrald_method("lora"));
        assert!(!is_zrald_method("grpo"));
        assert!(!is_zrald_method("qlora"));
        assert!(!is_zrald_method(""));
    }

    #[test]
    fn both_zrald_methods_are_dispatchable() {
        let expected = [(zrald::KEY, "zrald_train.py"), (zrald_offline::KEY, "zrald_offline.py")];
        for (method, script) in expected {
            let lora = base_lora();
            let run = run_with(method, lora.clone());
            let cmd = build_train_cmd(method, &run, &lora, "").expect("dispatch");
            assert!(!cmd.trim().is_empty(), "{method} produced an empty command");
            assert!(cmd.contains(script), "{method} must write {script}");
        }
    }

    #[test]
    fn run_status_serializes_as_expected() {
        // The simulation fixture depends on this spelling; if it changes, the
        // fixture silently stops exercising the intended path.
        let json = serde_json::to_string(&RunStatus::Training).expect("serialize");
        assert_eq!(json, "\"training\"");
    }

    /// Dump every method's real train command to `FT_DUMP_DIR`, one file each.
    ///
    /// Used to drive the on-GPU test matrix: the commands executed on the
    /// droplet are byte-for-byte what the app would send, so a method that
    /// works here works in the app.
    #[test]
    #[ignore = "writes files; run explicitly with FT_DUMP_DIR set"]
    fn dump_all_method_commands() {
        let Some(dir) = std::env::var("FT_DUMP_DIR").ok().filter(|d| !d.trim().is_empty()) else {
            eprintln!("skipping: set FT_DUMP_DIR");
            return;
        };
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir).expect("dump dir");

        let methods = [
            "lora", "qlora", "dora", "loraplus", "pissa", "unsloth",
            "full", "freeze", "galore", "badam",
        ];
        for method in methods {
            let mut lora = base_lora();
            lora.method = method.to_string();
            let run = run_with(method, lora.clone());
            match build_train_cmd(method, &run, &lora, "") {
                Ok(cmd) => {
                    let path = dir.join(format!("{method}.sh"));
                    std::fs::write(&path, &cmd).expect("write");
                    println!("wrote {}", path.display());
                }
                Err(e) => println!("{method}: build failed: {e}"),
            }
        }
    }

    /// Every non-RL method must produce a command that a real shell parses.
    #[test]
    fn every_training_method_emits_a_parseable_command() {
        for method in [
            "lora", "qlora", "dora", "loraplus", "pissa", "unsloth",
            "full", "freeze", "galore", "badam", "custom",
        ] {
            let mut lora = base_lora();
            lora.method = method.to_string();
            if method == "custom" {
                // `custom` refuses to build without at least one command.
                lora.custom_commands = vec!["llamafactory-cli train {train_yaml}".to_string()];
                lora.custom_method_name = "my-method".to_string();
            }
            let run = run_with(method, lora.clone());
            let cmd = build_train_cmd(method, &run, &lora, "").unwrap_or_else(|e| {
                panic!("{method}: build_train_cmd failed: {e}")
            });
            assert!(!cmd.trim().is_empty(), "{method} produced an empty command");
            assert_eq!(
                cmd.matches('\'').count() % 2,
                0,
                "{method}: unbalanced single quotes would break the wrapper"
            );
            // `custom` runs arbitrary user commands, so it has no train.yaml.
            if method != "custom" {
                assert!(
                    cmd.contains("train.yaml") || cmd.contains("llamafactory"),
                    "{method}: expected a LLaMA-Factory invocation"
                );
            }
        }
    }

    /// Every method must route through the isolated trainer venv.
    #[test]
    fn every_llamafactory_method_uses_the_isolated_venv() {
        for method in [
            "lora", "qlora", "dora", "loraplus", "pissa", "unsloth",
            "full", "freeze", "galore", "badam",
        ] {
            let mut lora = base_lora();
            lora.method = method.to_string();
            let run = run_with(method, lora.clone());
            let cmd = build_train_cmd(method, &run, &lora, "").expect("build");
            assert!(
                cmd.contains(".lf_venv"),
                "{method} must not install into the shared container env"
            );
            assert!(
                !cmd.contains("which llamafactory-cli"),
                "{method} still probes the system CLI instead of the venv"
            );
        }
    }
}
