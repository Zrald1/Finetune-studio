use crate::config::{runs_dir, TeacherConfig};
use crate::error::{AppError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::fs;
use ulid::Ulid;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Pending,
    TeacherLoading,
    GeneratingDataset,
    DatasetReady,
    Training,
    Done,
    Failed,
    Cancelled,
}

impl RunStatus {
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Done | Self::Failed | Self::Cancelled)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default, rename_all = "camelCase")]
pub struct LoraConfig {
    #[serde(default = "default_training_method")]
    pub method: String,
    #[serde(default)]
    pub custom_method_name: String,
    #[serde(default)]
    pub custom_commands: Vec<String>,
    #[serde(default = "default_unsloth_backend")]
    pub unsloth_backend: String,
    #[serde(default)]
    pub zrald_reward_endpoint: String,
    #[serde(default)]
    pub zrald_reward_model: String,
    #[serde(default = "default_zrald_train_questions")]
    pub zrald_train_questions: u32,
    #[serde(default = "default_zrald_benchmark_questions")]
    pub zrald_benchmark_questions: u32,
    #[serde(default = "default_zrald_num_generations")]
    pub zrald_num_generations: u32,
    #[serde(default)]
    pub zrald_reward_temperature: f32,
    #[serde(default = "default_zrald_max_completion_tokens")]
    pub zrald_max_completion_tokens: u32,
    #[serde(default = "default_zrald_dataset_source")]
    pub zrald_dataset_source: String, // "generate" | "huggingface"
    pub r: u32,
    pub alpha: u32,
    pub dropout: f32,
    pub learning_rate: f32,
    pub epochs: f32,
    pub batch_size: u32,
    pub gradient_accumulation: u32,
    pub cutoff_len: u32,
    #[serde(default = "default_save_steps")]
    pub save_steps: u32,
}

fn default_save_steps() -> u32 {
    100
}

fn default_training_method() -> String {
    "lora".to_string()
}

fn default_unsloth_backend() -> String {
    "cuda".to_string()
}

fn default_zrald_train_questions() -> u32 {
    1000
}

fn default_zrald_benchmark_questions() -> u32 {
    100
}

fn default_zrald_num_generations() -> u32 {
    4
}

fn default_zrald_max_completion_tokens() -> u32 {
    768
}

fn default_zrald_dataset_source() -> String {
    "generate".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default, rename_all = "camelCase")]
pub struct HubConfig {
    pub enabled: bool,
    pub model_id: String, // e.g. "Zrald/geodetic-lora-qwen-7b"
    pub private: bool,
    #[serde(default = "default_hub_strategy")]
    pub strategy: String, // "every_save" | "end" | "checkpoint"
    /// After training, merge LoRA into base weights and upload the full
    /// model to `merged_model_id` (or `<model_id>-merged` if blank).
    #[serde(default)]
    pub auto_merge: bool,
    #[serde(default)]
    pub merged_model_id: String,
    /// Destroy the GPU droplet after training completes (after merge+push if auto_merge is enabled).
    #[serde(default)]
    pub auto_destroy: bool,
    /// After merge, also convert to GGUF and upload to a dedicated repo for Ollama/llama.cpp.
    #[serde(default)]
    pub auto_convert_gguf: bool,
    /// Quantization type for GGUF conversion: "F16", "Q4_K_M", "Q5_K_M", "Q8_0".
    #[serde(default = "default_gguf_quantization")]
    pub gguf_quantization: String,
    /// Target GGUF repo ID (defaults to `<model_id>-gguf` if empty).
    #[serde(default)]
    pub gguf_repo_id: String,
}

fn default_gguf_quantization() -> String {
    "Q4_K_M".to_string()
}

fn default_hub_strategy() -> String {
    "every_save".to_string()
}

/// Auto-publish the generated Q&A dataset to a Hugging Face *dataset* repo
/// every N accepted pairs (checkpoint upload), and at end. Optionally seed
/// the run from an existing dataset repo to resume work without re-asking
/// the teacher about chunks we've already covered.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default, rename_all = "camelCase")]
pub struct HubDatasetConfig {
    /// Master enable flag. If false, all dataset hub features are skipped.
    pub enabled: bool,
    /// Target HF dataset repo, e.g. "Zrald/ge-reviewer-qa".
    pub repo_id: String,
    /// Train-Only mode: extra HF dataset repos to interleave alongside
    /// `repo_id`. When non-empty, LLaMA-Factory receives a comma-separated
    /// `dataset` field built from these (plus `repo_id` if it isn't already
    /// in the list). All datasets share the format/columns configured below.
    #[serde(default)]
    pub repo_ids: Vec<String>,
    /// Push to a private repo (default true).
    #[serde(default = "default_dataset_private")]
    pub private: bool,
    /// Push after every N accepted Q&A pairs. 0 / None → only at end.
    #[serde(default = "default_every_n")]
    pub every_n: u32,
    /// If non-empty, download this HF dataset repo at the start of the run
    /// and seed the local JSONL from it — chunks whose `source_chunk_id`
    /// is already present are skipped during generation. Defaults to
    /// `repo_id` when blank but `enabled` is true.
    #[serde(default)]
    pub resume_from: String,
    /// Skip dataset generation and train directly from Hugging Face dataset.
    #[serde(default)]
    pub train_only: bool,
    /// Dataset format: "sharegpt" or "alpaca".
    #[serde(default = "default_dataset_format")]
    pub dataset_format: String,
    /// Optional column mapping.
    #[serde(default)]
    pub dataset_columns: HashMap<String, String>,
}

fn default_dataset_format() -> String {
    "sharegpt".to_string()
}

fn default_dataset_private() -> bool {
    true
}
fn default_every_n() -> u32 {
    100
}

impl LoraConfig {
    #[cfg(test)]
    pub fn defaults() -> Self {
        Self {
            method: default_training_method(),
            custom_method_name: String::new(),
            custom_commands: Vec::new(),
            unsloth_backend: default_unsloth_backend(),
            zrald_reward_endpoint: String::new(),
            zrald_reward_model: String::new(),
            zrald_train_questions: default_zrald_train_questions(),
            zrald_benchmark_questions: default_zrald_benchmark_questions(),
            zrald_num_generations: default_zrald_num_generations(),
            zrald_reward_temperature: 0.0,
            zrald_max_completion_tokens: default_zrald_max_completion_tokens(),
            zrald_dataset_source: default_zrald_dataset_source(),
            r: 16,
            alpha: 32,
            dropout: 0.05,
            learning_rate: 5e-5,
            epochs: 2.0,
            batch_size: 4,
            gradient_accumulation: 4,
            cutoff_len: 4096,
            save_steps: 100,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TopicTarget {
    pub topic: String,
    #[serde(default)]
    pub total_questions: Option<u32>,
    #[serde(default)]
    pub tag: Option<String>,
    #[serde(default)]
    pub prompt_template: Option<String>,
    #[serde(default)]
    pub embedder_index: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Run {
    pub id: String,
    pub name: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub teacher_model: String,
    pub student_model: String,
    pub status: RunStatus,
    pub qa_total: u64,
    pub qa_kept: u64,
    pub qa_rejected: u64,
    #[serde(default)]
    pub train_loss_history: Vec<TrainPoint>,
    pub error: Option<String>,
    pub log_tail: String, // last ~8KB of streamed output, for the UI
    pub remote_dir: String,
    pub local_dir: String,
    pub lora: LoraConfig,
    pub teacher_cfg: TeacherConfig,
    #[serde(default)]
    pub hub: HubConfig,
    #[serde(default)]
    pub hub_dataset: HubDatasetConfig,
    /// True once the dataset jsonl has been generated and uploaded; resume can skip
    /// straight to training if true.
    #[serde(default)]
    pub dataset_ready: bool,
    /// Last completed step the trainer reported. Used as a heuristic resume marker.
    #[serde(default)]
    pub last_train_step: u32,
    /// Per-topic Q&A pair counts: maps topic label → number of kept pairs.
    /// Populated at the end of each topic generation loop.
    #[serde(default)]
    pub topic_stats: HashMap<String, u64>,
    /// Auto-destroy droplet after training completes (merge + upload).
    #[serde(default)]
    pub auto_destroy: bool,
    #[serde(default)]
    pub prompt_template: Option<String>,
    #[serde(default)]
    pub topics: Vec<TopicTarget>,
    #[serde(default)]
    pub dataset_format: Option<crate::generator::DatasetFormat>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainPoint {
    pub step: u32,
    pub loss: f32,
    pub epoch: f32,
}

impl Run {
    pub fn new(
        name: String,
        teacher_model: String,
        student_model: String,
        teacher_cfg: TeacherConfig,
        lora: LoraConfig,
        hub: HubConfig,
        hub_dataset: HubDatasetConfig,
    ) -> Self {
        let id = Ulid::new().to_string();
        let remote_dir = format!("/root/fine-tune/runs/{}", id);
        let local_dir = runs_dir()
            .map(|p| p.join(&id))
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|_| id.clone());
        Self {
            id,
            name,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            teacher_model,
            student_model,
            status: RunStatus::Pending,
            qa_total: 0,
            qa_kept: 0,
            qa_rejected: 0,
            train_loss_history: vec![],
            error: None,
            log_tail: String::new(),
            remote_dir,
            local_dir,
            lora,
            teacher_cfg,
            hub,
            hub_dataset,
            dataset_ready: false,
            last_train_step: 0,
            topic_stats: HashMap::new(),
            auto_destroy: false,
            prompt_template: None,
            topics: vec![],
            dataset_format: None,
        }
    }
}

fn run_path(id: &str) -> Result<PathBuf> {
    Ok(runs_dir()?.join(format!("{}.json", id)))
}

pub async fn save(run: &Run) -> Result<()> {
    fs::create_dir_all(runs_dir()?).await?;
    let txt = serde_json::to_string_pretty(run)?;
    fs::write(run_path(&run.id)?, txt).await?;
    Ok(())
}

pub async fn load(id: &str) -> Result<Run> {
    let path = run_path(id)?;
    if !path.exists() {
        return Err(AppError::NotFound(format!("run {}", id)));
    }
    let txt = fs::read_to_string(path).await?;
    let r: Run = serde_json::from_str(&txt)?;
    Ok(r)
}

pub async fn list() -> Result<Vec<Run>> {
    let dir = runs_dir()?;
    fs::create_dir_all(&dir).await?;
    let mut out = vec![];
    let mut rd = fs::read_dir(&dir).await?;
    while let Some(entry) = rd.next_entry().await? {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        if let Ok(txt) = fs::read_to_string(&path).await {
            if let Ok(r) = serde_json::from_str::<Run>(&txt) {
                out.push(r);
            }
        }
    }
    out.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    Ok(out)
}

pub fn append_log_tail(run: &mut Run, chunk: &str) {
    run.log_tail.push_str(chunk);
    // keep tail bounded to ~16 KB
    if run.log_tail.len() > 16 * 1024 {
        let mut cut = run.log_tail.len() - 16 * 1024;
        // String::drain panics if `cut` falls inside a multi-byte UTF-8 char
        // (training logs often contain box-drawing chars and ANSI escapes).
        while cut < run.log_tail.len() && !run.log_tail.is_char_boundary(cut) {
            cut += 1;
        }
        run.log_tail.drain(..cut);
    }
    run.updated_at = Utc::now();
}

/// On-disk live log path for a run. Lives next to the dataset/checkpoints so
/// the UI can re-hydrate the full history when the user switches tabs or
/// reopens the app, instead of relying on the 16 KB `log_tail` field.
pub fn run_log_path(run_id: &str, local_dir: &str) -> PathBuf {
    let dir = PathBuf::from(local_dir);
    if !dir.as_os_str().is_empty() {
        return dir.join("live.log");
    }
    runs_dir_sync().join(run_id).join("live.log")
}

fn runs_dir_sync() -> PathBuf {
    runs_dir().unwrap_or_else(|_| PathBuf::from("."))
}

/// Append a chunk to the per-run live log file. Uses sync I/O on purpose so
/// it can be called from non-async event emitters; failures are swallowed so
/// a missing directory never disrupts the pipeline itself.
pub fn append_log_file(run_id: &str, local_dir: &str, chunk: &str) {
    if chunk.is_empty() {
        return;
    }
    use std::fs::OpenOptions;
    use std::io::Write;
    let path = run_log_path(run_id, local_dir);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&path) {
        let _ = f.write_all(chunk.as_bytes());
    }
}

/// Read the last `max_bytes` bytes of a run's live log. Returns an empty
/// string when the file doesn't exist yet (a brand-new run).
pub async fn read_log_tail(run_id: &str, max_bytes: usize) -> Result<String> {
    let run = load(run_id).await?;
    let path = run_log_path(&run.id, &run.local_dir);
    if !path.exists() {
        return Ok(run.log_tail);
    }
    let bytes = fs::read(&path).await.unwrap_or_default();
    let start = bytes.len().saturating_sub(max_bytes.max(1));
    // Avoid splitting a UTF-8 sequence mid-byte.
    let mut s = start;
    while s < bytes.len() && (bytes[s] & 0b1100_0000) == 0b1000_0000 {
        s += 1;
    }
    Ok(String::from_utf8_lossy(&bytes[s..]).into_owned())
}
