#![allow(dead_code)]

use crate::config::{AppConfig, DockerConfig, EmbedderConfig, TeacherConfig};
use crate::error::{AppError, Result};
use crate::generator::{self, GeneratedPair, GeneratorConfig};
use crate::llamafactory;
use crate::runs::{self, HubConfig, HubDatasetConfig, LoraConfig, Run, RunStatus, TrainPoint};
use crate::ssh::{SshSession, StreamChunk};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tauri::{AppHandle, Emitter};
use tokio::fs;
use tokio::sync::mpsc;

pub const TEACHER_VRAM_WITH_SEMANTIC_EMBEDDER: f32 = 0.80;

pub fn is_protected_semantic_embedder(index: usize, embedder: &EmbedderConfig) -> bool {
    index == 0 || embedder.persistent
}

pub fn cap_teacher_vram_for_semantic_embedder(teacher: &mut TeacherConfig) -> Option<(f32, f32)> {
    if teacher.gpu_memory_utilization > TEACHER_VRAM_WITH_SEMANTIC_EMBEDDER {
        let previous = teacher.gpu_memory_utilization;
        teacher.gpu_memory_utilization = TEACHER_VRAM_WITH_SEMANTIC_EMBEDDER;
        Some((previous, teacher.gpu_memory_utilization))
    } else {
        None
    }
}

pub(crate) fn parse_arg_value(cmd: &str, arg: &str) -> Option<String> {
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    for i in 0..parts.len() {
        if parts[i] == arg {
            if i + 1 < parts.len() {
                return Some(
                    parts[i + 1]
                        .trim_matches('\'')
                        .trim_matches('"')
                        .to_string(),
                );
            }
        } else if parts[i].starts_with(arg) && parts[i].contains('=') {
            let subparts: Vec<&str> = parts[i].split('=').collect();
            if subparts.len() > 1 {
                return Some(subparts[1].trim_matches('\'').trim_matches('"').to_string());
            }
        }
    }
    None
}

pub(crate) fn extract_model_and_port(
    custom_cmd: &str,
    default_model: &str,
    default_port: u16,
) -> (String, u16) {
    let port = parse_arg_value(custom_cmd, "--port")
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(default_port);

    let model = parse_arg_value(custom_cmd, "--model")
        .or_else(|| {
            // Find positional argument after "serve" or "api_server"
            let parts: Vec<&str> = custom_cmd.split_whitespace().collect();
            if let Some(idx) = parts
                .iter()
                .position(|&s| s == "serve" || s.contains("api_server"))
            {
                for i in (idx + 1)..parts.len() {
                    let part = parts[i];
                    if !part.starts_with('-') && !part.contains('=') && part != "\\" {
                        return Some(part.trim_matches('\'').trim_matches('"').to_string());
                    }
                }
            }
            None
        })
        .unwrap_or_else(|| default_model.to_string());

    (model, port)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TopicTarget {
    pub topic: String,
    #[serde(default)]
    pub total_questions: Option<u32>,
    /// Optional payload-tag filter for Qdrant search. When set, the topic's
    /// candidate retrieval restricts to points whose ingest tag matches this
    /// string. Pair this with the per-batch tag in the Knowledge Base upload
    /// card to scope a topic to a curated subset.
    #[serde(default)]
    pub tag: Option<String>,
    /// Optional per-topic prompt template. Overrides `RunConfig.prompt_template`
    /// for just this topic's generation pass. `{topic}` and `{chunk_text}` are
    /// still substituted. Blank / `None` → fall back to the global template.
    #[serde(default)]
    pub prompt_template: Option<String>,
    /// Index into `cfg.embedders` for the embedder to use when retrieving
    /// candidate chunks for this topic. Each embedder owns its own Qdrant
    /// collection so picking the right one is essential.
    #[serde(default)]
    pub embedder_index: Option<usize>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum TeacherProvider {
    Vllm,
    Featherless,
}

impl Default for TeacherProvider {
    fn default() -> Self {
        Self::Vllm
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunConfig {
    pub name: String,
    pub teacher: TeacherConfig,
    pub student_model: String,
    pub lora: LoraConfig,
    pub prompt_template: Option<String>,
    pub max_pairs_per_chunk: u32,
    pub concurrency: u32,
    pub max_chunks: Option<u32>, // for smoke tests; None = process all
    /// Which inference backend supplies the Teacher. Default is the existing
    /// GPU/vLLM path (boots a local vLLM on the droplet). `Featherless` swaps
    /// the teacher source for the hosted Featherless `/v1/chat/completions`
    /// endpoint, so no GPU teacher boot is required.
    #[serde(default)]
    pub teacher_provider: TeacherProvider,
    /// Model id passed to the Featherless API when `teacher_provider` is
    /// `featherless`. Ignored otherwise.
    #[serde(default)]
    pub featherless_model: Option<String>,
    /// Single-topic mode (legacy). Injected into the prompt as `{topic}`; the
    /// teacher is told to skip chunks outside the topic. When `topics` is
    /// non-empty this field is ignored.
    #[serde(default)]
    pub topic: Option<String>,
    /// Single-topic hard cap on accepted Q&A pairs (legacy). Ignored when
    /// `topics` is non-empty.
    #[serde(default)]
    pub total_questions: Option<u32>,
    /// Multi-topic mode — when non-empty, generation runs once per entry with
    /// `{topic}` swapped in, each loop stopping at its own `total_questions`.
    /// The aggregate jsonl is the union of every loop's accepted pairs.
    #[serde(default)]
    pub topics: Vec<TopicTarget>,
    #[serde(default)]
    pub hub: HubConfig,
    /// HF *dataset* hub config (auto-upload every N pairs, optional resume seed).
    #[serde(default)]
    pub hub_dataset: HubDatasetConfig,
    #[serde(default)]
    pub generate_only: bool,
    // auto_destroy lives in hub.auto_destroy now — no separate field needed here
}

impl RunConfig {
    /// Returns the effective list of topic loops the wizard wants us to run.
    /// Falls back to a single entry derived from the legacy `topic`/`total_questions`
    /// fields if the new `topics` list is empty.
    pub fn effective_topics(&self) -> Vec<TopicTarget> {
        let cleaned: Vec<TopicTarget> = self
            .topics
            .iter()
            .filter_map(|t| {
                let topic = t.topic.trim().to_string();
                if topic.is_empty() {
                    return None;
                }
                Some(TopicTarget {
                    topic,
                    total_questions: t.total_questions.filter(|n| *n > 0),
                    tag: t
                        .tag
                        .as_ref()
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty()),
                    prompt_template: t
                        .prompt_template
                        .as_ref()
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty()),
                    embedder_index: t.embedder_index,
                })
            })
            .collect();
        if !cleaned.is_empty() {
            return cleaned;
        }
        let legacy_topic = self
            .topic
            .as_ref()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        // Always return at least one entry so the loop below can still run once
        // (in the no-topic case, `{topic}` falls back to a generic label).
        vec![TopicTarget {
            topic: legacy_topic.unwrap_or_default(),
            total_questions: self.total_questions.filter(|n| *n > 0),
            tag: None,
            prompt_template: None,
            embedder_index: None,
        }]
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LogEvent {
    pub run_id: String,
    pub line: String,
    pub kind: &'static str,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProgressEvent {
    pub run_id: String,
    pub scanned: u64,
    pub kept: u64,
    pub rejected: u64,
    pub status: RunStatus,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MetricEvent {
    pub run_id: String,
    pub step: u32,
    pub loss: f32,
    pub epoch: f32,
}

#[derive(Default)]
pub struct PipelineRegistry {
    pub cancel_flags: Mutex<HashMap<String, Arc<std::sync::atomic::AtomicBool>>>,
}

impl PipelineRegistry {
    pub fn cancel(&self, run_id: &str) {
        if let Some(flag) = self.cancel_flags.lock().get(run_id) {
            flag.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }

    pub fn register(&self, run_id: &str) -> Arc<std::sync::atomic::AtomicBool> {
        let flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        self.cancel_flags
            .lock()
            .insert(run_id.to_string(), flag.clone());
        flag
    }

    pub fn unregister(&self, run_id: &str) {
        self.cancel_flags.lock().remove(run_id);
    }
}

// Per-run local_dir registry so `emit_log` can mirror every emitted line
// onto the run's `live.log` file. Populated at the top of `start`/`resume`
// and dropped when the worker tears down via `PipelineRegistry::unregister`.
static LOG_DIRS: once_cell::sync::Lazy<Mutex<HashMap<String, String>>> =
    once_cell::sync::Lazy::new(|| Mutex::new(HashMap::new()));

pub fn register_log_dir(run_id: &str, local_dir: &str) {
    LOG_DIRS
        .lock()
        .insert(run_id.to_string(), local_dir.to_string());
}

pub fn unregister_log_dir(run_id: &str) {
    LOG_DIRS.lock().remove(run_id);
}

fn emit_log(app: &AppHandle, run_id: &str, line: &str, kind: &'static str) {
    // Mirror to disk so the UI can re-hydrate the full history after a tab
    // switch, page reload, or app restart. The `log_tail` field on Run is
    // capped at 16 KB and only persists at save points, which is why a
    // long-running teacher boot would otherwise vanish from the live log.
    let local_dir = LOG_DIRS.lock().get(run_id).cloned().unwrap_or_default();
    runs::append_log_file(run_id, &local_dir, line);
    let _ = app.emit(
        "pipeline://log",
        LogEvent {
            run_id: run_id.to_string(),
            line: line.to_string(),
            kind,
        },
    );
}

fn emit_progress(app: &AppHandle, run: &Run) {
    let _ = app.emit(
        "pipeline://progress",
        ProgressEvent {
            run_id: run.id.clone(),
            scanned: run.qa_total,
            kept: run.qa_kept,
            rejected: run.qa_rejected,
            status: run.status,
        },
    );
}

fn emit_metric(app: &AppHandle, run_id: &str, step: u32, loss: f32, epoch: f32) {
    let _ = app.emit(
        "pipeline://metric",
        MetricEvent {
            run_id: run_id.to_string(),
            step,
            loss,
            epoch,
        },
    );
}

async fn lexical_topic_candidates(
    qd_cfg: &crate::config::QdrantConfig,
    topic: &str,
    wanted: usize,
    scan_limit: usize,
    tag_filter: Option<&str>,
) -> Result<Vec<crate::qdrant::Chunk>> {
    let stop_words = [
        "and",
        "or",
        "the",
        "of",
        "in",
        "on",
        "for",
        "to",
        "a",
        "an",
        "rules",
        "rule",
        "regulations",
        "regulation",
    ];
    let terms: Vec<String> = topic
        .split(|c: char| !c.is_alphanumeric())
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| s.len() >= 3 && !stop_words.contains(&s.as_str()))
        .collect();

    let mut offset: Option<serde_json::Value> = None;
    let mut scanned = 0usize;
    let mut ranked: Vec<(i32, crate::qdrant::Chunk)> = Vec::new();
    let mut fill: Vec<crate::qdrant::Chunk> = Vec::new();

    while scanned < scan_limit && ranked.len() < wanted {
        let page = crate::qdrant::scroll(qd_cfg, 256, offset.take(), tag_filter).await?;
        if page.chunks.is_empty() {
            break;
        }
        scanned += page.chunks.len();
        offset = page.next_offset;

        for chunk in page.chunks {
            if fill.len() < wanted {
                fill.push(chunk.clone());
            }
            let haystack = format!("{} {} {}", chunk.text, chunk.file_name, chunk.file_path)
                .to_ascii_lowercase();
            let score = terms
                .iter()
                .filter(|term| haystack.contains(term.as_str()))
                .count() as i32;
            if score > 0 {
                ranked.push((score, chunk));
            }
        }

        if offset.is_none() {
            break;
        }
    }

    ranked.sort_by(|a, b| b.0.cmp(&a.0));
    let mut out: Vec<crate::qdrant::Chunk> =
        ranked.into_iter().take(wanted).map(|(_, c)| c).collect();
    if out.is_empty() {
        out = fill;
    }
    Ok(out)
}

pub async fn start(
    app: AppHandle,
    registry: Arc<PipelineRegistry>,
    cfg: AppConfig,
    run_cfg: RunConfig,
) -> Result<String> {
    let mut run = Run::new(
        run_cfg.name.clone(),
        run_cfg.teacher.repo_id.clone(),
        run_cfg.student_model.clone(),
        run_cfg.teacher.clone(),
        run_cfg.lora.clone(),
        run_cfg.hub.clone(),
        run_cfg.hub_dataset.clone(),
    );
    runs::save(&run).await?;
    fs::create_dir_all(&run.local_dir).await.ok();
    register_log_dir(&run.id, &run.local_dir);
    let cancel = registry.register(&run.id);
    let run_id = run.id.clone();

    // Spawn worker — the actual pipeline runs in the background.
    tokio::spawn(async move {
        let result = run_pipeline(&app, &cancel, &mut run, &cfg, &run_cfg, false).await;
        match result {
            Ok(()) => {
                if !run.status.is_terminal() {
                    run.status = RunStatus::Done;
                }
            }
            Err(AppError::Cancelled) => {
                run.status = RunStatus::Cancelled;
                run.error = Some("cancelled by user".to_string());
            }
            Err(e) => {
                run.status = RunStatus::Failed;
                run.error = Some(e.to_string());
                emit_log(&app, &run.id, &format!("\n[FATAL] {e}\n"), "fatal");
            }
        }
        let _ = runs::save(&run).await;
        emit_progress(&app, &run);
        registry.unregister(&run.id);
        unregister_log_dir(&run.id);
    });

    Ok(run_id)
}
/// Resume an existing run from wherever it left off:
///   - If dataset already on disk, skip generation, go straight to training with
///     `resume_from_checkpoint`.
///   - If training never started (or no checkpoint), redo training from scratch
///     but keep the existing dataset.
///   - If dataset was incomplete, generation continues (idempotent — JSONL file
///     append mode + Qdrant scroll is deterministic).
pub async fn resume(
    app: AppHandle,
    registry: Arc<PipelineRegistry>,
    cfg: AppConfig,
    run_id: String,
) -> Result<String> {
    let mut run = runs::load(&run_id).await?;
    if !run.status.is_terminal() && run.status != RunStatus::Pending {
        return Err(AppError::pipeline(format!(
            "run {} is already active (status={:?})",
            run_id, run.status
        )));
    }
    // Rebuild a RunConfig from the persisted Run record so the worker has the
    // same parameters it had originally.
    let run_cfg = RunConfig {
        name: run.name.clone(),
        teacher: run.teacher_cfg.clone(),
        student_model: run.student_model.clone(),
        lora: run.lora.clone(),
        prompt_template: None,
        max_pairs_per_chunk: 1,
        concurrency: 4,
        max_chunks: None,
        topic: None,
        total_questions: None,
        topics: vec![],
        hub: run.hub.clone(),
        hub_dataset: run.hub_dataset.clone(),
        generate_only: false,
        teacher_provider: TeacherProvider::default(),
        featherless_model: None,
    };
    // Reset transient error/status; the worker will set the right status.
    run.error = None;
    run.status = RunStatus::Pending;
    runs::save(&run).await?;
    register_log_dir(&run.id, &run.local_dir);
    let cancel = registry.register(&run.id);

    tokio::spawn(async move {
        let result = run_pipeline(&app, &cancel, &mut run, &cfg, &run_cfg, true).await;
        match result {
            Ok(()) => {
                if !run.status.is_terminal() {
                    run.status = RunStatus::Done;
                }
            }
            Err(AppError::Cancelled) => {
                run.status = RunStatus::Cancelled;
                run.error = Some("cancelled by user".to_string());
            }
            Err(e) => {
                run.status = RunStatus::Failed;
                run.error = Some(e.to_string());
                emit_log(&app, &run.id, &format!("\n[FATAL] {e}\n"), "fatal");
            }
        }
        let _ = runs::save(&run).await;
        emit_progress(&app, &run);
        registry.unregister(&run.id);
        unregister_log_dir(&run.id);
    });
    Ok(run_id)
}

async fn run_pipeline(
    app: &AppHandle,
    cancel: &Arc<std::sync::atomic::AtomicBool>,
    run: &mut Run,
    cfg: &AppConfig,
    run_cfg: &RunConfig,
    resume: bool,
) -> Result<()> {
    let mut effective_cfg = cfg.clone();
    crate::config::normalize_runtime_defaults(&mut effective_cfg);
    let cfg = &effective_cfg;

    // ── 0. Sanity checks ────────────────────────────────────────────────
    let training_only = run_cfg.hub_dataset.train_only;
    let needs_ssh = true;

    if needs_ssh && cfg.ssh.host.is_empty() {
        return Err(AppError::pipeline("SSH host not configured"));
    }
    if training_only {
        if !run_cfg.hub_dataset.enabled || run_cfg.hub_dataset.repo_id.trim().is_empty() {
            return Err(AppError::pipeline(
                "Training Only requires a Hugging Face dataset repo ID",
            ));
        }
    } else if cfg.qdrant.endpoint.is_empty() || cfg.qdrant.collection.is_empty() {
        return Err(AppError::pipeline("Qdrant not configured"));
    }

    // Silently apply AMD guide defaults if the student model matches a family
    // covered by the notebooks in `guide amd/`. Only fields still on schema
    // defaults are overridden — user-edited values are preserved.
    if let Some(g) = crate::guides::match_guide(&run.student_model) {
        crate::guides::apply_guide_defaults(&mut run.lora, g);
        emit_log(
            app,
            &run.id,
            &format!(
                "[guide] applied AMD recipe '{}' (source: {})\n",
                g.label, g.notebook
            ),
            "stage",
        );
    }

    let mut session_opt: Option<SshSession> = None;
    if needs_ssh {
        emit_log(app, &run.id, "[stage] connecting to droplet\n", "stage");
        // Race the SSH connect against the cancel flag so the user can abort a
        // stuck connection attempt instead of waiting for the 30-second timeout.
        let session = tokio::select! {
            res = SshSession::connect(&cfg.ssh) => { res? }
            _ = async {
                loop {
                    if cancel.load(std::sync::atomic::Ordering::SeqCst) {
                        break;
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(250)).await;
                }
            } => {
                return Err(AppError::Cancelled);
            }
        };
        emit_log(app, &run.id, "[ok] SSH connected\n", "stage");
        session_opt = Some(session);
    } else {
        emit_log(
            app,
            &run.id,
            "[stage] using local mode for dataset generation (Featherless)\n",
            "stage",
        );
    }

    let gpu_state = if let Some(ref session) = session_opt {
        crate::ssh::nvidia_smi(session).await.ok()
    } else {
        None
    };

    let docker_cfg = cfg.docker.clone();
    let mut container_name = docker_cfg.container_name.clone();

    // Hoisted flag: set during GPU cleanup, consumed by teacher boot (Phase 2).
    // True when there is a persistent embedder to preserve, so the teacher-boot
    // port cleanup stays port-targeted (never a blanket vLLM pkill that would
    // kill the persistent embedder serving semantic search).
    let mut embedder_1_alive = false;

    if docker_cfg.enabled && needs_ssh {
        // ── Pre-probe: is the teacher already serving? ─────────────────────
        // The teacher model runs INSIDE rocm-vllm. If we unconditionally stop
        // rocm-vllm here, we destroy the already-deployed teacher, causing the
        // downstream probe to miss it and triggering an expensive (and often
        // failing) re-boot. Instead:
        //  • Always stop paddleocr-vl   (only the OCR container — always safe)
        //  • Only stop rocm-vllm if the teacher is NOT already serving on the
        //    expected port (the boot block below will handle it when needed)
        let teacher_port = if let Some(ref cmd) = run_cfg
            .teacher
            .custom_serve_cmd
            .as_ref()
            .filter(|s| !s.is_empty())
        {
            let (_, p) =
                extract_model_and_port(cmd, &run_cfg.teacher.repo_id, run_cfg.teacher.vllm_port);
            p
        } else {
            run_cfg.teacher.vllm_port
        };

        // Quick host-level probe (does NOT require the container to be running)
        let teacher_already_up = if let Some(ref session) = session_opt {
            let probe = format!(
                "curl -s -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{}/v1/models 2>/dev/null || echo 000",
                teacher_port
            );
            session
                .exec_blocking(&probe)
                .await
                .map(|r| r.stdout.trim() == "200")
                .unwrap_or(false)
        } else {
            false
        };

        // ── GPU CLEANUP ─────────────────────────────────────────────────────
        // Rule: NEVER stop the embedding container (rocm-vllm) — keep it warm.
        // Only the PaddleOCR container is evicted, and only the *non-persistent*
        // embedder vLLM processes inside the container(s) are killed. Persistent
        // embedders (the "first embedder" doing semantic search) are preserved so
        // dataset generation can still retrieve via vector search while the
        // teacher boots on the remaining VRAM.
        {
            // Split configured embedders into preserve (persistent) vs stop.
            let non_persistent_ports: Vec<u16> = cfg
                .embedders
                .iter()
                .enumerate()
                .filter(|(idx, e)| !is_protected_semantic_embedder(*idx, e))
                .map(|(_, e)| e.port)
                .collect();
            let persistent_names: Vec<String> = cfg
                .embedders
                .iter()
                .enumerate()
                .filter(|(idx, e)| is_protected_semantic_embedder(*idx, e))
                .map(|(_, e)| e.name.clone())
                .collect();
            let non_persistent_names: Vec<String> = cfg
                .embedders
                .iter()
                .enumerate()
                .filter(|(idx, e)| !is_protected_semantic_embedder(*idx, e))
                .map(|(_, e)| e.name.clone())
                .collect();
            // Phase-2 boot keeps a port-targeted (non-blanket) pkill whenever a
            // persistent embedder must survive.
            embedder_1_alive = !persistent_names.is_empty();

            if teacher_already_up {
                emit_log(
                    app,
                    &run.id,
                    &format!(
                        "[GPU CLEANUP] teacher already serving on port {} — stopping only PaddleOCR (paddleocr-vl), preserving rocm-vllm container...\n",
                        teacher_port
                    ),
                    "stage",
                );
            } else {
                emit_log(
                    app,
                    &run.id,
                    &format!(
                        "[GPU CLEANUP] stopping PaddleOCR (paddleocr-vl); preserving rocm-vllm container; stopping {} non-persistent embedder(s) ({}), preserving persistent embedder(s) ({})\n",
                        non_persistent_names.len(),
                        if non_persistent_names.is_empty() { "none".to_string() } else { non_persistent_names.join(", ") },
                        if persistent_names.is_empty() { "none".to_string() } else { persistent_names.join(", ") },
                    ),
                    "stage",
                );
            }

            if let Some(ref session) = session_opt {
                // Always evict only the OCR container — never rocm-vllm.
                let _ = session.exec_blocking(
                    "docker stop paddleocr-vl 2>/dev/null; docker rm paddleocr-vl 2>/dev/null; true"
                ).await;

                // Stop non-persistent embedder processes (only when the teacher is
                // not already up — if it is, the embedders inside are already in
                // their steady state and the teacher needs no extra VRAM freed).
                if !teacher_already_up && !non_persistent_ports.is_empty() {
                    // Port-targeted only so protected embedders on other ports
                    // are preserved whether they run on the host or in Docker.
                    let host_kill = build_embedder_port_kill_cmd(&non_persistent_ports);
                    let _ = session.exec_blocking(&host_kill).await;

                    let inner_kill = build_embedder_port_kill_cmd(&non_persistent_ports);
                    if let Ok(ps_r) = session
                        .exec_blocking("docker ps --format '{{.Names}}' 2>/dev/null || true")
                        .await
                    {
                        for cname in ps_r
                            .stdout
                            .lines()
                            .map(|s| s.trim().to_string())
                            .filter(|s| !s.is_empty())
                        {
                            let inner = wrap_docker_cmd(&inner_kill, &cname);
                            let _ = session.exec_blocking(&inner).await;
                        }
                    }
                    emit_log(
                        app,
                        &run.id,
                        "[GPU CLEANUP] non-persistent embedders stopped\n",
                        "stage",
                    );
                }
            }
        }

        emit_log(
            app,
            &run.id,
            &format!(
                "[stage] [DOCKER] ensuring container '{}' is running\n",
                docker_cfg.container_name
            ),
            "stage",
        );
        container_name = ensure_container(session_opt.as_ref().unwrap(), &docker_cfg).await?;
        emit_log(
            app,
            &run.id,
            &format!("[ok] container '{}' is running\n", container_name),
            "stage",
        );

        // ── Re-deploy ONE embedder for semantic search ─────────────────────
        // The cleanup above stopped PaddleOCR and the embedder vLLM to free VRAM
        // for the teacher. Dataset generation still needs to embed topic queries
        // for vector search against Qdrant, so we bring a single embedder back up
        // on the remaining VRAM (the embedder is sized at ~0.084 GPU util, which
        // co-exists fine with the teacher).
        //
        // GATE: only deploy the embedder when the GPU server actually HAS a Qdrant
        // running (live probe on :6333). With no Qdrant there is nothing to search,
        // so deploying an embedder would just waste VRAM — skip it entirely.
        // Also skipped in train_only / resumed-dataset modes (no generation).
        let needs_embedder = !run_cfg.hub_dataset.train_only && !(resume && run.dataset_ready);
        if needs_embedder {
            let session = session_opt.as_ref().unwrap();

            // Live Qdrant probe on the GPU server.
            let qdrant_up = {
                let probe = "curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:6333/collections 2>/dev/null || echo 000";
                session
                    .exec_blocking(probe)
                    .await
                    .map(|r| r.stdout.trim() == "200")
                    .unwrap_or(false)
            };

            if !qdrant_up {
                emit_log(
                    app,
                    &run.id,
                    "[embedder] GPU server has no Qdrant on :6333 — skipping embedder deployment (no vector store to search)\n",
                    "stage",
                );
            } else if let Some((_embedder_index, embedder)) = cfg
                .embedders
                .iter()
                .enumerate()
                .find(|(idx, e)| is_protected_semantic_embedder(*idx, e))
            {
                // Embedder index 0 is the semantic-search embedder and is
                // protected even for older configs that predate `persistent`.
                // Keep it in the configured vLLM Docker runtime; cleanup is
                // port-targeted so protected embedders survive teacher deploys.
                let embedder_cfg = docker_cfg.clone();

                // Already serving?
                let already_up = crate::serve::health_check_embedder(
                    session,
                    &embedder_cfg,
                    "127.0.0.1",
                    embedder.port,
                )
                .await
                .map(|o| o.is_some())
                .unwrap_or(false);

                if already_up {
                    emit_log(
                        app,
                        &run.id,
                        &format!(
                            "[embedder] '{}' already serving on port {} — reusing for semantic search\n",
                            embedder.name, embedder.port
                        ),
                        "stage",
                    );
                } else {
                    let gpu_mem = embedder.gpu_memory_utilization as f64;
                    emit_log(
                        app,
                        &run.id,
                        &format!(
                            "[embedder] Qdrant present — deploying 1 embedder '{}' ({}) on port {} via docker ({} GPU util)\n",
                            embedder.name, embedder.model_id, embedder.port, gpu_mem
                        ),
                        "stage",
                    );
                    let hf_token = cfg.hf_token.as_deref().filter(|s| !s.is_empty());
                    match crate::serve::launch_embedder(
                        session,
                        &embedder_cfg,
                        embedder,
                        hf_token,
                        Some(gpu_mem),
                    )
                    .await
                    {
                        Ok(_) => emit_log(
                            app,
                            &run.id,
                            &format!(
                                "[embedder] '{}' launching in background on port {} — continuing; semantic search activates once it is ready (keyword scroll used until then)\n",
                                embedder.name, embedder.port
                            ),
                            "stage",
                        ),
                        Err(e) => emit_log(
                            app,
                            &run.id,
                            &format!(
                                "[embedder] launch failed (non-fatal, will fall back to keyword scroll): {}\n",
                                e
                            ),
                            "warn",
                        ),
                    }
                }

                // Block waiting for the embedder to serve /v1/models (up to
                // 600s = 10 min). An 8B model takes 3–5 min to load into VRAM.
                // If it becomes ready, semantic search works immediately.
                // Otherwise fall through to keyword-ranked Qdrant scroll.
                emit_log(
                    app,
                    &run.id,
                    &format!(
                        "[embedder] waiting up to 600s for '{}' on 127.0.0.1:{} to become ready...\n",
                        embedder.name, embedder.port
                    ),
                    "stage",
                );
                let embedder_ready = tokio::time::timeout(
                    std::time::Duration::from_secs(600),
                    crate::serve::wait_for_embedder(session, &embedder_cfg, embedder, 600, None),
                )
                .await;
                match embedder_ready {
                    Ok(Ok(ip)) => {
                        emit_log(
                            app,
                            &run.id,
                            &format!(
                                "[embedder] '{}' ready at {}:{} — semantic search active\n",
                                embedder.name, ip, embedder.port
                            ),
                            "stage",
                        );
                    }
                    Ok(Err(e)) => {
                        emit_log(
                            app,
                            &run.id,
                            &format!(
                                "[embedder] '{}' not ready within 600s (non-fatal, falling back to keyword scroll): {}\n",
                                embedder.name, e
                            ),
                            "warn",
                        );
                    }
                    Err(_) => {
                        emit_log(
                            app,
                            &run.id,
                            &format!(
                                "[embedder] '{}' wait timed out after 600s (non-fatal, falling back to keyword scroll)\n",
                                embedder.name
                            ),
                            "warn",
                        );
                    }
                }
            } else {
                emit_log(
                    app,
                    &run.id,
                    "[embedder] Qdrant present but no embedders configured — skipping embedder deployment\n",
                    "warn",
                );
            }
        }
    }

    // Prep remote directories
    let remote_data = format!("{}/data", run.remote_dir);
    if let Some(session) = session_opt.as_ref() {
        let cmd = format!(
            "mkdir -p {} {}/lora {} && (rocm-smi -i 2>/dev/null || amd-smi list 2>/dev/null || true)",
            run.remote_dir, run.remote_dir, remote_data
        );
        let r = session.exec_blocking(&cmd).await?;
        runs::append_log_tail(run, &r.stdout);
        emit_log(app, &run.id, &r.stdout, "stage");
    }

    // Resume short-circuit: if dataset already prepared on disk or if train_only is enabled,
    // skip teacher boot and generation, jump straight to training.
    let skip_dataset = (resume && run.dataset_ready) || run.hub_dataset.train_only;
    if skip_dataset {
        let session = session_opt
            .as_ref()
            .ok_or_else(|| AppError::pipeline("SSH session unavailable for training"))?;
        let msg = if run.hub_dataset.train_only {
            "[resume] train_only mode enabled — skipping teacher + generation, training from Hugging Face dataset\n"
        } else {
            "[resume] dataset already prepared — skipping teacher + generation\n"
        };
        emit_log(app, &run.id, msg, "stage");

        if run.hub_dataset.train_only {
            // Aggressive VRAM cleanup requested by user for this mode: ensure no
            // lingering vLLM or sglang processes are holding memory.
            emit_log(
                app,
                &run.id,
                "[stage] direct training: purging any existing vLLM and sglang processes to free VRAM\n",
                "stage",
            );
            let pkill_body = "pkill -f '[v]llm' 2>/dev/null; \
                              pkill -9 -f '[v]llm' 2>/dev/null; \
                              pkill -f 'sglang' 2>/dev/null; \
                              pkill -9 -f 'sglang' 2>/dev/null; \
                              true";
            let _ = session.exec_blocking(pkill_body).await;
            if docker_cfg.enabled {
                if let Ok(ps_r) = session
                    .exec_blocking("docker ps --format '{{.Names}}'")
                    .await
                {
                    let names: Vec<String> = ps_r
                        .stdout
                        .lines()
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect();
                    for cname in &names {
                        if cname != &container_name
                            && (cname.contains("vllm")
                                || cname.contains("sglang")
                                || cname.contains("paddleocr"))
                        {
                            emit_log(
                                app,
                                &run.id,
                                &format!("[GPU CLEANUP] stopping and removing container '{}' to free VRAM...\n", cname),
                                "stage",
                            );
                            let stop_cmd = format!(
                                "docker stop {} 2>/dev/null; docker rm {} 2>/dev/null; true",
                                cname, cname
                            );
                            let _ = session.exec_blocking(&stop_cmd).await;
                        } else {
                            let inner = wrap_docker_cmd(pkill_body, cname);
                            let _ = session.exec_blocking(&inner).await;
                        }
                    }
                }
            }
        }
    }

    let gpu_memory_total_mb = gpu_state.as_ref().map(|gpu| gpu.memory_total);
    let mut effective_teacher = run_cfg.teacher.resolved_for_gpu(gpu_memory_total_mb);
    let has_semantic_embedder = cfg
        .embedders
        .iter()
        .enumerate()
        .any(|(idx, e)| is_protected_semantic_embedder(idx, e));
    if has_semantic_embedder {
        if let Some((old, new)) = cap_teacher_vram_for_semantic_embedder(&mut effective_teacher) {
            emit_log(
                app,
                &run.id,
                &format!(
                    "[vram] protected embedder_1 is reserved for semantic search; capping teacher VRAM util from {:.2} to {:.2}\n",
                    old, new
                ),
                "stage",
            );
        }
    }
    if !skip_dataset
        && effective_teacher
            .custom_serve_cmd
            .as_ref()
            .map(|cmd| cmd.trim().is_empty())
            .unwrap_or(true)
    {
        if let Some(session) = session_opt.as_ref() {
            if let Some((limit, source)) = probe_model_context_limit(
                session,
                docker_cfg.enabled,
                &container_name,
                &effective_teacher.repo_id,
                cfg.hf_token.as_deref(),
            )
            .await
            {
                if effective_teacher.max_model_len > limit {
                    let old = effective_teacher.max_model_len;
                    effective_teacher.max_model_len = limit;
                    emit_log(
                        app,
                        &run.id,
                        &format!(
                            "[context] model config reports {}={}; capping --max-model-len from {} to {}\n",
                            source, limit, old, limit
                        ),
                        "stage",
                    );
                }
            }
        }
    }
    if run.teacher_cfg.repo_id != effective_teacher.repo_id
        || run.teacher_cfg.max_model_len != effective_teacher.max_model_len
        || run.teacher_cfg.dtype != effective_teacher.dtype
        || (run.teacher_cfg.gpu_memory_utilization - effective_teacher.gpu_memory_utilization).abs()
            > f32::EPSILON
    {
        run.teacher_cfg = effective_teacher.clone();
        let _ = runs::save(run).await;
    }
    let t = &effective_teacher;

    if !skip_dataset {
        // ── 1. Boot Teacher (vLLM, OpenAI-compatible) ───────────────────────
        run.status = RunStatus::TeacherLoading;
        runs::save(run).await?;
        emit_progress(app, run);

        let mut served_model_id = String::new();
        let mut port_to_check = t.vllm_port;
        let mut teacher_endpoint = format!("http://{}:{}", cfg.ssh.host, port_to_check);

        if true {
            let session = session_opt.as_ref().unwrap();
            let teacher_log = format!("{}/teacher.log", run.remote_dir);

            // Extract model name and port from custom command if possible, otherwise fallback
            let (model_to_check, mut port_to_check_inner) =
                if let Some(ref cmd) = t.custom_serve_cmd.as_ref().filter(|s| !s.is_empty()) {
                    extract_model_and_port(cmd, &t.repo_id, t.vllm_port)
                } else {
                    (t.repo_id.clone(), t.vllm_port)
                };

            // The actual model id we will pass to /v1/chat/completions. We start with
            // the configured value but overwrite it with whatever vLLM reports at
            // /v1/models once we detect a running server — vLLM normalises GGUF paths,
            // strips suffixes, etc., so the served id is rarely the same string the
            // user typed into the config.
            served_model_id = model_to_check.clone();

            // Check if the teacher model is already running on the remote port and serving the target model.
            // We run a comprehensive python script inline over SSH/Docker that reads listening ports
            // and probes them to find where our model (or any model, as a fallback) is serving.
            let check_script = format!(
                "python3 -c \"\
import urllib.request, json
def g():
    p = []
    for path in ['/proc/net/tcp', '/proc/net/tcp6']:
        try:
            with open(path) as f:
                for l in f.readlines()[1:]:
                    parts = l.split()
                    if len(parts) > 3 and parts[3] == '0A':
                        lp = int(parts[1].split(':')[1], 16)
                        if lp not in p: p.append(lp)
        except: pass
    return sorted(p)
m_cfg = '{}'.lower().replace('.gguf', '').split(':')[0]
p_cfg = {}
ports = g()
if p_cfg in ports:
    ports.remove(p_cfg)
    ports.insert(0, p_cfg)
exact, any_m = None, None
for p in ports:
    if p in [22, 53, 80, 443, 3306, 5432, 6333, 8888, 27017, 30000]: continue
    try:
        req = urllib.request.Request(f'http://127.0.0.1:{{p}}/v1/models', headers={{'User-Agent': 'vllm-probe'}})
        with urllib.request.urlopen(req, timeout=1.0) as res:
            if res.status == 200:
                data = json.loads(res.read().decode())
                m = data.get('data', [])
                if m:
                    mid = m[0].get('id', '')
                    cid = mid.lower().replace('.gguf', '').split(':')[0]
                    if m_cfg and cid == m_cfg:
                        exact = (p, mid)
                        break
                    elif not any_m:
                        any_m = (p, mid)
    except: pass
if exact: print(f'FOUND_PORT::{{exact[0]}}::{{exact[1]}}')
elif any_m: print(f'FIRST_PORT::{{any_m[0]}}::{{any_m[1]}}')
else: print('NOT_FOUND')\
\" 2>/dev/null || echo 'ERROR'",
                model_to_check.replace("\"", "\\\"").replace("'", "\\'"),
                port_to_check_inner
            );

            let check_probe = if docker_cfg.enabled {
                wrap_docker_cmd(&check_script, &container_name)
            } else {
                check_script
            };
            let mut already_running = false;
            if let Ok(probe_r) = session.exec_blocking(&check_probe).await {
                let out = probe_r.stdout.trim();
                let found_line = out
                    .lines()
                    .find(|l| l.starts_with("FOUND_PORT::") || l.starts_with("FIRST_PORT::"));
                if let Some(line) = found_line {
                    let parts: Vec<&str> = line.splitn(3, "::").collect();
                    if parts.len() == 3 {
                        if let Ok(port) = parts[1].parse::<u16>() {
                            already_running = true;
                            port_to_check_inner = port;
                            let id = parts[2].trim();
                            if !id.is_empty() {
                                served_model_id = id.to_string();
                            }
                            if line.starts_with("FIRST_PORT::") {
                                emit_log(
                                    app,
                                    &run.id,
                                    &format!(
                                        "[warn] configured teacher model '{}' not found at /v1/models; using actually-served model '{}' instead on port {}\n",
                                        model_to_check, served_model_id, port_to_check_inner
                                    ),
                                    "warn",
                                );
                            }
                        }
                    }
                }
            }

            if already_running {
                emit_log(
                    app,
                    &run.id,
                    &format!(
                        "[ok] teacher model '{}' is already serving on port {} — skipping boot\n",
                        served_model_id, port_to_check_inner
                    ),
                    "stage",
                );
            } else {
                // Pre-kill any existing vLLM processes AND free the target port. The
                // old `pkill A || pkill B` form short-circuits — when A matched, B
                // never ran, leaving orphan workers (and the bound socket) behind,
                // which surfaces 5 minutes later as `OSError: [Errno 98] Address
                // already in use`. Use `;` to guarantee every pattern is tried,
                // escalate to -9 after a grace period, then `fuser -k`/`ss` to kill
                // whatever still holds the port. Poll up to 10s for the port to
                // actually become free before declaring success.
                let pkill_body = format!(
            "pkill -f '[v]llm.entrypoints' 2>/dev/null; \
             pkill -f '[v]llm serve' 2>/dev/null; \
             pkill -f 'multiproc.*vllm' 2>/dev/null; \
             pkill -f 'python.*vllm' 2>/dev/null; \
             pkill -f 'sglang.launch_server' 2>/dev/null; \
             pkill -f 'python.*sglang' 2>/dev/null; \
             sleep 1; \
             pkill -9 -f '[v]llm.entrypoints' 2>/dev/null; \
             pkill -9 -f '[v]llm serve' 2>/dev/null; \
             pkill -9 -f 'multiproc.*vllm' 2>/dev/null; \
             pkill -9 -f 'python.*vllm' 2>/dev/null; \
             pkill -9 -f 'sglang.launch_server' 2>/dev/null; \
             pkill -9 -f 'python.*sglang' 2>/dev/null; \
             (command -v fuser >/dev/null 2>&1 && fuser -k {port}/tcp 2>/dev/null) || true; \
             (command -v ss >/dev/null 2>&1 && ss -ltnp 2>/dev/null | awk '/:{port} /{{print $0}}' | grep -oE 'pid=[0-9]+' | cut -d= -f2 | xargs -r kill -9 2>/dev/null) || true; \
             for i in 1 2 3 4 5 6 7 8 9 10; do \
                 if command -v ss >/dev/null 2>&1; then \
                     ss -ltn 2>/dev/null | awk '{{print $4}}' | grep -qE ':{port}$' || break; \
                 else \
                     (netstat -ltn 2>/dev/null || true) | awk '{{print $4}}' | grep -qE ':{port}$' || break; \
                 fi; \
                 sleep 1; \
             done; \
             true",
            port = port_to_check_inner,
        );
                // Ensure the PaddleOCR container is stopped (already done at
                // pipeline start, but force again here in case it was restarted
                // externally). NEVER stop rocm-vllm — the embedding container is
                // kept warm; only embedder *processes* are managed, and the
                // persistent embedder serving semantic search must survive.
                if docker_cfg.enabled {
                    let _ = session.exec_blocking("docker stop paddleocr-vl 2>/dev/null; docker rm paddleocr-vl 2>/dev/null; true").await;
                    container_name = ensure_container(session, &docker_cfg).await?;
                }
                // When embedder_1 is alive, use a port-targeted cleanup instead of
                // the blanket vLLM pkill (which would kill the embedder process too).
                let effective_pkill = if embedder_1_alive {
                    format!(
                        "(command -v fuser >/dev/null 2>&1 && fuser -k {port}/tcp 2>/dev/null) || true; \
                         (command -v ss >/dev/null 2>&1 && ss -ltnp 2>/dev/null | awk '/:{port} /{{print $0}}' | grep -oE 'pid=[0-9]+' | cut -d= -f2 | xargs -r kill -9 2>/dev/null) || true; \
                         for i in 1 2 3 4 5 6 7 8 9 10; do \
                             if command -v ss >/dev/null 2>&1; then \
                                 ss -ltn 2>/dev/null | awk '{{print $4}}' | grep -qE ':{port}$' || break; \
                             else \
                                 (netstat -ltn 2>/dev/null || true) | awk '{{print $4}}' | grep -qE ':{port}$' || break; \
                             fi; \
                             sleep 1; \
                         done; \
                         true",
                        port = port_to_check_inner,
                    )
                } else {
                    pkill_body
                };
                // First sweep on the host — covers any vLLM started outside docker
                // and any process holding the port directly on the bare metal.
                let _ = session.exec_blocking(&effective_pkill).await;
                tokio::time::sleep(std::time::Duration::from_millis(200)).await;

                // Then sweep across EVERY running container. ROCm/vLLM containers
                // typically use --network=host, so a vLLM running inside container A
                // will block port binding from container B. Iterate all running
                // containers and run the same kill body inside each.
                if docker_cfg.enabled {
                    if let Ok(ps_r) = session
                        .exec_blocking("docker ps --format '{{.Names}}'")
                        .await
                    {
                        let names: Vec<String> = ps_r
                            .stdout
                            .lines()
                            .map(|s| s.trim().to_string())
                            .filter(|s| !s.is_empty())
                            .collect();
                        for cname in &names {
                            let inner = wrap_docker_cmd(&effective_pkill, cname);
                            let _ = session.exec_blocking(&inner).await;
                        }
                    }
                }

                // Final targeted sweep in the container we'll actually use.
                let pkill_cmd = if docker_cfg.enabled {
                    wrap_docker_cmd(&effective_pkill, &container_name)
                } else {
                    effective_pkill
                };
                let _ = session.exec_blocking(&pkill_cmd).await;
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;

                // Final sanity check: is the port really free? If not, fail fast with
                // a clear error instead of waiting 5 minutes for vLLM to crash.
                let port_check_inner = format!(
            "if command -v ss >/dev/null 2>&1; then \
                 ss -ltn 2>/dev/null | awk '{{print $4}}' | grep -qE ':{port}$' && echo PORT_BUSY || echo PORT_FREE; \
             else \
                 (netstat -ltn 2>/dev/null || true) | awk '{{print $4}}' | grep -qE ':{port}$' && echo PORT_BUSY || echo PORT_FREE; \
             fi",
            port = port_to_check_inner,
        );
                let _port_check_cmd = if docker_cfg.enabled {
                    wrap_docker_cmd(&port_check_inner, &container_name)
                } else {
                    port_check_inner.clone()
                };
                // ── Always use a free port (avoid race condition where port  ──
                //    becomes busy between the check and vLLM's bind()).
                let find_port_script = "python3 -c \"import socket; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(('', 0)); print(s.getsockname()[1]); s.close()\" 2>/dev/null || echo 0";
                let find_cmd = if docker_cfg.enabled {
                    wrap_docker_cmd(find_port_script, &container_name)
                } else {
                    find_port_script.to_string()
                };
                if let Ok(fr) = session.exec_blocking(&find_cmd).await {
                    let found: u16 = fr.stdout.trim().parse().unwrap_or(port_to_check_inner);
                    if found != port_to_check_inner {
                        emit_log(app, &run.id, &format!("[port] configured port {} is not available, using free port {} instead\n", port_to_check_inner, found), "info");
                        port_to_check_inner = found;
                        if let Ok(mut app_cfg) = crate::config::load().await {
                            app_cfg.teacher.vllm_port = found;
                            let _ = crate::config::save(&app_cfg).await;
                        }
                    }
                }

                let boot_cmd = if let Some(ref cmd) =
                    t.custom_serve_cmd.as_ref().filter(|s| !s.is_empty())
                {
                    let custom_cmd_clean = cmd
                        .replace("\\\n", " ")
                        .replace("\\\r\n", " ")
                        .replace('\n', " ")
                        .replace('\r', " ");

                    let mut final_custom_cmd = custom_cmd_clean.clone();
                    if !final_custom_cmd.contains("HF_TOKEN")
                        && !final_custom_cmd.contains("HUGGING_FACE_HUB_TOKEN")
                    {
                        if let Some(tok) = cfg.hf_token.as_ref().filter(|s| !s.is_empty()) {
                            final_custom_cmd = format!(
                                "export HF_TOKEN={} HUGGING_FACE_HUB_TOKEN={}; {}",
                                tok, tok, final_custom_cmd
                            );
                        }
                    }

                    let mut display_cmd = final_custom_cmd.clone();
                    if let Some(idx) = display_cmd.find("HF_TOKEN=") {
                        let after_token = &display_cmd[idx + 9..];
                        if let Some(space_idx) = after_token.find(' ') {
                            display_cmd = format!(
                                "{}HF_TOKEN=***{}",
                                &display_cmd[..idx],
                                &after_token[space_idx..]
                            );
                        } else {
                            display_cmd = format!("{}HF_TOKEN=***", &display_cmd[..idx]);
                        }
                    }

                    emit_log(app, &run.id, "[stage] booting teacher vLLM\n", "stage");
                    emit_log(app, &run.id, &format!("[cmd] {}\n", display_cmd), "stage");

                    if docker_cfg.enabled {
                        let inner_cmd = format!(
                            "mkdir -p /root/hf-cache; \
                     mkdir -p $(dirname {teacher_log}); \
                     truncate -s 0 {teacher_log} 2>/dev/null || rm -f {teacher_log}; \
                     cd /app && {custom_cmd} \
                     > {log} 2>&1",
                            teacher_log = teacher_log,
                            custom_cmd = final_custom_cmd,
                            log = teacher_log,
                        );
                        wrap_docker_cmd_detached(&inner_cmd, &container_name)
                    } else {
                        format!(
                            "mkdir -p /root/hf-cache; \
                     truncate -s 0 {teacher_log} 2>/dev/null || rm -f {teacher_log}; \
                     nohup bash -lc 'cd /app && {custom_cmd} > {log} 2>&1' < /dev/null & \
                     echo TEACHER_LAUNCHED",
                            teacher_log = teacher_log,
                            custom_cmd = final_custom_cmd,
                            log = teacher_log,
                        )
                    }
                } else {
                    // Guess the base model for GGUF files to load compatible tokenizer from HF
                    let mut tokenizer_arg = String::new();
                    let repo_id_lower = t.repo_id.to_lowercase();
                    if repo_id_lower.contains("gguf") {
                        let parts: Vec<&str> = t.repo_id.split('/').collect();
                        let base_repo = if parts.len() >= 2 {
                            format!(
                                "{}/{}",
                                parts[0],
                                parts[1].split(':').next().unwrap_or(parts[1])
                            )
                        } else {
                            t.repo_id
                                .split(':')
                                .next()
                                .unwrap_or(&t.repo_id)
                                .to_string()
                        };
                        let base_model = base_repo
                            .replace("-GGUF", "")
                            .replace("-gguf", "")
                            .replace(".GGUF", "")
                            .replace(".gguf", "");
                        tokenizer_arg = format!("--tokenizer {}", base_model);
                    }

                    // Global optimizations for ROCm and AMD MI300X (192GB VRAM)
                    let vllm_env = {
                        let mut envs = format!(
                            "export PYTHONUNBUFFERED=1; \
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
                     export OMP_NUM_THREADS=4; "
                        );
                        if let Some(tok) = cfg.hf_token.as_ref().filter(|s| !s.is_empty()) {
                            envs.push_str(&format!(
                                "export HF_TOKEN={} HUGGING_FACE_HUB_TOKEN={}; ",
                                tok, tok
                            ));
                        }
                        envs
                    };
                    let extra_args = t.vllm_extra_args();
                    let runtime_prepare = t.vllm_runtime_prepare_cmd();

                    let serve_cmd_display = format!(
                        "HF_TOKEN=*** vllm serve {} --port {} --host 0.0.0.0 \
                         --max-model-len {} --dtype {} \
                         --download-dir /root/hf-cache \
                         --tensor-parallel-size {} --gpu-memory-utilization {} {} {}\n",
                        t.repo_id,
                        port_to_check_inner,
                        t.max_model_len,
                        t.dtype,
                        t.tensor_parallel,
                        t.gpu_memory_utilization,
                        tokenizer_arg,
                        extra_args
                    );
                    emit_log(app, &run.id, "[stage] booting teacher vLLM\n", "stage");
                    emit_log(
                        app,
                        &run.id,
                        &format!("[cmd] {}", serve_cmd_display),
                        "stage",
                    );

                    if docker_cfg.enabled {
                        let inner_cmd = format!(
                            "mkdir -p /root/hf-cache; \
                             mkdir -p $(dirname {teacher_log}); \
                             truncate -s 0 {teacher_log} 2>/dev/null || rm -f {teacher_log}; \
                             cd /app && {env}{runtime_prepare}vllm serve {model} --port {port} --host 0.0.0.0 \
                                --max-model-len {mml} --dtype {dtype} --download-dir /root/hf-cache \
                                --tensor-parallel-size {tp} --gpu-memory-utilization {gpu_mem} {tok_arg} {extra_args} \
                                > {log} 2>&1",
                            teacher_log = teacher_log,
                            env = vllm_env,
                            runtime_prepare = runtime_prepare,
                            model = t.repo_id,
                            port = port_to_check_inner,
                            mml = t.max_model_len,
                            dtype = t.dtype,
                            tp = t.tensor_parallel,
                            gpu_mem = t.gpu_memory_utilization,
                            tok_arg = tokenizer_arg,
                            extra_args = extra_args,
                            log = teacher_log,
                        );
                        wrap_docker_cmd_detached(&inner_cmd, &container_name)
                    } else {
                        format!(
                            "mkdir -p /root/hf-cache; \
                             truncate -s 0 {teacher_log} 2>/dev/null || rm -f {teacher_log}; \
                             nohup bash -lc 'cd /app && {env}{runtime_prepare}vllm serve {model} --port {port} --host 0.0.0.0 \
                                --max-model-len {mml} --dtype {dtype} --download-dir /root/hf-cache \
                                --tensor-parallel-size {tp} --gpu-memory-utilization {gpu_mem} {tok_arg} {extra_args} \
                                > {log} 2>&1' < /dev/null & \
                             echo TEACHER_LAUNCHED",
                            teacher_log = teacher_log,
                            env = vllm_env,
                            runtime_prepare = runtime_prepare,
                            model = t.repo_id,
                            port = port_to_check_inner,
                            mml = t.max_model_len,
                            dtype = t.dtype,
                            tp = t.tensor_parallel,
                            gpu_mem = t.gpu_memory_utilization,
                            tok_arg = tokenizer_arg,
                            extra_args = extra_args,
                            log = teacher_log,
                        )
                    }
                };

                let boot_r = session.exec_blocking(&boot_cmd).await?;
                if boot_r.exit_code != 0 {
                    return Err(AppError::pipeline(format!(
                        "failed to start teacher vLLM process (exit {}): {}",
                        boot_r.exit_code, boot_r.stderr
                    )));
                }

                // Poll /v1/models until ready (or timeout 20 min).
                // Stream teacher.log *incrementally* — track how many lines we've already
                // emitted and only fetch new ones each poll. This gives the user a
                // live feed of model weight download / layer load progress.
                let started = std::time::Instant::now();
                let timeout = std::time::Duration::from_secs(20 * 60);
                let mut log_line_offset: u64 = 1; // `tail -n +1` = from the very first line
                loop {
                    if cancel.load(std::sync::atomic::Ordering::SeqCst) {
                        return Err(AppError::Cancelled);
                    }
                    if started.elapsed() > timeout {
                        return Err(AppError::pipeline("teacher boot timeout (20 min)"));
                    }

                    // ── Fetch new lines from teacher.log and check API in one SSH call ─
                    let combo_cmd = if docker_cfg.enabled {
                        format!(
                    "docker exec {} tail -n +{} {} 2>/dev/null | head -n 500; \
                     echo '---PROBE---'; \
                     docker exec {} curl -s -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{}/v1/models || echo '000'",
                    container_name, log_line_offset, teacher_log, container_name, port_to_check_inner
                )
                    } else {
                        format!(
                    "tail -n +{} {} 2>/dev/null | head -n 500; \
                     echo '---PROBE---'; \
                     curl -s -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{}/v1/models || echo '000'",
                    log_line_offset, teacher_log, port_to_check_inner
                )
                    };
                    let r = session.exec_blocking(&combo_cmd).await?;

                    let parts: Vec<&str> = r.stdout.split("---PROBE---").collect();
                    let logs_part = parts.first().map(|s| *s).unwrap_or("");
                    let probe_part = parts.get(1).map(|s| s.trim()).unwrap_or("000");

                    if !logs_part.is_empty() {
                        let new_count = logs_part.lines().count() as u64;
                        log_line_offset += new_count;
                        runs::append_log_tail(run, logs_part);
                        emit_log(app, &run.id, logs_part, "teacher");

                        let lower = logs_part.to_lowercase();
                        if lower.contains("traceback")
                            || lower.contains("validationerror")
                            || lower.contains("does not recognize this architecture")
                            || lower.contains("vllm_failed")
                            || lower.contains("out of memory")
                            || lower.contains("hip out of memory")
                            || lower.contains("outofmemoryerror")
                        {
                            let engine_name = "vLLM";
                            let err_msg = if lower.contains("out of memory")
                                || lower.contains("hip out of memory")
                                || lower.contains("outofmemoryerror")
                            {
                                format!(
                                    "{} crashed during startup with an Out Of Memory (OOM) error. \
                                     Suggestions:\n\
                                     1. If you are deploying a massive model (like DeepSeek-V3 or R1) on a single GPU, it will not fit. Use a smaller model (e.g. DeepSeek-R1-Distill-Qwen-32B) or increase Tensor Parallelism in Settings -> Teacher.\n\
                                     2. If the model should fit, lower your 'GPU Memory Utilization' (gpuMemoryUtilization / --mem-fraction-static) in Settings to 0.85 or 0.80 to leave enough headroom for loading.\n\
                                     3. Ensure other GPU processes are stopped to free up VRAM.",
                                    engine_name
                                )
                            } else if lower.contains("deepseek_v4")
                                || lower.contains("does not recognize this architecture")
                            {
                                format!(
                                    "{} crashed because the container runtime does not support this model architecture yet. The deploy command now installs a Transformers build with DeepSeek V4 support before launching; deploy again to apply it.",
                                    engine_name
                                )
                            } else {
                                format!("{} crashed during startup; check the streamed traceback above for the root cause", engine_name)
                            };
                            return Err(AppError::pipeline(err_msg));
                        }
                    }

                    if probe_part == "200" {
                        // Drain any remaining log lines before marking ready.
                        let drain = if docker_cfg.enabled {
                            format!(
                                "docker exec {} tail -n +{} {} 2>/dev/null",
                                container_name, log_line_offset, teacher_log
                            )
                        } else {
                            format!("tail -n +{} {} 2>/dev/null", log_line_offset, teacher_log)
                        };
                        let drain_r = session.exec_blocking(&drain).await?;
                        if !drain_r.stdout.is_empty() {
                            runs::append_log_tail(run, &drain_r.stdout);
                            emit_log(app, &run.id, &drain_r.stdout, "teacher");
                        }
                        emit_log(app, &run.id, "[ok] teacher ready\n", "stage");
                        break;
                    }

                    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                }

                // Resolve the actual model id vLLM reports — for GGUF / local-path
                // servings the served id often differs from the configured repo_id,
                // and /v1/chat/completions will 404 unless we send the exact string.
                let id_probe = format!(
                    "python3 -c \"import urllib.request, json; \
             res=urllib.request.urlopen('http://127.0.0.1:{}/v1/models', timeout=3); \
             data=json.loads(res.read().decode()); \
             ids=[m.get('id','') for m in data.get('data', [])]; \
             print(ids[0] if ids else '')\" 2>/dev/null || echo ''",
                    port_to_check_inner
                );
                let id_probe_wrapped = if docker_cfg.enabled {
                    wrap_docker_cmd(&id_probe, &container_name)
                } else {
                    id_probe
                };
                if let Ok(id_r) = session.exec_blocking(&id_probe_wrapped).await {
                    let id = id_r
                        .stdout
                        .trim()
                        .lines()
                        .last()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if !id.is_empty() {
                        served_model_id = id.clone();
                        emit_log(
                            app,
                            &run.id,
                            &format!("[ok] teacher is serving as model id: {}\n", served_model_id),
                            "stage",
                        );
                    }
                }
            }

            port_to_check = port_to_check_inner;
            run.teacher_cfg.vllm_port = port_to_check;
            if !served_model_id.trim().is_empty() {
                run.teacher_cfg.repo_id = served_model_id.clone();
            }
            teacher_endpoint = format!("http://{}:{}", cfg.ssh.host, port_to_check);
        }

        // ── 2. Generate dataset ─────────────────────────────────────────────
        run.status = RunStatus::GeneratingDataset;
        runs::save(run).await?;
        emit_progress(app, run);

        // We hit the teacher over SSH local-port-forward.
        // Cheapest cross-platform implementation: SSH-port-forward externally
        // is complex with russh; instead we open the teacher port on the droplet's
        // *public* interface (--host 0.0.0.0) and talk to it via SSH-execed curl
        // from the *droplet itself*. To keep our Rust code talking native HTTP,
        // we run vLLM listening on 0.0.0.0 inside DO and let the user's reqwest
        // client hit `http://<droplet-ip>:<port>` directly. The droplet's firewall
        // needs the port open. As a fallback for closed firewalls, the user can
        // override `teacher_endpoint` (future enhancement).
        emit_log(
            app,
            &run.id,
            &format!(
                "[teacher] endpoint = {} | model = {}\n",
                teacher_endpoint, served_model_id
            ),
            "stage",
        );

        // Quick reachability probe from *this* machine — vLLM listens on the
        // droplet's 0.0.0.0:<port> and the user has to open that port in the
        // firewall. If the port isn't open the generator would silently hang on
        // every chunk for the full 180s teacher timeout; better to fail fast with
        // a clear error.
        {
            let probe_url = format!("{}/v1/models", teacher_endpoint);
            let probe_client = reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(8))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new());
            match probe_client.get(&probe_url).send().await {
                Ok(r) if r.status().is_success() => {
                    emit_log(
                        app,
                        &run.id,
                        "[ok] teacher endpoint reachable from app\n",
                        "stage",
                    );
                }
                Ok(r) => {
                    emit_log(
                        app,
                        &run.id,
                        &format!(
                            "[warn] teacher endpoint returned HTTP {} — generation may fail\n",
                            r.status()
                        ),
                        "warn",
                    );
                }
                Err(e) => {
                    let err_msg = format!(
                        "teacher endpoint {} not reachable from this machine: {}. \
                         Make sure port {} is open in the GPU server firewall, or \
                         run vLLM with --host 0.0.0.0 and expose the port publicly.",
                        teacher_endpoint, e, port_to_check
                    );
                    return Err(AppError::pipeline(err_msg));
                }
            }
        }

        let base_prompt = run_cfg
            .prompt_template
            .clone()
            .unwrap_or_else(|| generator::DEFAULT_GENERATOR_PROMPT.to_string());

        // Resolve the effective list of topic loops. Single-topic UI fills this
        // with one element; multi-topic UI fills it with N rows.
        let topics_planned = run_cfg.effective_topics();
        if topics_planned.len() > 1 {
            emit_log(
                app,
                &run.id,
                &format!("[plan] {} focus topics queued\n", topics_planned.len()),
                "stage",
            );
            for (i, t) in topics_planned.iter().enumerate() {
                let cap = t
                    .total_questions
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "all".to_string());
                emit_log(
                    app,
                    &run.id,
                    &format!("[plan]   {}. {} (target = {})\n", i + 1, t.topic, cap),
                    "stage",
                );
            }
        } else if let Some(t) = topics_planned.first() {
            if !t.topic.is_empty() {
                emit_log(
                    app,
                    &run.id,
                    &format!("[topic] focus = {}\n", t.topic),
                    "stage",
                );
            }
            if let Some(n) = t.total_questions {
                emit_log(
                    app,
                    &run.id,
                    &format!("[target] generate up to {n} accepted Q&A pairs\n"),
                    "stage",
                );
            }
        }

        // Probe Qdrant up front so we fail loudly if the collection is empty,
        // wrong-named, or unreachable — instead of silently sitting at 0 scanned.
        emit_log(
            app,
            &run.id,
            &format!(
                "[qdrant] endpoint = {} | collection = {}\n",
                cfg.qdrant.endpoint, cfg.qdrant.collection
            ),
            "stage",
        );
        match crate::qdrant::count(&cfg.qdrant).await {
            Ok(n) => {
                emit_log(
                    app,
                    &run.id,
                    &format!("[qdrant] collection has {} points\n", n),
                    "stage",
                );
                if n == 0 {
                    return Err(AppError::pipeline(format!(
                        "Qdrant collection '{}' is empty — ingest your reviewer PDFs first \
                     (run embed_reviewer.py) before generating a dataset.",
                        cfg.qdrant.collection
                    )));
                }
            }
            Err(e) => {
                return Err(AppError::pipeline(format!(
                    "Qdrant count failed for '{}' at {}: {}. Check the endpoint, \
                 api key, and collection name in the app config.",
                    cfg.qdrant.collection, cfg.qdrant.endpoint, e
                )));
            }
        }

        // Shared across every topic loop: the aggregated pair list, scan/keep/reject
        // counters, and the dedup set of source chunk IDs.
        let pairs = Arc::new(parking_lot::Mutex::new(Vec::<GeneratedPair>::new()));
        let scanned = Arc::new(parking_lot::Mutex::new(0u64));
        let kept = Arc::new(parking_lot::Mutex::new(0u64));
        let rejected = Arc::new(parking_lot::Mutex::new(0u64));
        // chunks we've already produced a pair for — used to skip them when
        // resuming from an existing HF dataset *and* across topic loops so the
        // same chunk isn't asked twice.
        let seen_chunk_ids: Arc<parking_lot::Mutex<std::collections::HashSet<String>>> =
            Arc::new(parking_lot::Mutex::new(std::collections::HashSet::new()));
        let seen_questions: Arc<parking_lot::Mutex<Vec<String>>> =
            Arc::new(parking_lot::Mutex::new(Vec::new()));

        let mut qd_cfg = cfg.qdrant.clone();
        let max_chunks = run_cfg.max_chunks;

        // Build a per-embedder EmbeddingConfig so run_pipeline can embed topics.
        // When no explicit embedder_index is set on any topic (i.e., we fall back
        // to the first/default embedder), the pipeline uses embedder_1 (port 8101)
        // which should search ALL collections so no reviewer knowledge is missed.
        let embed_cfg = if let Some(emb) = run_cfg
            .topics
            .iter()
            .find_map(|t| t.embedder_index.and_then(|i| cfg.embedders.get(i)))
        {
            crate::ingest::EmbeddingConfig {
                provider: crate::ingest::EmbeddingProvider::Vllm,
                api_url: emb.api_url(&cfg.ssh.host),
                api_key: String::new(),
                model_id: emb.model_id.clone(),
                concurrency: Some(emb.concurrency as usize),
            }
        } else if let Some(emb) = cfg.embedders.first() {
            crate::ingest::EmbeddingConfig {
                provider: crate::ingest::EmbeddingProvider::Vllm,
                api_url: emb.api_url(&cfg.ssh.host),
                api_key: String::new(),
                model_id: emb.model_id.clone(),
                concurrency: Some(emb.concurrency as usize),
            }
        } else {
            return Err(AppError::pipeline("no embedding embedders configured"));
        };

        // When no explicit embedder_index is set on any topic (default path),
        // override the Qdrant collection to "all" so the pipeline searches
        // across ALL collections, not just a single one. This ensures the
        // teacher can retrieve chunks from any ingested domain.
        if run_cfg.topics.iter().all(|t| t.embedder_index.is_none()) {
            qd_cfg.collection = "all".to_string();
        }
        let local_jsonl_path = std::path::Path::new(&run.local_dir).join("qa_dataset.jsonl");
        let _ = fs::write(&local_jsonl_path, b"").await;

        // ── 2a. Optional seed: download existing HF dataset and prefill ────
        let ds_cfg = &run_cfg.hub_dataset;
        let hf_token_opt = cfg.hf_token.as_ref().filter(|s| !s.is_empty()).cloned();
        if ds_cfg.enabled {
            let resume_repo = if !ds_cfg.resume_from.trim().is_empty() {
                ds_cfg.resume_from.trim().to_string()
            } else if !ds_cfg.repo_id.trim().is_empty() {
                ds_cfg.repo_id.trim().to_string()
            } else {
                String::new()
            };
            if !resume_repo.is_empty() {
                emit_log(
                    app,
                    &run.id,
                    &format!("[hf-dataset] attempting to resume from {resume_repo}\n"),
                    "stage",
                );
                match seed_from_hf_dataset(&resume_repo, hf_token_opt.as_deref()).await {
                    Ok(Some(seed_jsonl)) => {
                        // Parse all pairs first (no locks held across awaits).
                        let parsed: Vec<GeneratedPair> = seed_jsonl
                            .lines()
                            .filter(|l| !l.trim().is_empty())
                            .filter_map(|l| serde_json::from_str::<GeneratedPair>(l).ok())
                            .collect();
                        let seed_count = parsed.len() as u64;

                        // Append everything to local JSONL (await — no locks held).
                        use tokio::io::AsyncWriteExt;
                        if let Ok(mut f) = tokio::fs::OpenOptions::new()
                            .append(true)
                            .create(true)
                            .open(&local_jsonl_path)
                            .await
                        {
                            for p in &parsed {
                                if let Ok(row) = serde_json::to_string(p) {
                                    let _ = f.write_all(row.as_bytes()).await;
                                    let _ = f.write_all(b"\n").await;
                                }
                            }
                        }

                        // Now mutate the shared in-memory state in tight, sync scopes.
                        {
                            let mut seen_lock = seen_chunk_ids.lock();
                            let mut pairs_lock = pairs.lock();
                            for p in parsed {
                                if !p.source_chunk_id.is_empty() {
                                    seen_lock.insert(p.source_chunk_id.clone());
                                }
                                let normalized_question =
                                    generator::normalize_question_for_dedup(&p.question);
                                if !normalized_question.is_empty() {
                                    seen_questions.lock().push(normalized_question);
                                }
                                pairs_lock.push(p);
                            }
                        }
                        *kept.lock() = seed_count;
                        run.qa_kept = seed_count;
                        runs::save(run).await.ok();
                        emit_log(
                            app,
                            &run.id,
                            &format!(
                                "[hf-dataset] seeded {seed_count} pair(s) from {resume_repo} \
                             — will skip these chunks during generation\n"
                            ),
                            "stage",
                        );
                    }
                    Ok(None) => {
                        emit_log(
                            app,
                            &run.id,
                            "[hf-dataset] no qa_dataset.jsonl found in repo — starting fresh\n",
                            "stage",
                        );
                    }
                    Err(e) => {
                        emit_log(
                            app,
                            &run.id,
                            &format!("[hf-dataset] resume skipped: {e}\n"),
                            "warn",
                        );
                    }
                }
            }
        }
        // Snapshot of every_n for the closure; 0 → disabled.
        let push_every_n: u32 = if ds_cfg.enabled { ds_cfg.every_n } else { 0 };
        let push_repo: Option<String> = if ds_cfg.enabled && !ds_cfg.repo_id.trim().is_empty() {
            Some(ds_cfg.repo_id.trim().to_string())
        } else {
            None
        };
        let push_private = ds_cfg.private;

        // ── 2b. Dataset Generation ──────────────────────────────────────────
        // Skip entirely if train_only is set — assumes the user has already
        // seeded the dataset via resume_from/repo_id in step 2a.
        if ds_cfg.enabled && ds_cfg.train_only {
            emit_log(
                app,
                &run.id,
                "[skip] train_only enabled — skipping dataset generation loop\n",
                "stage",
            );
        } else {
            // Wrap the SSH session in an Arc-friendly mutex-like channel via a tokio
            // mutex so concurrent generator tasks can serialize hub pushes.
            let push_lock = Arc::new(tokio::sync::Mutex::new(()));
            // Track the kept-count at which we last pushed, so the trigger is
            // "kept - last_pushed >= every_n" instead of "kept % every_n == 0".
            // Modulo breaks under concurrency (parallel tasks can both observe
            // kept past the divisor and skip the push) and under resume (seed
            // count rarely lands on a multiple of every_n). Seed with the current
            // kept value so the first push happens after `every_n` *new* pairs.
            let push_last_pushed = Arc::new(parking_lot::Mutex::new(*kept.lock()));

            let seen_for_skip = seen_chunk_ids.clone();
            let hf_token_for_push = hf_token_opt.clone();
            let multi_topic_mode = topics_planned.len() > 1;

            // Run one generator pass per topic. In single-topic mode this loops once;
            // in multi-topic mode each pass swaps `{topic}` and tracks its own cap
            // (`kept_this_topic`) against the topic's `total_questions`.
            for (topic_idx, topic_target) in topics_planned.iter().enumerate() {
                if cancel.load(std::sync::atomic::Ordering::SeqCst) {
                    return Err(AppError::Cancelled);
                }

                let topic_label = topic_target.topic.clone();
                let topic_value = if topic_label.is_empty() {
                    "the subject".to_string()
                } else {
                    topic_label.clone()
                };
                let effective_prompt = topic_target
                    .prompt_template
                    .clone()
                    .filter(|p| !p.trim().is_empty())
                    .unwrap_or_else(|| base_prompt.clone());
                let prompt_with_topic = effective_prompt.replace("{topic}", &topic_value);
                let topic_cap = topic_target.total_questions;
                let mut topic_tag = topic_target
                    .tag
                    .as_ref()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty());

                if topic_tag.is_none() && !topic_label.is_empty() {
                    if let Some(first_word) = topic_label.split_whitespace().next() {
                        let sanitized = first_word
                            .chars()
                            .filter(|c| c.is_alphanumeric())
                            .collect::<String>()
                            .to_lowercase();
                        if !sanitized.is_empty() {
                            topic_tag = Some(sanitized);
                        }
                    }
                }

                if multi_topic_mode {
                    let cap_str = topic_cap
                        .map(|n| n.to_string())
                        .unwrap_or_else(|| "all".to_string());
                    emit_log(
                        app,
                        &run.id,
                        &format!(
                            "[topic {}/{}] {} (target = {})\n",
                            topic_idx + 1,
                            topics_planned.len(),
                            topic_label,
                            cap_str
                        ),
                        "stage",
                    );
                }

                let gen_cfg = GeneratorConfig {
                    teacher_endpoint: teacher_endpoint.clone(),
                    // Use the model id the running vLLM actually reports — vLLM returns
                    // 404 model_not_found if we send a near-miss string (common with
                    // GGUF servings where the served id is a local path).
                    teacher_model: served_model_id.clone(),
                    prompt_template: prompt_with_topic,
                    // Strict-schema board-exam: low temp, top_p clip, mild rep penalty.
                    temperature: 0.25,
                    top_p: 0.9,
                    repetition_penalty: 1.05,
                    // Match generator.rs default — reasoning teachers truncate at 1024.
                    max_tokens: 4096,
                    max_pairs_per_chunk: run_cfg.max_pairs_per_chunk.max(1),
                    concurrency: run_cfg.concurrency.max(1),
                    api_key: None,
                };

                // Per-topic counter: how many *new* pairs this loop has accepted. Lives
                // only for the duration of this for_each_chunk call so each topic can
                // stop independently of the others.
                let kept_this_topic = Arc::new(parking_lot::Mutex::new(0u64));

                let app_clone = app.clone();
                let run_id = run.id.clone();
                let cancel_c = cancel.clone();
                let gen_clone = gen_cfg.clone();

                let pairs_w = pairs.clone();
                let scanned_w = scanned.clone();
                let kept_w = kept.clone();
                let rejected_w = rejected.clone();
                let seen_questions_w = seen_questions.clone();
                let kept_topic_for_stop = kept_this_topic.clone();
                let scanned_for_stop = scanned.clone();

                let concurrency = run_cfg.concurrency.max(1) as usize;

                // ── Topic-first retrieval ────────────────────────────────────────
                // If we have a non-empty topic, embed the topic using the active self-hosted
                // embedding model and pull only the top-K most relevant chunks via Qdrant
                // vector search.
                let use_semantic = !topic_label.is_empty();

                let should_continue_fn = || {
                    if cancel_c.load(std::sync::atomic::Ordering::SeqCst) {
                        return false;
                    }
                    if let Some(cap) = max_chunks {
                        if *scanned_for_stop.lock() >= cap as u64 {
                            return false;
                        }
                    }
                    if let Some(target) = topic_cap {
                        if *kept_topic_for_stop.lock() >= target as u64 {
                            return false;
                        }
                    }
                    true
                };

                if use_semantic {
                    // Pull a generous candidate pool. Target × 5 is a good default:
                    // most chunks will pass the topic filter (we embedded the topic,
                    // so they're already topical) but a fraction will be rejected
                    // by format/short-answer checks.
                    let target = topic_cap.unwrap_or(100);
                    let k = std::cmp::min(
                        std::cmp::max(target.saturating_mul(5), 200),
                        run_cfg.max_chunks.unwrap_or(2000),
                    ) as u32;

                    emit_log(
                        app,
                        &run.id,
                        &format!(
                            "[retrieve] semantic search for topic '{}' (top-{} via Qdrant)\n",
                            topic_label, k
                        ),
                        "stage",
                    );

                    let (mut candidates, topic_vec_for_tag_retry): (Vec<_>, Option<Vec<f32>>) =
                        match crate::serve::embed_text(
                            &embed_cfg.api_url,
                            &embed_cfg.model_id,
                            &topic_label,
                        )
                        .await
                        {
                            Ok(topic_vec) => match crate::qdrant::search(
                                &qd_cfg,
                                &topic_vec,
                                k,
                                topic_tag.as_deref(),
                            )
                            .await
                            {
                                Ok(c) => (c, Some(topic_vec)),
                                Err(e) => {
                                    return Err(AppError::pipeline(format!(
                                        "Qdrant search failed: {}",
                                        e
                                    )));
                                }
                            },
                            Err(e) => {
                                emit_log(
                                    app,
                                    &run.id,
                                    &format!(
                                        "[retrieve] topic embedding failed for '{}' after retries: {} — falling back to keyword-ranked Qdrant scroll\n",
                                        topic_label, e
                                    ),
                                    "warn",
                                );
                                let scan_limit = std::cmp::max(k as usize * 5, 5000);
                                let fallback = lexical_topic_candidates(
                                    &qd_cfg,
                                    &topic_label,
                                    k as usize,
                                    scan_limit,
                                    topic_tag.as_deref(),
                                )
                                .await?;
                                emit_log(
                                    app,
                                    &run.id,
                                    &format!(
                                        "[retrieve] fallback selected {} candidate chunks for '{}'\n",
                                        fallback.len(),
                                        topic_label
                                    ),
                                    "stage",
                                );
                                let fallback = if fallback.is_empty() && topic_tag.is_some() {
                                    emit_log(
                                        app,
                                        &run.id,
                                        "[retrieve] tag-filtered fallback returned 0 chunks — retrying without tag filter\n",
                                        "warn",
                                    );
                                    lexical_topic_candidates(
                                        &qd_cfg,
                                        &topic_label,
                                        k as usize,
                                        scan_limit,
                                        None,
                                    )
                                    .await?
                                } else {
                                    fallback
                                };
                                if fallback.is_empty() {
                                    return Err(AppError::pipeline(format!(
                                        "topic embedding failed for '{}' and fallback retrieval found no chunks: {}",
                                        topic_label, e
                                    )));
                                }
                                (fallback, None)
                            }
                        };
                    emit_log(
                        app,
                        &run.id,
                        &format!(
                            "[retrieve] got {} candidate chunks for '{}'\n",
                            candidates.len(),
                            topic_label
                        ),
                        "stage",
                    );

                    // If a tag filter was supplied but matched nothing, the most
                    // likely cause is a tag mismatch (the user typed a tag that no
                    // ingested batch used). Falling back to an unfiltered search
                    // beats aborting the whole run with zero pairs — the teacher's
                    // SKIP step will still drop off-topic chunks downstream.
                    if candidates.is_empty()
                        && topic_tag.is_some()
                        && topic_vec_for_tag_retry.is_some()
                    {
                        let tag_str = topic_tag.as_deref().unwrap_or("");
                        emit_log(
                    app,
                    &run.id,
                    &format!(
                        "[retrieve] tag filter '{}' matched 0 points — retrying without the tag filter\n",
                        tag_str
                    ),
                    "stage",
                );
                        match crate::qdrant::search(
                            &qd_cfg,
                            topic_vec_for_tag_retry.as_ref().unwrap(),
                            k,
                            None,
                        )
                        .await
                        {
                            Ok(c) => {
                                emit_log(
                                    app,
                                    &run.id,
                                    &format!(
                                        "[retrieve] fallback got {} candidate chunks for '{}'\n",
                                        c.len(),
                                        topic_label
                                    ),
                                    "stage",
                                );
                                candidates = c;
                            }
                            Err(e) => {
                                return Err(AppError::pipeline(format!(
                                    "Qdrant fallback search failed: {}",
                                    e
                                )));
                            }
                        }
                    }

                    generator::for_each_in(
                candidates,
                concurrency,
                should_continue_fn,
            |chunk| {
                let gen_c = gen_clone.clone();
                let pairs_in = pairs_w.clone();
                let scanned_in = scanned_w.clone();
                let kept_in = kept_w.clone();
                let kept_topic_in = kept_this_topic.clone();
                let rej_in = rejected_w.clone();
                let app_in = app_clone.clone();
                let run_in = run_id.clone();
                let path_in = local_jsonl_path.clone();
                let seen_in = seen_for_skip.clone();
                let seen_questions_in = seen_questions_w.clone();
                let push_repo_in = push_repo.clone();
                let push_lock_in = push_lock.clone();
                let push_last_pushed_in = push_last_pushed.clone();
                let hf_token_in = hf_token_for_push.clone();
                // Clone topic_label for the async move so the outer variable
                // stays accessible after for_each_chunk returns.
                let topic_label_in = topic_label.clone();
                let topic_cap_in = topic_cap;
                async move {
                    *scanned_in.lock() += 1;
                    // Skip chunks we've already produced pairs for (resume or
                    // an earlier topic in this same run).
                    if seen_in.lock().contains(&chunk.id) {
                        emit_log(
                            &app_in,
                            &run_in,
                            &format!("[skip] chunk {} already covered\n", chunk.id),
                            "skip",
                        );
                        return Ok(());
                    }
                    let prompt = generator::build_prompt(&gen_c.prompt_template, &chunk);
                    match generator::ask_teacher(&gen_c, &prompt).await {
                        Ok(raw) => {
                            match generator::parse_pair(&raw, &chunk) {
                            Ok(mut pair) => {
                                pair.topic = topic_label_in.clone();
                                let duplicate_reason = {
                                    let mut question_lock = seen_questions_in.lock();
                                    if let Some(reason) = generator::duplicate_question_reason(
                                        &pair.question,
                                        question_lock.as_slice(),
                                    ) {
                                        Some(reason)
                                    } else {
                                        let normalized = generator::normalize_question_for_dedup(
                                            &pair.question,
                                        );
                                        if !normalized.is_empty() {
                                            question_lock.push(normalized);
                                        }
                                        None
                                    }
                                };
                                if let Some(reason) = duplicate_reason {
                                    *rej_in.lock() += 1;
                                    emit_log(
                                        &app_in,
                                        &run_in,
                                        &format!(
                                            "[reject] chunk {} — {}\n",
                                            chunk.id, reason
                                        ),
                                        "reject",
                                    );
                                    return Ok(());
                                }
                                *kept_in.lock() += 1;
                                *kept_topic_in.lock() += 1;
                                // Append to local JSONL
                                let row = serde_json::to_string(&pair).unwrap_or_default();
                                use tokio::io::AsyncWriteExt;
                                if let Ok(mut f) = tokio::fs::OpenOptions::new()
                                    .append(true)
                                    .create(true)
                                    .open(&path_in)
                                    .await
                                  {
                                    let _ = f.write_all(row.as_bytes()).await;
                                    let _ = f.write_all(b"\n").await;
                                }
                                let current_kept = *kept_in.lock();
                                let current_topic_kept = *kept_topic_in.lock();
                                let preview = if let Some(target) = topic_cap_in {
                                    if target > 0 {
                                        let pct = (current_topic_kept as f64 / target as f64) * 100.0;
                                        format!(
                                            "[pair {} | topic: {}/{} ({:.1}%)] Q: {}\n",
                                            current_kept,
                                            current_topic_kept,
                                            target,
                                            pct,
                                            pair.question.chars().take(120).collect::<String>()
                                        )
                                    } else {
                                        format!(
                                            "[pair {}] Q: {}\n",
                                            current_kept,
                                            pair.question.chars().take(120).collect::<String>()
                                        )
                                    }
                                } else {
                                    format!(
                                        "[pair {}] Q: {}\n",
                                        current_kept,
                                        pair.question.chars().take(120).collect::<String>()
                                    )
                                };
                                emit_log(&app_in, &run_in, &preview, "generated");
                                seen_in.lock().insert(pair.source_chunk_id.clone());
                                pairs_in.lock().push(pair);

                                // ── Auto-checkpoint to HF dataset every N ─
                                // Trigger when `current_kept - last_pushed >= every_n`.
                                // This is robust to:
                                //   • Concurrency — modulo can be skipped if two
                                //     parallel tasks both observe kept past the
                                //     divisor; the diff cannot.
                                //   • Resume — seed_count is usually not a
                                //     multiple of every_n, so modulo would push
                                //     after a partial interval; the diff seeds
                                //     last_pushed to the seed count and pushes
                                //     after a full `every_n` *new* pairs.
                                if let Some(repo) = push_repo_in.as_ref() {
                                    if push_every_n > 0 {
                                        let _guard = push_lock_in.lock().await;
                                        // Re-check under the lock so only one
                                        // concurrent task wins the trigger.
                                        let kept_now = *kept_in.lock();
                                        let should_push = {
                                            let mut last = push_last_pushed_in.lock();
                                            if kept_now.saturating_sub(*last)
                                                >= push_every_n as u64
                                            {
                                                *last = kept_now;
                                                true
                                            } else {
                                                false
                                            }
                                        };
                                        if should_push {
                                            // Read the current local jsonl and ship it to droplet, then push.
                                            if let Ok(snapshot) =
                                                tokio::fs::read_to_string(&path_in).await
                                            {
                                                emit_log(
                                                    &app_in,
                                                    &run_in,
                                                    &format!(
                                                        "[hf-dataset] checkpoint push @ {kept_now} pairs → {repo}\n"
                                                    ),
                                                    "stage",
                                                );
                                                if let Err(e) = push_jsonl_to_hf_dataset(
                                                    repo,
                                                    push_private,
                                                    hf_token_in.as_deref(),
                                                    &snapshot,
                                                    kept_now,
                                                )
                                                .await
                                                {
                                                    // On failure, roll last_pushed
                                                    // back so the next pair retriggers.
                                                    *push_last_pushed_in.lock() =
                                                        kept_now.saturating_sub(push_every_n as u64);
                                                    emit_log(
                                                        &app_in,
                                                        &run_in,
                                                        &format!("[hf-dataset] push failed: {e}\n"),
                                                        "warn",
                                                    );
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            Err(reason) => {
                                *rej_in.lock() += 1;
                                // Surface *why* the parser rejected — and on
                                // the first few rejects show a raw response
                                // preview so the user can spot a format issue.
                                let current_rej = *rej_in.lock();
                                let preview: String = raw.chars().take(220).collect();
                                let preview = preview.replace('\n', " ⏎ ");
                                if current_rej <= 5 {
                                    emit_log(
                                        &app_in,
                                        &run_in,
                                        &format!(
                                            "[reject] chunk {} — {} | raw: {}…\n",
                                            chunk.id,
                                            reason.label(),
                                            preview
                                        ),
                                        "reject",
                                    );
                                } else {
                                    emit_log(
                                        &app_in,
                                        &run_in,
                                        &format!(
                                            "[reject] chunk {} — {}\n",
                                            chunk.id,
                                            reason.label()
                                        ),
                                        "reject",
                                    );
                                }
                            }
                            }
                        }
                        Err(e) => {
                            *rej_in.lock() += 1;
                            emit_log(
                                &app_in,
                                &run_in,
                                &format!("[teacher-err] {}\n", e),
                                "error",
                            );
                        }
                    }
                    let _ = app_in.emit(
                        "pipeline://progress",
                        ProgressEvent {
                            run_id: run_in.clone(),
                            scanned: *scanned_in.lock(),
                            kept: *kept_in.lock(),
                            rejected: *rej_in.lock(),
                            status: RunStatus::GeneratingDataset,
                        },
                    );
                    Ok(())
                }
            },
        )
        .await?;
                } else {
                    emit_log(
                        app,
                        &run.id,
                        "[retrieve] no topic focus set — scrolling all chunks page-by-page from Qdrant\n",
                        "stage",
                    );

                    let mut offset: Option<serde_json::Value> = None;
                    let mut page_idx = 0;

                    loop {
                        if !should_continue_fn() {
                            break;
                        }

                        let page = match crate::qdrant::scroll(
                            &qd_cfg,
                            256,
                            offset.take(),
                            topic_tag.as_deref(),
                        )
                        .await
                        {
                            Ok(p) => p,
                            Err(e) => {
                                return Err(AppError::pipeline(format!(
                                    "scroll all failed: {}",
                                    e
                                )));
                            }
                        };

                        if page.chunks.is_empty() {
                            break;
                        }

                        page_idx += 1;
                        emit_log(
                            app,
                            &run.id,
                            &format!(
                                "[retrieve] processing chunk page {} ({} chunks)\n",
                                page_idx,
                                page.chunks.len()
                            ),
                            "stage",
                        );

                        offset = page.next_offset;

                        generator::for_each_in(
                            page.chunks,
                            concurrency,
                            should_continue_fn,
                            |chunk| {
                                let gen_c = gen_clone.clone();
                                let pairs_in = pairs_w.clone();
                                let scanned_in = scanned_w.clone();
                                let kept_in = kept_w.clone();
                                let kept_topic_in = kept_this_topic.clone();
                                let rej_in = rejected_w.clone();
                                let seen_questions_in = seen_questions_w.clone();
                                let app_in = app_clone.clone();
                                let run_in = run_id.clone();
                                let path_in = local_jsonl_path.clone();
                                let seen_in = seen_for_skip.clone();
                                let push_repo_in = push_repo.clone();
                                let push_lock_in = push_lock.clone();
                                let push_last_pushed_in = push_last_pushed.clone();
                                let hf_token_in = hf_token_for_push.clone();
                                let topic_label_in = topic_label.clone();
                                let topic_cap_in = topic_cap;
                                async move {
                                    *scanned_in.lock() += 1;
                                    if seen_in.lock().contains(&chunk.id) {
                                        emit_log(
                                            &app_in,
                                            &run_in,
                                            &format!("[skip] chunk {} already covered\n", chunk.id),
                                            "skip",
                                        );
                                        return Ok(());
                                    }
                                    let prompt = generator::build_prompt(&gen_c.prompt_template, &chunk);
                                    match generator::ask_teacher(&gen_c, &prompt).await {
                                        Ok(raw) => {
                                            match generator::parse_pair(&raw, &chunk) {
                                                Ok(mut pair) => {
                                                    pair.topic = topic_label_in.clone();
                                                    let duplicate_reason = {
                                                        let mut question_lock = seen_questions_in.lock();
                                                        if let Some(reason) = generator::duplicate_question_reason(
                                                            &pair.question,
                                                            question_lock.as_slice(),
                                                        ) {
                                                            Some(reason)
                                                        } else {
                                                            let normalized = generator::normalize_question_for_dedup(
                                                                &pair.question,
                                                            );
                                                            if !normalized.is_empty() {
                                                                question_lock.push(normalized);
                                                            }
                                                            None
                                                        }
                                                    };
                                                    if let Some(reason) = duplicate_reason {
                                                        *rej_in.lock() += 1;
                                                        emit_log(
                                                            &app_in,
                                                            &run_in,
                                                            &format!(
                                                                "[reject] chunk {} — {}\n",
                                                                chunk.id, reason
                                                            ),
                                                            "reject",
                                                        );
                                                        return Ok(());
                                                    }
                                                    *kept_in.lock() += 1;
                                                    *kept_topic_in.lock() += 1;
                                                    let row = serde_json::to_string(&pair).unwrap_or_default();
                                                    use tokio::io::AsyncWriteExt;
                                                    if let Ok(mut f) = tokio::fs::OpenOptions::new()
                                                        .append(true)
                                                        .create(true)
                                                        .open(&path_in)
                                                        .await
                                                    {
                                                        let _ = f.write_all(row.as_bytes()).await;
                                                        let _ = f.write_all(b"\n").await;
                                                    }
                                                    let current_kept = *kept_in.lock();
                                                    let current_topic_kept = *kept_topic_in.lock();
                                                    let preview = if let Some(target) = topic_cap_in {
                                                        if target > 0 {
                                                            let pct = (current_topic_kept as f64 / target as f64) * 100.0;
                                                            format!(
                                                                "[pair {} | topic: {}/{} ({:.1}%)] Q: {}\n",
                                                                current_kept,
                                                                current_topic_kept,
                                                                target,
                                                                pct,
                                                                pair.question.chars().take(120).collect::<String>()
                                                            )
                                                        } else {
                                                            format!(
                                                                "[pair {}] Q: {}\n",
                                                                current_kept,
                                                                pair.question.chars().take(120).collect::<String>()
                                                            )
                                                        }
                                                    } else {
                                                        format!(
                                                            "[pair {}] Q: {}\n",
                                                            current_kept,
                                                            pair.question.chars().take(120).collect::<String>()
                                                        )
                                                    };
                                                    emit_log(&app_in, &run_in, &preview, "generated");
                                                    seen_in.lock().insert(pair.source_chunk_id.clone());
                                                    pairs_in.lock().push(pair);
                                                    if let Some(repo) = push_repo_in.as_ref() {
                                                        if push_every_n > 0 {
                                                            let _guard = push_lock_in.lock().await;
                                                            let kept_now = *kept_in.lock();
                                                            let should_push = {
                                                                let mut last = push_last_pushed_in.lock();
                                                                if kept_now.saturating_sub(*last) >= push_every_n as u64 {
                                                                    *last = kept_now;
                                                                    true
                                                                } else {
                                                                    false
                                                                }
                                                            };
                                                            if should_push {
                                                                if let Ok(snapshot) = tokio::fs::read_to_string(&path_in).await {
                                                                    emit_log(
                                                                        &app_in,
                                                                        &run_in,
                                                                        &format!("[hf-dataset] checkpoint push @ {kept_now} pairs → {repo}\n"),
                                                                        "stage",
                                                                    );
                                                                    let _ = push_jsonl_to_hf_dataset(
                                                                        repo,
                                                                        push_private,
                                                                        hf_token_in.as_deref(),
                                                                        &snapshot,
                                                                        kept_now,
                                                                    ).await;
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                                Err(reason) => {
                                                    *rej_in.lock() += 1;
                                                    let current_rej = *rej_in.lock();
                                                    let preview: String = raw.chars().take(220).collect();
                                                    let preview = preview.replace('\n', " ⏎ ");
                                                    if current_rej <= 5 {
                                                        emit_log(&app_in, &run_in, &format!("[reject] chunk {} — {} | raw: {}…\n", chunk.id, reason.label(), preview), "reject");
                                                    } else {
                                                        emit_log(&app_in, &run_in, &format!("[reject] chunk {} — {}\n", chunk.id, reason.label()), "reject");
                                                    }
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            *rej_in.lock() += 1;
                                            emit_log(&app_in, &run_in, &format!("[teacher-err] {}\n", e), "error");
                                        }
                                    }
                                    let _ = app_in.emit(
                                        "pipeline://progress",
                                        ProgressEvent {
                                            run_id: run_in.clone(),
                                            scanned: *scanned_in.lock(),
                                            kept: *kept_in.lock(),
                                            rejected: *rej_in.lock(),
                                            status: RunStatus::GeneratingDataset,
                                        },
                                    );
                                    Ok(())
                                }
                            }
                        ).await?;

                        if offset.is_none() {
                            break;
                        }
                    }
                }

                if multi_topic_mode || !topic_label.is_empty() {
                    let got = *kept_this_topic.lock();
                    let total_so_far = *kept.lock();
                    // Record in the run's per-topic stats so the UI can show breakdown.
                    run.topic_stats.insert(
                        if topic_label.is_empty() {
                            "(general)".to_string()
                        } else {
                            topic_label.clone()
                        },
                        got,
                    );
                    emit_log(
                        app,
                        &run.id,
                        &format!(
                            "[topic {}/{}] '{}' → {} pair(s) | running total: {}\n",
                            topic_idx + 1,
                            topics_planned.len(),
                            if topic_label.is_empty() {
                                "(general)"
                            } else {
                                &topic_label
                            },
                            got,
                            total_so_far
                        ),
                        "stage",
                    );
                }
            }
        }

        let final_scanned = *scanned.lock();
        let final_kept = *kept.lock();
        let final_rej = *rejected.lock();
        run.qa_total = final_scanned;
        run.qa_kept = final_kept;
        run.qa_rejected = final_rej;
        runs::save(run).await?;

        // Final HF dataset push (always at end if enabled, regardless of every_n).
        if let Some(repo) = push_repo.as_ref() {
            if let Ok(snapshot) = tokio::fs::read_to_string(&local_jsonl_path).await {
                if !snapshot.trim().is_empty() {
                    emit_log(
                        app,
                        &run.id,
                        &format!("[hf-dataset] final push ({final_kept} pairs) → {repo}\n"),
                        "stage",
                    );
                    if let Err(e) = push_jsonl_to_hf_dataset(
                        repo,
                        push_private,
                        hf_token_opt.as_deref(),
                        &snapshot,
                        final_kept,
                    )
                    .await
                    {
                        emit_log(
                            app,
                            &run.id,
                            &format!("[hf-dataset] final push failed: {e}\n"),
                            "warn",
                        );
                    }
                }
            }
        }

        if cancel.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(AppError::Cancelled);
        }
        if final_kept == 0 && !ds_cfg.train_only {
            return Err(AppError::pipeline(
                "no valid Q&A pairs generated — aborting before training",
            ));
        }

        // ── 3. Teacher unload — free VRAM for Student ───────────────────────
        let zrald_keeps_local_reward_teacher = run.lora.method.trim().eq_ignore_ascii_case("zrald")
            && run.lora.zrald_reward_endpoint.trim().is_empty();
        if zrald_keeps_local_reward_teacher {
            emit_log(
                app,
                &run.id,
                "[zrald] preserving the local teacher as the reward model; use an external reward endpoint to free this VRAM before training\n",
                "warn",
            );
        } else if let Some(session) = session_opt.as_ref() {
            emit_log(
                app,
                &run.id,
                "[stage] unloading teacher and freeing VRAM\n",
                "stage",
            );
            let pkill_body = "pkill -f '[v]llm' 2>/dev/null; \
                          pkill -9 -f '[v]llm' 2>/dev/null; \
                          pkill -f 'sglang' 2>/dev/null; \
                          pkill -9 -f 'sglang' 2>/dev/null; \
                          true";

            // First sweep on the host — covers any vLLM/sglang started outside docker.
            let _ = session.exec_blocking(pkill_body).await;

            // Then sweep across EVERY running container to ensure VRAM is clear.
            if docker_cfg.enabled {
                if let Ok(ps_r) = session
                    .exec_blocking("docker ps --format '{{.Names}}'")
                    .await
                {
                    let names: Vec<String> = ps_r
                        .stdout
                        .lines()
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect();
                    for cname in &names {
                        if cname != &container_name
                            && (cname.contains("vllm")
                                || cname.contains("sglang")
                                || cname.contains("paddleocr"))
                        {
                            emit_log(
                                app,
                                &run.id,
                                &format!("[GPU CLEANUP] stopping and removing container '{}' to free VRAM...\n", cname),
                                "stage",
                            );
                            let stop_cmd = format!(
                                "docker stop {} 2>/dev/null; docker rm {} 2>/dev/null; true",
                                cname, cname
                            );
                            let _ = session.exec_blocking(&stop_cmd).await;
                        } else {
                            let inner = wrap_docker_cmd(pkill_body, cname);
                            let _ = session.exec_blocking(&inner).await;
                        }
                    }
                }
            }
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;

            // Delete teacher model files from /root/hf-cache to free up disk space
            let repo_clean = run_cfg
                .teacher
                .repo_id
                .split(':')
                .next()
                .unwrap_or(&run_cfg.teacher.repo_id);
            let folder_name = format!("models--{}", repo_clean.replace('/', "--"));
            emit_log(
                app,
                &run.id,
                &format!(
                    "[stage] deleting teacher model files from cache: {}\n",
                    folder_name
                ),
                "stage",
            );
            let mut rm_cmd = format!(
            "rm -rf /root/hf-cache/hub/{folder_name} /root/.cache/huggingface/hub/{folder_name} || true",
            folder_name = folder_name
        );
            if docker_cfg.enabled {
                rm_cmd = wrap_docker_cmd(&rm_cmd, &container_name);
            }
            let _ = session.exec_blocking(&rm_cmd).await;
        } else {
            emit_log(
                app,
                &run.id,
                "[stage] no droplet teacher to unload (Featherless local generation)\n",
                "stage",
            );
        }

        // ── 4. Compile dataset, upload ──────────────────────────────────────
        run.status = RunStatus::DatasetReady;
        runs::save(run).await?;
        emit_progress(app, run);

        let pairs_final = pairs.lock().clone();
        let (train_jsonl, val_jsonl) = llamafactory::build_jsonl(&pairs_final);
        let run_name = format!("ft_{}", &run.id[..8]);
        let has_hf_repos = !llamafactory::hub_dataset_repos(run).is_empty();
        let info = if run.hub_dataset.enabled && has_hf_repos {
            llamafactory::dataset_info_hf(run)
        } else {
            llamafactory::dataset_info(&run_name)
        };

        if let Some(session) = session_opt.as_ref() {
            let remote_data = format!("{}/data", run.remote_dir);
            write_file_auto(
                session,
                cfg.docker.enabled,
                &container_name,
                &format!("{}/train.jsonl", remote_data),
                &train_jsonl,
            )
            .await?;
            write_file_auto(
                session,
                cfg.docker.enabled,
                &container_name,
                &format!("{}/val.jsonl", remote_data),
                &val_jsonl,
            )
            .await?;
            write_file_auto(
                session,
                cfg.docker.enabled,
                &container_name,
                &format!("{}/dataset_info.json", remote_data),
                &info,
            )
            .await?;
            if let Ok(qa_jsonl) = fs::read_to_string(&local_jsonl_path).await {
                write_file_auto(
                    session,
                    cfg.docker.enabled,
                    &container_name,
                    &format!("{}/qa_dataset.jsonl", remote_data),
                    &qa_jsonl,
                )
                .await?;
            }
        } else {
            emit_log(app, &run.id, "[stage] wrote dataset artifacts locally; remote training files skipped in generate_only mode\n", "stage");
        }

        // Save a local copy too.
        let _ = fs::write(
            std::path::Path::new(&run.local_dir).join("train.jsonl"),
            &train_jsonl,
        )
        .await;
        let _ = fs::write(
            std::path::Path::new(&run.local_dir).join("val.jsonl"),
            &val_jsonl,
        )
        .await;

        // Mark dataset as durably prepared. Future resumes will skip generation.
        run.dataset_ready = true;
        runs::save(run).await?;
    } // end if !skip_dataset

    if run_cfg.generate_only {
        emit_log(
            app,
            &run.id,
            "[stage] dataset generation completed successfully (generate_only)\n",
            "stage",
        );
        run.status = RunStatus::DatasetReady;
        runs::save(run).await?;
        emit_progress(app, run);
        return Ok(());
    }

    let session = session_opt
        .as_ref()
        .ok_or_else(|| AppError::pipeline("SSH session unavailable for training"))?;

    // Detect if a checkpoint already exists in output_dir → forces resume_from_checkpoint.
    let ckpt_inner = format!(
        "ls -1d {}/lora/checkpoint-* 2>/dev/null | head -n1 || true",
        run.remote_dir
    );
    let ckpt_cmd = if docker_cfg.enabled {
        wrap_docker_cmd(&ckpt_inner, &container_name)
    } else {
        ckpt_inner
    };
    let ckpt_probe = session.exec_blocking(&ckpt_cmd).await?;
    let has_checkpoint = !ckpt_probe.stdout.trim().is_empty();
    let do_resume = resume && has_checkpoint;
    if has_checkpoint {
        emit_log(
            app,
            &run.id,
            &format!("[resume] found checkpoint: {}\n", ckpt_probe.stdout.trim()),
            "stage",
        );
    }

    // (Re)emit train.yaml — hub config and resume flag may have changed since
    // a previous attempt.
    // AMD ROCm GPUs don't support bf16 → use fp16 instead.
    let is_rocm = container_name.to_lowercase().contains("rocm");
    // Use `run.lora` (mutated by guide auto-apply) rather than `run_cfg.lora`.
    let train_yaml = llamafactory::train_yaml(run, &run.lora, &run_cfg.hub, do_resume, is_rocm)?;
    write_file_auto(
        &session,
        cfg.docker.enabled,
        &container_name,
        &format!("{}/train.yaml", run.remote_dir),
        &train_yaml,
    )
    .await?;
    let _ = fs::write(
        std::path::Path::new(&run.local_dir).join("train.yaml"),
        &train_yaml,
    )
    .await;

    // (Re)emit dataset_info.json on training start to ensure it is up-to-date
    // (especially if we skipped generation step because the dataset is already prepared).
    let run_name = format!("ft_{}", &run.id[..8]);
    let has_hf_repos = !llamafactory::hub_dataset_repos(run).is_empty();
    let info = if run.hub_dataset.enabled && has_hf_repos {
        llamafactory::dataset_info_hf(run)
    } else {
        llamafactory::dataset_info(&run_name)
    };
    write_file_auto(
        &session,
        cfg.docker.enabled,
        &container_name,
        &format!("{}/data/dataset_info.json", run.remote_dir),
        &info,
    )
    .await?;

    // ── 4b. Hugging Face Hub: login + pre-create repo (best-effort) ─────
    if run_cfg.hub.enabled && !run_cfg.hub.model_id.trim().is_empty() {
        if let Some(tok) = cfg.hf_token.as_ref().filter(|s| !s.is_empty()) {
            emit_log(
                app,
                &run.id,
                &format!(
                    "[stage] HF Hub: login + ensure repo {}\n",
                    run_cfg.hub.model_id
                ),
                "stage",
            );
            let private_flag = if run_cfg.hub.private { "--private" } else { "" };
            // huggingface_hub ships `huggingface-cli` (older) or `hf` (newer).
            // both login and repo create are idempotent. We mask the token in logs.
            let mut hub_cmd = format!(
                "pip install -q -U 'huggingface_hub[cli]<1.0' >/dev/null 2>&1 || true; \
                 (command -v hf >/dev/null 2>&1 && hf auth login --token {tok} >/dev/null 2>&1) || \
                 (huggingface-cli login --token {tok} --add-to-git-credential >/dev/null 2>&1) || \
                   echo 'login failed (continuing — Trainer will still try with HF_TOKEN env)'; \
                 (command -v hf >/dev/null 2>&1 && hf repo create {repo} --repo-type model {priv} 2>&1) || \
                 (huggingface-cli repo create {repo} --type model {priv} -y 2>&1) || true",
                tok = tok,
                repo = run_cfg.hub.model_id,
                priv = private_flag,
            );
            if docker_cfg.enabled {
                hub_cmd = wrap_docker_cmd(&hub_cmd, &container_name);
            }
            let r = session.exec_blocking(&hub_cmd).await?;
            // Don't append the raw stdout (contains nothing sensitive after CLI
            // masks it, but be safe).
            let safe = r
                .stdout
                .replace(tok.as_str(), "***")
                .replace(&r.stderr.replace(tok.as_str(), "***"), "");
            if !safe.trim().is_empty() {
                emit_log(app, &run.id, &format!("[hub] {}\n", safe.trim()), "stage");
            }
        } else {
            emit_log(
                app,
                &run.id,
                "[warn] hub.enabled = true but no HF token configured — push will fail\n",
                "warn",
            );
        }
    }

    // ── 5. Launch LLaMA-Factory training ────────────────────────────────
    run.status = RunStatus::Training;
    runs::save(run).await?;
    emit_progress(app, run);

    let hf_export = cfg
        .hf_token
        .as_ref()
        .filter(|s| !s.is_empty())
        .map(|t| {
            format!(
                "export HF_TOKEN={} HUGGING_FACE_HUB_TOKEN={}; ",
                sh_quote(t),
                sh_quote(t)
            )
        })
        .unwrap_or_default();
    // Read from `run.lora` so guide-applied defaults (and method auto-switch)
    // take effect for the rest of the training command builder.
    let method = run.lora.method.trim().to_lowercase();
    let custom_method = method == "custom";
    let grpo_method = method == "grpo";
    let zrald_method = method == "zrald";
    let mut train_cmd = if custom_method {
        build_custom_train_cmd(run, &run.lora, &hf_export)?
    } else if zrald_method {
        build_zrald_train_cmd(run, &run.lora, &hf_export)?
    } else if grpo_method {
        build_grpo_train_cmd(run, &run.lora, &hf_export)?
    } else {
        // QLoRA needs bitsandbytes for 4-bit quantization. Unsloth touches
        // bitsandbytes symbols at import time (bnb.functional.get_ptr,
        // bitsandbytes.cextension.ROCM_WARP_SIZE_64, bitsandbytes.nn.modules,
        // etc.), so it's required for *any* unsloth method too.
        let needs_bnb = method == "qlora" || method == "loftq" || method == "unsloth";
        // bitsandbytes 0.45–0.49.2 has a 4-bit decode NaN bug on every AMD GPU
        // (per unsloth AMD install docs). For unsloth on ROCm we install the
        // pre-release wheel via {unsloth_install} below (with --no-deps so pip
        // doesn't choke on the unconventional version string) and skip the
        // PyPI install here. For plain QLoRA/LoftQ we pin to >=0.49.1.
        let bnb_dep = if needs_bnb && method != "unsloth" {
            "'bitsandbytes>=0.49.1' "
        } else {
            ""
        };

        // Unsloth install is finicky. A plain `pip install unsloth` re-pins
        // transformers/peft/datasets/trl into ranges LLaMA-Factory 0.9.4 may not
        // like, but installing with --no-deps leaves unsloth missing peft/trl/
        // accelerate/bitsandbytes/sentencepiece at *import* time and the training
        // process crashes silently before producing a checkpoint.
        //
        // Strategy: install unsloth normally (so it brings in its companions), then
        // re-install our pinned versions of huggingface-hub, transformers, and
        // llamafactory as the *last* step — pip resolves transformers to a version
        // that satisfies LLaMA-Factory's pin and unsloth's tolerant range. On ROCm
        // we skip the upstream PyPI install (it pulls bitsandbytes/triton wheels
        // that don't build on AMD) and install only what's importable on ROCm.
        //
        // The probe imports a module that pulls in the heavy deps so a stale
        // install (missing peft, trl, etc.) is detected on resume and re-fixed.
        // It also checks that peft has `ensure_weight_tying` — required by recent
        // unsloth releases but absent in peft<=0.17.1 (which LLaMA-Factory 0.9.4's
        // declared dep range would otherwise pin us to). If the kwarg is missing
        // we re-run the install block, which ends by force-upgrading peft past
        // LF's cap.
        let unsloth_probe = "python3 -c 'import unsloth; from unsloth import FastLanguageModel; \
                         import inspect; from peft import LoraConfig; \
                         assert \"ensure_weight_tying\" in inspect.signature(LoraConfig).parameters' \
                         >/dev/null 2>&1";
        // Torch must be the ROCm/HIP build. The default PyPI wheel is CUDA-only;
        // installing it (directly or transitively via bitsandbytes/unsloth) leaves
        // `torch.version.hip is None`, which makes unsloth raise
        // "Unsloth detected signs of an AMD ROCm GPU, but your current PyTorch
        // build has no usable HIP accelerator" right after dataset prep.
        // This probe exits 0 only when a HIP-enabled torch is importable.
        let torch_hip_probe = "python3 -c 'import torch,sys; \
                         sys.exit(0 if getattr(torch.version,\"hip\",None) else 1)' \
                         >/dev/null 2>&1";
        // AMD ROCm unsloth install — follows the upstream guide at
        // https://unsloth.ai/docs/get-started/install/amd. Key points:
        //  • torch must be from the ROCm wheel index, version <2.11 (2.11+
        //    only has ROCm 7.2 wheels and would break the install).
        //  • `unsloth[amd]` installs the AMD-specific extras; we do NOT pass
        //    --no-deps here because the [amd] marker IS a dependency and
        //    --no-deps would skip it entirely.
        //  • bitsandbytes ≤0.49.2 has a 4-bit decode NaN bug on every AMD
        //    GPU; use the pre-release wheel from the bnb GitHub releases
        //    (continuous-release_main). Falls back to 0.49.1+ if the wheel
        //    URL is unreachable. Installed with --no-deps because pip/uv
        //    rejects the non-standard version string in the filename.
        //  • The rocm/vllm container often has ROCm torch only inside vLLM's
        //    venv, leaving system-Python torch absent or a CUDA wheel; the
        //    torch_probe + reinstall makes sure system-Python has HIP torch.
        let bnb_amd_wheel = "https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_main/bitsandbytes-1.33.7.preview-py3-none-manylinux_2_24_x86_64.whl";
        let (unsloth_check, unsloth_install) = if method == "unsloth" {
            (
            // Re-check torch HIP every run — if a previous step (or a stale
            // image) replaced torch with a CUDA wheel we want to repair before
            // training starts.
            format!(" && {} && {}", torch_hip_probe, unsloth_probe),
            format!(
                "&& {{ export UNSLOTH_IS_ROCM=1 PYTORCH_ROCM_ARCH=${{PYTORCH_ROCM_ARCH:-gfx1100}}; \
                      ({torch_probe}) || \
                      (pip install --no-cache-dir --upgrade --force-reinstall \
                                                --index-url https://download.pytorch.org/whl/rocm7.0 \
                                                'torch>=2.4,<2.11.0' 'torchvision<0.26.0' 'torchaudio<2.11.0'); \
                      {probe} || \
                      (pip install --no-cache-dir 'unsloth[amd]' 'unsloth_zoo' && \
                       (pip install --force-reinstall --no-cache-dir --no-deps '{bnb_wheel}' || \
                        pip install --force-reinstall --no-cache-dir --no-deps 'bitsandbytes>=0.49.1') && \
                       pip install --no-cache-dir 'peft>=0.19,<0.20' 'trl<0.10.0' 'accelerate>=0.34.0' \
                                                'sentencepiece>=0.2.0' 'datasets>=2.16.0' \
                                                'tyro' 'protobuf' 'hf_transfer' 'psutil' || true); }} ",
                torch_probe = torch_hip_probe,
                probe = unsloth_probe,
                bnb_wheel = bnb_amd_wheel,
            ),
        )
        } else {
            (String::new(), String::new())
        };
        let unsloth_env = if method == "unsloth" {
            "export UNSLOTH_IS_ROCM=1 PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH:-gfx1100}; "
        } else {
            ""
        };
        // GaLore is vendored inside LLaMA-Factory; BAdam and APOLLO are separate
        // PyPI packages that must be present at import time. On ROCm these are
        // pure-PyTorch packages (no CUDA-specific extensions), so a regular pip
        // install is the right path. If the install fails LF crashes with a clear
        // ImportError — better than the silent default-fallback we had before.
        let extra_optimizer = match method.as_str() {
            "badam" => " && pip install --no-cache-dir 'badam>=1.2.1'",
            "galore" => " && pip install --no-cache-dir 'galore-torch'",
            _ => "",
        };

        // Install order matters when method == unsloth:
        //   1) install/refresh unsloth (it may bump transformers / peft / trl)
        //   2) install LLaMA-Factory + our pinned huggingface-hub/transformers,
        //      which re-locks transformers to a version both LF 0.9.4 and unsloth
        //      can tolerate.
        //   3) force-upgrade peft past LF 0.9.4's `peft<=0.17.1` cap. Recent
        //      unsloth releases call `LoraConfig(ensure_weight_tying=...)`, a
        //      kwarg only present in peft>=0.19. We install peft last with
        //      `--no-deps` so it overrides LF's metadata constraint without
        //      disturbing the rest of the resolved environment.
        // For non-unsloth methods, {unsloth_install} is empty so this collapses
        // to the original single pip install and {peft_pin} is empty too.
        let peft_pin = if method == "unsloth" {
            " && pip install --no-cache-dir --no-deps --upgrade 'peft>=0.19,<0.20'"
        } else {
            ""
        };
        // Final guard: if anything in the install chain replaced torch with a
        // CUDA wheel, repair it one last time before `llamafactory-cli train`.
        // Versions are pinned to <2.11 to avoid pulling wheels that only exist
        // on the rocm7.2 index (per unsloth AMD install docs).
        // No-op when method != unsloth (the probe variable is empty and the
        // expression collapses to `true`).
        let torch_hip_guard = if method == "unsloth" {
            format!(
                " && ({probe} || pip install --no-cache-dir --upgrade --force-reinstall \
                       --index-url https://download.pytorch.org/whl/rocm7.0 \
                       'torch>=2.4,<2.11.0' 'torchvision<0.26.0' 'torchaudio<2.11.0')",
                probe = torch_hip_probe,
            )
        } else {
            String::new()
        };
        format!(
        "set -o pipefail; \
         python3 -c \"import site, os, shutil; [shutil.rmtree(os.path.join(p, 'triton_kernels'), ignore_errors=True) for p in (getattr(site, 'getsitepackages', lambda: [])() + [getattr(site, 'getusersitepackages', lambda: None)()]) if p]\" 2>/dev/null || true; \
         rm -rf ~/.triton/cache 2>/dev/null || true; \
         export DISABLE_VERSION_CHECK=1 HF_HOME=$HF_HOME && \
         {hf_export} cd {dir} && \
         {unsloth_env}\
         ((python3 -c 'import huggingface_hub; v=huggingface_hub.__version__; exit(0 if v.split(\".\")[0] == \"0\" else 1)' >/dev/null 2>&1 && which llamafactory-cli >/dev/null 2>&1{unsloth_check}) || \
         (true {unsloth_install} && \
          pip install --no-cache-dir 'huggingface-hub<1.0' 'transformers>=4.41.2,<4.58' 'llamafactory==0.9.4' {bnb_dep}{peft_pin}{extra_optimizer})){torch_hip_guard} && \
         rm -rf ~/.cache/huggingface/datasets 2>/dev/null || true && \
          : > {dir}/log.txt && : > {dir}/errorlog.txt && : > {dir}/train.log && \
          llamafactory-cli train {dir}/train.yaml \
            > >(tee -a {dir}/log.txt {dir}/train.log) \
            2> >(tee -a {dir}/errorlog.txt {dir}/train.log >&2)",
        hf_export = hf_export,
        dir = run.remote_dir,
        unsloth_env = unsloth_env,
        unsloth_check = unsloth_check,
        bnb_dep = bnb_dep,
        unsloth_install = unsloth_install,
        peft_pin = peft_pin,
        extra_optimizer = extra_optimizer,
        torch_hip_guard = torch_hip_guard,
    )
    };
    if docker_cfg.enabled {
        train_cmd = wrap_docker_cmd(&train_cmd, &container_name);
    }

    // ── Pre-flight GPU/HIP health check (unsloth on ROCm) ───────────────
    // The #1 field failure is `unsloth_zoo/device_type.py` raising
    // "no usable HIP accelerator" *after* dataset prep. Two root causes:
    //   (a) the GPU isn't visible in the container (rocm-smi empty, device=cpu)
    //   (b) torch got replaced by a CUDA/PyPI wheel (torch.version.hip is None)
    // Catch both here, before launch, so we repair (b) automatically and warn
    // clearly on (a) instead of burning minutes of dataset loading first.
    if method == "unsloth" {
        preflight_gpu_health(&session, cfg.docker.enabled, &container_name, run, app).await;
    }

    emit_log(app, &run.id, "[stage] training started\n", "stage");

    // Hook into the streaming output to parse metrics.
    let (tx, mut rx) = mpsc::unbounded_channel::<StreamChunk>();
    let cancel_c = cancel.clone();
    let watch_handle = tokio::spawn(async move {
        loop {
            if cancel_c.load(std::sync::atomic::Ordering::SeqCst) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        }
    });

    // ── Remote-tail fallback poller ─────────────────────────────────────
    // The SSH `exec_stream` channel can go silent for several reasons during
    // a long training run:
    //   • tqdm emits `\r`-terminated progress bars, not newlines, so the
    //     `for line in s.lines()` loop below only flushes on the next true
    //     newline (which only arrives at `logging_steps`, often 10–25).
    //   • NAT idle timeouts / network blips can stall the channel without
    //     killing it — bytes back up in the server's pipe buffer.
    //   • The frontend Reload button reads `live.log` on the host, which
    //     itself is only populated by the stream above. If the stream
    //     stalls, Reload sees the same stale file.
    // This poller mirrors the remote `train.log` into the same `emit_log`
    // pipeline every few seconds, tracking byte offsets so we never replay
    // data we already emitted. It runs concurrently with `exec_stream`,
    // and shuts down when training exits.
    let poller_done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let poller_done_c = poller_done.clone();
    let poller_session = session.clone();
    let poller_docker_enabled = cfg.docker.enabled;
    let poller_container_name = container_name.clone();
    let poller_remote_dir = run.remote_dir.clone();
    let poller_app = app.clone();
    let poller_run_id = run.id.clone();
    let poller_cancel = cancel.clone();
    let poller_handle = tokio::spawn(async move {
        let remote_log = format!("{}/train.log", poller_remote_dir);
        let mut offset: u64 = 0;
        // Slightly faster than tqdm's default refresh so a stalled stream
        // surfaces within ~5s in the UI. Cheap: one `tail -c +N` round-trip.
        let mut tick = tokio::time::interval(std::time::Duration::from_secs(5));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        // Heartbeat: if no new bytes arrive for this many seconds, emit a
        // single `[wait]` line so the UI doesn't look frozen during HF Hub
        // uploads or post-training shutdown (tqdm `\r`-bars don't change
        // the file size once they reach 100%).
        const IDLE_HEARTBEAT_SECS: u64 = 30;
        let mut last_data_at = std::time::Instant::now();
        let mut last_heartbeat_at: Option<std::time::Instant> = None;
        // Track which dataset-prep stage lines we've already announced so each
        // `[stage] getting datasets` / tokenizing message fires once even though
        // LLaMA-Factory's loader logs repeat per dataset and tqdm re-emits bars.
        let mut announced_stages: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        loop {
            tick.tick().await;
            if poller_done_c.load(std::sync::atomic::Ordering::SeqCst)
                || poller_cancel.load(std::sync::atomic::Ordering::SeqCst)
            {
                break;
            }
            // `tail -c +N+1` prints bytes from position N onward (1-indexed).
            // If the file doesn't exist yet (training still installing deps),
            // print nothing and try again next tick.
            let inner = format!(
                "if [ -f {p} ]; then tail -c +{n} {p}; fi",
                p = sh_quote(&remote_log),
                n = offset + 1
            );
            let cmd = if poller_docker_enabled {
                wrap_docker_cmd(&inner, &poller_container_name)
            } else {
                inner
            };
            let Ok(r) = poller_session.exec_blocking(&cmd).await else {
                continue;
            };
            if r.stdout.is_empty() {
                // No new bytes — emit a heartbeat every IDLE_HEARTBEAT_SECS
                // so the user knows training is alive but in a tqdm-only
                // phase (typically HF Hub upload or final checkpoint write).
                let idle = last_data_at.elapsed().as_secs();
                if idle >= IDLE_HEARTBEAT_SECS {
                    let due = match last_heartbeat_at {
                        Some(t) => t.elapsed().as_secs() >= IDLE_HEARTBEAT_SECS,
                        None => true,
                    };
                    if due {
                        emit_log(
                            &poller_app,
                            &poller_run_id,
                            &format!(
                                "[wait] training process is alive but quiet ({}s) — likely uploading to HF Hub or writing final checkpoint\n",
                                idle
                            ),
                            "stage",
                        );
                        last_heartbeat_at = Some(std::time::Instant::now());
                    }
                }
                continue;
            }
            last_data_at = std::time::Instant::now();
            last_heartbeat_at = None;
            let new_bytes = r.stdout.len() as u64;
            offset += new_bytes;
            // tqdm uses `\r` to overwrite the same line. Convert to `\n` so
            // the UI sees progress as discrete lines, and so `lines()` below
            // doesn't swallow a 5-minute stretch into one giant token.
            let normalized = r.stdout.replace('\r', "\n");
            for line in normalized.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                emit_log(&poller_app, &poller_run_id, &format!("{}\n", line), "train");
                // Surface dataset-prep lifecycle as friendly [stage] messages so
                // the user sees "getting datasets" before the first training step
                // (which can be minutes away while datasets download + tokenize).
                if let Some(stage_msg) = dataset_stage_message(line, &mut announced_stages) {
                    emit_log(&poller_app, &poller_run_id, &stage_msg, "stage");
                }
                if let Some(m) = llamafactory::parse_metric(line) {
                    emit_metric(&poller_app, &poller_run_id, m.step, m.loss, m.epoch);
                    // Persist the metric so the loss chart survives an app
                    // reload. Load-mutate-save is fine here — metrics fire
                    // every `logging_steps` (10–25 steps), not per token.
                    if let Ok(mut r) = runs::load(&poller_run_id).await {
                        if m.step > r.last_train_step {
                            r.last_train_step = m.step;
                        }
                        r.train_loss_history.push(TrainPoint {
                            step: m.step,
                            loss: m.loss,
                            epoch: m.epoch,
                        });
                        let _ = runs::save(&r).await;
                    }
                }
            }
        }
    });

    // Drain the SSH stream so the channel doesn't backpressure, but do NOT
    // emit logs or metrics from this path. The remote-tail poller above is
    // the single source of truth for training logs (it reads the same
    // `train.log` that `tee` is writing on the droplet, so we get exactly
    // one copy of each byte without de-dup logic). The collector still
    // updates the bounded `log_tail` field on Run for parity with older
    // resume code that snapshots it.
    let collector = async {
        let mut run_local = run.clone();
        while let Some(chunk) = rx.recv().await {
            match chunk {
                StreamChunk::Stdout(s) | StreamChunk::Stderr(s) => {
                    runs::append_log_tail(&mut run_local, &s);
                }
                StreamChunk::Done(_) => {}
            }
        }
        run_local
    };

    let cancel_for_stream = cancel.clone();
    let (_exit, final_run) = tokio::join!(
        session.exec_stream(&train_cmd, tx, Some(cancel_for_stream)),
        collector
    );
    *run = final_run;
    watch_handle.abort();
    // Tell the remote-tail poller to stop, then await it so we don't leak
    // a task that keeps tailing after training has exited.
    poller_done.store(true, std::sync::atomic::Ordering::SeqCst);
    let _ = poller_handle.await;
    sync_training_logs(&session, cfg.docker.enabled, &container_name, run, app).await;

    if cancel.load(std::sync::atomic::Ordering::SeqCst) {
        let mut kill_cmd = "pkill -f '[l]lamafactory' || true".to_string();
        if docker_cfg.enabled {
            kill_cmd = wrap_docker_cmd(&kill_cmd, &container_name);
        }
        let _ = session.exec_blocking(&kill_cmd).await;
        sync_training_logs(&session, cfg.docker.enabled, &container_name, run, app).await;
        return Err(AppError::Cancelled);
    }

    // ── 6. Verify adapter exists & done ─────────────────────────────────
    if docker_cfg.enabled {
        emit_log(
            app,
            &run.id,
            "[stage] copying weights from container back to host...\n",
            "stage",
        );
        let cp_back_cmd = format!(
            "docker cp {cn}:{dir}/lora {dir}_temp_copy && rm -rf {dir}/lora && mv {dir}_temp_copy {dir}/lora",
            dir = run.remote_dir,
            cn = container_name
        );
        let r_cp = session.exec_blocking(&cp_back_cmd).await?;
        if r_cp.exit_code != 0 {
            emit_log(
                app,
                &run.id,
                &format!(
                    "[warn] failed to copy weights back to host: {}\n",
                    r_cp.stderr
                ),
                "warn",
            );
        }
        sync_training_logs(&session, cfg.docker.enabled, &container_name, run, app).await;
    }

    let r = session
        .exec_blocking(&format!(
            "ls -1 {}/lora/adapter_model.safetensors 2>/dev/null || true",
            run.remote_dir
        ))
        .await?;
    let adapter_exists = !r.stdout.trim().is_empty();
    if !adapter_exists {
        if custom_method {
            let out_probe = session
                .exec_blocking(&format!(
                    "find {}/lora -mindepth 1 -maxdepth 2 -type f 2>/dev/null | head -n1 || true",
                    run.remote_dir
                ))
                .await?;
            if out_probe.stdout.trim().is_empty() {
                return Err(AppError::pipeline(format!(
                    "custom training finished but no output files were found in {}/lora",
                    run.remote_dir
                )));
            }
            if run_cfg.hub.enabled {
                emit_log(
                    app,
                    &run.id,
                    "[warn] custom method did not produce adapter_model.safetensors — built-in adapter upload and auto-merge skipped\n",
                    "warn",
                );
            }
            emit_log(
                app,
                &run.id,
                &format!(
                    "\n[done] Custom fine-tuning output saved at {}/lora\nRun `scp -r root@{}:{}/lora <local>` to pull outputs.\n",
                    run.remote_dir, cfg.ssh.host, run.remote_dir
                ),
                "stage",
            );
            run.status = RunStatus::Done;
            runs::save(run).await?;
            emit_progress(app, run);
            return Ok(());
        }
        return Err(AppError::pipeline(
            "training finished but adapter_model.safetensors not found",
        ));
    }
    let mut adapter_uploaded = false;
    if run_cfg.hub.enabled && !run_cfg.hub.model_id.trim().is_empty() {
        if let Some(token) = cfg.hf_token.as_ref().filter(|s| !s.is_empty()) {
            emit_log(
                app,
                &run.id,
                &format!(
                    "[hub] uploading adapter to {}\n",
                    run_cfg.hub.model_id.trim()
                ),
                "stage",
            );
            match upload_adapter(
                &session,
                cfg.docker.enabled,
                &container_name,
                run,
                token,
                run_cfg.hub.model_id.trim(),
                run_cfg.hub.private,
            )
            .await
            {
                Ok(()) => adapter_uploaded = true,
                Err(e) => emit_log(
                    app,
                    &run.id,
                    &format!(
                        "[warn] adapter upload failed, but training output is saved locally at {}/lora: {}\n",
                        run.remote_dir, e
                    ),
                    "warn",
                ),
            }
        } else {
            emit_log(
                app,
                &run.id,
                "[warn] hub.enabled = true but no HF token configured — adapter upload skipped\n",
                "warn",
            );
        }
    }
    let hub_note = if adapter_uploaded {
        format!(
            "\n[hub] adapter pushed to https://huggingface.co/{}\n",
            run_cfg.hub.model_id
        )
    } else {
        String::new()
    };
    emit_log(
        app,
        &run.id,
        &format!(
            "\n[done] {} saved at {}/lora{}\nRun `scp -r root@{}:{}/lora <local>` to pull weights.\n",
            if custom_method { "Custom adapter output" } else { "LoRA adapter" },
            run.remote_dir,
            hub_note,
            cfg.ssh.host,
            run.remote_dir
        ),
        "stage",
    );

    // ── 7. Optional: merge LoRA → base and upload full model ────────────
    // When the user wants a ready-to-load model (not just the adapter),
    // do the merge in-pipeline so they don't have to click "Merge & Upload"
    // afterwards. Requires `hub.enabled` + `hub.auto_merge` and a valid HF
    // token. Errors are logged but non-fatal — the adapter is already up.
    if run_cfg.hub.enabled && run_cfg.hub.auto_merge && !run_cfg.hub.model_id.trim().is_empty() {
        if let Some(token) = cfg.hf_token.as_ref().filter(|s| !s.is_empty()) {
            let merged_repo = if !run_cfg.hub.merged_model_id.trim().is_empty() {
                run_cfg.hub.merged_model_id.trim().to_string()
            } else {
                format!("{}-merged", run_cfg.hub.model_id.trim())
            };
            let gguf_repo = if !run_cfg.hub.gguf_repo_id.trim().is_empty() {
                run_cfg.hub.gguf_repo_id.trim().to_string()
            } else {
                format!("{}-gguf", run_cfg.hub.model_id.trim())
            };
            let gguf_quantization = if run_cfg.hub.gguf_quantization.trim().is_empty() {
                "Q4_K_M".to_string()
            } else {
                run_cfg.hub.gguf_quantization.trim().to_string()
            };
            emit_log(
                app,
                &run.id,
                &format!(
                    "\n[stage] auto-merge: merging LoRA into {} and uploading full model to {}\n",
                    crate::llamafactory::resolve_trainable_repo(&run.student_model),
                    merged_repo
                ),
                "stage",
            );
            match merge_and_upload(
                &session,
                cfg.docker.enabled,
                &container_name,
                run,
                token,
                &merged_repo,
                run_cfg.hub.private,
                app,
            )
            .await
            {
                Ok(url) => {
                    if run.hub.merged_model_id.trim().is_empty() {
                        run.hub.merged_model_id = merged_repo.clone();
                    }
                    emit_log(
                        app,
                        &run.id,
                        &format!("[done] merged model uploaded → {}\n", url),
                        "stage",
                    );
                    if run_cfg.hub.auto_convert_gguf {
                        emit_log(
                            app,
                            &run.id,
                            &format!(
                                "[stage] auto-gguf: converting merged model to {} and uploading to {}\n",
                                gguf_quantization, gguf_repo
                            ),
                            "stage",
                        );
                        match convert_and_upload_gguf(
                            &session,
                            cfg.docker.enabled,
                            &container_name,
                            run,
                            token,
                            &gguf_repo,
                            &gguf_quantization,
                            run_cfg.hub.private,
                            app,
                        )
                        .await
                        {
                            Ok(url) => {
                                run.hub.gguf_repo_id = gguf_repo.clone();
                                run.hub.gguf_quantization = gguf_quantization.clone();
                                emit_log(
                                    app,
                                    &run.id,
                                    &format!("[done] GGUF uploaded → {}\n", url),
                                    "stage",
                                );
                            }
                            Err(e) => emit_log(
                                app,
                                &run.id,
                                &format!(
                                    "[warn] auto-GGUF failed (merged model is still uploaded): {}\n",
                                    e
                                ),
                                "warn",
                            ),
                        }
                    }
                }
                Err(e) => emit_log(
                    app,
                    &run.id,
                    &format!(
                        "[warn] auto-merge failed (adapter is still uploaded): {}\n",
                        e
                    ),
                    "warn",
                ),
            }
        } else {
            emit_log(
                app,
                &run.id,
                "[warn] auto-merge requested but no HF token configured — skipped\n",
                "warn",
            );
        }
    }

    run.status = RunStatus::Done;
    runs::save(run).await?;
    emit_progress(app, run);
    Ok(())
}

async fn upload_adapter(
    session: &SshSession,
    docker_enabled: bool,
    container_name: &str,
    run: &Run,
    hf_token: &str,
    repo_id: &str,
    private: bool,
) -> Result<()> {
    let adapter_path = format!("{}/lora", run.remote_dir);
    let private_flag = if private { "True" } else { "False" };
    let script = format!(
        r#"set -e
cd {run_dir}
export HF_TOKEN={token}
export HUGGING_FACE_HUB_TOKEN={token}
python3 - <<'PY'
import json
import os
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo

base_model = {base_model}
adapter_path = Path({adapter_path})
stage_path = adapter_path.parent / "adapter_hub_upload"
repo_id = {repo}
private = {private}
token = os.environ.get("HF_TOKEN")

adapter_config_path = adapter_path / "adapter_config.json"
if not adapter_path.exists():
    raise SystemExit(f"adapter directory not found: {{adapter_path}}")
if not adapter_config_path.exists():
    raise SystemExit(f"adapter_config.json not found in {{adapter_path}}")
if not ((adapter_path / "adapter_model.safetensors").exists() or (adapter_path / "adapter_model.bin").exists()):
    raise SystemExit(f"adapter weights not found in {{adapter_path}}")

if stage_path.exists():
    shutil.rmtree(stage_path)
stage_path.mkdir(parents=True, exist_ok=True)

skip_files = {{
    "README.md",
    "adapter_config.json",
    "all_results.json",
    "eval_results.json",
    "optimizer.pt",
    "rng_state.pth",
    "scaler.pt",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
    "train_results.json",
}}
skip_suffixes = (".pt", ".pth")

for item in adapter_path.iterdir():
    if item.is_file() and item.name not in skip_files and not item.name.endswith(skip_suffixes):
        shutil.copy2(item, stage_path / item.name)

zrald_artifacts = adapter_path / "zrald_artifacts"
if zrald_artifacts.is_dir():
    shutil.copytree(zrald_artifacts, stage_path / "zrald_artifacts", dirs_exist_ok=True)

data = json.loads(adapter_config_path.read_text())
if not data.get("base_model_name_or_path"):
    data["base_model_name_or_path"] = base_model
(stage_path / "adapter_config.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

readme = adapter_path / "README.md"
body = ""
if readme.exists():
    text = readme.read_text()
    if text.startswith("---"):
        parts = text.split("---", 2)
        body = parts[2].lstrip() if len(parts) >= 3 else text
    else:
        body = text
metadata = f"""---
base_model: {base_model}
library_name: peft
tags:
- lora
- adapter
---
"""
(stage_path / "README.md").write_text(metadata + "\n" + body)

staged_files = [p for p in stage_path.rglob("*") if p.is_file()]
staged_bytes = sum(p.stat().st_size for p in staged_files)
print(f"[hub] staging {{len(staged_files)}} adapter file(s), {{staged_bytes / 1024 / 1024:.1f}} MiB; checkpoints excluded", flush=True)

create_repo(repo_id=repo_id, repo_type="model", private=private, token=token, exist_ok=True)
HfApi(token=token).upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path=str(stage_path),
    commit_message="Upload LoRA adapter",
    ignore_patterns=[
        "checkpoint-*",
        "checkpoint-*/*",
        "**/optimizer.pt",
        "**/rng_state.pth",
        "**/scaler.pt",
        "**/scheduler.pt",
        "**/trainer_state.json",
        "**/training_args.bin",
    ],
)
print(f"https://huggingface.co/{{repo_id}}")
PY"#,
        run_dir = sh_quote(&run.remote_dir),
        token = sh_quote(hf_token),
        base_model = serde_json::to_string(&crate::llamafactory::resolve_trainable_repo(
            &run.student_model
        ))
        .unwrap_or_else(|_| "\"\"".to_string()),
        adapter_path = serde_json::to_string(&adapter_path).unwrap_or_else(|_| "\"\"".to_string()),
        repo = serde_json::to_string(&repo_id).unwrap_or_else(|_| "\"\"".to_string()),
        private = private_flag,
    );
    let cmd = if docker_enabled {
        wrap_docker_cmd(&script, container_name)
    } else {
        script
    };
    let result = session.exec_blocking(&cmd).await?;
    if result.exit_code != 0 {
        let combined = format!(
            "{}{}",
            result.stderr.replace(hf_token, "***"),
            result.stdout.replace(hf_token, "***")
        )
        .replace('\r', "\n");
        let mut lines = combined
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .rev()
            .take(40)
            .collect::<Vec<_>>();
        lines.reverse();
        let output = if lines.is_empty() {
            "(no upload output captured)".to_string()
        } else {
            lines.join("\n")
        };
        return Err(AppError::pipeline(format!(
            "adapter upload failed with exit code {}:\n{}",
            result.exit_code, output
        )));
    }
    Ok(())
}

/// Merge LoRA adapter into base weights on the remote and push the full
/// merged model to a Hugging Face repo. Used both by the pipeline's
/// auto-merge step and by the on-demand `merge_and_upload_model` command.
async fn merge_and_upload(
    session: &SshSession,
    docker_enabled: bool,
    container_name: &str,
    run: &Run,
    hf_token: &str,
    repo_id: &str,
    private: bool,
    app: &AppHandle,
) -> Result<String> {
    let merged_dir = format!("{}/merged", run.remote_dir);
    let adapter_path = format!("{}/lora", run.remote_dir);
    let private_flag = if private { "True" } else { "False" };
    let script = format!(
        r#"set -e
cd {run_dir}
export HF_TOKEN={token}
export HUGGING_FACE_HUB_TOKEN={token}
python3 - <<'PY'
import os
import shutil
from pathlib import Path
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi, create_repo

base_model = {base_model}
adapter_path = Path({adapter_path})
merged_dir = Path({merged_dir})
repo_id = {repo}
private = {private}
token = os.environ.get("HF_TOKEN")

print(f"[merge] loading base {{base_model}}", flush=True)
os.makedirs(merged_dir, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
model_type = getattr(config, "model_type", "").lower()
if "vl" in model_type or "vision" in model_type:
    try:
        from transformers import AutoModelForImageTextToText
        model_cls = AutoModelForImageTextToText
    except Exception:
        from transformers import Qwen2_5_VLForConditionalGeneration
        model_cls = Qwen2_5_VLForConditionalGeneration
else:
    model_cls = AutoModelForCausalLM
try:
    base = model_cls.from_pretrained(
        base_model,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
except TypeError:
    base = model_cls.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
print("[merge] applying adapter", flush=True)
model = PeftModel.from_pretrained(base, str(adapter_path))
print("[merge] merging & unloading", flush=True)
merged = model.merge_and_unload()
print(f"[merge] saving to {{merged_dir}}", flush=True)
merged.save_pretrained(str(merged_dir), safe_serialization=True, max_shard_size="4GB")
tokenizer.save_pretrained(str(merged_dir))
try:
    AutoProcessor.from_pretrained(base_model, trust_remote_code=True).save_pretrained(str(merged_dir))
except Exception:
    pass

zrald_artifacts = adapter_path / "zrald_artifacts"
if zrald_artifacts.is_dir():
    shutil.copytree(zrald_artifacts, merged_dir / "zrald_artifacts", dirs_exist_ok=True)
    print("[merge] included ZRALD prompts, rewards, and benchmark artifacts", flush=True)

print(f"[merge] uploading to {{repo_id}}", flush=True)
create_repo(repo_id=repo_id, repo_type="model", private=private, token=token, exist_ok=True)
api = HfApi(token=token)
api.upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path=str(merged_dir),
    commit_message="Upload merged fine-tuned model",
)
print(f"https://huggingface.co/{{repo_id}}")
PY"#,
        run_dir = sh_quote(&run.remote_dir),
        token = sh_quote(hf_token),
        base_model = serde_json::to_string(&crate::llamafactory::resolve_trainable_repo(
            &run.student_model
        ))
        .unwrap_or_else(|_| "\"\"".to_string()),
        adapter_path = serde_json::to_string(&adapter_path).unwrap_or_else(|_| "\"\"".to_string()),
        merged_dir = serde_json::to_string(&merged_dir).unwrap_or_else(|_| "\"\"".to_string()),
        repo = serde_json::to_string(&repo_id).unwrap_or_else(|_| "\"\"".to_string()),
        private = private_flag,
    );
    let cmd = if docker_enabled {
        wrap_docker_cmd(&script, container_name)
    } else {
        script
    };
    // Stream the merge output so the user sees progress (downloads can
    // take 5–15 minutes on a 7B base).
    let (tx, mut rx) = mpsc::unbounded_channel::<StreamChunk>();
    let app_c = app.clone();
    let run_id_c = run.id.clone();
    let pump = async move {
        while let Some(chunk) = rx.recv().await {
            if let StreamChunk::Stdout(s) | StreamChunk::Stderr(s) = chunk {
                let normalized = s.replace('\r', "\n");
                for line in normalized.lines() {
                    if line.trim().is_empty() {
                        continue;
                    }
                    emit_log(&app_c, &run_id_c, &format!("{}\n", line), "train");
                }
            }
        }
    };
    let (exit, _) = tokio::join!(session.exec_stream(&cmd, tx, None), pump);
    let code = exit.unwrap_or(-1);
    if code != 0 {
        return Err(AppError::pipeline(format!(
            "merge/upload exited with code {}",
            code
        )));
    }
    Ok(format!("https://huggingface.co/{}", repo_id))
}

/// Convert the merged model to GGUF format and upload to a dedicated repo
/// for Ollama/llama.cpp compatibility.
pub async fn convert_and_upload_gguf(
    session: &SshSession,
    docker_enabled: bool,
    container_name: &str,
    run: &Run,
    hf_token: &str,
    gguf_repo_id: &str,
    quantization: &str,
    private: bool,
    app: &AppHandle,
) -> Result<String> {
    let merged_dir = format!("{}/merged", run.remote_dir);
    let gguf_dir = format!("{}/gguf", run.remote_dir);
    let private_flag = if private { "True" } else { "False" };

    let script = format!(
        r#"set -e
cd {run_dir}
export HF_TOKEN={token}
export HUGGING_FACE_HUB_TOKEN={token}
python3 - <<'PY'
import os
import shutil
import subprocess
import sys
from pathlib import Path

merged_dir = Path({merged_dir})
gguf_dir = Path({gguf_dir})
gguf_repo = {gguf_repo}
quantization = ({quantization} or "Q4_K_M").strip()
private = {private}
token = os.environ.get("HF_TOKEN")

def run(cmd, cwd=None):
    cmd = [str(part) for part in cmd]
    print("[gguf] $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)

if not merged_dir.is_dir():
    raise SystemExit("merged model directory not found: " + str(merged_dir))

gguf_dir.mkdir(parents=True, exist_ok=True)
llama_dir = gguf_dir / "llama.cpp"
build_dir = llama_dir / "build"
raw_gguf = gguf_dir / "model-f16.gguf"
final_gguf = gguf_dir / "model.gguf"

if not llama_dir.exists():
    print("[gguf] cloning llama.cpp", flush=True)
    run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp.git", llama_dir])
else:
    print("[gguf] updating llama.cpp", flush=True)
    try:
        run(["git", "-C", llama_dir, "pull", "--ff-only"])
    except subprocess.CalledProcessError:
        print("[gguf] warning: llama.cpp update failed; using existing checkout", flush=True)

req_candidates = [
    llama_dir / "requirements" / "requirements-convert_hf_to_gguf.txt",
    llama_dir / "requirements.txt",
]
for req in req_candidates:
    if req.exists():
        run([sys.executable, "-m", "pip", "install", "-r", req])
        break

converter = llama_dir / "convert_hf_to_gguf.py"
if not converter.exists():
    raise SystemExit("convert_hf_to_gguf.py not found in " + str(llama_dir))

quant_key = quantization.lower()
converter_outtypes = {{"f16": "f16", "bf16": "bf16", "q8": "q8_0", "q8_0": "q8_0"}}
if quant_key in converter_outtypes:
    out_gguf = final_gguf
    convert_outtype = converter_outtypes[quant_key]
else:
    out_gguf = raw_gguf
    convert_outtype = "f16"

if out_gguf.exists():
    out_gguf.unlink()
if final_gguf.exists() and final_gguf != out_gguf:
    final_gguf.unlink()

print("[gguf] converting " + str(merged_dir) + " to " + convert_outtype, flush=True)
run([sys.executable, converter, merged_dir, "--outfile", out_gguf, "--outtype", convert_outtype])

if out_gguf != final_gguf:
    quantize = None
    for candidate in [
        build_dir / "bin" / "llama-quantize",
        build_dir / "bin" / "quantize",
        llama_dir / "llama-quantize",
        llama_dir / "quantize",
    ]:
        if candidate.exists():
            quantize = candidate
            break
    if quantize is None:
        print("[gguf] building llama-quantize for " + quantization, flush=True)
        if shutil.which("cmake") is None:
            raise SystemExit("cmake is required to build llama-quantize for " + quantization)
        run(["cmake", "-S", llama_dir, "-B", build_dir, "-DLLAMA_CURL=OFF"])
        run([
            "cmake",
            "--build",
            build_dir,
            "--config",
            "Release",
            "-j",
            str(os.cpu_count() or 2),
            "--target",
            "llama-quantize",
        ])
        for candidate in [
            build_dir / "bin" / "llama-quantize",
            build_dir / "bin" / "quantize",
            llama_dir / "llama-quantize",
            llama_dir / "quantize",
        ]:
            if candidate.exists():
                quantize = candidate
                break
    if quantize is None:
        raise SystemExit("llama-quantize was not produced by the llama.cpp build")
    if final_gguf.exists():
        final_gguf.unlink()
    print("[gguf] quantizing to " + quantization.upper(), flush=True)
    run([quantize, out_gguf, final_gguf, quantization.upper()])

if not final_gguf.exists() or final_gguf.stat().st_size == 0:
    raise SystemExit("GGUF file was not created: " + str(final_gguf))

print("[gguf] uploading to " + gguf_repo, flush=True)
from huggingface_hub import HfApi, create_repo
create_repo(repo_id=gguf_repo, repo_type="model", private=private, token=token, exist_ok=True)
api = HfApi(token=token)
api.upload_file(
    repo_id=gguf_repo,
    repo_type="model",
    path_in_repo="model.gguf",
    path_or_fileobj=str(final_gguf),
    commit_message="Upload GGUF model for Ollama/llama.cpp",
)
print("https://huggingface.co/" + gguf_repo)
PY"#,
        run_dir = sh_quote(&run.remote_dir),
        token = sh_quote(hf_token),
        merged_dir = serde_json::to_string(&merged_dir).unwrap_or_else(|_| "\"\"".to_string()),
        gguf_dir = serde_json::to_string(&gguf_dir).unwrap_or_else(|_| "\"\"".to_string()),
        gguf_repo = serde_json::to_string(&gguf_repo_id).unwrap_or_else(|_| "\"\"".to_string()),
        quantization = serde_json::to_string(&quantization).unwrap_or_else(|_| "\"\"".to_string()),
        private = private_flag,
    );

    let cmd = if docker_enabled {
        wrap_docker_cmd(&script, container_name)
    } else {
        script
    };

    let (tx, mut rx) = mpsc::unbounded_channel::<StreamChunk>();
    let app_c = app.clone();
    let run_id_c = run.id.clone();
    let pump = async move {
        while let Some(chunk) = rx.recv().await {
            if let StreamChunk::Stdout(s) | StreamChunk::Stderr(s) = chunk {
                let normalized = s.replace('\r', "\n");
                for line in normalized.lines() {
                    if line.trim().is_empty() {
                        continue;
                    }
                    emit_log(&app_c, &run_id_c, &format!("{}\n", line), "train");
                }
            }
        }
    };
    let (exit, _) = tokio::join!(session.exec_stream(&cmd, tx, None), pump);
    let code = exit.unwrap_or(-1);
    if code != 0 {
        return Err(AppError::pipeline(format!(
            "gguf convert/upload exited with code {}",
            code
        )));
    }
    Ok(format!("https://huggingface.co/{}", gguf_repo_id))
}

/// Pre-flight health check for unsloth-on-ROCm runs. Verifies the GPU is
/// visible in the (container's) environment and that PyTorch has a usable HIP
/// runtime; auto-repairs a wrong torch wheel once, and emits a clear diagnosis
/// when the GPU itself is invisible (which a reinstall cannot fix).
///
/// Best-effort and non-fatal: on any SSH error we log a note and let training
/// proceed (the install block's own probes are a second line of defence).
async fn preflight_gpu_health(
    session: &SshSession,
    docker_enabled: bool,
    container_name: &str,
    run: &mut Run,
    app: &AppHandle,
) {
    let wrap = |inner: &str| -> String {
        if docker_enabled {
            wrap_docker_cmd(inner, container_name)
        } else {
            inner.to_string()
        }
    };

    emit_log(
        app,
        &run.id,
        "[stage] pre-flight: checking GPU visibility and PyTorch HIP runtime\n",
        "stage",
    );

    // 1) Is a GPU visible? rocm-smi (or amd-smi) should list a device. We check
    //    for a real device row, not just exit code, because rocm-smi can exit 0
    //    while printing an empty SMI log.
    let smi_cmd =
        wrap("(rocm-smi --showproductname 2>/dev/null || amd-smi list 2>/dev/null || true)");
    let gpu_visible = match session.exec_blocking(&smi_cmd).await {
        Ok(r) => {
            let o = r.stdout.to_lowercase();
            // Heuristics for "a GPU is actually listed".
            (o.contains("gpu")
                || o.contains("card")
                || o.contains("series")
                || o.contains("instinct")
                || o.contains("radeon"))
                && !o.contains("no gpu")
        }
        Err(_) => true, // can't tell — don't block training on a probe failure
    };

    if !gpu_visible {
        emit_log(
            app,
            &run.id,
            "[diagnosis] the GPU is not visible to the training environment (rocm-smi listed no device). Training would fall back to CPU and unsloth will abort.\n",
            "warn",
        );
        if docker_enabled {
            emit_log(app, &run.id, "[fix] recreate the container with GPU access: docker run --device=/dev/kfd --device=/dev/dri --group-add video --group-add render --security-opt seccomp=unconfined --ipc=host --shm-size 16G ... rocm/vllm:latest\n", "warn");
            emit_log(app, &run.id, &format!("[fix] then verify: docker exec {container_name} rocm-smi (must list the card)\n"), "warn");
        } else {
            emit_log(app, &run.id, "[fix] confirm the host sees the GPU with `rocm-smi`, and that the user is in the `video` and `render` groups.\n", "warn");
        }
        // Don't repair torch — it won't help an invisible GPU. Let training
        // proceed so the real crash + errorlog still surface, but the user now
        // already has the actionable fix above.
        return;
    }

    // 2) GPU is visible. Make sure torch has a working HIP runtime. This probe
    //    actually exercises HIP device discovery (not just torch.version.hip),
    //    matching what unsloth checks at startup.
    let probe = wrap(
        "python3 -c 'import torch,sys; sys.exit(0 if (getattr(torch.version,\"hip\",None) and torch.cuda.is_available()) else 1)' >/dev/null 2>&1 && echo HIP_OK || echo HIP_BAD",
    );
    let hip_ok = match session.exec_blocking(&probe).await {
        Ok(r) => r.stdout.contains("HIP_OK"),
        Err(_) => true, // probe failed to run — defer to the install block
    };

    if hip_ok {
        emit_log(
            app,
            &run.id,
            "[ok] pre-flight: GPU visible and PyTorch HIP runtime healthy\n",
            "stage",
        );
        return;
    }

    // 3) torch is present but HIP is unusable → reinstall the ROCm wheels once.
    emit_log(
        app,
        &run.id,
        "[stage] auto-repair: PyTorch HIP runtime unusable — reinstalling ROCm PyTorch wheels (<2.11)\n",
        "stage",
    );
    let repair = wrap(
        "export UNSLOTH_IS_ROCM=1 PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH:-gfx1100}; \
         pip install --no-cache-dir --upgrade --force-reinstall \
            --index-url https://download.pytorch.org/whl/rocm7.0 \
            'torch>=2.4,<2.11.0' 'torchvision<0.26.0' 'torchaudio<2.11.0'",
    );
    match session.exec_blocking(&repair).await {
        Ok(_) => {
            // Re-probe to confirm the repair took.
            let ok = match session.exec_blocking(&probe).await {
                Ok(r) => r.stdout.contains("HIP_OK"),
                Err(_) => false,
            };
            if ok {
                emit_log(
                    app,
                    &run.id,
                    "[ok] auto-repair: ROCm PyTorch reinstalled — HIP runtime now healthy\n",
                    "stage",
                );
            } else {
                emit_log(app, &run.id, "[warn] auto-repair ran but HIP is still unusable; training may fail. See the diagnosis after the run if it does.\n", "warn");
            }
        }
        Err(e) => {
            emit_log(
                app,
                &run.id,
                &format!("[warn] auto-repair failed to reinstall ROCm PyTorch: {e}\n"),
                "warn",
            );
        }
    }
}

/// Translate a raw LLaMA-Factory/datasets log line into a friendly `[stage]`
/// message about dataset preparation, or `None` if the line isn't a dataset
/// lifecycle event. `seen` dedups so each distinct stage fires only once
/// (the loader logs repeat per dataset; tqdm re-emits the same bar many times).
fn dataset_stage_message(
    line: &str,
    seen: &mut std::collections::HashSet<String>,
) -> Option<String> {
    let lower = line.to_lowercase();

    // "Loading dataset Zrald/GE-MATH-SET1..." — one [stage] per dataset name.
    if lower.contains("loading dataset") {
        // Pull the token after "Loading dataset " up to the trailing "...".
        let name = line
            .split("Loading dataset")
            .nth(1)
            .map(|s| s.trim().trim_end_matches('.').trim())
            .filter(|s| !s.is_empty())
            .unwrap_or("dataset")
            .to_string();
        let key = format!("load:{name}");
        if seen.insert(key) {
            return Some(format!("[stage] getting datasets: loading {name}\n"));
        }
        return None;
    }

    // "Running tokenizer on dataset (num_proc=4): ...%" — announce once.
    if lower.contains("running tokenizer on dataset") && seen.insert("tokenize".to_string()) {
        return Some("[stage] tokenizing dataset…\n".to_string());
    }

    // "Generating train split: 500 examples [..]" — report the final count once
    // per split line that carries an example count.
    if lower.contains("generating train split") && lower.contains("examples") {
        // Grab the integer that precedes "examples".
        if let Some(idx) = lower.find("examples") {
            let prefix = &line[..idx];
            let count: String = prefix
                .chars()
                .rev()
                .skip_while(|c| !c.is_ascii_digit())
                .take_while(|c| c.is_ascii_digit() || *c == ',')
                .collect::<String>()
                .chars()
                .rev()
                .collect();
            let count = count.trim().trim_matches(',');
            if !count.is_empty() && count != "0" {
                let key = format!("split:{count}");
                if seen.insert(key) {
                    return Some(format!("[stage] dataset split ready: {count} examples\n"));
                }
            }
        }
    }

    None
}

async fn sync_training_logs(
    session: &SshSession,
    docker_enabled: bool,
    container_name: &str,
    run: &mut Run,
    app: &AppHandle,
) {
    let local_dir = std::path::Path::new(&run.local_dir);
    let _ = fs::create_dir_all(local_dir).await;

    for name in [
        "log.txt",
        "errorlog.txt",
        "train.log",
        "lora/trainer_state.json",
    ] {
        let remote_path = format!("{}/{}", run.remote_dir, name);
        let cat_inner = format!("cat {} 2>/dev/null || true", sh_quote(&remote_path));
        let cat_cmd = if docker_enabled {
            wrap_docker_cmd(&cat_inner, container_name)
        } else {
            cat_inner
        };

        match session.exec_blocking(&cat_cmd).await {
            Ok(r) if !r.stdout.is_empty() => {
                let local_path = local_dir.join(name.replace('/', "_"));
                if let Some(parent) = local_path.parent() {
                    let _ = fs::create_dir_all(parent).await;
                }
                let _ = fs::write(local_path, &r.stdout).await;

                if name == "lora/trainer_state.json" {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&r.stdout) {
                        if let Some(step) = v.get("global_step").and_then(|s| s.as_u64()) {
                            run.last_train_step = run.last_train_step.max(step as u32);
                        }
                        if let Some(history) = v.get("log_history").and_then(|h| h.as_array()) {
                            for item in history {
                                let step =
                                    item.get("step").and_then(|s| s.as_u64()).unwrap_or(0) as u32;
                                let epoch =
                                    item.get("epoch").and_then(|e| e.as_f64()).unwrap_or(0.0)
                                        as f32;
                                let loss = item
                                    .get("loss")
                                    .or_else(|| item.get("train_loss"))
                                    .and_then(|l| l.as_f64())
                                    .map(|l| l as f32);
                                if step > 0 {
                                    run.last_train_step = run.last_train_step.max(step);
                                }
                                if let Some(loss) = loss {
                                    if !run.train_loss_history.iter().any(|p| p.step == step) {
                                        run.train_loss_history.push(TrainPoint {
                                            step,
                                            loss,
                                            epoch,
                                        });
                                        emit_metric(app, &run.id, step, loss, epoch);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Ok(_) => {}
            Err(e) => {
                emit_log(
                    app,
                    &run.id,
                    &format!("[warn] could not sync remote {}: {}\n", name, e),
                    "warn",
                );
            }
        }
    }

    if let Ok(err_txt) = fs::read_to_string(local_dir.join("errorlog.txt")).await {
        // Find the last block that looks like a real failure (Traceback / Error /
        // ModuleNotFoundError / CUDA OOM) and show the tail to the user. Stderr
        // from training is mostly tqdm progress bars; we want to skip that and
        // only surface the actual crash so people don't have to SSH in to debug.
        let lines: Vec<&str> = err_txt.lines().collect();
        let notable_idx = lines.iter().rposition(|line| {
            let lower = line.to_lowercase();
            lower.contains("traceback")
                || lower.contains("error:")
                || lower.contains("modulenotfounderror")
                || lower.contains("importerror")
                || lower.contains("runtimeerror")
                || lower.contains("cuda out of memory")
                || lower.contains("oserror")
                || lower.contains("valueerror")
        });
        if let Some(idx) = notable_idx {
            let start = idx.saturating_sub(2);
            let end = (idx + 40).min(lines.len());
            let snippet: Vec<&str> = lines[start..end]
                .iter()
                .filter(|l| {
                    // drop tqdm progress bars and other carriage-return noise
                    let t = l.trim();
                    !t.is_empty() && !t.starts_with("Map ") && !t.contains("it/s]")
                })
                .copied()
                .collect();
            if !snippet.is_empty() {
                emit_log(
                    app,
                    &run.id,
                    "[errorlog] ── training crashed; tail of errorlog.txt ──\n",
                    "warn",
                );
                for line in snippet {
                    emit_log(app, &run.id, &format!("[errorlog] {}\n", line), "warn");
                }
                emit_log(
                    app,
                    &run.id,
                    "[errorlog] ── end errorlog.txt tail ──\n",
                    "warn",
                );
            }
        }

        // Universal diagnosis: match the crash against known signatures and
        // print a plain-language cause + the fix steps, so a recurring error
        // always ships with its solution and the user needn't SSH in to debug.
        if let Some((diagnosis, fixes)) = diagnose_failure(&err_txt) {
            emit_log(app, &run.id, &format!("[diagnosis] {diagnosis}\n"), "warn");
            for fix in fixes {
                emit_log(app, &run.id, &format!("[fix] {fix}\n"), "warn");
            }
        }
    }

    let _ = runs::save(run).await;
}

/// Match a training crash log against known failure signatures and return a
/// `(diagnosis, fix_steps)` pair. Data-driven so new cases are one-line
/// additions to the table. Ordered most-specific first; the first match wins.
fn diagnose_failure(log_text: &str) -> Option<(String, Vec<String>)> {
    let lower = log_text.to_lowercase();
    let has = |needle: &str| lower.contains(needle);

    // 1) GPU not visible inside the container. The most important case from the
    //    field report: rocm-smi returns an empty log and training falls back to
    //    CPU. A torch reinstall cannot fix this — the container needs the device
    //    nodes and the right groups. Detect the empty-SMI banner / cpu fallback.
    let rocm_smi_empty = has("end of rocm smi log")
        && !lower.contains("gpu[0]")
        && !lower.contains("card series")
        && !lower.contains("vram");
    if rocm_smi_empty
        || lower.contains("device: cpu")
        || lower.contains("no usable hip accelerator")
    {
        // Distinguish "GPU invisible" from "torch is a CUDA wheel".
        if rocm_smi_empty || has("device: cpu") {
            return Some((
                "the GPU is not visible inside the Docker container (rocm-smi returned nothing / training fell back to CPU). A PyTorch reinstall cannot fix this — the container is missing the GPU device nodes or render/video group access.".to_string(),
                vec![
                    "On the host, confirm the GPU is healthy: `rocm-smi` (should list the card).".to_string(),
                    "Recreate the container with GPU access: `docker run --device=/dev/kfd --device=/dev/dri --group-add video --group-add render --security-opt seccomp=unconfined --ipc=host --shm-size 16G ... rocm/vllm:latest`".to_string(),
                    "Verify inside the container: `docker exec rocm rocm-smi` — it must show the GPU before training will use it.".to_string(),
                    "If running bare-metal (no Docker), disable the Docker toggle in Credentials so training runs directly on the host GPU.".to_string(),
                ],
            ));
        }
        // HIP present in build but torch is the wrong (CUDA/PyPI) wheel.
        return Some((
            "PyTorch has no usable HIP runtime — torch was likely replaced by a default CUDA/PyPI wheel during dependency install.".to_string(),
            vec![
                "Reinstall the ROCm PyTorch wheels (<2.11): `pip install --force-reinstall --index-url https://download.pytorch.org/whl/rocm7.0 'torch>=2.4,<2.11.0' 'torchvision<0.26.0' 'torchaudio<2.11.0'`".to_string(),
                "Set the env before training: `export UNSLOTH_IS_ROCM=1 PYTORCH_ROCM_ARCH=gfx1100` (use your actual arch).".to_string(),
                "Re-run training — the app now auto-repairs this once before failing.".to_string(),
            ],
        ));
    }

    // 2) bitsandbytes 4-bit decode NaN bug on AMD.
    if has("bitsandbytes")
        && (has("nan") || has("4-bit") || has("cextension") || has("rocm_warp_size"))
    {
        return Some((
            "bitsandbytes ≤0.49.2 has a 4-bit decode NaN bug on every AMD GPU.".to_string(),
            vec![
                "Install the bnb pre-release wheel: `pip install --force-reinstall --no-deps 'https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_main/bitsandbytes-1.33.7.preview-py3-none-manylinux_2_24_x86_64.whl'`".to_string(),
                "Fallback if the URL is unreachable: `pip install --no-deps 'bitsandbytes>=0.49.1'`".to_string(),
            ],
        ));
    }

    // 3) peft too old for recent unsloth (LoraConfig.ensure_weight_tying).
    if has("ensure_weight_tying") || (has("loraconfig") && has("unexpected keyword")) {
        return Some((
            "peft is older than unsloth requires (missing LoraConfig.ensure_weight_tying — needs peft ≥0.19).".to_string(),
            vec![
                "`pip install --no-cache-dir --no-deps --upgrade 'peft>=0.19,<0.20'`".to_string(),
            ],
        ));
    }

    // 4) VRAM OOM.
    if has("cuda out of memory") || has("hip out of memory") || has("out of memory") {
        return Some((
            "the GPU ran out of VRAM during training.".to_string(),
            vec![
                "Lower `per_device_train_batch_size` (e.g. to 1) and/or raise `gradient_accumulation_steps`.".to_string(),
                "Reduce `cutoff_len` / max sequence length.".to_string(),
                "Enable gradient checkpointing if not already on.".to_string(),
            ],
        ));
    }

    // 5) Missing Python dependency.
    if has("modulenotfounderror") || has("importerror") {
        // Try to name the module from "No module named 'x'".
        let module = log_text
            .split("No module named")
            .nth(1)
            .map(|s| s.trim().trim_start_matches(['\'', '"']))
            .and_then(|s| s.split(['\'', '"']).next())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string());
        let diag = match &module {
            Some(m) => format!("a required Python package is missing: `{m}`."),
            None => "a required Python package failed to import.".to_string(),
        };
        let fix = match &module {
            Some(m) => format!(
                "`pip install --no-cache-dir {m}` (inside the training container), then re-run."
            ),
            None => "Reinstall the training dependencies and re-run.".to_string(),
        };
        return Some((diag, vec![fix]));
    }

    None
}

// ── Docker Helpers ─────────────────────────────────────────────────────────

pub fn wrap_docker_cmd(cmd: &str, container_name: &str) -> String {
    format!(
        "docker exec -i {} bash -lc {}",
        container_name,
        sh_quote(cmd)
    )
}

pub fn wrap_docker_cmd_detached(cmd: &str, container_name: &str) -> String {
    format!(
        "docker exec -d {} bash -lc {} >/dev/null 2>&1",
        container_name,
        sh_quote(cmd)
    )
}

pub async fn probe_model_context_limit(
    session: &SshSession,
    docker_enabled: bool,
    container_name: &str,
    repo_id: &str,
    hf_token: Option<&str>,
) -> Option<(u32, String)> {
    let token_export = hf_token
        .filter(|token| !token.trim().is_empty())
        .map(|token| {
            format!(
                "export HF_TOKEN={token} HUGGING_FACE_HUB_TOKEN={token}; ",
                token = sh_quote(token)
            )
        })
        .unwrap_or_default();
    let script = format!(
        r#"set +e
{token_export}python3 - <<'PY'
import os

MODEL = {model}
TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None

def collect(cfg):
    values = []
    for name in [
        "max_position_embeddings",
        "n_positions",
        "seq_length",
        "max_seq_len",
        "max_sequence_length",
        "max_context_length",
    ]:
        value = getattr(cfg, name, None)
        if isinstance(value, bool):
            continue
        try:
            parsed = int(value)
        except Exception:
            continue
        if 0 < parsed < 10_000_000:
            values.append((name, parsed))
    return values

try:
    from transformers import AutoConfig
    kwargs = {{"trust_remote_code": True}}
    if TOKEN:
        kwargs["token"] = TOKEN
    cfg = AutoConfig.from_pretrained(MODEL, **kwargs)
    values = collect(cfg)
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None:
        values.extend(collect(text_cfg))
    if values:
        source, limit = max(values, key=lambda item: item[1])
        print(f"MAX_LEN::{{limit}}::{{source}}")
    else:
        print("MAX_LEN_UNKNOWN")
except Exception as exc:
    print("MAX_LEN_ERROR::" + repr(exc))
PY"#,
        token_export = token_export,
        model = serde_json::to_string(repo_id).unwrap_or_else(|_| "\"\"".to_string()),
    );
    let cmd = if docker_enabled {
        wrap_docker_cmd(&script, container_name)
    } else {
        script
    };
    let result = session.exec_blocking(&cmd).await.ok()?;
    for line in result.stdout.lines().map(str::trim) {
        if let Some(rest) = line.strip_prefix("MAX_LEN::") {
            let mut parts = rest.splitn(2, "::");
            let limit = parts.next()?.parse::<u32>().ok()?;
            let source = parts.next().unwrap_or("model config").to_string();
            return Some((limit, source));
        }
    }
    None
}

/// Per-port kill body: frees each given TCP port by `fuser`/`ss`. Safe to run on
/// the host or inside a container because it touches only the listed ports —
/// pass ONLY non-persistent embedder ports so protected embedders are untouched.
fn embedder_port_kill(non_persistent_ports: &[u16]) -> String {
    non_persistent_ports
        .iter()
        .map(|p| {
            format!(
                "(command -v fuser >/dev/null 2>&1 && fuser -k {p}/tcp 2>/dev/null) || true; \
                 (command -v ss >/dev/null 2>&1 && ss -ltnp 2>/dev/null | awk '/:{p} /{{print $0}}' | grep -oE 'pid=[0-9]+' | cut -d= -f2 | xargs -r kill -9 2>/dev/null) || true; ",
                p = p,
            )
        })
        .collect()
}

/// Build a shell command that stops only the embedder vLLM *processes* — never a
/// Docker container. It kills the generic embed-vLLM process patterns and force-
/// frees each given port. Broad pattern sweeps (`vllm.*pooling`, etc.) would
/// also match protected embedders, so cleanup stays port-targeted everywhere.
/// Pass only non-persistent ports.
/// Mirrors the cleanup in `main.rs::run_deploy_teacher_task`.
pub fn build_embedder_kill_cmd(non_persistent_ports: &[u16]) -> String {
    build_embedder_port_kill_cmd(non_persistent_ports)
}

/// Embedder stop: ONLY frees the given non-persistent ports (no broad pattern
/// sweep that could hit a protected embedder on another port).
pub fn build_embedder_port_kill_cmd(non_persistent_ports: &[u16]) -> String {
    format!("{}true", embedder_port_kill(non_persistent_ports))
}

fn build_custom_train_cmd(run: &Run, lora: &LoraConfig, hf_export: &str) -> Result<String> {
    let commands: Vec<String> = lora
        .custom_commands
        .iter()
        .map(|cmd| cmd.trim())
        .filter(|cmd| !cmd.is_empty())
        .map(|cmd| expand_custom_command(cmd, run))
        .collect();

    if commands.is_empty() {
        return Err(AppError::pipeline(
            "custom fine-tuning method selected but no commands were provided",
        ));
    }

    let train_yaml = format!("{}/train.yaml", run.remote_dir);
    let data_dir = format!("{}/data", run.remote_dir);
    let output_dir = format!("{}/lora", run.remote_dir);
    let base_model = llamafactory::resolve_trainable_repo(&run.student_model);
    let method_name = lora.custom_method_name.trim();
    let title = if method_name.is_empty() {
        "custom fine-tuning method"
    } else {
        method_name
    };

    let mut body = String::from("set -e\n");
    body.push_str(&format!(
        "echo {}\n",
        sh_quote(&format!("[custom] starting {title}"))
    ));
    for (idx, command) in commands.iter().enumerate() {
        body.push_str(&format!(
            "echo {}\n{}\n",
            sh_quote(&format!("[custom] step {}/{}", idx + 1, commands.len())),
            command
        ));
    }

    Ok(format!(
        "set -o pipefail; \
         {hf_export} cd {dir} && \
         export RUN_DIR={run_dir} TRAIN_YAML={train_yaml} DATA_DIR={data_dir} OUTPUT_DIR={output_dir} \
                STUDENT_MODEL={student_model} BASE_MODEL={base_model} \
                FT_LEARNING_RATE={learning_rate} FT_EPOCHS={epochs} FT_BATCH_SIZE={batch_size} \
                FT_GRADIENT_ACCUMULATION={gradient_accumulation} FT_CUTOFF_LEN={cutoff_len} \
                FT_SAVE_STEPS={save_steps} FT_LORA_R={rank} FT_LORA_ALPHA={alpha} FT_LORA_DROPOUT={dropout} && \
         mkdir -p {output_dir} && \
         : > {dir}/log.txt && : > {dir}/errorlog.txt && : > {dir}/train.log && \
         {{ {body} }} \
           > >(tee -a {dir}/log.txt {dir}/train.log) \
           2> >(tee -a {dir}/errorlog.txt {dir}/train.log >&2)",
        hf_export = hf_export,
        dir = sh_quote(&run.remote_dir),
        run_dir = sh_quote(&run.remote_dir),
        train_yaml = sh_quote(&train_yaml),
        data_dir = sh_quote(&data_dir),
        output_dir = sh_quote(&output_dir),
        student_model = sh_quote(&run.student_model),
        base_model = sh_quote(&base_model),
        learning_rate = sh_quote(&lora.learning_rate.to_string()),
        epochs = sh_quote(&lora.epochs.to_string()),
        batch_size = sh_quote(&lora.batch_size.to_string()),
        gradient_accumulation = sh_quote(&lora.gradient_accumulation.to_string()),
        cutoff_len = sh_quote(&lora.cutoff_len.to_string()),
        save_steps = sh_quote(&lora.save_steps.to_string()),
        rank = sh_quote(&lora.r.to_string()),
        alpha = sh_quote(&lora.alpha.to_string()),
        dropout = sh_quote(&lora.dropout.to_string()),
        body = body,
    ))
}

fn build_zrald_train_cmd(run: &Run, lora: &LoraConfig, hf_export: &str) -> Result<String> {
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
    let reward_model = if lora.zrald_reward_model.trim().is_empty() {
        run.teacher_cfg.repo_id.clone()
    } else {
        lora.zrald_reward_model.trim().to_string()
    };
    let train_questions = lora.zrald_train_questions.max(1);
    let benchmark_questions = lora.zrald_benchmark_questions.min(train_questions).max(1);
    let num_generations = lora.zrald_num_generations.clamp(2, 8);
    let max_completion_tokens = lora
        .zrald_max_completion_tokens
        .clamp(64, lora.cutoff_len.max(64));

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
from datasets import Dataset
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
if PER_DEVICE_BS < NUM_GENERATIONS or PER_DEVICE_BS % NUM_GENERATIONS != 0:
    adjusted = max(NUM_GENERATIONS, ((PER_DEVICE_BS + NUM_GENERATIONS - 1) // NUM_GENERATIONS) * NUM_GENERATIONS)
    print(f"[zrald] adjusting per-device batch size from {PER_DEVICE_BS} to {adjusted} so it is divisible by num_generations={NUM_GENERATIONS}", flush=True)
    PER_DEVICE_BS = adjusted

SYSTEM_PROMPT = (
    "You are the ZRALD student model. Answer with exactly two XML-style blocks: "
    "<thinking>brief reasoning</thinking><answer>final answer</answer>. "
    "Do not mention rewards, scoring, hidden references, or evaluator instructions."
)

def chat_prompt(question):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
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

def load_pool():
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
                messages = obj.get("messages")
                question = str(obj.get("question") or first_user(messages) or obj.get("instruction") or obj.get("prompt") or "").strip()
                reference = str(obj.get("answer") or strip_answer(first_assistant(messages)) or obj.get("output") or "").strip()
                if not question or not reference:
                    continue
                rag_context = str(obj.get("source_text") or obj.get("context") or obj.get("input") or "").strip()
                source = str(obj.get("source_chunk_id") or obj.get("source_file") or path.name)
                rubric = (
                    "Score from -1.0 to 1.0. 1.0 means the final answer is precise, complete, and supported. "
                    "0.0 means partially useful but needs correction. -1.0 means incorrect, contradictory, hallucinated, or empty. "
                    "Correctness and RAG faithfulness dominate style."
                )
                rows.append({
                    "prompt": chat_prompt(question),
                    "question": question,
                    "reference_answer": reference,
                    "rag_context": rag_context,
                    "rubric": rubric,
                    "source": source,
                })
        if rows:
            break
    dedup = {}
    for row in rows:
        key = hashlib.sha256((row["question"] + "\n" + row["reference_answer"]).encode("utf-8")).hexdigest()
        dedup.setdefault(key, row)
    rows = list(dedup.values())
    random.Random(3407).shuffle(rows)
    if not rows:
        raise SystemExit("[zrald] no usable RAG question rows found in qa_dataset/train/val JSONL")
    return rows

def dump_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def reward_url():
    if REWARD_ENDPOINT.endswith("/v1"):
        return REWARD_ENDPOINT + "/chat/completions"
    return REWARD_ENDPOINT + "/v1/chat/completions"

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
    ];
    for (needle, value) in replacements {
        py = py.replace(needle, &value);
    }

    let heredoc = format!(
        "cat > {script} <<'PYEOF'\n{py}\nPYEOF",
        script = sh_quote(&script_path),
        py = py
    );
    let bnb_amd_wheel = "https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_main/bitsandbytes-1.33.7.preview-py3-none-manylinux_2_24_x86_64.whl";
    let zrald_probe = "python3 -c 'import requests, datasets, unsloth; from trl import GRPOConfig, GRPOTrainer' >/dev/null 2>&1";
    let torch_hip_probe = "python3 -c 'import torch,sys; sys.exit(0 if getattr(torch.version,\"hip\",None) else 1)' >/dev/null 2>&1";

    Ok(format!(
        "set -o pipefail; \
         {hf_export} cd {dir} && \
         export UNSLOTH_IS_ROCM=1 PYTORCH_ROCM_ARCH=${{PYTORCH_ROCM_ARCH:-gfx1100}} && \
         ({torch_probe} || pip install --no-cache-dir --upgrade --force-reinstall \
             --index-url https://download.pytorch.org/whl/rocm7.0 \
             'torch>=2.4,<2.11.0' 'torchvision<0.26.0' 'torchaudio<2.11.0') && \
         ({probe} || \
             (pip install --no-cache-dir 'unsloth[amd]' 'unsloth_zoo' 'trl>=0.19.0' \
                 'datasets>=2.16.0' 'requests>=2.31.0' 'peft>=0.19,<0.20' \
                 'accelerate>=0.34.0' 'sentencepiece>=0.2.0' 'protobuf' 'hf_transfer' 'psutil' && \
              (pip install --force-reinstall --no-cache-dir --no-deps '{bnb_wheel}' || \
               pip install --force-reinstall --no-cache-dir --no-deps 'bitsandbytes>=0.49.1'))) && \
         mkdir -p {output_dir} && \
         : > {dir}/log.txt && : > {dir}/errorlog.txt && : > {dir}/train.log && \
         {heredoc} && \
         {{ echo {start_msg} && python3 {script}; }} \
           > >(tee -a {dir}/log.txt {dir}/train.log) \
           2> >(tee -a {dir}/errorlog.txt {dir}/train.log >&2)",
        hf_export = hf_export,
        dir = sh_quote(&run.remote_dir),
        torch_probe = torch_hip_probe,
        probe = zrald_probe,
        bnb_wheel = bnb_amd_wheel,
        output_dir = sh_quote(&output_dir),
        heredoc = heredoc,
        start_msg = sh_quote("[zrald] starting ZRALD RAG reward GRPO trainer"),
        script = sh_quote(&script_path),
    ))
}

/// Build a training command that runs unsloth's native GRPOTrainer instead
/// of LLaMA-Factory. GRPO isn't supported by LLaMA-Factory 0.9.4 (`stage` only
/// accepts pt|sft|rm|ppo|dpo|kto), so we shell out to a Python script that
/// mirrors the structure of the AMD GRPO notebooks in `guide amd/`:
///   - `Gemma4_(E2B)_Reinforcement_Learning_Sudoku_Game.ipynb`
///   - `gpt_oss_(20B)_Reinforcement_Learning_2048_Game_BF16.ipynb`
///
/// The reward functions in those notebooks are task-specific (Sudoku, 2048).
/// For a general-purpose GRPO run on the user's dataset we use a length+stop-
/// token reward as a sensible baseline. Users who need bespoke rewards should
/// switch to the `custom` method and paste their own reward script.
fn build_grpo_train_cmd(run: &Run, lora: &LoraConfig, hf_export: &str) -> Result<String> {
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

    // Write the Python script via a literal-marker heredoc so bash does no
    // expansion (single-quoted 'PYEOF'). The shell-side script then runs it.
    let heredoc = format!(
        "cat > {script} <<'PYEOF'\n{py}\nPYEOF",
        script = sh_quote(&script_path),
        py = py
    );

    // GRPO doesn't read train.yaml — the surrounding pipeline still writes it
    // for resume/inspect purposes, but the trainer ignores it.
    let _ = train_yaml;

    Ok(format!(
        "set -o pipefail; \
         {hf_export} cd {dir} && \
         mkdir -p {output_dir} && \
         : > {dir}/log.txt && : > {dir}/errorlog.txt && : > {dir}/train.log && \
         {heredoc} && \
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

fn expand_custom_command(command: &str, run: &Run) -> String {
    let base_model = llamafactory::resolve_trainable_repo(&run.student_model);
    command
        .replace("{run_dir}", &run.remote_dir)
        .replace("{train_yaml}", &format!("{}/train.yaml", run.remote_dir))
        .replace("{data_dir}", &format!("{}/data", run.remote_dir))
        .replace("{output_dir}", &format!("{}/lora", run.remote_dir))
        .replace("{student_model}", &run.student_model)
        .replace("{base_model}", &base_model)
}

/// Robustly write a file to the remote, ensuring it is visible inside the
/// container if Docker is enabled.
async fn write_file_auto(
    session: &SshSession,
    docker_enabled: bool,
    container_name: &str,
    remote_path: &str,
    content: &str,
) -> Result<()> {
    // Always write to host for persistence/log-gathering
    session.write_file(remote_path, content).await?;

    // If Docker is enabled, copy the file from host into the container using docker cp
    if docker_enabled {
        let dir = match remote_path.rsplit_once('/') {
            Some((d, _)) if !d.is_empty() => d.to_string(),
            _ => ".".to_string(),
        };
        // Create the parent directory inside the container
        let mkdir_inner = format!("mkdir -p \"{}\"", dir);
        let mkdir_cmd = wrap_docker_cmd(&mkdir_inner, container_name);
        let r_mkdir = session.exec_blocking(&mkdir_cmd).await?;
        if r_mkdir.exit_code != 0 {
            return Err(AppError::ssh(format!(
                "mkdir (docker) exit={} stderr={}",
                r_mkdir.exit_code, r_mkdir.stderr
            )));
        }

        // Copy from host path to container path
        let cp_cmd = format!(
            "docker cp {} {}:{}",
            remote_path, container_name, remote_path
        );
        let r_cp = session.exec_blocking(&cp_cmd).await?;
        if r_cp.exit_code != 0 {
            return Err(AppError::ssh(format!(
                "docker cp exit={} stderr={}",
                r_cp.exit_code, r_cp.stderr
            )));
        }
    }
    Ok(())
}

pub async fn ensure_container(session: &SshSession, cfg: &DockerConfig) -> Result<String> {
    if !cfg.enabled {
        return Ok(cfg.container_name.clone());
    }
    // Check if ANY container is running, with image info so we can detect a
    // compatible-but-differently-named container (e.g. user already has one
    // called `rocm` instead of the configured `rocm-vllm`). With ROCm
    // containers started using --network=host, two containers sharing the
    // host network can't both bind the same vLLM port, so reusing the
    // existing one is the correct behaviour.
    let check_cmd = "docker ps --format '{{.Names}}\t{{.Image}}'";
    let r = session.exec_blocking(check_cmd).await?;
    let running: Vec<(String, String)> = r
        .stdout
        .lines()
        .filter_map(|l| {
            let mut it = l.splitn(2, '\t');
            let n = it.next()?.trim().to_string();
            let img = it.next().unwrap_or("").trim().to_string();
            if n.is_empty() {
                None
            } else {
                Some((n, img))
            }
        })
        .collect();
    let running_names: Vec<String> = running.iter().map(|(n, _)| n.clone()).collect();

    // If our preferred container is running, use it
    if running_names.contains(&cfg.container_name) {
        return Ok(cfg.container_name.clone());
    }

    // Look for any *other* running container that appears to be a ROCm/vLLM
    // image. If we find one, reuse it instead of creating a sibling that
    // will fight for the same host port (because of --network=host).
    let cfg_img_lower = cfg.image_name.to_lowercase();
    let cfg_img_prefix = cfg_img_lower.split(':').next().unwrap_or("").to_string();
    let target_is_sglang = cfg_img_lower.contains("sglang");
    let target_is_vllm = cfg_img_lower.contains("vllm");
    let candidate = running.iter().find(|(_, img)| {
        let il = img.to_lowercase();
        if target_is_sglang {
            il.contains("sglang")
        } else if target_is_vllm {
            il.contains("vllm") || il.contains("rocm/vllm")
        } else {
            !cfg_img_prefix.is_empty() && il.starts_with(&cfg_img_prefix)
        }
    });
    if let Some((name, _img)) = candidate {
        return Ok(name.clone());
    }

    // Check if exists but stopped
    let check_exists = "docker ps -a --format '{{.Names}}'";
    let r_exists = session.exec_blocking(check_exists).await?;
    let exists = r_exists
        .stdout
        .lines()
        .any(|l| l.trim() == cfg.container_name);
    if exists {
        let start_cmd = format!("docker start {}", cfg.container_name);
        let r_start = session.exec_blocking(&start_cmd).await?;
        if r_start.exit_code != 0 {
            return Err(AppError::ssh(format!(
                "Failed to start container '{}': {}",
                cfg.container_name, r_start.stderr
            )));
        }
        return Ok(cfg.container_name.clone());
    }

    // Doesn't exist, run it
    let run_cmd = format!(
        "docker run -d --name {} {} --entrypoint sleep {} infinity",
        cfg.container_name, cfg.start_args, cfg.image_name
    );
    let r_run = session.exec_blocking(&run_cmd).await?;
    if r_run.exit_code != 0 {
        return Err(AppError::ssh(format!(
            "Failed to run container '{}': {}",
            cfg.container_name, r_run.stderr
        )));
    }
    Ok(cfg.container_name.clone())
}

// ── HF dataset helpers ─────────────────────────────────────────────────────

/// Quote a string so it's safe to embed inside single quotes in a bash command.
pub fn sh_quote(s: &str) -> String {
    // wrap in '...' and escape any embedded '
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for ch in s.chars() {
        if ch == '\'' {
            out.push_str("'\"'\"'");
        } else {
            out.push(ch);
        }
    }
    out.push('\'');
    out
}

/// Try to pull `qa_dataset.jsonl` from a HF *dataset* repo using the REST API.
/// Returns Ok(None) if the file isn't present in the repo.
async fn seed_from_hf_dataset(repo_id: &str, hf_token: Option<&str>) -> Result<Option<String>> {
    let repo = repo_id.trim();
    if repo.is_empty() {
        return Ok(None);
    }

    let url = format!(
        "https://huggingface.co/datasets/{}/resolve/main/qa_dataset.jsonl",
        repo
    );

    let mut req = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .map_err(|e| AppError::pipeline(format!("build http client: {e}")))?
        .get(&url);

    if let Some(tok) = hf_token.filter(|t| !t.trim().is_empty()) {
        req = req.bearer_auth(tok);
    }

    let res = req
        .send()
        .await
        .map_err(|e| AppError::pipeline(format!("hf download failed: {e}")))?;

    if res.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(None);
    }
    if !res.status().is_success() {
        return Err(AppError::pipeline(format!(
            "hf download returned HTTP {}",
            res.status()
        )));
    }

    let content = res
        .text()
        .await
        .map_err(|e| AppError::pipeline(format!("hf text failed: {e}")))?;
    if content.trim().is_empty() {
        return Ok(None);
    }

    Ok(Some(content))
}

/// Upload the current `qa_dataset.jsonl` to a HF *dataset* repo. Idempotent;
/// creates the repo if missing. `kept` is just embedded into the commit msg.
///
/// Implementation note: we now go straight to the HF REST API from the local
/// machine instead of shelling out to `huggingface-cli` over SSH on the
/// droplet. The CLI path silently failed on us when (a) the droplet image
/// had a `huggingface_hub` version where the `repo create` / `upload`
/// subcommands moved or got renamed, and (b) error output was suppressed
/// by the redirects in the chained shell command. The REST path is faster,
/// has no Python/pip dependency, and surfaces a real HTTP status when the
/// token is wrong or the repo can't be created.
///
/// The `session`, `docker_enabled`, `container_name`, and `remote_dir`
/// arguments are kept for now to preserve callers; they're unused by this
/// implementation.
async fn push_jsonl_to_hf_dataset(
    repo_id: &str,
    private: bool,
    hf_token: Option<&str>,
    jsonl: &str,
    kept: u64,
) -> Result<()> {
    use base64::Engine;

    let tok = hf_token.filter(|t| !t.trim().is_empty()).ok_or_else(|| {
        AppError::pipeline(
            "HF dataset upload requested but no HF token configured (Settings → Hugging Face)",
        )
    })?;
    let repo = repo_id.trim();
    if repo.is_empty() {
        return Err(AppError::pipeline("HF dataset push: empty repo id"));
    }

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10 * 60))
        .user_agent("fine-tune-tauri/0.1")
        .build()
        .map_err(|e| AppError::pipeline(format!("build http client: {e}")))?;

    let parts: Vec<&str> = repo.split('/').collect();
    let (org, name) = if parts.len() == 2 {
        let namespace = parts[0];
        let name = parts[1];

        // Fetch whoami to see if namespace is the username
        let mut is_user = true;
        if let Ok(res) = client
            .get("https://huggingface.co/api/whoami-v2")
            .bearer_auth(tok)
            .send()
            .await
        {
            if res.status().is_success() {
                if let Ok(user_info) = res.json::<serde_json::Value>().await {
                    if let Some(username) = user_info.get("name").and_then(|v| v.as_str()) {
                        if !username.eq_ignore_ascii_case(namespace) {
                            is_user = false;
                        }
                    }
                }
            }
        }

        if is_user {
            (None, name)
        } else {
            (Some(namespace), name)
        }
    } else if parts.len() > 2 {
        return Err(AppError::pipeline(format!(
            "HF dataset push: invalid repo id '{repo_id}' (expected '<user-or-org>/<name>')"
        )));
    } else {
        (None, repo)
    };

    // 1. Create the repo (idempotent — HF returns 409 if it already exists).
    let mut create_body = serde_json::json!({
        "type": "dataset",
        "name": name,
        "private": private,
    });
    if let Some(o) = org {
        if let Some(obj) = create_body.as_object_mut() {
            obj.insert("organization".to_string(), serde_json::json!(o));
        }
    }
    let create_res = client
        .post("https://huggingface.co/api/repos/create")
        .bearer_auth(tok)
        .json(&create_body)
        .send()
        .await
        .map_err(|e| AppError::pipeline(format!("HF create-repo network error: {e}")))?;
    let create_status = create_res.status();
    if !create_status.is_success() {
        let body = create_res.text().await.unwrap_or_default();
        let already_exists = body.to_lowercase().contains("already")
            || body.contains("RepoExistsError")
            || create_status.as_u16() == 409;
        if !already_exists {
            let safe = body.replace(tok, "***");
            return Err(AppError::pipeline(format!(
                "HF create-repo {create_status}: {safe}"
            )));
        }
    }

    // 2. Ask the Hub how this file should be uploaded. This mirrors
    // huggingface_hub's preupload step and prevents us from forcing small JSONL
    // checkpoints through LFS unnecessarily.
    use sha2::{Digest, Sha256};
    let content_bytes = jsonl.as_bytes();
    let size = content_bytes.len();

    let mut hasher = Sha256::new();
    hasher.update(content_bytes);
    let hash = format!("{:x}", hasher.finalize());

    let sample_len = content_bytes.len().min(512);
    let sample = base64::engine::general_purpose::STANDARD.encode(&content_bytes[..sample_len]);
    let preupload_url = format!(
        "https://huggingface.co/api/datasets/{}/preupload/main",
        repo
    );
    let preupload_body = serde_json::json!({
        "files": [
            {
                "path": "qa_dataset.jsonl",
                "sample": sample,
                "size": size,
            }
        ]
    });
    let preupload_res = client
        .post(&preupload_url)
        .bearer_auth(tok)
        .json(&preupload_body)
        .send()
        .await
        .map_err(|e| AppError::pipeline(format!("HF preupload network error: {e}")))?;
    let preupload_status = preupload_res.status();
    if !preupload_status.is_success() {
        let body = preupload_res.text().await.unwrap_or_default();
        let safe = body.replace(tok, "***");
        return Err(AppError::pipeline(format!(
            "HF preupload {preupload_status}: {safe}"
        )));
    }
    let preupload_json: serde_json::Value = preupload_res
        .json()
        .await
        .map_err(|e| AppError::pipeline(format!("HF preupload response not JSON: {e}")))?;

    let file_info = preupload_json
        .get("files")
        .and_then(|f| f.as_array())
        .and_then(|files| files.first())
        .ok_or_else(|| AppError::pipeline("HF preupload response missing files[0]"))?;
    if file_info
        .get("shouldIgnore")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        return Err(AppError::pipeline(
            "HF preupload says qa_dataset.jsonl is ignored by repo .gitignore",
        ));
    }
    let upload_mode = file_info
        .get("uploadMode")
        .and_then(|v| v.as_str())
        .unwrap_or("regular");

    if upload_mode == "lfs" {
        // 3. Pre-upload the blob via Git LFS. The correct dataset LFS route is
        // /datasets/<repo>.git/info/lfs/objects/batch, not /api/datasets/<repo>.git.
        let batch_url = format!(
            "https://huggingface.co/datasets/{}.git/info/lfs/objects/batch",
            repo
        );
        let batch_body = serde_json::json!({
            "operation": "upload",
            "transfers": ["basic"],
            "ref": { "name": "main" },
            "objects": [
                {
                    "oid": hash,
                    "size": size,
                }
            ],
            "hash_algo": "sha256",
        });

        let lfs_res = client
            .post(&batch_url)
            .header("Accept", "application/vnd.git-lfs+json")
            .header("Content-Type", "application/vnd.git-lfs+json")
            .bearer_auth(tok)
            .json(&batch_body)
            .send()
            .await
            .map_err(|e| AppError::pipeline(format!("HF LFS batch network error: {e}")))?;

        let lfs_status = lfs_res.status();
        if !lfs_status.is_success() {
            let body = lfs_res.text().await.unwrap_or_default();
            let safe = body.replace(tok, "***");
            return Err(AppError::pipeline(format!(
                "HF LFS batch {lfs_status}: {safe}"
            )));
        }

        let res_json: serde_json::Value = lfs_res
            .json()
            .await
            .map_err(|e| AppError::pipeline(format!("HF LFS batch response not JSON: {e}")))?;

        let obj = res_json
            .get("objects")
            .and_then(|o| o.as_array())
            .and_then(|objects| objects.first())
            .ok_or_else(|| AppError::pipeline("HF LFS batch response missing objects[0]"))?;
        if let Some(err_obj) = obj.get("error") {
            let err_msg = err_obj
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown LFS batch error");
            return Err(AppError::pipeline(format!(
                "HF LFS batch object error: {err_msg}"
            )));
        }
        if let Some(actions) = obj.get("actions") {
            if let Some(upload) = actions.get("upload") {
                let href = upload
                    .get("href")
                    .and_then(|h| h.as_str())
                    .ok_or_else(|| AppError::pipeline("HF LFS response missing upload href"))?;
                let headers = upload.get("header").and_then(|h| h.as_object());

                let mut req = client.put(href).body(content_bytes.to_vec());
                if let Some(headers_map) = headers {
                    for (k, v) in headers_map {
                        if let Some(v_str) = v.as_str() {
                            req = req.header(k, v_str);
                        }
                    }
                }

                let upload_res = req.send().await.map_err(|e| {
                    AppError::pipeline(format!("HF LFS upload PUT request network error: {e}"))
                })?;

                let upload_status = upload_res.status();
                if !upload_status.is_success() {
                    let body = upload_res.text().await.unwrap_or_default();
                    return Err(AppError::pipeline(format!(
                        "HF LFS upload PUT returned {upload_status}: {body}"
                    )));
                }
            }

            if let Some(verify) = actions.get("verify") {
                let href = verify
                    .get("href")
                    .and_then(|h| h.as_str())
                    .ok_or_else(|| AppError::pipeline("HF LFS response missing verify href"))?;
                let verify_res = client
                    .post(href)
                    .header("Accept", "application/vnd.git-lfs+json")
                    .header("Content-Type", "application/vnd.git-lfs+json")
                    .bearer_auth(tok)
                    .json(&serde_json::json!({
                        "oid": hash,
                        "size": size,
                    }))
                    .send()
                    .await
                    .map_err(|e| AppError::pipeline(format!("HF LFS verify network error: {e}")))?;
                let verify_status = verify_res.status();
                if !verify_status.is_success() {
                    let body = verify_res.text().await.unwrap_or_default();
                    let safe = body.replace(tok, "***");
                    return Err(AppError::pipeline(format!(
                        "HF LFS verify {verify_status}: {safe}"
                    )));
                }
            }
        }
    } else if upload_mode != "regular" {
        return Err(AppError::pipeline(format!(
            "HF preupload returned unsupported uploadMode '{upload_mode}'"
        )));
    }

    // 4. Commit. For LFS uploads, the Hub commit API expects an `lfsFile`
    // operation with oid/size metadata; for regular uploads it expects the file
    // content as base64.
    let summary = format!("auto dataset: {kept} pairs");
    let file_line = if upload_mode == "lfs" {
        serde_json::json!({
            "key": "lfsFile",
            "value": {
                "path": "qa_dataset.jsonl",
                "algo": "sha256",
                "oid": hash,
                "size": size,
            }
        })
    } else {
        let encoded = base64::engine::general_purpose::STANDARD.encode(content_bytes);
        serde_json::json!({
            "key": "file",
            "value": {
                "path": "qa_dataset.jsonl",
                "content": encoded,
                "encoding": "base64",
            }
        })
    };

    let header_line = serde_json::json!({
        "key": "header",
        "value": { "summary": summary }
    });
    let ndjson = format!("{}\n{}\n", header_line, file_line);

    let commit_url = format!("https://huggingface.co/api/datasets/{}/commit/main", repo);
    let commit_res = client
        .post(&commit_url)
        .bearer_auth(tok)
        .header("Content-Type", "application/x-ndjson")
        .body(ndjson)
        .send()
        .await
        .map_err(|e| AppError::pipeline(format!("HF commit network error: {e}")))?;

    let status = commit_res.status();
    if !status.is_success() {
        let body = commit_res.text().await.unwrap_or_default();
        let safe = body.replace(tok, "***");
        return Err(AppError::pipeline(format!("HF commit {status}: {safe}")));
    }
    Ok(())
}
