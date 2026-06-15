//! Robot capture queue + research/ingest pipeline.
//!
//! Flow for one unfamiliar object the robot uploads:
//!   1. `enqueue_capture`  — validate, dedupe, store image + metadata (status=Pending)
//!   2. `research_capture` — OCR the image (PaddleOCR-VL on the GPU), run web
//!      research, embed the cited packet into Qdrant (status=Researched)
//!   3. operator approves  — `set_status(Approved)`; the training run is then
//!      kicked off by the caller (server/desktop) via the existing pipeline.
//!
//! Training + Hugging Face publish are NEVER triggered automatically here —
//! they are gated on human approval per the whitepaper's safety guardrails.

use crate::config::{robot_dir, AppConfig, QdrantConfig};
use crate::error::{AppError, Result};
use crate::ingest::{self, EmbeddingConfig, EmbeddingProvider, IngestOptions, PaddleOcrOptions};
use crate::research;
use crate::ssh::SshSessionManager;
use base64::Engine;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use tokio::fs;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureStatus {
    Pending,
    Researching,
    Researched,
    Approved,
    Rejected,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Capture {
    pub id: String,
    pub robot_id: String,
    /// The robot's own label guess for the object, if any.
    pub label_guess: String,
    pub confidence: f32,
    pub scene_notes: String,
    /// Path to the stored image on the server.
    pub image_path: String,
    /// sha256 of the image bytes (used for dedupe).
    pub image_sha256: String,
    pub status: CaptureStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    /// OCR text extracted from the image.
    pub ocr_text: Option<String>,
    /// Sources cited during research.
    pub citations: Vec<research::Citation>,
    /// Number of chunks embedded into Qdrant for this capture.
    pub chunks_ingested: u64,
    pub error: Option<String>,
}

/// Incoming capture payload from the robot (image as base64).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CaptureInput {
    pub robot_id: String,
    #[serde(default)]
    pub label_guess: String,
    #[serde(default)]
    pub confidence: f32,
    #[serde(default)]
    pub scene_notes: String,
    /// Base64-encoded image bytes.
    pub image_base64: String,
    /// File extension hint, e.g. "jpg" | "png". Defaults to jpg.
    #[serde(default)]
    pub image_ext: Option<String>,
}

fn capture_json_path(id: &str) -> Result<PathBuf> {
    Ok(robot_dir()?.join(format!("{id}.json")))
}

async fn save_capture(c: &Capture) -> Result<()> {
    crate::config::ensure_dirs().await?;
    let txt = serde_json::to_string_pretty(c)?;
    fs::write(capture_json_path(&c.id)?, txt).await?;
    Ok(())
}

pub async fn load_capture(id: &str) -> Result<Capture> {
    let txt = fs::read_to_string(capture_json_path(id)?)
        .await
        .map_err(|_| AppError::NotFound(format!("capture '{id}'")))?;
    serde_json::from_str(&txt).map_err(|e| AppError::config(format!("parse capture {id}: {e}")))
}

pub async fn list_captures() -> Result<Vec<Capture>> {
    crate::config::ensure_dirs().await?;
    let dir = robot_dir()?;
    let mut out = Vec::new();
    let mut rd = match fs::read_dir(&dir).await {
        Ok(rd) => rd,
        Err(_) => return Ok(out),
    };
    while let Ok(Some(entry)) = rd.next_entry().await {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        if let Ok(txt) = fs::read_to_string(&path).await {
            if let Ok(c) = serde_json::from_str::<Capture>(&txt) {
                out.push(c);
            }
        }
    }
    out.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    Ok(out)
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

/// Validate, dedupe, and persist a new capture. Returns the stored capture in
/// `Pending` status. Does NOT research yet — call `research_capture` next.
pub async fn enqueue_capture(cfg: &AppConfig, input: CaptureInput) -> Result<Capture> {
    if cfg.robot.enabled
        && !cfg.robot.allowed_robot_ids.is_empty()
        && !cfg.robot.allowed_robot_ids.contains(&input.robot_id)
    {
        return Err(AppError::other(format!(
            "robot id '{}' is not in the allowlist",
            input.robot_id
        )));
    }
    if input.confidence < cfg.robot.min_capture_confidence {
        return Err(AppError::other(format!(
            "capture confidence {:.2} below minimum {:.2}",
            input.confidence, cfg.robot.min_capture_confidence
        )));
    }

    let bytes = base64::engine::general_purpose::STANDARD
        .decode(input.image_base64.trim())
        .map_err(|e| AppError::other(format!("decode image_base64: {e}")))?;
    if bytes.is_empty() {
        return Err(AppError::other("capture image is empty"));
    }
    let sha = sha256_hex(&bytes);

    // Dedupe: if an existing capture has the same image hash within the window
    // (and is not rejected), return it instead of creating a duplicate.
    let now = Utc::now();
    if cfg.robot.dedupe_window_secs > 0 {
        for existing in list_captures().await.unwrap_or_default() {
            if existing.image_sha256 == sha
                && existing.status != CaptureStatus::Rejected
                && (now - existing.created_at).num_seconds() >= 0
                && (now - existing.created_at).num_seconds() <= cfg.robot.dedupe_window_secs as i64
            {
                return Ok(existing);
            }
        }
    }

    let id = format!("cap-{}", Uuid::new_v4());
    let ext = input
        .image_ext
        .as_deref()
        .map(|s| s.trim_start_matches('.').to_lowercase())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "jpg".to_string());
    let image_path = robot_dir()?.join(format!("{id}.{ext}"));
    fs::write(&image_path, &bytes).await?;

    let capture = Capture {
        id: id.clone(),
        robot_id: input.robot_id,
        label_guess: input.label_guess,
        confidence: input.confidence,
        scene_notes: input.scene_notes,
        image_path: image_path.to_string_lossy().into_owned(),
        image_sha256: sha,
        status: CaptureStatus::Pending,
        created_at: now,
        updated_at: now,
        ocr_text: None,
        citations: vec![],
        chunks_ingested: 0,
        error: None,
    };
    save_capture(&capture).await?;
    Ok(capture)
}

/// Set a capture's status (used by the approval gate and rejection).
pub async fn set_status(id: &str, status: CaptureStatus) -> Result<Capture> {
    let mut c = load_capture(id).await?;
    c.status = status;
    c.updated_at = Utc::now();
    save_capture(&c).await?;
    Ok(c)
}

/// OCR the captured image, research the object online, and embed the cited
/// research packet into Qdrant. Moves the capture to `Researched` (or `Failed`).
///
/// Reuses the existing ingest pipeline (`ingest::ingest_files`) for the
/// embed→Qdrant path, exactly like document ingestion, so there is one code
/// path for chunking + embedding + upsert.
pub async fn research_capture(cfg: &AppConfig, id: &str) -> Result<Capture> {
    let mut c = load_capture(id).await?;
    c.status = CaptureStatus::Researching;
    c.updated_at = Utc::now();
    c.error = None;
    save_capture(&c).await?;

    let outcome = research_capture_inner(cfg, &c).await;
    match outcome {
        Ok((ocr_text, packet, chunks)) => {
            c.ocr_text = Some(ocr_text);
            c.citations = packet.citations;
            c.chunks_ingested = chunks;
            c.status = if packet.blocked {
                CaptureStatus::Rejected
            } else {
                CaptureStatus::Researched
            };
            c.error = packet.block_reason;
            c.updated_at = Utc::now();
            save_capture(&c).await?;
            Ok(c)
        }
        Err(e) => {
            c.status = CaptureStatus::Failed;
            c.error = Some(e.to_string());
            c.updated_at = Utc::now();
            save_capture(&c).await?;
            Err(e)
        }
    }
}

async fn research_capture_inner(
    cfg: &AppConfig,
    c: &Capture,
) -> Result<(String, research::ResearchPacket, u64)> {
    // --- 1. OCR the image via PaddleOCR-VL (through the GPU droplet) ---
    let ocr_opts = PaddleOcrOptions {
        enabled: true,
        host: cfg.ssh.host.clone(),
        port: if cfg.paddle_ocr.port != 0 {
            cfg.paddle_ocr.port
        } else {
            8118
        },
        model_name: cfg.paddle_ocr.model_name.clone(),
    };
    let ssh_mgr = if !cfg.ssh.host.is_empty() {
        Some(SshSessionManager::new(cfg.ssh.clone()))
    } else {
        None
    };
    let ocr_text = ingest::ocr_image(
        std::path::Path::new(&c.image_path),
        &ocr_opts,
        ssh_mgr.as_ref(),
    )
    .await
    .unwrap_or_default();

    // --- 2. Build the research query and search the web ---
    let mut query_parts: Vec<String> = Vec::new();
    if !c.label_guess.trim().is_empty() {
        query_parts.push(c.label_guess.trim().to_string());
    }
    if !ocr_text.trim().is_empty() {
        // keep the query short — first ~120 chars of OCR text
        query_parts.push(ocr_text.trim().chars().take(120).collect());
    }
    if query_parts.is_empty() {
        query_parts.push("unidentified object".to_string());
    }
    let query = query_parts.join(" ");
    let packet =
        research::research_object(&cfg.web_research, &query, &Utc::now().to_rfc3339()).await?;

    if packet.blocked {
        return Ok((ocr_text, packet, 0));
    }

    // --- 3. Embed the cited packet into Qdrant via the existing ingest path ---
    let collection = if cfg.robot.research_collection.trim().is_empty() {
        "kb_robot".to_string()
    } else {
        cfg.robot.research_collection.clone()
    };
    let qcfg = QdrantConfig {
        endpoint: if cfg.qdrant.endpoint.is_empty() {
            format!("http://{}:6333", cfg.ssh.host)
        } else {
            cfg.qdrant.endpoint.clone()
        },
        api_key: cfg.qdrant.api_key.clone(),
        collection: collection.clone(),
    };

    let emb = cfg
        .embedders
        .first()
        .ok_or_else(|| AppError::pipeline("no embedding embedders configured"))?;
    let embedding_config = EmbeddingConfig {
        provider: EmbeddingProvider::Vllm,
        api_url: emb.api_url(&cfg.ssh.host),
        api_key: String::new(),
        model_id: emb.model_id.clone(),
        concurrency: Some(emb.concurrency as usize),
    };

    // Write the research packet to a temp .txt and feed it through the same
    // ingest pipeline used for documents.
    let packet_path = robot_dir()?.join(format!("{}_packet.txt", c.id));
    fs::write(&packet_path, &packet.markdown).await?;

    let cancel = Arc::new(AtomicBool::new(false));
    let noop: ingest::ProgressFn = Box::new(|_, _, _, _| {});
    let summary = ingest::ingest_files(
        vec![packet_path.to_string_lossy().into_owned()],
        qcfg,
        embedding_config,
        IngestOptions {
            tag: Some(format!("robot:{}", c.id)),
            chunk_size: None,
            chunk_overlap: None,
            vector_dim: None,
        },
        cancel,
        noop,
        PaddleOcrOptions::default(),
        ssh_mgr.map(Arc::new),
    )
    .await?;

    let _ = fs::remove_file(&packet_path).await;
    Ok((ocr_text, packet, summary.total_chunks))
}
