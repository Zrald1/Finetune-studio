#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod config;
mod digitalocean;
mod droplet_usage;
mod error;
mod generator;
mod guides;
mod hf;
mod ingest;
mod llamafactory;
mod manifest;
mod pipeline;
mod qdrant;
mod research;
mod robot;
mod runs;
mod serve;
mod ssh;

use crate::config::{
    AppConfig, DigitalOceanConfig, DockerConfig, EmbedderConfig, QdrantConfig, SshConfig,
    TeacherConfig,
};
use crate::error::{AppError, Result};
use crate::pipeline::{PipelineRegistry, RunConfig};
use crate::qdrant::Chunk;
use crate::runs::Run;
use crate::ssh::{GpuState, SshSession, SshSessionManager, StreamChunk};
use parking_lot::Mutex;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;
use tauri::{AppHandle, Emitter, Manager, State};
use tokio::sync::mpsc;
use uuid::Uuid;

// ── shared state ───────────────────────────────────────────────────────────

#[derive(Default)]
struct AppState {
    pipeline: Arc<PipelineRegistry>,
    streams: Mutex<HashMap<String, Arc<std::sync::atomic::AtomicBool>>>,
}

// ── config commands ────────────────────────────────────────────────────────

#[tauri::command]
async fn save_config(cfg: AppConfig) -> Result<()> {
    config::save(&cfg).await
}

#[tauri::command]
async fn load_config() -> Result<AppConfig> {
    config::load().await
}

#[tauri::command]
async fn read_local_file_text(path: String) -> Result<String> {
    let content = tokio::fs::read_to_string(&path)
        .await
        .map_err(|e| AppError::config(format!("read local file `{}`: {}", path, e)))?;
    Ok(content)
}

fn is_ingestable_extension(ext: &str) -> bool {
    matches!(
        ext,
        "pdf"
            | "txt"
            | "md"
            | "docx"
            | "pptx"
            | "ppt"
            | "png"
            | "jpg"
            | "jpeg"
            | "webp"
            | "gif"
            | "bmp"
            | "tiff"
            | "tif"
    )
}

#[tauri::command]
async fn list_ingestable_files(folder_path: String) -> Result<Vec<String>> {
    let root = std::path::PathBuf::from(folder_path.trim());
    if root.as_os_str().is_empty() {
        return Err(AppError::config("folder path is empty"));
    }
    if !root.is_dir() {
        return Err(AppError::config(format!(
            "`{}` is not a folder",
            root.display()
        )));
    }

    tokio::task::spawn_blocking(move || {
        let mut files = Vec::<String>::new();
        let mut stack = vec![root];
        while let Some(dir) = stack.pop() {
            let entries = match std::fs::read_dir(&dir) {
                Ok(entries) => entries,
                Err(_) => continue,
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                    continue;
                }
                let Some(ext) = path
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|s| s.to_ascii_lowercase())
                else {
                    continue;
                };
                if is_ingestable_extension(&ext) {
                    files.push(path.to_string_lossy().into_owned());
                }
            }
        }
        files.sort_by_key(|p| p.to_ascii_lowercase());
        Ok(files)
    })
    .await
    .map_err(|e| AppError::config(format!("scan folder task failed: {e}")))?
}

#[tauri::command]
async fn save_ingest_state(state_json: String) -> Result<()> {
    config::ensure_dirs().await?;
    let path = config::app_dir()?.join("ingest_state.json");
    tokio::fs::write(&path, state_json)
        .await
        .map_err(|e| AppError::config(format!("write ingest state `{}`: {}", path.display(), e)))?;
    Ok(())
}

#[tauri::command]
async fn load_ingest_state() -> Result<String> {
    let path = config::app_dir()?.join("ingest_state.json");
    match tokio::fs::read_to_string(&path).await {
        Ok(content) => Ok(content),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok("{}".to_string()),
        Err(e) => Err(AppError::config(format!(
            "read ingest state `{}`: {}",
            path.display(),
            e
        ))),
    }
}

#[tauri::command]
async fn do_list_gpu_sizes(cfg: DigitalOceanConfig) -> Result<Vec<digitalocean::DoSize>> {
    digitalocean::list_gpu_sizes(&cfg).await
}

#[tauri::command]
async fn do_list_droplets(cfg: DigitalOceanConfig) -> Result<Vec<digitalocean::DoDroplet>> {
    digitalocean::list_droplets(&cfg).await
}

#[tauri::command]
async fn do_list_gpu_droplets(cfg: DigitalOceanConfig) -> Result<Vec<digitalocean::DoDroplet>> {
    digitalocean::list_gpu_droplets(&cfg).await
}

#[tauri::command]
async fn do_list_regions(cfg: DigitalOceanConfig) -> Result<Vec<digitalocean::DoRegion>> {
    digitalocean::list_regions(&cfg).await
}

#[tauri::command]
async fn do_list_images(cfg: DigitalOceanConfig) -> Result<Vec<digitalocean::DoImage>> {
    digitalocean::list_images(&cfg).await
}

#[tauri::command]
async fn do_list_ssh_keys(cfg: DigitalOceanConfig) -> Result<Vec<digitalocean::DoSshKey>> {
    digitalocean::list_ssh_keys(&cfg).await
}

#[tauri::command]
async fn do_list_projects(cfg: DigitalOceanConfig) -> Result<Vec<digitalocean::DoProject>> {
    digitalocean::list_projects(&cfg).await
}

#[tauri::command]
async fn do_get_account(cfg: DigitalOceanConfig) -> Result<digitalocean::DoAccount> {
    digitalocean::get_account(&cfg).await
}

#[tauri::command]
async fn do_create_gpu_droplet(cfg: DigitalOceanConfig) -> Result<digitalocean::DoDroplet> {
    digitalocean::create_droplet(&cfg).await
}

#[tauri::command]
async fn do_destroy_droplet(cfg: DigitalOceanConfig, droplet_id: u64) -> Result<()> {
    digitalocean::destroy_droplet(&cfg, droplet_id).await
}

#[tauri::command]
async fn do_list_droplet_usage() -> Result<Vec<droplet_usage::DropletUsageSummary>> {
    droplet_usage::list_summaries()
}

#[tauri::command]
async fn do_export_droplet_usage_csv() -> Result<String> {
    droplet_usage::export_csv()
}

// ── ssh commands ───────────────────────────────────────────────────────────

#[tauri::command]
async fn test_ssh(cfg: SshConfig) -> Result<String> {
    let s = SshSession::connect(&cfg).await?;
    let r = s.exec_blocking("uname -a && (rocm-smi -i 2>/dev/null || amd-smi list 2>/dev/null || echo 'no ROCm GPU tool')").await?;
    s.disconnect().await;
    Ok(format!("{}\n{}", r.stdout.trim(), r.stderr.trim()))
}

#[tauri::command]
async fn nvidia_smi(cfg: SshConfig) -> Result<GpuState> {
    if cfg.host.is_empty() {
        return Ok(GpuState::simulated());
    }
    let s = SshSession::connect(&cfg).await?;
    let st = ssh::nvidia_smi(&s).await?;
    s.disconnect().await;
    Ok(st)
}

#[tauri::command]
async fn ssh_exec_stream(
    state: State<'_, AppState>,
    app: AppHandle,
    cfg: SshConfig,
    cmd: String,
    stream_id: Option<String>,
) -> Result<String> {
    let app_cfg = config::load().await?;
    let stream_id = stream_id.unwrap_or_else(|| Uuid::new_v4().to_string());
    let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
    state
        .streams
        .lock()
        .insert(stream_id.clone(), cancel.clone());

    let id = stream_id.clone();
    let app_c = app.clone();
    tokio::spawn(async move {
        let session = match SshSession::connect(&cfg).await {
            Ok(s) => s,
            Err(e) => {
                let _ = app_c.emit(
                    "shell://log",
                    serde_json::json!({
                        "streamId": id,
                        "kind": "error",
                        "line": format!("[CONNECTION ERROR] {}\n", e)
                    }),
                );
                let _ = app_c.emit(
                    "shell://done",
                    serde_json::json!({ "streamId": id, "exitCode": -1 }),
                );
                return;
            }
        };

        let mut container_name = app_cfg.docker.container_name.clone();
        if app_cfg.docker.enabled {
            let _ = app_c.emit(
                "shell://log",
                serde_json::json!({
                    "streamId": id,
                    "kind": "info",
                    "line": format!("[DOCKER] Ensuring container '{}' is running...\n", app_cfg.docker.container_name)
                }),
            );
            match crate::pipeline::ensure_container(&session, &app_cfg.docker).await {
                Ok(name) => container_name = name,
                Err(e) => {
                    let _ = app_c.emit(
                        "shell://log",
                        serde_json::json!({
                            "streamId": id,
                            "kind": "error",
                            "line": format!("[DOCKER ERROR] {}\n", e)
                        }),
                    );
                    let _ = app_c.emit(
                        "shell://done",
                        serde_json::json!({ "streamId": id, "exitCode": -1 }),
                    );
                    return;
                }
            }
        }

        let final_cmd = if app_cfg.docker.enabled && !app_cfg.docker.bypass_terminal {
            crate::pipeline::wrap_docker_cmd(&cmd, &container_name)
        } else {
            cmd.clone()
        };

        let _ = app_c.emit(
            "shell://log",
            serde_json::json!({
                "streamId": id,
                "kind": "info",
                "line": format!("[SSH] Running: {}\n", final_cmd)
            }),
        );

        let (tx, mut rx) = mpsc::unbounded_channel::<StreamChunk>();
        let cancel_c = cancel.clone();

        let id_for_collect = id.clone();
        let app_for_collect = app_c.clone();
        let collector = tokio::spawn(async move {
            while let Some(c) = rx.recv().await {
                match c {
                    StreamChunk::Stdout(s) => {
                        let _ = app_for_collect.emit(
                            "shell://log",
                            serde_json::json!({
                                "streamId": id_for_collect,
                                "kind": "stdout",
                                "line": s
                            }),
                        );
                    }
                    StreamChunk::Stderr(s) => {
                        let _ = app_for_collect.emit(
                            "shell://log",
                            serde_json::json!({
                                "streamId": id_for_collect,
                                "kind": "stderr",
                                "line": s
                            }),
                        );
                    }
                    StreamChunk::Done(code) => {
                        let _ = app_for_collect.emit(
                            "shell://done",
                            serde_json::json!({
                                "streamId": id_for_collect,
                                "exitCode": code
                            }),
                        );
                    }
                }
            }
        });

        tokio::select! {
            _ = session.exec_stream(&final_cmd, tx, None) => {}
            _ = async {
                loop {
                    if cancel_c.load(std::sync::atomic::Ordering::SeqCst) {
                        break;
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(250)).await;
                }
            } => {
                let _ = app_c.emit(
                    "shell://log",
                    serde_json::json!({
                        "streamId": id,
                        "kind": "error",
                        "line": "\n[SSH] Connection cancelled by user.\n"
                    }),
                );
                let _ = app_c.emit(
                    "shell://done",
                    serde_json::json!({
                        "streamId": id,
                        "exitCode": -1
                    }),
                );
            }
        }
        let _ = collector.await;
        session.disconnect().await;
    });

    Ok(stream_id)
}

#[tauri::command]
fn ssh_stop_stream(state: State<'_, AppState>, stream_id: String) -> Result<()> {
    if let Some(flag) = state.streams.lock().get(&stream_id) {
        flag.store(true, std::sync::atomic::Ordering::SeqCst);
    }
    Ok(())
}

#[tauri::command]
async fn write_remote_file(cfg: SshConfig, file_path: String, content: String) -> Result<()> {
    let s = SshSession::connect(&cfg).await?;
    s.write_file(&file_path, &content).await?;
    s.disconnect().await;
    Ok(())
}

// ── qdrant commands ────────────────────────────────────────────────────────

#[tauri::command]
async fn qdrant_count(cfg: QdrantConfig) -> Result<u64> {
    qdrant::count(&cfg).await
}

#[tauri::command]
async fn qdrant_sample(cfg: QdrantConfig, n: u32) -> Result<Vec<Chunk>> {
    qdrant::sample(&cfg, n).await
}

#[tauri::command]
async fn qdrant_ensure_collection(cfg: QdrantConfig) -> Result<()> {
    qdrant::ensure_collection(&cfg).await
}

#[tauri::command]
async fn qdrant_sample_in_collection(
    cfg: QdrantConfig,
    collection: String,
    n: u32,
) -> Result<Vec<Chunk>> {
    qdrant::sample_in_collection(&cfg, &collection, n).await
}

#[tauri::command]
async fn qdrant_scroll_in_collection(
    cfg: QdrantConfig,
    collection: String,
    page_size: u32,
    offset: Option<serde_json::Value>,
) -> Result<qdrant::ScrollPage> {
    qdrant::scroll_in_collection(&cfg, &collection, page_size, offset, None).await
}

#[tauri::command]
async fn qdrant_scroll_all(cfg: QdrantConfig, n: u32) -> Result<Vec<Chunk>> {
    qdrant::scroll_all(&cfg, n).await
}

#[tauri::command]
async fn qdrant_scroll_all_in_collection(
    cfg: QdrantConfig,
    collection: String,
    n: u32,
) -> Result<Vec<Chunk>> {
    qdrant::scroll_all_in_collection(&cfg, &collection, n).await
}

#[tauri::command]
async fn list_qdrant_collections(cfg: QdrantConfig) -> Result<Vec<qdrant::CollectionInfo>> {
    qdrant::list_collections(&cfg).await
}

#[tauri::command]
async fn list_qdrant_snapshots(
    cfg: QdrantConfig,
    collection: String,
) -> Result<Vec<qdrant::SnapshotInfo>> {
    qdrant::list_snapshots(&cfg, &collection).await
}

#[tauri::command]
async fn create_qdrant_snapshot(
    cfg: QdrantConfig,
    collection: String,
) -> Result<qdrant::SnapshotInfo> {
    qdrant::create_snapshot(&cfg, &collection).await
}

#[tauri::command]
async fn restore_qdrant_snapshot(
    cfg: QdrantConfig,
    collection: String,
    snapshot_path: String,
) -> Result<()> {
    qdrant::restore_snapshot(&cfg, &collection, &snapshot_path).await
}

#[tauri::command]
async fn qdrant_upload_snapshot(
    cfg: QdrantConfig,
    collection: String,
    snapshot_path: String,
) -> Result<()> {
    qdrant::upload_snapshot(&cfg, &collection, std::path::Path::new(&snapshot_path)).await
}

#[tauri::command]
async fn qdrant_download_snapshot(
    cfg: QdrantConfig,
    collection: String,
    snapshot_name: String,
    local_path: String,
) -> Result<()> {
    qdrant::download_snapshot(
        &cfg,
        &collection,
        &snapshot_name,
        std::path::Path::new(&local_path),
    )
    .await?;
    Ok(())
}

#[tauri::command]
async fn create_all_qdrant_snapshots(
    cfg: QdrantConfig,
) -> Result<Vec<qdrant::CollectionSnapshotResult>> {
    qdrant::create_all_snapshots(&cfg).await
}

#[tauri::command]
async fn download_all_qdrant_snapshots(
    cfg: QdrantConfig,
    local_dir: String,
) -> Result<Vec<String>> {
    let paths = qdrant::download_all_snapshots(&cfg, std::path::Path::new(&local_dir)).await?;
    Ok(paths
        .into_iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect())
}

#[tauri::command]
async fn serve_ensure_qdrant(
    ssh: SshConfig,
    docker: DockerConfig,
    qdrant_port: u16,
    data_dir: String,
) -> Result<()> {
    let session = SshSession::connect(&ssh).await?;
    let result = serve::ensure_qdrant(&session, &docker, qdrant_port, &data_dir).await;
    session.disconnect().await;
    result
}

#[tauri::command]
async fn serve_boot_embedder(
    app: AppHandle,
    ssh: SshConfig,
    docker: DockerConfig,
    embedder: EmbedderConfig,
    hf_token: Option<String>,
) -> Result<String> {
    let session = SshSession::connect(&ssh).await?;
    let app_c = app.clone();
    let host = serve::boot_embedder(
        &session,
        &docker,
        &embedder,
        hf_token.as_deref(),
        embedder.gpu_memory_utilization,
        Some(&move |line| {
            let _ = app_c.emit("setup://log", serde_json::json!({ "line": line }));
        }),
    )
    .await?;
    session.disconnect().await;
    Ok(host)
}

#[tauri::command]
async fn serve_check_embedder(
    ssh: SshConfig,
    docker: DockerConfig,
    host: String,
    port: u16,
) -> Result<Option<String>> {
    let session = SshSession::connect(&ssh).await?;
    serve::health_check_embedder(&session, &docker, &host, port).await
}

#[tauri::command]
async fn serve_boot_paddleocr(
    app: AppHandle,
    ssh: SshConfig,
    docker: DockerConfig,
    paddle_ocr: config::PaddleOcrConfig,
) -> Result<String> {
    if !paddle_ocr.enabled {
        return Err(AppError::pipeline("PaddleOCR is not enabled in config"));
    }
    let session = SshSession::connect(&ssh).await?;
    let app_c = app.clone();
    let host = serve::boot_paddleocr(
        &session,
        &docker,
        &paddle_ocr,
        Some(&move |line| {
            let _ = app_c.emit("setup://log", serde_json::json!({ "line": line }));
        }),
    )
    .await?;
    session.disconnect().await;
    Ok(host)
}

async fn connect_ssh_with_setup_logs(app: &AppHandle, ssh: &SshConfig) -> Result<SshSession> {
    const MAX_ATTEMPTS: usize = 5;
    let mut last_err: Option<AppError> = None;
    for attempt in 1..=MAX_ATTEMPTS {
        let _ = app.emit(
            "setup://log",
            serde_json::json!({
                "line": format!("[stage] connecting to GPU server (attempt {attempt}/{MAX_ATTEMPTS})\n")
            }),
        );
        match SshSession::connect(ssh).await {
            Ok(session) => {
                let _ = app.emit(
                    "setup://log",
                    serde_json::json!({"line": "[ok] SSH connected\n"}),
                );
                return Ok(session);
            }
            Err(e) => {
                let msg = e.to_string();
                if attempt == MAX_ATTEMPTS {
                    let _ = app.emit(
                        "setup://log",
                        serde_json::json!({
                            "line": format!(
                                "[error] SSH connection failed after {MAX_ATTEMPTS} attempts: {msg}\n[fix] verify the droplet is powered on, IP address is correct, port 22 is open, and the firewall/security group allows SSH from this machine.\n"
                            )
                        }),
                    );
                    return Err(e);
                }
                let wait_secs = 3u64 * attempt as u64;
                let _ = app.emit(
                    "setup://log",
                    serde_json::json!({
                        "line": format!("[warn] SSH attempt {attempt}/{MAX_ATTEMPTS} failed: {msg}; retrying in {wait_secs}s\n")
                    }),
                );
                last_err = Some(e);
                tokio::time::sleep(std::time::Duration::from_secs(wait_secs)).await;
            }
        }
    }
    Err(last_err.unwrap_or_else(|| AppError::ssh("SSH connection failed after retries")))
}

#[tauri::command]
async fn serve_setup_all_embedders(
    app: AppHandle,
    ssh: SshConfig,
    docker: DockerConfig,
    mut embedders: Vec<EmbedderConfig>,
    hf_token: Option<String>,
    paddle_ocr: Option<config::PaddleOcrConfig>,
) -> Result<Vec<serde_json::Value>> {
    let session = connect_ssh_with_setup_logs(&app, &ssh).await?;
    config::normalize_embedders(&mut embedders);

    let mut resolved_docker = docker.clone();
    if docker.enabled {
        let _ = app.emit("setup://log", serde_json::json!({"line": format!("[stage] ensuring container '{}' is running...\n", docker.container_name)}));
        match crate::pipeline::ensure_container(&session, &docker).await {
            Ok(name) => {
                if name != docker.container_name {
                    let _ = app.emit("setup://log", serde_json::json!({"line": format!("[ok] using compatible running container: '{}'\n", name)}));
                } else {
                    let _ = app.emit("setup://log", serde_json::json!({"line": format!("[ok] container '{}' is ready\n", name)}));
                }
                resolved_docker.container_name = name;
            }
            Err(e) => {
                let _ = app.emit("setup://log", serde_json::json!({"line": format!("[warn] container setup check failed: {e}\n")}));
            }
        }
    }

    let _ = app.emit(
        "setup://log",
        serde_json::json!({"line": "[stage] ensuring Qdrant is running\n"}),
    );
    match serve::ensure_qdrant(&session, &resolved_docker, 6333, "/root").await {
        Ok(()) => {
            let _ = app.emit(
                "setup://log",
                serde_json::json!({"line": "[ok] Qdrant ready\n"}),
            );
        }
        Err(e) => {
            let _ = app.emit(
                "setup://log",
                serde_json::json!({"line": format!("[error] Qdrant setup failed: {e}\n")}),
            );
            return Err(e);
        }
    }

    if let Some(ref pocr) = paddle_ocr.filter(|p| p.enabled) {
        let _ = app.emit(
            "setup://log",
            serde_json::json!({"line": "[stage] checking PaddleOCR-VL\n"}),
        );
        match serve::health_check_paddleocr(&session, &resolved_docker, pocr.port).await {
            Ok(true) => {
                let _ = app.emit(
                    "setup://log",
                    serde_json::json!({"line": "[ok] PaddleOCR-VL already running\n"}),
                );
            }
            _ => {
                let _ = app.emit(
                    "setup://log",
                    serde_json::json!({"line": "[stage] booting PaddleOCR-VL\n"}),
                );
                let app_c = app.clone();
                match serve::boot_paddleocr(
                    &session,
                    &resolved_docker,
                    pocr,
                    Some(&move |line| {
                        let _ = app_c.emit("setup://log", serde_json::json!({ "line": line }));
                    }),
                )
                .await
                {
                    Ok(_) => {
                        let _ = app.emit(
                            "setup://log",
                            serde_json::json!({"line": "[ok] PaddleOCR-VL ready\n"}),
                        );
                    }
                    Err(e) => {
                        let _ = app.emit("setup://log", serde_json::json!({"line": format!("[warn] PaddleOCR boot failed (non-fatal): {e}\n")}));
                    }
                }
            }
        }
    }

    let mut results = vec![];
    for embedder in embedders.iter() {
        let embedder_cfg = resolved_docker.clone();
        let _ = app.emit("setup://log", serde_json::json!({"line": format!("[stage] checking embedder '{}' on port {} (persistent: {})\n", embedder.name, embedder.port, embedder.persistent)}));
        let collection_name = embedder.effective_collection();
        let qdrant_cfg = QdrantConfig {
            endpoint: format!("http://{}:6333", ssh.host),
            api_key: String::new(),
            collection: collection_name.clone(),
        };

        let has_points = match qdrant::count_in_collection(&qdrant_cfg, &collection_name).await {
            Ok(c) if c > 0 => Some(c),
            _ => None,
        };

        if let Some(c) = has_points {
            let _ = app.emit("setup://log", serde_json::json!({"line": format!("[ok] embeddings already exist ({} points) in collection '{}' for '{}'; ensuring model is live for semantic search.\n", c, collection_name, embedder.name)}));
        }

        let status = match serve::health_check_embedder(&session, &embedder_cfg, "127.0.0.1", embedder.port)
            .await
        {
            Ok(Some(_)) => {
                let _ = app.emit("setup://log", serde_json::json!({"line": format!("[ok] '{}' already running\n", embedder.name)}));
                "already_running".to_string()
            }
            _ => {
                let _ = app.emit("setup://log", serde_json::json!({"line": format!("[stage] booting '{}' with {}\n", embedder.name, embedder.model_id)}));
                let app_c = app.clone();
                match serve::boot_embedder(
                    &session,
                    &embedder_cfg,
                    embedder,
                    hf_token.as_deref(),
                    embedder.gpu_memory_utilization,
                    Some(&move |line| {
                        let _ = app_c.emit("setup://log", serde_json::json!({ "line": line }));
                    }),
                )
                .await
                {
                    Ok(_) => {
                        let _ = app.emit("setup://log", serde_json::json!({"line": format!("[ok] '{}' ready on port {}\n", embedder.name, embedder.port)}));
                        "booted".to_string()
                    }
                    Err(e) => {
                        let _ = app.emit("setup://log", serde_json::json!({"line": format!("[error] '{}' boot failed: {e}\n", embedder.name)}));
                        format!("error: {}", e)
                    }
                }
            }
        };
        results.push(serde_json::json!({
            "name": embedder.name,
            "model_id": embedder.model_id,
            "port": embedder.port,
            "status": status,
        }));
    }
    let _ = app.emit(
        "setup://log",
        serde_json::json!({"line": "[done] setup complete\n"}),
    );
    session.disconnect().await;
    Ok(results)
}

// ── knowledge-base ingestion ───────────────────────────────────────────────

/// Ingest one or more local files into Qdrant. Reads each file (PDF/TXT/MD),
/// chunks it, embeds each chunk via configurable provider (Featherless/Ollama/Llama.cpp),
/// and upserts to the configured collection. Streams `ingest://progress` events;
/// emits `ingest://done` once. Returns the stream id the UI uses to correlate events and cancel.
#[tauri::command]
async fn ingest_documents(
    state: State<'_, AppState>,
    app: AppHandle,
    files: Vec<String>,
    tag: Option<String>,
    vector_dim: Option<usize>,
    qdrant: QdrantConfig,
    embedding_config: ingest::EmbeddingConfig,
    paddle_ocr: Option<config::PaddleOcrConfig>,
) -> Result<String> {
    if files.is_empty() {
        return Err(AppError::pipeline("ingest: no files selected"));
    }
    let stream_id = format!("ingest-{}", Uuid::new_v4());
    let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
    state
        .streams
        .lock()
        .insert(stream_id.clone(), cancel.clone());

    let id_c = stream_id.clone();
    let app_c = app.clone();
    let app_for_progress = app.clone();
    let id_for_progress = stream_id.clone();

    let mut app_cfg = config::load().await.unwrap_or_default();
    config::normalize_runtime_defaults(&mut app_cfg);
    let ocr_opts = match paddle_ocr {
        Some(ref p) => ingest::PaddleOcrOptions {
            enabled: true,
            host: app_cfg.ssh.host.clone(),
            port: p.port,
            model_name: p.model_name.clone(),
        },
        _ => ingest::PaddleOcrOptions {
            enabled: app_cfg.paddle_ocr.enabled || app_cfg.paddle_ocr.port != 0,
            host: app_cfg.ssh.host.clone(),
            port: app_cfg.paddle_ocr.port,
            model_name: app_cfg.paddle_ocr.model_name.clone(),
        },
    };
    let ssh_cfg = app_cfg.ssh.clone();
    let need_ssh = ocr_opts.enabled && !ocr_opts.host.is_empty();

    tokio::spawn(async move {
        let ssh_session = if need_ssh {
            let mgr = SshSessionManager::new(ssh_cfg);
            // Verify initial connectivity
            if let Err(e) = mgr.get_session().await {
                let _ = app_c.emit(
                    "ingest://done",
                    serde_json::json!({
                        "streamId": id_c,
                        "success": false,
                        "error": format!("SSH connect for PaddleOCR: {}", e),
                    }),
                );
                return;
            }
            Some(Arc::new(mgr))
        } else {
            None
        };

        let on_progress: ingest::ProgressFn = Box::new(move |stage, file, done, total| {
            let _ = app_for_progress.emit(
                "ingest://progress",
                serde_json::json!({
                    "streamId": id_for_progress,
                    "stage": stage,
                    "file": file,
                    "done": done,
                    "total": total,
                }),
            );
        });
        let opts = ingest::IngestOptions {
            vector_dim,
            tag,
            chunk_size: None,
            chunk_overlap: None,
        };
        let res = ingest::ingest_files(
            files,
            qdrant,
            embedding_config,
            opts,
            cancel,
            on_progress,
            ocr_opts,
            ssh_session,
        )
        .await;
        match res {
            Ok(summary) => {
                let _ = app_c.emit(
                    "ingest://done",
                    serde_json::json!({
                        "streamId": id_c,
                        "success": true,
                        "summary": summary,
                    }),
                );
            }
            Err(e) => {
                let _ = app_c.emit(
                    "ingest://done",
                    serde_json::json!({
                        "streamId": id_c,
                        "success": false,
                        "error": e.to_string(),
                    }),
                );
            }
        }
    });

    Ok(stream_id)
}

/// Signal an in-flight ingest run to stop. Re-uses the same cancel registry
/// as `ssh_exec_stream` / `deploy_teacher` so cancellation is uniform.
#[tauri::command]
async fn cancel_ingest(state: State<'_, AppState>, stream_id: String) -> Result<()> {
    if let Some(c) = state.streams.lock().get(&stream_id) {
        c.store(true, std::sync::atomic::Ordering::SeqCst);
    }
    Ok(())
}

// ── runs / pipeline commands ───────────────────────────────────────────────

#[tauri::command]
async fn start_pipeline(
    state: State<'_, AppState>,
    app: AppHandle,
    run_cfg: RunConfig,
) -> Result<String> {
    let cfg = config::load().await?;
    pipeline::start(app, state.pipeline.clone(), cfg, run_cfg).await
}

#[tauri::command]
async fn cancel_run(state: State<'_, AppState>, run_id: String) -> Result<()> {
    state.pipeline.cancel(&run_id);
    Ok(())
}

#[tauri::command]
async fn resume_run(state: State<'_, AppState>, app: AppHandle, run_id: String) -> Result<String> {
    let cfg = config::load().await?;
    pipeline::resume(app, state.pipeline.clone(), cfg, run_id).await
}

#[tauri::command]
async fn update_run_config(
    run_id: String,
    student_model: String,
    lora: runs::LoraConfig,
    hub: runs::HubConfig,
) -> Result<()> {
    let mut run = runs::load(&run_id).await?;
    run.student_model = student_model;
    run.lora = lora;
    run.hub = hub;
    runs::save(&run).await?;
    Ok(())
}

#[tauri::command]
fn match_model_guide(student_model: String) -> Option<guides::MatchedGuideInfo> {
    guides::match_guide(&student_model).map(Into::into)
}

#[tauri::command]
async fn list_runs() -> Result<Vec<Run>> {
    let mut list = runs::list().await?;
    list.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    Ok(list)
}

#[tauri::command]
async fn get_run(run_id: String) -> Result<Run> {
    let run = runs::load(&run_id).await?;
    Ok(run)
}

#[tauri::command]
async fn list_local_dataset(run_id: String, limit: usize) -> Result<Vec<serde_json::Value>> {
    let run = runs::load(&run_id).await?;
    let path = std::path::Path::new(&run.local_dir).join("qa_dataset.jsonl");
    let mut out = vec![];
    if !path.exists() {
        return Ok(out);
    }
    let txt = tokio::fs::read_to_string(path).await?;
    for line in txt.lines().take(limit.max(1)) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            out.push(v);
        }
    }
    Ok(out)
}

#[tauri::command]
async fn list_local_dataset_page(
    run_id: String,
    offset: usize,
    limit: usize,
) -> Result<serde_json::Value> {
    let run = runs::load(&run_id).await?;
    let path = std::path::Path::new(&run.local_dir).join("qa_dataset.jsonl");
    if !path.exists() {
        return Ok(serde_json::json!({ "rows": [], "total": 0usize }));
    }
    let txt = tokio::fs::read_to_string(path).await?;
    let lines: Vec<&str> = txt.lines().filter(|l| !l.trim().is_empty()).collect();
    let total = lines.len();
    let rows: Vec<serde_json::Value> = lines
        .into_iter()
        .skip(offset)
        .take(limit.max(1))
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .collect();
    Ok(serde_json::json!({ "rows": rows, "total": total }))
}

#[tauri::command]
async fn open_runs_folder() -> Result<String> {
    let path = config::runs_dir()?;
    Ok(path.to_string_lossy().into_owned())
}

#[tauri::command]
async fn read_run_log(run_id: String, max_bytes: Option<usize>) -> Result<String> {
    runs::read_log_tail(&run_id, max_bytes.unwrap_or(256 * 1024)).await
}

// ── Hugging Face commands ─────────────────────────────────────────────────
//
// These are read-only helpers backing the wizard's "Dataset Repo ID auto-fill"
// and "Resume from <dropdown>" features. Both consult the saved hfToken; they
// fail with a clear error if no token is stored.

#[tauri::command]
async fn hf_whoami() -> Result<hf::HfWhoami> {
    let cfg = config::load().await?;
    let tok = cfg
        .hf_token
        .as_ref()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| AppError::other("no Hugging Face token configured"))?;
    hf::whoami(tok).await
}

#[tauri::command]
async fn hf_list_datasets() -> Result<Vec<hf::HfDatasetRepo>> {
    let cfg = config::load().await?;
    let tok_opt = cfg.hf_token.as_ref().filter(|s| !s.is_empty()).cloned();
    let token = match tok_opt {
        Some(t) => t,
        None => return Err(AppError::other("no Hugging Face token configured")),
    };
    // We need the username to scope the listing — fall back to an empty list
    // if whoami fails so the dropdown can still render (with a hint).
    let me = match hf::whoami(&token).await {
        Ok(w) => w,
        Err(e) => return Err(e),
    };
    hf::list_user_datasets(&token, &me.name).await
}

#[tauri::command]
async fn hf_list_models() -> Result<Vec<hf::HfModelRepo>> {
    let cfg = config::load().await?;
    let tok_opt = cfg.hf_token.as_ref().filter(|s| !s.is_empty()).cloned();
    let token = match tok_opt {
        Some(t) => t,
        None => return Err(AppError::other("no Hugging Face token configured")),
    };
    let me = match hf::whoami(&token).await {
        Ok(w) => w,
        Err(e) => return Err(e),
    };
    hf::list_user_models(&token, &me.name).await
}

#[derive(serde::Serialize)]
pub struct DatasetValidationResult {
    pub repo_id: String,
    pub valid: bool,
    pub sample_count: Option<usize>,
    pub format: Option<String>,
    pub columns: Vec<String>,
    pub error: Option<String>,
}

#[tauri::command]
async fn hf_validate_dataset(repo_id: String) -> Result<DatasetValidationResult> {
    let cfg = config::load().await?;
    let token = match cfg.hf_token.as_ref().filter(|s| !s.is_empty()) {
        Some(t) => t,
        None => return Err(AppError::other("no Hugging Face token configured")),
    };

    match hf::get_dataset_info(token, &repo_id).await {
        Ok(info) => {
            let columns: Vec<String> = info.splits.keys().cloned().collect();
            let sample_count = info.splits.values().map(|s| s.num_examples).sum::<usize>();
            Ok(DatasetValidationResult {
                repo_id: repo_id.clone(),
                valid: true,
                sample_count: Some(sample_count),
                format: None,
                columns,
                error: None,
            })
        }
        Err(e) => Ok(DatasetValidationResult {
            repo_id,
            valid: false,
            sample_count: None,
            format: None,
            columns: vec![],
            error: Some(e.to_string()),
        }),
    }
}

#[tauri::command]
async fn ping_teacher(endpoint: String) -> Result<bool> {
    let url = format!("{}/v1/models", endpoint.trim_end_matches('/'));
    let c = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .map_err(AppError::Http)?;
    Ok(c.get(url)
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false))
}

#[derive(Debug, Clone, Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct TeacherDeploymentStatus {
    port: u16,
    model_id: String,
    exact: bool,
}

#[tauri::command]
async fn check_teacher_deployed(
    ssh: SshConfig,
    docker: DockerConfig,
    teacher: TeacherConfig,
) -> Result<Option<TeacherDeploymentStatus>> {
    if ssh.host.is_empty() {
        return Ok(None);
    }
    let session = match SshSession::connect(&ssh).await {
        Ok(s) => s,
        Err(_) => return Ok(None),
    };

    let docker_cfg = docker.clone();

    let (model_to_check, port_to_check) =
        if let Some(ref cmd) = teacher.custom_serve_cmd.as_ref().filter(|s| !s.is_empty()) {
            pipeline::extract_model_and_port(cmd, &teacher.repo_id, teacher.vllm_port)
        } else {
            (teacher.repo_id.clone(), teacher.vllm_port)
        };

    // Scan all listening ports for a running vLLM teacher, same approach as pipeline.rs.
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
if exact: print('STATUS::' + json.dumps({{'port': exact[0], 'modelId': exact[1], 'exact': True}}))
elif any_m: print('STATUS::' + json.dumps({{'port': any_m[0], 'modelId': any_m[1], 'exact': False}}))
else: print('NOT_FOUND')\
\" 2>/dev/null || echo 'ERROR'",
        model_to_check.replace("\"", "\\\"").replace("'", "\\'"),
        port_to_check
    );

    let check_probe = if docker_cfg.enabled {
        let mut is_running = false;
        let mut resolved_name = docker_cfg.container_name.clone();
        if let Ok(ps_r) = session
            .exec_blocking("docker ps --format '{{.Names}}\t{{.Image}}'")
            .await
        {
            let running: Vec<(String, String)> = ps_r
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
            if running_names.contains(&docker_cfg.container_name) {
                is_running = true;
            } else {
                let cfg_img_lower = docker_cfg.image_name.to_lowercase();
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
                if let Some((name, _)) = candidate {
                    resolved_name = name.clone();
                    is_running = true;
                }
            }
        }
        if !is_running {
            session.disconnect().await;
            return Ok(None);
        }
        pipeline::wrap_docker_cmd(&check_script, &resolved_name)
    } else {
        check_script
    };

    let mut found_status = None;
    if let Ok(probe_r) = session.exec_blocking(&check_probe).await {
        for line in probe_r.stdout.lines().map(str::trim) {
            if let Some(json_text) = line.strip_prefix("STATUS::") {
                if let Ok(status) = serde_json::from_str::<TeacherDeploymentStatus>(json_text) {
                    found_status = Some(status);
                    break;
                }
            }
        }
    }

    session.disconnect().await;
    Ok(found_status)
}

async fn run_deploy_teacher_task(
    app: &AppHandle,
    id: &str,
    ssh: &SshConfig,
    docker: &DockerConfig,
    teacher: &TeacherConfig,
    hf_token: Option<&str>,
    cancel: &Arc<std::sync::atomic::AtomicBool>,
) -> Result<u16> {
    let mut session = SshSession::connect(ssh).await?;

    // --- GPU VRAM Cleanup ---
    // Stop paddleocr-vl container to free VRAM. Do NOT stop rocm-vllm — instead
    // kill only non-persistent embedder processes by port (see below).
    // This keeps the container warm so the teacher can reuse it without a full
    // re-pull/launch. Protected embedders are skipped by port, even when they
    // run inside rocm-vllm.
    let _ = app.emit("deploy://log", serde_json::json!({
        "streamId": id,
        "kind": "info",
        "line": "[GPU CLEANUP] stopping PaddleOCR (paddleocr-vl); preserving rocm-vllm container, will kill non-persistent embedders inside...\n"
    }));
    let cleanup_cmd =
        "docker stop paddleocr-vl 2>/dev/null; docker rm paddleocr-vl 2>/dev/null; true";
    let ocr_cleanup = session.exec_blocking(cleanup_cmd).await;
    if let Err(e) = ocr_cleanup {
        let _ = app.emit(
            "deploy://log",
            serde_json::json!({
                "streamId": id,
                "kind": "info",
                "line": format!("[GPU CLEANUP] PaddleOCR stop info: {}\n", e)
            }),
        );
    }

    let mut app_cfg = config::load().await.unwrap_or_default();
    config::normalize_runtime_defaults(&mut app_cfg);
    let has_protected_embedder = app_cfg
        .embedders
        .iter()
        .enumerate()
        .any(|(idx, e)| pipeline::is_protected_semantic_embedder(idx, e));
    if !app_cfg.embedders.is_empty() {
        let killable: Vec<_> = app_cfg
            .embedders
            .iter()
            .enumerate()
            .filter(|(idx, e)| !pipeline::is_protected_semantic_embedder(*idx, e))
            .map(|(_, e)| e)
            .collect();
        let persistent: Vec<_> = app_cfg
            .embedders
            .iter()
            .enumerate()
            .filter(|(idx, e)| pipeline::is_protected_semantic_embedder(*idx, e))
            .map(|(_, e)| e)
            .collect();
        let killable_names: Vec<String> = killable.iter().map(|e| e.name.clone()).collect();
        let persistent_names: Vec<String> = persistent.iter().map(|e| e.name.clone()).collect();
        let _ = app.emit("deploy://log", serde_json::json!({
            "streamId": id,
            "kind": "info",
            "line": format!(
                "[GPU CLEANUP] stopping {} embedder(s) ({}) and preserving persistent embedder(s) ({})\n",
                killable_names.len(),
                killable_names.join(", "),
                if persistent_names.is_empty() { "none".to_string() } else { persistent_names.join(", ") }
            )
        }));

        let killable_ports: Vec<u16> = killable.iter().map(|e| e.port).collect();
        // Port-targeted only — broad vLLM pattern sweeps would also match a
        // protected embedder, whether it is running on the host or in Docker.
        let host_kill = pipeline::build_embedder_port_kill_cmd(&killable_ports);
        let _ = session.exec_blocking(&host_kill).await;
        let inner_kill = pipeline::build_embedder_port_kill_cmd(&killable_ports);
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
                let inner = pipeline::wrap_docker_cmd(&inner_kill, &cname);
                let _ = session.exec_blocking(&inner).await;
            }
        }

        let _ = app.emit(
            "deploy://log",
            serde_json::json!({
                "streamId": id,
                "kind": "info",
                "line": "[GPU CLEANUP] embedders stopped\n"
            }),
        );
    }

    let gpu_state = ssh::nvidia_smi(&session).await.ok();
    let gpu_memory_total_mb = gpu_state.as_ref().map(|gpu| gpu.memory_total);
    let mut teacher = teacher.resolved_for_gpu(gpu_memory_total_mb);
    if has_protected_embedder {
        if let Some((old, new)) = pipeline::cap_teacher_vram_for_semantic_embedder(&mut teacher) {
            let _ = app.emit("deploy://log", serde_json::json!({
                "streamId": id,
                "kind": "info",
                "line": format!(
                    "[vram] protected embedder_1 is reserved for semantic search; capping teacher VRAM util from {:.2} to {:.2}\n",
                    old, new
                )
            }));
        }
    }
    let docker_cfg = docker.clone();
    let mut container_name = docker_cfg.container_name.clone();

    if docker_cfg.enabled {
        let _ = app.emit("deploy://log", serde_json::json!({
            "streamId": id,
            "kind": "info",
            "line": format!("[DOCKER] ensuring container '{}' is running\n", docker_cfg.container_name)
        }));
        match pipeline::ensure_container(&session, &docker_cfg).await {
            Ok(name) => container_name = name,
            Err(e) => {
                let _ = app.emit(
                    "deploy://log",
                    serde_json::json!({
                        "streamId": id,
                        "kind": "error",
                        "line": format!("[DOCKER ERROR] {}\n", e)
                    }),
                );
                return Err(e);
            }
        }
    }

    if teacher
        .custom_serve_cmd
        .as_ref()
        .map(|cmd| cmd.trim().is_empty())
        .unwrap_or(true)
    {
        if let Some((limit, source)) = pipeline::probe_model_context_limit(
            &session,
            docker_cfg.enabled,
            &container_name,
            &teacher.repo_id,
            hf_token,
        )
        .await
        {
            if teacher.max_model_len > limit {
                let old = teacher.max_model_len;
                teacher.max_model_len = limit;
                let _ = app.emit("deploy://log", serde_json::json!({
                    "streamId": id,
                    "kind": "info",
                    "line": format!(
                        "[context] model config reports {}={}; capping --max-model-len from {} to {}\n",
                        source, limit, old, limit
                    )
                }));
                if let Ok(mut app_cfg) = config::load().await {
                    if app_cfg.teacher.repo_id.trim() == teacher.repo_id.trim() {
                        app_cfg.teacher.max_model_len = limit;
                        let _ = config::save(&app_cfg).await;
                    }
                }
            }
        }
    }

    let (_model_to_check, _port_to_check) =
        if let Some(ref cmd) = teacher.custom_serve_cmd.as_ref().filter(|s| !s.is_empty()) {
            pipeline::extract_model_and_port(cmd, &teacher.repo_id, teacher.vllm_port)
        } else {
            (teacher.repo_id.clone(), teacher.vllm_port)
        };

    let teacher_log = "/root/fine-tune/runs/teacher_deploy.log".to_string();

    // Pre-kill any existing vLLM processes AND free the target port. The old
    // `pkill A || pkill B` form short-circuits — when A matched anything, B
    // never ran, leaving orphan workers behind. Use `;` to guarantee every
    // pattern is tried, escalate to -9 after a grace period, and finally
    // `fuser -k` whatever still holds the port (covers crashed parents whose
    // sockets are still bound). Then poll up to 10s for the port to actually
    // become free before declaring success.
    let port_to_free = if let Some(ref cmd) =
        teacher.custom_serve_cmd.as_ref().filter(|s| !s.is_empty())
    {
        let (_m, p) = pipeline::extract_model_and_port(cmd, &teacher.repo_id, teacher.vllm_port);
        p
    } else {
        teacher.vllm_port
    };

    let mut ports_to_free = vec![port_to_free];
    for (idx, emb) in app_cfg.embedders.iter().enumerate() {
        if !pipeline::is_protected_semantic_embedder(idx, emb) {
            ports_to_free.push(emb.port);
        }
    }
    ports_to_free.sort();
    ports_to_free.dedup();

    let mut port_cleanup_cmds = String::new();
    for p in &ports_to_free {
        port_cleanup_cmds.push_str(&format!(
            "(command -v fuser >/dev/null 2>&1 && fuser -k {port}/tcp 2>/dev/null) || true; \
             (command -v ss >/dev/null 2>&1 && ss -ltnp 2>/dev/null | awk '/:{port} /{{print $0}}' | grep -oE 'pid=[0-9]+' | cut -d= -f2 | xargs -r kill -9 2>/dev/null) || true; ",
            port = p
        ));
    }

    let mut port_wait_cmds = String::new();
    for p in &ports_to_free {
        port_wait_cmds.push_str(&format!(
            "for i in 1 2 3 4 5 6 7 8 9 10; do \
                 if command -v ss >/dev/null 2>&1; then \
                     ss -ltn 2>/dev/null | awk '{{print $4}}' | grep -qE ':{port}$' || break; \
                 else \
                     (netstat -ltn 2>/dev/null || true) | awk '{{print $4}}' | grep -qE ':{port}$' || break; \
                 fi; \
                 sleep 1; \
             done; ",
            port = p
        ));
    }

    let pkill_body = if has_protected_embedder {
        let vllm_pattern = pipeline::sh_quote(&format!("vllm serve {}", teacher.repo_id));
        let sglang_pattern =
            pipeline::sh_quote(&format!("sglang.launch_server.*{}", teacher.repo_id));
        format!(
            "pkill -f {vllm_pattern} 2>/dev/null; \
             pkill -f {sglang_pattern} 2>/dev/null; \
             sleep 1; \
             pkill -9 -f {vllm_pattern} 2>/dev/null; \
             pkill -9 -f {sglang_pattern} 2>/dev/null; \
             {port_cleanup} \
             {port_wait} \
             true",
            vllm_pattern = vllm_pattern,
            sglang_pattern = sglang_pattern,
            port_cleanup = port_cleanup_cmds,
            port_wait = port_wait_cmds,
        )
    } else {
        format!(
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
             {port_cleanup} \
             {port_wait} \
             true",
            port_cleanup = port_cleanup_cmds,
            port_wait = port_wait_cmds,
        )
    };
    // First sweep on the host — covers any vLLM started outside docker
    // and any process holding the port directly on the bare metal.
    let _ = session.exec_blocking(&pkill_body).await;
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
                let inner = pipeline::wrap_docker_cmd(&pkill_body, cname);
                let _ = session.exec_blocking(&inner).await;
            }
        }
    }

    // Final targeted sweep in the container we'll actually use.
    let pkill_cmd = if docker_cfg.enabled {
        pipeline::wrap_docker_cmd(&pkill_body, &container_name)
    } else {
        pkill_body.clone()
    };
    let _ = session.exec_blocking(&pkill_cmd).await;
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Final sanity check inside the container: is the port really free?
    // If not, emit a clear error so the user knows to manually clear it
    // instead of getting the cryptic "Address already in use" 5 minutes in.
    let port_check_inner = format!(
        "if command -v ss >/dev/null 2>&1; then \
             ss -ltn 2>/dev/null | awk '{{print $4}}' | grep -qE ':{port}$' && echo PORT_BUSY || echo PORT_FREE; \
         else \
             (netstat -ltn 2>/dev/null || true) | awk '{{print $4}}' | grep -qE ':{port}$' && echo PORT_BUSY || echo PORT_FREE; \
         fi",
        port = port_to_free,
    );
    let _port_check_cmd = if docker_cfg.enabled {
        pipeline::wrap_docker_cmd(&port_check_inner, &container_name)
    } else {
        port_check_inner.clone()
    };
    // ── Always use a free port (avoid race condition where port becomes busy ──
    //    between the check and vLLM's bind() during model loading).
    let find_port_script = "python3 -c \"import socket; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(('', 0)); print(s.getsockname()[1]); s.close()\" 2>/dev/null || echo 0";
    let find_port_cmd = if docker_cfg.enabled {
        pipeline::wrap_docker_cmd(find_port_script, &container_name)
    } else {
        find_port_script.to_string()
    };
    let actual_port: u16 = match session.exec_blocking(&find_port_cmd).await {
        Ok(r) => r.stdout.trim().parse().unwrap_or(port_to_free),
        Err(_) => port_to_free,
    };
    if actual_port != port_to_free {
        let _ = app.emit("deploy://log", serde_json::json!({
            "streamId": id,
            "kind": "info",
            "line": format!("[port] configured port {} is not available, using free port {} instead\n", port_to_free, actual_port)
        }));
        if let Ok(mut app_cfg) = config::load().await {
            app_cfg.teacher.vllm_port = actual_port;
            let _ = config::save(&app_cfg).await;
        }
    }

    // Override port_to_check to use the actual port for boot and polling
    let (_, port_to_check) =
        if let Some(ref cmd) = teacher.custom_serve_cmd.as_ref().filter(|s| !s.is_empty()) {
            pipeline::extract_model_and_port(cmd, &teacher.repo_id, actual_port)
        } else {
            (teacher.repo_id.clone(), actual_port)
        };

    let boot_cmd = if let Some(ref cmd) =
        teacher.custom_serve_cmd.as_ref().filter(|s| !s.is_empty())
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
            if let Some(tok) = hf_token {
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

        let _ = app.emit(
            "deploy://log",
            serde_json::json!({
                "streamId": id,
                "kind": "info",
                "line": "[stage] booting teacher vLLM\n"
            }),
        );
        let _ = app.emit(
            "deploy://log",
            serde_json::json!({
                "streamId": id,
                "kind": "info",
                "line": format!("[cmd] {}\n", display_cmd)
            }),
        );

        if docker_cfg.enabled {
            // Use foreground docker exec with nohup inside so we can detect
            // immediate failures. After 3s we check if vLLM is still alive.
            // This mirrors the user's manual flow: docker exec -it rocm bash
            // then running the serve command from inside.
            let inner_script = format!(
                "mkdir -p /root/hf-cache; \
                 mkdir -p $(dirname {log}); \
                 truncate -s 0 {log} 2>/dev/null || rm -f {log}; \
                 nohup bash -c {serve} </dev/null >>{log} 2>&1 & \
                 BGPID=$!; \
                 sleep 3; \
                 if kill -0 $BGPID 2>/dev/null; then \
                   echo VLLM_STARTED:$BGPID; \
                 else \
                   echo VLLM_FAILED; \
                   tail -n 30 {log} 2>/dev/null; \
                 fi",
                log = teacher_log,
                serve = pipeline::sh_quote(&final_custom_cmd),
            );
            format!(
                "docker exec {cn} bash -c {script}",
                cn = container_name,
                script = pipeline::sh_quote(&inner_script),
            )
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
        // ... (rest of the code for non-custom command)

        let mut tokenizer_arg = String::new();
        let repo_id_lower = teacher.repo_id.to_lowercase();
        if repo_id_lower.contains("gguf") {
            let parts: Vec<&str> = teacher.repo_id.split('/').collect();
            let base_repo = if parts.len() >= 2 {
                format!(
                    "{}/{}",
                    parts[0],
                    parts[1].split(':').next().unwrap_or(parts[1])
                )
            } else {
                teacher
                    .repo_id
                    .split(':')
                    .next()
                    .unwrap_or(&teacher.repo_id)
                    .to_string()
            };
            let base_model = base_repo
                .replace("-GGUF", "")
                .replace("-gguf", "")
                .replace(".GGUF", "")
                .replace(".gguf", "");
            tokenizer_arg = format!("--tokenizer {}", base_model);
        }

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
            if let Some(tok) = hf_token {
                envs.push_str(&format!(
                    "export HF_TOKEN={} HUGGING_FACE_HUB_TOKEN={}; ",
                    tok, tok
                ));
            }
            envs
        };
        let extra_args = teacher.vllm_extra_args();
        let runtime_prepare = teacher.vllm_runtime_prepare_cmd();

        let serve_cmd_display = format!(
            "HF_TOKEN=*** vllm serve {} --port {} --host 0.0.0.0 \
             --max-model-len {} --dtype {} \
             --download-dir /root/hf-cache \
             --tensor-parallel-size {} --gpu-memory-utilization {} {} {}\n",
            teacher.repo_id,
            port_to_check,
            teacher.max_model_len,
            teacher.dtype,
            teacher.tensor_parallel,
            teacher.gpu_memory_utilization,
            tokenizer_arg,
            extra_args
        );
        let _ = app.emit(
            "deploy://log",
            serde_json::json!({
                "streamId": id,
                "kind": "info",
                "line": "[stage] booting teacher vLLM\n"
            }),
        );
        let _ = app.emit(
            "deploy://log",
            serde_json::json!({
                "streamId": id,
                "kind": "info",
                "line": format!("[cmd] {}\n", serve_cmd_display)
            }),
        );

        let serve_cmd_inner = format!(
            "cd /app && {env}{runtime_prepare}vllm serve {model} --port {port} --host 0.0.0.0 \
               --max-model-len {mml} --dtype {dtype} --download-dir /root/hf-cache \
               --tensor-parallel-size {tp} --gpu-memory-utilization {gpu_mem} {tok_arg} {extra_args}",
            env = vllm_env,
            runtime_prepare = runtime_prepare,
            model = teacher.repo_id,
            port = port_to_check,
            mml = teacher.max_model_len,
            dtype = teacher.dtype,
            tp = teacher.tensor_parallel,
            gpu_mem = teacher.gpu_memory_utilization,
            tok_arg = tokenizer_arg,
            extra_args = extra_args,
        );

        if docker_cfg.enabled {
            // Foreground docker exec with nohup inside so we can detect
            // immediate failures instead of silently hanging forever.
            let inner_script = format!(
                "mkdir -p /root/hf-cache; \
                 mkdir -p $(dirname {log}); \
                 truncate -s 0 {log} 2>/dev/null || rm -f {log}; \
                 nohup bash -c {serve} </dev/null >>{log} 2>&1 & \
                 BGPID=$!; \
                 sleep 3; \
                 if kill -0 $BGPID 2>/dev/null; then \
                   echo VLLM_STARTED:$BGPID; \
                 else \
                   echo VLLM_FAILED; \
                   tail -n 30 {log} 2>/dev/null; \
                 fi",
                log = teacher_log,
                serve = pipeline::sh_quote(&serve_cmd_inner),
            );
            format!(
                "docker exec {cn} bash -c {script}",
                cn = container_name,
                script = pipeline::sh_quote(&inner_script),
            )
        } else {
            format!(
                "mkdir -p /root/hf-cache; \
                 truncate -s 0 {teacher_log} 2>/dev/null || rm -f {teacher_log}; \
                 nohup bash -lc '{serve} > {log} 2>&1' < /dev/null & \
                 echo TEACHER_LAUNCHED",
                teacher_log = teacher_log,
                serve = serve_cmd_inner,
                log = teacher_log,
            )
        }
    };

    let boot_r = session.exec_blocking(&boot_cmd).await?;

    // For docker mode: the boot script waits 3s then prints VLLM_STARTED or
    // VLLM_FAILED (plus the error log tail). Surface this immediately.
    let boot_stdout = boot_r.stdout.trim().to_string();
    let boot_stderr = boot_r.stderr.trim().to_string();

    if boot_r.exit_code != 0 {
        let err_msg = format!(
            "failed to start teacher (exit {}): {}{}",
            boot_r.exit_code,
            if !boot_stdout.is_empty() {
                format!("\n{}", boot_stdout)
            } else {
                String::new()
            },
            if !boot_stderr.is_empty() {
                format!("\n{}", boot_stderr)
            } else {
                String::new()
            },
        );
        let _ = app.emit(
            "deploy://log",
            serde_json::json!({
                "streamId": id,
                "kind": "error",
                "line": format!("[error] {}\n", err_msg)
            }),
        );
        return Err(AppError::pipeline(err_msg));
    }

    // Emit the boot script's stdout so the user sees VLLM_STARTED/VLLM_FAILED
    // and any error log tail immediately after the 3s startup check.
    if !boot_stdout.is_empty() {
        let failed = boot_stdout.contains("VLLM_FAILED");
        for line in boot_stdout.lines() {
            let kind = if line.contains("VLLM_FAILED")
                || line.contains("Error")
                || line.contains("Traceback")
            {
                "error"
            } else if line.contains("VLLM_STARTED") {
                "ok"
            } else {
                "stdout"
            };
            let _ = app.emit(
                "deploy://log",
                serde_json::json!({
                    "streamId": id,
                    "kind": kind,
                    "line": format!("{}\n", line)
                }),
            );
        }
        if failed {
            return Err(AppError::pipeline(
                "vLLM process exited within 3 seconds of launch — check the log lines above for the error"
            ));
        }
    }
    if !boot_stderr.is_empty() {
        let _ = app.emit(
            "deploy://log",
            serde_json::json!({
                "streamId": id,
                "kind": "error",
                "line": format!("[boot-stderr] {}\n", boot_stderr)
            }),
        );
    }

    // Emit startup: let user know polling has begun
    let _ = app.emit("deploy://log", serde_json::json!({
        "streamId": id,
        "kind": "info",
        "line": format!("[stage] vLLM process launched — polling {} for logs (this can take 5-20 min for large models)...\n", teacher_log)
    }));

    let started = std::time::Instant::now();
    let timeout = std::time::Duration::from_secs(20 * 60);
    let mut log_line_offset: u64 = 1;
    let mut last_heartbeat = std::time::Instant::now();
    let heartbeat_interval = std::time::Duration::from_secs(15);
    let poll_interval_secs = 3u64;

    loop {
        if cancel.load(std::sync::atomic::Ordering::SeqCst) {
            let _ = app.emit(
                "deploy://log",
                serde_json::json!({
                    "streamId": id,
                    "kind": "error",
                    "line": "Deployment cancelled by user\n"
                }),
            );
            return Err(AppError::Cancelled);
        }
        if started.elapsed() > timeout {
            let err_msg = "teacher boot timeout (20 min)";
            let _ = app.emit(
                "deploy://log",
                serde_json::json!({
                    "streamId": id,
                    "kind": "error",
                    "line": format!("{}\n", err_msg)
                }),
            );
            return Err(AppError::pipeline(err_msg));
        }

        // CRITICAL FIX: wrap in bash -c so that pipes, semicolons, echo and
        // shell redirection all work. session.exec_blocking() uses the raw SSH
        // exec channel (not a shell), so without this wrapper the pipe between
        // `docker exec ... tail` and `head -n 500`, the `;` sequencing, and
        // the `echo '---PROBE---'` call are never executed — the delimiter
        // never appears in stdout, parts.len() is always 1, and no log lines
        // are ever emitted.
        let inner_combo = if docker_cfg.enabled {
            format!(
                "docker exec {cn} tail -n +{off} {log} 2>/dev/null | head -n 500; \
                 echo '---PROBE---'; \
                 docker exec {cn} curl -s -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{port}/v1/models 2>/dev/null || echo '000'",
                cn = container_name,
                off = log_line_offset,
                log = teacher_log,
                port = port_to_check,
            )
        } else {
            format!(
                "tail -n +{off} {log} 2>/dev/null | head -n 500; \
                 echo '---PROBE---'; \
                 curl -s -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{port}/v1/models 2>/dev/null || echo '000'",
                off = log_line_offset,
                log = teacher_log,
                port = port_to_check,
            )
        };
        // Wrap in bash -c so pipes and semicolons are interpreted by a shell
        let combo_cmd = format!("bash -c {}", pipeline::sh_quote(&inner_combo));

        let r = loop {
            match session.exec_blocking(&combo_cmd).await {
                Ok(result) => break result,
                Err(e) => {
                    let _ = app.emit("deploy://log", serde_json::json!({
                        "streamId": id,
                        "kind": "warn",
                        "line": format!("[ssh-warn] session disconnected ({}), reconnecting...\n", e)
                    }));
                    let new_session = SshSession::connect(ssh).await?;
                    session = new_session;
                    let _ = app.emit(
                        "deploy://log",
                        serde_json::json!({
                            "streamId": id,
                            "kind": "info",
                            "line": "[ssh] session reconnected, retrying probe...\n"
                        }),
                    );
                }
            }
        };

        let parts: Vec<&str> = r.stdout.split("---PROBE---").collect();
        if parts.len() >= 2 {
            let log_chunk = parts[0];
            let probe_code = parts[1].trim();

            if !log_chunk.is_empty() {
                let lines: Vec<&str> = log_chunk.lines().collect();
                log_line_offset += lines.len() as u64;
                for line in lines {
                    let _ = app.emit(
                        "deploy://log",
                        serde_json::json!({
                            "streamId": id,
                            "kind": "stdout",
                            "line": format!("{}\n", line)
                        }),
                    );
                }
                let lower = log_chunk.to_lowercase();
                let engine_name = "vLLM";
                if lower.contains("traceback")
                    || lower.contains("validationerror")
                    || lower.contains("does not recognize this architecture")
                    || lower.contains("vllm_failed")
                    || lower.contains("out of memory")
                    || lower.contains("hip out of memory")
                    || lower.contains("outofmemoryerror")
                {
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
                    let _ = app.emit(
                        "deploy://log",
                        serde_json::json!({
                            "streamId": id,
                            "kind": "error",
                            "line": format!("[error] {}\n", err_msg)
                        }),
                    );
                    return Err(AppError::pipeline(err_msg));
                }
            }

            if probe_code == "200" {
                let _ = app.emit(
                    "deploy://log",
                    serde_json::json!({
                        "streamId": id,
                        "kind": "ok",
                        "line": format!("[ok] teacher model is serving on port {}\n", port_to_check)
                    }),
                );
                break;
            }

            // No new log lines and not ready yet — emit a heartbeat every 15s
            // so the UI shows the connection is alive during model download.
            if log_chunk.trim().is_empty() && last_heartbeat.elapsed() >= heartbeat_interval {
                last_heartbeat = std::time::Instant::now();
                let elapsed = started.elapsed().as_secs();
                let _ = app.emit("deploy://log", serde_json::json!({
                    "streamId": id,
                    "kind": "info",
                    "line": format!("[waiting] {}s elapsed — vLLM still starting (no output yet, model may be downloading)...\n", elapsed)
                }));
            }
        } else {
            // bash -c itself failed or returned no output — emit stderr for debug
            if !r.stderr.trim().is_empty() {
                let _ = app.emit(
                    "deploy://log",
                    serde_json::json!({
                        "streamId": id,
                        "kind": "error",
                        "line": format!("[poll-err] {}\n", r.stderr.trim())
                    }),
                );
            }
        }

        tokio::time::sleep(std::time::Duration::from_secs(poll_interval_secs)).await;
    }

    if let Some((_embedder_index, embedder)) = app_cfg
        .embedders
        .iter()
        .enumerate()
        .find(|(idx, e)| e.enabled && pipeline::is_protected_semantic_embedder(*idx, e))
    {
        let qdrant_up = session
            .exec_blocking("curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:6333/collections 2>/dev/null || echo 000")
            .await
            .map(|r| r.stdout.trim() == "200")
            .unwrap_or(false);

        if qdrant_up {
            let mut embedder_docker = docker_cfg.clone();
            embedder_docker.container_name = container_name.clone();
            let embedder_cfg = embedder_docker;

            let already_up =
                serve::health_check_embedder(&session, &embedder_cfg, "127.0.0.1", embedder.port)
                    .await
                    .map(|o| o.is_some())
                    .unwrap_or(false);

            if already_up {
                let _ = app.emit("deploy://log", serde_json::json!({
                    "streamId": id,
                    "kind": "ok",
                    "line": format!("[embedder] '{}' already serving on port {} — semantic search ready\n", embedder.name, embedder.port)
                }));
            } else {
                let _ = app.emit("deploy://log", serde_json::json!({
                    "streamId": id,
                    "kind": "info",
                    "line": format!("[embedder] Qdrant is running — starting semantic embedder '{}' ({}) on port {}\n", embedder.name, embedder.model_id, embedder.port)
                }));

                match serve::launch_embedder(
                    &session,
                    &embedder_cfg,
                    embedder,
                    hf_token,
                    Some(embedder.gpu_memory_utilization as f64),
                )
                .await
                {
                    Ok(_) => {
                        let _ = app.emit("deploy://log", serde_json::json!({
                            "streamId": id,
                            "kind": "info",
                            "line": format!("[embedder] waiting up to 600s for '{}' to become ready...\n", embedder.name)
                        }));
                        match tokio::time::timeout(
                            std::time::Duration::from_secs(600),
                            serve::wait_for_embedder(&session, &embedder_cfg, embedder, 600, None),
                        )
                        .await
                        {
                            Ok(Ok(ip)) => {
                                let _ = app.emit("deploy://log", serde_json::json!({
                                    "streamId": id,
                                    "kind": "ok",
                                    "line": format!("[embedder] '{}' ready at {}:{} — semantic search ready\n", embedder.name, ip, embedder.port)
                                }));
                            }
                            Ok(Err(e)) => {
                                let _ = app.emit("deploy://log", serde_json::json!({
                                    "streamId": id,
                                    "kind": "warn",
                                    "line": format!("[embedder] '{}' did not become ready: {}\n", embedder.name, e)
                                }));
                            }
                            Err(_) => {
                                let _ = app.emit("deploy://log", serde_json::json!({
                                    "streamId": id,
                                    "kind": "warn",
                                    "line": format!("[embedder] '{}' wait timed out after 600s\n", embedder.name)
                                }));
                            }
                        }
                    }
                    Err(e) => {
                        let _ = app.emit("deploy://log", serde_json::json!({
                            "streamId": id,
                            "kind": "warn",
                            "line": format!("[embedder] launch failed after teacher deploy: {}\n", e)
                        }));
                    }
                }
            }
        } else {
            let _ = app.emit("deploy://log", serde_json::json!({
                "streamId": id,
                "kind": "warn",
                "line": "[embedder] Qdrant is not running on :6333 — semantic embedder was not started\n"
            }));
        }
    }

    session.disconnect().await;
    Ok(port_to_check)
}

#[tauri::command]
async fn deploy_teacher(
    app: AppHandle,
    state: State<'_, AppState>,
    ssh: SshConfig,
    docker: DockerConfig,
    teacher: TeacherConfig,
    hf_token: Option<String>,
) -> Result<String> {
    let stream_id = format!("deploy-{}", uuid::Uuid::new_v4());
    let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
    state
        .streams
        .lock()
        .insert(stream_id.clone(), cancel.clone());

    let id_c = stream_id.clone();
    let app_c = app.clone();
    tokio::spawn(async move {
        let res = run_deploy_teacher_task(
            &app_c,
            &id_c,
            &ssh,
            &docker,
            &teacher,
            hf_token.as_deref(),
            &cancel,
        )
        .await;
        match res {
            Ok(actual_port) => {
                let _ = app_c.emit(
                    "deploy://done",
                    serde_json::json!({
                        "streamId": id_c,
                        "success": true,
                        "message": "Teacher model deployed successfully!",
                        "port": actual_port
                    }),
                );
            }
            Err(e) => {
                let _ = app_c.emit(
                    "deploy://done",
                    serde_json::json!({
                        "streamId": id_c,
                        "success": false,
                        "message": e.to_string()
                    }),
                );
            }
        }
    });

    Ok(stream_id)
}

#[tauri::command]
async fn teacher_chat(
    endpoint: String,
    model: String,
    messages: Vec<serde_json::Value>,
) -> Result<String> {
    let body = serde_json::json!({
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 4096
    });
    let c = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(AppError::Http)?;
    let res = c
        .post(format!(
            "{}/v1/chat/completions",
            endpoint.trim_end_matches('/')
        ))
        .json(&body)
        .send()
        .await
        .map_err(AppError::Http)?;
    if !res.status().is_success() {
        let s = res.status();
        let t = res.text().await.unwrap_or_default();
        return Err(AppError::pipeline(format!("teacher chat {s}: {t}")));
    }
    let v: serde_json::Value = res.json().await.map_err(AppError::Http)?;
    Ok(v.get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string())
}

#[tauri::command]
async fn test_trained_model(run_id: String, prompt: String) -> Result<String> {
    let cfg = config::load().await?;
    let run = runs::load(&run_id).await?;
    if run.status != runs::RunStatus::Done {
        return Err(AppError::pipeline("model test requires a completed run"));
    }
    if prompt.trim().is_empty() {
        return Err(AppError::pipeline("test prompt is empty"));
    }

    let session = SshSession::connect(&cfg.ssh).await?;
    let mut container_name = cfg.docker.container_name.clone();
    if cfg.docker.enabled {
        container_name = pipeline::ensure_container(&session, &cfg.docker).await?;
    }

    let adapter_path = format!("{}/lora", run.remote_dir);
    let check_inner = format!(
        "test -f {}/adapter_model.safetensors && echo OK || echo MISSING",
        pipeline::sh_quote(&adapter_path)
    );
    let check_cmd = if cfg.docker.enabled {
        pipeline::wrap_docker_cmd(&check_inner, &container_name)
    } else {
        check_inner
    };
    let check = session.exec_blocking(&check_cmd).await?;
    if !check.stdout.contains("OK") {
        return Err(AppError::pipeline(format!(
            "adapter not found at {adapter_path}/adapter_model.safetensors"
        )));
    }

    let script = format!(
        r#"set -e
cd {run_dir}
python3 - <<'PY'
import json
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = {base_model}
adapter_path = {adapter_path}
prompt = {prompt}

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
model = PeftModel.from_pretrained(base, adapter_path)
model.eval()

messages = [
    {{"role": "system", "content": "You are a helpful assistant. Answer using the training domain when relevant."}},
    {{"role": "user", "content": prompt}},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.eos_token_id,
    )
generated = output_ids[0][inputs.input_ids.shape[-1]:]
print(tokenizer.decode(generated, skip_special_tokens=True).strip())
PY"#,
        run_dir = pipeline::sh_quote(&run.remote_dir),
        base_model = serde_json::to_string(&crate::llamafactory::resolve_trainable_repo(
            &run.student_model
        ))
        .unwrap_or_else(|_| "\"\"".to_string()),
        adapter_path = serde_json::to_string(&adapter_path).unwrap_or_else(|_| "\"\"".to_string()),
        prompt = serde_json::to_string(&prompt).unwrap_or_else(|_| "\"\"".to_string()),
    );
    let cmd = if cfg.docker.enabled {
        pipeline::wrap_docker_cmd(&script, &container_name)
    } else {
        script
    };
    let result = session.exec_blocking(&cmd).await?;
    session.disconnect().await;
    if result.exit_code != 0 {
        return Err(AppError::pipeline(format!(
            "model test failed: {}{}",
            result.stderr, result.stdout
        )));
    }
    let answer = result.stdout.trim().to_string();
    if answer.is_empty() {
        return Err(AppError::pipeline("model test returned an empty answer"));
    }
    Ok(answer)
}

#[tauri::command]
async fn run_inference_benchmark(run_id: String, sample_size: Option<usize>) -> Result<String> {
    let cfg = config::load().await?;
    let run = runs::load(&run_id).await?;
    if run.status != runs::RunStatus::Done {
        return Err(AppError::pipeline("benchmark requires a completed run"));
    }
    let sample_size = sample_size.unwrap_or(100).min(500);
    let dataset_path = format!("{}/data/qa_dataset.jsonl", run.remote_dir);

    let session = SshSession::connect(&cfg.ssh).await?;
    let mut container_name = cfg.docker.container_name.clone();
    if cfg.docker.enabled {
        container_name = pipeline::ensure_container(&session, &cfg.docker).await?;
    }

    let adapter_path = format!("{}/lora", run.remote_dir);
    let check_inner = format!(
        "test -f {}/adapter_model.safetensors && echo OK || echo MISSING",
        pipeline::sh_quote(&adapter_path)
    );
    let check_cmd = if cfg.docker.enabled {
        pipeline::wrap_docker_cmd(&check_inner, &container_name)
    } else {
        check_inner
    };
    let check = session.exec_blocking(&check_cmd).await?;
    if !check.stdout.contains("OK") {
        return Err(AppError::pipeline(format!(
            "adapter not found at {adapter_path}/adapter_model.safetensors"
        )));
    }

    let script = format!(
        r#"set -e
cd {run_dir}
python3 - <<'PY'
import json, re, torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = "{base_model}"
adapter_path = "{adapter_path}"
dataset_path = "{dataset_path}"
hf_repo_id = "{hf_repo_id}"

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
    base = model_cls.from_pretrained(base_model, dtype=dtype, device_map="auto", trust_remote_code=True)
except TypeError:
    base = model_cls.from_pretrained(base_model, torch_dtype=dtype, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(base, adapter_path)
model.eval()

# Load eval samples from qa_dataset.jsonl
samples = []
from pathlib import Path
local_path = Path("{dataset_path}")
if local_path.exists():
    with open(local_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                if "question" in sample and "answer" in sample:
                    samples.append({{"question": sample["question"], "answer": sample["answer"]}})
            except Exception:
                pass
            if len(samples) >= {sample_size}:
                break
else:
    # Try downloading from HuggingFace Hub
    if hf_repo_id:
        try:
            from huggingface_hub import hf_hub_download
            hf_path = hf_hub_download(repo_id=hf_repo_id, filename="qa_dataset.jsonl", repo_type="dataset")
            with open(hf_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sample = json.loads(line)
                        if "question" in sample and "answer" in sample:
                            samples.append({{"question": sample["question"], "answer": sample["answer"]}})
                    except Exception:
                        pass
                    if len(samples) >= {sample_size}:
                        break
        except Exception:
            pass

total = len(samples)
if total == 0:
    print(json.dumps({{"error": "no valid samples found in dataset"}}))
    exit(0)

correct = 0
partial = 0
results = []

for i, s in enumerate(samples):
    question = s["question"]
    expected_answer = s["answer"]
    messages = [
        {{"role": "system", "content": "You are a helpful assistant. Answer using the training domain when relevant."}},
        {{"role": "user", "content": question}},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            repetition_penalty=1.05,
        )
    generated = output_ids[0][inputs.input_ids.shape[-1]:]
    model_answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    
    # Normalize for comparison
    expected_clean = re.sub(r'<[^>]+>', '', expected_answer).lower().strip()
    model_clean = re.sub(r'<[^>]+>', '', model_answer).lower().strip()
    
    # Exact match
    if expected_clean == model_clean:
        correct += 1
        match_type = "exact"
    # Keyword overlap check
    elif model_clean and expected_clean:
        expected_words = set(expected_clean.split())
        model_words = set(model_clean.split())
        overlap = len(expected_words & model_words)
        if overlap >= len(expected_words) * 0.6:
            partial += 1
            match_type = "partial"
        else:
            match_type = "miss"
    else:
        match_type = "miss"
    
    results.append({{
        "question": question[:200],
        "expected": expected_answer[:300],
        "model_answer": model_answer[:300],
        "match": match_type,
    }})

accuracy = (correct + partial * 0.5) / total * 100
summary = {{
    "total": total,
    "correct": correct,
    "partial": partial,
    "missed": total - correct - partial,
    "accuracy": round(accuracy, 2),
    "samples": results,
}}

print(json.dumps(summary, ensure_ascii=False))
PY"#,
        run_dir = pipeline::sh_quote(&run.remote_dir),
        base_model = crate::llamafactory::resolve_trainable_repo(&run.student_model),
        adapter_path = &adapter_path,
        dataset_path = &dataset_path,
        hf_repo_id = run
            .hub_dataset
            .enabled
            .then_some(run.hub_dataset.repo_id.as_str())
            .unwrap_or(""),
        sample_size = sample_size,
    );
    let cmd = if cfg.docker.enabled {
        pipeline::wrap_docker_cmd(&script, &container_name)
    } else {
        script
    };
    let result = session.exec_blocking(&cmd).await?;
    session.disconnect().await;
    if result.exit_code != 0 {
        return Err(AppError::pipeline(format!(
            "benchmark failed: {}{}",
            result.stderr, result.stdout
        )));
    }
    let output = result.stdout.trim().to_string();
    if output.is_empty() {
        return Err(AppError::pipeline("benchmark returned empty output"));
    }
    Ok(output)
}

#[tauri::command]
async fn merge_and_upload_model(run_id: String, target_repo: Option<String>) -> Result<String> {
    let cfg = config::load().await?;
    let mut run = runs::load(&run_id).await?;
    if run.status != runs::RunStatus::Done {
        return Err(AppError::pipeline("merge requires a completed run"));
    }
    let token = cfg
        .hf_token
        .as_ref()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            AppError::pipeline("Hugging Face token is required to upload a merged model")
        })?;

    let repo = target_repo
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| {
            if run.hub.model_id.trim().is_empty() {
                None
            } else {
                Some(format!("{}-merged", run.hub.model_id.trim()))
            }
        })
        .ok_or_else(|| AppError::pipeline("target merged model repo is empty"))?;

    let session = SshSession::connect(&cfg.ssh).await?;
    let mut container_name = cfg.docker.container_name.clone();
    if cfg.docker.enabled {
        container_name = pipeline::ensure_container(&session, &cfg.docker).await?;
    }

    let merged_dir = format!("{}/merged", run.remote_dir);
    let adapter_path = format!("{}/lora", run.remote_dir);
    let private_flag = if run.hub.private { "True" } else { "False" };
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
model = PeftModel.from_pretrained(base, str(adapter_path))
merged = model.merge_and_unload()
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
        run_dir = pipeline::sh_quote(&run.remote_dir),
        token = pipeline::sh_quote(token),
        base_model = serde_json::to_string(&crate::llamafactory::resolve_trainable_repo(
            &run.student_model
        ))
        .unwrap_or_else(|_| "\"\"".to_string()),
        adapter_path = serde_json::to_string(&adapter_path).unwrap_or_else(|_| "\"\"".to_string()),
        merged_dir = serde_json::to_string(&merged_dir).unwrap_or_else(|_| "\"\"".to_string()),
        repo = serde_json::to_string(&repo).unwrap_or_else(|_| "\"\"".to_string()),
        private = private_flag,
    );
    let cmd = if cfg.docker.enabled {
        pipeline::wrap_docker_cmd(&script, &container_name)
    } else {
        script
    };
    let result = session.exec_blocking(&cmd).await?;
    session.disconnect().await;
    if result.exit_code != 0 {
        return Err(AppError::pipeline(format!(
            "merge/upload failed: {}{}",
            result.stderr.replace(token, "***"),
            result.stdout.replace(token, "***")
        )));
    }
    run.hub.merged_model_id = repo.clone();
    runs::save(&run).await?;
    Ok(result.stdout.trim().replace(token, "***"))
}

#[tauri::command]
async fn merge_convert_upload_model(
    run_id: String,
    target_merged_repo: Option<String>,
    target_gguf_repo: Option<String>,
    gguf_quantization: Option<String>,
) -> Result<serde_json::Value> {
    let cfg = config::load().await?;
    let mut run = runs::load(&run_id).await?;
    if run.status != runs::RunStatus::Done {
        return Err(AppError::pipeline("merge requires a completed run"));
    }
    let token = cfg
        .hf_token
        .as_ref()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            AppError::pipeline("Hugging Face token is required to upload merged models")
        })?;

    let merged_repo = target_merged_repo
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| {
            if run.hub.model_id.trim().is_empty() {
                None
            } else {
                Some(format!("{}-merged", run.hub.model_id.trim()))
            }
        })
        .ok_or_else(|| AppError::pipeline("target merged model repo is empty"))?;

    let gguf_repo = target_gguf_repo
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| {
            if run.hub.model_id.trim().is_empty() {
                None
            } else {
                Some(format!("{}-gguf", run.hub.model_id.trim()))
            }
        })
        .ok_or_else(|| AppError::pipeline("target GGUF repo is empty"))?;

    let quantization = gguf_quantization.unwrap_or_else(|| "Q4_K_M".to_string());

    let session = SshSession::connect(&cfg.ssh).await?;
    let mut container_name = cfg.docker.container_name.clone();
    if cfg.docker.enabled {
        container_name = pipeline::ensure_container(&session, &cfg.docker).await?;
    }

    let merged_dir = format!("{}/merged", run.remote_dir);
    let adapter_path = format!("{}/lora", run.remote_dir);
    let gguf_dir = format!("{}/gguf", run.remote_dir);
    let private_flag = if run.hub.private { "True" } else { "False" };

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
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi, create_repo

base_model = {base_model}
adapter_path = Path({adapter_path})
merged_dir = Path({merged_dir})
gguf_dir = Path({gguf_dir})
merged_repo = {merged_repo}
gguf_repo = {gguf_repo}
quantization = ({quantization} or "Q4_K_M").strip()
private = {private}
token = os.environ.get("HF_TOKEN")

def run(cmd, cwd=None):
    cmd = [str(part) for part in cmd]
    print("[gguf] $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)

os.makedirs(merged_dir, exist_ok=True)
os.makedirs(gguf_dir, exist_ok=True)

print("[merge] loading base", flush=True)
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
    base = model_cls.from_pretrained(base_model, dtype=dtype, device_map="auto", trust_remote_code=True)
except TypeError:
    base = model_cls.from_pretrained(base_model, torch_dtype=dtype, device_map="auto", trust_remote_code=True)

print("[merge] applying adapter", flush=True)
model = PeftModel.from_pretrained(base, str(adapter_path))
print("[merge] merging", flush=True)
merged = model.merge_and_unload()
print("[merge] saving", flush=True)
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

print("[merge] uploading to " + merged_repo, flush=True)
create_repo(repo_id=merged_repo, repo_type="model", private=private, token=token, exist_ok=True)
api = HfApi(token=token)
api.upload_folder(repo_id=merged_repo, repo_type="model", folder_path=str(merged_dir), commit_message="Upload merged model")
print("[merge] done: https://huggingface.co/" + merged_repo)

if not merged_dir.is_dir():
    raise SystemExit("merged model directory not found: " + str(merged_dir))

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
create_repo(repo_id=gguf_repo, repo_type="model", private=private, token=token, exist_ok=True)
api.upload_file(repo_id=gguf_repo, repo_type="model", path_in_repo="model.gguf", path_or_fileobj=str(final_gguf), commit_message="Upload GGUF for Ollama/llama.cpp")
print("[gguf] done: https://huggingface.co/" + gguf_repo)
PY"#,
        run_dir = pipeline::sh_quote(&run.remote_dir),
        token = pipeline::sh_quote(token),
        base_model = serde_json::to_string(&crate::llamafactory::resolve_trainable_repo(
            &run.student_model
        ))
        .unwrap_or_else(|_| "\"\"".to_string()),
        adapter_path = serde_json::to_string(&adapter_path).unwrap_or_else(|_| "\"\"".to_string()),
        merged_dir = serde_json::to_string(&merged_dir).unwrap_or_else(|_| "\"\"".to_string()),
        gguf_dir = serde_json::to_string(&gguf_dir).unwrap_or_else(|_| "\"\"".to_string()),
        merged_repo = serde_json::to_string(&merged_repo).unwrap_or_else(|_| "\"\"".to_string()),
        gguf_repo = serde_json::to_string(&gguf_repo).unwrap_or_else(|_| "\"\"".to_string()),
        quantization = serde_json::to_string(&quantization).unwrap_or_else(|_| "\"\"".to_string()),
        private = private_flag,
    );

    let cmd = if cfg.docker.enabled {
        pipeline::wrap_docker_cmd(&script, &container_name)
    } else {
        script
    };

    let result = session.exec_blocking(&cmd).await?;
    session.disconnect().await;
    if result.exit_code != 0 {
        return Err(AppError::pipeline(format!(
            "merge+gguf failed: {}{}",
            result.stderr.replace(token, "***"),
            result.stdout.replace(token, "***")
        )));
    }

    run.hub.merged_model_id = merged_repo.clone();
    run.hub.gguf_repo_id = gguf_repo.clone();
    run.hub.gguf_quantization = quantization.clone();
    runs::save(&run).await?;

    Ok(serde_json::json!({
        "mergedUrl": format!("https://huggingface.co/{}", merged_repo),
        "ggufUrl": format!("https://huggingface.co/{}", gguf_repo),
    }))
}

// ── main ───────────────────────────────────────────────────────────────────

#[tauri::command]
async fn cleanup_vram(cfg: SshConfig, docker: DockerConfig) -> Result<String> {
    let session = SshSession::connect(&cfg).await?;

    let pkill_body = "pkill -f '[v]llm' 2>/dev/null; \
                      pkill -f '[l]lamafactory' 2>/dev/null; \
                      pkill -9 -f '[v]llm' 2>/dev/null; \
                      pkill -9 -f '[l]lamafactory' 2>/dev/null; \
                      true";

    // 1. Host sweep
    let _ = session.exec_blocking(pkill_body).await;

    // 2. Container sweep
    if docker.enabled {
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
                let inner = pipeline::wrap_docker_cmd(pkill_body, cname);
                let _ = session.exec_blocking(&inner).await;
            }
        }
    }

    session.disconnect().await;
    Ok("VRAM cleanup complete (all vLLM and LLaMA Factory processes terminated).".to_string())
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    tauri::Builder::default()
        .manage(AppState::default())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            if let Some(window) = app.get_webview_window("main") {
                match tauri::image::Image::from_bytes(include_bytes!("../icons/icon.png")) {
                    Ok(icon) => {
                        if let Err(error) = window.set_icon(icon) {
                            tracing::warn!("failed to set window icon: {error}");
                        }
                    }
                    Err(error) => tracing::warn!("failed to load window icon: {error}"),
                }
            }

            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                let _ = config::ensure_dirs().await;

                // ── Stale run cleanup ──────────────────────────────────
                // On app restart, any run that was in a non-terminal state
                // (Pending, TeacherLoading, GeneratingDataset, etc.) is
                // actually dead because the tokio task that was driving it
                // no longer exists. Mark them Failed so the UI doesn't
                // show phantom "Pending" or "Connecting" runs.
                if let Ok(all_runs) = runs::list().await {
                    for mut r in all_runs {
                        if !r.status.is_terminal() {
                            r.status = runs::RunStatus::Failed;
                            r.error = Some("interrupted: app was restarted while this run was active".to_string());
                            let _ = runs::save(&r).await;
                            tracing::info!("cleaned up stale run {} (was {:?})", r.id, r.status);
                        }
                    }
                }

                // Surface app dir to the UI so it can show the path in Settings.
                let _ = handle.emit(
                    "app://ready",
                    serde_json::json!({
                        "appDir": config::app_dir().map(|p| p.to_string_lossy().into_owned()).unwrap_or_default()
                    }),
                );
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            save_config,
            load_config,
            read_local_file_text,
            list_ingestable_files,
            save_ingest_state,
            load_ingest_state,
            do_list_gpu_sizes,
            do_list_droplets,
            do_list_gpu_droplets,
            do_list_regions,
            do_list_images,
            do_list_ssh_keys,
            do_list_projects,
            do_get_account,
            do_create_gpu_droplet,
            do_destroy_droplet,
            do_list_droplet_usage,
            do_export_droplet_usage_csv,
            test_ssh,
            nvidia_smi,
            ssh_exec_stream,
            ssh_stop_stream,
            write_remote_file,
            qdrant_count,
            qdrant_sample,
            qdrant_ensure_collection,
            qdrant_sample_in_collection,
            qdrant_scroll_in_collection,
            qdrant_scroll_all,
            qdrant_scroll_all_in_collection,
            list_qdrant_collections,
            list_qdrant_snapshots,
            create_qdrant_snapshot,
            restore_qdrant_snapshot,
            qdrant_upload_snapshot,
            qdrant_download_snapshot,
            create_all_qdrant_snapshots,
            download_all_qdrant_snapshots,
            ingest_documents,
            cancel_ingest,
            serve_ensure_qdrant,
            serve_boot_embedder,
            serve_check_embedder,
            serve_boot_paddleocr,
            serve_setup_all_embedders,
            start_pipeline,
            cancel_run,
            resume_run,
            list_runs,
            get_run,
            list_local_dataset,
            list_local_dataset_page,
            open_runs_folder,
            read_run_log,
ping_teacher,
            teacher_chat,
            test_trained_model,
            run_inference_benchmark,
            merge_and_upload_model,
            merge_convert_upload_model,
            hf_whoami,
            hf_list_datasets,
            hf_list_models,
            hf_validate_dataset,
            check_teacher_deployed,
            deploy_teacher,
            update_run_config,
            match_model_guide,
            cleanup_vram,
            ai_get_app_state,
            ai_get_runs_summary,
            ai_get_run_details,
            ai_cancel_run,
            ai_get_gpu_status,
            ai_trigger_pipeline_action,
            ai_get_config_summary,
            ai_proxy_chat,
            ai_list_models,
            robot_list_captures,
            robot_enqueue_capture,
            robot_research_capture,
            robot_approve_capture,
            robot_reject_capture,
            robot_list_manifests,
            robot_publish_manifest,
            robot_promote_model,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

// ── robotics commands (shared with the headless VPS server) ─────────────────

/// List all robot captures (pending/researched/approved/rejected/failed).
#[tauri::command]
async fn robot_list_captures() -> Result<Vec<robot::Capture>> {
    robot::list_captures().await
}

/// Enqueue a capture from the desktop (testing the robot flow without a robot).
#[tauri::command]
async fn robot_enqueue_capture(input: robot::CaptureInput) -> Result<robot::Capture> {
    let cfg = config::load().await?;
    robot::enqueue_capture(&cfg, input).await
}

/// OCR + web-research a capture and embed the cited packet into Qdrant.
#[tauri::command]
async fn robot_research_capture(id: String) -> Result<robot::Capture> {
    let cfg = config::load().await?;
    robot::research_capture(&cfg, &id).await
}

/// Approve a researched capture (clears it for training). Human approval gate.
#[tauri::command]
async fn robot_approve_capture(id: String) -> Result<robot::Capture> {
    robot::set_status(&id, robot::CaptureStatus::Approved).await
}

/// Reject a capture (it will not be used for training).
#[tauri::command]
async fn robot_reject_capture(id: String) -> Result<robot::Capture> {
    robot::set_status(&id, robot::CaptureStatus::Rejected).await
}

/// Full model-manifest store (all versions + the current served version).
#[tauri::command]
async fn robot_list_manifests() -> Result<manifest::ManifestStore> {
    manifest::load().await
}

/// Publish a new model manifest and make it the version the robot pulls.
#[tauri::command]
async fn robot_publish_manifest(manifest: manifest::ModelManifest) -> Result<manifest::ManifestStore> {
    crate::manifest::publish(manifest).await
}

/// Promote/roll back the served model to an already-published version.
#[tauri::command]
async fn robot_promote_model(version: String) -> Result<manifest::ManifestStore> {
    manifest::set_current(&version).await
}

#[tauri::command]
async fn ai_get_app_state() -> Result<serde_json::Value> {
    let cfg = config::load().await?;
    Ok(serde_json::json!({
        "sshConfigured": !cfg.ssh.host.is_empty(),
        "sshHost": cfg.ssh.host,
        "qdrantConfigured": !cfg.qdrant.endpoint.is_empty(),
        "hfTokenConfigured": cfg.hf_token.as_ref().filter(|s| !s.is_empty()).is_some(),
        "dockerEnabled": cfg.docker.enabled,
        "studentModel": cfg.student.repo_id,
        "gpuAvailable": true,
    }))
}

#[tauri::command]
async fn ai_get_runs_summary() -> Result<Vec<serde_json::Value>> {
    let runs = runs::list().await?;
    Ok(runs
        .into_iter()
        .map(|r| {
            serde_json::json!({
                "id": r.id,
                "name": r.name,
                "status": r.status,
                "teacherModel": r.teacher_model,
                "studentModel": r.student_model,
                "createdAt": r.created_at,
            })
        })
        .collect())
}

#[tauri::command]
async fn ai_get_run_details(run_id: String) -> Result<serde_json::Value> {
    let run = runs::load(&run_id).await?;
    Ok(serde_json::json!({
        "id": run.id,
        "name": run.name,
        "status": run.status,
        "error": run.error,
        "qaTotal": run.qa_total,
        "qaKept": run.qa_kept,
        "qaRejected": run.qa_rejected,
        "trainLossHistory": run.train_loss_history,
        "topicStats": run.topic_stats,
    }))
}

#[tauri::command]
async fn ai_cancel_run(_run_id: String) -> Result<()> {
    Ok(())
}

#[tauri::command]
async fn ai_get_gpu_status(ssh_cfg: SshConfig, docker_cfg: DockerConfig) -> Result<String> {
    let session = SshSession::connect(&ssh_cfg).await?;
    let cmd = if docker_cfg.enabled {
        pipeline::wrap_docker_cmd("rocm-smi --json", &docker_cfg.container_name)
    } else {
        "rocm-smi --json".to_string()
    };
    let r = session.exec_blocking(&cmd).await?;
    session.disconnect().await;
    Ok(r.stdout)
}

#[tauri::command]
async fn ai_trigger_pipeline_action(action: String, _params: serde_json::Value) -> Result<String> {
    match action.as_str() {
        "refresh_runs" => {
            let _ = runs::list().await?;
            Ok("Runs list refreshed".to_string())
        }
        _ => Err(AppError::pipeline(format!("Unknown action: {}", action))),
    }
}

#[tauri::command]
async fn ai_get_config_summary() -> Result<serde_json::Value> {
    let cfg = config::load().await?;
    Ok(serde_json::json!({
        "sshHost": cfg.ssh.host,
        "sshUsername": cfg.ssh.username,
        "qdrantEndpoint": cfg.qdrant.endpoint,
        "qdrantCollection": cfg.qdrant.collection,
        "dockerEnabled": cfg.docker.enabled,
        "dockerContainer": cfg.docker.container_name,
        "studentRepoId": cfg.student.repo_id,
        "teacherRepoId": cfg.teacher.repo_id,
        "vllmPort": cfg.teacher.vllm_port,
    }))
}

#[tauri::command]
async fn ai_proxy_chat(
    api_url: String,
    api_key: String,
    request_body: String,
    provider: Option<String>,
) -> Result<String> {
    let client = reqwest::Client::new();
    let mut req = client
        .post(&api_url)
        .header("Content-Type", "application/json")
        .body(request_body);

    // Anthropic uses x-api-key + a version header instead of Bearer auth.
    if provider.as_deref() == Some("anthropic") {
        req = req
            .header("x-api-key", &api_key)
            .header("anthropic-version", "2023-06-01");
    } else {
        req = req.header("Authorization", format!("Bearer {}", api_key));
    }

    let response = req
        .send()
        .await
        .map_err(|e| AppError::pipeline(e.to_string()))?;

    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|e| AppError::pipeline(e.to_string()))?;

    if !status.is_success() {
        return Err(AppError::pipeline(format!(
            "API error {}: {}",
            status, body
        )));
    }

    Ok(body)
}

/// List available model IDs from an OpenAI-compatible (or Anthropic) provider.
/// Runs server-side so it isn't subject to browser CORS restrictions — the
/// frontend used to `fetch()` these endpoints directly and got blocked.
#[tauri::command]
async fn ai_list_models(provider: String, api_url: String, api_key: String) -> Result<Vec<String>> {
    // Build the /models endpoint from whatever base URL the user configured.
    let trimmed = api_url.trim().trim_end_matches('/').to_string();
    let endpoint = if trimmed.contains("/models") {
        trimmed
    } else if trimmed.ends_with("/v1") {
        format!("{trimmed}/models")
    } else if provider == "anthropic" {
        format!("{trimmed}/v1/models")
    } else {
        format!("{trimmed}/models")
    };

    let client = reqwest::Client::new();
    let mut req = client
        .get(&endpoint)
        .header("Content-Type", "application/json");
    if provider == "anthropic" {
        req = req
            .header("x-api-key", &api_key)
            .header("anthropic-version", "2023-06-01");
    } else {
        req = req.header("Authorization", format!("Bearer {}", api_key));
    }

    let response = req
        .send()
        .await
        .map_err(|e| AppError::pipeline(e.to_string()))?;
    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|e| AppError::pipeline(e.to_string()))?;
    if !status.is_success() {
        return Err(AppError::pipeline(format!(
            "models API error {}: {}",
            status, body
        )));
    }

    let v: serde_json::Value = serde_json::from_str(&body)
        .map_err(|e| AppError::pipeline(format!("models response not JSON: {e}")))?;
    // Both OpenAI-compatible and Anthropic responses use {"data":[{"id":...}]}.
    // Capture each model's optional `created` timestamp (OpenAI/xAI/Gemini
    // include it) so we can surface the newest models first.
    let mut entries: Vec<(String, i64)> = v
        .get("data")
        .and_then(|d| d.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| {
                    let id = m.get("id").and_then(|i| i.as_str())?.to_string();
                    let created = m.get("created").and_then(|c| c.as_i64()).unwrap_or(0);
                    Some((id, created))
                })
                .collect::<Vec<(String, i64)>>()
        })
        .unwrap_or_default();

    // If any model carries a `created` timestamp, sort newest-first by it.
    // Otherwise preserve the provider's original ordering (a stable no-op sort).
    if entries.iter().any(|(_, created)| *created > 0) {
        entries.sort_by(|a, b| b.1.cmp(&a.1));
    }

    // Limit to at most 5 models per provider to keep the picker focused.
    let ids = entries
        .into_iter()
        .map(|(id, _)| id)
        .take(5)
        .collect::<Vec<String>>();
    Ok(ids)
}
