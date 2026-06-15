//! Headless VPS server for the robot↔FineTune bridge.
//!
//! Exposes a REST API the AMD robot calls to upload unfamiliar-object captures
//! and to poll for model updates, plus operator/dashboard endpoints the desktop
//! app (in Remote mode) uses to review the capture queue, approve captures, and
//! promote/roll back the served model.
//!
//! Reuses the `fine_tune` library crate for all business logic (config, SSH,
//! Qdrant, ingest, OCR, web research, manifests) — no logic is forked here.
//!
//! Run:  FT_DATA_DIR=/var/lib/fine-tune  cargo run --bin fine-tune-server
//! Put TLS in front of it with caddy/nginx; auth is bearer-token via config.

use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use fine_tune::config::{self, AppConfig};
use fine_tune::manifest::{self, ModelManifest};
use fine_tune::robot::{self, CaptureInput, CaptureStatus};
use serde_json::{json, Value};
use std::sync::Arc;
use tower_http::cors::CorsLayer;

#[derive(Clone)]
struct AppState {
    cfg: Arc<tokio::sync::RwLock<AppConfig>>,
}

#[tokio::main]
async fn main() {
    let cfg = config::load().await.unwrap_or_default();
    let port: u16 = std::env::var("FT_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8787);

    let state = AppState {
        cfg: Arc::new(tokio::sync::RwLock::new(cfg)),
    };

    let app = Router::new()
        .route("/health", get(health))
        // ── Robot API (robot bearer token) ──
        .route("/robot/capture", post(robot_capture))
        .route("/robot/captures", get(robot_list_captures))
        .route("/robot/model", get(robot_model))
        // ── Operator / dashboard API (dashboard bearer token) ──
        .route("/config", get(get_config).put(put_config))
        .route("/captures/:id/research", post(op_research_capture))
        .route("/captures/:id/approve", post(op_approve_capture))
        .route("/captures/:id/reject", post(op_reject_capture))
        .route("/model/manifests", get(op_list_manifests))
        .route("/model/publish", post(op_publish_manifest))
        .route("/model/promote/:version", post(op_promote))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("0.0.0.0:{port}");
    println!("[fine-tune-server] listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("bind listener");
    axum::serve(listener, app).await.expect("server run");
}

// ── auth helpers ────────────────────────────────────────────────────────────

fn bearer(headers: &HeaderMap) -> Option<String> {
    headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
        .map(|s| s.trim().to_string())
}

/// Returns Err(response) when auth fails. An empty configured token means that
/// surface is open (useful for first-run / local testing) — the operator sets
/// tokens in the Robotics widget to lock it down.
fn check(headers: &HeaderMap, expected: &str) -> Result<(), (StatusCode, Json<Value>)> {
    if expected.trim().is_empty() {
        return Ok(());
    }
    match bearer(headers) {
        Some(tok) if tok == expected => Ok(()),
        _ => Err((
            StatusCode::UNAUTHORIZED,
            Json(json!({ "error": "invalid or missing bearer token" })),
        )),
    }
}

fn err(status: StatusCode, msg: impl ToString) -> (StatusCode, Json<Value>) {
    (status, Json(json!({ "error": msg.to_string() })))
}

// ── handlers ─────────────────────────────────────────────────────────────────

async fn health() -> impl IntoResponse {
    Json(json!({ "ok": true, "service": "fine-tune-server" }))
}

async fn robot_capture(
    State(st): State<AppState>,
    headers: HeaderMap,
    Json(input): Json<CaptureInput>,
) -> impl IntoResponse {
    let cfg = st.cfg.read().await.clone();
    if let Err(e) = check(&headers, &cfg.robot.robot_api_token) {
        return e.into_response();
    }
    if !cfg.robot.enabled {
        return err(StatusCode::FORBIDDEN, "robot intake is disabled").into_response();
    }

    let capture = match robot::enqueue_capture(&cfg, input).await {
        Ok(c) => c,
        Err(e) => return err(StatusCode::BAD_REQUEST, e).into_response(),
    };

    // Optionally research immediately (training still gated on approval).
    if cfg.robot.auto_research_on_capture {
        let id = capture.id.clone();
        let cfg2 = cfg.clone();
        tokio::spawn(async move {
            if let Err(e) = robot::research_capture(&cfg2, &id).await {
                eprintln!("[robot] auto-research failed for {id}: {e}");
            }
        });
    }

    (
        StatusCode::ACCEPTED,
        Json(json!({ "captureId": capture.id, "status": capture.status })),
    )
        .into_response()
}

async fn robot_list_captures(State(st): State<AppState>, headers: HeaderMap) -> impl IntoResponse {
    let cfg = st.cfg.read().await.clone();
    if let Err(e) = check(&headers, &cfg.robot.robot_api_token) {
        return e.into_response();
    }
    match robot::list_captures().await {
        Ok(list) => Json(list).into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e).into_response(),
    }
}

async fn robot_model(State(st): State<AppState>, headers: HeaderMap) -> impl IntoResponse {
    let cfg = st.cfg.read().await.clone();
    if let Err(e) = check(&headers, &cfg.robot.robot_api_token) {
        return e.into_response();
    }
    match manifest::current().await {
        Ok(Some(m)) => Json(json!({ "available": true, "manifest": m })).into_response(),
        Ok(None) => Json(json!({ "available": false })).into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e).into_response(),
    }
}

async fn get_config(State(st): State<AppState>, headers: HeaderMap) -> impl IntoResponse {
    let cfg = st.cfg.read().await.clone();
    if let Err(e) = check(&headers, &cfg.robot.dashboard_api_token) {
        return e.into_response();
    }
    Json(cfg).into_response()
}

async fn put_config(
    State(st): State<AppState>,
    headers: HeaderMap,
    Json(new_cfg): Json<AppConfig>,
) -> impl IntoResponse {
    {
        let cur = st.cfg.read().await;
        if let Err(e) = check(&headers, &cur.robot.dashboard_api_token) {
            return e.into_response();
        }
    }
    if let Err(e) = config::save(&new_cfg).await {
        return err(StatusCode::INTERNAL_SERVER_ERROR, e).into_response();
    }
    *st.cfg.write().await = new_cfg;
    Json(json!({ "ok": true })).into_response()
}

async fn op_research_capture(
    State(st): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let cfg = st.cfg.read().await.clone();
    if let Err(e) = check(&headers, &cfg.robot.dashboard_api_token) {
        return e.into_response();
    }
    match robot::research_capture(&cfg, &id).await {
        Ok(c) => Json(c).into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e).into_response(),
    }
}

async fn op_approve_capture(
    State(st): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let cfg = st.cfg.read().await.clone();
    if let Err(e) = check(&headers, &cfg.robot.dashboard_api_token) {
        return e.into_response();
    }
    match robot::set_status(&id, CaptureStatus::Approved).await {
        Ok(c) => Json(c).into_response(),
        Err(e) => err(StatusCode::NOT_FOUND, e).into_response(),
    }
}

async fn op_reject_capture(
    State(st): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let cfg = st.cfg.read().await.clone();
    if let Err(e) = check(&headers, &cfg.robot.dashboard_api_token) {
        return e.into_response();
    }
    match robot::set_status(&id, CaptureStatus::Rejected).await {
        Ok(c) => Json(c).into_response(),
        Err(e) => err(StatusCode::NOT_FOUND, e).into_response(),
    }
}

async fn op_list_manifests(State(st): State<AppState>, headers: HeaderMap) -> impl IntoResponse {
    let cfg = st.cfg.read().await.clone();
    if let Err(e) = check(&headers, &cfg.robot.dashboard_api_token) {
        return e.into_response();
    }
    match manifest::load().await {
        Ok(store) => Json(store).into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e).into_response(),
    }
}

async fn op_publish_manifest(
    State(st): State<AppState>,
    headers: HeaderMap,
    Json(m): Json<ModelManifest>,
) -> impl IntoResponse {
    let cfg = st.cfg.read().await.clone();
    if let Err(e) = check(&headers, &cfg.robot.dashboard_api_token) {
        return e.into_response();
    }
    match manifest::publish(m).await {
        Ok(store) => Json(store).into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e).into_response(),
    }
}

async fn op_promote(
    State(st): State<AppState>,
    headers: HeaderMap,
    Path(version): Path<String>,
) -> impl IntoResponse {
    let cfg = st.cfg.read().await.clone();
    if let Err(e) = check(&headers, &cfg.robot.dashboard_api_token) {
        return e.into_response();
    }
    match manifest::set_current(&version).await {
        Ok(store) => Json(store).into_response(),
        Err(e) => err(StatusCode::NOT_FOUND, e).into_response(),
    }
}
