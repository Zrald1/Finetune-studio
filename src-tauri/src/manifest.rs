//! Model-manifest store served to the robot.
//!
//! Each published student model gets a versioned manifest with a checksum and a
//! pointer to its previous version (for rollback). The robot polls the
//! "current" manifest and downloads/pulls only when the version changes. The
//! store is a single JSON file under the app data dir, so it survives restarts
//! and is shared by the desktop app and the headless server.

use crate::config::manifest_path;
use crate::error::{AppError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelManifest {
    pub version: String,
    /// Hugging Face repo id the merged student model was uploaded to.
    pub hf_repo: String,
    /// Revision / commit on the HF repo.
    pub hf_revision: String,
    /// Checksum of the published artifact (sha256), if computed.
    pub sha256: String,
    pub eval_summary: Option<String>,
    pub created_at: DateTime<Utc>,
    /// Version this one supersedes — lets the robot/operator roll back.
    pub previous_version: Option<String>,
    /// Run id that produced this model.
    pub run_id: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ManifestStore {
    /// The version the robot should currently pull. None = nothing published yet.
    pub current_version: Option<String>,
    pub manifests: Vec<ModelManifest>,
}

pub async fn load() -> Result<ManifestStore> {
    let path = manifest_path()?;
    match fs::read_to_string(&path).await {
        Ok(txt) => serde_json::from_str(&txt)
            .map_err(|e| AppError::config(format!("parse model_manifests.json: {e}"))),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(ManifestStore::default()),
        Err(e) => Err(AppError::Io(e)),
    }
}

pub async fn save(store: &ManifestStore) -> Result<()> {
    crate::config::ensure_dirs().await?;
    let txt = serde_json::to_string_pretty(store)?;
    fs::write(manifest_path()?, txt).await?;
    Ok(())
}

/// Publish a new manifest and make it the current served version. Links the
/// previous current version for rollback.
pub async fn publish(mut manifest: ModelManifest) -> Result<ManifestStore> {
    let mut store = load().await?;
    manifest.previous_version = store.current_version.clone();
    store.current_version = Some(manifest.version.clone());
    // replace any existing manifest with the same version, else append
    if let Some(slot) = store.manifests.iter_mut().find(|m| m.version == manifest.version) {
        *slot = manifest;
    } else {
        store.manifests.push(manifest);
    }
    save(&store).await?;
    Ok(store)
}

/// Pin the served version to an arbitrary already-published version
/// (promote or roll back).
pub async fn set_current(version: &str) -> Result<ManifestStore> {
    let mut store = load().await?;
    if !store.manifests.iter().any(|m| m.version == version) {
        return Err(AppError::NotFound(format!("manifest version '{version}'")));
    }
    store.current_version = Some(version.to_string());
    save(&store).await?;
    Ok(store)
}

/// The manifest the robot should currently pull.
pub async fn current() -> Result<Option<ModelManifest>> {
    let store = load().await?;
    Ok(store
        .current_version
        .as_ref()
        .and_then(|v| store.manifests.iter().find(|m| &m.version == v).cloned()))
}
