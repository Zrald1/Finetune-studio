#![allow(dead_code)]

use crate::config::QdrantConfig;
use crate::error::{AppError, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: String,
    pub text: String,
    #[serde(default)]
    pub file_path: String,
    #[serde(default)]
    pub file_name: String,
    #[serde(default)]
    pub chunk_index: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
}

fn http() -> Client {
    static CLIENT: std::sync::OnceLock<Client> = std::sync::OnceLock::new();
    CLIENT
        .get_or_init(|| {
            Client::builder()
                .timeout(std::time::Duration::from_secs(30 * 60))
                .build()
                .unwrap()
        })
        .clone()
}

fn base(cfg: &QdrantConfig) -> Result<String> {
    if cfg.endpoint.is_empty() {
        return Err(AppError::qdrant("endpoint is empty"));
    }
    Ok(cfg.endpoint.trim_end_matches('/').to_string())
}

fn api_key(cfg: &QdrantConfig) -> String {
    cfg.api_key.clone()
}

pub async fn health(cfg: &QdrantConfig) -> Result<()> {
    let url = format!("{}/collections", base(cfg)?);
    let res = http()
        .get(&url)
        .header("api-key", api_key(cfg))
        .send()
        .await?;
    if !res.status().is_success() {
        let s = res.status();
        let body = res.text().await.unwrap_or_default();
        return Err(AppError::qdrant(format!(
            "Qdrant health check failed {s}: {body}"
        )));
    }
    Ok(())
}

pub async fn count_in_collection(cfg: &QdrantConfig, collection: &str) -> Result<u64> {
    let url = format!("{}/collections/{}/points/count", base(cfg)?, collection);
    let res = http()
        .post(url)
        .header("api-key", api_key(cfg))
        .json(&json!({ "exact": true }))
        .send()
        .await?;
    if !res.status().is_success() {
        let s = res.status();
        let body = res.text().await.unwrap_or_default();
        return Err(AppError::qdrant(format!("count failed {s}: {body}")));
    }
    let body = res
        .text()
        .await
        .map_err(|e| AppError::qdrant(format!("count body read failed: {e}")))?;
    let v: Value = serde_json::from_str(&body).map_err(|e| {
        let preview: String = body.chars().take(200).collect();
        AppError::qdrant(format!("count body not JSON: {e} — preview: {preview}"))
    })?;
    let count = v
        .get("result")
        .and_then(|r| r.get("count"))
        .and_then(|c| c.as_u64())
        .ok_or_else(|| AppError::qdrant("response missing result.count"))?;
    Ok(count)
}

pub async fn count(cfg: &QdrantConfig) -> Result<u64> {
    if cfg.collection == "all" {
        let cols = list_collections(cfg).await?;
        let mut total = 0;
        for col in cols {
            if let Ok(c) = count_in_collection(cfg, &col.name).await {
                total += c;
            }
        }
        Ok(total)
    } else {
        count_in_collection(cfg, &cfg.collection).await
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ScrollPage {
    pub chunks: Vec<Chunk>,
    pub next_offset: Option<Value>,
}

pub async fn scroll_in_collection(
    cfg: &QdrantConfig,
    collection: &str,
    page_size: u32,
    offset: Option<Value>,
    tag_filter: Option<&str>,
) -> Result<ScrollPage> {
    let url = format!("{}/collections/{}/points/scroll", base(cfg)?, collection);
    let mut body = json!({
        "limit": page_size,
        "with_payload": true,
        "with_vector": false,
    });
    if let Some(off) = offset {
        body["offset"] = off;
    }
    if let Some(tag) = tag_filter {
        if !tag.is_empty() {
            ensure_text_index(cfg, collection, "tag").await?;
            body["filter"] = json!({
                "must": [
                    { "key": "tag", "match": { "text": tag } }
                ]
            });
        }
    }
    let res = http()
        .post(url)
        .header("api-key", api_key(cfg))
        .json(&body)
        .send()
        .await?;
    if !res.status().is_success() {
        let s = res.status();
        let text = res.text().await.unwrap_or_default();
        return Err(AppError::qdrant(format!("scroll failed {s}: {text}")));
    }
    let body = res
        .text()
        .await
        .map_err(|e| AppError::qdrant(format!("scroll body read failed: {e}")))?;
    let v: Value = serde_json::from_str(&body).map_err(|e| {
        let preview: String = body.chars().take(200).collect();
        AppError::qdrant(format!("scroll body not JSON: {e} — preview: {preview}"))
    })?;
    let result = v
        .get("result")
        .ok_or_else(|| AppError::qdrant("response missing result"))?;
    let next_offset = result.get("next_page_offset").cloned();
    let chunks = parse_chunk_array(result.get("points"));
    Ok(ScrollPage {
        chunks,
        next_offset,
    })
}

pub async fn scroll(
    cfg: &QdrantConfig,
    page_size: u32,
    offset: Option<Value>,
    tag_filter: Option<&str>,
) -> Result<ScrollPage> {
    if cfg.collection == "all" {
        let cols = list_collections(cfg).await?;
        if cols.is_empty() {
            return Ok(ScrollPage {
                chunks: vec![],
                next_offset: None,
            });
        }

        let (col_index, col_offset) = match offset {
            Some(val) => {
                let idx = val.get("col_index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                let off = val.get("col_offset").cloned();
                let off = if off.as_ref().map(|v| v.is_null()).unwrap_or(true) {
                    None
                } else {
                    off
                };
                (idx, off)
            }
            None => (0, None),
        };

        let mut accumulated_chunks = Vec::new();
        let mut current_offset = col_offset;
        let mut current_index = col_index;

        while current_index < cols.len() && accumulated_chunks.len() < page_size as usize {
            let col = &cols[current_index];
            let needed = page_size as usize - accumulated_chunks.len();
            let page = scroll_in_collection(
                cfg,
                &col.name,
                needed as u32,
                current_offset.take(),
                tag_filter,
            )
            .await?;
            accumulated_chunks.extend(page.chunks);

            if let Some(next) = page.next_offset {
                return Ok(ScrollPage {
                    chunks: accumulated_chunks,
                    next_offset: Some(json!({
                        "col_index": current_index,
                        "col_offset": next
                    })),
                });
            } else {
                current_index += 1;
                current_offset = None;
            }
        }

        let next_offset = if current_index < cols.len() {
            Some(json!({
                "col_index": current_index,
                "col_offset": null
            }))
        } else {
            None
        };

        Ok(ScrollPage {
            chunks: accumulated_chunks,
            next_offset,
        })
    } else {
        scroll_in_collection(cfg, &cfg.collection, page_size, offset, tag_filter).await
    }
}

fn parse_chunk_array(arr: Option<&Value>) -> Vec<Chunk> {
    let mut chunks = vec![];
    if let Some(pts) = arr.and_then(|p| p.as_array()) {
        for p in pts {
            let id = p
                .get("id")
                .map(|i| i.to_string().trim_matches('"').to_string())
                .unwrap_or_default();
            let payload = p.get("payload").cloned().unwrap_or(json!({}));
            let text = payload
                .get("content")
                .and_then(|t| t.as_str())
                .or_else(|| payload.get("text").and_then(|t| t.as_str()))
                .or_else(|| payload.get("page_content").and_then(|t| t.as_str()))
                .or_else(|| payload.get("chunk_text").and_then(|t| t.as_str()))
                .unwrap_or("")
                .to_string();
            if text.is_empty() {
                continue;
            }
            let score = p.get("score").and_then(|s| s.as_f64());
            chunks.push(Chunk {
                id,
                text,
                file_path: payload
                    .get("file_path")
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string(),
                file_name: payload
                    .get("file_name")
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string(),
                chunk_index: payload
                    .get("chunk_index")
                    .and_then(|t| t.as_i64())
                    .unwrap_or(0),
                score,
            });
        }
    }
    chunks
}

pub async fn sample(cfg: &QdrantConfig, n: u32) -> Result<Vec<Chunk>> {
    let page = scroll(cfg, n.max(1), None, None).await?;
    Ok(page.chunks)
}

pub async fn sample_in_collection(
    cfg: &QdrantConfig,
    collection: &str,
    n: u32,
) -> Result<Vec<Chunk>> {
    let page = scroll_in_collection(cfg, collection, n.max(1), None, None).await?;
    Ok(page.chunks)
}

pub async fn scroll_all(cfg: &QdrantConfig, max_total: u32) -> Result<Vec<Chunk>> {
    let mut all_chunks = Vec::new();
    let mut offset: Option<Value> = None;
    let page_size = 256u32.min(max_total);
    while all_chunks.len() < max_total as usize {
        let page = scroll(cfg, page_size, offset, None).await?;
        let empty = page.chunks.is_empty();
        let next = page.next_offset;
        all_chunks.extend(page.chunks);
        if next.is_none() || empty {
            break;
        }
        offset = next;
    }
    if all_chunks.len() > max_total as usize {
        all_chunks.truncate(max_total as usize);
    }
    Ok(all_chunks)
}

pub async fn scroll_all_in_collection(
    cfg: &QdrantConfig,
    collection: &str,
    max_total: u32,
) -> Result<Vec<Chunk>> {
    let mut all_chunks = Vec::new();
    let mut offset: Option<Value> = None;
    let page_size = 256u32.min(max_total);
    while all_chunks.len() < max_total as usize {
        let page = scroll_in_collection(cfg, collection, page_size, offset, None).await?;
        let empty = page.chunks.is_empty();
        let next = page.next_offset;
        all_chunks.extend(page.chunks);
        if next.is_none() || empty {
            break;
        }
        offset = next;
    }
    if all_chunks.len() > max_total as usize {
        all_chunks.truncate(max_total as usize);
    }
    Ok(all_chunks)
}

/// Ensure a keyword payload index exists for `field` in the given collection.
pub async fn ensure_keyword_index(cfg: &QdrantConfig, collection: &str, field: &str) -> Result<()> {
    let url = format!("{}/collections/{}/index?wait=true", base(cfg)?, collection);
    let body = json!({
        "field_name": field,
        "field_schema": "keyword",
    });
    let res = http()
        .put(&url)
        .header("api-key", api_key(cfg))
        .json(&body)
        .send()
        .await?;
    if res.status().is_success() {
        return Ok(());
    }
    let status = res.status();
    let text = res.text().await.unwrap_or_default();
    if text.to_lowercase().contains("already exists") {
        return Ok(());
    }
    Err(AppError::qdrant(format!(
        "create payload index for '{field}' failed {status}: {text}"
    )))
}

/// Ensure a text payload index exists for `field` in the given collection.
/// If a conflicting keyword index exists, we delete it and recreate it as a text index.
pub async fn ensure_text_index(cfg: &QdrantConfig, collection: &str, field: &str) -> Result<()> {
    let url = format!("{}/collections/{}/index?wait=true", base(cfg)?, collection);
    let body = json!({
        "field_name": field,
        "field_schema": "text",
    });
    let res = http()
        .put(&url)
        .header("api-key", api_key(cfg))
        .json(&body)
        .send()
        .await?;
    if res.status().is_success() {
        return Ok(());
    }
    let status = res.status();
    let text = res.text().await.unwrap_or_default();
    if text.to_lowercase().contains("already exists") {
        return Ok(());
    }

    // If it conflicts (already indexed as keyword), delete it first and recreate.
    if text.to_lowercase().contains("already indexed")
        || text.to_lowercase().contains("conflict")
        || status.as_u16() == 400
    {
        let delete_url = format!(
            "{}/collections/{}/index/{}?wait=true",
            base(cfg)?,
            collection,
            field
        );
        let _ = http()
            .delete(&delete_url)
            .header("api-key", api_key(cfg))
            .send()
            .await;

        let res2 = http()
            .put(&url)
            .header("api-key", api_key(cfg))
            .json(&body)
            .send()
            .await?;
        if res2.status().is_success() {
            return Ok(());
        }
    }
    Err(AppError::qdrant(format!(
        "create text payload index for '{field}' failed {status}: {text}"
    )))
}

/// Get the configured vector size of an existing collection, or None if it doesn't exist.
async fn get_collection_dim(cfg: &QdrantConfig, collection: &str) -> Option<usize> {
    let url = format!(
        "{}/collections/{}",
        cfg.endpoint.trim_end_matches('/'),
        collection
    );
    let res = http()
        .get(&url)
        .header("api-key", api_key(cfg))
        .send()
        .await
        .ok()?;
    if !res.status().is_success() {
        return None;
    }
    let v: Value = res.json().await.ok()?;
    // Qdrant response: result.config.params.vectors.size
    v.pointer("/result/config/params/vectors/size")
        .and_then(|s| s.as_u64())
        .map(|s| s as usize)
}

/// Create a Qdrant collection with a specific vector dimension.
/// - If it does not exist → creates it.
/// - If it exists with the **same** dim → does nothing.
/// - If it exists with a **different** dim → deletes and recreates it with the correct dim.
pub async fn create_collection(cfg: &QdrantConfig, collection: &str, dim: usize) -> Result<()> {
    let base = base(cfg)?;
    let url = format!("{}/collections/{}", base, collection);

    // First try to create
    let body = json!({
        "vectors": {
            "size": dim,
            "distance": "Cosine"
        }
    });
    let res = http()
        .put(&url)
        .header("api-key", api_key(cfg))
        .json(&body)
        .send()
        .await?;

    if res.status().is_success() {
        return Ok(());
    }

    let status = res.status();
    let text = res.text().await.unwrap_or_default();

    // Collection already exists
    if status.as_u16() == 409 || text.to_lowercase().contains("already exists") {
        // Check if the existing collection has the right dimension
        if let Some(existing_dim) = get_collection_dim(cfg, collection).await {
            if existing_dim == dim {
                // Dimension matches — nothing to do
                return Ok(());
            }
            // Dimension mismatch: delete and recreate
            eprintln!(
                "[qdrant] Collection '{}' has dim {} but need {} — recreating",
                collection, existing_dim, dim
            );
            let del_res = http()
                .delete(&url)
                .header("api-key", api_key(cfg))
                .send()
                .await?;
            if !del_res.status().is_success() {
                let ds = del_res.status();
                let dt = del_res.text().await.unwrap_or_default();
                return Err(AppError::qdrant(format!(
                    "delete mismatched collection '{}' failed {ds}: {dt}",
                    collection
                )));
            }
        }
        // Recreate with correct dim
        let recreate_res = http()
            .put(&url)
            .header("api-key", api_key(cfg))
            .json(&body)
            .send()
            .await?;
        if recreate_res.status().is_success() {
            return Ok(());
        }
        let rs = recreate_res.status();
        let rt = recreate_res.text().await.unwrap_or_default();
        return Err(AppError::qdrant(format!(
            "recreate collection '{}' failed {rs}: {rt}",
            collection
        )));
    }

    Err(AppError::qdrant(format!(
        "create collection '{}' failed {}: {text}",
        collection, status
    )))
}

/// Ensure the collection exists. Uses the legacy fixed dim (1536) for the
/// default single collection. For multi-embedder flow prefer `create_collection`
/// with the detected dimension.
pub async fn ensure_collection(cfg: &QdrantConfig) -> Result<()> {
    create_collection(cfg, &cfg.collection, crate::ingest::TARGET_VECTOR_DIM).await
}

/// Run a vector search against a specific collection.
pub async fn search_in_collection(
    cfg: &QdrantConfig,
    collection: &str,
    vector: &[f32],
    limit: u32,
    tag_filter: Option<&str>,
) -> Result<Vec<Chunk>> {
    let url = format!("{}/collections/{}/points/search", base(cfg)?, collection);
    let mut body = json!({
        "vector": vector,
        "limit": limit,
        "with_payload": true,
        "with_vector": false,
    });
    if let Some(tag) = tag_filter {
        if !tag.is_empty() {
            ensure_text_index(cfg, collection, "tag").await?;
            body["filter"] = json!({
                "must": [
                    { "key": "tag", "match": { "text": tag } }
                ]
            });
        }
    }
    let res = http()
        .post(url)
        .header("api-key", api_key(cfg))
        .json(&body)
        .send()
        .await?;
    if !res.status().is_success() {
        let s = res.status();
        let txt = res.text().await.unwrap_or_default();
        return Err(AppError::qdrant(format!("search failed {s}: {txt}")));
    }
    let body = res
        .text()
        .await
        .map_err(|e| AppError::qdrant(format!("search body read failed: {e}")))?;
    let v: Value = serde_json::from_str(&body).map_err(|e| {
        let preview: String = body.chars().take(200).collect();
        AppError::qdrant(format!("search body not JSON: {e} — preview: {preview}"))
    })?;
    let chunks = parse_chunk_array(v.get("result"));
    Ok(chunks)
}

/// Legacy single-collection search (used by pipeline retrieval).
pub async fn search(
    cfg: &QdrantConfig,
    vector: &[f32],
    limit: u32,
    tag_filter: Option<&str>,
) -> Result<Vec<Chunk>> {
    if cfg.collection == "all" {
        let cols = list_collections(cfg).await?;
        let mut all_chunks = Vec::new();
        for col in cols {
            if let Ok(chunks) =
                search_in_collection(cfg, &col.name, vector, limit, tag_filter).await
            {
                all_chunks.extend(chunks);
            }
        }
        all_chunks.sort_by(|a, b| {
            let sa = a.score.unwrap_or(-1.0);
            let sb = b.score.unwrap_or(-1.0);
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });
        if all_chunks.len() > limit as usize {
            all_chunks.truncate(limit as usize);
        }
        Ok(all_chunks)
    } else {
        search_in_collection(cfg, &cfg.collection, vector, limit, tag_filter).await
    }
}

// ─── Collection listing ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionInfo {
    pub name: String,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub vectors_count: u64,
}

pub async fn list_collections(cfg: &QdrantConfig) -> Result<Vec<CollectionInfo>> {
    let url = format!("{}/collections", base(cfg)?);
    let res = http()
        .get(&url)
        .header("api-key", api_key(cfg))
        .send()
        .await?;
    if !res.status().is_success() {
        let s = res.status();
        let body = res.text().await.unwrap_or_default();
        return Err(AppError::qdrant(format!(
            "list collections failed {s}: {body}"
        )));
    }
    let v: Value = res
        .json()
        .await
        .map_err(|e| AppError::qdrant(format!("list collections JSON parse failed: {e}")))?;
    let arr = v
        .get("result")
        .and_then(|r| r.get("collections"))
        .and_then(|r| r.as_array())
        .cloned()
        .unwrap_or_default();
    let mut cols = vec![];
    for entry in arr {
        let name = entry
            .get("name")
            .and_then(|n| n.as_str())
            .unwrap_or("")
            .to_string();
        cols.push(CollectionInfo {
            name,
            status: String::new(),
            vectors_count: 0,
        });
    }
    Ok(cols)
}

// ─── Snapshot helpers ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct SnapshotList {
    pub result: Vec<SnapshotInfo>,
    #[serde(default)]
    pub time: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotInfo {
    pub name: String,
    #[serde(rename = "creation_time")]
    pub creation_time: Option<String>,
    pub size: u64,
}

pub async fn list_snapshots(cfg: &QdrantConfig, collection: &str) -> Result<Vec<SnapshotInfo>> {
    let url = format!("{}/collections/{}/snapshots", base(cfg)?, collection);
    let res = http()
        .get(&url)
        .header("api-key", api_key(cfg))
        .send()
        .await?;
    if !res.status().is_success() {
        let s = res.status();
        let body = res.text().await.unwrap_or_default();
        return Err(AppError::qdrant(format!(
            "list snapshots failed {s}: {body}"
        )));
    }
    let v: Value = res
        .json()
        .await
        .map_err(|e| AppError::qdrant(format!("list snapshots JSON parse failed: {e}")))?;
    let arr = v
        .get("result")
        .and_then(|r| r.as_array())
        .cloned()
        .unwrap_or_default();
    let mut snaps = vec![];
    for entry in arr {
        snaps.push(
            serde_json::from_value(entry).unwrap_or_else(|_| SnapshotInfo {
                name: String::new(),
                creation_time: None,
                size: 0,
            }),
        );
    }
    Ok(snaps)
}

pub async fn create_snapshot(cfg: &QdrantConfig, collection: &str) -> Result<SnapshotInfo> {
    let url = format!(
        "{}/collections/{}/snapshots?wait=true",
        base(cfg)?,
        collection
    );
    let res = http()
        .post(&url)
        .header("api-key", api_key(cfg))
        .send()
        .await?;
    if !res.status().is_success() && res.status().as_u16() != 409 {
        let s = res.status();
        let body = res.text().await.unwrap_or_default();
        return Err(AppError::qdrant(format!(
            "create snapshot failed {s}: {body}"
        )));
    }
    #[derive(Deserialize)]
    struct SnapResponse {
        result: SnapshotInfo,
    }
    let v: SnapResponse = res
        .json()
        .await
        .map_err(|e| AppError::qdrant(format!("create snapshot JSON parse failed: {e}")))?;
    Ok(v.result)
}

pub async fn download_snapshot(
    cfg: &QdrantConfig,
    collection: &str,
    snapshot_name: &str,
    local_path: &std::path::Path,
) -> Result<std::path::PathBuf> {
    use tokio::io::AsyncWriteExt;
    let url = format!(
        "{}/collections/{}/snapshots/{}",
        base(cfg)?,
        collection,
        snapshot_name
    );
    let res = http()
        .get(&url)
        .header("api-key", api_key(cfg))
        .send()
        .await?;
    if !res.status().is_success() {
        let s = res.status();
        let body = res.text().await.unwrap_or_default();
        return Err(AppError::qdrant(format!(
            "download snapshot failed {s}: {body}"
        )));
    }
    let bytes = res
        .bytes()
        .await
        .map_err(|e| AppError::qdrant(format!("read snapshot bytes failed: {e}")))?;
    let mut file = tokio::io::BufWriter::new(
        tokio::fs::File::create(local_path)
            .await
            .map_err(|e| AppError::qdrant(format!("create snapshot file failed: {e}")))?,
    );
    file.write_all(&bytes)
        .await
        .map_err(|e| AppError::qdrant(format!("write snapshot file failed: {e}")))?;
    file.flush()
        .await
        .map_err(|e| AppError::qdrant(format!("flush snapshot file failed: {e}")))?;
    Ok(local_path.to_path_buf())
}

pub async fn restore_snapshot(
    cfg: &QdrantConfig,
    collection: &str,
    snapshot_path: &str,
) -> Result<()> {
    let base_url = base(cfg)?;
    let url = format!(
        "{}/collections/{}/snapshots/recover?wait=true",
        base_url, collection
    );
    let location = if snapshot_path.starts_with("http://")
        || snapshot_path.starts_with("https://")
        || snapshot_path.starts_with("file://")
    {
        snapshot_path.to_string()
    } else {
        format!(
            "{}/collections/{}/snapshots/{}",
            base_url, collection, snapshot_path
        )
    };
    let mut body = json!({
        "location": location,
        "priority": "snapshot",
    });
    let key = api_key(cfg);
    if !key.trim().is_empty() {
        body["api_key"] = json!(key);
    }
    let res = http()
        .put(&url)
        .header("api-key", api_key(cfg))
        .json(&body)
        .send()
        .await?;
    if !res.status().is_success() {
        let s = res.status();
        let body = res.text().await.unwrap_or_default();
        return Err(AppError::qdrant(format!(
            "restore snapshot failed {s}: {body}"
        )));
    }
    Ok(())
}

pub async fn upload_snapshot(
    cfg: &QdrantConfig,
    collection: &str,
    snapshot_path: &std::path::Path,
) -> Result<()> {
    use reqwest::multipart;
    use tokio_util::io::ReaderStream;
    let url = format!(
        "{}/collections/{}/snapshots/upload?wait=true&priority=snapshot",
        base(cfg)?,
        collection
    );
    let file_name = snapshot_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("snapshot.snapshot")
        .to_string();

    let file = tokio::fs::File::open(snapshot_path)
        .await
        .map_err(|e| AppError::qdrant(format!("failed to open snapshot file for upload: {e}")))?;
    let file_size = file
        .metadata()
        .await
        .map_err(|e| AppError::qdrant(format!("failed to stat snapshot file for upload: {e}")))?
        .len();
    if file_size == 0 {
        return Err(AppError::qdrant("snapshot file is empty"));
    }
    let stream = ReaderStream::new(file);
    let body = reqwest::Body::wrap_stream(stream);

    let part = multipart::Part::stream_with_length(body, file_size)
        .file_name(file_name)
        .mime_str("application/octet-stream")
        .map_err(|e| AppError::qdrant(format!("failed to prepare snapshot upload: {e}")))?;

    let form = multipart::Form::new().part("snapshot", part);

    let res = http()
        .post(&url)
        .header("api-key", api_key(cfg))
        .multipart(form)
        .send()
        .await?;

    if !res.status().is_success() {
        let s = res.status();
        let body = res.text().await.unwrap_or_default();
        return Err(AppError::qdrant(format!(
            "snapshot upload/restore failed {s}: {body}"
        )));
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize)]
pub struct CollectionSnapshotResult {
    pub collection: String,
    pub snapshot_name: String,
    pub size: u64,
}

pub async fn create_all_snapshots(cfg: &QdrantConfig) -> Result<Vec<CollectionSnapshotResult>> {
    let cols = list_collections(cfg).await?;
    let mut results = vec![];
    for col in &cols {
        match create_snapshot(cfg, &col.name).await {
            Ok(info) => results.push(CollectionSnapshotResult {
                collection: col.name.clone(),
                snapshot_name: info.name,
                size: info.size,
            }),
            Err(e) => {
                results.push(CollectionSnapshotResult {
                    collection: col.name.clone(),
                    snapshot_name: format!("ERROR: {e}"),
                    size: 0,
                });
            }
        }
    }
    Ok(results)
}

pub async fn download_all_snapshots(
    cfg: &QdrantConfig,
    local_dir: &std::path::Path,
) -> Result<Vec<std::path::PathBuf>> {
    use tokio::io::AsyncWriteExt;
    let cols = list_collections(cfg).await?;
    tokio::fs::create_dir_all(local_dir)
        .await
        .map_err(|e| AppError::qdrant(format!("create download dir failed: {e}")))?;
    let mut paths = vec![];
    for col in &cols {
        let snaps = list_snapshots(cfg, &col.name).await.unwrap_or_default();
        for snap in &snaps {
            let url = format!(
                "{}/collections/{}/snapshots/{}",
                base(cfg)?,
                col.name,
                snap.name
            );
            let res = http()
                .get(&url)
                .header("api-key", api_key(cfg))
                .send()
                .await?;
            if !res.status().is_success() {
                continue;
            }
            let bytes = res
                .bytes()
                .await
                .map_err(|e| AppError::qdrant(format!("read snapshot bytes failed: {e}")))?;
            let file_path = local_dir.join(format!("{}__{}", col.name, snap.name));
            let mut file = tokio::io::BufWriter::new(
                tokio::fs::File::create(&file_path)
                    .await
                    .map_err(|e| AppError::qdrant(format!("create snapshot file failed: {e}")))?,
            );
            file.write_all(&bytes)
                .await
                .map_err(|e| AppError::qdrant(format!("write snapshot file failed: {e}")))?;
            file.flush()
                .await
                .map_err(|e| AppError::qdrant(format!("flush snapshot file failed: {e}")))?;
            paths.push(file_path);
        }
    }
    Ok(paths)
}

pub async fn scroll_filtered_in_collection(
    cfg: &QdrantConfig,
    collection: &str,
    file_path: &str,
    start_idx: i64,
    end_idx: i64,
) -> Result<Vec<Chunk>> {
    let url = format!("{}/collections/{}/points/scroll", base(cfg)?, collection);

    // Qdrant filter: file_path == value AND chunk_index in [start_idx, end_idx]
    let filter = json!({
        "must": [
            {
                "key": "file_path",
                "match": { "value": file_path }
            },
            {
                "key": "chunk_index",
                "range": {
                    "gte": start_idx,
                    "lte": end_idx
                }
            }
        ]
    });

    let body = json!({
        "limit": 100, // Safe upper limit for windows
        "with_payload": true,
        "with_vector": false,
        "filter": filter
    });

    let res = http()
        .post(url)
        .header("api-key", api_key(cfg))
        .json(&body)
        .send()
        .await?;

    if !res.status().is_success() {
        let s = res.status();
        let text = res.text().await.unwrap_or_default();
        return Err(AppError::qdrant(format!(
            "scroll_filtered failed {s}: {text}"
        )));
    }

    let body_text = res
        .text()
        .await
        .map_err(|e| AppError::qdrant(format!("scroll_filtered body read failed: {e}")))?;

    let v: Value = serde_json::from_str(&body_text).map_err(|e| {
        let preview: String = body_text.chars().take(200).collect();
        AppError::qdrant(format!(
            "scroll_filtered body not JSON: {e} — preview: {preview}"
        ))
    })?;

    let result = v
        .get("result")
        .ok_or_else(|| AppError::qdrant("response missing result"))?;

    let chunks = parse_chunk_array(result.get("points"));
    Ok(chunks)
}

pub async fn scroll_filtered(
    cfg: &QdrantConfig,
    file_path: &str,
    start_idx: i64,
    end_idx: i64,
) -> Result<Vec<Chunk>> {
    if cfg.collection == "all" {
        let cols = list_collections(cfg).await?;
        let mut all_chunks = Vec::new();
        for col in &cols {
            if let Ok(chunks) =
                scroll_filtered_in_collection(cfg, &col.name, file_path, start_idx, end_idx).await
            {
                all_chunks.extend(chunks);
            }
        }
        all_chunks.sort_by_key(|c| c.chunk_index);
        all_chunks.dedup_by(|a, b| a.id == b.id);
        Ok(all_chunks)
    } else {
        scroll_filtered_in_collection(cfg, &cfg.collection, file_path, start_idx, end_idx).await
    }
}
