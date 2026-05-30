// Lightweight wrapper around the Hugging Face Hub REST API.
//
// We only need two things for the UI:
//   1. Resolve the user's username from the stored token (so the Dataset Repo ID
//      placeholder can read "<username>/...").
//   2. List the user's dataset repos so the "Resume from" field can become a
//      dropdown of already-existing datasets they can pull progress from.
//
// We hit https://huggingface.co/api directly with reqwest — no Python needed.

use crate::error::{AppError, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

fn http() -> Client {
    Client::builder()
        .timeout(Duration::from_secs(20))
        .user_agent("fine-tune-tauri/0.1")
        .build()
        .expect("reqwest client")
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct HfWhoami {
    /// HF username — what the user types as the org/user portion of a repo id.
    pub name: String,
    /// Display name, if HF returned one.
    #[serde(default)]
    pub fullname: String,
    /// Avatar URL when available.
    #[serde(default)]
    pub avatar_url: String,
    /// Whether the token has write scope. We can use this to surface a warning
    /// if the user tries to push without write.
    #[serde(default)]
    pub can_pay: bool,
    /// `read` | `write` | `fineGrained` — surfaced as-is.
    #[serde(default)]
    pub token_role: String,
}

/// Hit `GET /api/whoami-v2` and reduce it to the few fields the UI cares about.
pub async fn whoami(token: &str) -> Result<HfWhoami> {
    if token.trim().is_empty() {
        return Err(AppError::other("no Hugging Face token configured"));
    }
    let res = http()
        .get("https://huggingface.co/api/whoami-v2")
        .bearer_auth(token)
        .send()
        .await
        .map_err(AppError::Http)?;
    if !res.status().is_success() {
        let s = res.status();
        let body = res.text().await.unwrap_or_default();
        return Err(AppError::other(format!("HF whoami {s}: {body}")));
    }
    let v: serde_json::Value = res.json().await.map_err(AppError::Http)?;
    let name = v
        .get("name")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    let fullname = v
        .get("fullname")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    let avatar_url = v
        .get("avatarUrl")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    // The auth block tells us the access token's scope.
    let token_role = v
        .pointer("/auth/accessToken/role")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    Ok(HfWhoami {
        name,
        fullname,
        avatar_url,
        can_pay: false,
        token_role,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HfDatasetRepo {
    pub id: String,           // e.g. "zrald/ge-reviewer-qa"
    #[serde(default)]
    pub private: bool,
    #[serde(default)]
    pub last_modified: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HfModelRepo {
    pub id: String, // e.g. "zrald/ge-reviewer-qwen-merged"
    #[serde(default)]
    pub private: bool,
    #[serde(default)]
    pub last_modified: String,
}

/// List dataset repos owned by `username` (and by orgs they belong to when
/// `include_orgs` is true and we can fetch them cheaply). The HF API endpoint
/// `GET /api/datasets?author=<name>` returns the list for a given author with
/// pagination via `limit` / `cursor`. We keep it simple and pull up to 200.
pub async fn list_user_datasets(token: &str, username: &str) -> Result<Vec<HfDatasetRepo>> {
    if username.trim().is_empty() {
        return Ok(vec![]);
    }
    let mut out = vec![];

    // Authors we want to query: the user themself, plus the orgs the whoami
    // response listed (best-effort — if the token can't see them we just skip).
    let mut authors: Vec<String> = vec![username.to_string()];
    if !token.trim().is_empty() {
        if let Ok(orgs) = list_user_orgs(token).await {
            for o in orgs {
                if !authors.iter().any(|a| a.eq_ignore_ascii_case(&o)) {
                    authors.push(o);
                }
            }
        }
    }

    for author in authors {
        // HF usernames + org names are restricted to [A-Za-z0-9-_], so plain
        // concatenation is safe — but be defensive and strip anything else.
        let safe_author: String = author
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
            .collect();
        if safe_author.is_empty() {
            continue;
        }
        let url = format!(
            "https://huggingface.co/api/datasets?author={}&limit=200&full=false",
            safe_author
        );
        let mut req = http().get(&url);
        if !token.trim().is_empty() {
            req = req.bearer_auth(token);
        }
        let res = match req.send().await {
            Ok(r) => r,
            Err(_) => continue, // skip this author on transient error
        };
        if !res.status().is_success() {
            continue;
        }
        let v: serde_json::Value = match res.json().await {
            Ok(v) => v,
            Err(_) => continue,
        };
        if let Some(arr) = v.as_array() {
            for item in arr {
                let id = item.get("id").and_then(|x| x.as_str()).unwrap_or("").to_string();
                if id.is_empty() {
                    continue;
                }
                let private = item.get("private").and_then(|x| x.as_bool()).unwrap_or(false);
                let last_modified = item
                    .get("lastModified")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string();
                out.push(HfDatasetRepo {
                    id,
                    private,
                    last_modified,
                });
            }
        }
    }

    // Most-recent first — the dropdown UX makes more sense that way.
    out.sort_by(|a, b| b.last_modified.cmp(&a.last_modified));
    // Dedup by id (in case the user shows up via multiple authors).
    out.dedup_by(|a, b| a.id == b.id);
    Ok(out)
}

/// List model repos owned by `username` and visible orgs. This feeds the
/// Student Model picker so previously merged full-weight repos can be reused
/// as the base for the next training run.
pub async fn list_user_models(token: &str, username: &str) -> Result<Vec<HfModelRepo>> {
    if username.trim().is_empty() {
        return Ok(vec![]);
    }
    let mut out = vec![];

    let mut authors: Vec<String> = vec![username.to_string()];
    if !token.trim().is_empty() {
        if let Ok(orgs) = list_user_orgs(token).await {
            for o in orgs {
                if !authors.iter().any(|a| a.eq_ignore_ascii_case(&o)) {
                    authors.push(o);
                }
            }
        }
    }

    for author in authors {
        let safe_author: String = author
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
            .collect();
        if safe_author.is_empty() {
            continue;
        }
        let url = format!(
            "https://huggingface.co/api/models?author={}&limit=200&full=false",
            safe_author
        );
        let mut req = http().get(&url);
        if !token.trim().is_empty() {
            req = req.bearer_auth(token);
        }
        let res = match req.send().await {
            Ok(r) => r,
            Err(_) => continue,
        };
        if !res.status().is_success() {
            continue;
        }
        let v: serde_json::Value = match res.json().await {
            Ok(v) => v,
            Err(_) => continue,
        };
        if let Some(arr) = v.as_array() {
            for item in arr {
                let id = item
                    .get("id")
                    .or_else(|| item.get("modelId"))
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string();
                if id.is_empty() {
                    continue;
                }
                let private = item.get("private").and_then(|x| x.as_bool()).unwrap_or(false);
                let last_modified = item
                    .get("lastModified")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string();
                out.push(HfModelRepo {
                    id,
                    private,
                    last_modified,
                });
            }
        }
    }

    out.sort_by(|a, b| b.last_modified.cmp(&a.last_modified));
    out.dedup_by(|a, b| a.id == b.id);
    Ok(out)
}

/// Returns the names of orgs the authenticated user belongs to. Best-effort —
/// empty vec if anything goes wrong.
async fn list_user_orgs(token: &str) -> Result<Vec<String>> {
    let res = http()
        .get("https://huggingface.co/api/whoami-v2")
        .bearer_auth(token)
        .send()
        .await
        .map_err(AppError::Http)?;
    if !res.status().is_success() {
        return Ok(vec![]);
    }
    let v: serde_json::Value = res.json().await.map_err(AppError::Http)?;
    let mut orgs = vec![];
    if let Some(arr) = v.get("orgs").and_then(|x| x.as_array()) {
        for o in arr {
            if let Some(n) = o.get("name").and_then(|x| x.as_str()) {
                if !n.is_empty() {
                    orgs.push(n.to_string());
                }
            }
        }
    }
Ok(orgs)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DatasetSplitInfo {
    pub num_examples: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DatasetInfo {
    pub id: String,
    pub private: bool,
    pub tags: Vec<String>,
    pub splits: std::collections::HashMap<String, DatasetSplitInfo>,
}

pub async fn get_dataset_info(token: &str, repo_id: &str) -> Result<DatasetInfo> {
    let url = format!("https://huggingface.co/api/datasets/{}", repo_id);
    let res = http()
        .get(&url)
        .bearer_auth(token)
        .send()
        .await
        .map_err(AppError::Http)?;
    if !res.status().is_success() {
        let s = res.status();
        let body = res.text().await.unwrap_or_default();
        return Err(AppError::other(format!("HF dataset info {s}: {body}")));
    }
    let v: serde_json::Value = res.json().await.map_err(AppError::Http)?;
    
    let splits = v.get("splits")
        .and_then(|s| s.as_object())
        .map(|obj| {
            let mut map = std::collections::HashMap::new();
            for (k, val) in obj {
                let num_examples = val.get("numExamples")
                    .or_else(|| val.get("num_examples"))
                    .and_then(|x| x.as_u64())
                    .unwrap_or(0) as usize;
                map.insert(k.clone(), DatasetSplitInfo { num_examples });
            }
            map
        })
        .unwrap_or_default();
    
    let id = v.get("id")
        .or_else(|| v.get("repo_id"))
        .and_then(|x| x.as_str())
        .unwrap_or(repo_id)
        .to_string();
    
    let private = v.get("private").and_then(|x| x.as_bool()).unwrap_or(false);
    
    let tags = v.get("tags")
        .and_then(|x| x.as_array())
        .map(|arr| arr.iter().filter_map(|t| t.as_str().map(String::from)).collect())
        .unwrap_or_default();
    
    Ok(DatasetInfo { id, private, tags, splits })
}
