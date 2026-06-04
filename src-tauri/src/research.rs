//! Pluggable web-research module for the robot capture pipeline.
//!
//! Given a query built from a captured object's OCR text + label guess, this
//! searches the web via a configurable provider (Brave / SerpAPI / Google CSE),
//! filters results by a domain allowlist, fetches readable page text, applies a
//! dangerous-topic blocklist, and returns a cited research packet ready to be
//! embedded into Qdrant and handed to the teacher model.

use crate::config::WebResearchConfig;
use crate::error::{AppError, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;

/// One cited source pulled during research.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Citation {
    pub url: String,
    pub title: String,
    pub snippet: String,
    /// Truncated readable text extracted from the page (empty if fetch failed).
    pub extract: String,
    pub fetched_at: String,
}

/// The assembled research packet for one captured object.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResearchPacket {
    pub query: String,
    pub citations: Vec<Citation>,
    /// Markdown body combining all citation extracts; this is what gets embedded.
    pub markdown: String,
    pub blocked: bool,
    pub block_reason: Option<String>,
}

const DANGEROUS_TERMS: &[&str] = &[
    "weapon", "firearm", "explosive", "bomb", "ammunition", "detonator",
    "improvised explosive", "nerve agent", "bioweapon", "meth synthesis",
    "how to kill", "self-harm", "suicide method",
];

fn http() -> Client {
    Client::builder()
        .timeout(Duration::from_secs(20))
        .user_agent("fine-tune-robot-research/0.1")
        .build()
        .unwrap_or_else(|_| Client::new())
}

fn host_of(url: &str) -> Option<String> {
    let after = url.split("://").nth(1).unwrap_or(url);
    let host = after.split('/').next().unwrap_or("");
    if host.is_empty() {
        None
    } else {
        Some(host.trim_start_matches("www.").to_lowercase())
    }
}

fn allowed(url: &str, allowlist: &[String]) -> bool {
    if allowlist.is_empty() {
        return true;
    }
    match host_of(url) {
        Some(h) => allowlist
            .iter()
            .any(|d| h == d.to_lowercase() || h.ends_with(&format!(".{}", d.to_lowercase()))),
        None => false,
    }
}

fn dangerous(text: &str) -> Option<String> {
    let lower = text.to_lowercase();
    DANGEROUS_TERMS
        .iter()
        .find(|t| lower.contains(*t))
        .map(|t| format!("matched blocked term '{}'", t))
}

/// Naive HTML → text: strip tags and collapse whitespace. Good enough to give
/// the teacher model usable context without pulling in a heavy HTML crate.
fn html_to_text(html: &str) -> String {
    let mut out = String::with_capacity(html.len() / 2);
    let mut in_tag = false;
    let lower = html.to_lowercase();
    // crude script/style stripping
    let cleaned = if lower.contains("<script") || lower.contains("<style") {
        let mut s = html.to_string();
        for tag in ["script", "style"] {
            loop {
                let l = s.to_lowercase();
                if let (Some(start), Some(end)) =
                    (l.find(&format!("<{}", tag)), l.find(&format!("</{}>", tag)))
                {
                    if end > start {
                        s.replace_range(start..end + tag.len() + 3, " ");
                        continue;
                    }
                }
                break;
            }
        }
        s
    } else {
        html.to_string()
    };
    for c in cleaned.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    // collapse whitespace
    let mut collapsed = String::with_capacity(out.len());
    let mut last_ws = false;
    for c in out.chars() {
        if c.is_whitespace() {
            if !last_ws {
                collapsed.push(' ');
                last_ws = true;
            }
        } else {
            collapsed.push(c);
            last_ws = false;
        }
    }
    collapsed.trim().to_string()
}

#[derive(Debug, Clone)]
struct SearchHit {
    url: String,
    title: String,
    snippet: String,
}

async fn search(cfg: &WebResearchConfig, query: &str) -> Result<Vec<SearchHit>> {
    if cfg.api_key.trim().is_empty() {
        return Err(AppError::other(
            "web research: no API key configured (set it in the Robotics widget)",
        ));
    }
    let n = cfg.max_results.clamp(1, 10) as usize;
    let c = http();
    match cfg.provider.as_str() {
        "brave" => {
            let res = c
                .get("https://api.search.brave.com/res/v1/web/search")
                .header("X-Subscription-Token", &cfg.api_key)
                .header("Accept", "application/json")
                .query(&[("q", query), ("count", &n.to_string())])
                .send()
                .await
                .map_err(AppError::Http)?;
            let v: Value = res.json().await.map_err(AppError::Http)?;
            let hits = v["web"]["results"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .take(n)
                        .map(|r| SearchHit {
                            url: r["url"].as_str().unwrap_or("").to_string(),
                            title: r["title"].as_str().unwrap_or("").to_string(),
                            snippet: r["description"].as_str().unwrap_or("").to_string(),
                        })
                        .collect()
                })
                .unwrap_or_default();
            Ok(hits)
        }
        "serpapi" => {
            let res = c
                .get("https://serpapi.com/search.json")
                .query(&[
                    ("q", query),
                    ("engine", "google"),
                    ("num", &n.to_string()),
                    ("api_key", &cfg.api_key),
                ])
                .send()
                .await
                .map_err(AppError::Http)?;
            let v: Value = res.json().await.map_err(AppError::Http)?;
            let hits = v["organic_results"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .take(n)
                        .map(|r| SearchHit {
                            url: r["link"].as_str().unwrap_or("").to_string(),
                            title: r["title"].as_str().unwrap_or("").to_string(),
                            snippet: r["snippet"].as_str().unwrap_or("").to_string(),
                        })
                        .collect()
                })
                .unwrap_or_default();
            Ok(hits)
        }
        "google_cse" => {
            let cse = cfg.cse_id.clone().unwrap_or_default();
            if cse.trim().is_empty() {
                return Err(AppError::other(
                    "google_cse provider requires a cseId in the Robotics widget",
                ));
            }
            let res = c
                .get("https://www.googleapis.com/customsearch/v1")
                .query(&[
                    ("key", cfg.api_key.as_str()),
                    ("cx", cse.as_str()),
                    ("q", query),
                    ("num", &n.to_string()),
                ])
                .send()
                .await
                .map_err(AppError::Http)?;
            let v: Value = res.json().await.map_err(AppError::Http)?;
            let hits = v["items"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .take(n)
                        .map(|r| SearchHit {
                            url: r["link"].as_str().unwrap_or("").to_string(),
                            title: r["title"].as_str().unwrap_or("").to_string(),
                            snippet: r["snippet"].as_str().unwrap_or("").to_string(),
                        })
                        .collect()
                })
                .unwrap_or_default();
            Ok(hits)
        }
        other => Err(AppError::other(format!(
            "unknown web research provider '{}'",
            other
        ))),
    }
}

/// Research a captured object online and return a cited packet.
/// `now_iso` is passed in so callers control the clock (and tests stay
/// deterministic).
pub async fn research_object(
    cfg: &WebResearchConfig,
    query: &str,
    now_iso: &str,
) -> Result<ResearchPacket> {
    let query = query.trim();
    if query.is_empty() {
        return Err(AppError::other("research: empty query"));
    }

    if cfg.block_dangerous_topics {
        if let Some(reason) = dangerous(query) {
            return Ok(ResearchPacket {
                query: query.to_string(),
                citations: vec![],
                markdown: String::new(),
                blocked: true,
                block_reason: Some(reason),
            });
        }
    }

    let hits = search(cfg, query).await?;
    let c = http();
    let mut citations = Vec::new();

    for hit in hits {
        if hit.url.is_empty() || !allowed(&hit.url, &cfg.domain_allowlist) {
            continue;
        }
        let extract = match c.get(&hit.url).send().await {
            Ok(resp) => match resp.text().await {
                Ok(body) => {
                    let text = html_to_text(&body);
                    text.chars().take(4000).collect::<String>()
                }
                Err(_) => String::new(),
            },
            Err(_) => String::new(),
        };

        if cfg.block_dangerous_topics {
            let combined = format!("{} {} {}", hit.title, hit.snippet, extract);
            if dangerous(&combined).is_some() {
                continue; // silently drop a dangerous source
            }
        }

        citations.push(Citation {
            url: hit.url,
            title: hit.title,
            snippet: hit.snippet,
            extract,
            fetched_at: now_iso.to_string(),
        });
    }

    if citations.is_empty() {
        return Err(AppError::other(
            "research: no usable sources (all filtered by allowlist/blocklist or fetch failed)",
        ));
    }

    let mut markdown = format!("# Research packet: {}\n\n", query);
    for (i, c) in citations.iter().enumerate() {
        markdown.push_str(&format!(
            "## Source {} — {}\nURL: {}\n\n{}\n\n{}\n\n---\n\n",
            i + 1,
            c.title,
            c.url,
            c.snippet,
            c.extract
        ));
    }

    Ok(ResearchPacket {
        query: query.to_string(),
        citations,
        markdown,
        blocked: false,
        block_reason: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allowlist_matches_subdomains() {
        let allow = vec!["wikipedia.org".to_string()];
        assert!(allowed("https://en.wikipedia.org/wiki/Cat", &allow));
        assert!(allowed("https://wikipedia.org/x", &allow));
        assert!(!allowed("https://evil.com/x", &allow));
    }

    #[test]
    fn empty_allowlist_allows_all() {
        assert!(allowed("https://anything.com", &[]));
    }

    #[test]
    fn dangerous_terms_detected() {
        assert!(dangerous("how to build a bomb").is_some());
        assert!(dangerous("a friendly orange cat").is_none());
    }

    #[test]
    fn html_stripped() {
        let t = html_to_text("<html><body><p>Hello   <b>world</b></p></body></html>");
        assert_eq!(t, "Hello world");
    }
}
