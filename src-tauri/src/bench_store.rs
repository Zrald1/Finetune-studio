//! Persistent benchmark history.
//!
//! Every benchmark run — teacher, student, or a Deploy-page serving — is
//! recorded here with a snapshot of the configuration it was measured under.
//! Keeping the config alongside the numbers is the whole point: it is what
//! lets the UI answer "how much did switching Standard → Optimized actually
//! buy me?" instead of showing a single unlabelled set of figures.
//!
//! Records live in `benchmarks.json` next to `config.json`, so they survive
//! app restarts and a droplet being destroyed and recreated.

use crate::error::{AppError, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tokio::fs;

/// What was benchmarked. Kept as a string so new kinds do not break old files.
pub const KIND_TEACHER: &str = "teacher";
pub const KIND_STUDENT: &str = "student";
pub const KIND_DEPLOYMENT: &str = "deployment";

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct BenchConfig {
    pub serving_profile: Option<String>,
    pub reasoning_effort: Option<String>,
    pub reasoning_parser: Option<String>,
    pub dtype: Option<String>,
    pub max_model_len: Option<u32>,
    pub gpu_memory_utilization: Option<f32>,
    pub max_num_seqs: Option<u32>,
    pub max_num_batched_tokens: Option<u32>,
    pub tensor_parallel: Option<u32>,
    pub quantization: Option<String>,
    pub kv_cache_dtype: Option<String>,
    pub prefix_caching: Option<bool>,
    /// Free-form extras (training method, LoRA rank, GPU name, …).
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct BenchMetrics {
    pub ttft_ms: Option<f64>,
    pub median_ttft_ms: Option<f64>,
    pub p95_ttft_ms: Option<f64>,
    pub tpot_ms: Option<f64>,
    pub itl_ms: Option<f64>,
    pub e2el_ms: Option<f64>,
    pub output_tokens_per_s: Option<f64>,
    pub total_tokens_per_s: Option<f64>,
    pub request_throughput: Option<f64>,
    /// Student runs report accuracy alongside generation speed.
    pub accuracy: Option<f64>,
    pub samples: Option<u32>,
    pub concurrency: Option<u32>,
    pub total_output_tokens: Option<u32>,
}

impl BenchMetrics {
    /// A record with nothing measurable in it is not worth keeping.
    pub fn is_meaningful(&self) -> bool {
        self.ttft_ms.is_some()
            || self.output_tokens_per_s.is_some()
            || self.accuracy.is_some()
            || self.request_throughput.is_some()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchRecord {
    pub id: String,
    /// One of `KIND_*`.
    pub kind: String,
    /// User-facing name shown in the history table.
    pub label: String,
    pub model: String,
    pub endpoint: String,
    pub config: BenchConfig,
    pub metrics: BenchMetrics,
    pub captured_at: String,
}

fn store_path() -> Result<PathBuf> {
    Ok(crate::config::app_dir()?.join("benchmarks.json"))
}

pub async fn load_all() -> Result<Vec<BenchRecord>> {
    let path = store_path()?;
    if !path.exists() {
        return Ok(Vec::new());
    }
    let text = fs::read_to_string(&path).await?;
    if text.trim().is_empty() {
        return Ok(Vec::new());
    }
    // A corrupt store must not take the app down — benchmarks are diagnostics,
    // not source of truth, so fall back to an empty history and let the next
    // save rewrite the file.
    Ok(serde_json::from_str(&text).unwrap_or_default())
}

async fn write_all(records: &[BenchRecord]) -> Result<()> {
    crate::config::ensure_dirs().await?;
    let text = serde_json::to_string_pretty(records)?;
    fs::write(store_path()?, text).await?;
    Ok(())
}

/// Append a record. `id` is assigned here if the caller left it blank.
pub async fn save(mut record: BenchRecord) -> Result<BenchRecord> {
    if !record.metrics.is_meaningful() {
        return Err(AppError::config(
            "refusing to store a benchmark with no metrics",
        ));
    }
    if record.id.trim().is_empty() {
        record.id = ulid::Ulid::new().to_string();
    }
    if record.captured_at.trim().is_empty() {
        record.captured_at = chrono::Local::now().to_rfc3339();
    }
    let mut all = load_all().await?;
    all.push(record.clone());
    write_all(&all).await?;
    Ok(record)
}

pub async fn delete(id: &str) -> Result<()> {
    let mut all = load_all().await?;
    let before = all.len();
    all.retain(|r| r.id != id);
    if all.len() == before {
        return Err(AppError::config(format!("no benchmark with id {id}")));
    }
    write_all(&all).await
}

pub async fn clear() -> Result<()> {
    write_all(&[]).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(kind: &str, label: &str, ttft: f64, tps: f64) -> BenchRecord {
        BenchRecord {
            id: String::new(),
            kind: kind.to_string(),
            label: label.to_string(),
            model: "Qwen/Qwen3.8-27B".to_string(),
            endpoint: "http://127.0.0.1:44319".to_string(),
            config: BenchConfig {
                serving_profile: Some("optimized".to_string()),
                ..Default::default()
            },
            metrics: BenchMetrics {
                ttft_ms: Some(ttft),
                output_tokens_per_s: Some(tps),
                ..Default::default()
            },
            captured_at: String::new(),
        }
    }

    #[test]
    fn rejects_a_record_with_no_metrics() {
        let mut r = record(KIND_TEACHER, "empty", 0.0, 0.0);
        r.metrics = BenchMetrics::default();
        assert!(!r.metrics.is_meaningful());
    }

    #[test]
    fn accepts_records_that_carry_any_metric() {
        let cases = [
            BenchMetrics { ttft_ms: Some(1.0), ..Default::default() },
            BenchMetrics { output_tokens_per_s: Some(1.0), ..Default::default() },
            BenchMetrics { accuracy: Some(1.0), ..Default::default() },
            BenchMetrics { request_throughput: Some(1.0), ..Default::default() },
        ];
        for m in cases {
            assert!(m.is_meaningful(), "should be storable: {m:?}");
        }
    }

    #[test]
    fn records_round_trip_through_json() {
        let r = record(KIND_TEACHER, "Standard", 1398.0, 47.5);
        let text = serde_json::to_string(&r).expect("serialize");
        let back: BenchRecord = serde_json::from_str(&text).expect("deserialize");
        assert_eq!(back.kind, KIND_TEACHER);
        assert_eq!(back.label, "Standard");
        assert_eq!(back.metrics.ttft_ms, Some(1398.0));
        assert_eq!(back.config.serving_profile.as_deref(), Some("optimized"));
    }

    #[test]
    fn legacy_records_without_new_fields_still_parse() {
        // Forward compatibility: a file written before a field existed must not
        // wipe the user's history.
        let legacy = r#"[
            {
              "id": "01ABC",
              "kind": "teacher",
              "label": "Old run",
              "model": "Qwen/Qwen3.8-27B",
              "endpoint": "http://127.0.0.1:8000",
              "config": { "servingProfile": "standard" },
              "metrics": { "ttftMs": 900.0 },
              "capturedAt": "2026-01-01T00:00:00Z"
            }
        ]"#;
        let parsed: Vec<BenchRecord> = serde_json::from_str(legacy).expect("must parse");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].metrics.ttft_ms, Some(900.0));
        assert_eq!(parsed[0].metrics.tpot_ms, None);
        assert!(parsed[0].config.notes.is_empty());
    }

    #[test]
    fn corrupt_store_falls_back_to_empty() {
        let parsed: Vec<BenchRecord> = serde_json::from_str("not json").unwrap_or_default();
        assert!(parsed.is_empty());
    }

    #[test]
    fn kinds_are_distinct_strings() {
        assert_ne!(KIND_TEACHER, KIND_STUDENT);
        assert_ne!(KIND_STUDENT, KIND_DEPLOYMENT);
        assert_ne!(KIND_TEACHER, KIND_DEPLOYMENT);
    }
}
