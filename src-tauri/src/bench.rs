//! Inference benchmarking for the deployed teacher.
//!
//! Metric definitions deliberately mirror vLLM's own `vllm bench serve` so the
//! numbers are comparable with the upstream tooling and with published
//! benchmarks:
//!
//! - **TTFT** — time from sending the request to the first streamed token.
//! - **TPOT** — `(e2el - ttft) / (output_tokens - 1)`. Decode-only per-token
//!   cost; for a single request this equals the mean inter-token latency.
//! - **ITL** — mean gap between successive streamed chunks. A chunk may carry
//!   more than one token, so this is an upper bound on true ITL.
//! - **E2EL** — wall time for one request, end to end.
//! - **Throughput** — output tokens over wall time for the whole run.
//!
//! TTFT cannot be observed from a non-streaming response, so every request sets
//! `stream: true` together with `stream_options.include_usage` to get exact
//! token counts back from the server instead of estimating them from text.

use crate::error::{AppError, Result};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Mixed prompt set: short answers, long answers and a reasoning-heavy item, so
/// the aggregate is not skewed by a single output length. Kept identical to the
/// set used by the Deploy page benchmark for comparability.
const BENCH_PROMPTS: &[&str] = &[
    "Write a Python function that reverses a linked list.",
    "Explain the difference between TCP and UDP in 2 sentences.",
    "What is 1337 * 42? Show your work step by step.",
    "Summarize the concept of gradient descent in one paragraph.",
    "List 3 advantages of transformer architecture over RNNs.",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchSample {
    pub index: u32,
    pub prompt: String,
    pub ttft_ms: f64,
    /// Time to the first *content* token. Differs from `ttft_ms` when the model
    /// emits a thinking block first, so the gap is the visible thinking cost.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttfc_ms: Option<f64>,
    pub e2el_ms: f64,
    pub prompt_tokens: u32,
    pub output_tokens: u32,
    pub tpot_ms: f64,
    pub output_tps: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchReport {
    pub endpoint: String,
    pub model: String,
    pub completed: u32,
    pub failed: u32,
    pub errors: Vec<String>,
    pub duration_s: f64,
    pub concurrency: u32,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    pub mean_ttft_ms: f64,
    pub median_ttft_ms: f64,
    pub p95_ttft_ms: f64,
    pub mean_tpot_ms: f64,
    pub mean_itl_ms: f64,
    pub mean_e2el_ms: f64,
    pub output_tokens_per_s: f64,
    pub total_tokens_per_s: f64,
    pub request_throughput: f64,
    pub total_output_tokens: u32,
    pub total_input_tokens: u32,
    pub samples: Vec<BenchSample>,
    pub captured_at: String,
}

pub struct BenchOptions {
    pub concurrency: u32,
    pub max_tokens: u32,
    pub reasoning_effort: Option<String>,
    pub timeout_s: u64,
}

impl Default for BenchOptions {
    fn default() -> Self {
        Self {
            concurrency: 1,
            max_tokens: 512,
            reasoning_effort: None,
            timeout_s: 300,
        }
    }
}

fn client(timeout_s: u64) -> Result<reqwest::Client> {
    Ok(reqwest::Client::builder()
        .user_agent("fine-tune-studio/0.1")
        .timeout(Duration::from_secs(timeout_s))
        // No redirect following: an endpoint that 301s is a misconfiguration,
        // and silently re-POSTing elsewhere would benchmark the wrong server.
        .redirect(reqwest::redirect::Policy::none())
        .build()?)
}

fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (pct / 100.0) * (sorted.len() as f64 - 1.0);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let weight = rank - lo as f64;
        sorted[lo] * (1.0 - weight) + sorted[hi] * weight
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

/// One streamed request. Returns `(ttft_ms, ttfc_ms, e2el_ms, prompt_tokens,
/// output_tokens, itl_ms)`.
async fn stream_one(
    http: &reqwest::Client,
    url: &str,
    model: &str,
    prompt: &str,
    max_tokens: u32,
    kwargs: Option<&serde_json::Value>,
) -> Result<(f64, Option<f64>, f64, u32, u32, Vec<f64>)> {
    let mut body = serde_json::json!({
        "model": model,
        "messages": [{ "role": "user", "content": prompt }],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": true,
        // Required for the final chunk to carry exact token counts.
        "stream_options": { "include_usage": true },
    });
    if let Some(kwargs) = kwargs {
        body["chat_template_kwargs"] = kwargs.clone();
    }

    let started = Instant::now();
    let res = http
        .post(url)
        .json(&body)
        .send()
        .await
        .map_err(|e| AppError::pipeline(format!("benchmark request failed: {e}")))?;

    if !res.status().is_success() {
        let status = res.status();
        let text = res.text().await.unwrap_or_default();
        return Err(AppError::pipeline(format!(
            "benchmark endpoint returned {status}: {text}"
        )));
    }

    let mut stream = res.bytes_stream();
    let mut pending = String::new();
    let mut ttft: Option<f64> = None;
    let mut ttfc: Option<f64> = None;
    let mut last_token_at: Option<Instant> = None;
    let mut itls: Vec<f64> = Vec::new();
    let mut prompt_tokens = 0u32;
    let mut output_tokens = 0u32;

    while let Some(chunk) = stream.next().await {
        let chunk =
            chunk.map_err(|e| AppError::pipeline(format!("benchmark stream error: {e}")))?;
        pending.push_str(&String::from_utf8_lossy(&chunk));

        // SSE events are newline-delimited; a chunk boundary can split a line,
        // so only consume complete lines and keep the remainder buffered.
        while let Some(newline) = pending.find('\n') {
            let line = pending[..newline].trim().to_string();
            pending.drain(..=newline);

            let Some(data) = line.strip_prefix("data:") else {
                continue;
            };
            let data = data.trim();
            if data.is_empty() || data == "[DONE]" {
                continue;
            }
            let Ok(event) = serde_json::from_str::<serde_json::Value>(data) else {
                continue;
            };

            let delta = event
                .get("choices")
                .and_then(|c| c.get(0))
                .and_then(|c| c.get("delta"));

            // Thinking tokens arrive in `reasoning_content` once a reasoning
            // parser is active; both fields count as generated output.
            let reasoning = delta
                .and_then(|d| d.get("reasoning_content"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let content = delta
                .and_then(|d| d.get("content"))
                .and_then(|v| v.as_str())
                .unwrap_or("");

            if !reasoning.is_empty() || !content.is_empty() {
                let now = Instant::now();
                if ttft.is_none() {
                    ttft = Some(started.elapsed().as_secs_f64() * 1000.0);
                } else if let Some(prev) = last_token_at {
                    itls.push(now.duration_since(prev).as_secs_f64() * 1000.0);
                }
                last_token_at = Some(now);
                if ttfc.is_none() && !content.is_empty() {
                    ttfc = Some(started.elapsed().as_secs_f64() * 1000.0);
                }
            }

            if let Some(usage) = event.get("usage").filter(|u| !u.is_null()) {
                if let Some(v) = usage.get("prompt_tokens").and_then(|v| v.as_u64()) {
                    prompt_tokens = v as u32;
                }
                if let Some(v) = usage.get("completion_tokens").and_then(|v| v.as_u64()) {
                    output_tokens = v as u32;
                }
            }
        }
    }

    let e2el_ms = started.elapsed().as_secs_f64() * 1000.0;
    let ttft_ms = ttft.ok_or_else(|| {
        AppError::pipeline("benchmark stream produced no tokens — is the model loaded?")
    })?;
    if output_tokens == 0 {
        // Older servers may omit usage on the final chunk; fall back to the
        // observed chunk count so the report still has a denominator.
        output_tokens = (itls.len() + 1) as u32;
    }

    Ok((ttft_ms, ttfc, e2el_ms, prompt_tokens, output_tokens, itls))
}

/// Benchmark an OpenAI-compatible endpoint. `concurrency > 1` measures
/// aggregate throughput under load; `1` measures single-stream latency, which
/// is what dataset generation actually experiences.
pub async fn benchmark_endpoint(
    endpoint: &str,
    model: &str,
    opts: &BenchOptions,
) -> Result<BenchReport> {
    let base = endpoint.trim().trim_end_matches('/');
    if base.is_empty() {
        return Err(AppError::config("teacher endpoint is not configured"));
    }
    if model.trim().is_empty() {
        return Err(AppError::config("teacher model id is not configured"));
    }
    let url = format!("{base}/v1/chat/completions");
    let http = client(opts.timeout_s)?;
    let kwargs = crate::config::chat_template_kwargs(opts.reasoning_effort.as_deref());

    let concurrency = opts.concurrency.max(1) as usize;
    let started = Instant::now();
    let mut samples: Vec<BenchSample> = Vec::new();
    let mut errors: Vec<String> = Vec::new();
    let mut observed_itls: Vec<f64> = Vec::new();

    for (batch_idx, batch) in BENCH_PROMPTS.chunks(concurrency).enumerate() {
        let futures = batch.iter().enumerate().map(|(offset, prompt)| {
            let http = http.clone();
            let url = url.clone();
            let model = model.to_string();
            let kwargs = kwargs.clone();
            let prompt = prompt.to_string();
            let index = (batch_idx * concurrency + offset) as u32;
            let max_tokens = opts.max_tokens;
            async move {
                let result = stream_one(
                    &http,
                    &url,
                    &model,
                    &prompt,
                    max_tokens,
                    kwargs.as_ref(),
                )
                .await;
                (index, prompt, result)
            }
        });

        for (index, prompt, result) in futures::future::join_all(futures).await {
            match result {
                Ok((ttft_ms, ttfc_ms, e2el_ms, prompt_tokens, output_tokens, itls)) => {
                    let decode_ms = (e2el_ms - ttft_ms).max(0.0);
                    let tpot_ms = if output_tokens > 1 {
                        decode_ms / (output_tokens - 1) as f64
                    } else {
                        0.0
                    };
                    observed_itls.extend(itls);
                    samples.push(BenchSample {
                        index,
                        prompt,
                        ttft_ms,
                        ttfc_ms,
                        e2el_ms,
                        prompt_tokens,
                        output_tokens,
                        tpot_ms,
                        output_tps: output_tokens as f64 / (e2el_ms / 1000.0).max(1e-9),
                    });
                }
                Err(e) => errors.push(e.to_string()),
            }
        }
    }

    let duration_s = started.elapsed().as_secs_f64();
    if samples.is_empty() {
        return Err(AppError::pipeline(format!(
            "benchmark produced no successful requests. {}",
            errors.first().cloned().unwrap_or_default()
        )));
    }

    let mut ttfts: Vec<f64> = samples.iter().map(|s| s.ttft_ms).collect();
    ttfts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let e2els: Vec<f64> = samples.iter().map(|s| s.e2el_ms).collect();
    let tpots: Vec<f64> = samples.iter().map(|s| s.tpot_ms).collect();
    let total_output: u32 = samples.iter().map(|s| s.output_tokens).sum();
    let total_input: u32 = samples.iter().map(|s| s.prompt_tokens).sum();
    let completed = samples.len() as u32;

    Ok(BenchReport {
        endpoint: base.to_string(),
        model: model.to_string(),
        completed,
        failed: errors.len() as u32,
        errors,
        duration_s,
        concurrency: concurrency as u32,
        max_tokens: opts.max_tokens,
        reasoning_effort: opts.reasoning_effort.clone(),
        mean_ttft_ms: mean(&ttfts),
        median_ttft_ms: percentile(&ttfts, 50.0),
        p95_ttft_ms: percentile(&ttfts, 95.0),
        mean_tpot_ms: mean(&tpots),
        // Prefer genuinely observed inter-token gaps; fall back to the TPOT mean
        // (which is equal to mean ITL for a single request) when the server
        // streamed the whole completion in one chunk.
        mean_itl_ms: if observed_itls.is_empty() {
            mean(&tpots)
        } else {
            mean(&observed_itls)
        },
        mean_e2el_ms: mean(&e2els),
        output_tokens_per_s: total_output as f64 / duration_s.max(1e-9),
        total_tokens_per_s: (total_output + total_input) as f64 / duration_s.max(1e-9),
        request_throughput: completed as f64 / duration_s.max(1e-9),
        total_output_tokens: total_output,
        total_input_tokens: total_input,
        samples,
        captured_at: chrono::Local::now().to_rfc3339(),
    })
}
