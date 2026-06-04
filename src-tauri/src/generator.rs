#![allow(dead_code)]

use crate::config::QdrantConfig;
use crate::error::{AppError, Result};
use crate::qdrant::{self, Chunk};
use regex::Regex;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub const DEFAULT_GENERATOR_PROMPT: &str = r#"FOCUS TOPIC: {topic}

ROLE: You are an expert tutor helper designed to explain complex topics and concepts to any student in a simple, easy-to-understand way.

TASK:
I will provide you with source material (treat it as your open notes / RAG database).
1. Identify the core concepts and facts inside the source material.
2. Write a NEW, original question based strictly on those concepts.
- The new question must be on the focus topic '{topic}'. If the source material has no meaningful connection to the focus topic, respond with exactly:
  SKIP: off-topic
- Do NOT copy the source material verbatim. Rephrase, change numbers if mathematical, or shift the angle (e.g. solve for a different variable).
- The question must be answerable strictly using facts from the source material.
- Do NOT repeat a question, fact pattern, or legal-provision angle that has already been used in this generation run.

3. Provide the final ANSWER along with a concise explanation of WHY it is correct.

Format your response EXACTLY like this, with no extra commentary before or after:

QUESTION: <the new question>

ANSWER: <the final answer, followed by a simplified explanation of why it is the correct answer>

Source material:
"""
{chunk_text}
"""
"#;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedPair {
    pub question: String,
    pub think: String,
    pub answer: String,
    pub source_chunk_id: String,
    pub source_file: String,
    #[serde(default)]
    pub source_text: String,
    /// The focus topic this pair was generated under. Empty = no topic filter.
    #[serde(default)]
    pub topic: String,
    #[serde(default)]
    pub messages: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratorConfig {
    pub teacher_endpoint: String, // OpenAI-compatible base, e.g. http://127.0.0.1:8000
    pub teacher_model: String,    // value passed as `model:` to /v1/chat/completions
    pub prompt_template: String,
    pub temperature: f32,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub max_tokens: u32,
    pub max_pairs_per_chunk: u32,
    pub concurrency: u32,
    pub api_key: Option<String>,
}

impl GeneratorConfig {
    pub fn defaults(teacher_endpoint: String, teacher_model: String) -> Self {
        Self {
            teacher_endpoint,
            teacher_model,
            prompt_template: DEFAULT_GENERATOR_PROMPT.to_string(),
            // 0.25 instead of 0.4: strict-schema board-exam prompts hit
            // fewer format failures at low temp without going fully greedy
            // (0.0 collapses distractor diversity).
            temperature: 0.25,
            top_p: 0.9,
            // Stops reasoning chains from looping the same derivation line.
            repetition_penalty: 1.05,
            // 4096 keeps room for board-exam style choices and explanations.
            // Some reasoning teachers still leak hidden reasoning despite the
            // prompt; parsing strips that output before the pair is persisted.
            max_tokens: 4096,
            max_pairs_per_chunk: 1,
            concurrency: 4,
            api_key: None,
        }
    }
}

fn http() -> Client {
    static CLIENT: std::sync::OnceLock<Client> = std::sync::OnceLock::new();
    CLIENT
        .get_or_init(|| {
            Client::builder()
                // Reasoning teachers at ~15 tok/s can take 4+ minutes to emit 4096
                // tokens; 180s was cutting them off mid-stream and surfacing as
                // [teacher-err] http error in the UI.
                .timeout(std::time::Duration::from_secs(600))
                .build()
                .unwrap()
        })
        .clone()
}

pub async fn ask_teacher(cfg: &GeneratorConfig, prompt: &str) -> Result<String> {
    let mut last_err: Option<AppError> = None;
    for attempt in 0..6 {
        match ask_teacher_once(cfg, prompt).await {
            Ok(raw) => return Ok(raw),
            Err(e) => {
                let msg = e.to_string();
                let retryable = msg.contains("429")
                    || msg.to_ascii_lowercase().contains("too many requests")
                    || msg.contains("concurrency_limit_exceeded")
                    || msg.contains("503")
                    || msg.contains("504")
                    || msg.to_ascii_lowercase().contains("error sending request")
                    || msg.to_ascii_lowercase().contains("connection")
                    || msg.to_ascii_lowercase().contains("timeout");
                if !retryable || attempt == 5 {
                    return Err(e);
                }
                last_err = Some(e);
                // Featherless high-cost models can occupy the whole account
                // concurrency budget with one request. Wait long enough for
                // an in-flight completion to finish before retrying.
                let backoff_secs = 8u64 * (attempt as u64 + 1);
                tokio::time::sleep(std::time::Duration::from_secs(backoff_secs)).await;
            }
        }
    }
    Err(last_err.unwrap_or_else(|| AppError::pipeline("teacher request failed")))
}

async fn ask_teacher_once(cfg: &GeneratorConfig, prompt: &str) -> Result<String> {
    let url = format!(
        "{}/v1/chat/completions",
        cfg.teacher_endpoint.trim_end_matches('/')
    );
    let body = json!({
        "model": cfg.teacher_model,
        "messages": [
            { "role": "user", "content": prompt }
        ],
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "repetition_penalty": cfg.repetition_penalty,
        "max_tokens": cfg.max_tokens,
    });
    let mut req = http().post(url).json(&body);
    if let Some(ref key) = cfg.api_key {
        if !key.trim().is_empty() {
            req = req.bearer_auth(key.trim());
        }
    }
    let res = req.send().await?;
    if !res.status().is_success() {
        let s = res.status();
        let txt = res.text().await.unwrap_or_default();
        return Err(AppError::pipeline(format!("teacher http {s}: {txt}")));
    }
    let v: Value = res.json().await?;
    let content = v
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c0| c0.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .ok_or_else(|| AppError::pipeline("teacher response missing choices[0].message.content"))?
        .to_string();
    Ok(content)
}

/// Reason a teacher response failed to parse — surfaced to the UI so the user
/// can see whether the model is skipping topics, emitting the wrong format,
/// or producing answers that are too short.
#[derive(Debug, Clone)]
pub enum ParseReject {
    Skip,
    NoQuestion,
    NoAnswer,
    AnswerTooShort(usize),
}

impl ParseReject {
    pub fn label(&self) -> String {
        match self {
            Self::Skip => "off-topic (teacher said SKIP)".to_string(),
            Self::NoQuestion => "no QUESTION: marker in response".to_string(),
            Self::NoAnswer => "no ANSWER: marker in response".to_string(),
            Self::AnswerTooShort(n) => format!("answer too short ({} chars)", n),
        }
    }
}

/// Parse the Teacher's response into a GeneratedPair. Returns `Err(reason)`
/// when the response can't be parsed so the caller can log *why*.
///
/// Handles both:
///   - the requested QUESTION:/ANSWER: format,
///   - the older Q1/A1 format used by some fine-tuned teachers,
///   - reasoning models that leak `<think>...</think>` or
///     `thinking ... response` wrappers before the usable answer.
pub fn parse_pair(raw: &str, chunk: &Chunk) -> std::result::Result<GeneratedPair, ParseReject> {
    let trimmed = raw.trim_start();
    if trimmed.to_ascii_uppercase().starts_with("SKIP:") {
        return Err(ParseReject::Skip);
    }
    // Some models emit "SKIP: off-topic" anywhere in the response.
    if trimmed.to_ascii_uppercase().contains("SKIP: OFF-TOPIC")
        && !trimmed.to_ascii_uppercase().contains("QUESTION:")
        && !trimmed.contains("Q1:")
    {
        return Err(ParseReject::Skip);
    }

    // Try the canonical QUESTION:/ANSWER: format first.
    let q_re =
        Regex::new(r"(?is)QUESTION\s*:\s*(.*?)(?:\n\s*(?:<?think>|thinking\b)|\nANSWER\s*:|\z)")
            .unwrap();
    let think_tag_re = Regex::new(r"(?is)<?think>\s*(.*?)\s*</think>").unwrap();
    let thinking_re = Regex::new(r"(?is)\bthinking\s*(.*?)\s*response").unwrap();
    let a_re = Regex::new(r"(?is)ANSWER\s*:\s*(.*)\z").unwrap();

    // Fallback: some fine-tuned teachers emit Q1:/A1: format instead.
    let q1_re = Regex::new(r"(?is)Q1\s*:\s*(.*?)(?:\n|A1\s*:|\z)").unwrap();
    let a1_re = Regex::new(r"(?is)A1\s*:\s*(.*)\z").unwrap();

    let has_question_fmt = trimmed.contains("QUESTION:");
    let has_q1_fmt = trimmed.contains("Q1:");

    let (q, a) = if has_question_fmt || !has_q1_fmt {
        // Use canonical QUESTION: format
        let question = match q_re.captures(raw) {
            Some(c) => c
                .get(1)
                .map(|m| m.as_str().trim().to_string())
                .unwrap_or_default(),
            None => return Err(ParseReject::NoQuestion),
        };
        if question.is_empty() {
            return Err(ParseReject::NoQuestion);
        }
        let answer = match a_re.captures(raw) {
            Some(c) => c
                .get(1)
                .map(|m| m.as_str().trim().to_string())
                .unwrap_or_default(),
            None => return Err(ParseReject::NoAnswer),
        };
        (
            strip_thinking_blocks(&question),
            strip_thinking_blocks(&answer),
        )
    } else {
        // Use Q1: / A1: fallback format
        let question = match q1_re.captures(raw) {
            Some(c) => c
                .get(1)
                .map(|m| m.as_str().trim().to_string())
                .unwrap_or_default(),
            None => return Err(ParseReject::NoQuestion),
        };
        if question.is_empty() {
            return Err(ParseReject::NoQuestion);
        }
        let answer = match a1_re.captures(raw) {
            Some(c) => c
                .get(1)
                .map(|m| m.as_str().trim().to_string())
                .unwrap_or_default(),
            None => {
                // If A1: not found, try everything after the last Q1: block as the answer
                raw.splitn(2, |c: char| c == '\n')
                    .nth(1)
                    .unwrap_or("")
                    .trim()
                    .to_string()
            }
        };
        (
            strip_thinking_blocks(&question),
            strip_thinking_blocks(&answer),
        )
    };

    if a.len() < 20 {
        return Err(ParseReject::AnswerTooShort(a.len()));
    }

    let t = think_tag_re
        .captures(raw)
        .and_then(|c| c.get(1).map(|m| m.as_str().trim().to_string()))
        .or_else(|| {
            thinking_re
                .captures(raw)
                .and_then(|c| c.get(1).map(|m| m.as_str().trim().to_string()))
        })
        .unwrap_or_default();

    let messages = vec![
        serde_json::json!({ "role": "user", "content": q }),
        serde_json::json!({ "role": "assistant", "content": a }),
    ];

    Ok(GeneratedPair {
        question: q,
        think: t,
        answer: a,
        source_chunk_id: chunk.id.clone(),
        source_file: if !chunk.file_name.is_empty() {
            chunk.file_name.clone()
        } else {
            chunk.file_path.clone()
        },
        source_text: chunk.text.clone(),
        topic: String::new(),
        messages: Some(messages),
    })
}

pub fn strip_thinking_blocks(text: &str) -> String {
    let mut out = text.to_string();
    let patterns = [
        r"(?is)<?think>\s*.*?\s*</think>",
        r"(?is)<thinking>\s*.*?\s*</thinking>",
        r"(?is)\bthinking\s+.*?\s+\bresponse\b",
    ];
    for pattern in patterns {
        if let Ok(re) = Regex::new(pattern) {
            out = re.replace_all(&out, "").to_string();
        }
    }
    out.trim().to_string()
}

pub fn normalize_question_for_dedup(question: &str) -> String {
    let stripped = strip_thinking_blocks(question);
    let mut out = String::with_capacity(stripped.len());
    let mut last_space = true;
    for ch in stripped.chars().flat_map(|c| c.to_lowercase()) {
        if ch.is_alphanumeric() {
            out.push(ch);
            last_space = false;
        } else if !last_space {
            out.push(' ');
            last_space = true;
        }
    }
    out.trim().to_string()
}

fn token_similarity(a: &str, b: &str) -> f32 {
    let a_tokens: std::collections::HashSet<&str> =
        a.split_whitespace().filter(|t| t.len() > 2).collect();
    let b_tokens: std::collections::HashSet<&str> =
        b.split_whitespace().filter(|t| t.len() > 2).collect();
    if a_tokens.is_empty() || b_tokens.is_empty() {
        return 0.0;
    }
    let intersection = a_tokens.intersection(&b_tokens).count() as f32;
    let union = a_tokens.union(&b_tokens).count() as f32;
    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

fn same_opening_fact_pattern(a: &str, b: &str) -> bool {
    let a_words: Vec<&str> = a
        .split_whitespace()
        .filter(|w| w.len() > 2)
        .take(14)
        .collect();
    let b_words: Vec<&str> = b
        .split_whitespace()
        .filter(|w| w.len() > 2)
        .take(14)
        .collect();
    if a_words.len() < 10 || b_words.len() < 10 {
        return false;
    }
    let same = a_words
        .iter()
        .zip(b_words.iter())
        .take_while(|(a, b)| a == b)
        .count();
    same >= 10
}

pub fn duplicate_question_reason(
    question: &str,
    accepted_normalized: &[String],
) -> Option<&'static str> {
    let normalized = normalize_question_for_dedup(question);
    if normalized.len() < 12 {
        return None;
    }
    for existing in accepted_normalized {
        if existing == &normalized {
            return Some("duplicate question");
        }
        let len_ratio = (normalized.len().min(existing.len()) as f32)
            / (normalized.len().max(existing.len()) as f32);
        let similarity = token_similarity(&normalized, existing);
        if (len_ratio > 0.72 && similarity >= 0.86)
            || (len_ratio > 0.84 && similarity >= 0.80)
            || same_opening_fact_pattern(&normalized, existing)
        {
            return Some("near-duplicate question");
        }
    }
    None
}

/// Build the prompt for a chunk by substituting `{chunk_text}`.
pub fn build_prompt(template: &str, chunk: &Chunk) -> String {
    template.replace("{chunk_text}", &chunk.text)
}

/// Process a pre-fetched list of chunks (e.g. from a semantic search) with up
/// to `concurrency` in-flight futures at a time. Identical concurrency model
/// to `for_each_chunk` but without paging.
pub async fn for_each_in<F, Fut>(
    chunks: Vec<Chunk>,
    concurrency: usize,
    mut should_continue: impl FnMut() -> bool,
    on_chunk: F,
) -> Result<u64>
where
    F: Fn(Chunk) -> Fut,
    Fut: std::future::Future<Output = Result<()>>,
{
    use futures::stream::{self, StreamExt};
    let cc = concurrency.max(1);
    if !should_continue() {
        return Ok(0);
    }
    let total = chunks.len() as u64;
    let mut s = stream::iter(chunks.into_iter().map(&on_chunk)).buffer_unordered(cc);
    while let Some(res) = s.next().await {
        res?;
        if !should_continue() {
            break;
        }
    }
    Ok(total)
}

/// Pull every chunk from Qdrant page by page and process them with up to
/// `concurrency` in-flight futures at a time, using `futures::stream::buffer_unordered`
/// so we never need `'static` task spawning — the closure can freely borrow
/// non-`'static` state (SSH session, app handle, mutex guards, etc.).
/// `should_continue` is consulted between pages so the topic cap / cancel flag /
/// max_chunks limit can stop the loop early.
pub async fn for_each_chunk<F, Fut, P>(
    qd: &QdrantConfig,
    page_size: u32,
    concurrency: usize,
    mut should_continue: impl FnMut() -> bool,
    on_chunk: F,
    mut on_page: P,
) -> Result<u64>
where
    F: Fn(Chunk) -> Fut,
    Fut: std::future::Future<Output = Result<()>>,
    P: FnMut(u32, usize, u64),
{
    use futures::stream::{self, StreamExt};

    let cc = concurrency.max(1);
    let mut offset: Option<Value> = None;
    let mut total: u64 = 0;
    let mut page_idx: u32 = 0;

    loop {
        if !should_continue() {
            break;
        }
        let page = qdrant::scroll(qd, page_size, offset.clone(), None).await?;
        let n = page.chunks.len();
        page_idx += 1;
        on_page(page_idx, n, total);
        total += n as u64;

        // Process this page with `cc` workers in flight.
        let mut s = stream::iter(page.chunks.into_iter().map(&on_chunk)).buffer_unordered(cc);
        while let Some(res) = s.next().await {
            res?;
        }

        match page.next_offset {
            Some(v) if v.is_null() => break,
            Some(v) => offset = Some(v),
            None => break,
        }
    }
    Ok(total)
}
