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

3. Explain how it should be solved step-by-step inside the <think></think> tags. Even if it involves logical reasoning, words, mathematics, formulas, functions, or identifications, explain them clearly so any student can easily follow and understand.

4. Provide the final ANSWER along with a simplified explanation of WHY it is the correct answer.

Format your response EXACTLY like this, with no extra commentary before or after:

QUESTION: <the new question>

<think>
<step-by-step simplified explanation of the reasoning, logic, functions, words, or mathematical steps needed to solve the question, written so any student can understand it>
</think>

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
    /// The focus topic this pair was generated under. Empty = no topic filter.
    #[serde(default)]
    pub topic: String,
    #[serde(default)]
    pub messages: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratorConfig {
    pub teacher_endpoint: String,    // OpenAI-compatible base, e.g. http://127.0.0.1:8000
    pub teacher_model: String,       // value passed as `model:` to /v1/chat/completions
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
            // 4096 because reasoning teachers (DeepSeek-R1-Distill etc.) emit
            // long free-form thinking *before* the QUESTION:/ANSWER: block.
            // At 1024 the response was getting truncated mid-think and the
            // QUESTION: marker never appeared, so every chunk was rejected.
            max_tokens: 4096,
            max_pairs_per_chunk: 1,
            concurrency: 4,
            api_key: None,
        }
    }
}

fn http() -> Client {
    static CLIENT: std::sync::OnceLock<Client> = std::sync::OnceLock::new();
    CLIENT.get_or_init(|| {
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
                    || msg.contains("504");
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
    let url = format!("{}/v1/chat/completions", cfg.teacher_endpoint.trim_end_matches('/'));
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
///   - the requested format (QUESTION → <think>…</think> → ANSWER), and
///   - reasoning models like DeepSeek-R1 that emit `<think>…</think>` at the
///     start before the QUESTION/ANSWER block.
/// `<think>` is now optional — if the model omits it we still keep the pair
/// (think defaults to empty); historically requiring it was the main reason
/// every chunk got rejected.
pub fn parse_pair(raw: &str, chunk: &Chunk) -> std::result::Result<GeneratedPair, ParseReject> {
    let trimmed = raw.trim_start();
    if trimmed.to_ascii_uppercase().starts_with("SKIP:") {
        return Err(ParseReject::Skip);
    }
    // Some models emit "SKIP: off-topic" anywhere in the response.
    if trimmed.to_ascii_uppercase().contains("SKIP: OFF-TOPIC")
        && !trimmed.to_ascii_uppercase().contains("QUESTION:")
    {
        return Err(ParseReject::Skip);
    }

    // QUESTION block: from "QUESTION:" up to (but not including) the next
    // <think>, ANSWER:, or end of string. Use lookahead-friendly non-greedy
    // match without lookahead support by stopping at the earliest of those.
    let q_re = Regex::new(r"(?is)QUESTION\s*:\s*(.*?)(?:\n\s*<think>|\nANSWER\s*:|\z)").unwrap();
    let t_re = Regex::new(r"(?is)<think>\s*(.*?)\s*</think>").unwrap();
    let a_re = Regex::new(r"(?is)ANSWER\s*:\s*(.*)\z").unwrap();

    let q = match q_re.captures(raw) {
        Some(c) => c.get(1).map(|m| m.as_str().trim().to_string()).unwrap_or_default(),
        None => return Err(ParseReject::NoQuestion),
    };
    if q.is_empty() {
        return Err(ParseReject::NoQuestion);
    }

    let a = match a_re.captures(raw) {
        Some(c) => c.get(1).map(|m| m.as_str().trim().to_string()).unwrap_or_default(),
        None => return Err(ParseReject::NoAnswer),
    };
    if a.len() < 20 {
        return Err(ParseReject::AnswerTooShort(a.len()));
    }

    // `<think>` is optional. R1-style reasoning models emit it at the top of
    // the message; the formatted spec puts it between QUESTION and ANSWER.
    // Either way, grab the first match if present.
    let t = t_re
        .captures(raw)
        .and_then(|c| c.get(1).map(|m| m.as_str().trim().to_string()))
        .unwrap_or_default();

    let messages = vec![
        serde_json::json!({ "role": "user", "content": q }),
        serde_json::json!({ "role": "assistant", "content": format!("<think>\n{}\n</think>\n{}", t, a) }),
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
        topic: String::new(),
        messages: Some(messages),
    })
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
        let mut s = stream::iter(page.chunks.into_iter().map(&on_chunk))
            .buffer_unordered(cc);
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
