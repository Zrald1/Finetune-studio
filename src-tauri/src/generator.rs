#![allow(dead_code)]

use crate::config::QdrantConfig;
use crate::error::{AppError, Result};
use crate::qdrant::{self, Chunk};
use regex::Regex;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DatasetFormat {
    SimpleQa,
    ReasoningQa,
    MultipleChoice,
    ChainOfThought,
    InstructionIo,
    Conversational,
}

impl DatasetFormat {
    pub fn all() -> Vec<Self> {
        vec![
            Self::SimpleQa,
            Self::ReasoningQa,
            Self::MultipleChoice,
            Self::ChainOfThought,
            Self::InstructionIo,
            Self::Conversational,
        ]
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::SimpleQa => "Simple Q&A",
            Self::ReasoningQa => "Q&A with Reasoning",
            Self::MultipleChoice => "Multiple Choice",
            Self::ChainOfThought => "Chain-of-Thought",
            Self::InstructionIo => "Instruction-Input-Output",
            Self::Conversational => "Conversational Multi-Turn",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::SimpleQa => "Direct question-answer pairs for FAQ and quick instruction tuning.",
            Self::ReasoningQa => {
                "Question-answer pairs with a <think> reasoning block for reasoning-model SFT."
            }
            Self::MultipleChoice => {
                "Four-option MCQs with plausible distractors for exams and benchmarks."
            }
            Self::ChainOfThought => {
                "Step-by-step problem solutions for math, logic, and procedural tasks."
            }
            Self::InstructionIo => "Alpaca-style instruction, input, and output triples.",
            Self::Conversational => "Multi-turn user and assistant dialogues for chat tuning.",
        }
    }

    pub fn default_prompt(&self) -> &'static str {
        match self {
            Self::SimpleQa => SIMPLE_QA_PROMPT,
            Self::ReasoningQa => REASONING_QA_PROMPT,
            Self::MultipleChoice => MCQ_PROMPT,
            Self::ChainOfThought => COT_PROMPT,
            Self::InstructionIo => INSTRUCTION_IO_PROMPT,
            Self::Conversational => CONVERSATIONAL_PROMPT,
        }
    }
}

pub const SIMPLE_QA_PROMPT: &str = r#"FOCUS TOPIC: {topic}

ROLE: You are a knowledgeable tutor creating clear, direct study questions.

TASK:
Using the source material below, write ONE original question and a concise, factually-grounded answer.

RULES:
- Question must be answerable strictly from the source.
- Do NOT copy the source verbatim. Rephrase or shift the angle.
- Answer should be 2-5 sentences, direct and complete.
- If the source is unrelated to '{topic}', respond EXACTLY: SKIP: off-topic

OUTPUT FORMAT (strict):
QUESTION:
ANSWER:

Source material:
"""
{chunk_text}
"""
"#;

pub const REASONING_QA_PROMPT: &str = r#"FOCUS TOPIC: {topic}

ROLE: You are an expert tutor creating reasoning-based training data for advanced LLMs.

TASK:
Using the source material below, write ONE original question, a reasoning chain, and a final answer.

RULES:
- Question must be answerable strictly from the source.
- REASONING should identify facts, connect them, and derive the answer.
- ANSWER should be the final, clean response.
- Reasoning should be 3-7 sentences using words like "because", "therefore", "since", or "this means".
- If source is unrelated to '{topic}', respond EXACTLY: SKIP: off-topic

OUTPUT FORMAT (strict):
QUESTION:
REASONING:
ANSWER:

Source material:
"""
{chunk_text}
"""
"#;

pub const MCQ_PROMPT: &str = r#"FOCUS TOPIC: {topic}

ROLE: You are an expert dataset curator and question writer. Your job is to generate high-quality training question-answer pairs with detailed reasoning from source material for any domain or subject.

TASK:
Given the source material below, generate ONE high-quality question-answer pair with step-by-step reasoning.

RULES:
1. Detect the domain from the source material and adapt your question style accordingly:
   - Quantitative/math-heavy material -> computation or problem-solving question (change given values slightly)
   - Legal/regulatory material -> scenario-based application question, NOT a definition question
   - Conceptual/theory material -> "which is MOST accurate / appropriate" question that tests understanding, not recall
   - Procedural material -> step-ordering or error-identification question
2. Provide exactly 4 choices (A-D) with plausible distractors based on common mistakes or misconceptions.
3. REASONING must be detailed:
   - Break the problem down step by step
   - Eliminate wrong choices explicitly and explain why each is wrong
   - Show the derivation, formula application, or rule being used
   - Conclude with why the correct answer is correct
4. ANSWER must state the correct letter and a concise summary of the reasoning (2-3 sentences max).
5. Do NOT copy the source verbatim. Rephrase, vary values, or shift the angle.
6. The question must be answerable strictly from the source material.
7. If the source material has no meaningful connection to '{topic}', respond with exactly: SKIP: off-topic

FORMAT (strictly - no extra text before or after):
QUESTION: <stem>
A. <choice>
B. <choice>
C. <choice>
D. <choice>
REASONING: <detailed step-by-step reasoning, distractor elimination, formula/rule application>
ANSWER: <correct letter> - <concise 2-3 sentence summary of why it is correct>

Source material:
"""
{chunk_text}
"""
"#;

pub const COT_PROMPT: &str = r#"FOCUS TOPIC: {topic}

ROLE: You are a methodical teacher creating step-by-step solution training data.

TASK:
Using the source material, write ONE problem and a numbered step-by-step solution leading to a final answer. Ideal for math, logic, procedures, or multi-step derivations.

RULES:
- Problem should require at least 3 reasoning steps.
- Each step must be explicit, atomic, and explained.
- End with a clearly labeled FINAL ANSWER.
- Use formulas, equations, or rule citations where applicable.
- If unrelated to '{topic}', respond EXACTLY: SKIP: off-topic

OUTPUT FORMAT (strict):
PROBLEM:
SOLUTION:
Step 1:
Step 2:
Step 3:
[continue as needed]
FINAL ANSWER:

Source material:
"""
{chunk_text}
"""
"#;

pub const INSTRUCTION_IO_PROMPT: &str = r#"FOCUS TOPIC: {topic}

ROLE: You are creating Alpaca-style instruction tuning data.

TASK:
Using the source material, generate ONE instruction-input-output triple.

RULES:
- INSTRUCTION: a clear task directive (for example, "Summarize the following", "Explain why...", "Calculate...").
- INPUT: relevant context/data the instruction operates on. Use "N/A" if the instruction is self-contained.
- OUTPUT: the complete, accurate response derived from the source.
- Vary instruction types: summarize, explain, classify, extract, compare, calculate.
- If unrelated to '{topic}', respond EXACTLY: SKIP: off-topic

OUTPUT FORMAT (strict):
INSTRUCTION:
INPUT:
OUTPUT:

Source material:
"""
{chunk_text}
"""
"#;

pub const CONVERSATIONAL_PROMPT: &str = r#"FOCUS TOPIC: {topic}

ROLE: You are scripting a realistic tutor-student dialogue for training a conversational AI.

TASK:
Using the source material, write a 3-4 turn dialogue between a curious USER and an expert ASSISTANT. The conversation should naturally explore a concept from the source.

RULES:
- Turn 1: USER asks a beginner-level question.
- Turn 2: ASSISTANT explains clearly using source facts.
- Turn 3: USER asks a follow-up (clarification, edge case, or deeper question).
- Turn 4: ASSISTANT gives a precise, source-grounded answer.
- Optional Turn 5-6: deeper exchange.
- Keep tone natural and helpful. No reasoning leakage.
- If unrelated to '{topic}', respond EXACTLY: SKIP: off-topic

OUTPUT FORMAT (strict):
USER:
ASSISTANT:
USER:
ASSISTANT:

Source material:
"""
{chunk_text}
"""
"#;

pub const DEFAULT_GENERATOR_PROMPT: &str = MCQ_PROMPT;

fn default_dataset_format() -> DatasetFormat {
    DatasetFormat::MultipleChoice
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedPair {
    #[serde(default = "default_dataset_format")]
    pub format: DatasetFormat,
    pub question: String,
    #[serde(default)]
    pub choices: Vec<String>,
    #[serde(default)]
    pub correct_letter: String,
    #[serde(default, alias = "think")]
    pub reasoning: String,
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

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instruction: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub explanation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turns: Option<Vec<DialogueTurn>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialogueTurn {
    pub role: String,
    pub content: String,
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
    #[serde(default)]
    pub enable_verification: bool,
    #[serde(default)]
    pub verifier_model: Option<String>,
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
            enable_verification: false,
            verifier_model: None,
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
    ReasoningTooShort(usize),
}

impl ParseReject {
    pub fn label(&self) -> String {
        match self {
            Self::Skip => "off-topic (teacher said SKIP)".to_string(),
            Self::NoQuestion => "no QUESTION: marker in response".to_string(),
            Self::NoAnswer => "no ANSWER: marker in response".to_string(),
            Self::AnswerTooShort(n) => format!("answer too short ({} chars)", n),
            Self::ReasoningTooShort(n) => format!("reasoning too short ({} chars)", n),
        }
    }
}

/// Parse the Teacher's response into a GeneratedPair. Returns `Err(reason)`
/// when the response can't be parsed so the caller can log *why*.
pub fn parse_simple_qa(
    raw: &str,
    chunk: &Chunk,
) -> std::result::Result<GeneratedPair, ParseReject> {
    let cleaned = strip_thinking_blocks(raw);
    let question_re = Regex::new(r"(?is)QUESTION\s*:\s*(.*?)(?:\n\s*ANSWER\s*:|\z)").unwrap();
    let answer_re = Regex::new(r"(?is)ANSWER\s*:\s*(.*)").unwrap();

    let question = question_re
        .captures(&cleaned)
        .and_then(|c| c.get(1).map(|m| m.as_str().trim().to_string()))
        .ok_or(ParseReject::NoQuestion)?;

    if question.is_empty() {
        return Err(ParseReject::NoQuestion);
    }

    let answer = answer_re
        .captures(&cleaned)
        .and_then(|c| c.get(1).map(|m| m.as_str().trim().to_string()))
        .ok_or(ParseReject::NoAnswer)?;

    if answer.is_empty() {
        return Err(ParseReject::NoAnswer);
    }
    if answer.len() < 20 {
        return Err(ParseReject::AnswerTooShort(answer.len()));
    }

    let messages = vec![
        json!({ "role": "user", "content": question }),
        json!({ "role": "assistant", "content": answer }),
    ];

    Ok(GeneratedPair {
        format: DatasetFormat::SimpleQa,
        question: question.clone(),
        choices: vec![],
        correct_letter: String::new(),
        reasoning: String::new(),
        answer: answer.clone(),
        source_chunk_id: chunk.id.clone(),
        source_file: if !chunk.file_name.is_empty() {
            chunk.file_name.clone()
        } else {
            chunk.file_path.clone()
        },
        source_text: chunk.text.clone(),
        topic: String::new(),
        messages: Some(messages),
        steps: None,
        instruction: None,
        input: None,
        output: None,
        explanation: None,
        turns: None,
    })
}

pub fn parse_reasoning_qa(
    raw: &str,
    chunk: &Chunk,
) -> std::result::Result<GeneratedPair, ParseReject> {
    let cleaned = strip_thinking_blocks(raw);
    let question_re = Regex::new(r"(?is)QUESTION\s*:\s*(.*?)(?:\n\s*REASONING\s*:|\z)").unwrap();
    let reasoning_re = Regex::new(r"(?is)REASONING\s*:\s*(.*?)(?:\n\s*ANSWER\s*:|\z)").unwrap();
    let answer_re = Regex::new(r"(?is)ANSWER\s*:\s*(.*)").unwrap();
    let think_tag_re = Regex::new(r"(?is)<think>\s*(.*?)\s*</think>").unwrap();

    let question = question_re
        .captures(&cleaned)
        .and_then(|c| c.get(1).map(|m| m.as_str().trim().to_string()))
        .ok_or(ParseReject::NoQuestion)?;

    if question.is_empty() {
        return Err(ParseReject::NoQuestion);
    }

    let reasoning = reasoning_re
        .captures(&cleaned)
        .and_then(|c| c.get(1).map(|m| m.as_str().trim().to_string()))
        .or_else(|| {
            think_tag_re
                .captures(raw)
                .and_then(|c| c.get(1).map(|m| m.as_str().trim().to_string()))
        })
        .unwrap_or_default();
    if !reasoning.is_empty() && reasoning.len() < 40 {
        return Err(ParseReject::ReasoningTooShort(reasoning.len()));
    }

    let answer = answer_re
        .captures(&cleaned)
        .and_then(|c| c.get(1).map(|m| m.as_str().trim().to_string()))
        .ok_or(ParseReject::NoAnswer)?;

    if answer.is_empty() {
        return Err(ParseReject::NoAnswer);
    }
    if answer.len() < 20 {
        return Err(ParseReject::AnswerTooShort(answer.len()));
    }

    let assistant_content = if !reasoning.is_empty() {
        format!("<think>\n{}\n</think>\n\n{}", reasoning, answer)
    } else {
        answer.clone()
    };

    let messages = vec![
        json!({ "role": "user", "content": question }),
        json!({ "role": "assistant", "content": assistant_content }),
    ];

    Ok(GeneratedPair {
        format: DatasetFormat::ReasoningQa,
        question: question.clone(),
        choices: vec![],
        correct_letter: String::new(),
        reasoning,
        answer: answer.clone(),
        source_chunk_id: chunk.id.clone(),
        source_file: if !chunk.file_name.is_empty() {
            chunk.file_name.clone()
        } else {
            chunk.file_path.clone()
        },
        source_text: chunk.text.clone(),
        topic: String::new(),
        messages: Some(messages),
        steps: None,
        instruction: None,
        input: None,
        output: None,
        explanation: None,
        turns: None,
    })
}

pub fn parse_mcq(raw: &str, chunk: &Chunk) -> std::result::Result<GeneratedPair, ParseReject> {
    let cleaned = strip_thinking_blocks(raw);

    let question_re = Regex::new(r"(?is)QUESTION\s*:\s*(.*?)(?:\n\s*A[\.\)\:]\s)").unwrap();
    let choice_re = Regex::new(r"(?im)^\s*([A-D])[\.\)\:]\s+(.+?)\s*$").unwrap();
    let reasoning_re = Regex::new(r"(?is)REASONING\s*:\s*(.*?)(?:\nANSWER\s*:|\z)").unwrap();
    let answer_re = Regex::new(r"(?is)ANSWER\s*:\s*([A-D])").unwrap();
    let explanation_re =
        Regex::new(r"(?is)ANSWER\s*:\s*[A-D][\.\)\:]?\s*.*?\s*[-—]\s*(.*)").unwrap();
    let think_tag_re = Regex::new(r"(?is)<think>\s*(.*?)\s*</think>").unwrap();

    let question = question_re
        .captures(&cleaned)
        .and_then(|c| c.get(1).map(|m| m.as_str().trim().to_string()))
        .ok_or(ParseReject::NoQuestion)?;

    if question.is_empty() {
        return Err(ParseReject::NoQuestion);
    }

    let mut choices_map: std::collections::BTreeMap<char, String> =
        std::collections::BTreeMap::new();
    for cap in choice_re.captures_iter(&cleaned) {
        let letter = cap.get(1).unwrap().as_str().chars().next().unwrap();
        let text = cap.get(2).unwrap().as_str().trim().to_string();
        choices_map.entry(letter).or_insert(text);
    }

    if choices_map.len() < 4 {
        return Err(ParseReject::NoAnswer);
    }

    let choices: Vec<String> = ['A', 'B', 'C', 'D']
        .iter()
        .filter_map(|c| choices_map.get(c).cloned())
        .collect();

    let reasoning = reasoning_re
        .captures(&cleaned)
        .and_then(|c| c.get(1).map(|m| m.as_str().trim().to_string()))
        .or_else(|| {
            think_tag_re
                .captures(raw)
                .and_then(|c| c.get(1).map(|m| m.as_str().trim().to_string()))
        })
        .unwrap_or_default();
    if !reasoning.is_empty() && reasoning.len() < 40 {
        return Err(ParseReject::ReasoningTooShort(reasoning.len()));
    }

    let correct_letter = answer_re
        .captures(&cleaned)
        .and_then(|c| c.get(1).map(|m| m.as_str().to_uppercase()))
        .ok_or(ParseReject::NoAnswer)?;

    let explanation = explanation_re
        .captures(&cleaned)
        .and_then(|c| c.get(1).map(|m| m.as_str().trim().to_string()));

    let answer_text = format!(
        "{}) {}",
        correct_letter,
        choices_map
            .get(&correct_letter.chars().next().unwrap())
            .cloned()
            .unwrap_or_default()
    );

    if answer_text.len() < 20 {
        return Err(ParseReject::AnswerTooShort(answer_text.len()));
    }

    let user_content = format!(
        "{}\n\nA) {}\nB) {}\nC) {}\nD) {}",
        question, choices[0], choices[1], choices[2], choices[3]
    );

    let assistant_content = format!("<think>\n{}\n</think>\n\n{}", reasoning, answer_text);

    let messages = vec![
        json!({ "role": "user", "content": user_content }),
        json!({ "role": "assistant", "content": assistant_content }),
    ];

    Ok(GeneratedPair {
        format: DatasetFormat::MultipleChoice,
        question,
        choices,
        correct_letter,
        reasoning,
        answer: answer_text,
        source_chunk_id: chunk.id.clone(),
        source_file: if !chunk.file_name.is_empty() {
            chunk.file_name.clone()
        } else {
            chunk.file_path.clone()
        },
        source_text: chunk.text.clone(),
        topic: String::new(),
        messages: Some(messages),
        steps: None,
        instruction: None,
        input: None,
        output: None,
        explanation,
        turns: None,
    })
}

pub fn parse_cot(raw: &str, chunk: &Chunk) -> std::result::Result<GeneratedPair, ParseReject> {
    let cleaned = strip_thinking_blocks(raw);
    let problem_re =
        Regex::new(r"(?is)PROBLEM\s*:\s*(.*?)(?:\n\s*(?:SOLUTION|STEPS)\s*:|\z)").unwrap();
    let steps_re =
        Regex::new(r"(?is)(?:SOLUTION|STEPS)\s*:\s*(.*?)(?:\n\s*(?:FINAL\s+ANSWER|ANSWER)\s*:|\z)")
            .unwrap();
    let answer_re = Regex::new(r"(?is)(?:FINAL\s+ANSWER|ANSWER)\s*:\s*(.*)").unwrap();

    let problem = problem_re
        .captures(&cleaned)
        .and_then(|c| c.get(1).map(|m| m.as_str().trim().to_string()))
        .ok_or(ParseReject::NoQuestion)?;

    if problem.is_empty() {
        return Err(ParseReject::NoQuestion);
    }

    let steps_block = steps_re
        .captures(&cleaned)
        .and_then(|c| c.get(1).map(|m| m.as_str().trim().to_string()))
        .unwrap_or_default();

    let answer = answer_re
        .captures(&cleaned)
        .and_then(|c| c.get(1).map(|m| m.as_str().trim().to_string()))
        .ok_or(ParseReject::NoAnswer)?;

    if answer.is_empty() {
        return Err(ParseReject::NoAnswer);
    }
    if answer.len() < 20 {
        return Err(ParseReject::AnswerTooShort(answer.len()));
    }

    let mut steps = Vec::new();
    for line in steps_block.lines() {
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            steps.push(trimmed.to_string());
        }
    }

    let assistant_content = if !steps_block.is_empty() {
        format!("<think>\n{}\n</think>\n\n{}", steps_block, answer)
    } else {
        answer.clone()
    };

    let messages = vec![
        json!({ "role": "user", "content": problem }),
        json!({ "role": "assistant", "content": assistant_content }),
    ];

    Ok(GeneratedPair {
        format: DatasetFormat::ChainOfThought,
        question: problem.clone(),
        choices: vec![],
        correct_letter: String::new(),
        reasoning: steps_block.clone(),
        answer: answer.clone(),
        source_chunk_id: chunk.id.clone(),
        source_file: if !chunk.file_name.is_empty() {
            chunk.file_name.clone()
        } else {
            chunk.file_path.clone()
        },
        source_text: chunk.text.clone(),
        topic: String::new(),
        messages: Some(messages),
        steps: Some(steps),
        instruction: None,
        input: None,
        output: None,
        explanation: None,
        turns: None,
    })
}

pub fn parse_instruction_io(
    raw: &str,
    chunk: &Chunk,
) -> std::result::Result<GeneratedPair, ParseReject> {
    let cleaned = strip_thinking_blocks(raw);
    let inst_re = Regex::new(r"(?is)INSTRUCTION\s*:\s*(.*?)(?:\n\s*INPUT\s*:|\z)").unwrap();
    let input_re = Regex::new(r"(?is)INPUT\s*:\s*(.*?)(?:\n\s*OUTPUT\s*:|\z)").unwrap();
    let output_re = Regex::new(r"(?is)OUTPUT\s*:\s*(.*)").unwrap();

    let instruction = inst_re
        .captures(&cleaned)
        .and_then(|c| c.get(1).map(|m| m.as_str().trim().to_string()))
        .ok_or(ParseReject::NoQuestion)?;

    if instruction.is_empty() {
        return Err(ParseReject::NoQuestion);
    }

    let input_val = input_re
        .captures(&cleaned)
        .and_then(|c| c.get(1).map(|m| m.as_str().trim().to_string()))
        .unwrap_or_default();

    let input = if input_val.is_empty() || input_val.to_ascii_uppercase() == "N/A" {
        None
    } else {
        Some(input_val)
    };

    let output = output_re
        .captures(&cleaned)
        .and_then(|c| c.get(1).map(|m| m.as_str().trim().to_string()))
        .ok_or(ParseReject::NoAnswer)?;

    if output.is_empty() {
        return Err(ParseReject::NoAnswer);
    }
    if output.len() < 20 {
        return Err(ParseReject::AnswerTooShort(output.len()));
    }

    let messages = if let Some(ref inp) = input {
        vec![
            json!({ "role": "user", "content": format!("{}\n\nContext:\n{}", instruction, inp) }),
            json!({ "role": "assistant", "content": output }),
        ]
    } else {
        vec![
            json!({ "role": "user", "content": instruction }),
            json!({ "role": "assistant", "content": output }),
        ]
    };

    Ok(GeneratedPair {
        format: DatasetFormat::InstructionIo,
        question: instruction.clone(),
        choices: vec![],
        correct_letter: String::new(),
        reasoning: String::new(),
        answer: output.clone(),
        source_chunk_id: chunk.id.clone(),
        source_file: if !chunk.file_name.is_empty() {
            chunk.file_name.clone()
        } else {
            chunk.file_path.clone()
        },
        source_text: chunk.text.clone(),
        topic: String::new(),
        messages: Some(messages),
        steps: None,
        instruction: Some(instruction),
        input,
        output: Some(output),
        explanation: None,
        turns: None,
    })
}

pub fn parse_conversational(
    raw: &str,
    chunk: &Chunk,
) -> std::result::Result<GeneratedPair, ParseReject> {
    let cleaned = strip_thinking_blocks(raw);
    let turn_re =
        Regex::new(r"(?is)(USER|ASSISTANT)\s*:\s*(.*?)(?=\n\s*(?:USER|ASSISTANT)\s*:|\z)").unwrap();

    let mut messages = Vec::new();
    let mut turns = Vec::new();
    let mut first_question = String::new();
    let mut first_answer = String::new();

    for cap in turn_re.captures_iter(&cleaned) {
        let role_str = cap.get(1).unwrap().as_str().to_ascii_lowercase();
        let content = cap.get(2).unwrap().as_str().trim().to_string();
        let role = if role_str == "user" {
            "user"
        } else {
            "assistant"
        };

        if role == "user" && first_question.is_empty() {
            first_question = content.clone();
        } else if role == "assistant" && first_answer.is_empty() {
            first_answer = content.clone();
        }

        messages.push(json!({
            "role": role,
            "content": content.clone()
        }));
        turns.push(DialogueTurn {
            role: role.to_string(),
            content,
        });
    }

    if messages.is_empty() {
        return Err(ParseReject::NoQuestion);
    }
    if first_question.is_empty() {
        return Err(ParseReject::NoQuestion);
    }
    if first_answer.is_empty() {
        return Err(ParseReject::NoAnswer);
    }

    Ok(GeneratedPair {
        format: DatasetFormat::Conversational,
        question: first_question,
        choices: vec![],
        correct_letter: String::new(),
        reasoning: String::new(),
        answer: first_answer,
        source_chunk_id: chunk.id.clone(),
        source_file: if !chunk.file_name.is_empty() {
            chunk.file_name.clone()
        } else {
            chunk.file_path.clone()
        },
        source_text: chunk.text.clone(),
        topic: String::new(),
        messages: Some(messages),
        steps: None,
        instruction: None,
        input: None,
        output: None,
        explanation: None,
        turns: Some(turns),
    })
}

pub fn parse_pair(
    raw: &str,
    chunk: &Chunk,
    format: DatasetFormat,
) -> std::result::Result<GeneratedPair, ParseReject> {
    let cleaned = strip_thinking_blocks(raw);
    let trimmed = cleaned.trim_start();

    if trimmed.to_ascii_uppercase().starts_with("SKIP:") {
        return Err(ParseReject::Skip);
    }
    if trimmed.to_ascii_uppercase().contains("SKIP: OFF-TOPIC")
        && !trimmed.to_ascii_uppercase().contains("QUESTION:")
        && !trimmed.to_ascii_uppercase().contains("PROBLEM:")
        && !trimmed.to_ascii_uppercase().contains("INSTRUCTION:")
        && !trimmed.to_ascii_uppercase().contains("USER:")
    {
        return Err(ParseReject::Skip);
    }

    match format {
        DatasetFormat::SimpleQa => parse_simple_qa(raw, chunk),
        DatasetFormat::ReasoningQa => parse_reasoning_qa(raw, chunk),
        DatasetFormat::MultipleChoice => parse_mcq(raw, chunk),
        DatasetFormat::ChainOfThought => parse_cot(raw, chunk),
        DatasetFormat::InstructionIo => parse_instruction_io(raw, chunk),
        DatasetFormat::Conversational => parse_conversational(raw, chunk),
    }
}

pub fn export_to_jsonl(pair: &GeneratedPair) -> Value {
    match pair.format {
        DatasetFormat::SimpleQa => {
            if let Some(ref messages) = pair.messages {
                json!({ "messages": messages })
            } else {
                json!({
                    "messages": [
                        { "role": "user", "content": pair.question },
                        { "role": "assistant", "content": pair.answer }
                    ]
                })
            }
        }
        DatasetFormat::ReasoningQa => {
            let assistant_content = if !pair.reasoning.is_empty() {
                format!("<think>\n{}\n</think>\n\n{}", pair.reasoning, pair.answer)
            } else {
                pair.answer.clone()
            };
            json!({
                "reasoning": pair.reasoning,
                "messages": [
                    { "role": "user", "content": pair.question },
                    { "role": "assistant", "content": assistant_content }
                ]
            })
        }
        DatasetFormat::MultipleChoice => {
            let user_content = if pair.choices.len() >= 4 {
                format!(
                    "{}\n\nA) {}\nB) {}\nC) {}\nD) {}",
                    pair.question,
                    pair.choices[0],
                    pair.choices[1],
                    pair.choices[2],
                    pair.choices[3]
                )
            } else {
                pair.question.clone()
            };
            let assistant_content = if !pair.reasoning.is_empty() {
                format!("<think>\n{}\n</think>\n\n{}", pair.reasoning, pair.answer)
            } else {
                pair.answer.clone()
            };
            json!({
                "question": pair.question,
                "choices": pair.choices,
                "answer": pair.correct_letter,
                "explanation": pair.explanation,
                "messages": [
                    { "role": "user", "content": user_content },
                    { "role": "assistant", "content": assistant_content }
                ]
            })
        }
        DatasetFormat::ChainOfThought => {
            let steps_text = pair
                .steps
                .as_ref()
                .map(|s| s.join("\n"))
                .unwrap_or_else(|| pair.reasoning.clone());
            let assistant_content = if !steps_text.is_empty() {
                format!("<think>\n{}\n</think>\n\n{}", steps_text, pair.answer)
            } else {
                pair.answer.clone()
            };
            json!({
                "problem": pair.question,
                "steps": pair.steps,
                "final_answer": pair.answer,
                "messages": [
                    { "role": "user", "content": pair.question },
                    { "role": "assistant", "content": assistant_content }
                ]
            })
        }
        DatasetFormat::InstructionIo => {
            json!({
                "instruction": pair.instruction.as_ref().unwrap_or(&pair.question),
                "input": pair.input.as_ref().map(|s| s.as_str()).unwrap_or("N/A"),
                "output": pair.output.as_ref().unwrap_or(&pair.answer)
            })
        }
        DatasetFormat::Conversational => {
            if let Some(ref msgs) = pair.messages {
                json!({
                    "messages": msgs
                })
            } else {
                json!({
                    "messages": [
                        { "role": "user", "content": pair.question },
                        { "role": "assistant", "content": pair.answer }
                    ]
                })
            }
        }
    }
}

pub fn strip_thinking_blocks(text: &str) -> String {
    let mut out = text.to_string();
    let patterns = [
        r"(?is)<think>.*?</think>",
        r"(?is)<thinking>.*?</thinking>",
        r"(?is)<reasoning>.*?</reasoning>",
        r"(?is)<\|begin_of_thought\|>.*?<\|end_of_thought\|>",
        r"(?is)\bthinking:\s*.*?\bresponse:",
        r"(?is)<?think>\s*.*?</think>",
        r"(?is)<thinking>\s*.*?</thinking>",
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

#[derive(Debug, Clone)]
pub enum QualityReject {
    NoReasoningKeywords,
    ChoicesDuplicated,
    ChoicesTooShort,
    AnswerNotInChoices,
    QuestionTooShort,
}

impl QualityReject {
    pub fn label(&self) -> String {
        match self {
            Self::NoReasoningKeywords => "reasoning lacks logical connectors".into(),
            Self::ChoicesDuplicated => "duplicate or near-identical choices".into(),
            Self::ChoicesTooShort => "one or more choices too short".into(),
            Self::AnswerNotInChoices => "answer letter not in A-D".into(),
            Self::QuestionTooShort => "question too short to be meaningful".into(),
        }
    }
}

pub fn validate_quality(pair: &GeneratedPair) -> std::result::Result<(), QualityReject> {
    // 1. Question length
    if pair.question.split_whitespace().count() < 6 {
        return Err(QualityReject::QuestionTooShort);
    }

    if pair.format == DatasetFormat::MultipleChoice {
        // 2. Answer letter validity
        if !["A", "B", "C", "D"].contains(&pair.correct_letter.as_str()) {
            return Err(QualityReject::AnswerNotInChoices);
        }

        // 3. Choices length
        if pair.choices.len() < 4
            || pair
                .choices
                .iter()
                .any(|c| c.split_whitespace().count() < 2)
        {
            return Err(QualityReject::ChoicesTooShort);
        }

        // 4. Choice uniqueness (case-insensitive token similarity)
        for i in 0..pair.choices.len() {
            for j in (i + 1)..pair.choices.len() {
                let sim = token_similarity(
                    &pair.choices[i].to_lowercase(),
                    &pair.choices[j].to_lowercase(),
                );
                if sim > 0.85 {
                    return Err(QualityReject::ChoicesDuplicated);
                }
            }
        }

        // 5. Reasoning quality: must contain logical connectors or math
        let reasoning_lc = pair.reasoning.to_lowercase();
        let has_reasoning = [
            "because",
            "therefore",
            "thus",
            "since",
            "so ",
            "implies",
            "leads to",
            "follows",
            "result",
        ]
        .iter()
        .any(|kw| reasoning_lc.contains(kw))
            || pair.reasoning.contains('=')
            || pair.reasoning.contains('×')
            || pair.reasoning.contains('+');

        if !has_reasoning && pair.reasoning.len() < 80 {
            return Err(QualityReject::NoReasoningKeywords);
        }
    } else if pair.format == DatasetFormat::ReasoningQa
        || pair.format == DatasetFormat::ChainOfThought
    {
        let reasoning_lc = pair.reasoning.to_lowercase();
        let has_reasoning = [
            "because",
            "therefore",
            "thus",
            "since",
            "so ",
            "implies",
            "leads to",
            "follows",
            "result",
        ]
        .iter()
        .any(|kw| reasoning_lc.contains(kw))
            || pair.reasoning.contains('=')
            || pair.reasoning.contains('×')
            || pair.reasoning.contains('+');

        if !has_reasoning && pair.reasoning.len() < 50 {
            return Err(QualityReject::NoReasoningKeywords);
        }
    }

    Ok(())
}

pub const VERIFIER_PROMPT: &str = r#"You are a strict fact-checker. Given a SOURCE PASSAGE and a QUESTION-ANSWER pair, verify:

1. Is the correct answer FULLY supported by the source? (yes/no)
2. Are the distractors plausible but clearly wrong per the source? (yes/no)
3. Is the reasoning consistent with the source facts? (yes/no)

Respond in EXACTLY this format:
SUPPORTED: <yes|no>
DISTRACTORS_OK: <yes|no>
REASONING_OK: <yes|no>
VERDICT: <accept|reject>
NOTES: <one short sentence>

SOURCE PASSAGE:
"""
{source}
"""

QUESTION-ANSWER:
{qa_block}
"#;

pub async fn verify_pair(cfg: &GeneratorConfig, pair: &GeneratedPair) -> Result<bool> {
    let qa_block = match pair.format {
        DatasetFormat::MultipleChoice => {
            format!(
                "Q: {}\nA) {}\nB) {}\nC) {}\nD) {}\nCorrect: {}\nReasoning: {}",
                pair.question,
                pair.choices.get(0).cloned().unwrap_or_default(),
                pair.choices.get(1).cloned().unwrap_or_default(),
                pair.choices.get(2).cloned().unwrap_or_default(),
                pair.choices.get(3).cloned().unwrap_or_default(),
                pair.correct_letter,
                pair.reasoning
            )
        }
        DatasetFormat::InstructionIo => {
            format!(
                "Instruction: {}\nInput: {}\nOutput: {}",
                pair.instruction.as_ref().unwrap_or(&pair.question),
                pair.input.as_ref().map(|s| s.as_str()).unwrap_or(""),
                pair.output.as_ref().unwrap_or(&pair.answer)
            )
        }
        DatasetFormat::Conversational => {
            let mut turns = String::new();
            if let Some(ref msgs) = pair.messages {
                for m in msgs {
                    if let (Some(role), Some(content)) = (
                        m.get("role").and_then(|r| r.as_str()),
                        m.get("content").and_then(|c| c.as_str()),
                    ) {
                        turns.push_str(&format!("{}: {}\n", role.to_uppercase(), content));
                    }
                }
            } else {
                turns = format!("USER: {}\nASSISTANT: {}\n", pair.question, pair.answer);
            }
            turns
        }
        _ => {
            format!(
                "Q: {}\nReasoning: {}\nA: {}",
                pair.question, pair.reasoning, pair.answer
            )
        }
    };

    let prompt = VERIFIER_PROMPT
        .replace("{source}", &pair.source_text)
        .replace("{qa_block}", &qa_block);

    let mut verifier_cfg = cfg.clone();
    if let Some(ref model) = cfg.verifier_model {
        if !model.trim().is_empty() {
            verifier_cfg.teacher_model = model.clone();
        }
    }

    let raw = ask_teacher(&verifier_cfg, &prompt).await?;
    let verdict_re = Regex::new(r"(?i)VERDICT\s*:\s*(accept|reject)").unwrap();

    let accepted = verdict_re
        .captures(&raw)
        .and_then(|c| c.get(1).map(|m| m.as_str().to_lowercase()))
        .map(|v| v == "accept")
        .unwrap_or(false);

    Ok(accepted)
}

/// Fetch consecutive chunks (i-1, i, i+1) from same document for richer context.
/// Preserves narrative flow vs. vector-similarity neighbors which may be unrelated.
pub async fn fetch_consecutive_bundle(
    qd: &QdrantConfig,
    chunk: &Chunk,
    window: usize, // 1 = ±1 neighbor, 2 = ±2 neighbors
) -> Result<String> {
    let chunk_idx = chunk.chunk_index;
    let file_path = &chunk.file_path;

    let start = chunk_idx.saturating_sub(window as i64);
    let end = chunk_idx + window as i64;

    // Qdrant filter: same file_path AND chunk_index in [start, end]
    let neighbors = qdrant::scroll_filtered(qd, file_path, start, end).await?;

    // Sort by chunk_index to preserve document order
    let mut sorted = neighbors;
    sorted.sort_by_key(|c| c.chunk_index);

    let bundled = sorted
        .iter()
        .map(|c| c.text.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");

    Ok(bundled)
}

pub async fn build_prompt_bundled(
    template: &str,
    chunk: &Chunk,
    qd: &QdrantConfig,
    bundle_window: usize,
) -> Result<String> {
    let bundled_text = if bundle_window > 0 {
        fetch_consecutive_bundle(qd, chunk, bundle_window)
            .await
            .unwrap_or_else(|_| chunk.text.clone())
    } else {
        chunk.text.clone()
    };
    Ok(template.replace("{chunk_text}", &bundled_text))
}
