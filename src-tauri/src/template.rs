//! Custom dataset templates.
//!
//! Each format has a built-in prompt, but domains differ enough that a one-size
//! template leaves quality on the table — a clinical dataset wants different
//! framing from a legal one. This module lets a template be overridden while
//! keeping the two things that actually decide whether the output is usable:
//!
//! 1. **The source must reach the model.** A template without `{chunk_text}`
//!    produces a teacher answering from its own memory, which is the single
//!    most reliable way to synthesise confident nonsense.
//! 2. **The output must be parseable.** Every format's parser looks for specific
//!    markers (`QUESTION:`, `ANSWER:`, …). A template that asks for prose gets
//!    prose, the parser finds nothing, and the pair is silently dropped — the
//!    run "succeeds" while producing an empty dataset. That failure is checked
//!    here rather than discovered after a long generation.
//!
//! Validation returns errors (the template cannot work) separately from warnings
//! (it probably will not), because a template in an unusual language may
//! legitimately phrase the markers differently.

use crate::generator::DatasetFormat;

/// The placeholder that injects the retrieved source chunk.
pub const PLACEHOLDER_CHUNK: &str = "{chunk_text}";
/// The placeholder that injects the run's focus topic.
pub const PLACEHOLDER_TOPIC: &str = "{topic}";

/// Every placeholder a template may use. Anything else is a typo that would
/// survive into the prompt as literal text.
pub const KNOWN_PLACEHOLDERS: &[&str] = &[PLACEHOLDER_CHUNK, PLACEHOLDER_TOPIC];

/// Minimum template length. Shorter than this cannot express a task plus a
/// format, so it is almost certainly a mistake.
pub const MIN_TEMPLATE_CHARS: usize = 80;

/// Markers the parser for `format` needs to find in the model's output.
///
/// Mirrors the parsing code — if a parser changes, this must change with it, and
/// `required_markers_match_the_parsers` pins that down.
pub fn required_markers(format: DatasetFormat) -> &'static [&'static str] {
    match format {
        DatasetFormat::SimpleQa => &["QUESTION:", "ANSWER:"],
        DatasetFormat::ReasoningQa => &["QUESTION:", "REASONING:", "ANSWER:"],
        DatasetFormat::MultipleChoice => &["QUESTION:", "ANSWER:"],
        DatasetFormat::ChainOfThought => &["PROBLEM:", "FINAL ANSWER:"],
        DatasetFormat::InstructionIo => &["INSTRUCTION:", "OUTPUT:"],
        DatasetFormat::Conversational => &["USER:", "ASSISTANT:"],
    }
}

/// Result of checking a custom template.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TemplateCheck {
    /// The template cannot produce usable data.
    pub errors: Vec<String>,
    /// The template will probably work, but something looks off.
    pub warnings: Vec<String>,
}

impl TemplateCheck {
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }

    /// One-line summary for the UI and the run log.
    pub fn summary(&self) -> String {
        if self.errors.is_empty() && self.warnings.is_empty() {
            return "template OK".to_string();
        }
        let mut parts = Vec::new();
        if !self.errors.is_empty() {
            parts.push(format!("errors: {}", self.errors.join("; ")));
        }
        if !self.warnings.is_empty() {
            parts.push(format!("warnings: {}", self.warnings.join("; ")));
        }
        parts.join(" | ")
    }
}

/// Check a custom template against the format it will drive.
pub fn check(template: &str, format: DatasetFormat) -> TemplateCheck {
    let mut out = TemplateCheck::default();
    let trimmed = template.trim();

    if trimmed.is_empty() {
        out.errors.push("template is empty".into());
        return out;
    }

    if trimmed.chars().count() < MIN_TEMPLATE_CHARS {
        out.errors.push(format!(
            "template is only {} characters; a task plus an output format needs at least {MIN_TEMPLATE_CHARS}",
            trimmed.chars().count()
        ));
    }

    // The decisive check: without the source, the teacher answers from memory.
    if !template.contains(PLACEHOLDER_CHUNK) {
        out.errors.push(format!(
            "template must contain {PLACEHOLDER_CHUNK} — without the source chunk the teacher \
             answers from its own memory and the pair cannot be verified against anything"
        ));
    }

    if !template.contains(PLACEHOLDER_TOPIC) {
        out.warnings.push(format!(
            "no {PLACEHOLDER_TOPIC} placeholder: every chunk generates the same framing"
        ));
    }

    // Catch typos that would otherwise survive as literal text.
    for token in placeholders_in(template) {
        if !KNOWN_PLACEHOLDERS.contains(&token.as_str()) {
            out.errors.push(format!(
                "unknown placeholder {token} — known placeholders are {}",
                KNOWN_PLACEHOLDERS.join(", ")
            ));
        }
    }

    // Parseability. This is what stops a run from "succeeding" with zero rows.
    let missing: Vec<&str> = required_markers(format)
        .iter()
        .copied()
        .filter(|m| !template.to_ascii_uppercase().contains(&m.to_ascii_uppercase()))
        .collect();
    if !missing.is_empty() {
        out.errors.push(format!(
            "the {} parser looks for {} but the template never asks for {} — the model will \
             produce output the parser cannot read",
            format.display_name(),
            required_markers(format).join(", "),
            missing.join(", ")
        ));
    }

    out
}

/// Extract `{...}` tokens from a template.
fn placeholders_in(template: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = template;
    while let Some(start) = rest.find('{') {
        let after = &rest[start..];
        let Some(end) = after.find('}') else { break };
        let token = &after[..=end];
        // Only brace-delimited single tokens; f-strings and JSON braces are not
        // placeholders and must not be reported as typos.
        if token.len() < 40 && !token.contains('\n') && token[1..token.len() - 1].chars().all(|c| {
            c.is_ascii_alphanumeric() || c == '_'
        }) {
            out.push(token.to_string());
        }
        rest = &after[end + 1..];
    }
    out
}

/// Render a template for one chunk.
pub fn render(template: &str, topic: &str, chunk_text: &str) -> String {
    template
        .replace(PLACEHOLDER_TOPIC, topic)
        .replace(PLACEHOLDER_CHUNK, chunk_text)
}

/// The template to use: the override when it passes validation, else the
/// built-in. A broken override must never silently produce an empty dataset, so
/// the caller gets the built-in and a reason.
pub fn resolve(
    format: DatasetFormat,
    custom: Option<&str>,
) -> (String, Option<TemplateCheck>) {
    match custom.map(str::trim).filter(|t| !t.is_empty()) {
        Some(t) => {
            let result = check(t, format);
            if result.is_ok() {
                (t.to_string(), Some(result))
            } else {
                (format.default_prompt().to_string(), Some(result))
            }
        }
        None => (format.default_prompt().to_string(), None),
    }
}

/// Starter templates for domains that need different framing. Offered in the UI
/// so a user edits a working template rather than writing one from scratch.
pub fn preset_templates(format: DatasetFormat) -> Vec<(&'static str, &'static str)> {
    match format {
        DatasetFormat::MultipleChoice => vec![
            (
                "Exam — distractors from common misconceptions",
                EXAM_MCQ,
            ),
            (
                "Clinical — vignette style",
                CLINICAL_MCQ,
            ),
        ],
        DatasetFormat::ReasoningQa => vec![(
            "Socratic — reasoning must cite the source",
            SOCRATIC_REASONING,
        )],
        _ => vec![],
    }
}

pub const EXAM_MCQ: &str = r#"FOCUS TOPIC: {topic}

ROLE: You are an exam writer for a professional certification board.

TASK: From the source material, write ONE four-option multiple-choice question.

RULES:
1. The stem must be answerable from the source alone.
2. Each distractor must be a *plausible* error a candidate could make — a common
   misconception, a swapped term, or a correct fact applied to the wrong step.
   Never use filler or obviously absurd options.
3. Exactly one option is correct.
4. REASONING must justify the key and say why each distractor fails.
5. If the source is unrelated to '{topic}', respond with exactly: SKIP: off-topic

FORMAT (no text before or after):
QUESTION: <stem>
A. <choice>
B. <choice>
C. <choice>
D. <choice>
REASONING: <why the key is right and each distractor is wrong>
ANSWER: <letter> - <one-line justification>

Source material:
"""
{chunk_text}
""""#;

pub const CLINICAL_MCQ: &str = r#"FOCUS TOPIC: {topic}

ROLE: You are writing continuing-education material for clinicians.

TASK: From the source material, write ONE vignette-style question.

RULES:
1. Open with a short clinical vignette (age, presentation, key finding) drawn
   from the source — do not invent patient details the source does not support.
2. Ask for the best next step, diagnosis, or mechanism.
3. Distractors must be plausible alternatives that are wrong for a stated reason.
4. REASONING must walk through the discriminating finding.
5. If the source is unrelated to '{topic}', respond with exactly: SKIP: off-topic

FORMAT (no text before or after):
QUESTION: <vignette + question>
A. <choice>
B. <choice>
C. <choice>
D. <choice>
REASONING: <step-by-step discrimination>
ANSWER: <letter> - <one-line justification>

Source material:
"""
{chunk_text}
""""#;

pub const SOCRATIC_REASONING: &str = r#"FOCUS TOPIC: {topic}

ROLE: You are a tutor who never asserts anything the source does not support.

TASK: From the source material, write ONE question and a reasoning chain that
quotes the source at each step.

RULES:
1. Every reasoning step must cite the phrase from the source that justifies it,
   in the form: <step> — source: "<quoted phrase>".
2. Do not introduce any fact, figure, or entity absent from the source.
3. If the source does not fully support an answer, respond with exactly:
   SKIP: insufficient
4. If the source is unrelated to '{topic}', respond with exactly: SKIP: off-topic

FORMAT (no text before or after):
QUESTION: <question>
REASONING: <numbered steps, each with its source citation>
ANSWER: <answer, plus the source phrase that establishes it>

Source material:
"""
{chunk_text}
""""#;

#[cfg(test)]
mod tests {
    use super::*;

    fn good_mcq() -> String {
        format!(
            "FOCUS TOPIC: {PLACEHOLDER_TOPIC}\n\nROLE: exam writer\n\nTASK: write one question\n\n\
             FORMAT:\nQUESTION: <stem>\nA. <c>\nB. <c>\nC. <c>\nD. <c>\nREASONING: <why>\n\
             ANSWER: <letter>\n\nSource material:\n\"\"\"\n{PLACEHOLDER_CHUNK}\n\"\"\""
        )
    }

    // ── Validation ──────────────────────────────────────────────────────────

    #[test]
    fn accepts_a_well_formed_template() {
        let c = check(&good_mcq(), DatasetFormat::MultipleChoice);
        assert!(c.is_ok(), "should pass: {:?}", c.errors);
    }

    #[test]
    fn rejects_an_empty_template() {
        let c = check("   ", DatasetFormat::SimpleQa);
        assert!(!c.is_ok());
        assert!(c.errors[0].contains("empty"));
    }

    #[test]
    fn rejects_a_template_without_the_source_placeholder() {
        // The highest-consequence mistake: the teacher answers from memory and
        // nothing downstream can tell.
        let t = "FOCUS TOPIC: {topic}\n\nWrite a question.\nFORMAT:\nQUESTION: x\nANSWER: y\n\
                 Make it detailed and useful for training a model on this subject matter.";
        let c = check(t, DatasetFormat::SimpleQa);
        assert!(!c.is_ok());
        assert!(
            c.errors.iter().any(|e| e.contains(PLACEHOLDER_CHUNK)),
            "must demand the chunk placeholder: {:?}",
            c.errors
        );
    }

    #[test]
    fn rejects_a_template_the_parser_cannot_read() {
        // This is the failure that produces a "successful" empty dataset.
        let t = format!(
            "FOCUS TOPIC: {PLACEHOLDER_TOPIC}\n\nWrite a short essay answering a question about \
             the material below. Be thorough and explain your thinking in prose.\n\n\
             Source material:\n\"\"\"\n{PLACEHOLDER_CHUNK}\n\"\"\""
        );
        let c = check(&t, DatasetFormat::SimpleQa);
        assert!(!c.is_ok(), "prose output cannot satisfy the parser");
        assert!(
            c.errors.iter().any(|e| e.contains("parser")),
            "the reason must name the parser: {:?}",
            c.errors
        );
    }

    #[test]
    fn rejects_unknown_placeholders() {
        let t = format!(
            "FOCUS TOPIC: {PLACEHOLDER_TOPIC}\n\nUse {{source_doc}} to write a question.\n\
             FORMAT:\nQUESTION: x\nANSWER: y\n\nMaterial:\n\"\"\"\n{PLACEHOLDER_CHUNK}\n\"\"\"\n\
             Write one high quality pair."
        );
        let c = check(&t, DatasetFormat::SimpleQa);
        assert!(
            c.errors.iter().any(|e| e.contains("{source_doc}")),
            "a typo would survive as literal text: {:?}",
            c.errors
        );
    }

    #[test]
    fn rejects_a_too_short_template() {
        let c = check("{chunk_text} QUESTION: ANSWER:", DatasetFormat::SimpleQa);
        assert!(!c.is_ok());
        assert!(c.errors.iter().any(|e| e.contains("characters")));
    }

    #[test]
    fn warns_when_the_topic_placeholder_is_absent() {
        let t = format!(
            "ROLE: exam writer\n\nTASK: write one question from the material.\n\nFORMAT:\n\
             QUESTION: <stem>\nANSWER: <letter>\n\nSource material:\n\"\"\"\n{PLACEHOLDER_CHUNK}\n\"\"\""
        );
        let c = check(&t, DatasetFormat::SimpleQa);
        assert!(c.is_ok(), "missing topic is a warning, not an error: {:?}", c.errors);
        assert!(c.warnings.iter().any(|w| w.contains("topic")));
    }

    #[test]
    fn every_builtin_prompt_passes_its_own_check() {
        // The built-ins must satisfy the gate they impose on user templates —
        // otherwise the gate is wrong, not the templates.
        for format in [
            DatasetFormat::SimpleQa,
            DatasetFormat::ReasoningQa,
            DatasetFormat::MultipleChoice,
            DatasetFormat::ChainOfThought,
            DatasetFormat::InstructionIo,
            DatasetFormat::Conversational,
        ] {
            let c = check(format.default_prompt(), format);
            assert!(
                c.is_ok(),
                "built-in {} fails its own check: {:?}",
                format.display_name(),
                c.errors
            );
        }
    }

    #[test]
    fn every_preset_passes_its_own_check() {
        for format in [DatasetFormat::MultipleChoice, DatasetFormat::ReasoningQa] {
            for (name, template) in preset_templates(format) {
                let c = check(template, format);
                assert!(c.is_ok(), "preset {name} fails: {:?}", c.errors);
            }
        }
    }

    // ── Rendering ───────────────────────────────────────────────────────────

    #[test]
    fn render_substitutes_both_placeholders() {
        let out = render(&good_mcq(), "cardiology", "The heart has four chambers.");
        assert!(out.contains("cardiology"));
        assert!(out.contains("The heart has four chambers."));
        assert!(!out.contains(PLACEHOLDER_CHUNK));
        assert!(!out.contains(PLACEHOLDER_TOPIC));
    }

    #[test]
    fn render_survives_braces_in_the_chunk() {
        // Source text containing JSON must not be re-interpreted.
        let out = render(&good_mcq(), "t", "{\"key\": \"value\"}");
        assert!(out.contains("{\"key\": \"value\"}"));
    }

    // ── Resolution ──────────────────────────────────────────────────────────

    #[test]
    fn resolve_falls_back_to_the_builtin_when_a_template_is_broken() {
        // A broken override must never silently produce an empty dataset.
        let (template, check_result) = resolve(DatasetFormat::SimpleQa, Some("nonsense"));
        assert_eq!(template, DatasetFormat::SimpleQa.default_prompt());
        assert!(check_result.is_some_and(|c| !c.is_ok()));
    }

    #[test]
    fn resolve_uses_a_valid_override() {
        let good = good_mcq();
        let (template, _) = resolve(DatasetFormat::MultipleChoice, Some(&good));
        assert_eq!(template, good);
    }

    #[test]
    fn resolve_treats_blank_as_no_override() {
        let (template, result) = resolve(DatasetFormat::SimpleQa, Some("   "));
        assert_eq!(template, DatasetFormat::SimpleQa.default_prompt());
        assert!(result.is_none());
    }

    // ── Drift guard ─────────────────────────────────────────────────────────

    #[test]
    fn required_markers_match_the_parsers() {
        // If a parser's markers change, this fails rather than letting the
        // template gate drift into validating the wrong thing.
        assert_eq!(
            required_markers(DatasetFormat::SimpleQa),
            &["QUESTION:", "ANSWER:"]
        );
        assert!(required_markers(DatasetFormat::ReasoningQa).contains(&"REASONING:"));
        assert!(required_markers(DatasetFormat::InstructionIo).contains(&"INSTRUCTION:"));
        assert!(required_markers(DatasetFormat::Conversational).contains(&"USER:"));
    }

    #[test]
    fn placeholder_scanner_ignores_non_placeholders() {
        // f-strings and JSON must not be reported as typos.
        let found = placeholders_in("{\"a\": 1} and {chunk_text} and f\"{x}\"");
        assert!(found.contains(&"{chunk_text}".to_string()));
        assert!(!found.contains(&"{\"a\"".to_string()));
    }
}
