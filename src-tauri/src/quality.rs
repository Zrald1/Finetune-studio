//! Dataset quality gates.
//!
//! Follows the synthetic-data literature rather than inventing heuristics:
//!
//! - **LIMA** (Zhou et al., 2023) showed 1,000 filtered examples beat 50,000
//!   noisy ones, because fine-tuning teaches *format* far more readily than
//!   knowledge. Filtering matters more than volume, so these gates reject
//!   aggressively and report what they rejected.
//! - **Provenance-preserving gating** (arXiv 2606.11127) found that gating
//!   against the exact source chunk beats post-hoc retrieval, and that
//!   hallucination gates and reward gates reject **largely disjoint** failure
//!   populations — so both are needed. `grounding` here is the hallucination
//!   gate; `generator::verify_pair` is the reward gate.
//! - **Self-Instruct** (Wang et al., 2022) deduplicates with a ROUGE-L overlap
//!   filter. Alpaca shipped ~20% near-duplicate instructions without one.
//!
//! Everything here is deterministic and cheap — it runs once per generated pair
//! on the orchestrator, not on the GPU.

use crate::generator::DatasetFormat;
use std::collections::{BTreeMap, HashSet};

/// Words carrying no topical signal. Kept short on purpose: an over-broad list
/// would inflate grounding scores by discarding the very terms we check.
const STOPWORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "that", "this", "these", "those",
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did", "doing", "have",
    "has", "had", "having", "of", "in", "on", "at", "to", "for", "with", "by", "from", "as", "it",
    "its", "they", "them", "their", "there", "here", "which", "who", "whom", "what", "when",
    "where", "why", "how", "not", "no", "nor", "so", "such", "can", "could", "will", "would",
    "shall", "should", "may", "might", "must", "about", "into", "over", "under", "between",
    "because", "therefore", "thus", "since", "also", "more", "most", "some", "any", "all", "each",
    "other", "only", "own", "same", "very", "just", "up", "down", "out", "off", "again", "further",
];

fn is_stopword(word: &str) -> bool {
    STOPWORDS.contains(&word)
}

/// Lowercased alphanumeric tokens.
pub fn tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_ascii_lowercase())
        .collect()
}

/// Tokens with stopwords removed — the words that carry meaning.
pub fn content_words(text: &str) -> Vec<String> {
    tokens(text)
        .into_iter()
        .filter(|t| !is_stopword(t))
        .collect()
}

// Note: near-duplicate detection lives in `generator::duplicate_question_reason`
// (token similarity plus length-ratio guards, the Self-Instruct ROUGE-L filter
// in spirit). Deliberately not duplicated here — a second, differently-tuned
// similarity function would drift from the one the pipeline actually uses.

/// Fraction of the answer's content words that also appear in the source chunk.
///
/// This is the hallucination gate. It is deliberately lexical: a cheap,
/// deterministic first line that catches the common failure — the model
/// inventing entities, names or values the source never contained — without
/// needing a second model in the loop. Paraphrase scores lower than copying, so
/// the threshold is set to tolerate rephrasing rather than demand extraction.
pub fn grounding_score(answer: &str, source: &str) -> f32 {
    if source.trim().is_empty() {
        // No provenance to check against. Reporting 1.0 would be a lie; the
        // caller decides whether an unchecked pair is acceptable.
        return f32::NAN;
    }
    let answer_words = content_words(answer);
    if answer_words.is_empty() {
        return 1.0;
    }
    let source_set: HashSet<String> = content_words(source).into_iter().collect();
    let hits = answer_words
        .iter()
        .filter(|w| source_set.contains(*w))
        .count();
    hits as f32 / answer_words.len() as f32
}

/// Numbers in the answer that do **not** appear anywhere in the source.
///
/// Numeric fabrication is the most damaging hallucination in study material —
/// a confidently wrong value teaches the model to be confidently wrong — and it
/// is also the cheapest to detect exactly. An empty result means every figure in
/// the answer is traceable to the source.
pub fn ungrounded_numbers(answer: &str, source: &str) -> Vec<String> {
    if source.trim().is_empty() {
        return Vec::new();
    }
    let source_nums: HashSet<String> = tokens(source)
        .into_iter()
        .filter(|t| t.chars().any(|c| c.is_ascii_digit()))
        .collect();
    let mut out: Vec<String> = tokens(answer)
        .into_iter()
        .filter(|t| t.chars().any(|c| c.is_ascii_digit()))
        .filter(|t| !source_nums.contains(t))
        .collect();
    out.sort();
    out.dedup();
    out
}

/// Minimum answer length per format.
///
/// Measured against a live teacher: MCQ answers of 12–13 characters
/// ("ATP and NADPH.") passed the old gate, and a 12-character answer teaches
/// almost nothing. The floor scales with how much reasoning a format implies.
pub fn min_answer_chars(format: DatasetFormat) -> usize {
    match format {
        DatasetFormat::SimpleQa => 40,
        DatasetFormat::ReasoningQa => 60,
        DatasetFormat::MultipleChoice => 30,
        DatasetFormat::ChainOfThought => 60,
        DatasetFormat::InstructionIo => 40,
        DatasetFormat::Conversational => 60,
    }
}

/// Minimum reasoning length for the formats that carry a reasoning block.
pub fn min_reasoning_chars(format: DatasetFormat) -> usize {
    match format {
        DatasetFormat::ReasoningQa | DatasetFormat::ChainOfThought => 120,
        DatasetFormat::MultipleChoice => 80,
        _ => 0,
    }
}

/// Grounding floor. 0.35 tolerates paraphrase while still rejecting answers that
/// introduce vocabulary the source never used.
pub const MIN_GROUNDING: f32 = 0.35;

// ── Trainer compatibility ────────────────────────────────────────────────────

/// Characters per token for English prose.
///
/// Deliberately pessimistic (real English averages closer to 4.2) so the
/// estimate errs toward "too long" rather than "fits". Under-estimating is the
/// dangerous direction: LLaMA-Factory truncates silently, so a pair we wrongly
/// bless gets its tail cut off and trains the model on a mangled example.
const CHARS_PER_TOKEN: f32 = 3.6;

/// Rough token count for a string, without loading a tokenizer.
///
/// The orchestrator has no tokenizer for the student model, and downloading one
/// per run just to size-check a dataset is not worth the round trip. This is
/// good enough to catch the failure that matters — an example that overruns
/// `cutoff_len` — and the margin over-counts on purpose.
pub fn estimate_tokens(text: &str) -> usize {
    let chars = text.chars().count();
    if chars == 0 {
        return 0;
    }
    (chars as f32 / CHARS_PER_TOKEN).ceil() as usize
}

/// Tokens a training example costs: prompt plus target, plus an allowance for
/// the chat template's control tokens.
pub fn example_tokens(question: &str, answer: &str, reasoning: &str) -> usize {
    const TEMPLATE_OVERHEAD: usize = 16;
    let mut total = estimate_tokens(question) + estimate_tokens(answer) + TEMPLATE_OVERHEAD;
    if !reasoning.trim().is_empty() {
        total += estimate_tokens(reasoning);
    }
    total
}

/// Whether an example fits inside the trainer's `cutoff_len`.
///
/// This exists because the failure is silent and expensive. LLaMA-Factory trims
/// an over-long example *after* applying the chat template, so the tail of the
/// answer — or the closing control tokens — is what disappears. The maintainers'
/// own description of the resulting run: *"get a bad result, and have zero idea
/// why."* Catching it here turns that into an explicit rejection.
pub fn fits_cutoff(question: &str, answer: &str, reasoning: &str, cutoff_len: usize) -> bool {
    if cutoff_len == 0 {
        // 0 means "unset" rather than "zero-length"; do not reject everything.
        return true;
    }
    example_tokens(question, answer, reasoning) <= cutoff_len
}

/// The `cutoff_len` a set of examples actually needs.
///
/// The app defaults `cutoff_len` to 1024, which reasoning-formatted pairs
/// routinely exceed — a 1200-character reasoning block plus a 300-character
/// answer is already past it. Surfacing the real requirement lets the UI say
/// "raise cutoff_len to N" rather than silently discarding good data.
pub fn required_cutoff_len(examples: &[(String, String, String)]) -> usize {
    examples
        .iter()
        .map(|(q, a, r)| example_tokens(q, a, r))
        .max()
        .unwrap_or(0)
}

/// What fraction of examples would be truncated at this `cutoff_len`.
pub fn truncation_rate(examples: &[(String, String, String)], cutoff_len: usize) -> f32 {
    if examples.is_empty() || cutoff_len == 0 {
        return 0.0;
    }
    let over = examples
        .iter()
        .filter(|(q, a, r)| !fits_cutoff(q, a, r, cutoff_len))
        .count();
    over as f32 / examples.len() as f32
}

// ── Coverage reporting ───────────────────────────────────────────────────────

/// What a generation run actually produced, and what it threw away.
///
/// A rejected-pair count alone is not actionable. This exists so the UI can say
/// *why* pairs were dropped and whether the survivors actually span the corpus,
/// which is the difference between "it ran" and "the dataset is usable".
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct DatasetReport {
    pub generated: usize,
    pub kept: usize,
    /// Rejection reason → count.
    pub rejected: BTreeMap<String, usize>,
    pub mean_answer_chars: f32,
    pub mean_reasoning_chars: f32,
    pub mean_grounding: f32,
    /// Distinct source chunks that contributed at least one kept pair.
    pub distinct_sources: usize,
    /// Distinct topics that contributed at least one kept pair.
    pub distinct_topics: usize,
    /// Distinct questions, as a share of kept pairs. 1.0 means no repetition.
    pub question_diversity: f32,
}

impl DatasetReport {
    /// Share of kept pairs that are near-duplicates of an earlier one.
    pub fn duplicate_rate(&self) -> f32 {
        let n = self.rejected.get("near-duplicate").copied().unwrap_or(0);
        if self.generated == 0 {
            0.0
        } else {
            n as f32 / self.generated as f32
        }
    }

    /// One-line summary for the run log.
    pub fn summary(&self) -> String {
        let mut reasons: Vec<String> = self
            .rejected
            .iter()
            .filter(|(_, n)| **n > 0)
            .map(|(k, n)| format!("{k}={n}"))
            .collect();
        reasons.sort();
        format!(
            "kept {}/{} · grounding {:.2} · answer {:.0} ch · {} source(s) · {} topic(s) · dup {:.0}%{}",
            self.kept,
            self.generated,
            if self.mean_grounding.is_nan() { 0.0 } else { self.mean_grounding },
            self.mean_answer_chars,
            self.distinct_sources,
            self.distinct_topics,
            self.duplicate_rate() * 100.0,
            if reasons.is_empty() {
                String::new()
            } else {
                format!(" · rejected: {}", reasons.join(", "))
            }
        )
    }
}

/// Accumulates a [`DatasetReport`] as pairs are accepted or rejected.
#[derive(Debug, Default)]
pub struct ReportBuilder {
    report: DatasetReport,
    answer_chars: Vec<f32>,
    reasoning_chars: Vec<f32>,
    groundings: Vec<f32>,
    sources: HashSet<String>,
    topics: HashSet<String>,
    questions: Vec<String>,
}

impl ReportBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn generated(&mut self) {
        self.report.generated += 1;
    }

    pub fn reject(&mut self, reason: &str) {
        *self.report.rejected.entry(reason.to_string()).or_insert(0) += 1;
    }

    pub fn keep(
        &mut self,
        question: &str,
        answer: &str,
        reasoning: &str,
        source_id: &str,
        topic: &str,
        grounding: f32,
    ) {
        self.report.kept += 1;
        self.answer_chars.push(answer.chars().count() as f32);
        self.reasoning_chars.push(reasoning.chars().count() as f32);
        if !grounding.is_nan() {
            self.groundings.push(grounding);
        }
        if !source_id.trim().is_empty() {
            self.sources.insert(source_id.to_string());
        }
        if !topic.trim().is_empty() {
            self.topics.insert(topic.to_string());
        }
        self.questions.push(question.to_string());
    }

    pub fn finish(mut self) -> DatasetReport {
        let mean = |v: &[f32]| {
            if v.is_empty() {
                0.0
            } else {
                v.iter().sum::<f32>() / v.len() as f32
            }
        };
        self.report.mean_answer_chars = mean(&self.answer_chars);
        self.report.mean_reasoning_chars = mean(&self.reasoning_chars);
        self.report.mean_grounding = mean(&self.groundings);
        self.report.distinct_sources = self.sources.len();
        self.report.distinct_topics = self.topics.len();

        // Distinct question *shapes*, not exact strings: a template that only
        // swaps a noun should not count as new coverage.
        let mut unique = HashSet::new();
        for q in &self.questions {
            unique.insert(content_words(q).join(" "));
        }
        self.report.question_diversity = if self.questions.is_empty() {
            0.0
        } else {
            unique.len() as f32 / self.questions.len() as f32
        };
        self.report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SRC: &str = "Photosynthesis converts light energy into chemical energy inside \
        chloroplasts. The light-dependent reactions produce ATP and NADPH; the Calvin cycle \
        fixes carbon dioxide into glucose using the enzyme RuBisCO.";

    // ── Tokenisation ────────────────────────────────────────────────────────

    #[test]
    fn tokenises_and_lowercases() {
        assert_eq!(tokens("Hello, World!"), vec!["hello", "world"]);
    }

    #[test]
    fn strips_stopwords_from_content_words() {
        let w = content_words("the cat is on a mat");
        assert_eq!(w, vec!["cat", "mat"]);
    }

    // ── Grounding ───────────────────────────────────────────────────────────

    #[test]
    fn grounded_answer_scores_high() {
        let score = grounding_score("ATP and NADPH are produced by the light reactions", SRC);
        assert!(score > 0.6, "expected strong grounding, got {score}");
    }

    #[test]
    fn hallucinated_answer_scores_low() {
        let score = grounding_score(
            "Mitochondria generate electricity through quantum tunnelling in zebrafish",
            SRC,
        );
        assert!(score < MIN_GROUNDING, "hallucination should fail, got {score}");
    }

    #[test]
    fn paraphrase_is_tolerated() {
        // The gate must not demand extraction — rephrasing is legitimate.
        let score = grounding_score(
            "Light energy becomes chemical energy, and the Calvin cycle turns CO2 into glucose",
            SRC,
        );
        assert!(score >= MIN_GROUNDING, "paraphrase should pass, got {score}");
    }

    #[test]
    fn missing_source_is_reported_as_unknown_not_perfect() {
        // Returning 1.0 would silently certify an unchecked pair.
        assert!(grounding_score("anything", "   ").is_nan());
    }

    #[test]
    fn empty_answer_is_vacuously_grounded() {
        assert_eq!(grounding_score("", SRC), 1.0);
    }

    // ── Numeric fabrication ─────────────────────────────────────────────────

    #[test]
    fn catches_invented_numbers() {
        let bad = ungrounded_numbers("The reaction yields 391 units of glucose", SRC);
        assert_eq!(bad, vec!["391"], "a fabricated figure must be flagged");
    }

    #[test]
    fn accepts_numbers_present_in_the_source() {
        let src = "The mass is 42 kilograms.";
        assert!(ungrounded_numbers("The mass is 42 kilograms", src).is_empty());
    }

    #[test]
    fn numeric_check_is_silent_without_a_source() {
        assert!(ungrounded_numbers("value 999", "").is_empty());
    }

    // ── Per-format floors ─-------------------------------------------------------------


    #[test]
    fn floors_reject_the_thin_answers_measured_live() {
        // 12–13 characters passed the old gate; it must not now.
        let thin = "ATP and NADPH.";
        assert!(thin.chars().count() < min_answer_chars(DatasetFormat::SimpleQa));
        assert!(thin.chars().count() < min_answer_chars(DatasetFormat::MultipleChoice));
    }

    #[test]
    fn reasoning_floors_apply_only_where_a_reasoning_block_exists() {
        assert_eq!(min_reasoning_chars(DatasetFormat::SimpleQa), 0);
        assert_eq!(min_reasoning_chars(DatasetFormat::InstructionIo), 0);
        assert!(min_reasoning_chars(DatasetFormat::ReasoningQa) > 0);
        assert!(min_reasoning_chars(DatasetFormat::MultipleChoice) > 0);
    }

    // ── Trainer compatibility ───────────────────────────────────────────────

    #[test]
    fn token_estimate_scales_with_length() {
        assert_eq!(estimate_tokens(""), 0);
        let short = estimate_tokens("hello world");
        let long = estimate_tokens(&"hello world ".repeat(50));
        assert!(long > short * 20, "should scale: {short} -> {long}");
    }

    #[test]
    fn token_estimate_over_counts_rather_than_under() {
        // Under-counting is the dangerous direction: a pair we wrongly bless
        // gets silently truncated by the trainer.
        let text = "The light-dependent reactions produce ATP and NADPH.";
        let estimate = estimate_tokens(text);
        // Real English runs ~4.2 chars/token, so 3.6 chars/token over-counts.
        assert!(
            estimate as f32 >= text.chars().count() as f32 / 4.2,
            "estimate {estimate} must not under-count for {} chars",
            text.chars().count()
        );
    }

    #[test]
    fn example_tokens_includes_reasoning_only_when_present() {
        let without = example_tokens("What fixes CO2?", "The Calvin cycle does it.", "");
        let with = example_tokens(
            "What fixes CO2?",
            "The Calvin cycle does it.",
            &"Because the enzyme RuBisCO catalyses the fixation step. ".repeat(8),
        );
        assert!(with > without, "reasoning must add to the budget");
    }

    #[test]
    fn cutoff_rejects_an_example_that_would_be_truncated() {
        // The real failure this guards: a long reasoning block overruns a
        // default cutoff_len and LLaMA-Factory trims it silently. At ~3.6
        // chars/token, 1024 tokens is roughly 3,700 characters, so the block
        // below is comfortably past it.
        let reasoning = "Because the light reactions generate the ATP and NADPH that the \
                         Calvin cycle then consumes to fix carbon dioxide into glucose. "
            .repeat(30);
        assert!(
            reasoning.chars().count() > 3_700,
            "fixture must actually exceed the cutoff: {} chars",
            reasoning.chars().count()
        );
        assert!(
            !fits_cutoff(
                "Explain how photosynthesis stores energy.",
                "It stores it as glucose.",
                &reasoning,
                1024
            ),
            "a long reasoning block must not fit a 1024 cutoff"
        );
        assert!(fits_cutoff("Short?", "Short.", "", 1024));
    }

    #[test]
    fn cutoff_accepts_the_answers_measured_from_the_live_teacher() {
        // Regression guard on real data: the strengthened prompt produced
        // 202-302 character answers, which must still fit the default cutoff.
        for len in [202, 253, 302] {
            let answer = "a".repeat(len);
            assert!(
                fits_cutoff("A typical question about the material?", &answer, "", 1024),
                "a {len}-char answer should fit the default cutoff"
            );
        }
    }

    #[test]
    fn cutoff_of_zero_means_unset_and_rejects_nothing() {
        // A missing cutoff must not silently discard the whole dataset.
        assert!(fits_cutoff("q", &"a".repeat(10_000), "", 0));
        assert_eq!(truncation_rate(&[("q".into(), "a".into(), "".into())], 0), 0.0);
    }

    #[test]
    fn required_cutoff_reports_the_largest_example() {
        let examples = vec![
            ("short".to_string(), "answer".to_string(), String::new()),
            (
                "a much longer question that goes on for a while".to_string(),
                "x".repeat(400),
                "y".repeat(800),
            ),
        ];
        let need = required_cutoff_len(&examples);
        assert!(need > 300, "should reflect the long example, got {need}");
        assert!(fits_cutoff(&examples[1].0, &examples[1].1, &examples[1].2, need));
    }

    #[test]
    fn truncation_rate_is_a_share_of_examples() {
        let long_reasoning = "z".repeat(4000);
        let examples = vec![
            ("q".to_string(), "a".to_string(), String::new()),
            ("q".to_string(), "a".to_string(), long_reasoning.clone()),
            ("q".to_string(), "a".to_string(), long_reasoning),
            ("q".to_string(), "a".to_string(), String::new()),
        ];
        let rate = truncation_rate(&examples, 512);
        assert!((rate - 0.5).abs() < 1e-6, "expected half to overflow, got {rate}");
    }

    #[test]
    fn empty_example_set_is_well_formed() {
        assert_eq!(required_cutoff_len(&[]), 0);
        assert_eq!(truncation_rate(&[], 1024), 0.0);
    }

    // ── Reporting ───────────────────────────────────────────────────────────

    #[test]
    fn report_tracks_rejections_and_coverage() {
        let mut b = ReportBuilder::new();
        b.generated();
        b.reject("near-duplicate");
        b.generated();
        b.keep(
            "What do the light reactions produce?",
            "They produce ATP and NADPH for the Calvin cycle.",
            "",
            "chunk-1",
            "biology",
            0.9,
        );
        b.generated();
        b.reject("answer-too-short");

        let r = b.finish();
        assert_eq!(r.generated, 3);
        assert_eq!(r.kept, 1);
        assert_eq!(r.rejected.get("near-duplicate"), Some(&1));
        assert_eq!(r.rejected.get("answer-too-short"), Some(&1));
        assert_eq!(r.distinct_sources, 1);
        assert_eq!(r.distinct_topics, 1);
        assert!(r.mean_answer_chars > 0.0);
    }

    #[test]
    fn duplicate_rate_is_a_share_of_generated() {
        let mut b = ReportBuilder::new();
        for _ in 0..4 {
            b.generated();
        }
        b.reject("near-duplicate");
        let r = b.finish();
        assert!((r.duplicate_rate() - 0.25).abs() < 1e-6, "got {}", r.duplicate_rate());
    }

    #[test]
    fn question_diversity_ignores_punctuation_only_changes() {
        let mut b = ReportBuilder::new();
        b.generated();
        b.keep("What fixes CO2?", "The Calvin cycle.", "", "c1", "t", 1.0);
        b.generated();
        b.keep("What fixes CO2!", "The Calvin cycle.", "", "c1", "t", 1.0);
        let r = b.finish();
        assert_eq!(r.kept, 2);
        assert!(
            (r.question_diversity - 0.5).abs() < 1e-6,
            "the two are the same question, got {}",
            r.question_diversity
        );
    }

    #[test]
    fn empty_report_is_well_formed() {
        let r = ReportBuilder::new().finish();
        assert_eq!(r.kept, 0);
        assert_eq!(r.duplicate_rate(), 0.0);
        assert!(r.summary().contains("kept 0/0"));
    }

    #[test]
    fn summary_names_the_rejection_reasons() {
        let mut b = ReportBuilder::new();
        b.generated();
        b.reject("not-grounded");
        let s = b.finish().summary();
        assert!(s.contains("not-grounded"), "summary must explain: {s}");
    }

    #[test]
    fn report_round_trips_through_json() {
        let mut b = ReportBuilder::new();
        b.generated();
        b.keep("Q?", "A long enough answer to matter here.", "", "c1", "t", 0.8);
        let r = b.finish();
        let text = serde_json::to_string(&r).expect("serialize");
        let back: DatasetReport = serde_json::from_str(&text).expect("deserialize");
        assert_eq!(back.kept, 1);
        assert_eq!(back.distinct_sources, 1);
    }
}
