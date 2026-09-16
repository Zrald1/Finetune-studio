//! PDF report generation for benchmark history.
//!
//! Deliberately dependency-free: the report is plain text in Helvetica, which
//! every PDF reader has built in, so there is no font embedding and no new
//! crate to audit. The writer emits the PDF container by hand and the tests
//! read the result back with the `pdf-extract` crate the app already uses, so
//! a malformed file fails the build rather than the user's PDF viewer.

use crate::bench_store::{BenchConfig, BenchMetrics, BenchRecord};

const PAGE_W: f32 = 595.0; // A4 portrait, in points
const PAGE_H: f32 = 842.0;
const MARGIN: f32 = 42.0;
const LINE_H: f32 = 13.0;

/// Escape the three characters that are special inside a PDF string literal.
fn esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '(' => out.push_str("\\("),
            ')' => out.push_str("\\)"),
            '\\' => out.push_str("\\\\"),
            // Helvetica is a Latin-1 font; anything outside that range is
            // dropped rather than emitted as a broken glyph.
            c if (c as u32) < 256 => out.push(c),
            _ => out.push('?'),
        }
    }
    out
}

fn fmt_opt(v: Option<f64>, unit: &str, decimals: usize) -> String {
    match v {
        Some(n) => format!("{n:.decimals$}{unit}"),
        None => "—".to_string(),
    }
}

fn fmt_f32(v: Option<f32>, unit: &str) -> String {
    match v {
        Some(n) => format!("{n}{unit}"),
        None => "—".to_string(),
    }
}

fn fmt_u32(v: Option<u32>, unit: &str) -> String {
    match v {
        Some(n) => format!("{n}{unit}"),
        None => "—".to_string(),
    }
}

fn fmt_str(v: Option<&String>) -> String {
    match v.map(|s| s.trim()).filter(|s| !s.is_empty()) {
        Some(s) => s.to_string(),
        None => "—".to_string(),
    }
}

fn fmt_bool(v: Option<bool>) -> String {
    match v {
        Some(true) => "enabled".to_string(),
        Some(false) => "disabled".to_string(),
        None => "—".to_string(),
    }
}

/// Label/value pairs describing the configuration a benchmark ran under.
pub fn config_rows(c: &BenchConfig) -> Vec<(String, String)> {
    let mut rows = vec![
        ("Serving profile".into(), fmt_str(c.serving_profile.as_ref())),
        ("Precision (dtype)".into(), fmt_str(c.dtype.as_ref())),
        ("Context length".into(), fmt_u32(c.max_model_len, " tokens")),
        ("GPU memory budget".into(), fmt_f32(c.gpu_memory_utilization, "")),
        ("Max sequences".into(), fmt_u32(c.max_num_seqs, "")),
        ("Batched tokens".into(), fmt_u32(c.max_num_batched_tokens, "")),
        ("Tensor parallel".into(), fmt_u32(c.tensor_parallel, "")),
        ("KV cache dtype".into(), fmt_str(c.kv_cache_dtype.as_ref())),
        ("Prefix caching".into(), fmt_bool(c.prefix_caching)),
        ("Quantization".into(), fmt_str(c.quantization.as_ref())),
        ("Reasoning parser".into(), fmt_str(c.reasoning_parser.as_ref())),
        ("Thinking effort".into(), fmt_str(c.reasoning_effort.as_ref())),
    ];
    for note in &c.notes {
        rows.push(("Note".into(), note.clone()));
    }
    rows
}

/// Label/value pairs for the measured metrics.
pub fn metric_rows(m: &BenchMetrics) -> Vec<(String, String)> {
    let mut rows = vec![
        ("TTFT (mean)".into(), fmt_opt(m.ttft_ms, " ms", 0)),
        ("TTFT (median)".into(), fmt_opt(m.median_ttft_ms, " ms", 0)),
        ("TTFT (p95)".into(), fmt_opt(m.p95_ttft_ms, " ms", 0)),
        ("TPOT".into(), fmt_opt(m.tpot_ms, " ms", 1)),
        ("ITL".into(), fmt_opt(m.itl_ms, " ms", 1)),
        ("End-to-end latency".into(), fmt_opt(m.e2el_ms, " ms", 0)),
        ("Output throughput".into(), fmt_opt(m.output_tokens_per_s, " tok/s", 1)),
        ("Total throughput".into(), fmt_opt(m.total_tokens_per_s, " tok/s", 1)),
        ("Request throughput".into(), fmt_opt(m.request_throughput, " req/s", 2)),
        ("Accuracy".into(), fmt_opt(m.accuracy, " %", 2)),
        ("Concurrency".into(), fmt_u32(m.concurrency, "")),
        ("Samples".into(), fmt_u32(m.samples, "")),
        ("Output tokens".into(), fmt_u32(m.total_output_tokens, "")),
    ];
    rows.retain(|(_, v)| v != "—");
    rows
}

/// Percentage change from `base` to `now`. Positive means the value rose.
fn pct_change(base: Option<f64>, now: Option<f64>) -> Option<f64> {
    let (b, n) = (base?, now?);
    if b.abs() < f64::EPSILON {
        return None;
    }
    Some((n - b) / b * 100.0)
}

fn fmt_delta(base: Option<f64>, now: Option<f64>, lower_is_better: bool, unit: &str) -> String {
    match pct_change(base, now) {
        Some(p) => {
            let improved = if lower_is_better { p < 0.0 } else { p > 0.0 };
            let arrow = if p.abs() < 0.05 {
                "="
            } else if improved {
                "IMPROVED"
            } else {
                "REGRESSED"
            };
            format!("{p:+.1}%{unit} ({arrow})")
        }
        None => "—".to_string(),
    }
}

/// Build the comparison table between the earliest and latest record of the
/// same kind+model — the "Standard vs Optimized" view.
pub fn comparison_rows(records: &[BenchRecord]) -> Vec<Vec<String>> {
    let mut out = Vec::new();
    let mut seen: Vec<(String, String)> = Vec::new();
    for r in records {
        let key = (r.kind.clone(), r.model.clone());
        if !seen.contains(&key) {
            seen.push(key);
        }
    }
    for (kind, model) in seen {
        let group: Vec<&BenchRecord> = records
            .iter()
            .filter(|r| r.kind == kind && r.model == model)
            .collect();
        if group.len() < 2 {
            continue;
        }
        let first = group.first().unwrap();
        let last = group.last().unwrap();
        let short = model.rsplit('/').next().unwrap_or(&model);
        out.push(vec![
            format!("{short} ({kind})"),
            format!("{} → {}", first.label, last.label),
            fmt_delta(first.metrics.ttft_ms, last.metrics.ttft_ms, true, ""),
            fmt_delta(
                first.metrics.output_tokens_per_s,
                last.metrics.output_tokens_per_s,
                false,
                "",
            ),
            fmt_delta(
                first.metrics.tpot_ms,
                last.metrics.tpot_ms,
                true,
                "",
            ),
            fmt_delta(first.metrics.accuracy, last.metrics.accuracy, false, ""),
        ]);
    }
    out
}

// ── Minimal PDF container ────────────────────────────────────────────────────

struct Page {
    ops: String,
    y: f32,
}

impl Page {
    fn new() -> Self {
        Self {
            ops: String::new(),
            y: PAGE_H - MARGIN,
        }
    }

    fn remaining(&self) -> f32 {
        self.y - MARGIN
    }

    fn text(&mut self, x: f32, size: f32, bold: bool, text: &str) {
        let font = if bold { "/F2" } else { "/F1" };
        self.ops.push_str(&format!(
            "BT {font} {size} Tf {x:.2} {:.2} Td ({}) Tj ET\n",
            self.y,
            esc(text)
        ));
    }

    /// Draw a line and advance. Wraps at `max_chars` rather than measuring
    /// glyph widths — good enough for a fixed-pitch report and far simpler.
    fn line(&mut self, text: &str, size: f32, bold: bool, indent: f32, max_chars: usize) {
        for chunk in wrap(text, max_chars) {
            if self.remaining() < LINE_H {
                return;
            }
            self.text(MARGIN + indent, size, bold, &chunk);
            self.y -= LINE_H;
        }
    }

    fn rule(&mut self) {
        if self.remaining() < LINE_H {
            return;
        }
        self.ops.push_str(&format!(
            "0.75 w 0.7 0.7 0.7 RG {:.2} {:.2} m {:.2} {:.2} l S\n",
            MARGIN,
            self.y + 4.0,
            PAGE_W - MARGIN,
            self.y + 4.0
        ));
        self.y -= 6.0;
    }

    fn gap(&mut self, pts: f32) {
        self.y -= pts;
    }
}

fn wrap(text: &str, max_chars: usize) -> Vec<String> {
    if text.is_empty() {
        return vec![String::new()];
    }
    let mut lines = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        if current.is_empty() {
            current = word.to_string();
        } else if current.chars().count() + 1 + word.chars().count() <= max_chars {
            current.push(' ');
            current.push_str(word);
        } else {
            lines.push(std::mem::take(&mut current));
            current = word.to_string();
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines
}

fn build_pdf(pages: &[Page]) -> Vec<u8> {
    // Object 1 = catalog, 2 = pages, 3/4 = fonts, then per page: page + content.
    let first_page_obj = 5;
    let page_count = pages.len().max(1);

    let mut objects: Vec<String> = Vec::new();
    objects.push("<< /Type /Catalog /Pages 2 0 R >>".to_string());

    let kids: Vec<String> = (0..page_count)
        .map(|i| format!("{} 0 R", first_page_obj + i * 2))
        .collect();
    objects.push(format!(
        "<< /Type /Pages /Kids [{}] /Count {page_count} >>",
        kids.join(" ")
    ));
    objects.push("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>".to_string());
    objects.push("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>".to_string());

    for (i, page) in pages.iter().enumerate() {
        let content_obj = first_page_obj + i * 2 + 1;
        objects.push(format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] \
             /Contents {content_obj} 0 R /Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> >>"
        ));
        let mut stream = page.ops.clone();
        if stream.is_empty() {
            stream.push('\n');
        }
        objects.push(format!(
            "<< /Length {} >>\nstream\n{}endstream",
            stream.len(),
            stream
        ));
    }

    let mut pdf = String::from("%PDF-1.4\n");
    let mut offsets = Vec::with_capacity(objects.len());
    for (i, body) in objects.iter().enumerate() {
        offsets.push(pdf.len());
        pdf.push_str(&format!("{} 0 obj\n{body}\nendobj\n", i + 1));
    }

    let xref_at = pdf.len();
    pdf.push_str(&format!("xref\n0 {}\n", objects.len() + 1));
    pdf.push_str("0000000000 65535 f \n");
    for off in &offsets {
        pdf.push_str(&format!("{off:010} 00000 n \n"));
    }
    pdf.push_str(&format!(
        "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n",
        objects.len() + 1
    ));

    pdf.into_bytes()
}

/// Render the full benchmark report.
pub fn render_report(records: &[BenchRecord], generated_at: &str) -> Vec<u8> {
    let mut pages: Vec<Page> = Vec::new();
    let mut page = Page::new();

    page.text(MARGIN, 19.0, true, "Fine-Tune Studio — Benchmark Report");
    page.y -= 22.0;
    page.line(
        &format!("Generated {generated_at} · {} record(s)", records.len()),
        9.0,
        false,
        0.0,
        110,
    );
    page.gap(6.0);
    page.rule();
    page.gap(8.0);

    if records.is_empty() {
        page.line("No benchmarks recorded yet.", 10.0, false, 0.0, 110);
    }

    // ── Comparison summary first: it is the reason to open the report ──────
    let comparisons = comparison_rows(records);
    if !comparisons.is_empty() {
        page.line("Optimization summary", 13.0, true, 0.0, 110);
        page.gap(2.0);
        page.line(
            "Earliest vs latest run of the same model. TTFT/TPOT: lower is better. Throughput/accuracy: higher is better.",
            8.0,
            false,
            0.0,
            130,
        );
        page.gap(4.0);
        for row in &comparisons {
            page.line(&row[0], 9.5, true, 0.0, 120);
            page.line(&format!("    {}", row[1]), 9.0, false, 0.0, 120);
            page.line(
                &format!(
                    "    TTFT {}    Throughput {}    TPOT {}    Accuracy {}",
                    row[2], row[3], row[4], row[5]
                ),
                9.0,
                false,
                0.0,
                130,
            );
            page.gap(5.0);
        }
        page.rule();
        page.gap(8.0);
    }

    // ── One section per record ────────────────────────────────────────────
    for (i, r) in records.iter().enumerate() {
        if page.remaining() < 170.0 {
            pages.push(page);
            page = Page::new();
        }
        page.line(&format!("{}. {}", i + 1, r.label), 12.0, true, 0.0, 110);
        page.line(
            &format!("{} · {} · {}", r.kind, r.model, r.captured_at),
            8.5,
            false,
            0.0,
            130,
        );
        if !r.endpoint.is_empty() {
            page.line(&format!("endpoint {}", r.endpoint), 8.5, false, 0.0, 130);
        }
        page.gap(3.0);

        page.line("Configuration", 10.0, true, 0.0, 110);
        for (k, v) in config_rows(&r.config) {
            page.line(&format!("{k}: {v}"), 9.0, false, 10.0, 110);
        }
        page.gap(3.0);

        let metrics = metric_rows(&r.metrics);
        page.line("Measured", 10.0, true, 0.0, 110);
        if metrics.is_empty() {
            page.line("(no metrics)", 9.0, false, 10.0, 110);
        }
        for (k, v) in metrics {
            page.line(&format!("{k}: {v}"), 9.0, false, 10.0, 110);
        }
        page.gap(8.0);
        page.rule();
        page.gap(8.0);
    }

    pages.push(page);
    build_pdf(&pages)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench_store::{BenchConfig, BenchMetrics, BenchRecord, KIND_TEACHER};

    fn rec(label: &str, profile: &str, ttft: f64, tps: f64) -> BenchRecord {
        BenchRecord {
            id: label.to_string(),
            kind: KIND_TEACHER.to_string(),
            label: label.to_string(),
            model: "Qwen/Qwen3.8-27B".to_string(),
            endpoint: "http://127.0.0.1:44319".to_string(),
            config: BenchConfig {
                serving_profile: Some(profile.to_string()),
                max_model_len: Some(262144),
                kv_cache_dtype: Some("fp8".to_string()),
                ..Default::default()
            },
            metrics: BenchMetrics {
                ttft_ms: Some(ttft),
                output_tokens_per_s: Some(tps),
                tpot_ms: Some(18.9),
                ..Default::default()
            },
            captured_at: "2026-09-16T00:00:00Z".to_string(),
        }
    }

    #[test]
    fn escapes_pdf_string_delimiters() {
        assert_eq!(esc("a(b)c"), "a\\(b\\)c");
        assert_eq!(esc("back\\slash"), "back\\\\slash");
    }

    #[test]
    fn wraps_long_text_without_splitting_words() {
        let lines = wrap("alpha beta gamma delta", 11);
        assert!(lines.iter().all(|l| l.chars().count() <= 11), "{lines:?}");
        assert_eq!(lines.join(" "), "alpha beta gamma delta");
    }

    #[test]
    fn produces_a_parseable_pdf() {
        let pdf = render_report(&[rec("Standard", "standard", 1398.0, 47.5)], "now");
        assert!(pdf.starts_with(b"%PDF-1.4"), "missing PDF magic");
        assert!(pdf.ends_with(b"%%EOF\n"), "missing EOF marker");
        let text = String::from_utf8_lossy(&pdf);
        assert!(text.contains("xref"), "missing xref table");
        assert!(text.contains("/Root 1 0 R"), "missing trailer root");
    }

    #[test]
    fn renders_an_empty_history_without_panicking() {
        let pdf = render_report(&[], "now");
        assert!(pdf.starts_with(b"%PDF-1.4"));
        assert!(!pdf.is_empty());
    }

    #[test]
    fn reports_improvement_direction_correctly() {
        // Lower TTFT is better; higher throughput is better.
        assert!(fmt_delta(Some(1000.0), Some(500.0), true, "").contains("IMPROVED"));
        assert!(fmt_delta(Some(1000.0), Some(1500.0), true, "").contains("REGRESSED"));
        assert!(fmt_delta(Some(40.0), Some(80.0), false, "").contains("IMPROVED"));
        assert!(fmt_delta(Some(80.0), Some(40.0), false, "").contains("REGRESSED"));
        assert_eq!(fmt_delta(None, Some(1.0), true, ""), "—");
    }

    #[test]
    fn comparison_needs_at_least_two_runs() {
        let single = vec![rec("Standard", "standard", 1398.0, 47.5)];
        assert!(comparison_rows(&single).is_empty());

        let two = vec![
            rec("Standard", "standard", 1398.0, 47.5),
            rec("Optimized", "optimized", 387.0, 94.2),
        ];
        let rows = comparison_rows(&two);
        assert_eq!(rows.len(), 1);
        assert!(rows[0][2].contains("IMPROVED"), "TTFT should improve: {:?}", rows[0]);
        assert!(rows[0][3].contains("IMPROVED"), "throughput should improve");
    }

    #[test]
    fn config_rows_omit_nothing_and_use_dashes_for_unset() {
        let rows = config_rows(&BenchConfig::default());
        assert!(rows.iter().all(|(_, v)| !v.is_empty()));
        assert!(rows.iter().any(|(_, v)| v == "—"));
    }

    #[test]
    fn metric_rows_skip_unset_metrics() {
        let rows = metric_rows(&BenchMetrics {
            ttft_ms: Some(100.0),
            ..Default::default()
        });
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].0, "TTFT (mean)");
    }

    #[test]
    fn generated_pdf_is_readable_by_a_real_pdf_parser() {
        // The strongest check available without opening a viewer: hand the file
        // to the same crate the app uses to READ PDFs. If the container, xref
        // table, or stream lengths were wrong, extraction would fail or come
        // back empty.
        let records = vec![
            rec("Teacher — Standard", "standard", 1398.0, 47.5),
            rec("Teacher — Optimized", "optimized", 387.0, 94.2),
        ];
        let pdf = render_report(&records, "2026-09-16T00:00:00Z");

        let dir = std::env::temp_dir().join("ft_bench_pdf_test");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("report.pdf");
        std::fs::write(&path, &pdf).expect("write pdf");

        let text = pdf_extract::extract_text(&path).expect("generated PDF must parse");
        assert!(
            text.contains("Benchmark Report"),
            "title missing from extracted text: {text:.400}"
        );
        assert!(text.contains("Qwen3.8-27B"), "model missing: {text:.400}");
        assert!(text.contains("Optimization summary"), "comparison missing");
        // The comparison must actually report the improvement, not just exist.
        assert!(
            text.contains("IMPROVED"),
            "expected an improvement verdict in: {text:.600}"
        );
    }

    #[test]
    fn paginates_long_histories() {
        let many: Vec<BenchRecord> = (0..40)
            .map(|i| rec(&format!("run {i}"), "standard", 1000.0, 40.0))
            .collect();
        let pdf = render_report(&many, "now");
        let text = String::from_utf8_lossy(&pdf);
        let page_objs = text.matches("/Type /Page ").count();
        assert!(page_objs > 1, "40 records should span multiple pages, got {page_objs}");
    }
}
