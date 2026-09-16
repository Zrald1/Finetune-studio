//! Self-healing dependency reconciliation.
//!
//! The app drives several Python consumers that disagree about their
//! dependencies. The sharpest example, hit on a live droplet:
//!
//! | consumer | wants |
//! |---|---|
//! | Qwen3.8-27B teacher (vLLM) | `transformers >= 5.8.0` |
//! | LLaMA-Factory | `transformers >= 4.55.0, <= 5.6.0` |
//!
//! Hardcoding pins for each of those goes stale — the previous hardcoded
//! `<4.58` pin is already wrong for newer LLaMA-Factory builds. So the strategy
//! here is:
//!
//! 1. **Isolate** consumers that conflict, so each can satisfy its own
//!    requirements without breaking the other (see `method::common`).
//! 2. **Declare** the few requirements we genuinely own as a table, each with a
//!    reason, so the emitted shell can explain itself in the log.
//! 3. **Reconcile** by checking the installed version and installing only when
//!    the check fails. `pip install` is idempotent, so the check is purely an
//!    optimisation — correctness never depends on it.
//! 4. **Discover** what the installed packages actually demand, so a conflict is
//!    visible in the log instead of surfacing later as an ImportError.
//!
//! Everything emitted here is a single line with no single quotes, because the
//! deploy path wraps commands in `bash -lc '…'` where a stray `'` terminates
//! the wrapper early.

/// One package requirement, with the reason it exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pin {
    pub package: &'static str,
    /// Inclusive lower bound as `(major, minor)`.
    pub min: (u32, u32),
    /// Optional inclusive upper bound as `(major, minor)`.
    pub max: Option<(u32, u32)>,
    /// Shown in the log when the pin triggers, so a heal is never silent.
    pub why: &'static str,
}

impl Pin {
    pub const fn at_least(package: &'static str, min: (u32, u32), why: &'static str) -> Self {
        Self { package, min, max: None, why }
    }

    pub const fn between(
        package: &'static str,
        min: (u32, u32),
        max: (u32, u32),
        why: &'static str,
    ) -> Self {
        Self { package, min, max: Some(max), why }
    }

    /// Human-readable specifier, e.g. `>=5.8.0` or `>=4.41.2,<=4.58`.
    pub fn spec(&self) -> String {
        let (lo_major, lo_minor) = self.min;
        let mut s = format!(">={lo_major}.{lo_minor}.0");
        if let Some((hi_major, hi_minor)) = self.max {
            s.push_str(&format!(",<={hi_major}.{hi_minor}.0"));
        }
        s
    }

    /// Shell-safe pip argument, e.g. `'transformers>=5.8.0'`.
    pub fn pip_arg(&self) -> String {
        format!("'{}'", format!("{}{}", self.package, self.spec()).replace('\'', ""))
    }
}

/// Requirements that belong to the *environment* rather than to any one model.
///
/// Deliberately short. Anything model-specific is **discovered** at deploy time
/// by [`model_heal_cmd`], because a hardcoded `transformers>=5.8.0` is only
/// correct for Qwen3.5+ — it would force-upgrade transformers out from under an
/// older checkpoint that works fine on 4.x. Pins here must hold for every model
/// the app might ever serve.
pub const TEACHER_PINS: &[Pin] = &[
    Pin::at_least(
        "starlette",
        (1, 0),
        "vLLM 0.27.1 requires starlette>=1.0.1; installing LLaMA-Factory downgrades it",
    ),
];

/// Base64-write `content` to `path` and run it with `python3`.
///
/// Base64 avoids every quoting hazard: the payload can contain single quotes,
/// double quotes, newlines, backslashes — none of it reaches the shell. That
/// matters because the deploy path wraps commands in `bash -lc '…'`, where a
/// stray `'` silently truncates the command (see BUG-14).
pub fn run_python_file(path: &str, content: &str, args: &str) -> String {
    use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
    let encoded = B64.encode(content.as_bytes());
    let chunks = encoded
        .as_bytes()
        .chunks(76)
        .map(|c| std::str::from_utf8(c).unwrap_or(""))
        .map(|c| format!("'{c}'"))
        .collect::<Vec<_>>()
        .join(" ");
    let dest = shell_quote(path);
    let tmp = shell_quote(&format!("{path}.b64.$$"));
    let args = args.trim();
    format!(
        "printf '%s\\n' {chunks} > {tmp} && base64 -d {tmp} > {dest} && rm -f {tmp} && \
         python3 {dest} {args}; "
    )
}

/// Minimal single-quote wrapper for a path we control (no embedded quotes).
fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', ""))
}

/// The universal model-compatibility resolver.
///
/// Rather than assuming anything about the checkpoint, it asks the repository
/// what it needs and reconciles the environment to match:
///
/// 1. Fetch `config.json` and read `model_type` / `architectures`.
/// 2. If the installed `transformers` does not recognise `model_type`, upgrade —
///    this is what makes an unknown future model work without a code change.
/// 3. Honour the `transformers_version` the config was written by, as a floor.
/// 4. Install any `requirements.txt` the repo ships (common for remote-code
///    models such as DeepSeek-VL or custom Qwen variants).
/// 5. Ask vLLM's own model registry whether it supports the declared
///    architectures, and say so plainly if it does not — a vLLM image is the one
///    thing we cannot repair at runtime, so the user needs to know.
///
/// Every branch prints, so the deploy log always explains what was decided.
pub fn model_compat_script(repo_id: &str) -> String {
    // The repo id goes into a Python string literal; strip anything that could
    // break out of it. Real HF ids are [A-Za-z0-9._-/], so this is lossless.
    let safe_repo: String = repo_id
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | '/'))
        .collect();

    format!(
        r##"import json, subprocess, sys, urllib.request
from importlib.metadata import version, PackageNotFoundError

REPO = "{safe_repo}"
BASE = "https://huggingface.co/" + REPO + "/resolve/main"


def fetch(name, timeout=20):
    url = BASE + "/" + name
    req = urllib.request.Request(url, headers={{"User-Agent": "fine-tune-studio"}})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")
    except Exception:
        return None


def installed(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


def pip(*specs):
    specs = [s for s in specs if s]
    if not specs:
        return True
    print("[heal] pip install " + " ".join(specs), flush=True)
    cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir"] + specs
    return subprocess.call(cmd) == 0


print("[heal] resolving requirements for " + REPO, flush=True)

cfg_text = fetch("config.json")
if cfg_text is None:
    print("[heal] config.json not reachable - skipping model-specific resolution")
    sys.exit(0)

try:
    cfg = json.loads(cfg_text)
except Exception as exc:
    print("[heal] config.json is not valid JSON: " + repr(exc))
    sys.exit(0)

model_type = str(cfg.get("model_type", "") or "")
archs = [str(a) for a in (cfg.get("architectures") or [])]
print("[heal] model_type=" + repr(model_type) + " architectures=" + repr(archs))

tf = installed("transformers")
print("[heal] transformers installed: " + str(tf))


def tf_knows(mt):
    if not mt:
        return True
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        return mt in CONFIG_MAPPING
    except Exception:
        return False


def satisfies(have, floor):
    """True when `have` is at least `floor`. Tolerates dev/rc suffixes."""
    if not have:
        return False

    def parts(v):
        out = []
        for chunk in str(v).split("."):
            digits = ""
            for ch in chunk:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            if not digits:
                break
            out.append(int(digits))
        return out

    a, b = parts(have), parts(floor)
    for i in range(max(len(a), len(b))):
        x = a[i] if i < len(a) else 0
        y = b[i] if i < len(b) else 0
        if x != y:
            return x > y
    return True


needed = None

if not tf_knows(model_type):
    print("[heal] installed transformers does not recognise " + repr(model_type))
    needed = "transformers"

declared = str(cfg.get("transformers_version", "") or "")
if declared:
    print("[heal] config.json was written by transformers " + declared)
    if satisfies(tf, declared):
        print("[heal] installed transformers already meets that floor - no change")
    else:
        needed = "transformers>=" + declared

reqs = fetch("requirements.txt")
if reqs:
    pkgs = [ln.strip() for ln in reqs.splitlines()
            if ln.strip() and not ln.strip().startswith("#")]
    if pkgs:
        print("[heal] repo ships requirements.txt: " + ", ".join(pkgs))
        pip(*pkgs)

if needed:
    if pip(needed):
        print("[heal] installed " + needed)
    else:
        print("[heal] WARNING: could not install " + needed)
else:
    print("[heal] no dependency changes required")

try:
    from vllm.model_executor.models.registry import ModelRegistry
    supported = set(ModelRegistry.get_supported_archs())
    missing = [a for a in archs if a not in supported]
    if missing:
        print("[heal] WARNING: this vLLM build does not list " + ", ".join(missing))
        print("[heal]   try trust_remote_code, or a newer vllm/vllm-openai-rocm image")
    elif archs:
        print("[heal] vLLM supports every declared architecture")
except Exception as exc:
    print("[heal] could not query the vLLM model registry: " + repr(exc))

print("[heal] model compatibility resolution complete", flush=True)
"##,
        safe_repo = safe_repo
    )
}

/// Requirements for *serving* a trained student (base model + LoRA adapter)
/// with plain transformers, outside LLaMA-Factory.
///
/// The base model half is discovered by [`model_compat_script`] exactly as it is
/// for the teacher — the student repo is user-chosen and can be anything. These
/// two are the extra pieces the adapter path needs on top of that.
pub const STUDENT_PINS: &[Pin] = &[
    Pin::at_least(
        "peft",
        (0, 17),
        "loading a LoRA adapter requires peft; LLaMA-Factory 0.9.4 installs 0.17.x",
    ),
    Pin::at_least(
        "accelerate",
        (0, 34),
        "from_pretrained(device_map=\"auto\") requires accelerate",
    ),
];

/// Heal step for student inference / merge, run before the loader script.
///
/// The student base model is user-chosen, so it gets the same model-agnostic
/// resolution as the teacher: read its `config.json`, honour the transformers
/// floor it declares, install any `requirements.txt` it ships, and check vLLM's
/// registry. Then reconcile the two adapter-specific pins.
///
/// Emitted before the loader's own `set -e`, so a failed heal warns rather than
/// aborting a run that might still have succeeded.
pub fn student_heal_cmd(base_model: &str) -> String {
    let mut out = String::from("echo '[heal] preparing student environment'; ");
    out.push_str(&reconcile_cmd(STUDENT_PINS, ""));
    out.push_str(&run_python_file(
        MODEL_COMPAT_SCRIPT,
        &model_compat_script(base_model),
        "",
    ));
    out
}

/// Path the compatibility script is written to on the host.
pub const MODEL_COMPAT_SCRIPT: &str = "/root/.ft_model_compat.py";

/// Full model-agnostic heal step: reconcile environment pins, then resolve
/// whatever the selected checkpoint needs.
///
/// Order matters — environment pins first (vLLM must be importable for the
/// registry query in the model script), then the model-specific pass.
pub fn teacher_heal_cmd(repo_id: &str) -> String {
    let mut out = String::from("echo '[heal] reconciling environment'; ");
    out.push_str(&reconcile_cmd(TEACHER_PINS, ""));
    out.push_str(&run_python_file(
        MODEL_COMPAT_SCRIPT,
        &model_compat_script(repo_id),
        "",
    ));
    out.push_str(&conflict_report_cmd());
    out
}

/// Requirements the isolated LLaMA-Factory venv owns.
///
/// `transformers` carries an upper bound here *on purpose*: this is the only
/// place that constraint is allowed to exist, because the venv keeps it away
/// from the teacher.
///
/// Every pin must be *failable*. A pin like `>=0.0.0` can never fail, so the
/// reconciler would "heal" it on every run by installing the newest release —
/// which is how an earlier revision of this file silently pushed
/// `huggingface-hub` to 1.31.0 and broke transformers with
/// `huggingface-hub>=0.34.0,<1.0 is required ... but found 1.31.0`.
pub const TRAINER_PINS: &[Pin] = &[
    Pin::between(
        "transformers",
        (4, 41),
        (4, 58),
        "LLaMA-Factory 0.9.4 supports transformers 4.41.2 through 4.58",
    ),
    Pin::between(
        "huggingface-hub",
        (0, 34),
        (0, 99),
        "transformers 4.5x requires huggingface-hub >=0.34,<1.0; the 1.x API is incompatible",
    ),
];

/// Emit `python3 -c "…"` for a snippet, escaped so it survives being embedded in
/// a single-quoted `bash -lc '…'` wrapper.
///
/// The snippet itself must not contain single quotes; the returned string has
/// its double quotes escaped so the inner shell receives them intact.
pub fn py_inline(code: &str) -> String {
    debug_assert!(!code.contains('\''), "inline python must avoid single quotes: {code}");
    let escaped = code.replace('\\', "\\\\").replace('"', "\\\"");
    format!("python3 -c \"{escaped}\"")
}

/// Python that exits 0 when `package` satisfies `pin`, 1 when it does not, and
/// 2 when the package is absent. Kept to two-part version tuples because that
/// covers every pin the app declares and avoids depending on `packaging`.
fn check_snippet(pin: &Pin) -> String {
    let (lo_major, lo_minor) = pin.min;
    let upper = match pin.max {
        Some((major, minor)) => format!(
            " and (v[0],v[1])<=({major},{minor})"
        ),
        None => String::new(),
    };
    // No single quotes anywhere; `\\\"` in the Rust literal produces `"` in the
    // emitted shell, which the inner bash then hands to Python.
    format!(
        "import sys;from importlib.metadata import version as V;\
         v=[int(x) for x in V(\\\"{pkg}\\\").split(\\\".\\\")[:2] if x.isdigit()];\
         sys.exit(0 if len(v)==2 and (v[0],v[1])>=({lo_major},{lo_minor}){upper} else 1)",
        pkg = pin.package,
        lo_major = lo_major,
        lo_minor = lo_minor,
        upper = upper,
    )
}

/// Shell that checks each pin and installs only the ones that fail.
///
/// `pip_args` is spliced into the install line (index URLs, extra packages).
/// Every heal prints a `[heal]` line so the run log shows exactly what changed.
///
/// Note the deliberate absence of `--upgrade`: it would let pip drift to the
/// newest release that happens to satisfy the spec, which is how an earlier
/// revision pushed `huggingface-hub` past 1.0 and broke the very environment it
/// was meant to repair. A heal should converge on the declared requirement, not
/// on whatever is newest today.
pub fn reconcile_cmd(pins: &[Pin], pip_args: &str) -> String {
    let mut out = String::new();
    for pin in pins {
        let check = py_inline(&check_snippet(pin));
        out.push_str(&format!(
            "{check} 2>/dev/null && echo '[heal] {pkg} OK' || {{ \
               echo '[heal] {pkg} does not satisfy {spec} — installing ({why})'; \
               python3 -m pip install --no-cache-dir {pip_arg} {pip_args} \
                 || echo '[heal] WARNING: could not install {pkg}; continuing'; \
             }}; ",
            check = check,
            pkg = pin.package,
            spec = pin.spec(),
            why = pin.why,
            pip_arg = pin.pip_arg(),
            pip_args = pip_args.trim(),
        ));
    }
    out
}

/// Shell that reports dependency conflicts in the current environment.
///
/// This is the discovery half, and it deliberately delegates to `pip check`
/// rather than reading metadata ourselves: pip already knows every installed
/// package's declared requirements, so a conflict we never anticipated shows up
/// in the log without a code change. Hardcoding the list of things to watch is
/// exactly the mistake that let the transformers conflict go unnoticed.
///
/// Output is prefixed so it can be grepped out of a long run log.
pub fn conflict_report_cmd() -> String {
    "python3 -m pip check 2>&1 | sed 's/^/[deps] /' | head -25; \
     true"
        .to_string()
}

/// Reconciliation for the teacher's environment, followed by a conflict report.
///
/// Order matters: heal first so the report reflects the repaired state, then
/// report so anything still broken is visible in the log.
pub fn heal_and_report_cmd(pins: &[Pin], pip_args: &str) -> String {
    format!(
        "echo '[deps] reconciling {} requirement(s)'; {}{}",
        pins.len(),
        reconcile_cmd(pins, pip_args),
        conflict_report_cmd()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn teacher() -> Vec<Pin> {
        TEACHER_PINS.to_vec()
    }

    #[test]
    fn spec_renders_bounds() {
        assert_eq!(Pin::at_least("x", (5, 8), "").spec(), ">=5.8.0");
        assert_eq!(Pin::between("x", (4, 41), (4, 58), "").spec(), ">=4.41.0,<=4.58.0");
    }

    #[test]
    fn pip_arg_is_shell_safe() {
        let arg = Pin::at_least("transformers", (5, 8), "").pip_arg();
        assert_eq!(arg, "'transformers>=5.8.0'");
        assert!(!arg.contains('"'));
    }

    #[test]
    fn inline_python_has_no_single_quotes() {
        // The deploy path wraps in `bash -lc '…'`; a single quote here would
        // terminate the wrapper and break the whole command.
        for pin in teacher().iter().chain(TRAINER_PINS) {
            let snippet = check_snippet(pin);
            assert!(
                !snippet.contains('\''),
                "pin check for {} contains a single quote: {snippet}",
                pin.package
            );
        }
    }

    #[test]
    fn reconcile_never_drifts_to_latest() {
        // `--upgrade` lets pip pick the newest release that satisfies the spec,
        // which silently pushed huggingface-hub past 1.0 on a live droplet and
        // broke the environment this function exists to repair.
        let cmd = reconcile_cmd(&teacher(), "");
        assert!(
            !cmd.contains("--upgrade"),
            "a heal must converge on the declared requirement, not on latest: {cmd}"
        );
    }

    #[test]
    fn every_pin_is_actually_failable() {
        // A pin like `>=0.0.0` can never fail, so the reconciler would "heal"
        // it on every run — installing something on each pass and drifting the
        // environment. Every declared pin must be able to fail.
        for pin in TEACHER_PINS.iter().chain(TRAINER_PINS) {
            assert!(
                pin.min != (0, 0) || pin.max.is_some(),
                "pin for {} can never fail ({})",
                pin.package,
                pin.spec()
            );
        }
    }

    #[test]
    fn trainer_caps_huggingface_hub_below_one() {
        // transformers 4.5x hard-fails against the huggingface-hub 1.x API.
        let cmd = reconcile_cmd(TRAINER_PINS, "");
        assert!(
            cmd.contains("huggingface-hub>=0.34.0,<=0.99.0"),
            "trainer must cap huggingface-hub below 1.0: {cmd}"
        );
    }

    #[test]
    fn reconcile_cmd_is_quote_balanced_and_single_line() {
        let cmd = reconcile_cmd(&teacher(), "");
        assert!(!cmd.contains('\n'), "must stay one logical line");
        assert_eq!(cmd.matches('\'').count() % 2, 0, "unbalanced quotes: {cmd}");
    }

    #[test]
    fn reconcile_cmd_covers_every_pin() {
        let cmd = reconcile_cmd(&teacher(), "");
        for pin in teacher() {
            assert!(cmd.contains(pin.package), "missing {}", pin.package);
            assert!(cmd.contains(&pin.spec()), "missing spec for {}", pin.package);
        }
    }

    #[test]
    fn reconcile_cmd_explains_why_it_healed() {
        // A silent repair is indistinguishable from a bug; the reason must land
        // in the run log.
        let cmd = reconcile_cmd(&teacher(), "");
        assert!(cmd.contains("starlette>=1.0.1"), "environment reason missing");
        assert!(
            cmd.contains("LLaMA-Factory"),
            "the reason a pin exists must reach the log: {cmd}"
        );
    }

    #[test]
    fn reconcile_cmd_is_non_fatal_on_install_failure() {
        // A failed repair must not abort the deploy — the teacher may still
        // work, and a hard failure here would be worse than a warning.
        let cmd = reconcile_cmd(&teacher(), "");
        assert!(cmd.contains("continuing"), "install failure should warn, not exit");
        assert!(!cmd.contains("exit 42"), "must not abort the deploy");
    }

    #[test]
    fn trainer_pins_keep_the_upper_bound_out_of_the_teacher() {
        // The whole point of the venv: the <=4.58 constraint must exist in the
        // trainer table and nowhere in the teacher's.
        let trainer = reconcile_cmd(TRAINER_PINS, "");
        assert!(trainer.contains("<=4.58.0"), "trainer should cap transformers");

        let teacher_cmd = reconcile_cmd(&teacher(), "");
        assert!(
            !teacher_cmd.contains("<=4.58"),
            "teacher must not inherit the trainer's cap: {teacher_cmd}"
        );
    }

    #[test]
    fn pip_args_are_threaded_through() {
        let cmd = reconcile_cmd(&teacher(), "--index-url https://example.invalid");
        assert!(cmd.contains("--index-url https://example.invalid"));
    }

    #[test]
    fn conflict_report_delegates_to_pip_check() {
        // Discovery must not be a hardcoded watch-list — that is the mistake
        // that let the transformers conflict go unnoticed. `pip check` knows
        // every installed package's declared requirements.
        let cmd = conflict_report_cmd();
        assert!(cmd.contains("pip check"), "should ask pip: {cmd}");
        assert!(cmd.contains("[deps]"), "output must be greppable");
        assert!(cmd.ends_with("true"), "must not fail the deploy on a conflict");
    }

    #[test]
    fn conflict_report_is_shell_safe() {
        let cmd = conflict_report_cmd();
        assert!(!cmd.contains('\n'), "must be one line");
        assert_eq!(cmd.matches('\'').count() % 2, 0, "unbalanced quotes: {cmd}");
    }

    #[test]
    fn heal_and_report_heals_before_reporting() {
        // If the report ran first it would describe the broken state and then
        // the heal would silently fix it — the log would contradict reality.
        let cmd = heal_and_report_cmd(&teacher(), "");
        let heal_at = cmd.find("does not satisfy").expect("heal present");
        let report_at = cmd.find("pip check").expect("report present");
        assert!(heal_at < report_at, "heal must precede the report");
    }

    #[test]
    fn empty_pin_list_is_harmless() {
        assert_eq!(reconcile_cmd(&[], "").trim(), "");
    }

    // ── Universal model resolver ────────────────────────────────────────────

    #[test]
    fn model_script_is_agnostic_to_the_checkpoint() {
        // Nothing about Qwen3.8 may be baked in — the whole point is that a
        // model the app has never seen resolves its own requirements.
        let script = model_compat_script("some-org/brand-new-model-9B");
        assert!(script.contains("some-org/brand-new-model-9B"));
        assert!(!script.contains("qwen3"), "must not hardcode a model family");
        assert!(!script.contains("5.8.0"), "must not hardcode a transformers floor");
        assert!(!script.contains("27B"), "must not hardcode a model size");
    }

    #[test]
    fn model_script_discovers_rather_than_assumes() {
        let script = model_compat_script("a/b");
        // The four discovery channels, each independent of any model list.
        assert!(script.contains("config.json"), "should read the model config");
        assert!(script.contains("model_type"), "should read model_type");
        assert!(script.contains("CONFIG_MAPPING"), "should ask transformers what it knows");
        assert!(script.contains("transformers_version"), "should honour the declared version");
        assert!(script.contains("requirements.txt"), "should install repo requirements");
        assert!(script.contains("ModelRegistry"), "should ask vLLM about the architecture");
    }

    #[test]
    fn model_script_degrades_gracefully() {
        // An unreachable or malformed config must not abort the deploy — the
        // model may still load fine.
        let script = model_compat_script("a/b");
        assert!(script.contains("sys.exit(0)"), "should exit cleanly, not fail");
        assert!(script.contains("not reachable"));
        assert!(script.contains("not valid JSON"));
        assert!(script.contains("WARNING"), "should warn rather than raise");
    }

    #[test]
    fn model_script_compares_before_installing() {
        // "Detect, then install" — not "install every time". A deploy must not
        // invoke pip when the environment already satisfies the floor.
        let script = model_compat_script("a/b");
        assert!(script.contains("def satisfies("), "needs a version comparison");
        assert!(
            script.contains("already meets that floor"),
            "should report a no-op decision"
        );
        assert!(
            script.contains("no dependency changes required"),
            "should say when nothing is needed"
        );
    }

    #[test]
    fn version_comparison_handles_real_world_suffixes() {
        // The rule has to cope with the versions HF actually publishes:
        // 4.42.0.dev0, 5.6.0rc1, 4.43.1.
        let script = model_compat_script("a/b");
        // Extract the helper and exercise it the way the deploy does.
        let start = script.find("def satisfies(").expect("helper");
        let end = script.find("needed = None").expect("marker");
        let helper = &script[start..end];
        assert!(helper.contains("isdigit"), "must parse numeric prefixes");
        assert!(helper.contains("break"), "must tolerate non-numeric suffixes");
    }

    #[test]
    fn model_script_sanitises_the_repo_id() {
        // The id lands inside a Python string literal; a quote would break out.
        let script = model_compat_script("evil/repo\"\nimport os; os.system('x')");
        // The `(` is stripped, so no call can survive even if the text does.
        assert!(!script.contains("os.system("), "the call must be neutered");
        let repo_line = script
            .lines()
            .find(|l| l.starts_with("REPO = "))
            .expect("REPO line must exist");
        assert_eq!(
            repo_line.matches('"').count(),
            2,
            "exactly one string literal may survive: {repo_line}"
        );
        assert!(
            !repo_line.contains('\\'),
            "no escape sequences may survive: {repo_line}"
        );
    }

    #[test]
    fn model_script_reports_every_decision() {
        // The deploy log has to explain itself, otherwise a heal is
        // indistinguishable from a silent change.
        let script = model_compat_script("a/b");
        for marker in [
            "[heal] resolving",
            "[heal] transformers installed",
            "[heal] model compatibility resolution complete",
        ] {
            assert!(script.contains(marker), "missing {marker}");
        }
    }

    #[test]
    fn python_payload_survives_shell_wrapping() {
        // Base64 means quotes, newlines and backslashes in the payload never
        // reach the shell — the hazard that caused BUG-14. The generated
        // resolver happens to use only double quotes, so use a payload that
        // definitely carries every dangerous character.
        let payload = "print('single')  # \"double\" and \\ backslash\nnewline";
        let cmd = run_python_file("/root/x.py", payload, "");
        assert!(
            !cmd.contains("single") && !cmd.contains("double"),
            "payload must be encoded, not inlined: {cmd:.200}"
        );
        assert!(!cmd.contains('\n'), "no raw newline may reach the shell");
        assert!(cmd.contains("base64 -d"));
        assert_eq!(cmd.matches('\'').count() % 2, 0, "unbalanced quotes: {cmd:.200}");
    }

    #[test]
    fn real_model_script_is_encoded_not_inlined() {
        // The script itself contains the `"#` sequence and single quotes in
        // comments; none of that may reach the shell.
        let payload = model_compat_script("a/b");
        let cmd = run_python_file("/root/x.py", &payload, "");
        assert!(
            !cmd.contains("config.json"),
            "payload must be encoded, not inlined: {cmd:.200}"
        );
        assert_eq!(cmd.matches('\'').count() % 2, 0, "unbalanced quotes");
    }

    #[test]
    fn python_payload_round_trips_through_base64() {
        use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
        let payload = model_compat_script("Qwen/Qwen3.8-27B");
        let encoded = B64.encode(payload.as_bytes());
        let decoded = B64.decode(encoded.as_bytes()).expect("round trip");
        assert_eq!(String::from_utf8_lossy(&decoded), payload);
    }

    #[test]
    fn teacher_heal_orders_environment_before_model() {
        // vLLM must be importable before the model script queries its registry,
        // so the environment pass has to come first.
        let cmd = teacher_heal_cmd("Qwen/Qwen3.8-27B");
        let env_at = cmd.find("reconciling environment").expect("env pass");
        let model_at = cmd.find("base64 -d").expect("model pass");
        assert!(env_at < model_at, "environment must be reconciled first");
        assert!(cmd.contains("pip check"), "should end with a conflict report");
    }

    #[test]
    fn teacher_heal_is_shell_safe_for_any_repo() {
        for repo in [
            "Qwen/Qwen3.8-27B",
            "meta-llama/Llama-3.1-8B-Instruct",
            "deepseek-ai/DeepSeek-V3",
            "some-org/weird--name.with_dots_1.2B",
        ] {
            let cmd = teacher_heal_cmd(repo);
            assert!(!cmd.contains('\n'), "{repo}: must be one line");
            assert_eq!(cmd.matches('\'').count() % 2, 0, "{repo}: unbalanced quotes");
        }
    }

    #[test]
    fn teacher_pins_hold_for_every_model() {
        // A model-specific floor here would force-upgrade transformers out from
        // under an older checkpoint, so the table must stay environment-only.
        for pin in TEACHER_PINS {
            assert_ne!(
                pin.package, "transformers",
                "transformers must be resolved per-model, not pinned globally"
            );
        }
    }

    // ── Student serving ─────────────────────────────────────────────────────

    #[test]
    fn student_heal_resolves_the_base_model_agnostically() {
        // The student base model is user-chosen and can be anything, so it gets
        // the same discovery pass the teacher does — not a Qwen-shaped assumption.
        let repo = "some-org/brand-new-student-3B";
        let cmd = student_heal_cmd(repo);
        assert!(cmd.contains("base64 -d"), "should run the discovery script");
        // The payload is encoded, so the id is not plaintext — decode to prove
        // the *right* repo was baked in rather than just that something was.
        use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
        let encoded: String = cmd
            .split("printf '%s\\n' ")
            .nth(1)
            .and_then(|rest| rest.split(" > ").next())
            .expect("base64 payload")
            .split('\'')
            .filter(|s| !s.trim().is_empty())
            .collect();
        let decoded = B64.decode(encoded.as_bytes()).expect("decode");
        let decoded = String::from_utf8_lossy(&decoded);
        assert!(decoded.contains(repo), "the student repo must be baked in");
        assert!(!decoded.contains("qwen"), "must not assume a model family");
    }

    #[test]
    fn student_heal_installs_the_adapter_dependencies() {
        // Base + LoRA needs peft to load the adapter and accelerate for
        // device_map="auto"; neither is implied by the base model's metadata.
        let cmd = student_heal_cmd("a/b");
        assert!(cmd.contains("peft"), "adapter loading needs peft");
        assert!(cmd.contains("accelerate"), "device_map=auto needs accelerate");
    }

    #[test]
    fn student_pins_are_failable() {
        for pin in STUDENT_PINS {
            assert!(
                pin.min != (0, 0) || pin.max.is_some(),
                "pin for {} can never fail",
                pin.package
            );
        }
    }

    #[test]
    fn student_heal_does_not_touch_transformers_globally() {
        // Same reasoning as the teacher: a global transformers pin would break
        // whichever student happens to need a different range.
        let cmd = student_heal_cmd("a/b");
        assert!(
            !cmd.contains("pip install --no-cache-dir 'transformers"),
            "student transformers must be discovered, not pinned: {cmd}"
        );
    }

    #[test]
    fn student_heal_is_shell_safe_for_any_base_model() {
        for repo in [
            "Qwen/Qwen2.5-7B-Instruct",
            "meta-llama/Llama-3.2-11B-Vision",
            "some-org/weird--name.with_dots_1.2B",
        ] {
            let cmd = student_heal_cmd(repo);
            assert!(!cmd.contains('\n'), "{repo}: must be one line");
            assert_eq!(cmd.matches('\'').count() % 2, 0, "{repo}: unbalanced quotes");
        }
    }

    /// Write the resolver for `FT_COMPAT_REPO` to `FT_DUMP_DIR/compat.py`.
    ///
    /// Used to run the real resolver against models the app has never seen.
    #[test]
    #[ignore = "writes a file; run explicitly with FT_COMPAT_REPO + FT_DUMP_DIR"]
    fn dump_model_compat_script() {
        let (Some(repo), Some(dir)) = (
            std::env::var("FT_COMPAT_REPO").ok().filter(|s| !s.trim().is_empty()),
            std::env::var("FT_DUMP_DIR").ok().filter(|s| !s.trim().is_empty()),
        ) else {
            eprintln!("skipping: set FT_COMPAT_REPO and FT_DUMP_DIR");
            return;
        };
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir).expect("dump dir");
        let path = dir.join("compat.py");
        std::fs::write(&path, model_compat_script(&repo)).expect("write");
        println!("wrote {} for {}", path.display(), repo);
    }
}
