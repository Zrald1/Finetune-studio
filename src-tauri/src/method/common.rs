/// Quote a string so it is safe to embed inside single quotes in a bash command.
pub fn sh_quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for ch in s.chars() {
        if ch == '\'' {
            out.push_str("'\"'\"'");
        } else {
            out.push(ch);
        }
    }
    out.push('\'');
    out
}

/// Shell prelude shared by both ZRALD trainers.
///
/// Two host quirks are handled here so the online and offline paths cannot drift
/// apart again (they previously disagreed on both):
///
/// 1. **`amdgpu.ids` stub.** libdrm reads `/opt/amdgpu/share/libdrm/amdgpu.ids`
///    to enumerate GPU IDs. When the file is absent it prints
///    `amdgpu.ids: No such file or directory` and GPU detection can fail. The
///    offline path carried this workaround; the online path did not, and the
///    warning reproduces on a stock DigitalOcean MI300X droplet.
///
/// 2. **`PYTORCH_ROCM_ARCH` detection.** The three previous call sites defaulted
///    to `gfx950`, `gfx1100`, and `gfx950` — none of which is `gfx942`, the
///    MI300X / MI325X architecture this app targets and tests against. Building
///    extensions for the wrong ISA fails late with an obscure device-function
///    error. Detect it from `rocminfo` and fall back to `gfx942`.
///
/// Emitted as a single line so callers can inline it into an `&&` chain.
pub fn rocm_prelude() -> &'static str {
    "if [ ! -f /opt/amdgpu/share/libdrm/amdgpu.ids ]; then \
       mkdir -p /opt/amdgpu/share/libdrm 2>/dev/null || true; \
       touch /opt/amdgpu/share/libdrm/amdgpu.ids 2>/dev/null || true; \
     fi; \
     export PYTORCH_ROCM_ARCH=\"${PYTORCH_ROCM_ARCH:-$(rocminfo 2>/dev/null | grep -m1 -o 'gfx[0-9]*' || true)}\"; \
     export PYTORCH_ROCM_ARCH=\"${PYTORCH_ROCM_ARCH:-gfx942}\"; "
}

/// Shared `llamafactory-cli train` command, isolated in its own venv.
///
/// **Why the venv.** LLaMA-Factory and the Qwen3.8 teacher have mutually
/// exclusive `transformers` requirements:
///
/// | consumer | requirement |
/// |---|---|
/// | Qwen3.8-27B teacher (vLLM) | `transformers >= 5.8.0` |
/// | LLaMA-Factory | `transformers >= 4.55.0, <= 5.6.0` |
///
/// Installing LLaMA-Factory into the shared container environment silently
/// downgraded transformers 5.15.0 → 5.6.0 and broke the teacher; repairing the
/// teacher then broke the trainer with
/// `ImportError: transformers>=4.55.0,<=5.6.0 is required ... but found 5.17.0`.
/// `DISABLE_VERSION_CHECK=1` papers over it, but the gap widens with every
/// transformers release — `warmup_ratio` was already removed from
/// `TrainingArguments` in 5.x, which hard-fails the config parser.
///
/// `--system-site-packages` is deliberate: the container already ships a
/// ROCm-matched torch, so the venv inherits it instead of pulling its own
/// (~5 GB, and a build that may not see the GPU). Only the conflicting packages
/// — transformers, llamafactory, huggingface-hub and friends — land in the
/// venv, where they shadow the system copies without touching them.
///
/// `extra_packages` is spliced into the install line for methods that need
/// bitsandbytes, galore, badam, and so on.
pub fn llamafactory_train_cmd(dir: &str, hf_export: &str, extra_packages: &str) -> String {
    let dir_q = sh_quote(dir);
    let venv = sh_quote(&format!("{dir}/.lf_venv"));
    let extra = extra_packages.trim();

    format!(
        "set -o pipefail; \
         rm -rf ~/.triton/cache 2>/dev/null || true; \
         {hf_export}cd {dir} && \
         (test -d {venv}/bin || python3 -m venv --system-site-packages {venv}) && \
         . {venv}/bin/activate && \
         export DISABLE_VERSION_CHECK=1 HF_HOME=${{HF_HOME:-/root/hf-cache}} PYTHONUNBUFFERED=1 && \
         (test -x {venv}/bin/llamafactory-cli || \
           (python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
            pip install --no-cache-dir 'huggingface-hub<1.0' 'transformers>=4.41.2,<4.58' \
              'llamafactory==0.9.4' {extra})) && \
         {reconcile}\
         rm -rf ~/.cache/huggingface/datasets 2>/dev/null || true; \
         : > {dir}/log.txt && : > {dir}/errorlog.txt && : > {dir}/train.log && \
         {venv}/bin/llamafactory-cli train {dir}/train.yaml \
           > >(tee -a {dir}/log.txt {dir}/train.log) \
           2> >(tee -a {dir}/errorlog.txt {dir}/train.log >&2)",
        hf_export = hf_export,
        dir = dir_q,
        venv = venv,
        extra = extra,
        reconcile = crate::deps::heal_and_report_cmd(crate::deps::TRAINER_PINS, ""),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rocm_prelude_stubs_amdgpu_ids() {
        let p = rocm_prelude();
        assert!(p.contains("/opt/amdgpu/share/libdrm/amdgpu.ids"));
        assert!(p.contains("touch"));
    }

    #[test]
    fn rocm_prelude_detects_arch_and_falls_back_to_gfx942() {
        let p = rocm_prelude();
        assert!(p.contains("rocminfo"), "should detect the real arch first");
        assert!(p.contains("gfx942"), "fallback must match the target hardware");
        assert!(!p.contains("gfx950"), "gfx950 was the wrong default");
        assert!(!p.contains("gfx1100"), "gfx1100 was the wrong default");
    }

    #[test]
    fn rocm_prelude_is_single_line_for_ampersand_chains() {
        // Callers splice this into `cmd && {prelude} cd ... && ...`, so an
        // embedded newline would truncate the command.
        assert!(!rocm_prelude().contains('\n'));
    }

    #[test]
    fn rocm_prelude_does_not_break_single_quoted_wrapping() {
        // The deploy path wraps commands in single quotes; an unbalanced quote
        // here would terminate the wrapper early.
        assert_eq!(rocm_prelude().matches('\'').count() % 2, 0);
    }

    #[test]
    fn sh_quote_escapes_single_quotes() {
        assert_eq!(sh_quote("a'b"), "'a'\"'\"'b'");
        assert_eq!(sh_quote("plain"), "'plain'");
    }

    // ── LLaMA-Factory trainer isolation ─────────────────────────────────────

    #[test]
    fn trainer_uses_its_own_venv() {
        let cmd = llamafactory_train_cmd("/root/run", "", "");
        assert!(cmd.contains(".lf_venv"), "trainer must not share the container env");
        assert!(cmd.contains("/bin/llamafactory-cli"), "must call the venv's CLI");
    }

    #[test]
    fn trainer_venv_inherits_system_site_packages() {
        // The container already has a ROCm-matched torch. Without
        // --system-site-packages the venv would pull its own ~5 GB torch and,
        // worse, a CPU/CUDA build that cannot see the GPU.
        let cmd = llamafactory_train_cmd("/root/run", "", "");
        assert!(
            cmd.contains("--system-site-packages"),
            "venv must inherit the container's ROCm torch: {cmd}"
        );
        assert!(
            !cmd.contains("pip install --no-cache-dir 'torch"),
            "must not reinstall torch into the venv"
        );
    }

    #[test]
    fn trainer_probes_for_the_venv_cli_not_the_module() {
        // With --system-site-packages the venv can *import* a system-wide
        // llamafactory, so an import probe passes while the venv has no
        // `llamafactory-cli` binary — the run then dies with
        // "No such file or directory". The probe must test the executable.
        let cmd = llamafactory_train_cmd("/root/run", "", "");
        assert!(
            cmd.contains("test -x") && cmd.contains("bin/llamafactory-cli"),
            "must probe for the venv's own CLI: {cmd}"
        );
        assert!(
            !cmd.contains("import llamafactory"),
            "an import probe is satisfied by the system copy: {cmd}"
        );
    }

    #[test]
    fn trainer_pins_transformers_inside_the_venv_only() {
        // The `<=4.58` cap is legitimate for LLaMA-Factory — but it must be
        // applied *inside* the venv, after activation, so it cannot leak back
        // into the container environment the teacher shares.
        let cmd = llamafactory_train_cmd("/root/run", "", "");
        // The venv path is shell-quoted, so match on the unquoted tail.
        let activate = cmd.find("bin/activate").expect("venv must be activated");
        let cap = cmd.find("<4.58").expect("trainer should cap transformers");
        assert!(
            activate < cap,
            "the transformers cap must be installed after venv activation: {cmd}"
        );

        // The teacher must not inherit that cap, and must not pin transformers
        // at all — its requirement is discovered per-model at deploy time, so a
        // global floor would force-upgrade transformers for older checkpoints.
        let teacher = crate::deps::teacher_heal_cmd("Qwen/Qwen3.8-27B");
        assert!(
            !teacher.contains("<4.58"),
            "teacher must not inherit the trainer's cap: {teacher}"
        );
        for pin in crate::deps::TEACHER_PINS {
            assert_ne!(pin.package, "transformers");
        }
    }

    #[test]
    fn trainer_heals_its_own_requirements() {
        // Self-healing applies to the trainer too, not just the teacher.
        let cmd = llamafactory_train_cmd("/root/run", "", "");
        assert!(cmd.contains("[heal]"), "trainer should report its heals");
        assert!(cmd.contains("pip check"), "trainer should report conflicts");
    }

    #[test]
    fn trainer_keeps_the_version_check_disabled() {
        let cmd = llamafactory_train_cmd("/root/run", "", "");
        assert!(cmd.contains("DISABLE_VERSION_CHECK=1"));
    }

    #[test]
    fn trainer_threads_extra_packages_through() {
        let cmd = llamafactory_train_cmd("/root/run", "", "'bitsandbytes>=0.49.1'");
        assert!(cmd.contains("bitsandbytes>=0.49.1"));
    }

    #[test]
    fn trainer_cmd_is_shell_parseable_and_quote_balanced() {
        let cmd = llamafactory_train_cmd("/root/fine-tune/runs/abc", "export HF_TOKEN=x; ", "");
        assert_eq!(cmd.matches('\'').count() % 2, 0, "unbalanced quotes: {cmd}");
        assert!(!cmd.contains('\n'), "must stay a single logical line");
    }

    #[test]
    fn trainer_writes_the_logs_the_ui_tails() {
        let cmd = llamafactory_train_cmd("/root/run", "", "");
        for f in ["log.txt", "errorlog.txt", "train.log"] {
            assert!(cmd.contains(f), "missing {f} in: {cmd}");
        }
    }
}
