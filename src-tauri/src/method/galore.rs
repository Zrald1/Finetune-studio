use crate::error::Result;
use crate::runs::{LoraConfig, Run};

use super::{LlamaFactoryYamlOptions, MethodOptions};

pub const KEY: &str = "galore";

pub fn yaml() -> LlamaFactoryYamlOptions {
    LlamaFactoryYamlOptions {
        use_galore: true,
        galore_layerwise: true,
        galore_target: Some("all"),
        galore_rank: Some(128),
        galore_update_interval: Some(200),
        galore_scale: Some(2.0),
        pure_bf16: true,
        ..LlamaFactoryYamlOptions::full_like()
    }
}

pub fn options() -> MethodOptions {
    MethodOptions {
        yaml: yaml(),
        extra_optimizer_install: " && pip install --no-cache-dir 'galore-torch'",
        ..MethodOptions::full_like(KEY)
    }
}

pub fn build_train_cmd(run: &Run, _lora: &LoraConfig, hf_export: &str) -> Result<String> {
    let extra_optimizer = options().extra_optimizer_install;

    Ok(format!(
        "set -o pipefail; \
         python3 -c \"import site, os, shutil; [shutil.rmtree(os.path.join(p, 'triton_kernels'), ignore_errors=True) for p in (getattr(site, 'getsitepackages', lambda: [])() + [getattr(site, 'getusersitepackages', lambda: None)()]) if p]\" 2>/dev/null || true; \
         rm -rf ~/.triton/cache 2>/dev/null || true; \
         export DISABLE_VERSION_CHECK=1 HF_HOME=$HF_HOME PYTHONUNBUFFERED=1 && \
         {hf_export} cd {dir} && \
         ((python3 -c 'import huggingface_hub; v=huggingface_hub.__version__; exit(0 if v.split(\".\")[0] == \"0\" else 1)' >/dev/null 2>&1 && which llamafactory-cli >/dev/null 2>&1) || \
          pip install --no-cache-dir 'huggingface-hub<1.0' 'transformers>=4.41.2,<4.58' 'llamafactory==0.9.4' {extra_optimizer}) && \
         rm -rf ~/.cache/huggingface/datasets 2>/dev/null || true && \
         : > {dir}/log.txt && : > {dir}/errorlog.txt && : > {dir}/train.log && \
         llamafactory-cli train {dir}/train.yaml \
           > >(tee -a {dir}/log.txt {dir}/train.log) \
           2> >(tee -a {dir}/errorlog.txt {dir}/train.log >&2)",
        hf_export = hf_export,
        dir = run.remote_dir,
        extra_optimizer = extra_optimizer,
    ))
}
