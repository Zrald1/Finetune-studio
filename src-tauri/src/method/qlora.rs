use crate::error::Result;
use crate::runs::{LoraConfig, Run};

use super::{LlamaFactoryYamlOptions, MethodOptions};

pub const KEY: &str = "qlora";

pub fn yaml() -> LlamaFactoryYamlOptions {
    LlamaFactoryYamlOptions {
        quantization_bit: Some(4),
        quantization_method: Some("bnb"),
        ..LlamaFactoryYamlOptions::lora_like()
    }
}

pub fn options() -> MethodOptions {
    MethodOptions {
        yaml: yaml(),
        needs_bitsandbytes: true,
        ..MethodOptions::lora_like(KEY)
    }
}

pub fn build_train_cmd(run: &Run, _lora: &LoraConfig, hf_export: &str) -> Result<String> {
    let bnb_dep = if options().needs_bitsandbytes {
        "'bitsandbytes>=0.49.1' "
    } else {
        ""
    };

    Ok(format!(
        "set -o pipefail; \
         python3 -c \"import site, os, shutil; [shutil.rmtree(os.path.join(p, 'triton_kernels'), ignore_errors=True) for p in (getattr(site, 'getsitepackages', lambda: [])() + [getattr(site, 'getusersitepackages', lambda: None)()]) if p]\" 2>/dev/null || true; \
         rm -rf ~/.triton/cache 2>/dev/null || true; \
         export DISABLE_VERSION_CHECK=1 HF_HOME=$HF_HOME PYTHONUNBUFFERED=1 && \
         {hf_export} cd {dir} && \
         ((python3 -c 'import huggingface_hub; v=huggingface_hub.__version__; exit(0 if v.split(\".\")[0] == \"0\" else 1)' >/dev/null 2>&1 && which llamafactory-cli >/dev/null 2>&1) || \
          pip install --no-cache-dir 'huggingface-hub<1.0' 'transformers>=4.41.2,<4.58' 'llamafactory==0.9.4' {bnb_dep}) && \
         rm -rf ~/.cache/huggingface/datasets 2>/dev/null || true && \
         : > {dir}/log.txt && : > {dir}/errorlog.txt && : > {dir}/train.log && \
         llamafactory-cli train {dir}/train.yaml \
           > >(tee -a {dir}/log.txt {dir}/train.log) \
           2> >(tee -a {dir}/errorlog.txt {dir}/train.log >&2)",
        hf_export = hf_export,
        dir = run.remote_dir,
        bnb_dep = bnb_dep,
    ))
}
