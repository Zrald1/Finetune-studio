use crate::error::Result;
use crate::runs::{LoraConfig, Run};

use super::{LlamaFactoryYamlOptions, MethodOptions};

pub const KEY: &str = "unsloth";

pub fn yaml() -> LlamaFactoryYamlOptions {
    LlamaFactoryYamlOptions {
        use_unsloth: true,
        ..LlamaFactoryYamlOptions::lora_like()
    }
}

pub fn options() -> MethodOptions {
    MethodOptions {
        yaml: yaml(),
        needs_bitsandbytes: true,
        needs_gpu_preflight: true,
        ..MethodOptions::lora_like(KEY)
    }
}

pub fn build_train_cmd(run: &Run, _lora: &LoraConfig, hf_export: &str) -> Result<String> {
    // Unsloth rides the same isolated venv as every other LLaMA-Factory
    // method. Installing it into the container's shared environment is what
    // downgraded transformers and broke the Qwen3.8 teacher.
    let unsloth_pkgs = "'unsloth[amd]' 'unsloth_zoo' 'peft>=0.19,<0.20' 'trl<0.10.0'         'accelerate>=0.34.0' 'sentencepiece>=0.2.0' 'datasets>=2.16.0'         'tyro' 'protobuf' 'hf_transfer' 'psutil'";

    let mut cmd = super::common::llamafactory_train_cmd(
        &run.remote_dir,
        hf_export,
        unsloth_pkgs,
    );

    // Unsloth only activates ROCm mode when these are exported at *runtime*,
    // not just install time; without them it falls back to CUDA/CPU, fails to
    // find the AMD GPU, and is killed with SIGTERM (exit 143).
    let prelude = format!("export UNSLOTH_IS_ROCM=1; {}cd ", super::common::rocm_prelude());
    cmd = cmd.replacen("cd ", &prelude, 1);

    Ok(cmd)

}
