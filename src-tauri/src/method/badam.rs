use crate::error::Result;
use crate::runs::{LoraConfig, Run};

use super::{LlamaFactoryYamlOptions, MethodOptions};

pub const KEY: &str = "badam";

pub fn yaml() -> LlamaFactoryYamlOptions {
    LlamaFactoryYamlOptions {
        use_badam: true,
        badam_mode: Some("layer"),
        badam_switch_mode: Some("ascending"),
        badam_switch_interval: Some(50),
        badam_verbose: Some(2),
        pure_bf16: true,
        ..LlamaFactoryYamlOptions::full_like()
    }
}

pub fn options() -> MethodOptions {
    MethodOptions {
        yaml: yaml(),
        extra_optimizer_install: " && pip install --no-cache-dir 'badam>=1.2.1'",
        ..MethodOptions::full_like(KEY)
    }
}

pub fn build_train_cmd(run: &Run, _lora: &LoraConfig, hf_export: &str) -> Result<String> {
    let extra_optimizer = options().extra_optimizer_install;

    Ok(super::common::llamafactory_train_cmd(
        &run.remote_dir,
        hf_export,
        &extra_optimizer,
    ))

}
