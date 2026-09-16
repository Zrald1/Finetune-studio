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

    Ok(super::common::llamafactory_train_cmd(
        &run.remote_dir,
        hf_export,
        &extra_optimizer,
    ))

}
