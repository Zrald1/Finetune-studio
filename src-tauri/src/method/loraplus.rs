use crate::error::Result;
use crate::runs::{LoraConfig, Run};

use super::{LlamaFactoryYamlOptions, MethodOptions};

pub const KEY: &str = "loraplus";

pub fn yaml() -> LlamaFactoryYamlOptions {
    LlamaFactoryYamlOptions {
        loraplus_lr_ratio: Some(16.0),
        ..LlamaFactoryYamlOptions::lora_like()
    }
}

pub fn options() -> MethodOptions {
    MethodOptions {
        yaml: yaml(),
        ..MethodOptions::lora_like(KEY)
    }
}

pub fn build_train_cmd(run: &Run, _lora: &LoraConfig, hf_export: &str) -> Result<String> {
    Ok(super::common::llamafactory_train_cmd(
        &run.remote_dir,
        hf_export,
        "",
    ))

}
