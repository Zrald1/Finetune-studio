use crate::error::Result;
use crate::runs::{LoraConfig, Run};

use super::{LlamaFactoryYamlOptions, MethodOptions};

pub const KEY: &str = "dora";

pub fn yaml() -> LlamaFactoryYamlOptions {
    LlamaFactoryYamlOptions {
        use_dora: true,
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
