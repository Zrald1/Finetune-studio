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

    Ok(super::common::llamafactory_train_cmd(
        &run.remote_dir,
        hf_export,
        &bnb_dep,
    ))

}
