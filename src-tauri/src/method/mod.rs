pub mod badam;
pub mod common;
pub mod custom;
pub mod dora;
pub mod freeze;
pub mod full;
pub mod galore;
pub mod grpo;
pub mod lora;
pub mod loraplus;
pub mod pissa;
pub mod qlora;
pub mod unsloth;
pub mod zrald;
pub mod zrald_offline;

use crate::error::Result;
use crate::runs::{LoraConfig, Run};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandKind {
    LlamaFactory,
    Custom,
    Grpo,
    Zrald,
    ZraldOffline,
}

#[derive(Debug, Clone, Copy)]
pub struct LlamaFactoryYamlOptions {
    pub finetuning_type: &'static str,
    pub is_lora_family: bool,
    pub quantization_bit: Option<u8>,
    pub quantization_method: Option<&'static str>,
    pub use_unsloth: bool,
    pub use_dora: bool,
    pub loraplus_lr_ratio: Option<f32>,
    pub pissa_init: bool,
    pub pissa_iter: Option<u32>,
    pub pissa_convert: bool,
    pub freeze_trainable_layers: Option<i32>,
    pub use_galore: bool,
    pub galore_layerwise: bool,
    pub galore_target: Option<&'static str>,
    pub galore_rank: Option<u32>,
    pub galore_update_interval: Option<u32>,
    pub galore_scale: Option<f32>,
    pub use_badam: bool,
    pub badam_mode: Option<&'static str>,
    pub badam_switch_mode: Option<&'static str>,
    pub badam_switch_interval: Option<u32>,
    pub badam_verbose: Option<u8>,
    pub pure_bf16: bool,
}

impl LlamaFactoryYamlOptions {
    pub fn lora_like() -> Self {
        Self {
            finetuning_type: "lora",
            is_lora_family: true,
            quantization_bit: None,
            quantization_method: None,
            use_unsloth: false,
            use_dora: false,
            loraplus_lr_ratio: None,
            pissa_init: false,
            pissa_iter: None,
            pissa_convert: false,
            freeze_trainable_layers: None,
            use_galore: false,
            galore_layerwise: false,
            galore_target: None,
            galore_rank: None,
            galore_update_interval: None,
            galore_scale: None,
            use_badam: false,
            badam_mode: None,
            badam_switch_mode: None,
            badam_switch_interval: None,
            badam_verbose: None,
            pure_bf16: false,
        }
    }

    pub fn full_like() -> Self {
        Self {
            finetuning_type: "full",
            is_lora_family: false,
            ..Self::lora_like()
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MethodOptions {
    pub command_kind: CommandKind,
    pub yaml: LlamaFactoryYamlOptions,
    pub needs_bitsandbytes: bool,
    pub extra_optimizer_install: &'static str,
    pub needs_gpu_preflight: bool,
}

impl MethodOptions {
    pub fn lora_like(_key: &'static str) -> Self {
        Self {
            command_kind: CommandKind::LlamaFactory,
            yaml: LlamaFactoryYamlOptions::lora_like(),
            needs_bitsandbytes: false,
            extra_optimizer_install: "",
            needs_gpu_preflight: false,
        }
    }

    pub fn full_like(key: &'static str) -> Self {
        Self {
            yaml: LlamaFactoryYamlOptions::full_like(),
            ..Self::lora_like(key)
        }
    }
}

pub fn options(method: &str) -> MethodOptions {
    match method.trim().to_ascii_lowercase().as_str() {
        lora::KEY | "peft" => lora::options(),
        qlora::KEY | "loftq" => qlora::options(),
        unsloth::KEY => unsloth::options(),
        full::KEY => full::options(),
        freeze::KEY => freeze::options(),
        dora::KEY => dora::options(),
        loraplus::KEY => loraplus::options(),
        pissa::KEY => pissa::options(),
        galore::KEY => galore::options(),
        badam::KEY => badam::options(),
        grpo::KEY => grpo::options(),
        zrald::KEY => zrald::options(),
        zrald_offline::KEY => zrald_offline::options(),
        custom::KEY => custom::options(),
        _ => lora::options(),
    }
}

pub fn command_kind(method: &str) -> CommandKind {
    options(method).command_kind
}

pub fn yaml(method: &str) -> LlamaFactoryYamlOptions {
    options(method).yaml
}

pub fn is_zrald_method(method: &str) -> bool {
    matches!(
        command_kind(method),
        CommandKind::Zrald | CommandKind::ZraldOffline
    )
}

/// True when the method trains and saves a *complete* model (no PEFT adapter):
/// `full` and `freeze`. These run through LLaMA-Factory like LoRA, but the
/// output dir holds `model.safetensors[.index.json]` + `config.json` instead of
/// `adapter_model.safetensors` + `adapter_config.json`, so the post-training
/// existence check, Hub upload, and "merge" steps must be handled differently
/// (the model is already merged — there is nothing to merge into a base).
pub fn is_full_model_method(method: &str) -> bool {
    let opts = options(method);
    opts.command_kind == CommandKind::LlamaFactory && !opts.yaml.is_lora_family
}

pub fn build_train_cmd(
    method: &str,
    run: &Run,
    lora: &LoraConfig,
    hf_export: &str,
) -> Result<String> {
    match method.trim().to_ascii_lowercase().as_str() {
        lora::KEY | "peft" => lora::build_train_cmd(run, lora, hf_export),
        qlora::KEY | "loftq" => qlora::build_train_cmd(run, lora, hf_export),
        unsloth::KEY => unsloth::build_train_cmd(run, lora, hf_export),
        full::KEY => full::build_train_cmd(run, lora, hf_export),
        freeze::KEY => freeze::build_train_cmd(run, lora, hf_export),
        dora::KEY => dora::build_train_cmd(run, lora, hf_export),
        loraplus::KEY => loraplus::build_train_cmd(run, lora, hf_export),
        pissa::KEY => pissa::build_train_cmd(run, lora, hf_export),
        galore::KEY => galore::build_train_cmd(run, lora, hf_export),
        badam::KEY => badam::build_train_cmd(run, lora, hf_export),
        grpo::KEY => grpo::build_train_cmd(run, lora, hf_export),
        zrald::KEY => zrald::build_train_cmd(run, lora, hf_export),
        zrald_offline::KEY => zrald_offline::build_train_cmd(run, lora, hf_export),
        custom::KEY => custom::build_train_cmd(run, lora, hf_export),
        _ => lora::build_train_cmd(run, lora, hf_export),
    }
}
