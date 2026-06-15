use crate::error::{AppError, Result};
use crate::llamafactory;
use crate::runs::{LoraConfig, Run};

use super::common::sh_quote;
use super::{CommandKind, LlamaFactoryYamlOptions, MethodOptions};

pub const KEY: &str = "custom";

pub fn yaml() -> LlamaFactoryYamlOptions {
    LlamaFactoryYamlOptions::lora_like()
}

pub fn options() -> MethodOptions {
    MethodOptions {
        command_kind: CommandKind::Custom,
        yaml: yaml(),
        ..MethodOptions::lora_like(KEY)
    }
}
pub fn build_train_cmd(run: &Run, lora: &LoraConfig, hf_export: &str) -> Result<String> {
    let commands: Vec<String> = lora
        .custom_commands
        .iter()
        .map(|cmd| cmd.trim())
        .filter(|cmd| !cmd.is_empty())
        .map(|cmd| expand_custom_command(cmd, run))
        .collect();

    if commands.is_empty() {
        return Err(AppError::pipeline(
            "custom fine-tuning method selected but no commands were provided",
        ));
    }

    let train_yaml = format!("{}/train.yaml", run.remote_dir);
    let data_dir = format!("{}/data", run.remote_dir);
    let output_dir = format!("{}/lora", run.remote_dir);
    let base_model = llamafactory::resolve_trainable_repo(&run.student_model);
    let method_name = lora.custom_method_name.trim();
    let title = if method_name.is_empty() {
        "custom fine-tuning method"
    } else {
        method_name
    };

    let mut body = String::from("set -e\n");
    body.push_str(&format!(
        "echo {}\n",
        sh_quote(&format!("[custom] starting {title}"))
    ));
    for (idx, command) in commands.iter().enumerate() {
        body.push_str(&format!(
            "echo {}\n{}\n",
            sh_quote(&format!("[custom] step {}/{}", idx + 1, commands.len())),
            command
        ));
    }

    Ok(format!(
        "set -o pipefail; \
         {hf_export} cd {dir} && \
         export RUN_DIR={run_dir} TRAIN_YAML={train_yaml} DATA_DIR={data_dir} OUTPUT_DIR={output_dir} \
                STUDENT_MODEL={student_model} BASE_MODEL={base_model} \
                FT_LEARNING_RATE={learning_rate} FT_EPOCHS={epochs} FT_BATCH_SIZE={batch_size} \
                FT_GRADIENT_ACCUMULATION={gradient_accumulation} FT_CUTOFF_LEN={cutoff_len} \
                FT_SAVE_STEPS={save_steps} FT_LORA_R={rank} FT_LORA_ALPHA={alpha} FT_LORA_DROPOUT={dropout} \
                PYTHONUNBUFFERED=1 && \
         mkdir -p {output_dir} && \
         : > {dir}/log.txt && : > {dir}/errorlog.txt && : > {dir}/train.log && \
         {{ {body} }} \
           > >(tee -a {dir}/log.txt {dir}/train.log) \
           2> >(tee -a {dir}/errorlog.txt {dir}/train.log >&2)",
        hf_export = hf_export,
        dir = sh_quote(&run.remote_dir),
        run_dir = sh_quote(&run.remote_dir),
        train_yaml = sh_quote(&train_yaml),
        data_dir = sh_quote(&data_dir),
        output_dir = sh_quote(&output_dir),
        student_model = sh_quote(&run.student_model),
        base_model = sh_quote(&base_model),
        learning_rate = sh_quote(&lora.learning_rate.to_string()),
        epochs = sh_quote(&lora.epochs.to_string()),
        batch_size = sh_quote(&lora.batch_size.to_string()),
        gradient_accumulation = sh_quote(&lora.gradient_accumulation.to_string()),
        cutoff_len = sh_quote(&lora.cutoff_len.to_string()),
        save_steps = sh_quote(&lora.save_steps.to_string()),
        rank = sh_quote(&lora.r.to_string()),
        alpha = sh_quote(&lora.alpha.to_string()),
        dropout = sh_quote(&lora.dropout.to_string()),
        body = body,
    ))
}

fn expand_custom_command(command: &str, run: &Run) -> String {
    let base_model = llamafactory::resolve_trainable_repo(&run.student_model);
    command
        .replace("{run_dir}", &run.remote_dir)
        .replace("{train_yaml}", &format!("{}/train.yaml", run.remote_dir))
        .replace("{data_dir}", &format!("{}/data", run.remote_dir))
        .replace("{output_dir}", &format!("{}/lora", run.remote_dir))
        .replace("{student_model}", &run.student_model)
        .replace("{base_model}", &base_model)
}
