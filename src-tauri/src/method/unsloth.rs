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
    let unsloth_probe = "python3 -c 'import unsloth; from unsloth import FastLanguageModel; \
                     import inspect; from peft import LoraConfig; \
                     assert \"ensure_weight_tying\" in inspect.signature(LoraConfig).parameters' \
                     >/dev/null 2>&1";
    let torch_hip_probe = "python3 -c 'import torch,sys; \
                     sys.exit(0 if getattr(torch.version,\"hip\",None) else 1)' \
                     >/dev/null 2>&1";
    let bnb_amd_wheel = "https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_main/bitsandbytes-1.33.7.preview-py3-none-manylinux_2_24_x86_64.whl";
    let unsloth_check = format!(" && {} && {}", torch_hip_probe, unsloth_probe);
    let unsloth_install = format!(
        "&& {{ export UNSLOTH_IS_ROCM=1 PYTORCH_ROCM_ARCH=${{PYTORCH_ROCM_ARCH:-gfx1100}}; \
              ({torch_probe}) || \
              (pip install --no-cache-dir --upgrade --force-reinstall \
                                        --index-url https://download.pytorch.org/whl/rocm7.0 \
                                        'torch>=2.4,<2.11.0' 'torchvision<0.26.0' 'torchaudio<2.11.0'); \
              {probe} || \
              (pip install --no-cache-dir 'unsloth[amd]' 'unsloth_zoo' && \
               (pip install --force-reinstall --no-cache-dir --no-deps '{bnb_wheel}' || \
                pip install --force-reinstall --no-cache-dir --no-deps 'bitsandbytes>=0.49.1') && \
               pip install --no-cache-dir 'peft>=0.19,<0.20' 'trl<0.10.0' 'accelerate>=0.34.0' \
                                        'sentencepiece>=0.2.0' 'datasets>=2.16.0' \
                                        'tyro' 'protobuf' 'hf_transfer' 'psutil' || true); }} ",
        torch_probe = torch_hip_probe,
        probe = unsloth_probe,
        bnb_wheel = bnb_amd_wheel,
    );
    let peft_pin = " && pip install --no-cache-dir --no-deps --upgrade 'peft>=0.19,<0.20'";
    let torch_hip_guard = format!(
        " && ({probe} || pip install --no-cache-dir --upgrade --force-reinstall \
               --index-url https://download.pytorch.org/whl/rocm7.0 \
               'torch>=2.4,<2.11.0' 'torchvision<0.26.0' 'torchaudio<2.11.0')",
        probe = torch_hip_probe,
    );

    Ok(format!(
        "set -o pipefail; \
         python3 -c \"import site, os, shutil; [shutil.rmtree(os.path.join(p, 'triton_kernels'), ignore_errors=True) for p in (getattr(site, 'getsitepackages', lambda: [])() + [getattr(site, 'getusersitepackages', lambda: None)()]) if p]\" 2>/dev/null || true; \
         rm -rf ~/.triton/cache 2>/dev/null || true; \
         export DISABLE_VERSION_CHECK=1 HF_HOME=$HF_HOME PYTHONUNBUFFERED=1 && \
         {hf_export} cd {dir} && \
         export UNSLOTH_IS_ROCM=1 PYTORCH_ROCM_ARCH=${{PYTORCH_ROCM_ARCH:-gfx1100}}; \
         ((python3 -c 'import huggingface_hub; v=huggingface_hub.__version__; exit(0 if v.split(\".\")[0] == \"0\" else 1)' >/dev/null 2>&1 && which llamafactory-cli >/dev/null 2>&1{unsloth_check}) || \
         (true {unsloth_install} && \
         pip install --no-cache-dir 'huggingface-hub<1.0' 'transformers>=4.41.2,<4.58' 'llamafactory==0.9.4' {peft_pin})){torch_hip_guard} && \
         rm -rf ~/.cache/huggingface/datasets 2>/dev/null || true && \
         : > {dir}/log.txt && : > {dir}/errorlog.txt && : > {dir}/train.log && \
         llamafactory-cli train {dir}/train.yaml \
           > >(tee -a {dir}/log.txt {dir}/train.log) \
           2> >(tee -a {dir}/errorlog.txt {dir}/train.log >&2)",
        hf_export = hf_export,
        dir = run.remote_dir,
        unsloth_check = unsloth_check,
        unsloth_install = unsloth_install,
        peft_pin = peft_pin,
        torch_hip_guard = torch_hip_guard,
    ))
}
