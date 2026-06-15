import type { MethodInfo } from "./types";

export const qloraMethod: MethodInfo = {
  key: "qlora",
  label: "QLoRA",
  desc: "4-bit quant",
  usesAdapterFields: true,
  copy: {
    title: "QLoRA 4-bit adapter training",
    detail: "Loads the base model in 4-bit with bitsandbytes, then trains LoRA adapters.",
    rankLabel: "QLoRA Rank (r)",
    alphaLabel: "QLoRA Alpha",
    dropoutLabel: "QLoRA Dropout",
    learningLabel: "QLoRA Learning Rate",
    batchLabel: "4-bit Batch Size",
    accumLabel: "4-bit Grad Accum",
    cutoffLabel: "4-bit Context Cutoff",
    saveLabel: "Checkpoint Push Steps",
    note: "Installs bitsandbytes>=0.49.1 for 4-bit quantization. ROCm compatibility still depends on the GPU/image pair.",
  },
};
