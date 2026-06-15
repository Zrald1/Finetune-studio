import type { MethodInfo } from "./types";

export const unslothMethod: MethodInfo = {
  key: "unsloth",
  label: "Unsloth",
  desc: "Fast inference",
  usesAdapterFields: true,
  copy: {
    title: "Unsloth accelerated LoRA",
    detail: "Adds use_unsloth: true to LLaMA-Factory and prepares the ROCm-specific Unsloth dependency path.",
    rankLabel: "Unsloth LoRA Rank (r)",
    alphaLabel: "Unsloth LoRA Alpha",
    dropoutLabel: "Unsloth Dropout",
    learningLabel: "Unsloth Learning Rate",
    batchLabel: "Unsloth Batch Size",
    accumLabel: "Unsloth Grad Accum",
    cutoffLabel: "Unsloth Max Seq Length",
    saveLabel: "Unsloth Save Steps",
    note: "Unsloth on ROCm is community-supported. MI300X is best-tested; consumer AMD cards may hit Triton-kernel issues.",
  },
};
