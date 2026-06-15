import type { MethodInfo } from "./types";

export const loraMethod: MethodInfo = {
  key: "lora",
  label: "LoRA",
  desc: "Adapter weights",
  usesAdapterFields: true,
  copy: {
    title: "LoRA adapter training",
    detail: "Trains standard low-rank adapters with the base model loaded normally.",
    rankLabel: "LoRA Rank (r)",
    alphaLabel: "LoRA Alpha",
    dropoutLabel: "LoRA Dropout",
    learningLabel: "LoRA Learning Rate",
    batchLabel: "Batch Size",
    accumLabel: "Grad Accum Steps",
    cutoffLabel: "Context Cutoff Length",
    saveLabel: "Save Steps (ckpt + hub push)",
    note: "Standard LoRA keeps VRAM use moderate and works broadly across supported LLaMA-Factory models.",
  },
};
