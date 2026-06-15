import type { MethodInfo } from "./types";

export const galoreMethod: MethodInfo = {
  key: "galore",
  label: "GaLore",
  desc: "Memory efficient",
  usesAdapterFields: false,
  copy: {
    title: "GaLore full tuning",
    detail: "Full fine-tune with GaLore low-rank gradient projection. Memory close to LoRA, about 30% slower per step.",
    rankLabel: "Unused Rank",
    alphaLabel: "Unused Alpha",
    dropoutLabel: "Unused Dropout",
    learningLabel: "GaLore Learning Rate",
    batchLabel: "GaLore Batch Size",
    accumLabel: "GaLore Grad Accum",
    cutoffLabel: "GaLore Cutoff Length",
    saveLabel: "GaLore Save Steps",
    note: "Layerwise mode + pure_bf16; works on ROCm as a pure PyTorch optimizer. Use Grad Accum = 1 with layerwise GaLore.",
  },
};
