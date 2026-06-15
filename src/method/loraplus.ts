import type { MethodInfo } from "./types";

export const loraplusMethod: MethodInfo = {
  key: "loraplus",
  label: "LoRA+",
  desc: "LR multiplier",
  usesAdapterFields: true,
  copy: {
    title: "LoRA+ adapter training",
    detail: "Uses LoRA with LoRA+ learning-rate ratio enabled.",
    rankLabel: "LoRA+ Rank (r)",
    alphaLabel: "LoRA+ Alpha",
    dropoutLabel: "LoRA+ Dropout",
    learningLabel: "LoRA+ Base LR",
    batchLabel: "LoRA+ Batch Size",
    accumLabel: "LoRA+ Grad Accum",
    cutoffLabel: "LoRA+ Cutoff Length",
    saveLabel: "LoRA+ Save Steps",
    note: "LoRA+ uses separate adapter learning-rate scaling while keeping normal LoRA outputs.",
  },
};
