import type { MethodInfo } from "./types";

export const freezeMethod: MethodInfo = {
  key: "freeze",
  label: "Freeze",
  desc: "Selective layers",
  usesAdapterFields: false,
  copy: {
    title: "Freeze tuning",
    detail: "Trains selected upper layers with finetuning_type: freeze.",
    rankLabel: "Unused Rank",
    alphaLabel: "Unused Alpha",
    dropoutLabel: "Unused Dropout",
    learningLabel: "Freeze Learning Rate",
    batchLabel: "Freeze Batch Size",
    accumLabel: "Freeze Grad Accum",
    cutoffLabel: "Freeze Cutoff Length",
    saveLabel: "Freeze Save Steps",
    note: "Freeze tuning trains part of the base model and sits between LoRA and full fine-tuning.",
  },
};
