import type { MethodInfo } from "./types";

export const fullMethod: MethodInfo = {
  key: "full",
  label: "Full",
  desc: "All weights",
  usesAdapterFields: false,
  copy: {
    title: "Full parameter fine-tuning",
    detail: "Trains all model weights with finetuning_type: full.",
    rankLabel: "Unused Rank",
    alphaLabel: "Unused Alpha",
    dropoutLabel: "Unused Dropout",
    learningLabel: "Full Fine-tune LR",
    batchLabel: "Full Batch Size",
    accumLabel: "Full Grad Accum",
    cutoffLabel: "Full Cutoff Length",
    saveLabel: "Full Checkpoint Steps",
    note: "Full fine-tuning needs much more VRAM and storage because all model weights are trainable.",
  },
};
