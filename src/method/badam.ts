import type { MethodInfo } from "./types";

export const badamMethod: MethodInfo = {
  key: "badam",
  label: "BAdam",
  desc: "Block-wise Adam",
  usesAdapterFields: false,
  copy: {
    title: "BAdam full tuning",
    detail: "Full fine-tune with block-wise Adam updating one layer block at a time. Lower peak VRAM than full Adam.",
    rankLabel: "Unused Rank",
    alphaLabel: "Unused Alpha",
    dropoutLabel: "Unused Dropout",
    learningLabel: "BAdam Learning Rate",
    batchLabel: "BAdam Batch Size",
    accumLabel: "BAdam Grad Accum",
    cutoffLabel: "BAdam Cutoff Length",
    saveLabel: "BAdam Save Steps",
    note: "Installs badam from PyPI. Layer mode + ascending switch + interval 50. ROCm-compatible as a pure PyTorch optimizer.",
  },
};
