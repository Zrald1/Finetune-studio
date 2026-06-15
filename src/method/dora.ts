import type { MethodInfo } from "./types";

export const doraMethod: MethodInfo = {
  key: "dora",
  label: "DoRA",
  desc: "Magnitude LoRA",
  usesAdapterFields: true,
  copy: {
    title: "DoRA adapter training",
    detail: "Uses LoRA with use_dora enabled for magnitude-aware adapter updates.",
    rankLabel: "DoRA Rank (r)",
    alphaLabel: "DoRA Alpha",
    dropoutLabel: "DoRA Dropout",
    learningLabel: "DoRA Learning Rate",
    batchLabel: "DoRA Batch Size",
    accumLabel: "DoRA Grad Accum",
    cutoffLabel: "DoRA Cutoff Length",
    saveLabel: "DoRA Save Steps",
    note: "DoRA is a LoRA variant, so adapter merge/upload works like regular LoRA.",
  },
};
