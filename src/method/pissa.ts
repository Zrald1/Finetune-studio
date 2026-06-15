import type { MethodInfo } from "./types";

export const pissaMethod: MethodInfo = {
  key: "pissa",
  label: "PiSSA",
  desc: "SVD init",
  usesAdapterFields: true,
  copy: {
    title: "PiSSA initialized LoRA",
    detail: "Uses LoRA with PiSSA SVD-based initialization; folds the residual into the adapter at save time.",
    rankLabel: "PiSSA Rank (r)",
    alphaLabel: "PiSSA Alpha",
    dropoutLabel: "PiSSA Dropout",
    learningLabel: "PiSSA Learning Rate",
    batchLabel: "PiSSA Batch Size",
    accumLabel: "PiSSA Grad Accum",
    cutoffLabel: "PiSSA Cutoff Length",
    saveLabel: "PiSSA Save Steps",
    note: "Emits pissa_init + pissa_iter=16 + pissa_convert=true. Saved adapter format matches plain LoRA.",
  },
};
