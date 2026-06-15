import type { MethodInfo } from "./types";

export const customMethod: MethodInfo = {
  key: "custom",
  label: "Add +",
  desc: "Command chain",
  usesAdapterFields: false,
  copy: {
    title: "Custom command method",
    detail: "Runs your saved shell commands in sequence on the remote training node.",
    rankLabel: "Custom Rank",
    alphaLabel: "Custom Alpha",
    dropoutLabel: "Custom Dropout",
    learningLabel: "Custom Learning Rate",
    batchLabel: "Custom Batch Size",
    accumLabel: "Custom Grad Accum",
    cutoffLabel: "Custom Cutoff Length",
    saveLabel: "Custom Save Steps",
    note: "Commands run from the run directory. Write adapter outputs to $OUTPUT_DIR for built-in upload and merge support.",
  },
};
