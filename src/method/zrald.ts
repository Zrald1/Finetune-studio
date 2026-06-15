import type { MethodInfo } from "./types";

export const zraldMethod: MethodInfo = {
  key: "zrald",
  label: "ZRALD",
  desc: "RAG reward RL",
  usesAdapterFields: true,
  copy: {
    title: "ZRALD RAG reward learning",
    detail: "Zero-shot Retrieval-Augmented Learning with Dynamic rewards. The RAG teacher builds questions, the student samples answers, and a reward teacher scores them for GRPO.",
    rankLabel: "ZRALD LoRA Rank (r)",
    alphaLabel: "ZRALD LoRA Alpha",
    dropoutLabel: "ZRALD Dropout",
    learningLabel: "ZRALD Learning Rate",
    batchLabel: "ZRALD Prompt Batch",
    accumLabel: "ZRALD Grad Accum",
    cutoffLabel: "ZRALD Max Seq Length",
    saveLabel: "ZRALD Save Steps",
    note: "Uses the generated RAG question pool, fixed before/after benchmarks, sampled completions per prompt, and reward-teacher scores clamped from -1 to 1.",
  },
};
