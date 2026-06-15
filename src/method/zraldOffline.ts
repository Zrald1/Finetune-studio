import type { MethodInfo } from "./types";

export const zraldOfflineMethod: MethodInfo = {
  key: "zrald_offline",
  label: "ZRALD Offline",
  desc: "Low-VRAM reward",
  usesAdapterFields: true,
  copy: {
    title: "ZRALD Offline preference distillation",
    detail: "Low-VRAM ZRALD. The teacher generates RAG Q&A, unloads, the student writes answers, unloads, then the teacher reloads to score saved answers before student-only adapter training.",
    rankLabel: "Offline ZRALD LoRA Rank (r)",
    alphaLabel: "Offline ZRALD LoRA Alpha",
    dropoutLabel: "Offline ZRALD Dropout",
    learningLabel: "Offline ZRALD Learning Rate",
    batchLabel: "Offline ZRALD Batch",
    accumLabel: "Offline ZRALD Grad Accum",
    cutoffLabel: "Offline ZRALD Max Seq Length",
    saveLabel: "Offline ZRALD Save Steps",
    note: "Teacher and student are never intentionally kept in VRAM together. The student answers from the question only; the teacher scores against stored RAG context.",
  },
};
