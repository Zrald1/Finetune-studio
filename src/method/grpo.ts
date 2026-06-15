import type { MethodInfo } from "./types";

export const grpoMethod: MethodInfo = {
  key: "grpo",
  label: "GRPO",
  desc: "Reinforcement RL",
  usesAdapterFields: true,
  copy: {
    title: "GRPO reinforcement learning",
    detail: "Group Relative Policy Optimization via Unsloth's GRPOTrainer. Trains LoRA adapters with reward feedback.",
    rankLabel: "GRPO Rank (r)",
    alphaLabel: "GRPO Alpha",
    dropoutLabel: "GRPO Dropout",
    learningLabel: "GRPO Learning Rate",
    batchLabel: "GRPO Batch Size",
    accumLabel: "GRPO Grad Accum",
    cutoffLabel: "GRPO Max Seq Length",
    saveLabel: "GRPO Save Steps",
    note: "GRPO uses reward functions for code/game-strategy tasks and writes training output directly under the run's lora folder.",
  },
};
