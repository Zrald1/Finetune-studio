// ── Shared vLLM serving profiles & helpers ───────────────────────────────────
// Imported by DeployPanel AND PipelineWizard so the logic lives in one place.

import React from "react";
import { Target, Scale, Zap } from "lucide-react";

// ── Quant detection ──────────────────────────────────────────────────────────
export interface QuantInfo {
  format: string; vllmFlag: string; label: string;
  colorClass: string; bgClass: string; borderClass: string; description: string;
}

const QUANT_PATTERNS: Array<{ pattern: RegExp; info: Omit<QuantInfo, "format"> }> = [
  { pattern: /\bawq\b/i, info: { vllmFlag: "awq", label: "AWQ", colorClass: "text-violet-300", bgClass: "bg-violet-500/10", borderClass: "border-violet-500/30", description: "Activation-Aware Weight Quantization — INT4 with Marlin kernel. Best accuracy/speed balance." } },
  { pattern: /\bgptq\b/i, info: { vllmFlag: "gptq", label: "GPTQ", colorClass: "text-blue-300", bgClass: "bg-blue-500/10", borderClass: "border-blue-500/30", description: "GPTQ post-training INT4/INT8. Use with Marlin kernel for best performance." } },
  { pattern: /\bfp8\b/i, info: { vllmFlag: "fp8", label: "FP8", colorClass: "text-cyan-300", bgClass: "bg-cyan-500/10", borderClass: "border-cyan-500/30", description: "FP8 — half-precision weights, ~1.6× faster than BF16 on FP8-capable accelerators." } },
  { pattern: /\b(gguf|q4_k_m|q5_k_m|q8_0|q4_0|ggml)\b/i, info: { vllmFlag: "gguf", label: "GGUF", colorClass: "text-amber-300", bgClass: "bg-amber-500/10", borderClass: "border-amber-500/30", description: "GGUF/llama.cpp format. Served via vLLM GGUF backend." } },
  { pattern: /\b(bnb|bitsandbytes|int8|8bit|8-bit)\b/i, info: { vllmFlag: "bitsandbytes", label: "BnB INT8", colorClass: "text-orange-300", bgClass: "bg-orange-500/10", borderClass: "border-orange-500/30", description: "BitsAndBytes INT8. Good for consumer GPUs with limited VRAM." } },
  { pattern: /\b(int4|4bit|4-bit)\b/i, info: { vllmFlag: "awq", label: "INT4", colorClass: "text-fuchsia-300", bgClass: "bg-fuchsia-500/10", borderClass: "border-fuchsia-500/30", description: "INT4 quantized. Auto-selecting AWQ kernel for best throughput." } },
  { pattern: /\b(exl2|exllamav2)\b/i, info: { vllmFlag: "exl2", label: "EXL2", colorClass: "text-pink-300", bgClass: "bg-pink-500/10", borderClass: "border-pink-500/30", description: "ExLlamaV2 quantization. Very fast on NVIDIA consumer GPUs." } },
];

export function detectQuant(repoId: string): QuantInfo | null {
  for (const { pattern, info } of QUANT_PATTERNS) {
    if (pattern.test(repoId)) return { format: info.vllmFlag, ...info };
  }
  return null;
}

// ── Serving Profiles ─────────────────────────────────────────────────────────
export interface ServingProfile {
  key: "precision" | "balanced" | "throughput";
  label: string;
  icon: React.ReactNode;
  gpuMemUtil: number; maxNumSeqs: number; maxNumBatchedTokens: number;
  enableChunkedPrefill: boolean; dtype: string; blockSize: number;
  swapSpaceGb: number; preemptionMode: "recompute" | "swap";
  description: string; badgeText: string; badgeClass: string;
  accentClass: string; borderClass: string; bgClass: string;
}

export const SERVING_PROFILES: ServingProfile[] = [
  {
    key: "precision", label: "Precision Focus", icon: React.createElement(Target, { className: "w-4 h-4" }),
    gpuMemUtil: 0.70, maxNumSeqs: 32, maxNumBatchedTokens: 4096, enableChunkedPrefill: false, dtype: "bfloat16", blockSize: 16, swapSpaceGb: 4, preemptionMode: "recompute",
    description: "Conservative memory, strict dtype — optimized for accuracy. Best for research and low-concurrency tasks.",
    badgeText: "Low load", badgeClass: "text-emerald-300 bg-emerald-500/10 border-emerald-500/30",
    accentClass: "text-emerald-400", borderClass: "border-emerald-500/30", bgClass: "bg-emerald-500/5",
  },
  {
    key: "balanced", label: "Smart Balance", icon: React.createElement(Scale, { className: "w-4 h-4" }),
    gpuMemUtil: 0.85, maxNumSeqs: 128, maxNumBatchedTokens: 16384, enableChunkedPrefill: true, dtype: "auto", blockSize: 16, swapSpaceGb: 8, preemptionMode: "recompute",
    description: "Chunked prefill for mixed request sizes. Handles moderate concurrent users while maintaining quality.",
    badgeText: "Recommended", badgeClass: "text-theme-accent bg-theme-accent/10 border-theme-accent/30",
    accentClass: "text-theme-accent", borderClass: "border-theme-accent/30", bgClass: "bg-theme-accent/5",
  },
  {
    key: "throughput", label: "Max Throughput", icon: React.createElement(Zap, { className: "w-4 h-4" }),
    gpuMemUtil: 0.95, maxNumSeqs: 512, maxNumBatchedTokens: 32768, enableChunkedPrefill: true, dtype: "auto", blockSize: 32, swapSpaceGb: 16, preemptionMode: "swap",
    description: "Maximizes GPU memory and block size. Swap preemption preserves partial work. Best for high-volume APIs.",
    badgeText: "Max users", badgeClass: "text-orange-300 bg-orange-500/10 border-orange-500/30",
    accentClass: "text-orange-400", borderClass: "border-orange-500/30", bgClass: "bg-orange-500/5",
  },
];

// ── Extra flags builder ───────────────────────────────────────────────────────
export interface VllmFlagParams {
  detectedQuant: QuantInfo | null;
  blockSize: number;
  kvCacheDtype: string;
  swapSpaceGb: number;
  cpuOffloadGb: number;
  schedulingPolicy: string;
  preemptionMode: string;
  enablePrefixCaching: boolean;
}

export function buildVllmExtraFlags(p: VllmFlagParams): string {
  const f: string[] = [];
  if (p.detectedQuant) f.push(`--quantization ${p.detectedQuant.vllmFlag}`);
  if (p.blockSize !== 16) f.push(`--block-size ${p.blockSize}`);
  if (p.kvCacheDtype !== "auto") f.push(`--kv-cache-dtype ${p.kvCacheDtype}`);
  if (p.swapSpaceGb > 0) f.push(`--swap-space ${p.swapSpaceGb}`);
  if (p.cpuOffloadGb > 0) f.push(`--cpu-offload-gb ${p.cpuOffloadGb}`);
  if (p.schedulingPolicy !== "fcfs") f.push(`--scheduling-policy ${p.schedulingPolicy}`);
  if (p.preemptionMode !== "recompute") f.push(`--preemption-mode ${p.preemptionMode}`);
  if (p.enablePrefixCaching) f.push("--enable-prefix-caching");
  return f.join(" ");
}

// ── Default state values ──────────────────────────────────────────────────────
export const DEFAULT_VLLM_STATE = {
  activeProfile: "balanced" as ServingProfile["key"],
  dtype: "auto",
  gpuMemUtil: 0.85,
  maxNumSeqs: 128,
  maxNumBatchedTokens: 16384,
  enableChunkedPrefill: true,
  blockSize: 16,
  kvCacheDtype: "auto",
  swapSpaceGb: 8,
  cpuOffloadGb: 0,
  schedulingPolicy: "fcfs" as "fcfs" | "priority",
  preemptionMode: "recompute" as "recompute" | "swap",
  enablePrefixCaching: true,
};
