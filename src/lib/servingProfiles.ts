// ── Serving-profile logic (shared, side-effect free) ─────────────────────────
//
// Extracted from PipelineWizard so it can be exercised without a browser. It is
// deliberately free of React and of any runtime import, which lets
// `scripts/simulate-deploy.ts` import this exact module under Node's type
// stripping — the simulation therefore tests the shipped code rather than a
// copy of it.
//
// Mirrors `TeacherConfig::resolved_for_gpu` in src-tauri/src/config.rs. When one
// side changes, change both: the Rust tests and this simulation assert the same
// table of values.

import type { GPUState, ServingProfile, TeacherConfig } from "../types.ts";

export interface PresetTeacherModel {
  value: string;
  label: string;
}

// Curated teacher roster. Qwen3.8-27B is the supported default: 27B dense
// hybrid-attention checkpoint, 262K native context, Apache 2.0, and the model
// vLLM publishes a verified recipe for on AMD Instinct MI300X/MI325X/MI355X.
// Everything else is reachable through "Custom Hugging Face Repo ID".
export const PRESET_TEACHER_MODELS: PresetTeacherModel[] = [
  { value: "Qwen/Qwen3.8-27B", label: "Qwen 3.8 27B (Recommended)" },
];

export const DEFAULT_SERVING_PROFILE: ServingProfile = "standard";

/** VRAM floor (GB) → context window for the optimized profile. */
export const OPTIMIZED_CONTEXT_LADDER: ReadonlyArray<readonly [number, number]> = [
  [140, 262144],
  [96, 131072],
  [48, 65536],
  [0, 32768],
];

// The optimized profile mirrors the vendor serving recipe. Context is clamped by
// VRAM so the same profile stays valid on a 48 GB card as well as a 256 GB MI325X.
export function optimizedMaxModelLen(memoryGb: number): number {
  for (const [floor, context] of OPTIMIZED_CONTEXT_LADDER) {
    if (memoryGb >= floor) return context;
  }
  return 32768;
}

export function optimizedMaxNumSeqs(memoryGb: number): number {
  return memoryGb > 0 && memoryGb < 48 ? 8 : 64;
}

/**
 * True for hybrid checkpoints mixing full attention with linear-attention
 * ("Mamba"-style) layers — Qwen3.5 and later, Jamba, and anything advertising a
 * Gated DeltaNet.
 *
 * Mirrors `is_hybrid_linear_attention` in src-tauri/src/config.rs. vLLM reports
 * prefix caching over Mamba layers as experimental, so the optimized profile
 * leaves it off for these models rather than quietly risking corrupted
 * training data.
 */
export function isHybridLinearAttention(repoId: string): boolean {
  const repo = (repoId || "").toLowerCase();
  if (["mamba", "gdn", "jamba", "deltanet"].some((m) => repo.includes(m))) return true;
  return ["qwen3.5", "qwen3.6", "qwen3.7", "qwen3.8", "qwen3_5", "qwen3_6", "qwen3_7", "qwen3_8"].some((m) =>
    repo.includes(m),
  );
}

export function optimizedProfileSummary(memoryGb: number, repoId = ""): string {
  const context = optimizedMaxModelLen(memoryGb);
  const prefix = isHybridLinearAttention(repoId) ? "" : "prefix caching, ";
  return `Optimized: FP8 KV cache, ${prefix}${context.toLocaleString()} ctx, ${optimizedMaxNumSeqs(memoryGb)} seqs`;
}

export function standardProfileSummary(config: TeacherConfig): string {
  return `Standard: ${config.dtype}, ${config.maxModelLen} ctx, ${config.maxNumBatchedTokens || 0} batched tokens, ${config.maxNumSeqs || 0} seqs`;
}

/** True when the repo id names a hybrid-thinking Qwen3.5+ checkpoint. */
export function isQwen3Family(repoId: string): boolean {
  return (repoId || "").toLowerCase().includes("qwen3");
}

/**
 * Qwen3.8-27B ships a vision tower but carries no "-vl"/"vision" marker in its
 * repo id, so name-based multimodal detection has to special-case it.
 */
export function isVisionFamily(repoId: string): boolean {
  const repo = (repoId || "").toLowerCase();
  return repo.includes("-vl") || repo.includes("vision") || repo.includes("qwen3.8") || repo.includes("qwen3_8");
}

export function autoTuneTeacherConfig(base: TeacherConfig, gpuStatus?: GPUState | null): TeacherConfig {
  if (!base.autoTune || (base.customServeCmd || "").trim()) return base;

  const repo = base.repoId || "";
  const memoryGb = (gpuStatus?.memoryTotal || 0) / 1024;
  const isQwen3 = isQwen3Family(repo);
  const isVision = isVisionFamily(repo);
  const isGguf = repo.toLowerCase().includes("gguf");

  // Hybrid-thinking Qwen3.5+ opens every assistant turn with ` thinking`. Without
  // a parser that block stays in `message.content` and the dataset generator
  // spends its output budget on text it then strips, so this is set in both
  // profiles rather than being treated as an optimization.
  const reasoningParser = isQwen3 ? "qwen3" : "";
  const tensorParallel = Math.max(base.tensorParallel || 1, 1);
  const toolCallParser = isQwen3 ? "qwen3_coder" : "";

  if (base.servingProfile === "optimized") {
    return {
      ...base,
      maxModelLen: optimizedMaxModelLen(memoryGb),
      dtype: "bfloat16",
      tensorParallel,
      gpuMemoryUtilization: 0.90,
      enableChunkedPrefill: true,
      maxNumBatchedTokens: 16384,
      maxNumSeqs: optimizedMaxNumSeqs(memoryGb),
      enableAutoToolChoice: isQwen3,
      toolCallParser,
      reasoningParser,
      servingEngine: "vllm",
    };
  }

  let maxModelLen = Math.max(base.maxModelLen || 32768, 32768);
  if (isQwen3 && isVision && memoryGb >= 180) maxModelLen = 100000;
  else if ((isQwen3 || isVision) && memoryGb >= 96) maxModelLen = 65536;
  else if (isGguf) maxModelLen = 32768;

  const maxNumBatchedTokens = memoryGb > 0 && memoryGb < 64 ? 4096 : 8192;
  const maxNumSeqs = memoryGb > 0 && memoryGb < 64 ? 4 : memoryGb > 0 && memoryGb < 128 ? 8 : 16;

  return {
    ...base,
    maxModelLen,
    dtype: "bfloat16",
    tensorParallel,
    gpuMemoryUtilization: 0.80,
    enableChunkedPrefill: true,
    maxNumBatchedTokens,
    maxNumSeqs,
    enableAutoToolChoice: isQwen3,
    toolCallParser,
    reasoningParser,
    servingEngine: "vllm",
  };
}

export function teacherConfigEquals(a: TeacherConfig, b: TeacherConfig): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}
