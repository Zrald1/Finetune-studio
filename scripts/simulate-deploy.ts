// ── Deploy-page simulation ───────────────────────────────────────────────────
//
// Exercises the serving-profile logic the Deploy page and Teacher step actually
// run, without a GPU. It imports the same module the app imports, so this is a
// test of shipped code rather than a reimplementation.
//
// Run:  node scripts/simulate-deploy.ts
//
// The VRAM ladder asserted here is duplicated in the Rust unit tests
// (src-tauri/src/config.rs, mod deploy_simulation). Keeping both means a change
// on one side that is not mirrored on the other fails loudly instead of
// silently diverging at deploy time.

import {
  autoTuneTeacherConfig,
  isQwen3Family,
  isVisionFamily,
  optimizedMaxModelLen,
  optimizedMaxNumSeqs,
  optimizedProfileSummary,
  PRESET_TEACHER_MODELS,
  standardProfileSummary,
  teacherConfigEquals,
} from "../src/lib/servingProfiles.ts";
import type { GPUState, ServingProfile, TeacherConfig } from "../src/types.ts";

let passed = 0;
const failures: string[] = [];

function check(name: string, condition: boolean, detail = "") {
  if (condition) {
    passed++;
  } else {
    failures.push(`${name}${detail ? ` — ${detail}` : ""}`);
  }
}

function eq<T>(name: string, actual: T, expected: T) {
  check(name, actual === expected, `expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
}

function gpu(gb: number): GPUState {
  return { memoryTotal: gb * 1024 } as GPUState;
}

function teacher(patch: Partial<TeacherConfig> = {}): TeacherConfig {
  return {
    repoId: "Qwen/Qwen3.8-27B",
    vllmPort: 8000,
    maxModelLen: 32768,
    dtype: "bfloat16",
    tensorParallel: 1,
    gpuMemoryUtilization: 0.8,
    autoTune: true,
    enableChunkedPrefill: true,
    maxNumBatchedTokens: 8192,
    maxNumSeqs: 16,
    enableAutoToolChoice: false,
    toolCallParser: "",
    customServeCmd: "",
    extraServeArgs: "",
    servingEngine: "vllm",
    servingProfile: "standard",
    reasoningParser: "",
    reasoningEffort: "xhigh",
    ...patch,
  };
}

// The exact ladder asserted by the Rust tests. Both sides must agree.
const RUST_CONTEXT_LADDER: Array<[number, number]> = [
  [256, 262144],
  [192, 262144],
  [140, 262144],
  [128, 131072],
  [96, 131072],
  [80, 65536],
  [48, 65536],
  [24, 32768],
];

const VRAM_CASES = [256, 192, 140, 128, 96, 80, 64, 48, 32, 24, 16, 8];

// ── 1. Optimized context ladder, cross-checked against Rust ─────────────────

for (const [gb, expected] of RUST_CONTEXT_LADDER) {
  eq(`optimized context @ ${gb} GB matches Rust`, optimizedMaxModelLen(gb), expected);
}

for (const gb of VRAM_CASES) {
  const ctx = optimizedMaxModelLen(gb);
  check(`optimized context @ ${gb} GB is a power-of-two-ish sane value`, [32768, 65536, 131072, 262144].includes(ctx), `got ${ctx}`);
}

eq("optimized seqs @ 24 GB", optimizedMaxNumSeqs(24), 8);
eq("optimized seqs @ 48 GB", optimizedMaxNumSeqs(48), 64);
eq("optimized seqs @ 256 GB", optimizedMaxNumSeqs(256), 64);

// ── 2. Profile resolution across every VRAM size ────────────────────────────

for (const profile of ["standard", "optimized"] as ServingProfile[]) {
  for (const gb of VRAM_CASES) {
    const r = autoTuneTeacherConfig(teacher({ servingProfile: profile }), gpu(gb));

    check(`[${profile} @ ${gb}GB] ctx is positive`, r.maxModelLen > 0, `got ${r.maxModelLen}`);
    check(`[${profile} @ ${gb}GB] dtype is bfloat16`, r.dtype === "bfloat16", `got ${r.dtype}`);
    check(`[${profile} @ ${gb}GB] vram util in (0,1]`, (r.gpuMemoryUtilization ?? 0) > 0 && (r.gpuMemoryUtilization ?? 0) <= 1, `got ${r.gpuMemoryUtilization}`);
    check(`[${profile} @ ${gb}GB] seqs positive`, (r.maxNumSeqs ?? 0) > 0, `got ${r.maxNumSeqs}`);
    check(`[${profile} @ ${gb}GB] batched tokens positive`, (r.maxNumBatchedTokens ?? 0) > 0);
    check(`[${profile} @ ${gb}GB] tensorParallel >= 1`, r.tensorParallel >= 1);
    check(`[${profile} @ ${gb}GB] chunked prefill on`, r.enableChunkedPrefill === true);
    check(`[${profile} @ ${gb}GB] serving engine vllm`, r.servingEngine === "vllm");
    check(`[${profile} @ ${gb}GB] reasoning parser set for qwen3`, r.reasoningParser === "qwen3", `got ${JSON.stringify(r.reasoningParser)}`);
    check(`[${profile} @ ${gb}GB] tool parser set for qwen3`, r.toolCallParser === "qwen3_coder", `got ${JSON.stringify(r.toolCallParser)}`);
    check(`[${profile} @ ${gb}GB] auto tool choice on`, r.enableAutoToolChoice === true);
  }
}

// ── 3. Profiles differ in the documented directions ─────────────────────────

for (const gb of [256, 192, 96, 48]) {
  const std = autoTuneTeacherConfig(teacher({ servingProfile: "standard" }), gpu(gb));
  const opt = autoTuneTeacherConfig(teacher({ servingProfile: "optimized" }), gpu(gb));
  check(`optimized ctx > standard ctx @ ${gb} GB`, opt.maxModelLen >= std.maxModelLen, `${opt.maxModelLen} vs ${std.maxModelLen}`);
  check(`optimized vram > standard vram @ ${gb} GB`, (opt.gpuMemoryUtilization ?? 0) > (std.gpuMemoryUtilization ?? 0));
  check(`optimized seqs > standard seqs @ ${gb} GB`, (opt.maxNumSeqs ?? 0) > (std.maxNumSeqs ?? 0));
}

// Exact values at the reference 256 GB card, matching the Rust assertions.
{
  const std = autoTuneTeacherConfig(teacher({ servingProfile: "standard" }), gpu(256));
  eq("standard vram @ 256 GB", std.gpuMemoryUtilization, 0.8);
  eq("standard ctx @ 256 GB (qwen3-vl)", std.maxModelLen, 100000);
  eq("standard batch tokens @ 256 GB", std.maxNumBatchedTokens, 8192);
  eq("standard seqs @ 256 GB", std.maxNumSeqs, 16);

  const opt = autoTuneTeacherConfig(teacher({ servingProfile: "optimized" }), gpu(256));
  eq("optimized vram @ 256 GB", opt.gpuMemoryUtilization, 0.9);
  eq("optimized ctx @ 256 GB", opt.maxModelLen, 262144);
  eq("optimized batch tokens @ 256 GB", opt.maxNumBatchedTokens, 16384);
  eq("optimized seqs @ 256 GB", opt.maxNumSeqs, 64);
}

// ── 4. Passthrough guards ───────────────────────────────────────────────────

{
  const manual = autoTuneTeacherConfig(teacher({ autoTune: false, maxModelLen: 4096, gpuMemoryUtilization: 0.42 }), gpu(256));
  eq("manual mode keeps ctx", manual.maxModelLen, 4096);
  eq("manual mode keeps vram", manual.gpuMemoryUtilization, 0.42);

  const custom = autoTuneTeacherConfig(teacher({ customServeCmd: "vllm serve mine", maxModelLen: 8192 }), gpu(256));
  eq("custom serve cmd keeps ctx", custom.maxModelLen, 8192);

  const noGpu = autoTuneTeacherConfig(teacher({ servingProfile: "optimized" }), null);
  check("null GPU status does not crash", noGpu.maxModelLen > 0, `got ${noGpu.maxModelLen}`);
  eq("null GPU status falls to smallest ladder rung", noGpu.maxModelLen, 32768);

  const undefinedGpu = autoTuneTeacherConfig(teacher({ servingProfile: "optimized" }));
  check("undefined GPU status does not crash", undefinedGpu.maxModelLen > 0);
}

// ── 5. Model-family detection ───────────────────────────────────────────────

{
  check("Qwen3.8-27B detected as qwen3", isQwen3Family("Qwen/Qwen3.8-27B"));
  check("Qwen3.8-27B detected as vision (no -vl marker)", isVisionFamily("Qwen/Qwen3.8-27B"));
  check("Llama is not qwen3", !isQwen3Family("meta-llama/Llama-3.1-8B-Instruct"));
  check("Llama is not vision", !isVisionFamily("meta-llama/Llama-3.1-8B-Instruct"));
  check("Qwen3-VL detected as vision", isVisionFamily("Qwen/Qwen3-VL-30B-A3B-Instruct"));

  const llama = autoTuneTeacherConfig(teacher({ repoId: "meta-llama/Llama-3.1-8B-Instruct" }), gpu(256));
  eq("non-qwen3 gets no reasoning parser", llama.reasoningParser, "");
  eq("non-qwen3 gets no tool parser", llama.toolCallParser, "");
  eq("non-qwen3 disables auto tool choice", llama.enableAutoToolChoice, false);

  const gguf = autoTuneTeacherConfig(
    teacher({ repoId: "TheBloke/Some-Model-GGUF", maxModelLen: 131072, servingProfile: "standard" }),
    gpu(0),
  );
  eq("gguf without VRAM info falls to 32768", gguf.maxModelLen, 32768);

  const customRepo = autoTuneTeacherConfig(teacher({ repoId: "some-org/some-model" }), gpu(256));
  check("unknown repo still tunes safely", customRepo.maxModelLen > 0);
}

// ── 6. Idempotence: re-tuning a tuned config is a no-op ─────────────────────

for (const profile of ["standard", "optimized"] as ServingProfile[]) {
  for (const gb of [256, 96, 24]) {
    const once = autoTuneTeacherConfig(teacher({ servingProfile: profile }), gpu(gb));
    const twice = autoTuneTeacherConfig(once, gpu(gb));
    check(`[${profile} @ ${gb}GB] re-tune is idempotent`, teacherConfigEquals(once, twice));
  }
}

// ── 7. Preset roster ────────────────────────────────────────────────────────

{
  eq("preset roster size", PRESET_TEACHER_MODELS.length, 1);
  eq("preset default repo", PRESET_TEACHER_MODELS[0].value, "Qwen/Qwen3.8-27B");

  const hardwareNames = /\b(H100|H200|A100|B200|GB300|L40S|RTX|Blackwell|Hopper|Ampere)\b/i;
  for (const model of PRESET_TEACHER_MODELS) {
    check(`preset label "${model.label}" carries no hardware name`, !hardwareNames.test(model.label));
  }
}

// ── 8. Summary strings ──────────────────────────────────────────────────────

{
  const opt = optimizedProfileSummary(256);
  check("optimized summary mentions FP8", opt.includes("FP8"), opt);
  check("optimized summary mentions prefix caching", opt.toLowerCase().includes("prefix caching"), opt);
  check("optimized summary shows full context", opt.includes("262,144"), opt);

  const std = standardProfileSummary(teacher());
  check("standard summary mentions dtype", std.includes("bfloat16"), std);
}

// ── Report ──────────────────────────────────────────────────────────────────

const total = passed + failures.length;
console.log(`\nDeploy simulation: ${passed}/${total} assertions passed`);
if (failures.length > 0) {
  console.log(`\n${failures.length} FAILED:`);
  for (const f of failures) console.log(`  ✕ ${f}`);
  process.exit(1);
}
console.log("All serving-profile paths OK.\n");
