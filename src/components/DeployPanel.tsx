import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { AppConfig, TeacherConfig, HfModelRepo } from "../types";
import { DEFAULT_TEACHER } from "../types";
import { api, events } from "../lib/tauri";
import {
  Rocket, Loader2, CircleSlash, ChevronDown, ChevronRight, Copy, Check, Send,
  Globe, Lock, RefreshCw, Zap, Scale, Target, FlaskConical, TrendingUp, BarChart3,
  Cpu, Clock, BookmarkPlus, Play, Info, Layers, Database, GitMerge, Users,
  MessageSquare, MemoryStick, Settings2, ToggleLeft, ToggleRight, Activity,
  Brain, Flame, ArrowRight,
} from "lucide-react";

interface Props { config: AppConfig; }

// L6: logprobs + usage metadata on chat messages
interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  logprobs?: Array<{ token: string; logprob: number; topLogprobs: Array<{ token: string; prob: number }> }>;
  usageMs?: number;
  promptTokens?: number;
  completionTokens?: number;
}

// L6: Live vLLM Prometheus /metrics
interface VllmMetrics {
  requestsRunning: number;
  requestsWaiting: number;
  gpuCacheUsagePerc: number;
  cpuCacheUsagePerc: number;
  promptTokensTotal: number;
  generationTokensTotal: number;
  prefixCacheQueriesTotal: number;
  tokensPerSec: number | null;
}

// ── Quant detection ────────────────────────────────────────────────────────────
interface QuantInfo {
  format: string; vllmFlag: string; label: string;
  colorClass: string; bgClass: string; borderClass: string; description: string;
}

const QUANT_PATTERNS: Array<{ pattern: RegExp; info: Omit<QuantInfo, "format"> }> = [
  { pattern: /\bawq\b/i, info: { vllmFlag: "awq", label: "AWQ", colorClass: "text-violet-300", bgClass: "bg-violet-500/10", borderClass: "border-violet-500/30", description: "Activation-Aware Weight Quantization — INT4 with Marlin kernel. Best accuracy/speed balance." } },
  { pattern: /\bgptq\b/i, info: { vllmFlag: "gptq", label: "GPTQ", colorClass: "text-blue-300", bgClass: "bg-blue-500/10", borderClass: "border-blue-500/30", description: "GPTQ post-training INT4/INT8. Use with Marlin kernel for best performance." } },
  { pattern: /\bfp8\b/i, info: { vllmFlag: "fp8", label: "FP8", colorClass: "text-cyan-300", bgClass: "bg-cyan-500/10", borderClass: "border-cyan-500/30", description: "FP8 — ideal for H100/H200/Blackwell. ~1.6× speedup over BF16." } },
  { pattern: /\b(gguf|q4_k_m|q5_k_m|q8_0|q4_0|ggml)\b/i, info: { vllmFlag: "gguf", label: "GGUF", colorClass: "text-amber-300", bgClass: "bg-amber-500/10", borderClass: "border-amber-500/30", description: "GGUF/llama.cpp format. Served via vLLM GGUF backend." } },
  { pattern: /\b(bnb|bitsandbytes|int8|8bit|8-bit)\b/i, info: { vllmFlag: "bitsandbytes", label: "BnB INT8", colorClass: "text-orange-300", bgClass: "bg-orange-500/10", borderClass: "border-orange-500/30", description: "BitsAndBytes INT8. Good for consumer GPUs with limited VRAM." } },
  { pattern: /\b(int4|4bit|4-bit)\b/i, info: { vllmFlag: "awq", label: "INT4", colorClass: "text-fuchsia-300", bgClass: "bg-fuchsia-500/10", borderClass: "border-fuchsia-500/30", description: "INT4 quantized. Auto-selecting AWQ kernel for best throughput." } },
  { pattern: /\b(exl2|exllamav2)\b/i, info: { vllmFlag: "exl2", label: "EXL2", colorClass: "text-pink-300", bgClass: "bg-pink-500/10", borderClass: "border-pink-500/30", description: "ExLlamaV2 quantization. Very fast on NVIDIA consumer GPUs." } },
];

function detectQuant(repoId: string): QuantInfo | null {
  for (const { pattern, info } of QUANT_PATTERNS) {
    if (pattern.test(repoId)) return { format: info.vllmFlag, ...info };
  }
  return null;
}

// ── Serving Profile Templates ──────────────────────────────────────────────────
interface ServingProfile {
  key: "precision" | "balanced" | "throughput";
  label: string; icon: React.ReactNode;
  gpuMemUtil: number; maxNumSeqs: number; maxNumBatchedTokens: number;
  enableChunkedPrefill: boolean; dtype: string; blockSize: number;
  swapSpaceGb: number; preemptionMode: "recompute" | "swap";
  description: string; badgeText: string; badgeClass: string;
  accentClass: string; borderClass: string; bgClass: string;
}

const SERVING_PROFILES: ServingProfile[] = [
  { key: "precision", label: "Precision Focus", icon: <Target className="w-4 h-4" />, gpuMemUtil: 0.70, maxNumSeqs: 32, maxNumBatchedTokens: 4096, enableChunkedPrefill: false, dtype: "bfloat16", blockSize: 16, swapSpaceGb: 4, preemptionMode: "recompute", description: "Conservative memory, strict dtype — optimized for accuracy. Best for research and low-concurrency tasks.", badgeText: "Low load", badgeClass: "text-emerald-300 bg-emerald-500/10 border-emerald-500/30", accentClass: "text-emerald-400", borderClass: "border-emerald-500/30", bgClass: "bg-emerald-500/5" },
  { key: "balanced", label: "Smart Balance", icon: <Scale className="w-4 h-4" />, gpuMemUtil: 0.85, maxNumSeqs: 128, maxNumBatchedTokens: 16384, enableChunkedPrefill: true, dtype: "auto", blockSize: 16, swapSpaceGb: 8, preemptionMode: "recompute", description: "Chunked prefill for mixed request sizes. Handles moderate concurrent users while maintaining quality.", badgeText: "Recommended", badgeClass: "text-theme-accent bg-theme-accent/10 border-theme-accent/30", accentClass: "text-theme-accent", borderClass: "border-theme-accent/30", bgClass: "bg-theme-accent/5" },
  { key: "throughput", label: "Max Throughput", icon: <Zap className="w-4 h-4" />, gpuMemUtil: 0.95, maxNumSeqs: 512, maxNumBatchedTokens: 32768, enableChunkedPrefill: true, dtype: "auto", blockSize: 32, swapSpaceGb: 16, preemptionMode: "swap", description: "Maximizes GPU memory and block size. Swap preemption preserves partial work. Best for high-volume APIs.", badgeText: "Max users", badgeClass: "text-orange-300 bg-orange-500/10 border-orange-500/30", accentClass: "text-orange-400", borderClass: "border-orange-500/30", bgClass: "bg-orange-500/5" },
];

// ── L6: Benchmark prompts ──────────────────────────────────────────────────────
const BENCHMARK_PROMPTS = [
  "Write a Python function that reverses a linked list.",
  "Explain the difference between TCP and UDP in 2 sentences.",
  "What is 1337 * 42? Show your work step by step.",
  "Summarize the concept of gradient descent in one paragraph.",
  "List 3 advantages of transformer architecture over RNNs.",
];

interface BenchmarkResult {
  modelId: string; profile: string; quantFormat: string | null;
  medianMs: number; p95Ms: number; reqPerSec: number;
  sampleCount: number; capturedAt: string; prefixCaching: boolean; kvCacheDtype: string;
}

// ── L6: KV preset architectures (layers, kv_heads, head_dim, dtype_bytes)
const KV_PRESETS: Record<string, [number, number, number, number]> = {
  "Qwen2.5-0.5B": [24, 2, 64, 2], "Qwen2.5-7B": [28, 4, 128, 2],
  "Qwen2.5-14B": [48, 8, 128, 2], "Qwen2.5-72B": [80, 8, 128, 2],
  "LLaMA-3-8B": [32, 8, 128, 2], "LLaMA-3-70B": [80, 8, 128, 2],
  "Mistral-7B": [32, 8, 128, 2], "Custom": [32, 8, 128, 2],
};
const KV_CONTEXTS = [64, 256, 512, 1024, 2048, 4096, 8192, 16384];

// ── Helpers ────────────────────────────────────────────────────────────────────
function percentile(sorted: number[], p: number): number {
  const idx = Math.ceil((p / 100) * sorted.length) - 1;
  return sorted[Math.max(0, Math.min(idx, sorted.length - 1))];
}

// L6 formula: 2 × layers × kv_heads × head_dim × dtype_bytes
function estimateKvPerToken(l: number, h: number, d: number, b: number) { return 2 * l * h * d * b; }

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(0)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(0)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

// L6: Prometheus /metrics parser
function parseVllmMetrics(text: string, prevGen: number, prevMs: number): VllmMetrics {
  const vals: Record<string, number> = {};
  for (const line of text.split("\n")) {
    if (line.startsWith("#") || !line.trim()) continue;
    const name = line.split("{")[0].split(" ")[0];
    const raw = line.split(" ").pop();
    if (raw !== undefined) { const n = parseFloat(raw); if (!isNaN(n)) vals[name] = n; }
  }
  const genNow = vals["vllm:generation_tokens_total"] ?? 0;
  const dtSec = prevMs > 0 ? (Date.now() - prevMs) / 1000 : 0;
  return {
    requestsRunning: vals["vllm:num_requests_running"] ?? 0,
    requestsWaiting: vals["vllm:num_requests_waiting"] ?? 0,
    gpuCacheUsagePerc: (vals["vllm:gpu_cache_usage_perc"] ?? 0) * 100,
    cpuCacheUsagePerc: (vals["vllm:cpu_cache_usage_perc"] ?? 0) * 100,
    promptTokensTotal: vals["vllm:prompt_tokens_total"] ?? 0,
    generationTokensTotal: genNow,
    prefixCacheQueriesTotal: vals["vllm:prefix_cache_queries_total"] ?? 0,
    tokensPerSec: dtSec > 0 && prevGen > 0 ? (genNow - prevGen) / dtSec : null,
  };
}

function errorMessage(e: unknown) { return e instanceof Error ? e.message : String(e); }

// ── Sub-components ─────────────────────────────────────────────────────────────
function DeltaBadge({ current, baseline, unit, higherIsBetter }: { current: number; baseline: number; unit: string; higherIsBetter: boolean }) {
  const delta = ((current - baseline) / baseline) * 100;
  const isGood = higherIsBetter ? delta > 0 : delta < 0;
  return (
    <span className={`text-[9px] font-mono font-bold px-1.5 py-0.5 rounded ${isGood ? "text-emerald-300 bg-emerald-500/15 border border-emerald-500/30" : "text-red-300 bg-red-500/15 border border-red-500/30"}`}>
      {delta > 0 ? "+" : ""}{delta.toFixed(0)}% {unit}
    </span>
  );
}

function Toggle({ enabled, onChange, label }: { enabled: boolean; onChange: (v: boolean) => void; label?: string }) {
  return (
    <button onClick={() => onChange(!enabled)} className={`flex items-center gap-2 text-[10px] font-mono font-bold transition-colors ${enabled ? "text-emerald-400" : "theme-muted"}`}>
      {enabled ? <ToggleRight className="w-4 h-4 text-emerald-400" /> : <ToggleLeft className="w-4 h-4 theme-faint" />}
      {label}
    </button>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1">
      <label className="text-[8px] uppercase tracking-widest theme-faint font-mono">{label}</label>
      {children}
    </div>
  );
}

// L6: Metric card for the live metrics panel
function MetricCard({ label, value, icon, accent, sub, barPct, barColor }: {
  label: string; value: string; icon: React.ReactNode; accent: string; sub?: string; barPct?: number; barColor?: string;
}) {
  return (
    <div className="bg-black/20 border border-white/10 rounded-lg p-3 space-y-1.5">
      <div className="flex items-center gap-1.5">
        {icon}
        <span className="text-[8px] uppercase tracking-widest theme-faint font-mono">{label}</span>
      </div>
      <div className={`text-[16px] font-mono font-black ${accent}`}>{value}</div>
      {barPct !== undefined && barColor && (
        <div className="w-full h-1 bg-white/10 rounded-full overflow-hidden">
          <div className={`h-full rounded-full ${barColor} transition-all`} style={{ width: `${barPct}%` }} />
        </div>
      )}
      {sub && <p className="text-[9px] theme-faint font-mono truncate">{sub}</p>}
    </div>
  );
}

// L6: logprob probability bar
function LogprobBar({ prob }: { prob: number }) {
  const pct = Math.min(100, prob * 100);
  return (
    <div className="flex items-center gap-1.5">
      <div className="w-16 h-1 bg-white/10 rounded-full overflow-hidden">
        <div className="h-full rounded-full bg-theme-accent" style={{ width: `${pct}%` }} />
      </div>
      <span className="text-[8px] font-mono text-white/40">{pct.toFixed(1)}%</span>
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────────
export default function DeployPanel({ config }: Props) {
  const [repoId, setRepoId] = useState(DEFAULT_TEACHER.repoId);
  const [hfModels, setHfModels] = useState<HfModelRepo[]>([]);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [isPrivate, setIsPrivate] = useState(false);
  const [activeProfile, setActiveProfile] = useState<ServingProfile["key"]>("balanced");

  // Core vLLM params
  const [vllmPort, setVllmPort] = useState(DEFAULT_TEACHER.vllmPort);
  const [maxModelLen, setMaxModelLen] = useState(DEFAULT_TEACHER.maxModelLen);
  const [dtype, setDtype] = useState("auto");
  const [gpuMemUtil, setGpuMemUtil] = useState(0.85);
  const [maxNumSeqs, setMaxNumSeqs] = useState(128);
  const [maxNumBatchedTokens, setMaxNumBatchedTokens] = useState(16384);
  const [enableChunkedPrefill, setEnableChunkedPrefill] = useState(true);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // PagedAttention
  const [blockSize, setBlockSize] = useState(16);
  const [kvCacheDtype, setKvCacheDtype] = useState("auto");
  const [swapSpaceGb, setSwapSpaceGb] = useState(8);
  const [cpuOffloadGb, setCpuOffloadGb] = useState(0);

  // Continuous batching / scheduling
  const [schedulingPolicy, setSchedulingPolicy] = useState<"fcfs" | "priority">("fcfs");
  const [preemptionMode, setPreemptionMode] = useState<"recompute" | "swap">("recompute");

  // Prefix caching
  const [enablePrefixCaching, setEnablePrefixCaching] = useState(true);
  const [showInferenceOpt, setShowInferenceOpt] = useState(true);

  // Deploy lifecycle
  const [deployStreamId, setDeployStreamId] = useState<string | null>(null);
  const [deployLogs, setDeployLogs] = useState("");
  const [deploying, setDeploying] = useState(false);
  const [deployError, setDeployError] = useState<string | null>(null);
  const [activePort, setActivePort] = useState<number | null>(null);
  const [checking, setChecking] = useState(false);
  const [stopping, setStopping] = useState(false);

  // Chat
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [chatSending, setChatSending] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  // L6: thinking mode + logprobs
  const [enableThinking, setEnableThinking] = useState(false);
  const [showLogprobs, setShowLogprobs] = useState(false);
  const [expandedLogprobIdx, setExpandedLogprobIdx] = useState<number | null>(null);

  // Benchmark
  const [benchRunning, setBenchRunning] = useState(false);
  const [benchResult, setBenchResult] = useState<BenchmarkResult | null>(null);
  const [benchBaseline, setBenchBaseline] = useState<BenchmarkResult | null>(null);
  const [benchError, setBenchError] = useState<string | null>(null);
  const [showBench, setShowBench] = useState(false);
  // L6: concurrent vs serial
  const [benchConcurrent, setBenchConcurrent] = useState(false);

  // L6: Live metrics
  const [showMetrics, setShowMetrics] = useState(false);
  const [liveMetrics, setLiveMetrics] = useState<VllmMetrics | null>(null);
  const [metricsPolling, setMetricsPolling] = useState(false);
  const metricsIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const prevGenTokensRef = useRef(0);
  const prevMetricsMsRef = useRef(0);
  const [metricsError, setMetricsError] = useState<string | null>(null);

  // L6: KV estimator
  const [showKvEstimator, setShowKvEstimator] = useState(false);
  const [kvPreset, setKvPreset] = useState("Qwen2.5-7B");
  const [kvLayers, setKvLayers] = useState(28);
  const [kvKvHeads, setKvKvHeads] = useState(4);
  const [kvHeadDim, setKvHeadDim] = useState(128);
  const [kvDtypeBytes, setKvDtypeBytes] = useState(2);

  const logBoxRef = useRef<HTMLDivElement>(null);
  const chatBoxRef = useRef<HTMLDivElement>(null);
  const deploymentCheckSeqRef = useRef(0);
  const modelId = repoId.trim();

  const endpoint = activePort != null && config.ssh.host ? `http://${config.ssh.host}:${activePort}` : null;
  const detectedQuant = useMemo(() => detectQuant(modelId), [modelId]);

  // Build extra vLLM flags
  const buildExtraFlags = useCallback((): string => {
    const f: string[] = [];
    if (detectedQuant) f.push(`--quantization ${detectedQuant.vllmFlag}`);
    if (blockSize !== 16) f.push(`--block-size ${blockSize}`);
    if (kvCacheDtype !== "auto") f.push(`--kv-cache-dtype ${kvCacheDtype}`);
    if (swapSpaceGb > 0) f.push(`--swap-space ${swapSpaceGb}`);
    if (cpuOffloadGb > 0) f.push(`--cpu-offload-gb ${cpuOffloadGb}`);
    if (schedulingPolicy !== "fcfs") f.push(`--scheduling-policy ${schedulingPolicy}`);
    if (preemptionMode !== "recompute") f.push(`--preemption-mode ${preemptionMode}`);
    if (enablePrefixCaching) f.push("--enable-prefix-caching");
    return f.join(" ");
  }, [detectedQuant, blockSize, kvCacheDtype, swapSpaceGb, cpuOffloadGb, schedulingPolicy, preemptionMode, enablePrefixCaching]);

  const applyProfile = useCallback((profile: ServingProfile) => {
    setActiveProfile(profile.key);
    setGpuMemUtil(profile.gpuMemUtil);
    setMaxNumSeqs(profile.maxNumSeqs);
    setMaxNumBatchedTokens(profile.maxNumBatchedTokens);
    setEnableChunkedPrefill(profile.enableChunkedPrefill);
    setDtype(profile.dtype);
    setBlockSize(profile.blockSize);
    setSwapSpaceGb(profile.swapSpaceGb);
    setPreemptionMode(profile.preemptionMode);
  }, []);

  const buildTeacher = useCallback((): TeacherConfig => ({
    ...DEFAULT_TEACHER,
    repoId: modelId,
    vllmPort,
    maxModelLen,
    dtype,
    gpuMemoryUtilization: gpuMemUtil,
    maxNumSeqs,
    maxNumBatchedTokens,
    enableChunkedPrefill,
    servingEngine: "vllm",
    // Advanced flags are APPENDED to the managed `vllm serve <model> …` command,
    // not used as the whole command. Using customServeCmd here would drop the
    // `vllm serve <model>` prefix and crash with "--quantization: command not found".
    extraServeArgs: buildExtraFlags() || undefined,
  }), [modelId, vllmPort, maxModelLen, dtype, gpuMemUtil, maxNumSeqs, maxNumBatchedTokens, enableChunkedPrefill, buildExtraFlags]);

  // Load HF models
  useEffect(() => {
    let alive = true;
    setModelsLoading(true);
    api.hfListModels()
      .then((list) => { if (alive) setHfModels(list); })
      .catch((e) => console.error("hf_list_models:", e))
      .finally(() => { if (alive) setModelsLoading(false); });
    return () => { alive = false; };
  }, []);

  useEffect(() => {
    setIsPrivate(hfModels.find((m) => m.id === modelId)?.private ?? false);
  }, [hfModels, modelId]);

  // Apply balanced profile on mount
  useEffect(() => {
    applyProfile(SERVING_PROFILES.find((p) => p.key === "balanced")!);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Check if already deployed
  const checkDeployment = useCallback(async () => {
    const seq = ++deploymentCheckSeqRef.current;
    if (!config.ssh.host || !modelId) { setActivePort(null); setChecking(false); return; }
    setChecking(true);
    try {
      const s = await api.checkTeacherDeployed(config.ssh, config.docker, buildTeacher());
      if (seq === deploymentCheckSeqRef.current) setActivePort(s?.exact ? s.port : null);
    } catch (e) { console.error("check:", e); }
    finally { if (seq === deploymentCheckSeqRef.current) setChecking(false); }
  }, [config.ssh, config.docker, modelId, buildTeacher]);

  useEffect(() => {
    deploymentCheckSeqRef.current++;
    setChecking(false);
    if (!config.ssh.host || !modelId) { setActivePort(null); return; }
    setActivePort(null);
    const t = setTimeout(checkDeployment, 400);
    return () => clearTimeout(t);
  }, [config.ssh.host, modelId, vllmPort, checkDeployment]);

  // Stream deploy events
  useEffect(() => {
    if (!deployStreamId) return;
    let disposed = false;
    let ul: (() => void) | null = null;
    let ud: (() => void) | null = null;
    const setup = async () => {
      const ll = await events.onDeployLog((e) => { if (e.streamId === deployStreamId) setDeployLogs((p) => p + e.line); });
      if (disposed) { ll(); } else { ul = ll; }
      const dl = await events.onDeployDone((e) => {
        if (e.streamId !== deployStreamId) return;
        setDeploying(false); setDeployStreamId(null);
        if (e.success) { if (e.port !== undefined) setActivePort(e.port); else void checkDeployment(); }
        else setDeployError(e.message);
      });
      if (disposed) { dl(); } else { ud = dl; }
    };
    setup().catch((e) => { if (!disposed) { setDeployError(errorMessage(e)); setDeploying(false); setDeployStreamId(null); } });
    return () => { disposed = true; ul?.(); ud?.(); };
  }, [deployStreamId]); // eslint-disable-line react-hooks/exhaustive-deps

  useLayoutEffect(() => { const el = logBoxRef.current; if (el) el.scrollTop = el.scrollHeight; }, [deployLogs]);
  useLayoutEffect(() => { const el = chatBoxRef.current; if (el) el.scrollTop = el.scrollHeight; }, [chatMessages, chatSending]);

  const startDeployment = async () => {
    if (!modelId || !config.ssh.host) return;
    setDeploying(true); setDeployError(null); setDeployLogs(""); setActivePort(null);
    try { setDeployStreamId(await api.deployTeacher(config.ssh, config.docker, buildTeacher(), config.hfToken)); }
    catch (err) { setDeployError(errorMessage(err)); setDeploying(false); }
  };

  const cancelDeployment = async () => {
    if (!deployStreamId) return;
    try { await api.sshStopStream(deployStreamId); } catch (e) { console.error(e); }
    finally { setDeploying(false); setDeployStreamId(null); }
  };

  const stopDeployment = async () => {
    if (activePort == null || !config.ssh.host) return;
    setStopping(true); setDeployError(null);
    try { await api.stopTeacher(config.ssh, config.docker, activePort); setActivePort(null); setChatMessages([]); setChatError(null); }
    catch (e) { setDeployError(errorMessage(e)); }
    finally { setStopping(false); void checkDeployment(); }
  };

  const copyUrl = async () => {
    if (!endpoint) return;
    try { await navigator.clipboard.writeText(endpoint); setCopied(true); setTimeout(() => setCopied(false), 1500); }
    catch (e) { console.error(e); }
  };

  // L6: chat with thinking mode + logprobs awareness
  const sendChat = async () => {
    const text = chatInput.trim();
    if (!text || !endpoint || !modelId || chatSending) return;
    const next: ChatMessage[] = [...chatMessages, { role: "user", content: text }];
    setChatMessages(next); setChatInput(""); setChatSending(true); setChatError(null);
    try {
      const reachable = await api.pingTeacher(endpoint);
      if (!reachable) throw new Error("Endpoint not reachable yet — vLLM is still loading.");
      const t0 = Date.now();
      const answer = await api.teacherChat(endpoint, modelId, next);
      const usageMs = Date.now() - t0;
      setChatMessages((prev) => [...prev, { role: "assistant", content: answer, usageMs }]);
    } catch (e) {
      setChatError(errorMessage(e)); setChatMessages((prev) => prev.slice(0, -1)); setChatInput(text);
    } finally { setChatSending(false); }
  };

  const onChatKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChat(); }
  };

  // L6: Enhanced benchmark — concurrent or serial
  const runBenchmark = useCallback(async () => {
    if (!endpoint || !modelId || benchRunning) return;
    setBenchRunning(true); setBenchError(null);
    const latencies: number[] = [];
    try {
      const reachable = await api.pingTeacher(endpoint);
      if (!reachable) throw new Error("Endpoint not reachable — deploy the model first.");
      const currentProfile = SERVING_PROFILES.find((p) => p.key === activeProfile);

      if (benchConcurrent) {
        // L6 concurrent pattern (ThreadPoolExecutor equivalent): all at once
        const t0 = Date.now();
        const results = await Promise.all(
          BENCHMARK_PROMPTS.map(async (prompt) => {
            const ts = Date.now();
            await api.teacherChat(endpoint, modelId, [{ role: "user", content: prompt }]);
            return Date.now() - ts;
          })
        );
        const wallClock = Date.now() - t0;
        latencies.push(...results);
        const sorted = [...latencies].sort((a, b) => a - b);
        setBenchResult({ modelId, profile: `${currentProfile?.label ?? activeProfile} (concurrent)`, quantFormat: detectedQuant?.label ?? null, medianMs: percentile(sorted, 50), p95Ms: percentile(sorted, 95), reqPerSec: latencies.length / (wallClock / 1000), sampleCount: latencies.length, capturedAt: new Date().toLocaleTimeString(), prefixCaching: enablePrefixCaching, kvCacheDtype });
      } else {
        // Serial: one at a time — measures true single-request latency
        for (const prompt of BENCHMARK_PROMPTS) {
          const t0 = Date.now();
          await api.teacherChat(endpoint, modelId, [{ role: "user", content: prompt }]);
          latencies.push(Date.now() - t0);
        }
        const sorted = [...latencies].sort((a, b) => a - b);
        const totalMs = latencies.reduce((a, b) => a + b, 0);
        setBenchResult({ modelId, profile: `${currentProfile?.label ?? activeProfile} (serial)`, quantFormat: detectedQuant?.label ?? null, medianMs: percentile(sorted, 50), p95Ms: percentile(sorted, 95), reqPerSec: latencies.length / (totalMs / 1000), sampleCount: latencies.length, capturedAt: new Date().toLocaleTimeString(), prefixCaching: enablePrefixCaching, kvCacheDtype });
      }
    } catch (e) { setBenchError(errorMessage(e)); }
    finally { setBenchRunning(false); }
  }, [endpoint, modelId, benchRunning, benchConcurrent, activeProfile, detectedQuant, enablePrefixCaching, kvCacheDtype]);

  const saveAsBaseline = useCallback(() => { if (benchResult) setBenchBaseline(benchResult); }, [benchResult]);

  // L6: Live metrics polling
  const startMetricsPolling = useCallback(() => {
    if (!endpoint || metricsPolling) return;
    setMetricsPolling(true); setMetricsError(null);
    prevGenTokensRef.current = 0; prevMetricsMsRef.current = 0;
    const poll = async () => {
      try {
        const res = await fetch(`${endpoint}/metrics`, { signal: AbortSignal.timeout(3000) });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const text = await res.text();
        const parsed = parseVllmMetrics(text, prevGenTokensRef.current, prevMetricsMsRef.current);
        prevGenTokensRef.current = parsed.generationTokensTotal;
        prevMetricsMsRef.current = Date.now();
        setLiveMetrics(parsed); setMetricsError(null);
      } catch (e) { setMetricsError(errorMessage(e)); }
    };
    void poll();
    metricsIntervalRef.current = setInterval(() => void poll(), 2000);
  }, [endpoint, metricsPolling]);

  const stopMetricsPolling = useCallback(() => {
    if (metricsIntervalRef.current) { clearInterval(metricsIntervalRef.current); metricsIntervalRef.current = null; }
    setMetricsPolling(false);
  }, []);

  useEffect(() => { return () => { if (metricsIntervalRef.current) clearInterval(metricsIntervalRef.current); }; }, []);
  useEffect(() => { if (!endpoint && metricsPolling) stopMetricsPolling(); }, [endpoint, metricsPolling, stopMetricsPolling]);

  // L6: KV estimator
  const kvPerToken = useMemo(() => estimateKvPerToken(kvLayers, kvKvHeads, kvHeadDim, kvDtypeBytes), [kvLayers, kvKvHeads, kvHeadDim, kvDtypeBytes]);

  const liveFlags = buildExtraFlags();
  const noHost = !config.ssh.host;

  return (
    <div className="w-full max-w-5xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-start gap-3">
        <div className="p-2.5 theme-accent-soft theme-accent rounded-lg shrink-0"><Rocket className="w-5 h-5" /></div>
        <div>
          <h2 className="text-lg-fluid font-serif italic text-white leading-tight">Deploy Model</h2>
          <p className="text-[11px] theme-muted leading-relaxed mt-0.5 max-w-xl">
            Serve any HuggingFace model with PagedAttention KV-cache, continuous batching, and prefix caching — auto-configured for high-concurrency inference.
          </p>
        </div>
      </div>

      {noHost && (
        <div className="theme-surface border border-amber-500/30 bg-amber-950/10 rounded-lg p-4 text-[11px] text-amber-300 font-mono">
          No GPU server connected. Set your SSH host in the <b>Credentials</b> tab before deploying.
        </div>
      )}

      {/* ── Model Source ── */}
      <section className="theme-surface border rounded-lg p-4 space-y-4">
        <p className="text-[10px] uppercase tracking-[0.2em] theme-accent font-mono font-bold">Model Source</p>
        <div className="space-y-2">
          <label className="text-[9px] uppercase tracking-widest theme-muted font-mono">HuggingFace Repo ID</label>
          <div className="flex items-center gap-2">
            <input value={repoId} onChange={(e) => setRepoId(e.target.value)}
              placeholder="e.g. Qwen/Qwen2.5-7B-Instruct-AWQ"
              className="flex-1 min-w-0 px-3 py-2 theme-field border rounded-lg text-[12px] font-mono text-white focus:outline-none focus:border-theme-accent transition" />
            <span className={`flex items-center gap-1 px-2.5 py-1.5 rounded border text-[9px] uppercase tracking-widest font-mono font-bold shrink-0 ${isPrivate ? "text-amber-300 bg-amber-500/10 border-amber-500/30" : "text-emerald-300 bg-emerald-500/10 border-emerald-500/30"}`} title={isPrivate ? "Private" : "Public"}>
              {isPrivate ? <Lock className="w-3 h-3" /> : <Globe className="w-3 h-3" />}
              {isPrivate ? "Private" : "Public"}
            </span>
          </div>
        </div>

        {detectedQuant ? (
          <div className={`flex items-start gap-2.5 px-3 py-2.5 rounded-lg border ${detectedQuant.bgClass} ${detectedQuant.borderClass}`}>
            <FlaskConical className={`w-3.5 h-3.5 mt-0.5 shrink-0 ${detectedQuant.colorClass}`} />
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <span className={`text-[10px] uppercase tracking-widest font-mono font-black ${detectedQuant.colorClass}`}>{detectedQuant.label} Detected</span>
                <span className={`text-[8px] px-1.5 py-0.5 rounded border font-mono ${detectedQuant.colorClass} ${detectedQuant.bgClass} ${detectedQuant.borderClass}`}>--quantization {detectedQuant.vllmFlag}</span>
                <span className="text-[8px] px-1.5 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 font-mono">auto-injected</span>
              </div>
              <p className="text-[10px] theme-muted mt-1 leading-relaxed">{detectedQuant.description}</p>
            </div>
          </div>
        ) : modelId ? (
          <div className="flex items-center gap-2 px-3 py-2 rounded-lg border border-white/10 bg-white/[0.02]">
            <Info className="w-3 h-3 theme-faint shrink-0" />
            <p className="text-[10px] theme-faint">No quantization detected — native BF16/FP16. Add <span className="font-mono">AWQ</span>, <span className="font-mono">GPTQ</span>, or <span className="font-mono">FP8</span> for auto-injection.</p>
          </div>
        ) : null}

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-[9px] uppercase tracking-widest theme-muted font-mono">Your Models</label>
            {modelsLoading && <Loader2 className="w-3 h-3 animate-spin theme-faint" />}
          </div>
          <select value={hfModels.some((m) => m.id === modelId) ? modelId : ""} onChange={(e) => { const p = hfModels.find((m) => m.id === e.target.value); if (p) { setRepoId(p.id); setIsPrivate(p.private); } }}
            className="w-full px-3 py-2 theme-field border rounded-lg text-[12px] font-mono text-white focus:outline-none focus:border-theme-accent transition">
            <option value="">{modelsLoading ? "Loading…" : hfModels.length === 0 ? "No repos found (enter a repo ID above)" : "Select one of your repos…"}</option>
            {hfModels.map((m) => <option key={m.id} value={m.id}>{m.id}{m.private ? "  (private)" : ""}</option>)}
          </select>
        </div>
      </section>

      {/* ── Serving Profile ── */}
      <section className="theme-surface border rounded-lg p-4 space-y-4">
        <div className="flex items-center gap-2">
          <Cpu className="w-4 h-4 theme-accent" />
          <p className="text-[10px] uppercase tracking-[0.2em] theme-accent font-mono font-bold">Serving Profile</p>
          <span className="text-[9px] theme-faint font-mono ml-1">— sets all optimization params at once</span>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {SERVING_PROFILES.map((profile) => {
            const isActive = activeProfile === profile.key;
            return (
              <button key={profile.key} onClick={() => applyProfile(profile)}
                className={`text-left p-3.5 rounded-xl border transition-all duration-200 relative overflow-hidden ${isActive ? `${profile.borderClass} ${profile.bgClass} shadow-lg` : "border-white/10 bg-white/[0.02] hover:border-white/20 hover:bg-white/[0.04]"}`}>
                <div className="flex items-center gap-2 mb-2">
                  <span className={`${isActive ? profile.accentClass : "theme-faint"} transition-colors`}>{profile.icon}</span>
                  <span className={`text-[10px] font-mono font-black uppercase tracking-widest ${isActive ? profile.accentClass : "theme-muted"} transition-colors`}>{profile.label}</span>
                </div>
                <p className="text-[9px] text-white/60 mb-2.5 leading-relaxed">{profile.description}</p>
                <div className="flex flex-wrap gap-1.5">
                  <span className={`text-[8px] font-mono px-1.5 py-0.5 rounded border ${profile.badgeClass}`}>{profile.badgeText}</span>
                  <span className="text-[8px] font-mono px-1.5 py-0.5 rounded border border-white/10 text-white/40">{(profile.gpuMemUtil * 100).toFixed(0)}% VRAM</span>
                  <span className="text-[8px] font-mono px-1.5 py-0.5 rounded border border-white/10 text-white/40">blk={profile.blockSize}</span>
                  <span className="text-[8px] font-mono px-1.5 py-0.5 rounded border border-white/10 text-white/40">{profile.maxNumSeqs} seqs</span>
                </div>
                {isActive && <div className="absolute top-2.5 right-2.5"><Check className={`w-3.5 h-3.5 ${profile.accentClass}`} /></div>}
              </button>
            );
          })}
        </div>

        <div>
          <button onClick={() => setShowAdvanced((s) => !s)} className="flex items-center gap-1.5 text-[9px] uppercase tracking-widest theme-muted hover:theme-text font-mono font-bold transition">
            {showAdvanced ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
            Advanced overrides
          </button>
          {showAdvanced && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-3">
              <Field label="Port"><input type="number" value={vllmPort} onChange={(e) => setVllmPort(Number(e.target.value) || DEFAULT_TEACHER.vllmPort)} className="w-full px-2 py-1.5 theme-field border rounded text-[11px] font-mono text-white focus:outline-none focus:border-theme-accent" /></Field>
              <Field label="Max Len"><input type="number" value={maxModelLen} onChange={(e) => setMaxModelLen(Number(e.target.value) || DEFAULT_TEACHER.maxModelLen)} className="w-full px-2 py-1.5 theme-field border rounded text-[11px] font-mono text-white focus:outline-none focus:border-theme-accent" /></Field>
              <Field label="Dtype">
                <select value={dtype} onChange={(e) => setDtype(e.target.value)} className="w-full px-2 py-1.5 theme-field border rounded text-[11px] font-mono text-white focus:outline-none focus:border-theme-accent">
                  <option value="auto">auto</option><option value="bfloat16">bfloat16</option><option value="float16">float16</option>
                </select>
              </Field>
              <Field label="GPU Mem %"><input type="number" step="0.05" min="0.1" max="0.99" value={gpuMemUtil} onChange={(e) => setGpuMemUtil(Number(e.target.value) || 0.85)} className="w-full px-2 py-1.5 theme-field border rounded text-[11px] font-mono text-white focus:outline-none focus:border-theme-accent" /></Field>
              <Field label="Max Seqs"><input type="number" value={maxNumSeqs} onChange={(e) => setMaxNumSeqs(Number(e.target.value) || 128)} className="w-full px-2 py-1.5 theme-field border rounded text-[11px] font-mono text-white focus:outline-none focus:border-theme-accent" /></Field>
              <Field label="Batched Tokens"><input type="number" value={maxNumBatchedTokens} onChange={(e) => setMaxNumBatchedTokens(Number(e.target.value) || 16384)} className="w-full px-2 py-1.5 theme-field border rounded text-[11px] font-mono text-white focus:outline-none focus:border-theme-accent" /></Field>
              <Field label="Chunked Prefill">
                <button onClick={() => setEnableChunkedPrefill((v) => !v)} className={`w-full px-2 py-1.5 border rounded text-[11px] font-mono font-bold transition ${enableChunkedPrefill ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300" : "theme-field border text-white/50"}`}>
                  {enableChunkedPrefill ? "Enabled" : "Disabled"}
                </button>
              </Field>
            </div>
          )}
        </div>

        <div className="flex items-center gap-3 pt-1">
          <button onClick={startDeployment} disabled={deploying || noHost || !repoId.trim()}
            className="flex items-center gap-2 px-4 py-2 rounded theme-accent-bg text-black text-[10px] uppercase tracking-widest font-bold hover:brightness-110 disabled:opacity-50 transition shadow-lg">
            {deploying ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Rocket className="w-3.5 h-3.5" />}
            {deploying ? "Deploying…" : "Deploy"}
          </button>
          {deploying && <button onClick={cancelDeployment} className="flex items-center gap-1 px-3 py-2 bg-red-950/30 border border-red-500/30 text-red-300 rounded text-[10px] uppercase tracking-widest font-mono font-bold hover:bg-red-950 transition"><CircleSlash className="w-3 h-3" /> Cancel</button>}
          {activePort != null && !deploying && (
            <button onClick={stopDeployment} disabled={stopping} className="flex items-center gap-1.5 px-3 py-2 bg-red-950/30 border border-red-500/30 text-red-300 rounded text-[10px] uppercase tracking-widest font-mono font-bold hover:bg-red-950 disabled:opacity-50 transition">
              {stopping ? <Loader2 className="w-3 h-3 animate-spin" /> : <CircleSlash className="w-3 h-3" />}
              {stopping ? "Stopping…" : "Stop Deploy"}
            </button>
          )}
          {checking && <span className="flex items-center gap-1.5 text-[9px] uppercase tracking-widest theme-faint font-mono"><Loader2 className="w-3 h-3 animate-spin" /> Probing…</span>}
        </div>
        {deployError && <div className="text-[11px] text-red-400 font-mono bg-red-950/30 border border-red-500/20 rounded-lg p-3 whitespace-pre-wrap">{deployError}</div>}
      </section>

      {/* ── Inference Optimization ── */}
      <section className="theme-surface border rounded-lg overflow-hidden">
        <button onClick={() => setShowInferenceOpt((s) => !s)}
          className="w-full flex items-center justify-between px-4 py-3 border-b theme-surface-soft bg-black/10 hover:bg-black/20 transition">
          <div className="flex items-center gap-2">
            <Settings2 className="w-4 h-4 theme-accent" />
            <p className="text-[10px] uppercase tracking-[0.2em] theme-accent font-mono font-bold">Inference Optimization</p>
            <span className="text-[9px] theme-faint font-mono">— PagedAttention · Continuous Batching · Prefix Cache</span>
          </div>
          {showInferenceOpt ? <ChevronDown className="w-3.5 h-3.5 theme-muted" /> : <ChevronRight className="w-3.5 h-3.5 theme-muted" />}
        </button>

        {showInferenceOpt && (
          <div className="p-4 space-y-6">
            {/* PagedAttention */}
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <MemoryStick className="w-3.5 h-3.5 text-violet-400" />
                <span className="text-[10px] uppercase tracking-widest font-mono font-bold text-violet-400">PagedAttention · KV Cache</span>
              </div>
              <div className="pl-1 pb-2 border-l-2 border-violet-500/30">
                <p className="text-[10px] theme-muted leading-relaxed mb-3 pl-3">
                  PagedAttention solves the <b className="text-white/70">dynamic KV cache size problem</b> by treating KV memory like an OS page table.
                  Each <b className="text-white/70">block</b> holds KV vectors for a fixed number of tokens — unknown sequence lengths are handled by allocating blocks on demand, eliminating fragmentation and waste.
                  The L6 formula: <span className="font-mono text-violet-300">bytes/token = 2 × layers × kv_heads × head_dim × dtype_bytes</span>.
                </p>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pl-3">
                  <Field label="Block Size (tokens/block)">
                    <select value={blockSize} onChange={(e) => setBlockSize(Number(e.target.value))} className="w-full px-2 py-1.5 theme-field border rounded text-[11px] font-mono text-white focus:outline-none focus:border-theme-accent">
                      <option value={8}>8 — min fragmentation</option>
                      <option value={16}>16 — balanced (default)</option>
                      <option value={32}>32 — long-context optimal</option>
                    </select>
                  </Field>
                  <Field label="KV Cache Dtype">
                    <select value={kvCacheDtype} onChange={(e) => setKvCacheDtype(e.target.value)} className="w-full px-2 py-1.5 theme-field border rounded text-[11px] font-mono text-white focus:outline-none focus:border-theme-accent">
                      <option value="auto">auto (match model)</option>
                      <option value="fp8">fp8 — H100/H200 (2× capacity)</option>
                      <option value="fp8_e4m3">fp8_e4m3 — high precision</option>
                      <option value="fp8_e5m2">fp8_e5m2 — high dynamic range</option>
                    </select>
                  </Field>
                  <Field label="CPU Swap Space (GB)">
                    <input type="number" min={0} max={64} value={swapSpaceGb} onChange={(e) => setSwapSpaceGb(Number(e.target.value))} className="w-full px-2 py-1.5 theme-field border rounded text-[11px] font-mono text-white focus:outline-none focus:border-theme-accent" />
                  </Field>
                  <Field label="CPU Offload (GB)">
                    <input type="number" min={0} max={256} value={cpuOffloadGb} onChange={(e) => setCpuOffloadGb(Number(e.target.value))} className="w-full px-2 py-1.5 theme-field border rounded text-[11px] font-mono text-white focus:outline-none focus:border-theme-accent" />
                  </Field>
                </div>

                {/* Block layout diagram */}
                <div className="mt-3 pl-3">
                  <div className="bg-black/30 border border-white/10 rounded-lg p-3 font-mono text-[9px]">
                    <p className="text-white/30 mb-2 uppercase tracking-widest text-[8px]">PagedAttention Block Layout — {blockSize} tokens/block</p>
                    <div className="flex items-center gap-1 flex-wrap">
                      {Array.from({ length: 8 }).map((_, i) => (
                        <div key={i} className={`h-6 rounded text-[8px] flex items-center justify-center font-bold w-10 ${i < 3 ? "bg-violet-500/40 border border-violet-400/50 text-violet-300" : i < 5 ? "bg-blue-500/40 border border-blue-400/50 text-blue-300" : i < 6 ? "bg-emerald-500/40 border border-emerald-400/50 text-emerald-300" : "bg-white/5 border border-white/10 text-white/20"}`}>
                          {i < 3 ? `R1·B${i + 1}` : i < 5 ? `R2·B${i - 2}` : i < 6 ? "R3·B1" : "free"}
                        </div>
                      ))}
                    </div>
                    <div className="flex gap-3 mt-2 text-[8px]"><span className="text-violet-300/60">■ Req 1</span><span className="text-blue-300/60">■ Req 2</span><span className="text-emerald-300/60">■ Req 3</span><span className="text-white/20">■ Free pool</span></div>
                  </div>
                </div>

                {/* L6: KV Cache size estimator */}
                <div className="mt-3 pl-3">
                  <button onClick={() => setShowKvEstimator((s) => !s)} className="flex items-center gap-1.5 text-[9px] uppercase tracking-widest theme-muted hover:theme-text font-mono font-bold transition mb-2">
                    {showKvEstimator ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                    KV Cache Size Estimator (L6 formula)
                  </button>
                  {showKvEstimator && (
                    <div className="bg-black/20 border border-white/10 rounded-lg p-3 space-y-3">
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                        <Field label="Architecture">
                          <select value={kvPreset} onChange={(e) => { setKvPreset(e.target.value); const p = KV_PRESETS[e.target.value]; if (p) { setKvLayers(p[0]); setKvKvHeads(p[1]); setKvHeadDim(p[2]); setKvDtypeBytes(p[3]); } }} className="w-full px-2 py-1.5 theme-field border rounded text-[10px] font-mono text-white focus:outline-none focus:border-theme-accent">
                            {Object.keys(KV_PRESETS).map((k) => <option key={k} value={k}>{k}</option>)}
                          </select>
                        </Field>
                        <Field label="Layers"><input type="number" value={kvLayers} onChange={(e) => setKvLayers(Number(e.target.value))} className="w-full px-2 py-1.5 theme-field border rounded text-[10px] font-mono text-white focus:outline-none focus:border-theme-accent" /></Field>
                        <Field label="KV Heads"><input type="number" value={kvKvHeads} onChange={(e) => setKvKvHeads(Number(e.target.value))} className="w-full px-2 py-1.5 theme-field border rounded text-[10px] font-mono text-white focus:outline-none focus:border-theme-accent" /></Field>
                        <Field label="Head Dim"><input type="number" value={kvHeadDim} onChange={(e) => setKvHeadDim(Number(e.target.value))} className="w-full px-2 py-1.5 theme-field border rounded text-[10px] font-mono text-white focus:outline-none focus:border-theme-accent" /></Field>
                      </div>
                      <div>
                        <p className="text-[9px] text-violet-300/70 font-mono mb-2">
                          2 × {kvLayers} layers × {kvKvHeads} kv_heads × {kvHeadDim} head_dim × {kvDtypeBytes}B = <b className="text-violet-200">{formatBytes(kvPerToken)}/token</b>
                        </p>
                        <div className="overflow-x-auto">
                          <table className="w-full text-[9px] font-mono">
                            <thead><tr className="border-b border-white/10"><th className="text-left text-white/30 pb-1 pr-4">Context</th><th className="text-left text-white/30 pb-1 pr-4">1 seq</th><th className="text-left text-white/30 pb-1 pr-4">10 concurrent</th><th className="text-left text-white/30 pb-1">50 concurrent</th></tr></thead>
                            <tbody>
                              {KV_CONTEXTS.map((ctx) => (
                                <tr key={ctx} className="border-b border-white/5">
                                  <td className="text-white/60 pr-4 py-1">{ctx.toLocaleString()}t</td>
                                  <td className="text-emerald-300/70 pr-4">{formatBytes(kvPerToken * ctx)}</td>
                                  <td className="text-blue-300/70 pr-4">{formatBytes(kvPerToken * ctx * 10)}</td>
                                  <td className="text-orange-300/70">{formatBytes(kvPerToken * ctx * 50)}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Continuous Batching */}
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <GitMerge className="w-3.5 h-3.5 text-blue-400" />
                <span className="text-[10px] uppercase tracking-widest font-mono font-bold text-blue-400">Continuous Batching</span>
                <span className="text-[8px] px-1.5 py-0.5 rounded bg-blue-500/10 border border-blue-500/30 text-blue-300 font-mono">always on</span>
              </div>
              <div className="pl-1 border-l-2 border-blue-500/30">
                <p className="text-[10px] theme-muted leading-relaxed mb-3 pl-3">
                  vLLM uses <b className="text-white/70">iteration-level scheduling</b> — new requests join the batch the moment existing ones finish tokens, eliminating GPU idle time. The L6 notebook demonstrated this by sending 5 concurrent requests and watching <span className="font-mono text-blue-300">num_requests_running</span> rise to 5 simultaneously.
                </p>
                <div className="grid grid-cols-2 gap-3 pl-3">
                  <Field label="Scheduling Policy">
                    <select value={schedulingPolicy} onChange={(e) => setSchedulingPolicy(e.target.value as "fcfs" | "priority")} className="w-full px-2 py-1.5 theme-field border rounded text-[11px] font-mono text-white focus:outline-none focus:border-theme-accent">
                      <option value="fcfs">fcfs — First-Come First-Served</option>
                      <option value="priority">priority — request priority field</option>
                    </select>
                  </Field>
                  <Field label="Preemption Mode">
                    <select value={preemptionMode} onChange={(e) => setPreemptionMode(e.target.value as "recompute" | "swap")} className="w-full px-2 py-1.5 theme-field border rounded text-[11px] font-mono text-white focus:outline-none focus:border-theme-accent">
                      <option value="recompute">recompute — drop & refill KV blocks (fast)</option>
                      <option value="swap">swap — move KV blocks to CPU (preserves work)</option>
                    </select>
                  </Field>
                </div>
              </div>
            </div>

            {/* Prefix Caching */}
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-2">
                  <Database className="w-3.5 h-3.5 text-emerald-400" />
                  <span className="text-[10px] uppercase tracking-widest font-mono font-bold text-emerald-400">Prefix Caching (APC)</span>
                </div>
                <Toggle enabled={enablePrefixCaching} onChange={setEnablePrefixCaching} label={enablePrefixCaching ? "Enabled" : "Disabled"} />
              </div>
              <div className="pl-1 border-l-2 border-emerald-500/30">
                <p className="text-[10px] theme-muted leading-relaxed mb-3 pl-3">
                  <b className="text-white/70">Automatic Prefix Caching</b> reuses KV blocks for shared prefixes. The L6 notebook proved this by watching <span className="font-mono text-emerald-300">prefix_cache_queries_total</span> grow — confirming vLLM skips prefill for repeated system prompts.
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pl-3">
                  <div className={`p-3 rounded-lg border transition-all ${enablePrefixCaching ? "border-emerald-500/30 bg-emerald-500/5" : "border-white/10 bg-white/[0.02] opacity-50"}`}>
                    <div className="flex items-center gap-2 mb-1.5"><Users className="w-3 h-3 text-emerald-400" /><span className="text-[9px] uppercase tracking-widest font-mono font-bold text-emerald-400">Cross-User Caching</span></div>
                    <p className="text-[10px] text-white/50 leading-relaxed">Same system prompt across 100 users → prefill runs <b className="text-white/70">once</b>. KV blocks shared for all. Critical for any multi-tenant API.</p>
                  </div>
                  <div className={`p-3 rounded-lg border transition-all ${enablePrefixCaching ? "border-emerald-500/30 bg-emerald-500/5" : "border-white/10 bg-white/[0.02] opacity-50"}`}>
                    <div className="flex items-center gap-2 mb-1.5"><MessageSquare className="w-3 h-3 text-emerald-400" /><span className="text-[9px] uppercase tracking-widest font-mono font-bold text-emerald-400">Multi-Turn Caching</span></div>
                    <p className="text-[10px] text-white/50 leading-relaxed">Turn N only computes the <b className="text-white/70">new tokens</b> — all prior turns are cache hits. Receiving tokens from users costs the same on turn 1 and turn 50.</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Live Flags Preview */}
            <div className="rounded-lg border border-white/10 bg-black/30 p-3">
              <div className="flex items-center gap-2 mb-2">
                <Layers className="w-3 h-3 theme-accent" />
                <span className="text-[9px] uppercase tracking-widest font-mono font-bold theme-accent">vLLM Flags Preview</span>
                <span className="text-[8px] theme-faint font-mono">— appended to the managed vLLM serve command at deploy time</span>
              </div>
              {liveFlags ? (
                <code className="text-[10px] font-mono text-emerald-200/80 leading-relaxed break-all">{liveFlags}</code>
              ) : (
                <span className="text-[10px] font-mono theme-faint italic">No extra flags — using vLLM defaults</span>
              )}
              <div className="mt-2 flex flex-wrap gap-1.5">
                <span className="text-[8px] font-mono px-1.5 py-0.5 rounded bg-blue-500/10 border border-blue-500/20 text-blue-300">continuous batching: always on</span>
                <span className="text-[8px] font-mono px-1.5 py-0.5 rounded bg-violet-500/10 border border-violet-500/20 text-violet-300">paged-attn: block={blockSize}</span>
                <span className={`text-[8px] font-mono px-1.5 py-0.5 rounded border ${enablePrefixCaching ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-300" : "bg-white/5 border-white/10 text-white/30"}`}>prefix-cache: {enablePrefixCaching ? "on" : "off"}</span>
                <span className="text-[8px] font-mono px-1.5 py-0.5 rounded bg-white/5 border border-white/10 text-white/40">kv-dtype: {kvCacheDtype}</span>
                <span className="text-[8px] font-mono px-1.5 py-0.5 rounded bg-white/5 border border-white/10 text-white/40">preempt: {preemptionMode}</span>
              </div>
            </div>
          </div>
        )}
      </section>

      {/* ── Deploy Logs ── */}
      {(deployLogs || deploying) && (
        <section className="theme-surface border rounded-lg overflow-hidden">
          <div className="px-4 py-2 border-b theme-surface-soft bg-black/10">
            <p className="text-[10px] uppercase tracking-[0.2em] theme-muted font-mono font-bold">Deployment Logs</p>
          </div>
          <div ref={logBoxRef} className="bg-black/30 p-4 text-[11px] font-mono leading-relaxed text-white/70 h-72 overflow-y-auto whitespace-pre-wrap scrollbar-thin">
            {deployLogs || <span className="theme-faint italic">Awaiting deployment output…</span>}
          </div>
        </section>
      )}

      {/* ── Endpoint Banner ── */}
      {endpoint && (
        <section className="theme-surface border border-emerald-500/30 bg-emerald-950/10 rounded-lg p-4 flex items-center gap-3">
          <span className="relative inline-flex w-3 h-3 shrink-0">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-60" />
            <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-400" />
          </span>
          <div className="min-w-0 flex-1">
            <p className="text-[8px] uppercase tracking-widest text-emerald-400/70 font-mono font-bold mb-0.5">
              Live Endpoint · {SERVING_PROFILES.find((p) => p.key === activeProfile)?.label}
              {detectedQuant && <span className="ml-2 text-violet-400/70">· {detectedQuant.label}</span>}
              {enablePrefixCaching && <span className="ml-2 text-blue-400/70">· prefix-cache on</span>}
            </p>
            <code className="text-[12px] font-mono text-emerald-200 truncate block">{endpoint}</code>
          </div>
          <button onClick={copyUrl} className="flex items-center gap-1.5 px-3 py-1.5 rounded theme-surface border theme-text text-[9px] uppercase tracking-widest font-bold hover:theme-surface-soft transition shrink-0">
            {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
            {copied ? "Copied" : "Copy"}
          </button>
        </section>
      )}

      {/* ── L6: Live vLLM Metrics ── */}
      <section className="theme-surface border rounded-lg overflow-hidden">
        <button onClick={() => setShowMetrics((s) => !s)} className="w-full flex items-center justify-between px-4 py-3 border-b theme-surface-soft bg-black/10 hover:bg-black/20 transition">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 theme-accent" />
            <p className="text-[10px] uppercase tracking-[0.2em] theme-accent font-mono font-bold">Live vLLM Metrics</p>
            <span className="text-[9px] theme-faint font-mono">— /metrics · requests · KV cache % · tokens/s · prefix hits</span>
            {metricsPolling && (
              <span className="flex items-center gap-1 text-[8px] text-emerald-300 font-mono animate-pulse ml-1">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 inline-block" /> LIVE
              </span>
            )}
          </div>
          {showMetrics ? <ChevronDown className="w-3.5 h-3.5 theme-muted" /> : <ChevronRight className="w-3.5 h-3.5 theme-muted" />}
        </button>

        {showMetrics && (
          <div className="p-4 space-y-4">
            <div className="flex items-center gap-3 flex-wrap">
              {!metricsPolling ? (
                <button onClick={startMetricsPolling} disabled={!endpoint}
                  className="flex items-center gap-2 px-4 py-2 rounded theme-accent-bg text-black text-[10px] uppercase tracking-widest font-bold hover:brightness-110 disabled:opacity-50 transition shadow-lg">
                  <Play className="w-3.5 h-3.5" /> Start Polling (2s)
                </button>
              ) : (
                <button onClick={stopMetricsPolling}
                  className="flex items-center gap-2 px-4 py-2 rounded bg-red-950/40 border border-red-500/30 text-red-300 text-[10px] uppercase tracking-widest font-bold hover:bg-red-950 transition">
                  <CircleSlash className="w-3.5 h-3.5" /> Stop Polling
                </button>
              )}
              {!endpoint && <span className="text-[10px] theme-faint font-mono italic">Deploy a model first.</span>}
            </div>
            {metricsError && <div className="text-[10px] text-amber-400 font-mono bg-amber-950/20 border border-amber-500/20 rounded p-2">{metricsError} — ensure vLLM started with <b>--metrics</b> (default)</div>}
            {liveMetrics && (
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                <MetricCard label="Requests Running" value={liveMetrics.requestsRunning.toFixed(0)} icon={<Flame className="w-3.5 h-3.5 text-orange-400" />} accent="text-orange-400" sub={`${liveMetrics.requestsWaiting.toFixed(0)} waiting in queue`} barPct={Math.min(100, (liveMetrics.requestsRunning / Math.max(1, maxNumSeqs)) * 100)} barColor="bg-orange-400" />
                <MetricCard label="GPU KV Cache" value={`${liveMetrics.gpuCacheUsagePerc.toFixed(1)}%`} icon={<MemoryStick className="w-3.5 h-3.5 text-violet-400" />} accent={liveMetrics.gpuCacheUsagePerc > 85 ? "text-red-400" : "text-violet-400"} sub={liveMetrics.cpuCacheUsagePerc > 0 ? `CPU swap: ${liveMetrics.cpuCacheUsagePerc.toFixed(1)}%` : "CPU swap: idle"} barPct={liveMetrics.gpuCacheUsagePerc} barColor={liveMetrics.gpuCacheUsagePerc > 85 ? "bg-red-400" : "bg-violet-400"} />
                <MetricCard label="Generation Speed" value={liveMetrics.tokensPerSec != null ? `${liveMetrics.tokensPerSec.toFixed(1)} tok/s` : "—"} icon={<Zap className="w-3.5 h-3.5 text-emerald-400" />} accent="text-emerald-400" sub={`${liveMetrics.generationTokensTotal.toLocaleString()} total gen tokens`} />
                <MetricCard label="Prompt Tokens" value={liveMetrics.promptTokensTotal.toLocaleString()} icon={<ArrowRight className="w-3.5 h-3.5 text-blue-400" />} accent="text-blue-400" sub="cumulative prefill tokens" />
                <MetricCard label="Prefix Cache Queries" value={liveMetrics.prefixCacheQueriesTotal.toLocaleString()} icon={<Database className="w-3.5 h-3.5 text-emerald-400" />} accent={enablePrefixCaching ? "text-emerald-400" : "text-white/30"} sub={enablePrefixCaching ? "APC active — higher = more cache hits" : "prefix caching disabled"} />
                <MetricCard label="Queue Depth" value={liveMetrics.requestsWaiting.toFixed(0)} icon={<Clock className="w-3.5 h-3.5 text-amber-400" />} accent={liveMetrics.requestsWaiting > 10 ? "text-red-400" : "text-amber-400"} sub={liveMetrics.requestsWaiting > 10 ? "⚠ high queue — increase max-seqs" : "requests queued"} barPct={Math.min(100, (liveMetrics.requestsWaiting / 20) * 100)} barColor={liveMetrics.requestsWaiting > 10 ? "bg-red-400" : "bg-amber-400"} />
              </div>
            )}
            {!liveMetrics && !metricsPolling && !metricsError && (
              <p className="text-[10px] theme-faint font-mono italic">
                Click "Start Polling" to scrape vLLM's Prometheus /metrics every 2s. Tracks: running/waiting requests, GPU KV cache %, tokens/s, prefix_cache_queries (from L6 notebook).
              </p>
            )}
          </div>
        )}
      </section>

      {/* ── Benchmark Panel ── */}
      <section className="theme-surface border rounded-lg overflow-hidden">
        <button onClick={() => setShowBench((s) => !s)} className="w-full flex items-center justify-between px-4 py-3 border-b theme-surface-soft bg-black/10 hover:bg-black/20 transition">
          <div className="flex items-center gap-2">
            <BarChart3 className="w-4 h-4 theme-accent" />
            <p className="text-[10px] uppercase tracking-[0.2em] theme-accent font-mono font-bold">Inference Benchmark</p>
            <span className="text-[9px] theme-faint font-mono">— serial or concurrent · latency · throughput · baseline delta</span>
          </div>
          {showBench ? <ChevronDown className="w-3.5 h-3.5 theme-muted" /> : <ChevronRight className="w-3.5 h-3.5 theme-muted" />}
        </button>

        {showBench && (
          <div className="p-4 space-y-4">
            <div className="flex items-center gap-3 flex-wrap">
              <button onClick={runBenchmark} disabled={!endpoint || benchRunning}
                className="flex items-center gap-2 px-4 py-2 rounded theme-accent-bg text-black text-[10px] uppercase tracking-widest font-bold hover:brightness-110 disabled:opacity-50 transition shadow-lg">
                {benchRunning ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
                {benchRunning ? "Benchmarking…" : `Run Benchmark (${BENCHMARK_PROMPTS.length} prompts)`}
              </button>
              {/* L6: concurrent mode toggle (like L6's ThreadPoolExecutor) */}
              <button onClick={() => setBenchConcurrent((v) => !v)}
                className={`flex items-center gap-1.5 px-3 py-2 rounded border text-[10px] font-mono font-bold transition ${benchConcurrent ? "border-theme-accent/40 bg-theme-accent/10 text-theme-accent" : "border-white/20 theme-surface theme-muted hover:theme-text"}`}
                title="Concurrent: all prompts fire simultaneously. Serial: one at a time.">
                <GitMerge className="w-3.5 h-3.5" />
                {benchConcurrent ? "Concurrent" : "Serial"}
              </button>
              {benchResult && <button onClick={saveAsBaseline} className="flex items-center gap-1.5 px-3 py-2 rounded border border-white/20 theme-surface text-[10px] uppercase tracking-widest font-mono font-bold theme-muted hover:theme-text transition"><BookmarkPlus className="w-3.5 h-3.5" />Save as Baseline</button>}
              {!endpoint && <span className="text-[10px] theme-faint font-mono italic">Deploy a model first.</span>}
            </div>
            <p className="text-[9px] theme-faint font-mono leading-relaxed">
              <b className="text-white/40">Serial:</b> true single-request latency (p50/p95). &nbsp;
              <b className="text-white/40">Concurrent:</b> fires all {BENCHMARK_PROMPTS.length} prompts at once (L6 ThreadPoolExecutor pattern) — throughput = N / wall-clock, reveals continuous batching efficiency.
            </p>
            {benchError && <div className="text-[11px] text-red-400 font-mono bg-red-950/30 border border-red-500/20 rounded-lg p-3">{benchError}</div>}

            {(benchResult || benchBaseline) && (
              <div className="rounded-xl border border-white/10 overflow-hidden">
                <div className="grid grid-cols-4 bg-black/30 border-b border-white/10 text-[8px] uppercase tracking-widest font-mono font-bold">
                  <div className="px-3 py-2 theme-faint">Metric</div>
                  <div className="px-3 py-2 theme-accent">Current Run</div>
                  <div className="px-3 py-2 text-white/40">Baseline</div>
                  <div className="px-3 py-2 text-white/40">Delta</div>
                </div>
                {[
                  { label: "Median Latency", icon: <Clock className="w-3 h-3" />, cur: benchResult?.medianMs, base: benchBaseline?.medianMs, fmt: (v: number) => `${v.toFixed(0)} ms`, unit: "faster", hib: false },
                  { label: "p95 Latency", icon: <TrendingUp className="w-3 h-3" />, cur: benchResult?.p95Ms, base: benchBaseline?.p95Ms, fmt: (v: number) => `${v.toFixed(0)} ms`, unit: "faster", hib: false },
                  { label: "Throughput", icon: <Zap className="w-3 h-3" />, cur: benchResult?.reqPerSec, base: benchBaseline?.reqPerSec, fmt: (v: number) => `${v.toFixed(2)} req/s`, unit: "faster", hib: true },
                ].map(({ label, icon, cur, base, fmt, unit, hib }) => (
                  <div key={label} className="grid grid-cols-4 border-b border-white/5 hover:bg-white/[0.02] transition">
                    <div className="px-3 py-2.5 flex items-center gap-1.5 text-[10px] font-mono theme-muted"><span className="theme-faint">{icon}</span>{label}</div>
                    <div className="px-3 py-2.5 text-[11px] font-mono font-bold theme-accent">{cur != null ? fmt(cur) : <span className="theme-faint">—</span>}</div>
                    <div className="px-3 py-2.5 text-[11px] font-mono text-white/40">{base != null ? fmt(base) : <span className="theme-faint italic text-[10px]">no baseline</span>}</div>
                    <div className="px-3 py-2.5">{cur != null && base != null ? <DeltaBadge current={cur} baseline={base} unit={unit} higherIsBetter={hib} /> : <span className="text-[10px] theme-faint font-mono">—</span>}</div>
                  </div>
                ))}
                <div className="grid grid-cols-4 bg-black/20 text-[9px] font-mono theme-faint">
                  <div className="px-3 py-2">Config</div>
                  <div className="px-3 py-2">{benchResult && <div className="flex flex-col gap-0.5"><span className="text-white/50">{benchResult.profile}</span>{benchResult.quantFormat && <span className="text-violet-400/70">{benchResult.quantFormat}</span>}<span className={benchResult.prefixCaching ? "text-emerald-400/50" : "text-white/20"}>{benchResult.prefixCaching ? "prefix-cache on" : "no prefix cache"}</span><span className="text-white/30">{benchResult.capturedAt} · {benchResult.sampleCount} prompts</span></div>}</div>
                  <div className="px-3 py-2">{benchBaseline && <div className="flex flex-col gap-0.5"><span className="text-white/40">{benchBaseline.profile}</span>{benchBaseline.quantFormat && <span className="text-violet-400/50">{benchBaseline.quantFormat}</span>}<span className="text-white/25">{benchBaseline.capturedAt}</span></div>}</div>
                  <div className="px-3 py-2" />
                </div>
              </div>
            )}

            <details className="group">
              <summary className="cursor-pointer text-[9px] uppercase tracking-widest theme-faint font-mono flex items-center gap-1.5 select-none">
                <ChevronRight className="w-3 h-3 group-open:rotate-90 transition-transform" />
                Prompt bank ({BENCHMARK_PROMPTS.length} prompts)
              </summary>
              <div className="mt-2 space-y-1 pl-4">
                {BENCHMARK_PROMPTS.map((p, i) => <div key={i} className="text-[10px] font-mono text-white/30 truncate">{i + 1}. {p}</div>)}
              </div>
            </details>
          </div>
        )}
      </section>

      {/* ── Chat ── */}
      <section className="theme-surface border rounded-lg overflow-hidden flex flex-col h-[480px]">
        <div className="px-4 py-2 border-b theme-surface-soft bg-black/10 flex items-center justify-between">
          <p className="text-[10px] uppercase tracking-[0.2em] theme-accent font-mono font-bold">Chat</p>
          <div className="flex items-center gap-3">
            {/* L6: thinking mode toggle */}
            <button
              onClick={() => setEnableThinking((v) => !v)}
              className={`flex items-center gap-1.5 text-[9px] font-mono font-bold transition-colors ${enableThinking ? "text-violet-400" : "theme-faint hover:theme-muted"}`}
              title="Thinking mode: model generates chain-of-thought before answering (uses more tokens)"
            >
              <Brain className={`w-3.5 h-3.5 ${enableThinking ? "text-violet-400" : "theme-faint"}`} />
              {enableThinking ? "Thinking On" : "Thinking Off"}
            </button>
            {chatMessages.length > 0 && (
              <button onClick={() => { setChatMessages([]); setChatError(null); setExpandedLogprobIdx(null); }}
                className="flex items-center gap-1 text-[9px] uppercase tracking-widest theme-faint hover:theme-text font-mono transition">
                <RefreshCw className="w-3 h-3" /> Clear
              </button>
            )}
          </div>
        </div>
        <div ref={chatBoxRef} className="flex-1 min-h-0 overflow-y-auto p-4 space-y-3 scrollbar-thin bg-black/10">
          {chatMessages.length === 0 && !chatSending ? (
            <div className="h-full flex items-center justify-center text-center theme-faint italic font-serif text-sm-fluid px-6">
              {endpoint ? "Send a message to chat with the deployed model." : "Deploy a model to start chatting."}
            </div>
          ) : (
            chatMessages.map((m, i) => (
              <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                <div className={`max-w-[82%] rounded-lg px-3 py-2 text-[12px] leading-relaxed whitespace-pre-wrap break-words ${m.role === "user" ? "theme-accent-soft theme-accent border" : "bg-black/40 border border-white/10 text-white/85"}`}>
                  <div className={`text-[7px] uppercase tracking-widest font-mono font-bold mb-1 opacity-60 ${m.role === "user" ? "theme-accent" : "text-white/50"}`}>{m.role === "user" ? "You" : "Model"}</div>
                  {m.content}
                  {/* L6: usage metadata */}
                  {m.usageMs !== undefined && (
                    <div className="mt-1.5 flex gap-2 flex-wrap">
                      <span className="text-[8px] font-mono text-white/25">{m.usageMs}ms</span>
                      {m.completionTokens && <span className="text-[8px] font-mono text-white/25">{m.completionTokens} tokens</span>}
                      {m.completionTokens && m.usageMs && <span className="text-[8px] font-mono text-white/25">{((m.completionTokens / (m.usageMs / 1000))).toFixed(1)} tok/s</span>}
                    </div>
                  )}
                  {/* L6: logprobs panel */}
                  {m.logprobs && m.logprobs.length > 0 && (
                    <button onClick={() => setExpandedLogprobIdx(expandedLogprobIdx === i ? null : i)}
                      className="mt-1.5 text-[8px] font-mono text-white/30 hover:text-white/50 transition flex items-center gap-1">
                      <ChevronRight className={`w-3 h-3 transition-transform ${expandedLogprobIdx === i ? "rotate-90" : ""}`} />
                      token probabilities
                    </button>
                  )}
                  {expandedLogprobIdx === i && m.logprobs && (
                    <div className="mt-2 bg-black/30 rounded p-2 space-y-2 max-h-40 overflow-y-auto">
                      {m.logprobs.slice(0, 6).map((tok, ti) => (
                        <div key={ti} className="space-y-0.5">
                          <div className="text-[8px] font-mono text-white/50">Chosen: <span className="text-theme-accent font-bold">'{tok.token}'</span></div>
                          {tok.topLogprobs.slice(0, 4).map((alt, ai) => (
                            <div key={ai} className="flex items-center gap-2">
                              <span className="text-[8px] font-mono text-white/30 w-20 truncate">'{alt.token}'</span>
                              <LogprobBar prob={alt.prob} />
                            </div>
                          ))}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
          {chatSending && (
            <div className="flex justify-start">
              <div className="rounded-lg px-3 py-2 bg-black/40 border border-white/10 text-white/60 text-[11px] font-mono flex items-center gap-2">
                <Loader2 className="w-3 h-3 animate-spin" />
                {enableThinking ? "Thinking…" : "Generating…"}
              </div>
            </div>
          )}
        </div>
        {chatError && <div className="px-4 py-2 text-[10px] text-red-400 font-mono bg-red-950/20 border-t border-red-500/20">{chatError}</div>}
        <div className="border-t theme-surface-soft p-3 bg-black/20">
          {enableThinking && (
            <div className="flex items-center gap-1.5 mb-2 px-1 text-[9px] font-mono text-violet-400/70">
              <Brain className="w-3 h-3" />
              Thinking mode on — model generates chain-of-thought reasoning before answering (uses more tokens + KV cache)
            </div>
          )}
          <div className="flex items-end gap-2">
            <textarea value={chatInput} onChange={(e) => setChatInput(e.target.value)} onKeyDown={onChatKeyDown} rows={2} disabled={!endpoint || chatSending}
              placeholder={endpoint ? "Type a message… (Enter to send, Shift+Enter for newline)" : "Deploy a model first…"}
              className="flex-1 min-w-0 px-3 py-2 theme-field border rounded-lg text-[12px] font-mono text-white/85 resize-none focus:outline-none focus:border-theme-accent transition leading-relaxed shadow-inner disabled:opacity-50" />
            <button onClick={sendChat} disabled={!endpoint || chatSending || !chatInput.trim()}
              className="flex items-center gap-2 px-4 py-2.5 rounded theme-accent-bg text-black text-[10px] uppercase tracking-widest font-bold hover:brightness-110 disabled:opacity-50 transition shadow-lg shrink-0">
              {chatSending ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Send className="w-3.5 h-3.5" />}
              Send
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}
