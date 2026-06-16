import React, { useEffect, useMemo, useRef, useState } from "react";
import type {
  AppConfig,
  Chunk,
  EmbedderConfig,
  HfDatasetRepo,
  HfModelRepo,
  GPUState,
  HubConfig,
  HubDatasetConfig,
  IngestStream,
  LoraConfig,
  PaddleOcrConfig,
  Run,
  RunConfig,
  TeacherConfig,
  TopicTarget,
} from "../types";
import {
  DEFAULT_EMBEDDER,
  DEFAULT_HUB,
  DEFAULT_HUB_DATASET,
  DEFAULT_LORA,
  DEFAULT_PADDLE_OCR,
  DEFAULT_TEACHER,
} from "../types";
import { api, events } from "../lib/tauri";
import { clearSetupLogs, getSetupLogSnapshot, subscribeSetupLogs } from "../lib/setupLogs";
import { stripModelThinking } from "../lib/textSanitize";
import TrainingConfigForm, { type StudentModelOption } from "./TrainingConfigForm";
import {
  CheckCircle2,
  ChevronRight,
  ChevronLeft,
  Database,
  Cpu,
  Filter,
  FileText,
  Sparkles,
  Loader2,
  Trash2,
  Play,
  RefreshCw,
  Layers,
  Plus,
  X,
  Upload,
  StopCircle,
  ShieldAlert,
  Zap,
  Server,
  Circle,
  Download,
  Save,
  Trash,
  HardDrive,
  FolderOpen,
  Wifi,
  WifiOff,
  ScanText,
} from "lucide-react";
import { open as openFileDialog, save as saveFileDialog } from "@tauri-apps/plugin-dialog";
import type { IngestDoneEvent, IngestProgressEvent } from "../types";

interface Props {
  config: AppConfig;
  gpuStatus?: GPUState | null;
  onConfigChange: (patch: Partial<AppConfig>) => void;
  onPipelineLaunched: (runId: string) => void;
  onStepChange?: (step: number) => void;
}

const STEPS = [
  { key: "kb", label: "Knowledge Base", icon: Database },
  { key: "teacher", label: "Teacher", icon: Cpu },
  { key: "dataset", label: "Dataset", icon: FileText },
  { key: "train", label: "Student & Train", icon: Layers },
] as const;

type PipelineMode = "rag" | "trainingOnly";

const DEFAULT_PROMPT = `FOCUS TOPIC: {topic}

ROLE: You are an expert tutor helper designed to explain complex topics and concepts to any student in a simple, easy-to-understand way.

TASK:
I will provide you with source material (treat it as your open notes / RAG database).
1. Identify the core concepts and facts inside the source material.
2. Write a NEW, original question based strictly on those concepts.
- The new question must be on the focus topic '{topic}'. If the source material has no meaningful connection to the focus topic, respond with exactly:
  SKIP: off-topic
- Do NOT copy the source material verbatim. Rephrase, change numbers if mathematical, or shift the angle (e.g. solve for a different variable).
- The question must be answerable strictly using facts from the source material.
- Do NOT repeat a question, fact pattern, or legal-provision angle that has already been used in this generation run.

3. Provide the final ANSWER along with a concise explanation of WHY it is correct.

Format your response EXACTLY like this, with no extra commentary before or after:

QUESTION: <the new question>

ANSWER: <the final answer, followed by a simplified explanation of why it is the correct answer>

Source material:
"""
{chunk_text}
"""`;

type DatasetFormatKey =
  | "simple_qa"
  | "reasoning_qa"
  | "multiple_choice"
  | "chain_of_thought"
  | "instruction_io"
  | "conversational";

const DEFAULT_DATASET_FORMAT: DatasetFormatKey = "multiple_choice";

const DATASET_FORMATS: Array<{
  key: DatasetFormatKey;
  label: string;
  desc: string;
  icon: React.ComponentType<{ className?: string }>;
  prompt: string;
}> = [
  {
    key: "simple_qa",
    label: "Simple Q&A",
    desc: "Direct question-answer pairs for FAQs and quick instruction tuning.",
    icon: FileText,
    prompt: `FOCUS TOPIC: {topic}

ROLE: You are a knowledgeable tutor creating clear, direct study questions.

TASK:
Using the source material below, write ONE original question and a concise, factually-grounded answer.

RULES:
- Question must be answerable strictly from the source.
- Do NOT copy the source verbatim. Rephrase or shift the angle.
- Answer should be 2-5 sentences, direct and complete.
- If the source is unrelated to '{topic}', respond EXACTLY: SKIP: off-topic

OUTPUT FORMAT (strict):
QUESTION:
ANSWER:

Source material:
"""
{chunk_text}
"""`
  },
  {
    key: "reasoning_qa",
    label: "Q&A with Reasoning",
    desc: "Emits <think> reasoning blocks for DeepSeek/Qwen-style SFT.",
    icon: Sparkles,
    prompt: `FOCUS TOPIC: {topic}

ROLE: You are an expert tutor creating reasoning-based training data for advanced LLMs.

TASK:
Using the source material below, write ONE original question, a reasoning chain, and a final answer.

RULES:
- Question must be answerable strictly from the source.
- REASONING should identify facts, connect them, and derive the answer.
- ANSWER should be the final, clean response.
- Reasoning should be 3-7 sentences using words like "because", "therefore", "since", or "this means".
- If source is unrelated to '{topic}', respond EXACTLY: SKIP: off-topic

OUTPUT FORMAT (strict):
QUESTION:
REASONING:
ANSWER:

Source material:
"""
{chunk_text}
"""`
  },
  {
    key: "multiple_choice",
    label: "Multiple Choice",
    desc: "Four-option MCQ with distractors, reasoning, and answer summary.",
    icon: Layers,
    prompt: `FOCUS TOPIC: {topic}

ROLE: You are an expert dataset curator and question writer. Your job is to generate high-quality training question-answer pairs with detailed reasoning from source material for any domain or subject.

TASK:
Given the source material below, generate ONE high-quality question-answer pair with step-by-step reasoning.

RULES:
1. Detect the domain from the source material and adapt your question style accordingly:
   - Quantitative/math-heavy material -> computation or problem-solving question (change given values slightly)
   - Legal/regulatory material -> scenario-based application question, NOT a definition question
   - Conceptual/theory material -> "which is MOST accurate / appropriate" question that tests understanding, not recall
   - Procedural material -> step-ordering or error-identification question
2. Provide exactly 4 choices (A-D) with plausible distractors based on common mistakes or misconceptions.
3. REASONING must be detailed:
   - Break the problem down step by step
   - Eliminate wrong choices explicitly and explain why each is wrong
   - Show the derivation, formula application, or rule being used
   - Conclude with why the correct answer is correct
4. ANSWER must state the correct letter and a concise summary of the reasoning (2-3 sentences max).
5. Do NOT copy the source verbatim. Rephrase, vary values, or shift the angle.
6. The question must be answerable strictly from the source material.
7. If the source material has no meaningful connection to '{topic}', respond with exactly: SKIP: off-topic

FORMAT (strictly - no extra text before or after):
QUESTION: <stem>
A. <choice>
B. <choice>
C. <choice>
D. <choice>
REASONING: <detailed step-by-step reasoning, distractor elimination, formula/rule application>
ANSWER: <correct letter> - <concise 2-3 sentence summary of why it is correct>

Source material:
"""
{chunk_text}
"""`
  },
  {
    key: "chain_of_thought",
    label: "Chain-of-Thought",
    desc: "Numbered step-by-step solutions for math, logic, and procedures.",
    icon: Zap,
    prompt: `FOCUS TOPIC: {topic}

ROLE: You are a methodical teacher creating step-by-step solution training data.

TASK:
Using the source material, write ONE problem and a numbered step-by-step solution leading to a final answer. Ideal for math, logic, procedures, or multi-step derivations.

RULES:
- Problem should require at least 3 reasoning steps.
- Each step must be explicit, atomic, and explained.
- End with a clearly labeled FINAL ANSWER.
- Use formulas, equations, or rule citations where applicable.
- If unrelated to '{topic}', respond EXACTLY: SKIP: off-topic

OUTPUT FORMAT (strict):
PROBLEM:
SOLUTION:
Step 1:
Step 2:
Step 3:
[continue as needed]
FINAL ANSWER:

Source material:
"""
{chunk_text}
"""`
  },
  {
    key: "instruction_io",
    label: "Alpaca Instruction",
    desc: "Instruction, input, and output triples for general SFT.",
    icon: Database,
    prompt: `FOCUS TOPIC: {topic}

ROLE: You are creating Alpaca-style instruction tuning data.

TASK:
Using the source material, generate ONE instruction-input-output triple.

RULES:
- INSTRUCTION: a clear task directive (for example, "Summarize the following", "Explain why...", "Calculate...").
- INPUT: relevant context/data the instruction operates on. Use "N/A" if the instruction is self-contained.
- OUTPUT: the complete, accurate response derived from the source.
- Vary instruction types: summarize, explain, classify, extract, compare, calculate.
- If unrelated to '{topic}', respond EXACTLY: SKIP: off-topic

OUTPUT FORMAT (strict):
INSTRUCTION:
INPUT:
OUTPUT:

Source material:
"""
{chunk_text}
"""`
  },
  {
    key: "conversational",
    label: "Multi-Turn Conversation",
    desc: "Source-grounded tutoring dialogue for chat model training.",
    icon: ScanText,
    prompt: `FOCUS TOPIC: {topic}

ROLE: You are scripting a realistic tutor-student dialogue for training a conversational AI.

TASK:
Using the source material, write a 3-4 turn dialogue between a curious USER and an expert ASSISTANT. The conversation should naturally explore a concept from the source.

RULES:
- Turn 1: USER asks a beginner-level question.
- Turn 2: ASSISTANT explains clearly using source facts.
- Turn 3: USER asks a follow-up (clarification, edge case, or deeper question).
- Turn 4: ASSISTANT gives a precise, source-grounded answer.
- Optional Turn 5-6: deeper exchange.
- Keep tone natural and helpful. No reasoning leakage.
- If unrelated to '{topic}', respond EXACTLY: SKIP: off-topic

OUTPUT FORMAT (strict):
USER:
ASSISTANT:
USER:
ASSISTANT:

Source material:
"""
{chunk_text}
"""`
  },
];

function promptForDatasetFormat(format: string): string {
  return DATASET_FORMATS.find((f) => f.key === format)?.prompt || DATASET_FORMATS.find((f) => f.key === DEFAULT_DATASET_FORMAT)?.prompt || DEFAULT_PROMPT;
}

const SELECT_CLASS =
  "w-full px-4 py-2.5 premium-input rounded-xl text-sm-fluid font-black font-mono focus:outline-none appearance-none cursor-pointer [color-scheme:dark]";
const OPTION_CLASS = "theme-surface theme-text";

const PRESET_TEACHER_MODELS = [
  { value: "Qwen/Qwen3.6-35B-A3B", label: "Qwen 3.6 35B A3B (Recommended MoE)" },
  { value: "Qwen/Qwen2.5-72B-Instruct", label: "Qwen 2.5 72B Instruct (High Parameter)" },
  { value: "Qwen/Qwen2.5-32B-Instruct", label: "Qwen 2.5 32B Instruct" },
  { value: "Qwen/Qwen2.5-14B-Instruct", label: "Qwen 2.5 14B Instruct" },
  { value: "Qwen/Qwen2.5-7B-Instruct", label: "Qwen 2.5 7B Instruct (Lightweight)" },
  { value: "Qwen/Qwen2.5-Coder-32B-Instruct", label: "Qwen 2.5 Coder 32B Instruct" },
  { value: "Qwen/Qwen2.5-Coder-7B-Instruct", label: "Qwen 2.5 Coder 7B Instruct" },
  { value: "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", label: "DeepSeek R1 Distill Llama 70B" },
];

function autoTuneTeacherConfig(base: TeacherConfig, gpuStatus?: GPUState | null): TeacherConfig {
  if (!base.autoTune || (base.customServeCmd || "").trim()) return base;

  const repo = (base.repoId || "").toLowerCase();
  const memoryGb = (gpuStatus?.memoryTotal || 0) / 1024;
  const isQwen3 = repo.includes("qwen3");
  const isVision = repo.includes("-vl") || repo.includes("vision");
  const isGguf = repo.includes("gguf");

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
    tensorParallel: Math.max(base.tensorParallel || 1, 1),
    gpuMemoryUtilization: 0.80,
    enableChunkedPrefill: true,
    maxNumBatchedTokens,
    maxNumSeqs,
    enableAutoToolChoice: isQwen3,
    toolCallParser: isQwen3 ? "qwen3_coder" : "",
    servingEngine: "vllm",
  };
}

function teacherConfigEquals(a: TeacherConfig, b: TeacherConfig): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

/** Multi-select HF dataset picker used in Training-Only mode. */
function MultiDatasetPicker(props: {
  selected: string[];
  onChange: (next: string[]) => void;
  hfDatasets: HfDatasetRepo[];
  hfUsername: string | null;
  hfTokenSet: boolean;
  hfLoading: boolean;
  hfError: string | null;
  onRefreshHf: () => void;
}) {
  const [draft, setDraft] = useState("");
  const [pickerOpen, setPickerOpen] = useState(false);
  const [search, setSearch] = useState("");
  const pickerRef = useRef<HTMLDivElement | null>(null);
  const selectedSet = useMemo(() => new Set(props.selected), [props.selected]);

  useEffect(() => {
    if (!pickerOpen) {
      setSearch("");
    }
  }, [pickerOpen]);

  useEffect(() => {
    if (!pickerOpen) return;
    const onDocClick = (e: MouseEvent) => {
      if (pickerRef.current && !pickerRef.current.contains(e.target as Node)) {
        setPickerOpen(false);
      }
    };
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [pickerOpen]);

  const add = (repo: string) => {
    const t = (repo || "").trim();
    if (!t) return;
    if (selectedSet.has(t)) return;
    props.onChange([...props.selected, t]);
  };
  const remove = (repo: string) => {
    props.onChange(props.selected.filter((r) => r !== repo));
  };
  const submitDraft = () => {
    if (!draft.trim()) return;
    add(draft);
    setDraft("");
  };

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return props.hfDatasets;
    return props.hfDatasets.filter((d) =>
      (d.id || "").toLowerCase().includes(q)
    );
  }, [props.hfDatasets, search]);

  return (
    <div className="col-span-2 space-y-4">
      <div className="flex items-center justify-between ml-1">
        <label className="text-[10px] uppercase tracking-widest theme-muted font-black">
          Training Datasets <span className="opacity-40 font-mono tracking-normal">({props.selected.length})</span>
        </label>
        <button
          type="button"
          onClick={props.onRefreshHf}
          disabled={!props.hfTokenSet || props.hfLoading}
          className="flex items-center gap-2 text-[9px] font-black uppercase tracking-widest theme-faint hover:theme-text transition-all group"
        >
          <RefreshCw className={`w-3.5 h-3.5 group-hover:rotate-180 transition-transform duration-500 ${props.hfLoading ? "animate-spin" : ""}`} />
          Refresh Cloud List
        </button>
      </div>

      {/* Selected chips */}
      {props.selected.length > 0 ? (
        <div className="flex flex-wrap gap-2 p-3 rounded-2xl border border-white/5 bg-black/20 shadow-inner">
          {props.selected.map((repo, idx) => (
            <span
              key={repo}
              className="inline-flex items-center gap-2 px-3 py-1.5 rounded-xl theme-accent-soft border border-theme-accent/30 text-[10px] font-black font-mono theme-accent shadow-sm animate-premium"
            >
              <span className="opacity-40">#{idx + 1}</span>
              <code className="text-white/90 lowercase tracking-tighter">{repo}</code>
              <button
                type="button"
                onClick={() => remove(repo)}
                className="ml-1 w-4 h-4 rounded-full bg-white/5 flex items-center justify-center hover:bg-red-500 hover:text-white transition-colors"
              >
                <X className="w-2.5 h-2.5" />
              </button>
            </span>
          ))}
        </div>
      ) : (
        <div className="p-4 rounded-2xl border border-dashed border-white/10 theme-faint text-[10px] font-black font-mono text-center uppercase tracking-widest">
          Awaiting Dataset Selection
        </div>
      )}

      {/* Picker: account datasets */}
      {props.hfDatasets.length > 0 && (
        <div className="relative" ref={pickerRef}>
          <button
            type="button"
            onClick={() => setPickerOpen((o) => !o)}
            className={`${SELECT_CLASS} flex items-center justify-between text-left`}
          >
            <span className="theme-text">— ATTACH ACCOUNT REPOSITORY —</span>
            <ChevronRight className={`w-4 h-4 opacity-40 transition-transform ${pickerOpen ? "rotate-[270deg]" : "rotate-90"}`} />
          </button>
          {pickerOpen && (
            <div className="absolute z-50 mt-2 w-full rounded-xl theme-surface border border-white/10 shadow-2xl animate-premium flex flex-col overflow-hidden">
              <div className="p-2 border-b border-white/5 bg-black/40 flex items-center gap-2">
                <svg className="w-3.5 h-3.5 opacity-40 shrink-0 ml-1" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5"><circle cx="11" cy="11" r="8"/><path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-4.35-4.35"/></svg>
                <input
                  type="text"
                  placeholder="Search repository..."
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter") e.preventDefault(); }}
                  className="w-full px-2 py-1.5 bg-transparent border-0 text-xs font-mono text-white placeholder-white/30 focus:outline-none"
                  autoFocus
                />
                {search && (
                  <button
                    type="button"
                    onClick={() => setSearch("")}
                    className="p-1 rounded-full hover:bg-white/5 text-white/50 hover:text-white transition-colors"
                  >
                    <X className="w-3 h-3" />
                  </button>
                )}
              </div>
              <div className="overflow-y-auto flex-1 max-h-[220px] divide-y divide-white/5 scrollbar-thin scrollbar-thumb-white/10">
                {filtered.length > 0 ? (
                  filtered.map((d) => {
                    const checked = selectedSet.has(d.id);
                    return (
                      <button
                        type="button"
                        key={d.id}
                        onClick={() => {
                          if (checked) remove(d.id);
                          else add(d.id);
                        }}
                        className="w-full flex items-center gap-3 px-4 py-2.5 text-left text-sm-fluid font-mono hover:bg-white/5 transition-colors"
                      >
                        <div className={`w-5 h-5 rounded border-2 flex items-center justify-center transition-all ${checked ? "bg-theme-accent border-theme-accent" : "border-white/20"}`}>
                          {checked && <CheckCircle2 className="w-4 h-4 text-black" />}
                        </div>
                        <span className="flex-1 truncate text-white/90 lowercase tracking-tighter">{d.id}</span>
                        {d.private && (
                          <span className="text-[9px] font-black font-mono theme-accent uppercase tracking-widest opacity-60">[PRIVATE]</span>
                        )}
                      </button>
                    );
                  })
                ) : (
                  <div className="px-4 py-6 text-xs theme-faint font-mono text-center uppercase tracking-widest opacity-60 select-none">
                    No repositories found
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Picker: free-form */}
      <div className="flex gap-3 animate-premium">
        <input
          type="text"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); submitDraft(); } }}
          placeholder="Or type manual identifier..."
          className="flex-1 px-4 py-3 premium-input rounded-xl text-sm-fluid font-mono text-white focus:outline-none shadow-inner"
        />
        <button
          type="button"
          onClick={submitDraft}
          disabled={!draft.trim()}
          className="px-6 py-3 rounded-xl text-[10px] uppercase tracking-[0.2em] font-black font-mono theme-accent-bg text-black hover:brightness-110 disabled:opacity-20 shadow-lg shadow-theme-accent/10 transition-all premium-button"
        >
          Add Repo
        </button>
      </div>

      {props.hfError && (
        <p className="text-[10px] text-red-400 font-black font-mono uppercase tracking-tighter ml-1">✕ {props.hfError}</p>
      )}
      <p className="text-[10px] theme-faint font-medium leading-relaxed ml-1 opacity-70 italic">
        Datasets will be interleaved during training. Ensure shared schema compatibility.
      </p>
    </div>
  );
}

// ── step 1 ────────────────────────────────────────────────────────────────
function GpuServerStatusCard({ gpuStatus, loading }: { gpuStatus: GPUState | null; loading: boolean }) {
  return (
    <div className="flex items-center gap-4">
      <div className="flex-1 premium-card rounded-2xl p-5 glass-panel relative overflow-hidden group shadow-lg border border-white/5">
        <div className="absolute top-0 left-0 w-1 h-full theme-accent-bg opacity-30" />
        <div className="flex items-center justify-between mb-3">
          <p className="text-[10px] uppercase tracking-widest theme-accent font-black font-mono flex items-center gap-2">
            <Server className="w-3.5 h-3.5" /> GPU Server
          </p>
          <div className={`w-2 h-2 rounded-full ${loading ? "animate-pulse bg-amber-400" : (gpuStatus?.success ? "bg-emerald-400 shadow-[0_0_6px_#4ade80]" : "bg-red-500")}`} />
        </div>
        {loading ? (
          <p className="text-sm font-mono theme-muted italic animate-pulse">Probing hardware telemetry...</p>
        ) : gpuStatus?.success ? (
          <div className="space-y-2">
            <p className="text-sm font-black font-mono text-white">{gpuStatus.gpuName}</p>
            <div className="flex items-center gap-4 text-[9px] font-mono theme-muted opacity-70">
              <span>VRAM {Math.round(gpuStatus.memoryUsed/1024)}/{Math.round(gpuStatus.memoryTotal/1024)} GB</span>
              <span>CUDA {gpuStatus.cudaVersion}</span>
              <span>{gpuStatus.utilizationGpu}% GPU</span>
            </div>
            {gpuStatus.systemInfo && (
              <p className="text-[8px] font-mono theme-muted opacity-50 truncate max-w-full" title={gpuStatus.systemInfo}>
                {gpuStatus.systemInfo.split('\n').slice(0, 1).join(' ')}
              </p>
            )}
          </div>
        ) : (
          <p className="text-sm font-mono theme-muted opacity-50">Not connected — configure SSH in Credentials</p>
        )}
      </div>
      <div className="flex flex-col items-center justify-center gap-2 p-4 rounded-2xl glass-panel border border-white/5">
        <div className={`p-2 rounded-lg ${gpuStatus?.success ? "bg-emerald-500/10 text-emerald-400" : "bg-white/5 text-theme-muted"}`}>
          <Database className="w-5 h-5" />
        </div>
        <p className="text-[8px] uppercase tracking-widest font-black font-mono theme-muted opacity-50">Qdrant</p>
      </div>
    </div>
  );
}

function EmbedderCard({
  embedder,
  idx,
  config,
  onUpdate,
  onRemove,
  onIngestComplete,
  showRemove,
  gpuStatus,
}: {
  embedder: EmbedderConfig;
  idx: number;
  config: AppConfig;
  onUpdate: (patch: Partial<EmbedderConfig>) => void;
  onRemove: () => void;
  onIngestComplete: () => void;
  showRemove?: boolean;
  gpuStatus?: GPUState | null;
}) {
  const [files, setFiles] = useState<string[]>([]);
  const [tag, setTag] = useState<string>("");
  const [streams, setStreams] = useState<IngestStream[]>([]);
  const [pointCount, setPointCount] = useState<number | null>(null);
  const [loadingPoints, setLoadingPoints] = useState(false);
  const [ingestLogs, setIngestLogs] = useState<string[]>([]);
  const [scanningFolder, setScanningFolder] = useState(false);
  const logContainerRef = useRef<HTMLDivElement | null>(null);
  const [currentStage, setCurrentStage] = useState<{ file: string; stage: string; done: number; total: number } | null>(null);
  const activeStreamIdsRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [ingestLogs]);
  const collection = embedder.collection || `kb_${embedder.name.toLowerCase().replace(/\s+/g, "_").replace(/[^a-z0-9_]/g, "")}`;
  const qdrantEndpoint = config.qdrant.endpoint || (config.ssh.host ? `http://${config.ssh.host}:6333` : "");
  const embeddingConfig = { provider: "vllm" as const, apiUrl: `http://${config.ssh.host}:${embedder.port}`, apiKey: "", modelId: embedder.modelId, concurrency: embedder.concurrency };
  const qdrantCfg = { ...config.qdrant, endpoint: qdrantEndpoint, collection };
  const ready = !!qdrantEndpoint && !!config.ssh.host;
  const ingesting = streams.some(s => !s.done);
  const activeStream = streams.find(s => !s.done);
  const currentFile = activeStream?.progress?.file || "";
  const aggProgress = streams.filter(s => !s.done).reduce((acc, s) => {
    if (s.progress) { acc.done += s.progress.done; acc.total += s.progress.total; }
    return acc;
  }, { done: 0, total: 0 });
  const pct = aggProgress.total > 0 ? Math.min(100, Math.round((aggProgress.done / aggProgress.total) * 100)) : 0;

  const setStreamsAndPersist = (newStreams: IngestStream[]) => {
    setStreams(newStreams);
    // Merge with other embedders' persisted state so we don't overwrite them
    api.loadIngestState().then(json => {
      try {
        const existing = json && json !== "{}" ? JSON.parse(json) : {};
        existing[collection] = newStreams;
        api.saveIngestState(JSON.stringify(existing)).catch(() => {});
      } catch {
        api.saveIngestState(JSON.stringify({ [collection]: newStreams })).catch(() => {});
      }
    }).catch(() => {
      api.saveIngestState(JSON.stringify({ [collection]: newStreams })).catch(() => {});
    });
  };

  useEffect(() => {
    api.loadIngestState().then(json => {
      if (!json || json === "{}") return;
      try {
        const parsed = JSON.parse(json);
        const saved = parsed[collection];
        if (Array.isArray(saved) && saved.length > 0) {
          setStreams(saved);
          for (const s of saved) {
            activeStreamIdsRef.current.add(s.id);
          }
        }
      } catch {}
    }).catch(() => {});
  }, [collection]);

  useEffect(() => {
    let active = true;
    let cleanupProgress: (() => void) | null = null;
    let cleanupDone: (() => void) | null = null;

    (async () => {
      const unlistenProgress = await events.onIngestProgress((e: IngestProgressEvent) => {
        if (!active) return;
        if (!activeStreamIdsRef.current.has(e.streamId)) return;

        setCurrentStage({ file: e.file, stage: e.stage, done: e.done, total: e.total });
        setStreams(prev => {
          const updated = prev.map(s => s.id === e.streamId ? { ...s, progress: { file: e.file, done: e.done, total: e.total } } : s);
          // Persist progress updates (debounced by only saving every 5th update)
          if (e.done % 5 === 0 || e.done === e.total) {
            api.loadIngestState().then(json => {
              try {
                const existing = json && json !== "{}" ? JSON.parse(json) : {};
                existing[collection] = updated;
                api.saveIngestState(JSON.stringify(existing)).catch(() => {});
              } catch {}
            }).catch(() => {});
          }
          return updated;
        });

        const timestamp = new Date().toLocaleTimeString();
        let logLine = "";
        if (e.stage === "read") {
          logLine = `[${timestamp}] [OCR/Read] Reading ${e.file}...`;
        } else if (e.stage === "ocr_start") {
          const gpuName = gpuStatus?.success && gpuStatus.gpuName ? gpuStatus.gpuName : "GPU";
          logLine = `[${timestamp}] [PaddleOCR] Starting remote OCR on ${gpuName} for ${e.file}...`;
        } else if (e.stage === "ocr_page") {
          logLine = `[${timestamp}] [PaddleOCR] Processed page ${e.done}/${e.total} of ${e.file}`;
        } else if (e.stage === "embed") {
          logLine = `[${timestamp}] [Embedder] Embedding chunk ${e.done}/${e.total} of ${e.file}`;
        } else if (e.stage === "upsert") {
          logLine = `[${timestamp}] [Qdrant] Upserting batch ${e.done}/${e.total} of ${e.file} to Qdrant`;
        } else if (e.stage === "done") {
          logLine = `[${timestamp}] [Done] Successfully ingested ${e.file} (${e.total} chunks)`;
        } else if (e.stage === "error") {
          logLine = `[${timestamp}] [Error] Failed to ingest ${e.file}`;
        } else if (e.stage === "warn") {
          logLine = `[${timestamp}] [Warning] ${e.file}`;
        }
        if (logLine) {
          setIngestLogs(prev => [...prev.slice(-99), logLine]);
        }
      });

      if (!active) {
        unlistenProgress();
      } else {
        cleanupProgress = unlistenProgress;
      }

      const unlistenDone = await events.onIngestDone((e: IngestDoneEvent) => {
        if (!active) return;
        if (!activeStreamIdsRef.current.has(e.streamId)) return;

        setCurrentStage(null);
        setStreams(prev => {
          const updated = prev.map(s => {
            if (s.id !== e.streamId) return s;
            if (e.success && e.summary) {
              onIngestComplete();
              const errs = e.summary.files.filter(f => f.error).map(f => ({ file: f.file_name, error: f.error || "" }));
              return { ...s, done: true, progress: null, chunks: e.summary.total_chunks, errors: errs, cancelled: e.summary.cancelled };
            }
            return { ...s, done: true, progress: null, error: e.error || "ingest failed" };
          });
          // Persist final state
          api.loadIngestState().then(json => {
            try {
              const existing = json && json !== "{}" ? JSON.parse(json) : {};
              existing[collection] = updated;
              api.saveIngestState(JSON.stringify(existing)).catch(() => {});
            } catch {}
          }).catch(() => {});
          return updated;
        });

        const timestamp = new Date().toLocaleTimeString();
        if (e.success && e.summary) {
          const fileErrors = e.summary.files.filter(f => f.error);
          if (fileErrors.length > 0) {
            setIngestLogs(prev => [
              ...prev,
              `[${timestamp}] [Done] Ingestion completed with errors.`,
              ...fileErrors.map(f => `  ✕ ${f.file_name}: ${f.error || "unknown error"}`)
            ]);
          } else {
            setIngestLogs(prev => [...prev, `[${timestamp}] [Done] Ingestion completed successfully!`]);
          }
        } else {
          setIngestLogs(prev => [...prev, `[${timestamp}] [Error] Ingestion failed: ${e.error || "unknown error"}`]);
        }
      });

      if (!active) {
        unlistenDone();
      } else {
        cleanupDone = unlistenDone;
      }
    })();

    return () => {
      active = false;
      if (cleanupProgress) cleanupProgress();
      if (cleanupDone) cleanupDone();
    };
  }, [onIngestComplete]);

  const refreshPoints = async () => {
    setLoadingPoints(true);
    try {
      const count = await api.qdrantCount(qdrantCfg);
      setPointCount(count);
    } catch { setPointCount(null); }
    finally { setLoadingPoints(false); }
  };

  useEffect(() => { refreshPoints(); }, []);

  async function pickFiles() {
    const sel = await openFileDialog({
      multiple: true,
      filters: [
        { name: "All Files", extensions: ["*"] },
        { name: "Documents & Images", extensions: ["pdf", "txt", "md", "docx", "pptx", "ppt", "png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff", "tif"] }
      ]
    });
    if (!sel) return;
    setFiles(Array.isArray(sel) ? sel : [sel]);
  }

  async function pickFolder() {
    const sel = await openFileDialog({
      multiple: false,
      directory: true,
    });
    if (!sel) return;
    const folderPath = Array.isArray(sel) ? sel[0] : sel;
    if (!folderPath) return;
    setScanningFolder(true);
    try {
      const found = await api.listIngestableFiles(folderPath);
      setFiles(found);
      const timestamp = new Date().toLocaleTimeString();
      setIngestLogs(prev => [
        ...prev,
        `[${timestamp}] [Folder] Found ${found.length} supported files in ${folderPath}`,
      ]);
    } catch (e) {
      const timestamp = new Date().toLocaleTimeString();
      setIngestLogs(prev => [...prev, `[${timestamp}] [Error] Folder scan failed: ${e}`]);
    } finally {
      setScanningFolder(false);
    }
  }

  async function startIngest() {
    if (!ready || files.length === 0) return;
    const batchFiles = files; const batchTag = tag.trim();
    setFiles([]);
    setIngestLogs([]);
    setCurrentStage(null);
    try {
      const id = await api.ingestDocuments(batchFiles, batchTag || null, null, qdrantCfg, embeddingConfig, config.paddleOcr ?? null);
      activeStreamIdsRef.current.add(id);
      setStreams(prev => [...prev, { id, files: batchFiles, tag: batchTag, progress: null, done: false, cancelled: false, chunks: 0, errors: [], error: null }]);
    } catch (e) {
      setFiles(prev => [...batchFiles, ...prev]);
      const timestamp = new Date().toLocaleTimeString();
      setIngestLogs(prev => [...prev, `[${timestamp}] [Error] Ingestion failed to start: ${e}`]);
    }
  }

  async function stopStream(streamId: string) { try { await api.cancelIngest(streamId); } catch {} }

  return (
    <div className="premium-card rounded-2xl p-5 glass-panel border border-white/5 space-y-4 animate-premium relative overflow-hidden group">
      <div className="absolute top-0 left-0 w-1 h-full theme-accent-bg opacity-20 group-hover:opacity-60 transition-opacity" />
      <div className="flex items-center justify-between">
        <div className="space-y-3 flex-1">
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${embedder.enabled ? "bg-emerald-400 shadow-[0_0_6px_#4ade80]" : "bg-white/10"}`} />
            <input
              type="text"
              value={embedder.name}
              onChange={e => onUpdate({ name: e.target.value })}
              className="text-[11px] font-black font-mono text-white bg-transparent border-b border-white/10 focus:border-theme-accent outline-none uppercase tracking-widest w-36"
            />
          </div>
          <div className="space-y-1">
            <label className="text-[8px] uppercase tracking-widest theme-muted font-black opacity-50">Model (HuggingFace)</label>
            <input
              type="text"
              value={embedder.modelId}
              onChange={e => onUpdate({ modelId: e.target.value })}
              placeholder="Qwen/Qwen3-Embedding-8B"
              className="w-full px-3 py-2 premium-input rounded-lg text-[10px] font-mono text-white focus:outline-none"
            />
          </div>
          <div className="flex items-center gap-3">
            <div className="space-y-1">
              <label className="text-[8px] uppercase tracking-widest theme-muted font-black opacity-50">Port</label>
              <input
                type="number"
                value={embedder.port}
                onChange={e => onUpdate({ port: parseInt(e.target.value) || 8100 })}
                className="w-20 px-3 py-2 premium-input rounded-lg text-[10px] font-mono text-white focus:outline-none"
              />
            </div>
            <div className="space-y-1">
              <label className="text-[8px] uppercase tracking-widest theme-muted font-black opacity-50">Concurrent</label>
              <input
                type="number"
                value={embedder.concurrency}
                onChange={e => onUpdate({ concurrency: parseInt(e.target.value) || 2 })}
                className="w-16 px-3 py-2 premium-input rounded-lg text-[10px] font-mono text-white focus:outline-none"
              />
            </div>
            <div className="flex-1 space-y-1">
              <label className="text-[8px] uppercase tracking-widest theme-muted font-black opacity-50">Collection</label>
              <p className="px-3 py-2 bg-black/30 rounded-lg text-[10px] font-mono text-theme-accent border border-white/5 truncate">{collection}</p>
            </div>
          </div>
        </div>
        <div className="flex flex-col items-end gap-2">
          <div className="flex items-center gap-2">
            {showRemove && (
              <button
                onClick={onRemove}
                className="p-1.5 rounded-lg hover:bg-red-500/15 text-red-400/60 hover:text-red-400 transition-all"
                title="Remove Embedder"
              >
                <X className="w-4 h-4" />
              </button>
            )}
            <span className="text-[8px] font-mono theme-muted opacity-50">~24 GB VRAM est.</span>
          </div>
          <div className="flex items-center gap-2">
            <button onClick={refreshPoints} disabled={loadingPoints} className="p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-all" title="Refresh point count">
              <RefreshCw className={`w-3.5 h-3.5 ${loadingPoints ? "animate-spin" : ""} theme-muted`} />
            </button>
          </div>
          <div className="flex items-center gap-2 text-[9px] font-black font-mono">
            <Circle className={`w-2 h-2 ${pointCount !== null && pointCount > 0 ? "fill-emerald-400 text-emerald-400" : "text-white/20"}`} />
            <span className="theme-muted">{pointCount !== null ? `${pointCount.toLocaleString()} pts` : "—"}</span>
          </div>
        </div>
      </div>

      <div className="border-t border-white/5 pt-4 space-y-3">
        <div className="grid grid-cols-[minmax(0,1fr)_7rem] gap-2 items-stretch">
          <div className="flex flex-col gap-2 min-w-0">
            <button
              onClick={pickFolder}
              disabled={!ready || scanningFolder}
              className="flex items-center justify-center gap-2 px-4 py-3 premium-input rounded-xl text-[10px] uppercase tracking-widest font-black font-mono theme-text disabled:opacity-20 premium-button transition-all whitespace-nowrap"
              title="Select a folder and recursively ingest supported documents and images"
            >
              {scanningFolder ? <Loader2 className="w-4 h-4 animate-spin" /> : <FolderOpen className="w-4 h-4" />}
              Upload Folder
            </button>
            <button
              onClick={pickFiles}
              disabled={!ready}
              className="flex items-center justify-center gap-2 px-4 py-3 premium-input rounded-xl text-[10px] uppercase tracking-widest font-black font-mono theme-text disabled:opacity-20 premium-button transition-all"
            >
              <Upload className="w-4 h-4" />
              {files.length === 0 ? "Upload Files" : `${files.length} FILES`}
            </button>
          </div>
          <input
            type="text"
            value={tag}
            onChange={e => setTag(e.target.value)}
            placeholder="TAG"
            className="w-full px-3 py-3 premium-input rounded-xl text-[10px] font-black font-mono text-white focus:outline-none uppercase"
          />
        </div>

        {files.length > 0 && (
          <div className="bg-black/40 border border-white/5 rounded-lg p-3 text-[9px] font-mono space-y-1 max-h-24 overflow-y-auto">
            {files.map(f => (
              <div key={f} className="flex items-center justify-between group/file">
                <span className="truncate flex-1 theme-muted opacity-60" title={f}>{f}</span>
                <button onClick={() => setFiles(prev => prev.filter(x => x !== f))} className="opacity-0 group-hover/file:opacity-100 text-red-500 p-1"><X className="w-3 h-3" /></button>
              </div>
            ))}
          </div>
        )}

        <div className="flex items-center gap-3">
          <button
            onClick={startIngest}
            disabled={!ready || files.length === 0 || ingesting}
            className="flex-1 flex items-center justify-center gap-2 px-4 py-3 theme-accent-bg text-black text-[10px] uppercase tracking-widest font-black rounded-xl hover:brightness-125 disabled:opacity-20 shadow-lg premium-button transition-all"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            {ingesting ? "INGESTING..." : "INGEST"}
          </button>
          {ingesting && streams.filter(s => !s.done).map(s => (
            <button key={s.id} onClick={() => stopStream(s.id)} className="p-3 bg-red-500/10 border border-red-500/20 text-red-400 rounded-xl hover:bg-red-500 hover:text-white transition-all">
              <StopCircle className="w-4 h-4" />
            </button>
          ))}
          {streams.length > 0 && (
            <button onClick={() => {
              setStreams([]);
              // Also clear persisted state for this collection
              api.loadIngestState().then(json => {
                try {
                  const existing = json && json !== "{}" ? JSON.parse(json) : {};
                  delete existing[collection];
                  api.saveIngestState(JSON.stringify(existing)).catch(() => {});
                } catch {}
              }).catch(() => {});
            }} className="p-3 bg-white/5 border border-white/10 theme-muted rounded-xl hover:bg-white/10 transition-all" title="Clear Logs">
              <Trash2 className="w-4 h-4" />
            </button>
          )}
        </div>

        {(pct > 0 || ingesting) && (
          <div className="space-y-3">
            <div className="space-y-1">
              <div className="flex justify-between text-[9px] font-black font-mono theme-muted opacity-60">
                <span>{currentFile ? `${currentFile} — ${pct}%` : `PROGRESS ${pct}%`}</span>
                <span>{pct}%</span>
              </div>
              <div className="h-1.5 bg-white/5 rounded-full overflow-hidden border border-white/5">
                <div className="h-full theme-accent-bg transition-all duration-700 ease-out" style={{ width: `${pct}%` }} />
              </div>
            </div>

            {/* Glowing Live Progression Pipeline Indicator */}
            <div className="flex items-center justify-between p-3 rounded-xl bg-black/40 border border-white/5 font-mono text-[9px] shadow-inner select-none transition-all">
              {/* PaddleOCR / Read Stage */}
              <div className={`flex flex-col items-center flex-1 text-center transition-all duration-300 ${
                (currentStage?.stage === "ocr_start" || currentStage?.stage === "ocr_page" || currentStage?.stage === "read")
                  ? "text-cyan-400 font-bold"
                  : "text-white/20"
              }`}>
                <span className="uppercase tracking-wider">1. PaddleOCR / Read</span>
                <span className="text-[8px] mt-0.5 opacity-80">
                  {(currentStage?.stage === "ocr_start" || currentStage?.stage === "ocr_page" || currentStage?.stage === "read")
                    ? (currentStage.stage === "read" ? "Reading File..." : `Page ${currentStage.done}/${currentStage.total}`)
                    : "—"
                  }
                </span>
              </div>

              <ChevronRight className={`w-3.5 h-3.5 text-white/10 shrink-0 mx-2 ${
                (currentStage?.stage === "ocr_start" || currentStage?.stage === "ocr_page" || currentStage?.stage === "read")
                  ? "text-cyan-400/40 animate-pulse"
                  : ""
              }`} />

              {/* Embedding Stage */}
              <div className={`flex flex-col items-center flex-1 text-center transition-all duration-300 ${
                currentStage?.stage === "embed"
                  ? "text-indigo-400 font-bold"
                  : "text-white/20"
              }`}>
                <span className="uppercase tracking-wider">2. Embedder Model</span>
                <span className="text-[8px] mt-0.5 opacity-80">
                  {currentStage?.stage === "embed"
                    ? `Chunk ${currentStage.done}/${currentStage.total}`
                    : "—"
                  }
                </span>
              </div>

              <ChevronRight className={`w-3.5 h-3.5 text-white/10 shrink-0 mx-2 ${
                currentStage?.stage === "embed"
                  ? "text-indigo-400/40 animate-pulse"
                  : ""
              }`} />

              {/* Qdrant Stage */}
              <div className={`flex flex-col items-center flex-1 text-center transition-all duration-300 ${
                currentStage?.stage === "upsert"
                  ? "text-purple-400 font-bold"
                  : "text-white/20"
              }`}>
                <span className="uppercase tracking-wider">3. Qdrant DB</span>
                <span className="text-[8px] mt-0.5 opacity-80">
                  {currentStage?.stage === "upsert"
                    ? `Batch ${currentStage.done}/${currentStage.total}`
                    : "—"
                  }
                </span>
              </div>
            </div>
          </div>
        )}

        <div className="space-y-1.5 animate-premium">
          <div className="flex justify-between items-center ml-1">
            <label className="text-[8px] uppercase tracking-widest theme-muted font-black opacity-50">Live Ingestion Logs</label>
            {ingestLogs.length > 0 && (
              <button
                type="button"
                onClick={() => setIngestLogs([])}
                className="text-[8px] uppercase tracking-widest theme-faint hover:theme-text transition-colors font-bold font-mono"
              >
                Clear Screen
              </button>
            )}
          </div>
          <div ref={logContainerRef} className="bg-black/40 border border-white/5 rounded-xl p-3.5 font-mono text-[9px] text-white/80 space-y-1 h-36 overflow-y-auto scrollbar-thin scrollbar-thumb-white/10 select-text animate-premium">
            {ingestLogs.length > 0 ? (
              ingestLogs.map((log, i) => {
                let colorClass = "text-white/60";
                if (log.includes("[Error]")) colorClass = "text-red-400 font-bold";
                else if (log.includes("[Warning]")) colorClass = "text-amber-400";
                else if (log.includes("[Done]")) colorClass = "text-emerald-400 font-bold";
                else if (log.includes("[PaddleOCR]")) colorClass = "text-cyan-400";
                else if (log.includes("[Embedder]")) colorClass = "text-indigo-300";
                else if (log.includes("[Qdrant]")) colorClass = "text-purple-300";
                
                return (
                  <div key={i} className={colorClass}>
                    {log}
                  </div>
                );
              })
            ) : (
              <div className="text-white/30 italic flex items-center justify-center h-full select-none">
                Awaiting ingestion task... Logs will appear here live.
              </div>
            )}
          </div>
        </div>

        {streams.filter(s => s.done).length > 0 && (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-[9px] font-black font-mono text-emerald-400">
              <CheckCircle2 className="w-3.5 h-3.5" />
              {streams.filter(s => s.done).map(s => `${s.chunks} chunks`).join(" · ")}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function QdrantDbPanel({ config, onConfigChange }: { config: AppConfig; onConfigChange?: (patch: Partial<AppConfig>) => void }) {
  const embedders = config.embedders && config.embedders.length > 0 ? config.embedders : [DEFAULT_EMBEDDER];
  // Build collection list from all embedders + default
  const allCollections = useMemo(() => {
    const cols: string[] = [];
    for (const emb of embedders) {
      const c = emb.collection || `kb_${emb.name.toLowerCase().replace(/\s+/g, "_").replace(/[^a-z0-9_]/g, "")}`;
      if (!cols.includes(c)) cols.push(c);
    }
    if (config.qdrant.collection && !cols.includes(config.qdrant.collection)) {
      cols.unshift(config.qdrant.collection);
    }
    if (cols.length === 0) cols.push("kb_default");
    return cols;
  }, [embedders, config.qdrant.collection]);

  const [selectedCollection, setSelectedCollection] = useState(allCollections[0] || "");
  const [snapshots, setSnapshots] = useState<{ name: string; creation_time?: string; size: number }[]>([]);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [savingAll, setSavingAll] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [downloading, setDownloading] = useState<string | null>(null);
  const [downloadingAll, setDownloadingAll] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [isError, setIsError] = useState(false);

  const [chunks, setChunks] = useState<any[]>([]);
  const [loadingChunks, setLoadingChunks] = useState(false);
  const [chunksError, setChunksError] = useState<string | null>(null);
  const [nextOffset, setNextOffset] = useState<any>(null);
  const [currentOffset, setCurrentOffset] = useState<any>(null);
  const [offsetsHistory, setOffsetsHistory] = useState<any[]>([]);

  const qdrantEndpoint = config.qdrant.endpoint || (config.ssh.host ? `http://${config.ssh.host}:6333` : "");
  const qdCfg = { ...config.qdrant, endpoint: qdrantEndpoint };
  const isQdrantOfflineError = (err: unknown) =>
    /error sending request|failed to fetch|network|timeout|timed out|connection|refused|10060|unreachable|could not connect|tcp/i.test(
      err instanceof Error ? err.message : String(err),
    );

  // Update selected when allCollections changes
  useEffect(() => {
    if (allCollections.length > 0 && !allCollections.includes(selectedCollection)) {
      setSelectedCollection(allCollections[0]);
    }
  }, [allCollections]);

  // Synchronize with global config collection
  useEffect(() => {
    if (config.qdrant.collection && config.qdrant.collection !== selectedCollection) {
      setSelectedCollection(config.qdrant.collection);
    }
  }, [config.qdrant.collection]);

  const loadSnapshots = async (coll?: string) => {
    if (!qdrantEndpoint) return;
    const col = coll || selectedCollection;
    if (!col) return;
    if (col === "all") {
      setSnapshots([]);
      setStatus(null);
      return;
    }
    setLoading(true); setStatus(null);
    try {
      const list = await api.qdrantListSnapshots(qdCfg, col);
      setSnapshots(list);
    } catch (e: any) {
      const msg = e.message || String(e);
      if (/doesn'?t exist|not found|404/i.test(msg)) {
        setSnapshots([]);
        setStatus(`Collection '${col}' doesn't exist yet — ingest some documents first.`);
        setIsError(false);
      } else if (isQdrantOfflineError(e)) {
        setSnapshots([]);
        setStatus(null);
        setIsError(false);
      } else {
        setStatus(msg); setIsError(true);
      }
    } finally { setLoading(false); }
  };

  const loadChunks = async (offsetVal: any = null) => {
    if (!qdrantEndpoint || !selectedCollection) return;
    if (selectedCollection === "all") {
      setChunks([]);
      setNextOffset(null);
      setChunksError(null);
      return;
    }
    setLoadingChunks(true); setChunksError(null);
    try {
      const res = await api.qdrantScrollInCollection(qdCfg, selectedCollection, 3, offsetVal);
      setChunks(res.chunks || []);
      setNextOffset(res.next_offset || null);
    } catch (e: any) {
      const msg = e.message || String(e);
      if (/doesn'?t exist|not found|404/i.test(msg)) {
        setChunks([]);
        setNextOffset(null);
      } else if (isQdrantOfflineError(e)) {
        setChunks([]);
        setNextOffset(null);
        setChunksError(null);
      } else {
        setChunksError(msg);
        setChunks([]);
        setNextOffset(null);
      }
    } finally { setLoadingChunks(false); }
  };

  const handleNextPage = () => {
    if (!nextOffset) return;
    const newHistory = [...offsetsHistory, currentOffset];
    setOffsetsHistory(newHistory);
    setCurrentOffset(nextOffset);
    loadChunks(nextOffset);
  };

  const handlePrevPage = () => {
    if (offsetsHistory.length === 0) return;
    const newHistory = [...offsetsHistory];
    const prevOffset = newHistory.pop();
    setOffsetsHistory(newHistory);
    setCurrentOffset(prevOffset);
    loadChunks(prevOffset);
  };

  const reloadSelectedCollection = async () => {
    if (!selectedCollection) return;
    setCurrentOffset(null);
    setOffsetsHistory([]);
    await Promise.all([
      loadSnapshots(selectedCollection),
      loadChunks(null),
    ]);
  };

  useEffect(() => {
    if (selectedCollection && qdrantEndpoint) {
      setCurrentOffset(null);
      setOffsetsHistory([]);
      void reloadSelectedCollection();
    } else {
      setChunks([]);
      setNextOffset(null);
      setOffsetsHistory([]);
    }
  }, [selectedCollection, qdrantEndpoint]);

  const saveSnapshot = async () => {
    if (!selectedCollection || selectedCollection === "all") return;
    setSaving(true); setStatus(null);
    try {
      const snap = await api.qdrantCreateSnapshot(qdCfg, selectedCollection);
      setStatus(`Snapshot saved: ${snap.name}`);
      setIsError(false);
      await reloadSelectedCollection();
    } catch (e: any) {
      if (isQdrantOfflineError(e)) {
        setStatus(null);
        setIsError(false);
      } else {
        setStatus(e.message || String(e)); setIsError(true);
      }
    } finally { setSaving(false); }
  };

  const saveAllSnapshots = async () => {
    setSavingAll(true); setStatus(null);
    try {
      const results = await api.createAllQdrantSnapshots(qdCfg);
      const ok = results.filter(r => !r.snapshot_name.startsWith("ERROR"));
      const fail = results.filter(r => r.snapshot_name.startsWith("ERROR"));
      if (fail.length > 0) {
        setStatus(`Saved ${ok.length}/${results.length} collections. ${fail.length} failed.`);
        setIsError(true);
      } else {
        setStatus(`Saved snapshots for all ${ok.length} collections.`);
        setIsError(false);
      }
      await reloadSelectedCollection();
    } catch (e: any) {
      if (isQdrantOfflineError(e)) {
        setStatus(null);
        setIsError(false);
      } else {
        setStatus(e.message || "Failed to save all snapshots"); setIsError(true);
      }
    } finally { setSavingAll(false); }
  };

  const uploadSnapshot = async () => {
    if (!selectedCollection || selectedCollection === "all") return;
    setStatus(null);
    try {
      const sel = await openFileDialog({
        multiple: true,
        filters: [{ name: "Qdrant Snapshot", extensions: ["snapshot", "tar"] }]
      });
      if (!sel) return;
      const snapshotPaths = (Array.isArray(sel) ? sel : [sel]).filter(Boolean);
      if (snapshotPaths.length === 0) return;

      setUploading(true);
      setIsError(false);

      for (let i = 0; i < snapshotPaths.length; i++) {
        const snapshotPath = snapshotPaths[i];
        setStatus(`Uploading and recovering snapshot ${i + 1}/${snapshotPaths.length}: ${snapshotPath.split(/[\\/]/).pop()}...`);
        await api.qdrantUploadSnapshot(qdCfg, selectedCollection, snapshotPath);
      }
      setStatus(`Successfully uploaded and restored ${snapshotPaths.length} snapshot${snapshotPaths.length === 1 ? "" : "s"}.`);
      setIsError(false);
      await reloadSelectedCollection();
    } catch (e: any) {
      if (isQdrantOfflineError(e)) {
        setStatus(null);
        setIsError(false);
      } else {
        setStatus(e.message || String(e));
        setIsError(true);
      }
    } finally {
      setUploading(false);
    }
  };

  const downloadSnapshot = async (snapshotName: string) => {
    if (!selectedCollection) return;
    try {
      const savePath = await saveFileDialog({
        defaultPath: snapshotName,
        filters: [{ name: "Qdrant Snapshot", extensions: ["snapshot"] }]
      });
      if (!savePath) return;
      setDownloading(snapshotName);
      setStatus(`Downloading ${snapshotName}...`);
      setIsError(false);
      await api.qdrantDownloadSnapshot(qdCfg, selectedCollection, snapshotName, savePath);
      setStatus(`Downloaded to ${savePath}`);
      setIsError(false);
    } catch (e: any) {
      if (isQdrantOfflineError(e)) {
        setStatus(null);
        setIsError(false);
      } else {
        setStatus(e.message || String(e));
        setIsError(true);
      }
    } finally {
      setDownloading(null);
    }
  };

  const downloadAllSnapshots = async () => {
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const dir = await open({ directory: true, multiple: false });
      if (!dir) return;
      const dirPath = dir as string;
      setDownloadingAll(true); setStatus(null);
      const paths = await api.downloadAllQdrantSnapshots(qdCfg, dirPath);
      setStatus(`Downloaded ${paths.length} snapshots to ${dirPath}`);
      setIsError(false);
    } catch (e: any) {
      if (isQdrantOfflineError(e)) {
        setStatus(null);
        setIsError(false);
      } else {
        setStatus(e.message || "Failed to download all snapshots"); setIsError(true);
      }
    } finally { setDownloadingAll(false); }
  };

  const formatBytes = (b: number) => b > 1e9 ? `${(b/1e9).toFixed(1)} GB` : `${(b/1e6).toFixed(0)} MB`;

  return (
    <div className="premium-card rounded-2xl p-5 glass-panel border border-white/5 space-y-4">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-3">
          <p className="text-[10px] uppercase tracking-[0.25em] theme-accent font-black font-mono flex items-center gap-2">
            <HardDrive className="w-3.5 h-3.5" /> Qdrant Database
          </p>
          <select
            value={selectedCollection}
            onChange={e => {
              const val = e.target.value;
              setSelectedCollection(val);
              onConfigChange?.({ qdrant: { ...config.qdrant, collection: val } });
            }}
            className="px-3 py-1.5 premium-input rounded-lg text-[10px] font-mono text-white focus:outline-none appearance-none cursor-pointer bg-black/40 border border-white/10 min-w-[160px]"
          >
            {allCollections.map(c => (
              <option key={c} value={c} className="theme-surface theme-text">{c}</option>
            ))}
          </select>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={reloadSelectedCollection} disabled={loading || loadingChunks} className="p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-all" title="Refresh">
            <RefreshCw className={`w-3.5 h-3.5 ${(loading || loadingChunks) ? "animate-spin" : ""} theme-muted`} />
          </button>
          <button
            onClick={downloadAllSnapshots}
            disabled={downloadingAll || !qdrantEndpoint}
            className="flex items-center gap-2 px-4 py-2 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-[10px] uppercase tracking-widest font-black rounded-xl hover:bg-emerald-500 hover:text-black disabled:opacity-20 transition-all"
            title="Backup all collections snapshots to local folder"
          >
            <Download className="w-3.5 h-3.5" />
            {downloadingAll ? "Downloading All..." : "Download All"}
          </button>
          <button
            onClick={saveAllSnapshots}
            disabled={savingAll || !qdrantEndpoint}
            className="flex items-center gap-2 px-4 py-2 bg-blue-500/10 border border-blue-500/20 text-blue-400 text-[10px] uppercase tracking-widest font-black rounded-xl hover:bg-blue-500 hover:text-white disabled:opacity-20 transition-all"
            title="Create snapshots for all collections on remote server"
          >
            <Save className="w-3.5 h-3.5" />
            {savingAll ? "Saving All..." : "Save All"}
          </button>
          <button
            onClick={uploadSnapshot}
            disabled={uploading || !qdrantEndpoint || !selectedCollection || selectedCollection === "all"}
            className="flex items-center gap-2 px-4 py-2 bg-white/5 border border-white/10 text-white text-[10px] uppercase tracking-widest font-black rounded-xl hover:bg-white/10 disabled:opacity-20 transition-all"
          >
            {uploading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Upload className="w-3.5 h-3.5" />}
            {uploading ? "Uploading..." : "Upload Snapshots"}
          </button>
          <button
            onClick={saveSnapshot}
            disabled={saving || !qdrantEndpoint || !selectedCollection || selectedCollection === "all"}
            className="flex items-center gap-2 px-4 py-2 theme-accent-bg text-black text-[10px] uppercase tracking-widest font-black rounded-xl hover:brightness-125 disabled:opacity-20 shadow-lg premium-button transition-all"
          >
            <Save className="w-3.5 h-3.5" />
            {saving ? "Saving..." : "Save Snapshot"}
          </button>
        </div>
      </div>

      {status && (
        <div className={`p-3 rounded-lg text-[10px] font-mono ${isError ? "border border-red-500/20 bg-red-500/5 text-red-300" : "border border-emerald-500/20 bg-emerald-500/5 text-emerald-300"}`}>
          {isError ? <span className="font-black uppercase">Error: </span> : <span className="font-black uppercase">OK: </span>}
          {status}
        </div>
      )}

      {selectedCollection === "all" ? (
        <div className="h-20 flex items-center justify-center border border-dashed border-white/10 rounded-xl">
          <p className="text-[10px] font-mono theme-muted italic opacity-30">Select a concrete collection to manage snapshots.</p>
        </div>
      ) : snapshots.length > 0 ? (
        <div className="space-y-2">
          <p className="text-[8px] uppercase tracking-widest theme-muted font-black opacity-40">Saved Snapshots · {selectedCollection}</p>
          {snapshots.map(s => (
            <div key={s.name} className="flex items-center justify-between bg-black/30 rounded-lg px-4 py-3 border border-white/5">
              <div className="flex items-center gap-3">
                <Database className="w-4 h-4 theme-muted opacity-40" />
                <div>
                  <p className="text-[10px] font-black font-mono text-white">{s.name}</p>
                  <p className="text-[8px] font-mono theme-muted opacity-50">{formatBytes(s.size)} · {s.creation_time?.split("T")[0] || "—"}</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => downloadSnapshot(s.name)}
                  disabled={downloading === s.name}
                  className="px-3 py-1.5 rounded-lg bg-blue-500/10 border border-blue-500/20 text-blue-400 text-[9px] font-black uppercase hover:bg-blue-500 hover:text-white transition-all flex items-center gap-1.5"
                >
                  {downloading === s.name ? <Loader2 className="w-3 h-3 animate-spin" /> : <Download className="w-3 h-3" />}
                  Download
                </button>
                <button onClick={async () => {
                  try {
                    setStatus(`Restoring ${s.name}...`);
                    setIsError(false);
                    await api.qdrantRestoreSnapshot(qdCfg, selectedCollection, s.name);
                    setStatus(`Restored: ${s.name}`);
                    await reloadSelectedCollection();
                  } catch (e: any) {
                    if (isQdrantOfflineError(e)) {
                      setStatus(null);
                      setIsError(false);
                    } else {
                      setStatus(e.message);
                      setIsError(true);
                    }
                  }
                }} className="px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-[9px] font-black uppercase hover:bg-emerald-500 hover:text-black transition-all">
                  Restore
                </button>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="h-20 flex items-center justify-center border border-dashed border-white/10 rounded-xl">
          <p className="text-[10px] font-mono theme-muted italic opacity-30">No snapshots for this collection.</p>
        </div>
      )}

      {/* Embedded Chunks Display */}
      <div className="border-t border-white/5 pt-4 space-y-3">
        <div className="flex items-center justify-between">
          <p className="text-[10px] uppercase tracking-widest theme-muted font-black opacity-45 flex items-center gap-2">
            <FileText className="w-3.5 h-3.5" /> Embedded Chunks
          </p>
          {chunks.length > 0 && (
            <div className="flex items-center gap-2">
              <button
                onClick={handlePrevPage}
                disabled={offsetsHistory.length === 0 || loadingChunks}
                className="p-1 rounded bg-white/5 border border-white/10 hover:bg-white/10 transition-all disabled:opacity-20 text-white"
                title="Previous Page"
                type="button"
              >
                <ChevronLeft className="w-3.5 h-3.5" />
              </button>
              <span className="text-[9px] font-mono theme-muted opacity-60">
                Page {offsetsHistory.length + 1}
              </span>
              <button
                onClick={handleNextPage}
                disabled={!nextOffset || loadingChunks}
                className="p-1 rounded bg-white/5 border border-white/10 hover:bg-white/10 transition-all disabled:opacity-20 text-white"
                title="Next Page"
                type="button"
              >
                <ChevronRight className="w-3.5 h-3.5" />
              </button>
            </div>
          )}
        </div>

        {loadingChunks ? (
          <div className="h-24 flex items-center justify-center">
            <Loader2 className="w-5 h-5 animate-spin theme-accent" />
          </div>
        ) : chunksError ? (
          <div className="p-3 rounded-lg border border-red-500/20 bg-red-500/5 text-red-300 text-[9px] font-mono">
            <span className="font-black uppercase">Failed to fetch data: </span>{chunksError}
          </div>
        ) : chunks.length > 0 ? (
          <div className="space-y-3">
            {chunks.map((c, i) => (
              <div key={c.id || i} className="bg-black/30 rounded-xl p-4 border border-white/5 space-y-2 hover:bg-black/40 transition-colors">
                <div className="flex items-center justify-between text-[9px] font-mono theme-muted opacity-50">
                  <span className="truncate max-w-[250px]" title={c.file_name || c.file_path}>
                    📄 {c.file_name || "Direct Embed"}
                  </span>
                  <span>
                    Idx: {c.chunk_index}
                  </span>
                </div>
                <p className="text-xs theme-text leading-relaxed font-medium line-clamp-4 break-words">
                  {c.text}
                </p>
              </div>
            ))}
          </div>
        ) : (
          <div className="h-20 flex items-center justify-center border border-dashed border-white/10 rounded-xl">
            <p className="text-[10px] font-mono theme-muted italic opacity-35">
              {selectedCollection === "all" ? "Select a concrete collection to preview chunks." : `No chunks found in '${selectedCollection}'. Ingest some files first.`}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

function KnowledgeBaseStep({
  gpuStatus: propGpuStatus,
  samples,
  loading,
  error,
  config,
  onConfigChange,
  onSkip,
}: {
  gpuStatus: GPUState | null;
  samples: Chunk[];
  loading: boolean;
  error: string | null;
  config: AppConfig;
  onConfigChange: (patch: Partial<AppConfig>) => void;
  onSkip?: () => void;
}) {
  const [gpuLoading, setGpuLoading] = useState(false);
  const [setupAllLoading, setSetupAllLoading] = useState(false);
  const [setupAllError, setSetupAllError] = useState<string | null>(null);
  const [setupOcrLoading, setSetupOcrLoading] = useState(false);
  const [setupOcrError, setSetupOcrError] = useState<string | null>(null);
  const [setupAllLog, setSetupAllLog] = useState(getSetupLogSnapshot());
  const [qdrantOnlyLoading, setQdrantOnlyLoading] = useState(false);
  const [qdrantOnlyLog, setQdrantOnlyLog] = useState<string[]>([]);
  const [qdrantOnlyError, setQdrantOnlyError] = useState<string | null>(null);
  const [embedderCounts, setEmbedderCounts] = useState<Record<number, number>>({});
  const [localGpuStatus, setLocalGpuStatus] = useState<GPUState | null>(null);
  const gpuStatus = localGpuStatus ?? propGpuStatus;

  const embedders = config.embedders && config.embedders.length > 0 ? config.embedders : [DEFAULT_EMBEDDER];
  const paddleOcr: PaddleOcrConfig = { ...DEFAULT_PADDLE_OCR, ...config.paddleOcr };

  useEffect(() => subscribeSetupLogs(setSetupAllLog), []);

  const detectGpu = async () => {
    if (!config.ssh.host) return;
    setGpuLoading(true);
    try {
      const result = await api.nvidiaSmi(config.ssh);
      setLocalGpuStatus(result);
    } catch {} finally { setGpuLoading(false); }
  };

  const refreshAllEmbedderCounts = async () => {
    if (embedders.length === 0) return;
    const counts: Record<number, number> = {};
    for (let i = 0; i < embedders.length; i++) {
      const embedder = embedders[i];
      const coll = embedder.collection || `kb_${embedder.name.toLowerCase().replace(/\s+/g, "_").replace(/[^a-z0-9_]/g, "")}`;
      const cfg = { ...config.qdrant, collection: coll };
      try {
        const c = await api.qdrantCount(cfg);
        counts[i] = c;
      } catch {
        counts[i] = 0;
      }
    }
    setEmbedderCounts(counts);
  };

  useEffect(() => {
    refreshAllEmbedderCounts();
  }, [embedders.length]);

  const setupAllEmbedders = async () => {
    if (embedders.length === 0) return;
    clearSetupLogs();
    setSetupAllLoading(true);
    setSetupAllError(null);
    setSetupOcrError(null);
    try {
      const results = await api.serveSetupAllEmbedders(config.ssh, config.docker, embedders, config.hfToken ?? null);
      const updatedEmbedders = embedders.map((e, i) => ({ ...e, enabled: results[i]?.status !== "error" }));
      onConfigChange({ embedders: updatedEmbedders });
      await refreshAllEmbedderCounts();
    } catch (e: any) { setSetupAllError(e.message || String(e)); }
    finally { setSetupAllLoading(false); }
  };

  const setupPaddleOcr = async () => {
    if (!config.ssh.host) return;
    clearSetupLogs();
    setSetupOcrLoading(true);
    setSetupOcrError(null);
    setSetupAllError(null);
    const nextPaddleOcr = { ...paddleOcr, enabled: true };
    if (!paddleOcr.enabled) {
      onConfigChange({ paddleOcr: nextPaddleOcr });
    }
    try {
      await api.serveBootPaddleocr(config.ssh, config.docker, nextPaddleOcr);
    } catch (e: any) {
      setSetupOcrError(e.message || String(e));
    } finally {
      setSetupOcrLoading(false);
    }
  };

  const installQdrantOnly = async () => {
    if (!config.ssh.host) return;
    setQdrantOnlyLoading(true);
    setQdrantOnlyError(null);
    setQdrantOnlyLog(["[stage] Connecting to GPU server..."]);
    try {
      // Extract port from the configured Qdrant endpoint URL, fall back to 6333
      let qdrantPort = 6333;
      const ep = config.qdrant?.endpoint ?? "";
      if (ep) {
        const portMatch = ep.match(/:(\d+)(\/|$)/);
        if (portMatch) qdrantPort = parseInt(portMatch[1], 10);
      }
      const dataDir = "/root/fine-tune";
      setQdrantOnlyLog(prev => [...prev, `[stage] Installing Qdrant on port ${qdrantPort}...`]);
      await api.serveEnsureQdrant(config.ssh, config.docker, qdrantPort, dataDir);
      setQdrantOnlyLog(prev => [...prev, "[ok] Qdrant is running — ready for semantic search."]);
      // Small delay so user sees the success message, then advance to next step
      setTimeout(() => { onSkip?.(); }, 1200);
    } catch (e: any) {
      setQdrantOnlyError(e.message || String(e));
      setQdrantOnlyLog(prev => [...prev, `[error] ${e.message || String(e)}`]);
    } finally {
      setQdrantOnlyLoading(false);
    }
  };

  const addEmbedder = () => {
    const newEmbedders = [...embedders, {
      name: `embedder_${embedders.length + 1}`,
      modelId: "Qwen/Qwen3-Embedding-8B",
      port: DEFAULT_EMBEDDER.port + embedders.length,
      collection: "",
      concurrency: 2,
      vectorDim: undefined,
      enabled: true,
      persistent: embedders.length === 0,
      gpuMemoryUtilization: 0.084,
    }];
    onConfigChange({ embedders: newEmbedders });
  };

  const updateEmbedder = (idx: number, patch: Partial<EmbedderConfig>) => {
    const updated = embedders.map((e, i) => i === idx ? { ...e, ...patch } : e);
    onConfigChange({ embedders: updated });
  };

const removeEmbedder = (idx: number) => {
    if (idx === 0) return;
    onConfigChange({ embedders: embedders.filter((_, i) => i !== idx) });
  };

  const totalPoints = Object.values(embedderCounts).reduce((a, b) => a + b, 0);
  const hasEmbedders = embedders.length > 0;
  const canProceed = totalPoints > 0;

  return (
    <div className="space-y-6 animate-premium">
      <div className="flex flex-col gap-1">
        <h3 className="text-base-fluid uppercase tracking-[0.25em] theme-accent font-black italic font-serif">
          Knowledge Base
        </h3>
        <p className="text-sm-fluid theme-muted font-medium opacity-80 leading-relaxed max-w-2xl">
          Configure one or more embedding models on your GPU server. Each embedder gets its own Qdrant collection for targeted knowledge retrieval.
        </p>
      </div>

      <div className="flex flex-col gap-3">
        <div className="flex items-center gap-3">
          <div className="flex-1">
            <GpuServerStatusCard gpuStatus={gpuStatus} loading={gpuLoading} />
          </div>
          <div className="flex flex-col gap-2">
            <button
              onClick={detectGpu}
              disabled={gpuLoading || !config.ssh.host}
              className="px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-[10px] uppercase tracking-widest font-black font-mono theme-text hover:border-theme-accent/30 disabled:opacity-20 transition-all"
            >
              {gpuLoading ? "Detecting..." : "Detect GPU"}
            </button>
            <button
              onClick={setupAllEmbedders}
              disabled={setupAllLoading || setupOcrLoading || embedders.length === 0 || !config.ssh.host}
              className="px-4 py-3 rounded-xl theme-accent-bg text-black text-[10px] uppercase tracking-widest font-black hover:brightness-125 disabled:opacity-20 shadow-lg shadow-theme-accent/10 transition-all flex items-center justify-center gap-2"
            >
              {setupAllLoading ? <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Installing...</> : <><Server className="w-3.5 h-3.5" /> Setup All Embedders</>}
            </button>
            <button
              onClick={setupPaddleOcr}
              disabled={setupOcrLoading || setupAllLoading || !config.ssh.host}
              className="px-4 py-3 rounded-xl bg-orange-500/10 border border-orange-500/20 text-orange-300 text-[10px] uppercase tracking-widest font-black hover:bg-orange-500/20 hover:border-orange-400/40 disabled:opacity-20 transition-all flex items-center justify-center gap-2"
              title="Deploy PaddleOCR-VL separately from the embedding models."
            >
              {setupOcrLoading
                ? <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Setting up OCR...</>
                : <><ScanText className="w-3.5 h-3.5" /> Setup Paddle OCR</>}
            </button>
            {/* Install Qdrant Only — for users with a pre-existing database */}
            <button
              onClick={installQdrantOnly}
              disabled={qdrantOnlyLoading || !config.ssh.host}
              className="px-4 py-3 rounded-xl bg-blue-500/10 border border-blue-500/20 text-blue-300 text-[10px] uppercase tracking-widest font-black hover:bg-blue-500/20 hover:border-blue-400/40 disabled:opacity-20 transition-all flex items-center justify-center gap-2"
              title="Install only Qdrant on the GPU server. No PaddleOCR or embedding models. Use this when you already have an existing database to upload."
            >
              {qdrantOnlyLoading
                ? <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Installing Qdrant...</>
                : <><Database className="w-3.5 h-3.5" /> Qdrant Only</>}
            </button>
            {onSkip && (
              <button
                onClick={onSkip}
                className="px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-[10px] uppercase tracking-widest font-black font-mono theme-text hover:border-white/20 transition-all"
              >
                Skip KB Config
              </button>
            )}
          </div>
        </div>
        {setupAllError && (
          <div className="p-3 rounded-lg border border-red-500/20 bg-red-500/5 text-red-300 text-[10px] font-mono">
            <span className="font-black uppercase">Setup Error: </span>{setupAllError}
          </div>
        )}
        {setupOcrError && (
          <div className="p-3 rounded-lg border border-red-500/20 bg-red-500/5 text-red-300 text-[10px] font-mono">
            <span className="font-black uppercase">Paddle OCR Error: </span>{setupOcrError}
          </div>
        )}
        {setupAllLog.trim().length > 0 && (
          <div className="bg-black/60 border border-white/10 rounded-xl p-4 max-h-48 overflow-y-auto font-mono text-[10px] leading-relaxed space-y-0.5 scrollbar-thin scrollbar-thumb-white/10">
            {setupAllLog.split(/\r?\n/).filter(Boolean).map((line, i) => (
              <div key={i} className={`${line.startsWith("[error]") ? "text-red-400" : line.startsWith("[ok]") ? "text-emerald-400" : line.startsWith("[stage]") ? "text-cyan-300" : "theme-text/60"}`}>
                {line}
              </div>
            ))}
            {(setupAllLoading || setupOcrLoading) && <span className="animate-pulse text-white/30">▊</span>}
          </div>
        )}
        {/* Qdrant-only install log */}
        {(qdrantOnlyLog.length > 0 || qdrantOnlyError) && (
          <div className="bg-black/60 border border-blue-500/20 rounded-xl p-4 max-h-36 overflow-y-auto font-mono text-[10px] leading-relaxed space-y-0.5">
            <p className="text-[8px] uppercase tracking-widest text-blue-400/60 font-black mb-1.5">Qdrant Install</p>
            {qdrantOnlyLog.map((line, i) => (
              <div key={i} className={`${line.startsWith("[error]") ? "text-red-400" : line.startsWith("[ok]") ? "text-emerald-400" : line.startsWith("[stage]") ? "text-cyan-300" : "theme-text/60"}`}>
                {line}
              </div>
            ))}
            {qdrantOnlyError && (
              <div className="text-red-400"><span className="font-black uppercase">Error: </span>{qdrantOnlyError}</div>
            )}
            {qdrantOnlyLoading && <span className="animate-pulse text-blue-300/40">▊</span>}
          </div>
        )}
      </div>


      {embedders.length === 0 ? (
        <div className="flex flex-col items-center justify-center gap-4 py-16 border-2 border-dashed border-white/10 rounded-2xl">
          <Sparkles className="w-10 h-10 theme-muted opacity-20" />
          <p className="text-sm font-mono theme-muted opacity-50 text-center max-w-xs leading-relaxed">
            No embedding models configured. Add one or more to start ingesting knowledge.
          </p>
          <button
            onClick={addEmbedder}
            className="flex items-center gap-2 px-6 py-3 theme-accent-bg text-black text-[11px] uppercase tracking-widest font-black rounded-xl hover:brightness-125 shadow-lg shadow-theme-accent/10 transition-all"
          >
            <Plus className="w-4 h-4" /> Add First Embedder
          </button>
        </div>
      ) : (
        <>
          <div className="flex items-center justify-between">
            <p className="text-[10px] uppercase tracking-widest theme-muted font-black font-mono opacity-50">
              {embedders.length} Embedder{embedders.length !== 1 ? "s" : ""} · {totalPoints.toLocaleString()} Total Points
            </p>
            <button
              onClick={addEmbedder}
              className="flex items-center gap-2 px-4 py-2 bg-white/5 border border-white/10 rounded-xl text-[10px] uppercase tracking-widest font-black font-mono theme-text hover:border-theme-accent/30 transition-all"
            >
              <Plus className="w-3.5 h-3.5" /> Add Embedder
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
            {embedders.map((embedder, idx) => (
              <EmbedderCard
                key={idx}
                embedder={embedder}
                idx={idx}
                config={config}
                gpuStatus={gpuStatus}
                onUpdate={patch => updateEmbedder(idx, patch)}
                onRemove={() => removeEmbedder(idx)}
                onIngestComplete={() => {
                  const coll = embedder.collection || `kb_${embedder.name.toLowerCase().replace(/\s+/g, "_").replace(/[^a-z0-9_]/g, "")}`;
                  const cfg = { ...config.qdrant, collection: coll };
                  api.qdrantCount(cfg).then(c => setEmbedderCounts(prev => ({ ...prev, [idx]: c }))).catch(() => {});
                }}
                showRemove={true}
              />
            ))}
          </div>
        </>
      )}

      {error && (
        <div className="p-4 rounded-xl border border-red-500/30 bg-red-500/5 text-red-300 text-sm font-mono animate-premium shadow-lg">
          <div className="flex items-center gap-2 mb-1">
            <div className="w-1.5 h-1.5 rounded-full bg-red-400 animate-pulse" />
            <span className="font-black uppercase tracking-widest text-[10px]">Probe Failure</span>
          </div>
          {error}
        </div>
      )}

      {embedders.length > 0 && <QdrantDbPanel config={config} onConfigChange={onConfigChange} />}

      {samples.length > 0 && (
        <div className="space-y-4">
          <div className="flex items-center gap-2 ml-1">
            <div className="w-1 h-3 theme-accent-bg rounded-full opacity-40" />
            <p className="text-[10px] uppercase tracking-widest theme-muted font-black font-mono">Sample Chunks</p>
          </div>
          <div className="grid grid-cols-1 gap-3">
            {samples.map(c => (
              <div key={c.id} className="premium-card rounded-2xl p-5 text-sm theme-text/80 group">
                <p className="text-[9px] font-black font-mono theme-accent uppercase tracking-widest mb-2 flex items-center gap-2">
                  <FileText className="w-3.5 h-3.5 opacity-40" /> {c.file_name || c.file_path || c.id}
                </p>
                <p className="line-clamp-3 leading-relaxed font-medium">{c.text}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── step 1 ────────────────────────────────────────────────────────────────

function TeacherStep({
  value,
  onChange,
  gpuStatus,
  hfToken,
  checkingTeacher,
  teacherDeployed,
  deployedTeacherModel,
  deploying,
  deployLogs,
  deployError,
  onCheckStatus,
  onDeploy,
  onCancelDeploy,
}: {
  value: TeacherConfig;
  onChange: (t: TeacherConfig) => void;
  gpuStatus?: GPUState | null;
  hfToken: string;
  checkingTeacher: boolean;
  teacherDeployed: boolean;
  deployedTeacherModel: string | null;
  deploying: boolean;
  deployLogs: string;
  deployError: string | null;
  onCheckStatus: () => void;
  onDeploy: () => void;
  onCancelDeploy: () => void;
}) {
  const set = <K extends keyof TeacherConfig>(k: K, v: TeacherConfig[K]) => {
    const next = { ...value, [k]: v };
    onChange(autoTuneTeacherConfig(next, gpuStatus));
  };

  useEffect(() => {
    const tuned = autoTuneTeacherConfig(value, gpuStatus);
    if (!teacherConfigEquals(tuned, value)) onChange(tuned);
  }, [value.repoId, value.autoTune, value.customServeCmd, gpuStatus?.memoryTotal]);

  const autoTuneActive = !!value.autoTune && !(value.customServeCmd || "").trim();
  const managedDisabled = !!value.customServeCmd || autoTuneActive;
  const canDeployTeacher = !!((value.customServeCmd || "").trim() || (value.repoId || "").trim());

  const deployLogRef = useRef<HTMLDivElement>(null);
  const stickToBottomRef = useRef(true);

  useEffect(() => {
    const el = deployLogRef.current;
    if (!el) return;
    if (stickToBottomRef.current) { el.scrollTop = el.scrollHeight; }
  }, [deployLogs, deployError]);

  const onDeployLogScroll = () => {
    const el = deployLogRef.current;
    if (!el) return;
    stickToBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
  };

  useEffect(() => { if (deploying && !deployLogs) stickToBottomRef.current = true; }, [deploying, deployLogs]);

  const coloredDeployLogs = useMemo(() => {
    const text = deployLogs || "";
    if (!text && !deployError) return null;
    return text.split("\n").map((line, i) => {
      let cls = "theme-text/70";
      if (line.startsWith("[ok]")) cls = "text-emerald-400 font-semibold";
      else if (line.startsWith("[stage]")) cls = "theme-accent font-semibold";
      else if (line.startsWith("[cmd]")) cls = "text-cyan-400";
      else if (line.startsWith("[FATAL]") || line.startsWith("[error]") || line.startsWith("DEPLOYMENT ERROR")) cls = "text-red-400 font-bold";
      else if (line.startsWith("[warn]")) cls = "text-amber-400";
      else if (line.startsWith("[DOCKER")) cls = "text-purple-300";
      else if (line.startsWith("[waiting]")) cls = "theme-faint italic";
      else if (line.startsWith("[poll-err]")) cls = "text-red-400";
      else if (line.startsWith("Deployment cancelled")) cls = "text-amber-300";
      else if (line.includes("Uvicorn running") || line.includes("Started server") || line.includes("Application startup complete")) cls = "text-emerald-400 font-semibold";
      else if (line.includes("INFO") && (line.includes("vllm") || line.includes("uvicorn") || line.includes("engine"))) cls = "text-emerald-300/80";
      else if (line.includes("WARNING") || line.includes("UserWarning")) cls = "text-amber-400/80";
      else if (line.includes("Traceback") || line.includes("raise ") || (line.includes("ERROR") && !line.startsWith("[ok]"))) cls = "text-red-400";
      else if (line.includes("Loading") || line.includes("Downloading") || line.includes("Fetching") || line.includes("Profiling") || line.includes("graph captured") || line.includes("model weights")) cls = "text-blue-300";
      return <span key={i} className={`block ${cls} break-all tracking-tight`}>{line || "\u00a0"}</span>;
    });
  }, [deployLogs, deployError]);

  return (
    <div className="space-y-8 animate-premium">
      <div className="flex flex-col gap-1">
        <h3 className="text-base-fluid uppercase tracking-[0.25em] theme-accent font-black italic font-serif">
          Teacher Configuration
        </h3>
        <p className="text-sm-fluid theme-muted font-medium opacity-80 leading-relaxed max-w-2xl">
          The Teacher model orchestrates the dataset generation. Deploy a high-parameter model locally.
        </p>
      </div>

      <div className="space-y-6 animate-premium">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-2.5">
            <div className="flex items-center justify-between ml-1">
              <label className="text-[10px] uppercase tracking-widest theme-muted font-black">Hugging Face Repository ID</label>
              {value.customServeCmd && (
                <span className="text-[8px] theme-accent font-black uppercase tracking-[0.2em] bg-theme-accent-soft px-3 py-1 rounded-full border border-theme-accent/30 shadow-sm animate-premium">Custom Binary Sequence Active</span>
              )}
            </div>
            <div className="relative">
              <select
                value={PRESET_TEACHER_MODELS.some((m) => m.value === value.repoId) ? value.repoId : "custom"}
                onChange={(e) => {
                  const val = e.target.value;
                  if (val === "custom") {
                    set("repoId", "");
                  } else {
                    set("repoId", val);
                  }
                }}
                disabled={!!value.customServeCmd}
                className={`w-full px-4 py-3.5 premium-input rounded-xl text-sm-fluid font-black font-mono text-white focus:outline-none shadow-inner appearance-none cursor-pointer transition-all ${
                  value.customServeCmd ? "opacity-20 grayscale scale-[0.98]" : ""
                }`}
              >
                {PRESET_TEACHER_MODELS.map((m) => (
                  <option key={m.value} className={OPTION_CLASS} value={m.value}>
                    {m.label} ({m.value.split("/").pop()})
                  </option>
                ))}
                <option className={OPTION_CLASS} value="custom">Custom Hugging Face Repo ID...</option>
              </select>
              <ChevronRight className="absolute right-4 top-1/2 -translate-y-1/2 w-4 h-4 rotate-90 pointer-events-none theme-faint opacity-50" />
            </div>
          </div>
          <div className="space-y-2.5">
            <label className="text-[10px] uppercase tracking-widest theme-muted font-black ml-1">Serving Engine</label>
            <div className="relative">
              <select
                value="vllm"
                onChange={() => set("servingEngine", "vllm")}
                disabled
                className="w-full px-4 py-3.5 premium-input rounded-xl text-sm-fluid font-black font-mono text-white focus:outline-none shadow-inner appearance-none opacity-80"
              >
                <option className={OPTION_CLASS} value="vllm">vLLM (Default)</option>
              </select>
              <ChevronRight className="absolute right-4 top-1/2 -translate-y-1/2 w-4 h-4 rotate-90 pointer-events-none theme-faint opacity-50" />
            </div>
          </div>
        </div>
        {(!PRESET_TEACHER_MODELS.some((m) => m.value === value.repoId) || value.repoId === "") && (
          <div className="space-y-2">
            <input
              type="text"
              value={value.repoId}
              onChange={(e) => set("repoId", e.target.value)}
              disabled={!!value.customServeCmd}
              placeholder="e.g. deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
              className="w-full px-4 py-3.5 premium-input rounded-xl text-sm-fluid font-mono text-white focus:outline-none shadow-inner transition-all animate-premium"
            />
            {!teacherDeployed && !(value.repoId || "").trim() && !value.customServeCmd?.trim() && (
              <p className="text-[10px] text-amber-300 font-mono uppercase tracking-tight ml-1">
                No live teacher detected. Enter a Hugging Face model id before deploying.
              </p>
            )}
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-[1fr_auto] gap-3 items-center rounded-xl border border-white/5 bg-white/[0.015] px-4 py-3">
          <div className="min-w-0">
            <div className="text-[10px] uppercase tracking-widest theme-muted font-black">Adaptive ROCm Serving Profile</div>
            <div className="mt-1 text-[9px] theme-faint font-mono uppercase tracking-tight">
              {autoTuneActive
                ? `Model and GPU tuned: ${value.dtype}, ${value.maxModelLen} ctx, ${value.maxNumBatchedTokens || 0} batched tokens, ${value.maxNumSeqs || 0} seqs`
                : "Manual vLLM parameters are active"}
            </div>
          </div>
          <label className="flex items-center gap-3 cursor-pointer select-none justify-end">
            <input
              type="checkbox"
              checked={!!value.autoTune}
              onChange={(e) => set("autoTune", e.target.checked)}
              disabled={!!value.customServeCmd}
              className="h-4 w-4 accent-[rgb(var(--app-accent-rgb))]"
            />
            <span className="text-[10px] uppercase tracking-widest theme-accent font-black">Auto Tune</span>
          </label>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div className="space-y-2">
            <label className="text-[9px] uppercase tracking-[0.15em] theme-muted font-black ml-1">TCP Port</label>
            <input type="number" value={value.vllmPort} onChange={(e) => set("vllmPort", Number(e.target.value))} disabled={!!value.customServeCmd} className="w-full px-4 py-3 premium-input rounded-xl text-[11px] font-black font-mono focus:outline-none" />
          </div>
          <div className="space-y-2">
            <label className="text-[9px] uppercase tracking-[0.15em] theme-muted font-black ml-1">Context</label>
            <input type="number" value={value.maxModelLen} onChange={(e) => set("maxModelLen", Number(e.target.value))} disabled={managedDisabled} className="w-full px-4 py-3 premium-input rounded-xl text-[11px] font-black font-mono focus:outline-none" />
          </div>
          <div className="space-y-2">
            <label className="text-[9px] uppercase tracking-[0.15em] theme-muted font-black ml-1">Precision</label>
            <div className="relative">
               <select value={value.dtype} onChange={(e) => set("dtype", e.target.value)} disabled={managedDisabled} className="w-full px-4 py-3 premium-input rounded-xl text-[11px] font-black font-mono focus:outline-none appearance-none cursor-pointer">
                  <option className={OPTION_CLASS} value="bfloat16">BF16</option>
                  <option className={OPTION_CLASS} value="float16">FP16</option>
                  <option className={OPTION_CLASS} value="auto">AUTO</option>
                </select>
                <ChevronRight className="absolute right-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 rotate-90 pointer-events-none theme-faint" />
            </div>
          </div>
          <div className="space-y-2">
            <label className="text-[9px] uppercase tracking-[0.15em] theme-muted font-black ml-1">Parallel</label>
            <input type="number" value={value.tensorParallel} onChange={(e) => set("tensorParallel", Number(e.target.value))} disabled={managedDisabled} className="w-full px-4 py-3 premium-input rounded-xl text-[11px] font-black font-mono focus:outline-none" />
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-5 gap-4">
          <div className="space-y-2">
            <label className="text-[9px] uppercase tracking-[0.15em] theme-muted font-black ml-1">VRAM Util</label>
            <input type="number" step="0.01" min="0.1" max="1" value={value.gpuMemoryUtilization ?? 0.80} onChange={(e) => set("gpuMemoryUtilization", Number(e.target.value))} disabled={managedDisabled} className="w-full px-4 py-3 premium-input rounded-xl text-[11px] font-black font-mono focus:outline-none" />
          </div>
          <div className="space-y-2">
            <label className="text-[9px] uppercase tracking-[0.15em] theme-muted font-black ml-1">Batch Tokens</label>
            <input type="number" value={value.maxNumBatchedTokens ?? 8192} onChange={(e) => set("maxNumBatchedTokens", Number(e.target.value))} disabled={managedDisabled} className="w-full px-4 py-3 premium-input rounded-xl text-[11px] font-black font-mono focus:outline-none" />
          </div>
          <div className="space-y-2">
            <label className="text-[9px] uppercase tracking-[0.15em] theme-muted font-black ml-1">Max Seqs</label>
            <input type="number" value={value.maxNumSeqs ?? 16} onChange={(e) => set("maxNumSeqs", Number(e.target.value))} disabled={managedDisabled} className="w-full px-4 py-3 premium-input rounded-xl text-[11px] font-black font-mono focus:outline-none" />
          </div>
          <label className="space-y-2 cursor-pointer select-none">
            <span className="block text-[9px] uppercase tracking-[0.15em] theme-muted font-black ml-1">Chunked</span>
            <span className="flex items-center gap-3 h-[42px] px-4 premium-input rounded-xl text-[10px] font-black font-mono">
              <input type="checkbox" checked={!!value.enableChunkedPrefill} onChange={(e) => set("enableChunkedPrefill", e.target.checked)} disabled={managedDisabled} className="h-4 w-4 accent-[rgb(var(--app-accent-rgb))]" />
              PREFILL
            </span>
          </label>
          <label className="space-y-2 cursor-pointer select-none">
            <span className="block text-[9px] uppercase tracking-[0.15em] theme-muted font-black ml-1">Tools</span>
            <span className="flex items-center gap-3 h-[42px] px-4 premium-input rounded-xl text-[10px] font-black font-mono">
              <input type="checkbox" checked={!!value.enableAutoToolChoice} onChange={(e) => set("enableAutoToolChoice", e.target.checked)} disabled={managedDisabled} className="h-4 w-4 accent-[rgb(var(--app-accent-rgb))]" />
              AUTO
            </span>
          </label>
        </div>

        {(value.enableAutoToolChoice || value.toolCallParser) && (
          <div className="space-y-2">
            <label className="text-[9px] uppercase tracking-[0.15em] theme-muted font-black ml-1">Tool Call Parser</label>
            <input
              type="text"
              value={value.toolCallParser || ""}
              onChange={(e) => set("toolCallParser", e.target.value)}
              disabled={managedDisabled}
              placeholder="qwen3_coder"
              className="w-full px-4 py-3 premium-input rounded-xl text-[11px] font-black font-mono focus:outline-none"
            />
          </div>
        )}

        <div className="space-y-3 pt-2">
           <div className="flex items-center justify-between ml-1">
            <label className="text-[10px] uppercase tracking-widest theme-muted font-black">Execution Binary Sequence <span className="opacity-40 font-mono tracking-normal text-[8px]">(OVERRIDE)</span></label>
            {value.customServeCmd && (
              <button onClick={() => set("customServeCmd", "")} className="text-[9px] theme-accent font-black hover:underline uppercase tracking-[0.15em]">Reset Context</button>
            )}
          </div>
          <textarea
            rows={4}
            value={value.customServeCmd || ""}
            onChange={(e) => set("customServeCmd", e.target.value)}
            placeholder="Inject raw vLLM startup sequence to bypass standard parameters..."
            className="w-full px-5 py-5 premium-input rounded-2xl text-[11px] font-mono text-white focus:outline-none shadow-xl resize-none leading-relaxed tracking-tight bg-black/20"
          />
        </div>

        <div className="premium-card rounded-2xl overflow-hidden glass-panel border border-white/5 shadow-2xl">
          <div className="px-8 py-5 border-b border-white/5 bg-white/[0.01] flex items-center justify-between">
             <div className="flex items-center gap-4">
              <div className={`w-3 h-3 rounded-full shadow-[0_0_10px_currentColor] transition-all duration-500 ${
                checkingTeacher ? "bg-amber-500 animate-pulse text-amber-400" : teacherDeployed ? "bg-emerald-500 text-emerald-400" : "bg-red-500 text-red-400"
              }`} />
              <div className="flex flex-col">
                <span className="text-[10px] uppercase tracking-[0.3em] font-black font-mono leading-none">NODE STATUS: {checkingTeacher ? "HANDSHAKING" : teacherDeployed ? "UPLINK_ESTABLISHED" : "CONTEXT_IDLE"}</span>
                <span className="text-[8px] theme-faint font-mono uppercase mt-1">
                  {teacherDeployed && deployedTeacherModel
                    ? `Serving: ${deployedTeacherModel}`
                    : teacherDeployed
                      ? "Live teacher endpoint detected"
                      : "No live teacher detected; deploy a Hugging Face model"}
                </span>
              </div>
            </div>
            <div className="flex gap-3">
              <button
                type="button"
                disabled={checkingTeacher || teacherDeployed}
                onClick={onCheckStatus}
                className="px-5 py-2 rounded-xl border border-white/10 theme-surface-soft theme-muted hover:theme-text hover:border-theme-accent/30 text-[10px] font-black uppercase tracking-widest transition-all shadow-sm"
              >
                Verify
              </button>
              {deploying ? (
                <button
                  type="button"
                  onClick={onCancelDeploy}
                  className="px-5 py-2 bg-red-500/10 border border-red-500/20 text-red-400 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all hover:bg-red-500 hover:text-white shadow-lg shadow-red-500/10"
                >
                  Abort
                </button>
              ) : (
                <button
                  type="button"
                  onClick={onDeploy}
                  disabled={!canDeployTeacher}
                  className="px-6 py-2 theme-accent-bg text-black rounded-xl text-[10px] font-black uppercase tracking-widest transition-all hover:brightness-125 disabled:opacity-20 shadow-xl shadow-theme-accent/20 premium-button"
                >
                  Deploy Teacher
                </button>
              )}
            </div>
          </div>

          {(deploying || deployLogs || deployError) && (
            <div className="animate-premium">
              <div ref={deployLogRef} onScroll={onDeployLogScroll} className="bg-black/60 p-6 h-72 overflow-y-auto font-mono text-[11px] leading-relaxed selection:bg-theme-selection scrollbar-thin scrollbar-thumb-white/10">
                {coloredDeployLogs ?? <span className="theme-faint italic opacity-50">Synchronizing secure buffers...</span>}
                {deployError && (
                  <div className="mt-5 p-4 rounded-xl border border-red-500/30 bg-red-500/5 text-red-400 font-bold animate-premium uppercase text-[10px] tracking-widest shadow-lg">
                    ✕ UPLINK INTERRUPTED: {deployError}
                  </div>
                )}
                {deploying && <span className="inline-block w-2.5 h-4.5 theme-accent-bg ml-2 animate-pulse align-middle shadow-[0_0_10px_currentColor]" />}
              </div>
              <div className="px-8 py-2 border-t border-white/5 bg-white/[0.01]">
                 <p className="text-[9px] theme-faint font-mono uppercase tracking-[0.3em] text-center opacity-40">Direct Telemetry Handover Sequence</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── step 3 ────────────────────────────────────────────────────────────────

function DatasetStep(props: {
  config: AppConfig;
  trainingOnly: boolean;
  onSwitchToGenerateDataset: () => void;
  topics: TopicTarget[];
  onTopicsChange: (t: TopicTarget[]) => void;
  prompt: string;
  onPromptChange: (s: string) => void;
  maxPairsPerChunk: number;
  onMaxPairsChange: (n: number) => void;
  concurrency: number;
  onConcurrencyChange: (n: number) => void;
  maxChunks: number | "all";
  onMaxChunksChange: (n: number | "all") => void;
  hubDataset: HubDatasetConfig;
  onHubDatasetChange: (h: HubDatasetConfig) => void;
  hfTokenSet: boolean;
  hfUsername: string | null;
  hfDatasets: HfDatasetRepo[];
  hfLoading: boolean;
  hfError: string | null;
  onRefreshHf: () => void;
  generating: boolean;
  generated: boolean;
  progress: { scanned: number; kept: number; rejected: number; status: string } | null;
  logs: string;
  error: string | null;
  onGenerate: () => void;
  onCancel: () => void;
  sshHostSet: boolean;
  onConfigChange?: (patch: Partial<AppConfig>) => void;
  method?: string;
  enableVerification: boolean;
  onEnableVerificationChange: (v: boolean) => void;
  bundleWindow: number;
  onBundleWindowChange: (n: number) => void;
  datasetFormat: string;
  onDatasetFormatChange: (s: string) => void;
}) {
  const isZraldOffline = props.method === "zrald_offline";
  const embedders = props.config.embedders && props.config.embedders.length > 0 ? props.config.embedders : [DEFAULT_EMBEDDER];
  const collections = useMemo(() => {
    const cols = ["all"];
    for (const emb of embedders) {
      const c = emb.collection || `kb_${emb.name.toLowerCase().replace(/\s+/g, "_").replace(/[^a-z0-9_]/g, "")}`;
      if (!cols.includes(c)) cols.push(c);
    }
    if (props.config.qdrant.collection && !cols.includes(props.config.qdrant.collection)) {
      cols.unshift(props.config.qdrant.collection);
    }
    return cols;
  }, [embedders, props.config.qdrant.collection]);

  const [showCustomCollection, setShowCustomCollection] = useState(() => {
    const presets = ["all"];
    for (const emb of embedders) {
      const c = emb.collection || `kb_${emb.name.toLowerCase().replace(/\s+/g, "_").replace(/[^a-z0-9_]/g, "")}`;
      if (!presets.includes(c)) presets.push(c);
    }
    return props.config.qdrant.collection ? !presets.includes(props.config.qdrant.collection) : false;
  });

  const handleCollectionChange = (value: string) => {
    if (props.onConfigChange) {
      props.onConfigChange({
        qdrant: {
          ...props.config.qdrant,
          collection: value,
        },
      });
    }
  };

  const hd = props.hubDataset;
  const setHd = <K extends keyof HubDatasetConfig>(k: K, v: HubDatasetConfig[K]) => props.onHubDatasetChange({ ...hd, [k]: v });
  const selectedFormat = DATASET_FORMATS.find((f) => f.key === props.datasetFormat) || DATASET_FORMATS.find((f) => f.key === DEFAULT_DATASET_FORMAT) || DATASET_FORMATS[0];

  const applyDatasetFormat = (format: DatasetFormatKey) => {
    const nextPrompt = promptForDatasetFormat(format);
    const previousPrompt = promptForDatasetFormat(props.datasetFormat);
    props.onDatasetFormatChange(format);
    props.onPromptChange(nextPrompt);
    props.onTopicsChange(
      props.topics.map((topic) => {
        const currentPrompt = topic.promptTemplate || "";
        if (!currentPrompt.trim() || currentPrompt === props.prompt || currentPrompt === previousPrompt) {
          return { ...topic, promptTemplate: nextPrompt };
        }
        return topic;
      })
    );
  };

  const selectedDatasets = useMemo<string[]>(() => {
    const list = (hd.repoIds && hd.repoIds.length > 0) ? hd.repoIds : (hd.repoId ? [hd.repoId] : []);
    const seen = new Set<string>();
    const out: string[] = [];
    for (const r of list) { const t = (r || "").trim(); if (t && !seen.has(t)) { seen.add(t); out.push(t); } }
    return out;
  }, [hd.repoIds, hd.repoId]);

  const totalTarget = props.topics.reduce((acc, t) => acc + (t.totalQuestions && t.totalQuestions > 0 ? t.totalQuestions : 0), 0);
  const anyUncapped = props.topics.some((t) => t.topic.trim().length > 0 && (!t.totalQuestions || t.totalQuestions <= 0));

  const setRow = (idx: number, patch: Partial<TopicTarget>) => {
    const next = props.topics.slice();
    next[idx] = { ...next[idx], ...patch };
    props.onTopicsChange(next);
  };
  const addRow = () => props.onTopicsChange([...props.topics, { topic: "", totalQuestions: undefined, promptTemplate: selectedFormat.prompt }]);
  const removeRow = (idx: number) => {
    if (props.topics.length <= 1) { props.onTopicsChange([{ topic: "", totalQuestions: undefined, promptTemplate: selectedFormat.prompt }]); return; }
    props.onTopicsChange(props.topics.filter((_, i) => i !== idx));
    setGeneratingTopicIdx((cur) => (cur === idx ? null : cur));
  };

  const [generatingTopicIdx, setGeneratingTopicIdx] = useState<number | null>(null);

  const generateTopicPromptWithAI = async (idx: number) => {
    const topicText = (props.topics[idx]?.topic || "").trim();
    if (!topicText) {
      alert("Please enter a topic name for this module first.");
      return;
    }
    // Use the AI agent configured in the Terminal/Credentials panel.
    // Whichever provider the user picked for the chat terminal (Anthropic, OpenAI, Gemini, custom, …).
    const agentConfig = props.config.aiAgent;
    if (!agentConfig || !agentConfig.apiKey || !agentConfig.apiUrl || !agentConfig.modelId) {
      alert(
        "No AI agent configured. Open the AI Terminal (or the Credentials tab) and set the provider, API URL, model ID, and API key — those credentials are then reused here."
      );
      return;
    }
    setGeneratingTopicIdx(idx);
    try {
      const provider = agentConfig.provider;
      const apiUrl = (agentConfig.apiUrl || "").trim();
      const apiKey = agentConfig.apiKey;
      const modelId = (agentConfig.modelId || "").trim();
      let endpoint = apiUrl;
      const headers: Record<string, string> = { "Content-Type": "application/json" };
      if (provider === "anthropic") {
        if (!endpoint.includes("/messages")) {
          const base = endpoint.replace(/\/+$/, "");
          endpoint = base.endsWith("/v1") ? `${base}/messages` : `${base}/v1/messages`;
        }
        headers["x-api-key"] = apiKey;
        headers["anthropic-version"] = "2023-06-01";
        headers["dangerously-allow-browser"] = "true";
      } else {
        if (!endpoint.includes("/chat/completions")) {
          const base = endpoint.replace(/\/+$/, "");
          endpoint = `${base}/chat/completions`;
        }
        headers["Authorization"] = `Bearer ${apiKey}`;
      }
      const userPrompt = `You are a system prompt engineering expert. Generate an engineering directive prompt for an LLM teacher model that will read source material chunks and generate ${selectedFormat.label} dataset rows focused exclusively on the topic: "${topicText}".

The template prompt MUST be written as an instruction system prompt for the teacher LLM, and it MUST contain the following exact placeholders:
- {topic} : representing the current focus topic (will resolve to "${topicText}").
- {chunk_text} : representing the raw text chunk to read.

Tailor the wording, examples, and rules specifically to "${topicText}" — assume the teacher will only ever generate questions about this single topic.

Use this selected dataset format as the structural baseline:
${selectedFormat.prompt}

Provide ONLY the final generated instruction system prompt text. Do not include markdown code fence formatting (like \`\`\`), <think> blocks, analysis, conversational intros, or explanations.`;
      const bodyData: any = provider === "anthropic"
        ? { model: modelId, messages: [{ role: "user", content: userPrompt }], max_tokens: 2048, temperature: 0.3 }
        : { model: modelId, messages: [{ role: "user", content: userPrompt }], temperature: 0.3 };
      const response = await fetch(endpoint, { method: "POST", headers, body: JSON.stringify(bodyData) });
      if (!response.ok) throw new Error(`API error: ${response.status} ${response.statusText}`);
      const data = await response.json();
      const answer = provider === "anthropic"
        ? (data.content?.[0]?.text || "")
        : (data.choices?.[0]?.message?.content || "");
      const cleaned = stripModelThinking(answer);
      if (cleaned) {
        setRow(idx, { promptTemplate: cleaned });
      } else {
        alert("Received empty prompt from the AI Agent.");
      }
    } catch (e: any) {
      alert(`Prompt Generation Failed: ${e.message || String(e)}`);
    } finally {
      setGeneratingTopicIdx(null);
    }
  };

  const datasetRepoPlaceholder = props.hfUsername ? `${props.hfUsername}/ge-reviewer-qa` : "your-account/repository-id";

  return (
    <div className="space-y-8 animate-premium">
       <div className="flex flex-col gap-1">
        <h3 className="text-base-fluid uppercase tracking-[0.25em] theme-accent font-black italic font-serif">
          {props.trainingOnly ? "Data Consolidation" : "Synthetic Engineering"}
        </h3>
        <p className="text-sm-fluid theme-muted font-medium opacity-80 leading-relaxed max-w-2xl">
          {props.trainingOnly ? <>Interleave high-fidelity datasets from the Hub to establish the training objective. Multi-source sampling enhances robustness.</> : <>Configure the prompt architecture and domain focus. Each fragment of knowledge is transformed into a unique synthetic sample context.</>}
        </p>
      </div>

      {props.trainingOnly && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
          <button onClick={props.onSwitchToGenerateDataset} className="text-left rounded-2xl border border-white/5 bg-white/[0.01] p-6 hover:bg-white/[0.03] hover:border-theme-accent/30 transition-all duration-500 premium-button group shadow-lg">
            <div className="flex items-center gap-4 text-[10px] uppercase tracking-[0.25em] font-black font-mono theme-text/80 group-hover:theme-accent transition-colors">
              <div className="p-2 rounded-xl bg-white/5 group-hover:bg-theme-accent group-hover:text-black transition-all shadow-inner"><Sparkles className="w-5 h-5" /></div>
              Pivotal Generation
            </div>
            <p className="mt-4 text-sm-fluid theme-muted opacity-60 leading-relaxed font-medium">Switch to the RAG Teacher workflow to engineer custom data before direct training.</p>
          </button>
          <div className="rounded-2xl border theme-accent-soft p-6 shadow-xl shadow-theme-accent/5 relative overflow-hidden glass-panel group">
             <div className="absolute top-0 left-0 w-1 h-full theme-accent-bg shadow-[0_0_15px_currentColor]" />
             <div className="flex items-center gap-4 text-[10px] uppercase tracking-[0.25em] font-black font-mono theme-accent">
                <div className="p-2 rounded-xl bg-theme-accent text-black shadow-lg"><Database className="w-5 h-5" /></div>
                Direct Dataset Context
             </div>
             <p className="mt-4 text-sm-fluid theme-muted opacity-90 leading-relaxed font-medium">Point to existing Hub repositories. Training will proceed without synthetic generation cycles.</p>
          </div>
        </div>
      )}

      {!props.trainingOnly && (
        <div className="space-y-4">
          <div className="flex items-center justify-between ml-1">
            <label className="text-[10px] uppercase tracking-[0.3em] theme-muted font-black font-mono">Dataset Format</label>
            <span className="text-[10px] font-black font-mono theme-accent uppercase tracking-widest">{selectedFormat.label}</span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {DATASET_FORMATS.map((format) => {
              const Icon = format.icon;
              const active = format.key === props.datasetFormat;
              return (
                <button
                  key={format.key}
                  type="button"
                  onClick={() => applyDatasetFormat(format.key)}
                  className={`text-left rounded-xl border p-4 transition-all duration-300 premium-button ${
                    active
                      ? "theme-accent-soft theme-accent border-theme-accent/40 shadow-lg shadow-theme-accent/5"
                      : "border-white/5 bg-white/[0.015] theme-muted hover:theme-text hover:border-white/15"
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${active ? "theme-accent-bg text-black" : "bg-white/5 text-white/40"}`}>
                      <Icon className="w-4 h-4" />
                    </div>
                    <span className="text-[10px] uppercase tracking-[0.18em] font-black font-mono">{format.label}</span>
                  </div>
                  <p className="mt-2 text-[10px] theme-muted opacity-70 leading-relaxed">{format.desc}</p>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {!props.trainingOnly && <div className="space-y-4">
        <div className="flex items-center justify-between ml-1">
          <label className="text-[10px] uppercase tracking-[0.3em] theme-muted font-black font-mono">Domain Specialization Grid</label>
          <div className="flex items-center gap-5 bg-white/[0.02] px-5 py-1.5 rounded-full border border-white/5 shadow-inner">
            <span className="text-[10px] font-black font-mono theme-accent uppercase tracking-widest">{props.topics.length} MODULES</span>
            <span className="w-1 h-1 rounded-full bg-white/20" />
            <span className="text-[10px] font-black font-mono theme-faint uppercase tracking-widest">QUOTA: {totalTarget || "UNCAPPED"}</span>
          </div>
        </div>

        <div className="premium-card rounded-2xl overflow-hidden glass-panel border border-white/5 shadow-2xl">
          <div className="grid grid-cols-[1fr,160px,120px,50px] gap-5 text-[9px] uppercase tracking-[0.3em] theme-muted font-black font-mono bg-white/[0.03] px-8 py-4 border-b border-white/5">
            <div>Target Domain</div>
            <div>Context Filter</div>
            <div>Token Quota</div>
            <div />
          </div>

          <div className="p-5 space-y-6">
            {props.topics.map((row, idx) => {
              const promptValue = row.promptTemplate || "";
              const promptMissing = promptValue.trim().length === 0;
              const missingPlaceholders: string[] = [];
              if (!promptMissing) {
                if (!promptValue.includes("{topic}")) missingPlaceholders.push("{topic}");
                if (!promptValue.includes("{chunk_text}")) missingPlaceholders.push("{chunk_text}");
              }
              const isGenerating = generatingTopicIdx === idx;
              return (
                <div key={idx} className="flex flex-col gap-3 group/row animate-premium">
                  <div className="grid grid-cols-[1fr,160px,120px,50px] gap-4 items-center">
                    <input type="text" value={row.topic} onChange={(e) => setRow(idx, { topic: e.target.value })} placeholder="Inject domain focus..." className="w-full px-5 py-3 premium-input rounded-xl text-sm-fluid font-mono text-white focus:outline-none bg-black/20" />
                    <input type="text" value={row.tag || ""} onChange={(e) => setRow(idx, { tag: e.target.value || undefined })} placeholder="NO_TAG" className="w-full px-5 py-3 premium-input rounded-xl text-[10px] font-black font-mono text-white focus:outline-none text-center bg-black/20" />
                    <input type="number" min="1" value={row.totalQuestions || ""} onChange={(e) => { const v = e.target.value.trim(); setRow(idx, { totalQuestions: v ? parseInt(v, 10) : undefined }); }} placeholder="MAX" className="w-full px-5 py-3 premium-input rounded-xl text-sm-fluid font-mono text-white focus:outline-none text-center bg-black/20" />
                    <button type="button" onClick={() => removeRow(idx)} className="w-12 h-12 flex items-center justify-center rounded-xl bg-red-500/5 text-red-500/30 hover:bg-red-500 hover:text-white transition-all duration-300 opacity-0 group-hover/row:opacity-100 scale-90 group-hover/row:scale-100 shadow-sm"><X className="w-6 h-6" /></button>
                  </div>
                  <div className={`rounded-xl border p-4 space-y-3 shadow-inner ${promptMissing ? "border-red-500/30 bg-red-500/[0.04]" : missingPlaceholders.length > 0 ? "border-amber-500/30 bg-amber-500/[0.03]" : "border-theme-accent/20 bg-black/30"}`}>
                    <div className="flex items-center justify-between flex-wrap gap-2">
                      <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.25em] font-black font-mono theme-accent">
                        <FileText className="w-3.5 h-3.5" />
                        <span>Engineering Directive Prompt <span className="theme-faint normal-case tracking-normal opacity-60">— required for "{row.topic.trim() || "this topic"}"</span></span>
                      </div>
                      <div className="flex items-center gap-2">
                        <button
                          type="button"
                          onClick={() => generateTopicPromptWithAI(idx)}
                          disabled={isGenerating}
                          className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-theme-accent/20 bg-theme-accent-soft theme-accent text-[9px] uppercase tracking-widest font-black font-mono hover:brightness-110 active:scale-[0.98] transition-all disabled:opacity-50"
                        >
                          {isGenerating ? (
                            <><Loader2 className="w-3.5 h-3.5 animate-spin" /><span>Generating...</span></>
                          ) : (
                            <><Sparkles className="w-3.5 h-3.5 animate-pulse" /><span>Generate with AI</span></>
                          )}
                        </button>
                        {!promptMissing && (
                          <button
                            type="button"
                            onClick={() => setRow(idx, { promptTemplate: undefined })}
                            className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-red-500/20 bg-red-500/5 text-red-400 text-[9px] uppercase tracking-widest font-black font-mono hover:bg-red-500/10 active:scale-[0.98] transition-all"
                          >
                            <X className="w-3.5 h-3.5" />
                            <span>Clear</span>
                          </button>
                        )}
                      </div>
                    </div>
                    <textarea
                      value={promptValue}
                      onChange={(e) => setRow(idx, { promptTemplate: e.target.value })}
                      rows={10}
                      required
                      placeholder={`Required. Engineering directive used to generate Q&A pairs for "${row.topic.trim() || "this topic"}".\nMust contain {topic} and {chunk_text} placeholders.`}
                      className={`w-full px-4 py-3 premium-input rounded-lg text-xs font-mono text-white/90 leading-relaxed resize-y focus:outline-none bg-black/30 min-h-[180px] ${promptMissing ? "border-red-500/40" : "border-white/10"}`}
                    />
                    <div className="flex items-center justify-between flex-wrap gap-2">
                      <p className="text-[9px] theme-faint font-mono uppercase tracking-[0.2em] opacity-60 italic">
                        Placeholders: <span className="theme-accent">{"{topic}"}</span> · <span className="theme-accent">{"{chunk_text}"}</span>
                      </p>
                      {promptMissing && (
                        <p className="text-[9px] text-red-400 font-mono uppercase tracking-[0.2em] font-black">Required — launch is blocked until set</p>
                      )}
                      {!promptMissing && missingPlaceholders.length > 0 && (
                        <p className="text-[9px] text-amber-400 font-mono uppercase tracking-[0.2em] font-black">Missing placeholder{missingPlaceholders.length > 1 ? "s" : ""}: {missingPlaceholders.join(" · ")}</p>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="px-8 py-5 border-t border-white/5 bg-white/[0.01] flex items-center justify-between">
            <button onClick={addRow} className="flex items-center gap-3 px-6 py-2.5 rounded-xl theme-accent-soft theme-accent text-[11px] font-black font-mono transition-all hover:brightness-125 shadow-lg border border-theme-accent/20 uppercase tracking-[0.15em] premium-button">
              <Plus className="w-5 h-5" /> Append Domain Module
            </button>
            <p className="text-[10px] theme-faint font-mono uppercase tracking-[0.2em] italic opacity-40">Sequential generation pass sequence</p>
          </div>
        </div>
      </div>}

      {!props.trainingOnly && (
        <div className="space-y-6 animate-premium">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div className="space-y-2.5">
              <label className="text-[10px] uppercase tracking-[0.2em] theme-accent font-black ml-1 font-mono tracking-widest">Source Collection</label>
              <div className="relative">
                <select
                  value={props.config.qdrant.collection || "all"}
                  onChange={(e) => {
                    if (e.target.value === "__custom__") {
                      setShowCustomCollection(true);
                    } else {
                      setShowCustomCollection(false);
                      handleCollectionChange(e.target.value);
                    }
                  }}
                  className="w-full px-5 py-3.5 premium-input rounded-xl text-xs font-mono text-white focus:outline-none appearance-none cursor-pointer bg-black/40 border border-white/10"
                >
                  {collections.map(c => (
                    <option key={c} value={c} className="theme-surface theme-text">{c}</option>
                  ))}
                  <option value="__custom__" className="theme-surface theme-text">Custom collection...</option>
                </select>
                <ChevronRight className="absolute right-4 top-1/2 -translate-y-1/2 w-4 h-4 rotate-90 pointer-events-none opacity-30" />
              </div>
            </div>
            <div className="space-y-2.5"><label className="text-[10px] uppercase tracking-widest theme-muted font-black ml-1 font-mono tracking-widest">Density</label><input type="number" min={1} value={props.maxPairsPerChunk} onChange={(e) => props.onMaxPairsChange(Math.max(1, Number(e.target.value)))} className="w-full px-5 py-3.5 premium-input rounded-xl text-sm-fluid font-mono focus:outline-none bg-black/40 border border-white/10" /></div>
            <div className="space-y-2.5"><label className="text-[10px] uppercase tracking-widest theme-muted font-black ml-1 font-mono tracking-widest">Concurrency</label><input type="number" min={1} value={props.concurrency} onChange={(e) => props.onConcurrencyChange(Math.max(1, Number(e.target.value)))} className="w-full px-5 py-3.5 premium-input rounded-xl text-sm-fluid font-mono focus:outline-none disabled:opacity-20 bg-black/40 border border-white/10" /></div>
            <div className="space-y-2.5"><label className="text-[10px] uppercase tracking-widest theme-muted font-black ml-1 font-mono tracking-widest">Cap</label><input type="text" value={props.maxChunks === "all" ? "all" : String(props.maxChunks)} onChange={(e) => { const v = e.target.value.trim().toLowerCase(); props.onMaxChunksChange((v === "" || v === "all") ? "all" : Number(v)); }} className="w-full px-5 py-3.5 premium-input rounded-xl text-sm-fluid font-mono focus:outline-none bg-black/40 border border-white/10" /></div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <label className={`flex items-center gap-4 rounded-xl border p-4 cursor-pointer transition-all ${props.enableVerification ? "theme-accent-soft theme-accent border-theme-accent/40" : "border-white/5 bg-white/[0.015] theme-muted hover:theme-text"}`}>
              <input
                type="checkbox"
                checked={props.enableVerification}
                onChange={(e) => props.onEnableVerificationChange(e.target.checked)}
                className="h-4 w-4 accent-[rgb(var(--app-accent-rgb))]"
              />
              <div>
                <div className="text-[10px] uppercase tracking-[0.2em] font-black font-mono">Teacher Verification</div>
                <p className="mt-1 text-[10px] opacity-70 leading-relaxed">Run a factuality judge pass before accepting each generated row.</p>
              </div>
            </label>
            <div className="rounded-xl border border-white/5 bg-white/[0.015] p-4 space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-[10px] uppercase tracking-[0.2em] theme-muted font-black font-mono">Bundle Window</label>
                <span className="text-[10px] font-black font-mono theme-accent">{props.bundleWindow}</span>
              </div>
              <input
                type="range"
                min={0}
                max={3}
                step={1}
                value={props.bundleWindow}
                onChange={(e) => props.onBundleWindowChange(Number(e.target.value))}
                className="w-full accent-[rgb(var(--app-accent-rgb))]"
              />
              <p className="text-[10px] theme-muted opacity-70 leading-relaxed">Include neighboring chunks from the same source document for richer context.</p>
            </div>
          </div>
          
          {showCustomCollection && (
            <div className="space-y-2.5 max-w-xs animate-premium">
              <label className="text-[10px] uppercase tracking-widest theme-muted font-black ml-1 font-mono tracking-widest">Custom Collection Name</label>
              <input
                type="text"
                value={props.config.qdrant.collection || "all"}
                onChange={(e) => handleCollectionChange(e.target.value)}
                placeholder="Enter Qdrant collection name"
                className="w-full px-5 py-3.5 premium-input rounded-xl text-sm-fluid font-mono focus:outline-none bg-black/40 border border-white/10"
              />
            </div>
          )}
        </div>
      )}

      {/* HF Dataset Uplink */}
      <div className="premium-card rounded-2xl p-8 glass-panel border border-white/5 space-y-8 relative group shadow-2xl">
        <div className="absolute top-0 left-0 w-1.5 h-full theme-accent-bg rounded-l-2xl opacity-30 group-hover:opacity-100 transition-opacity" />
        <div className="flex items-center justify-between">
          <div className="space-y-2">
            <p className="text-[11px] uppercase tracking-[0.3em] theme-accent font-black font-mono">Hugging Face Cloud Sink</p>
            <p className="text-sm-fluid theme-muted font-medium opacity-80 leading-relaxed max-w-2xl">
              {props.trainingOnly ? <>Define the upstream training source context. High-fidelity data is the bedrock of domain specialization.</> : <>Automatically commit synthetic telemetry to the Hub. Enables persistent datasets across distributed compute environments.</>}
            </p>
          </div>
          {!props.trainingOnly && (
            <label className={`shrink-0 inline-flex items-center gap-4 px-6 py-3 rounded-full border transition-all duration-500 cursor-pointer shadow-xl ${hd.enabled ? "theme-accent-soft theme-accent border-theme-accent/40" : "theme-surface-soft border-white/10 theme-muted"}`}>
               <span className={`w-2.5 h-2.5 rounded-full ${hd.enabled ? "theme-accent-bg shadow-[0_0_10px_currentColor] animate-pulse" : "bg-white/10"}`} />
               <span className="text-[11px] font-black uppercase tracking-[0.2em] font-mono">Uplink {hd.enabled ? "Online" : "Static"}</span>
               <input type="checkbox" checked={hd.enabled} onChange={(e) => setHd("enabled", e.target.checked)} className="hidden" />
            </label>
          )}
        </div>

        {(hd.enabled || props.trainingOnly) && (
          <div className="space-y-8 animate-premium">
            {!props.trainingOnly && <div className="space-y-3">
              <div className="flex items-center justify-between ml-1">
                <label className="text-[10px] uppercase tracking-widest theme-muted font-black font-mono opacity-60 tracking-[0.2em]">Target Repository Context</label>
                {props.hfUsername && <span className="text-[9px] font-black font-mono text-emerald-400 uppercase tracking-widest bg-emerald-500/10 px-3 py-1 rounded-full border border-emerald-500/20 shadow-sm">Authenticated // {props.hfUsername}</span>}
              </div>
              <div className="flex gap-4">
                <input type="text" value={hd.repoId} onChange={(e) => setHd("repoId", e.target.value)} placeholder={datasetRepoPlaceholder} className="flex-1 px-6 py-4 premium-input rounded-2xl text-sm-fluid font-mono text-white focus:outline-none shadow-2xl bg-black/40 border-white/10 tracking-tight" />
                {props.hfUsername && !hd.repoId && (
                  <button onClick={() => setHd("repoId", `${props.hfUsername}/ge-reviewer-qa`)} className="px-6 py-4 rounded-2xl border border-white/10 theme-surface-soft theme-text text-[11px] font-black font-mono transition-all hover:bg-white/5 hover:border-theme-accent/30 premium-button whitespace-nowrap shadow-lg">Map Profile Root</button>
                )}
              </div>
            </div>}
            
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 items-end">
              {!props.trainingOnly && <div className="space-y-3"><label className="text-[10px] uppercase tracking-widest theme-muted font-black ml-1 tracking-[0.2em] opacity-60">Sync Frequency <span className="opacity-30 text-[9px] font-mono">(TELEMETRY_PAIRS)</span></label><input type="number" min={0} value={hd.everyN} onChange={(e) => setHd("everyN", Math.max(0, Number(e.target.value)))} className="w-full px-6 py-4 premium-input rounded-2xl text-sm-fluid font-mono focus:outline-none bg-black/20" /></div>}
              <div className="flex gap-4 pb-1">
                <label className="flex-1 flex items-center justify-center gap-4 p-4 rounded-2xl border border-white/5 bg-white/[0.01] hover:bg-white/[0.03] transition-all cursor-pointer group shadow-xl"><div className={`w-5 h-5 rounded border-2 transition-all flex items-center justify-center ${hd.private ? "bg-theme-accent border-theme-accent" : "border-white/20 group-hover:border-white/40"}`}>{hd.private && <CheckCircle2 className="w-4 h-4 text-black" />}</div><span className="text-[10px] font-black font-mono theme-muted uppercase tracking-widest group-hover:theme-text">Private Visibility</span><input type="checkbox" checked={hd.private} onChange={(e) => setHd("private", e.target.checked)} className="hidden" /></label>
                <label className="flex-1 flex items-center justify-center gap-4 p-4 rounded-2xl border border-white/5 bg-white/[0.01] hover:bg-white/[0.03] transition-all cursor-pointer group shadow-xl"><div className={`w-5 h-5 rounded border-2 transition-all flex items-center justify-center ${hd.trainOnly ? "bg-theme-accent border-theme-accent" : "border-white/20 group-hover:border-white/40"}`}>{hd.trainOnly && <CheckCircle2 className="w-4 h-4 text-black" />}</div><span className="text-[10px] font-black font-mono theme-accent uppercase tracking-widest">Training Direct</span><input type="checkbox" checked={hd.trainOnly || false} onChange={(e) => setHd("trainOnly", e.target.checked)} className="hidden" /></label>
              </div>
            </div>

            <div className="pt-6 border-t border-white/5 animate-premium">
              {props.trainingOnly ? (
                <MultiDatasetPicker selected={selectedDatasets} onChange={(next) => { const cleaned = next.map((repo) => (repo || "").trim()).filter((repo, idx, all) => repo.length > 0 && all.indexOf(repo) === idx); props.onHubDatasetChange({ ...hd, enabled: true, trainOnly: true, repoId: cleaned[0] || "", repoIds: cleaned }); }} hfDatasets={props.hfDatasets} hfUsername={props.hfUsername} hfTokenSet={props.hfTokenSet} hfLoading={props.hfLoading} hfError={props.hfError} onRefreshHf={props.onRefreshHf} />
              ) : (
                <div className="space-y-5">
                  <div className="flex items-center justify-between ml-1"><label className="text-[11px] uppercase tracking-[0.3em] theme-muted font-black font-mono">Context Resumption</label><button onClick={props.onRefreshHf} disabled={!props.hfTokenSet || props.hfLoading} className="flex items-center gap-3 text-[10px] font-black uppercase tracking-widest theme-faint hover:theme-text transition-all group"><RefreshCw className={`w-4 h-4 group-hover:rotate-180 transition-transform duration-500 ${props.hfLoading ? "animate-spin" : ""}`} />Uplink Cloud Index</button></div>
                  <div className="grid grid-cols-1 gap-4">
                    {props.hfDatasets.length > 0 && <div className="relative"><select value={props.hfDatasets.some((d) => d.id === hd.resumeFrom) ? hd.resumeFrom : ""} onChange={(e) => setHd("resumeFrom", e.target.value)} className="w-full px-6 py-4 premium-input rounded-2xl text-[11px] font-black font-mono text-white focus:outline-none shadow-inner appearance-none cursor-pointer bg-black/40"><option className={OPTION_CLASS} value="">— INITIALIZE NEW SEQUENCE —</option>{props.hfDatasets.map((d) => (<option className={OPTION_CLASS} key={d.id} value={d.id}>{d.id}{d.private ? " [SECURE]" : ""}</option>))}</select><ChevronRight className="absolute right-5 top-1/2 -translate-y-1/2 w-4 h-4 rotate-90 pointer-events-none opacity-30" /></div>}
                    <input type="text" value={hd.resumeFrom} onChange={(e) => setHd("resumeFrom", e.target.value)} placeholder="Enter manual resume identifier or leave for profile defaults..." className="w-full px-6 py-4 premium-input rounded-2xl text-sm-fluid font-mono text-white focus:outline-none shadow-2xl bg-black/20 border-white/5" />
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {!props.trainingOnly && (
        isZraldOffline ? (
          <div className="pt-4 animate-premium">
            <div className="rounded-2xl border border-theme-accent/30 bg-theme-accent/5 p-6 space-y-4">
              <div className="flex items-center gap-3">
                <Sparkles className="w-5 h-5 theme-accent animate-pulse" />
                <h4 className="text-xs-fluid uppercase tracking-[0.2em] font-black font-mono theme-accent">ZRALD Offline Workflow Active</h4>
              </div>
              <p className="text-sm-fluid theme-muted font-medium leading-relaxed opacity-95">
                For ZRALD Offline, dataset generation and model training are executed together in a single pipeline run on the remote node. You do not need to generate the dataset beforehand.
              </p>
              <p className="text-[10px] theme-muted font-mono italic opacity-70">
                Proceed directly to the "Train" phase (Step 4) to specify training parameters and launch the combined pipeline.
              </p>
            </div>
          </div>
        ) : (
          <div className="space-y-6 pt-4">
            <div className="premium-card rounded-2xl overflow-hidden glass-panel border border-white/5 shadow-2xl relative">
              <div className="px-8 py-5 border-b border-white/5 bg-white/[0.02] flex items-center justify-between">
                 <div className="flex items-center gap-4">
                  <div className={`w-3.5 h-3.5 rounded-full shadow-[0_0_12px_currentColor] transition-all duration-500 ${props.generating ? "bg-amber-500 animate-pulse text-amber-400" : props.generated ? "bg-emerald-500 text-emerald-400" : "bg-red-500 text-red-400"}`} />
                  <div className="flex flex-col">
                     <span className="text-[11px] uppercase tracking-[0.3em] font-black font-mono theme-text/80">PIPELINE STATUS: {props.generating ? "EXECUTING_PASS" : props.generated ? "BUFFER_SYNCED" : "AWAITING_HANDSHAKE"}</span>
                     <span className="text-[9px] theme-faint font-mono uppercase mt-1">Telemetry Generation Context active</span>
                  </div>
                </div>
                <div className="flex gap-3">
                   {props.generating ? (
                    <button onClick={props.onCancel} className="px-6 py-2.5 bg-red-500/10 border border-red-500/20 text-red-400 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all hover:bg-red-500 hover:text-white shadow-lg shadow-red-500/10">Abort Sequence</button>
                  ) : (
                    <button onClick={props.onGenerate} disabled={!props.sshHostSet} className="px-8 py-2.5 theme-accent-bg text-black rounded-xl text-[10px] font-black uppercase tracking-widest transition-all hover:brightness-125 disabled:opacity-20 shadow-xl shadow-theme-accent/20 premium-button">Initiate Synthesis Pass</button>
                  )}
                </div>
              </div>

              {(props.generating || props.progress) && (
                <div className="grid grid-cols-3 divide-x divide-white/10 bg-black/30 text-center animate-premium shadow-inner">
                  <div className="py-8 px-6 group transition-colors hover:bg-white/[0.01]"><p className="text-[10px] uppercase tracking-[0.2em] theme-faint font-black font-mono mb-2 opacity-50">Ingested Fragments</p><p className="text-3xl font-black font-mono text-white tabular-nums tracking-tighter">{props.progress?.scanned || 0}</p></div>
                  <div className="py-8 px-6 group transition-colors hover:bg-white/[0.01]"><p className="text-[10px] uppercase tracking-[0.2em] theme-accent font-black font-mono mb-2">Accepted Contexts</p><p className="text-3xl font-black font-mono text-white tabular-nums tracking-tighter">{props.progress?.kept || 0}</p></div>
                  <div className="py-8 px-6 group transition-colors hover:bg-white/[0.01]"><p className="text-[10px] uppercase tracking-[0.2em] theme-faint font-black font-mono mb-2 opacity-50">Rejected Noise</p><p className="text-3xl font-black font-mono text-red-500/40 tabular-nums tracking-tighter">{props.progress?.rejected || 0}</p></div>
                </div>
              )}

              {(props.generating || props.logs || props.error) && (
                 <div className="animate-premium">
                   <div className="bg-black/60 p-8 h-80 overflow-y-auto font-mono text-[12px] leading-relaxed selection:bg-theme-selection scrollbar-thin scrollbar-thumb-white/10 tracking-tight">
                     {props.logs || <span className="theme-faint italic opacity-50">Establishing semantic handshake with remote node...</span>}
                     {props.error && <div className="mt-6 p-4 rounded-xl border border-red-500/30 bg-red-500/5 text-red-400 font-bold uppercase text-[11px] tracking-[0.15em] shadow-xl animate-premium">✕ CRITICAL CONTEXT FAULT: {props.error}</div>}
                     {props.generating && <span className="inline-block w-2.5 h-4.5 theme-accent-bg ml-3 animate-pulse align-middle shadow-[0_0_10px_currentColor]" />}
                   </div>
                   <div className="px-8 py-2 border-t border-white/5 bg-white/[0.01]"><p className="text-[9px] theme-faint font-mono uppercase tracking-[0.3em] text-center opacity-40">Direct pass telemetry stream</p></div>
                 </div>
              )}
            </div>
          </div>
        )
      )}
    </div>
  );
}

// ── step 4 ────────────────────────────────────────────────────────────────

function TrainStep(props: {
  trainingOnly: boolean;
  trainingDataset: React.ReactNode;
  runName: string;
  onRunNameChange: (s: string) => void;
  lora: LoraConfig;
  onLoraChange: (l: LoraConfig) => void;
  studentModel: string;
  onStudentChange: (s: string) => void;
  studentModelOptions: StudentModelOption[];
  hfLoading: boolean;
  hfTokenSet: boolean;
  onRefreshModels: () => void;
  hub: HubConfig;
  onHubChange: (h: HubConfig) => void;
  hfUsername: string | null;
  canLaunch: boolean;
  launching: boolean;
  launchError: string | null;
  onLaunch: () => void;
  validatingDataset: boolean;
  datasetsValidated: boolean;
  trainingOnlyDatasets: string[];
  hubDatasetValidation: Record<string, { valid: boolean; sampleCount?: number; error?: string; validatedAt?: number }>;
  onValidateDatasets: () => void;
  validateButtonRef?: React.RefObject<HTMLButtonElement | null>;
  requiresCloudTrainingDataset?: boolean;
  zraldUsesHf?: boolean;
}) {
  const hub = props.hub;
  const setHub = <K extends keyof HubConfig>(k: K, v: HubConfig[K]) => props.onHubChange({ ...hub, [k]: v });
  const modelRepoPlaceholder = props.hfUsername ? `${props.hfUsername}/geodetic-lora-production` : "profile-root/repository-id";

  return (
    <div className="space-y-10 animate-premium">
      <div className="flex flex-col gap-1">
        <h3 className="text-base-fluid uppercase tracking-[0.25em] theme-accent font-black italic font-serif">Engine Initialization</h3>
        <p className="text-sm-fluid theme-muted font-medium opacity-80 leading-relaxed max-w-2xl">{props.trainingOnly ? <>Finalize the student architecture and initiate the fine-tuning sequence on the remote MI300X node. High-parameter optimization active.</> : <>Dataset generation pass complete. Synchronizing semantic buffers with LLaMA-Factory context for LoRA acquisition sequence.</>}</p>
      </div>

      {props.trainingOnly && props.trainingDataset}

      <div className="space-y-4">
        <div className="flex items-center justify-between ml-1">
          <label className="text-[11px] uppercase tracking-[0.3em] theme-muted font-black font-mono leading-none">Session Identifier</label>
          {props.requiresCloudTrainingDataset ? (
            <button type="button" ref={props.validateButtonRef} onClick={props.onValidateDatasets} disabled={props.validatingDataset || props.datasetsValidated || props.trainingOnlyDatasets.length === 0} className={`flex items-center gap-3 px-5 py-2 rounded-xl text-[10px] uppercase tracking-[0.2em] font-black font-mono transition-all group shadow-sm ${props.datasetsValidated ? "bg-emerald-500/10 border border-emerald-500/20 text-emerald-400" : "theme-accent-soft theme-accent border-theme-accent/40 hover:bg-theme-accent hover:text-black"}`}>
              {props.validatingDataset ? <><Loader2 className="w-4 h-4 animate-spin" />VALIDATING...</> : props.datasetsValidated ? <><CheckCircle2 className="w-4 h-4" />VALIDATED</> : <><ShieldAlert className="w-4 h-4" />VALIDATE DATASETS</>}
            </button>
          ) : null}
        </div>
        <input type="text" value={props.runName} onChange={(e) => props.onRunNameChange(e.target.value)} placeholder="e.g. GEODETIC_LORA_PROD_V1" className="w-full px-6 py-5 premium-input rounded-2xl text-base-fluid font-black font-mono text-white focus:outline-none shadow-2xl tracking-[0.1em] bg-black/20 border-white/10 uppercase" />
      </div>

      <div className="premium-card rounded-2xl p-8 glass-panel border border-white/5 bg-white/[0.01] shadow-2xl relative overflow-hidden group">
         <div className="absolute top-0 left-0 w-1 h-full theme-accent-bg opacity-30 group-hover:opacity-100 transition-opacity" />
         <div className="flex items-center gap-3 mb-8 ml-1">
            <div className="w-1.5 h-6 theme-accent-bg rounded-full shadow-[0_0_10px_currentColor]" />
            <label className="text-[11px] uppercase tracking-[0.3em] theme-muted font-black font-mono">Hyper-Parameter Calibration</label>
          </div>
        <TrainingConfigForm
          value={props.lora}
          onChange={props.onLoraChange}
          studentModel={props.studentModel}
          onStudentChange={props.onStudentChange}
          studentModelOptions={props.studentModelOptions}
          hfLoading={props.hfLoading}
          hfTokenSet={props.hfTokenSet}
          onRefreshModels={props.onRefreshModels}
          directTraining={props.trainingOnly}
        />
      </div>

      <div className="premium-card rounded-2xl p-8 glass-panel border border-white/5 space-y-8 relative overflow-hidden group shadow-2xl">
         <div className="absolute top-0 left-0 w-1.5 h-full theme-accent-bg opacity-20 group-hover:opacity-100 transition-opacity" />
        <div className="flex items-center justify-between">
          <div className="space-y-2">
            <p className="text-[11px] uppercase tracking-[0.3em] theme-accent font-black font-mono">Weight Hub Synchronization</p>
            <p className="text-sm-fluid theme-muted font-medium opacity-80 leading-relaxed max-w-2xl">Real-time checkpointing to the Hugging Face Hub. Ensures model persistence and resumable training contexts.</p>
          </div>
          <label className={`shrink-0 inline-flex items-center gap-4 px-6 py-3 rounded-full border transition-all duration-500 cursor-pointer shadow-xl ${hub.enabled ? "theme-accent-soft theme-accent border-theme-accent/40" : "theme-surface-soft border-white/10 theme-muted"}`}><span className={`w-2.5 h-2.5 rounded-full ${hub.enabled ? "theme-accent-bg shadow-[0_0_10px_currentColor] animate-pulse" : "bg-white/10"}`} /><span className="text-[11px] font-black uppercase tracking-[0.2em] font-mono">Push {hub.enabled ? "ONLINE" : "OFFLINE"}</span><input type="checkbox" checked={hub.enabled} onChange={(e) => setHub("enabled", e.target.checked)} className="hidden" /></label>
        </div>

        {hub.enabled && (
          <div className="space-y-8 animate-premium">
            <div className="space-y-3">
              <div className="flex items-center justify-between ml-1"><label className="text-[10px] uppercase tracking-widest theme-muted font-black font-mono opacity-60 tracking-[0.2em]">Remote Model ID</label>{props.hfUsername && <span className="text-[9px] font-black font-mono text-emerald-400 uppercase tracking-widest bg-emerald-500/10 px-3 py-1 rounded-full border border-emerald-500/20 shadow-sm">Target Profile: {props.hfUsername}</span>}</div>
              <div className="flex gap-4">
                <input type="text" value={hub.modelId} onChange={(e) => setHub("modelId", e.target.value)} placeholder={modelRepoPlaceholder} className="flex-1 px-6 py-4 premium-input rounded-2xl text-sm-fluid font-mono text-white focus:outline-none shadow-2xl bg-black/40 border-white/10 tracking-tight" />
                 {props.hfUsername && !hub.modelId && (
                  <button onClick={() => setHub("modelId", `${props.hfUsername}/geodetic-lora-qwen-7b`)} className="px-6 py-4 rounded-2xl border border-white/10 theme-surface-soft theme-text text-[11px] font-black font-mono transition-all hover:bg-white/5 hover:border-theme-accent/30 premium-button whitespace-nowrap shadow-lg">Default Path</button>
                )}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-6 items-end">
              <div className="space-y-3"><label className="text-[10px] uppercase tracking-widest theme-muted font-black ml-1 tracking-[0.2em] opacity-60">Push Logic</label><div className="relative"><select value={hub.strategy} onChange={(e) => setHub("strategy", e.target.value as HubConfig["strategy"])} className="w-full px-6 py-4 premium-input rounded-2xl text-[11px] font-black font-mono text-white focus:outline-none shadow-inner appearance-none cursor-pointer bg-black/40"><option className={OPTION_CLASS} value="every_save">EACH_CHECKPOINT</option><option className={OPTION_CLASS} value="checkpoint">ROLLING_LATEST</option><option className={OPTION_CLASS} value="end">FINAL_ONLY</option></select><ChevronRight className="absolute right-5 top-1/2 -translate-y-1/2 w-4 h-4 rotate-90 pointer-events-none opacity-30" /></div></div>
              <div className="pb-1"><label className="flex items-center justify-center gap-4 p-4 rounded-2xl border border-white/5 bg-white/[0.01] hover:bg-white/[0.03] transition-colors cursor-pointer group shadow-xl"><div className={`w-5 h-5 rounded border-2 transition-all flex items-center justify-center ${hub.private ? "bg-theme-accent border-theme-accent" : "border-white/20 group-hover:border-white/40"}`}>{hub.private && <CheckCircle2 className="w-4 h-4 text-black" />}</div><span className="text-[10px] font-black font-mono theme-muted uppercase tracking-widest group-hover:theme-text">Private Adapter Repository</span><input type="checkbox" checked={hub.private} onChange={(e) => setHub("private", e.target.checked)} className="hidden" /></label></div>
            </div>

            <div className="space-y-6 pt-6 border-t border-white/5 animate-premium">
               <label className="flex items-center gap-6 p-5 rounded-2xl border border-white/5 bg-white/[0.01] hover:bg-white/[0.03] transition-all cursor-pointer group shadow-xl">
                  <div className={`w-6 h-6 rounded border-2 transition-all flex items-center justify-center ${hub.autoMerge ? "bg-theme-accent border-theme-accent" : "border-white/20 group-hover:border-white/40"}`}>{hub.autoMerge && <CheckCircle2 className="w-5 h-5 text-black" />}</div>
                  <div className="flex-1 space-y-1">
                    <span className="text-[11px] font-black font-mono theme-text uppercase tracking-[0.2em] group-hover:theme-accent transition-colors">Automated Engine Consolidation</span>
                    <p className="text-[9px] theme-faint font-black uppercase tracking-widest opacity-60 leading-none">Merge LoRA weights with base model and publish standalone repository</p>
                  </div>
                  <input type="checkbox" checked={!!hub.autoMerge} onChange={(e) => setHub("autoMerge", e.target.checked)} className="hidden" />
                </label>

{hub.autoMerge && (
                <div className="space-y-3 animate-premium pl-6 border-l-4 border-theme-accent/20 bg-white/[0.01] p-6 rounded-r-2xl shadow-inner">
                  <label className="text-[10px] uppercase tracking-widest theme-muted font-black ml-1 tracking-[0.2em]">Standalone Model ID <span className="opacity-30 font-mono tracking-normal text-[8px]">(AUTO_GENERATED)</span></label>
                  <input type="text" value={hub.mergedModelId ?? ""} onChange={(e) => setHub("mergedModelId", e.target.value)} placeholder={`${hub.modelId || "target-repo"}-merged`} className="w-full px-6 py-4 premium-input rounded-xl text-sm-fluid font-mono text-white focus:outline-none shadow-2xl bg-black/40" />
                  <div className="flex items-center gap-2 pt-2 opacity-50 italic text-[9px] theme-faint font-mono uppercase tracking-widest"><Info className="w-3.5 h-3.5" /> Merge duration: 5–15 minutes // Final size: ~15GB for 7B base</div>

                  <div className="pt-4 border-t border-white/5 mt-4">
                    <label className="flex items-center gap-4 p-4 rounded-xl border border-white/5 bg-white/[0.01] hover:bg-white/[0.03] transition-colors cursor-pointer group shadow-xl">
                      <div className={`w-5 h-5 rounded border-2 transition-all flex items-center justify-center ${hub.autoConvertGguf ? "bg-theme-accent border-theme-accent" : "border-white/20 group-hover:border-white/40"}`}>{hub.autoConvertGguf && <CheckCircle2 className="w-4 h-4 text-black" />}</div>
                      <div className="flex-1 space-y-1">
                        <span className="text-[10px] font-black font-mono theme-text uppercase tracking-[0.15em] group-hover:theme-accent transition-colors">Ollama/llama.cpp Ready</span>
                        <p className="text-[9px] theme-faint font-black uppercase tracking-widest opacity-60 leading-none">Convert merged model to GGUF for local inference</p>
                      </div>
                      <input type="checkbox" checked={!!hub.autoConvertGguf} onChange={(e) => setHub("autoConvertGguf", e.target.checked)} className="hidden" />
                    </label>
                  </div>

                  {hub.autoConvertGguf && (
                    <div className="space-y-3 pl-4 border-l-2 border-theme-accent/30 animate-premium">
                      <div className="space-y-1">
                        <label className="text-[9px] uppercase tracking-widest theme-muted font-black tracking-[0.2em]">Quantization</label>
                        <select value={hub.ggufQuantization ?? "Q4_K_M"} onChange={(e) => setHub("ggufQuantization", e.target.value)} className="w-full px-4 py-3 premium-input rounded-xl text-[11px] font-black font-mono text-white focus:outline-none shadow-xl bg-black/40">
                          <option value="F16">F16 (~14GB) - Lossless quality</option>
                          <option value="Q5_K_M">Q5 K/M (~5GB) - High quality</option>
                          <option value="Q4_K_M">Q4 K/M (~4GB) - Balanced (Recommended)</option>
                          <option value="Q8_0">Q8/0 (~7GB) - High quality, larger</option>
                        </select>
                      </div>
                      <div className="space-y-1">
                        <label className="text-[9px] uppercase tracking-widest theme-muted font-black tracking-[0.2em]">GGUF Repo ID <span className="opacity-30 font-mono tracking-normal text-[7px]">(OPTIONAL)</span></label>
                        <input type="text" value={hub.ggufRepoId ?? ""} onChange={(e) => setHub("ggufRepoId", e.target.value)} placeholder={`${hub.modelId || "target-repo"}-gguf`} className="w-full px-4 py-3 premium-input rounded-xl text-[11px] font-mono text-white focus:outline-none shadow-xl bg-black/40" />
                      </div>
                      <div className="flex items-center gap-2 opacity-50 italic text-[9px] theme-faint font-mono uppercase tracking-widest"><Info className="w-3 h-3" /> Run locally: ollama run hf.co/{hub.ggufRepoId || `${hub.modelId || "repo"}-gguf`}</div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {props.launchError && (
        <div className="p-6 rounded-2xl border border-red-500/30 bg-red-500/5 text-red-300 text-sm-fluid font-mono shadow-2xl animate-premium relative overflow-hidden">
           <div className="absolute top-0 left-0 w-1 h-full bg-red-500 shadow-[0_0_10px_#f87171]" />
           <div className="flex items-center gap-3 mb-3 ml-1">
             <ShieldAlert className="w-6 h-6 text-red-400" />
             <span className="font-black uppercase tracking-[0.3em] text-[11px]">System Launch Fault</span>
          </div>
          <p className="opacity-90 leading-relaxed font-black tracking-tight">{props.launchError}</p>
        </div>
      )}

      <div className="space-y-6 pt-6">
        <button disabled={!props.canLaunch || props.launching} onClick={props.onLaunch} className="w-full max-w-md mx-auto py-4 theme-accent-bg text-black text-[11px] font-black uppercase tracking-[0.25em] rounded-xl premium-button hover:brightness-125 active:scale-[0.99] transition-all disabled:opacity-20 flex items-center justify-center gap-3 shadow-lg shadow-theme-accent/20 group">
          {props.launching ? (
            <><Loader2 className="w-4 h-4 animate-spin text-black" /><span className="animate-pulse">SYNCHRONIZING ENGINE...</span></>
          ) : (
            <><Play className="w-4 h-4 fill-black group-hover:scale-110 transition-transform" /><span>INITIALIZE PIPELINE</span></>
          )}
        </button>
        {!props.canLaunch && (
          <div className="flex flex-col items-center gap-4 opacity-50 select-none">
            <p className="text-[10px] theme-muted font-black font-mono uppercase tracking-[0.3em] text-center max-w-lg leading-relaxed">System offline: Ensure SSH socket context, Knowledge Base alignment, and Student model target are defined for launch sequence.</p>
            <div className="w-24 h-0.5 bg-white/10 rounded-full" />
          </div>
        )}
      </div>
    </div>
  );
}

const Info = ({ className }: { className?: string }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>
);

export default function PipelineWizard({
  config,
  gpuStatus,
  onConfigChange,
  onPipelineLaunched,
  onStepChange,
}: Props) {
  const [step, setStep] = useState(0);

  useEffect(() => {
    onStepChange?.(step);
  }, [step, onStepChange]);
const [pipelineMode, setPipelineMode] = useState<PipelineMode>("rag");
  const isTrainingOnly = pipelineMode === "trainingOnly";
  const validateButtonRef = useRef<HTMLButtonElement | null>(null);

  const [chunkCount, setChunkCount] = useState<number | null>(null);
  const [samples, setSamples] = useState<Chunk[]>([]);
  const [loadingKb, setLoadingKb] = useState(false);
  const [kbError, setKbError] = useState<string | null>(null);
  // True when the probe failed specifically because the collection doesn't
  // exist yet (404) — distinct from a real connection/auth error, and offers a
  // "Create collection" action instead of a scary red failure.
  const [kbMissingCollection, setKbMissingCollection] = useState(false);
  const [creatingCollection, setCreatingCollection] = useState(false);

  const [teacher, setTeacher] = useState<TeacherConfig>(config.teacher ?? DEFAULT_TEACHER);
  const [teacherDeployed, setTeacherDeployed] = useState(false);
  const [deployedTeacherModel, setDeployedTeacherModel] = useState<string | null>(null);
  const [checkingTeacher, setCheckingTeacher] = useState(false);
  const [deployStreamId, setDeployStreamId] = useState<string | null>(null);
  const [deployLogs, setDeployLogs] = useState("");
  const [deploying, setDeploying] = useState(false);
  const [deployError, setDeployError] = useState<string | null>(null);

  const checkDeployment = React.useCallback(async (currentTeacher: TeacherConfig) => {
    if (!config.ssh.host) { setTeacherDeployed(false); setDeployedTeacherModel(null); return; }
    setCheckingTeacher(true);
    try {
      const activeTeacher = await api.checkTeacherDeployed(config.ssh, config.docker, currentTeacher);
      const exactTeacher = activeTeacher?.exact ? activeTeacher : null;
      setTeacherDeployed(exactTeacher !== null);
      setDeployedTeacherModel(exactTeacher?.modelId || null);
      if (exactTeacher !== null) {
        const updatedTeacher = {
          ...currentTeacher,
          vllmPort: exactTeacher.port,
          repoId: exactTeacher.modelId || currentTeacher.repoId,
        };
        if (!teacherConfigEquals(updatedTeacher, currentTeacher)) {
          setTeacher(updatedTeacher);
          onConfigChange({ teacher: updatedTeacher });
          await api.saveConfig({ ...config, teacher: updatedTeacher });
        }
      }
    } catch (e) { console.error(e); setTeacherDeployed(false); setDeployedTeacherModel(null); } finally { setCheckingTeacher(false); }
  }, [config, onConfigChange]);

  const startDeployment = async () => {
    setDeploying(true); setDeployError(null); setDeployLogs("");
    try { const streamId = await api.deployTeacher(config.ssh, config.docker, teacher, config.hfToken); setDeployStreamId(streamId); }
    catch (err: any) { setDeployError(err.message || String(err)); setDeploying(false); }
  };

  const cancelDeployment = async () => {
    if (!deployStreamId) return;
    try { await api.sshStopStream(deployStreamId); }
    catch (e) { console.error("Cancel deployment error:", e); }
    finally { setDeploying(false); setDeployStreamId(null); }
  };

  useEffect(() => { if (step === 1) { checkDeployment(teacher); } }, [step, teacher, checkDeployment]);

  useEffect(() => {
    if (!deployStreamId) return;
    let unlistenLog: (() => void) | null = null;
    let unlistenDone: (() => void) | null = null;
    const setupListeners = async () => {
      unlistenLog = await events.onDeployLog((e) => { if (e.streamId === deployStreamId) { setDeployLogs((prev) => prev + e.line); } });
      unlistenDone = await events.onDeployDone((e) => {
        if (e.streamId === deployStreamId) {
          setDeploying(false); setDeployStreamId(null);
          if (e.success) {
            setTeacherDeployed(true);
            setDeployedTeacherModel(teacher.repoId || null);
            if (e.port !== undefined) {
              const actualPort = e.port;
              setTeacher((prev) => {
                if (prev.vllmPort !== actualPort) {
                  const updated = { ...prev, vllmPort: actualPort };
                  onConfigChange({ teacher: updated });
                  api.saveConfig({ ...config, teacher: updated });
                  return updated;
                }
                return prev;
              });
            }
          } else {
            setDeployError(e.message);
          }
        }
      });
    };
    setupListeners();
    return () => { if (unlistenLog) unlistenLog(); if (unlistenDone) unlistenDone(); };
  }, [deployStreamId]);

  const [generatingDataset, setGeneratingDataset] = useState(false);
  const [datasetGenerated, setDatasetGenerated] = useState(false);
  const [generationRunId, setGenerationRunId] = useState<string | null>(null);
  const [generationLogs, setGenerationLogs] = useState("");
  const [generationProgress, setGenerationProgress] = useState<{ scanned: number; kept: number; rejected: number; status: string; } | null>(null);
  const [generationError, setGenerationError] = useState<string | null>(null);
  const [completedRunId, setCompletedRunId] = useState<string | null>(null);

  useEffect(() => {
    if (!generationRunId) return;
    let unlistenLog: (() => void) | null = null;
    let unlistenProgress: (() => void) | null = null;
    const setupListeners = async () => {
      unlistenLog = await events.onPipelineLog((e) => { if (e.runId === generationRunId) { setGenerationLogs((prev) => prev + e.line + "\n"); } });
      unlistenProgress = await events.onPipelineProgress((e) => {
        if (e.runId === generationRunId) {
          setGenerationProgress({ scanned: e.scanned, kept: e.kept, rejected: e.rejected, status: e.status });
          if (e.status === "done" || e.status === "dataset_ready") { setGeneratingDataset(false); setDatasetGenerated(true); setCompletedRunId(generationRunId); }
          else if (e.status === "failed" || e.status === "cancelled") { setGeneratingDataset(false); if (e.status === "failed") { setGenerationError("Dataset generation failed."); } }
        }
      });
    };
    setupListeners();
    return () => { if (unlistenLog) unlistenLog(); if (unlistenProgress) unlistenProgress(); };
  }, [generationRunId]);

  const [validatingDataset, setValidatingDataset] = useState(false);
  const [datasetValidationError, setDatasetValidationError] = useState<string | null>(null);
  const [showCleanModal, setShowCleanModal] = useState(false);

  // Validate opens the cleaning-options modal first; the modal's "Validate"
  // button persists the chosen cleaning options into hubDataset and then runs
  // the actual dataset validation.
  const openValidateModal = () => {
    if (trainingOnlyDatasets.length === 0) return;
    setShowCleanModal(true);
  };

  const confirmCleanAndValidate = async (opts: {
    cleanRemoveDuplicates: boolean;
    cleanRemoveShort: boolean;
    cleanMinChars: number;
  }) => {
    setHubDataset({ ...hubDataset, ...opts });
    setShowCleanModal(false);
    await validateDatasets();
  };

  const validateDatasets = async () => {
    const datasets = trainingOnlyDatasets;
    if (datasets.length === 0) return;
    setValidatingDataset(true);
    setDatasetValidationError(null);
    const newValidationResults: Record<string, { valid: boolean; sampleCount?: number; error?: string; validatedAt: number }> = {};
    let hasError = false;
    for (const repoId of datasets) {
      try {
        const result = await api.hfValidateDataset(repoId);
        if (result.valid) {
          newValidationResults[repoId] = { valid: true, sampleCount: result.sample_count, validatedAt: Date.now() };
        } else {
          newValidationResults[repoId] = { valid: false, error: result.error || "Validation failed", validatedAt: Date.now() };
          hasError = true;
        }
      } catch (e: any) {
        newValidationResults[repoId] = { valid: false, error: e.message || String(e), validatedAt: Date.now() };
        hasError = true;
      }
    }
    const currentValidation = hubDataset.validationResult || {} as Record<string, { valid: boolean; sampleCount?: number; error?: string; validatedAt: number }>;
    setHubDataset({ ...hubDataset, validationResult: { ...currentValidation, ...newValidationResults } });
    setValidatingDataset(false);
    if (hasError) {
      setDatasetValidationError("One or more datasets failed validation. Please check the highlighted datasets above.");
    }
  };

  const startDatasetGeneration = async () => {
    const namedTopics = topics.filter((t) => (t.topic || "").trim().length > 0);
    if (namedTopics.length === 0) {
      setGenerationError("Add at least one focus topic before generating.");
      return;
    }
    const missingPrompt = namedTopics.find((t) => !(t.promptTemplate || "").trim());
    if (missingPrompt) {
      setGenerationError(`Engineering Directive Prompt is required for every focus topic. Missing for: "${missingPrompt.topic}".`);
      return;
    }
    setGeneratingDataset(true); setDatasetGenerated(false); setGenerationError(null); setGenerationLogs(""); setGenerationProgress(null);
    try {
      const effectiveQdrant = { ...config.qdrant, endpoint: qdrantEndpoint, collection: config.qdrant.collection || "all" };
      const mergedConfig = { ...config, qdrant: effectiveQdrant, teacher };
      onConfigChange({ qdrant: effectiveQdrant, teacher });
      const legacyTopic = runTopics.length === 1 ? runTopics[0].topic : undefined;
      const legacyTotal = runTopics.length === 1 ? runTopics[0].totalQuestions : undefined;
      const generationHubDataset: HubDatasetConfig = isZraldMethod ? { ...hubDataset, trainOnly: false } : hubDataset;
      const rc: RunConfig = { name: `gen-${new Date().toISOString().slice(0, 19).replace(/:/g, "-")}`, teacher, studentModel, lora, promptTemplate: prompt, maxPairsPerChunk, concurrency, maxChunks: maxChunks === "all" ? undefined : (maxChunks as number), topic: legacyTopic, totalQuestions: legacyTotal, topics: runTopics.length > 0 ? runTopics : undefined, hub, hubDataset: generationHubDataset, generateOnly: true, enableVerification, bundleWindow, datasetFormat };
      const runId = await api.startPipeline(mergedConfig, rc); setGenerationRunId(runId);
      try { const initialLog = await api.readRunLog(runId, 64 * 1024); if (initialLog.trim()) { setGenerationLogs(initialLog); } const run = await api.getRun(runId); setGenerationProgress({ scanned: run.qaTotal || 0, kept: run.qaKept || 0, rejected: run.qaRejected || 0, status: run.status }); if (run.status === "dataset_ready" || run.status === "done") { setGeneratingDataset(false); setDatasetGenerated(true); setCompletedRunId(runId); } } catch {}
    } catch (e: any) { setGenerationError(e.message || String(e)); setGeneratingDataset(false); }
  };

  const cancelDatasetGeneration = async () => { if (!generationRunId) return; try { await api.cancelRun(generationRunId); } catch {} finally { setGeneratingDataset(false); } };

  const initialDatasetPrompt = config.promptTemplate || promptForDatasetFormat(DEFAULT_DATASET_FORMAT);
  const [datasetFormat, setDatasetFormat] = useState<string>(DEFAULT_DATASET_FORMAT);
  const [topics, setTopics] = useState<TopicTarget[]>([{ topic: "", totalQuestions: undefined, promptTemplate: initialDatasetPrompt }]);
  const [maxPairsPerChunk, setMaxPairsPerChunk] = useState(1);
  const [concurrency, setConcurrency] = useState(100);
  const [maxChunks, setMaxChunks] = useState<number | "all">("all");
  const [prompt, setPrompt] = useState(() => initialDatasetPrompt);
  const [enableVerification, setEnableVerification] = useState(false);
  const [bundleWindow, setBundleWindow] = useState(0);

  useEffect(() => {
    if (config.promptTemplate && config.promptTemplate !== prompt) {
      setPrompt(config.promptTemplate);
    }
  }, [config.promptTemplate]);

  const handlePromptChange = (nextPrompt: string) => {
    setPrompt(nextPrompt);
    onConfigChange({ promptTemplate: nextPrompt });
  };

  const [hubDataset, setHubDataset] = useState<HubDatasetConfig>(DEFAULT_HUB_DATASET);

  useEffect(() => { setHubDataset((current) => ({ ...current, enabled: isTrainingOnly ? true : current.enabled, trainOnly: isTrainingOnly })); if (isTrainingOnly && step !== 3) { setStep(3); } }, [isTrainingOnly, step]);

  const [studentModel, setStudentModel] = useState(config.student?.repoId || "Qwen/Qwen2.5-7B-Instruct");
  const [lora, setLora] = useState<LoraConfig>(DEFAULT_LORA);
  const isZraldMethod = (lora.method || "lora") === "zrald" || (lora.method || "lora") === "zrald_offline";
  const [hub, setHub] = useState<HubConfig>(DEFAULT_HUB);
  const [runName, setRunName] = useState("");
  const [launching, setLaunching] = useState(false);
  const [launchError, setLaunchError] = useState<string | null>(null);
  const [cleaningVram, setCleaningVram] = useState(false);

  useEffect(() => {
    if (lora.method !== "zrald" || !isTrainingOnly) return;
    setPipelineMode("rag");
    setHubDataset((current) => ({ ...current, enabled: false, trainOnly: false }));
    if (step === 3 && !datasetGenerated) {
      setStep(teacherDeployed ? 2 : 1);
    }
  }, [lora.method, isTrainingOnly, step, datasetGenerated, teacherDeployed]);

  const handleCleanupVram = async () => { if (!config.ssh.host) return; setCleaningVram(true); try { const msg = await api.cleanupVram(config.ssh, config.docker); alert(msg); } catch (e: any) { alert(`Cleanup failed: ${e}`); } finally { setCleaningVram(false); } };

  const [hfUsername, setHfUsername] = useState<string | null>(null);
  const [hfDatasets, setHfDatasets] = useState<HfDatasetRepo[]>([]);
  const [hfModels, setHfModels] = useState<HfModelRepo[]>([]);
  const [runsForStudentModels, setRunsForStudentModels] = useState<Run[]>([]);
  const [hfLoading, setHfLoading] = useState(false);
  const [hfError, setHfError] = useState<string | null>(null);

  const refreshHf = async () => {
    if (!config.hfToken) { setHfUsername(null); setHfDatasets([]); setHfModels([]); setHfError(null); setHfLoading(false); return; }
    setHfLoading(true); setHfError(null);
    try {
      const who = await api.hfWhoami();
      setHfUsername(who.name || null);
      const errors: string[] = [];
      try {
        const ds = await api.hfListDatasets();
        setHfDatasets(ds);
      } catch (e: any) {
        setHfDatasets([]);
        errors.push(`could not list datasets: ${e}`);
      }
      try {
        const models = await api.hfListModels();
        setHfModels(models);
      } catch (e: any) {
        setHfModels([]);
        errors.push(`could not list models: ${e}`);
      }
      setHfError(errors.length > 0 ? errors.join(" // ") : null);
    }
    catch (e: any) { setHfUsername(null); setHfDatasets([]); setHfModels([]); setHfError(String(e)); } finally { setHfLoading(false); }
  };

  const refreshRunModels = async () => {
    try {
      const list = await api.listRuns();
      setRunsForStudentModels(list);
    } catch (e) {
      console.error("list_runs for student models:", e);
    }
  };

  useEffect(() => { setTeacher(config.teacher ?? DEFAULT_TEACHER); }, [config.teacher]);
  useEffect(() => { refreshHf(); refreshRunModels(); }, [config.hfToken]);

  const refreshModelPickers = async () => {
    await Promise.all([refreshHf(), refreshRunModels()]);
  };

  const studentModelOptions = useMemo<StudentModelOption[]>(() => {
    const out: StudentModelOption[] = [];
    const seen = new Set<string>();
    const add = (id: string | undefined, label: string, source: StudentModelOption["source"]) => {
      const clean = (id || "").trim();
      if (!clean) return;
      const key = clean.toLowerCase();
      if (seen.has(key)) return;
      seen.add(key);
      out.push({ id: clean, label, source });
    };

    for (const run of runsForStudentModels) {
      if (run.status !== "done") continue;
      const explicitMerged = run.hub?.mergedModelId?.trim();
      const inferredMerged = run.hub?.autoMerge && run.hub?.modelId?.trim()
        ? `${run.hub.modelId.trim()}-merged`
        : "";
      add(explicitMerged || inferredMerged, run.name || "Completed run", "merged");
    }

    for (const model of hfModels) {
      const lower = model.id.toLowerCase();
      const label = lower.includes("merged") || lower.includes("merge") || lower.includes("full")
        ? "HF merged"
        : "HF model";
      add(model.id, label, "hf");
    }

    return out;
  }, [runsForStudentModels, hfModels]);

  const refreshKb = async () => {
    setLoadingKb(true);
    setKbError(null);
    setKbMissingCollection(false);
    try {
      const [c, s] = await Promise.all([api.qdrantCount(config.qdrant), api.qdrantScrollAll(config.qdrant, 1000)]);
      setChunkCount(c);
      setSamples(s);
    } catch (e: any) {
      const msg = String(e);
      // A 404 "doesn't exist" means the collection just hasn't been created yet
      // — that's a benign, fixable state, not a connection failure. Surface it
      // separately so the UI can offer a "Create collection" button.
      if (/doesn'?t exist|not found|404/i.test(msg)) {
        setKbMissingCollection(true);
        setChunkCount(null);
        setSamples([]);
      } else {
        setKbError(msg);
      }
    } finally {
      setLoadingKb(false);
    }
  };

  const createCollection = async () => {
    setCreatingCollection(true);
    try {
      await api.qdrantEnsureCollection(config.qdrant);
      await refreshKb();
    } catch (e: any) {
      setKbError(String(e));
    } finally {
      setCreatingCollection(false);
    }
  };

  const trainingOnlyDatasets = useMemo<string[]>(() => {
    const list = (hubDataset.repoIds && hubDataset.repoIds.length > 0) ? hubDataset.repoIds : (hubDataset.repoId ? [hubDataset.repoId] : []);
    return list.map((r) => (r || "").trim()).filter((r) => r.length > 0);
}, [hubDataset.repoIds, hubDataset.repoId]);

  const cleanedTopics = useMemo<TopicTarget[]>(() => topics.map((t) => ({ topic: (t.topic || "").trim(), tag: t.tag?.trim() || undefined, totalQuestions: t.totalQuestions && t.totalQuestions > 0 ? t.totalQuestions : undefined, promptTemplate: t.promptTemplate?.trim() ? t.promptTemplate.trim() : undefined })).filter((t) => t.topic.length > 0), [topics]);
  const runTopics = useMemo<TopicTarget[]>(() => {
    if (!isZraldMethod || cleanedTopics.length === 0) return cleanedTopics;
    const target = Math.max(1, lora.zraldTrainQuestions || DEFAULT_LORA.zraldTrainQuestions || 1000);
    const perTopicTarget = cleanedTopics.length === 1 ? target : Math.max(1, Math.ceil(target / cleanedTopics.length));
    return cleanedTopics.map((topic) => ({
      ...topic,
      totalQuestions: topic.totalQuestions ?? perTopicTarget,
    }));
  }, [cleanedTopics, isZraldMethod, lora.zraldTrainQuestions]);
  const allTopicsHavePrompt = useMemo(() => {
    const named = topics.filter((t) => (t.topic || "").trim().length > 0);
    if (named.length === 0) return false;
    return named.every((t) => (t.promptTemplate || "").trim().length > 0);
  }, [topics]);
  const datasetsValidated = useMemo(() => {
    if (!isTrainingOnly) return true;
    const validation = hubDataset.validationResult;
    if (!validation) return false;
    return trainingOnlyDatasets.every(repoId => validation[repoId]?.valid === true);
  }, [hubDataset.validationResult, trainingOnlyDatasets, isTrainingOnly]);

  useEffect(() => {
    if (isTrainingOnly && step === 3 && !datasetsValidated && trainingOnlyDatasets.length > 0 && validateButtonRef.current) {
      setTimeout(() => {
        validateButtonRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
        validateButtonRef.current?.classList.add('animate-pulse');
        setTimeout(() => validateButtonRef.current?.classList.remove('animate-pulse'), 2000);
      }, 100);
    }
  }, [step, isTrainingOnly, datasetsValidated, trainingOnlyDatasets]);

  const qdrantEndpoint = config.qdrant.endpoint || (config.ssh.host ? `http://${config.ssh.host}:6333` : "");
  const requiresCloudTrainingDataset = isTrainingOnly && (!isZraldMethod || lora.zraldDatasetSource === "huggingface");
  const zraldUsesHf = isZraldMethod && lora.zraldDatasetSource === "huggingface";
  const teacherModelReady = !!((teacher.repoId || "").trim() || (teacher.customServeCmd || "").trim());
  const zraldHfSourceReady = zraldUsesHf && hubDataset.enabled && trainingOnlyDatasets.length > 0 && datasetsValidated && teacherModelReady;
  const generationSourceReady = qdrantEndpoint && (config.qdrant.collection || "all") && teacherModelReady && allTopicsHavePrompt;
  const canLaunch = !!(config.ssh.host && studentModel && (requiresCloudTrainingDataset ? hubDataset.enabled && trainingOnlyDatasets.length > 0 && datasetsValidated : zraldUsesHf ? zraldHfSourceReady : generationSourceReady));

  const launch = async () => {
    setLaunching(true); setLaunchError(null);
    try {
      const mergedConfig = {
        ...config,
        qdrant: { ...config.qdrant, endpoint: qdrantEndpoint, collection: config.qdrant.collection || "all" },
        teacher,
        student: { ...config.student, repoId: studentModel },
      };
      onConfigChange({ qdrant: mergedConfig.qdrant, teacher, student: { ...config.student, repoId: studentModel } });
      const hubDatasetOut: HubDatasetConfig = requiresCloudTrainingDataset
        ? { ...hubDataset, enabled: true, trainOnly: true, repoId: trainingOnlyDatasets[0] || "", repoIds: trainingOnlyDatasets }
        : isZraldMethod
          ? {
              ...hubDataset,
              enabled: zraldUsesHf,
              trainOnly: false,
              repoId: zraldUsesHf ? (trainingOnlyDatasets[0] || "") : "",
              repoIds: zraldUsesHf ? trainingOnlyDatasets : [],
            }
          : { ...hubDataset, enabled: hubDataset.enabled, trainOnly: false };
      if (completedRunId) {
        await api.updateRunConfig(completedRunId, studentModel, lora, hub, hubDatasetOut, prompt, runTopics);
        await api.resumeRun(completedRunId);
        onPipelineLaunched(completedRunId);
      }
      else {
        const legacyTopic = runTopics.length === 1 ? runTopics[0].topic : undefined;
        const legacyTotal = runTopics.length === 1 ? runTopics[0].totalQuestions : undefined;
        const rc: RunConfig = { name: runName || `run-${new Date().toISOString().slice(0, 19)}`, teacher, studentModel, lora, promptTemplate: prompt, maxPairsPerChunk, concurrency, maxChunks: maxChunks === "all" ? undefined : (maxChunks as number), topic: legacyTopic, totalQuestions: legacyTotal, topics: runTopics.length > 0 ? runTopics : undefined, hub, hubDataset: hubDatasetOut, enableVerification, bundleWindow, datasetFormat };
        const runId = await api.startPipeline(mergedConfig, rc); onPipelineLaunched(runId);
      }
    } catch (e: any) { setLaunchError(e.message || String(e)); } finally { setLaunching(false); }
  };

  return (
    <div className="premium-card rounded-2xl animate-premium relative">
      <div className="border-b border-white/5 p-5 bg-white/[0.01] backdrop-blur-md rounded-t-2xl">
        <div className="grid grid-cols-2 gap-4">
          <button type="button" onClick={() => { setPipelineMode("rag"); setStep(0); }} className={`text-left rounded-2xl border p-5 transition-all duration-500 premium-button group ${!isTrainingOnly ? "theme-accent-soft theme-accent border-theme-accent/40 shadow-lg shadow-theme-accent/5" : "border-white/5 bg-white/[0.02] theme-muted hover:theme-text hover:bg-white/[0.04]"}`}>
            <div className="flex items-center gap-3 text-xs-fluid uppercase tracking-[0.2em] font-black font-mono"><div className={`p-1.5 rounded-lg transition-colors ${!isTrainingOnly ? "bg-theme-accent text-black" : "bg-white/5 text-white/40 group-hover:text-white"}`}><Sparkles className="w-4 h-4" /></div>RAG Teacher Student</div>
            <p className="mt-2 text-sm-fluid theme-muted font-medium opacity-80 leading-relaxed">Standard flow: Knowledge retrieval, Teacher-led generation, then Student training.</p>
          </button>
          <button type="button" onClick={() => { setPipelineMode("trainingOnly"); setStep(3); }} className={`text-left rounded-2xl border p-5 transition-all duration-500 premium-button group ${isTrainingOnly ? "theme-accent-soft theme-accent border-theme-accent/40 shadow-lg shadow-theme-accent/5" : "border-white/5 bg-white/[0.02] theme-muted hover:theme-text hover:bg-white/[0.04]"}`}>
            <div className="flex items-center gap-3 text-xs-fluid uppercase tracking-[0.2em] font-black font-mono"><div className={`p-1.5 rounded-lg transition-colors ${isTrainingOnly ? "bg-theme-accent text-black" : "bg-white/5 text-white/40 group-hover:text-white"}`}><Layers className="w-4 h-4" /></div>Direct Training</div>
            <p className="mt-2 text-sm-fluid theme-muted font-medium opacity-80 leading-relaxed">Skip generation. Connect an existing dataset and launch Student training immediately.</p>
          </button>
        </div>
      </div>

      <div className="flex items-stretch border-b border-white/5 bg-black/20 overflow-x-auto no-scrollbar">
        {STEPS.map((s, i) => {
          if (isTrainingOnly && i !== 3) return null;
          const Icon = s.icon; const active = step === i; const done = i < step;
          return (
            <button key={s.key} onClick={() => { if (!isTrainingOnly) { if (i === 2 && !teacherDeployed) return; if (i === 3 && (!teacherDeployed || (!datasetGenerated && lora.method !== "zrald_offline"))) return; } setStep(i); }} className={`flex-1 px-6 py-4 flex items-center justify-center gap-3 transition-all duration-300 border-r border-white/5 relative group ${active ? "bg-white/[0.03] theme-accent" : done ? "text-emerald-400 hover:bg-white/[0.02]" : "theme-muted hover:bg-white/[0.02]"}`}>
              <div className={`w-8 h-8 rounded-lg flex items-center justify-center transition-all duration-300 ${active ? "theme-accent-bg text-black shadow-lg shadow-theme-accent/20 scale-110" : done ? "bg-emerald-500/10 text-emerald-400" : "bg-white/5 text-white/20"}`}>{done ? <CheckCircle2 className="w-4 h-4" /> : <Icon className="w-4 h-4" />}</div>
              <div className="flex flex-col items-start min-w-0"><span className="text-[8px] uppercase tracking-widest font-black opacity-30 leading-none mb-1">Step 0{i + 1}</span><span className="text-xs font-black uppercase tracking-widest font-mono truncate">{s.label}</span></div>
              {active && <div className="absolute bottom-0 left-0 w-full h-0.5 theme-accent-bg shadow-[0_0_10px_currentColor]" />}
            </button>
          );
        })}
      </div>

      <div className="p-8 space-y-8 min-h-[520px]">
        {step === 0 && <KnowledgeBaseStep gpuStatus={gpuStatus ?? null} samples={samples} loading={loadingKb} error={kbError} config={config} onConfigChange={onConfigChange} onSkip={() => setStep(1)} />}
        {step === 1 && <TeacherStep value={teacher} onChange={(t) => { setTeacher(t); onConfigChange({ teacher: t }); }} gpuStatus={gpuStatus} hfToken={config.hfToken || ""} checkingTeacher={checkingTeacher} teacherDeployed={teacherDeployed} deployedTeacherModel={deployedTeacherModel} deploying={deploying} deployLogs={deployLogs} deployError={deployError} onCheckStatus={() => checkDeployment(teacher)} onDeploy={startDeployment} onCancelDeploy={cancelDeployment} />}
        {step === 2 && !isTrainingOnly && <DatasetStep config={config} onConfigChange={onConfigChange} trainingOnly={isTrainingOnly} onSwitchToGenerateDataset={() => { setPipelineMode("rag"); setStep(0); }} topics={topics} onTopicsChange={setTopics} prompt={prompt} onPromptChange={handlePromptChange} maxPairsPerChunk={maxPairsPerChunk} onMaxPairsChange={setMaxPairsPerChunk} concurrency={concurrency} onConcurrencyChange={setConcurrency} maxChunks={maxChunks} onMaxChunksChange={setMaxChunks} hubDataset={hubDataset} onHubDatasetChange={setHubDataset} hfTokenSet={!!config.hfToken} hfUsername={hfUsername} hfDatasets={hfDatasets} hfLoading={hfLoading} hfError={hfError} onRefreshHf={refreshHf} generating={generatingDataset} generated={datasetGenerated} progress={generationProgress} logs={generationLogs} error={generationError} onGenerate={startDatasetGeneration} onCancel={cancelDatasetGeneration} sshHostSet={!!config.ssh.host} method={lora.method} enableVerification={enableVerification} onEnableVerificationChange={setEnableVerification} bundleWindow={bundleWindow} onBundleWindowChange={setBundleWindow} datasetFormat={datasetFormat} onDatasetFormatChange={setDatasetFormat} />}
        {step === 3 && <TrainStep trainingOnly={isTrainingOnly} requiresCloudTrainingDataset={requiresCloudTrainingDataset} zraldUsesHf={zraldUsesHf} trainingDataset={(isTrainingOnly || zraldUsesHf) ? <DatasetStep config={config} onConfigChange={onConfigChange} trainingOnly={requiresCloudTrainingDataset} onSwitchToGenerateDataset={() => { setPipelineMode("rag"); setStep(0); }} topics={topics} onTopicsChange={setTopics} prompt={prompt} onPromptChange={handlePromptChange} maxPairsPerChunk={maxPairsPerChunk} onMaxPairsChange={setMaxPairsPerChunk} concurrency={concurrency} onConcurrencyChange={setConcurrency} maxChunks={maxChunks} onMaxChunksChange={setMaxChunks} hubDataset={hubDataset} onHubDatasetChange={setHubDataset} hfTokenSet={!!config.hfToken} hfUsername={hfUsername} hfDatasets={hfDatasets} hfLoading={hfLoading} hfError={hfError} onRefreshHf={refreshHf} generating={generatingDataset} generated={datasetGenerated} progress={generationProgress} logs={generationLogs} error={generationError} onGenerate={startDatasetGeneration} onCancel={cancelDatasetGeneration} sshHostSet={!!config.ssh.host} method={lora.method} enableVerification={enableVerification} onEnableVerificationChange={setEnableVerification} bundleWindow={bundleWindow} onBundleWindowChange={setBundleWindow} datasetFormat={datasetFormat} onDatasetFormatChange={setDatasetFormat} /> : null} runName={runName} onRunNameChange={setRunName} lora={lora} onLoraChange={setLora} studentModel={studentModel} onStudentChange={setStudentModel} studentModelOptions={studentModelOptions} hfLoading={hfLoading} hfTokenSet={!!config.hfToken} onRefreshModels={refreshModelPickers} hub={hub} onHubChange={setHub} hfUsername={hfUsername} canLaunch={!!canLaunch} launching={launching} launchError={launchError} onLaunch={launch} validatingDataset={validatingDataset} datasetsValidated={datasetsValidated} trainingOnlyDatasets={trainingOnlyDatasets} hubDatasetValidation={hubDataset.validationResult || {}} onValidateDatasets={openValidateModal} validateButtonRef={validateButtonRef} />}
      </div>

      {showCleanModal && (
        <DatasetCleanModal
          initial={{
            cleanRemoveDuplicates: hubDataset.cleanRemoveDuplicates ?? false,
            cleanRemoveShort: hubDataset.cleanRemoveShort ?? false,
            cleanMinChars: hubDataset.cleanMinChars ?? 30,
          }}
          datasetCount={trainingOnlyDatasets.length}
          onCancel={() => setShowCleanModal(false)}
          onConfirm={confirmCleanAndValidate}
        />
      )}

      <div className="border-t border-white/5 px-8 py-5 flex items-center justify-between bg-white/[0.01] rounded-b-2xl">
        <button disabled={isTrainingOnly || step === 0} onClick={() => setStep(step - 1)} className="flex items-center gap-2 px-6 py-2.5 rounded-xl border border-white/5 theme-surface-soft theme-muted hover:theme-text hover:border-theme-accent/30 disabled:opacity-0 transition-all font-black"><ChevronLeft className="w-5 h-5" /><span className="text-sm-fluid uppercase tracking-widest">Back</span></button>
        <div className="flex items-center gap-4">
          <div className="hidden sm:flex flex-col items-end"><p className="text-[10px] theme-muted uppercase tracking-[0.2em] font-black opacity-40 leading-none mb-1">Configuration Progress</p><p className="text-xs theme-text font-mono font-black">{Math.round(((step + 1) / STEPS.length) * 100)}% COMPLETE</p></div>
          <div className="w-32 h-1.5 bg-white/5 rounded-full overflow-hidden border border-white/5"><div className="h-full theme-accent-bg transition-all duration-500 shadow-[0_0_8px_currentColor]" style={{ width: `${((step + 1) / STEPS.length) * 100}%` }} /></div>
        </div>
        {!isTrainingOnly && step < STEPS.length - 1 ? (
          <button disabled={(step === 1 && !teacherDeployed && !isTrainingOnly) || (step === 2 && (!datasetGenerated && lora.method !== "zrald_offline") && !isTrainingOnly)} onClick={() => setStep(step + 1)} className="flex items-center gap-2 px-8 py-2.5 rounded-xl theme-accent-bg text-black font-black uppercase tracking-widest hover:brightness-110 disabled:opacity-20 transition-all shadow-xl shadow-theme-accent/10 premium-button"><span className="text-sm-fluid">Next Phase</span><ChevronRight className="w-5 h-5" /></button>
        ) : (<div className="w-[140px]" />)}
      </div>
    </div>
  );
}

/** Pop-up shown when the user clicks Validate. Lets them opt into pre-training
 *  dataset cleaning (remove duplicates / remove short paragraphs) so only
 *  high-quality samples are trained on. Choices are saved to the run config and
 *  applied automatically on the GPU server right before training starts. */
function DatasetCleanModal(props: {
  initial: { cleanRemoveDuplicates: boolean; cleanRemoveShort: boolean; cleanMinChars: number };
  datasetCount: number;
  onCancel: () => void;
  onConfirm: (opts: { cleanRemoveDuplicates: boolean; cleanRemoveShort: boolean; cleanMinChars: number }) => void;
}) {
  const [removeDup, setRemoveDup] = useState(props.initial.cleanRemoveDuplicates);
  const [removeShort, setRemoveShort] = useState(props.initial.cleanRemoveShort);
  const [minChars, setMinChars] = useState(props.initial.cleanMinChars || 30);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-6 bg-black/70 backdrop-blur-sm animate-premium" onClick={props.onCancel}>
      <div className="premium-card rounded-2xl w-full max-w-lg overflow-hidden shadow-2xl" onClick={(e) => e.stopPropagation()}>
        <div className="px-6 py-5 border-b border-white/5 bg-white/[0.02] flex items-start justify-between gap-4">
          <div>
            <div className="flex items-center gap-2.5 mb-1">
              <Filter className="w-4 h-4 theme-accent" />
              <h3 className="text-base-fluid font-black text-white tracking-tight">Dataset Quality Filters</h3>
            </div>
            <p className="text-xs-fluid theme-muted font-medium opacity-70 leading-relaxed">
              Applied on the GPU server right before training so the model only learns from high-quality samples. Covers all {props.datasetCount} selected dataset{props.datasetCount === 1 ? "" : "s"}.
            </p>
          </div>
          <button type="button" onClick={props.onCancel} className="p-1.5 rounded-lg border border-white/5 theme-surface-soft theme-muted hover:theme-text transition-all shrink-0">
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-6 space-y-4">
          <label className="flex items-start gap-3 p-4 rounded-xl border border-white/5 bg-white/[0.01] hover:bg-white/[0.03] transition-colors cursor-pointer group">
            <div className={`mt-0.5 w-5 h-5 rounded border-2 transition-all flex items-center justify-center shrink-0 ${removeDup ? "bg-theme-accent border-theme-accent" : "border-white/20 group-hover:border-white/40"}`}>
              {removeDup && <CheckCircle2 className="w-4 h-4 text-black" />}
            </div>
            <input type="checkbox" checked={removeDup} onChange={(e) => setRemoveDup(e.target.checked)} className="hidden" />
            <div>
              <span className="text-sm-fluid theme-text font-black font-mono tracking-tight group-hover:theme-accent transition-colors">Remove duplicate paragraphs</span>
              <p className="text-[11px] theme-muted opacity-70 mt-1 leading-relaxed">Drops exact-duplicate rows so the model isn't over-weighted on repeated samples.</p>
            </div>
          </label>

          <label className="flex items-start gap-3 p-4 rounded-xl border border-white/5 bg-white/[0.01] hover:bg-white/[0.03] transition-colors cursor-pointer group">
            <div className={`mt-0.5 w-5 h-5 rounded border-2 transition-all flex items-center justify-center shrink-0 ${removeShort ? "bg-theme-accent border-theme-accent" : "border-white/20 group-hover:border-white/40"}`}>
              {removeShort && <CheckCircle2 className="w-4 h-4 text-black" />}
            </div>
            <input type="checkbox" checked={removeShort} onChange={(e) => setRemoveShort(e.target.checked)} className="hidden" />
            <div className="flex-1">
              <span className="text-sm-fluid theme-text font-black font-mono tracking-tight group-hover:theme-accent transition-colors">Remove short paragraphs</span>
              <p className="text-[11px] theme-muted opacity-70 mt-1 leading-relaxed">Drops rows whose answer text is below the minimum length, removing low-signal stubs.</p>
            </div>
          </label>

          {removeShort && (
            <div className="flex items-center gap-3 ml-1 animate-premium">
              <label className="text-[10px] uppercase tracking-widest theme-muted font-black">Minimum answer length</label>
              <input
                type="number"
                min={1}
                value={minChars}
                onChange={(e) => setMinChars(parseInt(e.target.value) || 30)}
                className="w-24 px-3 py-2 premium-input rounded-lg text-sm-fluid font-mono text-white focus:outline-none shadow-inner bg-black/20"
              />
              <span className="text-[10px] theme-muted font-mono opacity-60">characters</span>
            </div>
          )}
        </div>

        <div className="px-6 py-4 border-t border-white/5 bg-white/[0.01] flex items-center justify-between gap-3">
          <p className="text-[10px] theme-muted font-mono opacity-50">
            {!removeDup && !removeShort ? "No filters — train on the dataset as-is." : "Filters run before training begins."}
          </p>
          <div className="flex items-center gap-2">
            <button type="button" onClick={props.onCancel} className="px-4 py-2 rounded-xl border border-white/5 theme-surface-soft text-[10px] uppercase tracking-widest theme-muted hover:theme-text font-black transition-all">
              Cancel
            </button>
            <button
              type="button"
              onClick={() => props.onConfirm({ cleanRemoveDuplicates: removeDup, cleanRemoveShort: removeShort, cleanMinChars: minChars })}
              className="flex items-center gap-2 px-5 py-2 rounded-xl theme-accent-bg text-black text-[10px] uppercase tracking-widest font-black hover:brightness-110 transition-all shadow-lg premium-button"
            >
              <ShieldAlert className="w-4 h-4" /> Validate Datasets
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
