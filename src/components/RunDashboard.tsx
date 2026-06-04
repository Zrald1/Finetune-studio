import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { Run, RunStatus, TrainPoint, GPUState } from "../types";
import DatasetPreview, { type PreviewPair } from "./DatasetPreview";
import { api } from "../lib/tauri";
import {
  getStream,
  hydrateFromDisk,
  setLogs as setStoreLogs,
  subscribe,
} from "../lib/runStreams";
import {
  CheckCircle2,
  XCircle,
  Loader2,
  CircleSlash,
  Folder,
  RefreshCw,
  ChevronRight,
  ChevronLeft,
  Play,
  BookOpen,
  Cpu,
  Database,
  GraduationCap,
  Zap,
  Send,
  Upload,
  Trash2,
} from "lucide-react";

interface Props {
  refreshKey: number;
  selectedRunId?: string | null;
  gpuStatus?: GPUState | null;
}

// ── Status config ──────────────────────────────────────────────────────────

const STATUS_COLOR: Record<RunStatus, string> = {
  pending: "theme-faint theme-surface-soft border",
  teacher_loading: "text-amber-400 bg-amber-950/20 border-amber-500/30",
  generating_dataset: "theme-accent theme-accent-soft border",
  dataset_ready: "text-blue-300 bg-blue-950/20 border-blue-500/30",
  training: "text-purple-300 bg-purple-950/20 border-purple-500/30",
  done: "text-emerald-300 bg-emerald-950/20 border-emerald-500/30",
  failed: "text-red-300 bg-red-950/20 border-red-500/30",
  cancelled: "theme-faint theme-surface-soft border",
};

const STATUS_LABEL: Record<RunStatus, string> = {
  pending: "Pending",
  teacher_loading: "Teacher Loading",
  generating_dataset: "Generating",
  dataset_ready: "Dataset Ready",
  training: "Training",
  done: "Done",
  failed: "Failed",
  cancelled: "Cancelled",
};

// Pipeline stages with order for the visual progress bar
const PIPELINE_STAGES: { key: RunStatus | "connecting"; label: string; icon: React.ElementType }[] = [
  { key: "pending", label: "Queued", icon: Zap },
  { key: "teacher_loading", label: "Teacher Boot", icon: Cpu },
  { key: "generating_dataset", label: "Generating", icon: Database },
  { key: "dataset_ready", label: "Dataset Ready", icon: BookOpen },
  { key: "training", label: "Training", icon: GraduationCap },
  { key: "done", label: "Complete", icon: CheckCircle2 },
];

const STAGE_ORDER: Record<string, number> = {
  pending: 0,
  teacher_loading: 1,
  generating_dataset: 2,
  dataset_ready: 3,
  training: 4,
  done: 5,
  failed: -1,
  cancelled: -1,
};

// ── Dashboard root ─────────────────────────────────────────────────────────

export default function RunDashboard({ refreshKey, selectedRunId, gpuStatus }: Props) {
  const [runs, setRuns] = useState<Run[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const reload = async () => {
    setLoading(true);
    try {
      const list = await api.listRuns();
      setRuns(list);
      if (!selectedId && list.length > 0) setSelectedId(list[0].id);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    reload();
  }, [refreshKey]);

  useEffect(() => {
    if (selectedRunId) setSelectedId(selectedRunId);
  }, [selectedRunId]);

  // Auto-refresh while there are non-terminal runs.
  useEffect(() => {
    const hasActive = runs.some(
      (r) => !["done", "failed", "cancelled"].includes(r.status),
    );
    if (!hasActive) return;
    const t = setInterval(reload, 4000);
    return () => clearInterval(t);
  }, [runs]);

  const selected = useMemo(
    () => runs.find((r) => r.id === selectedId) || null,
    [runs, selectedId],
  );

  return (
    <div className="w-full">
      {/* Detail */}
      <section className="w-full">
        {selected ? (
          <RunDetail run={selected} onChanged={reload} gpuStatus={gpuStatus} />
        ) : (
          <div className="theme-surface border rounded-lg p-12 text-center theme-faint italic font-serif">
            {loading ? "Loading run data..." : "No runs available. Launch one from the Pipeline tab."}
          </div>
        )}
      </section>
    </div>
  );
}

// ── Status dot ─────────────────────────────────────────────────────────────

function StatusDot({ status }: { status: RunStatus }) {
  if (status === "done") return <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />;
  if (["failed", "cancelled"].includes(status))
    return <XCircle className="w-4 h-4 text-red-400 shrink-0" />;
  return (
    <span className="relative inline-flex w-3 h-3 shrink-0">
      <span className="animate-ping absolute inline-flex h-full w-full rounded-full theme-accent-bg opacity-60" />
      <span className="relative inline-flex rounded-full h-3 w-3 theme-accent-bg" />
    </span>
  );
}

// ── Pipeline progress bar ──────────────────────────────────────────────────

function PipelineProgressBar({ status }: { status: RunStatus }) {
  const currentIdx = STAGE_ORDER[status] ?? -1;
  const isFailed = status === "failed" || status === "cancelled";

  return (
    <div className="px-5 py-5 border-b theme-surface-soft bg-black/10">
      <div className="flex items-center gap-0">
        {PIPELINE_STAGES.map((stage, i) => {
          const stageIdx = STAGE_ORDER[stage.key] ?? i;
          const isDone = !isFailed && currentIdx > stageIdx;
          const isActive = !isFailed && currentIdx === stageIdx;
          const Icon = stage.icon;

          // Animation logic: Only rotate if it's NOT the terminal 'done' state
          const shouldRotate = isActive && stage.key !== "done";
          const isComplete = isDone || (status === "done" && stage.key === "done");

          return (
            <React.Fragment key={stage.key}>
              <div className="flex flex-col items-center gap-1.5 flex-1 group">
                <div
                  className={`w-9 h-9 rounded-full flex items-center justify-center border-2 transition-all duration-300 ${
                    isComplete
                      ? "bg-emerald-500/20 border-emerald-500/50 text-emerald-400"
                      : isActive
                      ? "bg-theme-accent/20 border-theme-accent text-theme-accent shadow-[0_0_12px_rgba(var(--app-accent-rgb),0.3)]"
                      : "theme-surface-soft border theme-border text-white/20"
                  }`}
                >
                  {shouldRotate ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : isComplete ? (
                    <CheckCircle2 className="w-4 h-4" />
                  ) : (
                    <Icon className="w-4 h-4" />
                  )}
                </div>
                <span
                  className={`text-[9px] uppercase tracking-[0.15em] font-mono font-bold transition-colors ${
                    isComplete
                      ? "text-emerald-400/70"
                      : isActive
                      ? "theme-accent"
                      : "theme-faint"
                  }`}
                >
                  {stage.label}
                </span>
              </div>
              {i < PIPELINE_STAGES.length - 1 && (
                <div
                  className={`h-0.5 flex-1 mx-2 mb-4 transition-colors duration-500 ${
                    isComplete && i < STAGE_ORDER["done"] ? "bg-emerald-500/40" : "bg-white/10"
                  }`}
                />
              )}
            </React.Fragment>
          );
        })}
      </div>
      {isFailed && (
        <p className="text-xs-fluid font-mono text-red-400/80 text-center mt-3 bg-red-950/10 py-1 rounded border border-red-500/10">
          {status === "cancelled" ? "⊘ Pipeline Cancelled" : "✕ Execution Failed — click Resume to retry"}
        </p>
      )}
    </div>
  );
}

// ── Per-topic Q&A breakdown ────────────────────────────────────────────────

function TopicStats({ run }: { run: Run }) {
  const stats = run.topicStats;
  const hasTopics = stats && Object.keys(stats).length > 0;
  const isTrainOnly = run.hubDataset?.trainOnly;
  const isDone = run.status === "done";

  return (
    <div className="space-y-4">
      <p className="text-xs-fluid uppercase tracking-[0.2em] theme-muted font-mono pt-2 font-bold">
        {isTrainOnly ? "Training Config" : "Q&A Breakdown by Domain"}
      </p>
      {hasTopics ? (
        <div className="space-y-2.5">
          {Object.entries(stats!).map(([topic, count]) => {
            const pct = run.qaKept > 0 ? (count / run.qaKept) * 100 : 0;
            return (
              <div key={topic} className="space-y-1">
                <div className="flex items-center justify-between text-[11px] font-mono">
                  <span className="theme-text/75 truncate max-w-[70%]">
                    {topic === "(general)" ? "General Knowledge Base" : topic}
                  </span>
                  <span className="theme-accent font-bold">
                    {count.toLocaleString()}
                  </span>
                </div>
                <div className="h-1.5 theme-surface-soft rounded-full overflow-hidden border">
                  <div
                    className="h-full theme-accent-bg opacity-70 rounded-full transition-all duration-700"
                    style={{ width: `${pct}%` }}
                  />
                </div>
              </div>
            );
          })}
          <div className="flex items-center justify-between pt-2 mt-2 border-t theme-surface-soft">
            <span className="text-xs-fluid font-mono theme-muted uppercase tracking-widest">Aggregate Accepted</span>
            <span className="text-base-fluid font-mono font-bold text-white tabular-nums">{run.qaKept.toLocaleString()}</span>
          </div>
        </div>
      ) : isTrainOnly ? (
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <div className="theme-surface-soft border rounded p-3">
              <div className="text-[8px] uppercase tracking-widest text-theme-accent/60 font-mono mb-1">Student Model</div>
              <div className="text-[10px] font-mono text-white/80 truncate">{run.studentModel}</div>
            </div>
            <div className="theme-surface-soft border rounded p-3">
              <div className="text-[8px] uppercase tracking-widest text-theme-accent/60 font-mono mb-1">Teacher Model</div>
              <div className="text-[10px] font-mono text-white/80 truncate">{run.teacherModel}</div>
            </div>
          </div>
          {run.lora && (
            <div className="theme-surface-soft border rounded p-3">
              <div className="text-[8px] uppercase tracking-widest text-theme-accent/60 font-mono mb-2">LoRA Settings</div>
              <div className="grid grid-cols-3 gap-2 text-[10px] font-mono">
                <div><span className="text-white/50">r:</span> <span className="text-white/80">{run.lora.r}</span></div>
                <div><span className="text-white/50">alpha:</span> <span className="text-white/80">{run.lora.alpha}</span></div>
                <div><span className="text-white/50">lr:</span> <span className="text-white/80">{run.lora.learningRate}</span></div>
                <div><span className="text-white/50">epo:</span> <span className="text-white/80">{run.lora.epochs}</span></div>
                <div><span className="text-white/50">bs:</span> <span className="text-white/80">{run.lora.batchSize}</span></div>
                <div><span className="text-white/50">method:</span> <span className="text-white/80">{run.lora.method}</span></div>
              </div>
            </div>
          )}
          {run.hubDataset?.repoId && (
            <div className="theme-surface-soft border rounded p-3">
              <div className="text-[8px] uppercase tracking-widest text-theme-accent/60 font-mono mb-1">Dataset</div>
              {(run.hubDataset.repoIds && run.hubDataset.repoIds.length > 0 ? run.hubDataset.repoIds : [run.hubDataset.repoId]).map((repo, i) => (
                <div key={i} className="text-[10px] font-mono text-white/80 truncate">{repo}</div>
              ))}
              {run.hubDataset.sampleCount && (
                <div className="text-[9px] font-mono text-white/50 mt-1">{run.hubDataset.sampleCount.toLocaleString()} samples</div>
              )}
            </div>
          )}
          {isDone && run.lastTrainStep && (
            <div className="theme-surface-soft border rounded p-3">
              <div className="text-[8px] uppercase tracking-widest text-emerald-400/60 font-mono mb-1">Final Step</div>
              <div className="text-[14px] font-mono font-black text-emerald-400">{run.lastTrainStep.toLocaleString()}</div>
            </div>
          )}
        </div>
      ) : (
        <div className="theme-surface-soft border rounded p-6 text-center theme-faint italic text-sm-fluid">
          {isDone ? "Training complete" : "Initializing..."}
        </div>
      )}
    </div>
  );
}

// ── Run detail ─────────────────────────────────────────────────────────────

function RunDetail({ run, onChanged, gpuStatus }: { run: Run; onChanged: () => void; gpuStatus?: GPUState | null }) {
  const initial = getStream(run.id);
  const [logs, setLogs] = useState<string>(initial.logs);
  const [progress, setProgress] = useState({
    scanned: initial.progress?.scanned ?? run.qaTotal,
    kept: initial.progress?.kept ?? run.qaKept,
    rejected: initial.progress?.rejected ?? run.qaRejected,
  });
  const [metrics, setMetrics] = useState<TrainPoint[]>(
    initial.metrics.length > 0 ? initial.metrics : run.trainLossHistory || [],
  );
  const [preview, setPreview] = useState<PreviewPair[]>([]);
  const [previewTotal, setPreviewTotal] = useState(0);
  const [previewOffset, setPreviewOffset] = useState(0);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [reloading, setReloading] = useState(false);
  const [testPrompt, setTestPrompt] = useState(
    "What is the composition of the Professional Regulatory Board of Real Estate Service, and who appoints its members?",
  );
  const [testAnswer, setTestAnswer] = useState("");
  const [testing, setTesting] = useState(false);
  const [testError, setTestError] = useState<string | null>(null);
  const [mergeRepo, setMergeRepo] = useState(
    run.hub?.mergedModelId || (run.hub?.modelId ? `${run.hub.modelId}-merged` : ""),
);
  const [merging, setMerging] = useState(false);
  const [mergeResult, setMergeResult] = useState("");
  const [mergeError, setMergeError] = useState<string | null>(null);
  const [ggufRepo, setGgufRepo] = useState(
    run.hub?.ggufRepoId || (run.hub?.modelId ? `${run.hub.modelId}-gguf` : ""),
  );
  const [ggufQuantization, setGgufQuantization] = useState(run.hub?.ggufQuantization || "Q4_K_M");
  const [includeGguf, setIncludeGguf] = useState(run.hub?.autoConvertGguf || false);
  const logBoxRef = useRef<HTMLDivElement>(null);
  const stickToBottomRef = useRef(true);
  const prevLogLenRef = useRef(0);
  const isActive = !["done", "failed", "cancelled"].includes(run.status);

const prevStatusRef = useRef<RunStatus | null>(null);
  const previewPageSize = 5;

  useEffect(() => {
    setLogs(getStream(run.id).logs);
    const unsub = subscribe(run.id, (s) => {
      setLogs(s.logs);
      if (s.progress) setProgress(s.progress);
      if (s.metrics.length > 0) setMetrics(s.metrics);
    });
    const forceRehydrate = prevStatusRef.current === "done" && run.status === "training";
    hydrateFromDisk(run.id, forceRehydrate).catch(() => {});
    if (forceRehydrate) {
      setStoreLogs(run.id, "");
    }
    const current = getStream(run.id);
    if (!current.logs && run.logTail) {
      setStoreLogs(run.id, run.logTail);
    }
    prevStatusRef.current = run.status;
    return () => unsub();
  }, [run.id, run.status]);

  useEffect(() => {
    setProgress({
      scanned: run.qaTotal,
      kept: run.qaKept,
      rejected: run.qaRejected,
    });
    if ((run.trainLossHistory || []).length > 0) {
      setMetrics(run.trainLossHistory || []);
    }
  }, [run.id, run.qaTotal, run.qaKept, run.qaRejected, run.trainLossHistory]);

useEffect(() => {
    setMergeRepo(run.hub?.mergedModelId || (run.hub?.modelId ? `${run.hub.modelId}-merged` : ""));
  }, [run.id, run.hub?.mergedModelId, run.hub?.modelId]);

  useEffect(() => {
    setGgufRepo(run.hub?.ggufRepoId || (run.hub?.modelId ? `${run.hub.modelId}-gguf` : ""));
    setGgufQuantization(run.hub?.ggufQuantization || "Q4_K_M");
    setIncludeGguf(run.hub?.autoConvertGguf || false);
  }, [run.id, run.hub?.ggufRepoId, run.hub?.modelId, run.hub?.ggufQuantization, run.hub?.autoConvertGguf]);

  const syncRemoteWorkerRun = async () => {
    await hydrateFromDisk(run.id, true);
  };

  useLayoutEffect(() => {
    const el = logBoxRef.current;
    if (!el) return;
    if (stickToBottomRef.current || logs.length > prevLogLenRef.current) {
      requestAnimationFrame(() => {
        if (el) el.scrollTop = el.scrollHeight;
      });
    }
    prevLogLenRef.current = logs.length;
  }, [logs]);

  const onLogScroll = () => {
    const el = logBoxRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
    stickToBottomRef.current = atBottom;
  };

  const reloadLog = async () => {
    setReloading(true);
    try {
      await syncRemoteWorkerRun();
      onChanged();
      stickToBottomRef.current = true;
    } finally {
      setReloading(false);
    }
  };

  useEffect(() => {
    setPreviewOffset(0);
  }, [run.id]);

  useEffect(() => {
    let active = true;
    setPreviewLoading(true);
    (async () => {
      try {
        const page = await api.listLocalDatasetPage(run.id, previewOffset, previewPageSize);
        if (!active) return;
        setPreview(page.rows as PreviewPair[]);
        setPreviewTotal(page.total || 0);
      } catch {
        if (!active) return;
        setPreview([]);
        setPreviewTotal(0);
      } finally {
        if (active) setPreviewLoading(false);
      }
    })();
    return () => {
      active = false;
    };
  }, [run.id, run.status, progress.kept, previewOffset]);

  const cancel = async () => {
    await api.cancelRun(run.id);
    onChanged();
  };

  const [resuming, setResuming] = useState(false);
  const resume = async () => {
    setResuming(true);
    try {
      await api.resumeRun(run.id);
      onChanged();
    } catch (e) {
      console.error(e);
    } finally {
      setResuming(false);
    }
  };

  const runModelTest = async () => {
    setTesting(true);
    setTestError(null);
    setTestAnswer("");
    try {
      const answer = await api.testTrainedModel(run.id, testPrompt);
      setTestAnswer(answer);
    } catch (e: any) {
      setTestError(e.message || String(e));
    } finally {
      setTesting(false);
    }
  };

const mergeAndUpload = async () => {
    setMerging(true);
    setMergeError(null);
    setMergeResult("");
    try {
      const result = await api.mergeAndUploadModel(run.id, mergeRepo);
      setMergeResult(result);
      onChanged();
    } catch (e: any) {
      setMergeError(e.message || String(e));
    } finally {
      setMerging(false);
    }
  };

  const [benchRunning, setBenchRunning] = useState(false);
  const [benchResult, setBenchResult] = useState<{total: number; correct: number; partial: number; missed: number; accuracy: number; samples: any[]} | null>(null);
  const [benchError, setBenchError] = useState<string | null>(null);
  const [benchSampleSize, setBenchSampleSize] = useState(100);
  const [benchLogs, setBenchLogs] = useState<string[]>([]);

  const runBenchmark = async () => {
    if (run.status !== "done") return;
    setBenchRunning(true);
    setBenchError(null);
    setBenchResult(null);
    setBenchLogs(["Initializing benchmark...", "Loading adapter from LoRA weights...", "Loading dataset samples..."]);
    try {
      const output = await api.runInferenceBenchmark(run.id, benchSampleSize);
      setBenchLogs((prev) => [...prev, "Running inference on samples..."]);
      const parsed = JSON.parse(output);
      setBenchResult(parsed);
      setBenchLogs((prev) => [...prev, `Benchmark complete: ${parsed.accuracy}% accuracy`]);
    } catch (e: any) {
      setBenchError(e.message || String(e));
      setBenchLogs((prev) => [...prev, `Error: ${e.message}`]);
    } finally {
      setBenchRunning(false);
    }
  };

// Color-code log lines by kind prefix and highlight numbers/progress
  const coloredLogs = useMemo(() => {
    if (!logs) return null;
    return logs.split("\n").map((line, i) => {
      let cls = "text-white/60";
      if (line.startsWith("[ok]")) cls = "text-emerald-400";
      else if (line.startsWith("[stage]")) cls = "text-[#F27D26]";
      else if (line.startsWith("[cmd]")) cls = "text-cyan-400";
      else if (line.startsWith("[FATAL]") || line.startsWith("[error]")) cls = "text-red-400 font-bold";
      else if (line.startsWith("[warn]")) cls = "text-amber-400";
      else if (line.startsWith("[pair")) cls = "text-emerald-300";
      else if (line.startsWith("[skip]") || line.startsWith("[reject]")) cls = "text-white/30";
      else if (line.startsWith("[hf-dataset]")) cls = "text-blue-300";
      else if (line.startsWith("[topic")) cls = "text-purple-300";
      else if (line.startsWith("[vps]")) cls = "text-indigo-300";
      
      // Highlight numbers, progress bars, and percentages
      const coloredLine = line.replace(
        /(\d+\.?\d*%?)|(\[.*?\])|(✓|✗|~)|(━+|▓+░*|█+|░+)|(true|false|null|undefined)/gi,
        (match) => {
          if (match.match(/^\[.*\]$/)) return `<span class="text-yellow-300">${match}</span>`;
          if (match.match(/^\d+\.?\d*%$/)) return `<span class="text-cyan-300 font-bold">${match}</span>`;
          if (match.match(/^\d+\.?\d*$/)) return `<span class="text-emerald-300">${match}</span>`;
          if (match === "✓") return `<span class="text-emerald-400 font-bold">${match}</span>`;
          if (match === "✗") return `<span class="text-red-400 font-bold">${match}</span>`;
          if (match === "~") return `<span class="text-amber-400 font-bold">${match}</span>`;
          if (match.match(/^[━▓█]+$/)) return `<span class="text-emerald-400">${match}</span>`;
          if (match.match(/^[░]+$/)) return `<span class="text-white/30">${match}</span>`;
          if (match === "true") return `<span class="text-emerald-400">${match}</span>`;
          if (match === "false") return `<span class="text-red-400">${match}</span>`;
          if (match === "null" || match === "undefined") return `<span class="text-white/40">${match}</span>`;
          return `<span class="text-cyan-300">${match}</span>`;
        }
      );
      return (
        <span key={i} className={`block ${cls}`} dangerouslySetInnerHTML={{ __html: coloredLine }} />
      );
    });
  }, [logs]);

  return (
    <div className="theme-surface border rounded-lg overflow-hidden flex flex-col h-[calc(100vh-14rem)] min-h-[750px]">
      {/* Header */}
      <div className="px-5 py-3 border-b theme-surface-soft flex items-center justify-between gap-3 shrink-0">
        <div className="min-w-0">
          <h3 className="text-base-fluid font-serif italic text-white truncate">{run.name}</h3>
          <p className="text-[10px] theme-muted font-mono truncate uppercase tracking-wider">
            {run.teacherModel} → {run.studentModel}
          </p>
        </div>
<div className="flex items-center gap-2 shrink-0">
          <span
            className={`px-2 py-1 text-[10px] uppercase tracking-widest font-mono font-bold border rounded ${STATUS_COLOR[run.status]}`}
          >
            {STATUS_LABEL[run.status]}
          </span>
          {gpuStatus && (
            <div className="flex items-center gap-3 px-3 py-2 bg-black/40 border border-white/10 rounded-lg text-[9px] font-mono">
              <div className="flex flex-col items-center">
                <span className="text-[8px] uppercase tracking-widest text-white/40">VRAM</span>
                <span className="text-white font-bold text-[10px]">{Math.round((gpuStatus.memoryUsed / (gpuStatus.memoryTotal || 1)) * 100)}%</span>
                <span className="text-white/50 text-[8px]">{Math.round(gpuStatus.memoryUsed)}/{Math.round(gpuStatus.memoryTotal)}GB</span>
              </div>
              <div className="w-px h-8 bg-white/10" />
              <div className="flex flex-col items-center">
                <span className="text-[8px] uppercase tracking-widest text-white/40">GPU</span>
                <span className="text-white font-bold text-[10px]">{gpuStatus.utilizationGpu}%</span>
                <span className="text-white/50 text-[8px]">{gpuStatus.utilizationGpu >= 100 ? "MAX" : "TENSOR"}</span>
              </div>
              <div className="w-px h-8 bg-white/10" />
              <div className="flex flex-col items-center">
                <span className="text-[8px] uppercase tracking-widest text-white/40">TEMP</span>
                <span className="text-white font-bold text-[10px]">{gpuStatus.temperature}°C</span>
                <span className={`text-[8px] ${gpuStatus.temperature >= 80 ? "text-red-400" : "text-white/50"}`}>{gpuStatus.temperature >= 80 ? "HOT" : "OK"}</span>
              </div>
              <div className="w-px h-8 bg-white/10" />
              <div className="flex flex-col items-center">
                <span className="text-[8px] uppercase tracking-widest text-white/40">POWER</span>
                <span className="text-white font-bold text-[10px]">{Math.round((gpuStatus.powerDraw / (gpuStatus.powerLimit || 1)) * 100)}%</span>
                <span className="text-white/50 text-[8px]">{Math.round(gpuStatus.powerDraw)}W</span>
              </div>
              <span className={`ml-1 w-2 h-2 rounded-full ${gpuStatus.simulated ? "bg-amber-400" : "bg-emerald-400 animate-pulse"}`} />
            </div>
          )}
          {isActive && (
            <button
              onClick={cancel}
              className="flex items-center gap-1 px-3 py-1 bg-red-950/30 border border-red-500/30 text-red-300 rounded text-[10px] uppercase tracking-widest font-mono font-bold hover:bg-red-950 transition"
            >
              <CircleSlash className="w-3 h-3" /> Cancel
            </button>
          )}
          {["failed", "cancelled"].includes(run.status) && (
            <button
              onClick={resume}
              disabled={resuming}
              className="flex items-center gap-1 px-3 py-1 theme-accent-soft theme-accent rounded text-[10px] uppercase tracking-widest font-mono font-bold hover:bg-theme-accent/20 transition disabled:opacity-50"
            >
              {resuming ? <Loader2 className="w-3 h-3 animate-spin" /> : <Play className="w-3 h-3 fill-current" />}
              Resume
            </button>
          )}
        </div>
      </div>

      {/* Visual pipeline progress bar */}
      <div className="shrink-0">
        <PipelineProgressBar status={run.status} />
      </div>

      {/* Runs workspace */}
      <div className="flex-1 min-h-0 grid grid-rows-[minmax(0,1fr)_auto] theme-surface-soft overflow-hidden">
        <div className="min-h-0 grid grid-cols-1 xl:grid-cols-[380px_minmax(0,1fr)] grid-rows-[auto_minmax(0,1fr)] xl:grid-rows-1 overflow-hidden">
          <div className="p-4 space-y-4 border-b xl:border-b-0 xl:border-r theme-surface-soft bg-black/10 overflow-hidden">
            <section className="space-y-3">
              <div className="flex items-center justify-between">
                <p className="text-[10px] uppercase tracking-[0.2em] theme-muted font-mono font-bold">
                  Run Stats
                </p>
                {metrics.length > 0 && (
                  <span className="text-[9px] font-mono text-white/50 bg-white/5 px-2 py-0.5 rounded border border-white/5">
                    STEP {metrics[metrics.length - 1].step}
                  </span>
                )}
              </div>
              <div className="grid grid-cols-3 gap-2">
                <Stat label="Kept" value={progress.kept.toLocaleString()} accent />
                <Stat label="Scanned" value={progress.scanned.toLocaleString()} />
                <Stat label="Fail" value={progress.rejected.toLocaleString()} muted />
              </div>
            </section>

            <section className="space-y-3">
              <div className="flex items-center justify-between">
                <p className="text-[10px] uppercase tracking-[0.2em] theme-accent font-mono font-bold">
                  Learning Curve
                </p>
                <span className="text-[9px] theme-faint font-mono uppercase">
                  {metrics.length > 0 ? `${metrics.length} points` : "No telemetry"}
                </span>
              </div>
              <div className="h-44 min-h-0">
                <LossChart points={metrics} />
              </div>
            </section>

            <section className="theme-surface border rounded p-3">
              <TopicStats run={{ ...run, qaKept: progress.kept || run.qaKept, qaTotal: progress.scanned || run.qaTotal, qaRejected: progress.rejected || run.qaRejected }} />
            </section>
          </div>

          <div className="min-h-0 grid grid-rows-[minmax(0,1fr)_minmax(220px,36%)] divide-y theme-surface-soft">
            <section className="flex flex-col min-h-0">
              <div className="p-3 border-b theme-surface-soft flex items-center justify-between bg-black/10 shrink-0">
                <p className="text-[10px] uppercase tracking-[0.2em] theme-muted font-mono font-bold">
                  Execution Logs
                </p>
                <button
                  onClick={reloadLog}
                  disabled={reloading}
                  className="p-1 rounded theme-faint hover:theme-text hover:theme-surface-soft transition"
                >
                  <RefreshCw className={`w-3 h-3 ${reloading ? "animate-spin" : ""}`} />
                </button>
              </div>
              <div
                ref={logBoxRef}
                onScroll={onLogScroll}
                className="flex-1 min-h-0 bg-black/30 p-4 text-[11px] font-mono leading-relaxed overflow-y-auto selection:bg-theme-selection scrollbar-thin"
              >
                {coloredLogs ?? <span className="theme-faint italic">Awaiting secure session feedback...</span>}
              </div>
              <div className="px-4 py-1 border-t theme-surface-soft bg-black/20 text-center shrink-0">
                <p className="text-[8px] theme-faint font-mono tracking-widest uppercase">
                  SCROLL UP TO PAUSE • SCROLL TO BOTTOM TO TAIL
                </p>
              </div>
            </section>

            <section className="min-h-0 flex flex-col bg-black/20">
              <div className="px-4 py-2 border-b theme-surface-soft flex items-center justify-between gap-3 shrink-0">
                <div>
                  <p className="text-[10px] uppercase tracking-[0.2em] theme-accent font-mono font-bold">
                    Generated Q&A Samples
                  </p>
                  <p className="text-[9px] theme-faint font-mono uppercase tracking-widest">
                    {previewTotal > 0
                      ? `${previewOffset + 1}-${Math.min(previewOffset + previewPageSize, previewTotal)} of ${previewTotal}`
                      : previewLoading
                        ? "Loading samples"
                        : "No accepted pairs yet"}
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setPreviewOffset((o) => Math.max(0, o - previewPageSize))}
                    disabled={previewOffset === 0 || previewLoading}
                    className="p-2 rounded theme-surface-soft border theme-faint hover:theme-text disabled:opacity-25 transition"
                    title="Previous samples"
                  >
                    <ChevronLeft className="w-3.5 h-3.5" />
                  </button>
                  <button
                    onClick={() => setPreviewOffset((o) => Math.min(Math.max(0, previewTotal - previewPageSize), o + previewPageSize))}
                    disabled={previewOffset + previewPageSize >= previewTotal || previewLoading}
                    className="p-2 rounded theme-surface-soft border theme-faint hover:theme-text disabled:opacity-25 transition"
                    title="Next samples"
                  >
                    <ChevronRight className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
              <div className="flex-1 min-h-0 overflow-y-auto p-4 scrollbar-thin">
                {previewLoading ? (
                  <div className="h-24 flex items-center justify-center theme-faint font-mono text-[10px] uppercase tracking-widest">
                    <Loader2 className="w-4 h-4 animate-spin mr-2" />
                    Loading generated samples
                  </div>
                ) : (
                  <DatasetPreview pairs={preview} />
                )}
              </div>
            </section>
          </div>
        </div>

        <div className="border-t theme-surface-soft bg-black/20 p-4 shrink-0">
          <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1.1fr)_minmax(260px,0.8fr)_minmax(360px,1.1fr)] gap-4">
            <section className="theme-surface border rounded p-4 space-y-3 min-w-0">
              <div>
                <p className="text-[10px] uppercase tracking-[0.2em] theme-accent font-mono font-bold">
                  Inference Sandbox
                </p>
                <p className="text-[11px] theme-muted mt-1 leading-relaxed">
                  Test the trained Student LoRA on the live GPU to verify domain knowledge transfer.
                </p>
              </div>
              <div className="grid grid-cols-[minmax(0,1fr)_auto] gap-3 items-end">
                <textarea
                  value={testPrompt}
                  onChange={(e) => setTestPrompt(e.target.value)}
                  rows={3}
                  className="w-full px-3 py-2 theme-field border rounded-lg text-[12px] font-mono text-white/85 resize-none focus:outline-none focus:border-theme-accent transition leading-relaxed shadow-inner"
                />
                <button
                  onClick={runModelTest}
                  disabled={testing || !testPrompt.trim()}
                  className="flex items-center gap-2 px-3 py-2 rounded theme-accent-bg text-black text-[10px] uppercase tracking-widest font-bold hover:brightness-110 disabled:opacity-50 transition shadow-lg"
                >
                  {testing ? <Loader2 className="w-3 h-3 animate-spin" /> : <Send className="w-3 h-3" />}
                  Inference
                </button>
              </div>
              {testAnswer && (
                <div className="bg-black/40 border border-emerald-500/20 rounded-lg p-3 text-[11px] text-emerald-200/85 whitespace-pre-wrap leading-relaxed max-h-20 overflow-hidden">
                  <div className="text-[8px] uppercase tracking-widest text-emerald-400 font-bold mb-1 opacity-50 font-mono">
                    Response Output
                  </div>
                  {testAnswer}
                </div>
              )}
            </section>

            <section className="theme-surface border rounded p-4 space-y-3 min-w-0">
              <div>
                <p className="text-[10px] uppercase tracking-[0.2em] theme-accent font-mono font-bold">
                  Inference Benchmark
                </p>
                <p className="text-[11px] theme-muted mt-1 leading-relaxed">
                  Run {benchSampleSize}-sample evaluation against training dataset.
                </p>
              </div>
              <div className="flex gap-3 items-center">
                <input
                  type="number"
                  min={10}
                  max={500}
                  value={benchSampleSize}
                  onChange={(e) => setBenchSampleSize(Math.max(10, Math.min(500, Number(e.target.value))))}
                  className="w-20 px-3 py-2 theme-field border rounded-lg text-[12px] font-mono text-white focus:outline-none focus:border-theme-accent transition"
                />
                <button
                  onClick={runBenchmark}
                  disabled={benchRunning || run.status !== "done"}
                  className="flex items-center gap-2 px-4 py-2 rounded theme-accent-bg text-black text-[10px] uppercase tracking-widest font-bold hover:brightness-110 disabled:opacity-50 transition shadow-lg whitespace-nowrap"
                >
                  {benchRunning ? <Loader2 className="w-4 h-4 animate-spin" /> : <Cpu className="w-4 h-4" />}
                  {benchRunning ? "Running..." : "Run Benchmark"}
                </button>
              </div>
              {benchResult && (
                <div className="grid grid-cols-4 gap-2">
                  <div className="bg-black/40 border border-emerald-500/20 rounded-lg p-2 text-center">
                    <div className="text-lg font-mono font-black text-emerald-400">{benchResult.accuracy}%</div>
                    <div className="text-[8px] uppercase tracking-widest text-emerald-400/60 font-mono mt-1">Accuracy</div>
                  </div>
                  <div className="bg-black/40 border border-emerald-500/20 rounded-lg p-2 text-center">
                    <div className="text-lg font-mono font-black text-emerald-300">{benchResult.correct}</div>
                    <div className="text-[8px] uppercase tracking-widest text-emerald-400/60 font-mono mt-1">Correct</div>
                  </div>
                  <div className="bg-black/40 border border-amber-500/20 rounded-lg p-2 text-center">
                    <div className="text-lg font-mono font-black text-amber-300">{benchResult.partial}</div>
                    <div className="text-[8px] uppercase tracking-widest text-amber-400/60 font-mono mt-1">Partial</div>
                  </div>
                  <div className="bg-black/40 border border-red-500/20 rounded-lg p-2 text-center">
                    <div className="text-lg font-mono font-black text-red-300">{benchResult.missed}</div>
                    <div className="text-[8px] uppercase tracking-widest text-red-400/60 font-mono mt-1">Missed</div>
                  </div>
                </div>
              )}
              {(benchRunning || benchLogs.length > 0 || benchError) && (
                <div className="bg-black/50 border border-white/10 rounded-lg p-3 overflow-hidden">
                  {benchLogs.slice(-2).map((log, i) => (
                    <div key={i} className="text-[10px] font-mono text-white/70 py-0.5 flex items-center gap-2">
                      <span className="text-theme-accent/60">{">"}</span>
                      <span className="truncate">{log}</span>
                    </div>
                  ))}
                  {benchRunning && (
                    <div className="text-[10px] font-mono theme-accent animate-pulse py-0.5 flex items-center gap-2">
                      <Loader2 className="w-3 h-3 animate-spin" />
                      <span>Processing...</span>
                    </div>
                  )}
                  {benchError && <div className="text-[10px] text-red-400 font-mono truncate">{benchError}</div>}
                </div>
              )}
            </section>

            <section className="theme-surface border rounded p-4 space-y-3 min-w-0">
              <div>
                <p className="text-[10px] uppercase tracking-[0.2em] theme-text/75 font-mono font-bold">
                  Model Consolidation
                </p>
                <p className="text-[11px] theme-muted mt-1 leading-relaxed">
                  Merge the base model with LoRA weights and publish a standalone repository.
                </p>
              </div>
              <div className="grid grid-cols-[minmax(0,1fr)_auto_auto] gap-2 items-center">
                <input
                  value={mergeRepo}
                  onChange={(e) => setMergeRepo(e.target.value)}
                  placeholder="Zrald/GE-Ai-Zrald-2.5-merged"
                  className="min-w-0 px-3 py-2 theme-field border rounded-lg text-[12px] font-mono text-white focus:outline-none focus:border-theme-accent transition"
                />
                <label className="flex items-center gap-2 px-3 py-2 rounded theme-surface border theme-text text-[9px] uppercase tracking-widest font-bold cursor-pointer hover:theme-surface-soft transition shadow-lg">
                  <input type="checkbox" checked={includeGguf} onChange={(e) => setIncludeGguf(e.target.checked)} className="w-3.5 h-3.5 accent-theme-accent" />
                  +GGUF
                </label>
                <button
                  onClick={async () => {
                    setMerging(true);
                    setMergeError(null);
                    setMergeResult("");
                    try {
                      let result: string;
                      if (includeGguf) {
                        const r = await api.mergeConvertUploadModel(run.id, mergeRepo, ggufRepo, ggufQuantization);
                        result = "Merged: " + r.mergedUrl + "\nGGUF: " + r.ggufUrl;
                      } else {
                        result = await api.mergeAndUploadModel(run.id, mergeRepo);
                      }
                      setMergeResult(result);
                      onChanged();
                    } catch (e: any) {
                      setMergeError(e.message || String(e));
                    } finally {
                      setMerging(false);
                    }
                  }}
                  disabled={merging || !mergeRepo.trim()}
                  className="flex items-center justify-center gap-2 px-4 py-2 rounded theme-surface border theme-text text-[10px] uppercase tracking-widest font-bold hover:theme-surface-soft transition shadow-lg whitespace-nowrap disabled:opacity-50"
                >
                  {merging ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-3 h-3" />}
                  Merge & Upload
                </button>
              </div>
              {includeGguf && (
                <div className="grid grid-cols-[minmax(0,1fr)_96px] gap-2">
                  <input
                    value={ggufRepo}
                    onChange={(e) => setGgufRepo(e.target.value)}
                    placeholder="Zrald/GE-Ai-Zrald-2.5-gguf"
                    className="min-w-0 px-3 py-2 theme-field border rounded-lg text-[11px] font-mono text-white focus:outline-none focus:border-theme-accent transition"
                  />
                  <select
                    value={ggufQuantization}
                    onChange={(e) => setGgufQuantization(e.target.value)}
                    className="px-3 py-2 theme-field border rounded-lg text-[10px] font-mono text-white focus:outline-none focus:border-theme-accent transition"
                  >
                    <option value="F16">F16</option>
                    <option value="Q5_K_M">Q5</option>
                    <option value="Q4_K_M">Q4</option>
                    <option value="Q8_0">Q8</option>
                  </select>
                </div>
              )}
              <p className="text-[9px] theme-faint font-mono italic">
                Note: Full model weights can exceed 15GB. LoRA adapters are recommended for rapid iteration.
              </p>
              {mergeResult && (
                <div className="bg-emerald-950/20 border border-emerald-500/30 rounded-lg p-3 text-[11px] text-emerald-300 font-mono whitespace-pre-wrap break-words">
                  {mergeResult}
                </div>
              )}
              {mergeError && (
                <div className="text-[11px] text-red-400 font-mono bg-red-950/30 border border-red-500/20 rounded-lg p-3">
                  {mergeError}
                </div>
              )}
            </section>
          </div>
        </div>
      </div>

      {run.error && (
        <div className="m-4 p-4 rounded-lg border border-red-500/30 bg-red-950/20 text-red-300 text-sm-fluid font-mono whitespace-pre-wrap shadow-xl shrink-0">
          <div className="text-[8px] uppercase tracking-widest text-red-400 font-bold mb-1">Critical Exception</div>
          {run.error}
        </div>
      )}
    </div>
  );
}

// ── Sub-components ─────────────────────────────────────────────────────────

function Stat({ label, value, accent, muted }: { label: string; value: string; accent?: boolean; muted?: boolean }) {
  return (
    <div className="theme-surface border rounded p-3 text-center">
      <p className="text-xs-fluid uppercase tracking-widest theme-muted font-mono">{label}</p>
      <p
        className={`text-xl font-bold font-mono mt-1 ${
          accent ? "theme-accent" : muted ? "theme-faint" : "theme-text"
        }`}
      >
        {value}
      </p>
    </div>
  );
}

function formatEta(seconds?: number | null): string {
  if (!seconds || seconds <= 0) return "calculating";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (h > 0) return `${h}h ${m}m`;
  return `${Math.max(1, m)}m`;
}

function LossChart({ points }: { points: TrainPoint[] }) {
  if (points.length === 0) return (
    <div className="h-full w-full flex items-center justify-center theme-faint italic font-serif border border-dashed rounded-lg">
      Waiting for training telemetry...
    </div>
  );
  
  const w = 1000;
  const h = 240;
  const losses = points.map((p) => p.loss);
  const minL = Math.min(...losses);
  const maxL = Math.max(...losses);
  
  // Padding logic: ensure the line isn't crushed against the edges
  const margin = (maxL - minL) * 0.15 || 0.1; 
  const effectiveMin = minL - margin;
  const effectiveMax = maxL + margin;
  const range = effectiveMax - effectiveMin;
  
  const path = points
    .map((p, i) => {
      const x = (i / Math.max(1, points.length - 1)) * w;
      // Flip Y because SVG 0 is top
      const y = h - ((p.loss - effectiveMin) / range) * h;
      return `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");

  const last = points[points.length - 1];

  return (
    <div className="h-full w-full flex flex-col">
      <div className="flex-1 relative overflow-hidden rounded-lg bg-black/20">
        <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-full" preserveAspectRatio="none">
          {/* Grid lines - horizontal */}
          <line x1="0" y1={h*0.2} x2={w} y2={h*0.2} stroke="currentColor" className="theme-muted opacity-5" strokeWidth="1" />
          <line x1="0" y1={h*0.4} x2={w} y2={h*0.4} stroke="currentColor" className="theme-muted opacity-5" strokeWidth="1" />
          <line x1="0" y1={h*0.6} x2={w} y2={h*0.6} stroke="currentColor" className="theme-muted opacity-5" strokeWidth="1" />
          <line x1="0" y1={h*0.8} x2={w} y2={h*0.8} stroke="currentColor" className="theme-muted opacity-5" strokeWidth="1" />
          
          {/* Neon Gradient */}
          <defs>
            <linearGradient id="neonGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#ff00e5" />
              <stop offset="100%" stopColor="#00f2ff" />
            </linearGradient>
            {/* Simple subtle fill below curve */}
            <linearGradient id="curveFill" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#00f2ff" stopOpacity="0.1" />
              <stop offset="100%" stopColor="#00f2ff" stopOpacity="0" />
            </linearGradient>
          </defs>

          {/* Area fill */}
          <path 
            d={`${path} L ${w} ${h} L 0 ${h} Z`} 
            fill="url(#curveFill)" 
            className="pointer-events-none"
          />
          
          {/* The curve */}
          <path 
            d={path} 
            fill="none" 
            stroke="url(#neonGradient)" 
            strokeWidth="3" 
            strokeLinecap="round" 
            strokeLinejoin="round"
          />
        </svg>
      </div>
      <div className="flex justify-between text-[10px] font-mono theme-faint mt-3 pt-2 border-t border-white/5">
        <div className="flex gap-4">
          <span className="flex items-center gap-1.5"><span className="w-1.5 h-1.5 rounded-full bg-[#ff00e5]" /> START: {points[0].loss.toFixed(3)}</span>
          <span className="flex items-center gap-1.5"><span className="w-1.5 h-1.5 rounded-full bg-[#00f2ff]" /> LATEST: {last.loss.toFixed(3)}</span>
        </div>
        <div className="flex gap-4 uppercase">
          <span className="text-white/70">Epoch {last.epoch.toFixed(2)}</span>
          <span className="theme-accent font-bold">Step {last.step}</span>
        </div>
      </div>
    </div>
  );
}
