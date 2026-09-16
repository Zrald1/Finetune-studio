import { useCallback, useEffect, useMemo, useState } from "react";
import type { BenchKind, BenchRecord } from "../types";
import { api } from "../lib/tauri";
import { save as saveFileDialog } from "@tauri-apps/plugin-dialog";
import {
  ArrowDownRight,
  ArrowUpRight,
  BarChart3,
  CheckCircle2,
  Cpu,
  FileDown,
  GraduationCap,
  Loader2,
  Minus,
  RefreshCw,
  Rocket,
  Trash2,
} from "lucide-react";

const KIND_META: Record<BenchKind, { label: string; icon: React.ElementType; className: string }> = {
  teacher: { label: "Teacher", icon: Cpu, className: "text-cyan-300 border-cyan-500/30 bg-cyan-500/5" },
  student: { label: "Student", icon: GraduationCap, className: "text-violet-300 border-violet-500/30 bg-violet-500/5" },
  deployment: { label: "Deployment", icon: Rocket, className: "text-amber-300 border-amber-500/30 bg-amber-500/5" },
};

function num(v: number | null | undefined, decimals = 1, suffix = "") {
  return typeof v === "number" && Number.isFinite(v) ? `${v.toFixed(decimals)}${suffix}` : "—";
}

function fmtDate(raw: string) {
  const d = new Date(raw);
  return Number.isNaN(d.getTime()) ? raw : d.toLocaleString();
}

/** Percent change from `base` to `now`, or null when it cannot be computed. */
function pctChange(base?: number | null, now?: number | null): number | null {
  if (typeof base !== "number" || typeof now !== "number") return null;
  if (!Number.isFinite(base) || !Number.isFinite(now) || Math.abs(base) < 1e-9) return null;
  return ((now - base) / base) * 100;
}

/**
 * Render a delta with a direction indicator.
 *
 * `lowerIsBetter` flips the colour: a -70% TTFT is an improvement, while a -70%
 * throughput would be a regression. Getting this backwards would make the
 * report actively misleading, so the caller must state the polarity.
 */
function Delta({
  base,
  now,
  lowerIsBetter,
}: {
  base?: number | null;
  now?: number | null;
  lowerIsBetter: boolean;
}) {
  const pct = pctChange(base, now);
  if (pct === null) return <span className="theme-faint font-mono">—</span>;
  if (Math.abs(pct) < 0.05) {
    return (
      <span className="theme-faint font-mono inline-flex items-center gap-1">
        <Minus className="w-3 h-3" /> 0.0%
      </span>
    );
  }
  const improved = lowerIsBetter ? pct < 0 : pct > 0;
  const Icon = pct > 0 ? ArrowUpRight : ArrowDownRight;
  return (
    <span
      className={`font-mono font-black inline-flex items-center gap-1 ${
        improved ? "text-emerald-400" : "text-red-400"
      }`}
    >
      <Icon className="w-3 h-3" />
      {pct > 0 ? "+" : ""}
      {pct.toFixed(1)}%
    </span>
  );
}

interface Group {
  key: string;
  kind: BenchKind;
  model: string;
  records: BenchRecord[];
}

export default function BenchmarksPanel() {
  const [records, setRecords] = useState<BenchRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [filter, setFilter] = useState<BenchKind | "all">("all");

  const reload = useCallback(async () => {
    setLoading(true);
    try {
      setRecords(await api.benchList());
    } catch (e: any) {
      setMessage(String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void reload();
  }, [reload]);

  const visible = useMemo(
    () => (filter === "all" ? records : records.filter((r) => r.kind === filter)),
    [records, filter],
  );

  /**
   * Group by kind + model, oldest first. Comparing the first and last entry of
   * a group is the whole point of the page: it shows what an optimization
   * change actually bought for that specific model.
   */
  const groups = useMemo<Group[]>(() => {
    const map = new Map<string, Group>();
    for (const r of visible) {
      const key = `${r.kind}::${r.model}`;
      if (!map.has(key)) map.set(key, { key, kind: r.kind, model: r.model, records: [] });
      map.get(key)!.records.push(r);
    }
    for (const g of map.values()) {
      g.records.sort((a, b) => Date.parse(a.capturedAt) - Date.parse(b.capturedAt));
    }
    return Array.from(map.values());
  }, [visible]);

  const exportPdf = async () => {
    setExporting(true);
    setMessage(null);
    try {
      const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
      const dest = await saveFileDialog({
        defaultPath: `fine-tune-benchmarks-${stamp}.pdf`,
        filters: [{ name: "PDF", extensions: ["pdf"] }],
      });
      if (!dest) return;
      const written = await api.benchExportPdf(dest);
      setMessage(`Report written to ${written}`);
    } catch (e: any) {
      setMessage(String(e));
    } finally {
      setExporting(false);
    }
  };

  const removeRecord = async (id: string) => {
    try {
      await api.benchDelete(id);
      await reload();
    } catch (e: any) {
      setMessage(String(e));
    }
  };

  const clearAll = async () => {
    try {
      await api.benchClear();
      await reload();
      setMessage("Benchmark history cleared.");
    } catch (e: any) {
      setMessage(String(e));
    }
  };

  return (
    <div className="w-full max-w-6xl mx-auto space-y-6">
      <div className="premium-card rounded-2xl overflow-hidden">
        <div className="px-8 py-6 border-b theme-surface-soft bg-white/[0.02] flex items-start justify-between gap-4 flex-wrap">
          <div>
            <p className="text-[10px] uppercase tracking-[0.3em] theme-accent font-black font-mono">
              Benchmark History
            </p>
            <h2 className="text-2xl-fluid font-serif italic text-white tracking-tight font-black mt-1">
              Model Benchmarks
            </h2>
            <p className="text-sm-fluid theme-muted font-medium opacity-80 mt-2 max-w-2xl">
              Every benchmark you run is stored with the configuration it ran under, so a Standard
              deployment can be compared against an Optimized one for the same model.
            </p>
          </div>
          <div className="flex items-center gap-2 flex-wrap">
            <button
              type="button"
              onClick={reload}
              disabled={loading}
              className="px-4 py-2.5 rounded-xl border border-white/10 theme-surface-soft theme-text hover:border-theme-accent/40 text-[10px] uppercase tracking-widest font-black disabled:opacity-30 flex items-center gap-2"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
              Refresh
            </button>
            <button
              type="button"
              onClick={exportPdf}
              disabled={exporting || records.length === 0}
              className="px-4 py-2.5 rounded-xl theme-accent-bg text-black text-[10px] uppercase tracking-widest font-black disabled:opacity-30 flex items-center gap-2 premium-button"
            >
              {exporting ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <FileDown className="w-3.5 h-3.5" />}
              Export PDF
            </button>
          </div>
        </div>

        <div className="px-8 py-4 border-b border-white/5 flex items-center gap-2 flex-wrap">
          {(["all", "teacher", "student", "deployment"] as const).map((k) => {
            const count = k === "all" ? records.length : records.filter((r) => r.kind === k).length;
            const active = filter === k;
            return (
              <button
                key={k}
                type="button"
                onClick={() => setFilter(k)}
                className={`px-3.5 py-2 rounded-xl border text-[10px] uppercase tracking-widest font-black transition-all ${
                  active
                    ? "theme-accent-soft theme-accent border-theme-accent/40"
                    : "border-white/10 theme-muted hover:theme-text"
                }`}
              >
                {k === "all" ? "All" : KIND_META[k].label} ({count})
              </button>
            );
          })}
          {records.length > 0 && (
            <button
              type="button"
              onClick={clearAll}
              className="ml-auto text-[9px] uppercase tracking-widest theme-muted font-black hover:text-red-400 flex items-center gap-1.5"
            >
              <Trash2 className="w-3 h-3" /> Clear history
            </button>
          )}
        </div>

        {message && (
          <div className="px-8 py-3 text-[10px] font-mono theme-accent border-b border-white/5 break-all">{message}</div>
        )}

        <div className="p-8 space-y-8">
          {records.length === 0 && !loading && (
            <div className="p-10 text-center theme-muted font-serif italic rounded-2xl border border-white/5 bg-white/[0.015]">
              No benchmarks yet. Run <span className="theme-accent font-black not-italic">Benchmark</span> on the
              Teacher step, the Deploy page, or a completed run in the Runs tab — results land here.
            </div>
          )}

          {/* ── Comparison: earliest vs latest per model ── */}
          {groups.some((g) => g.records.length > 1) && (
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <BarChart3 className="w-4 h-4 theme-accent" />
                <span className="text-[10px] uppercase tracking-[0.2em] theme-accent font-black font-mono">
                  Optimization comparison
                </span>
                <span className="text-[9px] theme-faint font-mono">
                  earliest vs latest run of the same model
                </span>
              </div>
              <div className="overflow-x-auto rounded-2xl border border-white/5">
                <table className="w-full text-[11px] font-mono">
                  <thead>
                    <tr className="text-[9px] uppercase tracking-widest theme-muted bg-white/[0.02]">
                      <th className="text-left font-black px-4 py-3">Model</th>
                      <th className="text-left font-black px-3 py-3">Config change</th>
                      <th className="text-right font-black px-3 py-3">TTFT</th>
                      <th className="text-right font-black px-3 py-3">Throughput</th>
                      <th className="text-right font-black px-3 py-3">TPOT</th>
                      <th className="text-right font-black px-3 py-3">Accuracy</th>
                    </tr>
                  </thead>
                  <tbody>
                    {groups
                      .filter((g) => g.records.length > 1)
                      .map((g) => {
                        const first = g.records[0];
                        const last = g.records[g.records.length - 1];
                        const meta = KIND_META[g.kind];
                        const Icon = meta.icon;
                        return (
                          <tr key={g.key} className="border-t border-white/5">
                            <td className="px-4 py-3">
                              <div className="flex items-center gap-2">
                                <Icon className={`w-3.5 h-3.5 ${meta.className.split(" ")[0]}`} />
                                <div className="min-w-0">
                                  <div className="text-white font-black truncate">
                                    {g.model.split("/").pop()}
                                  </div>
                                  <div className="text-[9px] theme-faint uppercase">{meta.label}</div>
                                </div>
                              </div>
                            </td>
                            <td className="px-3 py-3 theme-muted">
                              {first.label} <span className="theme-faint">→</span> {last.label}
                            </td>
                            <td className="px-3 py-3 text-right">
                              <Delta base={first.metrics.ttftMs} now={last.metrics.ttftMs} lowerIsBetter />
                            </td>
                            <td className="px-3 py-3 text-right">
                              <Delta
                                base={first.metrics.outputTokensPerS}
                                now={last.metrics.outputTokensPerS}
                                lowerIsBetter={false}
                              />
                            </td>
                            <td className="px-3 py-3 text-right">
                              <Delta base={first.metrics.tpotMs} now={last.metrics.tpotMs} lowerIsBetter />
                            </td>
                            <td className="px-3 py-3 text-right">
                              <Delta base={first.metrics.accuracy} now={last.metrics.accuracy} lowerIsBetter={false} />
                            </td>
                          </tr>
                        );
                      })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* ── Full history ── */}
          {visible.length > 0 && (
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 theme-accent" />
                <span className="text-[10px] uppercase tracking-[0.2em] theme-accent font-black font-mono">
                  All runs
                </span>
                <span className="text-[9px] theme-faint font-mono">{visible.length} record(s)</span>
              </div>
              <div className="space-y-2">
                {[...visible].reverse().map((r) => {
                  const meta = KIND_META[r.kind] ?? KIND_META.deployment;
                  const Icon = meta.icon;
                  const cfg = [
                    r.config.servingProfile && `profile=${r.config.servingProfile}`,
                    r.config.dtype && `dtype=${r.config.dtype}`,
                    r.config.maxModelLen && `ctx=${r.config.maxModelLen}`,
                    r.config.gpuMemoryUtilization && `vram=${r.config.gpuMemoryUtilization}`,
                    r.config.maxNumSeqs && `seqs=${r.config.maxNumSeqs}`,
                    r.config.kvCacheDtype && `kv=${r.config.kvCacheDtype}`,
                    r.config.prefixCaching != null && `prefix=${r.config.prefixCaching ? "on" : "off"}`,
                    r.config.quantization && `quant=${r.config.quantization}`,
                    r.config.reasoningEffort && `think=${r.config.reasoningEffort}`,
                    ...(r.config.notes ?? []),
                  ].filter(Boolean) as string[];

                  return (
                    <div
                      key={r.id}
                      className="rounded-2xl border border-white/5 bg-white/[0.015] px-5 py-4 space-y-3"
                    >
                      <div className="flex items-start justify-between gap-3 flex-wrap">
                        <div className="flex items-center gap-3 min-w-0">
                          <span className={`px-2.5 py-1 rounded-lg border text-[9px] uppercase tracking-widest font-black flex items-center gap-1.5 ${meta.className}`}>
                            <Icon className="w-3 h-3" />
                            {meta.label}
                          </span>
                          <div className="min-w-0">
                            <div className="text-white font-black text-sm truncate">{r.label}</div>
                            <div className="text-[9px] theme-faint font-mono uppercase truncate">
                              {r.model} · {fmtDate(r.capturedAt)}
                            </div>
                          </div>
                        </div>
                        <button
                          type="button"
                          onClick={() => removeRecord(r.id)}
                          title="Delete this record"
                          className="theme-faint hover:text-red-400 transition-colors shrink-0"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>

                      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
                        {[
                          { l: "TTFT", v: num(r.metrics.ttftMs, 0, " ms") },
                          { l: "TPOT", v: num(r.metrics.tpotMs, 1, " ms") },
                          { l: "ITL", v: num(r.metrics.itlMs, 1, " ms") },
                          { l: "Out tok/s", v: num(r.metrics.outputTokensPerS, 1) },
                          { l: "Total tok/s", v: num(r.metrics.totalTokensPerS, 1) },
                          { l: "Accuracy", v: num(r.metrics.accuracy, 2, " %") },
                        ].map((m) => (
                          <div key={m.l} className="rounded-xl border border-white/5 bg-black/20 px-3 py-2">
                            <div className="text-[8px] uppercase tracking-[0.2em] theme-muted font-black">{m.l}</div>
                            <div className="text-[13px] font-black font-mono text-white tabular-nums mt-0.5">{m.v}</div>
                          </div>
                        ))}
                      </div>

                      {cfg.length > 0 && (
                        <div className="flex flex-wrap gap-1.5">
                          {cfg.map((c, i) => (
                            <span
                              key={i}
                              className="text-[9px] font-mono px-2 py-0.5 rounded border border-white/10 theme-muted"
                            >
                              {c}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
