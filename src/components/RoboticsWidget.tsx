import React, { useCallback, useEffect, useState } from "react";
import type {
  AppConfig,
  RobotConfig,
  WebResearchConfig,
  RobotCapture,
  ModelManifestStore,
} from "../types";
import { DEFAULT_ROBOT, DEFAULT_WEB_RESEARCH } from "../types";
import { api } from "../lib/tauri";
import {
  Bot,
  Search,
  ShieldCheck,
  RefreshCw,
  Check,
  X,
  Loader2,
  Cpu,
  Globe,
  KeyRound,
  Eye,
  EyeOff,
  ArrowUpCircle,
} from "lucide-react";

interface Props {
  config: AppConfig;
  onChange: (patch: Partial<AppConfig>) => void;
}

const STATUS_COLOR: Record<string, string> = {
  pending: "text-amber-300 bg-amber-500/10 border-amber-500/30",
  researching: "text-sky-300 bg-sky-500/10 border-sky-500/30",
  researched: "text-indigo-300 bg-indigo-500/10 border-indigo-500/30",
  approved: "text-emerald-300 bg-emerald-500/10 border-emerald-500/30",
  rejected: "text-red-300 bg-red-500/10 border-red-500/30",
  failed: "text-red-300 bg-red-500/10 border-red-500/30",
};

function randomToken(): string {
  const bytes = new Uint8Array(24);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

export default function RoboticsWidget({ config, onChange }: Props) {
  const robot: RobotConfig = { ...DEFAULT_ROBOT, ...(config.robot ?? {}) };
  const research: WebResearchConfig = { ...DEFAULT_WEB_RESEARCH, ...(config.webResearch ?? {}) };

  const [captures, setCaptures] = useState<RobotCapture[]>([]);
  const [manifests, setManifests] = useState<ModelManifestStore>({ manifests: [] });
  const [loading, setLoading] = useState(false);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [showRobotTok, setShowRobotTok] = useState(false);
  const [showDashTok, setShowDashTok] = useState(false);
  const [showResearchKey, setShowResearchKey] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const patchRobot = (p: Partial<RobotConfig>) => onChange({ robot: { ...robot, ...p } });
  const patchResearch = (p: Partial<WebResearchConfig>) =>
    onChange({ webResearch: { ...research, ...p } });

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [caps, mans] = await Promise.all([
        api.robotListCaptures(),
        api.robotListManifests(),
      ]);
      setCaptures(caps);
      setManifests(mans);
    } catch (e: any) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const research_ = async (id: string) => {
    setBusyId(id);
    setError(null);
    try {
      await api.robotResearchCapture(id);
      await refresh();
    } catch (e: any) {
      setError(String(e));
    } finally {
      setBusyId(null);
    }
  };
  const approve = async (id: string) => {
    setBusyId(id);
    try {
      await api.robotApproveCapture(id);
      await refresh();
    } catch (e: any) {
      setError(String(e));
    } finally {
      setBusyId(null);
    }
  };
  const reject = async (id: string) => {
    setBusyId(id);
    try {
      await api.robotRejectCapture(id);
      await refresh();
    } catch (e: any) {
      setError(String(e));
    } finally {
      setBusyId(null);
    }
  };
  const promote = async (version: string) => {
    try {
      const store = await api.robotPromoteModel(version);
      setManifests(store);
    } catch (e: any) {
      setError(String(e));
    }
  };

  const field =
    "w-full min-w-0 bg-black/30 border border-white/10 rounded-lg px-3 py-2 text-sm font-mono text-theme-text focus:border-theme-accent outline-none";
  const secretField = `${field} pr-10`;
  const label = "text-[10px] uppercase tracking-widest font-bold theme-muted mb-1 block";
  const tokenLabel = "text-[10px] uppercase tracking-widest font-bold theme-muted";
  const tokenAction =
    "px-2.5 py-1 text-[10px] uppercase tracking-widest rounded-md bg-white/5 border border-white/10 hover:border-theme-accent theme-muted shrink-0";

  return (
    <div className="theme-surface border border-white/10 rounded-2xl p-5 space-y-5 min-w-0 overflow-hidden">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div className="flex items-center gap-3 min-w-0">
          <div className="p-2 rounded-lg bg-theme-accent/15 text-theme-accent shrink-0">
            <Bot className="w-5 h-5" />
          </div>
          <div className="min-w-0">
            <h3 className="font-bold text-theme-text">Robotics Bridge</h3>
            <p className="text-[10px] uppercase tracking-widest theme-muted break-words">
              Robot capture → research → train → pull
            </p>
          </div>
        </div>
        <label className="flex items-center gap-2 text-xs theme-muted cursor-pointer shrink-0">
          <input
            type="checkbox"
            checked={robot.enabled}
            onChange={(e) => patchRobot({ enabled: e.target.checked })}
          />
          Enabled
        </label>
      </div>

      {error && (
        <div className="text-xs font-mono text-red-300 bg-red-500/10 border border-red-500/30 rounded-lg p-2 break-words">
          {error}
        </div>
      )}

      {/* ── Tokens ── */}
      <div className="space-y-3">
        <div className="flex items-center gap-2 text-xs font-bold theme-muted uppercase tracking-widest">
          <KeyRound className="w-3.5 h-3.5" /> API Tokens
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div className="min-w-0">
            <div className="mb-1 flex min-h-6 items-center justify-between gap-3">
              <span className={tokenLabel}>Robot API token</span>
              <button
                type="button"
                className={tokenAction}
                onClick={() => patchRobot({ robotApiToken: randomToken() })}
              >
                Generate
              </button>
            </div>
            <div className="relative">
              <input
                className={secretField}
                type={showRobotTok ? "text" : "password"}
                value={robot.robotApiToken}
                placeholder="bearer token the robot presents"
                onChange={(e) => patchRobot({ robotApiToken: e.target.value })}
              />
              <button
                type="button"
                className="absolute right-2 top-1/2 -translate-y-1/2 theme-muted"
                onClick={() => setShowRobotTok((s) => !s)}
              >
                {showRobotTok ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
          </div>
          <div className="min-w-0">
            <div className="mb-1 flex min-h-6 items-center justify-between gap-3">
              <span className={tokenLabel}>Dashboard API token</span>
              <button
                type="button"
                className={tokenAction}
                onClick={() => patchRobot({ dashboardApiToken: randomToken() })}
              >
                Generate
              </button>
            </div>
            <div className="relative">
              <input
                className={secretField}
                type={showDashTok ? "text" : "password"}
                value={robot.dashboardApiToken}
                placeholder="bearer token the desktop/dashboard presents"
                onChange={(e) => patchRobot({ dashboardApiToken: e.target.value })}
              />
              <button
                type="button"
                className="absolute right-2 top-1/2 -translate-y-1/2 theme-muted"
                onClick={() => setShowDashTok((s) => !s)}
              >
                {showDashTok ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* ── Capture / privacy ── */}
      <div className="space-y-3">
        <div className="flex items-center gap-2 text-xs font-bold theme-muted uppercase tracking-widest">
          <ShieldCheck className="w-3.5 h-3.5" /> Capture &amp; Privacy
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div>
            <span className={label}>Min confidence</span>
            <input
              className={field}
              type="number"
              step="0.05"
              min="0"
              max="1"
              value={robot.minCaptureConfidence}
              onChange={(e) => patchRobot({ minCaptureConfidence: parseFloat(e.target.value) || 0 })}
            />
          </div>
          <div>
            <span className={label}>Dedupe window (s)</span>
            <input
              className={field}
              type="number"
              min="0"
              value={robot.dedupeWindowSecs}
              onChange={(e) => patchRobot({ dedupeWindowSecs: parseInt(e.target.value) || 0 })}
            />
          </div>
          <div>
            <span className={label}>Research collection</span>
            <input
              className={field}
              value={robot.researchCollection}
              onChange={(e) => patchRobot({ researchCollection: e.target.value })}
            />
          </div>
          <div className="flex flex-col justify-end gap-2 text-xs theme-muted">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={robot.blurFacesPlates}
                onChange={(e) => patchRobot({ blurFacesPlates: e.target.checked })}
              />
              Blur faces / plates
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={robot.autoResearchOnCapture}
                onChange={(e) => patchRobot({ autoResearchOnCapture: e.target.checked })}
              />
              Auto-research on capture
            </label>
          </div>
        </div>
      </div>

      {/* ── Web research ── */}
      <div className="space-y-3">
        <div className="flex items-center gap-2 text-xs font-bold theme-muted uppercase tracking-widest">
          <Globe className="w-3.5 h-3.5" /> Web Research
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div>
            <span className={label}>Provider</span>
            <select
              className={field}
              value={research.provider}
              onChange={(e) => patchResearch({ provider: e.target.value as WebResearchConfig["provider"] })}
            >
              <option value="brave">Brave Search</option>
              <option value="serpapi">SerpAPI (Google)</option>
              <option value="google_cse">Google CSE</option>
            </select>
          </div>
          <div>
            <span className={label}>Max results</span>
            <input
              className={field}
              type="number"
              min="1"
              max="10"
              value={research.maxResults}
              onChange={(e) => patchResearch({ maxResults: parseInt(e.target.value) || 5 })}
            />
          </div>
        </div>
        <div>
          <span className={label}>API key</span>
          <div className="relative">
            <input
              className={secretField}
              type={showResearchKey ? "text" : "password"}
              value={research.apiKey}
              onChange={(e) => patchResearch({ apiKey: e.target.value })}
            />
            <button
              type="button"
              className="absolute right-2 top-1/2 -translate-y-1/2 theme-muted"
              onClick={() => setShowResearchKey((s) => !s)}
            >
              {showResearchKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
            </button>
          </div>
        </div>
        {research.provider === "google_cse" && (
          <div>
            <span className={label}>Google CSE id</span>
            <input
              className={field}
              value={research.cseId ?? ""}
              onChange={(e) => patchResearch({ cseId: e.target.value })}
            />
          </div>
        )}
        <div>
          <span className={label}>Domain allowlist (comma-separated, empty = all)</span>
          <input
            className={field}
            value={research.domainAllowlist.join(", ")}
            placeholder="wikipedia.org, britannica.com"
            onChange={(e) =>
              patchResearch({
                domainAllowlist: e.target.value
                  .split(",")
                  .map((s) => s.trim())
                  .filter(Boolean),
              })
            }
          />
        </div>
        <label className="flex items-center gap-2 text-xs theme-muted cursor-pointer">
          <input
            type="checkbox"
            checked={research.blockDangerousTopics}
            onChange={(e) => patchResearch({ blockDangerousTopics: e.target.checked })}
          />
          Block dangerous topics
        </label>
      </div>

      {/* ── Model pull ── */}
      <div className="space-y-2">
        <div className="flex items-center gap-2 text-xs font-bold theme-muted uppercase tracking-widest">
          <Cpu className="w-3.5 h-3.5" /> Robot Model Pull
        </div>
        <div className="text-xs font-mono theme-muted">
          Current served version:{" "}
          <span className="text-theme-accent">{manifests.currentVersion ?? "none published"}</span>
        </div>
        {manifests.manifests.length > 0 && (
          <div className="space-y-1 max-h-32 overflow-y-auto scrollbar-thin">
            {manifests.manifests.map((m) => (
              <div
                key={m.version}
                className="flex items-center justify-between text-xs bg-black/20 border border-white/10 rounded-lg px-3 py-1.5"
              >
                <div className="font-mono truncate">
                  <span className="text-theme-text">{m.version}</span>{" "}
                  <span className="theme-muted">{m.hfRepo}</span>
                </div>
                {manifests.currentVersion !== m.version && (
                  <button
                    className="flex items-center gap-1 text-theme-accent hover:underline"
                    onClick={() => promote(m.version)}
                  >
                    <ArrowUpCircle className="w-3.5 h-3.5" /> Serve
                  </button>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ── Capture queue ── */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-xs font-bold theme-muted uppercase tracking-widest">
            <Search className="w-3.5 h-3.5" /> Capture Queue ({captures.length})
          </div>
          <button
            className="flex items-center gap-1 text-xs theme-muted hover:text-theme-accent"
            onClick={refresh}
            disabled={loading}
          >
            {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <RefreshCw className="w-3.5 h-3.5" />}
            Refresh
          </button>
        </div>
        <div className="space-y-2 max-h-80 overflow-y-auto scrollbar-thin">
          {captures.length === 0 && (
            <div className="text-xs theme-muted italic py-4 text-center">
              No captures yet. The robot POSTs images to <code>/robot/capture</code>.
            </div>
          )}
          {captures.map((c) => (
            <div key={c.id} className="bg-black/20 border border-white/10 rounded-xl p-3 space-y-2">
              <div className="flex items-center justify-between">
                <div className="font-mono text-xs text-theme-text truncate">
                  {c.labelGuess || "unidentified"}{" "}
                  <span className="theme-muted">· {(c.confidence * 100).toFixed(0)}%</span>
                </div>
                <span
                  className={`text-[9px] uppercase tracking-widest font-bold px-2 py-0.5 rounded border ${
                    STATUS_COLOR[c.status] ?? ""
                  }`}
                >
                  {c.status}
                </span>
              </div>
              {c.ocrText && (
                <div className="text-[10px] font-mono theme-muted line-clamp-2">OCR: {c.ocrText}</div>
              )}
              {c.citations.length > 0 && (
                <div className="text-[10px] theme-muted">
                  {c.citations.length} source(s) · {c.chunksIngested} chunks embedded
                </div>
              )}
              {c.error && <div className="text-[10px] text-red-300 break-words">{c.error}</div>}
              <div className="flex items-center gap-2">
                {(c.status === "pending" || c.status === "failed") && (
                  <button
                    className="flex items-center gap-1 text-[11px] px-2 py-1 rounded-lg bg-sky-500/10 border border-sky-500/30 text-sky-300 disabled:opacity-50"
                    disabled={busyId === c.id}
                    onClick={() => research_(c.id)}
                  >
                    {busyId === c.id ? <Loader2 className="w-3 h-3 animate-spin" /> : <Search className="w-3 h-3" />}
                    Research
                  </button>
                )}
                {c.status === "researched" && (
                  <button
                    className="flex items-center gap-1 text-[11px] px-2 py-1 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 disabled:opacity-50"
                    disabled={busyId === c.id}
                    onClick={() => approve(c.id)}
                  >
                    <Check className="w-3 h-3" /> Approve for training
                  </button>
                )}
                {c.status !== "rejected" && (
                  <button
                    className="flex items-center gap-1 text-[11px] px-2 py-1 rounded-lg bg-red-500/10 border border-red-500/30 text-red-300 disabled:opacity-50"
                    disabled={busyId === c.id}
                    onClick={() => reject(c.id)}
                  >
                    <X className="w-3 h-3" /> Reject
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
