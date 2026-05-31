import React, { useRef, useState, useEffect } from "react";
import type { AppConfig, ConnectionStatus, PaddleOcrConfig, QdrantConfig, SSHConfig, AIAgentConfig, EmbeddingConfig, EmbedderConfig } from "../types";
import { DEFAULT_AI_AGENT, DEFAULT_DIGITAL_OCEAN, POPULAR_MODELS, DEFAULT_EMBEDDING, DEFAULT_EMBEDDER } from "../types";
import { CheckCircle2, Cloud, Container, Database, Download, Eye, EyeOff, Key, Loader2, LockKeyhole, RefreshCw, Save, Upload, Sparkles, Plus, Trash2, Server, Circle, ScanText, X, ChevronLeft, ChevronRight, FileText } from "lucide-react";
import { api, events } from "../lib/tauri";
import { open as openFileDialog } from "@tauri-apps/plugin-dialog";

interface Props {
  config: AppConfig;
  onChange: (patch: Partial<AppConfig>) => void;
  connection: ConnectionStatus;
  onTestConnection: () => Promise<void>;
}

export default function CredentialsPanel({
  config,
  onChange,
  connection,
  onTestConnection,
}: Props) {
  const [showPass, setShowPass] = useState(false);
  const [showKey, setShowKey] = useState(false);
  const [showHf, setShowHf] = useState(false);
  const [showFeather, setShowFeather] = useState(false);
  const [showQd, setShowQd] = useState(false);
  const [showDocker, setShowDocker] = useState(false);
  const [showDoKey, setShowDoKey] = useState(false);
  const [showCopilotConfig, setShowCopilotConfig] = useState(false);
  const [showCopilotKey, setShowCopilotKey] = useState(false);
  const [showEmbedding, setShowEmbedding] = useState(false);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [loadingModels, setLoadingModels] = useState(false);
  const [drag, setDrag] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const [settingUpEmbedders, setSettingUpEmbedders] = useState(false);
  const [embedderStatuses, setEmbedderStatuses] = useState<Record<number, "idle" | "booting" | "running" | "error">>({});
  const [bootingOcr, setBootingOcr] = useState(false);
  const [ocrStatus, setOcrStatus] = useState<"idle" | "booting" | "running" | "error">("idle");
  const [ocrLogs, setOcrLogs] = useState("");

  const [showQdrantDb, setShowQdrantDb] = useState(false);
  const [qdrantCollections, setQdrantCollections] = useState<{ name: string; status: string; vectors_count: number }[]>([]);
  const [loadingCollections, setLoadingCollections] = useState(false);
  const [qdrantSnapshots, setQdrantSnapshots] = useState<any[]>([]);
  const [loadingSnapshots, setLoadingSnapshots] = useState(false);
  const [downloadingSnapshot, setDownloadingSnapshot] = useState<string | null>(null);
  const [selectedCollection, setSelectedCollection] = useState("");
  const [snapshotSaving, setSnapshotSaving] = useState(false);
  const [snapshotUploading, setSnapshotUploading] = useState(false);
  const [snapshotStatus, setSnapshotStatus] = useState<string | null>(null);
  const [snapshotIsError, setSnapshotIsError] = useState(false);
  const [savingAll, setSavingAll] = useState(false);
  const [downloadingAll, setDownloadingAll] = useState(false);
  const [chunks, setChunks] = useState<any[]>([]);
  const [currentOffset, setCurrentOffset] = useState<any>(null);
  const [nextOffset, setNextOffset] = useState<any>(null);
  const [offsetsHistory, setOffsetsHistory] = useState<any[]>([]);
  const [loadingChunks, setLoadingChunks] = useState(false);
  const [chunksError, setChunksError] = useState<string | null>(null);

  const ssh = config.ssh;
  const qd = config.qdrant;
  const docker = config.docker;
  const digitalOcean = { ...DEFAULT_DIGITAL_OCEAN, ...(config.digitalOcean ?? {}) };

  const patchSsh = (p: Partial<SSHConfig>) =>
    onChange({ ssh: { ...ssh, ...p } });
  const patchQd = (p: Partial<QdrantConfig>) =>
    onChange({ qdrant: { ...qd, ...p } });
  const patchDocker = (p: Partial<typeof docker>) =>
    onChange({ docker: { ...docker, ...p } });

  const paddleOcr: PaddleOcrConfig = { enabled: false, port: 8118, modelName: "PaddleOCR-VL-1.6-0.9B", dockerImage: "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-amd-gpu", ...config.paddleOcr };
  const patchPaddleOcr = (p: Partial<PaddleOcrConfig>) =>
    onChange({ paddleOcr: { ...paddleOcr, ...p } });

  const embedding = { ...DEFAULT_EMBEDDING, ...(config.embedding ?? {}) };

  const patchEmbedding = (p: Partial<EmbeddingConfig>) => {
    onChange({ embedding: { ...embedding, ...p } });
  };

  const handleEmbeddingProviderChange = (provider: EmbeddingConfig["provider"]) => {
    const defaults: Record<EmbeddingConfig["provider"], Partial<EmbeddingConfig>> = {
      vllm: { provider: "vllm", apiUrl: "", apiKey: "", modelId: "Qwen/Qwen3-Embedding-8B" },
      ollama: { provider: "ollama", apiUrl: "http://localhost:11434", apiKey: "", modelId: "" },
      llamacpp: { provider: "llamacpp", apiUrl: "http://localhost:8080", apiKey: "", modelId: "" },
    };
    patchEmbedding(defaults[provider]);
  };

  const [detectingOllama, setDetectingOllama] = useState(false);
  const [detectedOllamaModels, setDetectedOllamaModels] = useState<string[]>([]);
  const [detectingLlamaCpp, setDetectingLlamaCpp] = useState(false);
  const [detectedLlamaCppModels, setDetectedLlamaCppModels] = useState<string[]>([]);

  const detectLocalModels = async () => {
    if (embedding.provider === "ollama" && embedding.apiUrl) {
      setDetectingOllama(true);
      try {
        const res = await fetch(`${embedding.apiUrl}/api/tags`);
        if (res.ok) {
          const data = await res.json();
          setDetectedOllamaModels((data.models || []).map((m: { name: string }) => m.name));
        }
      } catch { setDetectedOllamaModels([]); }
      finally { setDetectingOllama(false); }
    } else if (embedding.provider === "llamacpp" && embedding.apiUrl) {
      setDetectingLlamaCpp(true);
      try {
        const res = await fetch(`${embedding.apiUrl}/v1/models`);
        if (res.ok) {
          const data = await res.json();
          setDetectedLlamaCppModels((data.data || []).map((m: { id: string }) => m.id));
        }
      } catch { setDetectedLlamaCppModels([]); }
      finally { setDetectingLlamaCpp(false); }
    }
  };

  useEffect(() => { if (embedding.provider !== "vllm" && embedding.apiUrl) { detectLocalModels(); } }, [embedding.provider]);

  const embeddingModels = embedding.provider === "ollama" ? detectedOllamaModels : embedding.provider === "llamacpp" ? detectedLlamaCppModels : ["Qwen/Qwen3-Embedding-8B"];

  const embedders: EmbedderConfig[] = config.embedders ?? [];

  const addEmbedder = () => {
    const newIdx = embedders.length;
    onChange({
      embedders: [
        ...embedders,
        { ...DEFAULT_EMBEDDER, name: `embedder_${newIdx + 1}`, port: 8101 + newIdx }
      ]
    });
  };

  const updateEmbedder = (idx: number, patch: Partial<EmbedderConfig>) => {
    const updated = embedders.map((e, i) => i === idx ? { ...e, ...patch } : e);
    onChange({ embedders: updated });
  };

  const removeEmbedder = (idx: number) => {
    if (idx === 0) return;
    onChange({ embedders: embedders.filter((_, i) => i !== idx) });
  };

  const setupAllEmbedders = async () => {
    if (embedders.length === 0) return;
    setSettingUpEmbedders(true);
    const statuses: Record<number, "booting" | "running" | "error"> = {};
    embedders.forEach((_, i) => { statuses[i] = "booting"; });
    setEmbedderStatuses(statuses);
    try {
      const results = await api.serveSetupAllEmbedders(
        config.ssh, config.docker, embedders, config.hfToken ?? null, config.paddleOcr ?? null
      );
      const final: Record<number, "running" | "error"> = {};
      results.forEach((r, i) => {
        final[i] = r.status === "already_running" || r.status === "booted" || r.status === "existing_embeddings" ? "running" : "error";
      });
      setEmbedderStatuses(final);
    } catch (e) {
      embedders.forEach((_, i) => { statuses[i] = "error"; });
      setEmbedderStatuses(statuses);
    } finally {
      setSettingUpEmbedders(false);
    }
  };

  const loadQdrantCollections = async () => {
    setLoadingCollections(true);
    try {
      const cols = await api.qdrantListCollections(qd);
      setQdrantCollections(cols);
    } catch (e) {
      console.error("list collections:", e);
    } finally {
      setLoadingCollections(false);
    }
  };

  const loadQdrantSnapshots = async (collection: string) => {
    setLoadingSnapshots(true);
    try {
      const snaps = await api.qdrantListSnapshots(qd, collection);
      setQdrantSnapshots(snaps);
    } catch (e) {
      console.error("list snapshots:", e);
    } finally {
      setLoadingSnapshots(false);
    }
  };

  const loadChunks = async (offsetVal: any = null) => {
    if (!qd.endpoint || !selectedCollection) return;
    setLoadingChunks(true);
    setChunksError(null);
    try {
      const res = await api.qdrantScrollInCollection(qd, selectedCollection, 3, offsetVal);
      setChunks(res.chunks || []);
      setNextOffset(res.next_offset || null);
    } catch (e: any) {
      const msg = e.message || String(e);
      if (/doesn'?t exist|not found|404/i.test(msg)) {
        setChunks([]);
        setNextOffset(null);
      } else {
        console.error("Failed to load chunks:", e);
        setChunksError(msg);
        setChunks([]);
        setNextOffset(null);
      }
    } finally {
      setLoadingChunks(false);
    }
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

  useEffect(() => {
    if (selectedCollection && qd.endpoint) {
      loadQdrantSnapshots(selectedCollection);
      setCurrentOffset(null);
      setOffsetsHistory([]);
      loadChunks(null);
    } else {
      setChunks([]);
      setNextOffset(null);
      setOffsetsHistory([]);
    }
  }, [selectedCollection, qd.endpoint]);



  const downloadSnapshot = async (collection: string, name: string) => {
    setDownloadingSnapshot(name);
    try {
      const { save } = await import("@tauri-apps/plugin-dialog");
      const dest = await save({
        defaultPath: name,
        filters: [{ name: "Snapshot", extensions: ["tar", "snapshot"] }],
      });
      if (dest) {
        await api.qdrantDownloadSnapshot(qd, collection, name, dest);
      }
    } catch (e) {
      console.error("download snapshot:", e);
    } finally {
      setDownloadingSnapshot(null);
    }
  };

  const saveSnapshot = async () => {
    if (!selectedCollection) return;
    setSnapshotSaving(true);
    setSnapshotStatus(null);
    try {
      await api.qdrantCreateSnapshot(qd, selectedCollection);
      setSnapshotStatus(`Snapshot saved for ${selectedCollection}`);
      setSnapshotIsError(false);
      loadQdrantSnapshots(selectedCollection);
    } catch (e: any) {
      setSnapshotStatus(e.message || "Failed to save snapshot");
      setSnapshotIsError(true);
    } finally {
      setSnapshotSaving(false);
    }
  };

  const uploadSnapshotDb = async () => {
    if (!selectedCollection) return;
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const files = await open({
        multiple: false,
        filters: [{ name: "Snapshot", extensions: ["tar", "snapshot"] }],
      });
      if (!files) return;
      const filePath = files as string;
      setSnapshotUploading(true);
      setSnapshotStatus(null);
      await api.qdrantUploadSnapshot(qd, selectedCollection, filePath);
      setSnapshotStatus(`Snapshot uploaded to ${selectedCollection}`);
      setSnapshotIsError(false);
      loadQdrantSnapshots(selectedCollection);
    } catch (e: any) {
      setSnapshotStatus(e.message || "Failed to upload snapshot");
      setSnapshotIsError(true);
    } finally {
      setSnapshotUploading(false);
    }
  };

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
  };

  const saveAllSnapshots = async () => {
    setSavingAll(true);
    setSnapshotStatus(null);
    try {
      const results = await api.createAllQdrantSnapshots(qd);
      const ok = results.filter((r) => !r.snapshot_name.startsWith("ERROR"));
      const fail = results.filter((r) => r.snapshot_name.startsWith("ERROR"));
      if (fail.length > 0) {
        setSnapshotStatus(`Saved ${ok.length}/${results.length} collections. ${fail.length} failed.`);
        setSnapshotIsError(true);
      } else {
        setSnapshotStatus(`Saved snapshots for all ${ok.length} collections`);
        setSnapshotIsError(false);
      }
      if (selectedCollection) loadQdrantSnapshots(selectedCollection);
    } catch (e: any) {
      setSnapshotStatus(e.message || "Failed to save all snapshots");
      setSnapshotIsError(true);
    } finally {
      setSavingAll(false);
    }
  };

  const downloadAllSnapshots = async () => {
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const dir = await open({ directory: true, multiple: false });
      if (!dir) return;
      const dirPath = dir as string;
      setDownloadingAll(true);
      setSnapshotStatus(null);
      const paths = await api.downloadAllQdrantSnapshots(qd, dirPath);
      setSnapshotStatus(`Downloaded ${paths.length} snapshots to ${dirPath}`);
      setSnapshotIsError(false);
    } catch (e: any) {
      setSnapshotStatus(e.message || "Failed to download all snapshots");
      setSnapshotIsError(true);
    } finally {
      setDownloadingAll(false);
    }
  };

  const provider = config.aiAgent?.provider ?? "vllm";
  const staticModels = POPULAR_MODELS[provider] || [];
  const displayModels = Array.from(new Set([...staticModels, ...availableModels]));

  // Debounced model list fetching
  useEffect(() => {
    const provider = config.aiAgent?.provider ?? "vllm";
    const apiUrl = config.aiAgent?.apiUrl ?? "";
    const apiKey = config.aiAgent?.apiKey ?? "";
    
    if (!apiKey || !apiUrl) {
      setAvailableModels([]);
      return;
    }

    const timer = setTimeout(async () => {
      setLoadingModels(true);
      try {
        // Route through the Rust backend — fetching provider APIs directly from
        // the browser is blocked by CORS.
        const list = await api.aiListModels(provider, apiUrl, apiKey);
        if (Array.isArray(list) && list.length > 0) {
          setAvailableModels(list);
        }
      } catch (e) {
        console.error("Fetch models error:", e);
      } finally {
        setLoadingModels(false);
      }
    }, 800);

    return () => clearTimeout(timer);
  }, [config.aiAgent?.provider, config.aiAgent?.apiUrl, config.aiAgent?.apiKey]);



  const bootPaddleOcr = async () => {
    setBootingOcr(true);
    setOcrStatus("booting");
    setOcrLogs("");
    const unlisten = await events.onSetupLog(({ line }) => {
      setOcrLogs(prev => prev + line);
    });
    try {
      await api.serveBootPaddleocr(config.ssh, docker, paddleOcr);
      setOcrStatus("running");
    } catch (e: any) {
      console.error("boot paddleocr:", e);
      setOcrStatus("error");
      setOcrLogs(prev => prev + `[error] ${e}\n`);
    } finally {
      unlisten();
      setBootingOcr(false);
    }
  };

  const patchAiAgent = (p: Partial<AIAgentConfig>) => {
    const current = config.aiAgent ?? DEFAULT_AI_AGENT;
    const next = { ...current, ...p };
    if (p.provider && p.provider !== current.provider) {
      if (p.provider === "openai") {
        next.apiUrl = "https://api.openai.com/v1";
        next.modelId = "gpt-5.5-instant";
      } else if (p.provider === "anthropic") {
        next.apiUrl = "https://api.anthropic.com/v1";
        next.modelId = "claude-sonnet-4-6";
      } else if (p.provider === "gemini") {
        next.apiUrl = "https://generativelanguage.googleapis.com/v1beta/openai";
        next.modelId = "gemini-3.5-flash";
      } else if (p.provider === "groq") {
        next.apiUrl = "https://api.groq.com/openai/v1";
        next.modelId = "llama-3.3-70b-versatile";
      } else if (p.provider === "xai") {
        next.apiUrl = "https://api.x.ai/v1";
        next.modelId = "grok-4.3";
      } else if (p.provider === "vultr") {
        next.apiUrl = "https://api.vultrinference.com/v1";
        next.modelId = "deepseek-chat";
      } else if (p.provider === "custom") {
        next.apiUrl = next.apiUrl || "";
        next.modelId = next.modelId || "";
      }
    }
    onChange({ aiAgent: next });
  };

  const handleBrowseKey = async () => {
    try {
      const selected = await openFileDialog({
        multiple: false,
        directory: false,
        filters: [{ name: "Private Key", extensions: ["pem", "key", "*"] }]
      });
      if (selected && typeof selected === "string") {
        const content = await api.readLocalFileText(selected);
        patchSsh({
          privateKeyPath: selected,
          privateKey: content
        });
        setShowKey(true);
      }
    } catch (err) {
      console.error("Error choosing/reading key file:", err);
    }
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDrag(false);
    const f = e.dataTransfer.files?.[0];
    if (!f) return;
    const r = new FileReader();
    r.onload = (ev) => {
      const text = ev.target?.result as string;
      if (text) {
        patchSsh({ privateKey: text, privateKeyPath: undefined });
        setShowKey(true);
      }
    };
    r.readAsText(f);
  };

  return (
    <div className="xl:col-span-2 premium-card rounded-3xl overflow-hidden animate-premium relative">
      {/* Accent rail across the top of the whole panel */}
      <div className="absolute top-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-theme-accent/40 to-transparent" />
      <div className="px-8 py-7 border-b border-white/5 flex items-start justify-between gap-4 bg-gradient-to-br from-white/[0.04] via-white/[0.015] to-transparent backdrop-blur-md">
        <div className="space-y-2">
          <div className="flex items-center gap-2.5">
            <div className="w-1.5 h-6 theme-accent-bg rounded-full shadow-[0_0_12px_rgba(var(--app-accent-rgb),0.6)]" />
            <p className="text-[10px] uppercase tracking-[0.35em] theme-accent font-black font-mono">
              Access Control
            </p>
          </div>
          <h2 className="text-2xl-fluid font-serif italic text-white tracking-tight font-black leading-none">
            System Credentials
          </h2>
          <p className="text-sm-fluid theme-muted font-medium opacity-70">
            SSH, vector store, and model provider settings
          </p>
        </div>
        <div className={`shrink-0 inline-flex items-center gap-2.5 rounded-full border px-4 py-2 text-[10px] uppercase tracking-widest font-black font-mono transition-all duration-500 shadow-lg backdrop-blur-md ${
          connection.isConnected
            ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-400 shadow-emerald-500/10"
            : "theme-surface-soft theme-muted border-white/5"
        }`}>
          <span className={`w-2 h-2 rounded-full shadow-[0_0_8px_currentColor] ${connection.isConnected ? "bg-emerald-400 animate-pulse" : "theme-faint"}`} />
          {connection.isConnected ? "System Ready" : "Disconnected"}
        </div>
      </div>

      {/* Body: responsive two-column dashboard of credential cards */}
      <div className="p-6 sm:p-8 grid grid-cols-1 xl:grid-cols-2 gap-6 items-start">
      {/* SSH */}
      <div className="space-y-5 rounded-2xl border border-white/5 bg-white/[0.015] p-6 shadow-inner transition-colors hover:border-white/10 xl:row-span-2">
        <div className="flex items-center gap-3 pb-3 border-b border-white/5">
          <div className="w-10 h-10 rounded-xl theme-accent-soft theme-accent flex items-center justify-center glass-panel shadow-inner group transition-transform duration-300 hover:scale-105">
            <LockKeyhole className="w-5 h-5 group-hover:rotate-12 transition-transform" />
          </div>
          <div>
            <h3 className="text-base-fluid text-white font-black tracking-tight">Primary GPU SSH</h3>
            <p className="text-xs-fluid theme-muted font-mono uppercase tracking-wider opacity-60">Secure remote host access</p>
          </div>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="sm:col-span-2 space-y-2">
            <label className="text-[10px] uppercase tracking-widest theme-muted font-black ml-1">
              GPU Droplet IP
            </label>
            <input
              type="text"
              placeholder="0.0.0.0"
              value={ssh.host}
              onChange={(e) => patchSsh({ host: e.target.value })}
              className="w-full px-4 py-3 premium-input rounded-xl text-sm-fluid font-mono placeholder-white/10 focus:outline-none shadow-inner"
            />
          </div>
          <div className="space-y-2">
            <label className="text-[10px] uppercase tracking-widest theme-muted font-black ml-1">
              User
            </label>
            <input
              type="text"
              placeholder="root"
              value={ssh.username}
              onChange={(e) => patchSsh({ username: e.target.value })}
              className="w-full px-4 py-3 premium-input rounded-xl text-sm-fluid font-mono placeholder-white/10 focus:outline-none shadow-inner"
            />
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between ml-1">
            <label className="text-[10px] uppercase tracking-widest theme-muted font-black flex items-center gap-2">
              <Key className="w-3.5 h-3.5 theme-accent" /> SSH Private Key
            </label>
            <button
              type="button"
              onClick={() => setShowKey(!showKey)}
              className="px-3 py-1.5 rounded-lg border border-white/5 theme-surface-soft text-[10px] uppercase tracking-widest theme-muted hover:theme-text hover:border-theme-accent/30 font-black transition-all"
            >
              {showKey ? "Hide" : "Paste"}
            </button>
          </div>
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDrag(true);
            }}
            onDragLeave={() => setDrag(false)}
            onDrop={onDrop}
            onClick={handleBrowseKey}
            className={`cursor-pointer border-2 border-dashed rounded-2xl p-6 text-center transition-all duration-300 flex flex-col items-center justify-center space-y-2 min-h-[120px] glass-panel hover:bg-white/[0.02] group ${
              drag
                ? "border-theme-accent theme-accent-soft scale-[1.01]"
                : ssh.privateKey
                ? "border-emerald-500/30 bg-emerald-500/[0.02] shadow-[inset_0_0_20px_rgba(16,185,129,0.05)]"
                : "border-white/5 hover:border-white/10"
            }`}
          >
            <div className={`w-12 h-12 rounded-full flex items-center justify-center transition-colors duration-300 ${ssh.privateKey ? "bg-emerald-500/10 text-emerald-400" : "bg-white/5 theme-faint group-hover:bg-white/10"}`}>
              <Upload className={`w-6 h-6 transition ${ssh.privateKey ? "animate-bounce" : ""}`} />
            </div>
            <div className="text-sm-fluid">
              {ssh.privateKey ? (
                <div className="flex flex-col items-center gap-1">
                  <span className="inline-flex items-center gap-2 text-emerald-400 font-black font-mono">
                    <CheckCircle2 className="w-4 h-4" /> SECURE KEY LOADED
                  </span>
                  {ssh.privateKeyPath && (
                    <span className="text-[10px] theme-muted font-mono max-w-md truncate">
                      {ssh.privateKeyPath}
                    </span>
                  )}
                </div>
              ) : (
                <span className="theme-muted font-medium">
                  Drop key file or <span className="theme-accent font-black border-b border-theme-accent/30">click to browse</span>
                </span>
              )}
            </div>
          </div>
          {showKey && (
            <textarea
              rows={6}
              placeholder="-----BEGIN OPENSSH PRIVATE KEY-----"
              value={ssh.privateKey || ""}
              onChange={(e) => patchSsh({ privateKey: e.target.value })}
              className="w-full px-4 py-4 premium-input rounded-2xl text-sm-fluid font-mono placeholder-white/10 leading-relaxed resize-none focus:outline-none shadow-inner"
            />
          )}
        </div>

        <div className="space-y-2">
          <label className="text-[10px] uppercase tracking-widest theme-muted font-black ml-1">
            SSH Password <span className="opacity-40 font-medium">(Optional)</span>
          </label>
          <div className="relative group">
            <input
              type={showPass ? "text" : "password"}
              placeholder="••••••••"
              value={ssh.password || ""}
              onChange={(e) => patchSsh({ password: e.target.value })}
              className="w-full pl-4 pr-12 py-3 premium-input rounded-xl text-sm-fluid font-mono placeholder-white/10 focus:outline-none shadow-inner"
            />
            <button
              type="button"
              onClick={() => setShowPass(!showPass)}
              className="absolute right-4 top-1/2 -translate-y-1/2 theme-faint hover:theme-text transition-colors"
            >
              {showPass ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
            </button>
          </div>
        </div>
      </div>

      {/* DigitalOcean */}
      <div className="space-y-4 rounded-2xl border border-white/5 bg-white/[0.015] p-6 shadow-inner transition-colors hover:border-white/10">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl theme-surface-soft theme-muted flex items-center justify-center glass-panel shadow-inner transition-transform duration-300 hover:scale-105 border border-white/5">
            <Cloud className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base-fluid text-white font-black tracking-tight">DigitalOcean</h3>
            <p className="text-xs-fluid theme-muted font-mono uppercase tracking-wider opacity-60">API key only; GPU servers are managed in the GPU Servers tab</p>
          </div>
        </div>
        <div className="space-y-2">
          <div className="flex items-center justify-between ml-1">
            <label className="text-[10px] uppercase tracking-widest theme-muted font-black">API Token</label>
            <button
              type="button"
              onClick={() => setShowDoKey(!showDoKey)}
              className="text-[10px] uppercase tracking-widest theme-accent font-black hover:brightness-125"
            >
              {showDoKey ? "Mask" : "Unmask"}
            </button>
          </div>
          <input
            type={showDoKey ? "text" : "password"}
            placeholder="dop_v1_..."
            value={digitalOcean.apiKey}
            onChange={(e) => onChange({ digitalOcean: { ...digitalOcean, apiKey: e.target.value } })}
            className="w-full px-4 py-3 premium-input rounded-xl text-sm-fluid font-mono placeholder-white/10 focus:outline-none shadow-inner"
          />
        </div>
      </div>

      {/* Qdrant */}
      <div className="space-y-5 rounded-2xl border border-white/5 bg-white/[0.015] p-6 shadow-inner transition-colors hover:border-white/10">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl theme-accent-soft theme-accent flex items-center justify-center glass-panel shadow-inner transition-transform duration-300 hover:scale-105">
              <Database className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base-fluid text-white font-black tracking-tight">Qdrant</h3>
              <p className="text-xs-fluid theme-muted font-mono uppercase tracking-wider opacity-60">Knowledge-base collection</p>
            </div>
          </div>
          <button
            type="button"
            onClick={() => setShowQd(!showQd)}
            className="px-3 py-1.5 rounded-lg border border-white/5 theme-surface-soft text-[10px] uppercase tracking-widest theme-muted hover:theme-text transition-all font-black"
          >
            {showQd ? "Hide Config" : "Show Params"}
          </button>
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between ml-1">
            <label className="text-[10px] uppercase tracking-widest theme-muted font-black">Endpoint</label>
            {ssh.host && (
              <button
                type="button"
                onClick={() => patchQd({ endpoint: `http://${ssh.host}:6333` })}
                className="text-[9px] uppercase tracking-wider px-2 py-0.5 rounded-md bg-blue-500/20 text-blue-300 hover:bg-blue-500/30 transition-colors"
                title={`Set to http://${ssh.host}:6333`}
              >
                Use SSH Host
              </button>
            )}
          </div>
          <input
            type="text"
            placeholder={ssh.host ? `http://${ssh.host}:6333` : "https://xxx.qdrant.io"}
            value={qd.endpoint}
            onChange={(e) => patchQd({ endpoint: e.target.value })}
            className="w-full px-4 py-3 premium-input rounded-xl text-sm-fluid font-mono placeholder-white/10 focus:outline-none shadow-inner"
          />
          {qd.endpoint && ssh.host && !qd.endpoint.includes(ssh.host) && (
            <p className="text-[10px] text-amber-400 ml-1">⚠ Endpoint host differs from SSH host ({ssh.host}). Click "Use SSH Host" to fix.</p>
          )}
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="space-y-2">
            <label className="text-[10px] uppercase tracking-widest theme-muted font-black ml-1">API Key</label>
            <input
              type={showQd ? "text" : "password"}
              placeholder="eyJ…"
              value={qd.apiKey}
              onChange={(e) => patchQd({ apiKey: e.target.value })}
              className="w-full px-4 py-3 premium-input rounded-xl text-sm-fluid font-mono placeholder-white/10 focus:outline-none shadow-inner"
            />
          </div>
          <div className="space-y-2">
            <label className="text-[10px] uppercase tracking-widest theme-muted font-black ml-1">Collection</label>
            <input
              type="text"
              placeholder="collection name"
              value={qd.collection}
              onChange={(e) => patchQd({ collection: e.target.value })}
              className="w-full px-4 py-3 premium-input rounded-xl text-sm-fluid font-mono placeholder-white/10 focus:outline-none shadow-inner"
            />
          </div>
        </div>
      </div>

      {/* Qdrant Database Manager */}
      {qd.endpoint && (
        <div className="xl:col-span-2 space-y-5 rounded-2xl border border-white/5 bg-white/[0.015] p-6 shadow-inner transition-colors hover:border-white/10">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl theme-accent-soft theme-accent flex items-center justify-center glass-panel shadow-inner transition-transform duration-300 hover:scale-105">
                <Database className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-base-fluid text-white font-black tracking-tight">Database Manager</h3>
                <p className="text-xs-fluid theme-muted font-mono uppercase tracking-wider opacity-60">Collections, snapshots & restore</p>
              </div>
            </div>
            <button
              type="button"
              onClick={() => setShowQdrantDb(!showQdrantDb)}
              className="px-3 py-1.5 rounded-lg border border-white/5 theme-surface-soft text-[10px] uppercase tracking-widest theme-muted hover:theme-text transition-all font-black"
            >
              {showQdrantDb ? "Hide" : "Manage"}
            </button>
          </div>

          {showQdrantDb && (
            <div className="space-y-4">
              {/* Collection Select */}
              <div className="space-y-2">
                <label className="text-[10px] uppercase tracking-widest theme-muted font-black ml-1">Collection</label>
                <div className="flex gap-2">
                  <select
                    value={selectedCollection}
                    onChange={(e) => {
                      setSelectedCollection(e.target.value);
                      patchQd({ collection: e.target.value });
                    }}
                    className="flex-1 px-4 py-3 premium-input rounded-xl text-sm-fluid font-mono bg-black/20 focus:outline-none shadow-inner border border-white/5 text-white"
                  >
                    {qdrantCollections.length === 0 && <option value="">No collections found</option>}
                    {qdrantCollections.map((c) => (
                      <option key={c.name} value={c.name}>{c.name}</option>
                    ))}
                  </select>
                  <button onClick={loadQdrantCollections} disabled={loadingCollections} className="p-3 rounded-xl bg-white/5 border border-white/10 hover:bg-white/10 transition-all disabled:opacity-30">
                    <RefreshCw className={`w-4 h-4 ${loadingCollections ? "animate-spin" : ""} theme-muted`} />
                  </button>
                </div>
              </div>

              {/* Snapshot Actions */}
              <div className="flex gap-2 flex-wrap">
                <button
                  onClick={saveSnapshot}
                  disabled={snapshotSaving || !selectedCollection}
                  className="flex items-center gap-2 px-4 py-2 theme-accent-bg text-black text-[10px] uppercase tracking-widest font-black rounded-xl hover:brightness-125 disabled:opacity-20 shadow-lg premium-button transition-all"
                >
                  <Save className="w-3.5 h-3.5" />
                  {snapshotSaving ? "Saving..." : "Save Snapshot"}
                </button>
                <button
                  onClick={uploadSnapshotDb}
                  disabled={snapshotUploading || !selectedCollection}
                  className="flex items-center gap-2 px-4 py-2 bg-white/5 border border-white/10 text-white text-[10px] uppercase tracking-widest font-black rounded-xl hover:bg-white/10 disabled:opacity-20 transition-all"
                >
                  {snapshotUploading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Upload className="w-3.5 h-3.5" />}
                  {snapshotUploading ? "Uploading..." : "Upload Snapshot"}
                </button>
                <button
                  onClick={saveAllSnapshots}
                  disabled={savingAll}
                  className="flex items-center gap-2 px-4 py-2 bg-blue-500/10 border border-blue-500/20 text-blue-400 text-[10px] uppercase tracking-widest font-black rounded-xl hover:bg-blue-500 hover:text-black disabled:opacity-20 transition-all"
                >
                  <Save className="w-3.5 h-3.5" />
                  {savingAll ? "Saving All..." : "Save All"}
                </button>
                <button
                  onClick={downloadAllSnapshots}
                  disabled={downloadingAll}
                  className="flex items-center gap-2 px-4 py-2 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-[10px] uppercase tracking-widest font-black rounded-xl hover:bg-emerald-500 hover:text-black disabled:opacity-20 transition-all"
                >
                  <Download className="w-3.5 h-3.5" />
                  {downloadingAll ? "Downloading All..." : "Download All"}
                </button>
              </div>

              {snapshotStatus && (
                <div className={`p-3 rounded-lg text-[10px] font-mono ${snapshotIsError ? "border border-red-500/20 bg-red-500/5 text-red-300" : "border border-emerald-500/20 bg-emerald-500/5 text-emerald-300"}`}>
                  {snapshotIsError ? <span className="font-black uppercase">Error: </span> : <span className="font-black uppercase">OK: </span>}
                  {snapshotStatus}
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
                          <span className="truncate max-w-[200px]" title={c.file_name || c.file_path}>
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
                    <p className="text-[10px] font-mono theme-muted italic opacity-35">No chunks found in this collection. Ingest some files first.</p>
                  </div>
                )}
              </div>

              {/* Snapshots List */}
              {qdrantSnapshots.length > 0 ? (
                <div className="space-y-2">
                  <p className="text-[8px] uppercase tracking-widest theme-muted font-black opacity-40 flex items-center gap-2">
                    <Database className="w-3 h-3" /> Snapshots
                    <button onClick={() => loadQdrantSnapshots(selectedCollection)} className="p-1 rounded hover:bg-white/10 transition-all" title="Refresh">
                      <RefreshCw className="w-3 h-3" />
                    </button>
                  </p>
                  {qdrantSnapshots.map((s) => (
                    <div key={s.name} className="flex items-center justify-between bg-black/30 rounded-lg px-4 py-3 border border-white/5">
                      <div className="flex items-center gap-3 min-w-0">
                        <Database className="w-4 h-4 theme-muted opacity-40 shrink-0" />
                        <div className="min-w-0">
                          <p className="text-[10px] font-black font-mono text-white truncate">{s.name}</p>
                          <p className="text-[8px] font-mono theme-muted opacity-50">{formatBytes(s.size)} · {s.creation_time?.split("T")[0] || "—"}</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2 shrink-0">
                        <button onClick={() => downloadSnapshot(selectedCollection, s.name)} disabled={downloadingSnapshot === s.name} className="px-3 py-1.5 rounded-lg bg-blue-500/10 border border-blue-500/20 text-blue-400 text-[9px] font-black uppercase hover:bg-blue-500 hover:text-black transition-all disabled:opacity-30" title="Download snapshot">
                          <Download className="w-3 h-3" />
                        </button>
                        <button onClick={async () => {
                          try {
                            setSnapshotStatus(`Restoring ${s.name}...`);
                            setSnapshotIsError(false);
                            await api.qdrantRestoreSnapshot(qd, selectedCollection, s.name);
                            setSnapshotStatus(`Restored: ${s.name}`);
                          } catch (e: any) { setSnapshotStatus(e.message); setSnapshotIsError(true); }
                        }} className="px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-[9px] font-black uppercase hover:bg-emerald-500 hover:text-black transition-all">
                          Restore
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="h-16 flex items-center justify-center border border-dashed border-white/10 rounded-xl">
                  <p className="text-[10px] font-mono theme-muted italic opacity-30">No snapshots — click Save Snapshot to create one</p>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Tokens */}
      <div className="xl:col-span-2 space-y-6 rounded-2xl border border-white/5 bg-white/[0.015] p-6 shadow-inner transition-colors hover:border-white/10">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl theme-surface-soft theme-muted flex items-center justify-center glass-panel shadow-inner transition-transform duration-300 hover:scale-105 border border-white/5">
              <Cloud className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base-fluid text-white font-black tracking-tight">Provider Tokens</h3>
              <p className="text-xs-fluid theme-muted font-mono uppercase tracking-wider opacity-60">Hugging Face & Embedding API</p>
            </div>
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between ml-1">
            <label className="text-[10px] uppercase tracking-widest theme-muted font-black">Hugging Face Token</label>
            <button
              type="button"
              onClick={() => setShowHf(!showHf)}
              className="text-[10px] uppercase tracking-widest theme-accent font-black hover:brightness-125"
            >
              {showHf ? "Mask" : "Unmask"}
            </button>
          </div>
          <div className="relative group">
            <input
              type={showHf ? "text" : "password"}
              placeholder="hf_…"
              value={config.hfToken || ""}
              onChange={(e) => onChange({ hfToken: e.target.value })}
              className="w-full pl-4 pr-28 py-3 premium-input rounded-xl text-sm-fluid font-mono placeholder-white/10 focus:outline-none shadow-inner"
            />
            {config.hfToken && (
              <span className="absolute right-4 top-1/2 -translate-y-1/2 uppercase font-mono text-[9px] theme-accent-soft theme-accent border border-theme-accent/20 px-2 py-0.5 rounded-lg shadow-sm">
                Active Session
              </span>
            )}
          </div>
        </div>

        {/* Service config cards — responsive grid on wide screens */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-5 items-start">
        {/* Embedding Configuration */}
        <div className="space-y-4 glass-panel rounded-2xl p-5 border border-white/5 relative overflow-hidden group">
          <div className="absolute top-0 left-0 w-1 h-full theme-accent-bg opacity-30" />
          <div className="flex items-center justify-between">
            <label className="text-[10px] uppercase tracking-[0.2em] theme-accent font-black italic font-serif flex items-center gap-2">
              <Sparkles className="w-3.5 h-3.5" /> Embeddings <span className="text-[8px] opacity-40 font-mono tracking-normal">(vLLM on GPU Server)</span>
            </label>
            <button
              type="button"
              onClick={() => setShowEmbedding(!showEmbedding)}
              className="px-2 py-1 rounded-md border border-white/5 theme-surface-soft text-[9px] uppercase tracking-widest theme-muted hover:theme-text transition-all font-bold"
            >
              {showEmbedding ? "Hide" : "Configure"}
            </button>
          </div>
          <p className="text-sm-fluid theme-muted font-medium leading-relaxed opacity-80">
            Self-hosted embedding models served on your GPU server via <span className="text-white">vLLM --runner pooling</span>. Each model owns its own Qdrant collection.
          </p>

          {showEmbedding && (
            <div className="space-y-4 animate-premium">
              {embedders.length > 0 && (
                <div className="space-y-3">
                  {embedders.map((emb, idx) => {
                    const status = embedderStatuses[idx] ?? "idle";
                    return (
                      <div key={idx} className="glass-panel rounded-xl p-4 border border-white/5">
                        <div className="flex items-center gap-2 mb-3">
                          <span className={`w-2 h-2 rounded-full ${
                            status === "running" ? "bg-emerald-400" :
                            status === "booting" ? "bg-amber-400 animate-pulse" :
                            status === "error" ? "bg-red-400" : "bg-white/20"
                          }`} />
                          <input
                            type="text"
                            value={emb.name}
                            onChange={(e) => updateEmbedder(idx, { name: e.target.value })}
                            placeholder="e.g. Law, Math, Science"
                            className="flex-1 px-3 py-2 rounded-lg bg-black/30 border border-white/5 text-xs font-mono text-white focus:outline-none focus:border-theme-accent/30"
                          />
                          <input
                            type="number"
                            value={emb.port}
                            onChange={(e) => updateEmbedder(idx, { port: parseInt(e.target.value) || 8101 })}
                            className="w-20 px-3 py-2 rounded-lg bg-black/30 border border-white/5 text-xs font-mono text-white focus:outline-none focus:border-theme-accent/30 text-center"
                            placeholder="Port"
                          />
                          <button
                            type="button"
                            onClick={() => removeEmbedder(idx)}
                            className="p-2 rounded-lg border border-white/5 theme-surface-soft text-red-400 hover:text-red-300 hover:bg-red-500/10 transition-all"
                            title="Remove"
                          >
                            <X className="w-3.5 h-3.5" />
                          </button>
                        </div>
                        <div className="space-y-2">
                          <div className="flex gap-2">
                            <input
                              type="text"
                              value={emb.modelId}
                              onChange={(e) => updateEmbedder(idx, { modelId: e.target.value })}
                              placeholder="HuggingFace model id (e.g. Qwen/Qwen3-Embedding-8B)"
                              className="flex-1 px-3 py-2 rounded-lg bg-black/30 border border-white/5 text-xs font-mono text-white focus:outline-none focus:border-theme-accent/30"
                            />
                          </div>
                          <div className="flex items-center gap-3">
                            <label className="text-[9px] uppercase tracking-widest theme-muted font-black whitespace-nowrap">Concurrency</label>
                            <input
                              type="number"
                              min={1}
                              max={8}
                              value={emb.concurrency}
                              onChange={(e) => updateEmbedder(idx, { concurrency: parseInt(e.target.value) || 2 })}
                              className="w-16 px-3 py-2 rounded-lg bg-black/30 border border-white/5 text-xs font-mono text-white focus:outline-none focus:border-theme-accent/30 text-center"
                            />
                            <label className="text-[9px] uppercase tracking-widest theme-muted font-black whitespace-nowrap">Collection</label>
                            <input
                              type="text"
                              value={emb.collection}
                              onChange={(e) => updateEmbedder(idx, { collection: e.target.value })}
                              placeholder="kb_law (auto-generated if blank)"
                              className="flex-1 px-3 py-2 rounded-lg bg-black/30 border border-white/5 text-xs font-mono text-white focus:outline-none focus:border-theme-accent/30"
                            />
                          </div>
                        </div>
                        {status !== "idle" && (
                          <div className="mt-2 text-[9px] font-mono theme-muted">
                            {status === "running" ? "● Serving on GPU server" :
                             status === "booting" ? "◌ Bootstrapping model..." :
                             "✕ Failed to start embedder"}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}

              <button
                type="button"
                onClick={addEmbedder}
                className="flex items-center gap-2 px-4 py-3 rounded-xl border border-dashed border-white/20 text-[10px] uppercase tracking-widest font-black theme-muted hover:theme-text hover:border-white/30 transition-all w-full justify-center"
              >
                <Plus className="w-3.5 h-3.5" /> Add Embedding Model
              </button>

              {embedders.length > 0 && (
                <button
                  type="button"
                  onClick={setupAllEmbedders}
                  disabled={settingUpEmbedders || !config.ssh.host}
                  className={`flex items-center gap-2 px-5 py-3 rounded-xl border text-[10px] uppercase tracking-widest font-black transition-all w-full justify-center ${
                    settingUpEmbedders
                      ? "border-amber-400/30 bg-amber-400/10 text-amber-400"
                      : "theme-accent-bg text-black border-theme-accent shadow-lg hover:brightness-110"
                  } disabled:opacity-30`}
                >
                  <Server className="w-3.5 h-3.5" />
                  {settingUpEmbedders ? "Installing on GPU Server..." : "Setup All Embedding Models"}
                </button>
              )}
            </div>
          )}
        </div>

        {/* PaddleOCR-VL Configuration */}
        <div className="space-y-4 glass-panel rounded-2xl p-5 border border-white/5 relative overflow-hidden group">
          <div className="absolute top-0 left-0 w-1 h-full bg-orange-500/30" />
          <div className="flex items-center justify-between">
            <label className="text-[10px] uppercase tracking-[0.2em] text-orange-400 font-black italic font-serif flex items-center gap-2">
              <ScanText className="w-3.5 h-3.5" /> PaddleOCR-VL <span className="text-[8px] opacity-40 font-mono tracking-normal">(PDF OCR on GPU Server)</span>
            </label>
          </div>
          <p className="text-sm-fluid theme-muted font-medium leading-relaxed opacity-80">
            Deploys a PaddleOCR-VL container on the GPU server for extracting text from scanned/image-based PDFs during ingestion.
          </p>

          <label className="flex items-center justify-between p-4 rounded-xl border border-white/5 bg-white/[0.01] hover:bg-white/[0.03] transition-colors cursor-pointer group shadow-sm">
            <div className="flex items-center gap-3">
              <div className={`w-5 h-5 rounded border-2 transition-all flex items-center justify-center ${paddleOcr.enabled ? "bg-orange-500 border-orange-500" : "border-white/20 group-hover:border-white/40"}`}>
                {paddleOcr.enabled && <CheckCircle2 className="w-4 h-4 text-black" />}
              </div>
              <span className="text-sm-fluid theme-text font-black font-mono tracking-tight group-hover:text-orange-400 transition-colors">
                ENABLE PADDLEOCR-VL
              </span>
            </div>
            <input
              type="checkbox"
              checked={paddleOcr.enabled}
              onChange={(e) => patchPaddleOcr({ enabled: e.target.checked })}
              className="hidden"
            />
          </label>

          {paddleOcr.enabled && (
            <div className="space-y-3 animate-premium">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label className="text-[9px] uppercase tracking-widest theme-muted font-black ml-1">Port</label>
                  <input
                    type="number"
                    value={paddleOcr.port}
                    onChange={(e) => patchPaddleOcr({ port: parseInt(e.target.value) || 8118 })}
                    className="w-full px-3 py-2 rounded-lg bg-black/30 border border-white/5 text-xs font-mono text-white focus:outline-none focus:border-orange-500/30"
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-[9px] uppercase tracking-widest theme-muted font-black ml-1">Model Name</label>
                  <input
                    type="text"
                    value={paddleOcr.modelName}
                    onChange={(e) => patchPaddleOcr({ modelName: e.target.value })}
                    className="w-full px-3 py-2 rounded-lg bg-black/30 border border-white/5 text-xs font-mono text-white focus:outline-none focus:border-orange-500/30"
                  />
                </div>
              </div>
              <div className="space-y-2">
                <label className="text-[9px] uppercase tracking-widest theme-muted font-black ml-1">Docker Image</label>
                <input
                  type="text"
                  value={paddleOcr.dockerImage}
                  onChange={(e) => patchPaddleOcr({ dockerImage: e.target.value })}
                  className="w-full px-3 py-2 rounded-lg bg-black/30 border border-white/5 text-xs font-mono text-white focus:outline-none focus:border-orange-500/30"
                />
              </div>
              <button
                type="button"
                onClick={bootPaddleOcr}
                disabled={bootingOcr || !config.ssh.host}
                className={`flex items-center gap-2 px-5 py-3 rounded-xl border text-[10px] uppercase tracking-widest font-black transition-all w-full justify-center ${
                  bootingOcr
                    ? "border-amber-400/30 bg-amber-400/10 text-amber-400"
                    : "bg-orange-500 text-black border-orange-500 shadow-lg hover:brightness-110"
                } disabled:opacity-30`}
              >
                <ScanText className="w-3.5 h-3.5" />
                {bootingOcr ? "Deploying to GPU Server..." : "Boot PaddleOCR-VL"}
              </button>
              {ocrStatus !== "idle" && (
                <div className="text-[9px] font-mono theme-muted">
                  {ocrStatus === "running" ? "● PaddleOCR-VL serving on GPU server" :
                   ocrStatus === "error" ? "✕ Failed to boot PaddleOCR-VL" : null}
                </div>
              )}
              {ocrLogs && (
                <div className="mt-2 p-3 rounded-xl bg-black/60 border border-white/5 font-mono text-[10px] leading-relaxed theme-text/80 max-h-48 overflow-y-auto whitespace-pre-wrap scrollbar-thin scrollbar-thumb-white/10">
                  {ocrLogs}
                  {bootingOcr && <span className="inline-block w-1.5 h-3 bg-amber-400 ml-1 animate-pulse" />}
                </div>
              )}
            </div>
          )}
        </div>

        {/* AI Copilot Agent Configuration */}
        <div className="xl:col-span-2 space-y-4 glass-panel rounded-2xl p-5 border border-white/5 relative overflow-hidden group">
          <div className="absolute top-0 left-0 w-1 h-full theme-accent-bg opacity-30" />
          <div className="flex items-center justify-between">
            <label className="text-[10px] uppercase tracking-[0.2em] theme-accent font-black italic font-serif flex items-center gap-2">
              <Sparkles className="w-3.5 h-3.5" /> AI Copilot Agent <span className="text-[8px] opacity-40 font-mono tracking-normal">(Terminal Assistant)</span>
            </label>
            <button
              type="button"
              onClick={() => setShowCopilotConfig(!showCopilotConfig)}
              className="px-2 py-1 rounded-md border border-white/5 theme-surface-soft text-[9px] uppercase tracking-widest theme-muted hover:theme-text transition-all font-bold"
            >
              {showCopilotConfig ? "Hide Config" : "Configure"}
            </button>
          </div>
          <p className="text-sm-fluid theme-muted font-medium leading-relaxed opacity-80">
            Configure the AI provider, custom endpoints, and specific model ID utilized by the persistent side terminal.
          </p>

          {showCopilotConfig && (
            <div className="space-y-4 animate-premium">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label className="text-[10px] uppercase tracking-widest theme-muted font-black ml-1">Provider</label>
                  <select
                    value={config.aiAgent?.provider ?? "vultr"}
                    onChange={(e) => patchAiAgent({ provider: e.target.value as any })}
                    className="w-full px-4 py-3 premium-input rounded-xl text-sm-fluid font-mono bg-black/20 focus:outline-none shadow-inner border border-white/5 text-white"
                  >
                    <option value="vultr">Vultr</option>
                    <option value="openai">OpenAI (ChatGPT)</option>
                    <option value="anthropic">Anthropic (Claude)</option>
                    <option value="gemini">Google Gemini (OpenAI Endpoint)</option>
                    <option value="groq">Groq</option>
                    <option value="xai">xAI (X)</option>
                    <option value="custom">Custom OpenAI-Compatible API</option>
                  </select>
                </div>
                <div className="space-y-2">
                  <div className="flex items-center justify-between ml-1">
                    <label className="text-[10px] uppercase tracking-widest theme-muted font-black">Model ID</label>
                    {loadingModels && (
                      <span className="text-[8px] font-mono theme-accent animate-pulse">FETCHING MODELS...</span>
                    )}
                  </div>
                  {displayModels.length > 0 ? (
                    <div className="space-y-2">
                      <select
                        value={displayModels.includes(config.aiAgent?.modelId ?? "") ? (config.aiAgent?.modelId ?? "") : "custom"}
                        onChange={(e) => {
                          if (e.target.value === "custom") {
                            patchAiAgent({ modelId: "" });
                          } else {
                            patchAiAgent({ modelId: e.target.value });
                          }
                        }}
                        className="w-full px-4 py-3 premium-input rounded-xl text-sm-fluid font-mono bg-black/20 focus:outline-none shadow-inner border border-white/5 text-white"
                      >
                        {displayModels.map((m) => (
                          <option key={m} value={m}>{m}</option>
                        ))}
                        <option value="custom">Custom (Type manually)...</option>
                      </select>
                      {(!displayModels.includes(config.aiAgent?.modelId ?? "") || !config.aiAgent?.modelId) && (
                        <input
                          type="text"
                          placeholder="Type custom model ID..."
                          value={config.aiAgent?.modelId ?? ""}
                          onChange={(e) => patchAiAgent({ modelId: e.target.value })}
                          className="w-full px-4 py-3 premium-input rounded-xl text-sm-fluid font-mono placeholder-white/10 focus:outline-none shadow-inner bg-black/20"
                        />
                      )}
                    </div>
                  ) : (
                    <input
                      type="text"
                      placeholder="e.g. meta-llama/Meta-Llama-3.1-70B-Instruct"
                      value={config.aiAgent?.modelId ?? ""}
                      onChange={(e) => patchAiAgent({ modelId: e.target.value })}
                      className="w-full px-4 py-3 premium-input rounded-xl text-sm-fluid font-mono placeholder-white/10 focus:outline-none shadow-inner bg-black/20"
                    />
                  )}
                </div>
              </div>

              <div className="space-y-2">
                <label className="text-[10px] uppercase tracking-widest theme-muted font-black ml-1">API Endpoint URL</label>
                <input
                  type="text"
                  placeholder="https://api.openai.com/v1"
                  value={config.aiAgent?.apiUrl ?? ""}
                  onChange={(e) => patchAiAgent({ apiUrl: e.target.value })}
                  className="w-full px-4 py-3 premium-input rounded-xl text-sm-fluid font-mono placeholder-white/10 focus:outline-none shadow-inner bg-black/20"
                />
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between ml-1">
                  <label className="text-[10px] uppercase tracking-widest theme-muted font-black">API Key</label>
                  <button
                    type="button"
                    onClick={() => setShowCopilotKey(!showCopilotKey)}
                    className="text-[10px] uppercase tracking-widest theme-accent font-black hover:brightness-125"
                  >
                    {showCopilotKey ? "Mask" : "Unmask"}
                  </button>
                </div>
                <input
                  type={showCopilotKey ? "text" : "password"}
                  placeholder="Enter API key..."
                  value={config.aiAgent?.apiKey ?? ""}
                  onChange={(e) => patchAiAgent({ apiKey: e.target.value })}
                  className="w-full px-4 py-3 premium-input rounded-xl text-sm-fluid font-mono placeholder-white/10 focus:outline-none shadow-inner bg-black/20"
                />
              </div>
            </div>
          )}
        </div>
        </div>
      </div>

      {/* Docker */}
      <div className="space-y-6 rounded-2xl border border-white/5 bg-white/[0.015] p-6 shadow-inner transition-colors hover:border-white/10">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl theme-surface-soft theme-muted flex items-center justify-center glass-panel shadow-inner transition-transform duration-300 hover:scale-105 border border-white/5">
              <Container className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base-fluid text-white font-black tracking-tight">Containerization</h3>
              <p className="text-xs-fluid theme-muted font-mono uppercase tracking-wider opacity-60">Dockerized pipeline runtime</p>
            </div>
          </div>
          <button
            type="button"
            onClick={() => setShowDocker(!showDocker)}
            className="px-3 py-1.5 rounded-lg border border-white/5 theme-surface-soft text-[10px] uppercase tracking-widest theme-muted hover:theme-text transition-all font-black"
          >
            {showDocker ? "Simple View" : "Advanced"}
          </button>
        </div>

        <label className="flex items-center justify-between p-4 rounded-xl border border-white/5 bg-white/[0.01] hover:bg-white/[0.03] transition-colors cursor-pointer group shadow-sm">
          <div className="flex items-center gap-3">
            <div className={`w-5 h-5 rounded border-2 transition-all flex items-center justify-center ${docker?.enabled ? "bg-theme-accent border-theme-accent" : "border-white/20 group-hover:border-white/40"}`}>
              {docker?.enabled && <CheckCircle2 className="w-4 h-4 text-black" />}
            </div>
            <span className="text-sm-fluid theme-text font-black font-mono tracking-tight group-hover:theme-accent transition-colors">
              EXECUTE PIPELINE INSIDE DOCKER
            </span>
          </div>
          <input
            type="checkbox"
            checked={docker?.enabled ?? true}
            onChange={(e) => patchDocker({ enabled: e.target.checked })}
            className="hidden"
          />
        </label>

        {showDocker && (
          <div className="space-y-4 animate-premium">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-[10px] uppercase tracking-widest theme-muted font-black ml-1">Container Name</label>
                <input
                  type="text"
                  placeholder="rocm-vllm"
                  value={docker?.containerName ?? ""}
                  onChange={(e) => patchDocker({ containerName: e.target.value })}
                  className="w-full px-4 py-3 premium-input rounded-xl text-sm-fluid font-mono placeholder-white/10 focus:outline-none shadow-inner"
                />
              </div>
              <div className="space-y-2">
                <label className="text-[10px] uppercase tracking-widest theme-muted font-black ml-1">Docker Image</label>
                <input
                  type="text"
                  placeholder="rocm/vllm:latest"
                  value={docker?.imageName ?? ""}
                  onChange={(e) => patchDocker({ imageName: e.target.value })}
                  className="w-full px-4 py-3 premium-input rounded-xl text-sm-fluid font-mono placeholder-white/10 focus:outline-none shadow-inner"
                />
              </div>
            </div>
            <div className="space-y-2">
              <label className="text-[10px] uppercase tracking-widest theme-muted font-black ml-1">Runtime Arguments</label>
              <textarea
                rows={3}
                placeholder="--device=/dev/kfd --device=/dev/dri ..."
                value={docker?.startArgs ?? ""}
                onChange={(e) => patchDocker({ startArgs: e.target.value })}
                className="w-full px-4 py-4 premium-input rounded-2xl text-sm-fluid font-mono placeholder-white/10 leading-relaxed resize-none focus:outline-none shadow-inner"
              />
            </div>
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="xl:col-span-2 space-y-4 rounded-2xl border border-theme-accent/10 bg-gradient-to-br from-theme-accent/[0.04] to-transparent p-6 shadow-inner">
        <button
          onClick={onTestConnection}
          disabled={connection.isTesting || !ssh.host}
          className="w-full theme-accent-bg text-black text-center py-4 rounded-2xl font-black text-sm-fluid uppercase tracking-[0.25em] premium-button hover:brightness-110 active:scale-[0.98] transition-all disabled:opacity-20 shadow-xl shadow-theme-accent/20"
        >
          {connection.isTesting ? (
            <span className="flex items-center justify-center gap-3">
              <Loader2 className="w-5 h-5 animate-spin text-black" /> ESTABLISHING LINK…
            </span>
          ) : (
            "TEST SYSTEM UPLINK"
          )}
        </button>
        {connection.message && (
          <div
            className={`p-4 rounded-2xl border text-sm-fluid font-mono leading-relaxed shadow-lg animate-premium ${
              connection.isConnected
                ? "bg-green-500/5 border-green-500/30 text-green-300"
                : "bg-red-500/5 border-red-500/30 text-red-300"
            }`}
          >
            <div className="flex items-start gap-3">
              <div className={`mt-1 w-1.5 h-1.5 rounded-full shrink-0 ${connection.isConnected ? "bg-green-400" : "bg-red-400"}`} />
              <p className="opacity-90 break-words whitespace-pre-wrap">
                {connection.message}
              </p>
            </div>
          </div>
        )}
      </div>
      </div>
    </div>
  );
}
