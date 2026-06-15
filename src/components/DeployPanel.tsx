import React, { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import type { AppConfig, TeacherConfig, HfModelRepo } from "../types";
import { DEFAULT_TEACHER } from "../types";
import { api, events } from "../lib/tauri";
import {
  Rocket,
  Loader2,
  CircleSlash,
  ChevronDown,
  ChevronRight,
  Copy,
  Check,
  Send,
  Globe,
  Lock,
  RefreshCw,
} from "lucide-react";

interface Props {
  config: AppConfig;
}

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

// Dedicated "Deploy" page: stand up an arbitrary HuggingFace model (public or
// private) on the GPU server via the existing deploy_teacher backend, surface
// the live chat URL, and chat with it in-page. Deploy lifecycle mirrors the
// teacher-deploy flow in PipelineWizard.tsx; chat reuses teacher_chat. This
// panel keeps everything in LOCAL state and never mutates the shared
// config.teacher (which belongs to the pipeline).
export default function DeployPanel({ config }: Props) {
  // --- Model selection ---------------------------------------------------
  const [repoId, setRepoId] = useState(DEFAULT_TEACHER.repoId);
  const [hfModels, setHfModels] = useState<HfModelRepo[]>([]);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [isPrivate, setIsPrivate] = useState(false);

  // --- Advanced (collapsed) — seeded from DEFAULT_TEACHER (auto-tuned) ----
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [vllmPort, setVllmPort] = useState(DEFAULT_TEACHER.vllmPort);
  const [maxModelLen, setMaxModelLen] = useState(DEFAULT_TEACHER.maxModelLen);
  const [dtype, setDtype] = useState(DEFAULT_TEACHER.dtype);
  const [gpuMemUtil, setGpuMemUtil] = useState(DEFAULT_TEACHER.gpuMemoryUtilization ?? 0.8);

  // --- Deploy lifecycle --------------------------------------------------
  const [deployStreamId, setDeployStreamId] = useState<string | null>(null);
  const [deployLogs, setDeployLogs] = useState("");
  const [deploying, setDeploying] = useState(false);
  const [deployError, setDeployError] = useState<string | null>(null);
  const [activePort, setActivePort] = useState<number | null>(null);
  const [checking, setChecking] = useState(false);
  const [stopping, setStopping] = useState(false);

  // --- Chat --------------------------------------------------------------
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [chatSending, setChatSending] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const logBoxRef = useRef<HTMLDivElement>(null);
  const chatBoxRef = useRef<HTMLDivElement>(null);
  const deploymentCheckSeqRef = useRef(0);
  const modelId = repoId.trim();

  const endpoint =
    activePort != null && config.ssh.host
      ? `http://${config.ssh.host}:${activePort}`
      : null;

  // Assemble a TeacherConfig on demand. Local only — autoTune lets the backend
  // size context/VRAM to the GPU; the advanced fields override the defaults.
  const buildTeacher = useCallback(
    (): TeacherConfig => ({
      ...DEFAULT_TEACHER,
      repoId: modelId,
      vllmPort,
      maxModelLen,
      dtype,
      gpuMemoryUtilization: gpuMemUtil,
      servingEngine: "vllm",
    }),
    [modelId, vllmPort, maxModelLen, dtype, gpuMemUtil],
  );

  // --- Effect A: load the token owner's HF models (private repos selectable)
  useEffect(() => {
    let alive = true;
    setModelsLoading(true);
    api
      .hfListModels()
      .then((list) => {
        if (alive) setHfModels(list);
      })
      .catch((e) => console.error("hf_list_models:", e))
      .finally(() => {
        if (alive) setModelsLoading(false);
      });
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    const selected = hfModels.find((m) => m.id === modelId);
    setIsPrivate(selected?.private ?? false);
  }, [hfModels, modelId]);

  // --- Effect B: detect an already-running endpoint so chat works instantly.
  // Re-probe when the chosen model / port changes. Debounced lightly so we
  // don't hammer SSH on every keystroke in the repo field.
  const checkDeployment = useCallback(async () => {
    const checkSeq = ++deploymentCheckSeqRef.current;
    if (!config.ssh.host || !modelId) {
      setActivePort(null);
      setChecking(false);
      return;
    }
    setChecking(true);
    try {
      const status = await api.checkTeacherDeployed(config.ssh, config.docker, buildTeacher());
      if (checkSeq === deploymentCheckSeqRef.current) {
        setActivePort(status?.exact ? status.port : null);
      }
    } catch (e) {
      console.error("check_teacher_deployed:", e);
    } finally {
      if (checkSeq === deploymentCheckSeqRef.current) {
        setChecking(false);
      }
    }
  }, [config.ssh, config.docker, modelId, buildTeacher]);

  useEffect(() => {
    deploymentCheckSeqRef.current += 1;
    setChecking(false);
    if (!config.ssh.host || !modelId) {
      setActivePort(null);
      return;
    }
    setActivePort(null);
    const t = setTimeout(() => {
      checkDeployment();
    }, 400);
    return () => clearTimeout(t);
  }, [config.ssh.host, modelId, vllmPort, maxModelLen, dtype, gpuMemUtil, checkDeployment]);

  // --- Effect C: stream deploy log/done events for our stream only ---------
  useEffect(() => {
    if (!deployStreamId) return;
    let disposed = false;
    let unlistenLog: (() => void) | null = null;
    let unlistenDone: (() => void) | null = null;
    const setup = async () => {
      const logUnlisten = await events.onDeployLog((e) => {
        if (e.streamId === deployStreamId) {
          setDeployLogs((prev) => prev + e.line);
        }
      });
      if (disposed) {
        logUnlisten();
      } else {
        unlistenLog = logUnlisten;
      }

      const doneUnlisten = await events.onDeployDone((e) => {
        if (e.streamId !== deployStreamId) return;
        setDeploying(false);
        setDeployStreamId(null);
        if (e.success) {
          if (e.port !== undefined) {
            setActivePort(e.port);
          } else {
            // Port wasn't reported — re-probe to discover the live port.
            void checkDeployment();
          }
        } else {
          setDeployError(e.message);
        }
      });
      if (disposed) {
        doneUnlisten();
      } else {
        unlistenDone = doneUnlisten;
      }
    };
    setup().catch((e) => {
      if (disposed) return;
      console.error("deploy listeners:", e);
      setDeployError(errorMessage(e));
      setDeploying(false);
      setDeployStreamId(null);
    });
    return () => {
      disposed = true;
      if (unlistenLog) unlistenLog();
      if (unlistenDone) unlistenDone();
    };
  }, [deployStreamId]); // eslint-disable-line react-hooks/exhaustive-deps

  // Auto-tail the deploy log box.
  useLayoutEffect(() => {
    const el = logBoxRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [deployLogs]);

  // Auto-scroll the chat thread to the newest message.
  useLayoutEffect(() => {
    const el = chatBoxRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [chatMessages, chatSending]);

  const startDeployment = async () => {
    if (!modelId || !config.ssh.host) return;
    setDeploying(true);
    setDeployError(null);
    setDeployLogs("");
    setActivePort(null);
    try {
      const streamId = await api.deployTeacher(
        config.ssh,
        config.docker,
        buildTeacher(),
        config.hfToken,
      );
      setDeployStreamId(streamId);
    } catch (err: unknown) {
      setDeployError(errorMessage(err));
      setDeploying(false);
    }
  };

  const cancelDeployment = async () => {
    if (!deployStreamId) return;
    try {
      await api.sshStopStream(deployStreamId);
    } catch (e) {
      console.error("cancel deployment:", e);
    } finally {
      setDeploying(false);
      setDeployStreamId(null);
    }
  };

  const stopDeployment = async () => {
    if (activePort == null || !config.ssh.host) return;
    setStopping(true);
    setDeployError(null);
    try {
      await api.stopTeacher(config.ssh, config.docker, activePort);
      setActivePort(null);
      setChatMessages([]);
      setChatError(null);
    } catch (e: unknown) {
      setDeployError(errorMessage(e));
    } finally {
      setStopping(false);
      void checkDeployment();
    }
  };

  const copyUrl = async () => {
    if (!endpoint) return;
    try {
      await navigator.clipboard.writeText(endpoint);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch (e) {
      console.error("copy url:", e);
    }
  };

  const sendChat = async () => {
    const text = chatInput.trim();
    if (!text || !endpoint || !modelId || chatSending) return;
    const next: ChatMessage[] = [...chatMessages, { role: "user", content: text }];
    setChatMessages(next);
    setChatInput("");
    setChatSending(true);
    setChatError(null);
    try {
      const reachable = await api.pingTeacher(endpoint);
      if (!reachable) {
        throw new Error("Endpoint not reachable yet - vLLM is still loading.");
      }
      // Send the full thread so the model keeps multi-turn context.
      const answer = await api.teacherChat(endpoint, modelId, next);
      setChatMessages((prev) => [...prev, { role: "assistant", content: answer }]);
    } catch (e: unknown) {
      setChatError(errorMessage(e));
      // Roll back the optimistic user message so they can retry/edit.
      setChatMessages((prev) => prev.slice(0, -1));
      setChatInput(text);
    } finally {
      setChatSending(false);
    }
  };

  const onChatKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendChat();
    }
  };

  const noHost = !config.ssh.host;

  return (
    <div className="w-full max-w-5xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-start gap-3">
        <div className="p-2.5 theme-accent-soft theme-accent rounded-lg shrink-0">
          <Rocket className="w-5 h-5" />
        </div>
        <div>
          <h2 className="text-lg-fluid font-serif italic text-white leading-tight">Deploy Model</h2>
          <p className="text-[11px] theme-muted leading-relaxed mt-0.5 max-w-xl">
            Serve any HuggingFace model (public or private) on your GPU server, then chat with it
            directly. Private repos authenticate with your saved HF token.
          </p>
        </div>
      </div>

      {noHost && (
        <div className="theme-surface border border-amber-500/30 bg-amber-950/10 rounded-lg p-4 text-[11px] text-amber-300 font-mono">
          No GPU server connected. Set your SSH host in the <b>Credentials</b> tab before deploying.
        </div>
      )}

      {/* Model selection */}
      <section className="theme-surface border rounded-lg p-4 space-y-4">
        <p className="text-[10px] uppercase tracking-[0.2em] theme-accent font-mono font-bold">
          Model Source
        </p>

        <div className="space-y-2">
          <label className="text-[9px] uppercase tracking-widest theme-muted font-mono">
            HuggingFace Repo ID
          </label>
          <div className="flex items-center gap-2">
            <input
              value={repoId}
              onChange={(e) => setRepoId(e.target.value)}
              placeholder="e.g. Qwen/Qwen2.5-7B-Instruct"
              className="flex-1 min-w-0 px-3 py-2 theme-field border rounded-lg text-[12px] font-mono text-white focus:outline-none focus:border-theme-accent transition"
            />
            <span
              className={`flex items-center gap-1 px-2.5 py-1.5 rounded border text-[9px] uppercase tracking-widest font-mono font-bold shrink-0 ${
                isPrivate
                  ? "text-amber-300 bg-amber-500/10 border-amber-500/30"
                  : "text-emerald-300 bg-emerald-500/10 border-emerald-500/30"
              }`}
              title={isPrivate ? "Private — uses your saved HF token" : "Public"}
            >
              {isPrivate ? <Lock className="w-3 h-3" /> : <Globe className="w-3 h-3" />}
              {isPrivate ? "Private" : "Public"}
            </span>
          </div>
        </div>

        {/* Owner's repos — lets private models be selected in one click */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-[9px] uppercase tracking-widest theme-muted font-mono">
              Your Models
            </label>
            {modelsLoading && <Loader2 className="w-3 h-3 animate-spin theme-faint" />}
          </div>
          <select
            value={hfModels.some((m) => m.id === modelId) ? modelId : ""}
            onChange={(e) => {
              const picked = hfModels.find((m) => m.id === e.target.value);
              if (picked) {
                setRepoId(picked.id);
                setIsPrivate(picked.private);
              }
            }}
            className="w-full px-3 py-2 theme-field border rounded-lg text-[12px] font-mono text-white focus:outline-none focus:border-theme-accent transition"
          >
            <option value="">
              {modelsLoading
                ? "Loading your repos…"
                : hfModels.length === 0
                  ? "No repos found (enter a repo ID above)"
                  : "Select one of your repos…"}
            </option>
            {hfModels.map((m) => (
              <option key={m.id} value={m.id}>
                {m.id}
                {m.private ? "  (private)" : ""}
              </option>
            ))}
          </select>
        </div>

        {/* Advanced */}
        <div>
          <button
            onClick={() => setShowAdvanced((s) => !s)}
            className="flex items-center gap-1.5 text-[9px] uppercase tracking-widest theme-muted hover:theme-text font-mono font-bold transition"
          >
            {showAdvanced ? (
              <ChevronDown className="w-3 h-3" />
            ) : (
              <ChevronRight className="w-3 h-3" />
            )}
            Advanced (auto-tuned from GPU by default)
          </button>
          {showAdvanced && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-3">
              <Field label="Port">
                <input
                  type="number"
                  value={vllmPort}
                  onChange={(e) => setVllmPort(Number(e.target.value) || DEFAULT_TEACHER.vllmPort)}
                  className="w-full px-2 py-1.5 theme-field border rounded text-[11px] font-mono text-white focus:outline-none focus:border-theme-accent"
                />
              </Field>
              <Field label="Max Len">
                <input
                  type="number"
                  value={maxModelLen}
                  onChange={(e) =>
                    setMaxModelLen(Number(e.target.value) || DEFAULT_TEACHER.maxModelLen)
                  }
                  className="w-full px-2 py-1.5 theme-field border rounded text-[11px] font-mono text-white focus:outline-none focus:border-theme-accent"
                />
              </Field>
              <Field label="Dtype">
                <select
                  value={dtype}
                  onChange={(e) => setDtype(e.target.value)}
                  className="w-full px-2 py-1.5 theme-field border rounded text-[11px] font-mono text-white focus:outline-none focus:border-theme-accent"
                >
                  <option value="bfloat16">bfloat16</option>
                  <option value="float16">float16</option>
                  <option value="auto">auto</option>
                </select>
              </Field>
              <Field label="GPU Mem %">
                <input
                  type="number"
                  step="0.05"
                  min="0.1"
                  max="0.99"
                  value={gpuMemUtil}
                  onChange={(e) => setGpuMemUtil(Number(e.target.value) || 0.8)}
                  className="w-full px-2 py-1.5 theme-field border rounded text-[11px] font-mono text-white focus:outline-none focus:border-theme-accent"
                />
              </Field>
            </div>
          )}
        </div>

        {/* Deploy actions */}
        <div className="flex items-center gap-3 pt-1">
          <button
            onClick={startDeployment}
            disabled={deploying || noHost || !repoId.trim()}
            className="flex items-center gap-2 px-4 py-2 rounded theme-accent-bg text-black text-[10px] uppercase tracking-widest font-bold hover:brightness-110 disabled:opacity-50 transition shadow-lg"
          >
            {deploying ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Rocket className="w-3.5 h-3.5" />}
            {deploying ? "Deploying…" : "Deploy"}
          </button>
          {deploying && (
            <button
              onClick={cancelDeployment}
              className="flex items-center gap-1 px-3 py-2 bg-red-950/30 border border-red-500/30 text-red-300 rounded text-[10px] uppercase tracking-widest font-mono font-bold hover:bg-red-950 transition"
            >
              <CircleSlash className="w-3 h-3" /> Cancel
            </button>
          )}
          {activePort != null && !deploying && (
            <button
              onClick={stopDeployment}
              disabled={stopping}
              className="flex items-center gap-1.5 px-3 py-2 bg-red-950/30 border border-red-500/30 text-red-300 rounded text-[10px] uppercase tracking-widest font-mono font-bold hover:bg-red-950 disabled:opacity-50 transition"
            >
              {stopping ? (
                <Loader2 className="w-3 h-3 animate-spin" />
              ) : (
                <CircleSlash className="w-3 h-3" />
              )}
              {stopping ? "Stopping…" : "Stop Deploy"}
            </button>
          )}
          {checking && (
            <span className="flex items-center gap-1.5 text-[9px] uppercase tracking-widest theme-faint font-mono">
              <Loader2 className="w-3 h-3 animate-spin" /> Probing server…
            </span>
          )}
        </div>

        {deployError && (
          <div className="text-[11px] text-red-400 font-mono bg-red-950/30 border border-red-500/20 rounded-lg p-3 whitespace-pre-wrap">
            {deployError}
          </div>
        )}
      </section>

      {/* Deploy logs */}
      {(deployLogs || deploying) && (
        <section className="theme-surface border rounded-lg overflow-hidden">
          <div className="px-4 py-2 border-b theme-surface-soft bg-black/10">
            <p className="text-[10px] uppercase tracking-[0.2em] theme-muted font-mono font-bold">
              Deployment Logs
            </p>
          </div>
          <div
            ref={logBoxRef}
            className="bg-black/30 p-4 text-[11px] font-mono leading-relaxed text-white/70 h-72 overflow-y-auto whitespace-pre-wrap scrollbar-thin"
          >
            {deployLogs || (
              <span className="theme-faint italic">Awaiting deployment output…</span>
            )}
          </div>
        </section>
      )}

      {/* Chat URL banner */}
      {endpoint && (
        <section className="theme-surface border border-emerald-500/30 bg-emerald-950/10 rounded-lg p-4 flex items-center gap-3">
          <span className="relative inline-flex w-3 h-3 shrink-0">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-60" />
            <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-400" />
          </span>
          <div className="min-w-0 flex-1">
            <p className="text-[8px] uppercase tracking-widest text-emerald-400/70 font-mono font-bold mb-0.5">
              Live Chat Endpoint
            </p>
            <code className="text-[12px] font-mono text-emerald-200 truncate block">
              {endpoint}
            </code>
          </div>
          <button
            onClick={copyUrl}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded theme-surface border theme-text text-[9px] uppercase tracking-widest font-bold hover:theme-surface-soft transition shrink-0"
          >
            {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
            {copied ? "Copied" : "Copy"}
          </button>
        </section>
      )}

      {/* Chat box */}
      <section className="theme-surface border rounded-lg overflow-hidden flex flex-col h-[460px]">
        <div className="px-4 py-2 border-b theme-surface-soft bg-black/10 flex items-center justify-between">
          <p className="text-[10px] uppercase tracking-[0.2em] theme-accent font-mono font-bold">
            Chat
          </p>
          {chatMessages.length > 0 && (
            <button
              onClick={() => {
                setChatMessages([]);
                setChatError(null);
              }}
              className="flex items-center gap-1 text-[9px] uppercase tracking-widest theme-faint hover:theme-text font-mono transition"
              title="Clear conversation"
            >
              <RefreshCw className="w-3 h-3" /> Clear
            </button>
          )}
        </div>

        <div
          ref={chatBoxRef}
          className="flex-1 min-h-0 overflow-y-auto p-4 space-y-3 scrollbar-thin bg-black/10"
        >
          {chatMessages.length === 0 && !chatSending ? (
            <div className="h-full flex items-center justify-center text-center theme-faint italic font-serif text-sm-fluid px-6">
              {endpoint
                ? "Send a message to chat with the deployed model."
                : "Deploy a model (or connect to a running one) to start chatting."}
            </div>
          ) : (
            chatMessages.map((m, i) => (
              <div
                key={i}
                className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-[80%] rounded-lg px-3 py-2 text-[12px] leading-relaxed whitespace-pre-wrap break-words ${
                    m.role === "user"
                      ? "theme-accent-soft theme-accent border"
                      : "bg-black/40 border border-white/10 text-white/85"
                  }`}
                >
                  <div
                    className={`text-[7px] uppercase tracking-widest font-mono font-bold mb-1 opacity-60 ${
                      m.role === "user" ? "theme-accent" : "text-white/50"
                    }`}
                  >
                    {m.role === "user" ? "You" : "Model"}
                  </div>
                  {m.content}
                </div>
              </div>
            ))
          )}
          {chatSending && (
            <div className="flex justify-start">
              <div className="rounded-lg px-3 py-2 bg-black/40 border border-white/10 text-white/60 text-[11px] font-mono flex items-center gap-2">
                <Loader2 className="w-3 h-3 animate-spin" /> Thinking…
              </div>
            </div>
          )}
        </div>

        {chatError && (
          <div className="px-4 py-2 text-[10px] text-red-400 font-mono bg-red-950/20 border-t border-red-500/20">
            {chatError}
          </div>
        )}

        <div className="border-t theme-surface-soft p-3 bg-black/20">
          <div className="flex items-end gap-2">
            <textarea
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              onKeyDown={onChatKeyDown}
              rows={2}
              disabled={!endpoint || chatSending}
              placeholder={
                endpoint
                  ? "Type a message… (Enter to send, Shift+Enter for newline)"
                  : "Deploy a model first…"
              }
              className="flex-1 min-w-0 px-3 py-2 theme-field border rounded-lg text-[12px] font-mono text-white/85 resize-none focus:outline-none focus:border-theme-accent transition leading-relaxed shadow-inner disabled:opacity-50"
            />
            <button
              onClick={sendChat}
              disabled={!endpoint || chatSending || !chatInput.trim()}
              className="flex items-center gap-2 px-4 py-2.5 rounded theme-accent-bg text-black text-[10px] uppercase tracking-widest font-bold hover:brightness-110 disabled:opacity-50 transition shadow-lg shrink-0"
            >
              {chatSending ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Send className="w-3.5 h-3.5" />}
              Send
            </button>
          </div>
        </div>
      </section>
    </div>
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

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
