import React, { useCallback, useEffect, useRef, useState } from "react";
import type { AppConfig, GPUState, ShellDoneEvent, ShellLogEvent } from "./types";
import { DEFAULT_AI_AGENT, DEFAULT_DIGITAL_OCEAN, DEFAULT_EMBEDDER, DEFAULT_EMBEDDING, DEFAULT_LORA, DEFAULT_TEACHER } from "./types";
import { api, events } from "./lib/tauri";
import { startGlobalSubscription } from "./lib/runStreams";
import { startSetupLogSubscription } from "./lib/setupLogs";
import CredentialsPanel from "./components/CredentialsPanel";
import RoboticsWidget from "./components/RoboticsWidget";
import AITerminalPanel from "./components/AITerminalPanel";
import GPUStatsDashboard from "./components/GPUStatsDashboard";
import GpuServerManager from "./components/GpuServerManager";
import PipelineWizard from "./components/PipelineWizard";
import RunDashboard from "./components/RunDashboard";
import DeployPanel from "./components/DeployPanel";

import ThemeSwitcher from "./components/ThemeSwitcher";
import {
  Layers,
  Sparkles,
  Terminal as TerminalIcon,
  ListChecks,
  Wrench,
  Server,
  Rocket,
} from "lucide-react";

type Tab = "pipeline" | "gpu" | "credentials" | "terminal" | "runs" | "deploy";

const DEFAULT_CONFIG: AppConfig = {
  ssh: { host: "", port: 22, username: "root" },
  qdrant: { endpoint: "", apiKey: "", collection: "all" },
  digitalOcean: DEFAULT_DIGITAL_OCEAN,
  hfToken: "",
  teacher: DEFAULT_TEACHER,
  student: { repoId: "Qwen/Qwen2.5-7B-Instruct", outputDir: "/root/fine-tune/runs" },
  docker: {
    enabled: true,
    containerName: "rocm-vllm",
    imageName: "rocm/vllm:latest",
    startArgs: "--device=/dev/kfd --device=/dev/dri --network=host --ipc=host --group-add video -v /root:/root",
    bypassTerminal: false,
  },
  aiAgent: DEFAULT_AI_AGENT,
  embedding: DEFAULT_EMBEDDING,
  embedders: [DEFAULT_EMBEDDER],
};

function normalizeEmbedders(embedders?: AppConfig["embedders"]): AppConfig["embedders"] {
  const source = embedders && embedders.length > 0 ? embedders : [DEFAULT_EMBEDDER];
  return source.map((embedder, idx) => {
    const basePort = DEFAULT_EMBEDDER.port + idx;
    const name = embedder.name?.trim() || `embedder_${idx + 1}`;
    const port =
      idx === 0 && embedder.port === 8100 && name === "embedder_1"
        ? DEFAULT_EMBEDDER.port
        : embedder.port || basePort;
    return {
      ...DEFAULT_EMBEDDER,
      ...embedder,
      name,
      modelId: embedder.modelId?.trim() || DEFAULT_EMBEDDER.modelId,
      port,
      concurrency: embedder.concurrency || DEFAULT_EMBEDDER.concurrency,
      enabled: idx === 0 ? true : embedder.enabled ?? true,
      persistent: idx === 0 ? true : embedder.persistent ?? false,
      gpuMemoryUtilization: embedder.gpuMemoryUtilization || DEFAULT_EMBEDDER.gpuMemoryUtilization,
    };
  });
}

export default function App() {
  const [config, setConfig] = useState<AppConfig>(DEFAULT_CONFIG);
  const [tab, setTab] = useState<Tab>("pipeline");
  const [logs, setLogs] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [activeStreamId, setActiveStreamId] = useState<string | null>(null);
  const [connection, setConnection] = useState({
    isConnected: false,
    isTesting: false,
    message: "",
  });
  const [gpuStatus, setGpuStatus] = useState<GPUState | null>(null);
  const [autoPoll, setAutoPoll] = useState(false);
  const pollTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const [runsTick, setRunsTick] = useState(0); // bump to force RunDashboard reload
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [cwd, setCwd] = useState("/root");
  const [wizardStep, setWizardStep] = useState(0);
  const activeCmdOutputRef = useRef("");

  // --- Load persisted config on mount -----------------------------------
  useEffect(() => {
    (async () => {
      try {
        const loaded = await api.loadConfig();
        const merged = {
          ...DEFAULT_CONFIG,
          ...loaded,
          ssh: { ...DEFAULT_CONFIG.ssh, ...(loaded.ssh ?? {}) },
          qdrant: { ...DEFAULT_CONFIG.qdrant, ...(loaded.qdrant ?? {}), collection: loaded.qdrant?.collection || DEFAULT_CONFIG.qdrant.collection },
          digitalOcean: { ...DEFAULT_DIGITAL_OCEAN, ...(loaded.digitalOcean ?? {}) },
          teacher: { ...DEFAULT_TEACHER, ...(loaded.teacher ?? {}), servingEngine: "vllm" as const },
          student: { ...DEFAULT_CONFIG.student, ...(loaded.student ?? {}) },
          docker: { ...DEFAULT_CONFIG.docker, ...(loaded.docker ?? {}) },
          aiAgent: { ...DEFAULT_AI_AGENT, ...(loaded.aiAgent ?? {}) },
          // Always materialize the embedding block so downstream readiness
          // checks don't have to special-case a null value.
          embedding: {
            ...DEFAULT_EMBEDDING,
            ...(loaded.embedding ?? {}),
            apiKey: loaded.embedding?.apiKey || "",
          },
          embedders: normalizeEmbedders(loaded.embedders),
        };
        // Auto-align Qdrant endpoint to SSH host to ensure they use the correct GPU server Qdrant instance
        if (merged.ssh.host && (!merged.qdrant.endpoint || !merged.qdrant.endpoint.includes(merged.ssh.host))) {
          merged.qdrant.endpoint = `http://${merged.ssh.host}:6333`;
        }
        setConfig(merged);

        // Do NOT auto-test SSH on startup. The app must open instantly without
        // touching the GPU server — testing it on mount caused a long black
        // screen while waiting for the SSH handshake (or its timeout) when the
        // server was unreachable or slow. The user connects from the
        // Credentials tab when they're ready.
        setConnection({ isConnected: false, isTesting: false, message: "" });
      } catch (err) {
        console.error("load_config:", err);
      }
    })();
  }, []);

  // Automatically sync Qdrant endpoint when SSH host changes
  useEffect(() => {
    if (config.ssh.host) {
      const targetEndpoint = `http://${config.ssh.host}:6333`;
      if (config.qdrant.endpoint !== targetEndpoint) {
        setConfig((prev) => ({
          ...prev,
          qdrant: {
            ...prev.qdrant,
            endpoint: targetEndpoint,
          },
        }));
      }
    }
  }, [config.ssh.host]);

  // Persist whenever the user changes credentials/teacher.
  useEffect(() => {
    const t = setTimeout(() => {
      api.saveConfig(config).catch((e) => console.error("save_config:", e));
    }, 400);
    return () => clearTimeout(t);
  }, [config]);

  // One global subscription for pipeline events. Owned by App so logs keep
  // streaming into the in-memory store even when the user is on a different
  // tab and RunDashboard is unmounted.
  useEffect(() => {
    let teardown: (() => void) | null = null;
    let teardownSetup: (() => void) | null = null;
    startGlobalSubscription().then((fn) => {
      teardown = fn;
    });
    // Also mirror all setup://log events (teacher serving, embedder/OCR boot)
    // into a global buffer so the AI agent can read them on any page.
    startSetupLogSubscription().then((fn) => {
      teardownSetup = fn;
    });
    return () => {
      teardown?.();
      teardownSetup?.();
    };
  }, []);

  const fetchGpuStatus = useCallback(async () => {
    try {
      const s = await api.nvidiaSmi(config.ssh);
      setGpuStatus(s);
    } catch (e) {
      console.error("nvidia_smi:", e);
    }
  }, [config.ssh]);

  // --- Shell event subscriptions ----------------------------------------
  useEffect(() => {
    let active = true;
    let unsubLog: (() => void) | null = null;
    let unsubDone: (() => void) | null = null;

    (async () => {
      const uLog = await events.onShellLog((e: ShellLogEvent) => {
        if (!active) return;
        if (activeStreamId && e.streamId !== activeStreamId) return;
        
        activeCmdOutputRef.current += e.line;
        
        // Filter out the continuous directory tracking CWD blocks in real-time
        const currentAccum = activeCmdOutputRef.current;
        const marker = "---CWD---";
        const markerIdx = currentAccum.indexOf(marker);
        
        if (markerIdx === -1) {
          setLogs((prev) => prev + e.line);
        } else {
          const displayedRawLength = currentAccum.length - e.line.length;
          if (displayedRawLength < markerIdx) {
            const chunkPart = markerIdx - displayedRawLength;
            const appendText = e.line.substring(0, chunkPart);
            setLogs((prev) => prev + appendText);
          }
        }
      });
      if (!active) {
        uLog();
      } else {
        unsubLog = uLog;
      }

      const uDone = await events.onShellDone((e: ShellDoneEvent) => {
        if (!active) return;
        if (activeStreamId && e.streamId !== activeStreamId) return;

        // Parse out the CWD that resulted from command execution. We use
        // lastIndexOf because the SSH backend echoes the wrapped command back
        // to the log, and that echoed line literally contains "---CWD---" too.
        // The REAL pwd output is always the final occurrence.
        const rawOutput = activeCmdOutputRef.current;
        const marker = "---CWD---";
        const markerIdx = rawOutput.lastIndexOf(marker);
        if (markerIdx !== -1) {
          const rest = rawOutput.substring(markerIdx + marker.length).trim();
          if (rest) {
            const lines = rest.split(/[\r\n]+/);
            const newDir = lines[0].trim();
            // Only accept a plausible absolute POSIX path — guards against
            // garbage from a corrupted/echoed line ever poisoning the CWD.
            if (newDir && /^\/[^\s'"`;|&<>]*$/.test(newDir)) {
              setCwd(newDir);
            }
          }
        }

        setLogs((prev) => prev + `\n[SSH] Exit ${e.exitCode}\n`);
        setIsStreaming(false);
        setActiveStreamId(null);
        fetchGpuStatus();
      });
      if (!active) {
        uDone();
      } else {
        unsubDone = uDone;
      }
    })();

    return () => {
      active = false;
      if (unsubLog) unsubLog();
      if (unsubDone) unsubDone();
    };
  }, [activeStreamId, fetchGpuStatus]);

  // --- SSH actions ------------------------------------------------------
  const runCommand = useCallback(
    async (command: string, name: string) => {
      if (isStreaming) {
        setLogs((p) => p + `\n[busy] another stream running\n`);
        return;
      }
      setLogs((p) => p + `\n\n──────────────\n[RUN] ${name}\n──────────────\n`);
      setIsStreaming(true);
      activeCmdOutputRef.current = "";
      try {
        // hf token substitution
        let finalCmd =
          config.hfToken && command.includes("$HF_TOKEN")
            ? command.replace(/\$HF_TOKEN/g, config.hfToken)
            : command;
            
        // Wrap command to continuously track working directories.
        // The CWD marker is assembled at runtime via printf concatenation so
        // the literal string "---CWD---" never appears in the wrapper source
        // (the SSH backend echoes the full command back to the log, and we'd
        // otherwise find the marker inside the echoed line and parse garbage).
        // Also: if the saved cwd has somehow become invalid, fall back to /root.
        const safeCwd = /^\/[^\s'"`;|&<>]*$/.test(cwd) ? cwd : "/root";
        if (safeCwd !== cwd) {
          setCwd(safeCwd);
        }
        finalCmd = `cd "${safeCwd}" && ( ${finalCmd} ) ; CWD_EXIT_CODE=$? ; printf '%s%s\\n' '---' 'CWD---' ; pwd ; exit $CWD_EXIT_CODE`;
        
        const id = await api.sshStream(config.ssh, finalCmd);
        setActiveStreamId(id);
      } catch (e: any) {
        setLogs((p) => p + `\n[err] ${e.toString()}\n`);
        setIsStreaming(false);
      }
    },
    [config, isStreaming, cwd],
  );

  const stopStream = useCallback(async () => {
    if (activeStreamId) {
      try {
        await api.sshStopStream(activeStreamId);
      } catch (e) {
        console.error(e);
      }
    }
    setIsStreaming(false);
    setActiveStreamId(null);
  }, [activeStreamId]);

  const testConnection = useCallback(async () => {
    if (!config.ssh.host) {
      setConnection({
        isConnected: false,
        isTesting: false,
        message: "Enter a host first.",
      });
      return;
    }
    setConnection({ isConnected: false, isTesting: true, message: "" });
    try {
      const msg = await api.testSsh(config.ssh);
      setConnection({ isConnected: true, isTesting: false, message: msg });
      setLogs((p) => p + `\n[ssh-ok]\n${msg}\n`);
      fetchGpuStatus();
    } catch (e: any) {
      setConnection({
        isConnected: false,
        isTesting: false,
        message: String(e),
      });
    }
  }, [config.ssh]);

  // Auto-poll
  useEffect(() => {
    if (autoPoll) {
      fetchGpuStatus();
      pollTimer.current = setInterval(fetchGpuStatus, 5000);
    } else if (pollTimer.current) {
      clearInterval(pollTimer.current);
      pollTimer.current = null;
    }
    return () => {
      if (pollTimer.current) clearInterval(pollTimer.current);
    };
  }, [autoPoll, fetchGpuStatus]);

  const killProcess = async (pid: number) => {
    await runCommand(`kill -9 ${pid} && (rocm-smi -i 2>/dev/null || amd-smi list 2>/dev/null || true)`, `Kill PID ${pid}`);
  };

  // ----------------------------------------------------------------- UI
  return (
    <div className="h-screen overflow-hidden theme-app-bg flex flex-col font-sans theme-selection antialiased theme-text selection:bg-theme-accent/30 selection:text-white">
      <header className="theme-header border-b py-3 px-6 shrink-0 flex items-center justify-between select-none glass-panel z-50 sticky top-0 backdrop-blur-md">
        <div className="flex items-center space-x-4 group">
          <div className="p-2.5 theme-accent-bg text-black rounded-lg flex items-center justify-center premium-button shadow-lg shadow-theme-accent/20 group-hover:scale-105 transition-transform duration-300">
            <Layers className="w-5 h-5" />
          </div>
          <div>
            <h1 className="font-serif italic text-xl-fluid tracking-tight leading-none text-white font-black group-hover:text-theme-accent transition-colors duration-300">
              Fine-Tune Studio
            </h1>
            <p className="text-[10px] theme-muted uppercase tracking-[0.25em] mt-1.5 font-bold">
              Autonomous Synthetic Training Engine
            </p>
          </div>
        </div>
        <div className="flex items-center space-x-6">
          <ThemeSwitcher />
          <div className="hidden sm:flex items-center space-x-3 px-4 py-1.5 rounded-full theme-surface-soft border border-white/5 glass-panel">
            <span
              className={`w-2 h-2 rounded-full shadow-[0_0_8px_currentColor] ${
                connection.isConnected ? "bg-emerald-500 animate-pulse text-emerald-500" : "theme-accent-bg text-theme-accent"
              }`}
            />
            <span className="text-[10px] uppercase tracking-[0.1em] font-black theme-muted">
              {connection.isConnected ? "Droplet Live" : "Offline"}
            </span>
          </div>
        </div>
      </header>

      {/* Tab bar */}
      <div className="theme-tabbar border-b px-6 flex items-center justify-between gap-4 shrink-0 select-none h-11">
        <div className="flex items-center h-full">
          <TabButton active={tab === "pipeline"} onClick={() => setTab("pipeline")} icon={<Sparkles className="w-3.5 h-3.5" />} label="Pipeline" />
          <TabButton active={tab === "gpu"} onClick={() => setTab("gpu")} icon={<Server className="w-3.5 h-3.5" />} label="GPU Servers" />
          <TabButton active={tab === "credentials"} onClick={() => setTab("credentials")} icon={<Wrench className="w-3.5 h-3.5" />} label="Credentials" />
          <TabButton active={tab === "runs"} onClick={() => setTab("runs")} icon={<ListChecks className="w-3.5 h-3.5" />} label="Runs" />
          <TabButton active={tab === "deploy"} onClick={() => setTab("deploy")} icon={<Rocket className="w-3.5 h-3.5" />} label="Deploy" />
        </div>

        <div className="flex items-center space-x-3 text-xs-fluid font-mono h-full">
          {gpuStatus && (
            <div className="hidden md:flex items-center space-x-2 theme-surface-soft border border-white/5 px-3 py-1 rounded theme-muted">
              <span className="theme-faint uppercase font-bold text-[9px] tracking-wider">VRAM:</span>
              <span className="theme-text font-bold text-[9px]">
                {(gpuStatus.memoryUsed / 1024).toFixed(1)}GB / {(gpuStatus.memoryTotal / 1024).toFixed(0)}GB
              </span>
            </div>
          )}
        </div>
      </div>

      <main className="flex-1 p-6 lg:overflow-hidden overflow-y-auto theme-app-main min-h-0">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch lg:h-full">
          {/* Main Content Pane */}
          <section className="lg:col-span-8 flex flex-col lg:h-full lg:overflow-hidden">
            <div style={{ display: tab === "pipeline" ? "" : "none" }} className="w-full max-w-6xl mx-auto lg:h-full lg:overflow-y-auto pr-1 scrollbar-thin scrollbar-thumb-white/10">
              <PipelineWizard
                config={config}
                gpuStatus={gpuStatus}
                onConfigChange={(patch) => setConfig((c) => ({ ...c, ...patch }))}
                onPipelineLaunched={(runId) => {
                  setSelectedRunId(runId);
                  setRunsTick((t) => t + 1);
                  setTab("runs");
                }}
                onStepChange={(step) => setWizardStep(step)}
              />
            </div>
            <div style={{ display: tab === "credentials" ? "" : "none" }} className="max-w-7xl w-full mx-auto grid grid-cols-1 gap-6 items-start lg:h-full lg:overflow-y-auto pr-1 scrollbar-thin scrollbar-thumb-white/10">
              <CredentialsPanel
                config={config}
                onChange={(updated) => setConfig((prev) => ({ ...prev, ...updated }))}
                connection={connection}
                onTestConnection={testConnection}
              />
              <RoboticsWidget
                config={config}
                onChange={(patch) => setConfig((prev) => ({ ...prev, ...patch }))}
              />
            </div>
            <div style={{ display: tab === "gpu" ? "" : "none" }} className="w-full max-w-7xl mx-auto lg:h-full lg:overflow-y-auto pr-1 scrollbar-thin scrollbar-thumb-white/10">
              <GpuServerManager
                config={config}
                onConfigChange={(patch) => setConfig((prev) => ({ ...prev, ...patch }))}
              />
            </div>
            <div style={{ display: tab === "runs" ? "" : "none" }} className="w-full lg:h-full lg:overflow-y-auto pr-1 scrollbar-thin scrollbar-thumb-white/10">
              <RunDashboard refreshKey={runsTick} selectedRunId={selectedRunId} gpuStatus={gpuStatus} />
            </div>
            <div style={{ display: tab === "deploy" ? "" : "none" }} className="w-full lg:h-full lg:overflow-y-auto pr-1 scrollbar-thin scrollbar-thumb-white/10">
              <DeployPanel config={config} />
            </div>
          </section>

          {/* Persistent Sidebar (AI Terminal copilot) */}
          <aside className="lg:col-span-4 flex flex-col lg:h-full lg:overflow-y-auto pl-1 space-y-6 scrollbar-thin scrollbar-thumb-white/10">
            <AITerminalPanel
              logs={logs}
              isStreaming={isStreaming}
              onClearLogs={() => setLogs("")}
              onRunCustomCommand={(cmd) => runCommand(cmd, cmd)}
              onStopStreaming={stopStream}
              dockerEnabled={config.docker?.enabled ?? true}
              bypassTerminal={config.docker?.bypassTerminal ?? false}
              onToggleBypassTerminal={() =>
                setConfig((prev) => ({
                  ...prev,
                  docker: {
                    ...prev.docker,
                    bypassTerminal: !prev.docker.bypassTerminal,
                  },
                }))
              }
              cwd={cwd}
              config={config}
              onConfigChange={(patch) => setConfig((prev) => ({ ...prev, ...patch }))}
              activeTab={tab}
              wizardStep={wizardStep}
              gpuStatus={gpuStatus}
            />
          </aside>
        </div>
      </main>
    </div>
  );
}

function TabButton({ active, onClick, icon, label }: {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center space-x-2 px-5 h-full border-b-2 text-[10px] uppercase tracking-widest font-black font-mono transition-all duration-150 cursor-pointer ${
        active
          ? "border-theme-accent theme-accent bg-white/[0.015]"
          : "border-transparent theme-muted hover:theme-accent hover:bg-white/[0.005]"
      }`}
    >
      {icon}
      <span>{label}</span>
    </button>
  );
}
