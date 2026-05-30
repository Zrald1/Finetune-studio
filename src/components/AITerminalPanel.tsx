import React, { useRef, useEffect, useState } from "react";
import {
  Terminal as TerminalIcon,
  Play,
  Trash2,
  Copy,
  CircleSlash,
  Sparkles,
  CheckCircle2,
  Folder,
  Send,
  Loader2,
  Settings,
  Eye,
  EyeOff
} from "lucide-react";
import type { AppConfig, GPUState, AIAgentConfig, Run } from "../types";
import { DEFAULT_AI_AGENT, POPULAR_MODELS } from "../types";
import { api } from "../lib/tauri";
import { getStream as getRunStream, hydrateFromDisk as hydrateRunFromDisk } from "../lib/runStreams";
import { getSetupLogTail, hasSetupLogs } from "../lib/setupLogs";

interface Message {
  role: "user" | "assistant" | "system";
  content: string;
}

interface AITerminalPanelProps {
  logs: string;
  isStreaming: boolean;
  onClearLogs: () => void;
  onRunCustomCommand: (cmd: string) => void;
  onStopStreaming: () => void;
  dockerEnabled: boolean;
  bypassTerminal: boolean;
  onToggleBypassTerminal: () => void;
  cwd: string;
  config: AppConfig;
  onConfigChange: (patch: Partial<AppConfig>) => void;
  activeTab: string;
  wizardStep: number;
  gpuStatus: GPUState | null;
}

export default function AITerminalPanel({
  logs,
  isStreaming,
  onClearLogs,
  onRunCustomCommand,
  onStopStreaming,
  dockerEnabled,
  bypassTerminal,
  onToggleBypassTerminal,
  cwd,
  config,
  onConfigChange,
  activeTab,
  wizardStep,
  gpuStatus
}: AITerminalPanelProps) {
  const [inputVal, setInputVal] = useState("");
  const [mode, setMode] = useState<"ai" | "cmd">("ai");
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content:
        "Hello! I am your Fine-Tune Studio AI agent. I can **run commands on your GPU server over SSH** AND **read your live logs directly** — training runs (stage messages, step metrics, loss curves), teacher / embedder / OCR boot logs, and the visible terminal console. I automatically look at whatever page you're on. Ask me about the training run, the teacher serving, an embedder, `rocm-smi`, containers — anything."
    }
  ]);
  const [isLoadingResponse, setIsLoadingResponse] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [showApiKey, setShowApiKey] = useState(false);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [loadingModels, setLoadingModels] = useState(false);
  const [queueLength, setQueueLength] = useState(0);
  const [autoRunCommands, setAutoRunCommands] = useState(true);

  const abortControllerRef = useRef<AbortController | null>(null);
  const messageQueueRef = useRef<string[]>([]);
  const isProcessingRef = useRef(false);
  const commandHistoryRef = useRef<string[]>([]);
  const historyIndexRef = useRef(-1);
  const inputRef = useRef<HTMLInputElement>(null);

  // Tracks AI-initiated commands so we can pipe their output back to the model.
  // Commands are executed STRICTLY sequentially — the next one only fires after
  // the previous SSH stream has finished.
  const aiCommandQueueRef = useRef<string[]>([]);
  const currentAiCommandRef = useRef<string | null>(null);
  const aiResultsRef = useRef<Array<{ cmd: string; output: string }>>([]);
  const logsSnapshotRef = useRef<string>("");
  const wasStreamingRef = useRef<boolean>(false);
  const pendingAiFeedbackRef = useRef<boolean>(false);
  const lastAutoRunIndexRef = useRef<number>(-1);

  // Helper: extract bash/sh fenced blocks from an assistant message
  const extractCommands = (content: string): string[] => {
    const cmds: string[] = [];
    const regex = /```(\w*)\n([\s\S]*?)```/g;
    let m;
    while ((m = regex.exec(content)) !== null) {
      const lang = (m[1] || "").toLowerCase();
      if (lang === "sh" || lang === "bash") cmds.push(m[2].trim());
    }
    return cmds;
  };

  // Special marker the AI uses to request local logs (no SSH).
  // Syntax inside a ```bash block:
  //   __exec_logs__              → tail of the most recently active TRAINING run
  //   __exec_logs__ <runId>      → tail of a specific run
  //   __exec_logs__ <runId> 400  → last 400 lines of a specific run
  //   __exec_logs__ setup        → teacher serving / embedder / OCR boot logs
  //   __exec_logs__ terminal     → the live terminal console the user sees
  //   __exec_logs__ setup 600    → setup/terminal logs with a custom line count
  const EXEC_LOGS_MARKER = "__exec_logs__";
  const isExecLogsCommand = (cmd: string) => cmd.trim().startsWith(EXEC_LOGS_MARKER);

  // Resolve which run the AI is asking about. Returns null when nothing exists.
  const resolveRunForLogs = async (token: string | undefined): Promise<Run | null> => {
    try {
      const runs = await api.listRuns();
      if (!runs || runs.length === 0) return null;
      if (token) {
        const exact = runs.find((r) => r.id === token);
        if (exact) return exact;
        const partial = runs.find((r) => r.id.startsWith(token) || r.name === token);
        if (partial) return partial;
      }
      // Default: pick the run with the most recent updatedAt
      const sorted = [...runs].sort((a, b) => (b.updatedAt || "").localeCompare(a.updatedAt || ""));
      return sorted[0] || null;
    } catch (e) {
      console.warn("listRuns failed:", e);
      return null;
    }
  };

  // Pull the live training execution logs (in-memory stream + persisted tail)
  // and return a string ready to feed back to the AI.
  const fetchExecLogsForFeedback = async (rawCmd: string): Promise<string> => {
    const parts = rawCmd.trim().split(/\s+/).slice(1); // drop marker
    const runToken = parts[0];
    const tailLines = Math.min(parseInt(parts[1] || "300", 10) || 300, 2000);

    // --- Non-run log sources --------------------------------------------
    // "terminal" → the live console the user sees in the right pane.
    if (runToken === "terminal") {
      const text = logs || "";
      if (!text.trim()) {
        return "(the terminal console is empty — no command output yet.)";
      }
      const trimmed = text.split(/\r?\n/).slice(-tailLines).join("\n");
      return ["--- Live terminal console ---", trimmed].join("\n");
    }

    // "setup" → teacher serving / embedder / PaddleOCR / Qdrant boot logs.
    // These are emitted as setup://log events and captured globally so they
    // survive tab navigation.
    if (runToken === "setup") {
      if (!hasSetupLogs()) {
        return "(no setup/serving logs captured yet — nothing has been booted this session. Start a teacher, embedder, or OCR service from the Pipeline/Credentials tab, or check the SSH host is reachable.)";
      }
      const trimmed = getSetupLogTail(tailLines);
      return [
        "--- Teacher / embedder / OCR setup & serving logs ---",
        trimmed || "(setup log buffer empty)",
      ].join("\n");
    }

    const run = await resolveRunForLogs(runToken);
    if (!run) {
      return "(no runs found — there's nothing currently training. Start a run from the Pipeline tab, or check the SSH host is reachable.)";
    }

    // Ensure we have the persisted tail (no-op if already hydrated)
    await hydrateRunFromDisk(run.id);

    // Prefer the live in-memory stream (includes [stage]/metric chatter live)
    const stream = getRunStream(run.id);
    let logText = stream.logs || "";

    // Fallback: pull a fresh tail from disk if the in-memory buffer is empty
    if (!logText) {
      try {
        logText = await api.readRunLog(run.id, 96 * 1024);
      } catch {
        logText = "";
      }
    }

    const lines = logText.split(/\r?\n/);
    const trimmed = lines.slice(-tailLines).join("\n");

    const progress = stream.progress
      ? `scanned=${stream.progress.scanned} kept=${stream.progress.kept} rejected=${stream.progress.rejected}`
      : "n/a";
    const lastMetric = stream.metrics.length > 0 ? stream.metrics[stream.metrics.length - 1] : null;
    const metricLine = lastMetric
      ? `step=${lastMetric.step} loss=${lastMetric.loss?.toFixed?.(4) ?? lastMetric.loss} epoch=${lastMetric.epoch}`
      : "no train metrics yet";

    return [
      `Run: ${run.name} (id=${run.id})`,
      `Status: ${run.status}`,
      `Student: ${run.studentModel}  |  Teacher: ${run.teacherModel}`,
      `QA: total=${run.qaTotal} kept=${run.qaKept} rejected=${run.qaRejected}`,
      `Progress: ${progress}`,
      `Latest metric: ${metricLine}`,
      `Updated: ${run.updatedAt}`,
      "",
      "--- Execution log tail ---",
      trimmed || "(log buffer empty)",
    ].join("\n");
  };

  // Auto-abort request on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  const provider = config.aiAgent?.provider ?? "vultr";
  const staticModels = POPULAR_MODELS[provider] || [];
  // Show at most 5 models per provider. Prefer the live-fetched list (already
  // capped + freshest-first by the backend); fall back to the curated static
  // list when no models were fetched. The currently-selected model is always
  // kept in the list so the dropdown never drops the active selection.
  const displayModels = React.useMemo(() => {
    const source = availableModels.length > 0 ? availableModels : staticModels;
    const capped = Array.from(new Set(source)).slice(0, 5);
    const selected = config.aiAgent?.modelId ?? "";
    if (selected && !capped.includes(selected)) {
      return [selected, ...capped].slice(0, 6);
    }
    return capped;
  }, [staticModels, availableModels, provider, config.aiAgent?.modelId]);
  const PROVIDER_NAMES: Record<string, string> = {
    vultr: "Vultr",
    openai: "OpenAI",
    anthropic: "Anthropic",
    gemini: "Gemini",
    groq: "Groq",
    xai: "xAI",
    custom: "Custom",
  };
  const agentName = PROVIDER_NAMES[provider] || provider;

  // Debounced model list fetching
  useEffect(() => {
    const provider = config.aiAgent?.provider ?? "vultr";
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

  const patchAiAgent = (p: Partial<AIAgentConfig>) => {
    const current = config.aiAgent ?? DEFAULT_AI_AGENT;
    const next = { ...current, ...p };
    if (p.provider && p.provider !== current.provider) {
      if (p.provider === "vultr") {
        next.apiUrl = "https://api.vultrinference.com/v1";
        next.modelId = "deepseek-chat";
      } else if (p.provider === "openai") {
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
      } else if (p.provider === "custom") {
        next.apiUrl = next.apiUrl || "";
        next.modelId = next.modelId || "";
      }
    }
    onConfigChange({ aiAgent: next });
  };

  const terminalContainerRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  // Smart auto-scroll for terminal console
  useEffect(() => {
    const el = terminalContainerRef.current;
    if (!el) return;
    
    // Scroll to bottom only if the user is already near the bottom (within 60px)
    const isAtBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 60;
    if (isAtBottom || !logs) {
      el.scrollTop = el.scrollHeight;
    }
  }, [logs]);

  // Smart auto-scroll for AI chat history
  useEffect(() => {
    const el = chatContainerRef.current;
    if (!el) return;
    if (messages.length === 0) return;
    
    const lastMsg = messages[messages.length - 1];
    const isUser = lastMsg.role === "user";
    const isAtBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 60;
    
    // Always scroll to bottom for user-sent messages, otherwise only if already at the bottom
    if (isUser || isAtBottom) {
      el.scrollTop = el.scrollHeight;
    }
  }, [messages]);

  // Kicks off the next queued AI command IF the SSH lane is idle.
  // Special intercept: __exec_logs__ pulls local training logs synchronously
  // instead of going over SSH.
  // Safe to call repeatedly — it's a no-op when busy or empty.
  const pumpAiQueue = async () => {
    if (currentAiCommandRef.current !== null) return; // already running one
    if (isStreaming) return; // SSH lane busy
    const next = aiCommandQueueRef.current.shift();
    if (!next) return;

    // Local intercept — execution logs from in-memory stream, no SSH
    if (isExecLogsCommand(next)) {
      const output = await fetchExecLogsForFeedback(next);
      aiResultsRef.current.push({ cmd: next, output });
      setMessages((prev) => [
        ...prev,
        { role: "system", content: `✓ Read execution logs: ${next}` },
      ]);

      // More queued? Keep pumping. Else flush feedback to the AI.
      if (aiCommandQueueRef.current.length > 0) {
        setTimeout(pumpAiQueue, 50);
        return;
      }
      flushAiFeedback();
      return;
    }

    currentAiCommandRef.current = next;
    logsSnapshotRef.current = logs;
    onRunCustomCommand(next);
  };

  // Bundle the accumulated AI command results and feed them back to the model
  const flushAiFeedback = () => {
    if (!pendingAiFeedbackRef.current) return;
    pendingAiFeedbackRef.current = false;
    if (aiResultsRef.current.length === 0) return;

    const bundle = aiResultsRef.current
      .map(
        (r, i) =>
          `### Command ${i + 1}: \`${r.cmd}\`\n\`\`\`\n${r.output}\n\`\`\``,
      )
      .join("\n\n");

    const feedbackContent =
      `Here are the outputs from the ${aiResultsRef.current.length} command(s) you just ran. Interpret the results for the user, surface anything notable, and propose the next step.\n\n${bundle}`;

    aiResultsRef.current = [];

    if (isProcessingRef.current) {
      messageQueueRef.current.push(feedbackContent);
      setQueueLength(messageQueueRef.current.length);
    } else {
      setMessages((prev) => [...prev, { role: "user", content: feedbackContent }]);
      processAIMessage(feedbackContent);
    }
  };

  // Auto-execute fenced bash blocks in the most recent assistant message — exactly once per message
  useEffect(() => {
    if (!autoRunCommands) return;
    const lastIdx = messages.length - 1;
    if (lastIdx <= lastAutoRunIndexRef.current) return;
    const last = messages[lastIdx];
    if (!last || last.role !== "assistant") return;
    lastAutoRunIndexRef.current = lastIdx;

    const cmds = extractCommands(last.content);
    if (cmds.length === 0) return;

    // Enqueue and reset the result bundle for this batch
    aiResultsRef.current = [];
    cmds.forEach((c) => aiCommandQueueRef.current.push(c));
    pendingAiFeedbackRef.current = true;

    // Start the pipeline (a tiny delay lets the React tree settle)
    setTimeout(pumpAiQueue, 300);
  }, [messages, autoRunCommands]);

  // Track streaming transitions for AI-initiated commands. Capture each command's
  // output as it finishes, then drive the next one — or bundle results to the AI.
  useEffect(() => {
    // Streaming just started
    if (isStreaming && !wasStreamingRef.current) {
      wasStreamingRef.current = true;
      // If we just kicked an AI command, the snapshot was set in pumpAiQueue.
      // For user-initiated commands, snapshot here as a safety net.
      if (!currentAiCommandRef.current) {
        logsSnapshotRef.current = logs;
      }
      return;
    }

    // Streaming just finished
    if (!isStreaming && wasStreamingRef.current) {
      wasStreamingRef.current = false;

      // Was this an AI-initiated command? If not, do nothing.
      const cmd = currentAiCommandRef.current;
      if (!cmd) return;
      currentAiCommandRef.current = null;

      // Capture the output delta produced by this command
      const delta = logs.slice(logsSnapshotRef.current.length);
      const MAX_PER_CMD = 4000;
      let trimmed = delta.trim();
      if (trimmed.length > MAX_PER_CMD) {
        trimmed = trimmed.slice(0, MAX_PER_CMD) + `\n... [truncated ${delta.length - MAX_PER_CMD} chars]`;
      }
      aiResultsRef.current.push({ cmd, output: trimmed || "(no output)" });

      // Visible chat marker
      setMessages((prev) => [
        ...prev,
        { role: "system", content: `✓ Executed: ${cmd}` }
      ]);

      // More commands? Run the next one
      if (aiCommandQueueRef.current.length > 0) {
        setTimeout(pumpAiQueue, 200);
        return;
      }

      // Pipeline drained — package all results and feed them back to the AI
      flushAiFeedback();
    }
  }, [isStreaming, logs]);

  const copyToClipboard = () => {
    navigator.clipboard.writeText(logs);
  };

  const handleRunCommand = (cmd: string) => {
    if (!cmd.trim() || isStreaming) return;
    onRunCustomCommand(cmd);
  };

  const handleCancelThinking = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  };

  const triggerNextQueuedMessage = () => {
    if (messageQueueRef.current.length > 0) {
      const nextQuery = messageQueueRef.current.shift();
      if (nextQuery) {
        setQueueLength(messageQueueRef.current.length);
        processAIMessage(nextQuery);
      }
    }
  };

  const processAIMessage = async (query: string) => {
    setIsLoadingResponse(true);
    isProcessingRef.current = true;
    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    try {
      const agentConfig = config.aiAgent || {
        provider: "vultr" as const,
        apiUrl: "https://api.vultrinference.com/v1",
        apiKey: "",
        modelId: "deepseek-chat"
      };

      const apiKey = agentConfig.apiKey || "";
      if (!apiKey) {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content:
              "To enable autonomous SSH execution, please configure your **AI Agent** API key in the **Settings** panel (the gear icon in the terminal header).\n\nOnce configured, I can run commands directly on your GPU server — just ask."
          }
        ]);
        return;
      }

      const provider = agentConfig.provider || "vultr";
      const apiUrl = (agentConfig.apiUrl || "").trim();
      const modelId = (agentConfig.modelId || "").trim() || "meta-llama/Meta-Llama-3.1-70B-Instruct";

      // Build the endpoint path; auth headers are applied backend-side by the
      // ai_proxy_chat command (keeps API keys/CORS out of the browser).
      let endpoint = apiUrl;
      if (provider === "anthropic") {
        if (!endpoint.includes("/messages")) {
          endpoint = endpoint.replace(/\/+$/, "") + "/messages";
        }
      } else {
        if (!endpoint.includes("/chat/completions")) {
          endpoint = endpoint.replace(/\/+$/, "") + "/chat/completions";
        }
      }

      const stepLabels = ["Knowledge Base", "Teacher Configuration", "Dataset / Synthesis", "Student & Train"];
      const stepName = stepLabels[wizardStep] || `Step ${wizardStep}`;

      // Derive what the user is currently looking at, and which log source the
      // agent should reach for FIRST when they ask about "the logs" / progress.
      let pageContext: string;
      let preferredLogSource: string;
      if (activeTab === "credentials") {
        pageContext = "Credentials tab (SSH / Qdrant / embedders / OCR setup)";
        preferredLogSource = "`__exec_logs__ setup` (embedder / OCR / Qdrant boot logs)";
      } else if (activeTab === "gpu") {
        pageContext = "GPU Servers tab (droplet & GPU management)";
        preferredLogSource = "`__exec_logs__ setup` for any serving/boot output, or `__exec_logs__ terminal` for the visible console";
      } else if (activeTab === "runs") {
        pageContext = "Runs tab (training run dashboard)";
        preferredLogSource = "`__exec_logs__` (the most recent training run — metrics, loss, stages)";
      } else if (activeTab === "pipeline") {
        pageContext = `Pipeline tab → Step "${stepName}"`;
        if (wizardStep <= 1) {
          // Step 0 Knowledge Base / Step 1 Teacher → serving & setup logs.
          preferredLogSource = "`__exec_logs__ setup` (teacher serving + embedder boot logs)";
        } else {
          // Step 2 Dataset / Step 3 Student & Train → training run logs.
          preferredLogSource = "`__exec_logs__` (the active training run)";
        }
      } else {
        pageContext = activeTab;
        preferredLogSource = "`__exec_logs__` (training run) or `__exec_logs__ setup` / `__exec_logs__ terminal`";
      }

      const gpuDesc = gpuStatus
        ? `${gpuStatus.gpuName} | Temp: ${gpuStatus.temperature}°C | Util: ${gpuStatus.utilizationGpu}% | VRAM: ${(gpuStatus.memoryUsed / 1024).toFixed(1)}GB / ${(gpuStatus.memoryTotal / 1024).toFixed(0)}GB`
        : "Offline / Unknown";

      // Build a compact summary of recent runs + their live status. This gives
      // the AI awareness of what's currently training without it having to ask.
      let runsSummary = "(no run data available)";
      try {
        const runs = await api.listRuns();
        if (runs && runs.length > 0) {
          const sorted = [...runs].sort((a, b) => (b.updatedAt || "").localeCompare(a.updatedAt || ""));
          const lines: string[] = [];
          for (const r of sorted.slice(0, 5)) {
            const s = getRunStream(r.id);
            const lastMetric = s.metrics.length > 0 ? s.metrics[s.metrics.length - 1] : null;
            const metricStr = lastMetric
              ? `step=${lastMetric.step} loss=${typeof lastMetric.loss === "number" ? lastMetric.loss.toFixed(4) : lastMetric.loss} ep=${lastMetric.epoch}`
              : "no metrics";
            const progStr = s.progress
              ? `scanned=${s.progress.scanned} kept=${s.progress.kept} rej=${s.progress.rejected}`
              : "no progress";
            lines.push(`- id=${r.id}  status=${r.status}  student=${r.studentModel}  ${metricStr}  ${progStr}`);
          }
          runsSummary = lines.join("\n");
        }
      } catch (e) {
        runsSummary = `(listRuns failed: ${String(e)})`;
      }

      const systemPrompt = `You are a Fine-Tune Studio AI Agent with **direct SSH execution access** to the user's AMD ROCm GPU server AND **direct access to the local training Execution Logs**. You help them manage training pipelines, monitor GPU state, inspect live training runs, and operate the remote machine.

**HOW YOU EXECUTE COMMANDS — IMPORTANT:**
You have TWO execution channels. Use whichever fits the question:

**(A) Remote SSH commands** — any shell command in a fenced \`\`\`bash or \`\`\`sh block runs on the GPU server over SSH, and the stdout/stderr is streamed back to you in the next turn as a system message labeled "Command output".

**(B) Local log reader** — to inspect any logs WITHOUT going over SSH, emit a \`\`\`bash block whose ONLY content starts with the literal token \`__exec_logs__\`. There are THREE log sources:

1. **Training runs** (default):
\`\`\`bash
__exec_logs__
\`\`\`
→ tail of the MOST RECENTLY ACTIVE training run (metadata + status + metrics + last 300 lines).
\`__exec_logs__ <runId>\` targets a specific run (IDs in "Recent runs" below); \`__exec_logs__ <runId> 600\` sets the line count (max 2000).

2. **Teacher / embedder / OCR setup & serving logs**:
\`\`\`bash
__exec_logs__ setup
\`\`\`
→ the boot/serving output for the teacher (vLLM/SGLang), embedder models, PaddleOCR, and Qdrant — i.e. the logs shown while a model is loading on the Credentials/Pipeline pages. \`__exec_logs__ setup 600\` for more lines.

3. **The live terminal console** (what the user sees in the right pane):
\`\`\`bash
__exec_logs__ terminal
\`\`\`
→ the raw output of the last commands run in the terminal.

This reader works **even when the SSH server is unreachable** — it reads local in-memory buffers + on-disk tails. **When the user asks about logs / progress / "what's happening", pick the source that matches the page they're on (see "Current page" below) and use it BEFORE any SSH command.** For training/loss/step questions use source 1; for teacher-loading or embedder-boot questions use source 2; for "what did that command print" use source 3.

Rules for command execution:
1. When the user asks to check status, run diagnostics, inspect logs, restart services, or operate the server — **just run the command yourself**. Never tell the user to switch modes, click buttons, or copy/paste anything.
2. Emit exactly the commands you want to run inside \`\`\`bash blocks. They will execute in order.
3. Keep each block to ONE logical command (you can chain with && or use multi-line scripts when needed).
4. Prefer non-interactive commands. Add \`2>&1\` and \`| head -200\` to bound long output.
5. After output comes back, briefly interpret it for the user and propose the next step (or run it).
6. Never invent output. If you haven't run a command yet, say so and then run it.
7. **If SSH commands repeatedly fail with [CONNECTION ERROR] / timeout / os error 10060**, STOP retrying the same SSH command. Instead: (a) fall back to \`__exec_logs__\` for run state, and (b) tell the user the SSH host is unreachable and suggest checking the cloud console — do NOT keep hammering the dead host.
8. Do NOT say things like "switch to Shell Direct mode" — that guidance is obsolete. You ARE the executor.

**Useful SSH commands on this server:**
- \`rocm-smi\` — GPU status (temp, VRAM, utilization)
- \`docker ps -a\` — All containers
- \`ls -la /root/fine-tune/runs/\` — List training runs (on remote)
- \`tail -100 /root/fine-tune/runs/*/live.log\` — Live training output (remote)
- \`nvidia-smi\` / \`amd-smi list\` — Alternate GPU views
- \`df -h /root\` — Disk space
- \`free -h\` — System RAM

**Current System State:**
- SSH Host: ${config.ssh.host || "Not configured"}
- Docker: ${config.docker?.enabled ? "Enabled" : "Disabled"}
- Student: ${config.student?.repoId || "Not set"}
- Active GPU: ${gpuDesc}
- Working dir: ${cwd}
- Wizard step: ${stepName}

**Current page:** The user is currently viewing the **${pageContext}**. When they ask about "the logs", progress, or what's happening, prefer ${preferredLogSource} first. \`__exec_logs__ terminal\` is always available for the visible console.

**Recent runs (most recent first):**
${runsSummary}

Be proactive, concise, and act like an operator who runs the command first and explains the output second.`;

      let currentMessages: Message[] = [];
      setMessages((prev) => {
        currentMessages = prev;
        return prev;
      });

      const validHistory = currentMessages
        .slice(-10)
        .filter((m) => m.role === "user" || m.role === "assistant");

      let bodyData: any;
      if (provider === "anthropic") {
        const anthropicHistory = validHistory.map((m) => ({
          role: m.role === "assistant" ? ("assistant" as const) : ("user" as const),
          content: m.content
        }));
        bodyData = {
          model: modelId,
          system: systemPrompt,
          messages: [...anthropicHistory, { role: "user", content: query }],
          max_tokens: 4096,
          temperature: 0.2
        };
      } else {
        const apiHistory = validHistory.map((m) => ({
          role: m.role,
          content: m.content
        }));
        bodyData = {
          model: modelId,
          messages: [
            { role: "system", content: systemPrompt },
            ...apiHistory,
            { role: "user", content: query }
          ],
          temperature: 0.2
        };
      }

      // Always proxy through the Rust backend to dodge browser CORS — including
      // Anthropic, which the backend handles with x-api-key/version headers.
      const responseText = await api.aiProxyChat(endpoint, apiKey, bodyData, provider);

      const data = JSON.parse(responseText);
      let answer = "";
      if (provider === "anthropic") {
        answer = data.content?.[0]?.text || "No reply from Claude.";
      } else {
        answer = data.choices?.[0]?.message?.content || "No reply from AI agent.";
      }

      setMessages((prev) => [...prev, { role: "assistant", content: answer }]);
    } catch (err: any) {
      if (err.name === "AbortError") {
        setMessages((prev) => [
          ...prev,
          {
            role: "system",
            content: `✕ Agent thinking cancelled by user.`
          }
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: `✕ **Agent Error**: Failed to fetch response. Details: ${err.message || String(err)}`
          }
        ]);
      }
    } finally {
      abortControllerRef.current = null;
      setIsLoadingResponse(false);
      isProcessingRef.current = false;
      triggerNextQueuedMessage();
    }
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    const query = inputVal.trim();
    if (!query) return;

    if (mode === "cmd") {
      // Direct shell command
      handleRunCommand(query);
      setInputVal("");
      return;
    }

    // AI Mode
    const userMsg: Message = { role: "user", content: query };
    setMessages((prev) => [...prev, userMsg]);
    setInputVal("");

    if (isProcessingRef.current) {
      messageQueueRef.current.push(query);
      setQueueLength(messageQueueRef.current.length);
    } else {
      processAIMessage(query);
    }
  };

  const parseMessageContent = (msg: Message) => {
    const commands: string[] = [];
    const prompts: string[] = [];
    let match;
    const regex = /```(\w*)\n([\s\S]*?)```/g;
    
    while ((match = regex.exec(msg.content)) !== null) {
      const lang = (match[1] || "").toLowerCase();
      const content = match[2].trim();
      if (lang === "sh" || lang === "bash") {
        commands.push(content);
      } else if (content.includes("{topic}") && content.includes("{chunk_text}")) {
        prompts.push(content);
      } else {
        if (content.includes("{topic}") && content.includes("{chunk_text}")) {
          prompts.push(content);
        }
      }
    }

    // Auto-execution is handled by the dedicated useEffect that watches `messages`,
    // so we just render UI affordances here.

    const formattedContent = msg.content
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.*?)\*/g, "<em>$1</em>")
      .replace(/`([^`]+)`/g, "<code class='bg-white/10 px-1 rounded font-mono text-[10.5px]'>$1</code>");

    return (
      <div className="space-y-2">
        <p
          className="whitespace-pre-wrap text-[11px] leading-relaxed break-words font-medium opacity-90"
          dangerouslySetInnerHTML={{ __html: formattedContent }}
        />
        {autoRunCommands && commands.length > 0 && (
          <div className="text-[8px] theme-accent font-mono animate-pulse">
            ⏳ Auto-executing {commands.length} command(s)...
          </div>
        )}
        {!autoRunCommands && commands.map((cmd, idx) => (
          <button
            key={`cmd-${idx}`}
            type="button"
            onClick={() => handleRunCommand(cmd)}
            disabled={isStreaming}
            className="w-full flex items-center gap-2 mt-1.5 px-3 py-2 rounded-xl bg-theme-accent-soft border border-theme-accent/20 hover:brightness-110 active:scale-[0.99] transition-all text-left group"
          >
            <Play className="w-3.5 h-3.5 theme-accent shrink-0 fill-current group-hover:scale-110 transition-transform" />
            <div className="flex-1 min-w-0">
              <div className="text-[8px] font-black theme-accent uppercase tracking-widest font-mono">Run proposed command</div>
              <code className="text-[9.5px] font-mono text-white/95 truncate block">{cmd}</code>
            </div>
          </button>
        ))}
        {prompts.map((promptText, idx) => (
          <button
            key={`prompt-${idx}`}
            type="button"
            onClick={() => {
              onConfigChange({ promptTemplate: promptText });
              alert("Applied AI-generated template prompt to the Pipeline Wizard.");
            }}
            className="w-full flex items-center gap-2 mt-1.5 px-3 py-2 rounded-xl bg-theme-accent-soft border border-theme-accent/20 hover:brightness-110 active:scale-[0.99] transition-all text-left group"
          >
            <Sparkles className="w-3.5 h-3.5 theme-accent shrink-0 fill-current group-hover:scale-110 transition-transform animate-pulse" />
            <div className="flex-1 min-w-0">
              <div className="text-[8px] font-black theme-accent uppercase tracking-widest font-mono">Apply as Pipeline Prompt</div>
              <code className="text-[9.5px] font-mono text-white/95 truncate block">{promptText.split("\n")[0]}...</code>
            </div>
          </button>
        ))}
      </div>
    );
  };

  return (
    <div className="premium-card rounded-2xl overflow-hidden flex flex-col h-[760px] animate-premium glass-panel relative border border-white/5 shadow-2xl">
      <div className="absolute top-0 left-0 w-full h-1 theme-accent-bg opacity-30" />

      {/* ── TERMINAL HALF ────────────────────────────────────────────────── */}
      <div className="flex-1 min-h-[300px] flex flex-col border-b border-white/5 relative">
        {/* Terminal Header */}
        <div className="px-5 py-3.5 border-b border-white/5 flex items-center justify-between bg-white/[0.01] backdrop-blur-md">
          <div className="flex items-center space-x-3">
            <TerminalIcon className="w-4 h-4 theme-accent animate-pulse" />
            <div className="flex flex-col">
              <div className="flex items-center gap-2">
                <TerminalIcon className="w-4 h-4 theme-accent" />
                <span className="text-[9px] uppercase tracking-[0.25em] theme-accent font-black font-mono leading-none">TERMINAL</span>
                {gpuStatus && (
                  <div className="flex items-center gap-1.5 text-[8px] font-mono">
                    <span className={`w-1.5 h-1.5 rounded-full ${gpuStatus.simulated ? "bg-amber-400" : "bg-emerald-400 animate-pulse"}`} />
                    <span className="text-white/60">{gpuStatus.utilizationGpu}%</span>
                    <span className="text-white/40">|</span>
                    <span className="text-white/60">{gpuStatus.temperature}°C</span>
                  </div>
                )}
              </div>
              <span className="text-[8px] theme-faint font-mono uppercase mt-1 tracking-wider">{cwd}</span>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            {gpuStatus && (
              <div className="flex items-center gap-2 px-2 py-1 bg-black/30 rounded border border-white/5 text-[9px] font-mono">
                <div className="flex flex-col items-center">
                  <span className="text-[7px] text-white/40 uppercase">VRAM</span>
                  <span className="text-white font-bold">{Math.round((gpuStatus.memoryUsed / (gpuStatus.memoryTotal || 1)) * 100)}%</span>
                </div>
                <div className="w-px h-6 bg-white/10" />
                <div className="flex flex-col items-center">
                  <span className="text-[7px] text-white/40 uppercase">GPU</span>
                  <span className="text-white font-bold">{gpuStatus.utilizationGpu}%</span>
                </div>
              </div>
            )}
            {isStreaming && (
              <button
                onClick={onStopStreaming}
                className="flex items-center space-x-1.5 px-2 py-1 bg-red-500/10 border border-red-500/20 text-red-400 rounded text-[8px] font-black font-mono hover:bg-red-500 hover:text-white transition-all premium-button uppercase tracking-widest"
              >
                <CircleSlash className="w-2.5 h-2.5" />
                <span>Kill</span>
              </button>
            )}
            <div className="flex items-center gap-0.5 bg-white/5 rounded p-0.5 border border-white/5 shadow-inner">
              <button
                onClick={copyToClipboard}
                className="p-1 theme-faint hover:theme-text hover:bg-white/5 rounded transition-all group"
                title="Copy terminal logs"
              >
                <Copy className="w-3 h-3 group-hover:scale-110 transition-transform" />
              </button>
              <button
                onClick={onClearLogs}
                className="p-1 theme-faint hover:text-red-400 hover:bg-white/5 rounded transition-all group"
                title="Clear console"
              >
                <Trash2 className="w-3 h-3 group-hover:scale-110 transition-transform" />
              </button>
              <button
                onClick={() => setShowSettings(!showSettings)}
                className={`p-1 rounded transition-all group ${showSettings ? "theme-accent bg-white/5" : "theme-faint hover:theme-text hover:bg-white/5"}`}
                title="AI Agent settings"
              >
                <Settings className="w-3 h-3 group-hover:rotate-45 transition-transform duration-300" />
              </button>
            </div>
          </div>
        </div>

        {/* Collapsible Settings Panel */}
        {showSettings && (
          <div className="bg-black/95 border-b border-white/5 p-4 space-y-3 z-20 relative animate-premium text-xs font-mono">
            <div className="flex items-center justify-between pb-1 border-b border-white/5">
              <span className="text-[9px] uppercase tracking-wider theme-accent font-black flex items-center gap-1.5">
                <Settings className="w-3 h-3" /> AI Copilot Settings
              </span>
              <button
                onClick={() => setShowSettings(false)}
                className="text-[9px] theme-muted hover:theme-text uppercase font-bold"
              >
                Close
              </button>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1">
                <label className="text-[8px] uppercase tracking-wider theme-muted">Provider</label>
                <select
                  value={config.aiAgent?.provider ?? "vultr"}
                  onChange={(e) => patchAiAgent({ provider: e.target.value as any })}
                  className="w-full px-2 py-1 bg-black/40 border border-white/5 rounded text-[10px] focus:outline-none focus:border-theme-accent/50 text-white"
                >
                  <option value="vultr">Vultr</option>
                  <option value="openai">OpenAI (ChatGPT)</option>
                  <option value="anthropic">Anthropic (Claude)</option>
                  <option value="gemini">Google Gemini</option>
                  <option value="groq">Groq</option>
                  <option value="xai">xAI (X)</option>
                  <option value="custom">Custom API</option>
                </select>
              </div>

              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <label className="text-[8px] uppercase tracking-wider theme-muted">Model ID</label>
                  {loadingModels && (
                    <span className="text-[7px] font-mono theme-accent animate-pulse">FETCHING...</span>
                  )}
                </div>
                {displayModels.length > 0 ? (
                  <div className="space-y-1.5">
                    <select
                      value={displayModels.includes(config.aiAgent?.modelId ?? "") ? (config.aiAgent?.modelId ?? "") : "custom"}
                      onChange={(e) => {
                        if (e.target.value === "custom") {
                          patchAiAgent({ modelId: "" });
                        } else {
                          patchAiAgent({ modelId: e.target.value });
                        }
                      }}
                      className="w-full px-2 py-1 bg-black/40 border border-white/5 rounded text-[10px] focus:outline-none focus:border-theme-accent/50 text-white"
                    >
                      {displayModels.map((m) => (
                        <option key={m} value={m}>{m}</option>
                      ))}
                      <option value="custom">Custom (Type manually)...</option>
                    </select>
                    {(!displayModels.includes(config.aiAgent?.modelId ?? "") || !config.aiAgent?.modelId) && (
                      <input
                        type="text"
                        placeholder="Type custom model..."
                        value={config.aiAgent?.modelId ?? ""}
                        onChange={(e) => patchAiAgent({ modelId: e.target.value })}
                        className="w-full px-2 py-1 bg-black/40 border border-white/5 rounded text-[10px] focus:outline-none focus:border-theme-accent/50 text-white"
                      />
                    )}
                  </div>
                ) : (
                  <input
                    type="text"
                    placeholder="meta-llama/..."
                    value={config.aiAgent?.modelId ?? ""}
                    onChange={(e) => patchAiAgent({ modelId: e.target.value })}
                    className="w-full px-2 py-1 bg-black/40 border border-white/5 rounded text-[10px] focus:outline-none focus:border-theme-accent/50 text-white"
                  />
                )}
              </div>
            </div>

            <div className="space-y-1">
              <label className="text-[8px] uppercase tracking-wider theme-muted">API URL</label>
              <input
                type="text"
                placeholder="https://api.vultrinference.com/v1"
                value={config.aiAgent?.apiUrl ?? ""}
                onChange={(e) => patchAiAgent({ apiUrl: e.target.value })}
                className="w-full px-2 py-1 bg-black/40 border border-white/5 rounded text-[10px] focus:outline-none focus:border-theme-accent/50 text-white"
              />
            </div>

            <div className="space-y-1">
              <div className="flex items-center justify-between">
                <label className="text-[8px] uppercase tracking-wider theme-muted">API Key</label>
                <button
                  type="button"
                  onClick={() => setShowApiKey(!showApiKey)}
                  className="text-[8px] uppercase theme-accent hover:brightness-125"
                >
                  {showApiKey ? "Hide" : "Show"}
                </button>
              </div>
              <input
                type={showApiKey ? "text" : "password"}
                placeholder="rc_... or sk-..."
                value={config.aiAgent?.apiKey ?? ""}
                onChange={(e) => patchAiAgent({ apiKey: e.target.value })}
                className="w-full px-2 py-1 bg-black/40 border border-white/5 rounded text-[10px] focus:outline-none focus:border-theme-accent/50 text-white"
              />
            </div>
            <div className="flex items-center justify-between pt-2 border-t border-white/5">
              <label className="text-[8px] uppercase tracking-wider theme-muted">Auto-execute commands</label>
              <button
                type="button"
                onClick={() => setAutoRunCommands(!autoRunCommands)}
                className={`w-10 h-5 rounded-full transition-all ${autoRunCommands ? "bg-theme-accent" : "bg-white/20"} relative`}
              >
                <span className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-all ${autoRunCommands ? "left-5" : "left-0.5"}`} />
              </button>
            </div>
            <div className="flex gap-2 pt-2 border-t border-white/5">
              <button
                type="button"
                onClick={() => setShowSettings(false)}
                className="flex-1 px-3 py-2 bg-white/5 border border-white/10 rounded text-[10px] font-bold uppercase tracking-widest theme-text hover:bg-white/10 transition"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => {
                  onConfigChange({ aiAgent: config.aiAgent });
                  setShowSettings(false);
                }}
                className="flex-1 px-3 py-2 theme-accent-bg text-black rounded text-[10px] font-bold uppercase tracking-widest hover:brightness-110 transition"
              >
                Save & Use
              </button>
            </div>
          </div>
        )}

        {/* Terminal Logs Display */}
        <div ref={terminalContainerRef} className="flex-1 p-5 overflow-y-auto cursor-text font-mono text-[11px] leading-relaxed theme-text/80 space-y-1 bg-black/50 selection:bg-theme-selection scrollbar-thin scrollbar-thumb-white/10">
          {!logs ? (
            <div className="h-full flex flex-col items-center justify-center space-y-3 opacity-30 select-none py-8">
              <TerminalIcon className="w-6 h-6 text-white/50" />
              <div className="text-center">
                <p className="text-[9px] font-black font-mono uppercase tracking-widest">Console Inactive</p>
                <p className="text-[7.5px] theme-muted font-mono uppercase mt-1">Ready for input uplink...</p>
              </div>
            </div>
          ) : (
            <pre className="whitespace-pre-wrap break-all animate-premium max-w-full overflow-x-hidden" style={{ fontFamily: "inherit" }}>
              {logs}
              {isStreaming && (
                <span className="inline-block w-1.5 h-3 theme-accent-bg ml-1.5 animate-pulse" />
              )}
            </pre>
          )}
        </div>
      </div>

      {/* ── AI AGENT HALF ────────────────────────────────────────────────── */}
      <div className="flex-1 min-h-[260px] flex flex-col relative bg-black/10">
        {/* Chat Header */}
        <div className="px-5 py-3 border-b border-white/5 flex items-center justify-between bg-white/[0.01] backdrop-blur-md">
          <div className="flex items-center space-x-2">
            <Sparkles className="w-3.5 h-3.5 theme-accent" />
            <span className="text-[9px] uppercase tracking-[0.25em] theme-text font-black font-mono">{agentName} Agent</span>
            {queueLength > 0 && (
              <span className="text-[7.5px] font-mono theme-accent-soft theme-accent border border-theme-accent/20 px-1.5 py-0.2 rounded bg-theme-accent/5 ml-2 animate-pulse font-black">
                {queueLength} QUEUED
              </span>
            )}
          </div>
          {isLoadingResponse && (
            <div className="flex items-center gap-2">
              <div className="flex items-center gap-1.5 text-[8px] font-mono theme-accent animate-pulse">
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span>AGENT THINKING...</span>
              </div>
              <button
                type="button"
                onClick={handleCancelThinking}
                className="px-2 py-0.5 bg-red-500/10 border border-red-500/20 text-red-400 rounded text-[8px] font-black font-mono hover:bg-red-500 hover:text-white transition-all uppercase tracking-widest"
              >
                Cancel
              </button>
            </div>
          )}
        </div>

        {/* Chat Messages */}
        <div ref={chatContainerRef} className="flex-1 p-5 overflow-y-auto space-y-4 scrollbar-thin scrollbar-thumb-white/10">
          {messages.map((msg, idx) => {
            if (msg.role === "system") {
              return (
                <div key={idx} className="flex justify-center my-2 animate-premium">
                  <div className="px-3 py-1.5 rounded-xl border border-red-500/10 bg-red-500/5 text-red-400/90 text-[10px] font-mono tracking-tight shadow-sm max-w-[90%] text-center font-bold">
                    {msg.content}
                  </div>
                </div>
              );
            }
            const isAI = msg.role === "assistant";
            return (
              <div
                key={idx}
                className={`flex ${isAI ? "justify-start" : "justify-end"} animate-premium`}
              >
                <div
                  className={`max-w-[85%] rounded-2xl px-4 py-3 border text-xs shadow-md ${
                    isAI
                      ? "theme-surface border-white/5 bg-white/[0.015]"
                      : "theme-accent-soft theme-accent border-theme-accent/20 bg-theme-accent/5"
                  }`}
                >
                  <div className="flex items-center justify-between gap-4 mb-1.5 opacity-40 font-mono text-[8px] font-black uppercase tracking-wider">
                    <span>{isAI ? "AI Copilot" : "You"}</span>
                  </div>
                  {isAI ? parseMessageContent(msg) : <p className="whitespace-pre-wrap text-[11px] leading-relaxed break-words font-medium">{msg.content}</p>}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── BOTTOM INPUT PANEL ────────────────────────────────────────────── */}
      <div className="px-5 py-4 border-t border-white/5 bg-black/40 space-y-3">
        {/* Mode Selector Pill Toggle */}
        <div className="flex justify-between items-center gap-4">
          <div className="flex bg-white/5 rounded-xl p-0.5 border border-white/5 shadow-inner">
            <button
              type="button"
              onClick={() => setMode("ai")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[9px] font-black uppercase tracking-widest transition-all ${
                mode === "ai"
                  ? "theme-accent-soft theme-accent border border-theme-accent/20"
                  : "theme-faint hover:theme-text"
              }`}
            >
              <Sparkles className="w-3 h-3" />
              <span>AI Agent</span>
            </button>
            <button
              type="button"
              onClick={() => setMode("cmd")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[9px] font-black uppercase tracking-widest transition-all ${
                mode === "cmd"
                  ? "theme-accent-soft theme-accent border border-theme-accent/20"
                  : "theme-faint hover:theme-text"
              }`}
            >
              <TerminalIcon className="w-3 h-3" />
              <span>Shell Direct</span>
            </button>
          </div>

          <div className="flex items-center gap-1 text-[8.5px] font-mono theme-muted select-none">
            <Folder className="w-3 h-3 opacity-50 shrink-0" />
            <span className="truncate max-w-[120px] font-semibold">{cwd}</span>
          </div>
        </div>

        {/* Input Prompter */}
        <form onSubmit={handleSendMessage} className="flex items-center gap-2.5">
          {mode === "cmd" && (
            <div className="flex items-center gap-1 px-2.5 py-2 rounded-xl bg-theme-accent/10 border border-theme-accent/20 shadow-inner">
              <span className="text-[10px] font-black font-mono theme-accent select-none tracking-tighter">$</span>
            </div>
          )}
          <div className="relative flex-1">
            <input
              ref={inputRef}
              type="text"
              value={inputVal}
              onChange={(e) => {
                setInputVal(e.target.value);
                historyIndexRef.current = -1;
              }}
              onKeyDown={(e) => {
                if (e.key === "ArrowUp") {
                  e.preventDefault();
                  const hist = commandHistoryRef.current;
                  if (hist.length === 0) return;
                  const newIdx = historyIndexRef.current === -1 ? hist.length - 1 : Math.max(0, historyIndexRef.current - 1);
                  historyIndexRef.current = newIdx;
                  setInputVal(hist[newIdx] || "");
                } else if (e.key === "ArrowDown") {
                  e.preventDefault();
                  const hist = commandHistoryRef.current;
                  if (historyIndexRef.current === -1) return;
                  const newIdx = historyIndexRef.current + 1;
                  if (newIdx >= hist.length) {
                    historyIndexRef.current = -1;
                    setInputVal("");
                  } else {
                    historyIndexRef.current = newIdx;
                    setInputVal(hist[newIdx] || "");
                  }
                }
              }}
              disabled={isStreaming || (mode === "cmd" && isLoadingResponse)}
              placeholder={
                mode === "cmd"
                  ? isStreaming
                    ? "Executing..."
                    : "Enter command (↑↓ for history)..."
                  : "Ask AI Agent or type /runs, /gpu, /cancel..."
              }
              className="w-full bg-white/5 border border-white/5 rounded-xl px-4 py-3 text-sm-fluid font-mono text-white focus:outline-none focus:border-theme-accent/50 placeholder-white/20 disabled:opacity-30 tracking-tight pr-10"
              autoComplete="off"
            />
            {inputVal && (
              <button
                type="button"
                onClick={() => {
                  if (mode === "cmd" && inputVal.trim()) {
                    commandHistoryRef.current = [inputVal, ...commandHistoryRef.current.slice(0, 49)];
                  }
                  setInputVal("");
                }}
                className="absolute right-3 top-1/2 -translate-y-1/2 p-1 theme-faint hover:theme-text rounded transition-all"
              >
                <Trash2 className="w-3 h-3" />
              </button>
            )}
          </div>
          <button
            type="submit"
            disabled={isStreaming || (mode === "cmd" && isLoadingResponse) || !inputVal.trim()}
            className="p-3 theme-accent-bg text-black hover:brightness-125 border-none rounded-xl transition-all duration-300 disabled:opacity-20 premium-button shadow-lg shadow-theme-accent/20 shrink-0"
          >
            {isLoadingResponse ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Send className="w-3.5 h-3.5" />
            )}
          </button>
        </form>
      </div>
    </div>
  );
}
