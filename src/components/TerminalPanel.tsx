import React, { useRef, useEffect, useState } from "react";
import { Terminal, ShieldAlert, Play, Trash2, Copy, CircleSlash, ArrowRight, CheckCircle2 } from "lucide-react";

interface TerminalPanelProps {
  logs: string;
  isStreaming: boolean;
  onClearLogs: () => void;
  onRunCustomCommand: (cmd: string) => void;
  onStopStreaming: () => void;
  dockerEnabled: boolean;
  bypassTerminal: boolean;
  onToggleBypassTerminal: () => void;
}

export default function TerminalPanel({
  logs,
  isStreaming,
  onClearLogs,
  onRunCustomCommand,
  onStopStreaming,
  dockerEnabled,
  bypassTerminal,
  onToggleBypassTerminal,
}: TerminalPanelProps) {
  const [inputCmd, setInputCmd] = useState("");
  const terminalEndRef = useRef<HTMLDivElement>(null);

  // Scroll terminal logs automatically
  useEffect(() => {
    if (terminalEndRef.current) {
      terminalEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputCmd.trim() || isStreaming) return;
    onRunCustomCommand(inputCmd);
    setInputCmd("");
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(logs);
  };

  return (
    <div className="premium-card rounded-2xl overflow-hidden flex flex-col h-[600px] animate-premium glass-panel relative">
      <div className="absolute top-0 left-0 w-full h-1 theme-accent-bg opacity-20" />
      
      {/* Terminal Title Bar */}
      <div className="px-6 py-4 border-b border-white/5 flex items-center justify-between bg-white/[0.01] backdrop-blur-md">
        <div className="flex items-center space-x-4">
          {/* Virtual OS buttons */}
          <div className="flex space-x-2 shrink-0 select-none">
            <span className="w-3 h-3 rounded-full bg-red-500/30 border border-red-500/20 inline-block" />
            <span className="w-3 h-3 rounded-full bg-yellow-500/30 border border-yellow-500/20 inline-block" />
            <span className="w-3 h-3 rounded-full bg-green-500/30 border border-green-500/20 inline-block" />
          </div>
          <div className="flex items-center pl-4 border-l border-white/10">
            <div className="flex flex-col">
              <span className="text-[10px] uppercase tracking-[0.25em] theme-accent font-black font-mono leading-none">SSH Uplink</span>
              <span className="text-[8px] theme-faint font-mono uppercase mt-1">Status: Operational // Secure Tunnel</span>
            </div>
          </div>
        </div>

        {/* Action Controls */}
        <div className="flex items-center space-x-3">
          {dockerEnabled && (
            <label className="flex items-center space-x-2 px-3 py-1.5 rounded-lg border border-white/5 theme-surface-soft hover:bg-white/5 transition-all cursor-pointer group shadow-sm">
              <div className={`w-3.5 h-3.5 rounded border transition-all flex items-center justify-center ${!bypassTerminal ? "bg-theme-accent border-theme-accent" : "border-white/20 group-hover:border-white/40"}`}>
                {!bypassTerminal && <CheckCircle2 className="w-2.5 h-2.5 text-black" />}
              </div>
              <span className="text-[9px] font-black font-mono theme-muted uppercase tracking-widest group-hover:theme-text transition-colors">Wrap Docker</span>
              <input
                type="checkbox"
                checked={!bypassTerminal}
                onChange={onToggleBypassTerminal}
                className="hidden"
              />
            </label>
          )}
          {isStreaming && (
            <button
              id="terminal-stop-btn"
              onClick={onStopStreaming}
              className="flex items-center space-x-2 px-3 py-1.5 bg-red-500/10 border border-red-500/20 text-red-400 rounded-lg text-[9px] font-black font-mono hover:bg-red-500 hover:text-white transition-all premium-button shadow-lg shadow-red-500/10 uppercase tracking-widest"
              title="Terminate command execution"
            >
              <CircleSlash className="w-3 h-3 animate-pulse" />
              <span>Kill Session</span>
            </button>
          )}
          <div className="flex items-center gap-1 bg-white/5 rounded-lg p-1 border border-white/5 shadow-inner">
            <button
              id="terminal-copy-btn"
              onClick={copyToClipboard}
              className="p-1.5 theme-faint hover:theme-text hover:bg-white/5 rounded-md transition-all group"
              title="Copy logs"
            >
              <Copy className="w-3.5 h-3.5 group-hover:scale-110 transition-transform" />
            </button>
            <button
              id="terminal-clear-btn"
              onClick={onClearLogs}
              className="p-1.5 theme-faint hover:text-red-400 hover:bg-white/5 rounded-md transition-all group"
              title="Clear terminal text"
            >
              <Trash2 className="w-3.5 h-3.5 group-hover:scale-110 transition-transform" />
            </button>
          </div>
        </div>
      </div>

      {/* Terminal Display Logs */}
      <div className="flex-1 p-6 overflow-y-auto cursor-text font-mono text-sm-fluid leading-relaxed theme-text/80 space-y-1 bg-black/40 selection:bg-theme-selection scrollbar-thin scrollbar-thumb-white/10 scroll-smooth">
        {!logs ? (
          <div className="h-full flex flex-col items-center justify-center space-y-4 py-12 select-none opacity-40">
            <div className="w-16 h-16 rounded-full border-2 border-dashed border-white/20 flex items-center justify-center animate-[spin_10s_linear_infinite]">
              <Terminal className="w-8 h-8 text-white/50" />
            </div>
            <div className="text-center space-y-1">
              <p className="text-sm-fluid font-black font-mono uppercase tracking-[0.2em]">Terminal Idle</p>
              <p className="text-[10px] theme-muted font-mono uppercase tracking-widest">Awaiting Secure Link Initiation...</p>
            </div>
          </div>
        ) : (
          <div className="whitespace-pre-wrap animate-premium">
            {logs}
            {isStreaming && (
              <span className="inline-block w-2 h-4 theme-accent-bg ml-2 animate-pulse vertical-mid shadow-[0_0_8px_currentColor]" />
            )}
          </div>
        )}
        <div ref={terminalEndRef} />
      </div>

      {/* Terminal Manual CLI Prompt Input */}
      <form
        onSubmit={handleSubmit}
        className="px-6 py-4 border-t border-white/5 flex items-center space-x-3 bg-black/20"
      >
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 border border-white/5 shadow-inner">
          <span className="text-[11px] font-black font-mono theme-accent select-none tracking-tighter">root@remote</span>
          <span className="text-[11px] font-black font-mono theme-faint select-none">:~$</span>
        </div>
        <input
          id="terminal-prompt-input-field"
          type="text"
          value={inputCmd}
          onChange={(e) => setInputCmd(e.target.value)}
          placeholder={isStreaming ? "Pipeline streaming context active..." : "Enter remote shell command..."}
          disabled={isStreaming}
          className="flex-1 bg-transparent font-mono text-sm-fluid focus:outline-none text-white placeholder-white/10 disabled:opacity-30 tracking-tight"
          autoComplete="off"
        />
        <button
          id="terminal-prompt-submit-btn"
          type="submit"
          disabled={isStreaming || !inputCmd.trim()}
          className="p-2.5 theme-accent-bg text-black hover:brightness-125 border-none rounded-xl transition-all duration-300 disabled:opacity-20 premium-button shadow-lg shadow-theme-accent/20"
        >
          <ArrowRight className="w-4 h-4" />
        </button>
      </form>
    </div>
  );
}

