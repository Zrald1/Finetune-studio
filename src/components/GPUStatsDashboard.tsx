import { AppConfig, GPUState } from "../types";
import { RefreshCw, Cpu, Zap, Thermometer, Activity } from "lucide-react";

interface GPUStatsDashboardProps {
  status: GPUState | null;
  config: AppConfig;
  onConfigChange: (patch: Partial<AppConfig>) => void;
  isLoading: boolean;
  onRefresh: () => void;
  autoPoll: boolean;
  onToggleAutoPoll: () => void;
}

export default function GPUStatsDashboard({
  status,
  config,
  onConfigChange,
  isLoading,
  onRefresh,
  autoPoll,
  onToggleAutoPoll,
}: GPUStatsDashboardProps) {
  if (!status) {
    return null;
  }

  const vramPercent = Math.min(100, Math.round((status.memoryUsed / (status.memoryTotal || 100)) * 100));
  
  // Dynamic color for temperature
  const getTempColor = (temp: number) => {
    if (temp < 50) return { text: "text-green-400", progress: "bg-green-400" };
    if (temp < 75) return { text: "theme-accent", progress: "theme-accent-bg" };
    return { text: "text-red-400", progress: "bg-red-500" };
  };

  const tempCls = getTempColor(status.temperature);

  return (
    <div className="space-y-6 animate-premium">
      {/* Dashboard Top Header Bar */}
      <div className="premium-card rounded-2xl p-6 flex flex-col md:flex-row md:items-center justify-between gap-4 glass-panel relative overflow-hidden">
        <div className="absolute top-0 left-0 w-1.5 h-full theme-accent-bg opacity-50" />
        <div className="pl-2">
          <div className="flex items-center space-x-3">
            <span className={`w-2.5 h-2.5 rounded-full shadow-[0_0_10px_currentColor] ${status.simulated ? "bg-amber-400 text-amber-500" : "theme-accent-bg animate-pulse text-theme-accent"}`} />
            <h4 className="font-black text-white text-lg-fluid tracking-tight font-serif italic">
              {status.gpuName}
            </h4>
            {status.simulated && (
              <span className="text-[9px] font-black font-mono theme-accent-soft theme-accent border border-theme-accent/30 px-2 py-0.5 rounded-full tracking-[0.2em] uppercase bg-black/40">
                Emulated
              </span>
            )}
          </div>
          <p className="text-[10px] theme-muted mt-2 flex items-center gap-3 font-black uppercase tracking-[0.15em]">
            <span className="flex items-center gap-1.5"><span className="opacity-40">DRIVER:</span> <strong className="text-white">{status.driverVersion}</strong></span>
            <span className="w-1 h-1 rounded-full bg-white/10" />
            <span className="flex items-center gap-1.5"><span className="opacity-40">RUNTIME:</span> <strong className="text-white">{status.cudaVersion}</strong></span>
          </p>
        </div>

        {/* Polling & Fetch triggers */}
        <div className="flex items-center space-x-3 shrink-0 self-end md:self-auto">
          <button
            onClick={onToggleAutoPoll}
            className={`px-5 py-2.5 rounded-xl text-[10px] uppercase tracking-[0.2em] font-black transition-all duration-300 premium-button border ${
              autoPoll
                ? "theme-accent-soft theme-accent border-theme-accent/40 shadow-lg shadow-theme-accent/5"
                : "bg-white/5 border-white/10 theme-muted hover:theme-text hover:bg-white/10"
            }`}
          >
            {autoPoll ? "SYSTEM POLLING" : "MANUAL SYNC"}
          </button>

          <button
            id="btn-refresh-gpu-stats"
            onClick={onRefresh}
            disabled={isLoading}
            className="p-3 bg-white/5 hover:bg-white/10 border border-white/5 theme-text rounded-xl transition-all duration-300 disabled:opacity-30 premium-button group"
            title="Refresh diagnostics"
          >
            <RefreshCw className={`w-4 h-4 group-hover:rotate-180 transition-transform duration-500 ${isLoading ? "animate-spin" : ""}`} />
          </button>
        </div>
      </div>

      {/* Grid: Temperature, Core Utilization, Power details */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        {/* VRAM Card */}
        <div className="premium-card rounded-2xl p-5 flex flex-col justify-between group glass-panel">
          <div className="flex items-center justify-between mb-4">
            <span className="text-[10px] uppercase tracking-[0.2em] theme-muted font-black">Memory Pool</span>
            <Cpu className="w-4 h-4 theme-faint group-hover:theme-accent transition-colors" />
          </div>
          
          <div className="mb-4">
            <div className="flex items-baseline gap-1">
              <span className="text-3xl font-black tracking-tight text-white font-mono">
                {(status.memoryUsed / 1024).toFixed(2)}
              </span>
              <span className="text-xs theme-faint font-mono font-bold uppercase tracking-tighter">
                / {(status.memoryTotal / 1024).toFixed(1)} GB
              </span>
            </div>
          </div>

          <div className="space-y-2">
            <div className="w-full bg-white/5 rounded-full h-2 overflow-hidden border border-white/5 shadow-inner">
              <div
                className="theme-accent-bg h-full transition-all duration-1000 ease-out shadow-[0_0_10px_currentColor]"
                style={{ width: `${vramPercent}%` }}
              />
            </div>
            <div className="flex items-center justify-between text-[10px] font-black font-mono tracking-wider">
              <span className="theme-accent">{vramPercent}% LOAD</span>
              <span className="theme-muted opacity-50">AVAILABLE: {Math.max(0, Math.round((status.memoryTotal - status.memoryUsed)/1024))}GB</span>
            </div>
          </div>
        </div>

        {/* Temperature Card */}
        <div className="premium-card rounded-2xl p-5 flex flex-col justify-between group glass-panel">
          <div className="flex items-center justify-between mb-4">
            <span className="text-[10px] uppercase tracking-[0.2em] theme-muted font-black">Thermal Core</span>
            <Thermometer className="w-4 h-4 theme-faint group-hover:text-red-400 transition-colors" />
          </div>

          <div className="mb-4 flex items-baseline">
            <span className="text-3xl font-black tracking-tight font-mono text-white">
              {status.temperature}
            </span>
            <span className="text-sm font-black theme-muted font-mono ml-1">°C</span>
            <span className={`ml-3 text-[9px] tracking-[0.2em] uppercase font-black font-mono px-2 py-0.5 border rounded-lg bg-black/20 ${tempCls.text} ${tempCls.text.replace('text-', 'border-')}/30`}>
              {status.temperature < 50 ? "Stable" : status.temperature < 75 ? "Optimal" : "Threshold"}
            </span>
          </div>

          <div className="space-y-2">
            <div className="w-full bg-white/5 rounded-full h-2 overflow-hidden border border-white/5 shadow-inner">
              <div
                className={`h-full transition-all duration-500 ease-out shadow-[0_0_10px_currentColor] ${tempCls.progress}`}
                style={{ width: `${Math.min(100, (status.temperature / 100) * 100)}%` }}
              />
            </div>
            <div className="flex items-center text-[10px] font-black font-mono tracking-wider theme-muted opacity-50">
              CRITICAL LIMIT: 85°C
            </div>
          </div>
        </div>

        {/* GPU core engine workload */}
        <div className="premium-card rounded-2xl p-5 flex flex-col justify-between group glass-panel">
          <div className="flex items-center justify-between mb-4">
            <span className="text-[10px] uppercase tracking-[0.2em] theme-muted font-black">Compute Engine</span>
            <Activity className="w-4 h-4 theme-faint group-hover:text-cyan-400 transition-colors" />
          </div>

          <div className="mb-4">
            <span className="text-3xl font-black tracking-tight text-white font-mono">
              {status.utilizationGpu}
            </span>
            <span className="text-sm font-black theme-muted font-mono ml-1">%</span>
          </div>

          <div className="space-y-2">
            <div className="w-full bg-white/5 rounded-full h-2 overflow-hidden border border-white/5 shadow-inner">
              <div
                className="bg-cyan-500 h-full transition-all duration-500 ease-out shadow-[0_0_10px_#06b6d4]"
                style={{ width: `${status.utilizationGpu}%` }}
              />
            </div>
            <div className="flex items-center justify-between text-[10px] font-black font-mono tracking-wider text-cyan-400/70">
              PARALLEL TENSOR WORKLOAD
            </div>
          </div>
        </div>

        {/* Power drawing watt stats */}
        <div className="premium-card rounded-2xl p-5 flex flex-col justify-between group glass-panel">
          <div className="flex items-center justify-between mb-4">
            <span className="text-[10px] uppercase tracking-[0.2em] theme-muted font-black">Power Consumption</span>
            <Zap className="w-4 h-4 theme-faint group-hover:text-amber-400 transition-colors" />
          </div>

          <div className="mb-4">
            <div className="flex items-baseline gap-1">
              <span className="text-3xl font-black tracking-tight text-white font-mono">
                {Math.round(status.powerDraw)}
              </span>
              <span className="text-xs theme-faint font-mono font-bold uppercase tracking-tighter">
                / {Math.round(status.powerLimit)} W
              </span>
            </div>
          </div>

          <div className="space-y-2">
            <div className="w-full bg-white/5 rounded-full h-2 overflow-hidden border border-white/5 shadow-inner">
              <div
                className="bg-amber-500 h-full transition-all duration-500 ease-out shadow-[0_0_10px_#f59e0b]"
                style={{ width: `${Math.min(100, (status.powerDraw / (status.powerLimit || 250)) * 100)}%` }}
              />
            </div>
            <div className="flex items-center justify-between text-[10px] font-black font-mono tracking-wider text-amber-500/70">
              ENERGY EFFICIENCY: {Math.round((status.powerDraw / (status.powerLimit || 1)) * 100)}%
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
