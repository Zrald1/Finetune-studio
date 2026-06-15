import React, { useEffect, useState } from "react";
import type { LoraConfig, MatchedGuideInfo } from "../types";
import { api } from "../lib/tauri";
import { METHOD_OPTIONS, methodInfo } from "../method";
import type { FineTuneMethod } from "../method";
import { ChevronRight, Plus, RefreshCw, Trash2 } from "lucide-react";

export interface StudentModelOption {
  id: string;
  label: string;
  source: "merged" | "hf";
}

interface Props {
  value: LoraConfig;
  onChange: (next: LoraConfig) => void;
  studentModel: string;
  onStudentChange: (repoId: string) => void;
  studentModelOptions?: StudentModelOption[];
  hfLoading?: boolean;
  hfTokenSet?: boolean;
  onRefreshModels?: () => void;
  directTraining?: boolean;
}

export default function TrainingConfigForm({
  value,
  onChange,
  studentModel,
  onStudentChange,
  studentModelOptions = [],
  hfLoading = false,
  hfTokenSet = false,
  onRefreshModels,
  directTraining = false,
}: Props) {
  const set = <K extends keyof LoraConfig>(k: K, v: LoraConfig[K]) =>
    onChange({ ...value, [k]: v });
  const method: FineTuneMethod = value.method || "lora";
  const selectedMethod = methodInfo(method, value.customMethodName);
  const copy = selectedMethod.copy;
  const usesAdapterFields = selectedMethod.usesAdapterFields;
  const customCommands = value.customCommands?.length ? value.customCommands : [""];
  const mergedOptions = studentModelOptions.filter((o) => o.source === "merged");
  const hfOptions = studentModelOptions.filter((o) => o.source === "hf");
  const latestMerged = mergedOptions[0];
  const selectedOption = studentModelOptions.some((o) => o.id === studentModel) ? studentModel : "";
  const [matchedGuide, setMatchedGuide] = useState<MatchedGuideInfo | null>(null);
  const [guideLoading, setGuideLoading] = useState(false);

  useEffect(() => {
    if (!studentModel.trim()) { setMatchedGuide(null); return; }
    const timer = setTimeout(async () => {
      setGuideLoading(true);
      try { setMatchedGuide(await api.matchModelGuide(studentModel)); } catch { setMatchedGuide(null); }
      finally { setGuideLoading(false); }
    }, 500);
    return () => clearTimeout(timer);
  }, [studentModel]);

  const selectMethod = (nextMethod: LoraConfig["method"]) => {
    if (nextMethod === "custom" && (!value.customCommands || value.customCommands.length === 0)) {
      onChange({
        ...value,
        method: "custom",
        customMethodName: value.customMethodName || "My Fine-tune Method",
        customCommands: ["llamafactory-cli train $TRAIN_YAML"],
      });
      return;
    }
    set("method", nextMethod);
  };

  const setCustom = (patch: Pick<LoraConfig, "customMethodName" | "customCommands">) => {
    onChange({ ...value, ...patch, method: "custom" });
  };

  const setCommand = (idx: number, command: string) => {
    const next = [...customCommands];
    next[idx] = command;
    setCustom({ customCommands: next });
  };

  const addCommand = () => setCustom({ customCommands: [...customCommands, ""] });
  const removeCommand = (idx: number) => {
    const next = customCommands.filter((_, i) => i !== idx);
    setCustom({ customCommands: next.length ? next : [""] });
  };

  const NumField = ({ k, label, step }: { k: keyof LoraConfig; label: string; step?: number }) => (
    <div className="space-y-2">
      <label className="text-[10px] uppercase tracking-widest theme-muted font-black ml-1">{label}</label>
      <input
        type="number"
        step={step ?? 1}
        value={(value[k] ?? "") as number | ""}
        onChange={(e) => set(k, Number(e.target.value) as any)}
        className="w-full px-4 py-2.5 premium-input rounded-xl text-sm-fluid font-mono text-white focus:outline-none shadow-inner"
      />
    </div>
  );

  return (
    <div className="space-y-6 animate-premium">
      <div className="space-y-2">
        <label className="text-[10px] uppercase tracking-widest theme-muted font-black ml-1">Student Model <span className="opacity-40 font-mono tracking-normal">(HF REPO ID)</span></label>
        <input
          type="text"
          value={studentModel}
          onChange={(e) => onStudentChange(e.target.value)}
          placeholder="Qwen/Qwen2.5-7B-Instruct"
          className="w-full px-4 py-3 premium-input rounded-xl text-sm-fluid font-mono text-white focus:outline-none shadow-inner"
        />
        {matchedGuide && (
          <div className="mt-1.5 px-3 py-1.5 rounded-lg bg-theme-accent/10 border border-theme-accent/25 flex items-center gap-2">
            <span className="text-[9px] uppercase tracking-widest font-black font-mono theme-accent opacity-70">MATCHED</span>
            <span className="text-[10px] font-black font-mono text-theme-accent">{matchedGuide.label}</span>
            <span className="text-[9px] font-mono theme-muted opacity-50">from {matchedGuide.notebook}</span>
          </div>
        )}
        {(studentModelOptions.length > 0 || onRefreshModels) && (
          <div className="flex flex-col sm:flex-row gap-3">
            {studentModelOptions.length > 0 && (
              <div className="relative flex-1">
                <select
                  value={selectedOption}
                  onChange={(e) => {
                    if (e.target.value) onStudentChange(e.target.value);
                  }}
                  className="w-full px-4 py-3 pr-10 premium-input rounded-xl text-[11px] font-black font-mono text-white focus:outline-none shadow-inner appearance-none cursor-pointer bg-black/30"
                >
                  <option className="theme-surface theme-text" value="">
                    SELECT TRAINED OR ACCOUNT MODEL
                  </option>
                  {mergedOptions.length > 0 && (
                    <optgroup label="MERGED RUN OUTPUTS">
                      {mergedOptions.map((model) => (
                        <option className="theme-surface theme-text" key={`merged:${model.id}`} value={model.id}>
                          {model.label}: {model.id}
                        </option>
                      ))}
                    </optgroup>
                  )}
                  {hfOptions.length > 0 && (
                    <optgroup label="HUGGING FACE MODELS">
                      {hfOptions.map((model) => (
                        <option className="theme-surface theme-text" key={`hf:${model.id}`} value={model.id}>
                          {model.label}: {model.id}
                        </option>
                      ))}
                    </optgroup>
                  )}
                </select>
                <ChevronRight className="absolute right-4 top-1/2 -translate-y-1/2 w-4 h-4 rotate-90 pointer-events-none opacity-30" />
              </div>
            )}
            {latestMerged && (
              <button
                type="button"
                onClick={() => onStudentChange(latestMerged.id)}
                className="px-4 py-3 rounded-xl border border-theme-accent/30 theme-accent-soft theme-accent text-[10px] font-black font-mono uppercase tracking-widest hover:bg-theme-accent/20 transition-all premium-button whitespace-nowrap"
              >
                Use Latest Merged
              </button>
            )}
            {onRefreshModels && (
              <button
                type="button"
                onClick={onRefreshModels}
                disabled={hfLoading}
                title={hfTokenSet ? "Refresh model repositories" : "Refresh completed run models"}
                className="px-4 py-3 rounded-xl border border-white/10 theme-surface-soft theme-text hover:bg-white/5 transition-all premium-button disabled:opacity-30"
              >
                <RefreshCw className={`w-4 h-4 ${hfLoading ? "animate-spin" : ""}`} />
              </button>
            )}
          </div>
        )}
      </div>

      <div className="space-y-4">
        <div className="flex items-center gap-2 ml-1">
          <div className="w-1 h-4 theme-accent-bg rounded-full shadow-[0_0_8px_currentColor]" />
          <label className="text-[10px] uppercase tracking-[0.2em] theme-muted font-black">
            Fine-tuning Method
          </label>
        </div>
        <div className="grid grid-cols-2 xl:grid-cols-4 gap-3">
          {METHOD_OPTIONS.map(({ key, label, desc }) => {
            const active = method === key;
            return (
              <button
                key={key}
                type="button"
                onClick={() => selectMethod(key as LoraConfig["method"])}
                className={`text-left rounded-2xl border p-4 transition-all duration-300 premium-button group ${
                  active
                    ? "theme-accent-soft theme-accent border-theme-accent/40 shadow-lg shadow-theme-accent/5 scale-[1.02]"
                    : "border-white/5 bg-white/[0.01] theme-muted hover:theme-text hover:bg-white/[0.03]"
                }`}
              >
                <div className="text-[12px] uppercase tracking-widest font-black font-mono">
                  {label}
                </div>
                <p className="mt-1 text-[10px] theme-muted opacity-60 font-medium group-hover:opacity-100 transition-opacity">{desc}</p>
              </button>
            );
          })}
        </div>
        <div className="rounded-2xl border border-white/5 theme-surface-soft p-5 glass-panel relative overflow-hidden group">
           <div className="absolute top-0 left-0 w-1 h-full theme-accent-bg opacity-30 group-hover:opacity-100 transition-opacity" />
          <p className="text-[10px] uppercase tracking-widest font-black font-mono theme-accent">
            {copy.title}
          </p>
          <p className="mt-2 text-sm-fluid theme-muted leading-relaxed font-medium opacity-90">
            {copy.detail}
          </p>
        </div>

        {method === "custom" && (
          <div className="rounded-2xl border border-white/5 theme-surface-soft p-5 glass-panel space-y-5">
            <div className="space-y-2">
              <label className="text-[10px] uppercase tracking-widest theme-muted font-black ml-1">
                Method Name
              </label>
              <input
                type="text"
                value={value.customMethodName || ""}
                onChange={(e) => setCustom({ customMethodName: e.target.value })}
                placeholder="My ROCm fine-tune recipe"
                className="w-full px-4 py-3 premium-input rounded-xl text-sm-fluid font-mono text-white focus:outline-none shadow-inner"
              />
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between gap-3">
                <label className="text-[10px] uppercase tracking-widest theme-muted font-black ml-1">
                  Sequential Commands
                </label>
                <button
                  type="button"
                  onClick={addCommand}
                  className="inline-flex items-center gap-2 px-3 py-2 rounded-xl border border-white/10 theme-surface-soft theme-text text-[10px] font-black font-mono uppercase tracking-widest hover:bg-white/5 transition-colors"
                >
                  <Plus className="w-3.5 h-3.5" />
                  Command
                </button>
              </div>

              {customCommands.map((command, idx) => (
                <div key={idx} className="flex gap-3 items-start">
                  <div className="w-8 h-8 rounded-lg border border-white/10 bg-black/30 theme-muted font-mono text-[10px] font-black flex items-center justify-center shrink-0 mt-1">
                    {idx + 1}
                  </div>
                  <textarea
                    value={command}
                    onChange={(e) => setCommand(idx, e.target.value)}
                    rows={2}
                    spellCheck={false}
                    placeholder="pip install ... && llamafactory-cli train $TRAIN_YAML"
                    className="flex-1 min-h-[76px] px-4 py-3 premium-input rounded-xl text-[12px] font-mono text-white focus:outline-none shadow-inner leading-relaxed resize-y"
                  />
                  <button
                    type="button"
                    onClick={() => removeCommand(idx)}
                    className="w-9 h-9 rounded-xl border border-red-500/20 text-red-400 bg-red-500/5 hover:bg-red-500 hover:text-white transition-colors flex items-center justify-center shrink-0 mt-1"
                    title="Remove command"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              ))}

              <p className="text-[10px] theme-faint font-mono uppercase tracking-tight opacity-70 leading-relaxed">
                Available variables: $RUN_DIR, $TRAIN_YAML, $DATA_DIR, $OUTPUT_DIR, $STUDENT_MODEL, $BASE_MODEL, $HF_TOKEN, $FT_LEARNING_RATE, $FT_EPOCHS, $FT_BATCH_SIZE.
              </p>
            </div>
          </div>
        )}

        {(method === "zrald" || method === "zrald_offline") && (
          <div className="rounded-2xl border border-white/5 theme-surface-soft p-5 glass-panel space-y-5">
            <div className="rounded-xl border border-theme-accent/20 bg-theme-accent/10 p-4">
              <p className="text-[10px] uppercase tracking-widest font-black font-mono theme-accent">
                Teacher-generated reward loop
              </p>
              <p className="mt-2 text-[11px] theme-muted font-mono leading-relaxed">
                The GPU teacher generates the prompt/answer pool first, the current student is benchmarked on the configured sample, then ZRALD trains with reward-teacher scoring.
              </p>
            </div>
            <div className="space-y-2">
              <label className="text-[10px] uppercase tracking-widest theme-muted font-black ml-1">
                Dataset Source
              </label>
              <div className="grid grid-cols-2 gap-3">
                {([
                  ["generate", "Generate with teacher (Qdrant)"],
                  ["huggingface", "Use existing HF dataset"],
                ] as const).map(([key, label]) => {
                  const active = (value.zraldDatasetSource || "generate") === key;
                  return (
                    <button
                      type="button"
                      key={key}
                      onClick={() => set("zraldDatasetSource", key)}
                      className={`text-left rounded-xl border p-4 transition-all premium-button ${active ? "theme-accent-soft theme-accent border-theme-accent/40" : "border-white/5 bg-white/[0.02] theme-muted hover:theme-text hover:bg-white/[0.04]"}`}
                    >
                      <span className="text-[11px] font-mono font-black uppercase tracking-widest">{label}</span>
                    </button>
                  );
                })}
              </div>
              <p className="text-[10px] theme-muted font-mono italic opacity-70 ml-1 leading-relaxed">
                {(value.zraldDatasetSource || "generate") === "huggingface"
                  ? "ZRALD loads the question pool from the configured Hugging Face dataset repo(s); teacher generation is skipped. The student still only sees the question."
                  : "The teacher reads Qdrant/RAG to write the Q + reference answer, then ZRALD trains. The student is shown the question only — never the RAG context or the reference answer."}
              </p>
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-[10px] uppercase tracking-widest theme-muted font-black ml-1">
                  Reward Teacher Endpoint
                </label>
                <input
                  type="text"
                  value={value.zraldRewardEndpoint || ""}
                  onChange={(e) => set("zraldRewardEndpoint", e.target.value)}
                  placeholder={directTraining ? "Blank = auto-detect deployed teacher endpoint" : "Blank = use detected/deployed GPU teacher"}
                  className="w-full px-4 py-3 premium-input rounded-xl text-sm-fluid font-mono text-white focus:outline-none shadow-inner"
                />
              </div>
              <div className="space-y-2">
                <label className="text-[10px] uppercase tracking-widest theme-muted font-black ml-1">
                  Reward Teacher Model
                </label>
                <input
                  type="text"
                  value={value.zraldRewardModel || ""}
                  onChange={(e) => set("zraldRewardModel", e.target.value)}
                  placeholder="Blank = auto-detect served teacher model"
                  className="w-full px-4 py-3 premium-input rounded-xl text-sm-fluid font-mono text-white focus:outline-none shadow-inner"
                />
              </div>
            </div>
            <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
              <NumField k="zraldTrainQuestions" label="Train Questions" />
              <NumField k="zraldBenchmarkQuestions" label="Benchmark Questions" />
              <NumField k="zraldNumGenerations" label="Answers / Prompt" />
              <NumField k="zraldMaxCompletionTokens" label="Answer Tokens" />
              <NumField k="zraldRewardTemperature" label="Judge Temp" step={0.1} />
            </div>
          </div>
        )}
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
        {usesAdapterFields && (
          <>
            <NumField k="r" label={copy.rankLabel} />
            <NumField k="alpha" label={copy.alphaLabel} />
            <NumField k="dropout" label={copy.dropoutLabel} step={0.01} />
          </>
        )}
        <NumField k="learningRate" label={copy.learningLabel} step={0.00001} />
        <NumField k="epochs" label="Epochs" step={0.1} />
        <NumField k="batchSize" label={copy.batchLabel} />
        <NumField k="gradientAccumulation" label={copy.accumLabel} />
        <NumField k="cutoffLen" label={copy.cutoffLabel} />
        <NumField k="saveSteps" label={copy.saveLabel} />
      </div>
      <div className="p-4 rounded-xl border border-white/5 bg-white/[0.01] flex items-start gap-3">
        <div className="w-1.5 h-1.5 rounded-full theme-accent-bg mt-1.5 shrink-0" />
        <p className="text-[10px] theme-faint font-mono uppercase tracking-tight italic opacity-70">
          {copy.note}
        </p>
      </div>
    </div>
  );
}
