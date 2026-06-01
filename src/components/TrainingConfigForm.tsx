import React, { useEffect, useState } from "react";
import { ChevronRight, Plus, RefreshCw, Trash2 } from "lucide-react";
import type { LoraConfig, MatchedGuideInfo } from "../types";
import { api } from "../lib/tauri";

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
}: Props) {
  const set = <K extends keyof LoraConfig>(k: K, v: LoraConfig[K]) =>
    onChange({ ...value, [k]: v });
  const method = value.method || "lora";
  const methodCopy: Record<"lora" | "qlora" | "unsloth" | "full" | "freeze" | "dora" | "loraplus" | "pissa" | "galore" | "badam" | "grpo" | "zrald" | "custom", {
    title: string;
    detail: string;
    rankLabel: string;
    alphaLabel: string;
    dropoutLabel: string;
    learningLabel: string;
    batchLabel: string;
    accumLabel: string;
    cutoffLabel: string;
    saveLabel: string;
    note: string;
  }> = {
    lora: {
      title: "LoRA adapter training",
      detail: "Trains standard low-rank adapters with the base model loaded normally.",
      rankLabel: "LoRA Rank (r)",
      alphaLabel: "LoRA Alpha",
      dropoutLabel: "LoRA Dropout",
      learningLabel: "LoRA Learning Rate",
      batchLabel: "Batch Size",
      accumLabel: "Grad Accum Steps",
      cutoffLabel: "Context Cutoff Length",
      saveLabel: "Save Steps (ckpt + hub push)",
      note: "Standard LoRA keeps VRAM use moderate and works broadly across supported LLaMA-Factory models.",
    },
    qlora: {
      title: "QLoRA 4-bit adapter training",
      detail: "Loads the base model in 4-bit with bitsandbytes (multi-backend wheel), then trains LoRA adapters.",
      rankLabel: "QLoRA Rank (r)",
      alphaLabel: "QLoRA Alpha",
      dropoutLabel: "QLoRA Dropout",
      learningLabel: "QLoRA Learning Rate",
      batchLabel: "4-bit Batch Size",
      accumLabel: "4-bit Grad Accum",
      cutoffLabel: "4-bit Context Cutoff",
      saveLabel: "Checkpoint Push Steps",
      note: "Installs bitsandbytes>=0.45 multi-backend on ROCm. gfx942 (MI300X) is officially supported; gfx1100 is experimental.",
    },
    unsloth: {
      title: "Unsloth accelerated LoRA",
      detail: "Adds use_unsloth: true to LLaMA-Factory. ROCm-only install path: unsloth --no-deps + pinned peft/trl/accelerate.",
      rankLabel: "Unsloth LoRA Rank (r)",
      alphaLabel: "Unsloth LoRA Alpha",
      dropoutLabel: "Unsloth Dropout",
      learningLabel: "Unsloth Learning Rate",
      batchLabel: "Unsloth Batch Size",
      accumLabel: "Unsloth Grad Accum",
      cutoffLabel: "Unsloth Max Seq Length",
      saveLabel: "Unsloth Save Steps",
      note: "Unsloth on ROCm is community-supported. MI300X (gfx942) is best-tested; gfx1100 may hit Triton-kernel issues.",
    },
    full: {
      title: "Full parameter fine-tuning",
      detail: "Trains all model weights with finetuning_type: full.",
      rankLabel: "Unused Rank",
      alphaLabel: "Unused Alpha",
      dropoutLabel: "Unused Dropout",
      learningLabel: "Full Fine-tune LR",
      batchLabel: "Full Batch Size",
      accumLabel: "Full Grad Accum",
      cutoffLabel: "Full Cutoff Length",
      saveLabel: "Full Checkpoint Steps",
      note: "Full fine-tuning needs much more VRAM and storage because all model weights are trainable.",
    },
    freeze: {
      title: "Freeze tuning",
      detail: "Trains selected upper layers with finetuning_type: freeze.",
      rankLabel: "Unused Rank",
      alphaLabel: "Unused Alpha",
      dropoutLabel: "Unused Dropout",
      learningLabel: "Freeze Learning Rate",
      batchLabel: "Freeze Batch Size",
      accumLabel: "Freeze Grad Accum",
      cutoffLabel: "Freeze Cutoff Length",
      saveLabel: "Freeze Save Steps",
      note: "Freeze tuning trains part of the base model and sits between LoRA and full fine-tuning.",
    },
    dora: {
      title: "DoRA adapter training",
      detail: "Uses LoRA with use_dora enabled for magnitude-aware adapter updates.",
      rankLabel: "DoRA Rank (r)",
      alphaLabel: "DoRA Alpha",
      dropoutLabel: "DoRA Dropout",
      learningLabel: "DoRA Learning Rate",
      batchLabel: "DoRA Batch Size",
      accumLabel: "DoRA Grad Accum",
      cutoffLabel: "DoRA Cutoff Length",
      saveLabel: "DoRA Save Steps",
      note: "DoRA is a LoRA variant, so adapter merge/upload works like regular LoRA.",
    },
    loraplus: {
      title: "LoRA+ adapter training",
      detail: "Uses LoRA with LoRA+ learning-rate ratio enabled.",
      rankLabel: "LoRA+ Rank (r)",
      alphaLabel: "LoRA+ Alpha",
      dropoutLabel: "LoRA+ Dropout",
      learningLabel: "LoRA+ Base LR",
      batchLabel: "LoRA+ Batch Size",
      accumLabel: "LoRA+ Grad Accum",
      cutoffLabel: "LoRA+ Cutoff Length",
      saveLabel: "LoRA+ Save Steps",
      note: "LoRA+ uses separate adapter learning-rate scaling while keeping normal LoRA outputs.",
    },
    pissa: {
      title: "PiSSA initialized LoRA",
      detail: "Uses LoRA with PiSSA SVD-based initialization; folds the residual into the adapter at save time.",
      rankLabel: "PiSSA Rank (r)",
      alphaLabel: "PiSSA Alpha",
      dropoutLabel: "PiSSA Dropout",
      learningLabel: "PiSSA Learning Rate",
      batchLabel: "PiSSA Batch Size",
      accumLabel: "PiSSA Grad Accum",
      cutoffLabel: "PiSSA Cutoff Length",
      saveLabel: "PiSSA Save Steps",
      note: "Emits pissa_init + pissa_iter=16 + pissa_convert=true. Saved adapter format matches plain LoRA.",
    },
    galore: {
      title: "GaLore full tuning",
      detail: "Full fine-tune with GaLore low-rank gradient projection. Memory close to LoRA, ~30% slower per step.",
      rankLabel: "Unused Rank",
      alphaLabel: "Unused Alpha",
      dropoutLabel: "Unused Dropout",
      learningLabel: "GaLore Learning Rate",
      batchLabel: "GaLore Batch Size",
      accumLabel: "GaLore Grad Accum",
      cutoffLabel: "GaLore Cutoff Length",
      saveLabel: "GaLore Save Steps",
      note: "Layerwise mode + pure_bf16; works on ROCm (pure PyTorch). Use Grad Accum = 1 with layerwise GaLore.",
    },
    badam: {
      title: "BAdam full tuning",
      detail: "Full fine-tune with block-wise Adam updating one layer block at a time. Lower peak VRAM than full Adam.",
      rankLabel: "Unused Rank",
      alphaLabel: "Unused Alpha",
      dropoutLabel: "Unused Dropout",
      learningLabel: "BAdam Learning Rate",
      batchLabel: "BAdam Batch Size",
      accumLabel: "BAdam Grad Accum",
      cutoffLabel: "BAdam Cutoff Length",
      saveLabel: "BAdam Save Steps",
      note: "Installs `badam` from PyPI. Layer mode + ascending switch + interval 50. ROCm-compatible (pure PyTorch).",
    },
    grpo: {
      title: "GRPO reinforcement learning",
      detail: "Group Relative Policy Optimization via unsloth's GRPOTrainer. Trains LoRA adapters with reward feedback.",
      rankLabel: "GRPO Rank (r)",
      alphaLabel: "GRPO Alpha",
      dropoutLabel: "GRPO Dropout",
      learningLabel: "GRPO Learning Rate",
      batchLabel: "GRPO Batch Size",
      accumLabel: "GRPO Grad Accum",
      cutoffLabel: "GRPO Max Seq Length",
      saveLabel: "GRPO Save Steps",
      note: "GRPO via unsloth's GRPOTrainer. Uses reward functions for code/game-strategy tasks. ROCm-supported on MI300X.",
    },
    zrald: {
      title: "ZRALD RAG reward learning",
      detail: "Zero-shot Retrieval-Augmented Learning with Dynamic rewards. The RAG teacher builds questions, the student samples four answers, and a reward teacher scores them for GRPO.",
      rankLabel: "ZRALD LoRA Rank (r)",
      alphaLabel: "ZRALD LoRA Alpha",
      dropoutLabel: "ZRALD Dropout",
      learningLabel: "ZRALD Learning Rate",
      batchLabel: "ZRALD Prompt Batch",
      accumLabel: "ZRALD Grad Accum",
      cutoffLabel: "ZRALD Max Seq Length",
      saveLabel: "ZRALD Save Steps",
      note: "Uses the generated RAG question pool, fixed before/after benchmarks, four sampled completions per prompt, and reward-teacher scores clamped from -1 to 1.",
    },
    custom: {
      title: value.customMethodName?.trim() || "Custom command method",
      detail: "Runs your saved shell commands in sequence on the remote training node.",
      rankLabel: "Custom Rank",
      alphaLabel: "Custom Alpha",
      dropoutLabel: "Custom Dropout",
      learningLabel: "Custom Learning Rate",
      batchLabel: "Custom Batch Size",
      accumLabel: "Custom Grad Accum",
      cutoffLabel: "Custom Cutoff Length",
      saveLabel: "Custom Save Steps",
      note: "Commands run from the run directory. Write adapter outputs to $OUTPUT_DIR for built-in upload and merge support.",
    },
  };
  const copy = methodCopy[method];
  const usesAdapterFields = !["full", "freeze", "galore", "badam", "custom"].includes(method);
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
          {[
            ["lora", "LoRA", "Adapter weights"],
            ["qlora", "QLoRA", "4-bit quant"],
            ["unsloth", "Unsloth", "Fast inference"],
            ["full", "Full", "All weights"],
            ["freeze", "Freeze", "Selective layers"],
            ["dora", "DoRA", "Magnitude LoRA"],
            ["loraplus", "LoRA+", "LR multiplier"],
            ["pissa", "PiSSA", "SVD init"],
            ["galore", "GaLore", "Memory efficient"],
            ["badam", "BAdam", "Block-wise Adam"],
            ["grpo", "GRPO", "Reinforcement RL"],
            ["zrald", "ZRALD", "RAG reward RL"],
            ["custom", "Add +", "Command chain"],
          ].map(([key, label, desc]) => {
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

        {method === "zrald" && (
          <div className="rounded-2xl border border-white/5 theme-surface-soft p-5 glass-panel space-y-5">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-[10px] uppercase tracking-widest theme-muted font-black ml-1">
                  Reward Teacher Endpoint
                </label>
                <input
                  type="text"
                  value={value.zraldRewardEndpoint || ""}
                  onChange={(e) => set("zraldRewardEndpoint", e.target.value)}
                  placeholder="http://127.0.0.1:8000 or external OpenAI-compatible URL"
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
                  placeholder="Hugging Face model id or served model name"
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
