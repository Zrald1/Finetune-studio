import React from "react";
import { FileText, ChevronRight, Folder } from "lucide-react";


export interface PreviewPair {
  question: string;
  think?: string;
  answer: string;
  source_file?: string;
}

export default function DatasetPreview({ pairs }: { pairs: PreviewPair[] }) {
  if (pairs.length === 0) {
    return (
      <div className="h-48 flex flex-col items-center justify-center space-y-3 opacity-40 select-none glass-panel rounded-2xl border border-dashed border-white/10">
        <FileText className="w-8 h-8 theme-faint" />
        <p className="text-sm-fluid theme-faint italic font-serif">
          No telemetry samples captured yet.
        </p>
      </div>
    );
  }
  return (
    <div className="space-y-4 max-h-[500px] overflow-y-auto pr-2 scrollbar-thin animate-premium">
      {pairs.map((p, i) => (
        <details
          key={`${p.source_file || "sample"}-${i}-${p.question.slice(0, 32)}`}
          className="premium-card rounded-2xl p-4 border border-white/5 transition-all duration-300 group overflow-hidden"
        >
          <summary className="cursor-pointer theme-text leading-relaxed flex items-start gap-4 list-none outline-none">
            <div className="w-8 h-8 rounded-lg bg-white/5 border border-white/5 flex items-center justify-center shrink-0 group-hover:theme-accent-bg group-hover:text-black transition-all duration-300 shadow-inner font-mono text-[10px] font-black">
              {i + 1}
            </div>
            <div className="flex-1 min-w-0 pt-1">
              <span className="text-sm-fluid font-black tracking-tight line-clamp-2 group-hover:theme-accent transition-colors">{p.question}</span>
              <div className="flex items-center gap-2 mt-1 opacity-40">
                <span className="text-[8px] uppercase tracking-widest font-black font-mono">Sample {i + 1}</span>
              </div>
            </div>
            <ChevronRight className="w-4 h-4 theme-faint mt-2 group-open:rotate-90 transition-transform" />
          </summary>
          <div className="mt-5 space-y-4 pl-12 animate-premium relative">
            <div className="absolute left-4 top-0 w-0.5 h-full bg-white/5 rounded-full" />
            <div className="bg-emerald-500/5 border border-emerald-500/20 rounded-xl p-4 text-sm-fluid text-emerald-200/90 whitespace-pre-wrap shadow-lg relative group/answer">
              <div className="absolute -left-8 top-1/2 -translate-y-1/2 w-6 h-0.5 bg-emerald-500/20" />
              <span className="text-[9px] uppercase tracking-[0.2em] text-emerald-400 font-black mb-2 block opacity-60 group-hover/answer:opacity-100 transition-opacity">Ground Truth Answer</span>
              {p.answer}
            </div>

            {p.source_file && (
              <div className="flex items-center gap-2 text-[9px] font-mono theme-faint uppercase tracking-widest bg-white/5 w-fit px-3 py-1 rounded-full border border-white/5">
                <Folder className="w-3 h-3" />
                SRC: {p.source_file}
              </div>
            )}
          </div>
        </details>
      ))}
    </div>
  );
}
