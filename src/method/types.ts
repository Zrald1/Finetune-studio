import type { LoraConfig } from "../types";

export type FineTuneMethod = NonNullable<LoraConfig["method"]>;

export interface MethodCopy {
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
}

export interface MethodInfo {
  key: FineTuneMethod;
  label: string;
  desc: string;
  usesAdapterFields: boolean;
  copy: MethodCopy;
}
