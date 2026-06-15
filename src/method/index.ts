import { badamMethod } from "./badam";
import { customMethod } from "./custom";
import { doraMethod } from "./dora";
import { freezeMethod } from "./freeze";
import { fullMethod } from "./full";
import { galoreMethod } from "./galore";
import { grpoMethod } from "./grpo";
import { loraMethod } from "./lora";
import { loraplusMethod } from "./loraplus";
import { pissaMethod } from "./pissa";
import { qloraMethod } from "./qlora";
import type { FineTuneMethod, MethodInfo } from "./types";
import { unslothMethod } from "./unsloth";
import { zraldMethod } from "./zrald";
import { zraldOfflineMethod } from "./zraldOffline";

export type { FineTuneMethod, MethodCopy, MethodInfo } from "./types";

export const METHOD_OPTIONS: MethodInfo[] = [
  loraMethod,
  qloraMethod,
  unslothMethod,
  fullMethod,
  freezeMethod,
  doraMethod,
  loraplusMethod,
  pissaMethod,
  galoreMethod,
  badamMethod,
  grpoMethod,
  zraldMethod,
  zraldOfflineMethod,
  customMethod,
];

const METHOD_BY_KEY = METHOD_OPTIONS.reduce<Record<FineTuneMethod, MethodInfo>>((acc, method) => {
  acc[method.key] = method;
  return acc;
}, {} as Record<FineTuneMethod, MethodInfo>);

export function methodInfo(method: FineTuneMethod, customMethodName?: string): MethodInfo {
  const info = METHOD_BY_KEY[method] || loraMethod;
  if (method !== "custom") return info;

  const title = customMethodName?.trim() || customMethod.copy.title;
  return {
    ...info,
    copy: {
      ...info.copy,
      title,
    },
  };
}
