// Module-level store for per-run live data (logs, progress, train metrics).
//
// Why this exists: RunDashboard used to keep `logs`/`progress`/`metrics` in
// component state, which meant switching to the Pipeline/Terminal tab unmounted
// the component and discarded everything streamed so far. Lifting the buffer
// out of the React tree lets the live log keep accumulating in the background
// and survive tab navigation. App.tsx wires the global event subscription once
// at startup so events flow into this store even when no one is watching.

import { api, events } from "./tauri";
import type {
  PipelineLogEvent,
  PipelineMetricEvent,
  PipelineProgressEvent,
  TrainPoint,
} from "../types";

export interface RunStream {
  logs: string;
  progress: { scanned: number; kept: number; rejected: number } | null;
  metrics: TrainPoint[];
}

type Listener = (s: RunStream) => void;

const MAX_LOG_BYTES = 256 * 1024;

const streams = new Map<string, RunStream>();
const listeners = new Map<string, Set<Listener>>();
const hydrated = new Set<string>();
let started = false;

function emptyStream(): RunStream {
  return { logs: "", progress: null, metrics: [] };
}

function getOrCreate(runId: string): RunStream {
  let s = streams.get(runId);
  if (!s) {
    s = emptyStream();
    streams.set(runId, s);
  }
  return s;
}

function notify(runId: string) {
  const s = streams.get(runId);
  if (!s) return;
  const subs = listeners.get(runId);
  if (!subs) return;
  subs.forEach((fn) => fn(s));
}

export function getStream(runId: string): RunStream {
  return streams.get(runId) ?? emptyStream();
}

export function subscribe(runId: string, fn: Listener): () => void {
  let set = listeners.get(runId);
  if (!set) {
    set = new Set();
    listeners.set(runId, set);
  }
  set.add(fn);
  return () => {
    set!.delete(fn);
  };
}

/** Replace the log buffer for a run (used by Reload + initial hydrate). */
export function setLogs(runId: string, text: string) {
  const s = getOrCreate(runId);
  s.logs = text.length > MAX_LOG_BYTES ? text.slice(-MAX_LOG_BYTES) : text;
  notify(runId);
}

/** Fetch the persisted log tail from disk and seed the buffer. Safe to call
 *  multiple times — the first call wins unless `force` is set (Reload). */
export async function hydrateFromDisk(runId: string, force = false) {
  if (force) hydrated.delete(runId);
  if (!force && hydrated.has(runId)) return;
  hydrated.add(runId);
  try {
    const text = await api.readRunLog(runId, MAX_LOG_BYTES);
    if (text) setLogs(runId, text);
  } catch (e) {
    // A brand-new run may not have a log file yet — that's fine.
    console.warn("readRunLog failed for", runId, e);
  }
}

/** Initialize the global event subscription. Idempotent. Returns a teardown
 *  that should only be used in tests; in the app this lives for the whole
 *  session. */
export async function startGlobalSubscription(): Promise<() => void> {
  if (started) return () => {};
  started = true;
  const unsubs: Array<() => void> = [];

  unsubs.push(
    await events.onPipelineLog((e: PipelineLogEvent) => {
      const s = getOrCreate(e.runId);
      s.logs = (s.logs + e.line).slice(-MAX_LOG_BYTES);
      notify(e.runId);
    }),
  );
  unsubs.push(
    await events.onPipelineProgress((e: PipelineProgressEvent) => {
      const s = getOrCreate(e.runId);
      s.progress = { scanned: e.scanned, kept: e.kept, rejected: e.rejected };
      notify(e.runId);
    }),
  );
  unsubs.push(
    await events.onPipelineMetric((e: PipelineMetricEvent) => {
      const s = getOrCreate(e.runId);
      s.metrics = [...s.metrics, { step: e.step, loss: e.loss, epoch: e.epoch }];
      notify(e.runId);
    }),
  );

  return () => {
    unsubs.forEach((u) => u());
    started = false;
  };
}
