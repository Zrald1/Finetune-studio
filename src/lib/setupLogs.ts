// Global capture of `setup://log` events (teacher serving, embedder boot,
// PaddleOCR boot, Qdrant install, etc.).
//
// Why this exists: those logs are emitted as ephemeral Tauri events and were
// only ever appended to local component state in CredentialsPanel/PipelineWizard.
// That meant the moment the user navigated away from the page, the logs were
// gone — and the AI agent (which lives in the persistent sidebar) could never
// see teacher/embedder boot output. Lifting the buffer out of the React tree
// lets it survive tab navigation and become readable by the agent on demand.
//
// The inline UI in CredentialsPanel/PipelineWizard keeps its own local buffers
// for display; this store is an ADDITIONAL, app-wide mirror.

import { events } from "./tauri";

const MAX_LOG_BYTES = 128 * 1024;

let buffer = "";
let started = false;

/** Append a line to the rolling buffer (kept bounded to MAX_LOG_BYTES). */
function append(line: string) {
  buffer = (buffer + line).slice(-MAX_LOG_BYTES);
}

/**
 * Subscribe once to `setup://log` and mirror every line into the global
 * buffer. Idempotent — repeat calls are no-ops. Returns a teardown that is
 * only meaningful in tests; in the app this lives for the whole session.
 */
export async function startSetupLogSubscription(): Promise<() => void> {
  if (started) return () => {};
  started = true;
  const unsub = await events.onSetupLog(({ line }) => append(line));
  return () => {
    unsub();
    started = false;
  };
}

/** Return the tail of the captured setup logs (last `maxLines` lines). */
export function getSetupLogTail(maxLines = 300): string {
  if (!buffer) return "";
  const lines = buffer.split(/\r?\n/);
  return lines.slice(-maxLines).join("\n");
}

/** True when any setup log has been captured this session. */
export function hasSetupLogs(): boolean {
  return buffer.trim().length > 0;
}
