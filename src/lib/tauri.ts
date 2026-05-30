// Typed wrappers around Tauri's invoke/listen so the rest of the app
// doesn't have to repeat command names.

import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type {
  AppConfig,
  Chunk,
  GPUState,
  HfDatasetRepo,
  HfModelRepo,
  HfWhoami,
  IngestDoneEvent,
  IngestProgressEvent,
  PipelineLogEvent,
  PipelineMetricEvent,
  PipelineProgressEvent,
  PaddleOcrConfig,
  QdrantConfig,
  Run,
  RunConfig,
  SSHConfig,
  ShellDoneEvent,
  ShellLogEvent,
  LoraConfig,
  HubConfig,
  MatchedGuideInfo,
  DigitalOceanConfig,
  DigitalOceanAccount,
  DigitalOceanDroplet,
  DigitalOceanImage,
  DigitalOceanProject,
  DigitalOceanRegion,
  DigitalOceanSize,
  DigitalOceanSshKey,
  EmbedderConfig,
} from "../types";

export const api = {
  // config
  loadConfig: () => invoke<AppConfig>("load_config"),
  saveConfig: (cfg: AppConfig) => invoke<void>("save_config", { cfg }),
  readLocalFileText: (path: string) => invoke<string>("read_local_file_text", { path }),
  saveIngestState: (stateJson: string) => invoke<void>("save_ingest_state", { stateJson }),
  loadIngestState: () => invoke<string>("load_ingest_state"),

  // DigitalOcean GPU droplets
  doListGpuSizes: (cfg: DigitalOceanConfig) =>
    invoke<DigitalOceanSize[]>("do_list_gpu_sizes", { cfg }),
  doListDroplets: (cfg: DigitalOceanConfig) =>
    invoke<DigitalOceanDroplet[]>("do_list_droplets", { cfg }),
  doListGpuDroplets: (cfg: DigitalOceanConfig) =>
    invoke<DigitalOceanDroplet[]>("do_list_gpu_droplets", { cfg }),
  doListRegions: (cfg: DigitalOceanConfig) =>
    invoke<DigitalOceanRegion[]>("do_list_regions", { cfg }),
  doListImages: (cfg: DigitalOceanConfig) =>
    invoke<DigitalOceanImage[]>("do_list_images", { cfg }),
  doListSshKeys: (cfg: DigitalOceanConfig) =>
    invoke<DigitalOceanSshKey[]>("do_list_ssh_keys", { cfg }),
  doListProjects: (cfg: DigitalOceanConfig) =>
    invoke<DigitalOceanProject[]>("do_list_projects", { cfg }),
  doGetAccount: (cfg: DigitalOceanConfig) =>
    invoke<DigitalOceanAccount>("do_get_account", { cfg }),
  doCreateGpuDroplet: (cfg: DigitalOceanConfig) =>
    invoke<DigitalOceanDroplet>("do_create_gpu_droplet", { cfg }),
  doDestroyDroplet: (cfg: DigitalOceanConfig, dropletId: number) =>
    invoke<void>("do_destroy_droplet", { cfg, dropletId }),

  // ssh
  testSsh: (cfg: SSHConfig) => invoke<string>("test_ssh", { cfg }),
  nvidiaSmi: (cfg: SSHConfig) => invoke<GPUState>("nvidia_smi", { cfg }),
  sshStream: (cfg: SSHConfig, cmd: string, streamId?: string) =>
    invoke<string>("ssh_exec_stream", { cfg, cmd, streamId }),
  sshStopStream: (streamId: string) =>
    invoke<void>("ssh_stop_stream", { streamId }),
  writeRemoteFile: (cfg: SSHConfig, filePath: string, content: string) =>
    invoke<void>("write_remote_file", { cfg, filePath, content }),

  // qdrant
  qdrantCount: (cfg: QdrantConfig) => invoke<number>("qdrant_count", { cfg }),
  qdrantSample: (cfg: QdrantConfig, n: number) =>
    invoke<Chunk[]>("qdrant_sample", { cfg, n }),
  qdrantEnsureCollection: (cfg: QdrantConfig) =>
    invoke<void>("qdrant_ensure_collection", { cfg }),
  qdrantSampleInCollection: (cfg: QdrantConfig, collection: string, n: number) =>
    invoke<Chunk[]>("qdrant_sample_in_collection", { cfg, collection, n }),
  qdrantScrollInCollection: (cfg: QdrantConfig, collection: string, pageSize: number, offset: any) =>
    invoke<{ chunks: Chunk[]; next_offset: any }>("qdrant_scroll_in_collection", { cfg, collection, pageSize, offset }),
  qdrantScrollAll: (cfg: QdrantConfig, n: number) =>
    invoke<Chunk[]>("qdrant_scroll_all", { cfg, n }),
  qdrantScrollAllInCollection: (cfg: QdrantConfig, collection: string, n: number) =>
    invoke<Chunk[]>("qdrant_scroll_all_in_collection", { cfg, collection, n }),
  qdrantListCollections: (cfg: QdrantConfig) =>
    invoke<{ name: string; status: string; vectors_count: number }[]>("list_qdrant_collections", { cfg }),
  qdrantListSnapshots: (cfg: QdrantConfig, collection: string) =>
    invoke<{ name: string; creation_time?: string; size: number }[]>("list_qdrant_snapshots", { cfg, collection }),
  qdrantCreateSnapshot: (cfg: QdrantConfig, collection: string) =>
    invoke<{ name: string; creation_time?: string; size: number }>("create_qdrant_snapshot", { cfg, collection }),
  qdrantRestoreSnapshot: (cfg: QdrantConfig, collection: string, snapshotPath: string) =>
    invoke<void>("restore_qdrant_snapshot", { cfg, collection, snapshotPath }),
  qdrantUploadSnapshot: (cfg: QdrantConfig, collection: string, snapshotPath: string) =>
    invoke<void>("qdrant_upload_snapshot", { cfg, collection, snapshotPath }),
  qdrantDownloadSnapshot: (cfg: QdrantConfig, collection: string, snapshotName: string, localPath: string) =>
    invoke<void>("qdrant_download_snapshot", { cfg, collection, snapshotName, localPath }),
  createAllQdrantSnapshots: (cfg: QdrantConfig) =>
    invoke<{ collection: string; snapshot_name: string; size: number }[]>("create_all_qdrant_snapshots", { cfg }),
  downloadAllQdrantSnapshots: (cfg: QdrantConfig, localDir: string) =>
    invoke<string[]>("download_all_qdrant_snapshots", { cfg, localDir }),

  // serve — GPU-server managed embedder lifecycle
  serveEnsureQdrant: (ssh: SSHConfig, docker: AppConfig["docker"], qdrantPort: number, dataDir: string) =>
    invoke<void>("serve_ensure_qdrant", { ssh, docker, qdrantPort, dataDir }),
  serveBootEmbedder: (ssh: SSHConfig, docker: AppConfig["docker"], embedder: EmbedderConfig, hfToken?: string | null) =>
    invoke<string>("serve_boot_embedder", { ssh, docker, embedder, hfToken }),
  serveCheckEmbedder: (ssh: SSHConfig, docker: AppConfig["docker"], host: string, port: number) =>
    invoke<string | null>("serve_check_embedder", { ssh, docker, host, port }),
  serveSetupAllEmbedders: (ssh: SSHConfig, docker: AppConfig["docker"], embedders: EmbedderConfig[], hfToken?: string | null, paddleOcr?: PaddleOcrConfig | null) =>
    invoke<{ name: string; model_id: string; port: number; status: string }[]>("serve_setup_all_embedders", { ssh, docker, embedders, hfToken, paddleOcr }),
  serveBootPaddleocr: (ssh: SSHConfig, docker: AppConfig["docker"], paddleOcr: PaddleOcrConfig) =>
    invoke<string>("serve_boot_paddleocr", { ssh, docker, paddleOcr }),
  serveCreateCollection: (cfg: QdrantConfig, collection: string, vectorDim: number) =>
    invoke<void>("serve_create_collection", { cfg, collection, vectorDim }),

  // knowledge-base ingestion. Returns a streamId the caller correlates with
  // ingest://progress + ingest://done events. `tag` is optional — when set,
  // every point gets payload.tag = <tag> so later searches can filter to it.
  ingestDocuments: (
    files: string[],
    tag: string | null,
    vectorDim: number | null,
    qdrant: QdrantConfig,
    embeddingConfig: { provider: string; apiUrl?: string; apiKey?: string; modelId?: string; concurrency?: number },
    paddleOcr?: PaddleOcrConfig | null,
  ) =>
    invoke<string>("ingest_documents", { files, tag, vectorDim, qdrant, embeddingConfig, paddleOcr }),
  cancelIngest: (streamId: string) =>
    invoke<void>("cancel_ingest", { streamId }),

  // pipeline / runs
  startPipeline: (cfg: AppConfig, runCfg: RunConfig) =>
    invoke<string>("start_pipeline", { cfg, runCfg }),
  cancelRun: (runId: string) => invoke<void>("cancel_run", { runId }),
  resumeRun: (runId: string) => invoke<string>("resume_run", { runId }),
  listRuns: () => invoke<Run[]>("list_runs"),
  getRun: (runId: string) => invoke<Run>("get_run", { runId }),
  listLocalDataset: (runId: string, limit: number) =>
    invoke<unknown[]>("list_local_dataset", { runId, limit }),
  openRunsFolder: () => invoke<string>("open_runs_folder"),
  readRunLog: (runId: string, maxBytes?: number) =>
    invoke<string>("read_run_log", { runId, maxBytes }),
  pingTeacher: (endpoint: string) =>
    invoke<boolean>("ping_teacher", { endpoint }),
  teacherChat: (endpoint: string, model: string, messages: unknown[]) =>
    invoke<string>("teacher_chat", { endpoint, model, messages }),
  testTrainedModel: (runId: string, prompt: string) =>
    invoke<string>("test_trained_model", { runId, prompt }),
  runInferenceBenchmark: (runId: string, sampleSize?: number) =>
    invoke<string>("run_inference_benchmark", { runId, sampleSize }),
  mergeAndUploadModel: (runId: string, targetRepo?: string) =>
    invoke<string>("merge_and_upload_model", { runId, targetRepo }),
  mergeConvertUploadModel: (runId: string, targetMergedRepo?: string, targetGgufRepo?: string, ggufQuantization?: string) =>
    invoke<{ mergedUrl: string; ggufUrl: string }>("merge_convert_upload_model", { runId, targetMergedRepo, targetGgufRepo, ggufQuantization }),

  // hugging face — used by the wizard to auto-fill repo IDs and the
  // "Resume from" dropdown. Both read the stored hfToken on the Rust side.
  hfWhoami: () => invoke<HfWhoami>("hf_whoami"),
  hfListDatasets: () => invoke<HfDatasetRepo[]>("hf_list_datasets"),
  hfListModels: () => invoke<HfModelRepo[]>("hf_list_models"),
  hfValidateDataset: (repoId: string) =>
    invoke<{ repo_id: string; valid: boolean; sample_count?: number; format?: string; columns: string[]; error?: string }>("hf_validate_dataset", { repoId }),
  checkTeacherDeployed: (ssh: SSHConfig, docker: AppConfig["docker"], teacher: AppConfig["teacher"]) =>
    invoke<number | null>("check_teacher_deployed", { ssh, docker, teacher }),
  deployTeacher: (ssh: SSHConfig, docker: AppConfig["docker"], teacher: AppConfig["teacher"], hfToken?: string | null) =>
    invoke<string>("deploy_teacher", { ssh, docker, teacher, hfToken }),
  updateRunConfig: (runId: string, studentModel: string, lora: LoraConfig, hub: HubConfig) =>
    invoke<void>("update_run_config", { runId, studentModel, lora, hub }),
  cleanupVram: (cfg: SSHConfig, docker: AppConfig["docker"]) =>
    invoke<string>("cleanup_vram", { cfg, docker }),
  matchModelGuide: (studentModel: string) =>
    invoke<MatchedGuideInfo | null>("match_model_guide", { studentModel }),

  // AI Agent full app control
  aiGetAppState: () => invoke<any>("ai_get_app_state"),
  aiGetRunsSummary: () => invoke<any[]>("ai_get_runs_summary"),
  aiGetRunDetails: (runId: string) => invoke<any>("ai_get_run_details", { runId }),
  aiCancelRun: (runId: string) => invoke<void>("ai_cancel_run", { runId }),
  aiGetGpuStatus: (sshCfg: SSHConfig, dockerCfg: AppConfig["docker"]) =>
    invoke<string>("ai_get_gpu_status", { sshCfg, dockerCfg }),
  aiTriggerPipelineAction: (action: string, params?: any) =>
    invoke<string>("ai_trigger_pipeline_action", { action, params }),
  aiGetConfigSummary: () => invoke<any>("ai_get_config_summary"),
  aiProxyChat: (apiUrl: string, apiKey: string, requestBody: any, provider?: string) =>
    invoke<string>("ai_proxy_chat", { apiUrl, apiKey, requestBody: JSON.stringify(requestBody), provider }),
  // Server-side model listing (avoids browser CORS on provider APIs).
  aiListModels: (provider: string, apiUrl: string, apiKey: string) =>
    invoke<string[]>("ai_list_models", { provider, apiUrl, apiKey }),
};

export const events = {
  onShellLog: (handler: (e: ShellLogEvent) => void): Promise<UnlistenFn> =>
    listen<ShellLogEvent>("shell://log", (ev) => handler(ev.payload)),
  onShellDone: (handler: (e: ShellDoneEvent) => void): Promise<UnlistenFn> =>
    listen<ShellDoneEvent>("shell://done", (ev) => handler(ev.payload)),
  onDeployLog: (handler: (e: ShellLogEvent) => void): Promise<UnlistenFn> =>
    listen<ShellLogEvent>("deploy://log", (ev) => handler(ev.payload)),
  onDeployDone: (handler: (e: { streamId: string; success: boolean; message: string; port?: number }) => void): Promise<UnlistenFn> =>
    listen<{ streamId: string; success: boolean; message: string; port?: number }>("deploy://done", (ev) => handler(ev.payload)),
  onPipelineLog: (
    handler: (e: PipelineLogEvent) => void,
  ): Promise<UnlistenFn> =>
    listen<PipelineLogEvent>("pipeline://log", (ev) => handler(ev.payload)),
  onPipelineProgress: (
    handler: (e: PipelineProgressEvent) => void,
  ): Promise<UnlistenFn> =>
    listen<PipelineProgressEvent>("pipeline://progress", (ev) =>
      handler(ev.payload),
    ),
  onPipelineMetric: (
    handler: (e: PipelineMetricEvent) => void,
  ): Promise<UnlistenFn> =>
    listen<PipelineMetricEvent>("pipeline://metric", (ev) =>
      handler(ev.payload),
    ),
  onIngestProgress: (
    handler: (e: IngestProgressEvent) => void,
  ): Promise<UnlistenFn> =>
    listen<IngestProgressEvent>("ingest://progress", (ev) =>
      handler(ev.payload),
    ),
  onIngestDone: (
    handler: (e: IngestDoneEvent) => void,
  ): Promise<UnlistenFn> =>
    listen<IngestDoneEvent>("ingest://done", (ev) => handler(ev.payload)),
  onSetupLog: (handler: (e: { line: string }) => void): Promise<UnlistenFn> =>
    listen<{ line: string }>("setup://log", (ev) => handler(ev.payload)),
};
