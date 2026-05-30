// =====================================================================
// Shared types — mirror src-tauri/src/* serde structures (camelCase).
// =====================================================================

export interface SSHConfig {
  host: string;
  port?: number;
  username: string;
  privateKey?: string;
  privateKeyPath?: string;
  password?: string;
}

export interface QdrantConfig {
  endpoint: string;
  apiKey: string;
  collection: string;
}

export type ServingEngine = "vllm" | "sglang";

export interface TeacherConfig {
  repoId: string;
  vllmPort: number;
  maxModelLen: number;
  dtype: string;
  tensorParallel: number;
  gpuMemoryUtilization?: number;
  autoTune?: boolean;
  enableChunkedPrefill?: boolean;
  maxNumBatchedTokens?: number | null;
  maxNumSeqs?: number | null;
  enableAutoToolChoice?: boolean;
  toolCallParser?: string | null;
  customServeCmd?: string;
  servingEngine?: ServingEngine;
}

export interface StudentConfig {
  repoId: string;
  outputDir: string;
}

export interface DockerConfig {
  enabled: boolean;
  containerName: string;
  imageName: string;
  startArgs: string;
  bypassTerminal: boolean;
}



export interface DigitalOceanConfig {
  apiKey: string;
  dropletName: string;
  region: string;
  size: string;
  image: string;
  sshKeys: string;
  projectId: string;
  tags: string;
  backups: boolean;
  ipv6: boolean;
  monitoring: boolean;
  userData: string;
}

export interface DigitalOceanRegion {
  slug: string;
  name: string;
  available: boolean;
  sizes: string[];
  features: string[];
}

export interface DigitalOceanImage {
  id: number;
  name: string;
  distribution?: string;
  slug?: string | null;
  type?: string;
  public?: boolean;
  regions?: string[];
  minDiskSize?: number | null;
  description?: string | null;
}

export interface DigitalOceanSshKey {
  id: number;
  name: string;
  fingerprint: string;
  publicKey?: string;
}

export interface DigitalOceanProject {
  id: string;
  name: string;
  description?: string;
  purpose?: string;
  environment?: string;
  isDefault?: boolean;
}

export interface DigitalOceanAccount {
  name?: string | null;
  email?: string | null;
  uuid: string;
  status: string;
  team?: {
    name?: string | null;
    uuid?: string | null;
  } | null;
}

export interface DigitalOceanSize {
  slug: string;
  memory: number;
  vcpus: number;
  disk: number;
  transfer: number;
  priceMonthly?: number | null;
  priceHourly?: number | null;
  regions: string[];
  available: boolean;
  description: string;
  gpuInfo?: {
    count?: number;
    model?: string;
    vram?: {
      amount?: number;
      unit?: string;
    };
  } | null;
}

export interface DigitalOceanDroplet {
  id: number;
  name: string;
  status: string;
  urn?: string | null;
  region?: unknown;
  sizeSlug?: string | null;
  image?: unknown;
  networks?: {
    v4?: Array<{ ipAddress: string; type: string }>;
  };
  tags?: string[];
}

export interface EmbedderConfig {
  name: string;
  modelId: string;
  port: number;
  collection: string;
  concurrency: number;
  vectorDim?: number;
  enabled: boolean;
}

export interface PaddleOcrConfig {
  enabled: boolean;
  port: number;
  modelName: string;
  dockerImage: string;
}

export const DEFAULT_EMBEDDER: EmbedderConfig = {
  name: "",
  modelId: "Qwen/Qwen3-Embedding-8B",
  port: 8101,
  collection: "",
  concurrency: 2,
  vectorDim: undefined,
  enabled: true,
};

export interface EmbeddingConfig {
  provider: "vllm" | "ollama" | "llamacpp";
  apiUrl?: string;
  apiKey?: string;
  modelId?: string;
}

export const DEFAULT_EMBEDDING: EmbeddingConfig = {
  provider: "vllm",
  apiUrl: "",
  apiKey: "",
  modelId: "Qwen/Qwen3-Embedding-8B",
};

export interface AIAgentConfig {
  provider: "openai" | "anthropic" | "gemini" | "groq" | "xai" | "vultr" | "custom";
  apiUrl?: string;
  apiKey?: string;
  modelId?: string;
}

export const DEFAULT_AI_AGENT: AIAgentConfig = {
  provider: "vultr",
  apiUrl: "https://api.vultrinference.com/v1",
  apiKey: "",
  modelId: "deepseek-chat",
};

export const DEFAULT_DIGITAL_OCEAN: DigitalOceanConfig = {
  apiKey: "",
  dropletName: "",
  region: "",
  size: "",
  image: "220895104",
  sshKeys: "",
  projectId: "",
  tags: "",
  backups: false,
  ipv6: false,
  monitoring: true,
  userData: "",
};

export const POPULAR_MODELS: Record<string, string[]> = {
  openai: [
    "gpt-5.5",
    "gpt-5.5-instant",
    "gpt-5.4-mini",
    "gpt-4o",
    "gpt-4o-mini",
    "o1",
    "o1-mini"
  ],
  anthropic: [
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-latest",
    "claude-3-opus-latest"
  ],
  gemini: [
    "gemini-3.5-flash",
    "gemini-3.1-pro",
    "gemini-1.5-flash",
    "gemini-1.5-pro"
  ],
  groq: [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
  ],
  xai: [
    "grok-4.3",
    "grok-2",
    "grok-beta",
    "grok-2-mini"
  ],
  vultr: [
    "MiniMax-M2.7",
    "MiMo-V2.5-Pro",
    "Kimi-K2.6",
    "DeepSeek-V3.2-NVFP4",
    "Llama-3.1-Nemotron-Safety-Guard-8B-v3",
    "Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16",
    "Nemotron-Cascade-2-30B-A3B",
    "GLM-5.1-FP8"
  ],
  custom: []
};


export interface AppConfig {
  ssh: SSHConfig;
  digitalOcean?: DigitalOceanConfig;
  qdrant: QdrantConfig;
  hfToken?: string | null;
  embedders?: EmbedderConfig[];
  teacher: TeacherConfig;
  student: StudentConfig;
  docker: DockerConfig;
  paddleOcr?: PaddleOcrConfig;
  aiAgent?: AIAgentConfig;
  promptTemplate?: string;
  embedding?: EmbeddingConfig;
}

export interface LoraConfig {
  method?:
    | "lora"
    | "qlora"
    | "unsloth"
    | "full"
    | "freeze"
    | "dora"
    | "loraplus"
    | "pissa"
| "galore"
      | "badam"
      | "grpo"
      | "custom";
  customMethodName?: string;
  customCommands?: string[];
  r: number;
  alpha: number;
  dropout: number;
  learningRate: number;
  epochs: number;
  batchSize: number;
  gradientAccumulation: number;
  cutoffLen: number;
  saveSteps?: number;
}

export interface MatchedGuideInfo {
  label: string;
  notebook: string;
  family: string;
  recommendedMethod: string;
}

export interface HubConfig {
  enabled: boolean;
  modelId: string;
  private: boolean;
  strategy: "every_save" | "end" | "checkpoint";
  /** After training completes, also merge LoRA into the base model and
   *  upload the full merged weights to `mergedModelId` (or `<modelId>-merged
   *  if empty). Skipped when `enabled` is false. */
  autoMerge?: boolean;
  mergedModelId?: string;
  /** Destroy the GPU droplet after training completes (after merge+push if autoMerge is enabled). Saves GPU server costs. */
  autoDestroy?: boolean;
  /** After merge, also convert to GGUF and upload to a dedicated repo for Ollama/llama.cpp. */
  autoConvertGguf?: boolean;
  /** Quantization type for GGUF: "F16", "Q4_K_M" (default), "Q5_K_M", "Q8_0". */
  ggufQuantization?: string;
  /** Target GGUF repo ID (defaults to <modelId>-gguf). */
  ggufRepoId?: string;
}

/** Auto-upload the generated dataset to a HF *dataset* repo every N pairs. */
export interface HubDatasetConfig {
  /** Master switch. */
  enabled: boolean;
  /** Target HF dataset repo, e.g. "your-name/ge-reviewer-qa". */
  repoId: string;
  /** Train-Only mode: extra HF dataset repos to interleave alongside `repoId`.
   *  When non-empty, LLaMA-Factory receives a comma-separated `dataset` field
   *  built from these (plus `repoId` if it isn't already in the list). All
   *  datasets share the format/columns configured below. */
  repoIds?: string[];
  /** Private repo (default true). */
  private: boolean;
  /** Push after every N accepted pairs. 0 = only at end. */
  everyN: number;
  /** Optional: HF dataset repo to download + seed from before generating. */
  resumeFrom: string;
  /** Skip dataset generation and train directly from Hugging Face dataset. */
  trainOnly?: boolean;
  /** Dataset format: "sharegpt" (default) or "alpaca". */
  datasetFormat?: "sharegpt" | "alpaca";
  /** Optional column mapping for the dataset. */
  datasetColumns?: {
    prompt?: string;
    query?: string;
    response?: string;
    messages?: string;
  };
  /** Number of samples in the dataset (for training-only mode display). */
  sampleCount?: number;
  /** Dataset validation state (training-only mode). Keyed by repoId. */
  validationResult?: Record<string, {
    valid: boolean;
    sampleCount?: number;
    columns?: string[];
    error?: string;
    validatedAt?: number;
  }>;
}

export type RunStatus =
  | "pending"
  | "teacher_loading"
  | "generating_dataset"
  | "dataset_ready"
  | "training"
  | "done"
  | "failed"
  | "cancelled";

export interface TrainPoint {
  step: number;
  loss: number;
  epoch: number;
}

export interface Run {
  id: string;
  name: string;
  createdAt: string;
  updatedAt: string;
  teacherModel: string;
  studentModel: string;
  status: RunStatus;
  qaTotal: number;
  qaKept: number;
  qaRejected: number;
  trainLossHistory: TrainPoint[];
  error?: string | null;
  logTail: string;
  remoteDir: string;
  localDir: string;
  lora: LoraConfig;
  teacherCfg: TeacherConfig;
  hub?: HubConfig;
  hubDataset?: HubDatasetConfig;
  datasetReady?: boolean;
  lastTrainStep?: number;
  /** Per-topic Q&A pair counts: topic → number of accepted pairs. */
  topicStats?: Record<string, number>;
}

/** One row in the multi-topic editor: a focus topic + an optional per-topic
 *  cap on accepted Q&A pairs. The pipeline iterates these in order, swapping
 *  `{topic}` in the prompt and stopping each loop once its `totalQuestions`
 *  is reached. */
export interface TopicTarget {
  topic: string;
  totalQuestions?: number;
  tag?: string;
  /** Optional per-topic prompt template that overrides the global
   *  `RunConfig.promptTemplate` during this topic's generation pass.
   *  When omitted or blank, the global template is used. */
  promptTemplate?: string;
}

export interface RunConfig {
  name: string;
  teacher: TeacherConfig;
  studentModel: string;
  lora: LoraConfig;
  promptTemplate?: string;
  maxPairsPerChunk: number;
  concurrency: number;
  maxChunks?: number;
  /** Focus topic injected into the prompt as `{topic}` (single-topic mode). */
  topic?: string;
  /** Hard cap on accepted Q&A pairs (single-topic mode). */
  totalQuestions?: number;
  /** Multi-topic mode — if non-empty, overrides `topic`/`totalQuestions`. */
  topics?: TopicTarget[];
  hub?: HubConfig;
  hubDataset?: HubDatasetConfig;
  generateOnly?: boolean;
  /** Which provider supplies the Teacher API. Defaults to "vllm" if missing. */
  teacherProvider?: "vllm";
}

/** Result of `hf_whoami` — used to prefill repo-id placeholders. */
export interface HfWhoami {
  name: string;
  fullname?: string;
  avatarUrl?: string;
  tokenRole?: string;
}

/** One repo returned by `hf_list_datasets`. */
export interface HfDatasetRepo {
  id: string;
  private: boolean;
  lastModified?: string;
}

/** One model repo returned by `hf_list_models`. */
export interface HfModelRepo {
  id: string;
  private: boolean;
  lastModified?: string;
}

export interface ConnectionStatus {
  isConnected: boolean;
  isTesting: boolean;
  message: string;
}

export interface GPUProcess {
  pid: number;
  processName: string;
  memory: number;
}

export interface GPUState {
  success: boolean;
  simulated: boolean;
  gpuName: string;
  driverVersion: string;
  cudaVersion: string;
  temperature: number;
  utilizationGpu: number;
  utilizationMemory: number;
  memoryUsed: number;
  memoryTotal: number;
  powerDraw: number;
  powerLimit: number;
  fanSpeed: number;
  processes: GPUProcess[];
  systemInfo: string;
}

export interface Chunk {
  id: string;
  text: string;
  file_path: string;
  file_name: string;
  chunk_index: number;
}

export interface CommandPreset {
  id: string;
  name: string;
  category: "Serving" | "Monitoring" | "Fine-tuning" | "SystemSetup" | "AI-Agent" | "Custom";
  description: string;
  command: string;
  requiresHFToken?: boolean;
}

// Event payloads emitted from Rust
export interface ShellLogEvent {
  streamId: string;
  kind: "stdout" | "stderr" | "info" | "error";
  line: string;
}

export interface ShellDoneEvent {
  streamId: string;
  exitCode: number;
}

export interface PipelineLogEvent {
  runId: string;
  line: string;
  kind: string;
}

export interface PipelineProgressEvent {
  runId: string;
  scanned: number;
  kept: number;
  rejected: number;
  status: RunStatus;
}

export interface PipelineMetricEvent {
  runId: string;
  step: number;
  loss: number;
  epoch: number;
}

/** Per-stage tick emitted by `ingest_documents`. `done`/`total` are 0
 *  during the `read` stage (chunks unknown until parse). */
export interface IngestProgressEvent {
  streamId: string;
  /** "read" | "embed" | "upsert" | "done" | "error" */
  stage: string;
  file: string;
  done: number;
  total: number;
}

/** Per-file outcome inside `IngestDoneEvent.summary.files`. Mirrors
 *  `ingest::FileResult` on the Rust side. */
export interface IngestFileResult {
  file_path: string;
  file_name: string;
  chunks_ingested: number;
  error?: string | null;
}

export interface IngestSummary {
  total_files: number;
  total_chunks: number;
  files: IngestFileResult[];
  cancelled: boolean;
}

/** Final event for an ingest stream. On success, `summary` is populated;
 *  on failure, `error` carries the top-level error string. */
export interface IngestDoneEvent {
  streamId: string;
  success: boolean;
  summary?: IngestSummary;
  error?: string;
}

export const DEFAULT_LORA: LoraConfig = {
  method: "lora",
  r: 16,
  alpha: 32,
  dropout: 0.05,
  learningRate: 5e-5,
  epochs: 2,
  batchSize: 4,
  gradientAccumulation: 4,
  cutoffLen: 4096,
  saveSteps: 100,
};

export const DEFAULT_HUB: HubConfig = {
  enabled: false,
  modelId: "",
  private: true,
  strategy: "every_save",
  autoMerge: false,
  mergedModelId: "",
};

export const DEFAULT_HUB_DATASET: HubDatasetConfig = {
  enabled: false,
  repoId: "",
  repoIds: [],
  private: true,
  everyN: 100,
  resumeFrom: "",
};

export const DEFAULT_TEACHER: TeacherConfig = {
  repoId: "deepseek-ai/DeepSeek-V3",
  vllmPort: 8000,
  maxModelLen: 32768,
  dtype: "bfloat16",
  tensorParallel: 1,
  gpuMemoryUtilization: 0.95,
  autoTune: true,
  enableChunkedPrefill: true,
  maxNumBatchedTokens: 8192,
  maxNumSeqs: 16,
  enableAutoToolChoice: false,
  toolCallParser: "",
  customServeCmd: "",
  servingEngine: "sglang",
};

export interface IngestStream {
  id: string;
  files: string[];
  tag: string;
  progress: { file: string; done: number; total: number } | null;
  done: boolean;
  cancelled: boolean;
  chunks: number;
  errors: { file: string; error: string }[];
  error: string | null;
}
