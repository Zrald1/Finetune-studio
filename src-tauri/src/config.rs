use crate::error::{AppError, Result};
use crate::ingest;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tokio::fs;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct SshConfig {
    pub host: String,
    #[serde(default = "default_ssh_port")]
    pub port: u16,
    #[serde(default = "default_username")]
    pub username: String,
    pub private_key_path: Option<String>,
    pub private_key: Option<String>, // raw PEM contents
    pub password: Option<String>,
}

fn default_ssh_port() -> u16 {
    22
}
fn default_gpu_memory_utilization() -> f32 {
    0.084
}
fn default_embedder_port() -> u16 {
    8101
}
fn default_teacher_gpu_memory_utilization() -> f32 {
    0.80
}
fn default_username() -> String {
    "root".to_string()
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct QdrantConfig {
    /// Base URL of the Qdrant instance. For the self-hosted GPU-server flow this
    /// is `http://<droplet-ip>:6333`. `api_key` is blank for the local instance.
    pub endpoint: String,
    pub api_key: String,
    /// Default/legacy single-collection name. Multi-embedder ingest uses each
    /// embedder's own `collection` instead (see `EmbedderConfig`).
    pub collection: String,
}

/// One self-hosted embedding model served on the GPU server via
/// `vllm serve <model_id> --runner pooling --port <port>` (or `--task embed` on older vLLM). Each embedder owns its
/// own Qdrant collection (different models produce different vector dims, so
/// collections can't be shared). The user can add as many as the GPU allows.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct EmbedderConfig {
    /// Human label, e.g. "law", "math", "science". Drives the default collection.
    pub name: String,
    /// Hugging Face model id served on vLLM, e.g. "Qwen/Qwen3-Embedding-8B".
    pub model_id: String,
    /// Dedicated host port the embedder's vLLM `/v1/embeddings` listens on.
    pub port: u16,
    /// Qdrant collection that holds this embedder's chunks. Defaults to a slug
    /// of `name` (e.g. "kb_law") when blank.
    pub collection: String,
    /// In-flight embed requests during ingest for this embedder.
    pub concurrency: u32,
    /// Detected on first successful embed; used to create the collection with the
    /// matching vector size. `None` until the first ingest probes the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector_dim: Option<usize>,
    /// Whether this embedder participates in "Setup all embedding models".
    pub enabled: bool,
    /// If true, this embedder is protected from GPU cleanup during teacher
    /// deploy. Persistent embedders survive across teacher deployments so the
    /// pipeline can reuse them without the 3–5 minute VRAM load time on every
    /// dataset generation run.
    #[serde(default)]
    pub persistent: bool,
    /// GPU memory utilization (0.0–1.0) for this embedder's vLLM instance.
    /// Each embedder uses its own value independently — no subdivision logic.
    #[serde(default = "default_gpu_memory_utilization")]
    pub gpu_memory_utilization: f32,
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        default_semantic_embedder()
    }
}

pub fn default_semantic_embedder() -> EmbedderConfig {
    EmbedderConfig {
        name: "embedder_1".to_string(),
        model_id: "Qwen/Qwen3-Embedding-8B".to_string(),
        port: default_embedder_port(),
        collection: String::new(),
        concurrency: 2,
        vector_dim: None,
        enabled: true,
        persistent: true,
        gpu_memory_utilization: default_gpu_memory_utilization(),
    }
}

pub fn normalize_embedders(embedders: &mut Vec<EmbedderConfig>) {
    if embedders.is_empty() {
        embedders.push(default_semantic_embedder());
    }

    for (idx, embedder) in embedders.iter_mut().enumerate() {
        if embedder.name.trim().is_empty() {
            embedder.name = format!("embedder_{}", idx + 1);
        }
        if embedder.model_id.trim().is_empty() {
            embedder.model_id = "Qwen/Qwen3-Embedding-8B".to_string();
        }
        if embedder.port == 0 {
            embedder.port = default_embedder_port() + idx as u16;
        }
        if idx == 0 && embedder.port == 8100 && embedder.name.trim() == "embedder_1" {
            embedder.port = default_embedder_port();
        }
        if embedder.concurrency == 0 {
            embedder.concurrency = 2;
        }
        if embedder.gpu_memory_utilization <= 0.0 {
            embedder.gpu_memory_utilization = default_gpu_memory_utilization();
        }
        if idx == 0 {
            embedder.enabled = true;
            embedder.persistent = true;
        }
    }
}

pub fn normalize_runtime_defaults(cfg: &mut AppConfig) {
    if cfg.qdrant.collection.trim().is_empty() {
        cfg.qdrant.collection = "all".to_string();
    }
    if looks_like_non_teacher_service_model(&cfg.teacher.repo_id) {
        let default_teacher = TeacherConfig::default();
        cfg.teacher.repo_id = default_teacher.repo_id;
        if cfg.teacher.vllm_port == cfg.paddle_ocr.port || cfg.teacher.vllm_port == 8118 {
            cfg.teacher.vllm_port = default_teacher.vllm_port;
        }
    }
    normalize_embedders(&mut cfg.embedders);
}

fn looks_like_non_teacher_service_model(model_id: &str) -> bool {
    let id = model_id.trim().to_lowercase();
    !id.is_empty()
        && [
            "paddleocr",
            "paddle-ocr",
            "paddle_ocr",
            "paddleocr-vl",
            "paddleocr-vl-1.6-0.9b",
            "embedding",
            "jina-embeddings",
        ]
        .iter()
        .any(|marker| id.contains(marker))
}

impl EmbedderConfig {
    /// The collection name to use, falling back to a slug of `name`.
    pub fn effective_collection(&self) -> String {
        let c = self.collection.trim();
        if !c.is_empty() {
            return c.to_string();
        }
        let slug: String = self
            .name
            .trim()
            .to_lowercase()
            .chars()
            .map(|ch| if ch.is_alphanumeric() { ch } else { '_' })
            .collect();
        let slug = slug.trim_matches('_');
        if slug.is_empty() {
            "kb_default".to_string()
        } else {
            format!("kb_{}", slug)
        }
    }

    /// OpenAI-compatible base URL for this embedder against the given host.
    pub fn api_url(&self, host: &str) -> String {
        format!("http://{}:{}", host, self.port)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ServingEngine {
    Vllm,
    Sglang,
}

impl Default for ServingEngine {
    fn default() -> Self {
        Self::Vllm
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct TeacherConfig {
    pub repo_id: String,
    pub vllm_port: u16,
    pub max_model_len: u32,
    pub dtype: String,
    pub tensor_parallel: u32,
    #[serde(default = "default_teacher_gpu_memory_utilization")]
    pub gpu_memory_utilization: f32,
    pub auto_tune: bool,
    pub enable_chunked_prefill: bool,
    pub max_num_batched_tokens: Option<u32>,
    pub max_num_seqs: Option<u32>,
    pub enable_auto_tool_choice: bool,
    pub tool_call_parser: Option<String>,
    pub custom_serve_cmd: Option<String>,
    /// Extra vLLM flags appended to the managed `vllm serve` command (e.g.
    /// `--quantization gguf --block-size 32 --enable-prefix-caching`). Unlike
    /// `custom_serve_cmd`, this does NOT replace the command — the model, port,
    /// host, dtype and ROCm env vars stay intact and these flags are added on top.
    #[serde(default)]
    pub extra_serve_args: Option<String>,
    #[serde(default)]
    pub serving_engine: ServingEngine,
}

impl Default for TeacherConfig {
    fn default() -> Self {
        Self {
            repo_id: "deepseek-ai/DeepSeek-V3".to_string(),
            vllm_port: 8000,
            max_model_len: 32768,
            dtype: "bfloat16".to_string(),
            tensor_parallel: 1,
            gpu_memory_utilization: 0.80,
            auto_tune: true,
            enable_chunked_prefill: true,
            max_num_batched_tokens: Some(8192),
            max_num_seqs: Some(16),
            enable_auto_tool_choice: false,
            tool_call_parser: None,
            custom_serve_cmd: None,
            extra_serve_args: None,
            serving_engine: ServingEngine::Vllm,
        }
    }
}

impl TeacherConfig {
    pub fn resolved_for_gpu(&self, gpu_memory_total_mb: Option<f64>) -> Self {
        let mut resolved = self.clone();
        resolved.serving_engine = ServingEngine::Vllm;
        if !resolved.auto_tune
            || resolved
                .custom_serve_cmd
                .as_ref()
                .map(|s| !s.trim().is_empty())
                .unwrap_or(false)
        {
            return resolved;
        }

        let repo = resolved.repo_id.to_lowercase();
        let memory_gb = gpu_memory_total_mb.unwrap_or(0.0) / 1024.0;
        let is_qwen3 = repo.contains("qwen3");
        let is_vl = repo.contains("-vl") || repo.contains("vision");
        let is_gguf = repo.contains("gguf");

        resolved.dtype = "bfloat16".to_string();
        resolved.gpu_memory_utilization = 0.80;
        resolved.tensor_parallel = resolved.tensor_parallel.max(1);
        resolved.enable_chunked_prefill = true;

        resolved.max_model_len = if is_qwen3 && is_vl && memory_gb >= 180.0 {
            100000
        } else if (is_qwen3 || is_vl) && memory_gb >= 96.0 {
            65536
        } else if is_gguf {
            32768
        } else {
            resolved.max_model_len.max(32768)
        };

        resolved.max_num_batched_tokens = Some(if memory_gb > 0.0 && memory_gb < 64.0 {
            4096
        } else {
            8192
        });

        resolved.max_num_seqs = Some(if memory_gb > 0.0 && memory_gb < 64.0 {
            4
        } else if memory_gb > 0.0 && memory_gb < 128.0 {
            8
        } else {
            16
        });

        if is_qwen3 {
            resolved.enable_auto_tool_choice = true;
            resolved.tool_call_parser = Some("qwen3_coder".to_string());
        } else {
            resolved.enable_auto_tool_choice = false;
            resolved.tool_call_parser = None;
        }

        resolved
    }

    /// True when the configured teacher repo is a GGUF model. vLLM cannot load
    /// a bare GGUF *repo* (no config.json) — the caller must resolve the actual
    /// `.gguf` file path and serve that, using `gguf_base_model()` for the
    /// tokenizer / hf-config.
    pub fn is_gguf(&self) -> bool {
        self.repo_id.to_lowercase().contains("gguf")
    }

    /// The base (safetensors) model that produced a GGUF repo: strips the
    /// `-gguf`/`.gguf` suffix and any trailing `:Q4_K_M`-style quant tag. Used
    /// for `--tokenizer` and `--hf-config-path` so vLLM can build a config even
    /// when the GGUF repo itself ships none. Returns the repo unchanged when it
    /// isn't a GGUF repo.
    pub fn gguf_base_model(&self) -> String {
        if !self.is_gguf() {
            return self.repo_id.clone();
        }
        let parts: Vec<&str> = self.repo_id.split('/').collect();
        let base_repo = if parts.len() >= 2 {
            format!("{}/{}", parts[0], parts[1].split(':').next().unwrap_or(parts[1]))
        } else {
            self.repo_id
                .split(':')
                .next()
                .unwrap_or(&self.repo_id)
                .to_string()
        };
        base_repo
            .replace("-GGUF", "")
            .replace("-gguf", "")
            .replace(".GGUF", "")
            .replace(".gguf", "")
    }

    pub fn vllm_extra_args(&self) -> String {
        let mut args = Vec::new();
        if self.enable_chunked_prefill {
            args.push("--enable-chunked-prefill".to_string());
        }
        if let Some(tokens) = self.max_num_batched_tokens.filter(|n| *n > 0) {
            args.push(format!("--max-num-batched-tokens {}", tokens));
        }
        if let Some(seqs) = self.max_num_seqs.filter(|n| *n > 0) {
            args.push(format!("--max-num-seqs {}", seqs));
        }
        if self.enable_auto_tool_choice {
            args.push("--enable-auto-tool-choice".to_string());
            if let Some(parser) = self
                .tool_call_parser
                .as_ref()
                .filter(|s| !s.trim().is_empty())
            {
                args.push(format!("--tool-call-parser {}", parser.trim()));
            }
        }
        // User-supplied advanced flags from the Deploy page (quantization,
        // block-size, swap-space, kv-cache-dtype, prefix caching, …). Appended
        // last so they can override defaults set above.
        if let Some(extra) = self
            .extra_serve_args
            .as_ref()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
        {
            args.push(extra.to_string());
        }
        args.join(" ")
    }

    pub fn vllm_runtime_prepare_cmd(&self) -> String {
        let mut prepare = "python3 -c 'import torchvision' 2>&1 | grep -E -q 'nms|operator' && python3 -m pip uninstall -y torchvision || true; ".to_string();

        let repo = self.repo_id.to_lowercase();

        if repo.contains("deepseek-v4") || repo.contains("deepseek_v4") {
            prepare.push_str("python3 -c \"from transformers.models.auto.configuration_auto import CONFIG_MAPPING; import sys; sys.exit(0 if \\\"deepseek_v4\\\" in CONFIG_MAPPING else 1)\" || { echo [compat] installing Transformers with DeepSeek V4 support; python3 -m pip install --no-cache-dir --upgrade git+https://github.com/huggingface/transformers.git || exit 42; }; ");
        }

        let model_slug = self.repo_id.split('/').last().unwrap_or(&self.repo_id);
        prepare.push_str(&format!(
            "python3 -c \"\
               import json,urllib.request,sys; \
               url='https://huggingface.co/{repo}/raw/main/config.json'; \
               req=urllib.request.Request(url, headers={{'User-Agent':'fine-tune'}}); \
               cfg=json.load(urllib.request.urlopen(req, timeout=15)); \
               mt=cfg.get('model_type',''); \
               from transformers import AutoConfig; \
               from transformers.models.auto.configuration_auto import CONFIG_MAPPING; \
               if mt and mt not in CONFIG_MAPPING: \
                 print(f'[compat] transformers does not recognize model_type={{mt!r}} — upgrading from source'); sys.exit(1); \
             \" 2>/dev/null && echo '[compat] transformers OK' || {{ \
               echo '[compat] upgrading transformers from source for {model_slug}...'; \
               python3 -m pip install --no-cache-dir --upgrade git+https://github.com/huggingface/transformers.git || exit 42; \
             }}; ",
            repo = self.repo_id,
            model_slug = model_slug,
        ));

        prepare.push_str(
            "python3 -c \"import site,os; p=os.path.join(site.getsitepackages()[0],'zz_finetune_hetero_fix.pth'); open(p,'w').write('import transformers.configuration_utils as _tc; _tc.PretrainedConfig.allow_global_per_layer_attribute_access=True\\n'); print('[compat] heterogeneity fix installed')\" 2>/dev/null; ",
        );

        prepare
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct StudentConfig {
    pub repo_id: String,
    pub output_dir: String,
}

impl Default for StudentConfig {
    fn default() -> Self {
        Self {
            repo_id: "Qwen/Qwen2.5-7B-Instruct".to_string(),
            output_dir: "/root/fine-tune/runs".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct DockerConfig {
    pub enabled: bool,
    pub container_name: String,
    pub image_name: String,
    pub start_args: String,
    pub bypass_terminal: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct DigitalOceanConfig {
    pub api_key: String,
    pub droplet_name: String,
    pub region: String,
    pub size: String,
    pub hourly_rate_usd: Option<f64>,
    pub image: String,
    pub ssh_keys: String,
    pub project_id: String,
    pub tags: String,
    pub backups: bool,
    pub ipv6: bool,
    pub monitoring: bool,
    pub user_data: String,
}

impl Default for DigitalOceanConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            droplet_name: String::new(),
            region: String::new(),
            size: String::new(),
            hourly_rate_usd: None,
            image: "220895104".to_string(),
            ssh_keys: String::new(),
            project_id: String::new(),
            tags: String::new(),
            backups: false,
            ipv6: false,
            monitoring: true,
            user_data: String::new(),
        }
    }
}

impl Default for DockerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            container_name: "rocm-vllm".to_string(),
            image_name: "rocm/vllm:latest".to_string(),
            start_args: "--device=/dev/kfd --device=/dev/dri --network=host --ipc=host --group-add video -v /root:/root".to_string(),
            bypass_terminal: false,
        }
    }
}

/// Mirror of the frontend's AIAgentConfig (src/types.ts). Persisting this
/// server-side lets the API key the user types into the AI terminal panel
/// survive an app restart — previously the field was silently dropped here,
/// so on reload the frontend fell back to DEFAULT_AI_AGENT (provider:
/// featherless) and the user had to retype the key every session.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct AiAgentConfig {
    #[serde(default)]
    pub provider: String,
    #[serde(default)]
    pub api_url: String,
    #[serde(default)]
    pub api_key: String,
    #[serde(default)]
    pub model_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct PaddleOcrConfig {
    pub enabled: bool,
    #[serde(default)]
    pub port: u16,
    #[serde(default)]
    pub model_name: String,
    #[serde(default)]
    pub docker_image: String,
}

impl Default for PaddleOcrConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            port: 8118,
            model_name: "PaddleOCR-VL-1.6-0.9B".to_string(),
            docker_image: "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-amd-gpu".to_string(),
        }
    }
}

/// Configuration for the robot↔server bridge. Round-trips through the same
/// `config.json` as everything else, so the desktop app and the headless VPS
/// server share one source of truth. All fields are additive and default-safe
/// so existing config files keep parsing.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct RobotConfig {
    pub enabled: bool,
    /// Bearer token the robot must present on `/robot/*` endpoints.
    pub robot_api_token: String,
    /// Bearer token the desktop/dashboard client presents on operator endpoints.
    pub dashboard_api_token: String,
    /// If non-empty, only these robot ids may submit captures.
    pub allowed_robot_ids: Vec<String>,
    /// Captures below this detection confidence are rejected at intake.
    pub min_capture_confidence: f32,
    /// Skip duplicate captures of the same object within this window.
    pub dedupe_window_secs: u64,
    /// Blur faces / license plates before the image is stored (privacy guard).
    pub blur_faces_plates: bool,
    /// Qdrant collection robot research packets are embedded into.
    pub research_collection: String,
    /// Run web research automatically on capture. Training stays gated by
    /// human approval regardless of this flag.
    pub auto_research_on_capture: bool,
}

impl Default for RobotConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            robot_api_token: String::new(),
            dashboard_api_token: String::new(),
            allowed_robot_ids: Vec::new(),
            min_capture_confidence: 0.0,
            dedupe_window_secs: 300,
            blur_faces_plates: false,
            research_collection: "kb_robot".to_string(),
            auto_research_on_capture: true,
        }
    }
}

/// Pluggable web-research provider config. The robot pipeline uses this to
/// research an unfamiliar captured object online before building training data.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct WebResearchConfig {
    /// "brave" | "serpapi" | "google_cse".
    pub provider: String,
    pub api_key: String,
    /// Google Programmable Search engine id (google_cse only).
    pub cse_id: Option<String>,
    /// Only fetch result pages whose host matches one of these (empty = allow all).
    pub domain_allowlist: Vec<String>,
    pub max_results: u32,
    /// Drop results matching a built-in dangerous-topic blocklist.
    pub block_dangerous_topics: bool,
}

impl Default for WebResearchConfig {
    fn default() -> Self {
        Self {
            provider: "brave".to_string(),
            api_key: String::new(),
            cse_id: None,
            domain_allowlist: Vec::new(),
            max_results: 5,
            block_dangerous_topics: true,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct AppConfig {
    pub ssh: SshConfig,
    #[serde(default)]
    pub digital_ocean: DigitalOceanConfig,
    pub qdrant: QdrantConfig,
    pub hf_token: Option<String>,
    /// Deprecated: Featherless cloud embedding/teacher was removed in favour of
    /// self-hosted vLLM embedders on the GPU server. Kept here only so existing
    /// config.json files (which still carry this key) continue to parse. Never
    /// read, never written.
    #[allow(dead_code)]
    #[serde(default, skip_serializing)]
    pub featherless_api_key: Option<String>,
    /// Self-hosted embedding models served on the GPU server. Each owns a Qdrant
    /// collection. Replaces the old cloud-embedding config.
    #[serde(default)]
    pub embedders: Vec<EmbedderConfig>,
    pub teacher: TeacherConfig,
    pub student: StudentConfig,
    pub docker: DockerConfig,
    #[serde(default)]
    pub paddle_ocr: PaddleOcrConfig,
    #[serde(default)]
    pub ai_agent: Option<AiAgentConfig>,
    #[serde(default)]
    pub prompt_template: Option<String>,
    pub embedding: Option<ingest::EmbeddingConfig>,
    /// Robot↔server bridge settings (headless VPS mode + robotics widget).
    #[serde(default)]
    pub robot: RobotConfig,
    /// Web-research provider used by the robot capture pipeline.
    #[serde(default)]
    pub web_research: WebResearchConfig,
}

pub fn app_dir() -> Result<PathBuf> {
    // Headless server mode sets FT_DATA_DIR to a fixed path (e.g. /var/lib/fine-tune)
    // so it does not depend on an OS "user config dir". The desktop app leaves it
    // unset and uses the per-user config dir as before.
    if let Ok(dir) = std::env::var("FT_DATA_DIR") {
        if !dir.trim().is_empty() {
            return Ok(PathBuf::from(dir));
        }
    }
    let base = dirs::config_dir().ok_or_else(|| AppError::config("no OS config dir available"))?;
    Ok(base.join("fine-tune"))
}

/// Directory holding the robot capture queue (one JSON file per capture).
pub fn robot_dir() -> Result<PathBuf> {
    Ok(app_dir()?.join("robot"))
}

/// Path to the model-manifest store served to the robot.
pub fn manifest_path() -> Result<PathBuf> {
    Ok(app_dir()?.join("model_manifests.json"))
}

pub fn config_path() -> Result<PathBuf> {
    Ok(app_dir()?.join("config.json"))
}

pub fn runs_dir() -> Result<PathBuf> {
    Ok(app_dir()?.join("runs"))
}

pub async fn ensure_dirs() -> Result<()> {
    fs::create_dir_all(app_dir()?).await?;
    fs::create_dir_all(runs_dir()?).await?;
    fs::create_dir_all(robot_dir()?).await?;
    Ok(())
}

pub async fn load() -> Result<AppConfig> {
    ensure_dirs().await?;
    let path = config_path()?;
    if !path.exists() {
        let mut cfg = AppConfig::default();
        normalize_runtime_defaults(&mut cfg);
        return Ok(cfg);
    }
    let txt = fs::read_to_string(&path).await?;
    let mut cfg: AppConfig = serde_json::from_str(&txt)
        .map_err(|e| AppError::config(format!("parse config.json: {e}")))?;
    if cfg.qdrant.endpoint.is_empty() && !cfg.ssh.host.is_empty() {
        cfg.qdrant.endpoint = format!("http://{}:6333", cfg.ssh.host);
    }

    // Migrate old PaddleOCR config values: only fill in blank model_name.
    let default_pocr = PaddleOcrConfig::default();
    if cfg.paddle_ocr.model_name.trim().is_empty() {
        cfg.paddle_ocr.model_name = default_pocr.model_name.clone();
    }
    normalize_runtime_defaults(&mut cfg);
    Ok(cfg)
}

pub async fn save(cfg: &AppConfig) -> Result<()> {
    ensure_dirs().await?;
    let mut cfg = cfg.clone();
    normalize_runtime_defaults(&mut cfg);
    let txt = serde_json::to_string_pretty(&cfg)?;
    fs::write(config_path()?, txt).await?;
    Ok(())
}
