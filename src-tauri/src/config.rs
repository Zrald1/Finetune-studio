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

/// Serving presets for the teacher model.
///
/// `Standard` is the conservative, deterministic baseline: BF16 weights, a
/// 64K context and vLLM's default 0.80 memory budget — nothing exotic, and
/// nothing that depends on a particular checkpoint's published recipe.
///
/// `Optimized` applies the vendor-published serving recipe for the selected
/// model (for Qwen3.8-27B: the full 262K native context, an FP8 KV cache, and
/// prefix caching). It is a strict superset of `Standard` and can be reverted
/// at any time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ServingProfile {
    Standard,
    Optimized,
}

impl Default for ServingProfile {
    fn default() -> Self {
        Self::Standard
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
    /// Which serving preset to apply when `auto_tune` is on.
    #[serde(default)]
    pub serving_profile: ServingProfile,
    /// vLLM `--reasoning-parser` value. Required for hybrid-thinking models
    /// (Qwen3.5+) whose chat template opens every assistant turn with
    /// ` thinking`: without a parser the reasoning block lands in
    /// `message.content` and consumes the output budget before the answer
    /// starts. The dataset generator reads only `message.content`, so leaving
    /// this unset makes the teacher burn tokens on text that gets stripped.
    #[serde(default)]
    pub reasoning_parser: Option<String>,
    /// How hard the teacher should think before answering. Qwen3.8's chat
    /// template accepts only `xhigh`, `medium`, and `low`; `none` is this app's
    /// sentinel for turning thinking off entirely. Applied per request rather
    /// than via `--default-chat-template-kwargs` because that flag takes JSON,
    /// and the deploy command embeds its arguments inside a single-quoted
    /// `bash -lc '...'` where JSON quoting is fragile.
    #[serde(default)]
    pub reasoning_effort: Option<String>,
}

impl Default for TeacherConfig {
    fn default() -> Self {
        Self {
            repo_id: "Qwen/Qwen3.8-27B".to_string(),
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
            serving_profile: ServingProfile::Standard,
            reasoning_parser: None,
            reasoning_effort: None,
        }
    }
}

/// Qwen3.8-27B ships a vision tower (`Qwen3_5ForConditionalGeneration` with a
/// `vision_config`) but its repo id carries no `-vl` / `vision` marker, so
/// name-based multimodal detection has to special-case it.
fn is_qwen3_8_multimodal(repo_lower: &str) -> bool {
    repo_lower.contains("qwen3.8") || repo_lower.contains("qwen3_8")
}

/// True for hybrid checkpoints that mix full attention with linear-attention
/// ("Mamba"-style) layers — Qwen3.5 and later, Jamba, and anything advertising
/// a Gated DeltaNet.
///
/// This matters because vLLM warns that prefix caching over Mamba layers is
/// **experimental**:
///
/// ```text
/// WARNING [config.py:618] Mamba cache mode is set to 'align' for
///         Qwen3_5ForConditionalGeneration by default when prefix caching is enabled
/// INFO    [config.py:638] Prefix caching in Mamba cache 'align' mode is currently
///         enabled. Its support for Mamba layers is experimental.
/// ```
///
/// Silently opting the model into an experimental KV-reuse path risks quietly
/// corrupting generated training data, which is worse than generating slowly.
fn is_hybrid_linear_attention(repo_lower: &str) -> bool {
    const MARKERS: [&str; 4] = ["mamba", "gdn", "jamba", "deltanet"];
    if MARKERS.iter().any(|m| repo_lower.contains(m)) {
        return true;
    }
    // Qwen3.5 and every later generation use the 3:1 linear/full layer mix.
    ["qwen3.5", "qwen3.6", "qwen3.7", "qwen3.8", "qwen3_5", "qwen3_6", "qwen3_7", "qwen3_8"]
        .iter()
        .any(|m| repo_lower.contains(m))
}

/// The `chat_template_kwargs` object for a request to this teacher, or `None`
/// when no effort is configured so the checkpoint's own default applies.
///
/// Qwen3.8's `chat_template.jinja` validates the value and calls
/// `raise_exception` for anything outside `{xhigh, medium, low}`, which vLLM
/// surfaces as HTTP 500 rather than a 4xx — so unsupported values must never be
/// forwarded. `high` and `max` are folded into `xhigh` (the alias Qwen later
/// adopted upstream) instead of being sent verbatim.
pub fn chat_template_kwargs(effort: Option<&str>) -> Option<serde_json::Value> {
    let effort = effort?.trim().to_ascii_lowercase();
    match effort.as_str() {
        "" => None,
        "none" | "off" | "false" | "no_thinking" | "nothinking" => {
            Some(serde_json::json!({ "enable_thinking": false }))
        }
        "xhigh" | "medium" | "low" => Some(serde_json::json!({
            "enable_thinking": true,
            "reasoning_effort": effort,
        })),
        "high" | "max" => Some(serde_json::json!({
            "enable_thinking": true,
            "reasoning_effort": "xhigh",
        })),
        _ => None,
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
        let is_vl = repo.contains("-vl") || repo.contains("vision") || is_qwen3_8_multimodal(&repo);
        let is_gguf = repo.contains("gguf");

        resolved.tensor_parallel = resolved.tensor_parallel.max(1);
        resolved.enable_chunked_prefill = true;
        resolved.dtype = "bfloat16".to_string();

        // Hybrid-thinking Qwen3.5+ checkpoints open every assistant turn with
        // ` thinking`. Without a reasoning parser that block lands in
        // `message.content`, and the dataset generator — which reads only
        // `content` — spends its output budget on text it then strips. This is a
        // correctness fix, so it applies to both profiles.
        resolved.reasoning_parser = if is_qwen3 {
            Some("qwen3".to_string())
        } else {
            None
        };

        if resolved.serving_profile == ServingProfile::Optimized {
            // Vendor-published serving recipe: full native context, a larger
            // memory budget and higher concurrency. Clamped by VRAM so the same
            // profile stays valid on a 48 GB card as well as a 256 GB MI325X.
            resolved.gpu_memory_utilization = 0.90;
            resolved.max_model_len = if memory_gb >= 140.0 {
                262144
            } else if memory_gb >= 96.0 {
                131072
            } else if memory_gb >= 48.0 {
                65536
            } else {
                32768
            };
            resolved.max_num_batched_tokens = Some(16384);
            resolved.max_num_seqs = Some(if memory_gb > 0.0 && memory_gb < 48.0 {
                8
            } else {
                64
            });
        } else {
            resolved.gpu_memory_utilization = 0.80;
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
        }

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
        // Separates the model's ` thinking` block into `reasoning_content`.
        // Without it that text stays in `message.content` and eats the
        // generation budget before the answer starts.
        if let Some(parser) = self
            .reasoning_parser
            .as_ref()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
        {
            args.push(format!("--reasoning-parser {parser}"));
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

        if self.serving_profile == ServingProfile::Optimized {
            // Dataset generation sends the same prompt template for every chunk,
            // so prefix caching would be the single largest throughput win — but
            // vLLM marks it experimental for hybrid Mamba/linear-attention
            // checkpoints, and Qwen3.5+ (including the default Qwen3.8-27B) is
            // exactly that shape. Wrong KV reuse corrupts training data quietly,
            // so it stays off unless the user opts in via `extra_serve_args`.
            if !is_hybrid_linear_attention(&self.repo_id.to_lowercase()) {
                args.push("--enable-prefix-caching".to_string());
            }
            // FP8 KV cache is native on gfx942 (MI300X/MI325X) as the E4M3FNUZ
            // dialect and roughly doubles the KV pool, which is what lets the
            // raised --max-num-seqs actually stay resident.
            args.push("--kv-cache-dtype fp8".to_string());
            // Qwen3.8-27B carries a vision tower even for text-only prompts;
            // vLLM needs this to shard the encoder instead of replicating it.
            if is_qwen3_8_multimodal(&self.repo_id.to_lowercase()) {
                args.push("--mm-encoder-tp-mode data".to_string());
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
            prepare.push_str("python3 -c \"from transformers.models.auto.configuration_auto import CONFIG_MAPPING; import sys; sys.exit(0 if \\\"deepseek_v4\\\" in CONFIG_MAPPING else 1)\" || { echo [compat] installing Transformers with DeepSeek V4 support; python3 -m pip install --no-cache-dir --upgrade transformers || exit 42; }; ");
        }

        // Reconcile the environment, then resolve whatever THIS checkpoint
        // needs. The model-specific half is discovered at deploy time (config
        // metadata, requirements.txt, vLLM's architecture registry), so a model
        // the app has never seen still deploys without a code change.
        prepare.push_str(&crate::deps::teacher_heal_cmd(&self.repo_id));

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
    /// Optional control-plane override, e.g. an AMD Developer Cloud host for a
    /// tenant whose token is not a DigitalOcean personal access token. Empty
    /// means "use the standard DigitalOcean API", which also serves AMD-team
    /// tokens and the AMD Instinct MI-series plans.
    pub api_base: String,
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
            api_base: String::new(),
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

/// Docker arguments for the vLLM serving container.
///
/// Beyond the obvious device passthrough, this carries the two flags AMD lists
/// as **mandatory** for ROCm containers:
///
/// * `--cap-add=SYS_PTRACE` — ROCm JIT compilation requires ptrace. vLLM
///   compiles AITER kernels at startup, so without it the engine can die
///   mid-warmup.
/// * `--security-opt seccomp=unconfined` — ROCm's mmap variants are blocked by
///   the default seccomp profile.
///
/// Both were verified unnecessary on the MI300X test droplet, but AMD documents
/// them as required and the failure mode on a stricter host is an obscure JIT
/// crash rather than a clear error, so they ship by default.
pub fn default_docker_start_args() -> String {
    "--device=/dev/kfd --device=/dev/dri --network=host --ipc=host \
     --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
     -v /root:/root"
        .to_string()
}

/// Pinned vLLM ROCm image. Deliberately a release tag rather than `nightly`:
/// nightly changes daily, so a deploy that worked yesterday can break today
/// with no change on our side, and it forces a fresh multi-gigabyte pull even
/// when a known-good image is already on the host. v0.27.1 is verified
/// end-to-end against Qwen3.8-27B on gfx942 (vLLM >= 0.17.0 and
/// transformers >= 5.8.0 are the model's requirements).
pub const DEFAULT_VLLM_IMAGE: &str = "vllm/vllm-openai-rocm:v0.27.1";

impl Default for DockerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            container_name: "rocm-vllm".to_string(),
            image_name: DEFAULT_VLLM_IMAGE.to_string(),
            start_args: default_docker_start_args(),
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

// ── Deploy simulation ────────────────────────────────────────────────────────
//
// These exercise the same code paths the Deploy page drives, without a GPU:
// profile resolution against reported VRAM, the flags each profile emits, the
// reasoning-effort payload sent per request, and the shell-safety of the
// assembled argument string. Run with `cargo test`.
#[cfg(test)]
mod deploy_simulation {
    use super::*;

    const GB: f64 = 1024.0; // resolved_for_gpu takes MiB

    fn teacher(repo: &str, profile: ServingProfile) -> TeacherConfig {
        TeacherConfig {
            repo_id: repo.to_string(),
            auto_tune: true,
            serving_profile: profile,
            ..Default::default()
        }
    }

    fn qwen38(profile: ServingProfile) -> TeacherConfig {
        teacher("Qwen/Qwen3.8-27B", profile)
    }

    fn args(t: &TeacherConfig) -> Vec<String> {
        t.vllm_extra_args()
            .split_whitespace()
            .map(str::to_string)
            .collect()
    }

    fn has_arg(t: &TeacherConfig, needle: &str) -> bool {
        t.vllm_extra_args().contains(needle)
    }

    // ── Standard profile ────────────────────────────────────────────────────

    #[test]
    fn standard_profile_is_conservative() {
        let r = qwen38(ServingProfile::Standard).resolved_for_gpu(Some(256.0 * GB));
        assert_eq!(r.gpu_memory_utilization, 0.80, "standard keeps the 0.80 budget");
        assert_eq!(r.max_model_len, 100000, "standard caps Qwen3-VL below native");
        assert_eq!(r.max_num_batched_tokens, Some(8192));
        assert_eq!(r.max_num_seqs, Some(16));
        assert_eq!(r.dtype, "bfloat16");
    }

    #[test]
    fn standard_profile_omits_optimization_flags() {
        let t = qwen38(ServingProfile::Standard).resolved_for_gpu(Some(256.0 * GB));
        assert!(!has_arg(&t, "--enable-prefix-caching"), "prefix caching is opt-in");
        assert!(!has_arg(&t, "--kv-cache-dtype"), "FP8 KV cache is opt-in");
        assert!(!has_arg(&t, "--mm-encoder-tp-mode"));
    }

    #[test]
    fn standard_profile_still_sets_correctness_flags() {
        // The reasoning parser is a correctness fix, not an optimization, so it
        // must be present even in the conservative profile.
        let t = qwen38(ServingProfile::Standard).resolved_for_gpu(Some(256.0 * GB));
        assert!(has_arg(&t, "--reasoning-parser qwen3"));
        assert!(has_arg(&t, "--enable-auto-tool-choice"));
        assert!(has_arg(&t, "--tool-call-parser qwen3_coder"));
        assert!(has_arg(&t, "--enable-chunked-prefill"));
    }

    // ── Optimized profile ───────────────────────────────────────────────────

    #[test]
    fn optimized_profile_emits_vendor_recipe() {
        let t = qwen38(ServingProfile::Optimized).resolved_for_gpu(Some(256.0 * GB));
        assert_eq!(t.gpu_memory_utilization, 0.90);
        assert_eq!(t.max_model_len, 262144, "full native context on a 256 GB card");
        assert_eq!(t.max_num_batched_tokens, Some(16384));
        assert_eq!(t.max_num_seqs, Some(64));
        assert!(has_arg(&t, "--kv-cache-dtype fp8"));
        assert!(has_arg(&t, "--mm-encoder-tp-mode data"));
        assert!(has_arg(&t, "--reasoning-parser qwen3"));
    }

    #[test]
    fn hybrid_models_do_not_get_prefix_caching_by_default() {
        // vLLM reports prefix caching over Mamba/linear-attention layers as
        // experimental, and Qwen3.5+ (including the default Qwen3.8-27B) is
        // exactly that shape. Quietly corrupting training data would be worse
        // than generating slowly, so it stays off unless opted into.
        for repo in [
            "Qwen/Qwen3.8-27B",
            "Qwen/Qwen3.6-27B",
            "Qwen/Qwen3.5-122B-A10B",
            "ai21labs/Jamba-v2",
        ] {
            let t = teacher(repo, ServingProfile::Optimized).resolved_for_gpu(Some(256.0 * GB));
            assert!(
                !has_arg(&t, "--enable-prefix-caching"),
                "{repo} is hybrid and must not enable prefix caching by default"
            );
        }
    }

    #[test]
    fn non_hybrid_models_still_get_prefix_caching() {
        let t = teacher("meta-llama/Llama-3.1-8B-Instruct", ServingProfile::Optimized)
            .resolved_for_gpu(Some(256.0 * GB));
        assert!(
            has_arg(&t, "--enable-prefix-caching"),
            "dense attention models should keep the throughput win"
        );
    }

    #[test]
    fn user_can_force_prefix_caching_via_extra_args() {
        // The escape hatch matters: prefix caching is the biggest throughput
        // win for dataset generation, so an informed user must be able to
        // re-enable it after A/B-ing output equality themselves.
        let mut t = qwen38(ServingProfile::Optimized);
        t.extra_serve_args = Some("--enable-prefix-caching".to_string());
        let t = t.resolved_for_gpu(Some(256.0 * GB));
        assert!(has_arg(&t, "--enable-prefix-caching"));
    }

    #[test]
    fn optimized_context_is_clamped_by_vram() {
        let cases = [
            (256.0, 262144),
            (192.0, 262144),
            (140.0, 262144),
            (128.0, 131072),
            (96.0, 131072),
            (80.0, 65536),
            (48.0, 65536),
            (24.0, 32768),
        ];
        for (gb, expected) in cases {
            let t = qwen38(ServingProfile::Optimized).resolved_for_gpu(Some(gb * GB));
            assert_eq!(
                t.max_model_len, expected,
                "optimized context wrong at {gb} GB VRAM"
            );
        }
    }

    #[test]
    fn optimized_seq_count_drops_on_small_cards() {
        let small = qwen38(ServingProfile::Optimized).resolved_for_gpu(Some(24.0 * GB));
        assert_eq!(small.max_num_seqs, Some(8), "small VRAM gets fewer sequences");
        let large = qwen38(ServingProfile::Optimized).resolved_for_gpu(Some(256.0 * GB));
        assert_eq!(large.max_num_seqs, Some(64));
    }

    #[test]
    fn profiles_differ_in_the_expected_directions() {
        let std = qwen38(ServingProfile::Standard).resolved_for_gpu(Some(256.0 * GB));
        let opt = qwen38(ServingProfile::Optimized).resolved_for_gpu(Some(256.0 * GB));
        assert!(opt.max_model_len > std.max_model_len);
        assert!(opt.gpu_memory_utilization > std.gpu_memory_utilization);
        assert!(opt.max_num_seqs.unwrap() > std.max_num_seqs.unwrap());
        assert!(opt.max_num_batched_tokens.unwrap() > std.max_num_batched_tokens.unwrap());
    }

    // ── Model-family behaviour ──────────────────────────────────────────────

    #[test]
    fn multimodal_encoder_flag_only_for_qwen38() {
        // Qwen3.8-27B carries a vision tower with no "-vl" marker in its id.
        let q38 = qwen38(ServingProfile::Optimized).resolved_for_gpu(Some(256.0 * GB));
        assert!(has_arg(&q38, "--mm-encoder-tp-mode data"));

        let llama = teacher("meta-llama/Llama-3.1-8B-Instruct", ServingProfile::Optimized)
            .resolved_for_gpu(Some(256.0 * GB));
        assert!(!has_arg(&llama, "--mm-encoder-tp-mode"));
    }

    #[test]
    fn non_qwen_models_get_no_qwen_flags() {
        let t = teacher("meta-llama/Llama-3.1-8B-Instruct", ServingProfile::Optimized)
            .resolved_for_gpu(Some(256.0 * GB));
        assert!(!has_arg(&t, "--reasoning-parser"));
        assert!(!has_arg(&t, "--enable-auto-tool-choice"));
        assert!(!has_arg(&t, "--tool-call-parser"));
        assert!(t.reasoning_parser.is_none());
    }

    #[test]
    fn manual_mode_leaves_config_untouched() {
        let mut t = qwen38(ServingProfile::Optimized);
        t.auto_tune = false;
        t.max_model_len = 4096;
        t.gpu_memory_utilization = 0.42;
        let r = t.resolved_for_gpu(Some(256.0 * GB));
        assert_eq!(r.max_model_len, 4096, "manual settings must survive");
        assert_eq!(r.gpu_memory_utilization, 0.42);
    }

    #[test]
    fn custom_serve_cmd_disables_tuning() {
        let mut t = qwen38(ServingProfile::Optimized);
        t.custom_serve_cmd = Some("vllm serve my-own-thing".to_string());
        t.max_model_len = 8192;
        let r = t.resolved_for_gpu(Some(256.0 * GB));
        assert_eq!(r.max_model_len, 8192, "custom command wins over auto-tune");
    }

    #[test]
    fn unknown_vram_does_not_panic() {
        for profile in [ServingProfile::Standard, ServingProfile::Optimized] {
            let r = qwen38(profile).resolved_for_gpu(None);
            assert!(r.max_model_len > 0);
            assert!(r.max_num_seqs.unwrap() > 0);
        }
    }

    // ── Reasoning effort ────────────────────────────────────────────────────

    #[test]
    fn reasoning_effort_levels_map_to_valid_template_kwargs() {
        let cases = [
            ("xhigh", "xhigh"),
            ("medium", "medium"),
            ("low", "low"),
            ("XHIGH", "xhigh"),
            (" Medium ", "medium"),
        ];
        for (input, expected) in cases {
            let kw = chat_template_kwargs(Some(input)).expect("level should produce kwargs");
            assert_eq!(kw["reasoning_effort"], expected, "input {input:?}");
            assert_eq!(kw["enable_thinking"], true);
        }
    }

    #[test]
    fn no_thinking_disables_thinking_without_a_level() {
        for input in ["none", "off", "false", "no_thinking", "NOTHINKING"] {
            let kw = chat_template_kwargs(Some(input)).expect("should produce kwargs");
            assert_eq!(kw["enable_thinking"], false, "input {input:?}");
            assert!(
                kw.get("reasoning_effort").is_none(),
                "must not send a level alongside enable_thinking=false"
            );
        }
    }

    #[test]
    fn unsupported_levels_never_reach_the_template() {
        // Qwen3.8's chat template raises on anything outside xhigh/medium/low,
        // which vLLM reports as HTTP 500. "high" is the OpenAI spelling and is
        // folded into xhigh; junk is dropped so the model default applies.
        for input in ["high", "max", "HIGH"] {
            let kw = chat_template_kwargs(Some(input)).expect("high folds to xhigh");
            assert_eq!(kw["reasoning_effort"], "xhigh", "input {input:?}");
        }
        for input in ["bogus", "ultra", "7", ""] {
            assert!(
                chat_template_kwargs(Some(input)).is_none(),
                "input {input:?} must be dropped, not forwarded"
            );
        }
        assert!(chat_template_kwargs(None).is_none());
    }

    // ── Shell safety ────────────────────────────────────────────────────────

    #[test]
    fn extra_args_are_safe_inside_bash_lc_single_quotes() {
        // The deploy path embeds these args in `bash -lc '...'`, so a single
        // quote anywhere would terminate the wrapper and break the launch.
        for profile in [ServingProfile::Standard, ServingProfile::Optimized] {
            let t = qwen38(profile).resolved_for_gpu(Some(256.0 * GB));
            let args = t.vllm_extra_args();
            assert!(!args.contains('\''), "single quote in args: {args}");
            assert!(!args.contains('`'), "backtick in args: {args}");
            assert!(!args.contains("$("), "command substitution in args: {args}");
        }
    }

    #[test]
    fn every_emitted_flag_has_a_value() {
        let t = qwen38(ServingProfile::Optimized).resolved_for_gpu(Some(256.0 * GB));
        let toks = args(&t);
        let value_flags = [
            "--max-num-batched-tokens",
            "--max-num-seqs",
            "--reasoning-parser",
            "--tool-call-parser",
            "--kv-cache-dtype",
            "--mm-encoder-tp-mode",
        ];
        for (i, tok) in toks.iter().enumerate() {
            if value_flags.contains(&tok.as_str()) {
                assert!(
                    toks.get(i + 1).is_some_and(|v| !v.starts_with("--")),
                    "{tok} is missing its value in: {}",
                    t.vllm_extra_args()
                );
            }
        }
    }

    #[test]
    fn full_serve_command_assembles_cleanly() {
        // Mirrors the format string the deploy path uses, so a regression in
        // any single flag shows up as a malformed command here.
        for (profile, effort) in [
            (ServingProfile::Standard, "xhigh"),
            (ServingProfile::Optimized, "none"),
        ] {
            let mut t = qwen38(profile);
            t.reasoning_effort = Some(effort.to_string());
            let t = t.resolved_for_gpu(Some(256.0 * GB));
            let cmd = format!(
                "cd /app && vllm serve {model} --port {port} --host 0.0.0.0 \
                 --max-model-len {mml} --dtype {dtype} --download-dir /root/hf-cache \
                 --tensor-parallel-size {tp} --gpu-memory-utilization {gpu_mem} {extra}",
                model = t.repo_id,
                port = t.vllm_port,
                mml = t.max_model_len,
                dtype = t.dtype,
                tp = t.tensor_parallel,
                gpu_mem = t.gpu_memory_utilization,
                extra = t.vllm_extra_args(),
            );
            assert!(cmd.starts_with("cd /app && vllm serve Qwen/Qwen3.8-27B"));
            assert!(cmd.contains(&format!("--max-model-len {}", t.max_model_len)));
            assert!(cmd.contains(&format!("--gpu-memory-utilization {}", t.gpu_memory_utilization)));
            assert!(!cmd.contains("  "), "double space means an empty placeholder: {cmd}");
            assert!(!cmd.contains("None"), "unset Option leaked into command: {cmd}");
        }
    }

    // ── Fresh-install defaults ──────────────────────────────────────────────

    #[test]
    fn defaults_carry_no_credentials() {
        // A new install must start empty: no tokens, no keys, no hosts.
        let cfg = AppConfig::default();
        assert!(cfg.hf_token.is_none());
        assert!(cfg.ssh.host.is_empty());
        assert!(cfg.ssh.private_key.is_none());
        assert!(cfg.ssh.private_key_path.is_none());
        assert!(cfg.ssh.password.is_none());
        assert!(cfg.digital_ocean.api_key.is_empty());
        assert!(cfg.digital_ocean.project_id.is_empty());
        assert!(cfg.digital_ocean.ssh_keys.is_empty());
        assert!(cfg.qdrant.api_key.is_empty());
        assert!(cfg.qdrant.endpoint.is_empty());
        assert!(cfg.ai_agent.is_none());
        assert!(cfg.embedding.is_none());
    }

    #[test]
    fn teacher_default_targets_qwen38() {
        let t = TeacherConfig::default();
        assert_eq!(t.repo_id, "Qwen/Qwen3.8-27B");
        assert_eq!(t.serving_profile, ServingProfile::Standard);
        assert!(t.reasoning_effort.is_none());
        assert!(t.custom_serve_cmd.is_none());
    }

    #[test]
    fn docker_image_is_pinned_not_nightly() {
        // `nightly` changes daily, so a deploy that worked yesterday can break
        // today with no change on our side — and it forces a fresh multi-GB
        // pull even when a known-good image is already on the host.
        let d = DockerConfig::default();
        assert!(
            !d.image_name.contains("nightly"),
            "image must be a pinned release tag, got {}",
            d.image_name
        );
        assert_eq!(d.image_name, DEFAULT_VLLM_IMAGE);
        assert!(
            d.image_name.contains("v0.27.1"),
            "v0.27.1 is the version verified against Qwen3.8-27B on gfx942"
        );
    }

    #[test]
    fn docker_start_args_include_amd_mandatory_flags() {
        // AMD lists both of these as mandatory for ROCm containers: JIT
        // compilation needs ptrace, and ROCm's mmap variants are blocked by the
        // default seccomp profile. Verified unnecessary on the MI300X test
        // droplet, but the failure mode elsewhere is an obscure JIT crash.
        let d = DockerConfig::default();
        for required in [
            "--device=/dev/kfd",
            "--device=/dev/dri",
            "--group-add video",
            "--cap-add=SYS_PTRACE",
            "--security-opt seccomp=unconfined",
            "--ipc=host",
            "--network=host",
        ] {
            assert!(
                d.start_args.contains(required),
                "start_args missing {required}: {}",
                d.start_args
            );
        }
    }

    #[test]
    fn deploy_page_serving_carries_the_universal_heal() {
        // The Deploy page serves a model for production through exactly this
        // prepare step, so production serving must heal like the wizard does.
        for repo in [
            "Qwen/Qwen3.8-27B",
            "meta-llama/Llama-3.1-8B-Instruct",
            "some-org/model-the-app-has-never-seen",
        ] {
            let t = teacher(repo, ServingProfile::Optimized);
            let prep = t.vllm_runtime_prepare_cmd();
            assert!(
                prep.contains("reconciling environment"),
                "{repo}: deploy must reconcile environment pins"
            );
            assert!(
                prep.contains("base64 -d"),
                "{repo}: deploy must run the model-agnostic resolver"
            );
            assert!(
                prep.contains("pip check"),
                "{repo}: deploy must report residual conflicts"
            );
            assert!(
                !prep.contains('\n'),
                "{repo}: prepare must stay a single logical line"
            );
        }
    }

    #[test]
    fn deploy_page_flags_reach_the_serve_command() {
        // Every Deploy-page tuning knob is threaded through extra_serve_args;
        // if one silently stopped being emitted, the UI would lie about what
        // the server is running.
        let mut t = qwen38(ServingProfile::Optimized);
        t.extra_serve_args = Some(
            "--block-size 32 --swap-space 16 --scheduling-policy priority \
             --preemption-mode swap --cpu-offload-gb 4 --quantization awq"
                .to_string(),
        );
        let r = t.resolved_for_gpu(Some(256.0 * GB));
        for flag in [
            "--block-size 32",
            "--swap-space 16",
            "--scheduling-policy priority",
            "--preemption-mode swap",
            "--cpu-offload-gb 4",
            "--quantization awq",
        ] {
            assert!(
                has_arg(&r, flag),
                "deploy setting {flag} did not reach the command: {}",
                r.vllm_extra_args()
            );
        }
    }

    #[test]
    fn docker_start_args_avoid_group_add_render() {
        // `--group-add render` hard-fails `docker run` on hosts without a
        // `render` group, and AMD only lists it as needed "on many hosts".
        // Failing to start at all is worse than missing the group.
        let d = DockerConfig::default();
        assert!(
            !d.start_args.contains("--group-add render"),
            "render group would break hosts that lack it"
        );
    }

    #[test]
    fn config_roundtrips_without_secrets_leaking_into_unknown_fields() {
        // Old config.json files (written before servingProfile/reasoningEffort
        // existed) must still parse, with the new fields defaulted.
        let legacy = r#"{
            "teacher": { "repoId": "Qwen/Qwen3.8-27B", "autoTune": true },
            "ssh": { "host": "" }
        }"#;
        let cfg: AppConfig = serde_json::from_str(legacy).expect("legacy config must parse");
        assert_eq!(cfg.teacher.repo_id, "Qwen/Qwen3.8-27B");
        assert_eq!(cfg.teacher.serving_profile, ServingProfile::Standard);
        assert!(cfg.teacher.reasoning_effort.is_none());
        assert!(cfg.teacher.reasoning_parser.is_none());
    }
}
