use crate::config::{DockerConfig, EmbedderConfig, PaddleOcrConfig};
use crate::error::{AppError, Result};
use crate::ssh::SshSession;

fn sh_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\"'\"'"))
}

pub async fn ensure_qdrant(
    session: &SshSession,
    _cfg: &DockerConfig,
    qdrant_port: u16,
    data_dir: &str,
) -> Result<()> {
    // 1. Probe if Qdrant is already running and healthy on the host
    let probe = format!(
        "curl -s -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{}/collections 2>/dev/null || echo 000",
        qdrant_port
    );
    if let Ok(r) = session.exec_blocking(&probe).await {
        if r.stdout.trim() == "200" {
            return Ok(());
        }
    }

    // 2. Qdrant is not running or not healthy, so clean up any existing container
    // of the same name to prevent "name already in use" errors.
    let cleanup_cmd = "docker stop qdrant 2>/dev/null; docker rm qdrant 2>/dev/null; true";
    let _ = session.exec_blocking(cleanup_cmd).await;

    // 3. Start the Qdrant container directly on the host (with mapped port and volume)
    let run_cmd = format!(
        "mkdir -p {data_dir}/qdrant_storage && \
         docker run -d \
          --name qdrant \
          --restart unless-stopped \
          --ulimit nofile=65535:65535 \
          -p {port}:{port} \
          -v {data_dir}/qdrant_storage:/qdrant/storage \
          qdrant/qdrant:v1.7.4",
        port = qdrant_port,
        data_dir = data_dir,
    );
    let r = session.exec_blocking(&run_cmd).await?;
    if r.exit_code != 0 {
        return Err(AppError::pipeline(format!(
            "Failed to start Qdrant docker container (exit {}): {}",
            r.exit_code, r.stderr
        )));
    }

    // 4. Probe loop to wait for Qdrant to start responding
    for _ in 1..20 {
        if let Ok(r) = session.exec_blocking(&probe).await {
            if r.stdout.trim() == "200" {
                return Ok(());
            }
        }
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
    }

    Err(AppError::pipeline("Qdrant failed to start within 60s"))
}

pub fn wrap_docker_cmd(cmd: &str, container: &str) -> String {
    format!("docker exec -i {} bash -lc {}", container, sh_quote(cmd))
}

pub fn wrap_docker_cmd_detached(cmd: &str, container: &str) -> String {
    format!(
        "docker exec -d {} bash -lc {} >/dev/null 2>&1 && echo EMBEDDER_LAUNCHED",
        container,
        sh_quote(cmd)
    )
}

pub async fn health_check_embedder(
    session: &SshSession,
    cfg: &DockerConfig,
    host: &str,
    port: u16,
) -> Result<Option<String>> {
    let url = format!("http://{}:{}/v1/models", host, port);
    let probe = format!(
        "curl -s -o /dev/null -w '%{{http_code}}' '{}' 2>/dev/null || echo 000",
        url
    );
    let cmd = if cfg.enabled {
        wrap_docker_cmd(&probe, &cfg.container_name)
    } else {
        probe
    };
    let r = session.exec_blocking(&cmd).await?;
    if r.stdout.trim() == "200" {
        Ok(Some(host.to_string()))
    } else {
        Ok(None)
    }
}

/// Fire the embedder's vLLM process in the background (detached) and return as
/// soon as the launch command is accepted — WITHOUT waiting for the model to
/// finish loading. Use this when you must not block the caller (e.g. the
/// pipeline, where the teacher is already serving and the embedder only needs to
/// be ready by the time the first topic is embedded). Pair with
/// `wait_for_embedder` if you later need to confirm readiness.
pub async fn launch_embedder(
    session: &SshSession,
    cfg: &DockerConfig,
    embedder: &EmbedderConfig,
    hf_token: Option<&str>,
    gpu_memory_utilization: Option<f64>,
) -> Result<()> {
    let effective_gpu_mem = gpu_memory_utilization.unwrap_or(0.084);
    let max_num_seqs = " --max-num-seqs 100";

    let port = embedder.port;

    let pkill_body = format!(
        "pkill -f '[v]llm.*--port {port}' 2>/dev/null; \
         pkill -f '[v]llm.entrypoints' 2>/dev/null; \
         pkill -f 'sglang.*--port {port}' 2>/dev/null; \
         sleep 1; \
         pkill -9 -f '[v]llm.*--port {port}' 2>/dev/null; \
         pkill -9 -f '[v]llm.entrypoints' 2>/dev/null; \
         pkill -9 -f 'sglang.*--port {port}' 2>/dev/null; \
         (command -v fuser >/dev/null 2>&1 && fuser -k {port}/tcp 2>/dev/null) || true; \
         (command -v ss >/dev/null 2>&1 && ss -ltnp 2>/dev/null | awk '/:{port} /{{print $0}}' | grep -oE 'pid=[0-9]+' | cut -d= -f2 | xargs -r kill -9 2>/dev/null) || true; \
         for i in 1 2 3 4 5; do \
             if command -v ss >/dev/null 2>&1; then ss -ltn 2>/dev/null | awk '{{print $4}}' | grep -qE ':{port}$' || break; fi; \
             sleep 1; \
         done; \
         true",
        port = port,
    );

    let _ = session.exec_blocking(&pkill_body).await;
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    if cfg.enabled {
        let inner = format!("docker ps --format '{{{{.Names}}}}'");
        if let Ok(r) = session.exec_blocking(&inner).await {
            for cname in r
                .stdout
                .lines()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
            {
                let inner_kill = wrap_docker_cmd(&pkill_body, &cname);
                let _ = session.exec_blocking(&inner_kill).await;
            }
        }
    }

    let env = {
        let mut e = "export PYTHONUNBUFFERED=1; \
                     export VLLM_SLEEP_WHEN_IDLE=1; \
                     export VLLM_USE_DEEP_GEMM=0; \
                     export GLOO_SOCKET_IFNAME=lo; \
                     export NCCL_SOCKET_IFNAME=lo; \
                     export VLLM_HOST_IP=127.0.0.1; "
            .to_string();
        if let Some(tok) = hf_token.filter(|s| !s.is_empty()) {
            e.push_str(&format!(
                "export HF_TOKEN={} HUGGING_FACE_HUB_TOKEN={}; ",
                tok, tok
            ));
        }
        e
    };

    let model_id = sh_quote(&embedder.model_id);
    let model_slug = embedder.model_id.split('/').last().unwrap_or(&embedder.model_id);
    let compat_check = format!(
        "python3 -c \"\
           import json,urllib.request,sys; \
           url='https://huggingface.co/{repo}/raw/main/config.json'; \
           req=urllib.request.Request(url, headers={{'User-Agent':'fine-tune'}}); \
           cfg=json.load(urllib.request.urlopen(req, timeout=15)); \
           mt=cfg.get('model_type',''); \
           from transformers.models.auto.configuration_auto import CONFIG_MAPPING; \
           if mt and mt not in CONFIG_MAPPING: \
             print(f'[compat] transformers does not recognize model_type={{mt!r}} — upgrading from source'); sys.exit(1); \
         \" 2>/dev/null && echo '[compat] transformers OK' || {{ \
           echo '[compat] upgrading transformers from source for {slug}...'; \
           python3 -m pip install --no-cache-dir --upgrade git+https://github.com/huggingface/transformers.git || true; \
         }}; \
         python3 -c \"import site,os; p=os.path.join(site.getsitepackages()[0],'zz_finetune_hetero_fix.pth'); open(p,'w').write('import transformers.configuration_utils as _tc; _tc.PretrainedConfig.allow_global_per_layer_attribute_access=True\\n'); print('[compat] heterogeneity fix installed')\" 2>/dev/null; ",
        repo = embedder.model_id,
        slug = model_slug,
    );
    let serve_cmd = format!(
        "cd /root && {env} \
         MODEL_ID={model}; \
         python3 -c 'import torchvision' 2>&1 | grep -E -q 'nms|operator' && python3 -m pip uninstall -y torchvision || true; \
         {compat} \
         run_vllm() {{ \
           if command -v vllm >/dev/null 2>&1; then vllm serve \"$MODEL_ID\" \"$@\"; \
           elif python3 -c 'import vllm' >/dev/null 2>&1; then python3 -m vllm.entrypoints.openai.api_server --model \"$MODEL_ID\" \"$@\"; \
           elif python -c 'import vllm' >/dev/null 2>&1; then python -m vllm.entrypoints.openai.api_server --model \"$MODEL_ID\" \"$@\"; \
           else echo 'vLLM is not installed or not on PATH in this runtime' >&2; return 127; fi; \
         }}; \
         run_vllm \
           --runner pooling \
           --port {port} \
           --host 0.0.0.0 \
           --max-model-len 32768 \
           --dtype half \
           --download-dir /root/hf-cache \
           --tensor-parallel-size 1 \
           --gpu-memory-utilization {gpu_mem:.3}{max_seqs} \
         > /root/embedder_{port}.log 2>&1 || \
         run_vllm \
           --task embed \
           --port {port} \
           --host 0.0.0.0 \
           --max-model-len 32768 \
           --dtype half \
           --download-dir /root/hf-cache \
           --tensor-parallel-size 1 \
           --gpu-memory-utilization {gpu_mem:.3}{max_seqs} \
         > /root/embedder_{port}.log 2>&1",
        env = env,
        model = model_id,
        port = port,
        gpu_mem = effective_gpu_mem,
        max_seqs = max_num_seqs,
        compat = compat_check,
    );

    let boot_cmd = if cfg.enabled {
        let inner = format!(
            "mkdir -p /root/hf-cache /root; \
             truncate -s 0 /root/embedder_{port}.log 2>/dev/null || rm -f /root/embedder_{port}.log; \
             {cmd} > /root/embedder_{port}.log 2>&1",
            port = port,
            cmd = serve_cmd,
        );
        wrap_docker_cmd_detached(&inner, &cfg.container_name)
    } else {
        format!(
            "mkdir -p /root/hf-cache /root; \
             truncate -s 0 /root/embedder_{port}.log 2>/dev/null || rm -f /root/embedder_{port}.log; \
             nohup bash -lc '{cmd} > /root/embedder_{port}.log 2>&1' < /dev/null & \
             echo EMBEDDER_LAUNCHED",
            port = port,
            cmd = serve_cmd,
        )
    };

    let r = session.exec_blocking(&boot_cmd).await?;
    if r.exit_code != 0 && !r.stdout.contains("LAUNCHED") {
        return Err(AppError::pipeline(format!(
            "embedder boot failed (exit {}): {}",
            r.exit_code, r.stderr
        )));
    }
    Ok(())
}

/// Poll an already-launched embedder until it serves `/v1/models` (200) or the
/// timeout elapses, streaming new log lines to `on_log`. `timeout_secs` bounds
/// the wait; the pipeline uses a short bound so a slow embedder load never
/// blocks dataset generation (the retrieve phase falls back to keyword scroll).
pub async fn wait_for_embedder(
    session: &SshSession,
    cfg: &DockerConfig,
    embedder: &EmbedderConfig,
    timeout_secs: u64,
    on_log: Option<&(dyn Fn(String) + Send + Sync)>,
) -> Result<String> {
    let port = embedder.port;
    let started = std::time::Instant::now();
    let timeout = std::time::Duration::from_secs(timeout_secs);
    let mut printed_lines = 0;

    loop {
        if started.elapsed() > timeout {
            return Err(AppError::pipeline(format!(
                "embedder '{}' not ready within {}s on port {}",
                embedder.model_id, timeout_secs, port
            )));
        }

        // Stream new log lines of the embedder process to the UI callback
        let count_cmd = if cfg.enabled {
            wrap_docker_cmd(
                &format!("wc -l /root/embedder_{}.log 2>/dev/null", port),
                &cfg.container_name,
            )
        } else {
            format!("wc -l /root/embedder_{}.log 2>/dev/null", port)
        };

        if let Ok(count_res) = session.exec_blocking(&count_cmd).await {
            let first_token = count_res
                .stdout
                .trim()
                .split_whitespace()
                .next()
                .unwrap_or("0");
            if let Ok(total_lines) = first_token.parse::<usize>() {
                if total_lines > printed_lines {
                    let read_cmd = if cfg.enabled {
                        wrap_docker_cmd(
                            &format!("tail -n +{} /root/embedder_{}.log", printed_lines + 1, port),
                            &cfg.container_name,
                        )
                    } else {
                        format!("tail -n +{} /root/embedder_{}.log", printed_lines + 1, port)
                    };

                    if let Ok(read_res) = session.exec_blocking(&read_cmd).await {
                        for line in read_res.stdout.lines() {
                            let trimmed = line.trim();
                            if !trimmed.is_empty() {
                                if let Some(ref cb) = on_log {
                                    cb(format!("[embedder] {}\n", trimmed));
                                }
                            }
                        }
                    }
                    printed_lines = total_lines;
                }
            }
        }

        let url = format!("http://127.0.0.1:{}/v1/models", port);
        let probe = format!(
            "curl -s -o /dev/null -w '%{{http_code}}' '{}' 2>/dev/null || echo 000",
            url
        );
        let cmd = if cfg.enabled {
            wrap_docker_cmd(&probe, &cfg.container_name)
        } else {
            probe
        };
        if let Ok(r) = session.exec_blocking(&cmd).await {
            if r.stdout.trim() == "200" {
                return Ok("127.0.0.1".to_string());
            }
        }
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
    }
}

/// Launch the embedder and block until it is ready (200 on `/v1/models`) or
/// the 20-minute timeout elapses. This is the original blocking behaviour,
/// now expressed as `launch_embedder` + `wait_for_embedder` so callers that
/// must not block (the pipeline) can use the two pieces independently.
pub async fn boot_embedder(
    session: &SshSession,
    cfg: &DockerConfig,
    embedder: &EmbedderConfig,
    hf_token: Option<&str>,
    _gpu_memory_utilization: f32,
    on_log: Option<&(dyn Fn(String) + Send + Sync)>,
) -> Result<String> {
    launch_embedder(
        session,
        cfg,
        embedder,
        hf_token,
        Some(_gpu_memory_utilization as f64),
    )
    .await?;
    wait_for_embedder(session, cfg, embedder, 20 * 60, on_log).await
}

pub async fn embed_text(
    api_url: &str,
    model_id: &str,
    text: &str,
) -> crate::error::Result<Vec<f32>> {
    use crate::ingest::{normalize_vector, MAX_CHARS_PER_EMBED};
    let input: String = text.chars().take(MAX_CHARS_PER_EMBED).collect();
    let body = serde_json::json!({
        "model": if model_id.is_empty() { "default" } else { model_id },
        "input": input
    });
    let url = format!("{}/v1/embeddings", api_url.trim_end_matches('/'));
    static CLIENT: std::sync::OnceLock<reqwest::Client> = std::sync::OnceLock::new();
    let client = CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(90))
            .build()
            .unwrap()
    });
    let res =
        client.post(&url).json(&body).send().await.map_err(|e| {
            crate::error::AppError::pipeline(format!("embed request failed: {}", e))
        })?;
    if !res.status().is_success() {
        let s = res.status();
        let txt = res.text().await.unwrap_or_default();
        return Err(crate::error::AppError::pipeline(format!(
            "embed http {s}: {txt}"
        )));
    }
    let v: serde_json::Value = res
        .json()
        .await
        .map_err(|e| crate::error::AppError::pipeline(format!("embed JSON parse failed: {}", e)))?;
    let arr = v
        .get("data")
        .and_then(|d| d.get(0))
        .and_then(|d0| d0.get("embedding"))
        .and_then(|e| e.as_array())
        .ok_or_else(|| crate::error::AppError::pipeline("no data[0].embedding in response"))?;
    let mut vec: Vec<f32> = arr
        .iter()
        .filter_map(|x| x.as_f64().map(|f| f as f32))
        .collect();
    if vec.is_empty() {
        return Err(crate::error::AppError::pipeline("empty embedding vector"));
    }
    normalize_vector(&mut vec);
    Ok(vec)
}

pub async fn health_check_paddleocr(
    session: &SshSession,
    _cfg: &DockerConfig,
    port: u16,
) -> Result<bool> {
    let probe = format!(
        "curl -s -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{}/v1/models 2>/dev/null || echo 000",
        port
    );
    let r = session.exec_blocking(&probe).await?;
    Ok(r.stdout.trim() == "200")
}

pub async fn boot_paddleocr(
    session: &SshSession,
    cfg: &DockerConfig,
    paddle: &PaddleOcrConfig,
    on_log: Option<&(dyn Fn(String) + Send + Sync)>,
) -> Result<String> {
    let port = paddle.port;
    let container_name = "paddleocr-vl";

    let already = health_check_paddleocr(session, cfg, port)
        .await
        .unwrap_or(false);
    if already {
        if let Some(ref cb) = on_log {
            cb("[paddleocr] already running\n".to_string());
        }
        return Ok("127.0.0.1".to_string());
    }
    if let Some(ref cb) = on_log {
        cb(format!(
            "[paddleocr] booting PaddleOCR-VL on port {} (image: {})\n",
            port, paddle.docker_image
        ));
    }

    let cleanup = format!(
        "docker stop {} 2>/dev/null; docker rm {} 2>/dev/null; true",
        container_name, container_name
    );
    let _ = session.exec_blocking(&cleanup).await;

    let vllm_cfg_content = r#"tensor_parallel_size: 1
gpu_memory_utilization: 0.45
dtype: float16
max_num_seqs: 8
"#;
    let home = "/root";

    if let Some(ref cb) = on_log {
        cb("[paddleocr] uploading vLLM config...\n".to_string());
    }
    session
        .write_file("/root/vllm_config.yml", vllm_cfg_content)
        .await
        .map_err(|e| AppError::pipeline(format!("write vllm_config.yml to GPU server: {}", e)))?;

    // Pull the image explicitly (with up to 3 retries) before docker run.
    // This avoids the cryptic "exit -1" that occurs when docker run is killed
    // mid-pull by the OOM killer or a network interruption.
    if let Some(ref cb) = on_log {
        cb(format!(
            "[paddleocr] pulling image {} (this may take several minutes)...\n",
            paddle.docker_image
        ));
    }
    let pull_cmd = format!("docker pull {}", paddle.docker_image);
    let mut pull_ok = false;
    for attempt in 1u8..=3 {
        if let Some(ref cb) = on_log {
            if attempt > 1 {
                cb(format!("[paddleocr] pull attempt {}...\n", attempt));
            }
        }
        match session.exec_blocking(&pull_cmd).await {
            Ok(r) if r.exit_code == 0 => {
                pull_ok = true;
                break;
            }
            Ok(r) => {
                if let Some(ref cb) = on_log {
                    cb(format!(
                        "[paddleocr] pull attempt {} failed (exit {}): {}\n",
                        attempt,
                        r.exit_code,
                        r.stderr.trim()
                    ));
                }
            }
            Err(e) => {
                if let Some(ref cb) = on_log {
                    cb(format!(
                        "[paddleocr] pull attempt {} error: {}\n",
                        attempt, e
                    ));
                }
            }
        }
        if attempt < 3 {
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        }
    }
    if !pull_ok {
        return Err(AppError::pipeline(format!(
            "PaddleOCR image pull failed after 3 attempts: {}. \
             Check GPU server internet access or set a reachable docker_image in Settings.",
            paddle.docker_image
        )));
    }

    if let Some(ref cb) = on_log {
        cb("[paddleocr] image ready, starting container...\n".to_string());
    }

    let run_cmd = format!(
        "docker run -d \
         --name {} \
         --restart=no \
         --user root \
         --device=/dev/kfd \
         --device=/dev/dri \
         --security-opt seccomp=unconfined \
         --cap-add=SYS_PTRACE \
         --group-add video \
         --shm-size 64g \
         --ipc=host \
         --network host \
         -e GLOO_SOCKET_IFNAME=eth0 \
         -e NCCL_SOCKET_IFNAME=eth0 \
         -v {}/vllm_config.yml:/home/paddleocr/vllm_config.yml:ro \
         {} \
         paddleocr genai_server \
           --model_name {} \
           --host 0.0.0.0 \
           --port {} \
           --backend vllm \
           --backend_config /home/paddleocr/vllm_config.yml",
        container_name, home, paddle.docker_image, paddle.model_name, port,
    );

    let r = session.exec_blocking(&run_cmd).await?;
    if r.exit_code != 0 {
        return Err(AppError::pipeline(format!(
            "PaddleOCR docker run failed (exit {}): {}",
            r.exit_code, r.stderr
        )));
    }

    if let Some(ref cb) = on_log {
        cb("[paddleocr] container started, waiting for init...\n".to_string());
    }
    tokio::time::sleep(std::time::Duration::from_secs(5)).await;

    let started = std::time::Instant::now();
    let timeout = std::time::Duration::from_secs(20 * 60);

    loop {
        if started.elapsed() > timeout {
            return Err(AppError::pipeline(format!(
                "PaddleOCR boot timeout (20 min) on port {}",
                port
            )));
        }

        if let Some(ref cb) = on_log {
            let elapsed = started.elapsed().as_secs();
            cb(format!(
                "[paddleocr] waiting for vLLM ready... ({}s elapsed)\n",
                elapsed
            ));
        }

        if health_check_paddleocr(session, cfg, port)
            .await
            .unwrap_or(false)
        {
            if let Some(ref cb) = on_log {
                cb("[paddleocr] vLLM server ready!\n".to_string());
            }
            return Ok("127.0.0.1".to_string());
        }

        tokio::time::sleep(std::time::Duration::from_secs(10)).await;
    }
}
