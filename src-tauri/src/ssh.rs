use crate::config::SshConfig;
use crate::error::{AppError, Result};
use russh::client::{Config as RusshConfig, Handle, Handler};
use russh::keys::*;
use russh::*;
use serde::Serialize;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::mpsc;

// --- Handler ---------------------------------------------------------------

#[derive(Clone)]
struct Client;

#[async_trait::async_trait]
impl Handler for Client {
    type Error = russh::Error;

    async fn check_server_key(
        &mut self,
        _server_public_key: &russh_keys::key::PublicKey,
    ) -> std::result::Result<bool, Self::Error> {
        // TOFU: we do not pin host keys yet. The user is connecting to their own droplet.
        Ok(true)
    }
}

#[derive(Clone)]
pub struct SshSession {
    handle: Arc<Handle<Client>>,
}

impl SshSession {
    pub async fn connect(cfg: &SshConfig) -> Result<Self> {
        // --- Sanitize host string ---------------------------------------
        // Strip scheme (ssh://, https:// etc.), surrounding whitespace,
        // trailing slashes/paths, and any inline `user@` or `:port`. This
        // prevents Windows WSAEADDRNOTAVAIL (os error 10049) caused by
        // garbage being passed to the resolver.
        let mut host = cfg.host.trim().to_string();
        if host.is_empty() {
            return Err(AppError::ssh("host is empty"));
        }
        if let Some(idx) = host.find("://") {
            host = host[idx + 3..].to_string();
        }
        if let Some(idx) = host.find('@') {
            host = host[idx + 1..].to_string();
        }
        // strip path/query if user pasted a URL
        if let Some(idx) = host.find('/') {
            host = host[..idx].to_string();
        }
        // strip inline port (we use cfg.port instead)
        // careful with IPv6 literals `[::1]` — only split on `:` if not bracketed
        if !host.starts_with('[') {
            if let Some(idx) = host.find(':') {
                host = host[..idx].to_string();
            }
        }
        let host = host.trim().to_string();
        if host.is_empty() {
            return Err(AppError::ssh("host is empty after sanitization"));
        }
        let port = if cfg.port == 0 { 22 } else { cfg.port };

        // --- Resolve to a concrete SocketAddr ---------------------------
        // tokio's resolver returns a useful error if the name is bogus,
        // unlike russh's downstream connect which surfaces a raw winsock code.
        let addrs: Vec<std::net::SocketAddr> =
            tokio::net::lookup_host((host.as_str(), port))
                .await
                .map_err(|e| {
                    AppError::ssh(format!(
                        "cannot resolve host `{host}:{port}`: {e} \
                         (check the host field — no http://, no trailing /, no spaces)"
                    ))
                })?
                .collect();
        if addrs.is_empty() {
            return Err(AppError::ssh(format!(
                "DNS returned no addresses for `{host}:{port}`"
            )));
        }

        let conf = Arc::new(RusshConfig {
            inactivity_timeout: Some(std::time::Duration::from_secs(60 * 60)),
            // Send SSH keepalives every 15 seconds so the DigitalOcean SSH
            // daemon doesn't drop the connection between pipeline stages.
            // Without this, the server's ClientAliveInterval kicks in and
            // sends a disconnect, causing "ssh error: Disconnected" mid-run.
            keepalive_interval: Some(std::time::Duration::from_secs(15)),
            keepalive_max: 5,
            ..Default::default()
        });

        // Try each resolved address (IPv4 first if available) so an
        // unreachable IPv6 entry doesn't kill the whole connect.
        let mut sorted = addrs.clone();
        sorted.sort_by_key(|a| if a.is_ipv4() { 0 } else { 1 });

        // Wrap the entire TCP connect loop in a 30-second timeout so a
        // stalled host doesn't hang the pipeline at "connecting to droplet"
        // forever.  russh's `inactivity_timeout` only fires *after* the SSH
        // session is established, so we need an outer guard.
        let connect_timeout = std::time::Duration::from_secs(30);

        let connect_result = tokio::time::timeout(connect_timeout, async {
            let mut last_err: Option<russh::Error> = None;
            let mut handle = None;
            for addr in &sorted {
                match client::connect(conf.clone(), *addr, Client).await {
                    Ok(h) => {
                        handle = Some(h);
                        break;
                    }
                    Err(e) => {
                        last_err = Some(e);
                    }
                }
            }
            (handle, last_err)
        })
        .await;

        let (maybe_handle, last_err) = match connect_result {
            Ok(pair) => pair,
            Err(_elapsed) => {
                return Err(AppError::ssh(format!(
                    "tcp connect to `{host}:{port}` timed out after {connect_timeout:?} \
                     (tried {} address(es)). Check firewall, IP, and network connectivity.",
                    sorted.len()
                )));
            }
        };

        let mut handle = match maybe_handle {
            Some(h) => h,
            None => {
                let msg = last_err
                    .map(|e| e.to_string())
                    .unwrap_or_else(|| "unknown".to_string());
                return Err(AppError::ssh(format!(
                    "tcp connect to `{host}:{port}` failed (tried {} address(es)): {msg}",
                    sorted.len()
                )));
            }
        };

        // Auth: prefer private key (path), then raw PEM, then password.
        let authed = if let Some(path) = cfg.private_key_path.as_ref().filter(|s| !s.is_empty()) {
            let kp = load_secret_key(path, None)
                .map_err(|e| AppError::ssh(format!("load key {path}: {e}")))?;
            handle
                .authenticate_publickey(&cfg.username, Arc::new(kp))
                .await?
        } else if let Some(pem) = cfg.private_key.as_ref().filter(|s| !s.is_empty()) {
            let pem_norm = normalize_pem(pem);

            // Only catch the two unambiguous "wrong file" mistakes — anything
            // else, let russh's parser try and report its real error.
            if pem_norm.starts_with("PuTTY-User-Key-File-") {
                return Err(AppError::ssh(
                    "this looks like a PuTTY .ppk key — convert it to OpenSSH first \
                     (PuTTYgen → Conversions → Export OpenSSH key) and paste THAT content",
                ));
            }
            // A public key starts with `ssh-rsa AAAA…` / `ssh-ed25519 AAAA…` /
            // `ecdsa-sha2-… AAAA…` and contains no PEM block.
            let looks_like_public = {
                let head = pem_norm.split_ascii_whitespace().next().unwrap_or("");
                matches!(
                    head,
                    "ssh-rsa"
                        | "ssh-dss"
                        | "ssh-ed25519"
                        | "ecdsa-sha2-nistp256"
                        | "ecdsa-sha2-nistp384"
                        | "ecdsa-sha2-nistp521"
                        | "sk-ssh-ed25519@openssh.com"
                        | "sk-ecdsa-sha2-nistp256@openssh.com"
                ) && !pem_norm.contains("-----BEGIN")
            };
            if looks_like_public {
                return Err(AppError::ssh(
                    "this looks like a *public* key (id_rsa.pub / id_ed25519.pub) — \
                     paste the matching PRIVATE key file instead (id_rsa, id_ed25519, etc.)",
                ));
            }

            let kp = decode_secret_key(&pem_norm, None).map_err(|e| {
                AppError::ssh(format!(
                    "decode key: {e} — common causes: passphrase-encrypted key \
                     (not yet supported), truncated paste with missing BEGIN/END \
                     lines, or stray characters."
                ))
            })?;
            handle
                .authenticate_publickey(&cfg.username, Arc::new(kp))
                .await?
        } else if let Some(pw) = cfg.password.as_ref().filter(|s| !s.is_empty()) {
            handle.authenticate_password(&cfg.username, pw).await?
        } else {
            return Err(AppError::ssh(
                "no auth method provided (need private key or password)",
            ));
        };

        if !authed {
            return Err(AppError::ssh("authentication failed"));
        }

        Ok(SshSession { handle: Arc::new(handle) })
    }

    /// Run a command and collect stdout/stderr fully.
    pub async fn exec_blocking(&self, command: &str) -> Result<ExecResult> {
        let mut channel = self.handle.channel_open_session().await?;
        channel.exec(true, command).await?;

        let mut stdout = Vec::<u8>::new();
        let mut stderr = Vec::<u8>::new();
        let mut exit_code: Option<u32> = None;

        loop {
            let Some(msg) = channel.wait().await else {
                break;
            };
            match msg {
                ChannelMsg::Data { data } => stdout.extend_from_slice(&data),
                ChannelMsg::ExtendedData { data, ext } if ext == 1 => {
                    stderr.extend_from_slice(&data)
                }
                ChannelMsg::ExitStatus { exit_status } => exit_code = Some(exit_status),
                ChannelMsg::Eof => {}
                _ => {}
            }
        }
        let _ = channel.close().await;
        Ok(ExecResult {
            stdout: String::from_utf8_lossy(&stdout).into_owned(),
            stderr: String::from_utf8_lossy(&stderr).into_owned(),
            exit_code: exit_code.map(|c| c as i32).unwrap_or(-1),
        })
    }

    /// Run a command and collect stdout fully while streaming stderr chunks to a callback.
    pub async fn exec_collect_stderr<F>(
        &self,
        command: &str,
        mut on_stderr_data: F,
    ) -> Result<ExecResult>
    where
        F: FnMut(&[u8]) + Send,
    {
        let mut channel = self.handle.channel_open_session().await?;
        channel.exec(true, command).await?;

        let mut stdout = Vec::<u8>::new();
        let mut stderr = Vec::<u8>::new();
        let mut exit_code: Option<u32> = None;

        loop {
            let Some(msg) = channel.wait().await else {
                break;
            };
            match msg {
                ChannelMsg::Data { data } => stdout.extend_from_slice(&data),
                ChannelMsg::ExtendedData { data, ext } if ext == 1 => {
                    stderr.extend_from_slice(&data);
                    on_stderr_data(&data);
                }
                ChannelMsg::ExitStatus { exit_status } => exit_code = Some(exit_status),
                _ => {}
            }
        }
        let _ = channel.close().await;
        Ok(ExecResult {
            stdout: String::from_utf8_lossy(&stdout).into_owned(),
            stderr: String::from_utf8_lossy(&stderr).into_owned(),
            exit_code: exit_code.map(|c| c as i32).unwrap_or(-1),
        })
    }


/// Run a command and stream chunks to the receiver. Sends `None` once finished.
    /// Cancellation: drop the receiver to abort OR set the cancel flag.
    pub async fn exec_stream(
        &self,
        command: &str,
        tx: mpsc::UnboundedSender<StreamChunk>,
        cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    ) -> Result<i32> {
        let mut channel = self.handle.channel_open_session().await?;
        channel.exec(true, command).await?;

        let mut exit_code: i32 = -1;
        loop {
            // Check cancel flag periodically for faster cancellation response
            if let Some(ref c) = cancel {
                if c.load(std::sync::atomic::Ordering::SeqCst) {
                    let _ = channel.close().await;
                    return Err(AppError::Cancelled);
                }
            }
            let Some(msg) = channel.wait().await else {
                break;
            };
            match msg {
                ChannelMsg::Data { data } => {
                    let s = String::from_utf8_lossy(&data).into_owned();
                    if tx.send(StreamChunk::Stdout(s)).is_err() {
                        // receiver dropped: cancel
                        let _ = channel.close().await;
                        return Err(AppError::Cancelled);
                    }
                }
                ChannelMsg::ExtendedData { data, ext } if ext == 1 => {
                    let s = String::from_utf8_lossy(&data).into_owned();
                    if tx.send(StreamChunk::Stderr(s)).is_err() {
                        let _ = channel.close().await;
                        return Err(AppError::Cancelled);
                    }
                }
                ChannelMsg::ExitStatus { exit_status } => exit_code = exit_status as i32,
                ChannelMsg::Eof => {}
                _ => {}
            }
        }
        let _ = channel.close().await;
        let _ = tx.send(StreamChunk::Done(exit_code));
        Ok(exit_code)
    }

    /// Upload a file by streaming content to the remote process's stdin.
    /// This avoids the "Argument list too long" error caused by base64-injecting large files.
    pub async fn write_file(&self, remote_path: &str, content: &str) -> Result<()> {
        use tokio::io::AsyncWriteExt;

        let mut channel = self.handle.channel_open_session().await?;
        // dir-safe write
        let dir = match remote_path.rsplit_once('/') {
            Some((d, _)) if !d.is_empty() => d.to_string(),
            _ => ".".to_string(),
        };
        let cmd = format!("mkdir -p \"{dir}\" && cat > \"{remote_path}\"");
        channel.exec(true, cmd).await?;

        // Stream content to remote process stdin in chunks to prevent buffer overflow
        let mut writer = channel.make_writer();
        let bytes = content.as_bytes();
        let chunk_size = 65536; // 64KB chunks
        let mut pos = 0;
        while pos < bytes.len() {
            let end = (pos + chunk_size).min(bytes.len());
            writer.write_all(&bytes[pos..end]).await?;
            writer.flush().await?;
            pos = end;
            tokio::task::yield_now().await;
        }
        drop(writer);

        // Signal EOF to remote cat so it closes the file and exits
        channel.eof().await?;

        // Wait for process to exit and collect exit status
        let mut exit_code = None;
        loop {
            let Some(msg) = channel.wait().await else {
                break;
            };
            match msg {
                ChannelMsg::ExitStatus { exit_status } => exit_code = Some(exit_status),
                _ => {}
            }
        }
        let _ = channel.close().await;

        if let Some(code) = exit_code {
            if code != 0 {
                return Err(AppError::ssh(format!("write_file process exited with code {code}")));
            }
        }
        Ok(())
    }

    pub async fn disconnect(&self) {
        let _ = self
            .handle
            .disconnect(Disconnect::ByApplication, "bye", "en")
            .await;
    }
}

#[derive(Clone)]
pub struct SshSessionManager {
    cfg: SshConfig,
    session: Arc<tokio::sync::Mutex<Option<(SshSession, std::time::Instant)>>>,
}

impl SshSessionManager {
    pub fn new(cfg: SshConfig) -> Self {
        Self {
            cfg,
            session: Arc::new(tokio::sync::Mutex::new(None)),
        }
    }

    pub async fn get_session(&self) -> Result<SshSession> {
        let mut lock = self.session.lock().await;
        if let Some((ref s, ref mut last_checked)) = *lock {
            if last_checked.elapsed() < std::time::Duration::from_secs(10) {
                return Ok(s.clone());
            }
            // Verify if the session is still alive
            let check_fut = s.exec_blocking("true");
            match tokio::time::timeout(std::time::Duration::from_secs(3), check_fut).await {
                Ok(Ok(res)) if res.exit_code == 0 => {
                    *last_checked = std::time::Instant::now();
                    return Ok(s.clone());
                }
                _ => {
                    println!("SSH session health check failed. Reconnecting...");
                    let _ = s.disconnect().await;
                    *lock = None;
                }
            }
        }

        // Connect a new session with retries
        let mut last_err = None;
        for attempt in 0..3 {
            match SshSession::connect(&self.cfg).await {
                Ok(new_s) => {
                    // Only cache the session if the cached slot is empty (no active idle session)
                    if lock.is_none() {
                        *lock = Some((new_s.clone(), std::time::Instant::now()));
                    }
                    return Ok(new_s);
                }
                Err(e) => {
                    println!("SSH connection attempt {} failed: {}", attempt + 1, e);
                    last_err = Some(e);
                    tokio::time::sleep(std::time::Duration::from_secs(1 << attempt)).await;
                }
            }
        }
        Err(last_err.unwrap_or_else(|| AppError::ssh("Failed to connect after retries")))
    }

    pub async fn clear_session(&self) {
        let mut lock = self.session.lock().await;
        if let Some((ref s, _)) = *lock {
            let _ = s.disconnect().await;
        }
        *lock = None;
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ExecResult {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

/// Clean up a pasted/uploaded PEM private key so russh's parser accepts it.
/// - strips UTF-8 BOM
/// - converts CRLF / CR line endings to LF
/// - trims surrounding whitespace
/// - guarantees a single trailing newline (required by some PEM parsers)
fn normalize_pem(raw: &str) -> String {
    let mut s: &str = raw;
    // strip BOM
    if let Some(stripped) = s.strip_prefix('\u{feff}') {
        s = stripped;
    }
    let mut out = s
        .replace("\r\n", "\n")
        .replace('\r', "\n")
        .trim()
        .to_string();
    out.push('\n');
    out
}

#[derive(Debug, Clone)]
pub enum StreamChunk {
    Stdout(String),
    Stderr(String),
    Done(i32),
}

// --- GPU status parser ----------------------------------------------------

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuProcess {
    pub pid: u32,
    pub process_name: String,
    pub memory: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuState {
    pub success: bool,
    pub simulated: bool,
    pub gpu_name: String,
    pub driver_version: String,
    pub cuda_version: String,
    pub temperature: u32,
    pub utilization_gpu: u32,
    pub utilization_memory: u32,
    pub memory_used: f64,  // MB
    pub memory_total: f64, // MB
    pub power_draw: f64,
    pub power_limit: f64,
    pub fan_speed: u32,
    pub processes: Vec<GpuProcess>,
}

impl GpuState {
    pub fn simulated() -> Self {
        Self {
            success: true,
            simulated: true,
            gpu_name: "AMD Instinct MI300X 192GB (Simulation)".to_string(),
            driver_version: "ROCm 6.2".to_string(),
            cuda_version: "ROCm".to_string(),
            temperature: 42,
            utilization_gpu: 0,
            utilization_memory: 0,
            memory_used: 0.0,
            memory_total: 196608.0,
            power_draw: 95.0,
            power_limit: 750.0,
            fan_speed: 0,
            processes: vec![],
        }
    }

    /// A freshly-created cloud GPU droplet accepts SSH before it has finished
    /// provisioning. Instead of erroring on the boot banner, return a clearly
    /// labelled "not ready yet" state so the dashboard shows progress and the
    /// 5s auto-poll keeps retrying until real GPU stats appear.
    pub fn provisioning() -> Self {
        Self {
            success: false,
            simulated: false,
            gpu_name: "Provisioning… droplet still booting".to_string(),
            driver_version: String::new(),
            cuda_version: String::new(),
            temperature: 0,
            utilization_gpu: 0,
            utilization_memory: 0,
            memory_used: 0.0,
            memory_total: 0.0,
            power_draw: 0.0,
            power_limit: 0.0,
            fan_speed: 0,
            processes: vec![],
        }
    }
}

fn parse_first_number(s: &str) -> Option<f64> {
    let mut buf = String::new();
    let mut started = false;
    for ch in s.chars() {
        if ch.is_ascii_digit() || ch == '.' || (ch == '-' && !started) {
            buf.push(ch);
            started = true;
        } else if started {
            break;
        }
    }
    buf.parse::<f64>().ok()
}

fn value_as_number(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => parse_first_number(s),
        _ => None,
    }
}

fn value_as_string(v: &Value) -> Option<String> {
    match v {
        Value::String(s) if !s.trim().is_empty() => Some(s.trim().to_string()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

fn key_matches(key: &str, required: &[&str]) -> bool {
    let lower = key.to_lowercase();
    required.iter().all(|needle| lower.contains(needle))
}

fn find_number_by_key(v: &Value, required: &[&str]) -> Option<f64> {
    match v {
        Value::Object(map) => {
            for (key, value) in map {
                if key_matches(key, required) {
                    if let Some(n) = value_as_number(value) {
                        return Some(n);
                    }
                }
                if let Some(n) = find_number_by_key(value, required) {
                    return Some(n);
                }
            }
            None
        }
        Value::Array(items) => items.iter().find_map(|item| find_number_by_key(item, required)),
        _ => None,
    }
}

fn find_string_by_key(v: &Value, required: &[&str]) -> Option<String> {
    match v {
        Value::Object(map) => {
            for (key, value) in map {
                if key_matches(key, required) {
                    if let Some(s) = value_as_string(value) {
                        return Some(s);
                    }
                }
                if let Some(s) = find_string_by_key(value, required) {
                    return Some(s);
                }
            }
            None
        }
        Value::Array(items) => items.iter().find_map(|item| find_string_by_key(item, required)),
        _ => None,
    }
}

fn normalize_memory_mb(value: f64) -> f64 {
    if value > 10_000_000.0 {
        value / 1024.0 / 1024.0
    } else if value > 0.0 && value < 1024.0 {
        value * 1024.0
    } else {
        value
    }
}

fn parse_rocm_smi_json(stdout: &str) -> Option<GpuState> {
    let json: Value = serde_json::from_str(stdout).ok()?;
    let memory_total = find_number_by_key(&json, &["vram", "total", "memory"])
        .or_else(|| find_number_by_key(&json, &["memory", "total"]))
        .map(normalize_memory_mb)
        .unwrap_or(0.0);
    let memory_used = find_number_by_key(&json, &["vram", "used", "memory"])
        .or_else(|| find_number_by_key(&json, &["memory", "used"]))
        .map(normalize_memory_mb)
        .unwrap_or(0.0);
    let gpu_name = find_string_by_key(&json, &["card", "series"])
        .or_else(|| find_string_by_key(&json, &["product", "name"]))
        .or_else(|| find_string_by_key(&json, &["card", "model"]))
        .unwrap_or_else(|| "AMD ROCm GPU".to_string());
    let utilization_gpu = find_number_by_key(&json, &["gpu", "use"])
        .or_else(|| find_number_by_key(&json, &["gpu", "util"]))
        .unwrap_or(0.0)
        .round()
        .clamp(0.0, 100.0) as u32;
    let utilization_memory = if memory_total > 0.0 {
        ((memory_used / memory_total) * 100.0).round().clamp(0.0, 100.0) as u32
    } else {
        find_number_by_key(&json, &["memory", "use"])
            .unwrap_or(0.0)
            .round()
            .clamp(0.0, 100.0) as u32
    };

    Some(GpuState {
        success: true,
        simulated: false,
        gpu_name,
        driver_version: "ROCm".to_string(),
        cuda_version: "ROCm".to_string(),
        temperature: find_number_by_key(&json, &["temperature"])
            .or_else(|| find_number_by_key(&json, &["temp"]))
            .unwrap_or(0.0)
            .round()
            .max(0.0) as u32,
        utilization_gpu,
        utilization_memory,
        memory_used,
        memory_total,
        power_draw: find_number_by_key(&json, &["average", "power"])
            .or_else(|| find_number_by_key(&json, &["socket", "power"]))
            .or_else(|| find_number_by_key(&json, &["graphics", "package"]))
            .or_else(|| find_number_by_key(&json, &["power"]))
            .unwrap_or(0.0),
        power_limit: find_number_by_key(&json, &["max", "power"])
            .or_else(|| find_number_by_key(&json, &["power", "limit"]))
            .unwrap_or(0.0),
        fan_speed: find_number_by_key(&json, &["fan"])
            .unwrap_or(0.0)
            .round()
            .clamp(0.0, 100.0) as u32,
        processes: vec![],
    })
}

pub async fn nvidia_smi(session: &SshSession) -> Result<GpuState> {
    let rocm_cmd = "if command -v rocm-smi >/dev/null 2>&1; then rocm-smi --showproductname --showmeminfo vram --showuse --showtemp --showpower --showmaxpower --json; fi";
    if let Ok(out) = session.exec_blocking(rocm_cmd).await {
        if out.exit_code == 0 {
            if let Some(state) = parse_rocm_smi_json(out.stdout.trim()) {
                return Ok(state);
            }
        }
    }

    let query = "nvidia-smi --query-gpu=name,driver_version,vbios_version,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,fan.speed --format=csv,noheader,nounits";
    let proc = "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits";
    let cmd = format!("{query} && echo '---PROCS---' && {proc}");

    let out = session.exec_blocking(&cmd).await?;
    if out.exit_code != 0 && out.stdout.trim().is_empty() {
        return Err(AppError::ssh(format!("nvidia-smi failed: {}", out.stderr)));
    }
    let parts: Vec<&str> = out.stdout.split("---PROCS---").collect();
    let gpu_line = parts.first().map(|s| s.trim()).unwrap_or("");
    let proc_lines = parts.get(1).map(|s| s.trim()).unwrap_or("");

    // A just-created cloud droplet accepts SSH before nvidia-smi exists or the
    // GPU driver is up. DigitalOcean (and others) print a login/cloud-init
    // banner like "Please wait while we get your droplet ready..." which would
    // otherwise be mis-parsed as the GPU CSV line. Detect those and report a
    // provisioning state rather than a hard error so auto-poll keeps retrying.
    let lower = out.stdout.to_lowercase();
    let still_booting = lower.contains("please wait")
        || lower.contains("droplet ready")
        || lower.contains("droplet is being")
        || lower.contains("cloud-init")
        || lower.contains("not yet")
        || lower.contains("command not found"); // nvidia-smi not installed yet
    if still_booting {
        return Ok(GpuState::provisioning());
    }

    let specs: Vec<&str> = gpu_line.split(',').map(|s| s.trim()).collect();
    if specs.len() < 5 {
        return Err(AppError::ssh(format!(
            "unexpected nvidia-smi output: {gpu_line}"
        )));
    }

    let mut processes = vec![];
    for line in proc_lines.lines() {
        let cols: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if cols.len() >= 3 {
            processes.push(GpuProcess {
                pid: cols[0].parse().unwrap_or(0),
                process_name: cols[1].to_string(),
                memory: cols[2].parse().unwrap_or(0),
            });
        }
    }

    Ok(GpuState {
        success: true,
        simulated: false,
        gpu_name: specs[0].to_string(),
        driver_version: specs[1].to_string(),
        cuda_version: specs.get(2).map(|s| s.to_string()).unwrap_or_default(),
        temperature: specs.get(3).and_then(|s| s.parse().ok()).unwrap_or(0),
        utilization_gpu: specs.get(4).and_then(|s| s.parse().ok()).unwrap_or(0),
        utilization_memory: specs.get(5).and_then(|s| s.parse().ok()).unwrap_or(0),
        memory_used: specs.get(6).and_then(|s| s.parse().ok()).unwrap_or(0.0),
        memory_total: specs.get(7).and_then(|s| s.parse().ok()).unwrap_or(0.0),
        power_draw: specs.get(8).and_then(|s| s.parse().ok()).unwrap_or(0.0),
        power_limit: specs.get(9).and_then(|s| s.parse().ok()).unwrap_or(0.0),
        fan_speed: specs.get(10).and_then(|s| s.parse().ok()).unwrap_or(0),
        processes,
    })
}

