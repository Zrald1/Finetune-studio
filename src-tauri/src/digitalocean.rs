use crate::config::DigitalOceanConfig;
use crate::error::{AppError, Result};
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};

const API_BASE: &str = "https://api.digitalocean.com/v2";
// The legacy AMD Developer Cloud host `api-amd.digitalocean.com` now answers
// every request with `301 Moved Permanently` pointing at
// `https://api.devcloud.amd.com/v2`, and that host rejects DigitalOcean
// personal access tokens (`dop_v1_*`) with `401 Unable to authenticate you`.
// AMD-team tokens are therefore served by the standard control plane, which
// lists the AMD Instinct MI-series sizes and accepts plain (non-`-devcloud`)
// slugs. This host is kept only as an opt-in override for tenants holding a
// Developer Cloud-native token rather than a DigitalOcean PAT.
const AMD_DEVCLOUD_API_BASE: &str = "https://api.devcloud.amd.com/v2";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DoGpuInfo {
    pub count: Option<u32>,
    pub model: Option<String>,
    pub vram: Option<DoAmount>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DoAmount {
    pub amount: Option<f64>,
    pub unit: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoSize {
    pub slug: String,
    pub memory: u64,
    pub vcpus: u32,
    pub disk: u64,
    pub transfer: f64,
    #[serde(rename = "priceMonthly", alias = "price_monthly")]
    pub price_monthly: Option<f64>,
    #[serde(rename = "priceHourly", alias = "price_hourly")]
    pub price_hourly: Option<f64>,
    pub regions: Vec<String>,
    pub available: bool,
    pub description: String,
    #[serde(rename = "gpuInfo", alias = "gpu_info")]
    pub gpu_info: Option<DoGpuInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoNetworkAddress {
    #[serde(rename = "ipAddress", alias = "ip_address")]
    pub ip_address: String,
    #[serde(rename = "type")]
    pub kind: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DoNetworks {
    #[serde(default)]
    pub v4: Vec<DoNetworkAddress>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoDroplet {
    pub id: u64,
    pub name: String,
    pub status: String,
    pub urn: Option<String>,
    pub region: Option<serde_json::Value>,
    #[serde(rename = "sizeSlug", alias = "size_slug")]
    pub size_slug: Option<String>,
    pub image: Option<serde_json::Value>,
    #[serde(default)]
    pub networks: DoNetworks,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum SshKeyRef {
    Id(u64),
    Fingerprint(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoRegion {
    pub slug: String,
    pub name: String,
    pub available: bool,
    #[serde(default)]
    pub sizes: Vec<String>,
    #[serde(default)]
    pub features: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoImage {
    pub id: u64,
    pub name: String,
    pub distribution: Option<String>,
    pub slug: Option<String>,
    #[serde(rename = "type")]
    pub image_type: Option<String>,
    pub public: Option<bool>,
    #[serde(default)]
    pub regions: Vec<String>,
    #[serde(rename = "minDiskSize", alias = "min_disk_size")]
    pub min_disk_size: Option<u64>,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoSshKey {
    pub id: u64,
    pub name: String,
    pub fingerprint: String,
    #[serde(rename = "publicKey", alias = "public_key")]
    pub public_key: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoProject {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub purpose: Option<String>,
    pub environment: Option<String>,
    #[serde(rename = "isDefault", alias = "is_default")]
    pub is_default: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoTeam {
    pub name: Option<String>,
    pub uuid: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoAccount {
    pub name: Option<String>,
    pub email: Option<String>,
    pub uuid: String,
    pub status: String,
    pub team: Option<DoTeam>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum ImageRef {
    Id(u64),
    Slug(String),
}

#[derive(Debug, Clone, Serialize)]
struct CreateDropletRequest {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    region: Option<String>,
    size: String,
    image: ImageRef,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    ssh_keys: Vec<SshKeyRef>,
    backups: bool,
    ipv6: bool,
    private_networking: bool,
    public_networking: bool,
    monitoring: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tags: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user_data: Option<String>,
}

#[derive(Debug, Serialize)]
struct AssignProjectResourcesRequest {
    resources: Vec<String>,
}

#[derive(Debug)]
struct CreateAttemptError {
    status: StatusCode,
    body: String,
}

#[derive(Debug, Clone)]
struct CreateAttemptLog {
    size: String,
    region: String,
    status: StatusCode,
    body: String,
}

#[derive(Debug, Deserialize)]
struct SizesResponse {
    sizes: Vec<DoSize>,
    links: Option<DoLinks>,
}

#[derive(Debug, Deserialize)]
struct DropletsResponse {
    droplets: Vec<DoDroplet>,
    links: Option<DoLinks>,
}

#[derive(Debug, Deserialize)]
struct DropletResponse {
    droplet: DoDroplet,
}

#[derive(Debug, Deserialize)]
struct RegionsResponse {
    regions: Vec<DoRegion>,
    links: Option<DoLinks>,
}

#[derive(Debug, Deserialize)]
struct ImagesResponse {
    images: Vec<DoImage>,
    links: Option<DoLinks>,
}

#[derive(Debug, Deserialize)]
struct SshKeysResponse {
    ssh_keys: Vec<DoSshKey>,
    links: Option<DoLinks>,
}

#[derive(Debug, Deserialize)]
struct ProjectsResponse {
    projects: Vec<DoProject>,
    links: Option<DoLinks>,
}

#[derive(Debug, Deserialize)]
struct AccountResponse {
    account: DoAccount,
}

#[derive(Debug, Deserialize)]
struct DoLinks {
    pages: Option<DoPages>,
}

#[derive(Debug, Deserialize)]
struct DoPages {
    next: Option<String>,
}

fn token(cfg: &DigitalOceanConfig) -> Result<&str> {
    let token = cfg.api_key.trim();
    if token.is_empty() {
        return Err(AppError::config("DigitalOcean API key is not configured"));
    }
    Ok(token)
}

/// Effective control-plane base URL. Defaults to the standard DigitalOcean host
/// (which serves AMD, NVIDIA, and CPU plans alike, including AMD-team tokens);
/// an explicit `api_base` override wins so tenants on a different Developer
/// Cloud host can point the app at it without a code change.
fn api_base(cfg: &DigitalOceanConfig) -> String {
    let override_base = cfg.api_base.trim().trim_end_matches('/');
    if override_base.is_empty() {
        API_BASE.to_string()
    } else {
        override_base.to_string()
    }
}

/// Hosts to probe when discovering GPU plans. The override (if any) is tried
/// first, then the standard control plane, then the AMD Developer Cloud host.
/// Failures are tolerated so a token that only works on one host still lists
/// plans from that host.
fn discovery_bases(cfg: &DigitalOceanConfig) -> Vec<String> {
    let mut bases = vec![api_base(cfg)];
    push_unique(&mut bases, API_BASE.to_string());
    push_unique(&mut bases, AMD_DEVCLOUD_API_BASE.to_string());
    bases
}

fn client() -> Result<reqwest::Client> {
    Ok(reqwest::Client::builder()
        .user_agent("fine-tune-studio/0.1")
        .timeout(std::time::Duration::from_secs(30))
        // Control-plane hosts do not legitimately redirect. Following one would
        // rewrite the create POST into a GET (per RFC 9110) and silently return
        // a droplet *list* instead of provisioning — or, for the retired AMD
        // host, land on a different API that rejects the token with 401. Not
        // following keeps the failure explicit and the redirect visible.
        .redirect(reqwest::redirect::Policy::none())
        .build()?)
}

async fn parse_error(res: reqwest::Response, action: &str) -> AppError {
    let status = res.status();
    let body = res.text().await.unwrap_or_default();
    AppError::other(format!("DigitalOcean {action} failed ({status}): {body}"))
}

fn split_csv(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn parse_ssh_keys(raw: &str) -> Vec<SshKeyRef> {
    split_csv(raw)
        .into_iter()
        .map(|item| match item.parse::<u64>() {
            Ok(id) => SshKeyRef::Id(id),
            Err(_) => SshKeyRef::Fingerprint(item),
        })
        .collect()
}

fn parse_image(raw: &str) -> ImageRef {
    match raw.trim().parse::<u64>() {
        Ok(id) => ImageRef::Id(id),
        Err(_) => ImageRef::Slug(raw.trim().to_string()),
    }
}

async fn get_json<T: for<'de> Deserialize<'de>>(
    cfg: &DigitalOceanConfig,
    path: &str,
    action: &str,
) -> Result<T> {
    let res = client()?
        .get(format!("{}{path}", api_base(cfg)))
        .bearer_auth(token(cfg)?)
        .send()
        .await?;
    if !res.status().is_success() {
        return Err(parse_error(res, action).await);
    }
    Ok(res.json::<T>().await?)
}

async fn get_url<T: for<'de> Deserialize<'de>>(
    cfg: &DigitalOceanConfig,
    url: &str,
    action: &str,
) -> Result<T> {
    let res = client()?.get(url).bearer_auth(token(cfg)?).send().await?;
    if !res.status().is_success() {
        return Err(parse_error(res, action).await);
    }
    Ok(res.json::<T>().await?)
}

fn next_link(links: &Option<DoLinks>) -> Option<String> {
    links.as_ref()?.pages.as_ref()?.next.clone()
}

fn has_mi_series_token(text: &str) -> bool {
    text.split(|c: char| !c.is_ascii_alphanumeric())
        .any(|part| part.starts_with("mi") && part[2..].chars().any(|c| c.is_ascii_digit()))
}

fn has_amd_gpu_marker(text: &str) -> bool {
    let normalized = text.to_ascii_lowercase();
    normalized.contains("amd")
        || normalized.contains("instinct")
        || normalized.contains("radeon")
        || has_mi_series_token(&normalized)
}

fn is_amd_gpu_size(size: &DoSize) -> bool {
    if !size.slug.starts_with("gpu-") && size.gpu_info.is_none() {
        return false;
    }

    has_amd_gpu_marker(&format!(
        "{} {} {}",
        size.slug,
        size.description,
        size.gpu_info
            .as_ref()
            .and_then(|gpu| gpu.model.as_deref())
            .unwrap_or("")
    ))
}

async fn fetch_all_sizes(cfg: &DigitalOceanConfig, base: &str) -> Result<Vec<DoSize>> {
    let mut sizes = Vec::new();
    let mut url = format!("{base}/sizes?per_page=200");
    loop {
        let page = get_url::<SizesResponse>(cfg, &url, "list sizes").await?;
        sizes.extend(page.sizes);
        match next_link(&page.links) {
            Some(next) => url = next,
            None => break,
        }
    }
    Ok(sizes)
}

pub async fn list_gpu_sizes(cfg: &DigitalOceanConfig) -> Result<Vec<DoSize>> {
    let bases = discovery_bases(cfg);
    let mut sizes = fetch_all_sizes(cfg, &bases[0]).await?;

    // AMD plans are served by whichever host the token is entitled on, so merge
    // in extras from the remaining hosts. Failures are tolerated so a token that
    // only works on one host still lists that host's plans.
    for base in bases.iter().skip(1) {
        if let Ok(extra) = fetch_all_sizes(cfg, base).await {
            for size in extra {
                if !sizes.iter().any(|existing| existing.slug == size.slug) {
                    sizes.push(size);
                }
            }
        }
    }

    sizes.retain(is_amd_gpu_size);

    // Fallback catalog, used only for plans the control plane fails to report.
    // Specs, prices, and regions mirror the live AMD Instinct catalog: the
    // MI300X generation is no longer provisionable for current AMD teams, and
    // the MI350X/MI355X plans are spot-only. The retired `-devcloud` and
    // `-contracted` slugs are deliberately absent — sending them now yields
    // `422 "This size is unavailable."`.
    let hardcoded_sizes = vec![
        DoSize {
            slug: "gpu-mi325x1-256gb".to_string(),
            memory: 163840,
            vcpus: 20,
            disk: 720,
            transfer: 15000.0,
            price_monthly: Some(2827.2),
            price_hourly: Some(3.8),
            regions: vec!["nyc2".to_string(), "tor1".to_string()],
            available: true,
            description: "AMD Instinct MI325X (1 GPU)".to_string(),
            gpu_info: Some(DoGpuInfo {
                count: Some(1),
                model: Some("AMD Instinct MI325X".to_string()),
                vram: Some(DoAmount {
                    amount: Some(256.0),
                    unit: Some("GB".to_string()),
                }),
            }),
        },
        DoSize {
            slug: "gpu-mi325x8-2048gb".to_string(),
            memory: 1310720,
            vcpus: 160,
            disk: 2046,
            transfer: 60000.0,
            price_monthly: Some(22617.6),
            price_hourly: Some(30.4),
            regions: vec!["nyc2".to_string()],
            available: true,
            description: "AMD Instinct MI325X (8 GPUs)".to_string(),
            gpu_info: Some(DoGpuInfo {
                count: Some(8),
                model: Some("AMD Instinct MI325X".to_string()),
                vram: Some(DoAmount {
                    amount: Some(2048.0),
                    unit: Some("GB".to_string()),
                }),
            }),
        },
        DoSize {
            slug: "gpu-mi355x1-288gb-spot".to_string(),
            memory: 262144,
            vcpus: 24,
            disk: 720,
            transfer: 15000.0,
            price_monthly: Some(3348.0),
            price_hourly: Some(4.5),
            regions: vec!["mem1".to_string()],
            available: true,
            description: "AMD Instinct MI355X (1 GPU, Spot)".to_string(),
            gpu_info: Some(DoGpuInfo {
                count: Some(1),
                model: Some("AMD Instinct MI355X".to_string()),
                vram: Some(DoAmount {
                    amount: Some(288.0),
                    unit: Some("GB".to_string()),
                }),
            }),
        },
        DoSize {
            slug: "gpu-mi355x8-2304gb-spot".to_string(),
            memory: 2097152,
            vcpus: 192,
            disk: 2046,
            transfer: 60000.0,
            price_monthly: Some(26784.0),
            price_hourly: Some(36.0),
            regions: vec!["mem1".to_string()],
            available: true,
            description: "AMD Instinct MI355X (8 GPUs, Spot)".to_string(),
            gpu_info: Some(DoGpuInfo {
                count: Some(8),
                model: Some("AMD Instinct MI355X".to_string()),
                vram: Some(DoAmount {
                    amount: Some(2304.0),
                    unit: Some("GB".to_string()),
                }),
            }),
        },
    ];

    for s in hardcoded_sizes {
        if !sizes.iter().any(|existing| existing.slug == s.slug) {
            sizes.push(s);
        }
    }

    sizes.sort_by(|a, b| {
        let a_price = a.price_hourly.unwrap_or(f64::MAX);
        let b_price = b.price_hourly.unwrap_or(f64::MAX);
        a_price
            .partial_cmp(&b_price)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(sizes)
}

pub async fn list_droplets(cfg: &DigitalOceanConfig) -> Result<Vec<DoDroplet>> {
    let mut droplets = Vec::new();
    let mut url = format!("{}/droplets?per_page=200", api_base(cfg));
    loop {
        let page = get_url::<DropletsResponse>(cfg, &url, "list droplets").await?;
        droplets.extend(page.droplets);
        match next_link(&page.links) {
            Some(next) => url = next,
            None => break,
        }
    }
    Ok(droplets)
}

pub async fn list_gpu_droplets(cfg: &DigitalOceanConfig) -> Result<Vec<DoDroplet>> {
    let mut droplets = Vec::new();
    let mut url = format!("{}/droplets?type=gpus&per_page=200", api_base(cfg));
    loop {
        let page = get_url::<DropletsResponse>(cfg, &url, "list GPU droplets").await?;
        droplets.extend(page.droplets);
        match next_link(&page.links) {
            Some(next) => url = next,
            None => break,
        }
    }
    crate::droplet_usage::reconcile_active(cfg, &droplets)?;
    Ok(droplets)
}

pub async fn list_regions(cfg: &DigitalOceanConfig) -> Result<Vec<DoRegion>> {
    let mut regions = Vec::new();
    let mut url = format!("{}/regions?per_page=200", api_base(cfg));
    loop {
        let page = get_url::<RegionsResponse>(cfg, &url, "list regions").await?;
        regions.extend(page.regions);
        match next_link(&page.links) {
            Some(next) => url = next,
            None => break,
        }
    }
    regions.retain(|r| r.available);
    regions.sort_by(|a, b| a.slug.cmp(&b.slug));
    Ok(regions)
}

pub async fn list_images(cfg: &DigitalOceanConfig) -> Result<Vec<DoImage>> {
    let mut images = Vec::new();
    let mut url = format!("{}/images?per_page=200", api_base(cfg));
    loop {
        let page = get_url::<ImagesResponse>(cfg, &url, "list images").await?;
        images.extend(page.images);
        match next_link(&page.links) {
            Some(next) => url = next,
            None => break,
        }
    }
    images.retain(|img| {
        let text = format!(
            "{} {} {} {}",
            img.name,
            img.slug.as_deref().unwrap_or(""),
            img.distribution.as_deref().unwrap_or(""),
            img.description.as_deref().unwrap_or("")
        );
        text.to_ascii_lowercase().contains("rocm") || has_amd_gpu_marker(&text)
    });
    images.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(images)
}

pub async fn list_ssh_keys(cfg: &DigitalOceanConfig) -> Result<Vec<DoSshKey>> {
    let mut keys = Vec::new();
    let mut url = format!("{}/account/keys?per_page=200", api_base(cfg));
    loop {
        let page = get_url::<SshKeysResponse>(cfg, &url, "list SSH keys").await?;
        keys.extend(page.ssh_keys);
        match next_link(&page.links) {
            Some(next) => url = next,
            None => break,
        }
    }
    Ok(keys)
}

pub async fn list_projects(cfg: &DigitalOceanConfig) -> Result<Vec<DoProject>> {
    let mut projects = Vec::new();
    let mut url = format!("{}/projects?per_page=200", api_base(cfg));
    loop {
        let page = get_url::<ProjectsResponse>(cfg, &url, "list projects").await?;
        projects.extend(page.projects);
        match next_link(&page.links) {
            Some(next) => url = next,
            None => break,
        }
    }
    Ok(projects)
}

pub async fn get_account(cfg: &DigitalOceanConfig) -> Result<DoAccount> {
    Ok(get_json::<AccountResponse>(cfg, "/account", "get account")
        .await?
        .account)
}

async fn assign_project(cfg: &DigitalOceanConfig, droplet: &DoDroplet) -> Result<()> {
    if cfg.project_id.trim().is_empty() {
        return Ok(());
    }
    let resource = droplet
        .urn
        .clone()
        .unwrap_or_else(|| format!("do:droplet:{}", droplet.id));
    let res = client()?
        .post(format!(
            "{}/projects/{}/resources",
            api_base(cfg),
            cfg.project_id.trim()
        ))
        .bearer_auth(token(cfg)?)
        .json(&AssignProjectResourcesRequest {
            resources: vec![resource],
        })
        .send()
        .await?;
    if !res.status().is_success() {
        return Err(parse_error(res, "assign project").await);
    }
    Ok(())
}

async fn create_once(
    cfg: &DigitalOceanConfig,
    req: &CreateDropletRequest,
) -> std::result::Result<DoDroplet, CreateAttemptError> {
    // Every create goes to the effective control plane. AMD MI-series plans are
    // served there too — the old AMD-only host now redirects to a different API
    // that rejects DigitalOcean tokens.
    let base = api_base(cfg);
    let res = client()
        .map_err(|e| CreateAttemptError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            body: e.to_string(),
        })?
        .post(format!("{base}/droplets"))
        .bearer_auth(token(cfg).map_err(|e| CreateAttemptError {
            status: StatusCode::UNAUTHORIZED,
            body: e.to_string(),
        })?)
        .json(req)
        .send()
        .await
        .map_err(|e| CreateAttemptError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            body: e.to_string(),
        })?;

    let status = res.status();
    if !status.is_success() {
        return Err(CreateAttemptError {
            status,
            body: res.text().await.unwrap_or_default(),
        });
    }

    res.json::<DropletResponse>()
        .await
        .map(|response| response.droplet)
        .map_err(|e| CreateAttemptError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            body: e.to_string(),
        })
}

fn matching_image_regions(cfg: &DigitalOceanConfig, images: &[DoImage]) -> Vec<String> {
    let raw = cfg.image.trim();
    let image_id = raw.parse::<u64>().ok();

    images
        .iter()
        .find(|img| {
            image_id == Some(img.id)
                || img.slug.as_deref() == Some(raw)
                || img.name.eq_ignore_ascii_case(raw)
        })
        .map(|img| img.regions.clone())
        .unwrap_or_default()
}

async fn matching_size_regions(cfg: &DigitalOceanConfig) -> Vec<String> {
    let raw = cfg.size.trim();
    if raw.is_empty() {
        return Vec::new();
    }
    // Slugs saved by older builds may still carry a retired `-devcloud` /
    // `-contracted` suffix, so match the normalized form too.
    let normalized = normalize_amd_gpu_slug(raw);
    let mut url = format!("{}/sizes?per_page=200", api_base(cfg));
    loop {
        let page = match get_url::<SizesResponse>(cfg, &url, "list sizes").await {
            Ok(page) => page,
            Err(_) => return Vec::new(),
        };
        if let Some(size) = page
            .sizes
            .iter()
            .find(|s| s.slug == raw || s.slug == normalized)
        {
            return size.regions.clone();
        }
        match next_link(&page.links) {
            Some(next) => url = next,
            None => return Vec::new(),
        }
    }
}

fn push_unique(items: &mut Vec<String>, item: String) {
    if !item.trim().is_empty() && !items.iter().any(|existing| existing == &item) {
        items.push(item);
    }
}

fn region_prefix(region: &str) -> Option<String> {
    let prefix = region.trim_end_matches(|c: char| c.is_ascii_digit());
    if prefix.is_empty() || prefix == region {
        None
    } else {
        Some(prefix.to_string())
    }
}

fn is_amd_gpu_slug(slug: &str) -> bool {
    let lower = slug.to_ascii_lowercase();
    lower.starts_with("gpu-mi")
        && lower[6..]
            .chars()
            .next()
            .map(|c| c.is_ascii_digit())
            .unwrap_or(false)
}

/// Strip suffixes older builds appended to AMD MI-series slugs. The `-devcloud`
/// and `-contracted` variants were retired platform-wide — the control plane
/// now accepts the bare slug (`gpu-mi325x1-256gb`) and answers the suffixed
/// forms with `422 "This size is unavailable."` — so slugs saved by an older
/// build are normalized before use instead of being sent as-is.
fn normalize_amd_gpu_slug(slug: &str) -> String {
    let mut clean = slug.trim();
    for suffix in ["-devcloud", "-contracted"] {
        if let Some(stripped) = clean.strip_suffix(suffix) {
            clean = stripped;
            break;
        }
    }
    clean.to_string()
}

fn size_create_candidates(size: &str) -> Vec<String> {
    // The bare slug is the live, creatable form, so it is tried first. The
    // configured value is kept as a second candidate so a tenant on a host that
    // still expects a suffixed slug is not left without a fallback.
    let mut candidates = Vec::new();
    if is_amd_gpu_slug(size) {
        push_unique(&mut candidates, normalize_amd_gpu_slug(size));
    }
    push_unique(&mut candidates, size.trim().to_string());
    candidates
}

fn create_rejection_message(body: &str) -> String {
    serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|json| {
            json.get("message")
                .and_then(|msg| msg.as_str())
                .map(str::to_string)
        })
        .unwrap_or_else(|| body.to_string())
}

fn all_attempts_contain(attempts: &[CreateAttemptLog], needle: &str) -> bool {
    !attempts.is_empty()
        && attempts
            .iter()
            .all(|attempt| create_rejection_message(&attempt.body).to_ascii_lowercase().contains(needle))
}

fn all_size_unavailable(attempts: &[CreateAttemptLog]) -> bool {
    !attempts.is_empty()
        && attempts.iter().all(|attempt| {
            let msg = create_rejection_message(&attempt.body).to_ascii_lowercase();
            msg.contains("size is not available")
                || msg.contains("size is unavailable")
                || msg.contains("this size is unavailable")
                || msg.contains("invalid size")
                || msg.ends_with(" is unavailable.")
        })
}

/// Plans from the live catalog that publish at least one region, i.e. the ones
/// the control plane is actually willing to provision. Surfaced on failure so
/// the error names a working alternative instead of only reporting that
/// everything was rejected.
async fn creatable_alternatives(cfg: &DigitalOceanConfig, exclude: &str) -> Vec<String> {
    let exclude = normalize_amd_gpu_slug(exclude);
    match list_gpu_sizes(cfg).await {
        Ok(sizes) => sizes
            .into_iter()
            .filter(|size| {
                !size.regions.is_empty() && normalize_amd_gpu_slug(&size.slug) != exclude
            })
            .map(|size| format!("{} [{}]", size.slug, size.regions.join(", ")))
            .collect(),
        Err(_) => Vec::new(),
    }
}

fn summarize_attempts(attempts: &[CreateAttemptLog], limit: usize) -> String {
    let mut lines = attempts
        .iter()
        .take(limit)
        .map(|attempt| {
            format!(
                "{}@{} => {}: {}",
                attempt.size, attempt.region, attempt.status, attempt.body
            )
        })
        .collect::<Vec<_>>();

    if attempts.len() > limit {
        lines.push(format!(
            "... {} more attempts omitted",
            attempts.len() - limit
        ));
    }

    lines.join(" | ")
}

// GPU plan regions are sometimes missing from /v2/sizes even when the plan is
// provisionable, so fall back to the documented region set. This mirrors the
// live AMD Instinct catalog; keep it narrow so a failure points at the real
// account/capacity problem instead of burying it in irrelevant regions. Retired
// generations map to no region at all, which stops the caller from fanning out
// across every image region for a plan that can never be provisioned.
fn documented_amd_gpu_regions(size: &str) -> &'static [&'static str] {
    match normalize_amd_gpu_slug(size).as_str() {
        "gpu-mi325x1-256gb" => &["nyc2", "tor1"],
        "gpu-mi325x8-2048gb" => &["nyc2"],
        "gpu-mi355x1-288gb-spot" => &["mem1"],
        "gpu-mi355x8-2304gb-spot" => &["mem1"],
        _ => &[],
    }
}

async fn candidate_create_regions(cfg: &DigitalOceanConfig) -> (Vec<String>, Vec<String>) {
    let mut candidates = Vec::new();
    if !cfg.region.trim().is_empty() {
        push_unique(&mut candidates, cfg.region.trim().to_string());
    }

    let size_regions = matching_size_regions(cfg).await;

    let image_regions = match list_images(cfg).await {
        Ok(images) => matching_image_regions(cfg, &images),
        Err(_) => Vec::new(),
    };

    let account_regions = match list_regions(cfg).await {
        Ok(regions) => regions
            .into_iter()
            .map(|region| region.slug)
            .collect::<Vec<_>>(),
        Err(_) => Vec::new(),
    };

    let amd_gpu = is_amd_gpu_slug(&cfg.size);

    // 1. Size's own published regions are most authoritative when present.
    for region in &size_regions {
        push_unique(&mut candidates, region.clone());
    }

    if amd_gpu {
        // 2. AMD GPU plans are capacity-constrained and their region lists are
        //    frequently omitted, so fall back to the documented region set. Do
        //    NOT fan out across every region the image supports: that produces a
        //    burst of creates for a plan that is often simply not entitled, and
        //    buries the real account-level refusal in noise.
        for region in documented_amd_gpu_regions(&cfg.size) {
            push_unique(&mut candidates, (*region).to_string());
        }
    } else {
        // 3. Non-AMD plans: intersect with the image's regions where the plan
        //    publishes any, then add the rest.
        for region in &image_regions {
            if size_regions.is_empty() || size_regions.iter().any(|r| r == region) {
                push_unique(&mut candidates, region.clone());
            }
        }
        for region in image_regions {
            push_unique(&mut candidates, region);
        }

        // 4. Account-listed regions are a useful fallback for non-AMD plans.
        for region in account_regions {
            push_unique(&mut candidates, region);
        }

        // 5. Try trailing-digit-stripped aliases (some accounts accept "nyc" for "nyc3").
        let expanded = candidates.clone();
        for region in expanded {
            if let Some(prefix) = region_prefix(&region) {
                push_unique(&mut candidates, prefix);
            }
        }
    }

    (candidates, size_regions)
}

async fn create_context(cfg: &DigitalOceanConfig) -> String {
    let account = match get_account(cfg).await {
        Ok(account) => account
            .team
            .and_then(|team| team.name)
            .map(|name| format!("team={name}"))
            .unwrap_or_else(|| {
                format!(
                    "account={}",
                    account.name.or(account.email).unwrap_or(account.uuid)
                )
            }),
        Err(_) => "team=unknown".to_string(),
    };

    let project = if cfg.project_id.trim().is_empty() {
        "project=default".to_string()
    } else {
        match list_projects(cfg).await {
            Ok(projects) => projects
                .into_iter()
                .find(|project| project.id == cfg.project_id.trim())
                .map(|project| format!("project={}", project.name))
                .unwrap_or_else(|| format!("project_id={}", cfg.project_id.trim())),
            Err(_) => format!("project_id={}", cfg.project_id.trim()),
        }
    };

    format!(
        "{account}, {project}, size={}, image={}",
        cfg.size.trim(),
        cfg.image.trim()
    )
}

pub async fn create_droplet(cfg: &DigitalOceanConfig) -> Result<DoDroplet> {
    if cfg.droplet_name.trim().is_empty() {
        return Err(AppError::config("DigitalOcean droplet name is required"));
    }
    if cfg.size.trim().is_empty() || cfg.image.trim().is_empty() {
        return Err(AppError::config(
            "DigitalOcean GPU size and image are required",
        ));
    }

    let base_req = CreateDropletRequest {
        name: cfg.droplet_name.trim().to_string(),
        region: if cfg.region.trim().is_empty() {
            None
        } else {
            Some(cfg.region.trim().to_string())
        },
        size: cfg.size.trim().to_string(),
        image: parse_image(&cfg.image),
        ssh_keys: parse_ssh_keys(&cfg.ssh_keys),
        backups: cfg.backups,
        ipv6: cfg.ipv6,
        private_networking: false,
        public_networking: true,
        monitoring: cfg.monitoring,
        tags: split_csv(&cfg.tags),
        user_data: if cfg.user_data.trim().is_empty() {
            None
        } else {
            Some(cfg.user_data.clone())
        },
    };

    let (candidates, size_regions) = candidate_create_regions(cfg).await;
    let size_candidates = size_create_candidates(&cfg.size);

    let mut attempts = Vec::new();
    for size in &size_candidates {
        if candidates.is_empty() {
            let mut req = base_req.clone();
            req.size = size.clone();
            match create_once(cfg, &req).await {
                Ok(droplet) => {
                    // The Droplet now exists and is billing. A project-assignment
                    // failure (e.g. the project belongs to a different team, which
                    // AMD Developer Cloud resources cannot be moved into) must not
                    // be reported as a failed create, or the caller never learns
                    // the id of the server it is paying for.
                    let _ = assign_project(cfg, &droplet).await;
                    crate::droplet_usage::record_created(cfg, &droplet)?;
                    return Ok(droplet);
                }
                Err(err) => attempts.push(CreateAttemptLog {
                    size: size.clone(),
                    region: req.region.as_deref().unwrap_or("auto").to_string(),
                    status: err.status,
                    body: err.body,
                }),
            }
            continue;
        }

        for region in &candidates {
            let mut req = base_req.clone();
            req.size = size.clone();
            req.region = Some(region.clone());
            match create_once(cfg, &req).await {
                Ok(droplet) => {
                    let _ = assign_project(cfg, &droplet).await;
                    crate::droplet_usage::record_created(cfg, &droplet)?;
                    return Ok(droplet);
                }
                Err(err) => attempts.push(CreateAttemptLog {
                    size: size.clone(),
                    region: region.clone(),
                    status: err.status,
                    body: err.body,
                }),
            }
        }
    }

    let context = create_context(cfg).await;
    let alternatives = creatable_alternatives(cfg, &cfg.size).await;
    let alternatives_hint = if alternatives.is_empty() {
        "No GPU plan in this team's catalog currently publishes a creatable region.".to_string()
    } else {
        format!(
            "Plans that do publish a creatable region: {}.",
            alternatives.join("; ")
        )
    };

    if all_attempts_contain(&attempts, "exceed your gpu limit") {
        return Err(AppError::other(format!(
            "DigitalOcean refused '{size}' for this team because it would exceed the account GPU limit ({context}). \
             AMD Developer Cloud credits allow one 8-GPU Droplet or up to eight single-GPU Droplets, and the 8-GPU plans \
             additionally require the multi-node allocation to be enabled for the team. \
             {alternatives_hint} Reduce the plan to a single-GPU size, or ask DigitalOcean support to raise the GPU limit \
             for this team. Attempts: {attempts}",
            size = cfg.size.trim(),
            attempts = summarize_attempts(&attempts, 8),
        )));
    }

    if is_amd_gpu_slug(&cfg.size) && all_size_unavailable(&attempts) {
        return Err(AppError::other(format!(
            "DigitalOcean will not provision '{size}' for this team ({context}). \
             Every attempt came back 'size unavailable / not available in this region', which means the plan is retired, \
             not entitled for this account, or out of capacity in the regions tried — re-sending the same slug will not \
             change the result. {alternatives_hint} \
             Region slugs tried: [{regions}]. Size slugs tried: [{size_candidates}]. First attempts: {attempts}",
            size = cfg.size.trim(),
            regions = if candidates.is_empty() {
                "auto".to_string()
            } else {
                candidates.join(", ")
            },
            size_candidates = size_candidates.join(", "),
            attempts = summarize_attempts(&attempts, 8),
        )));
    }

    Err(AppError::other(format!(
        "DigitalOcean create droplet failed in every candidate region ({context}). \
         Size '{size}' is published as available in: [{published}]. \
         {alternatives_hint} \
         Size slugs tried: [{size_candidates}]. Attempts: {attempts}",
        size = cfg.size.trim(),
        size_candidates = size_candidates.join(", "),
        published = if size_regions.is_empty() {
            "none reported by API".to_string()
        } else {
            size_regions.join(", ")
        },
        attempts = summarize_attempts(&attempts, 32),
    )))
}

pub async fn destroy_droplet(cfg: &DigitalOceanConfig, droplet_id: u64) -> Result<()> {
    let res = client()?
        .delete(format!("{}/droplets/{droplet_id}", api_base(cfg)))
        .bearer_auth(token(cfg)?)
        .send()
        .await?;
    if res.status() == StatusCode::NO_CONTENT {
        crate::droplet_usage::record_destroyed(droplet_id)?;
        return Ok(());
    }
    if !res.status().is_success() {
        return Err(parse_error(res, "destroy droplet").await);
    }
    crate::droplet_usage::record_destroyed(droplet_id)?;
    Ok(())
}
