use crate::config::DigitalOceanConfig;
use crate::error::{AppError, Result};
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};

const API_BASE: &str = "https://api.digitalocean.com/v2";
// AMD Instinct MI-series GPU droplets are NOT served by the standard control
// plane. They live on the AMD Developer Cloud endpoint and use size slugs with
// a `-devcloud` suffix (e.g. `gpu-mi300x1-192gb-devcloud`). On the standard host
// these sizes appear in `/sizes` with an empty `regions` array and every create
// returns `422 "Size is not available in this region."`. Routing GPU `/sizes`
// and droplet creation to this host (with the devcloud slug) is what actually
// lets the create succeed. Verified end-to-end against a live AMD-team token.
const AMD_API_BASE: &str = "https://api-amd.digitalocean.com/v2";

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

fn client() -> Result<reqwest::Client> {
    Ok(reqwest::Client::builder()
        .user_agent("fine-tune-studio/0.1")
        .timeout(std::time::Duration::from_secs(30))
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

async fn get_json<T: for<'de> Deserialize<'de>>(cfg: &DigitalOceanConfig, path: &str, action: &str) -> Result<T> {
    let res = client()?
        .get(format!("{API_BASE}{path}"))
        .bearer_auth(token(cfg)?)
        .send()
        .await?;
    if !res.status().is_success() {
        return Err(parse_error(res, action).await);
    }
    Ok(res.json::<T>().await?)
}

async fn get_url<T: for<'de> Deserialize<'de>>(cfg: &DigitalOceanConfig, url: &str, action: &str) -> Result<T> {
    let res = client()?
        .get(url)
        .bearer_auth(token(cfg)?)
        .send()
        .await?;
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
    let mut sizes = fetch_all_sizes(cfg, API_BASE).await?;

    // AMD MI-series GPU sizes only appear (with real regions and the creatable
    // `-devcloud` slug) on the AMD Developer Cloud endpoint. Merge them in;
    // tolerate failure so a standard-host-only token still lists CPU/NVIDIA GPUs.
    if let Ok(amd_sizes) = fetch_all_sizes(cfg, AMD_API_BASE).await {
        for size in amd_sizes {
            if !sizes.iter().any(|existing| existing.slug == size.slug) {
                sizes.push(size);
            }
        }
    }

    sizes.retain(is_amd_gpu_size);

    let hardcoded_sizes = vec![
        DoSize {
            slug: "gpu-mi300x1-192gb".to_string(),
            memory: 245760,
            vcpus: 20,
            disk: 720,
            transfer: 15000.0,
            price_monthly: Some(1432.8),
            price_hourly: Some(1.99),
            regions: vec!["atl1".to_string()],
            available: true,
            description: "AMD Instinct MI300X (1 GPU)".to_string(),
            gpu_info: Some(DoGpuInfo {
                count: Some(1),
                model: Some("AMD Instinct MI300X".to_string()),
                vram: Some(DoAmount {
                    amount: Some(192.0),
                    unit: Some("GB".to_string()),
                }),
            }),
        },
        DoSize {
            slug: "gpu-mi300x8-1536gb".to_string(),
            memory: 1966080,
            vcpus: 160,
            disk: 2046,
            transfer: 60000.0,
            price_monthly: Some(11462.4),
            price_hourly: Some(15.92),
            regions: vec!["atl1".to_string()],
            available: true,
            description: "AMD Instinct MI300X (8 GPUs)".to_string(),
            gpu_info: Some(DoGpuInfo {
                count: Some(8),
                model: Some("AMD Instinct MI300X".to_string()),
                vram: Some(DoAmount {
                    amount: Some(1536.0),
                    unit: Some("GB".to_string()),
                }),
            }),
        },
        DoSize {
            slug: "gpu-mi325x1-256gb".to_string(),
            memory: 167936,
            vcpus: 20,
            disk: 720,
            transfer: 15000.0,
            price_monthly: Some(1648.8),
            price_hourly: Some(2.29),
            regions: vec!["atl1".to_string(), "nyc2".to_string(), "sfo3".to_string(), "tor1".to_string()],
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
            memory: 1341440,
            vcpus: 160,
            disk: 2046,
            transfer: 60000.0,
            price_monthly: Some(13190.4),
            price_hourly: Some(18.32),
            regions: vec!["atl1".to_string(), "nyc2".to_string(), "sfo3".to_string(), "tor1".to_string()],
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
            slug: "gpu-mi350x1-288gb".to_string(),
            memory: 262144,
            vcpus: 24,
            disk: 720,
            transfer: 15000.0,
            price_monthly: Some(3168.0),
            price_hourly: Some(4.4),
            regions: vec!["atl1".to_string(), "ric1".to_string()],
            available: true,
            description: "AMD Instinct MI350X (1 GPU)".to_string(),
            gpu_info: Some(DoGpuInfo {
                count: Some(1),
                model: Some("AMD Instinct MI350X".to_string()),
                vram: Some(DoAmount {
                    amount: Some(288.0),
                    unit: Some("GB".to_string()),
                }),
            }),
        },
        DoSize {
            slug: "gpu-mi350x8-2304gb".to_string(),
            memory: 2097152,
            vcpus: 192,
            disk: 2048,
            transfer: 60000.0,
            price_monthly: Some(25344.0),
            price_hourly: Some(35.2),
            regions: vec!["atl1".to_string(), "ric1".to_string()],
            available: true,
            description: "AMD Instinct MI350X (8 GPUs)".to_string(),
            gpu_info: Some(DoGpuInfo {
                count: Some(8),
                model: Some("AMD Instinct MI350X".to_string()),
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
    let mut url = format!("{API_BASE}/droplets?per_page=200");
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
    let mut url = format!("{API_BASE}/droplets?type=gpus&per_page=200");
    loop {
        let page = get_url::<DropletsResponse>(cfg, &url, "list GPU droplets").await?;
        droplets.extend(page.droplets);
        match next_link(&page.links) {
            Some(next) => url = next,
            None => break,
        }
    }
    Ok(droplets)
}

pub async fn list_regions(cfg: &DigitalOceanConfig) -> Result<Vec<DoRegion>> {
    let mut regions = Vec::new();
    let mut url = format!("{API_BASE}/regions?per_page=200");
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
    let mut url = format!("{API_BASE}/images?per_page=200");
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
    let mut url = format!("{API_BASE}/account/keys?per_page=200");
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
    let mut url = format!("{API_BASE}/projects?per_page=200");
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
    Ok(get_json::<AccountResponse>(cfg, "/account", "get account").await?.account)
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
        .post(format!("{API_BASE}/projects/{}/resources", cfg.project_id.trim()))
        .bearer_auth(token(cfg)?)
        .json(&AssignProjectResourcesRequest { resources: vec![resource] })
        .send()
        .await?;
    if !res.status().is_success() {
        return Err(parse_error(res, "assign project").await);
    }
    Ok(())
}

async fn create_once(cfg: &DigitalOceanConfig, req: &CreateDropletRequest) -> std::result::Result<DoDroplet, CreateAttemptError> {
    // Route to the host that can serve the size being requested. The size field
    // here already carries the `-devcloud` suffix for AMD candidates, so this
    // picks the AMD Developer Cloud endpoint for them and the standard control
    // plane for everything else.
    let base = size_api_base(&req.size);
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
    // For AMD slugs the authoritative regions live on the AMD host under the
    // `-devcloud` slug, so match both the raw and devcloud names there. Non-AMD
    // sizes are queried on the standard host as before.
    let base = size_api_base(raw);
    let devcloud = devcloud_amd_gpu_slug(raw);
    let mut url = format!("{base}/sizes?per_page=200");
    loop {
        let page = match get_url::<SizesResponse>(cfg, &url, "list sizes").await {
            Ok(page) => page,
            Err(_) => return Vec::new(),
        };
        if let Some(size) = page
            .sizes
            .iter()
            .find(|s| s.slug == raw || devcloud.as_deref() == Some(s.slug.as_str()))
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
    lower.starts_with("gpu-mi") && lower[6..].chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false)
}

/// The API host that can actually serve the configured size. AMD MI-series GPU
/// droplets are only creatable through the AMD Developer Cloud endpoint; every
/// other resource (and every non-AMD size) uses the standard control plane.
fn size_api_base(size: &str) -> &'static str {
    if is_amd_gpu_slug(size) {
        AMD_API_BASE
    } else {
        API_BASE
    }
}

fn contracted_amd_gpu_slug(slug: &str) -> Option<String> {
    let clean = slug.trim();
    if !is_amd_gpu_slug(clean) || clean.ends_with("-contracted") || clean.contains("-fabric-") {
        return None;
    }
    Some(format!("{clean}-contracted"))
}

/// AMD MI-series sizes must be requested with the `-devcloud` slug on the AMD
/// endpoint. Returns `None` for non-AMD slugs or ones already carrying the
/// suffix so we never double-append it.
fn devcloud_amd_gpu_slug(slug: &str) -> Option<String> {
    let clean = slug.trim();
    if !is_amd_gpu_slug(clean) || clean.ends_with("-devcloud") {
        return None;
    }
    // Strip a stale `-contracted` suffix first so we don't produce
    // `...-contracted-devcloud`, which is not a real slug.
    let core = strip_contracted_suffix(clean);
    Some(format!("{core}-devcloud"))
}

fn size_create_candidates(size: &str) -> Vec<String> {
    // Order matters: the `-devcloud` slug on the AMD endpoint is the one that
    // actually provisions AMD GPUs, so try it first. The bare and `-contracted`
    // slugs stay as fallbacks for any account/region where DO accepts them.
    let mut candidates = Vec::new();
    if let Some(devcloud) = devcloud_amd_gpu_slug(size) {
        push_unique(&mut candidates, devcloud);
    }
    push_unique(&mut candidates, size.trim().to_string());
    if let Some(contracted) = contracted_amd_gpu_slug(size) {
        push_unique(&mut candidates, contracted);
    }
    candidates
}

fn create_rejection_message(body: &str) -> String {
    serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|json| json.get("message").and_then(|msg| msg.as_str()).map(str::to_string))
        .unwrap_or_else(|| body.to_string())
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
        lines.push(format!("... {} more attempts omitted", attempts.len() - limit));
    }

    lines.join(" | ")
}

fn strip_contracted_suffix(slug: &str) -> &str {
    slug.trim().strip_suffix("-contracted").unwrap_or(slug.trim())
}

// GPU size regions are sometimes missing from /v2/sizes even when DigitalOcean
// documents the plan. Keep this list narrow so creation failures point at the
// real account/capacity problem instead of burying it in irrelevant regions.
fn documented_amd_gpu_regions(size: &str) -> &'static [&'static str] {
    match strip_contracted_suffix(size) {
        "gpu-mi300x1-192gb" | "gpu-mi300x8-1536gb" => &["atl1"],
        "gpu-mi325x1-256gb" | "gpu-mi325x8-2048gb" => &["atl1", "nyc2", "sfo3", "tor1"],
        "gpu-mi350x1-288gb" | "gpu-mi350x8-2304gb" => &["atl1", "ric1"],
        _ => &["atl1"],
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
        Ok(regions) => regions.into_iter().map(|region| region.slug).collect::<Vec<_>>(),
        Err(_) => Vec::new(),
    };

    // 1. Size's own published regions are most authoritative when present.
    for region in &size_regions {
        push_unique(&mut candidates, region.clone());
    }

    // 2. For AMD GPU sizes, always include known AMD-host regions. /v2/sizes often
    //    omits AMD GPU sizes for tokens that can still book them via the special
    //    AMD allocation, so size_regions can legitimately be empty.
    if is_amd_gpu_slug(&cfg.size) {
        for region in documented_amd_gpu_regions(&cfg.size) {
            push_unique(&mut candidates, (*region).to_string());
        }
    }

    // 3. Intersect with image regions where possible, then add the rest.
    for region in &image_regions {
        if size_regions.is_empty() || size_regions.iter().any(|r| r == region) {
            push_unique(&mut candidates, region.clone());
        }
    }
    for region in image_regions {
        push_unique(&mut candidates, region);
    }

    // 4. Account-listed regions are useful for non-GPU fallback, but AMD GPU
    //    creates should stay on documented GPU regions. Trying every CPU region
    //    creates noise and can mask an account-level GPU API refusal.
    if !is_amd_gpu_slug(&cfg.size) {
        for region in account_regions {
            push_unique(&mut candidates, region);
        }
    }

    // 5. Try trailing-digit-stripped aliases (some accounts accept "nyc" for "nyc3").
    if !is_amd_gpu_slug(&cfg.size) {
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
                    account
                        .name
                        .or(account.email)
                        .unwrap_or(account.uuid)
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

    format!("{account}, {project}, size={}, image={}", cfg.size.trim(), cfg.image.trim())
}

pub async fn create_droplet(cfg: &DigitalOceanConfig) -> Result<DoDroplet> {
    if cfg.droplet_name.trim().is_empty() {
        return Err(AppError::config("DigitalOcean droplet name is required"));
    }
    if cfg.size.trim().is_empty() || cfg.image.trim().is_empty() {
        return Err(AppError::config("DigitalOcean GPU size and image are required"));
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
                    assign_project(cfg, &droplet).await?;
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
                    assign_project(cfg, &droplet).await?;
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

    if is_amd_gpu_slug(&cfg.size) && size_regions.is_empty() && all_size_unavailable(&attempts) {
        return Err(AppError::other(format!(
            "DigitalOcean rejected AMD GPU Droplet creation for '{size}' on the AMD Developer Cloud endpoint ({amd_base}). \
             AMD MI-series GPUs are served there under the '-devcloud' slug in [{documented_regions}], and this app already \
             retried that endpoint, slug, and region set — every attempt came back 'size unavailable', which means there is no \
             AMD GPU capacity available for this team right now (or the team is not entitled to this size). \
             This is a capacity/entitlement issue, not a request-format problem. \
             Try a different AMD GPU size or region, or create the Droplet from the DigitalOcean control panel and click Sync Account + Use IP. \
             If the dashboard also reports no capacity, ask DigitalOcean support about AMD GPU availability for size '{size}' on team '{team}'. \
             Size slugs tried: [{size_candidates}]. First attempts: {attempts}",
            size = cfg.size.trim(),
            amd_base = AMD_API_BASE,
            team = create_context(cfg).await,
            documented_regions = documented_amd_gpu_regions(&cfg.size).join(", "),
            size_candidates = size_candidates.join(", "),
            attempts = summarize_attempts(&attempts, 8),
        )));
    }

    Err(AppError::other(format!(
        "DigitalOcean create droplet failed in every candidate region ({context}). \
         Size '{size}' is published as available in: [{published}]. \
         Note: AMD MI-series GPU sizes may report an empty regions list even when the dashboard can create them; \
         this app tried the known AMD GPU regions and any matching contracted AMD slug. \
         Size slugs tried: [{size_candidates}]. Attempts: {attempts}",
        size = cfg.size.trim(),
        size_candidates = size_candidates.join(", "),
        published = if size_regions.is_empty() {
            "none reported by API".to_string()
        } else {
            size_regions.join(", ")
        },
        attempts = summarize_attempts(&attempts, 32),
        context = create_context(cfg).await
    )))
}

pub async fn destroy_droplet(cfg: &DigitalOceanConfig, droplet_id: u64) -> Result<()> {
    let res = client()?
        .delete(format!("{API_BASE}/droplets/{droplet_id}"))
        .bearer_auth(token(cfg)?)
        .send()
        .await?;
    if res.status() == StatusCode::NO_CONTENT {
        return Ok(());
    }
    if !res.status().is_success() {
        return Err(parse_error(res, "destroy droplet").await);
    }
    Ok(())
}
