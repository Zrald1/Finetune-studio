use crate::config::{self, DigitalOceanConfig};
use crate::digitalocean::DoDroplet;
use crate::error::{AppError, Result};
use chrono::{DateTime, Utc};
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

static USAGE_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct DropletUsageRecord {
    pub droplet_id: u64,
    pub name: String,
    pub size_slug: String,
    pub region: String,
    pub ip_address: String,
    pub status: String,
    pub source: String,
    pub hourly_rate_usd: Option<f64>,
    pub local_started_at: DateTime<Utc>,
    pub last_seen_at: Option<DateTime<Utc>>,
    pub local_stopped_at: Option<DateTime<Utc>>,
}

impl Default for DropletUsageRecord {
    fn default() -> Self {
        Self {
            droplet_id: 0,
            name: String::new(),
            size_slug: String::new(),
            region: String::new(),
            ip_address: String::new(),
            status: String::new(),
            source: "unknown".to_string(),
            hourly_rate_usd: None,
            local_started_at: Utc::now(),
            last_seen_at: None,
            local_stopped_at: None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DropletUsageSummary {
    pub droplet_id: u64,
    pub name: String,
    pub size_slug: String,
    pub region: String,
    pub ip_address: String,
    pub status: String,
    pub source: String,
    pub hourly_rate_usd: Option<f64>,
    pub local_started_at: DateTime<Utc>,
    pub last_seen_at: Option<DateTime<Utc>>,
    pub local_stopped_at: Option<DateTime<Utc>>,
    pub duration_seconds: i64,
    pub duration_hours: f64,
    pub duration_hms: String,
    pub local_estimated_cost_usd: Option<f64>,
    pub cost_basis: String,
}

fn usage_json_path() -> Result<PathBuf> {
    Ok(config::app_dir()?.join("droplet_usage.json"))
}

fn usage_csv_path() -> Result<PathBuf> {
    Ok(config::app_dir()?.join("droplet_usage.csv"))
}

fn ensure_app_dir_sync() -> Result<()> {
    fs::create_dir_all(config::app_dir()?)?;
    Ok(())
}

fn load_records_sync() -> Result<Vec<DropletUsageRecord>> {
    ensure_app_dir_sync()?;
    let path = usage_json_path()?;
    if !path.exists() {
        return Ok(Vec::new());
    }
    let raw = fs::read_to_string(&path)?;
    if raw.trim().is_empty() {
        return Ok(Vec::new());
    }
    serde_json::from_str(&raw)
        .map_err(|e| AppError::config(format!("parse droplet_usage.json: {e}")))
}

fn save_records_sync(records: &[DropletUsageRecord]) -> Result<()> {
    ensure_app_dir_sync()?;
    let json = serde_json::to_string_pretty(records)?;
    fs::write(usage_json_path()?, json)?;
    write_csv_sync(records, Utc::now())?;
    Ok(())
}

fn write_csv_sync(records: &[DropletUsageRecord], now: DateTime<Utc>) -> Result<PathBuf> {
    ensure_app_dir_sync()?;
    let csv = usage_csv(records, now);
    let path = usage_csv_path()?;
    fs::write(&path, csv)?;
    Ok(path)
}

fn public_ip(droplet: &DoDroplet) -> String {
    droplet
        .networks
        .v4
        .iter()
        .find(|addr| addr.kind == "public")
        .map(|addr| addr.ip_address.clone())
        .unwrap_or_default()
}

fn region_slug(droplet: &DoDroplet) -> String {
    let Some(region) = droplet.region.as_ref() else {
        return String::new();
    };
    if let Some(slug) = region.get("slug").and_then(|value| value.as_str()) {
        return slug.to_string();
    }
    if let Some(slug) = region.as_str() {
        return slug.to_string();
    }
    String::new()
}

fn normalize_size_slug(slug: &str) -> String {
    slug.trim()
        .strip_suffix("-devcloud")
        .unwrap_or(slug.trim())
        .to_string()
}

fn known_hourly_rate_usd(size_slug: &str) -> Option<f64> {
    match normalize_size_slug(size_slug).as_str() {
        "gpu-mi300x1-192gb" => Some(1.99),
        "gpu-mi300x8-1536gb" => Some(15.92),
        "gpu-mi325x1-256gb" => Some(2.29),
        "gpu-mi325x8-2048gb" => Some(18.32),
        "gpu-mi350x1-288gb" => Some(4.40),
        "gpu-mi350x8-2304gb" => Some(35.20),
        _ => None,
    }
}

fn configured_hourly_rate_usd(cfg: &DigitalOceanConfig, size_slug: &str) -> Option<f64> {
    cfg.hourly_rate_usd
        .filter(|rate| rate.is_finite() && *rate > 0.0)
        .or_else(|| known_hourly_rate_usd(size_slug))
}

fn upsert_active_record(
    records: &mut Vec<DropletUsageRecord>,
    cfg: &DigitalOceanConfig,
    droplet: &DoDroplet,
    source: &str,
    now: DateTime<Utc>,
) {
    let size_slug = droplet
        .size_slug
        .clone()
        .unwrap_or_else(|| cfg.size.trim().to_string());
    let hourly_rate_usd = configured_hourly_rate_usd(cfg, &size_slug);

    if let Some(record) = records
        .iter_mut()
        .find(|record| record.droplet_id == droplet.id)
    {
        record.name = droplet.name.clone();
        if !size_slug.is_empty() {
            record.size_slug = size_slug.clone();
        }
        record.region = region_slug(droplet);
        record.ip_address = public_ip(droplet);
        record.status = droplet.status.clone();
        record.last_seen_at = Some(now);
        if record.hourly_rate_usd.is_none() {
            record.hourly_rate_usd = hourly_rate_usd;
        }
        return;
    }

    records.push(DropletUsageRecord {
        droplet_id: droplet.id,
        name: droplet.name.clone(),
        size_slug,
        region: region_slug(droplet),
        ip_address: public_ip(droplet),
        status: droplet.status.clone(),
        source: source.to_string(),
        hourly_rate_usd,
        local_started_at: now,
        last_seen_at: Some(now),
        local_stopped_at: None,
    });
}

pub fn record_created(cfg: &DigitalOceanConfig, droplet: &DoDroplet) -> Result<()> {
    let _guard = USAGE_LOCK.lock();
    let mut records = load_records_sync()?;
    let now = Utc::now();
    upsert_active_record(&mut records, cfg, droplet, "app_create", now);
    save_records_sync(&records)
}

pub fn record_destroyed(droplet_id: u64) -> Result<()> {
    let _guard = USAGE_LOCK.lock();
    let mut records = load_records_sync()?;
    let now = Utc::now();
    if let Some(record) = records
        .iter_mut()
        .find(|record| record.droplet_id == droplet_id && record.local_stopped_at.is_none())
    {
        record.local_stopped_at = Some(now);
        record.last_seen_at = Some(now);
        record.status = "deleted".to_string();
    }
    save_records_sync(&records)
}

pub fn reconcile_active(cfg: &DigitalOceanConfig, droplets: &[DoDroplet]) -> Result<()> {
    let _guard = USAGE_LOCK.lock();
    let mut records = load_records_sync()?;
    let now = Utc::now();
    let mut active_ids = HashSet::new();

    for droplet in droplets {
        active_ids.insert(droplet.id);
        upsert_active_record(&mut records, cfg, droplet, "sync_discovered", now);
    }

    for record in &mut records {
        if record.local_stopped_at.is_none()
            && record.droplet_id != 0
            && !active_ids.contains(&record.droplet_id)
        {
            record.local_stopped_at = Some(now);
            record.last_seen_at = Some(now);
            record.status = "missing_on_sync".to_string();
        }
    }

    save_records_sync(&records)
}

fn duration_seconds(record: &DropletUsageRecord, now: DateTime<Utc>) -> i64 {
    let end = record.local_stopped_at.unwrap_or(now);
    (end - record.local_started_at).num_seconds().max(0)
}

fn format_duration_hms(seconds: i64) -> String {
    let hours = seconds / 3600;
    let minutes = (seconds % 3600) / 60;
    let secs = seconds % 60;
    format!("{hours:02}:{minutes:02}:{secs:02}")
}

fn summarize(record: &DropletUsageRecord, now: DateTime<Utc>) -> DropletUsageSummary {
    let duration_seconds = duration_seconds(record, now);
    let duration_hours = duration_seconds as f64 / 3600.0;
    let local_estimated_cost_usd = record.hourly_rate_usd.map(|rate| duration_hours * rate);
    DropletUsageSummary {
        droplet_id: record.droplet_id,
        name: record.name.clone(),
        size_slug: record.size_slug.clone(),
        region: record.region.clone(),
        ip_address: record.ip_address.clone(),
        status: record.status.clone(),
        source: record.source.clone(),
        hourly_rate_usd: record.hourly_rate_usd,
        local_started_at: record.local_started_at,
        last_seen_at: record.last_seen_at,
        local_stopped_at: record.local_stopped_at,
        duration_seconds,
        duration_hours,
        duration_hms: format_duration_hms(duration_seconds),
        local_estimated_cost_usd,
        cost_basis: "local timestamps x configured hourly USD; DigitalOcean usage API not used"
            .to_string(),
    }
}

pub fn list_summaries() -> Result<Vec<DropletUsageSummary>> {
    let _guard = USAGE_LOCK.lock();
    let records = load_records_sync()?;
    let now = Utc::now();
    write_csv_sync(&records, now)?;
    let mut summaries: Vec<_> = records
        .iter()
        .map(|record| summarize(record, now))
        .collect();
    summaries.sort_by(|a, b| b.local_started_at.cmp(&a.local_started_at));
    Ok(summaries)
}

pub fn export_csv() -> Result<String> {
    let _guard = USAGE_LOCK.lock();
    let records = load_records_sync()?;
    let path = write_csv_sync(&records, Utc::now())?;
    Ok(path.to_string_lossy().into_owned())
}

fn csv_escape(raw: &str) -> String {
    if raw.contains(',') || raw.contains('"') || raw.contains('\n') || raw.contains('\r') {
        format!("\"{}\"", raw.replace('"', "\"\""))
    } else {
        raw.to_string()
    }
}

fn csv_row(fields: &[String]) -> String {
    fields
        .iter()
        .map(|field| csv_escape(field))
        .collect::<Vec<_>>()
        .join(",")
}

fn format_optional_rate(value: Option<f64>) -> String {
    value.map(|rate| format!("{rate:.4}")).unwrap_or_default()
}

fn format_optional_cost(value: Option<f64>) -> String {
    value.map(|cost| format!("{cost:.4}")).unwrap_or_default()
}

fn format_optional_time(value: Option<DateTime<Utc>>) -> String {
    value.map(|time| time.to_rfc3339()).unwrap_or_default()
}

fn usage_csv(records: &[DropletUsageRecord], now: DateTime<Utc>) -> String {
    let mut lines = Vec::new();
    lines.push(csv_row(&[
        "recordType".to_string(),
        "dropletId".to_string(),
        "name".to_string(),
        "sizeSlug".to_string(),
        "region".to_string(),
        "ipAddress".to_string(),
        "status".to_string(),
        "source".to_string(),
        "localStartedAt".to_string(),
        "localStoppedAt".to_string(),
        "lastSeenAt".to_string(),
        "hourlyRateUsd".to_string(),
        "durationSeconds".to_string(),
        "durationHms".to_string(),
        "durationHours".to_string(),
        "localEstimatedCostUsd".to_string(),
        "costBasis".to_string(),
    ]));

    let mut total_seconds = 0_i64;
    let mut total_cost = 0.0_f64;
    let mut all_costs_known = true;

    for record in records {
        let summary = summarize(record, now);
        total_seconds += summary.duration_seconds;
        match summary.local_estimated_cost_usd {
            Some(cost) => total_cost += cost,
            None => all_costs_known = false,
        }
        lines.push(csv_row(&[
            "droplet".to_string(),
            summary.droplet_id.to_string(),
            summary.name,
            summary.size_slug,
            summary.region,
            summary.ip_address,
            summary.status,
            summary.source,
            summary.local_started_at.to_rfc3339(),
            format_optional_time(summary.local_stopped_at),
            format_optional_time(summary.last_seen_at),
            format_optional_rate(summary.hourly_rate_usd),
            summary.duration_seconds.to_string(),
            summary.duration_hms,
            format!("{:.6}", summary.duration_hours),
            format_optional_cost(summary.local_estimated_cost_usd),
            summary.cost_basis,
        ]));
    }

    lines.push(csv_row(&[
        "total".to_string(),
        String::new(),
        "ALL_DROPLETS".to_string(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        "local_total".to_string(),
        String::new(),
        String::new(),
        now.to_rfc3339(),
        String::new(),
        total_seconds.to_string(),
        format_duration_hms(total_seconds),
        format!("{:.6}", total_seconds as f64 / 3600.0),
        if all_costs_known {
            format!("{total_cost:.4}")
        } else {
            String::new()
        },
        "Sum of local droplet rows; DigitalOcean usage API not used".to_string(),
    ]));

    let mut csv = lines.join("\n");
    csv.push('\n');
    csv
}
