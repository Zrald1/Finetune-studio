//! Library root for the Fine-Tune crate.
//!
//! The desktop Tauri binary (`main.rs`) declares these modules itself for its
//! own compilation root. This `lib.rs` re-exposes the same source files so the
//! headless server binary (`bin/server.rs`) can reuse every piece of business
//! logic — SSH, Qdrant, ingest, pipeline, research, robot intake, manifests —
//! without forking it. One source of truth, two compilation roots (the standard
//! shape for a Tauri app that also ships a headless binary).

pub mod config;
pub mod digitalocean;
pub mod droplet_usage;
pub mod error;
pub mod generator;
pub mod guides;
pub mod hf;
pub mod ingest;
pub mod llamafactory;
pub mod manifest;
pub mod method;
pub mod pipeline;
pub mod qdrant;
pub mod research;
pub mod robot;
pub mod runs;
pub mod serve;
pub mod ssh;
