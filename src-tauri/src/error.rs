use serde::{Serialize, Serializer};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("ssh error: {0}")]
    Ssh(String),

    #[error("qdrant error: {0}")]
    Qdrant(String),

    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serde error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("yaml error: {0}")]
    Yaml(#[from] serde_yaml::Error),

    #[error("pipeline error: {0}")]
    Pipeline(String),

    #[error("config error: {0}")]
    Config(String),

    #[error("not found: {0}")]
    NotFound(String),

    #[error("cancelled")]
    Cancelled,

    #[error("{0}")]
    Other(String),
}

impl AppError {
    pub fn other<S: Into<String>>(msg: S) -> Self {
        Self::Other(msg.into())
    }
    pub fn pipeline<S: Into<String>>(msg: S) -> Self {
        Self::Pipeline(msg.into())
    }
    pub fn ssh<S: Into<String>>(msg: S) -> Self {
        Self::Ssh(msg.into())
    }
    pub fn qdrant<S: Into<String>>(msg: S) -> Self {
        Self::Qdrant(msg.into())
    }
    pub fn config<S: Into<String>>(msg: S) -> Self {
        Self::Config(msg.into())
    }
}

impl Serialize for AppError {
    fn serialize<S>(&self, s: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        s.serialize_str(&self.to_string())
    }
}

impl From<anyhow::Error> for AppError {
    fn from(e: anyhow::Error) -> Self {
        AppError::Other(e.to_string())
    }
}

impl From<russh::Error> for AppError {
    fn from(e: russh::Error) -> Self {
        AppError::Ssh(e.to_string())
    }
}

impl From<russh_keys::Error> for AppError {
    fn from(e: russh_keys::Error) -> Self {
        AppError::Ssh(format!("keys: {}", e))
    }
}

pub type Result<T> = std::result::Result<T, AppError>;
