use crate::config::QdrantConfig;
use crate::error::{AppError, Result};
use crate::ssh::SshSessionManager;
use base64::Engine;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use uuid::Uuid;
use once_cell::sync::Lazy;
use tokio::sync::Semaphore;

static REMOTE_OCR_SEMAPHORE: Lazy<Semaphore> = Lazy::new(|| Semaphore::new(2));

pub const CHUNK_SIZE: usize = 800;
pub const CHUNK_OVERLAP: usize = 150;
pub const MAX_CHARS_PER_EMBED: usize = 8000;
/// Fallback vector dimension used only when the model cannot be probed.
pub const TARGET_VECTOR_DIM: usize = 4096;
pub const UPSERT_BATCH_SIZE: usize = 20;
pub const EMBED_CONCURRENCY: usize = 2;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct PaddleOcrOptions {
    pub enabled: bool,
    pub host: String,
    pub port: u16,
    pub model_name: String,
}

impl Default for PaddleOcrOptions {
    fn default() -> Self {
        Self {
            enabled: false,
            host: String::new(),
            port: 8118,
            model_name: "PaddleOCR-VL-1.6-0.9B".to_string(),
        }
    }
}

pub fn normalize_vector(vec: &mut Vec<f32>) {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingProvider {
    Vllm,
    Ollama,
    Llamacpp,
}

impl Default for EmbeddingProvider {
    fn default() -> Self {
        EmbeddingProvider::Vllm
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingConfig {
    pub provider: EmbeddingProvider,
    #[serde(default)]
    pub api_url: String,
    #[serde(default)]
    pub api_key: String,
    #[serde(default)]
    pub model_id: String,
    #[serde(default)]
    pub concurrency: Option<usize>,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        EmbeddingConfig {
            provider: EmbeddingProvider::Vllm,
            api_url: String::new(),
            api_key: String::new(),
            model_id: "Qwen/Qwen3-Embedding-8B".to_string(),
            concurrency: None,
        }
    }
}

const ID_NAMESPACE: Uuid = Uuid::from_bytes([
    0x6f, 0xe7, 0x1a, 0x2c, 0x49, 0x8d, 0x4b, 0x71,
    0x9a, 0x10, 0xc5, 0x3b, 0xd2, 0x88, 0x12, 0x55,
]);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestOptions {
    #[serde(default)]
    pub tag: Option<String>,
    #[serde(default)]
    pub chunk_size: Option<usize>,
    #[serde(default)]
    pub chunk_overlap: Option<usize>,
    pub vector_dim: Option<usize>,
}

impl Default for IngestOptions {
    fn default() -> Self {
        Self { tag: None, chunk_size: None, chunk_overlap: None, vector_dim: None }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileResult {
    pub file_path: String,
    pub file_name: String,
    pub chunks_ingested: u64,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestSummary {
    pub total_files: u64,
    pub total_chunks: u64,
    pub files: Vec<FileResult>,
    pub cancelled: bool,
    pub detected_dim: Option<usize>,
}

pub type ProgressFn = Box<dyn Fn(&str, &str, u64, u64) + Send + Sync>;

pub fn read_file_text(path: &Path) -> Result<String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "txt" | "md" => {
            std::fs::read_to_string(path).map_err(|e| {
                AppError::pipeline(format!("read {} failed: {}", path.display(), e))
            })
        }
        "pdf" => {
            let p = path.to_path_buf();
            let result = std::panic::catch_unwind(move || pdf_extract::extract_text(&p));
            match result {
                Ok(Ok(s)) if !s.trim().is_empty() => Ok(s),
                Ok(Ok(_)) => Err(AppError::pipeline(format!(
                    "pdf extract {} returned empty text (likely scanned/image-based PDF — enable PaddleOCR in Settings)",
                    path.display()
                ))),
                Ok(Err(e)) => Err(AppError::pipeline(format!(
                    "pdf extract {} failed: {}", path.display(), e
                ))),
                Err(_) => Err(AppError::pipeline(format!(
                    "pdf extract {} panicked (likely scanned/malformed — enable PaddleOCR in Settings)", path.display()
                ))),
            }
        }
        "docx" => {
            docx_lite::extract_text(path).map_err(|e| {
                AppError::pipeline(format!("docx extract {} failed: {}", path.display(), e))
            })
        }
        "pptx" | "ppt" => {
            let python_script = r#"
import sys
import os
import tempfile

def parse_pptx_file(pptx_path):
    from pptx import Presentation
    prs = Presentation(pptx_path)
    text_runs = []
    for i, slide in enumerate(prs.slides):
        slide_text = []
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                slide_text.append(shape.text.strip())
            elif shape.has_table:
                for row in shape.table.rows:
                    row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if row_text:
                        slide_text.append(" | ".join(row_text))
        try:
            if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                notes = slide.notes_slide.notes_text_frame.text.strip()
                if notes:
                    slide_text.append(f"Notes: {notes}")
        except Exception:
            pass
        if slide_text:
            text_runs.append(f"--- Slide {i+1} ---\n" + "\n".join(slide_text))
    return "\n\n".join(text_runs)

try:
    file_path = sys.argv[1]
    ext = os.path.splitext(file_path)[1].lower()
    temp_pptx = None
    
    if ext == ".ppt":
        try:
            import win32com.client
        except ImportError:
            win32com = None

        if win32com is not None:
            powerpoint = win32com.client.Dispatch("PowerPoint.Application")
            try:
                abs_ppt = os.path.abspath(file_path)
                fd, temp_pptx = tempfile.mkstemp(suffix=".pptx")
                os.close(fd)
                
                presentation = powerpoint.Presentations.Open(abs_ppt, WithWindow=False)
                presentation.SaveAs(temp_pptx, 24) # 24 = ppSaveAsOpenXMLPresentation
                presentation.Close()
                parse_target = temp_pptx
            finally:
                powerpoint.Quit()
        else:
            import subprocess
            fd, temp_pptx = tempfile.mkstemp(suffix=".pptx")
            os.close(fd)
            try:
                temp_dir = tempfile.gettempdir()
                os.remove(temp_pptx)
                result = subprocess.run([
                    "soffice", "--headless", "--convert-to", "pptx",
                    "--outdir", temp_dir, file_path
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                basename = os.path.basename(file_path)
                stem = os.path.splitext(basename)[0]
                expected_pptx = os.path.join(temp_dir, stem + ".pptx")
                
                if result.returncode == 0 and os.path.exists(expected_pptx):
                    temp_pptx = expected_pptx
                    parse_target = temp_pptx
                else:
                    raise RuntimeError("soffice conversion failed")
            except Exception as e:
                raise RuntimeError(
                    f"Legacy .ppt file support requires win32com on Windows or LibreOffice 'soffice' on Mac/Linux: {e}"
                )
    else:
        parse_target = file_path
        
    text = parse_pptx_file(parse_target)
    
    if temp_pptx and os.path.exists(temp_pptx):
        try:
            os.remove(temp_pptx)
        except Exception:
            pass
            
    sys.stdout.buffer.write(text.encode('utf-8'))
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
"#;
            let output = std::process::Command::new("python")
                .args(&[
                    "-c",
                    python_script,
                    path.to_str().unwrap_or_default(),
                ])
                .output();

            match output {
                Ok(out) => {
                    if out.status.success() {
                        let text = String::from_utf8_lossy(&out.stdout).to_string();
                        Ok(text)
                    } else {
                        let err_msg = String::from_utf8_lossy(&out.stderr).to_string();
                        Err(AppError::pipeline(format!("pptx/ppt extract {} failed: {}", path.display(), err_msg)))
                    }
                }
                Err(e) => {
                    Err(AppError::pipeline(format!(
                        "pptx/ppt extract {} failed: python runner failed to start (is python installed and on PATH?): {}",
                        path.display(),
                        e
                    )))
                }
            }
        }
        other => Err(AppError::pipeline(format!(
            "unsupported file extension '{}' for {}", other, path.display()
        ))),
    }
}

fn is_image_extension(ext: &str) -> bool {
    matches!(
        ext,
        "png" | "jpg" | "jpeg" | "webp" | "gif" | "bmp" | "tiff" | "tif"
    )
}

pub async fn read_file_text_with_ocr(
    path: &Path,
    ocr: &PaddleOcrOptions,
    ssh: Option<&SshSessionManager>,
    on_progress: &ProgressFn,
) -> Result<String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();

    if is_image_extension(&ext) {
        if ocr.enabled {
            return ocr_extract_file(path, ocr, ssh, on_progress).await;
        } else {
            return Err(AppError::pipeline(format!(
                "Image file {} cannot be read because PaddleOCR is disabled in Settings",
                path.display()
            )));
        }
    }

    // Try standard text extraction first
    let standard_res = read_file_text(path);
    match standard_res {
        Ok(text) => {
            let trimmed = text.trim();
            // If the text is empty or very short (< 120 characters) and it's a PDF and OCR is enabled, fallback to OCR
            if trimmed.len() < 120 && ocr.enabled && ext == "pdf" {
                match ocr_extract_file(path, ocr, ssh, on_progress).await {
                    Ok(ocr_text) => {
                        if !ocr_text.trim().is_empty() {
                            return Ok(ocr_text);
                        }
                    }
                    Err(e) => {
                        println!("OCR fallback failed for short PDF (using standard text): {:?}", e);
                    }
                }
            }
            Ok(text)
        }
        Err(e) => {
            // Standard extraction failed. If OCR is enabled and it's a PDF, try OCR.
            if ocr.enabled && ext == "pdf" {
                ocr_extract_file(path, ocr, ssh, on_progress).await
            } else {
                Err(e)
            }
        }
    }
}

/// Public single-image OCR helper used by the robot capture pipeline. Reuses
/// the same PaddleOCR-VL path as document ingestion. `ssh` should be `Some`
/// when the OCR server is reached through the GPU droplet (the usual case).
pub async fn ocr_image(
    path: &Path,
    ocr: &PaddleOcrOptions,
    ssh: Option<&SshSessionManager>,
) -> Result<String> {
    let noop: ProgressFn = Box::new(|_, _, _, _| {});
    ocr_extract_file(path, ocr, ssh, &noop).await
}

async fn ocr_extract_file(
    path: &Path,
    ocr: &PaddleOcrOptions,
    ssh: Option<&SshSessionManager>,
    on_progress: &ProgressFn,
) -> Result<String> {
    let Some(mgr) = ssh else {
        // Local Case: reject PDF, allow image via local HTTP request
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_default();
        let mime_type = match ext.as_str() {
            "pdf" => "application/pdf",
            "png" => "image/png",
            "jpg" | "jpeg" => "image/jpeg",
            "webp" => "image/webp",
            "gif" => "image/gif",
            "bmp" => "image/bmp",
            "tiff" | "tif" => "image/tiff",
            _ => "application/octet-stream",
        };
        if ext == "pdf" {
            return Err(AppError::pipeline(
                "Local PDF OCR is not supported (PDF page splitting requires remote GPU server with docker)".to_string()
            ));
        }

        let file_data = tokio::fs::read(path).await.map_err(|e| {
            AppError::pipeline(format!("read file bytes {}: {}", path.display(), e))
        })?;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&file_data);
        let url = format!("http://{}:{}/v1/chat/completions", ocr.host, ocr.port);
        let body = json!({
            "model": ocr.model_name,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "image_url",
                    "image_url": {
                        "url": format!("data:{};base64,{}", mime_type, b64)
                    }
                }]
            }]
        });
        let res = http().post(&url).json(&body).send().await.map_err(|e| {
            AppError::pipeline(format!("PaddleOCR HTTP request failed: {}", e))
        })?;
        if !res.status().is_success() {
            let s = res.status();
            let txt = res.text().await.unwrap_or_default();
            return Err(AppError::pipeline(format!("PaddleOCR HTTP {}: {}", s, txt)));
        }
        let v: Value = res.json().await.map_err(|e| {
            AppError::pipeline(format!("PaddleOCR response parse: {}", e))
        })?;
        let text = v
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_string();
        if text.is_empty() {
            return Err(AppError::pipeline(format!(
                "PaddleOCR returned empty text for {}", path.display()
            )));
        }
        return Ok(text);
    };

    // Remote Case: retry loop with auto-reconnect
    let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
    on_progress("ocr_start", &file_name, 0, 0);

    // Acquire OCR semaphore permit to limit concurrent remote OCR processing
    let _permit = REMOTE_OCR_SEMAPHORE.acquire().await.map_err(|e| {
        AppError::pipeline(format!("Failed to acquire OCR semaphore permit: {}", e))
    })?;

    let mut last_err = None;
    for attempt in 1..=3 {
        if attempt > 1 {
            on_progress("ocr_start", &format!("{} (Retry {})", file_name, attempt - 1), 0, 0);
        }
        match ocr_extract_file_inner(path, ocr, mgr, on_progress).await {
            Ok(text) => return Ok(text),
            Err(e) => {
                println!("OCR attempt {} failed for {}: {}. Retrying...", attempt, path.display(), e);
                let err_str = e.to_string();
                if err_str.contains("ssh error") || err_str.contains("Channel") || err_str.contains("Disconnected") || err_str.contains("timeout") {
                    mgr.clear_session().await;
                }
                last_err = Some(e);
                if attempt < 3 {
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                }
            }
        }
    }
    Err(last_err.unwrap_or_else(|| AppError::pipeline("OCR failed after retries")))
}

async fn ocr_extract_file_inner(
    path: &Path,
    ocr: &PaddleOcrOptions,
    mgr: &SshSessionManager,
    on_progress: &ProgressFn,
) -> Result<String> {
    let session = mgr.get_session().await?;
    let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("doc.dat");
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();

    let mime_type = match ext.as_str() {
        "pdf" => "application/pdf",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "gif" => "image/gif",
        "bmp" => "image/bmp",
        "tiff" | "tif" => "image/tiff",
        _ => "application/octet-stream",
    };

    let uuid = Uuid::new_v4().to_string();
    let remote_dir = format!("/tmp/paddleocr_ingest/{}", uuid);
    let remote_path = format!("{}/{}", remote_dir, file_name);

    let data = tokio::fs::read(path).await.map_err(|e| {
        AppError::pipeline(format!("read file bytes {}: {}", path.display(), e))
    })?;
    let b64 = base64::engine::general_purpose::STANDARD.encode(&data);
    
    // Ensure remote directory exists
    let mkdir_cmd = format!("mkdir -p \"{}\"", remote_dir);
    session.exec_blocking(&mkdir_cmd).await.map_err(|e| {
        AppError::pipeline(format!("create remote ocr temp dir: {}", e))
    })?;

    // Upload original file to remote server
    if b64.len() < 65_536 {
        let write_cmd = format!("echo '{}' | base64 -d > \"{}\"", b64, remote_path);
        session.exec_blocking(&write_cmd).await.map_err(|e| {
            AppError::pipeline(format!("upload file to GPU server: {}", e))
        })?;
    } else {
        let temp_b64_path = format!("{}.b64", remote_path);
        session.write_file(&temp_b64_path, &b64).await.map_err(|e| {
            AppError::pipeline(format!("write_file base64 to GPU server: {}", e))
        })?;
        let decode_cmd = format!("base64 -d < \"{}\" > \"{}\" && rm -f \"{}\"", temp_b64_path, remote_path, temp_b64_path);
        session.exec_blocking(&decode_cmd).await.map_err(|e| {
            AppError::pipeline(format!("decode base64 on GPU server: {}", e))
        })?;
    }

    let container_name = "paddleocr-vl";

    if ext == "pdf" {
        // PDF Case: Python page rendering + OCR inside container
        let python_script = r#"import sys
import os
import re
import base64
import requests
import pypdfium2
import time
from io import BytesIO

LOC_TOKEN_RE = re.compile(r'<\|LOC_\d+\|>')

def clean_ocr_text(text):
    """Remove PaddleOCR positional tokens and normalize whitespace."""
    text = LOC_TOKEN_RE.sub('', text)
    # Collapse multiple blank lines into one
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def ocr_pdf(pdf_path, port, model_name):
    if not os.path.exists(pdf_path):
        print(f"Error: file {pdf_path} not found", file=sys.stderr)
        sys.exit(1)
    try:
        pdf = pypdfium2.PdfDocument(pdf_path)
    except Exception as e:
        print(f"Error opening PDF: {e}", file=sys.stderr)
        sys.exit(1)

    extracted_texts = []
    for page_number in range(len(pdf)):
        try:
            page = pdf.get_page(page_number)
            # render() returns a PdfBitmap; use rev_byteorder for RGB
            bitmap = page.render(scale=2.0, rev_byteorder=True)
            pil_image = bitmap.to_pil().convert('RGB')
            buffered = BytesIO()
            pil_image.save(buffered, format="JPEG", quality=80)
            img_bytes = buffered.getvalue()
            b64_str = base64.b64encode(img_bytes).decode('utf-8')
            page.close()
        except Exception as e:
            print(f"Error rendering page {page_number}: {e}", file=sys.stderr)
            continue

        url = f"http://127.0.0.1:{port}/v1/chat/completions"
        payload = {
            "model": model_name,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_str}"
                    }
                }]
            }]
        }
        
        page_success = False
        for page_attempt in range(1, 4):
            try:
                res = requests.post(url, json=payload, timeout=120)
                if res.status_code == 200:
                    data = res.json()
                    content = data['choices'][0]['message']['content']
                    cleaned = clean_ocr_text(content)
                    if cleaned:
                        extracted_texts.append(cleaned)
                    print(f"PAGE_DONE: {page_number + 1} / {len(pdf)}", file=sys.stderr, flush=True)
                    page_success = True
                    break
                else:
                    print(f"Warning: OCR page {page_number} attempt {page_attempt} failed with status {res.status_code}: {res.text}", file=sys.stderr)
            except Exception as e:
                print(f"Warning: OCR request failed for page {page_number} attempt {page_attempt}: {e}", file=sys.stderr)
            if page_attempt < 3:
                time.sleep(2)
        if not page_success:
            print(f"Error: Page {page_number} failed after 3 attempts.", file=sys.stderr)

    pdf.close()
    result = "\n\n".join(extracted_texts)
    print(result)

if __name__ == "__main__":
    ocr_pdf(sys.argv[1], sys.argv[2], sys.argv[3])"#;

        let script_path = format!("{}/pdf_ocr.py", remote_dir);
        session.write_file(&script_path, python_script).await.map_err(|e| {
            AppError::pipeline(format!("write pdf_ocr.py to GPU server: {}", e))
        })?;

        // Copy PDF and script into container using unique directory to avoid concurrent overrides
        session.exec_blocking(&format!("docker exec {} mkdir -p \"/tmp/paddleocr_ingest/{}\"", container_name, uuid)).await?;
        session.exec_blocking(&format!("docker cp \"{}\" \"{}:/tmp/paddleocr_ingest/{}/{}\"", remote_path, container_name, uuid, file_name)).await?;
        session.exec_blocking(&format!("docker cp \"{}\" \"{}:/tmp/paddleocr_ingest/{}/pdf_ocr.py\"", script_path, container_name, uuid)).await?;

        // Execute script
        let run_script_cmd = format!(
            "docker exec {} python3 \"/tmp/paddleocr_ingest/{}/pdf_ocr.py\" \"/tmp/paddleocr_ingest/{}/{}\" {} {}",
            container_name, uuid, uuid, file_name, ocr.port, ocr.model_name
        );
        
        let mut line_buffer = String::new();
        let r = session.exec_collect_stderr(&run_script_cmd, |data| {
            if let Ok(s) = std::str::from_utf8(data) {
                line_buffer.push_str(s);
                while let Some(pos) = line_buffer.find('\n') {
                    let line = line_buffer[..pos].trim().to_string();
                    line_buffer = line_buffer[pos + 1..].to_string();
                    if line.starts_with("PAGE_DONE:") {
                        let parts: Vec<&str> = line["PAGE_DONE:".len()..].split('/').map(|p| p.trim()).collect();
                        if parts.len() == 2 {
                            if let (Ok(curr), Ok(tot)) = (parts[0].parse::<u64>(), parts[1].parse::<u64>()) {
                                on_progress("ocr_page", file_name, curr, tot);
                            }
                        }
                    }
                }
            }
        }).await.map_err(|e| AppError::ssh(e.to_string()))?;

        // Clean up files in container and on host
        let _ = session.exec_blocking(&format!("docker exec {} rm -rf \"/tmp/paddleocr_ingest/{}\"", container_name, uuid)).await;
        let _ = session.exec_blocking(&format!("rm -rf \"{}\"", remote_dir)).await;

        if r.exit_code != 0 {
            return Err(AppError::pipeline(format!(
                "PDF OCR script failed (exit {}): {}", r.exit_code, r.stderr
            )));
        }

        let text = r.stdout.trim().to_string();
        if text.is_empty() {
            return Err(AppError::pipeline(format!(
                "PaddleOCR returned empty text for PDF {}", path.display()
            )));
        }
        Ok(text)
    } else {
        // Image Case: Write JSON request to a temporary file on the host to avoid ARG_MAX issues
        let ocr_url = format!("http://127.0.0.1:{}/v1/chat/completions", ocr.port);
        let req_json_path = format!("{}/ocr_req.json", remote_dir);
        let body = json!({
            "model": ocr.model_name,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "image_url",
                    "image_url": {
                        "url": format!("data:{};base64,{}", mime_type, b64)
                    }
                }]
            }]
        });

        session.write_file(&req_json_path, &body.to_string()).await.map_err(|e| {
            AppError::pipeline(format!("write ocr request body to GPU server: {}", e))
        })?;

        let curl_cmd = format!(
            "curl -s -X POST '{}' -H 'Content-Type: application/json' -d @\"{}\"",
            ocr_url,
            req_json_path
        );
        let r = session.exec_blocking(&curl_cmd).await.map_err(|e| {
            AppError::pipeline(format!("PaddleOCR OCR request on GPU server: {}", e))
        })?;

        // Clean up host temp files
        let _ = session.exec_blocking(&format!("rm -rf \"{}\"", remote_dir)).await;

        if r.exit_code != 0 {
            return Err(AppError::pipeline(format!(
                "PaddleOCR curl failed (exit {}): {}", r.exit_code, r.stderr
            )));
        }

        let v: Value = serde_json::from_str(&r.stdout).map_err(|e| {
            AppError::pipeline(format!("PaddleOCR response parse error: {}, body was: {}", e, r.stdout))
        })?;

        let text = v
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_string();

        if text.is_empty() {
            return Err(AppError::pipeline(format!(
                "PaddleOCR returned empty text for {}", path.display()
            )));
        }
        Ok(text)
    }
}

pub fn chunk_text(text: &str, size: usize, overlap: usize) -> Vec<String> {
    let cleaned = collapse_whitespace(text);
    let chars: Vec<char> = cleaned.chars().collect();
    if chars.is_empty() {
        return vec![];
    }
    let step = size.saturating_sub(overlap).max(1);
    let mut out = Vec::with_capacity(chars.len() / step + 1);
    let mut start = 0;
    while start < chars.len() {
        let end = (start + size).min(chars.len());
        let slice: String = chars[start..end].iter().collect();
        let trimmed = slice.trim();
        if !trimmed.is_empty() {
            out.push(trimmed.to_string());
        }
        if end == chars.len() {
            break;
        }
        start += step;
    }
    out
}

fn collapse_whitespace(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut last_ws = false;
    for c in s.chars() {
        if c.is_whitespace() {
            if !last_ws {
                out.push(' ');
                last_ws = true;
            }
        } else {
            out.push(c);
            last_ws = false;
        }
    }
    out
}

fn http() -> Client {
    Client::builder()
        .timeout(std::time::Duration::from_secs(90))
        .build()
        .unwrap()
}

fn parse_embedding_response(v: Value, _expected_dim: usize) -> Result<Vec<f32>> {
    let arr = v
        .get("data")
        .and_then(|d| d.get(0))
        .and_then(|d0| d0.get("embedding"))
        .and_then(|e| e.as_array())
        .ok_or_else(|| AppError::qdrant("embed: no data[0].embedding in response"))?;
    let mut vec: Vec<f32> = arr
        .iter()
        .filter_map(|x| x.as_f64().map(|f| f as f32))
        .collect();
    if vec.is_empty() {
        return Err(AppError::qdrant("embed: empty embedding vector"));
    }
    normalize_vector(&mut vec);
    Ok(vec)
}

pub async fn embed_chunk(config: &EmbeddingConfig, text: &str) -> Result<Vec<f32>> {
    let input: String = text.chars().take(MAX_CHARS_PER_EMBED).collect();
    match config.provider {
        EmbeddingProvider::Vllm => {
            let url = format!("{}/v1/embeddings", config.api_url.trim_end_matches('/'));
            let model_name = if config.model_id.is_empty() {
                "default"
            } else {
                &config.model_id
            };
            let body = json!({ "model": model_name, "input": input });
            let res = http()
                .post(&url)
                .json(&body)
                .send()
                .await?;
            if !res.status().is_success() {
                let s = res.status();
                let txt = res.text().await.unwrap_or_default();
                return Err(AppError::qdrant(format!("vllm embed http {s}: {txt}")));
            }
            let v: Value = res.json().await?;
            parse_embedding_response(v, TARGET_VECTOR_DIM)
        }
        EmbeddingProvider::Ollama => {
            let url = format!("{}/api/embeddings", config.api_url.trim_end_matches('/'));
            let body = json!({ "model": config.model_id, "prompt": input });
            let res = http().post(&url).json(&body).send().await?;
            if !res.status().is_success() {
                let s = res.status();
                let txt = res.text().await.unwrap_or_default();
                return Err(AppError::qdrant(format!("ollama embed http {s}: {txt}")));
            }
            let v: Value = res.json().await?;
            let arr = v.get("embedding")
                .and_then(|e| e.as_array())
                .ok_or_else(|| AppError::qdrant("ollama: no embedding in response"))?;
            let mut vec: Vec<f32> = arr
                .iter()
                .filter_map(|x| x.as_f64().map(|f| f as f32))
                .collect();
            if vec.is_empty() {
                return Err(AppError::qdrant("ollama: empty embedding vector"));
            }
            normalize_vector(&mut vec);
            Ok(vec)
        }
        EmbeddingProvider::Llamacpp => {
            let url = format!("{}/v1/embeddings", config.api_url.trim_end_matches('/'));
            let model_name = if config.model_id.is_empty() { "default" } else { &config.model_id };
            let body = json!({ "model": model_name, "input": input });
            let res = http().post(&url).json(&body).send().await?;
            if !res.status().is_success() {
                let s = res.status();
                let txt = res.text().await.unwrap_or_default();
                return Err(AppError::qdrant(format!("llamacpp embed http {s}: {txt}")));
            }
            let v: Value = res.json().await?;
            parse_embedding_response(v, TARGET_VECTOR_DIM)
        }
    }
}

async fn embed_chunk_retrying(config: &EmbeddingConfig, text: &str) -> Result<Vec<f32>> {
    let mut last_err: Option<AppError> = None;
    for attempt in 0..3 {
        match embed_chunk(config, text).await {
            Ok(v) => return Ok(v),
            Err(e) => {
                last_err = Some(e);
                let backoff_ms = 500u64 * (1u64 << attempt);
                tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
            }
        }
    }
    Err(last_err.unwrap_or_else(|| AppError::qdrant("embed: unknown failure")))
}

fn deterministic_point_id(file_path: &str, chunk_index: usize) -> String {
    let key = format!("{}::{}", file_path, chunk_index);
    Uuid::new_v5(&ID_NAMESPACE, key.as_bytes()).to_string()
}

fn sanitize_filename_to_tag(filename: &str) -> String {
    let stem = std::path::Path::new(filename)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(filename);
    let mut sanitized = String::new();
    for c in stem.chars() {
        if c.is_alphanumeric() {
            sanitized.push(c.to_ascii_lowercase());
        } else if sanitized.ends_with('_') {
            // Avoid multiple underscores
        } else {
            sanitized.push('_');
        }
    }
    let trimmed = sanitized.trim_matches('_').to_string();
    if trimmed.is_empty() {
        "auto_doc".to_string()
    } else {
        trimmed
    }
}

async fn upsert_batch_to_collection(
    endpoint: &str,
    api_key: &str,
    collection: &str,
    points: Vec<Value>,
) -> Result<()> {
    if points.is_empty() {
        return Ok(());
    }
    let url = format!(
        "{}/collections/{}/points?wait=true",
        endpoint.trim_end_matches('/'),
        collection
    );
    let body = json!({ "points": points });
    let res = http()
        .put(url)
        .header("api-key", api_key)
        .json(&body)
        .send()
        .await?;
    if !res.status().is_success() {
        let s = res.status();
        let txt = res.text().await.unwrap_or_default();
        return Err(AppError::qdrant(format!("upsert failed {s}: {txt}")));
    }
    Ok(())
}

pub async fn ingest_files(
    files: Vec<String>,
    qdrant: QdrantConfig,
    embedding_config: EmbeddingConfig,
    opts: IngestOptions,
    cancel: Arc<AtomicBool>,
    on_progress: ProgressFn,
    ocr: PaddleOcrOptions,
    ssh: Option<Arc<SshSessionManager>>,
) -> Result<IngestSummary> {
    use futures::stream::{self, StreamExt};

    if qdrant.endpoint.is_empty() {
        return Err(AppError::qdrant("ingest: qdrant endpoint not set"));
    }
    let collection = if !qdrant.collection.is_empty() {
        qdrant.collection.clone()
    } else {
        "kb_default".to_string()
    };

    let chunk_size = opts.chunk_size.unwrap_or(CHUNK_SIZE);
    let chunk_overlap = opts.chunk_overlap.unwrap_or(CHUNK_OVERLAP);
    let tag = opts.tag.clone();
    let endpoint = qdrant.endpoint.trim_end_matches('/').to_string();
    let api_key = qdrant.api_key.clone();

    // Probe the embedding model to get the actual vector dimension before creating the collection.
    // This is critical because Qwen3-Embedding-8B produces 4096-dim vectors, not 1536.
    let probe_dim: usize = if let Some(d) = opts.vector_dim {
        d
    } else {
        on_progress("read", "Probing embedding model dimension...", 0, 0);
        match embed_chunk(&embedding_config, "dimension probe").await {
            Ok(v) => v.len(),
            Err(e) => {
                on_progress("warn", &format!("Could not probe embedding dim (using {TARGET_VECTOR_DIM}): {e}"), 0, 0);
                TARGET_VECTOR_DIM
            }
        }
    };
    let detected_dim: Option<usize> = Some(probe_dim);

    crate::qdrant::create_collection(&qdrant, &collection, probe_dim).await?;

    if let Err(e) = crate::qdrant::ensure_text_index(&qdrant, &collection, "tag").await {
        on_progress("warn", &format!("payload index for 'tag' not created: {e}"), 0, 0);
    }

    let total_files = files.len() as u64;
    let cancelled_flag = Arc::new(AtomicBool::new(false));
    let on_progress_arc = Arc::new(on_progress);

    // Reuse widget concurrency for file ingestion concurrency
    let file_concurrency = embedding_config.concurrency.unwrap_or(4).max(1);

    let results: Vec<FileResult> = stream::iter(files.into_iter())
        .map(|path_str| {
            let path = Path::new(&path_str).to_path_buf();
            let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
            let file_path_str = path.to_string_lossy().to_string();

            let ocr = ocr.clone();
            let ssh = ssh.clone();
            let embedding_config = embedding_config.clone();
            let tag = tag.clone();
            let endpoint = endpoint.clone();
            let api_key = api_key.clone();
            let collection = collection.clone();
            let cancel = cancel.clone();
            let cancelled_flag = cancelled_flag.clone();
            let on_progress = on_progress_arc.clone();

            async move {
                if cancel.load(Ordering::SeqCst) || cancelled_flag.load(Ordering::SeqCst) {
                    cancelled_flag.store(true, Ordering::SeqCst);
                    return FileResult {
                        file_path: file_path_str,
                        file_name,
                        chunks_ingested: 0,
                        error: Some("cancelled".to_string()),
                    };
                }

                on_progress("read", &file_name, 0, 0);

                let text = match read_file_text_with_ocr(&path, &ocr, ssh.as_ref().map(|s| s.as_ref()), &on_progress).await {
                    Ok(t) => t,
                    Err(e) => {
                        on_progress("error", &file_name, 0, 0);
                        return FileResult {
                            file_path: file_path_str,
                            file_name,
                            chunks_ingested: 0,
                            error: Some(e.to_string()),
                        };
                    }
                };

                let chunks = chunk_text(&text, chunk_size, chunk_overlap);
                if chunks.is_empty() {
                    on_progress("error", &file_name, 0, 0);
                    return FileResult {
                        file_path: file_path_str,
                        file_name,
                        chunks_ingested: 0,
                        error: Some("no extractable text (empty after parse)".to_string()),
                    };
                }

                let n_chunks = chunks.len() as u64;
                on_progress("embed", &file_name, 0, n_chunks);

                // Use a standard internal concurrency (e.g. 2) for chunk embedding within a single file
                let chunk_concurrency = EMBED_CONCURRENCY;
                let embed_done = std::sync::atomic::AtomicU64::new(0);
                let embed_done_ref = &embed_done;
                let on_progress_ref = &on_progress;
                let file_name_ref = &file_name;
                let n_chunks_u64 = n_chunks;

                let embedded: Vec<std::result::Result<(usize, String, Vec<f32>), (usize, String, AppError)>> =
                    stream::iter(chunks.into_iter().enumerate().map(|(i, txt)| {
                        let c = embedding_config.clone();
                        async move {
                            let res = embed_chunk_retrying(&c, &txt).await;
                            let prev = embed_done_ref.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                            on_progress_ref("embed", file_name_ref, prev + 1, n_chunks_u64);
                            match res {
                                Ok(v) => Ok((i, txt, v)),
                                Err(e) => Err((i, txt, e)),
                            }
                        }
                    }))
                    .buffer_unordered(chunk_concurrency)
                    .collect()
                    .await;

                if cancel.load(Ordering::SeqCst) || cancelled_flag.load(Ordering::SeqCst) {
                    cancelled_flag.store(true, Ordering::SeqCst);
                    return FileResult {
                        file_path: file_path_str,
                        file_name,
                        chunks_ingested: 0,
                        error: Some("cancelled".to_string()),
                    };
                }

                let mut points: Vec<Value> = Vec::with_capacity(embedded.len());
                let mut embed_errors: Vec<String> = vec![];
                for r in embedded.into_iter() {
                    match r {
                        Ok((i, text, vector)) => {
                            let id = deterministic_point_id(&file_path_str, i);
                            let mut payload = json!({
                                "content": text,
                                "file_path": file_path_str.clone(),
                                "file_name": file_name.clone(),
                                "chunk_index": i as i64,
                            });
                            let file_tag = if let Some(ref t) = tag {
                                if !t.trim().is_empty() {
                                    t.trim().to_string()
                                } else {
                                    sanitize_filename_to_tag(&file_name)
                                }
                            } else {
                                sanitize_filename_to_tag(&file_name)
                            };
                            payload["tag"] = json!(file_tag);
                            points.push(json!({
                                "id": id,
                                "vector": vector,
                                "payload": payload,
                            }));
                        }
                        Err((i, _, e)) => {
                            embed_errors.push(format!("chunk {}: {}", i, e));
                        }
                    }
                }
                let ok_count = points.len() as u64;

                let mut upsert_err: Option<String> = None;
                let mut done: u64 = 0;
                for batch in points.chunks(UPSERT_BATCH_SIZE) {
                    if cancel.load(Ordering::SeqCst) || cancelled_flag.load(Ordering::SeqCst) {
                        cancelled_flag.store(true, Ordering::SeqCst);
                        break;
                    }
                    match upsert_batch_to_collection(&endpoint, &api_key, &collection, batch.to_vec()).await {
                        Ok(()) => {
                            done += batch.len() as u64;
                            on_progress("upsert", &file_name, done, n_chunks);
                        }
                        Err(e) => {
                            upsert_err = Some(e.to_string());
                            break;
                        }
                    }
                }

                let err_msg = if let Some(u) = upsert_err {
                    Some(format!("upsert failed after {} chunks: {}", done, u))
                } else if !embed_errors.is_empty() && done < ok_count {
                    Some(format!("{} chunk embed errors; first: {}", embed_errors.len(), embed_errors[0]))
                } else if !embed_errors.is_empty() {
                    Some(format!("{} chunks failed to embed (rest ingested)", embed_errors.len()))
                } else {
                    None
                };
                on_progress(if err_msg.is_some() { "error" } else { "done" }, &file_name, done, n_chunks);

                FileResult {
                    file_path: file_path_str,
                    file_name,
                    chunks_ingested: done,
                    error: err_msg,
                }
            }
        })
        .buffer_unordered(file_concurrency)
        .collect()
        .await;

    let total_chunks: u64 = results.iter().map(|r| r.chunks_ingested).sum();
    let was_cancelled = cancel.load(Ordering::SeqCst) || cancelled_flag.load(Ordering::SeqCst);

    Ok(IngestSummary {
        total_files,
        total_chunks,
        files: results,
        cancelled: was_cancelled,
        detected_dim,
    })
}