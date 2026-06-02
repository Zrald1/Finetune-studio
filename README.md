# Fine-Tune Studio

A **Tauri 2 desktop app** that drives the entire LLM fine-tuning pipeline from a single window — no notebooks, no glue scripts, no SSH gymnastics.

```
Qdrant chunks → Teacher LLM (vLLM) → JSONL dataset → LoRA training (LLaMA-Factory) → adapter
```

All heavy ML work runs on a **remote GPU droplet over SSH** (built and tested on an MI300X DigitalOcean instance). The local app is a pure **orchestrator**: it generates the scripts, streams the logs, parses the metrics, and persists run history. Your laptop never needs a GPU.

---

## Table of contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Quick start](#quick-start)
- [Using the app](#using-the-app)
- [Reusing it for any domain or model](#reusing-it-for-any-domain-or-model)
- [Troubleshooting](#troubleshooting)
- [Tech stack](#tech-stack)
- [License](#license)

---

## What it does

Fine-Tune Studio turns a raw knowledge base into a fine-tuned LoRA adapter with a guided, 4-step wizard:

1. **Knowledge Base** — pulls document chunks from a Qdrant collection.
2. **Teacher** — boots a large "teacher" model with vLLM on the remote GPU.
3. **Dataset** — has the teacher generate Q&A pairs from your chunks → a JSONL training set.
4. **Student & Train** — runs LoRA fine-tuning (LLaMA-Factory) on a smaller "student" model.

Throughout, you get **live logs, kept/scanned/rejected counters, a dataset preview, and a live loss curve**. The app automatically unloads the teacher before training so the full GPU VRAM is free for LoRA.

---

## Architecture

```
   React UI            Rust core              Remote GPU droplet
   ────────            ──────────             ─────────────────────
   PipelineWizard ──▶  pipeline.rs   ──SSH──▶ vllm serve <teacher>
   RunDashboard   ◀──  ssh.rs        ─https─▶ Qdrant (knowledge base)
   Live Logs      ◀──  generator.rs
   LoRA chart     ◀──  llamafactory.rs ─SSH─▶ llamafactory-cli train
```

The local app stores config + run history in `%APPDATA%/fine-tune/` (Windows)
or `~/.config/fine-tune/` (Linux). Each run gets its own folder:

```
%APPDATA%/fine-tune/
├── config.json              # SSH, Qdrant, HF token, defaults
└── runs/
    └── 01HXYZ.../           # ULID per run
        ├── qa_dataset.jsonl
        ├── train.jsonl
        ├── val.jsonl
        └── train.yaml
```

The same run folder is mirrored on the droplet at `/root/fine-tune/runs/{id}/`.

> **Note on secrets:** SSH keys, Qdrant API keys, and Hugging Face tokens are entered in the app at runtime and stored locally in `config.json`. **Nothing sensitive is committed to this repo.**

---

## Project structure

```
FineTune/
├── README.md
├── index.html                       # Vite entry point
├── package.json                     # npm scripts + JS deps
├── tsconfig.json
├── vite.config.ts                   # Vite + Tailwind + React config
│
├── src/                             # ── Frontend (React + TypeScript) ──
│   ├── main.tsx                     # React root
│   ├── App.tsx                      # Top-level state, config load/save, tabs
│   ├── types.ts                     # Shared TS types + config defaults
│   ├── index.css                    # Tailwind entry
│   │
│   ├── components/
│   │   ├── PipelineWizard.tsx       # The 4-step fine-tuning wizard
│   │   ├── RunDashboard.tsx         # Run list + detail (logs, progress, loss chart)
│   │   ├── CredentialsPanel.tsx     # SSH / Qdrant / HF token entry + Test SSH
│   │   ├── TrainingConfigForm.tsx   # LoRA hyperparameter form
│   │   ├── DatasetPreview.tsx       # Inline JSONL sample viewer
│   │   ├── GpuServerManager.tsx     # Droplet / GPU server controls
│   │   ├── GPUStatsDashboard.tsx    # Live nvidia-smi / GPU stats
│   │   ├── EmbeddingWidget.tsx      # Embedding / knowledge-base helpers
│   │   ├── AITerminalPanel.tsx      # AI-assisted terminal
│   │   ├── TerminalPanel.tsx        # Raw SSH terminal panel
│   │   └── ThemeSwitcher.tsx        # Light/dark theme toggle
│   │
│   └── lib/
│       ├── tauri.ts                 # Typed invoke()/listen() wrappers to Rust
│       └── runStreams.ts            # Live log/metric stream handling
│
└── src-tauri/                       # ── Backend (Rust / Tauri core) ──
    ├── Cargo.toml                   # Rust deps
    ├── tauri.conf.json              # App identity, window, bundle config
    ├── build.rs
    ├── capabilities/default.json    # Tauri permission capabilities
    ├── icons/                       # App icons
    └── src/
        ├── main.rs                  # Tauri command surface + event wiring
        ├── config.rs                # Load/save config.json
        ├── ssh.rs                   # russh client (connect, exec, stream, nvidia-smi)
        ├── qdrant.rs                # Qdrant HTTP scroll/count
        ├── generator.rs             # Teacher prompt + OpenAI-compat call + parse
        ├── llamafactory.rs          # JSONL → ShareGPT, dataset_info.json, train.yaml, metrics
        ├── pipeline.rs              # State machine: ssh ▶ teacher_up ▶ generate ▶ unload ▶ train ▶ done
        ├── runs.rs                  # One JSON per run, durable across restarts
        ├── serve.rs                 # Model serving / embedder boot helpers
        ├── ingest.rs                # Document ingestion
        ├── hf.rs                    # Hugging Face whoami / list models & datasets
        ├── digitalocean.rs          # Droplet management
        ├── guides.rs                # In-app guidance content
        └── error.rs                 # Error types
```

---

## Quick start

### Prerequisites

| Component | Version |
|---|---|
| Node.js | 20+ |
| Rust toolchain | 1.75+ (`rustup default stable`) |
| MSVC build tools (Windows) | Visual Studio 2022 with the C++ workload |
| WebView2 runtime | ships with Windows 11 |

On the remote GPU droplet:

```bash
pip install vllm llamafactory torch transformers accelerate datasets peft trl
```

### Run in dev

```bash
cd FineTune
npm install
npm run dev        # spawns Vite + Tauri together
```

The app window opens. The first panel asks for SSH host, Qdrant endpoint + key, and Hugging Face token.

### Build the installer

```bash
npm run build      # produces .msi / .exe under src-tauri/target/release/bundle/
```

---

## Using the app

1. **Credentials panel** (left): paste your SSH host, drop in an SSH key file, paste the Qdrant endpoint + key, and your HF token. Click **Test SSH** — the droplet's `uname -a` + GPU detection appears.

2. **Pipeline tab → Step 1 Knowledge Base**: click **Refresh** to see the total chunk count + 3 sample chunks from Qdrant.

3. **Step 2 Teacher**: choose an HF repo (e.g. `Qwen/Qwen2.5-7B-Instruct` for fast tests, or a 70B distill for production quality), the serve port, dtype, and max length.

4. **Step 3 Dataset**: edit the prompt template, set concurrency and the pairs-per-chunk target. Use a small **Max Chunks** for smoke tests.

5. **Step 4 Student & Train**: pick the student HF repo, tune the LoRA params, click **Start Pipeline**.

6. **Runs tab** (opens automatically): watch live logs, the kept/scanned/rejected counters, the dataset preview, and the loss curve as it trains. When status = `done`, your adapter lives at `/root/fine-tune/runs/{id}/lora/` on the droplet — pull it with:

   ```bash
   scp -r root@<host>:/root/fine-tune/runs/<id>/lora ./adapter
   ```

The pipeline automatically **unloads the teacher (`pkill -f vllm`) before launching the student** so the full GPU VRAM is available for LoRA training.

---

## Reusing it for any domain or model

Nothing in the code is hard-coded to a single domain. To fine-tune a new one:

1. Build a new **Qdrant collection** of your raw documents.
2. Change the **Collection** field in the Credentials panel.
3. Rewrite the **prompt template** in Step 3 to fit the new domain.
4. Pick any **Teacher + Student** HF repos.
5. Click **Start**.

No code changes are needed for new domains, models, or datasets.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `teacher boot timeout (20 min)` | vLLM weight download slow or wrong dtype | Try `--dtype auto` or a smaller model. Watch `/root/fine-tune/runs/<id>/teacher.log` on the droplet. |
| Generator errors `Connection refused` | Firewall blocks the vLLM port | Open the port (`ufw allow 8000/tcp`) or change the port in Step 2. |
| `no valid Q&A pairs generated` | Teacher ignoring the response format | Tighten the prompt template, lower temperature, or use an instruct-tuned model. |
| `adapter_model.safetensors not found` | Training crashed (probably OOM) | Check `train.log` on the droplet. Lower `batch_size` or `cutoff_len`. |
| SSH "no auth method" | Both key and password fields empty | Drop a key file in the Credentials panel or paste a password. |

---

## Tech stack

- **Frontend:** React 19, TypeScript, Tailwind CSS 4, Vite 6, Motion, lucide-react
- **Backend:** Rust, Tauri 2, russh (SSH), Qdrant HTTP, OpenAI-compatible vLLM client
- **Remote ML:** vLLM (teacher serving), LLaMA-Factory (LoRA training), Hugging Face Hub

---
## About

Fine-Tune Model is a desktop fine-tuning workspace for building LoRA adapters from a knowledge base. The main app, Fine-Tune Studio, is a Tauri 2 desktop application that orchestrates Qdrant retrieval, vLLM teacher serving, JSONL dataset generation, LLaMA-Factory training, and adapter output on a remote GPU droplet over SSH.

The local machine stays lightweight while the GPU server handles model serving and training. The app also records local droplet usage time and estimated cost without relying on provider usage reports.

## License

[MIT](./LICENSE) © Zrald1
