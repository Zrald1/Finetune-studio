# Fine-Tune Studio

A **Tauri 2 desktop app** that drives the entire LLM fine-tuning pipeline from a single window — no notebooks, no glue scripts, no SSH gymnastics.

```
Qdrant chunks → Teacher LLM (vLLM) → JSONL dataset → LoRA training (LLaMA-Factory) → adapter
```

All heavy ML work runs on a **remote GPU droplet over SSH** (built and tested on AMD Instinct MI300X / MI325X DigitalOcean instances). The local app is a pure **orchestrator**: it generates the scripts, streams the logs, parses the metrics, and persists run history. Your laptop never needs a GPU.

---

## Table of contents

- [What it does](#what-it-does)
- [Teacher model](#teacher-model)
- [Serving profiles](#serving-profiles)
- [Self-healing dependencies](#self-healing-dependencies)
- [Dataset quality](#dataset-quality)
- [Benchmarking](#benchmarking)
- [ZRALD post-training](#zrald-post-training)
- [Training methods](#training-methods)
- [GPU provisioning](#gpu-provisioning)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Quick start](#quick-start)
- [Using the app](#using-the-app)
- [Testing](#testing)
- [Privacy and credentials](#privacy-and-credentials)
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

Six top-level tabs: **Pipeline**, **GPU Servers**, **Credentials**, **Runs**, **Deploy**, **Robot Vision**, plus an AI-assisted terminal in the sidebar.

---

## Teacher model

The teacher is the model that writes your dataset. The curated roster is a single entry:

| Preset | Why |
|---|---|
| `Qwen/Qwen3.8-27B` | 27B dense hybrid-attention checkpoint, 262K native context, native vision-language, Apache 2.0, and the model vLLM publishes a verified serving recipe for on AMD Instinct MI300X / MI325X / MI355X. |
| *Custom Hugging Face Repo ID* | Any other repo — GGUF, quantized, or a different family. Auto-tuning adapts to what it can detect. |

Anything else is reachable through the Custom entry. Nothing in the code is hard-coded to one family; the profile logic keys off the repo id (see `src/lib/servingProfiles.ts`).

### Thinking effort

Hybrid-thinking Qwen3.5+ checkpoints open every assistant turn with a ` thinking` block. Two settings control that:

- **Reasoning parser** (`--reasoning-parser qwen3`) — splits the thinking into `reasoning_content` so it stops consuming the dataset generator's output budget. Set automatically for any `qwen3*` repo, in **both** serving profiles, because it is a correctness fix rather than an optimization.
- **Thinking effort** — `XHigh` / `Medium` / `Low` / `No Thinking`, applied per request via `chat_template_kwargs`, so changing it needs no redeploy.

> **Why there is no `High` option.** Qwen3.8's `chat_template.jinja` validates the value and calls `raise_exception` for anything outside `{xhigh, medium, low}`, which vLLM surfaces as **HTTP 500** rather than a 4xx. `high` is the OpenAI/Anthropic spelling, not a Qwen3.8 one. If it reaches the backend from a hand-edited config it is folded into `xhigh` rather than forwarded.

---

## Serving profiles

The Teacher step exposes two presets, switchable at any time:

| | **Standard** | **Optimized** |
|---|---|---|
| Intent | Conservative, deterministic baseline | Vendor-published serving recipe |
| Context (≥140 GB VRAM) | 100,000 | **262,144** (native) |
| GPU memory budget | 0.80 | 0.90 |
| Batched tokens | 8,192 | 16,384 |
| Max sequences | 16 | 64 |
| Prefix caching | — | ✓ |
| FP8 KV cache | — | ✓ |
| Vision-encoder sharding | — | ✓ |

The optimized context is **clamped by reported VRAM** so the same profile stays valid on smaller cards:

| VRAM | Context | Sequences |
|---|---|---|
| ≥ 140 GB | 262,144 | 64 |
| ≥ 96 GB | 131,072 | 64 |
| ≥ 48 GB | 65,536 | 64 |
| < 48 GB | 32,768 | 8 |

Switching to Optimized shows a one-line summary of what changed, and a **Revert to Standard** link puts it back. Manual mode (Auto Tune off) and a custom serve command both bypass tuning entirely.

FP8 KV cache is **native on gfx942** (MI300X / MI325X) as the E4M3FNUZ dialect, so it roughly doubles the KV pool at no emulation cost.

---

## Self-healing dependencies

Serving a model requires a Python environment that agrees with it. That is harder than
it sounds, because the components disagree with each other:

| Consumer | Wants |
|---|---|
| Qwen3.8-27B teacher (vLLM) | `transformers >= 5.8.0` |
| LLaMA-Factory 0.9.4 | `transformers >= 4.41.2, <= 4.58` |

Those are **mutually exclusive**, so the app does not try to satisfy both in one place.
It isolates and reconciles instead:

- **Isolation.** The trainer runs in `<run>/.lf_venv`, created with
  `--system-site-packages` so it inherits the container's ROCm-matched torch rather than
  pulling its own. Only the conflicting packages land in the venv.
- **Discovery, not assumptions.** Before serving anything, the app reads the checkpoint's
  own `config.json`, honours the `transformers_version` it was written by, installs any
  `requirements.txt` the repo ships, and asks vLLM's `ModelRegistry` whether it supports
  the declared architectures. A model the app has never seen resolves its own
  requirements — nothing about Qwen3.8 is hardcoded.
- **Detect before installing.** A heal only runs when a check actually fails. Nothing
  drifts to "latest" behind your back.
- **Failures degrade.** An unreachable or malformed `config.json` warns and continues; the
  model may load fine, and refusing to deploy over a metadata fetch would be worse.

This applies to **every** serving path: the teacher wizard, the **Deploy** page for
production serving, and all four student paths (inference test, benchmark, merge, and
merge+convert). The student base model is user-chosen, so it gets the same resolver.

The one thing that cannot self-heal is the **vLLM image** — architecture support lives in
the container. The resolver says so plainly rather than pretending to fix it.

---

## Dataset quality

Synthetic data fails quietly. A bad pair does not crash a run — it teaches the model
something wrong, and you find out months later. The gates here follow the published
practice rather than inventing heuristics:

- **LIMA** (Zhou et al., 2023): 1,000 filtered examples beat 50,000 noisy ones, because
  fine-tuning teaches *format* far more readily than knowledge. Filtering matters more
  than volume.
- **Provenance-preserving gating** (arXiv 2606.11127): gating against the exact source
  chunk beats post-hoc retrieval, and hallucination gates and reward gates reject
  **disjoint** failure populations — so both are needed.
- **Self-Instruct** (Wang et al., 2022): deduplicate with an overlap filter. Alpaca
  shipped ~20% near-duplicates without one.

### The gates

| Gate | Catches | Threshold |
|---|---|---|
| **Grounding** | Answers whose vocabulary the source never used | ≥ 0.35 of content words traceable |
| **Numeric fabrication** | Invented figures — the most damaging hallucination, and the cheapest to detect exactly | every number must appear in the source |
| **Answer length** | Terse answers that teach nothing (measured live: 12–13 chars passing) | 30–60 chars by format |
| **Reasoning length** | Perfunctory reasoning blocks | 80–120 chars where the format has one |
| **Near-duplicate** | The Alpaca failure mode | token similarity + length ratio |
| **Structural** | Malformed choices, invalid answer letters, missing reasoning | — |

Provenance is exact: `source_text` is the chunk that induced the pair, captured at
generation time, so the grounding gate observes the real evidence relation instead of
approximating it with retrieval.

### Prompt shape matters more than the filter

Measured against a live Qwen3.8-27B, the single largest quality lever was not a gate at
all — it was the output-format block. A bare `ANSWER:` with nothing after it gave the
model no length signal:

| Format block | Answer length | Yield |
|---|---|---|
| `ANSWER:` | **7 chars** (`RuBisCO`) | **0/6** |
| `ANSWER: <a complete self-contained answer of 2-5 sentences that names the entities involved…>` | **202–302 chars** | **6/6** |

Every built-in prompt now shows the expected *shape* of each field. A length floor alone
would have rejected the entire run rather than fixing it. Training loss on the same source
fell from **1.2281 to 0.4087** once answers carried real content.

**Known limitation:** the teacher tends to cover one concept per chunk repeatedly rather than
spanning the source. All four test pairs addressed the same fact, so the trained student
answered that fact correctly and hallucinated an untested one. The fix is to vary the
extraction angle across calls on the same chunk; until then, `question diversity` in the
coverage report is the signal to watch.

### Custom templates

Any format's prompt can be overridden per topic. Two things are enforced, because they
decide whether the output is usable at all:

1. **The template must contain `{chunk_text}`.** Without it the teacher answers from its
   own memory, and nothing downstream can tell.
2. **The template must ask for the markers the parser reads** (`QUESTION:`, `ANSWER:`, …).
   A template requesting prose gets prose, the parser finds nothing, and the run
   "succeeds" with an empty dataset. That failure is caught at validation, not after a
   long generation.

A template that fails validation falls back to the built-in with the reason in the run
log — a broken override can never silently produce nothing. Starter presets are provided
for exam-style and clinical MCQs, and for Socratic reasoning that must cite the source.

### Trainer compatibility

LLaMA-Factory truncates an example that overruns `cutoff_len` **after** applying the chat
template, so what disappears is the tail of the answer — or the closing control tokens. The
maintainers' own description of the resulting run: *"get a bad result, and have zero idea
why."*

The app measures the real requirement and says so:

```
[quality] WARNING: 2 of 4 example(s) (50%) exceed cutoff_len=128 and will be silently
          truncated by the trainer — the tail of the answer is what gets cut.
          Raise cutoff_len to at least 150 in the Train step to train on them intact.
```

### Coverage reporting

A rejection count alone is not actionable. Each run reports what it kept, *why* it
dropped the rest, mean grounding, mean answer length, distinct sources and topics, and
question diversity — the difference between "it ran" and "the dataset is usable".

---

## Benchmarking

### Teacher — token-level, from the Teacher step

A **Benchmark** button sits beside *Verify* / *Deploy Teacher*. Metric definitions deliberately mirror vLLM's own `vllm bench serve` so results are comparable with upstream tooling:

| Metric | Definition |
|---|---|
| **TTFT** | Time to first streamed token (mean / median / p95) |
| **TPOT** | `(e2el − ttft) / (output_tokens − 1)` — decode-only per-token cost |
| **ITL** | Mean observed inter-token gap |
| **Output tok/s** | Generation throughput |
| **E2EL** | End-to-end latency per request |

TTFT is not observable from a non-streaming response, so every request sets `stream: true` with `stream_options.include_usage` to get exact token counts back from the server instead of estimating them. A per-sample table adds a **Think** column — the gap between the first token and the first *content* token, i.e. what the thinking phase actually costs you.

A **Serial / Concurrent ×4** toggle switches between single-stream latency (what dataset generation experiences) and aggregate throughput under load.

### Student — accuracy plus generation speed

The Runs tab benchmarks a completed run against its own dataset. It reports **accuracy** (exact / partial / keyword-overlap) and, since the streaming probe was added, **TTFT, TPOT and tokens/sec** measured with `TextIteratorStreamer` — a plain `model.generate()` call only yields total wall time, from which TTFT cannot be separated.

---

## ZRALD post-training

**Zero-shot Retrieval-Augmented Learning with Dynamic rewards** extends the RAG dataset pipeline into a teacher-student post-training loop. A teacher reads retrieved chunks, generates grounded questions and reference answers, then scores student answers against the stored source context. The best answers become training signal for a smaller student.

Two paths are maintained:

- **ZRALD Online** (`zrald`) — GRPO via Unsloth + TRL, keeping the reward teacher available during student training.
- **ZRALD Offline** (`zrald_offline`) — stages teacher generation, student candidate generation, teacher scoring, and student LoRA training separately, so a single AMD ROCm GPU can run the whole loop without holding both models at once.

Both build their trainer and environment at runtime:

- The GRPO stack installs into an **isolated `.zrald_venv`**. A `--force-reinstall torch` in the container's system Python would overwrite the torch the resident vLLM teacher's prebuilt C extensions need (undefined symbol / triton `constexpr_function` crashes).
- The offline runner **only boots a local reward teacher when no external reward endpoint is configured** — otherwise it would fight the deployed teacher for VRAM.
- `gpt-oss` students skip 4-bit loading; their MXFP4 weights cannot be re-quantized.

See [../method.md](../method.md) for the technical method notes.

---

## Training methods

Fourteen methods ship, dispatched by `src-tauri/src/method/mod.rs`:

| Family | Methods |
|---|---|
| LoRA family | `lora`, `qlora`, `dora`, `loraplus`, `pissa`, `unsloth` |
| Full-parameter | `full`, `freeze` |
| Memory-efficient optimizers | `galore`, `badam` |
| RL | `grpo`, `zrald`, `zrald_offline` |
| Escape hatch | `custom` (arbitrary command sequence) |

`full` and `freeze` save a **complete** model (`model.safetensors` + `config.json`) rather than a PEFT adapter, so the post-training existence check, Hub upload, and merge steps treat them differently — there is nothing to merge into a base.

---

## GPU provisioning

The **GPU Servers** tab manages DigitalOcean droplets directly: plan selection, images, SSH keys, projects, live GPU droplet listing, and local usage/cost accounting.

**AMD Developer Cloud notes.** `dop_v1_` personal access tokens are served by the **standard control plane** (`api.digitalocean.com/v2`), which lists the AMD Instinct MI-series plans. The legacy `api-amd.digitalocean.com` host now 301-redirects to `api.devcloud.amd.com`, which rejects DigitalOcean tokens with `401`, so it is no longer used by default. An optional **API Base** override exists in the Credentials panel for tenants holding a Developer Cloud-native token.

Two behaviours worth knowing:

- Plan availability is **per team**. A plan can report `available: true` with an empty `regions` array and still refuse every create — the app retries the documented regions, then fails with an actionable message naming plans that *do* publish a creatable region.
- Older `-devcloud` / `-contracted` slug suffixes are **retired** and are normalised away automatically, so configs saved by earlier builds keep working.

---

## Architecture

```
   React UI            Rust core              Remote GPU droplet
   ────────            ──────────             ───────────────────
   PipelineWizard ──▶  pipeline.rs   ──SSH──▶ vllm serve <teacher>
   DeployPanel    ──▶  bench.rs      ─https─▶ /v1/chat/completions
   RunDashboard   ◀──  ssh.rs        ─https─▶ Qdrant (knowledge base)
   Live Logs      ◀──  generator.rs
   LoRA chart     ◀──  llamafactory.rs ─SSH─▶ llamafactory-cli train
   GPU Servers    ──▶  digitalocean.rs ─https─▶ DigitalOcean API
```

The local app stores config + run history in `%APPDATA%/fine-tune/` (Windows)
or `~/.config/fine-tune/` (Linux). Each run gets its own folder:

```
%APPDATA%/fine-tune/
├── config.json              # SSH, Qdrant, HF token, defaults
├── droplet_usage.json       # local droplet time + estimated cost
├── droplet_usage.csv
└── runs/
    └── 01HXYZ.../           # ULID per run
        ├── qa_dataset.jsonl
        ├── train.jsonl
        ├── val.jsonl
        └── train.yaml
```

The same run folder is mirrored on the droplet at `/root/fine-tune/runs/{id}/`.

Training and serving run inside a **`rocm-vllm` Docker container** on the host (`--device=/dev/kfd --device=/dev/dri --network=host --ipc=host --group-add video -v /root:/root`), so the app can pipe commands through `docker exec` without a separate ROCm install on the droplet. The default image is `vllm/vllm-openai-rocm:nightly`.

---

## Project structure

```
FineTune/
├── README.md
├── index.html                       # Vite entry point
├── package.json                     # npm scripts + JS deps
├── tsconfig.json
├── vite.config.ts
│
├── scripts/
│   └── simulate-deploy.ts           # Deploy-page simulation (see Testing)
│
├── src/                             # ── Frontend (React + TypeScript) ──
│   ├── main.tsx                     # React root + context-menu suppression
│   ├── App.tsx                      # Top-level state, config load/save, tabs
│   ├── types.ts                     # Shared TS types + config defaults
│   ├── index.css                    # Tailwind entry
│   │
│   ├── components/
│   │   ├── PipelineWizard.tsx       # 4-step wizard + Teacher/Dataset/Train steps
│   │   ├── DeployPanel.tsx          # Serving profiles, vLLM tuning, chat, metrics
│   │   ├── BenchmarksPanel.tsx      # Benchmark history + optimization comparison
│   │   ├── RunDashboard.tsx         # Run detail, logs, loss chart, student benchmark
│   │   ├── CredentialsPanel.tsx     # SSH / Qdrant / HF / DO / AI agent entry
│   │   ├── GpuServerManager.tsx     # DigitalOcean droplet provisioning + usage
│   │   ├── GPUStatsDashboard.tsx    # Live GPU stats
│   │   ├── TrainingConfigForm.tsx   # LoRA hyperparameter form
│   │   ├── DatasetPreview.tsx       # Inline JSONL sample viewer
│   │   ├── EmbeddingWidget.tsx      # Embedder lifecycle helpers
│   │   ├── AITerminalPanel.tsx      # AI-assisted terminal (sidebar)
│   │   ├── TerminalPanel.tsx        # ⚠ legacy, not imported anywhere
│   │   ├── RoboticsWidget.tsx       # Robot capture / model-pull bridge
│   │   └── ThemeSwitcher.tsx        # Light/dark theme toggle
│   │
│   └── lib/
│       ├── tauri.ts                 # Typed invoke()/listen() wrappers
│       ├── runStreams.ts            # Live log/metric stream handling
│       ├── servingProfiles.ts       # Standard/Optimized profile logic (pure)
│       ├── setupLogs.ts             # Setup log buffer
│       ├── textSanitize.ts          # Model-output sanitising
│       └── vllmProfiles.ts          # ⚠ legacy duplicate, currently unused
│
└── src-tauri/                       # ── Backend (Rust / Tauri core) ──
    ├── Cargo.toml
    ├── tauri.conf.json              # App identity, window, bundle config
    ├── capabilities/default.json
    └── src/
        ├── main.rs                  # Tauri command surface + event wiring
        ├── lib.rs                   # Shared library (headless server reuses it)
        ├── config.rs                # Config load/save + serving-profile resolution
        ├── quality.rs               # Dataset quality gates + coverage report
        ├── template.rs              # Custom dataset templates + validation
        ├── deps.rs                  # Self-healing dependency reconciler
        ├── bench.rs                 # Token-level inference benchmark (TTFT/TPOT/ITL)
        ├── bench_store.rs           # Persistent benchmark history
        ├── bench_pdf.rs             # Dependency-free PDF report writer
        ├── pipeline.rs              # State machine: ssh ▶ teacher ▶ generate ▶ train ▶ done
        ├── generator.rs             # Teacher prompt + OpenAI-compat call + parse
        ├── ssh.rs                   # russh client (connect, exec, stream, GPU stats)
        ├── llamafactory.rs          # JSONL → ShareGPT, dataset_info.json, train.yaml
        ├── digitalocean.rs          # Droplet CRUD, plans, images, regions
        ├── droplet_usage.rs         # Local usage/cost ledger
        ├── runs.rs                  # One JSON per run, durable across restarts
        ├── serve.rs                 # Embedder / model serving helpers
        ├── ingest.rs                # Document ingestion + embedding
        ├── qdrant.rs                # Qdrant HTTP scroll/count
        ├── hf.rs                    # Hugging Face whoami / list models & datasets
        ├── guides.rs                # Model-specific hyperparameter guidance
        ├── research.rs              # Web-research provider (robot pipeline)
        ├── robot.rs                 # Robot↔server bridge
        ├── manifest.rs              # Model manifest store
        ├── error.rs                 # Error types
        │
        ├── method/                  # Training-method strategies
        │   ├── mod.rs               # Dispatch + ZRALD simulation tests
        │   ├── lora.rs  qlora.rs  dora.rs  loraplus.rs  pissa.rs  unsloth.rs
        │   ├── full.rs  freeze.rs
        │   ├── galore.rs  badam.rs
        │   ├── grpo.rs  zrald.rs  zrald_offline.rs
        │   └── custom.rs
        │
        └── bin/server.rs            # Headless VPS server (robot intake + REST API)
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

On the remote GPU host, the app expects Docker with ROCm access. vLLM, LLaMA-Factory, and the training stack are installed into the container by the app as needed.

### Run in dev

```bash
cd FineTune
npm install
npm run dev        # spawns Vite + Tauri together
```

### Build the installer

```bash
npm run build      # produces .msi / .exe under src-tauri/target/release/bundle/
```

---

## Using the app

1. **Credentials panel**: SSH host + key, Qdrant endpoint + key, HF token. Click **Test SSH** — the host's `uname -a` and GPU detection appear.

2. **Pipeline → Step 1 Knowledge Base**: **Refresh** to see the chunk count and sample chunks from Qdrant.

3. **Step 2 Teacher**: pick `Qwen/Qwen3.8-27B` (or Custom), choose a **serving profile**, set the thinking effort, then **Deploy Teacher**. Optionally hit **Benchmark** to record TTFT / tok/s before generating.

4. **Step 3 Dataset**: edit the prompt template, set concurrency and pairs-per-chunk. Use a small **Max Chunks** for smoke tests.

5. **Step 4 Student & Train**: pick the student repo, choose a training method, tune the hyperparameters, **Start Pipeline**.

6. **Runs tab**: watch live logs, the kept/scanned/rejected counters, the dataset preview, and the loss curve. When status = `done`, the adapter is at `/root/fine-tune/runs/{id}/lora/` on the host:

   ```bash
   scp -r root@<host>:/root/fine-tune/runs/<id>/lora ./adapter
   ```

The pipeline automatically unloads the teacher before launching the student so the full GPU VRAM is available for training.

### Reusing it for any domain

1. Build a new **Qdrant collection** of your raw documents.
2. Change the **Collection** field in the Credentials panel.
3. Rewrite the **prompt template** in Step 3.
4. Pick any **Teacher + Student** repos.

No code changes are needed for new domains, models, or datasets.

---

## Testing

**197 Rust tests and 343 frontend assertions**, all passing.

There is no browser test framework. Verification is split between Rust unit tests and a Node-driven simulation of the frontend profile logic.

```bash
npm test               # everything: Rust suite + deploy simulation
npm run test:rust      # cargo test only
npm run simulate       # deploy-page simulation only
npm run simulate:zrald # ZRALD simulation only
npm run lint           # tsc --noEmit
```

| Suite | Count | Covers |
|---|---|---|
| `config::deploy_simulation` | 24 | Standard/Optimized resolution across 8 VRAM sizes, reasoning-effort mapping, shell safety of emitted flags, fresh-install defaults, legacy-config roundtrip |
| `method::zrald_simulation` | 26 | Placeholder substitution, heredoc integrity, generated Python **compiled with a real interpreter**, generated shell **parsed with a real `bash -n`**, clamping, reward-endpoint fallback, venv isolation |
| `deps` | 30 | Reconciler, universal resolver, student healer, shell safety |
| `quality` | 27 | Grounding, numeric fabrication, length floors, trainer-compatibility, coverage reporting |
| `template` | 16 | Custom-template validation, placeholder handling, preset integrity |
| `scripts/simulate-deploy.ts` | 343 assertions | The frontend profile logic, imported from the same module the app uses — not a reimplementation |

The deploy simulation and the Rust tests assert the **same VRAM ladder**, so a change on one side that isn't mirrored on the other fails loudly instead of silently diverging at deploy time.

> **Scope.** These validate that generated artifacts are structurally sound and syntactically valid. They do **not** prove vLLM accepts the flags, that the model loads, or that GRPO trains — those need real hardware.

---

## Privacy and credentials

**No credentials ship with the app.** Verified:

- The installer bundles **only icons** — `tauri.conf.json` declares no `resources`, so there is no `config.json` inside the build.
- Config is created per-user at first run in the OS config dir. `AppConfig` derives `Default`, so every field starts empty.
- No tokens are hard-coded anywhere in the source tree.

A `defaults_carry_no_credentials` test locks this in, so it cannot silently regress.

SSH keys, Qdrant API keys, Hugging Face tokens, and DigitalOcean tokens are entered at runtime and stored locally in `config.json`. **Nothing sensitive is committed to this repo**, and `.gitignore` excludes `.env` files.

The webview's right-click context menu is suppressed so the app reads as a native desktop tool; editable fields are exempted so right-click paste still works in text boxes.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `teacher boot timeout` | vLLM weight download slow, or wrong dtype | Try `--dtype auto` or a smaller model. Watch `/root/fine-tune/runs/<id>/teacher.log`. |
| Generator errors `Connection refused` | Firewall blocks the vLLM port | Open the port (`ufw allow <port>/tcp`) or change it in Step 2. |
| `no valid Q&A pairs generated` | Teacher ignoring the response format | Tighten the prompt template, lower temperature, or use an instruct-tuned model. Check the **Reasoning parser** is set — without it the thinking block can consume the whole output budget. |
| `adapter_model.safetensors not found` | Training crashed (usually OOM) | Check `train.log`. Lower `batch_size` or `cutoff_len`, or switch to a smaller serving profile. |
| SSH "no auth method" | Both key and password fields empty | Drop a key file in the Credentials panel or paste a password. |
| `422 Size is not available in this region` | The plan is retired or not entitled for this team | The error names plans that *do* publish a creatable region — pick one of those. |
| `422 This size is unavailable` | A retired `-devcloud` / `-contracted` slug | Slugs are normalised automatically; re-sync the account to refresh the plan list. |
| `401 Unable to authenticate you` | A Developer Cloud host was set as **API Base** with a `dop_v1_` token | Clear the API Base field — DigitalOcean tokens are served by the standard control plane. |
| Teacher output truncated mid-answer | Thinking consumed the token budget | Lower the thinking effort, or raise the generator's `max_tokens`. |
| Answers are a single word | The prompt's format block gave no length signal | Fixed in this version; if using a custom template, show the expected shape after each marker. |
| `[template] … rejected — using the built-in` | A custom template cannot work (no `{chunk_text}`, or missing parser markers) | The reason is in the log; the run continues with the built-in. |
| Most pairs rejected as `not-grounded` | The teacher is answering from memory, not the source | Check the chunk actually reaches the prompt, and that retrieval is returning relevant text. |
| `[quality] WARNING: … exceed cutoff_len` | LLaMA-Factory will silently truncate those examples | Raise `cutoff_len` in the Train step to the reported value. |
| Student answers correctly on one fact, invents another | The dataset covered only part of the source | Watch `question diversity`; the fix is varying the extraction angle per chunk. |
| `error: unrecognized arguments: --swap-space` | Those flags were removed from vLLM | The Deploy page no longer emits them; update to this version. |
| `RuntimeError: reshape_and_cache, cache_kernels.hip` | `--kv-cache-dtype fp8_e5m2` is broken on gfx942 | Use `fp8` (native E4M3FNUZ) or `auto`; the UI no longer offers e5m2. |
| `ValueError: Free memory ... is less than desired GPU memory utilization` | Another model is still resident (often the teacher) | Unload it first — two vLLM engines cannot share the card. |
| `huggingface-hub>=0.34.0,<1.0 is required ... found 1.31.0` | Something upgraded huggingface-hub past 1.0 | Self-healed: the trainer venv pins `<=0.99` and the reconciler never installs `--upgrade`. |

---

## Tech stack

- **Frontend:** React 19, TypeScript, Tailwind CSS 4, Vite 6, Motion, lucide-react
- **Backend:** Rust, Tauri 2, russh (SSH), reqwest, Qdrant HTTP, OpenAI-compatible vLLM client
- **Remote ML:** vLLM (teacher serving), LLaMA-Factory (supervised training), Unsloth + TRL (GRPO / ZRALD), Hugging Face Hub
- **Infrastructure:** DigitalOcean GPU droplets (AMD Instinct MI300X / MI325X), Docker + ROCm

---

## License

[MIT](./LICENSE) © Zrald1
