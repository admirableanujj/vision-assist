# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VisionAssist (Lost & Found AI) is a Streamlit web app that lets users locate misplaced belongings via voice commands and computer vision. The pipeline is: voice → Whisper STT → Ollama intent classifier → LLM response → gTTS audio playback. YOLO/YOLOE camera scanning is implemented and enabled. Login/registration is backed by PostgreSQL (see `user_module/user_manager.py`); items registered or detected are persisted to Postgres, not `st.session_state`.

**Entry point (UI code):** `app/app.py`. **Actual container entrypoint:** `app/main.py` — runs DB migrations/seeding (`precheck_db.py`), then launches Streamlit as a subprocess and propagates its real exit code (important for diagnosing crashes via `docker inspect`'s `ExitCode` — see Key Design Patterns).

## Development Environment

Four Docker services (`docker-compose.yml`), credentials via `.env` + Docker secrets (`./secrets/`, gitignored — see `README.md`'s Secrets section):

| Service | Container | Port | Role |
|---|---|---|---|
| `app` | `vision_assist_app` | 8501 | Streamlit app (runs `python main.py`) |
| `ollama` | `vision_assist_llm_local` | 11434 | Local LLM (llama3) |
| `postgres` | `vision_assist_db` | 5432 | User auth + item persistence (hostname `db`) |
| `qdrant` | `vision_assist_qdrant` | 6333/6334 | Vector DB — **not wired into any application code yet**, infra-only |

```bash
# Requires .env (OPENAI_API_KEY, POSTGRES_USER/PASSWORD/DB) and secrets/ (see README.md)
docker-compose up -d --build
docker exec -it vision_assist_app bash

# Pull the local LLM (first time only)
docker exec -it vision_assist_llm_local ollama pull llama3

# Verify environment
python verify_env.py   # PyTorch, OpenCV, YOLO, ChromaDB, API key
python test_env.py     # quick CV matrix check
```

Minimum footprint for just the vision/auth path (skips `ollama`/`qdrant`): `docker compose up -d --no-deps app postgres` — see `TESTING_LOCAL_YOLO.md`.

Add dependencies to `app/requirements.txt` and rebuild with `docker-compose build app`.

## Testing

Tests live in `app/tests/`. Run from the `app/` directory:

```bash
# Run with coverage report (80% gate enforced)
python3 -m pytest -v

# Inside Docker container
docker exec -it vision_assist_app bash -c "cd /workspace && python3 -m pytest -v"
```

Coverage is configured in `app/pytest.ini` — currently scoped to `ml_engine` and `vision_engine`. Extend the `--cov` flag as new modules gain tests.

Tests run locally without Docker: `app/tests/conftest.py` stubs the Docker-only deps (`ollama`, `dotenv`, `langchain_openai`, `langchain_core`, `cv2`, `ultralytics`) via `sys.modules` so the suite works on any machine with `pytest` and `pytest-cov` installed.

### Pre-commit hook

A pre-commit hook in `.githooks/pre-commit` blocks commits when Python files are staged and tests fail or coverage drops below 80%. First-time setup (one-off per clone):

```bash
git config core.hooksPath .githooks
```

## Architecture

### Repository Layout

```
vision-assist/
├── app/                          # All runnable application code
│   ├── app.py                    # Streamlit UI + pipeline orchestration (imported by main.py's subprocess)
│   ├── main.py                   # Real container entrypoint — migrations, then launches Streamlit
│   ├── precheck_db.py            # DB migrations + first-boot seeding (test_user/admin accounts)
│   ├── user_module/
│   │   └── user_manager.py       # UserManager — Postgres-backed register/authenticate (salted SHA-256)
│   ├── .streamlit/
│   │   └── config.toml           # fileWatcherType="none" — dev-mode hot-reload disabled, see Key Design Patterns
│   ├── ml_engine/
│   │   ├── ml_base_engine.py     # ABC: tokenize_text(), generate_response()
│   │   ├── ml_engine.py          # OllamaMLEngine — local LLM + OpenAI cloud fallback
│   │   └── query_classifier.py   # Intent routing via Ollama structured JSON
│   ├── voice_engine/
│   │   ├── voice_base.py         # ABC: initialize_engine(), execute()
│   │   ├── voice_stt.py          # Whisper (OpenAI API) → Faster-Whisper (offline fallback)
│   │   └── voice_tts.py          # gTTS → MP3 → browser autoplay
│   └── vision_engine/
│       ├── vision_base.py        # ABC: scan_frame()
│       └── vision_engine.py      # YOLOVisionEngine + YOLOEVisionEngine (open-vocab, VISION_MODEL_TYPE=yoloe) + FallbackVisionEngine (mock degrade path)
├── Database/
│   └── DatabaseScript_PostgreSQL.sql   # Full PostgreSQL schema (13 tables)
└── Documents/
    ├── VisionAssist_Architecture_UML.pptx
    ├── VisionAssist_Technical_Guide.docx
    └── diagram_assets/           # 14 UML diagram PNGs
```

### Database Schema (`Database/DatabaseScript_PostgreSQL.sql`)

13 tables. `users`, `user_login`, `user_login_history`, and `items` **are** wired to the app (`user_module/user_manager.py` for auth, `app.py`'s `fetch_user_items`/`register_db_item` for inventory) — the rest (`cameras`, `zones`, `detections`, `reminders`, `alerts`, `query_logs`, `item_embeddings`, `roles`/`permission` beyond the default role) are schema-only, not yet written to by application code.

| Table | Purpose | Wired? |
|---|---|---|
| `users` | User accounts (guid, username, role FK, is_active) | Yes |
| `user_login` | Salted/hashed passwords (separate from users) | Yes |
| `user_login_history` | Login attempt audit trail | Yes |
| `items` | Tracked belongings (owner FK, item_name, description) | Yes |
| `roles` / `permission` | RBAC — role-based access control | Partial (default role only, no permission checks in app code) |
| `cameras` | Camera sources (owner FK, location, source URL) | No |
| `zones` | Bounding-box regions within a camera frame | No |
| `detections` | Detection events (camera FK, confidence, bbox, timestamp) | No — detections currently only reach `items`, not a `detections` row |
| `reminders` | Scheduled reminder records | No |
| `alerts` | Real-time breach/zone alert log | No |
| `query_logs` | LLM query trace (intent, found, latency_ms) | No |
| `item_embeddings` | Vector embeddings per item (for Qdrant parity) | No — Qdrant itself isn't called from app code either |

### Request Flow

Every user interaction reruns `app.py` top-to-bottom (Streamlit's execution model) behind an auth gate:

```
st.session_state.authenticated? → no → login/register tabs (UserManager, Postgres) → st.stop()
                                 → yes → dashboard renders, both vision engines pre-warmed (see below)

User speaks → st.audio_input()
  → SpeechToTextConverter.execute()       # Whisper cloud or local Faster-Whisper
  → QueryClassifier.classify()            # Ollama llama3 → JSON intent: locate/note/alarm/general
  → OllamaMLEngine.generate_response()   # for "locate" intent (uses tracking context)
  OR
  → OllamaMLEngine.generate_general_response()  # for "general" intent (OpenAI, falls back to Ollama)
  → TextToSpeechConverter.execute()       # gTTS → response_vocal.mp3
  → st.audio(..., autoplay=True)

User takes photo → st.camera_input()
  → get_vision_engine(engine_choice)      # cached per YOLO/YOLOE choice, both pre-warmed at login
  → tracker.scan_frame(cam_frame)         # detections + annotated_frame (custom-drawn, not Results.plot())
  → register_db_item(...) per detection   # persisted to Postgres `items`
```

### Key Design Patterns

- Every engine module follows **ABC → concrete class**: `BaseMLEngine`, `BaseVisionEngine`, `VoiceEngineAC`
- `OllamaMLEngine` resolves its host from the `OLLAMA_HOST` env var (set by docker-compose), falling back to `http://vision_assist_llm_local:11434` if unset
- `QueryClassifier` uses `format="json"` and `temperature=0.0` in Ollama to force deterministic structured output
- Non-vision engines are initialized once via `@st.cache_resource`-wrapped `boot_system_core()` in `app.py` — avoid stateful side effects in constructors
- `VISION_ENABLED = True` in `app.py` enables camera scanning; set to `False` to disable
- `YOLOVisionEngine` runs real YOLO inference with confidence-threshold filtering (default 0.5, see `DEFAULT_CONFIDENCE_THRESHOLD` in `vision_engine.py`); falls back to `FallbackVisionEngine`'s mock pool if weights fail to load or inference raises at runtime
- `BaseVisionEngine.scan_frame()` returns a `dict` — `{"detections": [{"label", "confidence", "box"}, ...], "annotated_frame": <rendered image or None>}`, sorted by confidence descending. `_UltralyticsScanMixin` (shared by `YOLOVisionEngine` and `YOLOEVisionEngine`) renders `annotated_frame` via its own `_draw_detections()` helper — plain `cv2.rectangle`/`cv2.putText` calls driven directly from the same `detections` list, not `ultralytics.Results.plot()` — so the drawn boxes can never diverge from what's reported as text, regardless of model task type (segmentation checkpoints like YOLOE's render masks/etc. differently via `.plot()`, which this sidesteps). Converted BGR→RGB for direct use with `st.image`. `FallbackVisionEngine` always returns `annotated_frame: None` (no real image to draw on) and `box: None` per detection
- `YOLOVisionEngine` resolves its weights file the same way `OllamaMLEngine` resolves its host: constructor arg > `YOLO_MODEL_PATH` env var > hardcoded default (`DEFAULT_LOCAL_WEIGHTS = "yolo26n.pt"`) — swapping to a bigger model in a cloud deployment is a config change, not a code change
- `YOLOVisionEngine` and `YOLOEVisionEngine` share their `scan_frame()` implementation via `_UltralyticsScanMixin` in `vision_engine.py` — only model loading (`YOLO(...)` vs `YOLOE(...)` + `set_classes()`) differs between them, so the two can't drift out of sync
- `VISION_MODEL_TYPE` env var (`"yolo"` default, or `"yoloe"`) selects which real engine backs `VisionTracker` (used outside the app's own UI, e.g. scripts/tests); **inside `app.py` itself**, a live `st.radio` "Detection engine" selector in the camera panel lets a logged-in user switch between YOLO/YOLOE at runtime for side-by-side comparison, backed by `get_vision_engine(engine_choice)`. `VISION_CUSTOM_CLASSES` (comma-separated) sets which classes `YOLOEVisionEngine` is prompted to detect, defaulting to `DEFAULT_CUSTOM_CLASSES` — see `YOLO_VS_YOLOE_GUIDE.md` for when to use which
- `get_vision_engine(engine_choice)` is `@st.cache_resource`-wrapped, keyed on the engine name — both YOLO and YOLOE end up cached and resident simultaneously once each has been selected, so switching is instant. Both are also eagerly pre-warmed right after login (before the dashboard renders) so even the *first* selection of either is instant, not just subsequent ones. This trades a slightly slower first render after login for zero-wait switching afterward — confirmed to have comfortable memory headroom on a 4-core/16GB Codespace; an earlier version evicted the previous engine on switch to control memory, but the actual crash that motivated that turned out to be unrelated (see the file-watcher point below) and it was reverted
- `app/.streamlit/config.toml` sets `fileWatcherType = "none"`, disabling Streamlit's dev-mode hot-reload file watcher. This isn't cosmetic: the watcher was observed (twice, in production logs) throwing an exception while probing `torch.classes.__path__` — a known Streamlit/PyTorch incompatibility — and was the actual cause of a real production bug where the whole Streamlit process would cleanly self-exit mid-session (confirmed via `docker inspect`'s `ExitCode`, once `main.py` was fixed to stop swallowing it — see next point) right after the vision pipeline touched files under the watched `/workspace` mount, manifesting to users as a WebSocket "CONNECTING" state and a browser-side 502. There's no legitimate need for hot-reload in a deployed container other people's browsers connect to, so this is a straightforward net positive, not just a workaround
- `main.py` (the real container entrypoint) launches Streamlit via `subprocess.Popen` and must call `sys.exit(process.wait())` — **not** just `process.wait()` — to propagate Streamlit's actual exit code/signal. Without this, `main()` falls through and the container always reports a clean `ExitCode=0` to Docker regardless of how Streamlit actually died, making `docker inspect` useless for diagnosing crashes. This was the root blocker in tracking down the file-watcher bug above; keep it this way if `main.py` is ever touched again
- `docker-compose.yml`'s `app` service must explicitly list `YOLO_MODEL_PATH`/`VISION_MODEL_TYPE`/`VISION_CUSTOM_CLASSES` under `environment:` with `${VAR:-default}` syntax for them to reach the container at all — Compose doesn't auto-forward arbitrary `.env` vars, and omitting the `:-default` fallback would pass an *empty string* (not "unset") when a var is missing from `.env`, silently overriding the Python-side default

### Vision Model Selection

- Local default is `yolo26n.pt` (Ultralytics YOLO26, nano, released January 2026) — chosen over the previously-used `yolo11n.pt` for fewer parameters (2.4M vs 2.6M), higher mAP (40.9 vs 39.5), and ~30% faster CPU inference (38.9ms vs 56.1ms) at the same size class. `requirements.txt`'s existing `ultralytics==8.4.106` pin already supports YOLO26 — no further version bump was needed.
- Override per-environment with `YOLO_MODEL_PATH` (e.g. a larger `m`/`l` variant on a GPU deployment) without touching `vision_engine.py`.
- Every version in the Ultralytics lineage (v5/v8/v9/v10/v11) ships under the same AGPL-3.0/Enterprise dual license — switching versions doesn't change licensing exposure.
- Standard COCO-pretrained weights (any YOLO version) do **not** include `keys`, `wallet`, or `sunglasses` as classes — only `backpack`, `handbag`, and `cell phone` overlap with `FallbackVisionEngine.simulated_pool`. Closing that gap needs fine-tuning, YOLOE's open-vocabulary detection (`VISION_MODEL_TYPE=yoloe`), or the embeddings-based custom-object approach (see Milestone 4 tasks) — not a different pretrained closed-set model.

### LLM Strategy

| Path | Trigger | Backend |
|---|---|---|
| Local inference | `generate_response()` on "locate" intent | Ollama (llama3) |
| General Q&A | `generate_general_response()` on "general" intent | LangChain + OpenAI gpt-4o-mini, falls back to Ollama (llama3) if no API key or on cloud failure |
| Offline STT | No `OPENAI_API_KEY` | Faster-Whisper "tiny" CPU model |

## Known Issues

1. **Dead commented-out code in `ml_engine.py`** — two earlier `tokenize_text()` implementations (Ollama-native tokenize, embed-based) are commented out above the single active definition (plain `ord()`-based fake tokenization). Harmless but worth deleting rather than leaving commented out next time that file is touched.
2. **Most of the PostgreSQL schema is unused** — `users`/`user_login`/`user_login_history`/`items` are wired (auth + inventory), but `cameras`, `zones`, `detections`, `reminders`, `alerts`, `query_logs`, `item_embeddings` are schema-only. Detections currently land in `items`, not a dedicated `detections` row with camera/bbox/timestamp — fine for the current single-camera-panel UX, but would need wiring if per-camera/zone tracking becomes a real requirement.
3. **Qdrant runs but is never called** — `docker-compose.yml` starts a `qdrant` service and `qdrant-client` is in `requirements.txt`, but no application code imports or calls it. `item_embeddings` (the table that would back it) is likewise unwritten.

Design assets (UML diagrams, technical guide) are in `Documents/`.
