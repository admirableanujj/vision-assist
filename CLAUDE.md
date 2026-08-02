# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VisionAssist (Lost & Found AI) is a Streamlit web app that lets users locate misplaced belongings via voice commands and computer vision. The pipeline is: voice → Whisper STT → Ollama intent classifier → LLM response → gTTS audio playback. YOLO camera scanning is implemented and enabled.

**Entry point:** `app/app.py` (`app/main.py` is empty — ignore it)

## Development Environment

Two Docker services (`docker-compose.yml`):

| Service | Container | Port | Role |
|---|---|---|---|
| `app` | `vision_assist_app` | 8501 | Streamlit app |
| `ollama` | `vision_assist_llm` | 11434 | Local LLM (llama3) |

> **Note:** `docker-compose.yml` maps port 8000 but Streamlit runs on 8501. Use 8501 to access the UI.

```bash
# Requires .env with OPENAI_API_KEY=sk-...
docker-compose up -d --build
docker exec -it vision_assist_app bash

# Inside container — run the app
streamlit run app.py --server.port=8501 --server.address=0.0.0.0

# Pull the local LLM (first time only)
docker exec -it vision_assist_llm ollama pull llama3

# Verify environment
python verify_env.py   # PyTorch, OpenCV, YOLO, ChromaDB, API key
python test_env.py     # quick CV matrix check
```

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

Tests run locally without Docker: `app/tests/conftest.py` stubs the Docker-only deps (`ollama`, `langchain_openai`, `dotenv`) via `sys.modules` so the suite works on any machine with `pytest` and `pytest-cov` installed.

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
│   ├── app.py                    # Streamlit UI + pipeline orchestration
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

13 tables covering the full domain. Not yet wired to the app (see Known Issues).

| Table | Purpose |
|---|---|
| `users` | User accounts (guid, email, role FK, is_active) |
| `user_login` | Hashed passwords (separate from users) |
| `roles` / `permission` | RBAC — role-based access control |
| `items` | Tracked belongings (owner FK, object_class, home_zone FK) |
| `cameras` | Camera sources (owner FK, location, source URL) |
| `zones` | Bounding-box regions within a camera frame |
| `detections` | YOLO detection events (camera FK, confidence, bbox, timestamp) |
| `reminders` | Scheduled reminder records |
| `alerts` | Real-time breach/zone alert log |
| `query_logs` | LLM query trace (intent, found, latency_ms) |
| `item_embeddings` | Vector embeddings per item (for ChromaDB parity) |

### Request Flow

```
User speaks → st.audio_input()
  → SpeechToTextConverter.execute()       # Whisper cloud or local Faster-Whisper
  → QueryClassifier.classify()            # Ollama llama3 → JSON intent: locate/note/alarm/general
  → OllamaMLEngine.generate_response()   # for "locate" intent (uses tracking context)
  OR
  → OllamaMLEngine.generate_general_response()  # for "general" intent (OpenAI, falls back to Ollama)
  → TextToSpeechConverter.execute()       # gTTS → response_vocal.mp3
  → st.audio(..., autoplay=True)
```

### Key Design Patterns

- Every engine module follows **ABC → concrete class**: `BaseMLEngine`, `BaseVisionEngine`, `VoiceEngineAC`
- `OllamaMLEngine` resolves its host from the `OLLAMA_HOST` env var (set by docker-compose), falling back to `http://vision_assist_llm_local:11434` if unset
- `QueryClassifier` uses `format="json"` and `temperature=0.0` in Ollama to force deterministic structured output
- All engines are initialized once via `@st.cache_resource` in `app.py` — avoid stateful side effects in constructors
- `VISION_ENABLED = True` in `app.py` enables camera scanning; set to `False` to disable
- `YOLOVisionEngine` runs real YOLO inference with confidence-threshold filtering (default 0.5, see `DEFAULT_CONFIDENCE_THRESHOLD` in `vision_engine.py`); falls back to `FallbackVisionEngine`'s mock pool if weights fail to load or inference raises at runtime
- `BaseVisionEngine.scan_frame()` returns a `dict` — `{"detections": [{"label", "confidence", "box"}, ...], "annotated_frame": <rendered image or None>}`, sorted by confidence descending. `_UltralyticsScanMixin` (shared by `YOLOVisionEngine` and `YOLOEVisionEngine`) renders `annotated_frame` via its own `_draw_detections()` helper — plain `cv2.rectangle`/`cv2.putText` calls driven directly from the same `detections` list, not `ultralytics.Results.plot()` — so the drawn boxes can never diverge from what's reported as text, regardless of model task type (segmentation checkpoints like YOLOE's render masks/etc. differently via `.plot()`, which this sidesteps). Converted BGR→RGB for direct use with `st.image`. `FallbackVisionEngine` always returns `annotated_frame: None` (no real image to draw on) and `box: None` per detection
- `YOLOVisionEngine` resolves its weights file the same way `OllamaMLEngine` resolves its host: constructor arg > `YOLO_MODEL_PATH` env var > hardcoded default (`DEFAULT_LOCAL_WEIGHTS = "yolo26n.pt"`) — swapping to a bigger model in a cloud deployment is a config change, not a code change
- `YOLOVisionEngine` and `YOLOEVisionEngine` share their `scan_frame()` implementation via `_UltralyticsScanMixin` in `vision_engine.py` — only model loading (`YOLO(...)` vs `YOLOE(...)` + `set_classes()`) differs between them, so the two can't drift out of sync
- `VISION_MODEL_TYPE` env var (`"yolo"` default, or `"yoloe"`) selects which real engine backs `VisionTracker`; `VISION_CUSTOM_CLASSES` (comma-separated) sets which classes `YOLOEVisionEngine` is prompted to detect, defaulting to `DEFAULT_CUSTOM_CLASSES` — see `YOLO_VS_YOLOE_GUIDE.md` for when to use which
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

1. **`tokenize_text()` defined 3× in `ml_engine.py`** — only the last definition (fake `ord()` version) is active; first two are dead code.
2. **No persistence** — items are stored in `st.session_state.tracked_items` and lost on restart. PostgreSQL schema exists in `Database/DatabaseScript_PostgreSQL.sql` but no SQLAlchemy ORM layer is wired to the app yet.
3. **`requirements.txt` incomplete** — missing `streamlit`, `faster-whisper`, `gtts`, `langchain`, `langchain-openai`, `langchain-core`.

Design assets (UML diagrams, technical guide) are in `Documents/`. Markdown design docs are in `docs/`.
