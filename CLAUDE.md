# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VisionAssist (Lost & Found AI) is a Streamlit web app that lets users locate misplaced belongings via voice commands and computer vision. The pipeline is: voice → Whisper STT → Ollama intent classifier → LLM response → gTTS audio playback. YOLO camera scanning is implemented but currently disabled.

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

Coverage is configured in `app/pytest.ini` — currently scoped to `ml_engine`. Extend the `--cov` flag as new modules gain tests.

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
│       └── vision_engine.py      # YOLOVisionEngine (stubbed) + FallbackVisionEngine (random mock)
├── Database/
│   └── DatabaseScript_PostgreSQL.sql   # Full PostgreSQL schema (13 tables)
└── Documents/
    ├── VisionAssist_Architecture_UML.pptx
    ├── VisionAssist_Technical_Guide.docx
    └── diagram_assets/           # 14 UML diagram PNGs
```

### Database Schema (`Database/DatabaseScript_PostgreSQL.sql`)

13 tables covering the full domain. Key tables Shubham's DB layer must implement as SQLAlchemy models:

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
  → OllamaMLEngine.generate_cloud_response()  # for "general" intent (OpenAI fallback)
  → TextToSpeechConverter.execute()       # gTTS → response_vocal.mp3
  → st.audio(..., autoplay=True)
```

### Key Design Patterns

- Every engine module follows **ABC → concrete class**: `BaseMLEngine`, `BaseVisionEngine`, `VoiceEngineAC`
- `OllamaMLEngine` connects to Ollama at `http://vision_assist_llm_local:11434` — this must match the Docker service hostname
- `QueryClassifier` uses `format="json"` and `temperature=0.0` in Ollama to force deterministic structured output
- All engines are initialized once via `@st.cache_resource` in `app.py` — avoid stateful side effects in constructors
- `VISION_ENABLED = False` in `app.py` disables camera scanning; set to `True` to enable

### LLM Strategy

| Path | Trigger | Backend |
|---|---|---|
| Local inference | `generate_response()` on "locate" intent | Ollama (llama3) |
| Cloud fallback | `generate_cloud_response()` on "general" intent | LangChain + OpenAI gpt-4o-mini |
| Offline STT | No `OPENAI_API_KEY` | Faster-Whisper "tiny" CPU model |

Cloud fallback activates only if `OPENAI_API_KEY` is set in `.env`.

## Known Issues (as of July 2026)

1. **`generate_cloud_response()` is undefined** — `app.py:128` calls it but `OllamaMLEngine` has no such method → `AttributeError` on any "general" intent query.
2. **`tokenize_text()` defined 3× in `ml_engine.py`** — only the last definition (fake `ord()` version) is active; first two are dead code.
3. **Ollama hostname** — `OllamaMLEngine` defaults to `http://vision_assist_llm_local:11434` but the Docker container is named `vision_assist_llm`. Update the host or the compose service name to match.
4. **YOLO not wired** — `YOLOVisionEngine` has `self.model = None` and YOLO import commented out; real detection not yet active.
5. **No persistence** — items are stored in `st.session_state.tracked_items` and lost on restart. PostgreSQL schema exists in `Database/DatabaseScript_PostgreSQL.sql` but SQLAlchemy ORM layer is not yet written or wired to the app.
6. **`requirements.txt` incomplete** — missing `streamlit`, `faster-whisper`, `gtts`, `langchain`, `langchain-openai`, `langchain-core`.

## Pending Work (Shubham — Milestone 3)

See `docs/milestone3-overview.md` for full details and rubric alignment.

1. **Implement `generate_cloud_response()`** in `ml_engine.py` — the missing general Q&A method
2. **DB layer** — Write SQLAlchemy ORM models from `Database/DatabaseScript_PostgreSQL.sql` schema; use SQLite for dev. Priority tables: `items`, `detections`, `query_logs`. Replace `st.session_state.tracked_items` with DB-backed queries.
3. **Fix the 3 bugs listed above** before any new feature work

Design assets (UML diagrams, technical guide) are in `Documents/`. All markdown design docs are in `docs/` — start with `docs/links.md` for all project references.
