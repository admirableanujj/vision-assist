# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vision-Assist is an AI-powered voice-vision assistant that combines:
- **Object detection** via YOLOv8 (ultralytics) and OpenCV
- **Speech I/O** via SpeechRecognition and pyttsx3
- **LLM backends**: OpenAI API (cloud) and Ollama (local, `http://ollama:11434`)
- **Vector memory** via ChromaDB
- **Orchestration** via LangChain

All application code lives in `app/`. The entry point is `app/main.py`.

## Development Environment

The project runs entirely inside Docker. There is **no local Python environment** — all commands should be run inside the container.

### Start the environment

```bash
# Requires a .env file with OPENAI_API_KEY=sk-...
docker-compose up -d
docker exec -it vision_assist_app bash
```

### Verify the environment (inside container)

```bash
python verify_env.py   # full diagnostics: PyTorch, OpenCV, YOLO, ChromaDB, API key
python test_env.py     # quick CV matrix sanity check
```

### Install a new dependency

Add to `app/requirements.txt`, then rebuild:
```bash
docker-compose build app
```

## Architecture

Two Docker services defined in `docker-compose.yml`:

| Service | Container | Role |
|---|---|---|
| `app` | `vision_assist_app` | Python app (port 8000) |
| `ollama` | `vision_assist_llm` | Local LLM server (port 11434) |

The `ollama_storage` named volume persists downloaded model weights across rebuilds.

### Two-stage Docker build (`app/Dockerfile`)

- **Stage 1 (`ai_base`):** Installs heavy, slow-changing deps (PyTorch CPU, OpenCV, ChromaDB, ultralytics). This layer is cached and rarely rebuilt.
- **Stage 2 (`ai_dev`):** Copies `requirements.txt`, installs remaining deps, pre-downloads `yolov8n.pt`. This is the active dev layer.

The `./app` directory is bind-mounted to `/workspace` inside the container, so code changes take effect immediately without rebuilding.

## Key Constraints

- **CPU-only PyTorch** — the image installs the `--index-url https://download.pytorch.org/whl/cpu` build. No GPU in this environment.
- **Headless OpenCV** (`opencv-python-headless`) — no display/GUI available inside Docker; use file I/O or numpy arrays for image data.
- **YOLO model:** `yolov8n.pt` (nano) is pre-cached in the image at build time.
- **Ollama models** must be pulled manually after first `docker-compose up` if not already in the volume: `docker exec vision_assist_llm ollama pull <model>`.
