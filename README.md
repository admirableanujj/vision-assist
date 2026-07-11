# Vision Assist

An AI-powered voice-vision assistant that combines real-time object detection, speech I/O, and LLM reasoning — running fully inside Docker.

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (v24+) — includes both Docker Engine and Docker Compose
- An OpenAI API key (for cloud LLM calls)

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/admirableanujj/vision-assist.git
cd vision-assist
```

### 2. Create a `.env` file

```bash
cp .env.example .env   # if it exists, otherwise create manually
```

Add your key to `.env`:

```
OPENAI_API_KEY=sk-...

# PostgreSQL credentials
POSTGRES_USER=admin
POSTGRES_PASSWORD=changeme
POSTGRES_DB=vision_assist
```

> **Note:** The Postgres credentials above are only applied when the database is
> **first initialized**. If you change `POSTGRES_USER`, `POSTGRES_PASSWORD`, or
> `POSTGRES_DB` after the `postgres_storage` volume already exists, the new
> values are ignored and you'll get `password authentication failed`. Wipe the
> volume to re-initialize with the new credentials:
>
> ```bash
> docker-compose down
> docker volume rm vision-assist_postgres_storage   # confirm name: docker volume ls | grep postgres
> docker-compose up -d
> ```
>
> To change credentials **without** losing data, see
> [`docs/issues/postgres-password-auth-failed.md`](docs/issues/postgres-password-auth-failed.md).

### 3. Build and start the containers

```bash
docker-compose up -d --build
```

This will:
- Build the `app` image (downloads PyTorch, OpenCV, YOLO weights — takes ~5–10 min on first run)
- Pull and start the `ollama` LLM server
- Mount `./app` into `/workspace` in the container

### 4. Verify the environment

```bash
docker exec -it vision_assist_app python verify_env.py
```

Expected output confirms: Python 3.11, PyTorch, OpenCV, YOLOv8n loaded, ChromaDB initialized, and `OPENAI_API_KEY` present.

### 5. (Optional) Pull a local LLM model via Ollama

```bash
docker exec -it vision_assist_llm ollama pull llama3
```

Models are persisted in the `ollama_storage` Docker volume across restarts.

## Running the App

```bash
# Access the UI at http://localhost:8501
docker exec -it vision_assist_app streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

Or drop into an interactive shell:

```bash
docker exec -it vision_assist_app bash
```

## Running Tests

Tests run locally without Docker (deps are stubbed automatically):

```bash
cd app
python3 -m pytest -v   # runs with 80% coverage gate
```

Or inside the container:

```bash
docker exec -it vision_assist_app bash -c "cd /workspace && python3 -m pytest -v"
```

### Pre-commit hook (one-time setup per clone)

Commits that touch Python files are blocked if tests fail or coverage drops below 80%:

```bash
git config core.hooksPath .githooks
```

## Services

| Service | Port | Description |
|---|---|---|
| `vision_assist_app` | 8501 | Streamlit web UI |
| `vision_assist_llm_local` | 11434 | Ollama local LLM server |

## Stack

| Component | Library |
|---|---|
| Frontend | Streamlit |
| Speech input | OpenAI Whisper API / Faster-Whisper (offline) |
| Speech output | gTTS → MP3 → browser autoplay |
| Object detection | YOLOv8 (ultralytics), OpenCV |
| Local LLM | Ollama (llama3) |
| Cloud LLM | LangChain + OpenAI gpt-4o-mini |
| Vector memory | ChromaDB (planned) |
