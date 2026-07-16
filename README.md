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

# PostgreSQL — non-sensitive values only; the password lives in a Docker secret
POSTGRES_USER=admin
POSTGRES_DB=vision_assist
```

The Postgres **password** and the Qdrant **API key** are not stored in `.env`.
They are managed as Docker secrets — see [Secrets](#secrets) below.

> **Note:** The Postgres credentials are only applied when the database is
> **first initialized**. If you change the user, password, or DB name after the
> `postgres_storage` volume already exists, the new values are ignored and you'll
> get `password authentication failed`. Wipe the volume to re-initialize:
>
> ```bash
> docker-compose down
> docker volume rm vision-assist_postgres_storage   # confirm name: docker volume ls | grep postgres
> docker-compose up -d
> ```
>
> To change credentials **without** losing data, see
> [`docs/issues/postgres-password-auth-failed.md`](docs/issues/postgres-password-auth-failed.md).

### Secrets

Sensitive credentials are injected via Docker secrets (files under `./secrets/`,
which is git-ignored). Create both files before starting the stack:

```bash
mkdir -p secrets

# PostgreSQL password (base64-encoded — see note below)
printf '%s' 'my_secure_password' | base64 > secrets/postgres_password.txt

# Qdrant API key (any strong random string)
python3 -c "import secrets; print(secrets.token_urlsafe(32))" | tr -d '\n' > secrets/qdrant_api_key.txt
```

| Secret file | Mounted at | Injected as |
|---|---|---|
| `secrets/postgres_password.txt` | `/run/secrets/postgres_password` | `POSTGRES_PASSWORD_FILE` |
| `secrets/qdrant_api_key.txt` | `/run/secrets/qdrant_api_key` | `QDRANT__SERVICE__API_KEY` |

> **Base64 caveat:** Postgres reads the secret file **verbatim** — it does *not*
> decode base64. So the effective login password is the encoded string itself
> (e.g. `bXlfc2VjdXJlX3Bhc3N3b3Jk`), not `my_secure_password`. If you prefer to
> log in with the plaintext value, store the plaintext directly in the file.

**Qdrant** has no native `*_FILE` support, so its container entrypoint reads the
secret file and exports `QDRANT__SERVICE__API_KEY` before launching. Unlike the
Postgres password, the API key is **not** tied to volume init — a restart is
enough to pick up a new key. Every request must send it in the `api-key` header:

```bash
# from the host
curl -H "api-key: $(cat secrets/qdrant_api_key.txt)" http://localhost:6333/collections
```

**Health check from the app container.** The `app` image has no `curl`, so use
Python (`requests` is already installed). `/healthz` needs no API key:

```bash
docker exec -it vision_assist_app python -c "import requests; print(requests.get('http://qdrant:6333/healthz').text)"
```

Expected output: `healthz check passed`. To also verify the API key is accepted,
read the mounted secret and hit an authenticated endpoint:

```bash
docker exec -it vision_assist_app python -c "
import requests
key = open('/run/secrets/qdrant_api_key').read().strip()
r = requests.get('http://qdrant:6333/collections', headers={'api-key': key})
print(r.status_code, r.json())
"
```

Python client:

```python
from qdrant_client import QdrantClient
client = QdrantClient(url="http://qdrant:6333", api_key="<your-api-key>")
```

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
| `vision_assist_app` | 8501 | Main Python application (Streamlit) |
| `vision_assist_llm` | 11434 | Ollama local LLM server |
| `vision_assist_db` | 5432 | PostgreSQL database (hostname `db`) |
| `vision_assist_qdrant` | 6333 / 6334 | Qdrant vector database (hostname `qdrant`) |

## Stack

| Component | Library |
|---|---|
| Frontend | Streamlit |
| Speech input | OpenAI Whisper API / Faster-Whisper (offline) |
| Speech output | gTTS → MP3 → browser autoplay |
| Object detection | YOLOv8 (ultralytics), OpenCV |
| Speech input | SpeechRecognition |
| Speech output | pyttsx3 |
| Cloud LLM | OpenAI API |
| Local LLM | Ollama |
| Orchestration | LangChain |
| Vector memory | Qdrant |
| Relational store | PostgreSQL |
