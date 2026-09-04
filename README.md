Here is the fully updated and detailed `README.md`. It incorporates all the fixes for Docker networking, secure secrets handling, Postgres initialization, and the automated Streamlit startup via `main.py`.

```markdown
# Vision Assist

An AI-powered voice-vision assistant that combines real-time object detection, speech I/O, and LLM reasoning — running fully inside Docker.

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (v24+) — includes both Docker Engine and Docker Compose
- An OpenAI API key (for cloud LLM calls)

## Setup

### 1. Clone the repo

```bash
git clone [https://github.com/admirableanujj/vision-assist.git](https://github.com/admirableanujj/vision-assist.git)
cd vision-assist

```

### 2. Create a `.env` file

```bash
cp .env.example .env   # if it exists, otherwise create manually

```

Add your credentials to `.env`:

```env
OPENAI_API_KEY=sk-...

# PostgreSQL — non-sensitive values only; the password lives in a Docker secret
POSTGRES_USER=postgres
POSTGRES_DB=vision_assist
POSTGRES_HOST=vision_assist_db

```

The Postgres **password** and the Qdrant **API key** are not stored in `.env`.
They are managed as Docker secrets — see [Secrets](#secrets) below.

> **Note:** The Postgres credentials are only applied when the database is **first initialized**. If you change the user, password, or DB name after the `postgres_storage` volume already exists, the new values are ignored and you'll get `password authentication failed`.
> To wipe the volume and re-initialize:
> ```bash
> docker-compose down
> docker volume rm vision-assist_postgres_storage
> docker-compose up -d
> 
> ```
> 
> 
> To change credentials **without** losing data, see [`docs/issues/postgres-password-auth-failed.md`](https://www.google.com/search?q=docs/issues/postgres-password-auth-failed.md).

To manually connect to the PostgreSQL database via CLI:

```bash
docker compose exec vision_assist_db psql -U postgres -d vision_assist

```

### 3. Configure Secrets

Sensitive credentials are injected via Docker secrets (files under `./secrets/`, which is git-ignored). Create both files before starting the stack:

```bash
mkdir -p secrets

# PostgreSQL password (base64-encoded — see note below)
printf '%s' 'my_secure_password' | base64> secrets/postgres_password.txt

> **Base64 caveat:** Postgres reads the secret file **verbatim** — it does *not*
> decode base64. So the effective login password is the encoded string itself
> (e.g. `bXlfc2VjdXJlX3Bhc3N3b3Jk`), not `my_secure_password`. If you prefer to
> log in with the plaintext value, store the plaintext directly in the file.

# Qdrant API key (any strong random string)
python3 -c "import secrets; print(secrets.token_urlsafe(32))" | tr -d '\n' > secrets/qdrant_api_key.txt

```

| Secret file | Mounted at | Injected as |
| --- | --- | --- |
| `secrets/postgres_password.txt` | `/run/secrets/postgres_password` | `POSTGRES_PASSWORD_FILE` |
| `secrets/qdrant_api_key.txt` | `/run/secrets/qdrant_api_key` | `QDRANT__SERVICE__API_KEY` / `QDRANT_API_KEY` |

> **Password Note:** Postgres reads secret files verbatim. Store the plain password string directly in `secrets/postgres_password.txt` without extra line breaks, spaces, or base64 encoding.

**Qdrant** has no native `*_FILE` support for its container variables, so its entrypoint script reads the secret file and exports `QDRANT__SERVICE__API_KEY` before launching. In the `app` container, `main.py` automatically reads the secret from `/run/secrets/qdrant_api_key` at startup and binds it to the environment.

Every request to Qdrant from external tools must send the key in the `api-key` header:

```bash
# From the host machine
curl -H "api-key: $(cat secrets/qdrant_api_key.txt)" http://localhost:6333/collections

```

### 4. Build and start the containers

```bash
docker-compose up -d --build

```

This command will:

1. Build the `app` image (downloads PyTorch, OpenCV, YOLO weights — takes ~5–10 min on first run).
2. Start `postgres`, `qdrant`, and `ollama`, waiting for their health checks to pass.
3. Run `app/main.py`, executing database schema migrations and data seeding.
4. Automatically launch the Streamlit frontend on port `8501`.

### 5. Verify the environment

```bash
docker exec -it vision_assist_app python verify_env.py

```

Expected output confirms: Python 3.11, PyTorch, OpenCV, YOLO26n loaded, database connection verified, and `OPENAI_API_KEY` present.

To check Qdrant health from inside the app container:

```bash
docker exec -it vision_assist_app python -c "import requests; print(requests.get('http://qdrant:6333/healthz').text)"

```

### 6. (Optional) Pull a local LLM model via Ollama

```bash
docker exec -it vision_assist_llm_local ollama pull llama3

```

Models are persisted in the `ollama_storage` Docker volume and will survive container restarts.

---

## Vision Model Configuration (YOLO / YOLOE)

`YOLOVisionEngine` defaults to `yolo26n.pt` (Ultralytics YOLO26, nano) — chosen for fewer parameters, higher accuracy, and ~30% faster CPU inference. To use a different model (e.g., a bigger variant on a GPU deployment), set an environment variable in your `.env` or Compose file:

```bash
YOLO_MODEL_PATH=yolo26m.pt

```

**Open-vocabulary detection (YOLOE):**
Set `VISION_MODEL_TYPE=yoloe` to switch to `YOLOEVisionEngine`, which detects arbitrary text-prompted classes instead of a fixed pretrained set. It tracks ~297 classes (COCO + curated household items) by default.

```bash
VISION_MODEL_TYPE=yoloe
# Optional — narrows detection to just these classes instead of the ~297-class default:
VISION_CUSTOM_CLASSES=keys,wallet,sunglasses,phone,backpack

```

**Easier: switch from the UI directly.** The camera panel has a live "Detection engine" radio button (YOLO/YOLOE) — no environment variables or restarts needed. Both engines are pre-warmed at login so switching is instant. See `YOLO_VS_YOLOE_GUIDE.md` for a full comparison.

---

## Running the App

The application launches automatically when the `app` container starts. Access the Streamlit interface directly in your browser:

👉 **`http://localhost:8501`**

**Useful container commands:**

```bash
# View application logs in real-time
docker logs -f vision_assist_app

# Drop into an interactive shell inside the running container
docker exec -it vision_assist_app bash

# Restart the application without rebuilding
docker-compose restart app

```

---

## Running Tests

Tests run locally without Docker (dependencies are stubbed automatically):

```bash
cd app
python3 -m pytest -v   # runs with 80% coverage gate

```

Or inside the container:

```bash
docker exec -it vision_assist_app bash -c "cd /workspace && python3 -m pytest -v"

```

### Pre-commit hook (one-time setup)

Commits that touch Python files are blocked if tests fail or coverage drops below 80%:

```bash
git config core.hooksPath .githooks

```

---

## Services Map

| Service | Port | Description |
| --- | --- | --- |
| `vision_assist_app` | 8501 | Main Python application orchestrator & Streamlit frontend |
| `vision_assist_llm_local` | 11434 | Ollama local LLM server |
| `vision_assist_db` | 5432 | PostgreSQL database (hostname `vision_assist_db`) |
| `vision_assist_qdrant` | 6333 / 6334 | Qdrant vector database (hostname `qdrant`) |

## Tech Stack

| Component | Technology |
| --- | --- |
| **Frontend** | Streamlit |
| **Speech Input** | OpenAI Whisper API / Faster-Whisper (offline fallback) |
| **Speech Output** | gTTS → MP3 → browser autoplay |
| **Object Detection** | YOLO26 (ultralytics), OpenCV, YOLOE (open-vocabulary) |
| **Cloud LLM** | OpenAI API |
| **Local LLM** | Ollama |
| **Orchestration** | LangChain |
| **Vector Memory** | Qdrant |
| **Relational Store** | PostgreSQL (auth, user configurations, persistence) |