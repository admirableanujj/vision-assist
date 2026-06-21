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
docker exec -it vision_assist_app python main.py
```

Or drop into an interactive shell:

```bash
docker exec -it vision_assist_app bash
```

## Services

| Service | Port | Description |
|---|---|---|
| `vision_assist_app` | 8000 | Main Python application |
| `vision_assist_llm` | 11434 | Ollama local LLM server |

## Stack

| Component | Library |
|---|---|
| Object detection | YOLOv8 (ultralytics), OpenCV |
| Speech input | SpeechRecognition |
| Speech output | pyttsx3 |
| Cloud LLM | OpenAI API |
| Local LLM | Ollama |
| Orchestration | LangChain |
| Vector memory | ChromaDB |
