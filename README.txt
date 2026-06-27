vision-assist/
├── docker-compose.yml
├── app/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── main.py
======================================
visionassistLFAI_local/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── app/                         <-- Main Application Domain
    ├── app.py                   <-- Core UI & Event Orchestrator
    ├── vision_engine.py         <-- Computer Vision (YOLO) Module
    ├── voice_engine/            <-- Audio Processing Domain
    │   ├── __init__.py
    │   ├── voice_stt.py
    │   └── voice_tts.py
    └── ml_engine/               <-- Core Analytics Domain (App Level)
        ├── __init__.py          <-- Exposes engines cleanly
        └── ml_engine.py         <-- BaseMLEngine & OllamaMLEngine
======================================
Hello team! Follow these exact steps to spin up the local development environment on your Windows machine so we are all coding on identical systems.

Step 1: Baseline Prerequisites
Open PowerShell as Admin and run wsl --install to ensure WSL2 is ready. (Restart if prompted).

Download and install Docker Desktop for Windows. Ensure WSL 2 engine is selected in the settings.

Create a file named .env in the root project directory and add your key:

Plaintext
OPENAI_API_KEY=your_actual_key_here
Step 2: Spin Up the Stack
Open your terminal (Command Prompt, PowerShell, or Git Bash) inside the project root folder and execute:

Bash
# Build the application layer and download the LLM service
==> docker compose up -d --build
Note: The initial build will take a few minutes as it downloads the base environments and YOLO weights.

Step 3: Download the Local LLM
We are offloading our LLM processing to the local ollama container. Let's pull a lightweight model (like Llama 3) into it:

Bash
==>docker exec -it vision_assist_llm ollama run llama3:8b

Once the download hits 100%, you can test typing to it. Type /exit to close the interactive LLM prompt.

Step 4: Interact and Code
Your local files are linked directly inside the container. Any changes you make in VS Code or your local text editor will instantly update inside Docker.

To run or debug your python scripts inside the synchronized dev space, enter the application container:

Bash
==> docker exec -it vision_assist_app bash

You are now at a Linux command prompt inside our core app. You can run python code using:

Bash
python main.py