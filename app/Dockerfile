# ==========================================
# STAGE 1: The Pristine Base (Heavy Dependencies)
# ==========================================
# Using the official Python 3.11 slim image as our bedrock
FROM python:3.11-slim AS ai_base

# Install system utilities required for OpenCV, audio processing, and building C++ extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    portaudio19-dev \
    python3-pyaudio \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Pre-install core, heavy AI libraries so they are cached forever
RUN pip install --no-cache-dir --upgrade pip 
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  
RUN pip install --no-cache-dir \
    opencv-python-headless \
    ultralytics \
    chromadb \
    openai

# ==========================================
# STAGE 2: Your Active Development Environment
# ==========================================
FROM ai_base AS ai_dev

WORKDIR /workspace

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the small YOLO model so your team doesn't experience launch lag
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Keep container alive/interactive during development
CMD ["bash"]