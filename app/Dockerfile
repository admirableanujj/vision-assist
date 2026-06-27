# ====================================================================
# STAGE 1: The Pristine Base (Heavy Dependencies Layer)
# ====================================================================
FROM python:3.11-slim AS ai_base

# Install system utilities required for OpenCV, audio hardware mapping, and eSpeak
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    portaudio19-dev \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    espeak \
    espeak-ng \
    alsa-utils \
    && rm -rf /var/lib/apt/lists/*

# Upgrade packaging pipeline tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Pre-install core, heavy AI libraries to keep them permanently cached
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  

RUN pip install --no-cache-dir \
    opencv-python-headless \
    ultralytics \
    chromadb \
    openai

# ====================================================================
# STAGE 2: Active Development Environment & App Layer
# ====================================================================
FROM ai_base AS ai_dev

WORKDIR /workspace

# Copy and install consolidated application-level dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# # Securely pre-download the YOLOv8 model by setting the legacy unpickling override
RUN python -c "import torch; _old_load = torch.load; torch.load = lambda *args, **kwargs: _old_load(*args, **{**kwargs, 'weights_only': False}); from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Keep container alive/interactive during development
CMD ["bash"]