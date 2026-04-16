# ============================================================
# Deepfake Detection – Reproducible Environment
# ============================================================
# Build:  docker build -t deepfake-detection .
# Run:    docker run --gpus all -it deepfake-detection
# Test:   docker run deepfake-detection pytest
# ============================================================

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ---- System dependencies ----
RUN sed -i 's|http://archive.ubuntu.com|http://de.archive.ubuntu.com|g' /etc/apt/sources.list \
    && apt-get -o Acquire::Retries=5 update \
    && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# ---- Set python3.11 as default ----
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# ---- Working directory ----
WORKDIR /workspace

# ---- Install Python dependencies ----
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-dev.txt

# ---- Copy project source ----
COPY . .

# ---- Default command ----
CMD ["python", "-c", "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"]
