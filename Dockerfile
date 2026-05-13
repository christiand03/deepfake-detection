# ============================================================
# Deepfake Detection – Multi-Stage Production Image
# ============================================================
# Build:   docker build -t deepfake-detection .
# Serve:   docker run --gpus all -p 8000:8000 deepfake-detection
# Compose: docker compose up --build
# Test:    docker run deepfake-detection pytest
# ============================================================

# ── Stage 1: Build the React / Vite frontend ─────────────────────────────────
FROM node:24-slim AS frontend-build

WORKDIR /app/frontend

# Install deps first for layer caching
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --prefer-offline

# Build the production bundle
COPY frontend/ ./
RUN npm run build


# ── Stage 2: Python / CUDA runtime ───────────────────────────────────────────
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
        libgl1 \
        libglib2.0-0 \
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

# ---- Copy compiled frontend from Stage 1 ----
COPY --from=frontend-build /app/frontend/dist /workspace/frontend/dist

# ---- Expose API port ----
EXPOSE 8000

# ---- Start FastAPI server (serves API + static frontend) ----
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
