# ===========================
# PRISM Dockerfile
# Multi-stage build for the PRISM platform
# ===========================

# ---- Stage 1: Builder ----
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    libffi-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry==1.8.4

# Copy dependency files first (for Docker layer caching)
COPY pyproject.toml poetry.lock* ./

# Configure Poetry: no virtualenvs in container
RUN poetry config virtualenvs.create false

# Install dependencies (without dev deps for production)
RUN poetry install --no-interaction --no-ansi --without dev

# ---- Stage 2: Runtime ----
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create directories for runtime data
RUN mkdir -p /app/vector_store /app/logs /app/models/checkpoints

# Expose ports
EXPOSE 8000 8501

# Default: start the backend API
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
