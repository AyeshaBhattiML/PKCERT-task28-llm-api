# ============================================================
# TASK 28 - PART C
# Multi-Stage Dockerfile
# ============================================================


# ============================================================
# STAGE 1: BUILDER
# ============================================================

FROM python:3.11-slim AS builder

# Prevent Python from creating .pyc files
# and make Python output appear immediately.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory.
WORKDIR /app

# Install build dependencies.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        && rm -rf /var/lib/apt/lists/*

# Copy dependency file first.
# This improves Docker build caching.
COPY requirements.txt .

# Install Python dependencies into /install.
RUN pip install --no-cache-dir \
    --prefix=/install \
    -r requirements.txt


# ============================================================
# STAGE 2: PRODUCTION IMAGE
# ============================================================

FROM python:3.11-slim AS production

# Python configuration.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory.
WORKDIR /app

# Copy installed Python packages
# from the builder stage.
COPY --from=builder /install /usr/local

# Copy application source code.
COPY app.py .

# Create a dedicated non-root user.
RUN useradd \
    --create-home \
    --shell /bin/bash \
    appuser

# Give the application user ownership
# of the application directory.
RUN chown -R appuser:appuser /app

# Switch from root to non-root user.
USER appuser

# Application port.
EXPOSE 8000

# Health check.
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"
# Start FastAPI using Uvicorn.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]