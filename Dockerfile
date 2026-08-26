# ============================================================
# TASK 28 - PART C
# Lightweight Production Dockerfile
# FastAPI + Hugging Face Remote Inference + Render
# ============================================================

FROM python:3.11-slim

# Prevent Python from creating .pyc files
# and make logs appear immediately.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Render provides PORT at runtime.
# 10000 is Render's default port.
ENV PORT=10000

# Application working directory
WORKDIR /app

# Copy requirements first for Docker layer caching.
COPY requirements.txt .

# Install lightweight Python dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code.
COPY app.py .

# Create a non-root application user.
RUN useradd --create-home --shell /bin/bash appuser

# Give the application user ownership of the application files.
RUN chown -R appuser:appuser /app

# Run the application as a non-root user.
USER appuser

# Document the port used by the application.
EXPOSE 10000

# Docker health check.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:' + os.getenv('PORT', '10000') + '/healthz')"

# Render-compatible startup command.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}"]