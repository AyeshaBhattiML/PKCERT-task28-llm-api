# ============================================================

# TASK 28 - PART C

# Lightweight Production Dockerfile

# ============================================================

FROM python:3.11-slim

# Prevent Python from creating .pyc files

# and make logs appear immediately.

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Render provides PORT at runtime.

# 10000 is Render's default, while 8000 is useful locally.

ENV PORT=10000

WORKDIR /app

# Copy requirements first for Docker layer caching.

COPY requirements.txt .

# Install Python dependencies.

RUN pip install --no-cache-dir -r requirements.txt

# Copy application.

COPY app.py .

# Create a non-root application user.

RUN useradd 
--create-home 
--shell /bin/bash 
appuser

RUN chown -R appuser:appuser /app

USER appuser

# Document the application port.

EXPOSE 10000

# Docker health check.

HEALTHCHECK --interval=30s 
--timeout=10s 
--start-period=20s 
--retries=3 
CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:' + os.getenv('PORT', '10000') + '/healthz')"

# Render-compatible startup command.

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}"]
