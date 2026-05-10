# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# System dependencies (faiss-cpu needs libgomp)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY api/requirements.txt /tmp/api-requirements.txt
RUN pip install --no-cache-dir -r /tmp/api-requirements.txt

# Copy application code
COPY api/        /app/api/
COPY worker/     /app/worker/
COPY logger/     /app/logger/
COPY db/         /app/db/
COPY eval/       /app/eval/
COPY docs/       /app/docs/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]