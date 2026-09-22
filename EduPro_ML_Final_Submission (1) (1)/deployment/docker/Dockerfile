FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV EDUPRO_BASE_DIR=/app/edupro_ml_system

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY api /app/edupro_ml_system/api
COPY serving /app/edupro_ml_system/serving
COPY model_registry /app/edupro_ml_system/model_registry
COPY models /app/edupro_ml_system/models
COPY retraining /app/edupro_ml_system/retraining

ENV PYTHONPATH=/app/edupro_ml_system/api

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "production_api_v2:app", "--host", "0.0.0.0", "--port", "8000"]