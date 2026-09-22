# EduPro ML Production API

## Endpoints

GET /health

GET /api-info

GET /model-status

POST /predict/instructor

POST /predict/course

POST /predict/instructor-batch

POST /predict/course-batch

## Authentication

Protected endpoints use the X-API-Key header.

Development key:

EDUPRO-DEMO-API-KEY-2026

Production deployment should use the EDUPRO_API_KEY
environment variable or a proper secret manager.

## Run

uvicorn api.production_api:app --host 0.0.0.0 --port 8000

## Important

For public deployment use HTTPS, a reverse proxy,
secure secret management, network controls and
production monitoring.