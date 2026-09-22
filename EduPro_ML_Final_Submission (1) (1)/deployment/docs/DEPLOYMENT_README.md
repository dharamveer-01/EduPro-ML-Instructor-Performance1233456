# EduPro ML Production Deployment

## Overview

Production deployment package for the EduPro Instructor Performance ML system.

## Components

- FastAPI prediction service
- Exact model registry
- Exact feature schemas
- Instructor prediction model
- Course prediction model
- API authentication
- Batch prediction
- Docker health check

## Local Deployment

Copy `.env.example` to `.env`.

Set a secure API key.

Run:

docker compose up --build

API:

http://localhost:8000

## Health

GET /health

## Model Status

GET /model-status

Requires the X-API-Key header.

## Instructor Prediction

POST /predict/instructor

## Course Prediction

POST /predict/course

## Batch Prediction

Maximum batch size: 500

## Security

Do not use the demonstration API key in a real deployment.

Use a strong randomly generated secret.

## Production Notes

Actual deployment should additionally use:

- HTTPS/TLS
- Secure secret management
- Network restrictions
- Monitoring
- Backups
- Access control
- Infrastructure testing