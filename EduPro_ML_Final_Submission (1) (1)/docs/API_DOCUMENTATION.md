# EduPro Production API

## Base URL

http://localhost:8000

## Authentication

Protected endpoints use the X-API-Key HTTP header.

Example:

X-API-Key: YOUR_API_KEY

## Endpoints

GET /health

GET /api-info

GET /model-status

POST /predict/instructor

POST /predict/course

POST /predict/instructor-batch

POST /predict/course-batch

## Batch Limit

The production API configuration limits batch prediction requests
to a maximum of 500 records.

## Security Note

Production deployments should use a securely managed secret instead
of the demonstration API key.