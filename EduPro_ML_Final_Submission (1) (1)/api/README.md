# EduPro ML Prediction API

## Project

Instructor Performance and Course Quality Evaluation.

## API

FastAPI-based ML prediction service.

## Endpoints

### Health

GET `/health`

### Model Status

GET `/model-status`

### Instructor Prediction

POST `/predict/instructor`

### Course Prediction

POST `/predict/course`

### Instructor Risk

POST `/predict/instructor-risk`

### Course Risk

POST `/predict/course-risk`

### Instructor Cluster

POST `/predict/instructor-cluster`

### Course Cluster

POST `/predict/course-cluster`

### Instructor Anomaly

POST `/predict/instructor-anomaly`

### Batch Prediction

POST `/predict/batch`

## Documentation

When server is running:

`/docs`

and

`/redoc`

## Important

Prediction inputs must use the same feature structure
expected by the trained model.

The API is a project prototype and should undergo
schema validation, authentication, monitoring and
security hardening before production deployment.