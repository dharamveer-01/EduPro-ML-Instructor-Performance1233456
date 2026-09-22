
import json
import time
import uuid

from pathlib import Path
from typing import Dict, Any, List

from fastapi import (
    FastAPI,
    Depends,
    HTTPException
)

from pydantic import BaseModel, Field

from security.security import verify_api_key
from logger import log_request
from validation import (
    validate_numeric_features,
    validate_batch_size
)


BASE = Path(__file__).resolve().parent.parent


app = FastAPI(
    title="EduPro ML Production API",
    description=(
        "Production API for EduPro instructor "
        "and course machine-learning predictions."
    ),
    version="1.0.0"
)


# ============================================================
# REQUEST MODELS
# ============================================================

class PredictionRequest(BaseModel):

    features: Dict[str, Any] = Field(
        default_factory=dict
    )


class BatchPredictionRequest(BaseModel):

    records: List[Dict[str, Any]]


# ============================================================
# MODEL STORAGE
# ============================================================

MODEL_DIR = BASE / "models"
RETRAIN_DIR = BASE / "retraining" / "models"

loaded_models = {}


def load_pickle(path):

    try:

        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)

    except Exception:

        return None


def discover_models():

    candidates = {}

    for root in [MODEL_DIR, RETRAIN_DIR]:

        if not root.exists():
            continue

        for path in root.rglob("*.pkl"):

            name = path.stem.lower()

            if "instructor" in name:
                candidates["instructor"] = path

            elif "course" in name:
                candidates["course"] = path

    return candidates


def load_production_models():

    discovered = discover_models()

    for entity, path in discovered.items():

        model = load_pickle(path)

        if model is not None:

            loaded_models[entity] = {
                "model": model,
                "path": str(path)
            }


load_production_models()


# ============================================================
# PREDICTION ENGINE
# ============================================================

def predict_entity(entity, features):

    if entity not in loaded_models:

        raise HTTPException(
            status_code=503,
            detail=f"{entity} model is not available."
        )

    cleaned = validate_numeric_features(
        features
    )

    model = loaded_models[entity]["model"]

    try:

        import pandas as pd

        schema_path = (
            BASE /
            "serving" /
            f"{entity}_feature_schema.json"
        )

        if schema_path.exists():

            with open(schema_path, "r") as f:
                schema = json.load(f)

            feature_names = schema.get(
                "feature_names",
                schema.get("features", [])
            )

            medians = schema.get(
                "feature_medians",
                {}
            )

            row = {}

            for feature in feature_names:

                if feature in cleaned:
                    row[feature] = cleaned[feature]

                else:
                    row[feature] = float(
                        medians.get(feature, 0.0)
                    )

            X = pd.DataFrame(
                [row],
                columns=feature_names
            )

        else:

            X = pd.DataFrame([cleaned])

        prediction = model.predict(X)

        return {
            "entity": entity,
            "prediction": float(prediction[0]),
            "features_used": int(X.shape[1])
        }

    except Exception as e:

        raise HTTPException(
            status_code=422,
            detail=f"Prediction failed: {str(e)}"
        )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "service": "EduPro ML API",
        "models_loaded": list(
            loaded_models.keys()
        )
    }


# ============================================================
# API INFO
# ============================================================

@app.get("/api-info")
async def api_info():

    return {
        "name": "EduPro ML Production API",
        "version": "1.0.0",
        "authentication": "X-API-Key",
        "batch_limit": 500,
        "endpoints": [
            "/health",
            "/api-info",
            "/model-status",
            "/predict/instructor",
            "/predict/course",
            "/predict/instructor-batch",
            "/predict/course-batch"
        ]
    }


# ============================================================
# MODEL STATUS
# ============================================================

@app.get("/model-status")
async def model_status(
    authenticated: bool = Depends(
        verify_api_key
    )
):

    result = {}

    for entity, info in loaded_models.items():

        result[entity] = {
            "loaded": True,
            "path": info["path"]
        }

    return {
        "status": "ok",
        "models": result
    }


# ============================================================
# INSTRUCTOR PREDICTION
# ============================================================

@app.post("/predict/instructor")
async def instructor_prediction(
    request: PredictionRequest,
    authenticated: bool = Depends(
        verify_api_key
    )
):

    request_id = str(uuid.uuid4())
    start = time.time()

    try:

        result = predict_entity(
            "instructor",
            request.features
        )

        latency = (
            time.time() - start
        ) * 1000

        log_request(
            endpoint="/predict/instructor",
            status_code=200,
            latency_ms=latency,
            request_id=request_id
        )

        result["request_id"] = request_id
        result["latency_ms"] = round(
            latency,
            3
        )

        return result

    except Exception as e:

        latency = (
            time.time() - start
        ) * 1000

        log_request(
            endpoint="/predict/instructor",
            status_code=500,
            latency_ms=latency,
            request_id=request_id,
            error=str(e)
        )

        raise


# ============================================================
# COURSE PREDICTION
# ============================================================

@app.post("/predict/course")
async def course_prediction(
    request: PredictionRequest,
    authenticated: bool = Depends(
        verify_api_key
    )
):

    request_id = str(uuid.uuid4())
    start = time.time()

    try:

        result = predict_entity(
            "course",
            request.features
        )

        latency = (
            time.time() - start
        ) * 1000

        log_request(
            endpoint="/predict/course",
            status_code=200,
            latency_ms=latency,
            request_id=request_id
        )

        result["request_id"] = request_id
        result["latency_ms"] = round(
            latency,
            3
        )

        return result

    except Exception as e:

        latency = (
            time.time() - start
        ) * 1000

        log_request(
            endpoint="/predict/course",
            status_code=500,
            latency_ms=latency,
            request_id=request_id,
            error=str(e)
        )

        raise


# ============================================================
# INSTRUCTOR BATCH
# ============================================================

@app.post("/predict/instructor-batch")
async def instructor_batch(
    request: BatchPredictionRequest,
    authenticated: bool = Depends(
        verify_api_key
    )
):

    validate_batch_size(
        request.records,
        500
    )

    results = []

    for index, record in enumerate(
        request.records
    ):

        try:

            result = predict_entity(
                "instructor",
                record
            )

            result["record_index"] = index
            result["status"] = "success"

            results.append(result)

        except Exception as e:

            results.append({
                "record_index": index,
                "status": "failed",
                "error": str(e)
            })

    return {
        "entity": "instructor",
        "count": len(results),
        "results": results
    }


# ============================================================
# COURSE BATCH
# ============================================================

@app.post("/predict/course-batch")
async def course_batch(
    request: BatchPredictionRequest,
    authenticated: bool = Depends(
        verify_api_key
    )
):

    validate_batch_size(
        request.records,
        500
    )

    results = []

    for index, record in enumerate(
        request.records
    ):

        try:

            result = predict_entity(
                "course",
                record
            )

            result["record_index"] = index
            result["status"] = "success"

            results.append(result)

        except Exception as e:

            results.append({
                "record_index": index,
                "status": "failed",
                "error": str(e)
            })

    return {
        "entity": "course",
        "count": len(results),
        "results": results
    }
