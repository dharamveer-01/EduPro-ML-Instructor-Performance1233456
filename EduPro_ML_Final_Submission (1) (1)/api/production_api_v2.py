
import os
import uuid
import time
from datetime import datetime

from fastapi import (
    FastAPI,
    Header,
    HTTPException
)

from pydantic import BaseModel, Field

from exact_model_server import (
    ExactModelServer
)


BASE_DIR = os.getenv(
    "EDUPRO_BASE_DIR",
    "/content/edupro_ml_system"
)

API_KEY = os.getenv(
    "EDUPRO_API_KEY",
    "EDUPRO-DEMO-API-KEY-2026"
)


app = FastAPI(
    title="EduPro Exact ML Production API",
    version="2.0.0",
    description=(
        "Production prediction API using "
        "exact registered models and schemas."
    )
)


server = ExactModelServer(
    BASE_DIR
)


class PredictionRequest(BaseModel):

    features: dict = Field(
        default_factory=dict
    )


class BatchPredictionRequest(BaseModel):

    records: list = Field(
        default_factory=list
    )


def request_id():

    return str(uuid.uuid4())


def authenticate(
    api_key
):

    if api_key != API_KEY:

        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "edupro-exact-ml-api",
        "version": "2.0.0",
        "timestamp":
            datetime.now().isoformat()
    }


@app.get("/api-info")
def api_info():

    return {
        "service":
            "EduPro Exact ML Production API",

        "version":
            "2.0.0",

        "registry":
            str(
                server.registry_path
            ),

        "authentication":
            "X-API-Key",

        "batch_limit":
            500
    }


@app.get("/model-status")
def model_status(
    x_api_key: str = Header(
        default=None
    )
):

    authenticate(x_api_key)

    return {
        "status": "ok",
        "models":
            server.status()
    }


@app.post("/predict/instructor")
def predict_instructor(
    request: PredictionRequest,
    x_api_key: str = Header(
        default=None
    )
):

    rid = request_id()

    authenticate(x_api_key)

    start = time.time()

    try:

        result = server.predict(
            "instructor",
            request.features
        )

        result["request_id"] = rid
        result["latency_ms"] = round(
            (time.time() - start) * 1000,
            3
        )

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail={
                "request_id": rid,
                "error": str(e)
            }
        )


@app.post("/predict/course")
def predict_course(
    request: PredictionRequest,
    x_api_key: str = Header(
        default=None
    )
):

    rid = request_id()

    authenticate(x_api_key)

    start = time.time()

    try:

        result = server.predict(
            "course",
            request.features
        )

        result["request_id"] = rid
        result["latency_ms"] = round(
            (time.time() - start) * 1000,
            3
        )

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail={
                "request_id": rid,
                "error": str(e)
            }
        )


@app.post("/predict/instructor-batch")
def predict_instructor_batch(
    request: BatchPredictionRequest,
    x_api_key: str = Header(
        default=None
    )
):

    authenticate(x_api_key)

    try:

        return server.batch_predict(
            "instructor",
            request.records
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


@app.post("/predict/course-batch")
def predict_course_batch(
    request: BatchPredictionRequest,
    x_api_key: str = Header(
        default=None
    )
):

    authenticate(x_api_key)

    try:

        return server.batch_predict(
            "course",
            request.records
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
