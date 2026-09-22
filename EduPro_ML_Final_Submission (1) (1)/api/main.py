
import os
import json
import pickle
import logging
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = "/content/edupro_ml_system"

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

LOG_DIR = os.path.join(
    BASE_DIR,
    "logs"
)

os.makedirs(
    LOG_DIR,
    exist_ok=True
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    filename=os.path.join(
        LOG_DIR,
        "prediction_api.log"
    ),
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    )
)

logger = logging.getLogger(
    "EduProMLAPI"
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(

    title="EduPro ML Prediction API",

    description=(
        "Production-style prediction API for "
        "Instructor Performance and Course Quality "
        "Evaluation."
    ),

    version="1.0.0"
)


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_PATHS = {

    "instructor":
        os.path.join(
            MODEL_DIR,
            "ensemble",
            "best_instructor_ensemble_model.pkl"
        ),

    "course":
        os.path.join(
            MODEL_DIR,
            "ensemble",
            "best_course_ensemble_model.pkl"
        ),

    "instructor_risk":
        os.path.join(
            MODEL_DIR,
            "risk",
            "instructor_risk_model.pkl"
        ),

    "course_risk":
        os.path.join(
            MODEL_DIR,
            "risk",
            "course_risk_model.pkl"
        ),

    "instructor_cluster":
        os.path.join(
            MODEL_DIR,
            "clustering",
            "instructor_kmeans.pkl"
        ),

    "course_cluster":
        os.path.join(
            MODEL_DIR,
            "clustering",
            "course_kmeans.pkl"
        ),

    "instructor_anomaly":
        os.path.join(
            MODEL_DIR,
            "risk",
            "instructor_isolation_forest.pkl"
        )
}


# ============================================================
# MODEL CACHE
# ============================================================

MODEL_CACHE = {}


def load_model(
    model_name: str
):

    if model_name in MODEL_CACHE:

        return MODEL_CACHE[
            model_name
        ]

    path = MODEL_PATHS.get(
        model_name
    )

    if path is None:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    if not os.path.exists(
        path
    ):

        raise FileNotFoundError(
            f"Model not found: {path}"
        )

    with open(
        path,
        "rb"
    ) as file:

        model = pickle.load(
            file
        )

    MODEL_CACHE[
        model_name
    ] = model

    logger.info(
        f"Model loaded: {model_name}"
    )

    return model


# ============================================================
# MODEL AVAILABILITY
# ============================================================

def model_status():

    result = {}

    for name, path in MODEL_PATHS.items():

        result[name] = {

            "available":
                os.path.exists(path),

            "path":
                path,

            "cached":
                name in MODEL_CACHE
        }

    return result


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class PredictionRequest(
    BaseModel
):

    features: Dict[
        str,
        float
    ] = Field(
        default_factory=dict
    )


class BatchPredictionRequest(
    BaseModel
):

    records: list[
        Dict[str, float]
    ] = Field(
        default_factory=list
    )


class HealthResponse(
    BaseModel
):

    status: str

    service: str

    timestamp: str


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_features(
    features: Dict[str, float]
):

    if not features:

        raise ValueError(
            "At least one feature is required."
        )

    cleaned = {}

    for key, value in features.items():

        try:

            cleaned[
                str(key)
            ] = float(value)

        except Exception:

            raise ValueError(
                f"Feature '{key}' "
                f"must be numeric."
            )

    frame = pd.DataFrame(
        [cleaned]
    )

    frame = frame.replace(
        [np.inf, -np.inf],
        np.nan
    )

    frame = frame.fillna(
        0
    )

    return frame


# ============================================================
# GENERIC PREDICTION
# ============================================================

def perform_prediction(
    model_name,
    features
):

    model = load_model(
        model_name
    )

    X = prepare_features(
        features
    )

    try:

        prediction = model.predict(
            X
        )

    except Exception as error:

        logger.error(
            f"Prediction failed: "
            f"{model_name} | {error}"
        )

        raise ValueError(
            "Input features do not match "
            "the trained model schema. "
            "Use the same feature structure "
            "used during model training."
        )

    result = prediction.tolist()

    return result


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse
)
def health():

    return {

        "status":
            "healthy",

        "service":
            "EduPro ML Prediction API",

        "timestamp":
            datetime.utcnow().isoformat()
    }


# ============================================================
# MODEL STATUS ENDPOINT
# ============================================================

@app.get(
    "/model-status"
)
def get_model_status():

    return {

        "timestamp":
            datetime.utcnow().isoformat(),

        "models":
            model_status()
    }


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {

        "application":
            "EduPro ML Prediction API",

        "version":
            "1.0.0",

        "status":
            "running",

        "endpoints": [

            "/health",

            "/model-status",

            "/predict/instructor",

            "/predict/course",

            "/predict/instructor-risk",

            "/predict/course-risk",

            "/predict/instructor-cluster",

            "/predict/course-cluster",

            "/predict/instructor-anomaly",

            "/predict/batch"
        ]
    }


# ============================================================
# INSTRUCTOR PREDICTION
# ============================================================

@app.post(
    "/predict/instructor"
)
def predict_instructor(
    request: PredictionRequest
):

    prediction = perform_prediction(
        "instructor",
        request.features
    )

    logger.info(
        "Instructor prediction generated"
    )

    return {

        "model":
            "best_instructor_ensemble_model",

        "prediction":
            prediction,

        "timestamp":
            datetime.utcnow().isoformat()
    }


# ============================================================
# COURSE PREDICTION
# ============================================================

@app.post(
    "/predict/course"
)
def predict_course(
    request: PredictionRequest
):

    prediction = perform_prediction(
        "course",
        request.features
    )

    logger.info(
        "Course prediction generated"
    )

    return {

        "model":
            "best_course_ensemble_model",

        "prediction":
            prediction,

        "timestamp":
            datetime.utcnow().isoformat()
    }


# ============================================================
# INSTRUCTOR RISK
# ============================================================

@app.post(
    "/predict/instructor-risk"
)
def predict_instructor_risk(
    request: PredictionRequest
):

    prediction = perform_prediction(
        "instructor_risk",
        request.features
    )

    logger.info(
        "Instructor risk prediction generated"
    )

    return {

        "model":
            "instructor_risk",

        "risk_prediction":
            prediction,

        "timestamp":
            datetime.utcnow().isoformat()
    }


# ============================================================
# COURSE RISK
# ============================================================

@app.post(
    "/predict/course-risk"
)
def predict_course_risk(
    request: PredictionRequest
):

    prediction = perform_prediction(
        "course_risk",
        request.features
    )

    logger.info(
        "Course risk prediction generated"
    )

    return {

        "model":
            "course_risk",

        "risk_prediction":
            prediction,

        "timestamp":
            datetime.utcnow().isoformat()
    }


# ============================================================
# INSTRUCTOR CLUSTER
# ============================================================

@app.post(
    "/predict/instructor-cluster"
)
def predict_instructor_cluster(
    request: PredictionRequest
):

    prediction = perform_prediction(
        "instructor_cluster",
        request.features
    )

    logger.info(
        "Instructor cluster generated"
    )

    return {

        "model":
            "instructor_kmeans",

        "cluster":
            prediction,

        "timestamp":
            datetime.utcnow().isoformat()
    }


# ============================================================
# COURSE CLUSTER
# ============================================================

@app.post(
    "/predict/course-cluster"
)
def predict_course_cluster(
    request: PredictionRequest
):

    prediction = perform_prediction(
        "course_cluster",
        request.features
    )

    logger.info(
        "Course cluster generated"
    )

    return {

        "model":
            "course_kmeans",

        "cluster":
            prediction,

        "timestamp":
            datetime.utcnow().isoformat()
    }


# ============================================================
# INSTRUCTOR ANOMALY
# ============================================================

@app.post(
    "/predict/instructor-anomaly"
)
def predict_instructor_anomaly(
    request: PredictionRequest
):

    prediction = perform_prediction(
        "instructor_anomaly",
        request.features
    )

    logger.info(
        "Instructor anomaly generated"
    )

    return {

        "model":
            "instructor_isolation_forest",

        "anomaly":
            prediction,

        "timestamp":
            datetime.utcnow().isoformat()
    }


# ============================================================
# BATCH PREDICTION
# ============================================================

@app.post(
    "/predict/batch"
)
def predict_batch(
    request: BatchPredictionRequest
):

    if not request.records:

        raise HTTPException(
            status_code=400,
            detail="No records supplied."
        )

    results = []

    for record in request.records:

        try:

            prediction = (
                perform_prediction(
                    "instructor",
                    record
                )
            )

            results.append({

                "status":
                    "success",

                "prediction":
                    prediction
            })

        except Exception as error:

            results.append({

                "status":
                    "error",

                "message":
                    str(error)
            })

    return {

        "count":
            len(results),

        "results":
            results,

        "timestamp":
            datetime.utcnow().isoformat()
    }


# ============================================================
# EXCEPTION HANDLER
# ============================================================

@app.exception_handler(
    Exception
)
async def global_exception_handler(
    request,
    exc
):

    logger.error(
        f"Unhandled exception: {exc}"
    )

    return {

        "error":
            "Internal server error",

        "message":
            str(exc),

        "timestamp":
            datetime.utcnow().isoformat()
    }


# ============================================================
# API INFORMATION
# ============================================================

@app.get(
    "/api-info"
)
def api_info():

    return {

        "project":
            "EduPro Instructor Performance "
            "and Course Quality Evaluation",

        "api":
            "ML Prediction API",

        "version":
            "1.0.0",

        "ml_capabilities": [

            "Instructor rating prediction",

            "Course quality prediction",

            "Instructor risk classification",

            "Course risk classification",

            "Instructor clustering",

            "Course clustering",

            "Instructor anomaly detection",

            "Batch prediction"
        ],

        "documentation":
            "/docs",

        "redoc":
            "/redoc"
    }
