import os
import tempfile
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

from src.config.schema import AppConfig
from src.inference.engine import AvianInferenceEngine
from src.features.registry import list_registered_extractors
from src.models.model_registry import list_registered_models
from src.utils.logging import setup_logger

logger = setup_logger("FastAPI_Service")

config = AppConfig()
engine = AvianInferenceEngine(config)

app = FastAPI(
    title="Avian Passive Acoustic Monitoring (PAM) REST API",
    description="Production Machine Learning REST API for Bird Population Estimation.",
    version=config.project.version,
    docs_url="/docs",
    redoc_url="/redoc",
)


class HealthResponse(BaseModel):
    status: str
    environment: str
    project: str


class PredictionResponse(BaseModel):
    filename: str
    predicted_bird_count: float
    estimated_integer_count: int
    duration_sec: float
    inference_latency_ms: float
    feature_count_extracted: int
    model_used: str


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Health check endpoint confirming API service status."""
    return HealthResponse(
        status="HEALTHY",
        environment=config.project.environment,
        project=config.project.name,
    )


@app.get("/version", tags=["Metadata"])
def get_version():
    """Returns project version and seed metadata."""
    return {
        "version": config.project.version,
        "seed": config.project.seed,
        "environment": config.project.environment,
    }


@app.get("/model_info", tags=["Metadata"])
def get_model_info():
    """Returns details of loaded inference model and registered algorithms."""
    return {
        "active_model_name": engine.model.name,
        "active_model_version": engine.model.version,
        "is_fitted": engine.model.is_fitted,
        "registered_models": list_registered_models(),
    }


@app.get("/feature_info", tags=["Metadata"])
def get_feature_info():
    """Returns information on active feature extractor plugins."""
    return {
        "active_extractors": config.features.active_extractors,
        "registered_extractors": list_registered_extractors(),
        "store_format": config.features.store_format,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_audio(file: UploadFile = File(...)):
    """Predicts bird count for an uploaded audio file (.wav, .flac, .mp3)."""
    if not file.filename.endswith((".wav", ".flac", ".ogg", ".mp3")):
        raise HTTPException(
            status_code=400,
            detail="Invalid audio file format. Must be .wav, .flac, .ogg, or .mp3.",
        )

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(file.filename).suffix
    ) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        res = engine.predict_audio_file(tmp_path)
        res["filename"] = file.filename
        return PredictionResponse(**res)
    except Exception as e:
        logger.error(f"Prediction error on '{file.filename}': {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
