"""Real TF-C HAR->Gesture checkpoint inference; numeric class ID only."""
from contextlib import asynccontextmanager
from math import isfinite
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from docker_app.verify_model import infer, load_models

_model = None
_classifier = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _classifier
    torch.set_num_threads(2)
    _model, _classifier = load_models()
    yield
    _model = _classifier = None

app = FastAPI(title="TF-C HAR to Gesture API", version="2.0.0", lifespan=lifespan)

class PredictRequest(BaseModel):
    samples: list[float] = Field(..., min_length=206, max_length=206, description="206 PREPROCESSED Gesture time-series values")

@app.get("/")
def index():
    return {"model": "Original TF-C", "protocol": "HAR pretraining -> Gesture fine-tuning", "predict": "/predict", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None and _classifier is not None}

@app.get("/model-info")
def model_info():
    return {"architecture": "Original TF-C dual TransformerEncoders + target_classifier", "input_length": 206, "num_classes": 8, "preprocessing": "Input must match training-time Gesture preprocessing", "class_names": None, "model_loaded": _model is not None and _classifier is not None}

@app.post("/predict")
def predict(payload: PredictRequest):
    if _model is None or _classifier is None:
        raise HTTPException(503, "Model unavailable")
    if not all(isfinite(v) for v in payload.samples):
        raise HTTPException(422, "All input values must be finite")
    class_id, probs = infer(_model, _classifier, payload.samples)
    return {"class_id": class_id, "probabilities": probs, "note": "Meaningful predictions require the exact Gesture training preprocessing and correct label mapping."}
