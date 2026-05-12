"""
Prediction API for FlipPhone trick classification.

Usage:
    python server.py                    # starts on port 8000
    PORT=5000 python server.py          # custom port
"""

import os
import pickle

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from train import extract_features

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "rf_model.pkl")


class Sample(BaseModel):
    t: float
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float


class PredictRequest(BaseModel):
    samples: list[Sample]


class PredictResponse(BaseModel):
    trick: str
    confidence: float
    probabilities: dict[str, float]


class EmbedRecording(BaseModel):
    id: str
    samples: list[Sample]


class BatchEmbedRequest(BaseModel):
    recordings: list[EmbedRecording]


class EmbedResult(BaseModel):
    id: str
    features: list[float] | None


# ── Load model ──────────────────────────────────────────────────────

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("No model found. Run train.py first.")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


model_data = load_model()
clf = model_data["clf"]
le = model_data["label_encoder"]
feature_cols = model_data["feature_cols"]

# ── App ─────────────────────────────────────────────────────────────

app = FastAPI(title="FlipPhone Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if len(req.samples) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 samples")

    df = pd.DataFrame([s.model_dump() for s in req.samples])
    features = extract_features(df)

    X = np.array([[features[col] for col in feature_cols]])
    proba = clf.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))

    return PredictResponse(
        trick=le.classes_[pred_idx],
        confidence=round(float(proba[pred_idx]), 4),
        probabilities={name: round(float(p), 4) for name, p in zip(le.classes_, proba)},
    )


@app.post("/batch_embed", response_model=list[EmbedResult])
def batch_embed(req: BatchEmbedRequest):
    results = []
    for rec in req.recordings:
        try:
            if len(rec.samples) < 2:
                raise ValueError("need at least 2 samples")
            df = pd.DataFrame([s.model_dump() for s in rec.samples])
            feats = extract_features(df)
            vec = [float(feats[col]) for col in feature_cols]
            results.append(EmbedResult(id=rec.id, features=vec))
        except Exception:
            results.append(EmbedResult(id=rec.id, features=None))
    return results


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
