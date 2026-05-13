"""
Prediction API for FlipPhone trick classification.

Usage:
    python server.py                    # starts on port 8000
    PORT=5000 python server.py          # custom port
"""

import json
import os
import pickle
import sqlite3
import threading
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from train import DATA_PATH, MODEL_DIR, embed_3d, extract_features, fetch_tricks, load_features

# ── Paths ─────────────────────────────────────────────────────────────

FLIPPHONE_URL = os.environ.get("FLIPPHONE_URL", "").rstrip("/")
FLIPPHONE_API_KEY = os.environ.get("FLIPPHONE_API_KEY", "")

DB_PATH = os.path.join(os.path.dirname(__file__), "models.db")
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "rf_model.pkl")
_AUGMENTED_PATH = os.path.join(os.path.dirname(__file__), "data", "dataset_augmented.csv")
_DATA_PATH = _AUGMENTED_PATH if os.path.exists(_AUGMENTED_PATH) else DATA_PATH


# ── Pydantic models ───────────────────────────────────────────────────

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


class EmbedCoord(BaseModel):
    id: str
    x: float
    y: float
    z: float


class TrainingRunSummary(BaseModel):
    id: int
    name: str | None
    model_type: str
    tricks: list[str]
    source_users: list[str] | None
    n_samples: int | None
    n_classes: int | None
    accuracy: float | None
    model_path: str | None
    status: str
    error: str | None
    started_at: str | None
    finished_at: str | None
    created_at: str
    is_active: bool


class TrainingRunDetail(TrainingRunSummary):
    metrics_json: dict | None


class ActivateResponse(BaseModel):
    ok: bool
    model: TrainingRunSummary


class TrainRequest(BaseModel):
    name: str | None = None
    model_type: str               # "random_forest" | "svm" | "mlp"
    tricks: list[str]
    source_users: list[str] | None = None


class TrainJobResponse(BaseModel):
    job_id: int


# ── DB helpers ────────────────────────────────────────────────────────

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS training_runs (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                name         TEXT,
                model_type   TEXT NOT NULL,
                tricks       TEXT NOT NULL,
                source_users TEXT,
                n_samples    INTEGER,
                n_classes    INTEGER,
                accuracy     REAL,
                metrics_json TEXT,
                model_path   TEXT,
                status       TEXT DEFAULT 'pending',
                error        TEXT,
                started_at   DATETIME,
                finished_at  DATETIME,
                created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS active_model (
                id     INTEGER PRIMARY KEY CHECK (id = 1),
                run_id INTEGER REFERENCES training_runs(id),
                set_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)


def _get_active_run_id(conn: sqlite3.Connection) -> int | None:
    row = conn.execute("SELECT run_id FROM active_model WHERE id = 1").fetchone()
    return row["run_id"] if row else None


def _row_to_summary(row: sqlite3.Row, active_run_id: int | None) -> dict:
    d = dict(row)
    d["tricks"] = json.loads(d["tricks"]) if d.get("tricks") else []
    d["source_users"] = json.loads(d["source_users"]) if d.get("source_users") else None
    d["is_active"] = active_run_id is not None and d["id"] == active_run_id
    d.pop("metrics_json", None)
    return d


# ── Active-model state ────────────────────────────────────────────────

_active: dict = {}


def _load_pkl(path: str) -> None:
    with open(path, "rb") as f:
        data = pickle.load(f)
    # "model" is the current key; "clf" is the legacy key from train.py CLI output
    clf = data.get("model") or data["clf"]
    # Legacy MLP pkls stored scaler separately — wrap into a Pipeline so
    # predict() works without knowing the model type.
    if "scaler" in data:
        clf = Pipeline([("scaler", data["scaler"]), ("clf", clf)])
    _active["clf"] = clf
    _active["le"] = data["label_encoder"]
    _active["feature_cols"] = data["feature_cols"]


def _require_model() -> None:
    if not _active:
        raise HTTPException(
            status_code=503,
            detail="No active model. Use POST /models/<id>/activate first.",
        )


# ── Background training ───────────────────────────────────────────────

def _build_estimator(model_type: str):
    if model_type == "random_forest":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    if model_type == "mlp":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                                  max_iter=500, random_state=42)),
        ])
    if model_type == "svm":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=42)),
        ])
    raise ValueError(f"Unknown model_type: {model_type!r}. Choose random_forest, mlp or svm.")


_MIN_SAMPLES_PER_CLASS = 20


def _run_training(
    run_id: int,
    model_type: str,
    tricks: list[str],
    source_users: list[str] | None,
    name: str | None,
) -> None:
    def _db(sql, params=()):
        with get_db() as c:
            c.execute(sql, params)

    now = lambda: datetime.now(timezone.utc).isoformat()

    try:
        _db("UPDATE training_runs SET status='running', started_at=? WHERE id=?", (now(), run_id))

        if FLIPPHONE_URL and FLIPPHONE_API_KEY:
            from fetch_data import fetch as _fetch_data
            try:
                _fetch_data(FLIPPHONE_URL, FLIPPHONE_API_KEY)
            except Exception as fetch_err:
                raise RuntimeError(
                    f"Failed to fetch dataset from {FLIPPHONE_URL}: {fetch_err}"
                ) from fetch_err
        elif not os.path.exists(_DATA_PATH):
            raise RuntimeError(
                f"No dataset at {_DATA_PATH} and FLIPPHONE_URL / FLIPPHONE_API_KEY are not "
                "configured. Set these environment variables on the training server."
            )

        resolved_tricks = tricks or [t["id"] for t in fetch_tricks(FLIPPHONE_URL or None)]
        feat_df, feature_cols = load_features(
            data_path=_DATA_PATH,
            selected_tricks=resolved_tricks or None,
            selected_collectors=source_users,
        )

        # Enforce minimum samples per class before doing any work
        counts = feat_df["trick"].value_counts()
        for trick_name, n in counts.items():
            if n < _MIN_SAMPLES_PER_CLASS:
                raise ValueError(
                    f"Not enough samples for class {trick_name}: {n} samples"
                    f" (minimum {_MIN_SAMPLES_PER_CLASS})"
                )

        X = feat_df[feature_cols].values
        le = LabelEncoder()
        y = le.fit_transform(feat_df["trick"])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        estimator = _build_estimator(model_type)
        estimator.fit(X_train, y_train)

        y_pred = estimator.predict(X_test)
        report = classification_report(
            y_test, y_pred,
            labels=list(range(len(le.classes_))),
            target_names=le.classes_.tolist(),
            output_dict=True,
            zero_division=0,
        )
        metrics = {
            "accuracy": float(report["accuracy"]),
            "macro_f1": float(report["macro avg"]["f1-score"]),
            "classification_report": report,
        }

        # Retrain on full dataset before saving
        estimator.fit(X, y)

        safe_name = (name or model_type).replace(" ", "_")
        model_path = os.path.join(MODEL_DIR, f"{run_id}_{safe_name}.pkl")
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": estimator,
                "label_encoder": le,
                "tricks": tricks,
                "feature_cols": feature_cols,
            }, f)

        _db(
            """UPDATE training_runs
               SET status='done', accuracy=?, metrics_json=?, model_path=?,
                   n_samples=?, n_classes=?, finished_at=?
               WHERE id=?""",
            (metrics["accuracy"], json.dumps(metrics), model_path,
             len(feat_df), len(le.classes_), now(), run_id),
        )

    except Exception as exc:
        _db(
            "UPDATE training_runs SET status='failed', error=?, finished_at=? WHERE id=?",
            (str(exc), now(), run_id),
        )


# ── Startup ───────────────────────────────────────────────────────────

init_db()

with get_db() as _conn:
    _startup_run_id = _get_active_run_id(_conn)
    if _startup_run_id:
        _r = _conn.execute(
            "SELECT model_path FROM training_runs WHERE id = ?", (_startup_run_id,)
        ).fetchone()
        if _r and _r["model_path"] and os.path.exists(_r["model_path"]):
            _load_pkl(_r["model_path"])

if not _active and os.path.exists(DEFAULT_MODEL_PATH):
    _load_pkl(DEFAULT_MODEL_PATH)


# ── App ───────────────────────────────────────────────────────────────

app = FastAPI(title="FlipPhone Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Trick endpoints ───────────────────────────────────────────────────

@app.get("/tricks")
def list_tricks():
    tricks = fetch_tricks(FLIPPHONE_URL or None)
    if not tricks:
        raise HTTPException(status_code=503, detail="No trick list available")
    return tricks


# ── Training endpoints ────────────────────────────────────────────────

@app.post("/train", response_model=TrainJobResponse, status_code=202)
def start_training(req: TrainRequest):
    valid_types = {"random_forest", "mlp", "svm"}
    if req.model_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"model_type must be one of {sorted(valid_types)}")

    with get_db() as conn:
        running = conn.execute(
            "SELECT id FROM training_runs WHERE status = 'running' LIMIT 1"
        ).fetchone()
        if running:
            raise HTTPException(status_code=409, detail=f"Job {running['id']} is already running")

        cur = conn.execute(
            "INSERT INTO training_runs (name, model_type, tricks, source_users, status) VALUES (?,?,?,?,?)",
            (req.name, req.model_type, json.dumps(req.tricks),
             json.dumps(req.source_users) if req.source_users else None, "pending"),
        )
        run_id = cur.lastrowid

    threading.Thread(
        target=_run_training,
        args=(run_id, req.model_type, req.tricks, req.source_users, req.name),
        daemon=True,
    ).start()

    return TrainJobResponse(job_id=run_id)


@app.get("/train/{run_id}")
def get_training_status(run_id: int):
    with get_db() as conn:
        row = conn.execute(
            "SELECT status, accuracy, n_classes, n_samples, error FROM training_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Run not found")

    result: dict = {"status": row["status"]}
    if row["status"] == "done":
        result.update({
            "accuracy": row["accuracy"],
            "n_classes": row["n_classes"],
            "n_samples": row["n_samples"],
        })
    elif row["status"] == "failed":
        result["error"] = row["error"]
    return result


# ── Model-management endpoints ────────────────────────────────────────

@app.get("/models", response_model=list[TrainingRunSummary])
def list_models():
    with get_db() as conn:
        active_run_id = _get_active_run_id(conn)
        rows = conn.execute(
            "SELECT * FROM training_runs ORDER BY created_at DESC"
        ).fetchall()
    return [_row_to_summary(r, active_run_id) for r in rows]


@app.get("/models/{run_id}", response_model=TrainingRunDetail)
def get_model(run_id: int):
    with get_db() as conn:
        active_run_id = _get_active_run_id(conn)
        row = conn.execute(
            "SELECT * FROM training_runs WHERE id = ?", (run_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Run not found")
    d = _row_to_summary(row, active_run_id)
    raw = dict(row).get("metrics_json")
    d["metrics_json"] = json.loads(raw) if raw else None
    return d


@app.post("/models/{run_id}/activate", response_model=ActivateResponse)
def activate_model(run_id: int):
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM training_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Run not found")
        if row["status"] != "done":
            raise HTTPException(
                status_code=400,
                detail=f"Run status is '{row['status']}', must be 'done' to activate",
            )
        path = row["model_path"]
        if not path or not os.path.exists(path):
            raise HTTPException(status_code=400, detail="model_path missing or file not found")

        _load_pkl(path)

        conn.execute(
            "INSERT INTO active_model (id, run_id, set_at) VALUES (1, ?, ?)"
            " ON CONFLICT(id) DO UPDATE SET run_id = excluded.run_id, set_at = excluded.set_at",
            (run_id, datetime.now(timezone.utc).isoformat()),
        )
        summary = _row_to_summary(row, run_id)

    return ActivateResponse(ok=True, model=TrainingRunSummary(**summary))


# ── Prediction endpoints ──────────────────────────────────────────────

@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    _require_model()
    if len(req.samples) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 samples")

    df = pd.DataFrame([s.model_dump() for s in req.samples])
    features = extract_features(df)

    X = np.array([[features[col] for col in _active["feature_cols"]]])
    proba = _active["clf"].predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    le = _active["le"]

    return PredictResponse(
        trick=le.classes_[pred_idx],
        confidence=round(float(proba[pred_idx]), 4),
        probabilities={name: round(float(p), 4) for name, p in zip(le.classes_, proba)},
    )


@app.post("/batch_embed", response_model=list[EmbedCoord])
def batch_embed(req: BatchEmbedRequest):
    _require_model()
    valid_ids: list[str] = []
    valid_vecs: list[list[float]] = []

    for rec in req.recordings:
        try:
            if len(rec.samples) < 2:
                raise ValueError("need at least 2 samples")
            df = pd.DataFrame([s.model_dump() for s in rec.samples])
            feats = extract_features(df)
            valid_ids.append(rec.id)
            valid_vecs.append([float(feats[col]) for col in _active["feature_cols"]])
        except Exception:
            pass

    if not valid_ids:
        return []

    coords = embed_3d(np.array(valid_vecs), method="umap", seed=42)
    return [
        EmbedCoord(id=rid, x=float(c[0]), y=float(c[1]), z=float(c[2]))
        for rid, c in zip(valid_ids, coords)
    ]


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
