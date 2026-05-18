"""
Train trick classifiers on FlipPhone IMU data.

Usage:
    python3 train.py                      # Random Forest (default)
    python3 train.py --model rf           # Random Forest only
    python3 train.py --model nn           # Neural Network only
    python3 train.py --model compare      # both + side-by-side comparison
    python3 train.py --model compare --data data/dataset_augmented.csv
"""

import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "dataset.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


# ── 3-D embedding ────────────────────────────────────────────────────


def embed_3d(X: np.ndarray, method: str = "umap", seed: int = 42) -> np.ndarray:
    """Reduce feature matrix X to 3 dimensions via UMAP or PCA."""
    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler().fit_transform(X)

    if method == "umap":
        try:
            import umap

            return umap.UMAP(n_components=3, random_state=seed, n_neighbors=15, min_dist=0.1).fit_transform(Xs)
        except ImportError:
            print("umap-learn not found, falling back to PCA.")

    from sklearn.decomposition import PCA

    return PCA(n_components=3, random_state=seed).fit_transform(Xs)


# ── Trick helpers ────────────────────────────────────────────────────


def _norm(s: str) -> str:
    """Canonical slug: lowercase, spaces/hyphens → underscores."""
    return s.strip().lower().replace(" ", "_").replace("-", "_")


def fetch_tricks(flipphone_url: str | None = None) -> list[dict]:
    """Return tricks from the backend API or fall back to the local CSV.

    Each entry: {"id": "kickflip", "name": "Kickflip"}
    """
    if flipphone_url:
        import requests
        resp = requests.get(f"{flipphone_url}/game/api/tricks", timeout=10)
        resp.raise_for_status()
        return resp.json()
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH, usecols=["trick"])
        return [{"id": _norm(t), "name": t} for t in sorted(df["trick"].unique())]
    return []


# ── Feature extraction ──────────────────────────────────────────────


def extract_features(group: pd.DataFrame) -> dict:
    """Extract features from a single recording's sample rows."""
    t = group["t"].values.astype(float)
    t0 = t[0]
    duration_ms = float(t[-1] - t0) if len(t) > 1 else 1.0
    t_norm = (t - t0) / duration_ms  # normalised time axis [0, 1]

    features = {}

    # Per-axis statistics
    for col in ["ax", "ay", "az", "gx", "gy", "gz"]:
        v = group[col].values
        features[f"{col}_mean"] = float(v.mean())
        features[f"{col}_std"] = float(v.std())
        features[f"{col}_min"] = float(v.min())
        features[f"{col}_max"] = float(v.max())
        features[f"{col}_range"] = float(v.max() - v.min())

    # Acceleration magnitude
    acc_mag = np.sqrt(group["ax"].values ** 2 + group["ay"].values ** 2 + group["az"].values ** 2)
    features["acc_mag_mean"] = float(acc_mag.mean())
    features["acc_mag_std"]  = float(acc_mag.std())
    features["acc_mag_max"]  = float(acc_mag.max())

    # Rotation magnitude
    gyro_mag = np.sqrt(group["gx"].values ** 2 + group["gy"].values ** 2 + group["gz"].values ** 2)
    features["gyro_mag_mean"] = float(gyro_mag.mean())
    features["gyro_mag_std"]  = float(gyro_mag.std())
    features["gyro_mag_max"]  = float(gyro_mag.max())

    # Duration-invariant net rotation per axis:
    #   trapezoid on normalised time [0,1] → not scaled by window length.
    #   Sign encodes rotation direction (e.g. heelflip vs kickflip).
    # Zero-crossings count direction reversals (discriminates flip from spin).
    for axis in ["gx", "gy", "gz"]:
        v = group[axis].values
        features[f"{axis}_net_angle"]      = float(np.trapezoid(v, t_norm))
        features[f"{axis}_zero_crossings"] = int(np.sum(np.diff(np.sign(v)) != 0))

    # When (as a fraction of the window) does peak acceleration occur?
    peak_idx = int(np.argmax(acc_mag))
    features["peak_time_ratio"] = float(t_norm[peak_idx])

    # Front-load vs back-load: energy ratio between first and second halves.
    mid = len(acc_mag) // 2
    e1 = float(np.mean(acc_mag[:mid] ** 2)) if mid > 0 else 0.0
    e2 = float(np.mean(acc_mag[mid:] ** 2)) if mid > 0 else 0.0
    features["half_energy_ratio"] = e1 / (e2 + 1e-6)

    return features


# ── Shared data loading ───────────────────────────────────────────────


def load_features(
    data_path: str = DATA_PATH,
    selected_tricks: list | None = None,
    selected_collectors: list | None = None,
) -> tuple:
    """Load CSV, filter, extract features. Returns (feat_df, feature_cols)."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No dataset found at {data_path}. Run fetch_data.py first.")

    df = pd.read_csv(data_path)

    if selected_tricks:
        normalized = {_norm(t) for t in selected_tricks}
        df = df[df["trick"].apply(_norm).isin(normalized)]
        if df.empty:
            raise ValueError(f"No data found for selected tricks: {selected_tricks}")

    if selected_collectors:
        df = df[df["collector"].isin(selected_collectors)]
        if df.empty:
            raise ValueError(f"No data found for selected collectors: {selected_collectors}")

    records = []
    for rec_id, group in df.groupby("id"):
        feats = extract_features(group)
        feats["id"] = rec_id
        feats["trick"] = group["trick"].iloc[0]
        feats["collector"] = group["collector"].iloc[0]
        feats["timestamp"] = group["timestamp"].iloc[0]
        records.append(feats)

    feat_df = pd.DataFrame(records)
    feature_cols = [c for c in feat_df.columns if c not in ("id", "trick", "collector", "timestamp")]
    return feat_df, feature_cols


# ── Model trainers ────────────────────────────────────────────────────


def train_rf(X_train, y_train, seed: int) -> tuple[RandomForestClassifier, float]:
    t0 = time.perf_counter()
    clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    clf.fit(X_train, y_train)
    return clf, time.perf_counter() - t0


def train_nn(
    X_train, y_train, seed: int
) -> tuple[MLPClassifier, StandardScaler, float]:
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_train)
    t0 = time.perf_counter()
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=500,
        random_state=seed,
    )
    clf.fit(X_s, y_train)
    return clf, scaler, time.perf_counter() - t0


# ── Report helpers ────────────────────────────────────────────────────


def print_confusion(cm: np.ndarray, class_names: list[str]) -> None:
    max_name = max(len(n) for n in class_names)
    header = " " * (max_name + 2) + "  ".join(f"{n[:6]:>6}" for n in class_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>6}" for v in row)
        print(f"{class_names[i]:>{max_name}}  {row_str}")


def print_misclassified(
    y_test, y_pred, test_ids, feat_df: pd.DataFrame, class_names: list[str]
) -> None:
    rec_meta = feat_df.set_index("id")[["collector", "timestamp"]]
    misclassified = y_test != y_pred
    if misclassified.any():
        print("\n── Misclassified Recordings ──")
        for idx in np.where(misclassified)[0]:
            true_label = class_names[y_test[idx]]
            pred_label = class_names[y_pred[idx]]
            rec_id = test_ids[idx]
            meta = rec_meta.loc[rec_id]
            ts = pd.to_datetime(meta["timestamp"]).strftime("%Y-%m-%d %H:%M")
            print(f"  {rec_id}: true={true_label}, predicted={pred_label} (by {meta['collector']}, {ts})")


def report_rf(clf, X_test, y_test, test_ids, feat_df, le, feature_cols) -> dict:
    y_pred = clf.predict(X_test)
    class_names = list(le.classes_)

    print("\n── RF — Classification Report ──")
    print(classification_report(y_test, y_pred, labels=range(len(class_names)), target_names=class_names))
    print("── RF — Confusion Matrix ──")
    print_confusion(confusion_matrix(y_test, y_pred), class_names)
    print_misclassified(y_test, y_pred, test_ids, feat_df, class_names)

    importances = sorted(zip(feature_cols, clf.feature_importances_), key=lambda x: -x[1])
    print("\n── RF — Top 10 Features ──")
    for name, imp in importances[:10]:
        print(f"  {name:<25} {imp:.4f}")

    return {
        "accuracy": float((y_pred == y_test).mean()),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "per_class_f1": {
            cls: float(v)
            for cls, v in zip(
                class_names,
                np.asarray(f1_score(y_test, y_pred, average=None, zero_division=0)),
            )
        },
    }


def report_nn(clf, scaler, X_test, y_test, test_ids, feat_df, le) -> dict:
    y_pred = clf.predict(scaler.transform(X_test))
    class_names = list(le.classes_)

    print("\n── NN — Classification Report ──")
    print(classification_report(y_test, y_pred, labels=range(len(class_names)), target_names=class_names))
    print("── NN — Confusion Matrix ──")
    print_confusion(confusion_matrix(y_test, y_pred), class_names)
    print_misclassified(y_test, y_pred, test_ids, feat_df, class_names)

    return {
        "accuracy": float((y_pred == y_test).mean()),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "per_class_f1": {
            cls: float(v)
            for cls, v in zip(
                class_names,
                np.asarray(f1_score(y_test, y_pred, average=None, zero_division=0)),
            )
        },
    }


def print_comparison(rf_metrics: dict, nn_metrics: dict, rf_time: float, nn_time: float) -> None:
    print("\n" + "=" * 52)
    print("=== Model Comparison ===")
    print("=" * 52)
    print(f"  {'Metric':<20} {'Random Forest':>14} {'Neural Network':>14}")
    print("  " + "-" * 48)
    print(f"  {'Accuracy':<20} {rf_metrics['accuracy']:>14.4f} {nn_metrics['accuracy']:>14.4f}")
    print(f"  {'Macro F1':<20} {rf_metrics['macro_f1']:>14.4f} {nn_metrics['macro_f1']:>14.4f}")
    print(f"  {'Training time':<20} {rf_time:>13.1f}s {nn_time:>13.1f}s")
    print()
    print(f"  {'Per-class F1:'}")

    all_classes = sorted(set(rf_metrics["per_class_f1"]) | set(nn_metrics["per_class_f1"]))
    for cls in all_classes:
        rf_f1 = rf_metrics["per_class_f1"].get(cls, float("nan"))
        nn_f1 = nn_metrics["per_class_f1"].get(cls, float("nan"))
        print(f"    {cls:<22} {rf_f1:>10.4f} {nn_f1:>14.4f}")
    print("=" * 52)


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FlipPhone trick classifier")
    parser.add_argument("--model", choices=["rf", "nn", "compare"], default="rf",
                        help="Which model(s) to train (default: rf)")
    parser.add_argument("--data", default=None,
                        help="Override CSV path (default: data/dataset.csv or dataset_augmented.csv)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--tricks", nargs="*", default=None,
                        help="Tricks to train on (slugs). Default: all from backend or CSV.")
    parser.add_argument("--url", default=os.environ.get("FLIPPHONE_URL", ""),
                        help="FlipPhone server URL (overrides FLIPPHONE_URL env var)")
    args = parser.parse_args()

    data_path = args.data or DATA_PATH

    # Resolve trick list: CLI flag → backend API → all in CSV
    if args.tricks:
        selected_tricks = args.tricks
    else:
        tricks = fetch_tricks(args.url or None)
        selected_tricks = [t["id"] for t in tricks] if tricks else None

    print("Loading data …")
    try:
        feat_df, feature_cols = load_features(
            data_path=data_path,
            selected_tricks=selected_tricks,
        )
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return

    print(f"  {feat_df['trick'].value_counts().sum()} recordings, {feat_df['trick'].nunique()} tricks")
    if selected_tricks:
        print(f"  Tricks: {selected_tricks}")
    print("\nRecordings per trick:")
    for trick, count in feat_df["trick"].value_counts().items():
        print(f"  {trick}: {count}")

    X: np.ndarray = feat_df[feature_cols].values
    le = LabelEncoder()
    y: np.ndarray = le.fit_transform(feat_df["trick"])
    print(f"\n  {X.shape[0]} recordings, {X.shape[1]} features, {len(le.classes_)} classes")

    idx_all = np.arange(len(y))
    idx_train, idx_test = train_test_split(
        idx_all, test_size=0.2, random_state=args.seed, stratify=y,
    )
    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]
    test_ids: np.ndarray = np.asarray(feat_df["id"].values)[idx_test]
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    rf_metrics: dict = {}
    nn_metrics: dict = {}
    rf_time: float = 0.0
    nn_time: float = 0.0

    # ── Random Forest ──
    if args.model in ("rf", "compare"):
        print("\nTraining Random Forest …")
        rf_clf, rf_time = train_rf(X_train, y_train, args.seed)
        rf_metrics = report_rf(rf_clf, X_test, y_test, test_ids, feat_df, le, feature_cols)

        # Retrain on all data, save
        rf_clf.fit(X, y)
        model_path = os.path.join(MODEL_DIR, "rf_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump({"clf": rf_clf, "label_encoder": le, "feature_cols": feature_cols}, f)
        print(f"\n  RF model saved → {model_path}")

    # ── Neural Network ──
    if args.model in ("nn", "compare"):
        print("\nTraining Neural Network …")
        nn_clf, nn_scaler, nn_time = train_nn(X_train, y_train, args.seed)
        nn_metrics = report_nn(nn_clf, nn_scaler, X_test, y_test, test_ids, feat_df, le)

        # Retrain on all data, save
        nn_scaler_full = StandardScaler()
        nn_clf.fit(nn_scaler_full.fit_transform(X), y)
        model_path = os.path.join(MODEL_DIR, "nn_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump({
                "clf": nn_clf, "scaler": nn_scaler_full,
                "label_encoder": le, "feature_cols": feature_cols,
            }, f)
        print(f"\n  NN model saved → {model_path}")

    # ── Comparison table ──
    if args.model == "compare":
        print_comparison(rf_metrics, nn_metrics, rf_time, nn_time)


if __name__ == "__main__":
    main()
