"""
Train MLP neural network classifiers on FlipPhone trick data and compare them.

Usage:
    python fetch_data.py --url ... --key ...   # first, fetch the data
    python train_nn.py                          # then, train
"""

import os
import pickle

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

import os

from train import MODEL_DIR, fetch_tricks, load_features

# ── Model definitions ────────────────────────────────────────────────
# Each entry: (name, MLPClassifier kwargs)
MODELS = [
    ("MLP-small",  dict(hidden_layer_sizes=(64,),        max_iter=500)),
    ("MLP-medium", dict(hidden_layer_sizes=(128, 64),    max_iter=500)),
    ("MLP-large",  dict(hidden_layer_sizes=(256, 128, 64), max_iter=500)),
]


# ── Helpers ───────────────────────────────────────────────────────────


def print_confusion(cm: np.ndarray, class_names: list[str]) -> None:
    max_name = max(len(n) for n in class_names)
    header = " " * (max_name + 2) + "  ".join(f"{n[:6]:>6}" for n in class_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>6}" for v in row)
        print(f"{class_names[i]:>{max_name}}  {row_str}")


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    url = os.environ.get("FLIPPHONE_URL", "")
    tricks = fetch_tricks(url or None)
    selected_tricks = [t["id"] for t in tricks] if tricks else None

    print("Loading data …")
    try:
        feat_df, feature_cols = load_features(selected_tricks=selected_tricks)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return

    print(f"  {feat_df['trick'].value_counts().sum()} recordings, {feat_df['trick'].nunique()} tricks")

    X = feat_df[feature_cols].values
    le = LabelEncoder()
    y = le.fit_transform(feat_df["trick"])

    print(f"  {X.shape[0]} recordings, {X.shape[1]} features, {len(le.classes_)} classes")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}\n")

    # MLPs need scaled input
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = []

    for name, kwargs in MODELS:
        print(f"Training {name} …")
        clf = MLPClassifier(random_state=42, **kwargs)
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)

        print(f"\n── {name} — Classification Report ──")
        print(classification_report(y_test, y_pred, labels=range(len(le.classes_)), target_names=le.classes_))

        print(f"── {name} — Confusion Matrix ──")
        cm = confusion_matrix(y_test, y_pred)
        print_confusion(cm, list(le.classes_))
        print()

        acc = (y_pred == y_test).mean()
        results.append((name, acc, clf))

    # Summary
    print("── Summary ──")
    print(f"  {'Model':<14} {'Accuracy':>10}")
    for name, acc, _ in sorted(results, key=lambda x: -x[1]):
        print(f"  {name:<14} {acc:>10.1%}")

    # Save best model
    best_name, best_acc, best_clf = max(results, key=lambda x: x[1])
    print(f"\nBest model: {best_name} ({best_acc:.1%}) — retraining on all data …")
    best_clf.fit(scaler.fit_transform(X), y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "nn_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            "clf": best_clf,
            "scaler": scaler,
            "label_encoder": le,
            "feature_cols": feature_cols,
            "model_name": best_name,
        }, f)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
