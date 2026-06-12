"""
Hibrit sınıflandırıcı: audio (Logistic Regression) + tür önceliği.
Streamlit slider'larının tahmine görünür etki etmesini sağlar.
Tam RF modeli (best_classifier_rf.pkl) notebook'tan ayrı kalır.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "model_ready"
MODELS = ROOT / "models"
POPULARITY_THRESHOLD = 50
AUDIO_BLEND_WEIGHT = 0.48


def binarize_popularity(y):
    return (np.asarray(y).ravel() >= POPULARITY_THRESHOLD).astype(int)


def genre_priors_from_train(X_train: pd.DataFrame, y_train: np.ndarray, genre_cols: list[str]):
    train_genre = X_train[genre_cols].idxmax(axis=1).str.replace("genre_", "", regex=False)
    genre_df = pd.DataFrame({"genre": train_genre, "y": y_train})
    genre_stats = genre_df.groupby("genre")["y"].agg(["mean", "count"])
    global_prior = float(y_train.mean())
    priors = {}
    for g in [c.replace("genre_", "") for c in genre_cols]:
        if g in genre_stats.index and genre_stats.loc[g, "count"] >= 30:
            priors[g] = float(genre_stats.loc[g, "mean"])
        else:
            priors[g] = global_prior
    return priors, global_prior


def blend_proba(p_audio: np.ndarray, p_genre: np.ndarray, weight: float) -> np.ndarray:
    return np.clip(weight * p_audio + (1.0 - weight) * p_genre, 0.0, 1.0)


def main():
    feature_names = json.loads((ROOT / "feature_names.json").read_text(encoding="utf-8"))
    audio_cols = [c for c in feature_names if not c.startswith("genre_")]
    genre_cols = [c for c in feature_names if c.startswith("genre_")]

    X_train = pd.read_csv(DATA / "X_train_clean.csv")
    X_test = pd.read_csv(DATA / "X_test_clean.csv")
    y_train = binarize_popularity(pd.read_csv(DATA / "y_train.csv").values)
    y_test = binarize_popularity(pd.read_csv(DATA / "y_test.csv").values)

    scaler = joblib.load(MODELS / "preprocessing_pipeline_scaler.pkl")
    X_train_s = pd.DataFrame(scaler.transform(X_train), columns=feature_names)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=feature_names)

    audio_model = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)
    audio_model.fit(X_train_s[audio_cols], y_train)

    genre_priors, global_prior = genre_priors_from_train(X_train, y_train, genre_cols)

    def genre_proba(X_raw: pd.DataFrame) -> np.ndarray:
        active = X_raw[genre_cols].idxmax(axis=1).str.replace("genre_", "", regex=False)
        return np.array([genre_priors.get(g, global_prior) for g in active])

    audio_proba_test = audio_model.predict_proba(X_test_s[audio_cols])[:, 1]
    genre_proba_test = genre_proba(X_test)
    hybrid_proba_test = blend_proba(audio_proba_test, genre_proba_test, AUDIO_BLEND_WEIGHT)
    hybrid_pred_test = (hybrid_proba_test >= 0.5).astype(int)

    # RF-only karşılaştırma (eski davranış)
    rf_metrics = {}
    rf_path = MODELS / "best_classifier_rf.pkl"
    if rf_path.exists():
        rf = joblib.load(rf_path)
        rf_proba = rf.predict_proba(X_test_s)[:, 1]
        rf_pred = (rf_proba >= 0.5).astype(int)
        rf_metrics = {
            "accuracy": float(accuracy_score(y_test, rf_pred)),
            "f1_weighted": float(f1_score(y_test, rf_pred, average="weighted")),
            "roc_auc": float(roc_auc_score(y_test, rf_proba)),
        }

    metrics = {
        "audio_only": {
            "accuracy": float(accuracy_score(y_test, (audio_proba_test >= 0.5).astype(int))),
            "f1_weighted": float(
                f1_score(y_test, (audio_proba_test >= 0.5).astype(int), average="weighted")
            ),
            "roc_auc": float(roc_auc_score(y_test, audio_proba_test)),
        },
        "hybrid_interactive": {
            "accuracy": float(accuracy_score(y_test, hybrid_pred_test)),
            "f1_weighted": float(f1_score(y_test, hybrid_pred_test, average="weighted")),
            "roc_auc": float(roc_auc_score(y_test, hybrid_proba_test)),
        },
        "random_forest_full": rf_metrics,
        "audio_blend_weight": AUDIO_BLEND_WEIGHT,
        "global_prior": global_prior,
        "model_type": "logistic_regression_audio + genre_prior",
    }

    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(audio_model, MODELS / "audio_classifier_lr.pkl")
    (MODELS / "genre_priors.json").write_text(
        json.dumps(
            {
                "threshold": POPULARITY_THRESHOLD,
                "global_prior": global_prior,
                "priors": genre_priors,
                "audio_blend_weight": AUDIO_BLEND_WEIGHT,
                "audio_feature_names": audio_cols,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (MODELS / "hybrid_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print("Hybrid (interactive) test metrics:", metrics["hybrid_interactive"])
    if rf_metrics:
        print("RF full model metrics:", rf_metrics)
    print(f"Saved: {MODELS / 'audio_classifier_lr.pkl'}")
    print(f"Saved: {MODELS / 'genre_priors.json'}")


if __name__ == "__main__":
    main()
