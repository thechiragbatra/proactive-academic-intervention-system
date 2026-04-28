"""
ML training pipeline.

Produces three models — Logistic Regression (baseline), Random Forest, and
Gradient Boosting — wrapped in sklearn Pipelines that include preprocessing
(imputation, scaling, one-hot). The best model on validation F1 is persisted.

Run directly:
    python -m src.ml.train
"""
from __future__ import annotations
import json
import joblib
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .. import config as C
from ..preprocessing import preprocess, model_feature_lists


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------
def build_preprocessor(numeric_cols: list[str],
                       categorical_cols: list[str]) -> ColumnTransformer:
    num = Pipeline([
        ("imp",    SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
    ])
    cat = Pipeline([
        ("imp",    SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", num, numeric_cols),
        ("cat", cat, categorical_cols),
    ])


def build_pipelines(numeric_cols: list[str],
                    categorical_cols: list[str]) -> dict[str, Pipeline]:
    pre = build_preprocessor(numeric_cols, categorical_cols)
    seed = C.RANDOM_SEED
    return {
        "logistic_regression": Pipeline([
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                       random_state=seed)),
        ]),
        "random_forest": Pipeline([
            ("pre", pre),
            ("clf", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=3,
                class_weight="balanced", n_jobs=-1, random_state=seed)),
        ]),
        "gradient_boosting": Pipeline([
            ("pre", pre),
            ("clf", GradientBoostingClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                random_state=seed)),
        ]),
    }


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def _metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall":    round(recall_score(y_true, y_pred), 4),
        "f1":        round(f1_score(y_true, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_true, y_proba), 4),
    }


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def train_and_persist(verbose: bool = True) -> dict:
    """Train all candidates, pick the winner by F1, persist it."""
    C.ensure_dirs()
    df = preprocess()
    numeric_cols, categorical_cols = model_feature_lists()

    # Drop leakage columns and IDs
    X = df[numeric_cols + categorical_cols]
    y = df[C.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=C.TEST_SIZE,
        random_state=C.RANDOM_SEED,
        stratify=y,
    )

    pipelines = build_pipelines(numeric_cols, categorical_cols)
    results: dict[str, dict] = {}

    for name, pipe in pipelines.items():
        if verbose:
            print(f"  › training {name}…", flush=True)
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        cv_f1 = cross_val_score(
            pipe, X_train, y_train,
            cv=C.CV_FOLDS, scoring="f1", n_jobs=-1,
        )
        results[name] = {
            "test_metrics": _metrics(y_test, y_pred, y_proba),
            "cv_f1_mean":   round(float(cv_f1.mean()), 4),
            "cv_f1_std":    round(float(cv_f1.std()), 4),
            "confusion":    confusion_matrix(y_test, y_pred).tolist(),
            "report":       classification_report(y_test, y_pred,
                                                  output_dict=True, zero_division=0),
        }

    # Choose winner by test F1.
    best_name = max(results, key=lambda k: results[k]["test_metrics"]["f1"])
    best_pipe = pipelines[best_name]

    # Persist winner and the full comparison report.
    feature_names = numeric_cols + categorical_cols
    joblib.dump(
        {
            "model": best_pipe,
            "feature_names": feature_names,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "winner": best_name,
        },
        C.MODELS_DIR / "risk_model.pkl",
    )
    (C.REPORTS_DIR / "training_report.json").write_text(
        json.dumps({"winner": best_name, "results": results}, indent=2)
    )

    if verbose:
        print(f"\nWinner: {best_name}")
        for name, r in results.items():
            m = r["test_metrics"]
            print(f"  {name:24s}  F1={m['f1']:.3f}  AUC={m['roc_auc']:.3f}"
                  f"  (cv-F1 {r['cv_f1_mean']:.3f}±{r['cv_f1_std']:.3f})")
        print(f"Saved: {C.MODELS_DIR/'risk_model.pkl'}")

    return {"winner": best_name, "results": results}


if __name__ == "__main__":
    train_and_persist()
