"""
ML evaluation utilities — loads the persisted model and produces figures.

Generates:
    reports/figures/confusion_matrix.png
    reports/figures/feature_importance.png
    reports/figures/roc_curve.png

Run directly:
    python -m src.ml.evaluate
"""
from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless-safe; must come before pyplot import
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    classification_report,
)
from sklearn.model_selection import train_test_split

from .. import config as C
from ..preprocessing import preprocess, model_feature_lists


def _load_bundle():
    return joblib.load(C.MODELS_DIR / "risk_model.pkl")


def _rebuild_test_split():
    df = preprocess()
    numeric_cols, categorical_cols = model_feature_lists()
    X = df[numeric_cols + categorical_cols]
    y = df[C.TARGET_COLUMN]
    return train_test_split(
        X, y,
        test_size=C.TEST_SIZE,
        random_state=C.RANDOM_SEED,
        stratify=y,
    )


def plot_confusion(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Safe", "At-risk"])
    ax.set_yticklabels(["Safe", "At-risk"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max()/2 else "black", fontsize=14)
    fig.colorbar(im)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_roc(y_true, y_proba, path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC Curve"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_feature_importance(pipe, feature_names, path):
    """Extract either tree importances or linear coefficients."""
    clf = pipe.named_steps["clf"]
    pre = pipe.named_steps["pre"]
    try:
        expanded = pre.get_feature_names_out()
    except Exception:
        expanded = np.asarray(feature_names)

    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        imp = np.abs(clf.coef_[0])
    else:
        return   # unknown model

    idx = np.argsort(imp)[::-1][:15]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh(range(len(idx))[::-1], imp[idx][::-1])
    ax.set_yticks(range(len(idx))[::-1])
    ax.set_yticklabels([str(expanded[i]) for i in idx][::-1], fontsize=9)
    ax.set_xlabel("Importance")
    ax.set_title("Top 15 Features")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def evaluate_model() -> dict:
    C.ensure_dirs()
    bundle = _load_bundle()
    pipe = bundle["model"]
    feature_names = bundle["feature_names"]

    X_train, X_test, y_train, y_test = _rebuild_test_split()
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    fig_dir = C.FIGURES_DIR
    plot_confusion(y_test, y_pred, fig_dir / "confusion_matrix.png")
    plot_roc(y_test, y_proba, fig_dir / "roc_curve.png")
    plot_feature_importance(pipe, feature_names,
                            fig_dir / "feature_importance.png")

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    print("Classification report (test set):")
    print(classification_report(y_test, y_pred, zero_division=0))
    return report


if __name__ == "__main__":
    evaluate_model()
