"""
PAIS — Data preprocessing pipeline.

Responsibilities
----------------
1. Load raw CSV.
2. Handle missing values (Parent_Education_Level has ~20% nulls).
3. Build the binary target `at_risk` using both grade and score rules.
4. Engineer a handful of derived features that capture behavioural signal.
5. Persist a clean, model-ready CSV to data/processed/.

The module is idempotent: run it twice, get the same output.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

from . import config as C


# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------
def build_target(df: pd.DataFrame) -> pd.Series:
    """Return a 0/1 Series: 1 = at-risk, 0 = safe."""
    grade_flag = df["Grade"].isin(C.AT_RISK_GRADES)
    score_flag = df["Total_Score"] < C.AT_RISK_TOTAL_SCORE
    return (grade_flag | score_flag).astype(int)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Parent_Education_Level — impute missing with explicit "Unknown" category,
    # which preserves the "missingness = signal" idea without dropping rows.
    df["Parent_Education_Level"] = df["Parent_Education_Level"].fillna("Unknown")

    # Everything else is already non-null per our earlier audit, but be safe.
    for col in C.NUMERIC_FEATURES:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    for col in C.CATEGORICAL_FEATURES:
        if df[col].isna().any():
            df[col] = df[col].fillna("Unknown")

    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Academic composite (before final) — a simple weighted average of the
    # early signals. Gives the model a pre-aggregated proxy it can lean on.
    df["early_academic_avg"] = (
        0.40 * df["Midterm_Score"]
        + 0.25 * df["Assignments_Avg"]
        + 0.20 * df["Quizzes_Avg"]
        + 0.15 * df["Projects_Score"]
    )

    # Engagement score: study hours normalised, penalised by high stress and
    # poor sleep. Bounded to [0, 10] roughly.
    df["engagement_index"] = (
        df["Study_Hours_per_Week"].clip(0, 40) / 4          # ~0-10
        - (df["Stress_Level (1-10)"] - 5) * 0.3             # stress > 5 drags
        + (df["Sleep_Hours_per_Night"] - 6) * 0.4           # sleep helps
    ).clip(0, 15)

    # Attendance risk: distance below the 75% threshold commonly used in India.
    df["attendance_deficit"] = (75.0 - df["Attendance (%)"]).clip(lower=0)

    # Interaction: low attendance AND low midterm is the canonical failure mode.
    df["low_att_low_mid"] = (
        (df["Attendance (%)"] < 70) & (df["Midterm_Score"] < 50)
    ).astype(int)

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Load the raw dataset as a DataFrame."""
    path = Path(path) if path else C.DATA_RAW
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame | None = None, *, persist: bool = True) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline.

    Parameters
    ----------
    df : optional pre-loaded DataFrame. If None, reads from DATA_RAW.
    persist : if True, writes the processed frame to DATA_PROCESSED.

    Returns
    -------
    The processed DataFrame, including the `at_risk` column.
    """
    C.ensure_dirs()
    if df is None:
        df = load_raw()

    df = _clean(df)
    df[C.TARGET_COLUMN] = build_target(df)
    df = _engineer(df)

    if persist:
        df.to_csv(C.DATA_PROCESSED, index=False)

    return df


# Features actually fed into the model (numeric + engineered). Categoricals
# are handled separately inside the sklearn pipeline.
def model_feature_lists() -> tuple[list[str], list[str]]:
    engineered = ["early_academic_avg", "engagement_index",
                  "attendance_deficit", "low_att_low_mid"]
    numeric = C.NUMERIC_FEATURES + engineered
    return numeric, list(C.CATEGORICAL_FEATURES)


if __name__ == "__main__":
    out = preprocess()
    print(f"Processed {len(out)} rows → {C.DATA_PROCESSED}")
    print(f"At-risk rate: {out[C.TARGET_COLUMN].mean():.1%}")
