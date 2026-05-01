from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

from . import config as C


def build_target(df: pd.DataFrame) -> pd.Series:
    grade_flag = df["Grade"].isin(C.AT_RISK_GRADES)
    score_flag = df["Total_Score"] < C.AT_RISK_TOTAL_SCORE
    return (grade_flag | score_flag).astype(int)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()


    df["Parent_Education_Level"] = df["Parent_Education_Level"].fillna("Unknown")


    for col in C.NUMERIC_FEATURES:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    for col in C.CATEGORICAL_FEATURES:
        if df[col].isna().any():
            df[col] = df[col].fillna("Unknown")

    return df


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()


    df["early_academic_avg"] = (
        0.40 * df["Midterm_Score"]
        + 0.25 * df["Assignments_Avg"]
        + 0.20 * df["Quizzes_Avg"]
        + 0.15 * df["Projects_Score"]
    )


    df["engagement_index"] = (
        df["Study_Hours_per_Week"].clip(0, 40) / 4
        - (df["Stress_Level (1-10)"] - 5) * 0.3
        + (df["Sleep_Hours_per_Night"] - 6) * 0.4
    ).clip(0, 15)


    df["attendance_deficit"] = (75.0 - df["Attendance (%)"]).clip(lower=0)


    df["low_att_low_mid"] = (
        (df["Attendance (%)"] < 70) & (df["Midterm_Score"] < 50)
    ).astype(int)

    return df


def load_raw(path: Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else C.DATA_RAW
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame | None = None, *, persist: bool = True) -> pd.DataFrame:
    C.ensure_dirs()
    if df is None:
        df = load_raw()

    df = _clean(df)
    df[C.TARGET_COLUMN] = build_target(df)
    df = _engineer(df)

    if persist:
        df.to_csv(C.DATA_PROCESSED, index=False)

    return df


def model_feature_lists() -> tuple[list[str], list[str]]:
    engineered = ["early_academic_avg", "engagement_index",
                  "attendance_deficit", "low_att_low_mid"]
    numeric = C.NUMERIC_FEATURES + engineered
    return numeric, list(C.CATEGORICAL_FEATURES)


if __name__ == "__main__":
    out = preprocess()
    print(f"Processed {len(out)} rows → {C.DATA_PROCESSED}")
    print(f"At-risk rate: {out[C.TARGET_COLUMN].mean():.1%}")
