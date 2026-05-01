from __future__ import annotations
import pandas as pd


def compute_gradient(df: pd.DataFrame) -> pd.Series:
    required = {"Midterm_Score", "early_academic_avg"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"compute_gradient missing columns: {missing}")
    return df["Midterm_Score"] - df["early_academic_avg"]


def rank_by_gradient(df: pd.DataFrame, *, ascending: bool = False,
                     top_n: int | None = None) -> pd.DataFrame:
    out = df[["Student_ID", "Midterm_Score", "early_academic_avg"]].copy()
    out["gradient"] = compute_gradient(df)
    out = out.sort_values("gradient", ascending=ascending,
                          kind="mergesort")
    if top_n:
        out = out.head(top_n)
    return out.reset_index(drop=True)
