"""
DSA #6 — Sorting by improvement / decline gradient.

Problem
-------
Faculty want to see "who is improving the fastest" and "who is falling the
fastest" after each evaluation cycle. This is a classic ranking problem;
we compute a per-student gradient and sort in O(n log n).

Gradient definition
-------------------
    gradient = midterm_score - early_academic_avg

`midterm_score` is the latest major assessment; `early_academic_avg` is the
blended pre-midterm signal (from preprocessing.py). Positive = improving.
Negative = declining.

Python's `sorted` uses Timsort (O(n log n), stable) which satisfies the
synopsis requirement of "stable O(n log n) sort".
"""
from __future__ import annotations
import pandas as pd


def compute_gradient(df: pd.DataFrame) -> pd.Series:
    """Return a Series of gradients aligned with df.index."""
    required = {"Midterm_Score", "early_academic_avg"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"compute_gradient missing columns: {missing}")
    return df["Midterm_Score"] - df["early_academic_avg"]


def rank_by_gradient(df: pd.DataFrame, *, ascending: bool = False,
                     top_n: int | None = None) -> pd.DataFrame:
    """
    Rank students by their improvement gradient.

    Parameters
    ----------
    ascending : False → fastest improvers first; True → fastest decliners first.
    top_n : optional cap.

    Returns a DataFrame keyed by Student_ID with columns
    [Student_ID, Midterm_Score, early_academic_avg, gradient].
    """
    out = df[["Student_ID", "Midterm_Score", "early_academic_avg"]].copy()
    out["gradient"] = compute_gradient(df)
    out = out.sort_values("gradient", ascending=ascending,
                          kind="mergesort")    # explicit stable sort
    if top_n:
        out = out.head(top_n)
    return out.reset_index(drop=True)
