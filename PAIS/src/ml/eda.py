from __future__ import annotations
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .. import config as C
from ..preprocessing import preprocess, load_raw, build_target

EDA_DIR = C.FIGURES_DIR / "eda"


def _setup():
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    })


def class_balance(df: pd.DataFrame) -> dict:
    counts = df[C.TARGET_COLUMN].value_counts().sort_index()
    rate = counts.iloc[1] / counts.sum()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Safe", "At-risk"], counts.values,
                  color=["#22c55e", "#ef4444"], edgecolor="black", linewidth=0.4)
    ax.set_title(f"Target distribution  ·  at-risk rate {rate:.1%}")
    ax.set_ylabel("Students")
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 30,
                f"{v:,}", ha="center", fontsize=10)
    fig.savefig(EDA_DIR / "class_balance.png")
    plt.close(fig)
    return {"safe": int(counts.iloc[0]), "at_risk": int(counts.iloc[1]),
            "at_risk_rate": round(float(rate), 4)}


def missingness(df: pd.DataFrame) -> dict:
    miss = (df.isna().mean() * 100).sort_values(ascending=False)
    miss = miss[miss > 0]
    if miss.empty:
        return {"columns_with_nulls": 0}
    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(miss))))
    ax.barh(miss.index, miss.values, color="#3b82f6", edgecolor="black", linewidth=0.4)
    ax.set_xlabel("% missing")
    ax.set_title("Missingness by column")
    ax.invert_yaxis()
    fig.savefig(EDA_DIR / "missingness.png")
    plt.close(fig)
    return {"columns_with_nulls": int(len(miss)),
            "max_missing_pct": round(float(miss.max()), 2)}


def numeric_distributions(df: pd.DataFrame) -> dict:
    cols = [c for c in C.NUMERIC_FEATURES if c in df.columns]
    n = len(cols)
    cols_per_row = 4
    rows = (n + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(4 * cols_per_row, 3.2 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if cols_per_row == 1 else axes
    for i, col in enumerate(cols):
        ax = axes[i]
        safe = df[df[C.TARGET_COLUMN] == 0][col].dropna()
        risk = df[df[C.TARGET_COLUMN] == 1][col].dropna()
        ax.hist(safe, bins=25, alpha=0.55, label="Safe", color="#22c55e")
        ax.hist(risk, bins=25, alpha=0.55, label="At-risk", color="#ef4444")
        ax.set_title(col, fontsize=10)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.legend(fontsize=8)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Numeric feature distributions  ·  by target class", y=1.005, fontsize=13)
    fig.savefig(EDA_DIR / "distributions.png")
    plt.close(fig)
    return {"plotted_columns": n}


def correlation_heatmap(df: pd.DataFrame) -> dict:
    cols = [c for c in C.NUMERIC_FEATURES if c in df.columns] + [C.TARGET_COLUMN]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True,
                linewidths=0.4, cbar_kws={"shrink": 0.7}, ax=ax)
    ax.set_title("Pearson correlation  ·  numeric features and target")
    fig.savefig(EDA_DIR / "correlation.png")
    plt.close(fig)
    target_corr = corr[C.TARGET_COLUMN].drop(C.TARGET_COLUMN).sort_values(key=abs, ascending=False)
    return {"top_correlated_with_target":
            {k: round(float(v), 3) for k, v in target_corr.head(8).items()}}


def categorical_breakdown(df: pd.DataFrame) -> dict:
    cols = [c for c in C.CATEGORICAL_FEATURES if c in df.columns]
    n = len(cols)
    if n == 0:
        return {}
    cols_per_row = 3
    rows = (n + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(5 * cols_per_row, 3 * rows))
    axes = axes.flatten() if rows > 1 else axes
    out = {}
    for i, col in enumerate(cols):
        ax = axes[i]
        risk_by = df.groupby(col)[C.TARGET_COLUMN].mean().sort_values(ascending=False)
        ax.bar(range(len(risk_by)), risk_by.values, color="#3b82f6",
               edgecolor="black", linewidth=0.4)
        ax.set_xticks(range(len(risk_by)))
        ax.set_xticklabels(risk_by.index, rotation=30, ha="right", fontsize=8)
        ax.set_title(col, fontsize=10)
        ax.set_ylabel("at-risk rate")
        ax.set_ylim(0, 1)
        out[col] = {k: round(float(v), 3) for k, v in risk_by.items()}
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("At-risk rate by category", y=1.005, fontsize=13)
    fig.savefig(EDA_DIR / "categorical.png")
    plt.close(fig)
    return out


def leakage_check(df: pd.DataFrame) -> dict:
    out = {}
    for col in C.LEAKAGE_COLUMNS:
        if col not in df.columns:
            continue
        s = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(s) and s.dtype != object
        if is_numeric:
            try:
                corr = s.astype(float).corr(df[C.TARGET_COLUMN].astype(float))
                out[col] = {"corr_with_target": round(float(corr), 4)}
            except Exception:
                out[col] = {"note": "non-comparable"}
        else:
            ct = pd.crosstab(s, df[C.TARGET_COLUMN], normalize="index")
            out[col] = {str(k): {str(c): round(float(v), 3) for c, v in row.items()}
                        for k, row in ct.iterrows()}
    return out


def run_eda() -> dict:
    _setup()
    raw = load_raw()
    raw[C.TARGET_COLUMN] = build_target(raw)
    summary = {
        "rows": int(len(raw)),
        "columns": int(raw.shape[1]),
        "class_balance": class_balance(raw),
        "missingness": missingness(raw),
        "numeric_distributions": numeric_distributions(raw),
        "correlation": correlation_heatmap(raw),
        "categorical": categorical_breakdown(raw),
        "leakage_check": leakage_check(raw),
    }
    out_path = C.REPORTS_DIR / "eda_report.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"EDA artefacts -> {EDA_DIR}")
    print(f"EDA summary  -> {out_path}")
    return summary


if __name__ == "__main__":
    run_eda()
