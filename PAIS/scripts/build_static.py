"""
PAIS — Static build script for Netlify deployment.

Extracts the trained sklearn pipeline into a JSON model specification that
runs in the browser, and serialises all cohort + analytics data as static
JSON files.

Run once, after every retraining or data refresh:
    python scripts/build_static.py
"""
from __future__ import annotations
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C
from src.preprocessing import preprocess, load_raw
from src.daily_logs import generate_daily_logs
from src.oop.student_record import StudentCohort
from src.oop.risk_predictor import RiskPredictor
from src.dsa.sliding_window import detect_attendance_anomalies
from src.dsa.resource_graph import ResourceGraph
from src.dsa.sorter import rank_by_gradient

OUT = ROOT / "netlify_site"
OUT_DATA = OUT / "data"
OUT_FIG = OUT / "figures"


# ----------------------------------------------------------------------------
# Extract sklearn Pipeline → pure JSON the JS engine can execute
# ----------------------------------------------------------------------------
def _extract_model_spec(bundle) -> dict:
    pipe = bundle["model"]
    numeric_cols = bundle["numeric_cols"]
    categorical_cols = bundle["categorical_cols"]
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    num_t = pre.named_transformers_["num"]
    cat_t = pre.named_transformers_["cat"]
    return {
        "numeric": {
            "columns": list(numeric_cols),
            "impute":  num_t.named_steps["imp"].statistics_.tolist(),
            "mean":    num_t.named_steps["scale"].mean_.tolist(),
            "scale":   num_t.named_steps["scale"].scale_.tolist(),
        },
        "categorical": {
            "columns":    list(categorical_cols),
            "impute":     [str(x) for x in cat_t.named_steps["imp"].statistics_.tolist()],
            "categories": [[str(c) for c in cats]
                           for cats in cat_t.named_steps["onehot"].categories_],
        },
        "model": {
            "type": "logistic_regression",
            "coef": clf.coef_[0].tolist(),
            "intercept": float(clf.intercept_[0]),
        },
        "bands": [{"threshold": t, "label": l} for t, l in C.RISK_BANDS],
    }


def _verify(bundle, sample_df, n=20):
    """Sanity check: replicating the pipeline manually must match sklearn."""
    pipe = bundle["model"]
    spec = _extract_model_spec(bundle)
    cols = spec["numeric"]["columns"] + spec["categorical"]["columns"]
    X = sample_df[cols].head(n)
    sk = pipe.predict_proba(X)[:, 1]

    def manual(row):
        v = []
        for i, col in enumerate(spec["numeric"]["columns"]):
            x = row[col]
            if pd.isna(x):
                x = spec["numeric"]["impute"][i]
            v.append((x - spec["numeric"]["mean"][i]) / spec["numeric"]["scale"][i])
        for i, col in enumerate(spec["categorical"]["columns"]):
            x = row[col]
            if pd.isna(x) or x == "":
                x = spec["categorical"]["impute"][i]
            for c in spec["categorical"]["categories"][i]:
                v.append(1.0 if str(x) == c else 0.0)
        z = float(np.dot(v, spec["model"]["coef"]) + spec["model"]["intercept"])
        return 1.0 / (1.0 + np.exp(-z))

    mine = np.array([manual(r) for _, r in X.iterrows()])
    diff = float(np.max(np.abs(sk - mine)))
    print(f"  JS-replication max diff vs sklearn: {diff:.2e} "
          f"({'OK' if diff < 1e-6 else 'MISMATCH'})")


def _summary(r) -> dict:
    return {
        "student_id": r.student_id, "name": r.full_name,
        "department": r.department, "gender": r.gender, "age": r.age,
        "attendance":      round(r.attendance or 0, 2),
        "midterm":         round(r.midterm or 0, 2),
        "assignments_avg": round(r.assignments_avg or 0, 2),
        "quizzes_avg":     round(r.quizzes_avg or 0, 2),
        "participation":   round(r.participation or 0, 2),
        "projects":        round(r.projects or 0, 2),
        "grade":           r.grade,
        "risk_score":      round(r.risk_score or 0, 4),
        "risk_band":       r.risk_band,
    }


def build():
    if not (C.MODELS_DIR / "risk_model.pkl").exists():
        print("No trained model found. Run `python main.py` first.")
        sys.exit(1)

    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    (OUT_DATA / "student").mkdir(parents=True, exist_ok=True)

    print("1/6 Loading data…")
    df = preprocess(persist=False)
    if C.DATA_DAILY_LOGS.exists():
        logs = pd.read_csv(C.DATA_DAILY_LOGS)
    else:
        logs = generate_daily_logs(load_raw())
        logs.to_csv(C.DATA_DAILY_LOGS, index=False)

    print("2/6 Extracting model spec…")
    bundle = joblib.load(C.MODELS_DIR / "risk_model.pkl")
    spec = _extract_model_spec(bundle)
    _verify(bundle, df)
    (OUT_DATA / "model.json").write_text(json.dumps(spec, indent=2))

    print("3/6 Scoring cohort + DSA…")
    predictor = RiskPredictor.load()
    cohort = StudentCohort.from_dataframe(df)
    predictor.score_cohort(cohort)
    graph = ResourceGraph().build_from_logs(logs)
    anomalies_df = detect_attendance_anomalies(logs)
    anom_by_sid = {row["Student_ID"]: dict(row)
                   for _, row in anomalies_df.iterrows()}

    print("4/6 Writing summaries + stats…")
    summaries = [_summary(r) for r in cohort]
    (OUT_DATA / "students.json").write_text(json.dumps(summaries))

    bands = {b: 0 for b in ("CRITICAL", "HIGH", "MODERATE", "LOW", "SAFE")}
    dept_counts, dept_risk = {}, {}
    for r in cohort:
        if r.risk_band in bands:
            bands[r.risk_band] += 1
        dept_counts[r.department] = dept_counts.get(r.department, 0) + 1
        dept_risk.setdefault(r.department, []).append(r.risk_score or 0)

    stats = {
        "total": len(cohort),
        "bands": bands,
        "critical_plus_high": bands["CRITICAL"] + bands["HIGH"],
        "dept_counts": dept_counts,
        "dept_avg_risk": {d: round(sum(v)/len(v), 3)
                          for d, v in dept_risk.items()},
        "avg_risk": round(sum((r.risk_score or 0) for r in cohort)
                          / max(1, len(cohort)), 3),
        "graph_summary": graph.summary(),
        "anomaly_count": len(anomalies_df),
    }
    tr_path = C.REPORTS_DIR / "training_report.json"
    if tr_path.exists():
        tr = json.loads(tr_path.read_text())
        stats["winner"] = tr.get("winner")
        stats["test_metrics"] = (tr.get("results", {})
                                 .get(tr.get("winner"), {})
                                 .get("test_metrics", {}))
    (OUT_DATA / "stats.json").write_text(json.dumps(stats, indent=2))

    print("5/6 Writing per-student detail…")
    logs_by_sid = {}
    for sid, group in logs.sort_values(["Student_ID", "day"]).groupby("Student_ID"):
        logs_by_sid[sid] = [
            {"day": int(d), "attended": int(a), "resource_hits": int(h)}
            for d, a, h in group[["day", "attended", "resource_hits"]].values
        ]

    for i, r in enumerate(cohort):
        sid = r.student_id
        detail = {
            **_summary(r),
            "email": r.email, "first_name": r.first_name,
            "last_name": r.last_name,
            "study_hours": r.study_hours, "stress": r.stress, "sleep": r.sleep,
            "extracurricular": r.extracurricular,
            "internet_access": r.internet_access,
            "parent_education": r.parent_education,
            "family_income": r.family_income,
            "final_score": r.final_score, "total_score": r.total_score,
            "graph_engagement": round(graph.engagement_score(sid), 3),
            "anomaly": (anom_by_sid.get(sid) and {
                "start_day": int(anom_by_sid[sid]["start_day"]),
                "end_day":   int(anom_by_sid[sid]["end_day"]),
                "mean_attended": float(anom_by_sid[sid]["mean_attended"]),
                "std_attended":  float(anom_by_sid[sid]["std_attended"]),
                "reason":    anom_by_sid[sid]["reason"],
            }),
            "daily_logs": logs_by_sid.get(sid, []),
        }
        (OUT_DATA / "student" / f"{sid}.json").write_text(
            json.dumps(detail, separators=(",", ":")))
        if (i + 1) % 1000 == 0:
            print(f"    {i+1:,} / {len(cohort):,}")

    print("6/6 Writing analytics + report…")
    (OUT_DATA / "anomalies.json").write_text(
        json.dumps(anomalies_df.head(50).to_dict(orient="records"),
                   default=str))
    gradient = rank_by_gradient(df)
    (OUT_DATA / "gradient.json").write_text(json.dumps({
        "improvers": gradient.head(10).to_dict(orient="records"),
        "decliners": gradient.tail(10).iloc[::-1].to_dict(orient="records"),
    }))
    if tr_path.exists():
        shutil.copy(tr_path, OUT_DATA / "model_report.json")
    for fname in ("confusion_matrix.png", "roc_curve.png",
                  "feature_importance.png"):
        src = C.FIGURES_DIR / fname
        if src.exists():
            shutil.copy(src, OUT_FIG / fname)

    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"\nDone. Static site payload: {total/1024/1024:.1f} MB → {OUT}")


if __name__ == "__main__":
    build()
