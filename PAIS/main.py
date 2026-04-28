"""
PAIS — End-to-end pipeline entry point.

Runs preprocessing → daily-log synthesis → training → evaluation →
DSA analytics → notification simulation in one shot.

Usage:
    python main.py              # run everything
    python main.py --skip-train # reuse an existing model
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Make `from src...` work no matter where we're invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config as C
from src.preprocessing import preprocess, load_raw
from src.daily_logs import generate_daily_logs, persist as persist_logs
from src.ml.train import train_and_persist
from src.ml.evaluate import evaluate_model
from src.oop.student_record import StudentCohort
from src.oop.risk_predictor import RiskPredictor
from src.dsa.sliding_window import detect_attendance_anomalies
from src.dsa.resource_graph import ResourceGraph
from src.dsa.sorter import rank_by_gradient
from src.dsa.hash_aggregator import StudentHashIndex
from src.business.alerts import AlertService
import pandas as pd


def section(title: str) -> None:
    print(f"\n{'='*72}\n{title}\n{'='*72}")


def main(skip_train: bool = False) -> None:
    C.ensure_dirs()

    # ------------------------------------------------------------------
    section("1. Preprocessing")
    df = preprocess()
    print(f"Processed {len(df):,} students  |  "
          f"at-risk rate = {df['at_risk'].mean():.1%}")

    # ------------------------------------------------------------------
    section("2. Daily engagement log synthesis")
    if not C.DATA_DAILY_LOGS.exists():
        logs = generate_daily_logs(load_raw())
        persist_logs(logs)
    else:
        logs = pd.read_csv(C.DATA_DAILY_LOGS)
    print(f"Loaded {len(logs):,} daily log rows.")

    # ------------------------------------------------------------------
    section("3. Model training & selection")
    if not skip_train or not (C.MODELS_DIR / "risk_model.pkl").exists():
        train_and_persist()
    else:
        print("  › --skip-train set and model exists, skipping.")

    # ------------------------------------------------------------------
    section("4. Evaluation & figures")
    evaluate_model()

    # ------------------------------------------------------------------
    section("5. DSA analytics")

    # Sliding window — worst-attendance windows per student.
    anomalies = detect_attendance_anomalies(logs)
    anom_path = C.REPORTS_DIR / "attendance_anomalies.csv"
    anomalies.to_csv(anom_path, index=False)
    print(f"  Sliding-window anomalies detected  : {len(anomalies):,}  → {anom_path}")

    # Bipartite graph — isolated students.
    g = ResourceGraph().build_from_logs(logs)
    summary = g.summary()
    print(f"  Resource graph                     : {summary}")

    # Gradient sort — top improvers and decliners.
    sorted_by_grad = rank_by_gradient(df)
    improvers = sorted_by_grad.head(10)
    decliners = sorted_by_grad.tail(10)
    print("\n  Top 10 improvers (midterm vs early avg):")
    print(improvers.to_string(index=False))
    print("\n  Top 10 decliners:")
    print(decliners.to_string(index=False))

    # ------------------------------------------------------------------
    section("6. Cohort scoring, ranking, notifications")
    cohort = StudentCohort.from_dataframe(df)
    service = AlertService()
    result = service.run_full_pipeline(cohort, marks_reflected_pct=55.0)

    print(f"  Scored cohort            : {result['scored']}")
    print(f"  Notifications simulated  : {result['notifications_sent']}")
    print("\n  Top-10 at-risk (max-heap peek):")
    for sid, score, meta in result["top_at_risk"]:
        print(f"    {sid}  {meta.get('name',''):<30s}  "
              f"score={score:.3f}  band={meta.get('band','')}")

    # ------------------------------------------------------------------
    section("7. Hash index demo")
    idx = StudentHashIndex().build_from_dataframe(df)
    idx.attach_logs(logs)
    sample = df["Student_ID"].iloc[0]
    profile = idx.get_profile(sample)
    print(f"  O(1) lookup for {sample}: grade={profile.get('Grade')} "
          f"days_attended={profile.get('days_attended')} "
          f"total_hits={profile.get('total_resource_hits')}")

    section("Done. Artifacts:")
    print(f"  • Model            → {C.MODELS_DIR/'risk_model.pkl'}")
    print(f"  • Training report  → {C.REPORTS_DIR/'training_report.json'}")
    print(f"  • Figures          → {C.FIGURES_DIR}/")
    print(f"  • Notifications    → {C.REPORTS_DIR/'notifications.jsonl'}")
    print(f"  • Anomalies CSV    → {anom_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true",
                        help="Reuse the persisted model instead of retraining.")
    args = parser.parse_args()
    main(skip_train=args.skip_train)
