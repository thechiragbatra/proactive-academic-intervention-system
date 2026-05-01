from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Iterable

from .. import config as C
from .student_record import StudentRecord, StudentCohort


class RiskPredictor:

    def __init__(self, model=None, feature_names: list[str] | None = None,
                 *, ml_weight: float = 0.7) -> None:
        self.model = model
        self.feature_names = feature_names or []
        self.ml_weight = ml_weight
        self.rule_weight = 1.0 - ml_weight


    @classmethod
    def load(cls, path: Path | None = None) -> "RiskPredictor":
        path = path or (C.MODELS_DIR / "risk_model.pkl")
        bundle = joblib.load(path)
        return cls(
            model=bundle["model"],
            feature_names=bundle.get("feature_names", []),
        )


    def rule_based_score(self, record: StudentRecord) -> float:
        w = C.RISK_WEIGHTS

        def clamp(x): return max(0.0, min(1.0, x))

        comp = {
            "attendance":    clamp((75 - (record.attendance or 0)) / 75),
            "midterm":       clamp((60 - (record.midterm or 0)) / 60),
            "assignments":   clamp((60 - (record.assignments_avg or 0)) / 60),
            "quizzes":       clamp((60 - (record.quizzes_avg or 0)) / 60),
            "participation": clamp((60 - (record.participation or 0)) / 60),
            "projects":      clamp((60 - (record.projects or 0)) / 60),

            "engagement":    clamp(
                ((record.stress or 5) - 5) / 10
                + (7 - (record.sleep or 7)) / 10
            ),
        }
        return sum(w[k] * comp[k] for k in w)

    def ml_probability(self, rows: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("No ML model loaded. Call RiskPredictor.load() first.")
        return self.model.predict_proba(rows)[:, 1]


    def score_cohort(self, cohort: StudentCohort) -> None:

        rows = pd.DataFrame([r.to_dict() for r in cohort])
        rows = self._prepare_for_model(rows)

        ml_probs = self.ml_probability(rows)

        for record, p in zip(cohort, ml_probs):
            rule = self.rule_based_score(record)
            blended = self.ml_weight * p + self.rule_weight * rule
            record.risk_score = float(round(blended, 4))
            record.risk_band = self._band_for(blended)

    def _prepare_for_model(self, df: pd.DataFrame) -> pd.DataFrame:


        col_rename = {
            "attendance":        "Attendance (%)",
            "midterm":           "Midterm_Score",
            "assignments_avg":   "Assignments_Avg",
            "quizzes_avg":       "Quizzes_Avg",
            "participation":     "Participation_Score",
            "projects":          "Projects_Score",
            "study_hours":       "Study_Hours_per_Week",
            "stress":            "Stress_Level (1-10)",
            "sleep":             "Sleep_Hours_per_Night",
            "gender":            "Gender",
            "age":               "Age",
            "department":        "Department",
            "extracurricular":   "Extracurricular_Activities",
            "internet_access":   "Internet_Access_at_Home",
            "parent_education":  "Parent_Education_Level",
            "family_income":     "Family_Income_Level",
        }
        df = df.rename(columns=col_rename)


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

        return df[self.feature_names]


    @staticmethod
    def _band_for(score: float) -> str:
        for threshold, label in C.RISK_BANDS:
            if score >= threshold:
                return label
        return "SAFE"
