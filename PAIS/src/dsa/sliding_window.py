from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass

from .. import config as C


@dataclass
class AnomalyWindow:
    student_id: str
    start_day: int
    end_day: int
    mean_attended: float
    std_attended: float
    reason: str


def _detect_for_student(student_id: str, days: list[int],
                        attended: list[int],
                        window: int) -> list[AnomalyWindow]:
    if len(attended) < window:
        return []

    anomalies: list[AnomalyWindow] = []
    running_sum = sum(attended[:window])

    for i in range(len(attended) - window + 1):
        if i > 0:
            running_sum += attended[i + window - 1] - attended[i - 1]

        chunk = attended[i:i + window]
        mean = running_sum / window
        std = float(np.std(chunk))

        reasons = []
        if mean < 0.5:
            reasons.append("attendance dropped below 50%")
        if mean > 0 and (std / mean) > C.ATTENDANCE_VARIANCE_FLAG:
            reasons.append("high variance (erratic)")

        if reasons:
            anomalies.append(AnomalyWindow(
                student_id=student_id,
                start_day=days[i],
                end_day=days[i + window - 1],
                mean_attended=round(mean, 3),
                std_attended=round(std, 3),
                reason="; ".join(reasons),
            ))
    return anomalies


def detect_attendance_anomalies(
    logs: pd.DataFrame,
    *,
    window: int = C.SLIDING_WINDOW_DAYS,
) -> pd.DataFrame:
    all_anomalies: list[AnomalyWindow] = []
    for sid, group in logs.sort_values(["Student_ID", "day"]).groupby("Student_ID"):
        all_anomalies.extend(_detect_for_student(
            sid,
            days=group["day"].tolist(),
            attended=group["attended"].tolist(),
            window=window,
        ))

    if not all_anomalies:
        return pd.DataFrame(columns=[
            "Student_ID", "start_day", "end_day",
            "mean_attended", "std_attended", "reason"
        ])

    out = pd.DataFrame([a.__dict__ for a in all_anomalies])

    worst = out.sort_values("mean_attended").drop_duplicates("student_id")
    worst = worst.rename(columns={"student_id": "Student_ID"})
    return worst.reset_index(drop=True)
