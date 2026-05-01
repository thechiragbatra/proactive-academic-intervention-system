from __future__ import annotations
import pandas as pd
from typing import Any


class StudentHashIndex:

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}


    def build_from_dataframe(self, df: pd.DataFrame,
                             id_col: str = "Student_ID") -> "StudentHashIndex":
        self._store = {row[id_col]: row.to_dict() for _, row in df.iterrows()}
        return self

    def upsert(self, student_id: str, **fields: Any) -> None:
        profile = self._store.setdefault(student_id, {"Student_ID": student_id})
        profile.update(fields)

    def attach_logs(self, logs: pd.DataFrame) -> None:
        agg = logs.groupby("Student_ID").agg(
            days_attended=("attended", "sum"),
            total_days=("attended", "count"),
            total_resource_hits=("resource_hits", "sum"),
            avg_daily_hits=("resource_hits", "mean"),
        )
        for sid, row in agg.iterrows():
            if sid in self._store:
                self._store[sid].update(row.to_dict())


    def get_profile(self, student_id: str) -> dict[str, Any] | None:
        return self._store.get(student_id)

    def __contains__(self, student_id: str) -> bool:
        return student_id in self._store

    def __len__(self) -> int:
        return len(self._store)

    def ids(self) -> list[str]:
        return list(self._store.keys())

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._store.values())
