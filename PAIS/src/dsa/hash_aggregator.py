"""
DSA #3 — Hash Map / Dictionary-backed student index.

Problem
-------
The system pulls data from multiple sources (academic rows, daily logs,
notifications). Scanning a DataFrame every time a mentor clicks a student
is O(n). A hash index gives O(1) lookup and update.

We deliberately wrap Python's dict so the usage site reads clearly
("index.get_profile(sid)" instead of "df[df.Student_ID == sid].iloc[0]")
and to enforce that only one profile object exists per student.
"""
from __future__ import annotations
import pandas as pd
from typing import Any


class StudentHashIndex:
    """
    Dictionary-backed index from Student_ID to a profile dict.

    Using a plain dict is the right call here: Python dicts are open-addressed
    hash tables with O(1) amortized ops, and we need nothing more.
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Build / update
    # ------------------------------------------------------------------
    def build_from_dataframe(self, df: pd.DataFrame,
                             id_col: str = "Student_ID") -> "StudentHashIndex":
        """Bulk-build from a DataFrame. O(n)."""
        self._store = {row[id_col]: row.to_dict() for _, row in df.iterrows()}
        return self

    def upsert(self, student_id: str, **fields: Any) -> None:
        """Create or merge fields into a profile. O(1)."""
        profile = self._store.setdefault(student_id, {"Student_ID": student_id})
        profile.update(fields)

    def attach_logs(self, logs: pd.DataFrame) -> None:
        """Aggregate per-student daily logs into the profile."""
        agg = logs.groupby("Student_ID").agg(
            days_attended=("attended", "sum"),
            total_days=("attended", "count"),
            total_resource_hits=("resource_hits", "sum"),
            avg_daily_hits=("resource_hits", "mean"),
        )
        for sid, row in agg.iterrows():
            if sid in self._store:
                self._store[sid].update(row.to_dict())

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
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
