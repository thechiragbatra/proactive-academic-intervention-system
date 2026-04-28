"""
OOP — Domain entity classes.

StudentRecord
    A single student's academic + behavioural profile. Encapsulates
    identity, grades, and engagement state. Provides derived properties
    (e.g., `attendance_risk`) so business logic doesn't have to recompute.

StudentCohort
    A collection of StudentRecords with factory methods to build from a
    DataFrame. Iterable, indexable by Student_ID, and supports bulk filters.

Design notes
------------
- `__slots__` keeps memory predictable for large cohorts.
- Identity fields are mangled with leading underscores + properties, so
  accidental mutation is caught (mentors should never be able to overwrite
  a Student_ID).
- StudentRecord is intentionally a small data object; heavy work lives in
  services (RiskPredictor, NotificationEngine) to keep concerns separated.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Iterable
import pandas as pd


class StudentRecord:
    """Encapsulates all available state for one student."""

    __slots__ = (
        "_student_id", "_first_name", "_last_name", "_email",
        "gender", "age", "department",
        "attendance", "midterm", "assignments_avg", "quizzes_avg",
        "participation", "projects", "final_score", "total_score", "grade",
        "study_hours", "stress", "sleep",
        "extracurricular", "internet_access",
        "parent_education", "family_income",
        "risk_score", "risk_band",
    )

    def __init__(self, **kwargs) -> None:
        # Identity (immutable via property)
        self._student_id    = kwargs["Student_ID"]
        self._first_name    = kwargs.get("First_Name", "")
        self._last_name     = kwargs.get("Last_Name", "")
        self._email         = kwargs.get("Email", "")
        # Demographics
        self.gender         = kwargs.get("Gender")
        self.age            = kwargs.get("Age")
        self.department     = kwargs.get("Department")
        # Academics
        self.attendance     = kwargs.get("Attendance (%)")
        self.midterm        = kwargs.get("Midterm_Score")
        self.assignments_avg = kwargs.get("Assignments_Avg")
        self.quizzes_avg    = kwargs.get("Quizzes_Avg")
        self.participation  = kwargs.get("Participation_Score")
        self.projects       = kwargs.get("Projects_Score")
        self.final_score    = kwargs.get("Final_Score")
        self.total_score    = kwargs.get("Total_Score")
        self.grade          = kwargs.get("Grade")
        # Behaviour
        self.study_hours    = kwargs.get("Study_Hours_per_Week")
        self.stress         = kwargs.get("Stress_Level (1-10)")
        self.sleep          = kwargs.get("Sleep_Hours_per_Night")
        # Context
        self.extracurricular = kwargs.get("Extracurricular_Activities")
        self.internet_access = kwargs.get("Internet_Access_at_Home")
        self.parent_education = kwargs.get("Parent_Education_Level")
        self.family_income  = kwargs.get("Family_Income_Level")
        # Filled by downstream services
        self.risk_score: float | None = None
        self.risk_band: str | None = None

    # ------------------------------------------------------------------
    # Read-only identity (encapsulation)
    # ------------------------------------------------------------------
    @property
    def student_id(self) -> str:   return self._student_id
    @property
    def first_name(self) -> str:   return self._first_name
    @property
    def last_name(self) -> str:    return self._last_name
    @property
    def email(self) -> str:        return self._email
    @property
    def full_name(self) -> str:
        return f"{self._first_name} {self._last_name}".strip()

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------
    @property
    def attendance_risk(self) -> str:
        """Coarse label driven purely by attendance %."""
        if self.attendance is None:    return "UNKNOWN"
        if self.attendance < 60:       return "CRITICAL"
        if self.attendance < 75:       return "HIGH"
        if self.attendance < 85:       return "MODERATE"
        return "LOW"

    @property
    def is_first_gen_college(self) -> bool:
        """Heuristic used downstream for tailored recommendations."""
        return self.parent_education in {"High School", "Unknown", None}

    def to_dict(self) -> dict:
        """Serialise to a plain dict (JSON-safe for downstream use)."""
        return {slot.lstrip("_"): getattr(self, slot) for slot in self.__slots__}

    def __repr__(self) -> str:
        return (f"StudentRecord({self._student_id}, {self.full_name!r}, "
                f"grade={self.grade}, risk={self.risk_band})")


# ---------------------------------------------------------------------------
# Cohort
# ---------------------------------------------------------------------------
class StudentCohort:
    """A keyed collection of StudentRecord objects."""

    def __init__(self, records: Iterable[StudentRecord] = ()) -> None:
        self._records: dict[str, StudentRecord] = {
            r.student_id: r for r in records
        }

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "StudentCohort":
        records = (StudentRecord(**row) for _, row in df.iterrows())
        return cls(records)

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[StudentRecord]:
        return iter(self._records.values())

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, student_id: str) -> StudentRecord:
        return self._records[student_id]

    def __contains__(self, student_id: str) -> bool:
        return student_id in self._records

    # ------------------------------------------------------------------
    # Bulk ops
    # ------------------------------------------------------------------
    def filter(self, predicate) -> "StudentCohort":
        return StudentCohort(r for r in self if predicate(r))

    def by_department(self, dept: str) -> "StudentCohort":
        return self.filter(lambda r: r.department == dept)

    def at_risk(self) -> "StudentCohort":
        return self.filter(
            lambda r: r.risk_band in {"CRITICAL", "HIGH"}
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(r.to_dict() for r in self)
