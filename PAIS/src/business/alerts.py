"""
Business logic — AlertService orchestrator.

Wires the RiskPredictor + GradeOptimizer + NotificationEngine into a single
service the dashboard and the CLI can both consume.
"""
from __future__ import annotations
from pathlib import Path

from ..oop.student_record import StudentCohort
from ..oop.risk_predictor import RiskPredictor
from ..oop.notification_engine import (
    NotificationEngine, ConsoleDispatcher, JsonlDispatcher,
)
from ..dsa.priority_queue import RiskHeap
from ..dsa.grade_optimizer import GradeOptimizer
from .recommendations import build_recommendation_text
from .. import config as C


class AlertService:
    """Top-level façade — one object, three verbs: score / rank / notify."""

    def __init__(self, predictor: RiskPredictor | None = None,
                 engine: NotificationEngine | None = None) -> None:
        self.predictor = predictor or RiskPredictor.load()
        self.engine = engine or NotificationEngine(
            dispatcher=JsonlDispatcher(
                C.REPORTS_DIR / "notifications.jsonl"
            )
        )
        self.optimizer = GradeOptimizer()

    def score(self, cohort: StudentCohort) -> None:
        self.predictor.score_cohort(cohort)

    def rank(self, cohort: StudentCohort, top_k: int | None = None) -> list[tuple]:
        top_k = top_k or C.TOP_K_AT_RISK
        heap = RiskHeap()
        for r in cohort:
            if r.risk_score is None:
                continue
            heap.push(r.student_id, r.risk_score,
                      metadata={"name": r.full_name, "band": r.risk_band})
        return heap.peek_top(top_k)

    def notify(self, cohort: StudentCohort, *,
               notify_parents_for: set[str] | None = None,
               marks_reflected_pct: float = 55.0) -> int:
        """
        Build recommendations and fire off batch notifications.

        Returns the count of notifications sent (students + parents).
        """
        recs = {
            r.student_id: build_recommendation_text(r, self.optimizer)
            for r in cohort
        }
        before = len(self.engine.sent_log)
        self.engine.batch_notify(
            cohort, recs,
            notify_parents_for=notify_parents_for,
            marks_reflected_pct=marks_reflected_pct,
        )
        return len(self.engine.sent_log) - before

    def run_full_pipeline(self, cohort: StudentCohort,
                          *, marks_reflected_pct: float = 55.0) -> dict:
        self.score(cohort)
        top = self.rank(cohort)
        sent = self.notify(cohort, marks_reflected_pct=marks_reflected_pct)
        return {
            "scored": len(cohort),
            "top_at_risk": top,
            "notifications_sent": sent,
        }
