from __future__ import annotations
from dataclasses import dataclass

from .. import config as C


@dataclass
class GradeRecommendation:
    target_grade: str
    cutoff: float
    already_earned_weighted: float
    required_from_remaining: float
    per_eval_score_needed: dict[str, float]
    feasible: bool
    note: str


class GradeOptimizer:

    def __init__(self,
                 cutoffs: dict[str, float] | None = None,
                 remaining_weights: dict[str, float] | None = None) -> None:
        self.cutoffs = cutoffs or C.GRADE_CUTOFFS
        self.remaining_weights = remaining_weights or C.REMAINING_EVAL_WEIGHTS
        self.remaining_total_weight = sum(self.remaining_weights.values())
        assert 0 < self.remaining_total_weight <= 1.0, \
            "remaining weights must be positive and sum to ≤ 1.0"

    def earned_weighted_marks(self, midterm: float, assignments: float,
                              quizzes: float, projects: float,
                              earned_weight: float | None = None) -> float:
        earned_weight = earned_weight or (1.0 - self.remaining_total_weight)
        avg = (0.40 * midterm + 0.25 * assignments
               + 0.20 * quizzes + 0.15 * projects)
        return avg * earned_weight

    def recommend_for_grade(self, target_grade: str,
                            earned: float) -> GradeRecommendation:
        cutoff = self.cutoffs[target_grade]
        gap = cutoff - earned

        if gap <= 0:
            return GradeRecommendation(
                target_grade=target_grade,
                cutoff=cutoff,
                already_earned_weighted=round(earned, 2),
                required_from_remaining=0.0,
                per_eval_score_needed={k: 0.0 for k in self.remaining_weights},
                feasible=True,
                note="Already locked in — just don't drop off.",
            )


        required_pct = gap / self.remaining_total_weight
        feasible = required_pct <= 100.0

        per_eval = {k: round(required_pct, 2) for k in self.remaining_weights}
        note = ("Feasible — consistent effort required."
                if feasible else
                f"Target out of reach: would need {required_pct:.1f}% "
                f"on every remaining evaluation.")

        return GradeRecommendation(
            target_grade=target_grade,
            cutoff=cutoff,
            already_earned_weighted=round(earned, 2),
            required_from_remaining=round(gap, 2),
            per_eval_score_needed=per_eval,
            feasible=feasible,
            note=note,
        )

    def full_roadmap(self, midterm: float, assignments: float,
                     quizzes: float, projects: float) -> list[GradeRecommendation]:
        earned = self.earned_weighted_marks(midterm, assignments,
                                            quizzes, projects)
        grades_by_cutoff = sorted(self.cutoffs.items(),
                                  key=lambda x: -x[1])
        return [self.recommend_for_grade(g, earned) for g, _ in grades_by_cutoff]

    def best_achievable_grade(self, midterm: float, assignments: float,
                              quizzes: float, projects: float) -> str:
        earned = self.earned_weighted_marks(midterm, assignments,
                                            quizzes, projects)
        best = "F"
        for grade, cutoff in sorted(self.cutoffs.items(), key=lambda x: -x[1]):
            gap = cutoff - earned
            if gap <= 0 or gap / self.remaining_total_weight <= 100.0:
                return grade
        return best
