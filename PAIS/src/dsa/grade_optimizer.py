"""
DSA #5 — Greedy optimizer for personalised marks recommendation.

Problem
-------
A student has their midterm + early marks on the portal, plus a set of
remaining evaluations with fixed weights. What's the minimum total score
(and minimum per-evaluation score) they need on what's left to reach a
target letter grade?

Formulation
-----------
Let C be the target cutoff (e.g., 70 for grade A).
Let s_earned be marks already contributing to the final tally.
Let remaining evaluations e_1..e_n have weights w_1..w_n.

The student needs:   s_earned + Σ (x_i · w_i) ≥ C        (on a 100 scale)

We use a *greedy* strategy: spread the required additional marks evenly
across remaining evaluations weighted by their weight. This minimises the
maximum burden on any single evaluation while proving feasibility.

If the required per-evaluation score exceeds 100, the target is infeasible
and we report the best achievable grade instead.
"""
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
    """Compute minimum marks needed for each target grade."""

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
        """
        Marks already contributing to the final tally on a 100 scale.

        We treat the already-earned portion as an average × its combined
        weight (everything not in `remaining_weights`).
        """
        earned_weight = earned_weight or (1.0 - self.remaining_total_weight)
        avg = (0.40 * midterm + 0.25 * assignments
               + 0.20 * quizzes + 0.15 * projects)
        return avg * earned_weight

    def recommend_for_grade(self, target_grade: str,
                            earned: float) -> GradeRecommendation:
        """Return the marks-needed plan for one target grade."""
        cutoff = self.cutoffs[target_grade]
        gap = cutoff - earned                     # how many weighted points left

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

        # Greedy: same percentile score across remaining evals.
        # gap = p * remaining_total_weight  →  p = gap / remaining_total_weight
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
        """
        Produce recommendations for every grade, in cutoff-descending order.
        The student / mentor can then pick the most ambitious feasible grade.
        """
        earned = self.earned_weighted_marks(midterm, assignments,
                                            quizzes, projects)
        grades_by_cutoff = sorted(self.cutoffs.items(),
                                  key=lambda x: -x[1])
        return [self.recommend_for_grade(g, earned) for g, _ in grades_by_cutoff]

    def best_achievable_grade(self, midterm: float, assignments: float,
                              quizzes: float, projects: float) -> str:
        """The highest grade whose required_pct ≤ 100."""
        earned = self.earned_weighted_marks(midterm, assignments,
                                            quizzes, projects)
        best = "F"
        for grade, cutoff in sorted(self.cutoffs.items(), key=lambda x: -x[1]):
            gap = cutoff - earned
            if gap <= 0 or gap / self.remaining_total_weight <= 100.0:
                return grade
        return best
