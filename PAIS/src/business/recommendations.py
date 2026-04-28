"""
Business logic — personalised recommendation text generator.

Produces human-readable, actionable advice for a student based on:
1. Which of their academic dimensions are weakest (attendance / midterm /
   assignments / quizzes / participation / projects).
2. The greedy grade optimizer's minimum-marks roadmap.
3. Their behavioural signals (stress, sleep, study hours).

Kept text-first (not ML-generated) so mentors can audit and edit easily.
"""
from __future__ import annotations
from dataclasses import dataclass

from ..oop.student_record import StudentRecord
from ..dsa.grade_optimizer import GradeOptimizer


# ---------------------------------------------------------------------------
# Rule snippets — one per diagnosable condition.
# ---------------------------------------------------------------------------
TIPS = {
    "attendance_low": (
        "• Your attendance is below 75%. Book a meeting with your mentor "
        "this week — universities flag attendance gaps on transcripts."
    ),
    "attendance_mod": (
        "• Your attendance is slipping toward the 75% threshold. Attend every "
        "lecture this week to create a safety buffer."
    ),
    "midterm_low": (
        "• Midterm score is weak. Ask your faculty for the answer key review "
        "session and rework the two topics you lost the most marks on."
    ),
    "assignments_low": (
        "• Assignment average is pulling your overall score down. Submit the "
        "next two assignments 48 hours early to allow revision cycles."
    ),
    "quizzes_low": (
        "• Quiz scores indicate shallow prep. Use a daily 20-minute retrieval "
        "practice block (flashcards / past quizzes) for the next 3 weeks."
    ),
    "participation_low": (
        "• Classroom participation score is low. Aim to ask one real question "
        "per class and contribute to at least one discussion thread weekly."
    ),
    "projects_low": (
        "• Project marks are weak. Schedule one 90-minute block each week "
        "specifically for project work — treat it like a fixed class."
    ),
    "stress_high": (
        "• Your reported stress is high. The campus counselling cell is free "
        "and confidential — one session can materially reduce exam-week collapse."
    ),
    "sleep_low": (
        "• Less than 6 hours of sleep significantly reduces recall. Target "
        "7+ hours for at least 5 nights/week through the exam window."
    ),
    "study_hours_low": (
        "• You're studying fewer than 10 hours/week outside class. Even moving "
        "to 15 meaningful hours, distributed across 5 days, measurably lifts outcomes."
    ),
}


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
def _diagnose(r: StudentRecord) -> list[str]:
    tips: list[str] = []
    if r.attendance is not None:
        if r.attendance < 60:   tips.append(TIPS["attendance_low"])
        elif r.attendance < 75: tips.append(TIPS["attendance_mod"])
    if r.midterm is not None and r.midterm < 55:
        tips.append(TIPS["midterm_low"])
    if r.assignments_avg is not None and r.assignments_avg < 60:
        tips.append(TIPS["assignments_low"])
    if r.quizzes_avg is not None and r.quizzes_avg < 60:
        tips.append(TIPS["quizzes_low"])
    if r.participation is not None and r.participation < 55:
        tips.append(TIPS["participation_low"])
    if r.projects is not None and r.projects < 55:
        tips.append(TIPS["projects_low"])
    if r.stress is not None and r.stress >= 8:
        tips.append(TIPS["stress_high"])
    if r.sleep is not None and r.sleep < 6:
        tips.append(TIPS["sleep_low"])
    if r.study_hours is not None and r.study_hours < 10:
        tips.append(TIPS["study_hours_low"])
    if not tips:
        tips.append("• Profile looks healthy — keep the current routine and "
                    "scan the grade optimizer below for any stretch targets.")
    return tips


def build_recommendation_text(r: StudentRecord,
                              optimizer: GradeOptimizer | None = None) -> str:
    """
    Compose the full personalised recommendation for a student.
    """
    optimizer = optimizer or GradeOptimizer()
    diagnostics = _diagnose(r)

    # Greedy roadmap — show the top 3 feasible grades.
    roadmap = optimizer.full_roadmap(
        midterm=r.midterm or 0,
        assignments=r.assignments_avg or 0,
        quizzes=r.quizzes_avg or 0,
        projects=r.projects or 0,
    )
    feasible = [g for g in roadmap if g.feasible][:3]

    roadmap_lines = []
    for g in feasible:
        per_eval = next(iter(g.per_eval_score_needed.values()))
        roadmap_lines.append(
            f"  → Grade {g.target_grade:<2s} (cutoff {g.cutoff:.0f}): need "
            f"≈ {per_eval:.0f}% on each remaining evaluation."
        )

    parts = [
        "Diagnostics:",
        "\n".join(diagnostics),
        "",
        "Grade roadmap (minimum consistent % on remaining evaluations):",
        "\n".join(roadmap_lines) if roadmap_lines
        else "  → All ambitious targets are currently out of reach. "
             "Focus on ensuring a clean pass, then rebuild.",
    ]
    return "\n".join(parts)


def recommend_for_student(r: StudentRecord) -> str:
    return build_recommendation_text(r)
