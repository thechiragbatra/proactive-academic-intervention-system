"""
PAIS — Central configuration.

All paths, thresholds, feature lists, and tunable constants live here so
the rest of the codebase stays clean. Importing anything from here must
not have side effects beyond reading PROJECT_ROOT.
"""
from __future__ import annotations
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "students.csv"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed" / "students_processed.csv"
DATA_DAILY_LOGS = PROJECT_ROOT / "data" / "processed" / "daily_engagement.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ---------------------------------------------------------------------------
# Target definition
# ---------------------------------------------------------------------------
# A student is "at-risk" if they end up with a failing/near-failing grade.
# Grade D is included because in Indian universities D typically means a
# pass with remediation required — still an intervention target.
AT_RISK_GRADES = {"D", "F"}
AT_RISK_TOTAL_SCORE = 50.0   # backup rule: any student under 50 marks is at-risk

# ---------------------------------------------------------------------------
# Features available 4-6 weeks before finals (synopsis requirement)
# Final_Score and Total_Score are EXCLUDED to prevent target leakage.
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "Age",
    "Attendance (%)",
    "Midterm_Score",
    "Assignments_Avg",
    "Quizzes_Avg",
    "Participation_Score",
    "Projects_Score",
    "Study_Hours_per_Week",
    "Stress_Level (1-10)",
    "Sleep_Hours_per_Night",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Department",
    "Extracurricular_Activities",
    "Internet_Access_at_Home",
    "Parent_Education_Level",
    "Family_Income_Level",
]

ID_COLUMNS = ["Student_ID", "First_Name", "Last_Name", "Email"]
TARGET_COLUMN = "at_risk"

# Columns that MUST be dropped when training — they leak the outcome.
LEAKAGE_COLUMNS = ["Final_Score", "Total_Score", "Grade"]

# ---------------------------------------------------------------------------
# Risk scoring weights (composite, non-ML score used alongside model output)
# Weights sum to 1.0. Calibrated against ML output during training.
# ---------------------------------------------------------------------------
RISK_WEIGHTS = {
    "attendance":     0.25,
    "midterm":        0.25,
    "assignments":    0.15,
    "quizzes":        0.10,
    "participation":  0.10,
    "projects":       0.10,
    "engagement":     0.05,   # derived from study_hours + sleep
}

RISK_BANDS = [
    (0.80, "CRITICAL"),
    (0.60, "HIGH"),
    (0.40, "MODERATE"),
    (0.20, "LOW"),
    (0.00, "SAFE"),
]

# ---------------------------------------------------------------------------
# DSA & business rules
# ---------------------------------------------------------------------------
SLIDING_WINDOW_DAYS = 7
ATTENDANCE_VARIANCE_FLAG = 0.25   # flag if std / mean exceeds this within window
SIMULATED_TERM_DAYS = 84          # ~12 weeks

# Greedy grade optimizer — cutoffs used at Indian universities (common scheme).
GRADE_CUTOFFS = {
    "O":  90.0,
    "A+": 80.0,
    "A":  70.0,
    "B+": 60.0,
    "B":  50.0,
    "C":  40.0,
    "P":  35.0,   # pass
}

# Weightage of remaining evaluations after midterm (must sum to <= 1.0).
# Used by the grade optimizer to compute minimum marks needed.
REMAINING_EVAL_WEIGHTS = {
    "final_exam":       0.40,
    "final_project":    0.20,
    "remaining_quiz":   0.10,
    "remaining_assign": 0.10,
}
# Together with the 20% already captured in Midterm + early assignments.

# ---------------------------------------------------------------------------
# ML training
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# ---------------------------------------------------------------------------
# Notification thresholds
# ---------------------------------------------------------------------------
AUDIT_TRIGGER_MARKS_PCT = 50   # send audit mail once this % of marks reflected
TOP_K_AT_RISK = 10              # how many students to surface in dashboards


def ensure_dirs() -> None:
    """Create any output directories that don't exist yet."""
    for p in [DATA_PROCESSED.parent, MODELS_DIR, FIGURES_DIR]:
        p.mkdir(parents=True, exist_ok=True)
