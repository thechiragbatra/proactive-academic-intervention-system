from __future__ import annotations
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "students.csv"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed" / "students_processed.csv"
DATA_DAILY_LOGS = PROJECT_ROOT / "data" / "processed" / "daily_engagement.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


AT_RISK_GRADES = {"D", "F"}
AT_RISK_TOTAL_SCORE = 50.0


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


LEAKAGE_COLUMNS = ["Final_Score", "Total_Score", "Grade"]


RISK_WEIGHTS = {
    "attendance":     0.25,
    "midterm":        0.25,
    "assignments":    0.15,
    "quizzes":        0.10,
    "participation":  0.10,
    "projects":       0.10,
    "engagement":     0.05,
}

RISK_BANDS = [
    (0.80, "CRITICAL"),
    (0.60, "HIGH"),
    (0.40, "MODERATE"),
    (0.20, "LOW"),
    (0.00, "SAFE"),
]


SLIDING_WINDOW_DAYS = 7
ATTENDANCE_VARIANCE_FLAG = 0.25
SIMULATED_TERM_DAYS = 84


GRADE_CUTOFFS = {
    "O":  90.0,
    "A+": 80.0,
    "A":  70.0,
    "B+": 60.0,
    "B":  50.0,
    "C":  40.0,
    "P":  35.0,
}


REMAINING_EVAL_WEIGHTS = {
    "final_exam":       0.40,
    "final_project":    0.20,
    "remaining_quiz":   0.10,
    "remaining_assign": 0.10,
}


RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


AUDIT_TRIGGER_MARKS_PCT = 50
TOP_K_AT_RISK = 10


def ensure_dirs() -> None:
    for p in [DATA_PROCESSED.parent, MODELS_DIR, FIGURES_DIR]:
        p.mkdir(parents=True, exist_ok=True)
