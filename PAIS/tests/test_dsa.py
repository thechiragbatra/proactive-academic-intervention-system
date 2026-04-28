"""
Unit tests for DSA modules.

Run:  pytest -q tests/
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dsa.priority_queue import RiskHeap
from src.dsa.sliding_window import detect_attendance_anomalies
from src.dsa.hash_aggregator import StudentHashIndex
from src.dsa.resource_graph import ResourceGraph
from src.dsa.grade_optimizer import GradeOptimizer
from src.dsa.sorter import rank_by_gradient


# ---------------------------------------------------------------------------
# Priority queue
# ---------------------------------------------------------------------------
def test_heap_orders_by_descending_score():
    h = RiskHeap()
    h.push("S1", 0.2); h.push("S2", 0.9); h.push("S3", 0.5)
    top = [sid for sid, _, _ in h.peek_top(3)]
    assert top == ["S2", "S3", "S1"]


def test_heap_update_in_place():
    h = RiskHeap()
    h.push("S1", 0.1); h.push("S2", 0.9)
    h.push("S1", 0.95)          # update — should now lead
    top = [sid for sid, _, _ in h.peek_top(2)]
    assert top == ["S1", "S2"]
    assert h.get("S1") == 0.95


def test_heap_pop_drains_in_order():
    h = RiskHeap()
    for sid, score in [("A", 0.1), ("B", 0.3), ("C", 0.2)]:
        h.push(sid, score)
    order = [h.pop_top()[0] for _ in range(3)]
    assert order == ["B", "C", "A"]


# ---------------------------------------------------------------------------
# Sliding window
# ---------------------------------------------------------------------------
def test_sliding_window_flags_gap():
    logs = pd.DataFrame([
        ("S1", d, 1 if d < 15 else 0, 0)
        for d in range(1, 30)
    ], columns=["Student_ID", "day", "attended", "resource_hits"])
    anomalies = detect_attendance_anomalies(logs, window=7)
    assert not anomalies.empty
    assert anomalies.iloc[0]["Student_ID"] == "S1"
    assert anomalies.iloc[0]["mean_attended"] <= 0.5


def test_sliding_window_quiet_student_no_anomaly():
    logs = pd.DataFrame([
        ("S1", d, 1, 5)
        for d in range(1, 30)
    ], columns=["Student_ID", "day", "attended", "resource_hits"])
    anomalies = detect_attendance_anomalies(logs, window=7)
    assert anomalies.empty


# ---------------------------------------------------------------------------
# Hash index
# ---------------------------------------------------------------------------
def test_hash_index_o1_lookup():
    df = pd.DataFrame({
        "Student_ID": ["S1", "S2"],
        "Attendance (%)": [80.0, 55.0],
    })
    idx = StudentHashIndex().build_from_dataframe(df)
    assert idx.get_profile("S1")["Attendance (%)"] == 80.0
    assert idx.get_profile("S2")["Attendance (%)"] == 55.0
    assert idx.get_profile("S3") is None


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
def test_resource_graph_isolation():
    g = ResourceGraph()
    g.add_edge("S1", "R1", 3)
    g.add_edge("S1", "R2", 2)
    g.add_edge("S2", "R1", 1)   # S2 only touches R1
    isolated = g.isolated_students(min_edges=2)
    assert "S2" in isolated
    assert "S1" not in isolated


def test_resource_graph_bfs():
    g = ResourceGraph()
    g.add_edge("S1", "R1"); g.add_edge("S2", "R2"); g.add_edge("S3", "R1")
    reached = g.bfs_reachable_students(["R1"])
    assert reached == {"S1", "S3"}


# ---------------------------------------------------------------------------
# Grade optimizer
# ---------------------------------------------------------------------------
def test_optimizer_returns_zero_for_already_passed():
    opt = GradeOptimizer()
    rec = opt.full_roadmap(midterm=95, assignments=95, quizzes=95, projects=95)
    # Grade O should be feasible (probably already locked in).
    o_rec = next(g for g in rec if g.target_grade == "O")
    assert o_rec.feasible


def test_optimizer_infeasible_when_too_far_behind():
    opt = GradeOptimizer()
    rec = opt.recommend_for_grade("O",
                                  earned=opt.earned_weighted_marks(
                                      midterm=5, assignments=5,
                                      quizzes=5, projects=5))
    assert not rec.feasible


# ---------------------------------------------------------------------------
# Sorter
# ---------------------------------------------------------------------------
def test_gradient_sort_identifies_improver():
    df = pd.DataFrame({
        "Student_ID": ["S1", "S2"],
        "Midterm_Score": [90, 20],
        "early_academic_avg": [50, 50],
    })
    ranked = rank_by_gradient(df)
    assert ranked.iloc[0]["Student_ID"] == "S1"   # improver at top
    assert ranked.iloc[-1]["Student_ID"] == "S2"   # decliner at bottom
