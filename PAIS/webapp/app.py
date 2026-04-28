"""
PAIS — Flask web application.

Serves a multi-page analytics dashboard backed by the trained ML model,
the DSA modules, and the OOP service layer.

Run:
    python webapp/app.py
    # then open http://127.0.0.1:5000
"""
from __future__ import annotations
import json
import sys
import traceback
from pathlib import Path
from functools import lru_cache
from datetime import datetime

import pandas as pd
from flask import (
    Flask, render_template, request, jsonify,
    redirect, url_for, flash, send_from_directory,
)
from werkzeug.utils import secure_filename

# Make `src.*` imports work when running this file directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C
from src.preprocessing import preprocess, load_raw, model_feature_lists
from src.daily_logs import generate_daily_logs
from src.oop.student_record import StudentCohort, StudentRecord
from src.oop.risk_predictor import RiskPredictor
from src.oop.notification_engine import (
    NotificationEngine, JsonlDispatcher,
)
from src.dsa.priority_queue import RiskHeap
from src.dsa.sliding_window import detect_attendance_anomalies
from src.dsa.hash_aggregator import StudentHashIndex
from src.dsa.resource_graph import ResourceGraph
from src.dsa.grade_optimizer import GradeOptimizer
from src.dsa.sorter import rank_by_gradient
from src.business.recommendations import build_recommendation_text


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Flask(__name__,
            template_folder=str(ROOT / "webapp" / "templates"),
            static_folder=str(ROOT / "webapp" / "static"))
app.config["SECRET_KEY"] = "pais-dev-secret-change-in-prod"
app.config["UPLOAD_FOLDER"] = str(ROOT / "webapp" / "uploads")
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024   # 20 MB
Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"csv"}


# ---------------------------------------------------------------------------
# Lazy, cached pipeline loader
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _pipeline_state():
    """Load the processed dataset, train-time artefacts, and DSA structures once."""
    df = preprocess(persist=False)

    logs_path = C.DATA_DAILY_LOGS
    if logs_path.exists():
        logs = pd.read_csv(logs_path)
    else:
        logs = generate_daily_logs(load_raw())
        logs_path.parent.mkdir(parents=True, exist_ok=True)
        logs.to_csv(logs_path, index=False)

    predictor = RiskPredictor.load()
    cohort = StudentCohort.from_dataframe(df)
    predictor.score_cohort(cohort)

    heap = RiskHeap()
    for r in cohort:
        if r.risk_score is not None:
            heap.push(r.student_id, r.risk_score,
                      metadata={"name": r.full_name, "band": r.risk_band,
                                "department": r.department})

    hash_idx = StudentHashIndex().build_from_dataframe(df)
    hash_idx.attach_logs(logs)

    graph = ResourceGraph().build_from_logs(logs)
    anomalies = detect_attendance_anomalies(logs)

    training_report_path = C.REPORTS_DIR / "training_report.json"
    training_report = (json.loads(training_report_path.read_text())
                       if training_report_path.exists() else {})

    state = {
        "df": df,
        "logs": logs,
        "cohort": cohort,
        "predictor": predictor,
        "heap": heap,
        "hash_idx": hash_idx,
        "graph": graph,
        "anomalies": anomalies,
        "training_report": training_report,
        "optimizer": GradeOptimizer(),
    }

    # Replay any persisted edits so changes survive restart.
    replayed = _replay_edits_on_cohort(state)
    if replayed:
        print(f"  replayed {replayed} persisted edit(s) from edits_overlay.jsonl")

    return state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _cohort_stats(cohort: StudentCohort) -> dict:
    bands = {"CRITICAL": 0, "HIGH": 0, "MODERATE": 0, "LOW": 0, "SAFE": 0}
    dept_counts: dict[str, int] = {}
    dept_risk: dict[str, list[float]] = {}
    for r in cohort:
        if r.risk_band in bands:
            bands[r.risk_band] += 1
        dept_counts[r.department] = dept_counts.get(r.department, 0) + 1
        dept_risk.setdefault(r.department, []).append(r.risk_score or 0)

    dept_avg = {d: round(sum(v) / len(v), 3) for d, v in dept_risk.items()}
    return {
        "total": len(cohort),
        "bands": bands,
        "critical_plus_high": bands["CRITICAL"] + bands["HIGH"],
        "dept_counts": dept_counts,
        "dept_avg_risk": dept_avg,
        "avg_risk": round(
            sum((r.risk_score or 0) for r in cohort) / max(1, len(cohort)), 3),
    }


def _record_to_dict(r: StudentRecord) -> dict:
    return {
        "student_id": r.student_id, "name": r.full_name, "email": r.email,
        "department": r.department, "gender": r.gender, "age": r.age,
        "attendance": r.attendance, "midterm": r.midterm,
        "assignments_avg": r.assignments_avg, "quizzes_avg": r.quizzes_avg,
        "participation": r.participation, "projects": r.projects,
        "study_hours": r.study_hours, "stress": r.stress, "sleep": r.sleep,
        "final_score": r.final_score, "total_score": r.total_score,
        "grade": r.grade, "risk_score": r.risk_score, "risk_band": r.risk_band,
        "parent_education": r.parent_education, "family_income": r.family_income,
        "internet_access": r.internet_access, "extracurricular": r.extracurricular,
    }


# ---------------------------------------------------------------------------
# HTML routes
# ---------------------------------------------------------------------------
@app.route("/")
def dashboard():
    state = _pipeline_state()
    stats = _cohort_stats(state["cohort"])
    top10 = state["heap"].peek_top(10)
    winner = state["training_report"].get("winner", "—")
    top_metrics = (state["training_report"]
                   .get("results", {})
                   .get(winner, {})
                   .get("test_metrics", {}))
    return render_template(
        "dashboard.html",
        stats=stats, top10=top10, winner=winner, top_metrics=top_metrics,
        graph_summary=state["graph"].summary(),
        anomaly_count=len(state["anomalies"]),
    )


@app.route("/students")
def students_list():
    return render_template("students.html")


@app.route("/students/<student_id>")
def student_detail(student_id: str):
    state = _pipeline_state()
    try:
        record = state["cohort"][student_id]
    except KeyError:
        flash(f"Student {student_id} not found.", "error")
        return redirect(url_for("students_list"))

    rec_text = build_recommendation_text(record, state["optimizer"])
    roadmap = state["optimizer"].full_roadmap(
        midterm=record.midterm or 0,
        assignments=record.assignments_avg or 0,
        quizzes=record.quizzes_avg or 0,
        projects=record.projects or 0,
    )
    log_rows = state["logs"][
        state["logs"]["Student_ID"] == student_id
    ].sort_values("day").to_dict(orient="records")
    agg = state["hash_idx"].get_profile(student_id) or {}
    graph_score = state["graph"].engagement_score(student_id)
    anom_row = state["anomalies"][state["anomalies"]["Student_ID"] == student_id]
    anomaly = anom_row.iloc[0].to_dict() if not anom_row.empty else None

    return render_template(
        "student_detail.html",
        r=record, recommendation=rec_text, roadmap=roadmap,
        daily_logs=log_rows, agg=agg, graph_score=graph_score, anomaly=anomaly,
    )


@app.route("/analytics")
def analytics():
    state = _pipeline_state()
    stats = _cohort_stats(state["cohort"])
    anomalies = state["anomalies"].head(50).to_dict(orient="records")
    gradient = rank_by_gradient(state["df"])
    improvers = gradient.head(10).to_dict(orient="records")
    decliners = gradient.tail(10).iloc[::-1].to_dict(orient="records")
    return render_template(
        "analytics.html",
        stats=stats, anomalies=anomalies,
        improvers=improvers, decliners=decliners,
        graph_summary=state["graph"].summary(),
    )


@app.route("/model")
def model_page():
    state = _pipeline_state()
    return render_template("model.html", report=state["training_report"])


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            flash("No file selected.", "error")
            return redirect(url_for("upload"))
        if not ("." in file.filename
                and file.filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS):
            flash("Only CSV files are supported.", "error")
            return redirect(url_for("upload"))

        fname = secure_filename(file.filename)
        dest = Path(app.config["UPLOAD_FOLDER"]) / fname
        file.save(dest)

        try:
            new_df = pd.read_csv(dest)
            required_raw = (set(C.NUMERIC_FEATURES) | set(C.CATEGORICAL_FEATURES)
                            | {"Student_ID"})
            missing = required_raw - set(new_df.columns)
            if missing:
                flash(f"Uploaded CSV missing columns: {sorted(missing)}", "error")
                return redirect(url_for("upload"))

            predictor = RiskPredictor.load()
            new_df["Parent_Education_Level"] = new_df["Parent_Education_Level"].fillna("Unknown")
            new_df["early_academic_avg"] = (
                0.40 * new_df["Midterm_Score"] + 0.25 * new_df["Assignments_Avg"]
                + 0.20 * new_df["Quizzes_Avg"] + 0.15 * new_df["Projects_Score"])
            new_df["engagement_index"] = (
                new_df["Study_Hours_per_Week"].clip(0, 40) / 4
                - (new_df["Stress_Level (1-10)"] - 5) * 0.3
                + (new_df["Sleep_Hours_per_Night"] - 6) * 0.4
            ).clip(0, 15)
            new_df["attendance_deficit"] = (75.0 - new_df["Attendance (%)"]).clip(lower=0)
            new_df["low_att_low_mid"] = (
                (new_df["Attendance (%)"] < 70) & (new_df["Midterm_Score"] < 50)
            ).astype(int)

            probs = predictor.model.predict_proba(new_df[predictor.feature_names])[:, 1]
            new_df["risk_score"] = probs
            new_df["risk_band"] = [RiskPredictor._band_for(p) for p in probs]

            out_path = Path(app.config["UPLOAD_FOLDER"]) / f"scored_{fname}"
            new_df.to_csv(out_path, index=False)

            return render_template(
                "upload.html",
                results=new_df.head(100).to_dict(orient="records"),
                n_scored=len(new_df),
                n_at_risk=int((new_df["risk_band"].isin(["CRITICAL", "HIGH"])).sum()),
                download_file=out_path.name,
            )
        except Exception as e:
            traceback.print_exc()
            flash(f"Scoring failed: {e}", "error")
            return redirect(url_for("upload"))

    return render_template("upload.html")


@app.route("/uploads/<filename>")
def download_upload(filename: str):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename,
                               as_attachment=True)


@app.route("/figures/<filename>")
def serve_figure(filename: str):
    return send_from_directory(C.FIGURES_DIR, filename)


# ---------------------------------------------------------------------------
# Edit — form, simulate, update, persistence
# ---------------------------------------------------------------------------
EDIT_FIELDS_NUMERIC = {
    "attendance":      ("Attendance (%)",         0.0, 100.0, float),
    "midterm":         ("Midterm_Score",          0.0, 100.0, float),
    "assignments_avg": ("Assignments_Avg",        0.0, 100.0, float),
    "quizzes_avg":     ("Quizzes_Avg",            0.0, 100.0, float),
    "participation":   ("Participation_Score",    0.0, 100.0, float),
    "projects":        ("Projects_Score",         0.0, 100.0, float),
    "study_hours":     ("Study_Hours_per_Week",   0.0,  80.0, float),
    "stress":          ("Stress_Level (1-10)",    1,   10,    int),
    "sleep":           ("Sleep_Hours_per_Night",  0.0,  14.0, float),
    "age":             ("Age",                    15,  80,    int),
}

EDIT_FIELDS_CATEGORICAL = {
    "gender":                 ("Gender", ["Male", "Female", "Other"]),
    "department":             ("Department", ["CS", "Engineering", "Mathematics", "Business"]),
    "extracurricular":        ("Extracurricular_Activities", ["Yes", "No"]),
    "internet_access":        ("Internet_Access_at_Home", ["Yes", "No"]),
    "parent_education":       ("Parent_Education_Level",
                               ["None", "High School", "Bachelor's", "Master's", "PhD", "Unknown"]),
    "family_income":          ("Family_Income_Level", ["Low", "Medium", "High"]),
}

EDITS_OVERLAY_FILE = None  # resolved at runtime via config


def _edits_overlay_path():
    return C.REPORTS_DIR / "edits_overlay.jsonl"


def _apply_edits_from_form(record, form) -> dict[str, str]:
    """
    Parse form values, validate, and apply them to `record`.
    Returns a dict of {field: error_msg} for any fields that failed validation.
    Only writes valid fields; partial updates OK.
    """
    errors: dict[str, str] = {}
    for attr, (_col, lo, hi, caster) in EDIT_FIELDS_NUMERIC.items():
        raw = form.get(attr, "").strip()
        if raw == "":
            continue
        try:
            val = caster(raw)
        except (TypeError, ValueError):
            errors[attr] = f"not a valid {caster.__name__}"
            continue
        if not (lo <= val <= hi):
            errors[attr] = f"must be between {lo} and {hi}"
            continue
        setattr(record, attr, val)

    for attr, (_col, choices) in EDIT_FIELDS_CATEGORICAL.items():
        raw = form.get(attr, "").strip()
        if raw == "":
            continue
        if raw not in choices:
            errors[attr] = f"must be one of {choices}"
            continue
        setattr(record, attr, raw)

    return errors


def _rescore_and_update_state(state, record) -> None:
    """Re-score a single record, update heap and df."""
    # Re-score
    state["predictor"].score_cohort(StudentCohort([record]))

    # Update heap (push handles in-place updates via stale-entry marking)
    state["heap"].push(
        record.student_id, record.risk_score or 0,
        metadata={"name": record.full_name,
                  "band": record.risk_band,
                  "department": record.department},
    )

    # Mirror edits into state['df'] so any downstream re-reads stay consistent.
    df = state["df"]
    idx = df.index[df["Student_ID"] == record.student_id]
    if len(idx):
        i = idx[0]
        for attr, (col, *_rest) in EDIT_FIELDS_NUMERIC.items():
            df.at[i, col] = getattr(record, attr)
        for attr, (col, _choices) in EDIT_FIELDS_CATEGORICAL.items():
            df.at[i, col] = getattr(record, attr)
        # Recompute engineered columns
        df.at[i, "early_academic_avg"] = (
            0.40 * (record.midterm or 0) + 0.25 * (record.assignments_avg or 0)
            + 0.20 * (record.quizzes_avg or 0) + 0.15 * (record.projects or 0))
        df.at[i, "engagement_index"] = max(0, min(15,
            (record.study_hours or 0) / 4
            - ((record.stress or 5) - 5) * 0.3
            + ((record.sleep or 7) - 6) * 0.4))
        df.at[i, "attendance_deficit"] = max(0.0, 75.0 - (record.attendance or 0))
        df.at[i, "low_att_low_mid"] = int(
            (record.attendance or 100) < 70 and (record.midterm or 100) < 50)


def _persist_edit_overlay(record) -> None:
    """
    Append the edit to a JSONL overlay file. On app startup, overlays
    are replayed so edits survive restarts without rewriting the raw CSV.
    """
    path = _edits_overlay_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "student_id": record.student_id,
        "timestamp": datetime.utcnow().isoformat(),
        "fields": {
            attr: getattr(record, attr)
            for attr in list(EDIT_FIELDS_NUMERIC) + list(EDIT_FIELDS_CATEGORICAL)
        },
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _replay_edits_on_cohort(state) -> int:
    """Replay persisted edits onto the freshly-loaded cohort. Returns count."""
    path = _edits_overlay_path()
    if not path.exists():
        return 0
    applied = 0
    # Later edits win — keep only the most recent per student.
    latest: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        try:
            e = json.loads(line)
            latest[e["student_id"]] = e
        except json.JSONDecodeError:
            continue
    for sid, edit in latest.items():
        if sid not in state["cohort"]:
            continue
        record = state["cohort"][sid]
        for attr, val in edit.get("fields", {}).items():
            if hasattr(record, attr):
                setattr(record, attr, val)
        _rescore_and_update_state(state, record)
        applied += 1
    return applied


@app.route("/students/<student_id>/edit", methods=["GET", "POST"])
def edit_student(student_id: str):
    state = _pipeline_state()
    if student_id not in state["cohort"]:
        flash(f"Student {student_id} not found.", "error")
        return redirect(url_for("students_list"))
    record = state["cohort"][student_id]

    if request.method == "POST":
        errors = _apply_edits_from_form(record, request.form)
        if errors:
            return render_template(
                "student_edit.html", r=record,
                numeric_fields=EDIT_FIELDS_NUMERIC,
                categorical_fields=EDIT_FIELDS_CATEGORICAL,
                errors=errors,
            )

        _rescore_and_update_state(state, record)
        _persist_edit_overlay(record)

        flash(
            f"Updated. New risk score: {record.risk_score:.3f} "
            f"(band: {record.risk_band})", "success",
        )
        return redirect(url_for("student_detail", student_id=student_id))

    return render_template(
        "student_edit.html", r=record,
        numeric_fields=EDIT_FIELDS_NUMERIC,
        categorical_fields=EDIT_FIELDS_CATEGORICAL,
        errors={},
    )


@app.route("/api/student/<student_id>/simulate", methods=["POST"])
def api_simulate(student_id: str):
    """
    What-if scoring. Takes a JSON payload of field overrides and returns the
    resulting risk score and band WITHOUT mutating or persisting anything.
    """
    state = _pipeline_state()
    if student_id not in state["cohort"]:
        return jsonify({"error": "not found"}), 404

    # Clone the record's dict, apply overrides, build a throwaway record.
    original = state["cohort"][student_id]
    snapshot = {slot.lstrip("_"): getattr(original, slot)
                for slot in StudentRecord.__slots__}

    overrides = request.get_json(silent=True) or {}
    for k, v in overrides.items():
        if k in snapshot and v not in (None, ""):
            # Coerce numerics sent as strings
            try:
                if isinstance(snapshot[k], (int, float)) or k in EDIT_FIELDS_NUMERIC:
                    caster = EDIT_FIELDS_NUMERIC.get(k, (None, None, None, float))[3]
                    snapshot[k] = caster(v)
                else:
                    snapshot[k] = v
            except (TypeError, ValueError):
                pass   # ignore bad values silently — simulate is best-effort

    # Re-hydrate into a StudentRecord using the CSV column names it expects.
    kwargs = {
        "Student_ID":                 original.student_id,
        "First_Name":                 original.first_name,
        "Last_Name":                  original.last_name,
        "Email":                      original.email,
        "Gender":                     snapshot["gender"],
        "Age":                        snapshot["age"],
        "Department":                 snapshot["department"],
        "Attendance (%)":             snapshot["attendance"],
        "Midterm_Score":              snapshot["midterm"],
        "Final_Score":                snapshot["final_score"],
        "Assignments_Avg":            snapshot["assignments_avg"],
        "Quizzes_Avg":                snapshot["quizzes_avg"],
        "Participation_Score":        snapshot["participation"],
        "Projects_Score":             snapshot["projects"],
        "Total_Score":                snapshot["total_score"],
        "Grade":                      snapshot["grade"],
        "Study_Hours_per_Week":       snapshot["study_hours"],
        "Extracurricular_Activities": snapshot["extracurricular"],
        "Internet_Access_at_Home":    snapshot["internet_access"],
        "Parent_Education_Level":     snapshot["parent_education"],
        "Family_Income_Level":        snapshot["family_income"],
        "Stress_Level (1-10)":        snapshot["stress"],
        "Sleep_Hours_per_Night":      snapshot["sleep"],
    }
    throwaway = StudentRecord(**kwargs)
    state["predictor"].score_cohort(StudentCohort([throwaway]))

    return jsonify({
        "risk_score": round(throwaway.risk_score or 0, 3),
        "risk_band":  throwaway.risk_band,
        "delta":      round((throwaway.risk_score or 0) - (original.risk_score or 0), 3),
    })


@app.route("/notifications")
def notifications_page():
    log_path = C.REPORTS_DIR / "notifications.jsonl"
    items: list[dict] = []
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    items.reverse()
    return render_template("notifications.html", items=items[:200])


# ---------------------------------------------------------------------------
# DSA Engine — standalone tools (no model dependency, pure UI / algorithms)
# ---------------------------------------------------------------------------
@app.route("/grade-calculator")
def grade_calculator():
    """Standalone UPES grade calculator — pure client-side math."""
    return render_template("grade_calculator.html")


@app.route("/dsa-visualizer")
def dsa_visualizer():
    """Interactive visualisations of the four DSA algorithms used in PAIS."""
    return render_template("dsa_visualizer.html")


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------
@app.route("/api/students")
def api_students():
    state = _pipeline_state()
    q = (request.args.get("q") or "").lower().strip()
    band = request.args.get("band") or ""
    department = request.args.get("department") or ""
    limit = int(request.args.get("limit", 200))
    offset = int(request.args.get("offset", 0))

    rows = []
    for r in state["cohort"]:
        if band and r.risk_band != band: continue
        if department and r.department != department: continue
        if q and (q not in r.full_name.lower() and q not in r.student_id.lower()):
            continue
        rows.append({
            "student_id": r.student_id, "name": r.full_name,
            "department": r.department,
            "attendance": round(r.attendance or 0, 1),
            "midterm": round(r.midterm or 0, 1),
            "grade": r.grade,
            "risk_score": round(r.risk_score or 0, 3),
            "risk_band": r.risk_band,
        })
    rows.sort(key=lambda x: -x["risk_score"])
    return jsonify({"total": len(rows), "rows": rows[offset:offset + limit]})


@app.route("/api/top-at-risk")
def api_top_at_risk():
    k = int(request.args.get("k", 10))
    state = _pipeline_state()
    top = state["heap"].peek_top(k)
    return jsonify([
        {"student_id": sid, "risk_score": round(score, 3),
         "name": meta.get("name"), "band": meta.get("band"),
         "department": meta.get("department")}
        for sid, score, meta in top
    ])


@app.route("/api/student/<student_id>")
def api_student(student_id: str):
    state = _pipeline_state()
    if student_id not in state["cohort"]:
        return jsonify({"error": "not found"}), 404
    return jsonify(_record_to_dict(state["cohort"][student_id]))


@app.route("/api/student/<student_id>/roadmap")
def api_student_roadmap(student_id: str):
    state = _pipeline_state()
    if student_id not in state["cohort"]:
        return jsonify({"error": "not found"}), 404
    r = state["cohort"][student_id]
    roadmap = state["optimizer"].full_roadmap(
        midterm=r.midterm or 0, assignments=r.assignments_avg or 0,
        quizzes=r.quizzes_avg or 0, projects=r.projects or 0)
    return jsonify([g.__dict__ for g in roadmap])


@app.route("/api/notify/<student_id>", methods=["POST"])
def api_notify(student_id: str):
    state = _pipeline_state()
    if student_id not in state["cohort"]:
        return jsonify({"error": "not found"}), 404
    r = state["cohort"][student_id]
    engine = NotificationEngine(
        dispatcher=JsonlDispatcher(C.REPORTS_DIR / "notifications.jsonl"))
    rec = build_recommendation_text(r, state["optimizer"])
    engine.notify_student(r, rec)
    notify_parent = request.json.get("notify_parent", False) if request.is_json else False
    if notify_parent:
        engine.notify_parent(r)
    return jsonify({"ok": True, "sent": len(engine.sent_log)})


@app.route("/api/stats")
def api_stats():
    state = _pipeline_state()
    return jsonify(_cohort_stats(state["cohort"]))


@app.route("/api/model-report")
def api_model_report():
    state = _pipeline_state()
    return jsonify(state["training_report"])


@app.route("/api/gradient-rankings")
def api_gradient():
    state = _pipeline_state()
    ranked = rank_by_gradient(state["df"])
    improvers = ranked.head(20).to_dict(orient="records")
    decliners = ranked.tail(20).iloc[::-1].to_dict(orient="records")
    return jsonify({"improvers": improvers, "decliners": decliners})


# ---------------------------------------------------------------------------
# Errors + globals
# ---------------------------------------------------------------------------
@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", code=404, message="Page not found"), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("error.html", code=500, message="Internal server error"), 500


@app.context_processor
def inject_globals():
    return {
        "current_year": datetime.now().year,
        "app_name": "PAIS",
        "app_tagline": "Proactive Academic Intervention System",
    }


if __name__ == "__main__":
    import os
    print("Warming pipeline cache...")
    _pipeline_state()
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"Ready. Open http://{host}:{port}")
    # debug=True only locally; Render sets PORT so this branch is safe.
    app.run(host=host, port=port, debug=(os.environ.get("FLASK_ENV") != "production"))
