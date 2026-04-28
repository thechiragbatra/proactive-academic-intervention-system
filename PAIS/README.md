# PAIS — Proactive Academic Intervention System

> An ML-powered early-warning platform that predicts student exam failure
> 4–6 weeks in advance and generates personalised recovery plans.

**Minor Project · BTech Cloud Computing · UPES**
**Domain:** Data Science, Machine Learning, DSA, OOP
**Stack:** Python · scikit-learn · Flask · Chart.js

---

## What it does

Traditional academic portals are *descriptive* — they report marks after the
evaluation is over. **PAIS is prescriptive**: it ingests behavioural and
academic indicators, predicts exam failure risk early, ranks students who need
urgent intervention, and produces per-student recovery roadmaps telling each
student exactly what marks they need on remaining evaluations to hit their
target grade.

It's built as a full local web app — not a notebook — so mentors can actually
use it.

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full ML + DSA pipeline (preprocess → train → evaluate → rank)
python main.py

# 3. Start the web app
python webapp/app.py

# 4. Open http://127.0.0.1:5000
```

That's it. The pipeline trains three models, picks the best one, produces
figures, generates daily engagement logs, and writes everything the web app
needs to `models/` and `reports/`.

**If you only want the website** (skip retraining because the model is already
trained and checked in): just run step 3.

---

## Deploying to Render (free tier)

The project ships with `render.yaml`, `Procfile`, `runtime.txt`, and `wsgi.py`
already set up. To deploy:

1. **Push the project to GitHub**
   ```bash
   cd PAIS
   git init
   git add .
   git commit -m "Initial PAIS deploy"
   git branch -M main
   # Create an empty repo on GitHub called `PAIS`, then:
   git remote add origin https://github.com/YOUR_USERNAME/PAIS.git
   git push -u origin main
   ```

2. **Go to [render.com](https://render.com)** → sign in with GitHub → click
   **New → Blueprint** → select your PAIS repo → click **Apply**.

3. **Wait 3–5 minutes** for the first build. Render installs dependencies,
   runs `python main.py` to regenerate processed data, then starts gunicorn.

4. **Open the URL Render gives you** — something like
   `https://pais-xxxx.onrender.com`.

### Things to know about the free tier

- **Cold starts (~30 seconds):** the free plan spins the service down after 15
  minutes of inactivity. The first request after idle will take ~30s while the
  container wakes up and the pipeline cache warms. Subsequent requests are fast.
- **No persistent disk:** any edits saved via the edit page are stored in
  `reports/edits_overlay.jsonl` inside the container, which is wiped on every
  redeploy. For viva-day use this is fine; for real deployment you'd add a
  Render Disk (paid) or swap the overlay for a managed database.
- **512 MB RAM:** pandas + scikit-learn + the 5000-student cohort fit
  comfortably in this limit (~180 MB at peak).
- **Region:** `render.yaml` is set to `singapore` for low latency from India.
  Change to `oregon` or `frankfurt` in `render.yaml` if you prefer.

### For the viva

**Wake the app 60 seconds before the demo** by hitting the URL once — that
way the cold start is absorbed before the examiner sees it. Keep a local copy
running (`python webapp/app.py`) as a backup in case the Wi-Fi is flaky.

---

## Dataset

The project ships with a 5,000-row synthetic academic dataset in
`data/raw/students.csv`. Columns include:

- Academics: `Attendance (%)`, `Midterm_Score`, `Assignments_Avg`,
  `Quizzes_Avg`, `Participation_Score`, `Projects_Score`, `Grade`, `Total_Score`
- Behaviour: `Study_Hours_per_Week`, `Stress_Level (1-10)`, `Sleep_Hours_per_Night`
- Context: `Department`, `Parent_Education_Level`, `Family_Income_Level`,
  `Internet_Access_at_Home`, `Extracurricular_Activities`

### Target definition

A student is labelled `at_risk = 1` when their final Grade is **D or F** OR
their `Total_Score < 50`. This gives a ~40% positive class rate — balanced
enough for meaningful learning without being trivially imbalanced.

### Leakage prevention

`Final_Score`, `Total_Score`, and `Grade` are **explicitly excluded** from
training features. These *are* the outcome; using them would give a fake
accuracy. The model uses only features available 4–6 weeks before the final
(matching the synopsis's early-prediction requirement).

---

## Architecture

```
PAIS/
├── data/
│   ├── raw/students.csv                     # source dataset
│   └── processed/                           # pipeline outputs
│       ├── students_processed.csv
│       └── daily_engagement.csv             # synthesised 84-day log
│
├── src/                                     # ← all Python domain code
│   ├── config.py                            # thresholds, paths, weights
│   ├── preprocessing.py                     # cleaning + target + features
│   ├── daily_logs.py                        # daily log synthesis
│   │
│   ├── dsa/                                 # ← 6 DSA modules
│   │   ├── priority_queue.py                # max-heap (risk ranking)
│   │   ├── sliding_window.py                # O(d) attendance anomaly detect
│   │   ├── hash_aggregator.py               # O(1) student profile lookup
│   │   ├── resource_graph.py                # bipartite graph + BFS
│   │   ├── grade_optimizer.py               # greedy min-marks solver
│   │   └── sorter.py                        # stable O(n log n) gradient rank
│   │
│   ├── oop/                                 # ← OOP classes per synopsis
│   │   ├── student_record.py                # StudentRecord + StudentCohort
│   │   ├── risk_predictor.py                # RiskPredictor (ML + rules)
│   │   └── notification_engine.py           # NotificationEngine + Dispatchers
│   │
│   ├── ml/                                  # ← ML pipeline
│   │   ├── train.py                         # LogReg / RF / GB comparison
│   │   └── evaluate.py                      # CM, ROC, feature importance
│   │
│   └── business/
│       ├── recommendations.py               # personalised advice generator
│       └── alerts.py                        # AlertService façade
│
├── webapp/                                  # ← Flask web application
│   ├── app.py                               # 17 HTML + API routes
│   ├── templates/                           # Jinja2 templates
│   │   ├── base.html                        # sidebar shell, typography
│   │   ├── dashboard.html                   # KPIs + top-10 + dept chart
│   │   ├── students.html                    # filterable list
│   │   ├── student_detail.html              # drill-down + roadmap
│   │   ├── analytics.html                   # DSA analytics
│   │   ├── model.html                       # ML report
│   │   ├── upload.html                      # CSV batch scoring
│   │   ├── notifications.html               # alert log
│   │   └── error.html
│   └── static/
│       ├── css/main.css                     # editorial design system
│       └── js/main.js                       # Chart.js defaults
│
├── models/risk_model.pkl                    # trained winner
├── reports/
│   ├── training_report.json                 # metrics for all 3 candidates
│   ├── notifications.jsonl                  # dispatched alert log
│   ├── attendance_anomalies.csv             # sliding-window output
│   └── figures/                             # confusion matrix, ROC, etc.
│
├── tests/test_dsa.py                        # 11 unit tests (all pass)
├── main.py                                  # orchestrator
└── requirements.txt
```

---

## How each requirement is met

### ✅ ML Objectives (from synopsis)

| Requirement | Where in code |
|---|---|
| Predict exam failure with classification | `src/ml/train.py` — three models compared |
| Generate risk scores | `src/oop/risk_predictor.py` — blended ML + rule score |
| Identify decline trends | `src/dsa/sorter.py` — gradient ranking |
| Recommend interventions | `src/business/recommendations.py` |

**Winning model:** Logistic Regression, **F1 = 0.763, ROC AUC = 0.887, Accuracy = 0.80**
on a held-out 20% test split with 5-fold CV (0.748 ± 0.017).

### ✅ DSA — all six required concepts

| # | Concept | Module | What it actually does |
|---|---|---|---|
| 1 | **Priority Queue (Max-Heap)** | `dsa/priority_queue.py` | Ranks students by risk — `peek_top(k)` in O(k log n), `push` in O(log n) with stale-entry marking for in-place updates |
| 2 | **Sliding Window** | `dsa/sliding_window.py` | Detects 7-day attendance collapses using a running-sum pass — O(d) per student |
| 3 | **Hash Map / Dict** | `dsa/hash_aggregator.py` | `StudentHashIndex` — O(1) amortised profile lookup + log aggregation |
| 4 | **Graph Theory + BFS** | `dsa/resource_graph.py` | Bipartite student↔resource graph; BFS to find students unreachable from core resources |
| 5 | **Greedy Optimization** | `dsa/grade_optimizer.py` | Given remaining-evaluation weights, solves the constraint `earned + Σ(x_i·w_i) ≥ cutoff` |
| 6 | **Stable Sorting** | `dsa/sorter.py` | Timsort on `midterm − early_avg` gradient; surfaces top improvers / decliners |

All six are covered by `pytest tests/` — **11 tests, all pass**.

### ✅ OOP — per-synopsis classes with proper design

- **`StudentRecord`** (`src/oop/student_record.py`) — encapsulates identity (read-only via `@property`), academics, behaviour. Uses `__slots__` for memory efficiency. Derived properties (`attendance_risk`, `is_first_gen_college`).
- **`StudentCohort`** — keyed collection; iterable, indexable by ID, supports bulk filters (`cohort.by_department("CS")`, `cohort.at_risk()`).
- **`RiskPredictor`** — holds the trained pipeline, exposes both `ml_probability()` and `rule_based_score()`, blends them 70/30.
- **`NotificationEngine`** — decides who to notify (MODERATE+ after 50-mark audit), builds tailored messages. Polymorphism via `Dispatcher` base class with `ConsoleDispatcher`, `JsonlDispatcher`, `SMTPDispatcher` concrete implementations.

### ✅ Business workflow (synopsis "Prescriptive Intervention Workflow")

1. **Continuous monitoring** — daily logs + sliding-window anomaly detection run every load
2. **The 50-Mark Audit** — `NotificationEngine.should_notify()` triggers once `marks_reflected_pct >= 50`
3. **Grade targeting** — `GradeOptimizer.full_roadmap()` returns per-grade min-marks plan
4. **Dual-notification** — `batch_notify(..., notify_parents_for={"CRITICAL"})` sends both student and parent alerts for critical cases

---

## Web app — pages & routes

| Page | Route | What's on it |
|---|---|---|
| Dashboard | `/` | KPIs, risk-band distribution chart, top-10 heap peek, department risk chart, system snapshot |
| Students | `/students` | Full roster table, filter by band/department, search by name/ID |
| Student detail | `/students/<id>` | Academic vitals, 84-day engagement chart, grade roadmap, behavioural signals, personalised recommendation, notify buttons |
| **Edit student** | `/students/<id>/edit` | **Edit any field with a LIVE what-if risk preview — as you drag a slider or type a value, the risk score recalculates on every keystroke. Perfect for showing "if this student raises their midterm to 65, they drop from CRITICAL to MODERATE". Changes persist across restarts via `reports/edits_overlay.jsonl`.** |
| Analytics | `/analytics` | Top improvers & decliners, sliding-window anomalies, graph stats |
| Model | `/model` | 3-way candidate comparison, confusion matrix, ROC, feature importance, classification report |
| Score new cohort | `/upload` | CSV upload → batch predict → downloadable scored CSV |
| Notifications | `/notifications` | Audit log of every alert dispatched |

**JSON API** endpoints also exposed (`/api/*`) for integrations.

---

## Design philosophy

- **Editorial, not SaaS.** Warm off-white background (#faf8f3), serif display
  type (Fraunces), careful use of color. Risk severity shown through a
  calibrated palette (oxblood / burnt amber / mustard / olive / forest) rather
  than the generic red/yellow/green you see everywhere.
- **Interpretable over clever.** The ML model is blended with a transparent
  rule-based score so mentors can explain *why* a student was flagged —
  crucial for trust in an educational setting.
- **Leakage-safe by construction.** Leakage-prone columns are hard-coded in
  `config.LEAKAGE_COLUMNS` and filtered out in preprocessing, so the "4–6 weeks
  ahead" claim is actually true.

---

## Viva talking points

1. **Problem framing** — "Current portals show marks after the fact. PAIS flags
   the student while there's still time to act."
2. **Why F1 and not accuracy** — accuracy would be misleading on a 40% positive
   class; F1 balances precision (don't cry wolf on healthy students) and
   recall (don't miss anyone at risk).
3. **Why blend ML + rules** — ML catches subtle patterns; rules give mentors
   something they can read. The 70/30 blend is a product decision not a
   technical one.
4. **The bipartite graph is novel** — framing student↔resource as a graph lets
   us use BFS to find truly isolated students (0 or 1 resource edges) rather
   than just those with low engagement numbers. This opens the door to future
   work on resource recommendation via collaborative filtering on the same
   graph.
5. **The greedy optimizer is honest** — it tells a student "target A+ is out
   of reach" instead of pretending everything's achievable. That's a real
   product decision made ethically.
6. **Sliding-window vs total percentage** — a student at 85% overall can still
   have a 7-day blackout that predicts disengagement. Windows catch sudden
   drops that totals hide.
7. **Scalability claim** — priority queue operations are O(log n), hash index
   is O(1), graph construction O(E). System handles millions of students
   without architectural change.

---

## Deploying to Render.com

PAIS ships with a Render blueprint. In ~10 minutes you can have a public URL
to share with your mentor / put on your CV. See [`DEPLOY.md`](DEPLOY.md) for
the step-by-step guide.

The short version:
1. Push this repo to GitHub (free)
2. Go to https://dashboard.render.com → **New → Blueprint**
3. Point it at your repo → click **Apply**
4. Wait ~5 minutes. Done.

Free tier gives you `$0/month` with the tradeoff that the service sleeps
after 15 min of idle. Warm it up 2 minutes before your viva by opening the
URL and waiting for the first load.

---

## Author

**Chirag Yadav** · SAP 500122577
BTech Cloud Computing, UPES Bidholi (Dehradun)

---

## License

Academic / educational use. Dataset and code ship together for reproducibility.
