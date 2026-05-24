# 🎓 Proactive Academic Intervention System (PAIS)

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey?logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn)
![Deployed on Render](https://img.shields.io/badge/Deployed-Render-46E3B7?logo=render)
![Static Mirror](https://img.shields.io/badge/Mirror-Netlify-00C7B7?logo=netlify)
![License](https://img.shields.io/badge/License-MIT-green)

> **Identify at-risk students 4–6 weeks before final exams — while there's still time to act.**

PAIS is a full-stack web application that combines a **hybrid ML risk scorer** with **six classical DSA modules** to give faculty an actionable, ranked view of their cohort. It scores 5,000 students end-to-end and exposes the results through a dark-themed Flask dashboard deployed live on Render.

---

## 🔴 The Problem

By the time end-semester results arrive, the intervention window is already closed. Most institutional dashboards report aggregate attendance percentages and cohort means — they don't rank students by predicted risk, catch sudden 2-week absence gaps, or tell a student what score they need on the final to recover a target grade.

PAIS fills that gap.

---

## ✅ Key Results

| Metric | Value |
|---|---|
| Model | Logistic Regression (selected over RF & GBM) |
| ROC-AUC | **0.887** |
| F1 Score | **0.764** |
| CV-F1 (5-fold) | 0.748 ± 0.017 |
| Accuracy | 79.8% |
| At-risk students identified | 1,089 / 5,000 (21.8%) |
| CRITICAL band | 331 students (6.6%) |
| Recall on at-risk class | **79.9%** |

---

## 🏗️ System Architecture

```
Raw CSV (5,000 students × 23 features)
        │
        ▼
 Feature Engineering (27 → 38 dims after one-hot)
        │
        ▼
 ML Pipeline (Logistic Regression, sklearn)
        │
        ▼
 Hybrid Risk Score = 0.70 × ML_prob + 0.30 × Rule_score
        │
        ▼
 Risk Bands: SAFE → LOW → MODERATE → HIGH → CRITICAL
        │
        ▼
 DSA Engine  ──────────────────────────────────────────┐
  ├── Max-Heap Priority Queue (top-k retrieval)         │
  ├── 7-Day Sliding Window (attendance anomaly)         │
  ├── Hash Map (O(1) student lookup)                    │
  ├── Bipartite Graph + BFS (engagement analysis)       │
  ├── Greedy Grade-Roadmap Planner                      │
  └── Stable Mergesort (improvement ranking)            │
        │                                               │
        ▼                                               │
 Flask Web App (12 routes, dark-theme UI) ◄─────────────┘
        │
        ▼
 Deployed: Render (live) + Netlify (static mirror)
```

---

## ⚙️ Tech Stack

| Layer | Technologies |
|---|---|
| **Backend** | Python 3.11, Flask 3.0 |
| **ML** | scikit-learn 1.3, pandas 2.1, numpy, joblib |
| **Frontend** | Jinja2, Vanilla JS, Chart.js, Custom CSS (dark theme) |
| **Testing** | pytest (11 DSA unit tests) |
| **Deployment** | Render (Flask app), Netlify (static mirror) |
| **Data** | Mahmoud Elhemaly's Students Grading Dataset (5,000 records, 23 features) |

---

## 🧠 ML Pipeline

### Feature Engineering (4 derived features)
- **`early_academic_avg`** — weighted blend of midterm (40%), assignments (25%), quizzes (20%), projects (15%)
- **`engagement_index`** — study hours, stress, and sleep combined into a single behavioural index
- **`attendance_deficit`** — piecewise penalty for falling below 75% (Indian university threshold)
- **`low_att_low_mid`** — binary flag when both attendance < 70% AND midterm < 50% simultaneously

### Hybrid Risk Score
```
risk_score = clip(0.70 × ML_probability + 0.30 × rule_score, 0, 1)
```
Rules cover: attendance < 70%, midterm < 50%, assignments + quizzes < 100.
The rule layer acts as a **safety net against ML overconfidence** — a student satisfying all three rules always gets a minimum risk score of 0.30.

### Model Comparison
| Model | F1 | ROC-AUC | CV-F1 |
|---|---|---|---|
| **Logistic Regression ✅** | **0.7635** | **0.8874** | 0.748 ± 0.017 |
| Random Forest | 0.7497 | 0.8737 | 0.732 ± 0.020 |
| Gradient Boosting | 0.7441 | 0.8814 | 0.725 ± 0.017 |

Logistic Regression selected for highest F1 — preferred over AUC because **missing an at-risk student is worse than a false positive**.

---

## 🔧 DSA Engine

| Module | Operation | Complexity |
|---|---|---|
| Max-Heap Priority Queue | `peek_top(k)` / `push` / `pop` | O(1) / O(log n) / O(log n) |
| Sliding Window | 7-day attendance scan per student | O(d) |
| Hash Map | Student lookup by ID | O(1) average |
| Bipartite Graph + BFS | Engagement reachability | O(V + E) |
| Greedy Grade Planner | Per-student remediation plan | O(g) |
| Stable Mergesort | Rank by improvement gradient | O(n log n) |

Each module is independently tested in `tests/test_dsa.py` (11 pytest assertions, full suite runs in 1.2s).

---

## 🖥️ Web Application

9 HTML routes + 4 JSON API endpoints:

| Page | What it does |
|---|---|
| **Dashboard** | 4 KPIs + risk band distribution + top-10 at-risk list |
| **Priority Queue** | Live max-heap binary tree visualisation |
| **Sliding Window** | 30-day animated attendance stream + anomaly alerts |
| **Hash Map** | Live O(1) student lookup with 150ms debounce |
| **Grade Calculator** | Single-student + CSV bulk roadmap planner |
| **Analytics** | Dept, gender, study-hours, grade distribution charts |
| **At-Risk Students** | Filterable, sortable list of all 5,000 students |
| **Model Page** | ROC curves, confusion matrix, metric summary |
| **Upload** | Score a new cohort CSV end-to-end |

---

## 📁 Project Structure

```
PAIS/
├── data/
│   ├── raw/students.csv              # 5,000 students × 23 cols
│   └── processed/                    # cached preprocessing artefacts
├── models/
│   └── risk_model.pkl                # trained logistic regression
├── src/
│   ├── preprocessing.py              # feature engineering
│   ├── ml/
│   │   ├── train.py                  # training orchestrator
│   │   └── evaluate.py               # evaluation figures
│   ├── dsa/
│   │   ├── priority_queue.py         # max-heap
│   │   ├── sliding_window.py         # attendance anomaly detector
│   │   ├── hash_aggregator.py        # O(1) lookup
│   │   ├── resource_graph.py         # engagement graph BFS
│   │   ├── grade_optimizer.py        # greedy roadmap planner
│   │   └── sorter.py                 # stable mergesort
├── tests/
│   └── test_dsa.py                   # 11 pytest assertions
├── webapp/
│   ├── app.py                        # Flask routes
│   ├── templates/                    # Jinja2 templates
│   └── static/                       # CSS + JS
├── netlify_site/                     # pre-rendered static mirror
├── render.yaml                       # Render deployment config
└── main.py                           # CLI training entry point
```

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/thechiragbatra/proactive-academic-intervention-system.git
cd proactive-academic-intervention-system/PAIS

# Install dependencies
pip install -r requirements.txt

# Train the model and generate all artefacts
python main.py

# Start the Flask app
python webapp/app.py
# → Open http://localhost:5000
```

---

## 🧪 Run Tests

```bash
pytest tests/test_dsa.py -v
# 11 tests, ~1.2s, all pass deterministically
```

---

## ⚠️ Limitations

- Daily attendance logs are **synthesised** from aggregate percentages (Bernoulli sampling with injected absence clusters). A real deployment would ingest LMS logs directly.
- The 40.8% at-risk rate in the source dataset is higher than real institutional rates (typically 5–15%). Band thresholds would need recalibration.
- Demographic features (family income, parent education, internet access) are included in the ML layer. Section 8.2 of the project report discusses the ethical implications and mitigation measures taken.

---

## 👥 Team

| Name | Roll No. |
|---|---|
| Chirag Batra | 500119174 |
| Shikhar Singh | 500121748 |
| Anjali Mamgain | — |

**Guide:** Dr. Khursheed Ahmad Bhat, Assistant Professor, School of Computer Science, UPES Dehradun

**Submitted:** May 2026 — B.Tech Computer Science & Engineering (Data Science), UPES Dehradun

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

> *"The contribution is not a novel learning algorithm. It is a complete end-to-end system that takes student data, produces an interpretable risk score, exposes it through complexity-appropriate algorithmic structures, and gives faculty a concrete plan of attention."*
