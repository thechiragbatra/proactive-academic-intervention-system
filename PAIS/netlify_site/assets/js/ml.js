/* ============================================================================
   PAIS — Client-side ML inference
   ----------------------------------------------------------------------------
   Replicates the sklearn pipeline from data/model.json using pure JavaScript:
     1. SimpleImputer(median) -> StandardScaler for numeric columns
     2. SimpleImputer(mode)   -> OneHotEncoder for categorical columns
     3. Logistic regression: sigmoid(W·x + b)
   Validated against sklearn to ~1e-16 absolute error.
   ============================================================================ */

class PAISModel {
  constructor(spec) { this.spec = spec; }

  /** Engineered features that mirror src/preprocessing.py::_engineer(). */
  static engineer(row) {
    const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
    const attendance      = Number(row['Attendance (%)']         ?? 0);
    const midterm         = Number(row['Midterm_Score']          ?? 0);
    const assignments_avg = Number(row['Assignments_Avg']        ?? 0);
    const quizzes_avg     = Number(row['Quizzes_Avg']            ?? 0);
    const projects        = Number(row['Projects_Score']         ?? 0);
    const study_hours     = Number(row['Study_Hours_per_Week']   ?? 0);
    const stress          = Number(row['Stress_Level (1-10)']    ?? 5);
    const sleep           = Number(row['Sleep_Hours_per_Night']  ?? 7);

    return {
      early_academic_avg:
        0.40 * midterm + 0.25 * assignments_avg
        + 0.20 * quizzes_avg + 0.15 * projects,
      engagement_index: clamp(
        clamp(study_hours, 0, 40) / 4
        - (stress - 5) * 0.3
        + (sleep - 6) * 0.4, 0, 15),
      attendance_deficit: Math.max(0, 75.0 - attendance),
      low_att_low_mid: (attendance < 70 && midterm < 50) ? 1 : 0,
    };
  }

  /** Translate snake_case form fields to the CSV column names sklearn learned. */
  static normaliseKeys(row) {
    const out = Object.assign({}, row);
    const map = {
      attendance: 'Attendance (%)', midterm: 'Midterm_Score',
      assignments_avg: 'Assignments_Avg', quizzes_avg: 'Quizzes_Avg',
      participation: 'Participation_Score', projects: 'Projects_Score',
      study_hours: 'Study_Hours_per_Week', stress: 'Stress_Level (1-10)',
      sleep: 'Sleep_Hours_per_Night', age: 'Age', gender: 'Gender',
      department: 'Department', extracurricular: 'Extracurricular_Activities',
      internet_access: 'Internet_Access_at_Home',
      parent_education: 'Parent_Education_Level',
      family_income: 'Family_Income_Level',
    };
    for (const [k, v] of Object.entries(map)) {
      if (k in row && !(v in row)) out[v] = row[k];
    }
    return out;
  }

  featurise(rawRow) {
    const row = Object.assign({}, rawRow);
    const eng = PAISModel.engineer(row);
    for (const k of Object.keys(eng)) {
      if (row[k] == null) row[k] = eng[k];
    }
    const num = this.spec.numeric;
    const cat = this.spec.categorical;
    const features = [];
    for (let i = 0; i < num.columns.length; i++) {
      const col = num.columns[i];
      let v = row[col];
      if (v == null || v === '' || Number.isNaN(Number(v))) v = num.impute[i];
      features.push((Number(v) - num.mean[i]) / num.scale[i]);
    }
    for (let i = 0; i < cat.columns.length; i++) {
      const col = cat.columns[i];
      let v = row[col];
      if (v == null || v === '') v = cat.impute[i];
      for (const c of cat.categories[i]) {
        features.push(String(v) === String(c) ? 1.0 : 0.0);
      }
    }
    return features;
  }

  predictProba(rawRow) {
    const features = this.featurise(PAISModel.normaliseKeys(rawRow));
    let logit = this.spec.model.intercept;
    for (let i = 0; i < features.length; i++) {
      logit += features[i] * this.spec.model.coef[i];
    }
    return 1.0 / (1.0 + Math.exp(-logit));
  }

  riskBand(prob) {
    for (const { threshold, label } of this.spec.bands) {
      if (prob >= threshold) return label;
    }
    return 'SAFE';
  }

  score(rawRow) {
    const prob = this.predictProba(rawRow);
    return { risk_score: prob, risk_band: this.riskBand(prob) };
  }
}

async function loadPAISModel(url = 'data/model.json') {
  const spec = await fetchJSON(url);
  return new PAISModel(spec);
}
