/* ============================================================================
   PAIS — shared utilities (sidebar, fetchers, formatters, storage)
   ============================================================================
   Central JS module loaded by every page. Provides:
     - renderSidebar()  : renders left navigation with DSA Engine section
     - DATA.*           : cached fetchers for static JSON files
     - bandBadge / gradeBadge / fmtScore / fmtPct : formatters
     - readStore / writeStore + STORE_EDITS / STORE_NOTIFS : localStorage
     - applyEditsTo / applyEditsToSummaries : merge edits into fetched data
     - flash()          : toast notifications
     - initChartDefaults() : Chart.js theming
     - RISK_COLORS      : palette
   ============================================================================ */

const RISK_COLORS = {
  CRITICAL: '#8b1e1e', HIGH: '#b25000', MODERATE: '#8b6f00',
  LOW: '#4a6b3d',      SAFE: '#2d5a3d',
};

/* Navigation is split into 3 logical groups in the sidebar.
   Each item has a `match` regex to detect when it's the active page. */
const NAV_ITEMS = [
  { label: 'Dashboard',         href: 'index.html',           match: /(^\/?$|\/index\.html$)/ },
  { label: 'Students',          href: 'students.html',        match: /\/(students|student|edit)\.html/ },
  { label: 'Analytics',         href: 'analytics.html',       match: /\/analytics\.html/ },
  { label: 'ML Model',          href: 'model.html',           match: /\/model\.html/ },
  { label: 'Score New Cohort',  href: 'upload.html',          match: /\/upload\.html/ },
  { label: 'Notifications',     href: 'notifications.html',   match: /\/notifications\.html/ },
  // DSA Engine group — new in this release
  { label: 'Algorithm Visualizer', href: 'dsa-visualizer.html', match: /\/dsa-visualizer\.html/ },
  { label: 'Grade Calculator',     href: 'grade-calculator.html', match: /\/grade-calculator\.html/ },
];

function renderSidebar(containerId = 'sidebar') {
  const el = document.getElementById(containerId);
  if (!el) return;
  const path = window.location.pathname;
  // Three groups: Overview (analytics), System (model + ops), DSA Engine (new)
  const groups = [
    { name: 'Overview',   items: ['Dashboard', 'Students', 'Analytics'] },
    { name: 'System',     items: ['ML Model', 'Score New Cohort', 'Notifications'] },
    { name: 'DSA Engine', items: ['Algorithm Visualizer', 'Grade Calculator'] },
  ];
  el.innerHTML = `
    <div class="brand">
      <div class="brand-mark">P</div>
      <div>
        <div class="brand-name">PAIS</div>
        <div class="brand-sub">Academic Intervention</div>
      </div>
    </div>
    <nav class="nav">
      ${groups.map(g => `
        <div class="nav-group">${g.name}</div>
        ${g.items.map(label => {
          const item = NAV_ITEMS.find(i => i.label === label);
          const active = item.match.test(path);
          return `<a href="${item.href}" class="nav-link${active ? ' active' : ''}">${label}</a>`;
        }).join('')}
      `).join('')}
    </nav>
    <div class="sidebar-footer">
      <div class="footer-line footer-author">Chirag Batra</div>
      <div class="footer-line footer-line-muted">PAIS · v1.0</div>
    </div>
  `;
}

/* ---------- JSON fetchers with simple in-memory cache ---------- */
const _cache = new Map();
async function fetchJSON(url) {
  if (_cache.has(url)) return _cache.get(url);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Fetch failed: ${url} (${res.status})`);
  const data = await res.json();
  _cache.set(url, data);
  return data;
}

const DATA = {
  stats:      () => fetchJSON('data/stats.json'),
  students:   () => fetchJSON('data/students.json'),
  anomalies:  () => fetchJSON('data/anomalies.json'),
  gradient:   () => fetchJSON('data/gradient.json'),
  model:      () => fetchJSON('data/model.json'),
  report:     () => fetchJSON('data/model_report.json'),
  student: (id) => fetchJSON(`data/student/${id}.json`),
};

/* ---------- Formatters ---------- */
function fmtScore(x, d = 3) { return x == null ? '—' : Number(x).toFixed(d); }
function fmtPct(x, d = 1)   { return x == null ? '—' : Number(x).toFixed(d) + '%'; }
function bandBadge(band)    { return `<span class="badge badge-${band}">${band}</span>`; }
function gradeBadge(grade)  { return `<span class="grade-badge grade-${grade || 'F'}">${grade || '—'}</span>`; }

function initChartDefaults() {
  if (!window.Chart) return;
  Chart.defaults.font.family = "'IBM Plex Sans', sans-serif";
  Chart.defaults.font.size = 12;
  Chart.defaults.color = '#4b5057';
  Chart.defaults.borderColor = '#e5e1d6';
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
  Chart.defaults.plugins.legend.labels.boxWidth = 8;
  Chart.defaults.plugins.legend.labels.padding = 14;
  Chart.defaults.plugins.tooltip.padding = 10;
  Chart.defaults.plugins.tooltip.backgroundColor = '#1a1d23';
  Chart.defaults.plugins.tooltip.cornerRadius = 4;
}

/* ---------- Toast / flash messages ---------- */
function flash(message, category = 'message') {
  let el = document.querySelector('.flash-stack');
  if (!el) {
    el = document.createElement('div');
    el.className = 'flash-stack';
    const main = document.querySelector('.main');
    if (main) main.insertBefore(el, main.firstChild);
  }
  const item = document.createElement('div');
  item.className = `flash flash-${category}`;
  item.textContent = message;
  el.appendChild(item);
  setTimeout(() => item.remove(), 5000);
}

/* ---------- localStorage-backed edit & notification persistence ---------- */
const STORE_EDITS  = 'pais_edits_v1';
const STORE_NOTIFS = 'pais_notifs_v1';

function readStore(key) {
  try { return JSON.parse(localStorage.getItem(key) || '{}'); }
  catch { return {}; }
}
function writeStore(key, data) {
  localStorage.setItem(key, JSON.stringify(data));
}

function applyEditsTo(detail) {
  const e = readStore(STORE_EDITS)[detail.student_id];
  if (!e) return detail;
  for (const [k, v] of Object.entries(e.fields || {})) {
    if (k in detail) detail[k] = v;
  }
  if (e.risk_score != null) detail.risk_score = e.risk_score;
  if (e.risk_band)          detail.risk_band  = e.risk_band;
  return detail;
}

function applyEditsToSummaries(rows) {
  const edits = readStore(STORE_EDITS);
  for (const r of rows) {
    const e = edits[r.student_id];
    if (!e) continue;
    for (const [k, v] of Object.entries(e.fields || {})) {
      if (k in r) r[k] = v;
    }
    if (e.risk_score != null) r.risk_score = e.risk_score;
    if (e.risk_band)          r.risk_band  = e.risk_band;
  }
  return rows;
}

document.addEventListener('DOMContentLoaded', () => {
  renderSidebar('sidebar');
  initChartDefaults();
});
