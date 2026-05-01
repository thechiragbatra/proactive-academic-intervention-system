/* PAIS — Frontend helpers */

if (window.Chart) {
  Chart.defaults.font.family = "'IBM Plex Sans', sans-serif";
  Chart.defaults.font.size = 12;
  Chart.defaults.color = '#b3b8c0';
  Chart.defaults.borderColor = '#232a36';
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
  Chart.defaults.plugins.legend.labels.boxWidth = 8;
  Chart.defaults.plugins.legend.labels.padding = 14;
  Chart.defaults.plugins.tooltip.padding = 10;
  Chart.defaults.plugins.tooltip.backgroundColor = '#0a0e13';
  Chart.defaults.plugins.tooltip.titleFont = { weight: '600' };
  Chart.defaults.plugins.tooltip.cornerRadius = 4;
}

const RISK_COLORS = {
  CRITICAL: '#ef4444',
  HIGH:     '#f59e0b',
  MODERATE: '#eab308',
  LOW:      '#84cc16',
  SAFE:     '#22c55e',
};

function fmtPct(x) {
  if (x == null) return '—';
  return (x * 100).toFixed(0) + '%';
}

function fmtScore(x, digits = 3) {
  if (x == null) return '—';
  return Number(x).toFixed(digits);
}

async function postJSON(url, body = {}) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return res.json();
}

async function getJSON(url) {
  const res = await fetch(url);
  return res.json();
}
