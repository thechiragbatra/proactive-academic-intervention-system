/* PAIS — Frontend helpers */

if (window.Chart) {
  Chart.defaults.font.family = "'IBM Plex Sans', sans-serif";
  Chart.defaults.font.size = 12;
  Chart.defaults.color = '#4b5057';
  Chart.defaults.borderColor = '#e5e1d6';
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
  Chart.defaults.plugins.legend.labels.boxWidth = 8;
  Chart.defaults.plugins.legend.labels.padding = 14;
  Chart.defaults.plugins.tooltip.padding = 10;
  Chart.defaults.plugins.tooltip.backgroundColor = '#1a1d23';
  Chart.defaults.plugins.tooltip.titleFont = { weight: '600' };
  Chart.defaults.plugins.tooltip.cornerRadius = 4;
}

const RISK_COLORS = {
  CRITICAL: '#8b1e1e',
  HIGH:     '#b25000',
  MODERATE: '#8b6f00',
  LOW:      '#4a6b3d',
  SAFE:     '#2d5a3d',
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
