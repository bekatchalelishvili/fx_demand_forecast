// ══════════════════════════════════════════════════════════════════════════════
//  NAVIGATION
// ══════════════════════════════════════════════════════════════════════════════

function goBack() {
  window.location.href = "web.html";
}

function goAboutUs() {
  window.location.href = "about.html";
}

function goMethodology() {
  window.location.href = "methodology.html";
}

// In {ABOUT US} layouts are invisible unless a button is clicked
function showLayout(layoutId) {
  document.querySelectorAll('.layout').forEach(layout => {
    layout.style.display = 'none';
  });
  document.getElementById(layoutId).style.display = 'block';
  if (layoutId === 'top5') {
    renderTop5Charts();
  }
}


// ══════════════════════════════════════════════════════════════════════════════
//  PIPELINE RUNNER
//  Spawns the Python pipeline via a small backend endpoint (POST /run-pipeline).
//  The endpoint should shell-out to:  python pipeline.py
//  and return { "status": "ok" } when done.
//
//  If you have no backend, replace the fetch() call with a note to the user
//  that the pipeline must be run manually from the terminal.
// ══════════════════════════════════════════════════════════════════════════════

/**
 * runPipeline()
 * Triggers the Python TFT pipeline and writes fx_forecast_results.json.
 * Attaches to any element with id="run-pipeline-btn".
 */
async function runPipeline() {
  const btn        = document.getElementById("run-pipeline-btn");
  const statusEl   = document.getElementById("pipeline-status");

  if (btn)      btn.disabled = true;
  if (statusEl) statusEl.innerHTML ='Running TFT model... <span class="spinner"></span>';

  try {
    const response = await fetch("/run-pipeline", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
    });

    if (!response.ok) throw new Error(`Server returned ${response.status}`);

    const data = await response.json();

    if (data.status === "ok") {
      if (statusEl) statusEl.textContent = "✅  Pipeline complete. Loading results…";
      await loadForecastChart();                   // auto-refresh chart
    } else {
      throw new Error(data.message || "Unknown error from server");
    }

  } catch (err) {
    console.error("Pipeline run failed:", err);
    if (statusEl) {
      statusEl.textContent =
        "❌  Pipeline error: " + err.message +
        " — run  python pipeline.py  manually then refresh.";
    }
  } finally {
    if (btn) btn.disabled = false;
  }
}


// ══════════════════════════════════════════════════════════════════════════════
//  FORECAST CHART LOADER
//  Reads fx_forecast_results.json and renders:
//    1. A quantile fan chart  (canvas id="forecast-chart")
//    2. Model performance KPI cards  (container id="perf-cards")
//    3. Variable importance bar chart (canvas id="importance-chart")
// ══════════════════════════════════════════════════════════════════════════════

/**
 * loadForecastChart()
 * Fetches fx_forecast_results.json and draws all dashboard elements.
 * Safe to call multiple times — destroys existing Chart.js instances first.
 */
async function loadForecastChart() {
  const statusEl = document.getElementById("pipeline-status");

  try {
    // Cache-bust so we always get the freshest file after a pipeline run
    const res  = await fetch("/static/fx_forecast_results.json?t=" + Date.now());
    if (!res.ok) throw new Error("/static/fx_forecast_results.json not found");
    const data = await res.json();

    _renderPerformanceCards(data.performance, data.meta);
    _renderForecastChart(data.forecast);
    _renderImportanceChart(data.variable_importance);
    _renderHyperparams(data.tuned_hyperparameters);

    if (statusEl) statusEl.textContent = "✅  Results loaded successfully.";

  } catch (err) {
    console.error("loadForecastChart error:", err);
    if (statusEl) {
      statusEl.textContent =
        "⚠️  Could not load results: " + err.message +
        " — run the pipeline first.";
    }
  }
}


// ──────────────────────────────────────────────────────────────────────────────
//  INTERNAL RENDERERS
// ──────────────────────────────────────────────────────────────────────────────

// Track Chart.js instances so we can destroy before re-render
const _chartInstances = {};

/**
 * _renderPerformanceCards(perf, meta)
 * Fills #perf-cards with KPI tiles.
 */
function _renderPerformanceCards(perf, meta) {
  const container = document.getElementById("perf-cards");
  if (!container) return;

  const beatsColor  = perf.beats_baseline  ? "#22c55e" : "#ef4444";
  const improvedColor = perf.improved_vs_initial ? "#22c55e" : "#f59e0b";
  const diffLabel   = perf.beats_baseline
    ? `${Math.abs(perf.mae_vs_baseline_pct).toFixed(1)}% better than baseline`
    : `${Math.abs(perf.mae_vs_baseline_pct).toFixed(1)}% worse than baseline`;

  container.innerHTML = `
    <div class="kpi-card">
      <span class="kpi-label">Baseline MAE</span>
      <span class="kpi-value">${perf.baseline_mae.toFixed(2)}</span>
      <span class="kpi-sub">Naïve (last-value) forecast</span>
    </div>
    <div class="kpi-card">
      <span class="kpi-label">Initial MAE</span>
      <span class="kpi-value">${perf.initial_mae.toFixed(2)}</span>
      <span class="kpi-sub">Default params</span>
    </div>
    <div class="kpi-card">
      <span class="kpi-label">Tuned MAE</span>
      <span class="kpi-value" style="color:${beatsColor}">${perf.tuned_mae.toFixed(2)}</span>
      <span class="kpi-sub">${diffLabel}</span>
    </div>
    <div class="kpi-card">
      <span class="kpi-label">Beats Baseline</span>
      <span class="kpi-value" style="color:${beatsColor}">
        ${perf.beats_baseline ? "✅ Yes" : "❌ No"}
      </span>
      <span class="kpi-sub">Horizon: ${meta.prediction_horizon} months</span>
    </div>
    <div class="kpi-card">
      <span class="kpi-label">vs Initial</span>
      <span class="kpi-value" style="color:${improvedColor}">
        ${perf.improved_vs_initial ? "✅ Improved" : "⚠️ No gain"}
      </span>
      <span class="kpi-sub">After Optuna (${meta.optuna_trials} trials)</span>
    </div>
  `;
}


/**
 * _renderForecastChart(forecastRows)
 * Draws a quantile fan chart on canvas#forecast-chart.
 * Requires Chart.js ≥ 3 loaded in the page.
 */
function _renderForecastChart(forecastRows) {
  const canvas = document.getElementById("forecast-chart");
  if (!canvas) return;

  if (_chartInstances.forecast) {
    _chartInstances.forecast.destroy();
  }

  const labels = forecastRows.map(r => r.month);
  const get    = key => forecastRows.map(r => r[key]);

  const ctx = canvas.getContext("2d");

  _chartInstances.forecast = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        // P2–P98 outer band
        {
          label:           "P98",
          data:            get("P98"),
          borderColor:     "rgba(99,179,237,0.0)",
          backgroundColor: "rgba(99,179,237,0.12)",
          fill:            "+5",   // fill to P2 dataset index below
          pointRadius:     0,
          tension:         0.4,
        },
        // P10–P90 inner band
        {
          label:           "P90",
          data:            get("P90"),
          borderColor:     "rgba(99,179,237,0.35)",
          backgroundColor: "rgba(99,179,237,0.18)",
          borderDash:      [4, 3],
          fill:            "+3",   // fill to P10
          pointRadius:     3,
          tension:         0.4,
        },
        // P25–P75 core band
        {
          label:           "P75",
          data:            get("P75"),
          borderColor:     "rgba(56,161,105,0.45)",
          backgroundColor: "rgba(56,161,105,0.22)",
          borderDash:      [3, 2],
          fill:            "+2",   // fill to P25
          pointRadius:     3,
          tension:         0.4,
        },
        // P50 median — primary line
        {
          label:           "P50 (Median)",
          data:            get("P50"),
          borderColor:     "#2b6cb0",
          backgroundColor: "#2b6cb0",
          borderWidth:     2.5,
          fill:            false,
          pointRadius:     5,
          pointHoverRadius: 7,
          tension:         0.4,
        },
        // P25 lower core
        {
          label:           "P25",
          data:            get("P25"),
          borderColor:     "rgba(56,161,105,0.45)",
          backgroundColor: "transparent",
          borderDash:      [3, 2],
          fill:            false,
          pointRadius:     3,
          tension:         0.4,
        },
        // P10 lower inner
        {
          label:           "P10",
          data:            get("P10"),
          borderColor:     "rgba(99,179,237,0.35)",
          backgroundColor: "transparent",
          borderDash:      [4, 3],
          fill:            false,
          pointRadius:     3,
          tension:         0.4,
        },
        // P2 outer lower
        {
          label:           "P2",
          data:            get("P2"),
          borderColor:     "rgba(99,179,237,0.0)",
          backgroundColor: "transparent",
          fill:            false,
          pointRadius:     0,
          tension:         0.4,
        },
      ],
    },
    options: {
      responsive:          true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          labels: {
            filter: item => !["P2", "P98"].includes(item.text),
          },
        },
        title: {
          display: true,
          text:    "FX Demand Forecast — Quantile Fan (USD M)",
          font:    { size: 15, weight: "600" },
        },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)} M`,
          },
        },
      },
      scales: {
        x: { grid: { color: "rgba(0,0,0,0.06)" } },
        y: {
          grid:  { color: "rgba(0,0,0,0.06)" },
          title: { display: true, text: "USD Million" },
        },
      },
    },
  });
}


/**
 * _renderImportanceChart(importanceRows)
 * Draws a horizontal bar chart on canvas#importance-chart.
 * Shows top 20 variables, colour-coded by encoder vs decoder.
 */
function _renderImportanceChart(importanceRows) {
  const canvas = document.getElementById("importance-chart");
  if (!canvas) return;

  if (_chartInstances.importance) {
    _chartInstances.importance.destroy();
  }

  // Top 20 by importance
  const top = importanceRows
    .slice()
    .sort((a, b) => b.importance - a.importance)
    .slice(0, 20);

  const labels = top.map(r => r.variable);
  const values = top.map(r => r.importance);
  const colors = top.map(r =>
    r.suggest_drop
      ? "rgba(239,68,68,0.65)"
      : r.type === "decoder (known)"
        ? "rgba(56,161,105,0.75)"
        : "rgba(66,153,225,0.75)"
  );

  const ctx = canvas.getContext("2d");

  _chartInstances.importance = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label:           "Importance Score",
        data:            values,
        backgroundColor: colors,
        borderRadius:    4,
      }],
    },
    options: {
      indexAxis:           "y",
      responsive:          true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        title: {
          display: true,
          text:    "Variable Importance — Top 20  (🟢 decoder · 🔵 encoder · 🔴 suggested drop)",
          font:    { size: 14, weight: "600" },
        },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.parsed.x.toFixed(4)}`,
          },
        },
      },
      scales: {
        x: {
          title: { display: true, text: "Importance (sum-reduced attention)" },
          grid:  { color: "rgba(0,0,0,0.06)" },
        },
        y: {
          ticks: { font: { size: 11 } },
          grid:  { display: false },
        },
      },
    },
  });
}


/**
 * _renderHyperparams(params)
 * Fills #hyperparam-table with a simple key/value table.
 */
function _renderHyperparams(params) {
  const container = document.getElementById("hyperparam-table");
  if (!container || !params) return;

  const rows = Object.entries(params)
    .map(([k, v]) => `
      <tr>
        <td class="hp-key">${k.replace(/_/g, " ")}</td>
        <td class="hp-val">${typeof v === "number" ? v.toFixed(4) : v}</td>
      </tr>`)
    .join("");

  container.innerHTML = `
    <table class="hp-table">
      <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}


// ══════════════════════════════════════════════════════════════════════════════
//  AUTO-LOAD on page ready
//  Attempts to load existing results as soon as the page opens.
//  If the JSON doesn't exist yet it fails silently (status message shown).
// ══════════════════════════════════════════════════════════════════════════════
document.addEventListener("DOMContentLoaded", () => {
  loadForecastChart();
});

