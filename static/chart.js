let forecastChart, performanceChart, importanceChart;

function renderFXCharts() {

  if (forecastChart) forecastChart.destroy();
  if (performanceChart) performanceChart.destroy();
  if (importanceChart) importanceChart.destroy();

  const forecastCtx = document.getElementById("forecastChart").getContext("2d");
  const perfCtx = document.getElementById("performanceChart").getContext("2d");
  const impCtx = document.getElementById("importanceChart").getContext("2d");

  fetch("fx_forecast_result.json")
    .then(res => res.json())
    .then(data => {

      // =========================
      // FORECAST QUANTILE CHART
      // =========================
      const months = data.forecast.map(f => f.month);

      forecastChart = new Chart(forecastCtx, {
        type: "line",
        data: {
          labels: months,
          datasets: [
            {
              label: "Median (P50)",
              data: data.forecast.map(f => f.P50),
              borderColor: "rgba(54,162,235,1)",
              backgroundColor: "rgba(54,162,235,0.2)",
              borderWidth: 3,
              tension: 0.3
            },
            {
              label: "P10",
              data: data.forecast.map(f => f.P10),
              borderColor: "rgba(255,159,64,1)",
              backgroundColor: "rgba(255,159,64,0.2)",
              borderWidth: 2,
              borderDash: [5,5],
              tension: 0.3
            },
            {
              label: "P90",
              data: data.forecast.map(f => f.P90),
              borderColor: "rgba(75,192,192,1)",
              backgroundColor: "rgba(75,192,192,0.2)",
              borderWidth: 2,
              borderDash: [5,5],
              tension: 0.3
            }
          ]
        },
        options: {
          responsive: true,
          plugins: {
            title: {
              display: true,
              text: "FX Demand Forecast (Quantiles)"
            }
          }
        }
      });

      // =========================
      // MODEL PERFORMANCE
      // =========================
      performanceChart = new Chart(perfCtx, {
        type: "bar",
        data: {
          labels: ["Baseline", "Initial Model", "Tuned Model"],
          datasets: [{
            label: "MAE",
            data: [
              data.performance.baseline_mae,
              data.performance.initial_mae,
              data.performance.tuned_mae
            ],
            backgroundColor: [
              "rgba(255,99,132,0.5)",
              "rgba(54,162,235,0.5)",
              "rgba(75,192,192,0.5)"
            ],
            borderColor: [
              "rgba(255,99,132,1)",
              "rgba(54,162,235,1)",
              "rgba(75,192,192,1)"
            ],
            borderWidth: 2
          }]
        },
        options: {
          responsive: true,
          plugins: {
            title: {
              display: true,
              text: "Model Performance (MAE)"
            }
          }
        }
      });

      // =========================
      // VARIABLE IMPORTANCE
      // =========================

      const topVars = data.variable_importance
        .slice(0, 15); // show top 15

      importanceChart = new Chart(impCtx, {
        type: "bar",
        data: {
          labels: topVars.map(v => v.variable),
          datasets: [{
            label: "Importance",
            data: topVars.map(v => v.importance),
            backgroundColor: "rgba(54,162,235,0.4)",
            borderColor: "rgba(54,162,235,1)",
            borderWidth: 2
          }]
        },
        options: {
          indexAxis: "y",
          responsive: true,
          plugins: {
            title: {
              display: true,
              text: "Top Variable Importance (TFT)"
            }
          }
        }
      });

    });
}
