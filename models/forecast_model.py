# forecast_model.py
import numpy as np
import pandas as pd

# your helper/model imports (make sure these are accessible on PYTHONPATH / same folder)
from get_data import get_revenue
from algorithms.cagr_meth import cagr_forecast
from algorithms.average_meth import average_forecast
from algorithms.arima_meth import arima_forecast
from algorithms.movingavg_meth import moving_average_forecast
from algorithms.gbm_meth import gbm_forecast
from algorithms.exposmoth_meth import exp_smoothing_forecast
from algorithms.holtwint_meth import holt_winters_forecast
from algorithms.fourier_meth import fourier_forecast
from algorithms.gru_meth import gru_forecast
from algorithms.randfor_meth import random_forest_forecast
from algorithms.dump_holt import damped_trend_forecast
from algorithms.prophet_meth import prophet_forecast
from algorithms.gaussian_process import gp_forecast

# Assuming these are also forecasting models, even if "reversed"
from algorithms.reversed_cagr import reversed_cagr_forecast
from algorithms.reversed_arima_meth import reversed_arima_forecast
from algorithms.reversed_holtwint_meth import reversed_holt_winters_forecast
from algorithms.reversed_dump_holt import reversed_damped_trend_forecast
from algorithms.reversed_exponsmoth_meth import reversed_exp_smoothing_forecast

ALL_MODELS = {
    "CAGR": cagr_forecast,
    "Average": average_forecast,
    "ARIMA": arima_forecast,
    "MovingAvg": moving_average_forecast,
    "GBM": gbm_forecast,
    "Expon_Smooth": exp_smoothing_forecast,
    "Holt-Winters": holt_winters_forecast,
    "Fourier": fourier_forecast,
    "GRU": gru_forecast,
    "RandomForest": random_forest_forecast,
    "DampedHolt": damped_trend_forecast,
    "Prophet": prophet_forecast,
    "Gaussian": gp_forecast,

    "Reversed_ARIMA": reversed_arima_forecast,
    "Reversed_CAGR": reversed_cagr_forecast,
    "Reversed_ExpSmooth": reversed_exp_smoothing_forecast,
    "Reversed_Dump_Holt": reversed_damped_trend_forecast,
    "Reversed_Holt": reversed_holt_winters_forecast
}


def safe_float_array(x):
    """Ensure data is a flat numpy array of floats."""
    return np.ravel(np.asarray(x, dtype=float)).astype(np.float64, copy=False)


def safe_mape(fc, act):
    """Calculate Mean Absolute Percentage Error, safely handling zeros."""
    fc, act = safe_float_array(fc), safe_float_array(act)
    mask = act != 0
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((act[mask] - fc[mask]) / act[mask])) * 100


def theils_u(act, fc):
    """Calculate Theil's U statistic."""
    fc, act = safe_float_array(fc), safe_float_array(act)
    if len(act) < 2 or len(fc) < 2:
        return np.inf
    num = np.sqrt(np.mean((fc - act) ** 2))
    den = np.sqrt(np.mean((act[1:] - act[:-1]) ** 2))
    return num / den if den != 0 else np.inf

def forecast_classical(future_horizon, ticker, dataframe, max_p=3, max_q=3, alpha=0.05, custom_models=None):
    """
    Returns a dict with:
      - 'forecast' : numpy array (future_horizon)
      - 'selected_models' : list[str]
      - 'weights' : list[float]
      - 'mean_mape' : dict[str -> float or nan]
      - 'mean_theils_u' : dict[str -> float or inf]
      - 'intervals' : str
    """
    U_THRESHOLD = 1.0

    # 1. Get Data
    # get_revenue from get_data.py
    # 1. Get Data
    if dataframe is not None and not dataframe.empty:
        # Use revenue column; ensure 1D numpy array
        revenue_data = np.ravel(np.asarray(dataframe['Revenue'], dtype=float))
    else:
        # fallback to scraping
        revenue_data, _ = get_revenue(ticker)
        revenue_data = np.nan_to_num(np.asarray(revenue_data, dtype=float), nan=0.0)

    if len(revenue_data) < 10:
        raise ValueError(f"Not enough data for ticker {ticker} (requires at least 10 points)")

    # 2. Split
    split_index = int(len(revenue_data) * 0.8)
    train_data = revenue_data[:split_index]
    test_data = revenue_data[split_index:]
    eval_horizon = len(test_data)

    MODEL_FUNCS = custom_models.copy() if custom_models is not None else ALL_MODELS.copy()

    # 3. Evaluate
    eval_mape = {}
    eval_theils_u = {}
    for name, func in MODEL_FUNCS.items():
        try:
            if name == "ARIMA":
                fc = func(train_data, eval_horizon, max_p=max_p, max_q=max_q)
            else:
                fc = func(train_data, eval_horizon)

            fc = safe_float_array(fc)
            if fc.size != eval_horizon:
                # pad or trim
                if fc.size < eval_horizon:
                    fc = np.pad(fc, (0, eval_horizon - fc.size), 'constant', constant_values=np.nan)
                else:
                    fc = fc[:eval_horizon]

            eval_mape[name] = safe_mape(fc, test_data)
            eval_theils_u[name] = theils_u(test_data, fc)
        except Exception:
            eval_mape[name] = np.nan
            eval_theils_u[name] = np.inf

    # 4. Select good models: Theil's U <= 1.0
    candidate_models = [k for k in eval_theils_u if np.isfinite(eval_theils_u[k])]
    good_models = [k for k in candidate_models if eval_theils_u[k] <= U_THRESHOLD]

    if not good_models:
        # Fallback: top 3 best by Theil's U
        sorted_by_u = sorted(candidate_models, key=lambda k: eval_theils_u[k])
        good_models = sorted_by_u[:3]

    if not good_models:
        raise ValueError("No models produced finite forecasts. Cannot proceed.")

    # --------------------------------------------------------------------- #
    # 5. WEIGHTS – inverse-MAPE weighting (only on models with finite MAPE)
    # --------------------------------------------------------------------- #
    valid_models = [
        m for m in good_models
        if m in eval_mape and np.isfinite(eval_mape[m]) and eval_mape[m] > 0
    ]

    if valid_models:
        # inverse MAPE → higher weight for lower error
        mapes = np.array([eval_mape[m] for m in valid_models])
        inv_mape = 1.0 / mapes
        norm = inv_mape.sum()
        if norm == 0:  # safety (should never happen)
            weights = np.ones(len(valid_models)) / len(valid_models)
        else:
            weights = inv_mape / norm
    else:
        # No model has a usable MAPE → fall back to uniform weights
        weights = np.ones(len(good_models)) / len(good_models)

    # --------------------------------------------------------------------- #
    # 6. RETRAIN & FORECAST with the *selected* models
    # --------------------------------------------------------------------- #
    final_forecasts = {}
    for name in good_models:
        try:
            func = MODEL_FUNCS[name]
            if name == "ARIMA":
                fc = func(revenue_data, future_horizon,
                          max_p=max_p, max_q=max_q)
            else:
                fc = func(revenue_data, future_horizon)

            fc = safe_float_array(fc)
            # pad / trim to exact horizon
            if fc.size < future_horizon:
                fc = np.pad(fc, (0, future_horizon - fc.size),
                            constant_values=np.nan)
            fc = fc[:future_horizon]
            final_forecasts[name] = fc
        except Exception:
            final_forecasts[name] = np.full(future_horizon, np.nan)

    # --------------------------------------------------------------------- #
    # 7. WEIGHTED ENSEMBLE
    # --------------------------------------------------------------------- #
    fc_weighted = np.zeros(future_horizon)

    for i, name in enumerate(good_models):
        # np.nan_to_num → 0.0 for any NaN that slipped through
        fc_i = np.nan_to_num(final_forecasts[name], nan=0.0)
        fc_weighted += weights[i] * fc_i  # ← weights[i] is now ALWAYS a float

    # --------------------------------------------------------------------- #
    # 8. CONFORMAL INTERVALS (using *test-set* residuals – correct!)
    # --------------------------------------------------------------------- #
    test_residuals = []
    for name in good_models:
        # Re-use the forecasts we already computed on the *test* split
        # (they are stored in the evaluation loop – we keep them here)
        try:
            # `eval_fc` was built in the evaluation loop; we reconstruct it safely
            func = MODEL_FUNCS[name]
            if name == "ARIMA":
                fc_test = func(train_data, eval_horizon, max_p=max_p, max_q=max_q)
            else:
                fc_test = func(train_data, eval_horizon)
            fc_test = safe_float_array(fc_test)[:eval_horizon]
            test_residuals.extend(np.abs(fc_test - test_data))
        except Exception:
            pass

    if test_residuals:
        q = np.quantile(test_residuals, 1 - alpha)
    else:
        q = 0.0

    interval_low = fc_weighted - q
    interval_high = fc_weighted + q

    interval_str = "\n".join(
        f"Horizon {i + 1}: [{interval_low[i]:.2f}, {interval_high[i]:.2f}]"
        for i in range(future_horizon)
    )
    interval_str = f"Conformal Prediction Intervals (alpha={alpha * 100:.0f}%):\n{interval_str}"

    accuracy_values = [1 - eval_mape[m] / 100 for m in good_models]  # convert MAPE % to "accuracy fraction"
    average_accuracy = float(np.sum(weights * np.array(accuracy_values)))

    return {
        "forecast": np.round(fc_weighted, 3),
        "intervals": interval_str,
        "selected_models": good_models,
        "weights": weights,
        "mean_mape": eval_mape,
        "mean_theils_u": eval_theils_u
    }
