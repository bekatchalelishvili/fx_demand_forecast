import copy
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import torch

# Hypertuning
import pickle

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, QuantileLoss

# to parse data from XLSX
from get_data import parse_data

data = parse_data()

# --------------------- LOAD & CLEAN DATA ---------------------------------------
data["date"] = pd.to_datetime(data["date"])
data = data.sort_values("date").reset_index(drop=True)

data["group_id"] = "GEO_FX"

# Time index: strictly monotonic integer
data["time_idx"] = data["date"].dt.year * 12 + data["date"].dt.month
data["time_idx"] -= data["time_idx"].min()
data["time_idx"] = data["time_idx"].astype(int)

# Calendar features
data["month"] = data["date"].dt.month.astype(str).astype("category")
data["quarter"] = data["quarter"].astype(str).astype("category")

# Log-transform target
data["log_fx_demand"] = np.log(data["fx_demand_usd_m"] + 1e-8)

# Aggregated features — group only by month/quarter (not time_idx) to avoid NaN on single-row groups
data["avg_fx_demand_by_month"] = data.groupby(
    ["month"], observed=True
)["fx_demand_usd_m"].transform("mean")

data["avg_fx_demand_by_quarter"] = data.groupby(
    ["quarter"], observed=True
)["fx_demand_usd_m"].transform("mean")

# Encode binary event columns as categorical
special_events = [
    "election_month",
    "major_election_month",
    "covid_period",
    "nbg_intervention",
    "is_quarter_end",
    "is_year_end",
    "is_year_start",
]
for col in special_events:
    data[col] = data[col].map({0: "-", 1: col}).astype("category")

# Encode remaining categoricals
data["speculation_level"] = data["speculation_level"].astype(str).astype("category")
data["intervention_direction"] = data["intervention_direction"].astype(str).astype("category")

# Fill any NaNs (forward then backward fill)
real_cols = [
    "fx_demand_usd_m", "log_fx_demand",
    "fx_demand_lag1m", "fx_demand_lag3m", "fx_demand_lag12m",
    "fx_demand_3m_ma", "fx_demand_6m_ma", "fx_demand_yoy_chg_pct",
    "usd_gel_rate", "eur_gel_rate", "usd_gel_mom_chg_pct", "usd_gel_yoy_chg_pct",
    "fx_volatility_3m", "nbg_policy_rate_pct", "fed_funds_rate_pct",
    "rate_differential_geo_us", "cpi_georgia_yoy_pct", "ppi_georgia_yoy_pct",
    "cpi_us_yoy_pct", "inflation_differential", "gdp_growth_yoy_pct",
    "unemployment_rate_pct", "fiscal_deficit_pct_gdp", "public_debt_pct_gdp",
    "public_debt_growth_yoy_pct", "m2_growth_yoy_pct", "dollarization_ratio_pct",
    "trade_balance_usd_m", "exports_usd_m", "imports_usd_m",
    "remittances_usd_m", "fdi_usd_m", "current_account_pct_gdp",
    "tourism_receipts_usd_m", "tourism_index", "nbg_reserves_usd_bn",
    "reserves_import_coverage_months", "russian_capital_inflow_proxy_usd_m",
    "geopolitical_shock_index", "nbg_intervention_size_usd_m",
    "speculation_index", "vix_index", "dxy_index", "brent_crude_usd",
    "em_capital_flow_index", "avg_fx_demand_by_month", "avg_fx_demand_by_quarter",
]
data[real_cols] = data[real_cols].ffill().bfill()

# Final clean reset index — critical for TimeSeriesDataSet
data = data.reset_index(drop=True)

print(f"Data shape: {data.shape}")
print(f"Time index range: {data['time_idx'].min()} to {data['time_idx'].max()}")
nan_check = data[real_cols].isna().sum()
nan_check = nan_check[nan_check > 0]
if len(nan_check):
    print(f"Remaining NaNs:\n{nan_check}")
else:
    print("No NaNs remaining.")

# --------------------- DATASET SETUP ----------------------------------
max_prediction_length = 6    # forecast 6 months ahead
max_encoder_length = 24      # use up to 24 months of history

training_cutoff = data["time_idx"].max() - max_prediction_length

time_varying_known_categoricals = [
    "month",
    "election_month",
    "major_election_month",
    "is_quarter_end",
    "is_year_end",
    "is_year_start",
]

time_varying_known_reals = [
    "time_idx",
    "avg_fx_demand_by_month",
    "avg_fx_demand_by_quarter",
]

time_varying_unknown_categoricals = [
    "covid_period",
    "nbg_intervention",
    "speculation_level",
    "intervention_direction",
]

time_varying_unknown_reals = [
    "fx_demand_usd_m",
    "log_fx_demand",
    "fx_demand_lag1m",
    "fx_demand_lag3m",
    "fx_demand_lag12m",
    "fx_demand_3m_ma",
    "fx_demand_6m_ma",
    "fx_demand_yoy_chg_pct",
    "usd_gel_rate",
    "eur_gel_rate",
    "usd_gel_mom_chg_pct",
    "usd_gel_yoy_chg_pct",
    "fx_volatility_3m",
    "nbg_policy_rate_pct",
    "fed_funds_rate_pct",
    "rate_differential_geo_us",
    "cpi_georgia_yoy_pct",
    "ppi_georgia_yoy_pct",
    "cpi_us_yoy_pct",
    "inflation_differential",
    "gdp_growth_yoy_pct",
    "unemployment_rate_pct",
    "fiscal_deficit_pct_gdp",
    "public_debt_pct_gdp",
    "public_debt_growth_yoy_pct",
    "m2_growth_yoy_pct",
    "dollarization_ratio_pct",
    "trade_balance_usd_m",
    "exports_usd_m",
    "imports_usd_m",
    "remittances_usd_m",
    "fdi_usd_m",
    "current_account_pct_gdp",
    "tourism_receipts_usd_m",
    "tourism_index",
    "nbg_reserves_usd_bn",
    "reserves_import_coverage_months",
    "russian_capital_inflow_proxy_usd_m",
    "geopolitical_shock_index",
    "nbg_intervention_size_usd_m",
    "speculation_index",
    "vix_index",
    "dxy_index",
    "brent_crude_usd",
    "em_capital_flow_index",
]

# Split and reset index on training slice — prevents the ValueError
training_data = data[data["time_idx"] <= training_cutoff].reset_index(drop=True)

training = TimeSeriesDataSet(
    training_data,
    time_idx="time_idx",
    target="fx_demand_usd_m",
    group_ids=["group_id"],
    min_encoder_length=max_encoder_length // 2,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["group_id"],
    static_reals=[],
    time_varying_known_categoricals=time_varying_known_categoricals,
    time_varying_known_reals=time_varying_known_reals,
    time_varying_unknown_categoricals=time_varying_unknown_categoricals,
    time_varying_unknown_reals=time_varying_unknown_reals,
    target_normalizer=GroupNormalizer(
        groups=["group_id"],
        transformation="softplus",
    ),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True,
)

# Validation dataset uses full data but inherits training parameters
validation = TimeSeriesDataSet.from_dataset(
    training,
    data.reset_index(drop=True),
    predict=True,
    stop_randomization=True,
)

# Dataloaders
batch_size = 32
train_dataloader = training.to_dataloader(
    train=True, batch_size=batch_size, num_workers=0
)
val_dataloader = validation.to_dataloader(
    train=False, batch_size=batch_size * 10, num_workers=0
)

print("Datasets created successfully.")
print(f"Training samples: {len(training)}, Validation samples: {len(validation)}")

# --------------------- BASELINE ---------------------------------------
baseline_predictions = Baseline().predict(val_dataloader, return_y=True)
print(
    f"Baseline MAE: {MAE()(baseline_predictions.output, baseline_predictions.y).item():.4f}"
)

# --------------------- TRAINER ----------------------------------------
pl.seed_everything(42)

early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=1e-4,
    patience=10,
    verbose=False,
    mode="min",
)
lr_logger = LearningRateMonitor()
logger = TensorBoardLogger("lightning_logs")

trainer = pl.Trainer(
    max_epochs=50,
    accelerator="auto",
    enable_model_summary=True,
    gradient_clip_val=0.1,
    limit_train_batches=50,
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,
)

# --------------------- MODEL ------------------------------------------
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=32,
    attention_head_size=2,
    dropout=0.1,
    hidden_continuous_size=16,
    loss=QuantileLoss(),
    log_interval=10,
    optimizer="Ranger",
    reduce_on_plateau_patience=4,
)

print(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")

# --------------------- TRAIN ------------------------------------------
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

# --------------------- EVALUATE ---------------------------------------
best_model_path = trainer.checkpoint_callback.best_model_path
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

predictions = best_tft.predict(
    val_dataloader,
    return_y=True,
    trainer_kwargs=dict(accelerator="auto"),
)
print(
    f"Validation MAE: {MAE()(predictions.output, predictions.y).item():.4f}"
)

# Raw predictions with attention weights for interpretation
raw_predictions = best_tft.predict(
    val_dataloader,
    mode="raw",
    return_x=True,
)

# --------------------- INTERPRET --------------------------------------
interpretation = best_tft.interpret_output(
    raw_predictions.output, reduction="sum"
)
best_tft.plot_interpretation(interpretation)

best_tft.plot_prediction(
    raw_predictions.x,
    raw_predictions.output,
    idx=0,
    add_loss_to_title=True,
)

# - - -- - -Tuning - -- - -
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

# create study
study = optimize_hyperparameters(
    train_dataloader,
    val_dataloader,
    model_path="optuna_test",
    n_trials=25,
    max_epochs=50,
    gradient_clip_val_range=(0.01, 1.0),
    hidden_size_range=(8, 128),
    hidden_continuous_size_range=(8, 128),
    attention_head_size_range=(1, 4),
    learning_rate_range=(0.001, 0.1),
    dropout_range=(0.1, 0.3),
    trainer_kwargs=dict(limit_train_batches=30),
    reduce_on_plateau_patience=4,
    use_learning_rate_finder=False,  # use Optuna to find ideal learning rate or use in-built learning rate finder
)

# save study results - also we can resume tuning at a later point in time
with open("test_study.pkl", "wb") as fout:
    pickle.dump(study, fout)

# show best hyperparameters
print(study.best_trial.params)

best_model_path = trainer.checkpoint_callback.best_model_path
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

predictions = best_tft.predict(
    val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="cpu")
)
MAE()(predictions.output, predictions.y)

raw_predictions = best_tft.predict(
    val_dataloader, mode="raw", return_x=True, trainer_kwargs=dict(accelerator="cpu")
)

# --------------------- OUT-OF-SAMPLE PREDICTION -----------------------

# Step 1: Define your externally forecasted values for the next 6 months
# These are the months AFTER your data ends (Jan 2026 - Jun 2026)
last_date = data["date"].max()
future_dates = pd.date_range(
    start=last_date + pd.DateOffset(months=1),
    periods=max_prediction_length,
    freq="MS"  # Month Start
)

future_df = pd.DataFrame({
    "date": future_dates,
    "group_id": "GEO_FX",
})

# ── Fill in your known/forecasted variables here ──────────────────────
# Replace these with your actual external forecasts
future_df["usd_gel_rate"]             = [2.91, 2.92, 2.93, 2.94, 2.95, 2.96]
future_df["eur_gel_rate"]             = [3.10, 3.11, 3.12, 3.13, 3.14, 3.15]
future_df["nbg_policy_rate_pct"]      = [8.25, 8.25, 8.00, 8.00, 7.75, 7.75]
future_df["fed_funds_rate_pct"]       = [4.25, 4.25, 4.00, 4.00, 3.75, 3.75]
future_df["cpi_georgia_yoy_pct"]      = [4.0,  3.9,  3.8,  3.7,  3.6,  3.5]
future_df["cpi_us_yoy_pct"]           = [2.1,  2.1,  2.0,  2.0,  1.9,  1.9]
future_df["vix_index"]                = [18.0, 17.5, 17.0, 16.5, 16.0, 15.5]
future_df["dxy_index"]                = [107., 106., 105., 104., 103., 102.]
future_df["brent_crude_usd"]          = [72.0, 73.0, 74.0, 73.0, 72.0, 71.0]
future_df["gdp_growth_yoy_pct"]       = [3.2,  3.2,  3.3,  3.3,  3.4,  3.4]
# ... add any other variables you have forecasts for

# ── Fill unknown variables with last known value (model ignores future)
unknown_cols = [c for c in real_cols if c not in future_df.columns
                and c not in ["avg_fx_demand_by_month", "avg_fx_demand_by_quarter",
                               "log_fx_demand"]]
last_row = data.iloc[-1]
for col in unknown_cols:
    future_df[col] = last_row[col]

# ── Target column must exist but value doesn't matter (ignored at inference)
future_df["fx_demand_usd_m"] = float("nan")

# ── Reconstruct all derived/categorical columns ────────────────────────
future_df["time_idx"] = future_df["date"].dt.year * 12 + future_df["date"].dt.month
future_df["time_idx"] -= data["time_idx"].min() + (
    data["date"].dt.year * 12 + data["date"].dt.month
).min()

future_df["month"] = future_df["date"].dt.month.astype(str).astype("category")
future_df["quarter"] = future_df["date"].dt.quarter.astype(str).astype("category")
future_df["log_fx_demand"] = 0.0  # placeholder, not used in decoder

future_df["avg_fx_demand_by_month"] = future_df["month"].map(
    data.groupby("month", observed=True)["fx_demand_usd_m"].mean()
)
future_df["avg_fx_demand_by_quarter"] = future_df["quarter"].map(
    data.groupby("quarter", observed=True)["fx_demand_usd_m"].mean()
)

# Calendar event flags — set to "-" (no-event) unless you know otherwise
for col in special_events:
    future_df[col] = "-"

# If you know an election is coming, set it:
# future_df.loc[future_df["date"] == "2026-04-01", "election_month"] = "election_month"

future_df["speculation_level"]     = "low"
future_df["intervention_direction"] = "none"

# Step 2: Concatenate historical + future data
data_with_future = pd.concat([data, future_df], ignore_index=True).reset_index(drop=True)

# Fix time_idx to be clean sequential integers
data_with_future["time_idx"] = (
    data_with_future["date"].dt.year * 12 + data_with_future["date"].dt.month
)
data_with_future["time_idx"] -= data_with_future["time_idx"].min()
data_with_future["time_idx"] = data_with_future["time_idx"].astype(int)

# ffill any remaining NaNs in future rows
data_with_future[real_cols] = data_with_future[real_cols].ffill().bfill()

# Align category dtypes with training dataset
for col in special_events + ["speculation_level", "intervention_direction", "month", "quarter"]:
    data_with_future[col] = data_with_future[col].astype("category")

data_with_future = data_with_future.reset_index(drop=True)

# Step 3: Create prediction dataset inheriting training parameters
pred_dataset = TimeSeriesDataSet.from_dataset(
    training,                        # inherits all scalers and encoders
    data_with_future,
    predict=True,
    stop_randomization=True,
)

pred_dataloader = pred_dataset.to_dataloader(
    train=False, batch_size=1, num_workers=0
)

# Step 4: Generate forecast
oos_predictions = best_tft.predict(
    pred_dataloader,
    mode="raw",
    return_x=True,
    trainer_kwargs=dict(accelerator="auto"),
)

# Step 5: Extract and display results
pred_tensor = oos_predictions.output.prediction  # shape: [1, 6, 7]
quantile_names = ["P2", "P10", "P25", "P50", "P75", "P90", "P98"]

results = pd.DataFrame(
    pred_tensor[0].numpy(),
    columns=quantile_names,
    index=future_dates.strftime("%Y-%m")
)
results.index.name = "Month"

print("\n========== OUT-OF-SAMPLE FORECAST ==========")
print(results.round(1).to_string())
print(f"\nPoint forecast (P50): {results['P50'].values}")


