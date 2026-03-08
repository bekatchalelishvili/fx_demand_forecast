"""
prepare_data.py  —  shared constants + data loading function
"""

import numpy as np
import pandas as pd
from get_data import parse_data

# ── Constants ─────────────────────────────────────────────────────────────────
REAL_COLS = [
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

SPECIAL_EVENTS = [
    "election_month", "major_election_month", "covid_period",
    "nbg_intervention", "is_quarter_end", "is_year_end", "is_year_start",
]

TIME_VARYING_KNOWN_CATEGORICALS = [
    "month", "election_month", "major_election_month",
    "is_quarter_end", "is_year_end", "is_year_start",
]
TIME_VARYING_KNOWN_REALS = [
    "time_idx", "avg_fx_demand_by_month", "avg_fx_demand_by_quarter",
]
TIME_VARYING_UNKNOWN_CATEGORICALS = [
    "covid_period", "nbg_intervention", "speculation_level", "intervention_direction",
]
TIME_VARYING_UNKNOWN_REALS = [
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
    "em_capital_flow_index",
]

MAX_PREDICTION_LENGTH = 6
MAX_ENCODER_LENGTH    = 24


def load_and_prepare() -> pd.DataFrame:
    data = parse_data()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").reset_index(drop=True)
    data["group_id"] = "GEO_FX"

    data["time_idx"] = data["date"].dt.year * 12 + data["date"].dt.month
    data["time_idx"] -= data["time_idx"].min()
    data["time_idx"] = data["time_idx"].astype(int)

    data["month"]   = data["date"].dt.month.astype(str).astype("category")
    data["quarter"] = data["quarter"].astype(str).astype("category")

    data["log_fx_demand"] = np.log(data["fx_demand_usd_m"] + 1e-8)

    data["avg_fx_demand_by_month"] = data.groupby(
        ["month"], observed=True)["fx_demand_usd_m"].transform("mean")
    data["avg_fx_demand_by_quarter"] = data.groupby(
        ["quarter"], observed=True)["fx_demand_usd_m"].transform("mean")

    for col in SPECIAL_EVENTS:
        data[col] = data[col].map({0: "-", 1: col}).astype("category")

    data["speculation_level"]      = data["speculation_level"].astype(str).astype("category")
    data["intervention_direction"] = data["intervention_direction"].astype(str).astype("category")

    data[REAL_COLS] = data[REAL_COLS].ffill().bfill()
    data = data.reset_index(drop=True)

    print(f"Data shape     : {data.shape}")
    print(f"Time idx range : {data['time_idx'].min()} to {data['time_idx'].max()}")
    nan_check = data[REAL_COLS].isna().sum()
    nan_check = nan_check[nan_check > 0]
    print("NaN check      :", "OK" if len(nan_check) == 0 else f"\n{nan_check}")
    return data


def build_datasets(data: pd.DataFrame):
    """Returns (training_dataset, validation_dataset, train_dl, val_dl)."""
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer

    training_cutoff = data["time_idx"].max() - MAX_PREDICTION_LENGTH
    training_data   = data[data["time_idx"] <= training_cutoff].reset_index(drop=True)

    training = TimeSeriesDataSet(
        training_data,
        time_idx="time_idx",
        target="fx_demand_usd_m",
        group_ids=["group_id"],
        min_encoder_length=MAX_ENCODER_LENGTH // 2,
        max_encoder_length=MAX_ENCODER_LENGTH,
        min_prediction_length=1,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        static_categoricals=["group_id"],
        static_reals=[],
        time_varying_known_categoricals=TIME_VARYING_KNOWN_CATEGORICALS,
        time_varying_known_reals=TIME_VARYING_KNOWN_REALS,
        time_varying_unknown_categoricals=TIME_VARYING_UNKNOWN_CATEGORICALS,
        time_varying_unknown_reals=TIME_VARYING_UNKNOWN_REALS,
        target_normalizer=GroupNormalizer(groups=["group_id"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, data.reset_index(drop=True), predict=True, stop_randomization=True,
    )

    train_dl = training.to_dataloader(train=True,  batch_size=32,  num_workers=0)
    val_dl   = validation.to_dataloader(train=False, batch_size=320, num_workers=0)

    return training, validation, train_dl, val_dl