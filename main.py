import warnings
warnings.filterwarnings("ignore")

import os
import json
import pickle
import numpy as np
import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

from get_data import parse_data

# ══════════════════════════════════════════════════════════════════════════════
# ▶ EDIT THESE BEFORE RUNNING
# ══════════════════════════════════════════════════════════════════════════════

EXTERNAL_FORECASTS = {
    "usd_gel_rate":         [2.78,   2.77,   2.76],
    "eur_gel_rate":         [3.02,   3.03,   3.04],
    "nbg_policy_rate_pct":  [8.0,    8.0,    8.00],
    "fed_funds_rate_pct":   [4.25,   4.25,   4.25],
    "cpi_georgia_yoy_pct":  [2.85,   2.7,    2.75],
    "cpi_us_yoy_pct":       [2.7,    2.7,    2.6],
    "vix_index":            [18.0,   21.5,   23.0],
    "dxy_index":            [107.,   106.,   105.],
    "brent_crude_usd":      [72.0,   80.0,   85.0],
    "gdp_growth_yoy_pct":   [3.2,    3.2,    3.3],
}

OPTUNA_TRIALS  = 200
MAX_EPOCHS     = 200
PATIENCE       = 20

# ══════════════════════════════════════════════════════════════════════════════

MAX_PREDICTION_LENGTH = 3
MAX_ENCODER_LENGTH    = 36

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
    "covid_period", "nbg_intervention", "intervention_direction",
]
TIME_VARYING_UNKNOWN_REALS = [
    "fx_demand_usd_m",
    "fx_demand_lag1m", "fx_demand_lag3m", "fx_demand_lag12m",
    "fx_demand_6m_ma",
    "usd_gel_rate", "eur_gel_rate", "usd_gel_mom_chg_pct",
    "fx_volatility_3m", "nbg_policy_rate_pct", "fed_funds_rate_pct",
    "rate_differential_geo_us", "cpi_georgia_yoy_pct",
    "cpi_us_yoy_pct", "gdp_growth_yoy_pct",
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

INITIAL_PARAMS = {
    "learning_rate":          0.005,
    "hidden_size":            32,
    "attention_head_size":    2,
    "dropout":                0.1,
    "hidden_continuous_size": 20,
    "gradient_clip_val":      0.1,
}


# ══════════════════════════════════════════════════════════════════════════════
class FXForecastPipeline:

    def __init__(self):
        self._data_loaded    = False
        self._step1_done     = False
        self._step2_done     = False
        self._step3_done     = False
        self._step4_done     = False
        self._step5_done     = False

        self.data            = None
        self.training        = None
        self.train_dl        = None
        self.val_dl          = None
        self.best_tft        = None
        self.best_path       = None
        self.tuned_params    = None

        self.baseline_mae    = None
        self.initial_mae     = None
        self.tuned_mae       = None
        self.results         = None

    def run(self) -> "FXForecastPipeline":
        self.step5_forecast()
        return self

    def _load_data(self):
        if self._data_loaded:
            return

        self._print_header("Loading & preparing data")

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

        print(f"  Shape     : {data.shape}")
        print(f"  Time idx  : {data['time_idx'].min()} → {data['time_idx'].max()}")

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

        self.data      = data
        self.training  = training
        self.train_dl  = training.to_dataloader(train=True,  batch_size=32,  num_workers=0)
        self.val_dl    = validation.to_dataloader(train=False, batch_size=320, num_workers=0)

        baseline = Baseline().predict(self.val_dl, return_y=True)
        self.baseline_mae = MAE()(baseline.output, baseline.y).item()
        print(f"  Baseline MAE : {self.baseline_mae:.4f}")

        self._data_loaded = True

    def _train(self, params: dict) -> str:
        pl.seed_everything(42)

        tft = TemporalFusionTransformer.from_dataset(
            self.training,
            learning_rate=params["learning_rate"],
            hidden_size=params["hidden_size"],
            attention_head_size=params["attention_head_size"],
            dropout=params["dropout"],
            hidden_continuous_size=params["hidden_continuous_size"],
            loss=QuantileLoss(),
            log_interval=10,
            optimizer="Ranger",
            reduce_on_plateau_patience=4,
        )
        print(f"  Parameters : {tft.size() / 1e3:.1f}k")

        trainer = pl.Trainer(
            max_epochs=MAX_EPOCHS,
            accelerator="auto",
            enable_model_summary=False,
            gradient_clip_val=params["gradient_clip_val"],
            callbacks=[
                EarlyStopping(monitor="val_loss", min_delta=1e-4,
                              patience=PATIENCE, mode="min"),
                LearningRateMonitor(),
            ],
            logger=TensorBoardLogger("models/lightning_logs"),
        )
        trainer.fit(tft, train_dataloaders=self.train_dl, val_dataloaders=self.val_dl)

        best_path = trainer.checkpoint_callback.best_model_path

        with open("models/best_model_path.txt", "w") as f:
            f.write(best_path)
        with open("models/training_dataset.pkl", "wb") as f:
            pickle.dump(self.training, f)

        return best_path

    @staticmethod
    def _print_header(title: str):
        print(f"\n{'─'*55}")
        print(f"  {title}")
        print(f"{'─'*55}")

    def step1_train(self) -> "FXForecastPipeline":
        if self._step1_done:
            print("  ✓ Step 1 already done — skipping")
            return self

        self._print_header("STEP 1 — Initial training")
        self._load_data()

        self.best_path = self._train(INITIAL_PARAMS)
        self.best_tft  = TemporalFusionTransformer.load_from_checkpoint(self.best_path)

        preds = self.best_tft.predict(
            self.val_dl, return_y=True, trainer_kwargs=dict(accelerator="auto")
        )
        self.initial_mae = MAE()(preds.output, preds.y).item()
        print(f"  Initial MAE  : {self.initial_mae:.4f}")
        print(f"  Baseline MAE : {self.baseline_mae:.4f}")

        self._step1_done = True
        return self

    def step2_tune(self) -> "FXForecastPipeline":
        if self._step2_done:
            print("  ✓ Step 2 already done — skipping")
            return self

        self.step1_train()

        self._print_header("STEP 2 — Hyperparameter tuning (Optuna)")
        print(f"  Running {OPTUNA_TRIALS} trials — this may take 10–20 min...")

        study = optimize_hyperparameters(
            self.train_dl,
            self.val_dl,
            model_path="models/optuna_test",
            n_trials=OPTUNA_TRIALS,
            max_epochs=100,
            gradient_clip_val_range=(0.01, 1.0),
            hidden_size_range=(8, 64),
            hidden_continuous_size_range=(8, 64),
            attention_head_size_range=(1, 4),
            learning_rate_range=(0.001, 0.05),
            dropout_range=(0.1, 0.3),
            trainer_kwargs=dict(limit_train_batches=30),
            reduce_on_plateau_patience=4,
            use_learning_rate_finder=False,
        )

        with open("models/test_study.pkl", "wb") as f:
            pickle.dump(study, f)

        p = study.best_trial.params
        self.tuned_params = {
            "learning_rate":          p.get("learning_rate", 0.001),
            "hidden_size":            p.get("hidden_size", 16),
            "attention_head_size":    p.get("attention_head_size", 2),
            "dropout":                p.get("dropout", 0.1),
            "hidden_continuous_size": p.get("hidden_continuous_size", 16),
            "gradient_clip_val":      p.get("gradient_clip_val", 0.1),
        }

        print(f"\n  Best trial val_loss : {study.best_trial.value:.4f}")
        print(f"  Best params:")
        for k, v in self.tuned_params.items():
            print(f"    {k:<35} {v}")

        self._step2_done = True
        return self

    def step3_retrain(self) -> "FXForecastPipeline":
        if self._step3_done:
            print("  ✓ Step 3 already done — skipping")
            return self

        self.step2_tune()

        self._print_header("STEP 3 — Retraining with tuned parameters")

        self.training  = None
        self.train_dl  = None
        self.val_dl    = None
        self._data_loaded = False
        self._load_data()

        self.best_path = self._train(self.tuned_params)
        self.best_tft  = TemporalFusionTransformer.load_from_checkpoint(self.best_path)

        preds = self.best_tft.predict(
            self.val_dl, return_y=True, trainer_kwargs=dict(accelerator="auto")
        )
        self.tuned_mae = MAE()(preds.output, preds.y).item()

        print(f"\n  Initial MAE  : {self.initial_mae:.4f}")
        print(f"  Tuned MAE    : {self.tuned_mae:.4f}")
        print(f"  Baseline MAE : {self.baseline_mae:.4f}")
        improved = self.tuned_mae < self.initial_mae
        beat     = self.tuned_mae < self.baseline_mae
        print(f"  vs Initial   : {'✅ improved' if improved else '⚠️  no improvement'}")
        print(f"  vs Baseline  : {'✅ beats baseline' if beat else '❌ still below baseline'}")

        self._step3_done = True
        return self

    def step4_evaluate(self) -> "FXForecastPipeline":
        if self._step4_done:
            print("  ✓ Step 4 already done — skipping")
            return self

        self.step3_retrain()

        self._print_header("STEP 4 — Model evaluation")

        raw_preds = self.best_tft.predict(
            self.val_dl, mode="raw", return_x=True,
            trainer_kwargs=dict(accelerator="auto"),
        )

        interpretation = self.best_tft.interpret_output(raw_preds.output, reduction="sum")
        self.best_tft.plot_interpretation(interpretation)
        self.best_tft.plot_prediction(
            raw_preds.x, raw_preds.output, idx=0, add_loss_to_title=True
        )

        print(f"  Plots displayed.")
        print(f"  Final MAE : {self.tuned_mae:.4f}  (baseline: {self.baseline_mae:.4f})")

        self._step4_done = True
        return self

    def step5_forecast(self) -> "FXForecastPipeline":
        if self._step5_done:
            print("  ✓ Step 5 already done — skipping")
            return self

        self.step4_evaluate()

        self._print_header("STEP 5 — Out-of-sample forecast")

        last_date    = self.data["date"].max()
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=MAX_PREDICTION_LENGTH,
            freq="MS",
        )

        future_df = pd.DataFrame({"date": future_dates, "group_id": "GEO_FX"})

        for col, vals in EXTERNAL_FORECASTS.items():
            future_df[col] = vals

        last_row = self.data.iloc[-1]
        for col in REAL_COLS:
            if col not in future_df.columns and col not in [
                "avg_fx_demand_by_month", "avg_fx_demand_by_quarter", "log_fx_demand"
            ]:
                future_df[col] = last_row[col]

        future_df["fx_demand_usd_m"] = float("nan")
        future_df["log_fx_demand"]   = 0.0
        future_df["month"]   = future_df["date"].dt.month.astype(str).astype("category")
        future_df["quarter"] = future_df["date"].dt.quarter.astype(str).astype("category")

        future_df["avg_fx_demand_by_month"] = future_df["month"].map(
            self.data.groupby("month", observed=True)["fx_demand_usd_m"].mean()
        )
        future_df["avg_fx_demand_by_quarter"] = future_df["quarter"].map(
            self.data.groupby("quarter", observed=True)["fx_demand_usd_m"].mean()
        )

        for col in SPECIAL_EVENTS:
            future_df[col] = "-"

        future_df.loc[future_df["date"].dt.month.isin([3,6,9,12]), "is_quarter_end"] = "is_quarter_end"
        future_df.loc[future_df["date"].dt.month == 12, "is_year_end"]   = "is_year_end"
        future_df.loc[future_df["date"].dt.month == 1,  "is_year_start"] = "is_year_start"
        future_df["speculation_level"]      = "low"
        future_df["intervention_direction"] = "none"

        data_with_future = pd.concat(
            [self.data, future_df], ignore_index=True
        ).reset_index(drop=True)

        data_with_future["time_idx"] = (
            data_with_future["date"].dt.year * 12 + data_with_future["date"].dt.month
        )
        data_with_future["time_idx"] -= data_with_future["time_idx"].min()
        data_with_future["time_idx"]  = data_with_future["time_idx"].astype(int)
        data_with_future[REAL_COLS]   = data_with_future[REAL_COLS].ffill().bfill()

        for col in SPECIAL_EVENTS + ["speculation_level", "intervention_direction", "month", "quarter"]:
            data_with_future[col] = data_with_future[col].astype("category")

        data_with_future = data_with_future.reset_index(drop=True)

        pred_dataset = TimeSeriesDataSet.from_dataset(
            self.training, data_with_future, predict=True, stop_randomization=True,
        )
        pred_dl = pred_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

        oos = self.best_tft.predict(
            pred_dl, mode="raw", return_x=True,
            trainer_kwargs=dict(accelerator="auto"),
        )

        quantile_names = ["P2", "P10", "P25", "P50", "P75", "P90", "P98"]
        self.results = pd.DataFrame(
            oos.output.prediction[0].numpy(),
            columns=quantile_names,
            index=future_dates.strftime("%Y-%m"),
        )
        self.results.index.name = "Month"

        self.results.to_csv("oos_forecast.csv")

        print(f"\n{'='*55}")
        print(f"  FINAL FORECAST (USD M)")
        print(f"{'='*55}")
        print(self.results.round(1).to_string())
        print(f"\n  P50 point forecast : {self.results['P50'].round(1).tolist()}")
        print(f"  P10 lower bound    : {self.results['P10'].round(1).tolist()}")
        print(f"  P90 upper bound    : {self.results['P90'].round(1).tolist()}")
        print(f"\n  Saved → oos_forecast.csv")

        self._step5_done = True
        return self

    def get_variable_importance(self) -> pd.DataFrame:
        if not self._step4_done:
            self.step4_evaluate()

        raw_preds = self.best_tft.predict(
            self.val_dl, mode="raw", return_x=True,
            trainer_kwargs=dict(accelerator="auto")
        )
        interpretation = self.best_tft.interpret_output(
            raw_preds.output, reduction="sum"
        )

        enc_names = (
                self.training.time_varying_unknown_reals
                + self.training.time_varying_unknown_categoricals
                + self.training.static_categoricals
                + self.training.static_reals
        )
        dec_names = (
                self.training.time_varying_known_reals
                + self.training.time_varying_known_categoricals
        )

        enc_vals = interpretation["encoder_variables"].numpy()
        dec_vals = interpretation["decoder_variables"].numpy()

        enc_df = pd.DataFrame({
            "variable": enc_names[:len(enc_vals)],
            "importance": enc_vals[:len(enc_names)],
            "type": "encoder (unknown)",
        })
        dec_df = pd.DataFrame({
            "variable": dec_names[:len(dec_vals)],
            "importance": dec_vals[:len(dec_names)],
            "type": "decoder (known)",
        })

        result = pd.concat([enc_df, dec_df]).sort_values(
            "importance", ascending=False
        ).reset_index(drop=True)

        threshold = result["importance"].mean() * 0.1
        result["suggest_drop"] = result["importance"] < threshold

        print("\n=== VARIABLE IMPORTANCE (ranked) ===")
        print(result.to_string(index=False))
        print(f"\nSuggested drops (importance < {threshold:.4f}):")
        print(result[result["suggest_drop"]]["variable"].tolist())

        return result

    # ──────────────────────────────────────────────────────────────────────
    # NEW: Export all essential results to a single JSON file
    # ──────────────────────────────────────────────────────────────────────
    def export_results_json(self, path: str = "f/static/x_forecast_results.json") -> dict:
        """
        Exports a structured JSON with:
          - model performance vs baseline
          - tuned hyperparameters
          - full quantile forecast table
          - variable importance ranking
        """
        if not self._step5_done:
            raise RuntimeError("Run the full pipeline first (pipeline.run())")

        importance_df = self.get_variable_importance()

        # ── Build forecast rows ───────────────────────────────────────────
        forecast_rows = []
        for month, row in self.results.iterrows():
            forecast_rows.append({
                "month": month,
                "P2":  round(float(row["P2"]),  2),
                "P10": round(float(row["P10"]), 2),
                "P25": round(float(row["P25"]), 2),
                "P50": round(float(row["P50"]), 2),
                "P75": round(float(row["P75"]), 2),
                "P90": round(float(row["P90"]), 2),
                "P98": round(float(row["P98"]), 2),
            })

        # ── Build variable importance rows ────────────────────────────────
        importance_rows = []
        for _, r in importance_df.iterrows():
            importance_rows.append({
                "variable":     r["variable"],
                "importance":   round(float(r["importance"]), 6),
                "type":         r["type"],
                "suggest_drop": bool(r["suggest_drop"]),
            })

        # ── Assemble output ───────────────────────────────────────────────
        output = {
            "meta": {
                "model":              "Temporal Fusion Transformer",
                "target":             "fx_demand_usd_m",
                "prediction_horizon": MAX_PREDICTION_LENGTH,
                "encoder_length":     MAX_ENCODER_LENGTH,
                "optuna_trials":      OPTUNA_TRIALS,
            },
            "performance": {
                "baseline_mae":         round(self.baseline_mae, 4),
                "initial_mae":          round(self.initial_mae,  4),
                "tuned_mae":            round(self.tuned_mae,    4),
                "beats_baseline":       bool(self.tuned_mae < self.baseline_mae),
                "improved_vs_initial":  bool(self.tuned_mae < self.initial_mae),
                "mae_vs_baseline_pct":  round(
                    (self.baseline_mae - self.tuned_mae) / self.baseline_mae * 100, 2
                ),
            },
            "tuned_hyperparameters": {
                k: round(v, 6) if isinstance(v, float) else v
                for k, v in (self.tuned_params or {}).items()
            },
            "forecast": forecast_rows,
            "variable_importance": importance_rows,
        }

        with open(path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\n  ✅ Results exported → {path}")
        return output


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    pipeline = FXForecastPipeline()
    pipeline.run()

    # Export structured JSON for the web dashboard
    pipeline.export_results_json("/static/fx_forecast_results.json")

    # Results also available in Python:
    # pipeline.results              → forecast DataFrame
    # pipeline.baseline_mae         → naive baseline MAE
    # pipeline.initial_mae          → MAE after step 1
    # pipeline.tuned_mae            → MAE after step 3 (tuned)
    # pipeline.tuned_params         → best Optuna hyperparameters