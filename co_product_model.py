"""
CO-in-Product Prediction Model — SMR Plant.
Grey-box hybrid: first-principles features + gradient boosting.

Usage
-----
  # Train and save model
  python co_product_model.py

  # Train with custom data path
  python co_product_model.py --data path/to/Combined_Data_with_KPIs.csv

Real-time integration
---------------------
  from co_product_model import COPredictor

  predictor = COPredictor.load("co_predictor.pkl")
  result = predictor.predict_realtime({
      "CO Slip (Syngas GC)": 0.82,
      "Shift dT (HTS Temperature Difference)": 72.0,
      "PSA Recovery": 88.5,
      "Plant Rate": 94.2,
      "S/C Ratio (Steam-to-Carbon)": 3.1,
      "Tube Outlet Temperature": 1565.0,
      "Excess O2 in Flue Gas": 2.4,
      "Purge Gas Buffer Vessel Pressure": 285.0,
  })
  print(result["co_ppm_predicted"], result["alert_level"])
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] xgboost not found. pip install xgboost  (recommended for best performance)")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("[WARN] shap not found. pip install shap  (needed for feature attribution)")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from feature_engineering import (
    KPI_FEATURE_COLS,
    PHYSICS_FEATURE_COLS,
    TARGET_COL,
    extract_feature_matrix,
    build_feature_row,
    compute_physics_features,
)

# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_DATA      = Path(__file__).parent / "Combined_Data_with_KPIs.csv"
MODEL_SAVE_PATH   = Path(__file__).parent / "co_predictor.pkl"
PLOT_DIR          = Path(__file__).parent / "model_plots"

CO_ALERT_SOFT = 5.0    # ppm — amber alert threshold
CO_ALERT_HARD = 10.0   # ppm — red alert (spec exceedance risk)

# Use log1p transform of target for training (reduces skew, improves RMSE)
USE_LOG_TARGET = True

# Temporal split: fraction of time range used for test
TEST_FRACTION = 0.20


# ── Utility ──────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    print(f"\n{'='*70}\n  {title}\n{'='*70}")


def _save_plot(fig: "plt.Figure", name: str) -> None:
    PLOT_DIR.mkdir(exist_ok=True)
    path = PLOT_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict[str, float]:
    """Compute regression metrics on original (ppm) scale."""
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 0.1, None))) * 100
    if label:
        print(f"\n  [{label}]")
        print(f"    RMSE : {rmse:.3f} ppm")
        print(f"    MAE  : {mae:.3f} ppm")
        print(f"    R²   : {r2:.4f}")
        print(f"    MAPE : {mape:.2f}%")
    return dict(rmse=rmse, mae=mae, r2=r2, mape=mape)


# ── Data preparation ──────────────────────────────────────────────────────────

def load_and_prepare(data_path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
    """
    Load KPI-enriched CSV, extract feature matrix and target.

    Returns:
        X      : feature DataFrame (NaN retained — imputed in pipeline)
        y      : target Series in original ppm scale
        timestamps: Timestamp series for temporal splitting (or None)
    """
    print(f"\nLoading: {data_path}")
    df = pd.read_csv(data_path, encoding="latin-1", low_memory=False)
    print(f"  Raw shape: {df.shape}")

    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.sort_values("Timestamp").reset_index(drop=True)

    X, y = extract_feature_matrix(df)

    # Drop rows where target is missing
    valid = y.notna()
    X, y = X[valid].reset_index(drop=True), y[valid].reset_index(drop=True)

    timestamps = (
        df.loc[valid, "Timestamp"].reset_index(drop=True)
        if "Timestamp" in df.columns else None
    )

    # Drop features that are >90% missing (imputer cannot learn a useful statistic)
    miss_rate = X.isna().mean()
    drop_cols = miss_rate[miss_rate > 0.90].index.tolist()
    if drop_cols:
        print(f"  Dropping {len(drop_cols)} features with >90% missing: {drop_cols}")
        X = X.drop(columns=drop_cols)

    # Remove rows where ALL remaining features are NaN
    all_nan = X.isna().all(axis=1)
    X, y = X[~all_nan].reset_index(drop=True), y[~all_nan].reset_index(drop=True)
    if timestamps is not None:
        timestamps = timestamps[~all_nan].reset_index(drop=True)

    print(f"  Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"  Feature columns: {list(X.columns)}")
    return X, y, timestamps


def temporal_split(
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series | None,
    test_fraction: float = TEST_FRACTION,
) -> tuple:
    """
    Time-based train/test split.
    Test set = the LAST `test_fraction` of the time-ordered data.
    This avoids data leakage that would occur with random splitting on time-series data.
    """
    n = len(X)
    split = int(n * (1 - test_fraction))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    ts_info = ""
    if timestamps is not None:
        ts_info = (
            f"\n  Train: {timestamps.iloc[0]} to {timestamps.iloc[split-1]}"
            f"\n  Test : {timestamps.iloc[split]} to {timestamps.dropna().iloc[-1]}"
        )
    print(f"\n  Train rows: {len(X_train):,} | Test rows: {len(X_test):,}{ts_info}")
    return X_train, X_test, y_train, y_test


# ── Model pipelines ───────────────────────────────────────────────────────────

def build_ridge_pipeline() -> Pipeline:
    """
    Ridge regression — interpretable linear baseline.
    Useful to validate that physics features have the expected sign.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler",  StandardScaler()),
        ("model",   Ridge(alpha=10.0)),
    ])


def build_rf_pipeline() -> Pipeline:
    """
    Random Forest — captures non-linearities, robust to outliers and multicollinearity.
    Heavier regularisation (shallow trees, high min_samples_leaf) for short time-series datasets.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("model", RandomForestRegressor(
            n_estimators=400,
            max_depth=6,           # shallower to avoid overfitting on ~19 days of data
            min_samples_leaf=20,   # each leaf must have 20+ samples
            max_features=0.5,
            n_jobs=-1,
            random_state=42,
        )),
    ])


def build_xgb_pipeline() -> Pipeline | None:
    """
    XGBoost — typically best predictive performance.
    Strong regularisation for short time-series: shallow trees, high lambda, low LR.
    """
    if not HAS_XGB:
        return None
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("model", xgb.XGBRegressor(
            n_estimators=800,
            max_depth=4,           # shallower trees to reduce overfitting
            learning_rate=0.02,    # lower LR compensated by more trees
            subsample=0.7,
            colsample_bytree=0.6,
            reg_alpha=5.0,         # stronger L1 regularisation
            reg_lambda=10.0,       # stronger L2 regularisation
            min_child_weight=20,   # high min_child_weight prevents small leaves
            gamma=1.0,             # minimum loss reduction to split
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )),
    ])


# ── Training & evaluation ─────────────────────────────────────────────────────

def _y_transform(y: pd.Series) -> np.ndarray:
    return np.log1p(y.clip(lower=0).values) if USE_LOG_TARGET else y.values


def _y_inverse(y_pred: np.ndarray) -> np.ndarray:
    return np.expm1(y_pred) if USE_LOG_TARGET else y_pred


def train_and_evaluate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    """
    Train all three model variants, evaluate on test set, return results dict.
    """
    y_tr = _y_transform(y_train)
    y_te = _y_transform(y_test)
    y_test_ppm = y_test.values

    results = {}
    pipelines = {
        "Ridge (linear baseline)": build_ridge_pipeline(),
        "Random Forest":           build_rf_pipeline(),
    }
    if HAS_XGB:
        pipelines["XGBoost"] = build_xgb_pipeline()

    _section("MODEL TRAINING & EVALUATION")
    print(f"\n  Target transform: {'log1p' if USE_LOG_TARGET else 'none'}")
    print(f"  Training on {len(X_train):,} rows | Evaluating on {len(X_test):,} rows")

    for name, pipe in pipelines.items():
        if pipe is None:
            continue
        print(f"\n  Training: {name} …")
        pipe.fit(X_train, y_tr)
        pred_tr = _y_inverse(pipe.predict(X_train))
        pred_te = _y_inverse(pipe.predict(X_test))

        train_m = _metrics(y_train.values, pred_tr)
        test_m  = _metrics(y_test_ppm, pred_te, label=name)
        results[name] = {
            "pipeline": pipe,
            "train_metrics": train_m,
            "test_metrics":  test_m,
            "y_pred_test":   pred_te,
        }

    return results


def cross_validate_best(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> None:
    """Time-series cross-validation on the full dataset."""
    _section("TIME-SERIES CROSS-VALIDATION (best model)")

    y_t = _y_transform(y)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_rmse, fold_r2 = [], []

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X)):
        pipeline.fit(X.iloc[tr_idx], y_t[tr_idx])
        pred = _y_inverse(pipeline.predict(X.iloc[te_idx]))
        actual = y.iloc[te_idx].values
        rmse = math.sqrt(mean_squared_error(actual, pred))
        r2   = r2_score(actual, pred)
        fold_rmse.append(rmse)
        fold_r2.append(r2)
        print(f"  Fold {fold+1}: RMSE = {rmse:.3f} ppm   R² = {r2:.4f}")

    print(f"\n  Mean RMSE : {np.mean(fold_rmse):.3f} ± {np.std(fold_rmse):.3f} ppm")
    print(f"  Mean R²   : {np.mean(fold_r2):.4f} ± {np.std(fold_r2):.4f}")


# ── Feature importance & SHAP ─────────────────────────────────────────────────

def _get_model_feature_names(pipe: Pipeline, input_cols: list[str]) -> list[str]:
    """Return feature names as seen by the model (after imputer transformation)."""
    try:
        return list(pipe.named_steps["imputer"].get_feature_names_out(input_cols))
    except Exception:
        return input_cols


def plot_feature_importance(results: dict[str, Any], feature_names: list[str]) -> None:
    if not HAS_MPL:
        return

    _section("FEATURE IMPORTANCE")

    for model_name in ("XGBoost", "Random Forest"):
        if model_name not in results:
            continue
        pipe = results[model_name]["pipeline"]
        model = pipe.named_steps["model"]

        try:
            importances = model.feature_importances_
        except AttributeError:
            continue

        # Use post-imputation names to match the length of importances
        model_feature_names = _get_model_feature_names(pipe, feature_names)
        if len(importances) != len(model_feature_names):
            model_feature_names = [f"feature_{i}" for i in range(len(importances))]

        imp_series = pd.Series(importances, index=model_feature_names).sort_values(ascending=False).head(20)
        print(f"\n  {model_name} — top feature importances:")
        for feat, imp in imp_series.items():
            bar = "#" * int(imp * 200)
            print(f"    {feat:<55s} {imp:.4f}  {bar}")

        fig, ax = plt.subplots(figsize=(10, max(4, len(imp_series) * 0.38)))
        imp_series.plot(kind="barh", ax=ax, color="#FF7043", edgecolor="white")
        ax.set_title(f"{model_name} — Feature Importance (top 20)", fontsize=12)
        ax.set_xlabel("Importance score")
        ax.invert_yaxis()
        _save_plot(fig, f"model_{model_name.lower().replace(' ', '_')}_importance")


def run_shap_analysis(results: dict[str, Any], X_test: pd.DataFrame) -> None:
    if not HAS_SHAP or not HAS_MPL:
        return
    if "XGBoost" not in results:
        return

    _section("SHAP FEATURE ATTRIBUTION (XGBoost)")

    pipe = results["XGBoost"]["pipeline"]
    model = pipe.named_steps["model"]
    imputer = pipe.named_steps["imputer"]
    imp_cols = _get_model_feature_names(pipe, list(X_test.columns))
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=imp_cols)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_imp)

        print("  SHAP summary computed. Saving plots …")

        # Summary plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test_imp, show=False)
        _save_plot(plt.gcf(), "shap_summary")

        # Bar plot (mean |SHAP|)
        shap.summary_plot(shap_values, X_test_imp, plot_type="bar", show=False)
        _save_plot(plt.gcf(), "shap_mean_importance")

    except Exception as e:
        print(f"  [WARN] SHAP failed: {e}")


def plot_actual_vs_predicted(results: dict[str, Any], y_test: pd.Series) -> None:
    if not HAS_MPL:
        return

    _section("ACTUAL vs PREDICTED PLOTS")
    y_true = y_test.values

    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), squeeze=False)

    for i, (name, res) in enumerate(results.items()):
        ax = axes[0][i]
        y_pred = res["y_pred_test"]
        m = res["test_metrics"]

        lim = max(y_true.max(), y_pred.max()) * 1.05
        ax.scatter(y_true, y_pred, alpha=0.3, s=8, color="#1976D2")
        ax.plot([0, lim], [0, lim], "r--", linewidth=1.2, label="Perfect prediction")
        ax.axhline(CO_ALERT_HARD, color="gray", linewidth=0.8, linestyle=":")
        ax.axvline(CO_ALERT_HARD, color="gray", linewidth=0.8, linestyle=":")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("Actual CO (ppm)", fontsize=10)
        ax.set_ylabel("Predicted CO (ppm)", fontsize=10)
        ax.set_title(
            f"{name}\nRMSE={m['rmse']:.2f}  R²={m['r2']:.3f}",
            fontsize=10,
        )
        ax.legend(fontsize=8)

    plt.suptitle("Actual vs Predicted — CO in Product", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_plot(fig, "actual_vs_predicted")


def plot_residuals(results: dict[str, Any], y_test: pd.Series) -> None:
    if not HAS_MPL:
        return

    y_true = y_test.values
    best_name = max(results, key=lambda k: results[k]["test_metrics"]["r2"])
    y_pred = results[best_name]["y_pred_test"]
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Residual Analysis — {best_name}", fontsize=12, fontweight="bold")

    axes[0].scatter(y_pred, residuals, alpha=0.3, s=6, color="#7B1FA2")
    axes[0].axhline(0, color="red", linewidth=1.2)
    axes[0].set_xlabel("Predicted CO (ppm)")
    axes[0].set_ylabel("Residual (ppm)")
    axes[0].set_title("Residuals vs Predicted")

    axes[1].hist(residuals, bins=60, color="#7B1FA2", alpha=0.75, edgecolor="white")
    axes[1].axvline(0, color="red", linewidth=1.2)
    axes[1].set_xlabel("Residual (ppm)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Residual Distribution  (mean={residuals.mean():.2f}, std={residuals.std():.2f})")

    _save_plot(fig, "residual_analysis")


# ── Model persistence ─────────────────────────────────────────────────────────

def save_model(pipeline: Pipeline, feature_names: list[str], save_path: Path) -> None:
    bundle = {"pipeline": pipeline, "feature_names": feature_names}
    joblib.dump(bundle, save_path)
    print(f"\n  Model saved → {save_path}")


def load_model(save_path: Path) -> tuple[Pipeline, list[str]]:
    bundle = joblib.load(save_path)
    return bundle["pipeline"], bundle["feature_names"]


# ── COPredictor — real-time prediction class ──────────────────────────────────

class COPredictor:
    """
    Real-time CO-in-product predictor for plant DCS integration.

    Wraps the trained model pipeline with:
      - Automatic physics feature computation
      - Alert level classification
      - Feature contribution attribution (SHAP or RF importances)
      - Confidence indication based on data completeness

    Example usage in a real-time loop:
    ───────────────────────────────────
        predictor = COPredictor.load("co_predictor.pkl")

        # Every N minutes, collect current readings from DCS
        readings = {
            "CO Slip (Syngas GC)":                  0.82,
            "Shift dT (HTS Temperature Difference)": 72.0,
            "PSA Recovery":                          88.5,
            "Plant Rate":                            94.2,
            "S/C Ratio (Steam-to-Carbon)":            3.1,
            "Tube Outlet Temperature":             1565.0,
            "Excess O2 in Flue Gas":                  2.4,
            "Purge Gas Buffer Vessel Pressure":      285.0,
        }

        result = predictor.predict_realtime(readings)
        print(f"Predicted CO: {result['co_ppm_predicted']:.1f} ppm  [{result['alert_level']}]")
        for feat, contrib in result['top_drivers']:
            print(f"  {feat}: {contrib:+.2f} ppm contribution")
    """

    def __init__(
        self,
        pipeline: Pipeline | None = None,
        feature_names: list[str] | None = None,
        alert_soft: float = CO_ALERT_SOFT,
        alert_hard: float = CO_ALERT_HARD,
    ):
        self.pipeline      = pipeline
        self.feature_names = feature_names or []
        self.alert_soft    = alert_soft
        self.alert_hard    = alert_hard
        self._shap_explainer = None
        self._rf_importances: dict[str, float] = {}

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_type: str = "xgboost",
    ) -> "COPredictor":
        """
        Train the predictor.
        model_type: 'xgboost' | 'rf' | 'ridge'
        """
        self.feature_names = list(X_train.columns)
        y_t = _y_transform(y_train)

        if model_type == "xgboost" and HAS_XGB:
            self.pipeline = build_xgb_pipeline()
        elif model_type == "rf":
            self.pipeline = build_rf_pipeline()
        else:
            self.pipeline = build_ridge_pipeline()

        print(f"  Fitting COPredictor ({model_type}) on {len(X_train):,} rows …")
        self.pipeline.fit(X_train, y_t)

        # Pre-build SHAP explainer if available
        self._build_shap_explainer(X_train)

        # Cache RF importances
        try:
            model = self.pipeline.named_steps["model"]
            if hasattr(model, "feature_importances_"):
                self._rf_importances = dict(
                    zip(self.feature_names, model.feature_importances_)
                )
        except Exception:
            pass

        return self

    def _build_shap_explainer(self, X_ref: pd.DataFrame) -> None:
        if not HAS_SHAP:
            return
        try:
            model = self.pipeline.named_steps["model"]
            imputer = self.pipeline.named_steps["imputer"]
            X_imp = pd.DataFrame(
                imputer.transform(X_ref),
                columns=X_ref.columns,
            )
            self._shap_explainer = shap.TreeExplainer(model, X_imp)
        except Exception as e:
            print(f"  [WARN] SHAP explainer not built: {e}")

    def _predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Predict on a feature DataFrame, returning ppm."""
        X_aligned = X.reindex(columns=self.feature_names)
        pred_t = self.pipeline.predict(X_aligned)
        return np.clip(_y_inverse(pred_t), 0, None)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Batch predict on a feature DataFrame. Returns array of CO (ppm)."""
        return self._predict_raw(X)

    def _get_top_drivers(
        self,
        X_row: pd.DataFrame,
        n: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Return top N feature contributions for a single prediction.
        Uses SHAP values if available, otherwise falls back to feature importance × deviation.
        """
        if self._shap_explainer is not None:
            try:
                imputer = self.pipeline.named_steps["imputer"]
                X_imp = pd.DataFrame(
                    imputer.transform(X_row.reindex(columns=self.feature_names)),
                    columns=self.feature_names,
                )
                sv = self._shap_explainer.shap_values(X_imp)[0]
                # Convert log-space SHAP to approximate ppm impact
                base = _y_inverse(np.array([self._shap_explainer.expected_value]))[0]
                contributions = {}
                for feat, sv_val in zip(self.feature_names, sv):
                    contributions[feat] = float(sv_val)  # log-space; sign is reliable
                sorted_c = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
                return sorted_c[:n]
            except Exception:
                pass

        # Fallback: importance-weighted deviation from training median
        if self._rf_importances:
            row_vals = X_row.reindex(columns=self.feature_names).iloc[0]
            contribs = {
                feat: float(imp * row_vals.get(feat, 0))
                for feat, imp in self._rf_importances.items()
                if pd.notna(row_vals.get(feat))
            }
            return sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:n]

        return []

    def predict_realtime(
        self,
        sensor_readings: dict[str, float],
        n_drivers: int = 5,
    ) -> dict[str, Any]:
        """
        Primary real-time prediction interface.

        Parameters
        ----------
        sensor_readings : dict
            Current sensor values keyed by KPI column name or DCS tag string.
            Missing keys are imputed from training medians.
        n_drivers : int
            Number of top contributing features to report.

        Returns
        -------
        dict with keys:
            co_ppm_predicted   : float — predicted CO in ppm
            co_ppm_low         : float — conservative estimate (predicted × 0.7)
            co_ppm_high        : float — conservative upper bound (predicted × 1.5)
            alert_level        : 'green' | 'amber' | 'red'
            alert_message      : str — human-readable status
            top_drivers        : list of (feature_name, contribution) tuples
            confidence         : 'high' | 'medium' | 'low'
            n_features_present : int — number of non-NaN input features
        """
        if self.pipeline is None:
            raise RuntimeError("Model not trained. Call .fit() or .load() first.")

        # Build feature row (adds physics features automatically)
        row_df = build_feature_row(sensor_readings)
        X_row  = row_df.reindex(columns=self.feature_names)

        # Count how many features are present
        n_present = int(X_row.notna().sum(axis=1).iloc[0])
        n_total   = len(self.feature_names)
        pct_present = n_present / n_total

        # Confidence tier
        if pct_present >= 0.8:
            confidence = "high"
        elif pct_present >= 0.5:
            confidence = "medium"
        else:
            confidence = "low"

        # Predict
        co_pred = float(self._predict_raw(X_row)[0])

        # Simple uncertainty envelope (heuristic; replace with quantile regression for production)
        co_low  = max(0.0, co_pred * 0.75)
        co_high = co_pred * 1.40

        # Alert classification
        if co_pred < self.alert_soft:
            alert_level   = "green"
            alert_message = f"CO {co_pred:.1f} ppm — within normal operating range (< {self.alert_soft} ppm)."
        elif co_pred < self.alert_hard:
            alert_level   = "amber"
            alert_message = (
                f"CO {co_pred:.1f} ppm — approaching specification limit. "
                f"Monitor closely. Spec limit: {self.alert_hard} ppm."
            )
        else:
            alert_level   = "red"
            alert_message = (
                f"CO {co_pred:.1f} ppm — AT OR ABOVE SPEC LIMIT ({self.alert_hard} ppm). "
                f"Immediate operator review required."
            )

        # Feature attribution
        top_drivers = self._get_top_drivers(X_row, n=n_drivers)

        return {
            "co_ppm_predicted":    round(co_pred, 2),
            "co_ppm_low":          round(co_low,  2),
            "co_ppm_high":         round(co_high, 2),
            "alert_level":         alert_level,
            "alert_message":       alert_message,
            "top_drivers":         top_drivers,
            "confidence":          confidence,
            "n_features_present":  n_present,
            "n_features_total":    n_total,
        }

    def print_prediction(self, result: dict[str, Any]) -> None:
        """Pretty-print a predict_realtime() result to console."""
        alert_icons = {"green": "✓", "amber": "!", "red": "✖"}
        icon = alert_icons.get(result["alert_level"], "?")
        print(f"\n  ┌─ CO-in-Product Prediction ───────────────────────────────────")
        print(f"  │  Predicted   : {result['co_ppm_predicted']:.2f} ppm")
        print(f"  │  Range       : {result['co_ppm_low']:.2f} – {result['co_ppm_high']:.2f} ppm")
        print(f"  │  Alert       : [{icon}] {result['alert_level'].upper()}")
        print(f"  │  Message     : {result['alert_message']}")
        print(f"  │  Confidence  : {result['confidence']}  ({result['n_features_present']}/{result['n_features_total']} features available)")
        print(f"  │")
        print(f"  │  Top drivers:")
        for feat, contrib in result["top_drivers"]:
            direction = "↑" if contrib > 0 else "↓"
            print(f"  │    {direction} {feat:<50s} ({contrib:+.3f})")
        print(f"  └──────────────────────────────────────────────────────────────")

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Return evaluation metrics on a held-out set."""
        y_pred = self._predict_raw(X)
        return _metrics(y.values, y_pred)

    def save(self, path: Path | str = MODEL_SAVE_PATH) -> None:
        bundle = {
            "pipeline":         self.pipeline,
            "feature_names":    self.feature_names,
            "alert_soft":       self.alert_soft,
            "alert_hard":       self.alert_hard,
            "_rf_importances":  self._rf_importances,
        }
        joblib.dump(bundle, path)
        print(f"  COPredictor saved → {path}")

    @classmethod
    def load(cls, path: Path | str = MODEL_SAVE_PATH) -> "COPredictor":
        bundle = joblib.load(path)
        obj = cls(
            pipeline      = bundle["pipeline"],
            feature_names = bundle["feature_names"],
            alert_soft    = bundle.get("alert_soft", CO_ALERT_SOFT),
            alert_hard    = bundle.get("alert_hard", CO_ALERT_HARD),
        )
        obj._rf_importances = bundle.get("_rf_importances", {})
        print(f"  COPredictor loaded ← {path}")
        return obj


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main(data_path: Path = DEFAULT_DATA) -> COPredictor:
    _section("CO-IN-PRODUCT PREDICTION PIPELINE — SMR Plant")
    print(f"  Grey-box model: first-principles features + gradient boosting")
    print(f"  Target transform: {'log1p(CO)' if USE_LOG_TARGET else 'none'}")

    # ── 1. Load data
    X, y, timestamps = load_and_prepare(data_path)

    # ── 2. Train/test split (temporal)
    X_train, X_test, y_train, y_test = temporal_split(X, y, timestamps)

    # ── 3. Train all model variants
    results = train_and_evaluate(X_train, X_test, y_train, y_test)

    # ── 4. Select best model
    best_name = max(results, key=lambda k: results[k]["test_metrics"]["r2"])
    _section(f"BEST MODEL: {best_name}")
    m = results[best_name]["test_metrics"]
    print(f"  Test RMSE: {m['rmse']:.3f} ppm")
    print(f"  Test R²  : {m['r2']:.4f}")
    print(f"  Test MAE : {m['mae']:.3f} ppm")

    # ── 5. Time-series cross-validation on best model
    best_pipe = results[best_name]["pipeline"]
    cross_validate_best(best_pipe, X, y)

    # ── 6. Visualisation
    plot_feature_importance(results, list(X.columns))
    plot_actual_vs_predicted(results, y_test)
    plot_residuals(results, y_test)

    # ── 7. SHAP analysis
    if HAS_SHAP and HAS_XGB and "XGBoost" in results:
        run_shap_analysis(results, X_test)

    # ── 8. Wrap best model in COPredictor and save
    predictor = COPredictor(
        pipeline      = best_pipe,
        feature_names = list(X.columns),
    )
    predictor._build_shap_explainer(X_train)
    try:
        predictor._rf_importances = dict(
            zip(list(X.columns), best_pipe.named_steps["model"].feature_importances_)
        )
    except Exception:
        pass
    predictor.save(MODEL_SAVE_PATH)

    # ── 9. Demo real-time prediction using last row of test set
    _section("DEMO — Real-time prediction on last test row")
    last_row = X_test.iloc[-1].to_dict()
    result = predictor.predict_realtime(last_row)
    predictor.print_prediction(result)
    print(f"\n  Actual CO for that row: {y_test.iloc[-1]:.2f} ppm")

    _section("COMPLETE")
    print(f"  Model     → {MODEL_SAVE_PATH}")
    print(f"  Plots     → {PLOT_DIR}/")
    print(f"""
  To use in real-time:
    from co_product_model import COPredictor
    predictor = COPredictor.load("co_predictor.pkl")
    result = predictor.predict_realtime({{
        "CO Slip (Syngas GC)": <current_value>,
        "Shift dT (HTS Temperature Difference)": <current_value>,
        "PSA Recovery": <current_value>,
        "Plant Rate": <current_value>,
        ... (fill as many as available)
    }})
    predictor.print_prediction(result)
""")
    return predictor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CO-in-product prediction model")
    parser.add_argument(
        "--data", type=str, default=str(DEFAULT_DATA),
        help="Path to Combined_Data_with_KPIs.csv",
    )
    args = parser.parse_args()
    main(Path(args.data))
