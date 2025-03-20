"""
Process Optimization Module — SMR Plant
================================================
Adds post-ML optimization KPIs and generates:

  1. Post-ML KPIs  — columns appended to the enriched DataFrame
       CO Spec Headroom (Predicted), HTS Catalyst Utilization,
       Efficiency Gap to Design, Steam Cost Index, Optimal SC Flag

  2. Trade-off curves — CO predicted vs S/C, HTS temp, Plant Rate
       Uses the trained COPredictor to sweep each parameter while
       holding all others at their current operating point.

  3. 2D Operating Envelope — Plant Rate vs S/C, coloured by CO ppm
       Shows the feasible operating region and constraint boundaries.

  4. Recommendation engine — rule-based, ranked by economic impact
       Reads the latest operating point and outputs a priority list
       of specific, quantified process adjustments.

  5. Diagnostic plots — saved to model_plots/optimization/

Usage
-----
  from optimization import OptimizationAnalyzer
  analyzer = OptimizationAnalyzer(predictor)          # pass trained COPredictor
  df_opt   = analyzer.score_dataframe(df_enriched)   # adds opt KPI columns
  analyzer.run_analysis(df_opt)                       # curves + plots + report
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_ROOT    = Path(__file__).resolve().parent
PLOT_DIR = _ROOT / "model_plots" / "optimization"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Plant constants ───────────────────────────────────────────────────────────
CO_SPEC_LIMIT    = 10.0    # ppm — product specification
CO_ALERT_SOFT    = 5.0     # ppm — early-warning threshold
SC_MIN_NO_COKING = 2.7     # mol/mol — coking prevention floor
SC_MAX_PRACTICAL = 5.5     # mol/mol — practical upper limit (steam drum capacity)
DESIGN_GROSS_EFF = 285.0   # BTU/SCF — typical SMR design gross efficiency
HTS_MIN_TEMP_F   = 350.0   # °F — minimum HTS catalyst activity temperature
HTS_MAX_TEMP_F   = 510.0   # °F — maximum HTS operating temperature
TUBE_TEMP_LIMIT  = 1650.0  # °F — metallurgical limit for reformer tubes
RATE_MIN         = 40.0    # % plant rate — practical minimum
RATE_MAX         = 110.0   # % plant rate — practical maximum
EXCESS_O2_MIN    = 1.5     # % — safety combustion floor
EXCESS_O2_DESIGN = 3.0     # % — design target for excess O2


# ── Helper ────────────────────────────────────────────────────────────────────

def _safe_float(val, default=np.nan):
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _find_col(df: pd.DataFrame, *candidates) -> Optional[str]:
    """Return the first candidate column name that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ── Post-ML KPI computation ───────────────────────────────────────────────────

def compute_opt_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add post-ML optimization KPIs that require CO_Predicted_ppm or
    physics features (available only after the ML stage runs).
    Returns df with new columns appended in-place on a copy.
    """
    result = df.copy()

    # ── CO spec headrooms ─────────────────────────────────────────────────────
    co_pred_col = _find_col(result, "CO_Predicted_ppm")
    if co_pred_col:
        co_pred = pd.to_numeric(result[co_pred_col], errors="coerce")
        result["CO Spec Headroom (Predicted)"] = CO_SPEC_LIMIT - co_pred
        result["CO Spec Headroom (Predicted)"] = result["CO Spec Headroom (Predicted)"]

    co_meas_col = _find_col(result, "CO in Product")
    if co_meas_col:
        co_meas = pd.to_numeric(result[co_meas_col], errors="coerce")
        result["CO Spec Headroom (Measured)"] = CO_SPEC_LIMIT - co_meas

    # ── HTS catalyst utilization ───────────────────────────────────────────────
    aeq_col = _find_col(result, "approach_to_eq")
    if aeq_col:
        result["HTS Catalyst Utilization (%)"] = (
            pd.to_numeric(result[aeq_col], errors="coerce") * 100.0
        ).clip(0, 100)

    # ── Efficiency gap to design ───────────────────────────────────────────────
    eff_col = _find_col(result, "Gross Efficiency")
    if eff_col:
        gross_eff = pd.to_numeric(result[eff_col], errors="coerce")
        result["Efficiency Gap to Design (BTU/SCF)"] = gross_eff - DESIGN_GROSS_EFF

    # ── Steam Cost Index — S/C normalised to coking minimum ───────────────────
    sc_col = _find_col(result, "S/C Ratio (Steam-to-Carbon)", "S/C Ratio")
    if sc_col:
        sc = pd.to_numeric(result[sc_col], errors="coerce")
        # Index: 100 = exactly at minimum safe S/C; 150 = 50% more steam than needed
        result["Steam Cost Index"] = (sc / SC_MIN_NO_COKING * 100.0).clip(0, 300)

    # ── Optimal S/C flag (1 = using more steam than CO spec requires) ─────────
    if co_pred_col and sc_col:
        co_pred = pd.to_numeric(result[co_pred_col], errors="coerce")
        sc      = pd.to_numeric(result[sc_col], errors="coerce")
        # If CO is well below spec AND S/C is above minimum, there is steam saving potential
        result["Steam Reduction Opportunity"] = (
            (co_pred < CO_SPEC_LIMIT - 2.0) & (sc > SC_MIN_NO_COKING + 0.3)
        ).astype(int)

    return result


# ── Trade-off curve engine ────────────────────────────────────────────────────

class OptimizationAnalyzer:
    """
    Generates trade-off curves, operating envelopes, and recommendations
    by sweeping the trained COPredictor across parameter ranges.
    """

    # Feature column names as they appear in the enriched DataFrame
    _SC_COL     = "S/C Ratio (Steam-to-Carbon)"
    _RATE_COL   = "Plant Rate"
    _PSA_COL    = "PSA Recovery"
    _GROSS_COL  = "Gross Efficiency"
    _NET_COL    = "Net Efficiency"
    _HTS_T_COL  = "hts_outlet_temp_c"   # physics feature (°C)
    _APPROACH_COL = "approach_to_eq"
    _CO_PRED_COL  = "CO_Predicted_ppm"

    def __init__(self, predictor):
        """predictor — fitted COPredictor instance from co_product_model.py"""
        self.predictor = predictor

    # ── Internal: feature sweep ───────────────────────────────────────────────

    def _get_baseline_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Extract the latest valid feature vector from the enriched DataFrame.
        Returns (X_row, feature_names) where X_row has shape (1, n_features).
        """
        from feature_engineering import extract_feature_matrix
        X, _ = extract_feature_matrix(df)

        feat_names = list(X.columns)
        # Find the last row with no all-NaN features
        X_num = X.apply(pd.to_numeric, errors="coerce")
        valid_rows = X_num.dropna(how="all")
        if valid_rows.empty:
            raise ValueError("No valid feature rows found in DataFrame")

        last_row = valid_rows.iloc[-1].values.astype(float)
        # Replace NaN with column median (same imputation as training)
        col_medians = np.nanmedian(X_num.values, axis=0)
        nans = np.isnan(last_row)
        last_row[nans] = col_medians[nans]

        return last_row.reshape(1, -1), feat_names

    def _predict_feature_vector(self, X: np.ndarray) -> float:
        """
        Predict CO ppm from a raw feature vector, bypassing tag resolution.
        Uses the pipeline's imputer + model directly.
        """
        pipe = self.predictor.model
        # Reindex to trained feature names and impute
        feat_df = pd.DataFrame(X, columns=self.predictor.feature_names[: X.shape[1]])
        co_log1p = pipe.predict(feat_df)[0]
        return float(np.expm1(co_log1p))

    def _sweep_feature(
        self,
        baseline: np.ndarray,
        feat_names: List[str],
        feature_key: str,
        sweep_values: np.ndarray,
        linked_updater=None,
    ) -> np.ndarray:
        """
        Sweep one feature across sweep_values, returning predicted CO ppm array.
        linked_updater(X_copy, val) can update derived features (e.g. K_eq from T).
        """
        try:
            idx = feat_names.index(feature_key)
        except ValueError:
            return np.full(len(sweep_values), np.nan)

        preds = []
        for val in sweep_values:
            X_copy = baseline.copy()
            X_copy[0, idx] = val
            if linked_updater is not None:
                X_copy = linked_updater(X_copy, val, feat_names)
            try:
                co = self._predict_feature_vector(X_copy)
            except Exception:
                co = np.nan
            preds.append(co)
        return np.array(preds)

    @staticmethod
    def _hts_temp_updater(X: np.ndarray, hts_t_c: float, feat_names: List[str]) -> np.ndarray:
        """When sweeping HTS outlet temperature, update K_eq and approx_eq_co_pct."""
        from feature_engineering import hts_equilibrium_constant
        T_k = hts_t_c + 273.15
        k_eq = hts_equilibrium_constant(hts_t_c)
        for fname, val in [
            ("hts_k_eq", k_eq),
            ("hts_outlet_temp_c", hts_t_c),
        ]:
            if fname in feat_names:
                X[0, feat_names.index(fname)] = val
        return X

    # ── Trade-off curves ──────────────────────────────────────────────────────

    def generate_tradeoff_curves(self, df: pd.DataFrame) -> Dict:
        """
        Sweep S/C, HTS temperature, and Plant Rate independently.
        Returns a dict of {curve_name: {"x": array, "y_co_ppm": array, "current": float}}.
        """
        baseline, feat_names = self._get_baseline_features(df)

        # Current operating values
        def _cur(col):
            s = pd.to_numeric(df.get(col, pd.Series(dtype=float)), errors="coerce").dropna()
            return float(s.iloc[-1]) if len(s) else np.nan

        cur_sc   = _cur(self._SC_COL)
        cur_rate = _cur(self._RATE_COL)
        cur_hts  = _cur(self._HTS_T_COL)     # already °C in physics features
        cur_psa  = _cur(self._PSA_COL)

        curves = {}

        # ── Curve 1: CO vs S/C Ratio ─────────────────────────────────────────
        sc_sweep = np.arange(2.5, 5.6, 0.1)
        co_sc = self._sweep_feature(baseline, feat_names, self._SC_COL, sc_sweep)
        curves["sc_ratio"] = {
            "x": sc_sweep, "y_co_ppm": co_sc,
            "current_x": cur_sc, "current_co": _cur(self._CO_PRED_COL),
            "xlabel": "S/C Ratio (mol/mol)",
            "title": "CO Predicted vs S/C Ratio\n(all other parameters held constant)",
        }

        # ── Curve 2: CO vs HTS Outlet Temperature ────────────────────────────
        hts_sweep_c = np.arange(175.0, 265.0, 2.0)   # °C (≈ 350–510 °F)
        co_hts = self._sweep_feature(
            baseline, feat_names, "hts_outlet_temp_c", hts_sweep_c,
            linked_updater=self._hts_temp_updater,
        )
        curves["hts_temp"] = {
            "x": hts_sweep_c * 9/5 + 32,   # convert to °F for display
            "y_co_ppm": co_hts,
            "current_x": cur_hts * 9/5 + 32 if not np.isnan(cur_hts) else np.nan,
            "current_co": _cur(self._CO_PRED_COL),
            "xlabel": "HTS Outlet Temperature (°F)",
            "title": "CO Predicted vs HTS Outlet Temperature\n(thermodynamic + model effect)",
        }

        # ── Curve 3: CO vs Plant Rate ─────────────────────────────────────────
        rate_sweep = np.arange(50.0, 111.0, 2.0)
        co_rate = self._sweep_feature(baseline, feat_names, self._RATE_COL, rate_sweep)
        curves["plant_rate"] = {
            "x": rate_sweep, "y_co_ppm": co_rate,
            "current_x": cur_rate, "current_co": _cur(self._CO_PRED_COL),
            "xlabel": "Plant Rate (%)",
            "title": "CO Predicted vs Plant Rate\n(PSA contact-time effect)",
        }

        # ── Curve 4: CO vs PSA Recovery ──────────────────────────────────────
        psa_sweep = np.arange(70.0, 96.0, 1.0)
        co_psa = self._sweep_feature(baseline, feat_names, self._PSA_COL, psa_sweep)
        curves["psa_recovery"] = {
            "x": psa_sweep, "y_co_ppm": co_psa,
            "current_x": cur_psa, "current_co": _cur(self._CO_PRED_COL),
            "xlabel": "PSA Recovery (%)",
            "title": "CO Predicted vs PSA Recovery",
        }

        return curves

    def generate_operating_envelope(self, df: pd.DataFrame) -> Dict:
        """
        2D grid: Plant Rate (x) vs S/C Ratio (y), coloured by predicted CO ppm.
        Returns dict with meshgrid arrays and CO matrix for plotting.
        """
        baseline, feat_names = self._get_baseline_features(df)

        rate_vals = np.arange(50.0, 111.0, 5.0)
        sc_vals   = np.arange(2.5,  5.6,  0.25)

        rate_idx = feat_names.index(self._RATE_COL) if self._RATE_COL in feat_names else None
        sc_idx   = feat_names.index(self._SC_COL)   if self._SC_COL   in feat_names else None

        co_grid = np.full((len(sc_vals), len(rate_vals)), np.nan)

        if rate_idx is not None and sc_idx is not None:
            for i, sc in enumerate(sc_vals):
                for j, rate in enumerate(rate_vals):
                    X = baseline.copy()
                    X[0, sc_idx]   = sc
                    X[0, rate_idx] = rate
                    try:
                        co_grid[i, j] = self._predict_feature_vector(X)
                    except Exception:
                        co_grid[i, j] = np.nan

        RATE_GRID, SC_GRID = np.meshgrid(rate_vals, sc_vals)

        def _cur(col):
            s = pd.to_numeric(df.get(col, pd.Series(dtype=float)), errors="coerce").dropna()
            return float(s.iloc[-1]) if len(s) else np.nan

        return {
            "rate_grid": RATE_GRID,
            "sc_grid":   SC_GRID,
            "co_grid":   co_grid,
            "rate_vals": rate_vals,
            "sc_vals":   sc_vals,
            "current_rate": _cur(self._RATE_COL),
            "current_sc":   _cur(self._SC_COL),
            "current_co":   _cur(self._CO_PRED_COL),
        }

    # ── Recommendation engine ─────────────────────────────────────────────────

    def generate_recommendations(self, df: pd.DataFrame) -> List[Dict]:
        """
        Rule-based recommendations from the latest operating point.
        Each item: {priority, category, finding, action, saving}.
        """
        def _latest(col, default=np.nan):
            s = pd.to_numeric(df.get(col, pd.Series(dtype=float)), errors="coerce").dropna()
            return float(s.iloc[-1]) if len(s) else default

        co_pred   = _latest("CO_Predicted_ppm")
        co_meas   = _latest("CO in Product")
        sc        = _latest(self._SC_COL)
        rate      = _latest(self._RATE_COL)
        psa_rec   = _latest(self._PSA_COL)
        gross_eff = _latest(self._GROSS_COL)
        net_eff   = _latest(self._NET_COL)
        hts_cat   = _latest("HTS Catalyst Utilization (%)")
        approach  = _latest("approach_to_eq")
        excess_o2 = _latest("Excess O2 in Flue Gas")
        tube_temp = _latest("Tube Outlet Temperature")
        h2_purge  = _latest("H2 Lost to Purge")
        sc_excess = _latest("S/C Excess over Coking Min")
        h2_ng     = _latest("H2/NG Yield Ratio")
        steam_idx = _latest("Steam Efficiency Index")
        carb_eff  = _latest("Carbon Efficiency")
        prod_val  = _latest("Production Value Index")
        eff_gap   = _latest("Efficiency Gap to Design (BTU/SCF)")
        co_head   = _latest("CO Spec Headroom (Predicted)")

        recs = []

        # ── 1. Steam over-use (highest economic impact in most SMR plants) ────
        if not np.isnan(sc) and not np.isnan(co_pred):
            if sc > SC_MIN_NO_COKING + 0.5 and co_pred < CO_SPEC_LIMIT - 2.0:
                steam_saving = sc - (SC_MIN_NO_COKING + 0.3)
                recs.append({
                    "priority": 1,
                    "category": "Steam Optimisation",
                    "finding": (
                        f"S/C ratio {sc:.2f} mol/mol is {sc - SC_MIN_NO_COKING:.2f} above "
                        f"the coking floor ({SC_MIN_NO_COKING} mol/mol). "
                        f"Predicted CO {co_pred:.1f} ppm — {co_head:.1f} ppm of headroom to spec."
                    ),
                    "action": (
                        f"Reduce S/C ratio by ~{steam_saving:.1f} mol/mol toward {SC_MIN_NO_COKING + 0.3:.1f}. "
                        f"Monitor CO prediction and measured CO during reduction."
                    ),
                    "saving": "Reduces steam consumption; improves Net Efficiency.",
                })

        # ── 2. CO approaching spec — S/C may need to increase ─────────────────
        if not np.isnan(co_pred) and co_pred > CO_SPEC_LIMIT - 1.5:
            recs.append({
                "priority": 1,
                "category": "Product Quality",
                "finding": (
                    f"CO predicted at {co_pred:.1f} ppm — only {CO_SPEC_LIMIT - co_pred:.1f} ppm "
                    f"from spec limit ({CO_SPEC_LIMIT} ppm)."
                ),
                "action": (
                    "Consider increasing S/C ratio by 0.2–0.5 mol/mol to improve WGS conversion. "
                    "Check HTS inlet temperature — low temperature reduces catalyst activity."
                ),
                "saving": "Prevents off-spec product and potential customer interruption.",
            })

        # ── 3. Poor HTS catalyst utilization ─────────────────────────────────
        if not np.isnan(approach) and approach < 0.80:
            recs.append({
                "priority": 2,
                "category": "HTS Reactor",
                "finding": (
                    f"Approach-to-equilibrium {approach:.2f} — HTS is operating at "
                    f"{approach*100:.0f}% of thermodynamic potential. "
                    f"Significant unconverted CO is leaving the HTS bed."
                ),
                "action": (
                    "Check HTS inlet temperature (min 350°F for catalyst activity). "
                    "Verify S/C ratio is adequate for WGS. "
                    "If temperature and S/C are correct, catalyst may be deactivated."
                ),
                "saving": "Recovering HTS efficiency could reduce CO by 2–5 ppm.",
            })

        # ── 4. High excess O2 — fuel waste ────────────────────────────────────
        if not np.isnan(excess_o2) and excess_o2 > 4.0:
            o2_excess = excess_o2 - EXCESS_O2_DESIGN
            recs.append({
                "priority": 2,
                "category": "Combustion Efficiency",
                "finding": (
                    f"Excess O2 at {excess_o2:.1f}% — {o2_excess:.1f}% above design target "
                    f"({EXCESS_O2_DESIGN}%). Excess air carries heat out of the firebox."
                ),
                "action": (
                    f"Reduce excess O2 toward {EXCESS_O2_DESIGN:.0f}% by trimming combustion air. "
                    f"Maintain minimum {EXCESS_O2_MIN:.1f}% for safe combustion."
                ),
                "saving": f"Each 1% O2 reduction ≈ 0.5–1.0 BTU/SCF efficiency improvement.",
            })

        # ── 5. PSA recovery below target ──────────────────────────────────────
        if not np.isnan(psa_rec) and psa_rec < 85.0:
            recs.append({
                "priority": 2,
                "category": "PSA Optimisation",
                "finding": (
                    f"PSA recovery {psa_rec:.1f}% — below the 85% reference target. "
                    f"H2 lost to purge: {h2_purge:.2f} MSCFH."
                ),
                "action": (
                    "Check PGB pressure and PSA cycle timing. "
                    "Verify purge gas vent valve position. "
                    "If PGB pressure is low, raise setpoint to improve H2 retention."
                ),
                "saving": f"Recovering 1% PSA recovery adds ~{0.45 * rate / 100:.2f} MSCFH H2 production.",
            })

        # ── 6. Low plant rate with CO headroom (throughput opportunity) ────────
        if not np.isnan(rate) and not np.isnan(co_pred):
            if rate < 90.0 and co_pred < CO_SPEC_LIMIT - 3.0:
                recs.append({
                    "priority": 3,
                    "category": "Throughput Optimisation",
                    "finding": (
                        f"Plant rate {rate:.1f}% with {CO_SPEC_LIMIT - co_pred:.1f} ppm CO "
                        f"headroom to spec. There is capacity to increase throughput."
                    ),
                    "action": (
                        f"Consider increasing plant rate toward 90–95%. "
                        f"Monitor CO prediction closely as rate increases — "
                        f"PSA contact time reduces at higher rates."
                    ),
                    "saving": "Additional H2 production at marginal cost.",
                })

        # ── 7. Efficiency far from design ─────────────────────────────────────
        if not np.isnan(eff_gap) and eff_gap > 20.0:
            recs.append({
                "priority": 3,
                "category": "Overall Efficiency",
                "finding": (
                    f"Gross efficiency {gross_eff:.1f} BTU/SCF — "
                    f"{eff_gap:.1f} BTU/SCF above design ({DESIGN_GROSS_EFF} BTU/SCF). "
                    f"Net efficiency: {net_eff:.1f} BTU/SCF."
                ),
                "action": (
                    "Review all efficiency losses: excess O2, steam venting, heat exchanger fouling, "
                    "and catalyst performance. Cross-check material balances for metering errors."
                ),
                "saving": "Significant — each 10 BTU/SCF efficiency gain reduces operating cost.",
            })

        # ── 8. Low H2/NG yield ────────────────────────────────────────────────
        if not np.isnan(h2_ng) and h2_ng < 2.8:
            recs.append({
                "priority": 3,
                "category": "Conversion Efficiency",
                "finding": (
                    f"H2/NG yield ratio {h2_ng:.2f} SCF/SCF — below typical range of 2.8–3.5. "
                    f"Carbon efficiency: {carb_eff:.1f}%."
                ),
                "action": (
                    "Check reformer tube outlet temperature for adequate methane conversion. "
                    "Verify S/C ratio is above 3.0 for good steam reforming. "
                    "Review catalyst activity — may require regeneration or replacement."
                ),
                "saving": "Each 0.1 improvement in H2/NG ratio = significant feed cost reduction.",
            })

        # Sort by priority (1 = most urgent)
        recs.sort(key=lambda r: r["priority"])
        return recs

    # ── Plots ─────────────────────────────────────────────────────────────────

    def plot_tradeoff_curves(self, curves: Dict):
        """4-panel trade-off curves: CO ppm vs S/C, HTS Temp, Plant Rate, PSA Recovery."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("CO-in-Product Trade-off Curves\n(each curve: one parameter swept, all others at current operating point)",
                     fontsize=12, fontweight="bold")

        panel_keys = ["sc_ratio", "hts_temp", "plant_rate", "psa_recovery"]
        for ax, key in zip(axes.flat, panel_keys):
            if key not in curves:
                ax.set_visible(False)
                continue
            c = curves[key]
            x, y = c["x"], c["y_co_ppm"]
            valid = np.isfinite(y)
            ax.plot(x[valid], y[valid], color="#1f77b4", linewidth=1.8)
            ax.axhline(CO_SPEC_LIMIT, color="red",    linestyle="--", linewidth=1.0, label=f"Spec {CO_SPEC_LIMIT} ppm")
            ax.axhline(CO_ALERT_SOFT, color="orange", linestyle="--", linewidth=1.0, label=f"Alert {CO_ALERT_SOFT} ppm")

            # Current operating point marker
            cx, cy = c.get("current_x"), c.get("current_co")
            if cx is not None and not np.isnan(cx) and cy is not None and not np.isnan(cy):
                ax.scatter([cx], [cy], color="red", zorder=5, s=60, label="Current")

            ax.set_xlabel(c["xlabel"], fontsize=9)
            ax.set_ylabel("CO Predicted (ppm)", fontsize=9)
            ax.set_title(c["title"], fontsize=9)
            ax.legend(fontsize=7)
            ax.set_ylim(bottom=0)

        fig.tight_layout()
        path = PLOT_DIR / "01_tradeoff_curves.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path.name}")

    def plot_operating_envelope(self, env: Dict):
        """2D operating envelope: Plant Rate vs S/C, coloured by CO ppm."""
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.suptitle("Operating Envelope: Plant Rate vs S/C Ratio\n(colour = predicted CO ppm)",
                     fontsize=12, fontweight="bold")

        co_grid = env["co_grid"]
        vmin, vmax = 0, max(20, np.nanmax(co_grid))
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "co_cmap", ["#2ca02c", "#ffff00", "#ff7f0e", "#d62728"]
        )
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        im = ax.pcolormesh(
            env["rate_grid"], env["sc_grid"], co_grid,
            cmap=cmap, norm=norm, shading="auto", alpha=0.85,
        )
        cbar = fig.colorbar(im, ax=ax, label="CO Predicted (ppm)")
        cbar.ax.axhline(CO_SPEC_LIMIT, color="red",    linewidth=1.5, linestyle="--")
        cbar.ax.axhline(CO_ALERT_SOFT, color="orange", linewidth=1.0, linestyle="--")

        # Constraint boundary: CO = 10 ppm
        try:
            ax.contour(env["rate_grid"], env["sc_grid"], co_grid,
                       levels=[CO_SPEC_LIMIT], colors=["red"], linewidths=2,
                       linestyles=["--"])
            ax.contour(env["rate_grid"], env["sc_grid"], co_grid,
                       levels=[CO_ALERT_SOFT], colors=["orange"], linewidths=1.5,
                       linestyles=["--"])
        except Exception:
            pass

        # Coking floor
        ax.axhline(SC_MIN_NO_COKING, color="brown", linestyle=":", linewidth=1.5,
                   label=f"Min S/C (coking) = {SC_MIN_NO_COKING}")

        # Current operating point
        cx, cy = env.get("current_rate"), env.get("current_sc")
        if cx and cy and not np.isnan(cx) and not np.isnan(cy):
            ax.scatter([cx], [cy], color="white", s=120, zorder=6,
                       edgecolors="black", linewidths=1.5, label="Current operating point")

        ax.set_xlabel("Plant Rate (%)", fontsize=11)
        ax.set_ylabel("S/C Ratio (mol/mol)", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")

        # Annotations
        ax.text(0.02, 0.97,
                f"Red dashed  = CO {CO_SPEC_LIMIT} ppm spec limit\n"
                f"Orange dashed = CO {CO_ALERT_SOFT} ppm alert threshold\n"
                f"Green region = low CO (good quality)",
                transform=ax.transAxes, fontsize=7,
                verticalalignment="top", bbox=dict(boxstyle="round", alpha=0.3))

        fig.tight_layout()
        path = PLOT_DIR / "02_operating_envelope.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path.name}")

    def plot_opt_kpi_trends(self, df: pd.DataFrame):
        """Time-series of the key optimization KPIs over the dataset period."""
        ts = pd.to_datetime(df.get("Timestamp", pd.Series(dtype=str)), errors="coerce")

        fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
        fig.suptitle("Optimization KPI Trends", fontsize=12, fontweight="bold")

        panels = [
            ("CO Spec Headroom (Predicted)",    "ppm",     "CO Spec Headroom\n(+ve = margin, -ve = over spec)",  "red",     0),
            ("HTS Catalyst Utilization (%)",    "%",       "HTS Catalyst Utilization\n(100% = at equilibrium)",  "steelblue", None),
            ("Steam Cost Index",                "index",   "Steam Cost Index\n(100 = minimum safe S/C)",          "darkorange", 100),
            ("H2/NG Yield Ratio",               "SCF/SCF", "H2/NG Yield Ratio\n(higher = better)",               "green",   None),
            ("Efficiency Gap to Design (BTU/SCF)", "BTU/SCF", "Efficiency Gap to Design\n(0 = at design)",        "purple",  0),
            ("Production Value Index",          "%",       "Production Value Index\n(Rate × Recovery / 100)",     "#1f77b4", None),
        ]

        import matplotlib.dates as mdates
        for ax, (col, unit, title, color, ref_line) in zip(axes.flat, panels):
            if col not in df.columns:
                ax.set_visible(False)
                continue
            s = pd.to_numeric(df[col], errors="coerce")
            ax.plot(ts, s, color=color, linewidth=0.7, alpha=0.8)
            if ref_line is not None:
                ax.axhline(ref_line, color="gray", linestyle="--", linewidth=0.8)
            ax.set_ylabel(unit, fontsize=8)
            ax.set_title(title, fontsize=9)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))

        fig.tight_layout()
        path = PLOT_DIR / "03_opt_kpi_trends.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path.name}")

    # ── Full analysis pipeline ────────────────────────────────────────────────

    def run_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Run the complete optimization analysis pipeline.
        Returns a dict with curves, envelope, and recommendations.
        Saves all plots to model_plots/optimization/.
        """
        print("\nGenerating trade-off curves ...")
        curves = self.generate_tradeoff_curves(df)
        self.plot_tradeoff_curves(curves)

        print("Generating 2D operating envelope ...")
        envelope = self.generate_operating_envelope(df)
        self.plot_operating_envelope(envelope)

        print("Generating optimization KPI trends ...")
        self.plot_opt_kpi_trends(df)

        print("Generating recommendations ...")
        recs = self.generate_recommendations(df)
        _print_recommendations(recs)

        return {"curves": curves, "envelope": envelope, "recommendations": recs}


# ── Console output ────────────────────────────────────────────────────────────

def _print_recommendations(recs: List[Dict]):
    print("\n" + "=" * 70)
    print("  PROCESS OPTIMISATION RECOMMENDATIONS")
    print("=" * 70)
    if not recs:
        print("  No recommendations — plant operating near optimal conditions.")
        return
    for i, r in enumerate(recs, 1):
        tag = {1: "[HIGH]", 2: "[MED ]", 3: "[LOW ]"}.get(r["priority"], "[    ]")
        print(f"\n  {i}. {tag}  {r['category']}")
        print(f"     Finding : {r['finding']}")
        print(f"     Action  : {r['action']}")
        print(f"     Saving  : {r['saving']}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main(data_path: Path = None, predictor=None) -> Dict:
    """
    Standalone entry point — trains / loads COPredictor then runs full analysis.
    If predictor is provided, uses it directly.
    """
    from pathlib import Path as _Path
    if data_path is None:
        data_path = _ROOT / "Combined_Data_with_KPIs.csv"
    if not data_path.exists():
        data_path = _ROOT / "Combined_Data.csv"

    print(f"\nLoading data: {data_path.name}")
    df = pd.read_csv(data_path, encoding="latin-1")
    df["Timestamp"] = pd.to_datetime(df.get("Timestamp", df.get("Date:", "")), errors="coerce")

    if predictor is None:
        from co_product_model import COPredictor
        pkl = _ROOT / "co_predictor.pkl"
        if pkl.exists():
            predictor = COPredictor.load(pkl)
        else:
            from co_product_model import main as train_co
            predictor = train_co(data_path)

    # Add post-ML opt KPIs
    print("Computing post-ML optimization KPIs ...")
    df = compute_opt_kpis(df)

    analyzer = OptimizationAnalyzer(predictor)
    result   = analyzer.run_analysis(df)
    result["df_opt"] = df
    return result


if __name__ == "__main__":
    main()
