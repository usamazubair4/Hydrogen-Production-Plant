"""
Compressor Reliability ML Module
==================================
Unsupervised anomaly detection and composite health scoring for
Compressors A, B and C on the SMR Plant.

Approach — no labelled failure data is available, so we use:

  1. Composite Health Index (0-100, higher = healthier)
       Weighted combination of four sub-scores:
         Bearing thermal score  — 35 %
         Vibration score        — 25 %
         Oil system score       — 25 %
         Compression eff. score — 15 %

  2. Isolation Forest (sklearn)
       Trained on the first 60% of each compressor's running hours
       as a "nominally healthy baseline".  Scored on the full dataset.
       anomaly_score in [-1, 0]: closer to -1 = more anomalous

  3. Rolling EWMA control chart
       Exponentially weighted moving average + 3-sigma band on the
       composite health index; deviation from band triggers drift alert.

Output columns appended to the DataFrame:
  Compressor_A_Health      — 0-100 composite health index
  Compressor_A_Anomaly     — Isolation Forest flag (0=normal, 1=anomaly)
  Compressor_A_Alert       — green / amber / red
  Compressor_A_Vib_Score   — vibration sub-score (0-100)
  Compressor_A_Bear_Score  — bearing thermal sub-score (0-100)
  Compressor_A_Oil_Score   — oil system sub-score (0-100)
  Compressor_A_Cr_Score    — compression efficiency sub-score (0-100)
  (and same for B, C)

Usage
------
  from compressor_reliability import CompressorHealthMonitor, main
  monitor = main("Combined_Data_with_KPIs.csv")   # trains & saves
  # — or —
  monitor = CompressorHealthMonitor.load("compressor_monitor.pkl")
  result  = monitor.predict_realtime({"Compressor A Motor Current": 530, ...})
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

_ROOT    = Path(__file__).resolve().parent
DATA_CSV = _ROOT / "Combined_Data.csv"
MODEL_PKL= _ROOT / "compressor_monitor.pkl"
PLOT_DIR = _ROOT / "model_plots" / "compressor"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

COMPRESSORS = ["A", "B", "C"]

# ── Sensor map identical to compressor_eda.py ─────────────────────────────────
SENSOR_MAP: Dict[str, Dict[str, str]] = {}
for _comp in COMPRESSORS:
    c = f"Compressor {_comp}"
    SENSOR_MAP[_comp] = {
        "current":        f"{c} Motor Current",
        "vib_motor_de":   f"{c} Motor DE Vibration",
        "vib_motor_ode":  f"{c} Motor ODE Vibration",
        "vib_frame_de":   f"{c} Frame DE Vibration",
        "vib_frame_ode":  f"{c} Frame ODE Vibration",
        "vib_ic1":        f"{c} Interstage Cooler Vibration",
        "vib_ic2":        f"{c} Interstage Cooler Vibration.1",
        "bear_hot":       f"{c} Hottest Bearing Temperature",
        "bear_1":         f"{c} Bearing Temperature #1",
        "bear_2":         f"{c} Bearing Temperature #2",
        "bear_3":         f"{c} Bearing Temperature #3",
        "bear_4":         f"{c} Bearing Temperature #4",
        "bear_5":         f"{c} Bearing Temperature #5",
        "bear_6":         f"{c} Bearing Temperature #6",
        "stator_hot":     f"{c} Hottest Stator Temperature",
        "oil_dp":         f"{c} Oil Filter dP",
        "oil_press":      f"{c} Oil Pressure",
        "oil_temp":       f"{c} Oil Temperature",
        "cyl_fg1":        f"{c} 1st Stage Feed Gas Cylinder Temperature",
        "cyl_fg2":        f"{c} 2nd Stage Feed Gas Cylinder Temperature",
        "cyl_h2_1":       f"{c} 1st Stage H2 Cylinder Temperature",
        "cyl_h2_2":       f"{c} 2nd Stage H2 Cylinder Temperature",
        "cyl_h2_3":       f"{c} 3rd Stage H2 Cylinder Temperature",
        "cr_h2_1":        f"{c} 1st Stage H2 Compression Ratio",
        "cr_h2_2":        f"{c} 2nd Stage H2 Compression Ratio",
        "cr_h2_3":        f"{c} 3rd Stage H2 Compression Ratio",
    }

# Alarm thresholds
VIB_ALERT_MM_S  = 0.71    # ISO 10816 Zone B/C boundary
VIB_TRIP_MM_S   = 1.12    # ISO 10816 Zone D
BEAR_ALERT_F    = 180.0   # bearing temperature alert (°F)
BEAR_TRIP_F     = 220.0   # bearing temperature trip  (°F)
OIL_DP_ALERT    = 12.0    # oil filter dP alert (psi)
OIL_DP_MAX      = 15.0    # oil filter dP trip / max
OIL_TEMP_ALERT  = 170.0   # oil temp alert (°F)
OIL_TEMP_MAX    = 185.0   # oil temp max (°F)
OIL_PRESS_MIN   = 55.0    # oil pressure low alarm (psig)
CR_MIN_NORMAL   = 1.2     # compression ratio below this = poor efficiency
RUNNING_THR: Dict[str, float] = {"A": 100.0, "B": 400.0, "C": 200.0}

# Health index weights
WEIGHTS = {"bear": 0.35, "vib": 0.25, "oil": 0.25, "cr": 0.15}

# Isolation Forest hyper-parameters
IF_CONTAMINATION = 0.05   # expected fraction of anomalies in running data
IF_N_ESTIMATORS  = 200
BASELINE_FRACTION= 0.60   # first 60% of running rows used for training


# ── Feature engineering ───────────────────────────────────────────────────────

def _get(row_or_series, col_name: str, default: float = np.nan) -> float:
    """Safe column accessor for both dict and Series."""
    try:
        val = row_or_series[col_name]
        return float(val) if pd.notna(val) else default
    except (KeyError, TypeError, ValueError):
        return default


def compute_compressor_features(df: pd.DataFrame, comp: str) -> pd.DataFrame:
    """
    Build a feature DataFrame for one compressor.
    Returns one row per input row with derived reliability features.
    """
    sm = SENSOR_MAP[comp]

    def gcol(key):
        col = sm.get(key, "")
        return pd.to_numeric(df[col], errors="coerce") if col and col in df.columns else pd.Series(np.nan, index=df.index)

    feats = pd.DataFrame(index=df.index)

    # ── Vibration features ────────────────────────────────────────────────────
    vib_cols = ["vib_motor_de", "vib_motor_ode", "vib_frame_de", "vib_frame_ode", "vib_ic1", "vib_ic2"]
    vib_mat  = pd.concat([gcol(k) for k in vib_cols], axis=1)
    feats["vib_max"]  = vib_mat.max(axis=1)
    feats["vib_mean"] = vib_mat.mean(axis=1)
    feats["vib_rms"]  = np.sqrt((vib_mat ** 2).mean(axis=1))

    # ── Bearing thermal features ───────────────────────────────────────────────
    bear_keys = ["bear_1", "bear_2", "bear_3", "bear_4", "bear_5", "bear_6"]
    bear_mat  = pd.concat([gcol(k) for k in bear_keys], axis=1)
    feats["bear_mean"]   = bear_mat.mean(axis=1)
    feats["bear_max"]    = bear_mat.max(axis=1)
    feats["bear_spread"] = bear_mat.max(axis=1) - bear_mat.min(axis=1)  # asymmetry indicator
    feats["bear_hot"]    = gcol("bear_hot")

    # Rolling rate of change for hottest bearing (30-step = ~7.5 h at 15-min data)
    feats["bear_delta"]  = feats["bear_hot"].diff(30)

    # ── Oil system features ────────────────────────────────────────────────────
    feats["oil_dp"]      = gcol("oil_dp")
    feats["oil_press"]   = gcol("oil_press")
    feats["oil_temp"]    = gcol("oil_temp")

    # Rolling trend of oil filter dP (fouling leading indicator)
    feats["oil_dp_delta"] = feats["oil_dp"].diff(48)  # 12-hour rate

    # ── Compression efficiency features ────────────────────────────────────────
    feats["cr_h2_1"] = gcol("cr_h2_1")
    feats["cr_h2_2"] = gcol("cr_h2_2")
    feats["cr_h2_3"] = gcol("cr_h2_3")
    feats["cr_mean"] = pd.concat([gcol("cr_h2_1"), gcol("cr_h2_2"), gcol("cr_h2_3")], axis=1).mean(axis=1)

    # ── Motor load ─────────────────────────────────────────────────────────────
    feats["current"]       = gcol("current")
    feats["current_delta"] = feats["current"].diff(8)  # 2-hour rate

    return feats


# ── Health sub-scores (0-100) ─────────────────────────────────────────────────

def _clip_score(value, low_bad, high_good, invert=False) -> float:
    """Map value linearly from [low_bad, high_good] → [0, 100].
    invert=True: low value is good (e.g. vibration, temperature)."""
    if pd.isna(value):
        return 50.0  # neutral if data missing
    if invert:
        # high value = bad score
        score = 100.0 * (1.0 - np.clip((value - low_bad) / (high_good - low_bad), 0, 1))
    else:
        score = 100.0 * np.clip((value - low_bad) / (high_good - low_bad), 0, 1)
    return float(score)


def vibration_score(row: pd.Series) -> float:
    """0–100; penalises for high peak vibration and high RMS."""
    peak = row.get("vib_max",  0.0) or 0.0
    rms  = row.get("vib_rms",  0.0) or 0.0
    s_peak = _clip_score(peak, 0.0,  VIB_TRIP_MM_S,  invert=True)
    s_rms  = _clip_score(rms,  0.0,  VIB_ALERT_MM_S, invert=True)
    return 0.6 * s_peak + 0.4 * s_rms


def bearing_score(row: pd.Series) -> float:
    """0–100; penalises for high bearing temperature, spread, and rate of rise."""
    temp   = row.get("bear_max",    np.nan)
    spread = row.get("bear_spread", np.nan)
    delta  = row.get("bear_delta",  0.0) or 0.0

    s_temp   = _clip_score(temp,   100.0, BEAR_TRIP_F,  invert=True)
    s_spread = _clip_score(spread,   0.0, 30.0,          invert=True)  # >30°F spread = asymmetric load
    s_delta  = _clip_score(abs(delta), 0.0, 15.0,         invert=True)  # >15°F/7.5h = rapid rise

    return 0.5 * s_temp + 0.3 * s_spread + 0.2 * s_delta


def oil_score(row: pd.Series) -> float:
    """0–100; penalises for high filter dP, low oil pressure, high oil temp."""
    dp       = row.get("oil_dp",    np.nan)
    pressure = row.get("oil_press", np.nan)
    temp     = row.get("oil_temp",  np.nan)
    dp_trend = row.get("oil_dp_delta", 0.0) or 0.0

    s_dp      = _clip_score(dp,       0.0,  OIL_DP_MAX,    invert=True)
    s_press   = _clip_score(pressure, OIL_PRESS_MIN, 70.0,  invert=False)
    s_temp    = _clip_score(temp,     100.0, OIL_TEMP_MAX,  invert=True)
    s_dp_trnd = _clip_score(abs(dp_trend), 0.0, 5.0,        invert=True)

    return 0.35 * s_dp + 0.25 * s_press + 0.25 * s_temp + 0.15 * s_dp_trnd


def compression_score(row: pd.Series) -> float:
    """0–100; rewards stable inter-stage compression ratios near design point."""
    cr1 = row.get("cr_h2_1", np.nan)
    cr2 = row.get("cr_h2_2", np.nan)
    cr3 = row.get("cr_h2_3", np.nan)
    scores = []
    for cr in [cr1, cr2, cr3]:
        if pd.isna(cr):
            scores.append(50.0)
        elif cr < 1.0:
            scores.append(20.0)   # reversal — very bad
        elif cr < CR_MIN_NORMAL:
            scores.append(_clip_score(cr, 1.0, CR_MIN_NORMAL, invert=False) * 0.6)
        else:
            scores.append(min(100.0, 60.0 + 40.0 * min(1.0, (cr - CR_MIN_NORMAL) / 1.0)))
    return float(np.mean(scores))


def composite_health_index(feats_row: pd.Series) -> Tuple[float, float, float, float, float]:
    """Return (HI, bear_s, vib_s, oil_s, cr_s) all in 0-100."""
    bear_s = bearing_score(feats_row)
    vib_s  = vibration_score(feats_row)
    oil_s  = oil_score(feats_row)
    cr_s   = compression_score(feats_row)
    hi     = (WEIGHTS["bear"] * bear_s + WEIGHTS["vib"] * vib_s +
               WEIGHTS["oil"] * oil_s  + WEIGHTS["cr"] * cr_s)
    return round(hi, 1), round(bear_s, 1), round(vib_s, 1), round(oil_s, 1), round(cr_s, 1)


def _alert_level(hi: float) -> str:
    if hi >= 75:
        return "green"
    if hi >= 50:
        return "amber"
    return "red"


# ── Isolation Forest wrapper ───────────────────────────────────────────────────

class IFWrapper:
    """Per-compressor Isolation Forest with fitted scaler."""

    FEATURE_KEYS = [
        "vib_max", "vib_rms",
        "bear_max", "bear_spread", "bear_delta",
        "oil_dp", "oil_press", "oil_temp", "oil_dp_delta",
        "cr_h2_1", "cr_h2_2", "cr_h2_3",
        "current",
    ]

    def __init__(self):
        self.scaler = RobustScaler()
        self.model  = IsolationForest(
            n_estimators=IF_N_ESTIMATORS,
            contamination=IF_CONTAMINATION,
            random_state=42,
            n_jobs=-1,
        )
        self.is_fitted = False
        self.baseline_stats: Dict[str, float] = {}

    def _get_matrix(self, feats: pd.DataFrame) -> np.ndarray:
        avail = [k for k in self.FEATURE_KEYS if k in feats.columns]
        X = feats[avail].values.astype(float)
        X = np.where(np.isfinite(X), X, np.nanmedian(X, axis=0))
        return X

    def fit(self, feats: pd.DataFrame, running_mask: pd.Series):
        run_feats = feats[running_mask].dropna(how="all").reset_index(drop=True)
        n_base    = int(len(run_feats) * BASELINE_FRACTION)
        baseline  = run_feats.iloc[:n_base]

        X_base = self._get_matrix(baseline)
        X_scaled = self.scaler.fit_transform(X_base)
        self.model.fit(X_scaled)
        self.is_fitted = True

        # Store baseline statistics for reference
        for col in self.FEATURE_KEYS:
            if col in baseline.columns:
                self.baseline_stats[col] = float(baseline[col].median())

        return self

    def score(self, feats: pd.DataFrame, running_mask: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          anomaly_flag   — 1=anomaly, 0=normal (NaN where not running)
          anomaly_score  — raw IF decision function [-1, 0], lower=more anomalous
        """
        flag  = np.full(len(feats), np.nan)
        score = np.full(len(feats), np.nan)

        if not self.is_fitted:
            return flag, score

        run_idx = feats.index[running_mask]
        if len(run_idx) == 0:
            return flag, score

        X = self._get_matrix(feats.loc[run_idx])
        X_scaled = self.scaler.transform(X)

        preds   = self.model.predict(X_scaled)        # +1 normal, -1 anomaly
        scores_ = self.model.decision_function(X_scaled)  # higher = more normal

        for i, idx in enumerate(run_idx):
            flag[feats.index.get_loc(idx)]  = 1 if preds[i] == -1 else 0
            score[feats.index.get_loc(idx)] = scores_[i]

        return flag, score


# ── Main monitor class ─────────────────────────────────────────────────────────

class CompressorHealthMonitor:
    """
    Train once; score every row in the dataset.
    Real-time inference via predict_realtime(sensor_dict).
    """

    def __init__(self):
        self.if_models: Dict[str, IFWrapper] = {c: IFWrapper() for c in COMPRESSORS}
        self.is_fitted = False
        self.feature_cache: Dict[str, pd.DataFrame] = {}

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "CompressorHealthMonitor":
        print("  Training Isolation Forest per compressor ...")
        for comp in COMPRESSORS:
            cur_col = SENSOR_MAP[comp].get("current", "")
            if cur_col not in df.columns:
                print(f"    Comp {comp}: current column missing — skipping IF training")
                continue

            running_mask = pd.to_numeric(df[cur_col], errors="coerce") > RUNNING_THR[comp]
            feats        = compute_compressor_features(df, comp)
            self.if_models[comp].fit(feats, running_mask)
            n_run = running_mask.sum()
            n_base= int(n_run * BASELINE_FRACTION)
            print(f"    Comp {comp}: {n_run:,} running rows  |  {n_base:,} baseline rows")

        self.is_fitted = True
        return self

    # ── Score full DataFrame ──────────────────────────────────────────────────

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute health index, sub-scores, and anomaly flags for every row.
        Returns df with new columns appended.
        """
        result = df.copy()

        for comp in COMPRESSORS:
            cur_col      = SENSOR_MAP[comp].get("current", "")
            running_mask = (
                pd.to_numeric(result[cur_col], errors="coerce") > RUNNING_THR[comp]
                if cur_col in result.columns else
                pd.Series(False, index=result.index)
            )

            feats = compute_compressor_features(result, comp)
            self.feature_cache[comp] = feats

            # Per-row health scores
            hi_vals    = []
            bear_vals  = []
            vib_vals   = []
            oil_vals   = []
            cr_vals    = []
            alert_vals = []

            for idx in result.index:
                if running_mask[idx]:
                    hi, bs, vs, os, cs = composite_health_index(feats.loc[idx])
                else:
                    hi, bs, vs, os, cs = np.nan, np.nan, np.nan, np.nan, np.nan
                hi_vals.append(hi)
                bear_vals.append(bs)
                vib_vals.append(vs)
                oil_vals.append(os)
                cr_vals.append(cs)
                alert_vals.append(_alert_level(hi) if pd.notna(hi) else "offline")

            result[f"Compressor_{comp}_Health"]     = hi_vals
            result[f"Compressor_{comp}_Bear_Score"] = bear_vals
            result[f"Compressor_{comp}_Vib_Score"]  = vib_vals
            result[f"Compressor_{comp}_Oil_Score"]  = oil_vals
            result[f"Compressor_{comp}_Cr_Score"]   = cr_vals
            result[f"Compressor_{comp}_Alert"]      = alert_vals

            # Isolation Forest anomaly scores
            flag, score = self.if_models[comp].score(feats, running_mask)
            result[f"Compressor_{comp}_Anomaly"]       = flag
            result[f"Compressor_{comp}_Anomaly_Score"] = np.round(score, 4)

        return result

    # ── Real-time single-row inference ────────────────────────────────────────

    def predict_realtime(self, sensor_dict: dict) -> dict:
        """
        Real-time health assessment from a dict of sensor readings.

        sensor_dict keys should be the exact DCS tag names, e.g.:
          {"Compressor A Motor Current": 530.0,
           "Compressor A Hottest Bearing Temperature": 165.0, ...}

        Returns a dict with per-compressor health summary.
        """
        row = pd.Series(sensor_dict)
        output = {}

        for comp in COMPRESSORS:
            cur_col = SENSOR_MAP[comp]["current"]
            current = _get(row, cur_col, 0.0)
            running = current > RUNNING_THR[comp]

            if not running:
                output[comp] = {
                    "status": "offline",
                    "health_index": None,
                    "alert": "offline",
                    "bear_score": None,
                    "vib_score": None,
                    "oil_score": None,
                    "cr_score": None,
                    "anomaly": None,
                    "key_concerns": [],
                }
                continue

            # Build feature row
            sm = SENSOR_MAP[comp]
            def g(key): return _get(row, sm.get(key, ""), np.nan)

            vib_vals = [g(k) for k in ["vib_motor_de", "vib_motor_ode",
                                        "vib_frame_de", "vib_frame_ode",
                                        "vib_ic1",      "vib_ic2"]]
            vib_valid = [v for v in vib_vals if not np.isnan(v)]
            bear_vals = [g(f"bear_{i}") for i in range(1, 7)]
            bear_valid = [v for v in bear_vals if not np.isnan(v)]
            cr_vals   = [g(f"cr_h2_{i}") for i in range(1, 4)]

            feat_row = pd.Series({
                "vib_max":       max(vib_valid)           if vib_valid  else np.nan,
                "vib_mean":      np.mean(vib_valid)       if vib_valid  else np.nan,
                "vib_rms":       np.sqrt(np.mean(np.square(vib_valid))) if vib_valid else np.nan,
                "bear_max":      max(bear_valid)           if bear_valid else np.nan,
                "bear_mean":     np.mean(bear_valid)       if bear_valid else np.nan,
                "bear_spread":   max(bear_valid) - min(bear_valid) if len(bear_valid) >= 2 else np.nan,
                "bear_hot":      g("bear_hot"),
                "bear_delta":    0.0,   # no history available in real-time
                "oil_dp":        g("oil_dp"),
                "oil_press":     g("oil_press"),
                "oil_temp":      g("oil_temp"),
                "oil_dp_delta":  0.0,   # no history available in real-time
                "cr_h2_1":       cr_vals[0],
                "cr_h2_2":       cr_vals[1],
                "cr_h2_3":       cr_vals[2],
                "cr_mean":       np.nanmean(cr_vals),
                "current":       current,
                "current_delta": 0.0,
            })

            hi, bs, vs, os, cs = composite_health_index(feat_row)
            alert = _alert_level(hi)

            # Identify key concerns
            concerns = []
            if not np.isnan(feat_row["vib_max"]) and feat_row["vib_max"] > VIB_ALERT_MM_S:
                concerns.append(f"High vibration: {feat_row['vib_max']:.3f} mm/s (alert>{VIB_ALERT_MM_S})")
            if not np.isnan(feat_row["bear_max"]) and feat_row["bear_max"] > BEAR_ALERT_F:
                concerns.append(f"High bearing temp: {feat_row['bear_max']:.0f}F (alert>{BEAR_ALERT_F})")
            if not np.isnan(feat_row["bear_spread"]) and feat_row["bear_spread"] > 20:
                concerns.append(f"Bearing temp spread: {feat_row['bear_spread']:.0f}F (>20F = asymmetric load)")
            if not np.isnan(feat_row["oil_dp"]) and feat_row["oil_dp"] > OIL_DP_ALERT:
                concerns.append(f"Oil filter dP high: {feat_row['oil_dp']:.1f} psi (alert>{OIL_DP_ALERT})")
            if not np.isnan(feat_row["oil_temp"]) and feat_row["oil_temp"] > OIL_TEMP_ALERT:
                concerns.append(f"Oil temp high: {feat_row['oil_temp']:.0f}F (alert>{OIL_TEMP_ALERT})")
            if not np.isnan(feat_row["oil_press"]) and feat_row["oil_press"] < OIL_PRESS_MIN:
                concerns.append(f"Oil pressure low: {feat_row['oil_press']:.0f} psig (min={OIL_PRESS_MIN})")

            # Isolation Forest score (if fitted and scaler available)
            if_score = None
            if self.if_models[comp].is_fitted:
                try:
                    fdf = feat_row.to_frame().T
                    avail = [k for k in IFWrapper.FEATURE_KEYS if k in fdf.columns]
                    X = fdf[avail].values.astype(float)
                    X = np.where(np.isfinite(X), X, 0.0)
                    X_sc = self.if_models[comp].scaler.transform(X)
                    if_score = float(self.if_models[comp].model.decision_function(X_sc)[0])
                    if_flag  = int(self.if_models[comp].model.predict(X_sc)[0] == -1)
                except Exception:
                    if_flag = None
            else:
                if_flag = None

            output[comp] = {
                "status":        "running",
                "health_index":  hi,
                "alert":         alert,
                "bear_score":    bs,
                "vib_score":     vs,
                "oil_score":     os,
                "cr_score":      cs,
                "anomaly":       if_flag,
                "if_score":      if_score,
                "key_concerns":  concerns,
                "motor_current": round(current, 1),
            }

        return output

    def print_health_report(self, result: dict, timestamp: str = "Latest"):
        print(f"\n  {'='*65}")
        print(f"  COMPRESSOR FLEET HEALTH REPORT  —  {timestamp}")
        print(f"  {'='*65}")
        for comp in COMPRESSORS:
            r = result[comp]
            if r["status"] == "offline":
                print(f"  Compressor {comp}:  OFFLINE / STANDBY")
            else:
                alert_sym = {"green": "[OK]", "amber": "[!!]", "red": "[XX]"}[r["alert"]]
                print(f"\n  Compressor {comp}:  HI={r['health_index']:.0f}/100  {alert_sym}  {r['alert'].upper()}")
                print(f"    Bear Score : {r['bear_score']:.0f}   Vib Score: {r['vib_score']:.0f}"
                      f"   Oil Score: {r['oil_score']:.0f}   CR Score: {r['cr_score']:.0f}")
                if r["anomaly"] is not None:
                    print(f"    Isolation Forest: {'ANOMALY' if r['anomaly'] else 'Normal'}  (IF score={r['if_score']:.3f})")
                if r["key_concerns"]:
                    print(f"    Concerns:")
                    for c in r["key_concerns"]:
                        print(f"      - {c}")
        print(f"  {'='*65}\n")

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: Path = MODEL_PKL):
        joblib.dump(self, path)
        print(f"  CompressorHealthMonitor saved -> {path}")

    @classmethod
    def load(cls, path: Path = MODEL_PKL) -> "CompressorHealthMonitor":
        obj = joblib.load(path)
        print(f"  CompressorHealthMonitor loaded <- {path}")
        return obj


# ── Diagnostic plots ──────────────────────────────────────────────────────────

def plot_health_timeseries(df_scored: pd.DataFrame, ts_col: str = "Timestamp"):
    """Plot health index for A/B/C over time with anomaly markers."""
    colors = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("Compressor Health Index (0-100) Over Time", fontsize=13, fontweight="bold")

    for ax, comp in zip(axes, COMPRESSORS):
        hi_col   = f"Compressor_{comp}_Health"
        anom_col = f"Compressor_{comp}_Anomaly"

        ts = pd.to_datetime(df_scored[ts_col], errors="coerce")
        hi = pd.to_numeric(df_scored[hi_col], errors="coerce")

        ax.plot(ts, hi, color=colors[comp], linewidth=0.8, alpha=0.8, label=f"Comp {comp} HI")

        # Anomaly markers
        if anom_col in df_scored.columns:
            anom_mask = df_scored[anom_col] == 1
            ax.scatter(ts[anom_mask], hi[anom_mask], color="red", s=8, alpha=0.6, label="Anomaly", zorder=5)

        ax.axhline(75, color="green",  linestyle="--", linewidth=0.8, alpha=0.6, label="HI=75 (green)")
        ax.axhline(50, color="orange", linestyle="--", linewidth=0.8, alpha=0.6, label="HI=50 (amber)")
        ax.set_ylim(-5, 105)
        ax.set_ylabel(f"Comp {comp}\nHealth Index", fontsize=9)
        ax.legend(fontsize=7, loc="lower left", ncol=4)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))

    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    path = PLOT_DIR / "01_health_index_timeline.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def plot_sub_scores(df_scored: pd.DataFrame, ts_col: str = "Timestamp"):
    """Stacked area chart of sub-scores for each compressor."""
    colors = {"bear": "#d62728", "vib": "#ff7f0e", "oil": "#1f77b4", "cr": "#2ca02c"}

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("Health Sub-Scores Over Time (area = weight x score)", fontsize=13, fontweight="bold")

    for ax, comp in zip(axes, COMPRESSORS):
        ts   = pd.to_datetime(df_scored[ts_col], errors="coerce")
        bear = pd.to_numeric(df_scored.get(f"Compressor_{comp}_Bear_Score"), errors="coerce") * WEIGHTS["bear"]
        vib  = pd.to_numeric(df_scored.get(f"Compressor_{comp}_Vib_Score"),  errors="coerce") * WEIGHTS["vib"]
        oil  = pd.to_numeric(df_scored.get(f"Compressor_{comp}_Oil_Score"),  errors="coerce") * WEIGHTS["oil"]
        cr   = pd.to_numeric(df_scored.get(f"Compressor_{comp}_Cr_Score"),   errors="coerce") * WEIGHTS["cr"]

        ax.stackplot(ts, bear.fillna(0), vib.fillna(0), oil.fillna(0), cr.fillna(0),
                     labels=["Bear (35%)", "Vib (25%)", "Oil (25%)", "CR (15%)"],
                     colors=[colors["bear"], colors["vib"], colors["oil"], colors["cr"]],
                     alpha=0.7)
        ax.set_ylim(0, 105)
        ax.set_ylabel(f"Comp {comp}", fontsize=9)
        if comp == "A":
            ax.legend(fontsize=7, loc="lower left", ncol=4)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))

    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    path = PLOT_DIR / "02_health_sub_scores.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def plot_anomaly_timeline(df_scored: pd.DataFrame, ts_col: str = "Timestamp"):
    """Bar chart of anomaly fraction per day."""
    ts_s = pd.to_datetime(df_scored[ts_col], errors="coerce")
    df_day = df_scored.copy()
    df_day["_date"] = ts_s.dt.date

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    fig.suptitle("Anomaly Fraction per Day — Isolation Forest", fontsize=13, fontweight="bold")

    colors = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}

    for ax, comp in zip(axes, COMPRESSORS):
        anom_col = f"Compressor_{comp}_Anomaly"
        if anom_col not in df_scored.columns:
            continue
        daily = df_day.groupby("_date")[anom_col].mean()
        ax.bar(daily.index, daily.values * 100, color=colors[comp], alpha=0.75, width=0.8)
        ax.set_ylabel(f"Comp {comp}\n% anomalous", fontsize=9)
        ax.axhline(IF_CONTAMINATION * 100, color="red", linestyle="--",
                   linewidth=0.8, label=f"Contamination={IF_CONTAMINATION*100:.0f}%")
        ax.legend(fontsize=7)
        ax.set_ylim(0, 100)

    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    path = PLOT_DIR / "03_anomaly_fraction_by_day.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def print_fleet_summary(df_scored: pd.DataFrame, ts_col: str = "Timestamp"):
    print("\n" + "=" * 70)
    print("  COMPRESSOR FLEET RELIABILITY SUMMARY")
    print("=" * 70)

    for comp in COMPRESSORS:
        hi_col   = f"Compressor_{comp}_Health"
        anom_col = f"Compressor_{comp}_Anomaly"
        alrt_col = f"Compressor_{comp}_Alert"

        hi   = pd.to_numeric(df_scored.get(hi_col), errors="coerce").dropna()
        anom = df_scored.get(anom_col, pd.Series(dtype=float)).dropna()
        alrt = df_scored.get(alrt_col, pd.Series(dtype=str))

        if len(hi) == 0:
            print(f"\n  Compressor {comp}: no running data")
            continue

        green_pct  = (alrt == "green").sum()  / len(alrt) * 100
        amber_pct  = (alrt == "amber").sum()  / len(alrt) * 100
        red_pct    = (alrt == "red").sum()    / len(alrt) * 100
        offline_pct= (alrt == "offline").sum()/ len(alrt) * 100
        anom_pct   = (anom == 1).sum() / max(len(anom), 1) * 100

        print(f"\n  Compressor {comp}:")
        print(f"    Mean Health Index : {hi.mean():.1f}  |  Min: {hi.min():.1f}  |  Max: {hi.max():.1f}")
        print(f"    Green             : {green_pct:.1f}%   Amber: {amber_pct:.1f}%   Red: {red_pct:.1f}%   Offline: {offline_pct:.1f}%")
        print(f"    Anomaly (IF)      : {anom_pct:.1f}% of running hours")

        # Report worst day
        ts_s = pd.to_datetime(df_scored[ts_col], errors="coerce")
        hi_s = pd.to_numeric(df_scored.get(hi_col), errors="coerce")
        if hi_s.notna().any():
            worst_idx = hi_s.idxmin()
            worst_ts  = ts_s.iloc[worst_idx]
            worst_hi  = hi_s.iloc[worst_idx]
            print(f"    Worst point       : HI={worst_hi:.0f}  at {worst_ts}")

    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def load_data_for_training(data_path: Path) -> pd.DataFrame:
    """Load CSV (raw or KPI-enriched) and prepare for reliability analysis."""
    df_head = pd.read_csv(data_path, encoding="latin-1", nrows=2)
    ts_col  = next((c for c in df_head.columns if "timestamp" in c.lower() or c.strip().lower() == "timestamp"),
                   df_head.columns[0])

    needed = {ts_col}
    for sm in SENSOR_MAP.values():
        for col in sm.values():
            if col in df_head.columns:
                needed.add(col)

    df = pd.read_csv(data_path, encoding="latin-1", usecols=list(needed))
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.rename(columns={ts_col: "Timestamp"}).sort_values("Timestamp").reset_index(drop=True)
    return df


def main(data_path: Path = DATA_CSV, save_path: Path = MODEL_PKL) -> CompressorHealthMonitor:
    """
    Full training pipeline:
      1. Load data
      2. Fit CompressorHealthMonitor (Isolation Forest per compressor)
      3. Score all rows and produce diagnostic plots
      4. Save the fitted monitor to pickle
    Returns the fitted monitor.
    """
    print(f"\nLoading data: {data_path.name}")
    df = load_data_for_training(data_path)
    ts_start = df["Timestamp"].dropna().iloc[0]
    ts_end   = df["Timestamp"].dropna().iloc[-1]
    print(f"  {len(df):,} rows  |  {ts_start.date()} to {ts_end.date()}")

    monitor = CompressorHealthMonitor()
    monitor.fit(df)

    print("\nScoring all rows ...")
    df_scored = monitor.score_dataframe(df)

    print("\nGenerating diagnostic plots ...")
    plot_health_timeseries(df_scored)
    plot_sub_scores(df_scored)
    plot_anomaly_timeline(df_scored)

    print_fleet_summary(df_scored)

    monitor.save(save_path)

    # Print real-time snapshot for the latest running row
    last_row = df.iloc[-1].to_dict()
    rt_result = monitor.predict_realtime(last_row)
    monitor.print_health_report(rt_result, timestamp=str(df["Timestamp"].dropna().iloc[-1]))

    return monitor


if __name__ == "__main__":
    main()
