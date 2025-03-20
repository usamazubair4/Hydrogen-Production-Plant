"""
Compressor Reliability EDA
===========================
Exploratory data analysis for Compressors A, B, C on the SMR Plant.

Sensor categories analysed:
  - Vibration       : Motor DE/ODE, Frame DE/ODE, Interstage Cooler
  - Bearing health  : 6-point bearing temperatures + hottest bearing
  - Oil system      : Filter dP, oil pressure, oil temperature
  - Cylinder temps  : Stage-by-stage heat rise
  - Compression eff.: Stage compression ratios
  - Motor load      : Motor current

Outputs (saved to eda_plots/compressor/):
  01_motor_current_timeline.png
  02_vibration_comparison.png
  03_bearing_temps_heatmap.png
  04_oil_system_trends.png
  05_cylinder_temp_stages.png
  06_compression_ratios.png
  07_running_hours_breakdown.png
  08_health_signal_correlations.png
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT    = Path(__file__).resolve().parent
DATA_CSV = _ROOT / "Combined_Data.csv"
OUT_DIR  = _ROOT / "eda_plots" / "compressor"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Per-compressor sensor maps ────────────────────────────────────────────────
COMPRESSORS = ["A", "B", "C"]

SENSOR_MAP = {}
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

# Running threshold: compressor is ON when current exceeds this value (Amps)
RUNNING_THRESHOLD = {"A": 100, "B": 400, "C": 200}

# Alarm / trip reference limits (literature / ISO 13709 / plant practice)
VIB_ALERT_MM_S  = 0.71   # ISO 10816 Zone B/C boundary
VIB_TRIP_MM_S   = 1.12   # ISO 10816 Zone D
BEAR_ALERT_F    = 180.0  # bearing temperature alert (°F)
BEAR_TRIP_F     = 220.0  # bearing temperature trip  (°F)
OIL_DP_ALERT    = 12.0   # oil filter dP alert (psi)
OIL_TEMP_ALERT  = 170.0  # oil temp alert (°F)

COLORS = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}
ALPHA  = 0.7


# ── Data loading ──────────────────────────────────────────────────────────────

def load_compressor_data() -> pd.DataFrame:
    """Load raw CSV, parse timestamp, keep only compressor + timestamp cols."""
    all_cols_df = pd.read_csv(DATA_CSV, encoding="latin-1", nrows=2)
    all_cols    = list(all_cols_df.columns)

    ts_col = next((c for c in all_cols if "timestamp" in c.lower() or c.strip().lower() == "timestamp"), all_cols[0])

    needed = {ts_col}
    for comp_sensors in SENSOR_MAP.values():
        for col in comp_sensors.values():
            if col in all_cols:
                needed.add(col)

    df = pd.read_csv(DATA_CSV, encoding="latin-1", usecols=list(needed))
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.rename(columns={ts_col: "Timestamp"}).sort_values("Timestamp").reset_index(drop=True)

    # Coerce all sensor columns to numeric
    for comp_sensors in SENSOR_MAP.values():
        for col in comp_sensors.values():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    # Add running flag per compressor
    for comp in COMPRESSORS:
        cur_col = SENSOR_MAP[comp]["current"]
        if cur_col in df.columns:
            df[f"running_{comp}"] = df[cur_col] > RUNNING_THRESHOLD[comp]
        else:
            df[f"running_{comp}"] = False

    return df


def _col(comp: str, key: str, df: pd.DataFrame):
    """Return a Series for a sensor key; NaN series if column missing."""
    col_name = SENSOR_MAP[comp].get(key, "")
    if col_name and col_name in df.columns:
        return df[col_name]
    return pd.Series(np.nan, index=df.index)


def _save(fig, name: str):
    path = OUT_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Plot 1: Motor Current Timeline ────────────────────────────────────────────

def plot_motor_current(df: pd.DataFrame):
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Motor Current — Compressor A / B / C", fontsize=13, fontweight="bold")

    for ax, comp in zip(axes, COMPRESSORS):
        s = _col(comp, "current", df)
        ax.plot(df["Timestamp"], s, color=COLORS[comp], linewidth=0.6, alpha=0.8)
        ax.axhline(RUNNING_THRESHOLD[comp], color="gray", linestyle="--", linewidth=0.8, label="Running threshold")
        ax.set_ylabel(f"Comp {comp}\n(Amps)", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))

    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    _save(fig, "01_motor_current_timeline.png")


# ── Plot 2: Vibration Comparison ──────────────────────────────────────────────

def plot_vibration(df: pd.DataFrame):
    vib_keys = ["vib_motor_de", "vib_motor_ode", "vib_frame_de", "vib_frame_ode", "vib_ic1", "vib_ic2"]
    vib_labels = ["Motor DE", "Motor ODE", "Frame DE", "Frame ODE", "IC Vib 1", "IC Vib 2"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Vibration Levels — Compressor A / B / C  (mm/s)", fontsize=13, fontweight="bold")

    for ax, comp in zip(axes, COMPRESSORS):
        run_mask = df[f"running_{comp}"]
        for key, label in zip(vib_keys, vib_labels):
            s = _col(comp, key, df)
            ax.plot(df.loc[run_mask, "Timestamp"], s[run_mask],
                    linewidth=0.7, alpha=0.75, label=label)
        ax.axhline(VIB_ALERT_MM_S, color="orange", linestyle="--", linewidth=1.0, label=f"Alert {VIB_ALERT_MM_S}")
        ax.axhline(VIB_TRIP_MM_S,  color="red",    linestyle="--", linewidth=1.0, label=f"Trip {VIB_TRIP_MM_S}")
        ax.set_ylabel(f"Comp {comp}", fontsize=9)
        ax.legend(fontsize=6, loc="upper left", ncol=4)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))

    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    _save(fig, "02_vibration_comparison.png")


# ── Plot 3: Bearing Temperature Heatmap (latest snapshot + time trend) ────────

def plot_bearing_heatmap(df: pd.DataFrame):
    bear_keys   = ["bear_1", "bear_2", "bear_3", "bear_4", "bear_5", "bear_6"]
    bear_labels = [f"Bearing #{i+1}" for i in range(6)]

    # Resample to hourly means for the heatmap
    df_h = df.set_index("Timestamp").resample("1h").mean(numeric_only=True)
    ts_index = df_h.index

    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
    fig.suptitle("Bearing Temperature (°F) — 6-Point Heatmap per Compressor", fontsize=13, fontweight="bold")

    for ax, comp in zip(axes, COMPRESSORS):
        mat = []
        for key in bear_keys:
            col_name = SENSOR_MAP[comp].get(key, "")
            if col_name and col_name in df_h.columns:
                mat.append(df_h[col_name].values)
            else:
                mat.append(np.full(len(df_h), np.nan))

        mat = np.array(mat)  # shape: (6, T)
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r",
                       vmin=60, vmax=220,
                       extent=[mdates.date2num(ts_index[0]), mdates.date2num(ts_index[-1]), 0, 6])
        ax.set_yticks(np.arange(0.5, 6.5))
        ax.set_yticklabels(bear_labels[::-1], fontsize=8)
        ax.set_title(f"Compressor {comp}", fontsize=9, loc="left")
        plt.colorbar(im, ax=ax, label="°F")

    axes[-1].xaxis_date()
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    _save(fig, "03_bearing_temps_heatmap.png")


# ── Plot 4: Oil System Trends ─────────────────────────────────────────────────

def plot_oil_system(df: pd.DataFrame):
    fig, axes = plt.subplots(3, 3, figsize=(16, 9), sharex=True)
    fig.suptitle("Oil System Health — Filter dP, Oil Pressure, Oil Temperature", fontsize=13, fontweight="bold")

    for row_idx, comp in enumerate(COMPRESSORS):
        run_mask = df[f"running_{comp}"]

        # Column 0: Oil filter dP
        ax = axes[row_idx][0]
        dp = _col(comp, "oil_dp", df)
        ax.plot(df.loc[run_mask, "Timestamp"], dp[run_mask], color=COLORS[comp], linewidth=0.7)
        ax.axhline(OIL_DP_ALERT, color="orange", linestyle="--", linewidth=1.0, label=f"Alert {OIL_DP_ALERT}")
        ax.set_ylabel(f"Comp {comp}", fontsize=9)
        if row_idx == 0:
            ax.set_title("Oil Filter dP (psi)", fontsize=9)
        ax.legend(fontsize=7)

        # Column 1: Oil pressure
        ax = axes[row_idx][1]
        op = _col(comp, "oil_press", df)
        ax.plot(df.loc[run_mask, "Timestamp"], op[run_mask], color=COLORS[comp], linewidth=0.7)
        if row_idx == 0:
            ax.set_title("Oil Pressure (psig)", fontsize=9)

        # Column 2: Oil temperature
        ax = axes[row_idx][2]
        ot = _col(comp, "oil_temp", df)
        ax.plot(df.loc[run_mask, "Timestamp"], ot[run_mask], color=COLORS[comp], linewidth=0.7)
        ax.axhline(OIL_TEMP_ALERT, color="orange", linestyle="--", linewidth=1.0, label=f"Alert {OIL_TEMP_ALERT}")
        if row_idx == 0:
            ax.set_title("Oil Temperature (°F)", fontsize=9)
        ax.legend(fontsize=7)

    for col_axes in axes:
        for ax in col_axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))

    fig.tight_layout()
    _save(fig, "04_oil_system_trends.png")


# ── Plot 5: Cylinder Temperature Stages ───────────────────────────────────────

def plot_cylinder_temps(df: pd.DataFrame):
    stage_keys   = ["cyl_fg1", "cyl_fg2", "cyl_h2_1", "cyl_h2_2", "cyl_h2_3"]
    stage_labels = ["NG Stage 1", "NG Stage 2", "H2 Stage 1", "H2 Stage 2", "H2 Stage 3"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("Cylinder Temperatures by Stage (°F)", fontsize=13, fontweight="bold")

    for ax, comp in zip(axes, COMPRESSORS):
        run_mask = df[f"running_{comp}"]
        for key, label in zip(stage_keys, stage_labels):
            s = _col(comp, key, df)
            ax.plot(df.loc[run_mask, "Timestamp"], s[run_mask],
                    linewidth=0.7, alpha=0.8, label=label)
        ax.set_ylabel(f"Comp {comp}\n(°F)", fontsize=9)
        ax.legend(fontsize=7, loc="upper left", ncol=3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))

    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    _save(fig, "05_cylinder_temp_stages.png")


# ── Plot 6: Compression Ratios (efficiency proxy) ─────────────────────────────

def plot_compression_ratios(df: pd.DataFrame):
    cr_keys   = ["cr_h2_1", "cr_h2_2", "cr_h2_3"]
    cr_labels = ["H2 Stage 1", "H2 Stage 2", "H2 Stage 3"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("H2 Compression Ratios — Compressor A / B / C  (higher=more efficient)", fontsize=13, fontweight="bold")

    for ax, comp in zip(axes, COMPRESSORS):
        run_mask = df[f"running_{comp}"]
        for key, label in zip(cr_keys, cr_labels):
            s = _col(comp, key, df)
            ax.plot(df.loc[run_mask, "Timestamp"], s[run_mask],
                    linewidth=0.7, alpha=0.8, label=label)
        ax.axhline(1.0, color="red", linestyle=":", linewidth=0.8, label="Ratio = 1 (no compression)")
        ax.set_ylabel(f"Comp {comp}", fontsize=9)
        ax.legend(fontsize=7, loc="upper left", ncol=2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))

    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    _save(fig, "06_compression_ratios.png")


# ── Plot 7: Running Hours Breakdown (pie + bar) ───────────────────────────────

def plot_running_hours(df: pd.DataFrame):
    fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Compressor Operating Mode Breakdown", fontsize=13, fontweight="bold")

    run_counts, stop_counts = [], []
    for comp in COMPRESSORS:
        run_mask = df[f"running_{comp}"]
        run_counts.append(run_mask.sum())
        stop_counts.append((~run_mask).sum())

    x = np.arange(3)
    ax_bar.bar(x - 0.2, run_counts,  0.4, label="Running", color="#2ca02c", alpha=0.8)
    ax_bar.bar(x + 0.2, stop_counts, 0.4, label="Standby/Off", color="#d62728", alpha=0.6)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"Comp {c}" for c in COMPRESSORS])
    ax_bar.set_ylabel("Data points (15-min intervals)")
    ax_bar.set_title("Running vs Standby Count")
    ax_bar.legend()

    # Fleet running overlap pie
    any_running   = (df["running_A"] | df["running_B"] | df["running_C"]).sum()
    all_running   = (df["running_A"] & df["running_B"] & df["running_C"]).sum()
    two_running   = ((df["running_A"].astype(int) + df["running_B"].astype(int) + df["running_C"].astype(int)) == 2).sum()
    one_running   = ((df["running_A"].astype(int) + df["running_B"].astype(int) + df["running_C"].astype(int)) == 1).sum()
    none_running  = (~df["running_A"] & ~df["running_B"] & ~df["running_C"]).sum()

    ax_pie.pie(
        [all_running, two_running, one_running, none_running],
        labels=["3 running", "2 running", "1 running", "0 running"],
        autopct="%1.0f%%",
        colors=["#2ca02c", "#98df8a", "#ffbb78", "#d62728"],
        startangle=90,
    )
    ax_pie.set_title("Fleet-Level Operating Mode")

    fig.tight_layout()
    _save(fig, "07_running_hours_breakdown.png")


# ── Plot 8: Key Health Signal Correlations ────────────────────────────────────

def plot_health_correlations(df: pd.DataFrame):
    """Pearson correlation matrix of key health signals (running periods only)."""
    key_cols = {}
    for comp in COMPRESSORS:
        run_mask = df[f"running_{comp}"]
        for key, label in [
            ("vib_frame_de",  f"C{comp} Frame DE Vib"),
            ("vib_ic2",       f"C{comp} IC Vib 2"),
            ("bear_hot",      f"C{comp} Hottest Bear"),
            ("oil_dp",        f"C{comp} Oil dP"),
            ("oil_temp",      f"C{comp} Oil Temp"),
            ("current",       f"C{comp} Current"),
        ]:
            s = _col(comp, key, df).copy()
            s[~run_mask] = np.nan
            key_cols[label] = s

    corr_df = pd.DataFrame(key_cols)
    corr_mat = corr_df.corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    sns.heatmap(corr_mat, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax, annot_kws={"size": 7},
                linewidths=0.4)
    ax.set_title("Compressor Health Signal Correlation Matrix (running periods only)", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", labelsize=8, rotation=45)
    ax.tick_params(axis="y", labelsize=8, rotation=0)
    fig.tight_layout()
    _save(fig, "08_health_signal_correlations.png")


# ── Summary statistics ────────────────────────────────────────────────────────

def print_eda_summary(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("  COMPRESSOR EDA SUMMARY")
    print("=" * 70)
    total_hrs = (df["Timestamp"].dropna().iloc[-1] - df["Timestamp"].dropna().iloc[0]).total_seconds() / 3600

    for comp in COMPRESSORS:
        run_mask  = df[f"running_{comp}"]
        run_hrs   = run_mask.sum() * 0.25  # 15-min intervals
        avail_pct = 100 * run_hrs / total_hrs

        print(f"\n  Compressor {comp}:")
        print(f"    Running time    : {run_hrs:.0f} h  ({avail_pct:.1f}% availability)")
        print(f"    Motor current   : mean={_col(comp, 'current', df)[run_mask].mean():.0f}A  max={_col(comp, 'current', df)[run_mask].max():.0f}A")

        vib_max = max(
            _col(comp, k, df)[run_mask].max()
            for k in ["vib_motor_de", "vib_motor_ode", "vib_frame_de", "vib_frame_ode", "vib_ic1", "vib_ic2"]
        )
        print(f"    Peak vibration  : {vib_max:.3f} mm/s  ({'ALERT' if vib_max > VIB_ALERT_MM_S else 'OK'})")

        bear_max = _col(comp, "bear_hot", df)[run_mask].max()
        print(f"    Max bearing temp: {bear_max:.1f} F  ({'ALERT' if bear_max > BEAR_ALERT_F else 'OK'})")

        oil_dp_max = _col(comp, "oil_dp", df)[run_mask].max()
        print(f"    Max oil filt dP : {oil_dp_max:.1f} psi  ({'ALERT' if oil_dp_max > OIL_DP_ALERT else 'OK'})")

        oil_temp_max = _col(comp, "oil_temp", df)[run_mask].max()
        print(f"    Max oil temp    : {oil_temp_max:.1f} F  ({'ALERT' if oil_temp_max > OIL_TEMP_ALERT else 'OK'})")

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading data from {DATA_CSV.name} ...")
    df = load_compressor_data()
    print(f"  {len(df):,} rows  |  {df['Timestamp'].dropna().iloc[0].date()} to {df['Timestamp'].dropna().iloc[-1].date()}")

    print("\nGenerating EDA plots ...")
    plot_motor_current(df)
    plot_vibration(df)
    plot_bearing_heatmap(df)
    plot_oil_system(df)
    plot_cylinder_temps(df)
    plot_compression_ratios(df)
    plot_running_hours(df)
    plot_health_correlations(df)

    print_eda_summary(df)

    print(f"All plots saved to: {OUT_DIR}")
    return df


if __name__ == "__main__":
    main()
