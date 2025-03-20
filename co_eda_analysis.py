"""
Exploratory Data Analysis — CO in Product prediction.
SMR Plant, H2 Production Facility.

Run this script AFTER generating Combined_Data_with_KPIs.csv via main.py.

Outputs
-------
  * Console: statistical summary, correlation ranking, feature rationale,
             multicollinearity (VIF), key EDA findings.
  * ./eda_plots/: PNG plots for every major EDA step.

Usage
-----
  python co_eda_analysis.py
  python co_eda_analysis.py --data "path/to/Combined_Data_with_KPIs.csv"
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Optional imports (degrade gracefully if not installed) ───────────────────
try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend (safe for scripts)
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not found — plots will be skipped. pip install matplotlib")

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False
    print("[WARN] seaborn not found — some plots will be skipped. pip install seaborn")

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAS_SM = True
except ImportError:
    HAS_SM = False
    print("[WARN] statsmodels not found — VIF analysis will be skipped. pip install statsmodels")

from feature_engineering import (
    KPI_FEATURE_COLS,
    PHYSICS_FEATURE_COLS,
    TARGET_COL,
    FEATURE_RATIONALE,
    extract_feature_matrix,
    compute_physics_features,
)

# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_DATA = Path(__file__).parent / "Combined_Data_with_KPIs.csv"
PLOT_DIR     = Path(__file__).parent / "eda_plots"

CO_SPEC_SOFT = 5.0   # ppm — typical soft alert threshold
CO_SPEC_HARD = 10.0  # ppm — typical hard specification limit


# ── Helpers ──────────────────────────────────────────────────────────────────

def _save(fig: "plt.Figure", name: str) -> None:
    PLOT_DIR.mkdir(exist_ok=True)
    path = PLOT_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


def _section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ── Data loading ─────────────────────────────────────────────────────────────

def load_enriched_data(path: Path) -> pd.DataFrame:
    """Load the KPI-enriched CSV and return a cleaned DataFrame."""
    print(f"\nLoading data from: {path}")
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    print(f"  Raw shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Parse timestamp
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.sort_values("Timestamp").reset_index(drop=True)
        n_bad_ts = df["Timestamp"].isna().sum()
        if n_bad_ts:
            print(f"  [WARN] {n_bad_ts} rows with unparseable Timestamp dropped.")
        df = df.dropna(subset=["Timestamp"]).reset_index(drop=True)

    print(f"  Clean shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    if "Timestamp" in df.columns:
        print(f"  Date range: {df['Timestamp'].min()}  to  {df['Timestamp'].max()}")
    return df


# ── Target analysis ──────────────────────────────────────────────────────────

def analyse_target(df: pd.DataFrame) -> None:
    _section("TARGET VARIABLE — CO in Product (ppm)")

    if TARGET_COL not in df.columns:
        print(f"  [ERROR] '{TARGET_COL}' column not found in data.")
        return

    y = pd.to_numeric(df[TARGET_COL], errors="coerce")
    y_clean = y.dropna()

    print(f"\n  Sample count     : {len(y_clean):,} (of {len(y):,} total rows)")
    print(f"  Missing / bad    : {y.isna().sum():,} ({y.isna().mean()*100:.1f}%)")
    print(f"\n  Min              : {y_clean.min():.2f} ppm")
    print(f"  5th percentile   : {y_clean.quantile(0.05):.2f} ppm")
    print(f"  25th percentile  : {y_clean.quantile(0.25):.2f} ppm")
    print(f"  Median           : {y_clean.median():.2f} ppm")
    print(f"  Mean             : {y_clean.mean():.2f} ppm")
    print(f"  75th percentile  : {y_clean.quantile(0.75):.2f} ppm")
    print(f"  95th percentile  : {y_clean.quantile(0.95):.2f} ppm")
    print(f"  Max              : {y_clean.max():.2f} ppm")
    print(f"  Std dev          : {y_clean.std():.2f} ppm")
    print(f"  Skewness         : {y_clean.skew():.3f}")
    print(f"  Kurtosis         : {y_clean.kurt():.3f}")

    n_above_soft = (y_clean > CO_SPEC_SOFT).sum()
    n_above_hard = (y_clean > CO_SPEC_HARD).sum()
    print(f"\n  > {CO_SPEC_SOFT} ppm (soft alert) : {n_above_soft:,} rows ({n_above_soft/len(y_clean)*100:.1f}%)")
    print(f"  > {CO_SPEC_HARD} ppm (hard spec)   : {n_above_hard:,} rows ({n_above_hard/len(y_clean)*100:.1f}%)")

    if not HAS_MPL:
        return

    # Plot 1: Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CO in Product — Distribution Analysis", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.hist(y_clean, bins=60, color="#2196F3", alpha=0.75, edgecolor="white", linewidth=0.5)
    ax.axvline(CO_SPEC_SOFT, color="#FF9800", linewidth=1.5, linestyle="--", label=f"Soft alert {CO_SPEC_SOFT} ppm")
    ax.axvline(CO_SPEC_HARD, color="#F44336", linewidth=1.5, linestyle="--", label=f"Hard spec {CO_SPEC_HARD} ppm")
    ax.axvline(y_clean.median(), color="#4CAF50", linewidth=1.5, linestyle="-", label=f"Median {y_clean.median():.1f} ppm")
    ax.set_xlabel("CO in Product (ppm)")
    ax.set_ylabel("Count")
    ax.set_title("Frequency Distribution")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.hist(np.log1p(y_clean.clip(lower=0)), bins=60, color="#9C27B0", alpha=0.75, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("log(1 + CO in Product) (ppm)")
    ax.set_ylabel("Count")
    ax.set_title("Log-transformed Distribution (reduces skew for modelling)")

    _save(fig, "01_target_distribution")

    # Plot 2: Time series
    if "Timestamp" in df.columns:
        fig, ax = plt.subplots(figsize=(16, 4))
        ax.plot(df["Timestamp"], y, color="#2196F3", linewidth=0.5, alpha=0.8)
        ax.axhline(CO_SPEC_SOFT, color="#FF9800", linewidth=1.2, linestyle="--", label=f"Soft {CO_SPEC_SOFT} ppm")
        ax.axhline(CO_SPEC_HARD, color="#F44336", linewidth=1.2, linestyle="--", label=f"Hard {CO_SPEC_HARD} ppm")
        ax.set_title("CO in Product — Time Series", fontsize=12)
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("CO (ppm)")
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()
        _save(fig, "02_target_timeseries")

    # Plot 3: Rolling 24-h mean
    if "Timestamp" in df.columns:
        ts_series = y.copy()
        ts_series.index = df["Timestamp"]
        ts_series = ts_series.sort_index()
        roll = ts_series.rolling("24h").mean()

        fig, ax = plt.subplots(figsize=(16, 4))
        ax.fill_between(ts_series.index, ts_series, alpha=0.2, color="#2196F3")
        ax.plot(roll.index, roll, color="#1565C0", linewidth=1.2, label="24-h rolling mean")
        ax.axhline(CO_SPEC_SOFT, color="#FF9800", linewidth=1.2, linestyle="--", label=f"Soft {CO_SPEC_SOFT} ppm")
        ax.axhline(CO_SPEC_HARD, color="#F44336", linewidth=1.2, linestyle="--", label=f"Hard {CO_SPEC_HARD} ppm")
        ax.set_title("CO in Product — 24-hour Rolling Mean", fontsize=12)
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("CO (ppm)")
        ax.legend()
        fig.autofmt_xdate()
        _save(fig, "03_target_rolling_mean")


# ── Feature extraction ───────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix and target; drop rows where target is NaN."""
    X, y = extract_feature_matrix(df)
    mask = y.notna() if y is not None else pd.Series(False, index=df.index)
    X = X[mask]
    y = y[mask]
    print(f"\n  Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"  Target series : {len(y):,} valid rows")
    return X, y


# ── Missing value analysis ────────────────────────────────────────────────────

def analyse_missing(X: pd.DataFrame) -> None:
    _section("MISSING VALUE ANALYSIS")

    miss = X.isna().mean().sort_values(ascending=False) * 100
    miss = miss[miss > 0]

    if miss.empty:
        print("\n  No missing values in feature matrix.")
        return

    print(f"\n  Features with missing data ({len(miss)} of {X.shape[1]}):")
    for feat, pct in miss.items():
        bar = "█" * int(pct / 2)
        print(f"    {feat:<55s} {pct:5.1f}%  {bar}")

    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(miss) * 0.35)))
    miss.plot(kind="barh", ax=ax, color="#FF7043", edgecolor="white")
    ax.set_title("Missing Data per Feature (%)", fontsize=12)
    ax.set_xlabel("Missing (%)")
    ax.invert_yaxis()
    ax.axvline(20, color="red", linestyle="--", linewidth=1, label="20% threshold")
    ax.legend()
    _save(fig, "04_missing_values")


# ── Correlation analysis ─────────────────────────────────────────────────────

def analyse_correlations(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    _section("CORRELATION WITH TARGET — CO in Product")

    corr = (
        X.apply(pd.to_numeric, errors="coerce")
         .corrwith(y.rename("CO_ppm"))
         .dropna()
         .sort_values(key=abs, ascending=False)
    )

    print(f"\n  {'Feature':<55s} {'Pearson r':>10}  Direction")
    print(f"  {'-'*55}  {'-'*10}  {'-'*15}")
    for feat, r in corr.items():
        direction = "↑ increases CO" if r > 0 else "↓ decreases CO"
        star = "***" if abs(r) > 0.5 else ("**" if abs(r) > 0.3 else ("*" if abs(r) > 0.15 else ""))
        print(f"  {feat:<55s} {r:>10.4f}  {direction}  {star}")

    if HAS_MPL:
        top_n = min(20, len(corr))
        top = corr.abs().nlargest(top_n)
        colors = ["#F44336" if corr[f] > 0 else "#2196F3" for f in top.index]

        fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.38)))
        top.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
        ax.set_title(f"Top {top_n} Feature Correlations with CO in Product (|Pearson r|)", fontsize=12)
        ax.set_xlabel("|Pearson r|")
        ax.invert_yaxis()
        ax.axvline(0.3, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        # Legend
        from matplotlib.patches import Patch
        legend = [Patch(color="#F44336", label="Positive (increases CO)"),
                  Patch(color="#2196F3", label="Negative (decreases CO)")]
        ax.legend(handles=legend, fontsize=9)
        _save(fig, "05_feature_correlations")

    return corr


# ── Correlation heatmap ───────────────────────────────────────────────────────

def plot_correlation_heatmap(X: pd.DataFrame, y: pd.Series) -> None:
    if not HAS_MPL or not HAS_SNS:
        return
    _section("FEATURE-FEATURE CORRELATION HEATMAP")

    # Use top-15 most correlated with target to keep heatmap readable
    corr_with_y = X.corrwith(y).abs().nlargest(15)
    top_cols = corr_with_y.index.tolist()
    combined = X[top_cols].copy()
    combined["CO_in_Product"] = y.values

    corr_matrix = combined.corr()
    print("\n  Heatmap of top-15 features + target (saved to file).")

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.zeros_like(corr_matrix, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        annot=True, fmt=".2f",
        annot_kws={"size": 7},
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap (top-15 features by |r| with CO in Product)", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    _save(fig, "06_correlation_heatmap")


# ── Scatter plots ─────────────────────────────────────────────────────────────

def plot_scatter_top_features(X: pd.DataFrame, y: pd.Series, corr: pd.Series) -> None:
    if not HAS_MPL:
        return
    _section("SCATTER PLOTS — TOP FEATURES vs CO in Product")

    top_features = corr.abs().nlargest(9).index.tolist()
    n_cols = 3
    n_rows = (len(top_features) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
    axes = axes.flatten()

    for i, feat in enumerate(top_features):
        ax = axes[i]
        x_vals = pd.to_numeric(X[feat], errors="coerce")
        mask = x_vals.notna() & y.notna()
        r = corr.get(feat, np.nan)

        ax.scatter(x_vals[mask], y[mask], alpha=0.25, s=4,
                   color="#F44336" if r > 0 else "#2196F3")

        # Trend line
        try:
            z = np.polyfit(x_vals[mask], y[mask], 1)
            p = np.poly1d(z)
            xs = np.linspace(x_vals[mask].min(), x_vals[mask].max(), 200)
            ax.plot(xs, p(xs), color="black", linewidth=1.2, alpha=0.7)
        except Exception:
            pass

        ax.axhline(CO_SPEC_HARD, color="#F44336", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_xlabel(feat, fontsize=8)
        ax.set_ylabel("CO in Product (ppm)", fontsize=8)
        ax.set_title(f"r = {r:.3f}", fontsize=9)
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Top Features vs CO in Product", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, "07_scatter_top_features")


# ── Feature distributions ─────────────────────────────────────────────────────

def plot_feature_distributions(X: pd.DataFrame, corr: pd.Series) -> None:
    if not HAS_MPL:
        return

    top_features = corr.abs().nlargest(9).index.tolist()
    n_cols = 3
    n_rows = (len(top_features) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3.5))
    axes = axes.flatten()

    for i, feat in enumerate(top_features):
        ax = axes[i]
        vals = pd.to_numeric(X[feat], errors="coerce").dropna()
        ax.hist(vals, bins=50, color="#7986CB", alpha=0.8, edgecolor="white", linewidth=0.3)
        ax.axvline(vals.mean(),   color="black", linewidth=1.2, linestyle="--", label=f"Mean {vals.mean():.2f}")
        ax.axvline(vals.median(), color="orange", linewidth=1.2, linestyle="-", label=f"Median {vals.median():.2f}")
        ax.set_title(feat, fontsize=8)
        ax.set_xlabel("Value", fontsize=7)
        ax.set_ylabel("Count", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions (top-9 by |correlation| with CO)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, "08_feature_distributions")


# ── VIF — Multicollinearity ───────────────────────────────────────────────────

def analyse_vif(X: pd.DataFrame) -> None:
    if not HAS_SM:
        return
    _section("MULTICOLLINEARITY — Variance Inflation Factor (VIF)")
    print("\n  VIF > 10 → severe multicollinearity (consider dropping or combining).")
    print("  VIF 5-10 → moderate multicollinearity (monitor).\n")

    Xn = X.apply(pd.to_numeric, errors="coerce").dropna()
    if Xn.empty or Xn.shape[1] < 2:
        print("  Insufficient data for VIF analysis.")
        return

    vif_data = []
    for i, col in enumerate(Xn.columns):
        try:
            vif_val = variance_inflation_factor(Xn.values, i)
            vif_data.append((col, vif_val))
        except Exception:
            vif_data.append((col, np.nan))

    vif_df = pd.DataFrame(vif_data, columns=["Feature", "VIF"]).sort_values("VIF", ascending=False)
    for _, row in vif_df.iterrows():
        flag = "  *** HIGH ***" if row["VIF"] > 10 else ("  ** moderate **" if row["VIF"] > 5 else "")
        print(f"  {row['Feature']:<55s} VIF = {row['VIF']:7.2f}{flag}")

    print("\n  Recommendation: Features with VIF > 10 carry redundant information.")
    print("  Tree-based models (RF, XGBoost) handle multicollinearity naturally,")
    print("  but for linear models, consider dropping high-VIF features.")


# ── Outlier detection ─────────────────────────────────────────────────────────

def analyse_outliers(X: pd.DataFrame, y: pd.Series) -> None:
    _section("OUTLIER ANALYSIS")

    y_clean = y.dropna()
    Q1, Q3 = y_clean.quantile(0.25), y_clean.quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - 3 * IQR, Q3 + 3 * IQR
    outliers = ((y_clean < lo) | (y_clean > hi)).sum()
    print(f"\n  Target (CO in Product):")
    print(f"    3×IQR bounds: [{lo:.2f}, {hi:.2f}] ppm")
    print(f"    Outlier rows: {outliers:,} ({outliers/len(y_clean)*100:.1f}%)")
    print(f"\n  Note: Plant upsets, startups, and shutdowns produce legitimate")
    print(f"  high-CO periods. Do NOT blindly drop outliers — they may represent")
    print(f"  the exact operating conditions where prediction is most valuable.")


# ── Feature rationale report ──────────────────────────────────────────────────

def print_feature_rationale() -> None:
    _section("FEATURE SELECTION RATIONALE (Engineering Basis)")

    categories = {}
    for feat, info in FEATURE_RATIONALE.items():
        cat = info.get("category", "Other")
        categories.setdefault(cat, []).append((feat, info))

    for cat, items in sorted(categories.items()):
        print(f"\n  [{cat}]")
        for feat, info in items:
            print(f"\n    Feature : {feat}  [{info.get('unit', '?')}]")
            print(f"    Basis   : {info.get('mechanism', '—')}")
            print(f"    Effect  : {info.get('expected_relationship', '—')}")


# ── Key findings summary ──────────────────────────────────────────────────────

def print_key_findings(corr: pd.Series, X: pd.DataFrame, y: pd.Series) -> None:
    _section("KEY EDA FINDINGS & MODELLING RECOMMENDATIONS")

    top3 = corr.abs().nlargest(3)
    print(f"""
  1. Target variable
     ─────────────────
     CO in product is typically right-skewed (occasional high spikes during
     upsets).  Consider log-transforming the target for model training, then
     back-transforming predictions.  Evaluate with RMSE on original scale.

  2. Top correlated features
     ──────────────────────────
     {top3.index[0]:<50s} |r| = {top3.iloc[0]:.3f}
     {top3.index[1]:<50s} |r| = {top3.iloc[1]:.3f}
     {top3.index[2]:<50s} |r| = {top3.iloc[2]:.3f}

     These should be included as primary features in every model variant.

  3. Temporal structure
     ───────────────────
     The data is a time series — always use a time-based train/test split
     (e.g., last 20% of the time range as test).  Random splits cause data
     leakage through temporal autocorrelation.

  4. Missing data strategy
     ──────────────────────
     "bad" quality readings from the DCS appear as NaN after ingestion.
     Use median imputation per feature (fitted on train set only).
     XGBoost handles NaN natively, which is an additional advantage.

  5. Model architecture recommendation
     ────────────────────────────────────
     Grey-box approach (first-principles features + gradient boosting):
     - Ridge regression with physics features only → interpretable baseline
     - Random Forest → handles non-linearities, robust to outliers
     - XGBoost → best predictive performance, handles missing values
     Compare all three; use SHAP to verify that XGBoost has learned
     physically sensible feature dependencies.

  6. Real-time alerting thresholds
     ──────────────────────────────
     Soft alert  : predicted CO > {CO_SPEC_SOFT} ppm  (operator awareness)
     Hard alert  : predicted CO > {CO_SPEC_HARD} ppm  (immediate action)
     Use prediction interval (10th–90th percentile from ensemble) to
     communicate uncertainty to operators.

  7. Lag features (future improvement)
     ─────────────────────────────────
     PSA and HTS dynamics introduce 5–30 minute delays.  Adding lagged
     versions of "CO Slip (Syngas GC)" and "Shift dT" may improve RMSE
     by capturing how upstream upsets propagate to product quality.
""")


# ── Main ─────────────────────────────────────────────────────────────────────

def run_eda(data_path: Path) -> None:
    df = load_enriched_data(data_path)
    analyse_target(df)

    X, y = prepare_features(df)
    analyse_missing(X)
    corr = analyse_correlations(X, y)
    plot_correlation_heatmap(X, y)
    plot_scatter_top_features(X, y, corr)
    plot_feature_distributions(X, corr)
    analyse_vif(X)
    analyse_outliers(X, y)
    print_feature_rationale()
    print_key_findings(corr, X, y)

    print(f"\n{'='*70}")
    print(f"  EDA complete.  Plots saved to: {PLOT_DIR}")
    print(f"  Next step: run  python co_product_model.py  to train the model.")
    print(f"{'='*70}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA for CO-in-product ML model")
    parser.add_argument(
        "--data", type=str, default=str(DEFAULT_DATA),
        help="Path to Combined_Data_with_KPIs.csv"
    )
    args = parser.parse_args()
    run_eda(Path(args.data))


if __name__ == "__main__":
    main()
