"""
SMR Plant — Full Data Pipeline
H2 Production Facility

Pipeline stages
---------------
1. INGEST       Load & clean raw DCS CSV                        (Data_Ingestion)
2. KPI          Calculate all engineering KPIs incl. opt KPIs  (kpi_formulas)
3. MODEL        Load saved CO predictor or train                (co_product_model)
4. PREDICT      Append CO predictions to every row
5. RELIABILITY  Load/train CompressorHealthMonitor; score rows  (compressor_reliability)
6. OPTIMISE     Add post-ML opt KPIs + run trade-off analysis   (optimization)
7. SAVE         Write final enriched CSV
8. REPORT       Real-time CO alert + fleet health + opt recs

Usage
-----
  # Standard run — load existing models (or train first time):
  python main.py

  # Force retraining of all models:
  python main.py --train

  # Skip compressor reliability scoring:
  python main.py --no-reliability

  # Skip optimization analysis (still runs CO prediction):
  python main.py --no-optimisation

  # KPI calculation only (skip all ML):
  python main.py --no-prediction

  # Custom file paths:
  python main.py --input path/to/raw.csv --output path/to/output.csv

  # Custom model paths:
  python main.py --model path/to/co_predictor.pkl --comp-model path/to/compressor_monitor.pkl
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from Data_Ingestion import load_and_clean_plant_data
from kpi_formulas import add_kpi_columns

# ── Resolve project root so imports work regardless of working directory ─────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Default file paths ───────────────────────────────────────────────────────
_DEFAULT_INPUT      = _ROOT / "Combined_Data.csv"
_DEFAULT_OUTPUT     = _ROOT / "Combined_Data_with_KPIs.csv"
_DEFAULT_MODEL      = _ROOT / "co_predictor.pkl"
_DEFAULT_COMP_MODEL = _ROOT / "compressor_monitor.pkl"

# Alert thresholds (ppm) — adjust here if spec limits change
CO_ALERT_SOFT = 5.0
CO_ALERT_HARD = 10.0


# ── Helpers ──────────────────────────────────────────────────────────────────

def _banner(stage: str, detail: str = "") -> None:
    pad = 70
    print(f"\n{'─'*pad}")
    print(f"  [{stage}]  {detail}")
    print(f"{'─'*pad}")


def _elapsed(t0: float) -> str:
    return f"{time.time() - t0:.1f}s"


def _alert_level(co_ppm: float) -> str:
    if co_ppm >= CO_ALERT_HARD:
        return "red"
    if co_ppm >= CO_ALERT_SOFT:
        return "amber"
    return "green"


# ── Stage 1 & 2: Ingest + KPI ────────────────────────────────────────────────

def run_ingest_and_kpi(input_path: Path) -> pd.DataFrame:
    """
    Load raw CSV, clean headers/timestamps, compute all KPI columns.
    Returns the fully enriched DataFrame (not yet saved to disk).
    """
    _banner("STAGE 1 — INGEST", f"Reading {input_path.name}")
    t0 = time.time()

    if not input_path.is_file():
        print(f"  ERROR: Input file not found: {input_path}")
        sys.exit(1)

    cleaned = load_and_clean_plant_data(input_path)
    print(f"  Loaded {len(cleaned):,} rows × {cleaned.shape[1]} columns  ({_elapsed(t0)})")

    _banner("STAGE 2 — KPI CALCULATION", "Computing all engineering KPIs row-by-row")
    t0 = time.time()
    enriched = add_kpi_columns(cleaned)
    kpi_cols_added = enriched.shape[1] - cleaned.shape[1]
    print(f"  Added {kpi_cols_added} KPI columns → {enriched.shape[1]} total  ({_elapsed(t0)})")

    return enriched


# ── Stage 3: Load or train model ─────────────────────────────────────────────

def get_predictor(
    model_path: Path,
    kpi_csv_path: Path,
    force_train: bool = False,
):
    """
    Load COPredictor from pickle if it exists, otherwise train on the KPI CSV.
    Returns a fitted COPredictor instance.
    """
    from co_product_model import COPredictor, main as train_model

    if model_path.is_file() and not force_train:
        _banner("STAGE 3 — MODEL", f"Loading saved model from {model_path.name}")
        predictor = COPredictor.load(model_path)
        print(f"  Feature count : {len(predictor.feature_names)}")
        print(f"  Alert thresholds: amber > {predictor.alert_soft} ppm  |  red > {predictor.alert_hard} ppm")
        return predictor

    _banner(
        "STAGE 3 — MODEL TRAINING",
        f"{'Forced retraining' if force_train else 'No saved model found — training now'}",
    )
    print(f"  Training data : {kpi_csv_path}")
    print(f"  Model will be saved to: {model_path}")
    print()

    # train_model() runs the full training pipeline and saves co_predictor.pkl
    predictor = train_model(kpi_csv_path)

    # Verify the pickle was written
    if not model_path.is_file():
        print(f"  ERROR: Model training completed but {model_path} not found.")
        sys.exit(1)

    return predictor


# ── Stage 4: Batch prediction ─────────────────────────────────────────────────

def run_batch_prediction(enriched: pd.DataFrame, predictor) -> pd.DataFrame:
    """
    Build the feature matrix from the enriched DataFrame, predict CO in ppm
    for every row, and append CO_Predicted_ppm + CO_Alert_Level columns.
    Also persists physics feature columns (hts_k_eq, approach_to_eq, etc.)
    to the enriched DataFrame so they appear in the saved CSV.
    """
    from feature_engineering import extract_feature_matrix, compute_physics_features, PHYSICS_FEATURE_COLS

    _banner("STAGE 4 — BATCH PREDICTION", "Predicting CO in product for every row")
    t0 = time.time()

    # Compute physics features and append them to enriched
    enriched = compute_physics_features(enriched)
    phys_added = [c for c in PHYSICS_FEATURE_COLS if c in enriched.columns]
    print(f"  Physics features appended: {phys_added}")

    # Extract feature matrix — same row count as enriched
    X, _ = extract_feature_matrix(enriched)

    # The predictor internally reindexes to its trained feature names,
    # so extra or missing columns are handled automatically.
    co_pred = predictor.predict(X)

    enriched["CO_Predicted_ppm"] = co_pred.round(2)
    enriched["CO_Alert_Level"] = [_alert_level(v) for v in co_pred]

    n_red   = (enriched["CO_Alert_Level"] == "red").sum()
    n_amber = (enriched["CO_Alert_Level"] == "amber").sum()
    n_green = (enriched["CO_Alert_Level"] == "green").sum()

    print(f"  Predictions complete  ({_elapsed(t0)})")
    print(f"  GREEN  (< {CO_ALERT_SOFT} ppm)   : {n_green:>6,} rows  ({n_green/len(enriched)*100:.1f}%)")
    print(f"  AMBER  ({CO_ALERT_SOFT}–{CO_ALERT_HARD} ppm) : {n_amber:>6,} rows  ({n_amber/len(enriched)*100:.1f}%)")
    print(f"  RED    (> {CO_ALERT_HARD} ppm)  : {n_red:>6,} rows  ({n_red/len(enriched)*100:.1f}%)")
    print(f"  Predicted CO — mean: {co_pred.mean():.2f} ppm  |  max: {co_pred.max():.2f} ppm  |  min: {co_pred.min():.2f} ppm")

    return enriched


# ── Stage 5: Compressor reliability ──────────────────────────────────────────

def get_comp_monitor(
    comp_model_path: Path,
    input_df_or_csv,
    force_train: bool = False,
):
    """
    Load CompressorHealthMonitor from pickle if it exists, otherwise train.
    input_df_or_csv can be a DataFrame or a Path to a CSV.
    Returns a fitted CompressorHealthMonitor.
    """
    from compressor_reliability import CompressorHealthMonitor, main as train_monitor

    if comp_model_path.is_file() and not force_train:
        _banner("STAGE 5 — COMPRESSOR MONITOR", f"Loading saved monitor from {comp_model_path.name}")
        monitor = CompressorHealthMonitor.load(comp_model_path)
        return monitor

    _banner(
        "STAGE 5 — COMPRESSOR MONITOR TRAINING",
        f"{'Forced retraining' if force_train else 'No saved monitor — training now'}",
    )

    if isinstance(input_df_or_csv, Path):
        train_path = input_df_or_csv
    else:
        # Save the DataFrame temporarily so the trainer can load it
        train_path = comp_model_path.parent / "_tmp_comp_training.csv"
        input_df_or_csv.to_csv(train_path, index=False, encoding="latin-1")

    monitor = train_monitor(train_path, comp_model_path)
    return monitor


def run_compressor_scoring(enriched: pd.DataFrame, monitor) -> pd.DataFrame:
    """
    Score every row with compressor health index, sub-scores, and anomaly flags.
    Appends ~9 new columns per compressor (27 total).
    """
    from compressor_reliability import print_fleet_summary

    _banner("STAGE 5b — COMPRESSOR SCORING", "Health-scoring all rows for Compressors A, B, C")
    t0 = time.time()

    scored = monitor.score_dataframe(enriched)

    # Print per-compressor summary
    print_fleet_summary(scored)

    # Count of red / amber / green rows per compressor
    for comp in ["A", "B", "C"]:
        alrt_col = f"Compressor_{comp}_Alert"
        if alrt_col in scored.columns:
            vc = scored[alrt_col].value_counts()
            print(f"  Compressor {comp} alert summary — "
                  f"green: {vc.get('green', 0):,}  amber: {vc.get('amber', 0):,}  "
                  f"red: {vc.get('red', 0):,}  offline: {vc.get('offline', 0):,}")

    new_cols = scored.shape[1] - enriched.shape[1]
    print(f"\n  Added {new_cols} reliability columns  ({_elapsed(t0)})")
    return scored


# ── Stage 6: Optimisation ────────────────────────────────────────────────────

def run_optimisation(enriched: pd.DataFrame, predictor) -> pd.DataFrame:
    """
    Add post-ML optimisation KPIs and run the full trade-off analysis:
      - CO Spec Headroom (Predicted), HTS Catalyst Utilization,
        Efficiency Gap to Design, Steam Cost Index, Steam Reduction Opportunity
      - Trade-off curves (CO vs S/C, HTS temp, plant rate, PSA recovery)
      - 2D operating envelope (Plant Rate × S/C)
      - Prioritised process recommendations
    Saves plots to model_plots/optimization/.
    """
    from optimization import compute_opt_kpis, OptimizationAnalyzer

    _banner("STAGE 6 — OPTIMISATION", "Post-ML KPIs + trade-off curves + recommendations")
    t0 = time.time()

    enriched = compute_opt_kpis(enriched)
    new_cols = [c for c in enriched.columns if c in (
        "CO Spec Headroom (Predicted)", "HTS Catalyst Utilization (%)",
        "Efficiency Gap to Design (BTU/SCF)", "Steam Cost Index",
        "Steam Reduction Opportunity", "CO Spec Headroom (Measured)",
    )]
    print(f"  Added {len(new_cols)} post-ML optimisation KPIs")

    analyzer = OptimizationAnalyzer(predictor)
    analyzer.run_analysis(enriched)
    print(f"  Optimisation analysis complete  ({_elapsed(t0)})")
    return enriched


# ── Stage 7: Save ─────────────────────────────────────────────────────────────

def save_output(df: pd.DataFrame, output_path: Path) -> None:
    _banner("STAGE 7 — SAVE", f"Writing enriched CSV to {output_path.name}")
    t0 = time.time()
    df.to_csv(output_path, index=False, encoding="latin-1")
    size_mb = output_path.stat().st_size / 1_048_576
    print(f"  Saved {len(df):,} rows × {df.shape[1]} columns  →  {size_mb:.1f} MB  ({_elapsed(t0)})")


# ── Stage 8: Real-time report for the latest row ─────────────────────────────

def print_latest_prediction(enriched: pd.DataFrame, predictor, monitor=None) -> None:
    _banner("STAGE 8 — LATEST ROW REAL-TIME PREDICTION")

    # Find the most recent non-NaN timestamp row
    if "Timestamp" in enriched.columns:
        ts_col = pd.to_datetime(enriched["Timestamp"], errors="coerce")
        last_idx = ts_col.last_valid_index()
        if last_idx is None:
            last_idx = len(enriched) - 1
        ts_label = ts_col.iloc[last_idx]
    else:
        last_idx = len(enriched) - 1
        ts_label = f"row {last_idx}"

    print(f"  Timestamp : {ts_label}")

    # Build sensor reading dict from the last row's KPI and raw columns
    last_row = enriched.iloc[last_idx].to_dict()

    result = predictor.predict_realtime(last_row)
    predictor.print_prediction(result)

    # Show actual vs predicted if the measured CO column is present
    actual = enriched.get("CO in Product", pd.Series(dtype=float)).iloc[last_idx]
    if pd.notna(actual):
        actual = float(actual)
        err = abs(result["co_ppm_predicted"] - actual)
        print(f"\n  Actual measured CO : {actual:.2f} ppm")
        print(f"  Absolute error     : {err:.2f} ppm")

    # Compressor fleet health for the same latest row
    if monitor is not None:
        fleet_result = monitor.predict_realtime(last_row)
        monitor.print_health_report(fleet_result, timestamp=str(ts_label))


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SMR Plant — Full pipeline: ingest → KPI → CO prediction → compressor reliability",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input", type=str,
        default=str(_DEFAULT_INPUT),
        help="Path to raw DCS data CSV file (default: Combined_Data.csv)",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(_DEFAULT_OUTPUT),
        help="Path for enriched output CSV (default: Combined_Data_with_KPIs.csv)",
    )
    parser.add_argument(
        "--model", type=str,
        default=str(_DEFAULT_MODEL),
        help="Path to COPredictor pickle file (default: co_predictor.pkl)",
    )
    parser.add_argument(
        "--comp-model", type=str,
        default=str(_DEFAULT_COMP_MODEL),
        help="Path to CompressorHealthMonitor pickle (default: compressor_monitor.pkl)",
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Force retraining of both CO predictor and compressor monitor",
    )
    parser.add_argument(
        "--no-prediction", action="store_true",
        help="Skip all ML — output KPIs only (backward-compatible mode)",
    )
    parser.add_argument(
        "--no-reliability", action="store_true",
        help="Skip compressor reliability scoring; CO prediction still runs",
    )
    parser.add_argument(
        "--no-optimisation", action="store_true",
        help="Skip optimisation analysis; CO prediction and reliability still run",
    )
    args = parser.parse_args()

    input_path      = Path(args.input)
    output_path     = Path(args.output)
    model_path      = Path(args.model)
    comp_model_path = Path(args.comp_model)
    skip_opt        = args.no_optimisation or args.no_prediction

    pipeline_start = time.time()
    print("\n" + "=" * 70)
    print("  SMR Plant — DATA PIPELINE")
    print("  H2 Production Facility")
    print("=" * 70)
    print(f"  Input        : {input_path}")
    print(f"  Output       : {output_path}")
    print(f"  CO Model     : {model_path}")
    print(f"  Comp Monitor : {comp_model_path}")
    print(f"  CO Predict   : {'NO (--no-prediction)' if args.no_prediction else 'YES'}")
    print(f"  Reliability  : {'NO' if (args.no_prediction or args.no_reliability) else 'YES'}")
    print(f"  Optimisation : {'NO' if skip_opt else 'YES'}")
    print(f"  Train        : {'FORCE RETRAIN' if args.train else 'load if exists'}")

    # ── Stages 1 & 2: Ingest + KPI ──────────────────────────────────────────
    enriched = run_ingest_and_kpi(input_path)

    if args.no_prediction:
        save_output(enriched, output_path)
        print(f"\n  Pipeline complete (KPI only)  |  Total time: {_elapsed(pipeline_start)}")
        return

    # ── Stage 3: Load or train CO model ─────────────────────────────────────
    if args.train or not model_path.is_file():
        _banner("SAVING KPI CSV (pre-training)", output_path.name)
        enriched.to_csv(output_path, index=False, encoding="latin-1")
        print(f"  KPI CSV saved ({len(enriched):,} rows)")

    predictor = get_predictor(model_path, output_path, force_train=args.train)

    # ── Stage 4: Batch CO predictions ───────────────────────────────────────
    enriched = run_batch_prediction(enriched, predictor)

    # ── Stage 5: Compressor reliability ─────────────────────────────────────
    monitor = None
    if not args.no_reliability:
        # Pass raw input_path so load_data_for_training() gets friendly column names
        monitor = get_comp_monitor(comp_model_path, input_path, force_train=args.train)
        enriched = run_compressor_scoring(enriched, monitor)

    # ── Stage 6: Optimisation ────────────────────────────────────────────────
    if not skip_opt:
        enriched = run_optimisation(enriched, predictor)

    # ── Stage 7: Save final CSV ──────────────────────────────────────────────
    save_output(enriched, output_path)

    # ── Stage 8: Real-time report for the latest row ─────────────────────────
    print_latest_prediction(enriched, predictor, monitor)

    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE  |  Total time: {_elapsed(pipeline_start)}")
    print(f"  Output saved      : {output_path}")
    print(f"  CO model          : {model_path}")
    if monitor is not None:
        print(f"  Compressor monitor: {comp_model_path}")
    if not skip_opt:
        print(f"  Opt plots         : model_plots/optimization/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
