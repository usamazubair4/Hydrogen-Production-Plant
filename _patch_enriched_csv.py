"""
Fast patch — adds all missing columns to Combined_Data_with_KPIs.csv
without re-running the 638-second KPI computation stage.

Adds:
  1. 8 new kpi_formulas optimization KPIs
  2. 5 physics features (hts_k_eq, approach_to_eq, etc.)
  3. 27 compressor health columns (loads saved compressor_monitor.pkl)
  4. 5 post-ML optimization KPIs (CO Spec Headroom Predicted, etc.)
"""
import sys, time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

ENRICHED_CSV  = ROOT / "Combined_Data_with_KPIs.csv"
COMP_MODEL    = ROOT / "compressor_monitor.pkl"
OUT_CSV       = ENRICHED_CSV  # overwrite in-place

t_start = time.time()

def elapsed(t0): return f"{time.time()-t0:.1f}s"
def banner(msg): print(f"\n{'-'*65}\n  {msg}\n{'-'*65}")

# ── Load enriched CSV ─────────────────────────────────────────────────
banner("LOADING enriched CSV")
t0 = time.time()
df = pd.read_csv(ENRICHED_CSV, encoding="latin-1")
print(f"  {len(df):,} rows × {df.shape[1]} columns  ({elapsed(t0)})")

# ── 1. New kpi_formulas optimization KPIs ────────────────────────────
banner("STEP 1 — 8 new kpi_formulas optimization KPIs")
t0 = time.time()
from kpi_formulas import (
    h2_ng_yield_ratio, h2_lost_to_purge, co_spec_headroom_measured,
    sc_excess_over_minimum, carbon_efficiency, production_value_index,
    steam_efficiency_index, reformer_severity_index,
)
new_kpi_map = {
    "H2/NG Yield Ratio (SCF/SCF)":    h2_ng_yield_ratio,
    "H2 Lost to Purge (MSCFH)":       h2_lost_to_purge,
    "CO Spec Headroom (Measured)":     co_spec_headroom_measured,
    "S/C Excess over Coking Min":      sc_excess_over_minimum,
    "Carbon Efficiency (%)":           carbon_efficiency,
    "Production Value Index (%)":      production_value_index,
    "Steam Efficiency Index":          steam_efficiency_index,
    "Reformer Severity Index":         reformer_severity_index,
}
added_kpi = []
for col_name, fn in new_kpi_map.items():
    if col_name not in df.columns:
        df[col_name] = df.apply(fn, axis=1)
        added_kpi.append(col_name)
    else:
        print(f"  [skip] {col_name} already present")

print(f"  Added {len(added_kpi)} KPI columns: {added_kpi}  ({elapsed(t0)})")

# Show coverage
for col in added_kpi:
    s = pd.to_numeric(df[col], errors="coerce")
    pct = 100 * s.notna().mean()
    print(f"    {col:<45}  {pct:.1f}% valid  mean={s.mean():.3f}")

# ── 2. Physics features ───────────────────────────────────────────────
banner("STEP 2 — Physics features (hts_k_eq, approach_to_eq, ...)")
t0 = time.time()
from feature_engineering import compute_physics_features, PHYSICS_FEATURE_COLS

df_phys = compute_physics_features(df)
added_phys = []
for col in PHYSICS_FEATURE_COLS:
    if col in df_phys.columns:
        df[col] = df_phys[col].values
        added_phys.append(col)

print(f"  Added {len(added_phys)} physics columns: {added_phys}  ({elapsed(t0)})")
for col in added_phys:
    s = pd.to_numeric(df[col], errors="coerce")
    pct = 100 * s.notna().mean()
    print(f"    {col:<35}  {pct:.1f}% valid  mean={s.mean():.4f}")

# ── 3. Compressor health scoring ──────────────────────────────────────
banner("STEP 3 — Compressor health scoring (load compressor_monitor.pkl)")
t0 = time.time()
if not COMP_MODEL.is_file():
    print(f"  [SKIP] {COMP_MODEL.name} not found — run main.py to train first")
else:
    import compressor_reliability as _cr
    from compressor_reliability import CompressorHealthMonitor, print_fleet_summary, SENSOR_MAP
    # The pkl was saved when compressor_reliability ran as __main__, so IFWrapper
    # was pickled under __main__. Inject it so joblib can deserialise it.
    import __main__
    __main__.IFWrapper = _cr.IFWrapper
    __main__.CompressorHealthMonitor = _cr.CompressorHealthMonitor
    monitor = CompressorHealthMonitor.load(COMP_MODEL)
    print(f"  Monitor loaded  ({elapsed(t0)})")

    # The enriched CSV uses DCS tag column names; compressor sensor data is
    # identified by FRIENDLY names in the raw CSV (file row 0 = pd.read_csv headers).
    # Load the raw CSV with pd.read_csv() to get friendly-named sensor columns,
    # skip the units/tag metadata rows (iloc[0] and iloc[1]), score with the
    # monitor, then merge the health columns into the enriched df by row index.
    print("  Loading raw CSV for compressor scoring (friendly column names) ...")
    t1 = time.time()
    RAW_CSV = ROOT / "Combined_Data.csv"
    df_raw_friendly = pd.read_csv(RAW_CSV, encoding="latin-1")
    # Collect which friendly sensor columns exist
    all_sensor_cols = {col for sm in SENSOR_MAP.values() for col in sm.values()}
    friendly_cols = [c for c in df_raw_friendly.columns if c in all_sensor_cols]
    ts_col_raw = df_raw_friendly.columns[0]  # "Date:"
    score_input = df_raw_friendly[[ts_col_raw] + friendly_cols].copy()
    # Rows iloc[0]=units, iloc[1]=tags are metadata; actual data starts at iloc[2]
    score_input = score_input.iloc[2:].reset_index(drop=True)
    score_input = score_input.rename(columns={ts_col_raw: "Timestamp"})
    score_input["Timestamp"] = pd.to_datetime(score_input["Timestamp"], errors="coerce")
    for c in friendly_cols:
        score_input[c] = pd.to_numeric(score_input[c], errors="coerce")
    print(f"  Scoring input: {len(score_input):,} rows x {len(friendly_cols)} sensor cols  ({elapsed(t1)})")

    t2 = time.time()
    scored = monitor.score_dataframe(score_input)
    health_cols = [c for c in scored.columns if c.startswith("Compressor_")]
    # Align by row index — enriched and raw data rows are in the same order
    if len(scored) == len(df):
        for c in health_cols:
            df[c] = scored[c].values
        print(f"  Added {len(health_cols)} compressor health columns  ({elapsed(t2)})")
    else:
        # Fall back to timestamp merge if row counts differ
        print(f"  Row count mismatch ({len(scored)} vs {len(df)}), merging by Timestamp ...")
        scored_slim = scored[["Timestamp"] + health_cols].copy()
        df = df.merge(scored_slim, on="Timestamp", how="left", suffixes=("", "_scored"))
        print(f"  Merged by Timestamp  ({elapsed(t2)})")

    print_fleet_summary(df)

    for comp in ["A", "B", "C"]:
        alrt = f"Compressor_{comp}_Alert"
        health = f"Compressor_{comp}_Health"
        if alrt in df.columns:
            vc = df[alrt].value_counts()
            hi = pd.to_numeric(df[health], errors="coerce")
            print(f"  Comp {comp}  health={hi.mean():.1f}  "
                  f"green={vc.get('green',0):,}  amber={vc.get('amber',0):,}  "
                  f"red={vc.get('red',0):,}  offline={vc.get('offline',0):,}")

# ── 4. Post-ML optimization KPIs ─────────────────────────────────────
banner("STEP 4 — Post-ML optimization KPIs (needs CO_Predicted_ppm)")
t0 = time.time()
if "CO_Predicted_ppm" not in df.columns:
    print("  [SKIP] CO_Predicted_ppm not in df — run main.py with prediction first")
else:
    from optimization import compute_opt_kpis
    df = compute_opt_kpis(df)
    opt_new = ["CO Spec Headroom (Predicted)", "HTS Catalyst Utilization (%)",
               "Efficiency Gap to Design (BTU/SCF)", "Steam Cost Index",
               "Steam Reduction Opportunity"]
    print(f"  Computed post-ML KPIs  ({elapsed(t0)})")
    for col in opt_new:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            pct = 100 * s.notna().mean()
            print(f"    {col:<45}  {pct:.1f}% valid  mean={s.mean():.3f}")
        else:
            print(f"    [MISSING] {col}")

# ── 5. Save updated CSV ───────────────────────────────────────────────
banner("STEP 5 — Saving updated CSV")
t0 = time.time()
df.to_csv(OUT_CSV, index=False, encoding="latin-1")
size_mb = OUT_CSV.stat().st_size / 1_048_576
print(f"  Saved {len(df):,} rows × {df.shape[1]} columns  →  {size_mb:.1f} MB  ({elapsed(t0)})")

print(f"\n{'='*65}")
print(f"  PATCH COMPLETE  |  Total time: {elapsed(t_start)}")
print(f"  Output: {OUT_CSV}")
print(f"{'='*65}\n")
