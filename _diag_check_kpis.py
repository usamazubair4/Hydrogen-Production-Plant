"""
KPI / Tag diagnostic — checks every dashboard parameter for data coverage.
Loads the existing enriched CSV and tests all new kpi_formulas KPIs
against the raw data as well.
"""
import math, sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

ENRICHED_CSV = ROOT / "Combined_Data_with_KPIs.csv"
RAW_CSV      = ROOT / "Combined_Data.csv"

print("=" * 70)
print("  KPI / TAG DIAGNOSTIC — SMR Plant Dashboard")
print("=" * 70)

# ── Load enriched CSV (has old KPIs + CO predictions) ────────────────────────
print(f"\nLoading enriched CSV: {ENRICHED_CSV.name}")
df = pd.read_csv(ENRICHED_CSV, encoding="latin-1")
df["Timestamp"] = pd.to_datetime(df.get("Timestamp", df.get("Date:", "")), errors="coerce")
df = df.sort_values("Timestamp").reset_index(drop=True)
print(f"  Rows: {len(df):,}   Columns: {df.shape[1]}")
ts_range = f"{df['Timestamp'].dropna().iloc[0].date()} to {df['Timestamp'].dropna().iloc[-1].date()}"
print(f"  Period: {ts_range}")

# ── Load raw CSV for new kpi_formulas tests ───────────────────────────────────
print(f"\nLoading raw CSV: {RAW_CSV.name}")
df_raw = pd.read_csv(RAW_CSV, encoding="latin-1")
ts_col_raw = next((c for c in df_raw.columns if "timestamp" in c.lower() or c.strip().lower() == "timestamp"), df_raw.columns[0])
df_raw = df_raw.rename(columns={ts_col_raw: "Timestamp"})

# ── Apply new kpi_formulas KPIs to raw data ───────────────────────────────────
print("\nApplying NEW kpi_formulas optimization KPIs to raw data ...")
from kpi_formulas import (
    h2_ng_yield_ratio, h2_lost_to_purge, co_spec_headroom_measured,
    sc_excess_over_minimum, carbon_efficiency, production_value_index,
    steam_efficiency_index, reformer_severity_index,
)
new_kpi_funcs = {
    "H2/NG Yield Ratio":           h2_ng_yield_ratio,
    "H2 Lost to Purge":            h2_lost_to_purge,
    "CO Spec Headroom (Measured)": co_spec_headroom_measured,
    "S/C Excess over Coking Min":  sc_excess_over_minimum,
    "Carbon Efficiency":           carbon_efficiency,
    "Production Value Index":      production_value_index,
    "Steam Efficiency Index":      steam_efficiency_index,
    "Reformer Severity Index":     reformer_severity_index,
}
for name, fn in new_kpi_funcs.items():
    df_raw[name] = df_raw.apply(fn, axis=1)

# Apply post-ML opt KPIs to enriched df
from optimization import compute_opt_kpis
df = compute_opt_kpis(df)

# ── Helper ────────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"

def check_col(source_df, col_name, label=None):
    label = label or col_name
    if col_name not in source_df.columns:
        return {"col": label, "status": "MISSING", "pct_valid": 0, "min": None, "mean": None, "max": None, "sample": None}
    s = pd.to_numeric(source_df[col_name], errors="coerce")
    n_valid = s.notna().sum()
    pct = 100 * n_valid / len(s)
    if n_valid == 0:
        status = "ALL NaN"
    elif pct < 10:
        status = "SPARSE"
    elif pct < 80:
        status = "PARTIAL"
    else:
        status = "OK"
    sample = s.dropna().iloc[-1] if n_valid > 0 else None
    return {
        "col": label, "status": status, "pct_valid": pct,
        "min": s.min() if n_valid else None,
        "mean": s.mean() if n_valid else None,
        "max": s.max() if n_valid else None,
        "sample": sample,
    }

def print_section(title):
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")
    print(f"  {'Parameter':<45} {'Status':<10} {'Valid%':>7}  {'Min':>9}  {'Mean':>9}  {'Max':>9}  Latest")

def print_row(r):
    status = r["status"]
    color = GREEN if status == "OK" else (YELLOW if status in ("PARTIAL","SPARSE") else RED)
    pct   = f"{r['pct_valid']:.1f}%" if r['pct_valid'] else "—"
    mn    = f"{r['min']:.2f}"   if r['min']  is not None else "—"
    mu    = f"{r['mean']:.2f}"  if r['mean'] is not None else "—"
    mx    = f"{r['max']:.2f}"   if r['max']  is not None else "—"
    samp  = f"{r['sample']:.2f}" if r['sample'] is not None else "—"
    print(f"  {r['col']:<45} {color}{status:<10}{RESET} {pct:>7}  {mn:>9}  {mu:>9}  {mx:>9}  {samp}")

def check_string_col(source_df, col_name, label=None):
    label = label or col_name
    if col_name not in source_df.columns:
        return {"col": label, "status": "MISSING", "pct_valid": 0, "counts": {}}
    s = source_df[col_name].dropna().astype(str)
    pct = 100 * len(s) / len(source_df)
    vc = s.value_counts().to_dict()
    status = "OK" if pct >= 80 else ("PARTIAL" if pct >= 10 else "SPARSE")
    return {"col": label, "status": status, "pct_valid": pct, "counts": vc}

def print_cat_row(r):
    color = GREEN if r["status"] == "OK" else (YELLOW if r["status"] == "PARTIAL" else RED)
    pct   = f"{r['pct_valid']:.1f}%"
    cats  = "  |  ".join(f"{k}: {v:,}" for k, v in list(r["counts"].items())[:5])
    print(f"  {r['col']:<45} {color}{r['status']:<10}{RESET} {pct:>7}  {cats}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
print_section("PAGE 1 — OVERVIEW")
for c in ["Plant Rate", "CO in Product", "PSA Recovery", "Production Value Index",
          "Gross Efficiency", "Efficiency Gap to Design (BTU/SCF)"]:
    src = df_raw if c in ["Production Value Index"] else df
    print_row(check_col(df if c in df.columns else df_raw, c))

for comp in ["A","B","C"]:
    print_row(check_col(df, f"Compressor_{comp}_Health"))
    r = check_string_col(df, f"Compressor_{comp}_Alert")
    print_cat_row(r)

print_row(check_col(df, "CO_Predicted_ppm"))
print_row(check_col(df, "CO Spec Headroom (Predicted)"))

r = check_string_col(df, "CO_Alert_Level")
print_cat_row(r)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
print_section("PAGE 2 — PERFORMANCE")
perf_cols = [
    ("Gross Efficiency", df), ("Net Efficiency", df), ("Burner Efficiency", df),
    ("Efficiency Gap to Design (BTU/SCF)", df),
    ("Tube Outlet Temperature", df), ("Reformer Severity Index", df_raw),
    ("Reformer Differential Pressure (Reformer DP)", df),
    ("Excess O2 in Flue Gas", df), ("S/C Ratio (Steam-to-Carbon)", df),
    ("S/C Excess over Coking Min", df_raw),
    ("Shift dT (HTS Temperature Difference)", df),
    ("Hydrotreater Outlet Temperature", df), ("Hydrotreater Out A", df),
    ("PSA Recovery", df), ("H2 Lost to Purge", df_raw),
    ("Purge Gas Buffer Vessel Pressure", df),
    ("CO Slip (Syngas GC)", df), ("Methane Slip (Syngas GC)", df), ("CH4  Syngas GC", df),
    ("Purge Gas Vent", df), ("Midplant Vent", df),
    ("PSA Vent (SMR PSA Vent)", df), ("Product Vent", df), ("Steam Vent", df),
    ("H2/NG Yield Ratio", df_raw), ("Carbon Efficiency", df_raw),
    ("Steam Efficiency Index", df_raw), ("Production Value Index", df_raw),
    # Material balances
    ("NG Check (Material Balance)", df), ("Steam Balance (Material Balance)", df),
    ("RFG Agreement (Material Balance)", df),
    ("Hydrocarbon (HC) Balance (Material Balance)", df),
    ("Hydrcarbon/Recycle H2 (HC/H2) Balance (Material Balance)", df),
    ("Mix Tee Balance (Material Balance)", df), ("Burner Balance (Material Balance)", df),
    ("PSA Balance (Material Balance)", df), ("Coker Agreement (Material Balance)", df),
    ("Hydrogen Balance (Material Balance)", df), ("NG Balance", df),
]
for col, src in perf_cols:
    print_row(check_col(src, col))

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
print_section("PAGE 3 — PREDICTION (CO Quality)")
pred_cols = [
    "CO_Predicted_ppm", "CO Spec Headroom (Predicted)", "CO Spec Headroom (Measured)",
    "CO in Product",
    "hts_outlet_temp_c", "hts_k_eq", "approx_eq_co_pct", "approach_to_eq",
    "HTS Catalyst Utilization (%)", "psa_space_vel_proxy",
]
for c in pred_cols:
    print_row(check_col(df, c))
r = check_string_col(df, "CO_Alert_Level")
print_cat_row(r)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — RELIABILITY
# ══════════════════════════════════════════════════════════════════════════════
print_section("PAGE 4 — RELIABILITY (Compressor Health)")
for comp in ["A", "B", "C"]:
    print(f"\n  -- Compressor {comp} --")
    c_cols = [
        f"Compressor_{comp}_Health", f"Compressor_{comp}_Bear_Score",
        f"Compressor_{comp}_Vib_Score", f"Compressor_{comp}_Oil_Score",
        f"Compressor_{comp}_Cr_Score", f"Compressor_{comp}_Anomaly",
        f"Compressor_{comp}_Anomaly_Score",
        f"Compressor {comp} Motor Current",
        f"Compressor {comp} Hottest Bearing Temperature",
        f"Compressor {comp} Oil Filter dP",
        f"Compressor {comp} Oil Pressure",
        f"Compressor {comp} Oil Temperature",
        f"Compressor {comp} Motor DE Vibration",
        f"Compressor {comp} Frame DE Vibration",
        f"Compressor {comp} Interstage Cooler Vibration",
        f"Compressor {comp} Interstage Cooler Vibration.1",
        f"Compressor {comp} 1st Stage H2 Compression Ratio",
        f"Compressor {comp} 2nd Stage H2 Compression Ratio",
        f"Compressor {comp} 3rd Stage H2 Compression Ratio",
    ]
    for col in c_cols:
        # Try enriched df first, then raw
        src = df if col in df.columns else df_raw
        print_row(check_col(src, col))
    r = check_string_col(df, f"Compressor_{comp}_Alert")
    print_cat_row(r)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — OPTIMISATION
# ══════════════════════════════════════════════════════════════════════════════
print_section("PAGE 6 — OPTIMISATION")
opt_cols = [
    ("CO Spec Headroom (Predicted)", df), ("CO Spec Headroom (Measured)", df),
    ("HTS Catalyst Utilization (%)", df),
    ("Efficiency Gap to Design (BTU/SCF)", df),
    ("Steam Cost Index", df), ("Steam Reduction Opportunity", df),
    ("H2/NG Yield Ratio", df_raw), ("Carbon Efficiency", df_raw),
    ("H2 Lost to Purge", df_raw), ("Production Value Index", df_raw),
    ("Steam Efficiency Index", df_raw), ("S/C Excess over Coking Min", df_raw),
    ("Reformer Severity Index", df_raw),
    ("S/C Ratio (Steam-to-Carbon)", df), ("Plant Rate", df),
    ("Gross Efficiency", df), ("Net Efficiency", df), ("PSA Recovery", df),
    ("approach_to_eq", df), ("hts_k_eq", df),
]
for col, src in opt_cols:
    print_row(check_col(src, col))

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  SUMMARY")
print(f"{'='*70}")

all_checks = []
for c in df.columns:
    s = pd.to_numeric(df[c], errors="coerce")
    pct = 100 * s.notna().mean()
    all_checks.append((c, pct))

ok      = sum(1 for _, p in all_checks if p >= 80)
partial = sum(1 for _, p in all_checks if 10 <= p < 80)
sparse  = sum(1 for _, p in all_checks if 0 < p < 10)
empty   = sum(1 for _, p in all_checks if p == 0)

print(f"  Total columns in enriched CSV : {len(all_checks):,}")
print(f"  OK     (>= 80% populated)     : {ok:,}")
print(f"  Partial (10–80% populated)    : {partial:,}")
print(f"  Sparse  (< 10% populated)     : {sparse:,}")
print(f"  All NaN (0%)                  : {empty:,}")
print()

# Show the key dashboard KPIs that are ALL NaN or MISSING
print("  KEY DASHBOARD PARAMETERS WITH ISSUES:")
dashboard_key = [
    "H2/NG Yield Ratio", "H2 Lost to Purge", "CO Spec Headroom (Measured)",
    "S/C Excess over Coking Min", "Carbon Efficiency", "Production Value Index",
    "Steam Efficiency Index", "Reformer Severity Index",
    "CO Spec Headroom (Predicted)", "HTS Catalyst Utilization (%)",
    "Efficiency Gap to Design (BTU/SCF)", "Steam Cost Index",
    "Steam Reduction Opportunity",
    "CO_Predicted_ppm", "CO_Alert_Level",
    "Compressor_A_Health", "Compressor_B_Health", "Compressor_C_Health",
    "Gross Efficiency", "Net Efficiency", "PSA Recovery", "Plant Rate",
    "S/C Ratio (Steam-to-Carbon)", "CO in Product",
    "hts_k_eq", "approach_to_eq",
]
issues = []
for col in dashboard_key:
    src = df_raw if col in df_raw.columns and col not in df.columns else df
    if col not in src.columns:
        issues.append((col, "MISSING FROM CSV"))
    else:
        s = pd.to_numeric(src[col], errors="coerce")
        pct = 100 * s.notna().mean()
        if pct < 80:
            issues.append((col, f"{pct:.1f}% valid"))

if not issues:
    print(f"  {GREEN}All key dashboard parameters are generating values (>= 80% coverage){RESET}")
else:
    for col, reason in issues:
        print(f"  {RED}[ISSUE]{RESET}  {col:<50} {reason}")
