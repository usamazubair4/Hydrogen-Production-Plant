"""Quick diagnostic of all key dashboard parameters."""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

df = pd.read_csv(ROOT / "Combined_Data_with_KPIs.csv", encoding="latin-1")
print(f"Enriched CSV: {len(df):,} rows  x  {df.shape[1]} columns")

CHECKS = [
    ("Plant Rate",                              "KPI"),
    ("CO in Product",                           "KPI"),
    ("PSA Recovery",                            "KPI"),
    ("Gross Efficiency",                        "KPI"),
    ("Net Efficiency",                          "KPI"),
    ("S/C Ratio (Steam-to-Carbon)",             "KPI"),
    ("Tube Outlet Temperature",                 "KPI"),
    ("Shift dT (HTS Temperature Difference)",   "KPI"),
    ("Excess O2 in Flue Gas",                   "KPI"),
    ("Reformer Differential Pressure (Reformer DP)", "KPI"),
    ("Purge Gas Buffer Vessel Pressure",        "KPI"),
    ("CO Slip (Syngas GC)",                     "KPI"),
    ("CO_Predicted_ppm",                        "ML"),
    ("CO_Alert_Level",                          "ML"),
    ("hts_k_eq",                                "Physics"),
    ("approx_eq_co_pct",                        "Physics"),
    ("approach_to_eq",                          "Physics"),
    ("psa_space_vel_proxy",                     "Physics"),
    ("hts_outlet_temp_c",                       "Physics"),
    ("H2/NG Yield Ratio (SCF/SCF)",             "OptKPI"),
    ("H2 Lost to Purge (MSCFH)",                "OptKPI"),
    ("CO Spec Headroom (Measured)",             "OptKPI"),
    ("S/C Excess over Coking Min",              "OptKPI"),
    ("Carbon Efficiency (%)",                   "OptKPI"),
    ("Production Value Index (%)",              "OptKPI"),
    ("Steam Efficiency Index",                  "OptKPI"),
    ("Reformer Severity Index",                 "OptKPI"),
    ("Compressor_A_Health",                     "Reliability"),
    ("Compressor_A_Alert",                      "Reliability"),
    ("Compressor_A_Bear_Score",                 "Reliability"),
    ("Compressor_A_Vib_Score",                  "Reliability"),
    ("Compressor_A_Oil_Score",                  "Reliability"),
    ("Compressor_A_Cr_Score",                   "Reliability"),
    ("Compressor_A_Anomaly",                    "Reliability"),
    ("Compressor_A_Anomaly_Score",              "Reliability"),
    ("Compressor_B_Health",                     "Reliability"),
    ("Compressor_B_Alert",                      "Reliability"),
    ("Compressor_B_Anomaly",                    "Reliability"),
    ("Compressor_C_Health",                     "Reliability"),
    ("Compressor_C_Alert",                      "Reliability"),
    ("Compressor_C_Anomaly",                    "Reliability"),
    ("CO Spec Headroom (Predicted)",            "PostML"),
    ("HTS Catalyst Utilization (%)",            "PostML"),
    ("Efficiency Gap to Design (BTU/SCF)",      "PostML"),
    ("Steam Cost Index",                        "PostML"),
    ("Steam Reduction Opportunity",             "PostML"),
]

print()
hdr = f"  {'Parameter':<48} {'Cat':<12} {'Valid%':>7}  {'Min':>8}  {'Mean':>8}  {'Max':>8}  Status"
print(hdr)
print("  " + "-" * 100)

issues = []
for col, cat in CHECKS:
    if col not in df.columns:
        print(f"  {col:<48} {cat:<12}  MISSING")
        issues.append((col, cat, "MISSING"))
        continue

    is_cat = cat in ("ML", "Reliability") and ("Alert" in col or col == "CO_Alert_Level")
    if is_cat:
        s = df[col].dropna().astype(str)
        pct = 100 * len(s) / len(df)
        vc = s.value_counts().to_dict()
        cats_str = "  ".join(f"{k}:{v:,}" for k, v in list(vc.items())[:4])
        status = "OK" if pct >= 80 else "PARTIAL"
        print(f"  {col:<48} {cat:<12} {pct:>7.1f}%  {cats_str}")
    else:
        s = pd.to_numeric(df[col], errors="coerce")
        pct = 100 * s.notna().mean()
        mn, mu, mx = s.min(), s.mean(), s.max()

        if pct == 0:
            status = "ALL_ZERO"
        elif s.nunique() <= 1 and pct > 0:
            status = "CONSTANT"
        elif pct < 80:
            status = "SPARSE"
        else:
            status = "OK"

        if status not in ("OK",):
            issues.append((col, cat, f"{pct:.1f}% {status}"))

        print(f"  {col:<48} {cat:<12} {pct:>7.1f}%  {mn:>8.3f}  {mu:>8.3f}  {mx:>8.3f}  {status}")

print()
print("=" * 60)
print("  SUMMARY")
print("=" * 60)
ok = sum(1 for c, cat, note in issues if "MISSING" not in note)
missing = sum(1 for c, cat, note in issues if "MISSING" in note)
print(f"  Total parameters checked : {len(CHECKS)}")
print(f"  OK (>=80% populated)     : {len(CHECKS) - len(issues)}")
print(f"  Issues (zero/sparse)     : {ok}")
print(f"  Missing from CSV         : {missing}")
print()
if not issues:
    print("  All key dashboard parameters generating values!")
else:
    print("  Parameters with issues:")
    for col, cat, note in issues:
        print(f"    [{cat}] {col}: {note}")
