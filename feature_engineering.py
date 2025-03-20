"""
Feature engineering for CO-in-product prediction — SMR Plant.

Hybrid "grey-box" model philosophy
===================================
Pure data-driven models can overfit to historical patterns and fail when the
plant moves to a new operating regime.  Pure first-principles models require
full composition analysis that is not always available in real time.

This module sits between: it translates fundamental chemical-engineering
knowledge (Water-Gas Shift thermodynamics, PSA mass balance) into numeric
features that give a gradient-boosting model physically meaningful axes along
which to generalise.

Process overview
----------------
  Natural gas + steam
        │
        ▼ Reformer (800–900 °C)
  Syngas (H2, CO, CO2, CH4, H2O)
        │
        ▼ HTS reactor (300–450 °C)   ← WGS: CO + H2O → CO2 + H2
  Shifted syngas (lower CO)
        │
        ▼ PSA unit
  H2 product (target: low CO in ppm) + purge gas

Key CO drivers (in causal order):
  1. CO entering PSA  ← set by reformer + HTS performance
  2. PSA CO removal   ← set by space velocity, purge quality, recovery target
  3. Net CO in product = CO_feed × (1 − removal_efficiency)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

try:
    from kpi_formulas import find_column
except ImportError:
    def find_column(columns: Any, key: str) -> str | None:  # type: ignore[misc]
        key_n = "".join(c.lower() if c.isalnum() else "_" for c in str(key)).strip("_")
        avail = {"".join(c.lower() if c.isalnum() else "_" for c in str(col)).strip("_"): col
                 for col in columns}
        return avail.get(key_n)


# ── DCS tag strings (primary key for find_column resolver) ──────────────────
_TAG_HTS_OUTLET = "PLANT_01:12TI0633_S/ALM1/PV.CV"
_TAG_HTS_INLET  = "PLANT_01:11TIC0060/PID1/PV.CV"
_TAG_SYNGAS_PSA = "PLANT_01:12FIC0050/PID1/PV.CV"
_TAG_PSA_H2     = "PLANT_01:18FI0121/AI1/PV.CV"
_TAG_CO_SLIP    = "PLANT_01:70AI_0275D/AI1/PV.CV"

# ── KPI column names (present in Combined_Data_with_KPIs.csv) ───────────────
# These are the pre-computed engineering KPIs — use as model input features.
# "CO in Product" is excluded here; it is the target.
KPI_FEATURE_COLS: list[str] = [
    "CO Slip (Syngas GC)",               # % CO in syngas → primary upstream driver
    "Methane Slip (Syngas GC)",          # reformer performance indicator
    "CH4  Syngas GC",                    # duplicate channel (different GC)
    "Shift dT (HTS Temperature Difference)",  # HTS conversion proxy
    "PSA Recovery",                      # PSA operating point
    "Plant Rate",                        # throughput / PSA space velocity driver
    "S/C Ratio (Steam-to-Carbon)",       # WGS equilibrium driver
    "Tube Outlet Temperature",           # reformer severity → sets initial CO
    "Excess O2 in Flue Gas",             # firing conditions
    "Purge Gas Buffer Vessel Pressure",  # PSA purge quality driver
    "Reformer Differential Pressure (Reformer DP)",  # equipment health
    "Hydrotreater Outlet Temperature",   # feed pretreatment quality
    "Hydrotreater Out A",                # individual sensor
    "PSA Vent (SMR PSA Vent)",           # PSA vent valve position
    "Purge Gas Vent",                    # PGB vent valve position
    "Gross Efficiency",                  # overall plant efficiency (regime indicator)
    "Net Efficiency",                    # net efficiency (accounts for steam export)
    "PSA Balance (Material Balance)",    # data quality / PSA integrity check
    "Steam Balance (Material Balance)",  # steam system integrity
    "NG Balance",                        # NG metering consistency
    "S/C/OUT.CV",                        # steam-to-carbon controller output
]

TARGET_COL = "CO in Product"  # ppm  ← prediction target

# Physics-derived features added by compute_physics_features()
PHYSICS_FEATURE_COLS: list[str] = [
    "hts_k_eq",            # WGS equilibrium constant at HTS outlet T
    "approx_eq_co_pct",    # estimated equilibrium CO% at HTS outlet T
    "approach_to_eq",      # actual CO / equilibrium CO (catalyst health index)
    "psa_space_vel_proxy", # syngas_to_PSA / PSA_H2_flow (PSA loading intensity)
    "hts_outlet_temp_c",   # HTS outlet in °C (for thermodynamic context)
]

# ── Feature documentation ────────────────────────────────────────────────────
# Structured rationale for every selected feature — used in EDA report.
FEATURE_RATIONALE: dict[str, dict[str, str]] = {
    "CO Slip (Syngas GC)": {
        "unit": "%",
        "category": "Primary — upstream mass balance",
        "mechanism": (
            "By simple molar balance around the PSA: CO_product = CO_syngas × (1 − η_PSA). "
            "The PSA must adsorb all CO in its feed on each cycle. Higher feed CO means "
            "larger adsorptive load per cycle and greater risk of breakthrough."
        ),
        "expected_relationship": "Strong positive correlation with CO in product.",
        "lag_sensitivity": "Immediate — GC reads syngas composition continuously.",
    },
    "Shift dT (HTS Temperature Difference)": {
        "unit": "°F",
        "category": "Primary — HTS conversion",
        "mechanism": (
            "CO + H2O → CO2 + H2 is exothermic (ΔH ≈ −41 kJ/mol). "
            "Temperature rise across the HTS bed is directly proportional to moles of CO converted. "
            "Low ΔT signals catalyst deactivation, flow maldistribution, or insufficient steam."
        ),
        "expected_relationship": "Negative — higher ΔT means less CO entering PSA.",
        "lag_sensitivity": "Low lag — reflects current catalyst activity.",
    },
    "PSA Recovery": {
        "unit": "%",
        "category": "Primary — PSA operating point",
        "mechanism": (
            "Recovery = H2_product / (syngas × y_H2). As recovery increases, the purge-to-feed "
            "ratio falls, adsorbent beds are less completely regenerated, and CO accumulates "
            "in the beds over successive cycles until eventual breakthrough."
        ),
        "expected_relationship": "Positive — higher recovery → higher CO in product (non-linear).",
        "lag_sensitivity": "Medium — PSA cycle dynamics introduce ~minutes delay.",
    },
    "Plant Rate": {
        "unit": "%",
        "category": "Primary — PSA space velocity",
        "mechanism": (
            "Higher plant rate → higher volumetric feed to PSA per unit bed volume → "
            "shorter adsorption contact time per cycle → less complete CO capture. "
            "Effect is most pronounced above ~90% of design rate."
        ),
        "expected_relationship": "Positive at high plant rates; weak effect at low rates.",
        "lag_sensitivity": "Immediate effect on PSA loading.",
    },
    "S/C Ratio (Steam-to-Carbon)": {
        "unit": "mol/mol",
        "category": "Primary — WGS equilibrium driver",
        "mechanism": (
            "Higher steam stoichiometry pushes WGS equilibrium to the right (Le Chatelier). "
            "Also prevents carbon deposition in reformer tubes, which would degrade catalyst "
            "and eventually restrict flow. S/C < 2.5 risks coking."
        ),
        "expected_relationship": "Negative — more steam → lower equilibrium CO → less CO in product.",
        "lag_sensitivity": "Low lag — equilibrium response is fast at HTS temperatures.",
    },
    "Tube Outlet Temperature": {
        "unit": "°F",
        "category": "Secondary — reformer severity",
        "mechanism": (
            "Reforming equilibrium: CH4 + H2O ⇌ CO + 3H2 is endothermic. "
            "Higher tube outlet T → higher CH4 conversion → more CO in syngas before HTS. "
            "This sets the baseline CO load that HTS must reduce."
        ),
        "expected_relationship": "Positive (moderated by downstream HTS response).",
        "lag_sensitivity": "Medium — affects syngas composition entering HTS.",
    },
    "Excess O2 in Flue Gas": {
        "unit": "%",
        "category": "Secondary — firing conditions",
        "mechanism": (
            "O2 < 1%: risk of incomplete combustion, cold spots, poor reforming. "
            "O2 > 4%: over-fired, excess heat may cause tube hot spots and maldistribution. "
            "Optimal range ≈ 2–3% for uniform reformer performance."
        ),
        "expected_relationship": "Non-linear; deviations from optimum correlate with CO excursions.",
        "lag_sensitivity": "Immediate effect on reformer heat balance.",
    },
    "Purge Gas Buffer Vessel Pressure": {
        "unit": "psig",
        "category": "Secondary — PSA purge quality",
        "mechanism": (
            "PGB pressure is the driving force for the PSA purge step. "
            "Low PGB pressure → weak purge flow → adsorbent beds carry residual CO into "
            "the next adsorption cycle → progressive CO accumulation → breakthrough."
        ),
        "expected_relationship": "Negative — lower pressure → weaker purge → more CO.",
        "lag_sensitivity": "Medium — effect builds over multiple PSA cycles.",
    },
    "hts_k_eq": {
        "unit": "dimensionless",
        "category": "First-Principles — WGS thermodynamics",
        "mechanism": (
            "K_eq(T) = exp(4577.8/T_K − 4.33) [Moe 1962]. "
            "This is the thermodynamic ceiling: higher K_eq (lower T) means more CO can be "
            "converted at equilibrium. The gap between K_eq and actual performance reveals "
            "how close the reactor is to its thermodynamic limit."
        ),
        "expected_relationship": "Negative — higher K_eq → lower equilibrium CO → less CO in product.",
        "lag_sensitivity": "Reflects steady-state thermodynamic condition.",
    },
    "approx_eq_co_pct": {
        "unit": "%",
        "category": "First-Principles — WGS equilibrium CO estimate",
        "mechanism": (
            "Computed by solving WGS quadratic (K_eq = (y_CO2+ε)(y_H2+ε)/((y_CO-ε)(y_H2O-ε))) "
            "with assumed typical SMR syngas composition at HTS inlet. "
            "Represents the minimum CO achievable at the current HTS outlet temperature. "
            "Note: uses fixed assumed inlet composition — actual accuracy depends on reformer conditions."
        ),
        "expected_relationship": "Positive — higher equilibrium CO → more CO reaching PSA.",
        "lag_sensitivity": "Steady-state thermodynamic estimate.",
    },
    "approach_to_eq": {
        "unit": "ratio (actual CO% / equilibrium CO%)",
        "category": "First-Principles — HTS catalyst health index",
        "mechanism": (
            "Ratio of measured syngas CO to thermodynamic equilibrium CO at the same temperature. "
            "  ~ 1.0: HTS operating near equilibrium — catalyst healthy, good flow distribution. "
            "  >> 1.0: kinetically limited — catalyst may be deactivating (sulfur poisoning, "
            "           sintering) or flow is maldistributed (bypassing catalyst)."
            "A rising trend in this ratio is a leading indicator of HTS degradation."
        ),
        "expected_relationship": "Positive — poor HTS performance → more CO into PSA → higher CO in product.",
        "lag_sensitivity": "Steady-state indicator; trend over days/weeks is most meaningful.",
    },
    "psa_space_vel_proxy": {
        "unit": "dimensionless (syngas flow / H2 product flow)",
        "category": "First-Principles — PSA loading intensity",
        "mechanism": (
            "A higher ratio means more syngas (with CO impurity) is being processed "
            "per unit of H2 delivered. This proxy captures the combined effect of "
            "plant rate and PSA recovery on CO loading per cycle."
        ),
        "expected_relationship": "Positive — higher loading → more CO in product.",
        "lag_sensitivity": "Immediate.",
    },
    "Methane Slip (Syngas GC)": {
        "unit": "%",
        "category": "Secondary — reformer performance",
        "mechanism": (
            "Residual CH4 in syngas means incomplete reforming. At low reformer severity, "
            "less CH4 is converted to CO/H2, so syngas CO content is typically lower "
            "(un-reformed CH4 has not become CO). Useful as a reformer regime indicator."
        ),
        "expected_relationship": "Generally negative correlation with CO% in syngas.",
        "lag_sensitivity": "Immediate — GC measures continuously.",
    },
    "Reformer Differential Pressure (Reformer DP)": {
        "unit": "psid",
        "category": "Diagnostic — equipment health",
        "mechanism": (
            "Rising DP signals tube fouling, coking, or catalyst degradation. "
            "These conditions cause flow maldistribution, creating hot/cold tube zones "
            "that produce non-uniform syngas composition."
        ),
        "expected_relationship": "Operating regime indicator; sharp increases correlate with excursions.",
        "lag_sensitivity": "Slow — develops over days to weeks.",
    },
    "PSA Balance (Material Balance)": {
        "unit": "%",
        "category": "Data Quality — closure check",
        "mechanism": (
            "Closure of the molar balance around the PSA unit. "
            "Poor closure (>5%) may indicate instrument errors, valve leaks, or "
            "abnormal PSA behaviour that could also produce anomalous CO readings."
        ),
        "expected_relationship": "Quality flag — high imbalance rows may need filtering.",
        "lag_sensitivity": "Instantaneous calculated value.",
    },
}


# ── Thermodynamic helpers ────────────────────────────────────────────────────

def fahrenheit_to_celsius(t_f: float) -> float:
    return (t_f - 32.0) * 5.0 / 9.0


def hts_equilibrium_constant(T_celsius: float) -> float:
    """
    Water-Gas Shift equilibrium constant at temperature T (Celsius).
    K_eq = exp(4577.8 / T_K − 4.33)   [Moe 1962, simplified]
    Valid range: 200–600 °C (typical HTS: 300–450 °C).
    """
    if math.isnan(T_celsius):
        return math.nan
    T_K = T_celsius + 273.15
    if T_K < 400:
        return math.nan
    return math.exp(4577.8 / T_K - 4.33)


def approx_equilibrium_co_pct(
    T_celsius: float,
    y_co_in:  float = 0.15,
    y_h2o_in: float = 0.12,
    y_co2_in: float = 0.08,
    y_h2_in:  float = 0.65,
) -> float:
    """
    Estimate equilibrium CO mol% (dry basis) at HTS outlet temperature.

    Solves WGS quadratic for ε (moles CO converted per mole feed):
      K_eq = (y_CO2 + ε)(y_H2 + ε) / ((y_CO − ε)(y_H2O − ε))

    Default inlet mole fractions are representative of typical SMR syngas
    entering the HTS reactor.  Actual values shift with reformer operating
    conditions, so this is an engineering approximation, not an exact
    calculation — but it is sufficient to define a thermodynamic reference.

    Returns: equilibrium CO% on dry syngas basis, or nan if not solvable.
    """
    K_eq = hts_equilibrium_constant(T_celsius)
    if math.isnan(K_eq) or K_eq <= 0:
        return math.nan

    # Quadratic coefficients
    a = K_eq - 1.0
    b = -(K_eq * (y_co_in + y_h2o_in) + y_co2_in + y_h2_in)
    c = K_eq * y_co_in * y_h2o_in - y_co2_in * y_h2_in

    if abs(a) < 1e-10:
        eps = -c / b if abs(b) > 1e-10 else 0.0
    else:
        disc = b ** 2 - 4.0 * a * c
        if disc < 0:
            return math.nan
        sq = math.sqrt(disc)
        limit = min(y_co_in, y_h2o_in)
        candidates = [(-b + sq) / (2 * a), (-b - sq) / (2 * a)]
        valid = [e for e in candidates if 0.0 <= e <= limit]
        if not valid:
            return math.nan
        eps = max(valid)

    return max(0.0, (y_co_in - eps) * 100.0)


# ── DataFrame feature computation ────────────────────────────────────────────

def _col_to_numeric(df: pd.DataFrame, col: str | None) -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def compute_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append first-principles thermodynamic features to a copy of df.

    Works with both:
      - Raw cleaned DataFrame (columns are DCS tag strings)
      - KPI-enriched DataFrame (also contains friendly-named KPI columns)
    """
    out = df.copy()

    # Resolve raw sensor columns via tag-name resolver
    hts_out_col = find_column(df.columns, _TAG_HTS_OUTLET)
    hts_in_col  = find_column(df.columns, _TAG_HTS_INLET)
    syngas_col  = find_column(df.columns, _TAG_SYNGAS_PSA)
    psa_h2_col  = find_column(df.columns, _TAG_PSA_H2)
    co_slip_col = find_column(df.columns, _TAG_CO_SLIP)

    # Prefer KPI-column names for CO slip if raw tag not resolved
    if co_slip_col is None and "CO Slip (Syngas GC)" in df.columns:
        co_slip_col = "CO Slip (Syngas GC)"

    # --- HTS outlet temperature in °C -----------------------------------------
    hts_f = _col_to_numeric(df, hts_out_col)
    if hts_out_col is None and "Shift dT (HTS Temperature Difference)" in df.columns and hts_in_col:
        # Reconstruct outlet from inlet + ΔT if raw tag missing
        shift_dt = _col_to_numeric(df, "Shift dT (HTS Temperature Difference)")
        hts_in_f = _col_to_numeric(df, hts_in_col)
        hts_f = hts_in_f + shift_dt

    def _f2c(t: float) -> float:
        return fahrenheit_to_celsius(t) if pd.notna(t) and not math.isnan(float(t)) else math.nan

    hts_c = hts_f.apply(_f2c)
    out["hts_outlet_temp_c"] = hts_c

    # --- WGS equilibrium constant ----------------------------------------------
    out["hts_k_eq"] = hts_c.apply(
        lambda t: hts_equilibrium_constant(t) if pd.notna(t) else math.nan
    )

    # --- Approximate equilibrium CO% ------------------------------------------
    out["approx_eq_co_pct"] = hts_c.apply(
        lambda t: approx_equilibrium_co_pct(t) if pd.notna(t) else math.nan
    )

    # --- Approach to equilibrium -----------------------------------------------
    co_slip = _col_to_numeric(df, co_slip_col)
    eq_co   = out["approx_eq_co_pct"].replace(0.0, np.nan)
    out["approach_to_eq"] = (co_slip / eq_co).clip(0.0, 10.0)

    # --- PSA space velocity proxy ---------------------------------------------
    syngas = _col_to_numeric(df, syngas_col)
    psa_h2 = _col_to_numeric(df, psa_h2_col)
    # Fall back to KPI-column proxies if raw tags missing
    if syngas_col is None and "PSA Balance (Material Balance)" in df.columns:
        pass  # cannot reconstruct without raw flow tags
    out["psa_space_vel_proxy"] = (syngas / psa_h2.replace(0.0, np.nan)).clip(0.0, 20.0)

    return out


def extract_feature_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Build the ML feature matrix X and target series y.

    Returns:
        X: DataFrame of numeric features (NaN retained — imputed in model pipeline)
        y: Series of CO in product (ppm), or None if TARGET_COL not in df
    """
    df2 = compute_physics_features(df)

    feature_cols = (
        [c for c in KPI_FEATURE_COLS if c in df2.columns and c != TARGET_COL]
        + [c for c in PHYSICS_FEATURE_COLS if c in df2.columns]
    )

    X = df2[feature_cols].copy().apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df2[TARGET_COL], errors="coerce") if TARGET_COL in df2.columns else None

    return X, y


def build_feature_row(sensor_readings: dict[str, float]) -> pd.DataFrame:
    """
    Build a single-row feature DataFrame from a dict of current sensor values.
    Used for real-time prediction in COPredictor.

    sensor_readings keys must match KPI column names (as in KPI_FEATURE_COLS)
    plus optionally the raw DCS tag strings for physics feature computation.

    Example:
        row = build_feature_row({
            "CO Slip (Syngas GC)": 0.82,
            "Shift dT (HTS Temperature Difference)": 72.0,
            "PSA Recovery": 88.5,
            "Plant Rate": 94.2,
            "S/C Ratio (Steam-to-Carbon)": 3.1,
            "Tube Outlet Temperature": 1565.0,
            "Excess O2 in Flue Gas": 2.4,
            "Purge Gas Buffer Vessel Pressure": 285.0,
            "PLANT_01:12TI0633_S/ALM1/PV.CV": 720.0,  # HTS outlet °F
            "PLANT_01:12FIC0050/PID1/PV.CV": 38.5,    # syngas to PSA
            "PLANT_01:18FI0121/AI1/PV.CV":   30.2,    # PSA H2 flow
        })
    """
    row_df = pd.DataFrame([sensor_readings])
    return compute_physics_features(row_df)
