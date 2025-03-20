import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Quality Prediction | SMR Dashboard",
                   page_icon=":material/science:", layout="wide")

from utils.data_loader import (
    load_enriched, val, latest_str, ALERT_COLORS, CO_STATUS_LABELS,
    C_BLUE, C_GREEN, C_AMBER, C_RED,
)
from utils.charts import timeseries
from utils.components import nav_sidebar, kpi_tile, alert_banner, section_title

df_full = load_enriched()
df = nav_sidebar(df_full)

st.markdown("<h1 style='font-size:26px;font-weight:700;color:#111827;margin-bottom:4px;'>Quality Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#9CA3AF;font-size:13px;margin-bottom:20px;'>ML model CO prediction, CO spec margin monitoring, and HTS thermodynamic physics features.</p>", unsafe_allow_html=True)

# ── Derived values ────────────────────────────────────────────────────────────
alert   = latest_str(df, "CO_Alert_Level", "unknown").lower()
color   = ALERT_COLORS.get(alert, "#6B7280")
co_pred = val(df, "CO_Predicted_ppm")
co_meas = val(df, "CO in Product")
hroom_p = val(df, "CO Spec Headroom (Predicted)")
hroom_m = val(df, "CO Spec Headroom (Measured)")
a_label = CO_STATUS_LABELS.get(alert, alert.upper())

# ── Alert banner ──────────────────────────────────────────────────────────────
st.markdown(
    alert_banner(
        f"CO {a_label}  ·  Predicted {co_pred:.2f} ppm  ·  "
        f"Measured {co_meas:.2f} ppm  ·  Spec limit 10 ppm",
        color,
    ),
    unsafe_allow_html=True,
)

# ── KPI tiles — 4 equal columns (no gauge) ────────────────────────────────────
t1, t2, t3, t4 = st.columns(4)
t1.markdown(
    kpi_tile("CO Predicted",        f"{co_pred:.2f} ppm",
             f"Δ {co_pred - 10:.2f} ppm vs spec limit", color),
    unsafe_allow_html=True,
)
t2.markdown(
    kpi_tile("CO Measured",         f"{co_meas:.2f} ppm",
             f"Δ {co_meas - 10:.2f} ppm vs spec limit", C_AMBER),
    unsafe_allow_html=True,
)
t3.markdown(
    kpi_tile("Spec Margin (Pred.)", f"{hroom_p:.2f} ppm",
             "ppm below 10 ppm spec", C_GREEN if hroom_p >= 0 else C_RED),
    unsafe_allow_html=True,
)
t4.markdown(
    kpi_tile("Spec Margin (Meas.)", f"{hroom_m:.2f} ppm",
             "ppm below 10 ppm spec", C_GREEN if hroom_m >= 0 else C_RED),
    unsafe_allow_html=True,
)

# ── Alert distribution ────────────────────────────────────────────────────────
if "CO_Alert_Level" in df.columns:
    vc    = df["CO_Alert_Level"].dropna().value_counts()
    total = max(vc.sum(), 1)
    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:10px;font-weight:700;letter-spacing:0.09em;"
        "color:#9CA3AF;text-transform:uppercase;'>Alert Distribution — Filtered Period</p>",
        unsafe_allow_html=True,
    )
    a1, a2, a3 = st.columns(3)
    a1.markdown(kpi_tile("In Specification",   f"{vc.get('green', 0):,}", f"{100*vc.get('green', 0)/total:.1f}% of period", C_GREEN), unsafe_allow_html=True)
    a2.markdown(kpi_tile("Elevated — Caution", f"{vc.get('amber', 0):,}", f"{100*vc.get('amber', 0)/total:.1f}% of period", C_AMBER), unsafe_allow_html=True)
    a3.markdown(kpi_tile("Off-Specification",  f"{vc.get('red',   0):,}", f"{100*vc.get('red',   0)/total:.1f}% of period", C_RED),   unsafe_allow_html=True)

st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

# ── CO Measured vs Predicted — full-width time series ─────────────────────────
st.markdown(
    "<p style='font-size:10px;font-weight:700;letter-spacing:0.1em;"
    "color:#9CA3AF;text-transform:uppercase;'>CO Measured vs Predicted — Time Series</p>",
    unsafe_allow_html=True,
)
fig = timeseries(
    df,
    ["CO in Product", "CO_Predicted_ppm"],
    ["CO Measured (ppm)", "CO Predicted (ppm)"],
    colors=[C_AMBER, C_BLUE], y_label="ppm", height=300,
)
fig.add_hline(y=10, line_dash="dash", line_color=C_RED,   annotation_text="Spec limit 10 ppm (Off-Spec)")
fig.add_hline(y=5,  line_dash="dot",  line_color=C_AMBER, annotation_text="Elevated ≥ 5 ppm (Caution)")
fig.add_hrect(y0=10, y1=25, fillcolor=C_RED,   opacity=0.05, layer="below", line_width=0)
fig.add_hrect(y0=5,  y1=10, fillcolor=C_AMBER, opacity=0.05, layer="below", line_width=0)
fig.add_hrect(y0=0,  y1=5,  fillcolor=C_GREEN, opacity=0.04, layer="below", line_width=0)
st.plotly_chart(fig, use_container_width=True, key="co_ts_pred")

# ── CO Spec Margin  |  HTS Physics ───────────────────────────────────────────
col_h1, col_h2 = st.columns(2)

with col_h1:
    st.markdown(
        "<p style='font-size:10px;font-weight:700;letter-spacing:0.1em;"
        "color:#9CA3AF;text-transform:uppercase;'>CO Spec Margin — ppm below 10 ppm CO spec limit</p>",
        unsafe_allow_html=True,
    )
    fig_hr = timeseries(
        df,
        ["CO Spec Headroom (Predicted)", "CO Spec Headroom (Measured)"],
        ["Spec Margin (Predicted)", "Spec Margin (Measured)"],
        colors=[C_BLUE, C_AMBER], y_label="ppm below 10 ppm spec", height=280,
    )
    fig_hr.add_hline(y=0, line_dash="dash", line_color=C_RED, annotation_text="Spec boundary")
    fig_hr.add_hrect(y0=-10, y1=0, fillcolor=C_RED, opacity=0.06, layer="below", line_width=0)
    st.plotly_chart(fig_hr, use_container_width=True, key="margin_ts")

with col_h2:
    st.markdown(section_title("Physics Features — HTS Section", "Water-Gas Shift thermodynamics at HTS reactor outlet"), unsafe_allow_html=True)
    ph1, ph2, ph3 = st.columns(3)
    ph1.markdown(kpi_tile("HTS Outlet Temp",  f"{val(df,'hts_outlet_temp_c'):.1f} °C",              "Converted from °F",          C_AMBER), unsafe_allow_html=True)
    ph2.markdown(kpi_tile("WGS K_eq",         f"{val(df,'hts_k_eq'):.3f}",                          "exp(4577.8/T_K − 4.33)",     C_BLUE),  unsafe_allow_html=True)
    ph3.markdown(kpi_tile("Eq. CO%",          f"{val(df,'approx_eq_co_pct'):.2f}%",                 "Thermodynamic equilibrium",   C_BLUE),  unsafe_allow_html=True)
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    ph4, ph5, ph6 = st.columns(3)
    ph4.markdown(kpi_tile("Approach to Eq.",  f"{val(df,'approach_to_eq'):.3f}",                    "CO Slip / Eq. CO%",           C_GREEN), unsafe_allow_html=True)
    ph5.markdown(kpi_tile("HTS Cat. Util.",   f"{val(df,'HTS Catalyst Utilization (%)'):.1f}%",      "Zero — CO GC offline",        C_AMBER), unsafe_allow_html=True)
    ph6.markdown(kpi_tile("PSA Space Vel.",   f"{val(df,'psa_space_vel_proxy'):.3f}",               "Syngas / PSA H₂ flow",        C_BLUE),  unsafe_allow_html=True)

# ── HTS Thermodynamics  |  PSA Space Velocity ─────────────────────────────────
st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
ch1, ch2 = st.columns(2)
with ch1:
    st.markdown(
        "<p style='font-size:10px;font-weight:700;letter-spacing:0.1em;"
        "color:#9CA3AF;text-transform:uppercase;'>HTS Thermodynamics — Outlet Temp vs Eq. CO%</p>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        timeseries(df,
            ["hts_outlet_temp_c", "approx_eq_co_pct"],
            ["HTS Outlet Temp (°C)", "Eq. CO% at T"],
            colors=[C_RED, C_BLUE], height=240),
        use_container_width=True, key="hts_ts",
    )
with ch2:
    st.markdown(
        "<p style='font-size:10px;font-weight:700;letter-spacing:0.1em;"
        "color:#9CA3AF;text-transform:uppercase;'>PSA Space Velocity Proxy</p>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        timeseries(df,
            ["psa_space_vel_proxy"], ["PSA Space Velocity Proxy"],
            colors=[C_GREEN], height=240),
        use_container_width=True, key="psavel_ts",
    )

with st.expander("Model & Alert Details"):
    st.markdown("""
| Parameter | Value |
|---|---|
| Model type | Ridge / Random Forest / XGBoost voting ensemble |
| Target variable | CO in product (ppm) |
| Green threshold | < 5 ppm |
| Amber threshold | 5 – 10 ppm |
| Red threshold (spec limit) | ≥ 10 ppm |
| Physics features | HTS outlet temp, WGS K_eq, equilibrium CO%, PSA space velocity proxy |
| CO GC data gap | Approach-to-equilibrium and HTS Catalyst Utilisation will populate when syngas CO analyser comes online |
    """)
