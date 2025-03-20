import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="SMR Plant Dashboard",
    page_icon=":material/dashboard:",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.data_loader import (
    load_enriched, val, latest_str, ALERT_COLORS,
    C_BLUE, C_GREEN, C_AMBER, C_RED,
    CO_STATUS_LABELS, CO_STATUS_SUBTITLE, COMP_STATUS_LABELS, DESIGN_MMSCFD,
)
from utils.charts import timeseries, health_gauge
from utils.components import nav_sidebar, kpi_tile, health_tile

df_full = load_enriched()
df = nav_sidebar(df_full)

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='margin-bottom:4px;font-size:26px;font-weight:700;color:#111827;'>Plant Overview</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#9CA3AF;font-size:13px;margin-bottom:20px;'>"
    "Real-time snapshot of key plant KPIs, CO prediction status, and compressor fleet health.</p>",
    unsafe_allow_html=True,
)

# ── Derived values ────────────────────────────────────────────────────────────
alert       = latest_str(df, "CO_Alert_Level", "unknown").lower()
co_latest   = val(df, "CO_Predicted_ppm")
co_meas     = val(df, "CO in Product")
headroom    = val(df, "CO Spec Headroom (Predicted)")
a_color     = ALERT_COLORS.get(alert, "#6B7280")
plant_rate  = val(df, "Plant Rate")
psa_rec_pct = val(df, "PSA Recovery (%)")
gross_eff   = val(df, "Gross Efficiency")
pvi         = val(df, "Production Value Index (%)")
co_alert_display = CO_STATUS_LABELS.get(alert, "Status Unknown")

# ── ① KPI tile row ───────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.markdown(
    kpi_tile("Plant Rate",   f"{plant_rate:.1f}%",
             f"Design: {DESIGN_MMSCFD:.0f} MMSCFD", C_BLUE),
    unsafe_allow_html=True,
)
k2.markdown(
    kpi_tile("CO Predicted", f"{co_latest:.2f} ppm",
             f"Measured: {co_meas:.2f} ppm", a_color),
    unsafe_allow_html=True,
)
k3.markdown(
    kpi_tile("Gross SHC",    f"{gross_eff:.1f}",
             "BTU/SCF  ·  Design: 285", C_BLUE),
    unsafe_allow_html=True,
)
k4.markdown(
    kpi_tile("PSA Recovery", f"{psa_rec_pct:.1f}%",
             "H₂ captured from syngas", C_GREEN),
    unsafe_allow_html=True,
)
k5.markdown(
    kpi_tile("Prod. Value Index", f"{pvi:.1f}%",
             "Rate × PSA Recovery", C_BLUE),
    unsafe_allow_html=True,
)

st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

# ── ② Time series charts ──────────────────────────────────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    st.markdown(
        "<p style='font-size:10px;font-weight:700;letter-spacing:0.1em;color:#9CA3AF;"
        "text-transform:uppercase;'>CO — Measured vs Predicted</p>",
        unsafe_allow_html=True,
    )
    fig = timeseries(
        df,
        ["CO in Product", "CO_Predicted_ppm"],
        ["CO Measured (ppm)", "CO Predicted (ppm)"],
        colors=[C_AMBER, C_BLUE], y_label="ppm",
    )
    fig.add_hline(y=10, line_dash="dash", line_color=C_RED,   annotation_text="Spec limit 10 ppm")
    fig.add_hline(y=5,  line_dash="dot",  line_color=C_AMBER, annotation_text="Elevated ≥ 5 ppm")
    st.plotly_chart(fig, use_container_width=True, key="co_ts")

with col_r:
    st.markdown(
        "<p style='font-size:10px;font-weight:700;letter-spacing:0.1em;color:#9CA3AF;"
        "text-transform:uppercase;'>Plant Rate & PSA Recovery</p>",
        unsafe_allow_html=True,
    )
    fig2 = timeseries(
        df,
        ["Plant Rate", "PSA Recovery (%)"],
        ["Plant Rate (%)", "PSA Recovery (%)"],
        colors=[C_BLUE, C_GREEN], y_label="%",
    )
    st.plotly_chart(fig2, use_container_width=True, key="rate_ts")

st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

# ── ③ Compressors Health  +  Current Operating Conditions ────────────────────
fleet_col, table_col = st.columns([3, 2])

with fleet_col:
    st.markdown(
        "<p style='font-size:10px;font-weight:700;letter-spacing:0.1em;color:#9CA3AF;"
        "text-transform:uppercase;margin-bottom:12px;'>Compressors Health</p>",
        unsafe_allow_html=True,
    )
    fc = st.columns(3)
    for i, comp in enumerate(["A", "B", "C"]):
        health = val(df, f"Compressor_{comp}_Health")
        st_c   = latest_str(df, f"Compressor_{comp}_Alert", "offline").lower()
        a_c    = ALERT_COLORS.get(st_c, "#6B7280")
        bear   = val(df, f"Compressor_{comp}_Bear_Score")
        vib    = val(df, f"Compressor_{comp}_Vib_Score")
        oil    = val(df, f"Compressor_{comp}_Oil_Score")
        cr     = val(df, f"Compressor_{comp}_Cr_Score")
        fc[i].markdown(health_tile(comp, health, st_c, a_c, bear, vib, oil, cr), unsafe_allow_html=True)

with table_col:
    st.markdown(
        "<p style='font-size:10px;font-weight:700;letter-spacing:0.1em;color:#9CA3AF;"
        "text-transform:uppercase;margin-bottom:8px;'>Current Operating Conditions</p>",
        unsafe_allow_html=True,
    )
    latest_data = {
        "Parameter": [
            "Plant Rate",
            "CO Predicted",
            "CO Measured",
            "CO Spec Margin",
            "Gross SHC",
            "Net SHC",
            "PSA Recovery",
            "Production Value Index",
            "S/C Ratio",
            "Tube Outlet Temp",
        ],
        "Value": [
            f"{val(df,'Plant Rate'):.1f} %",
            f"{val(df,'CO_Predicted_ppm'):.2f} ppm",
            f"{val(df,'CO in Product'):.2f} ppm",
            f"{val(df,'CO Spec Headroom (Predicted)'):.2f} ppm",
            f"{val(df,'Gross Efficiency'):.1f} BTU/SCF",
            f"{val(df,'Net Efficiency'):.1f} BTU/SCF",
            f"{val(df,'PSA Recovery (%)'):.1f} %",
            f"{val(df,'Production Value Index (%)'):.1f} %",
            f"{val(df,'S/C Ratio (Steam-to-Carbon)'):.3f}",
            f"{val(df,'Tube Outlet Temperature'):.1f} °F",
        ],
    }
    st.dataframe(
        pd.DataFrame(latest_data),
        use_container_width=True,
        hide_index=True,
        height=360,
    )
