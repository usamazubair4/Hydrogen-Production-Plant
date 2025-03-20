import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Reliability | SMR Dashboard",
                   page_icon=":material/engineering:", layout="wide")

from utils.data_loader import (
    load_enriched, load_raw_sensors, val, latest_str,
    ALERT_COLORS, COMP_STATUS_LABELS, C_BLUE, C_GREEN, C_AMBER, C_RED,
)
from utils.charts import timeseries
from utils.components import nav_sidebar, kpi_tile, health_tile, section_title

df_full = load_enriched()
df = nav_sidebar(df_full)
df_raw = load_raw_sensors()

st.markdown("<h1 style='font-size:26px;font-weight:700;color:#111827;margin-bottom:4px;'>Reliability</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#9CA3AF;font-size:13px;margin-bottom:20px;'>Compressor fleet health overview, health trend, and raw sensor diagnostics.</p>", unsafe_allow_html=True)

# ── Fleet overview cards (summary for all 3 compressors) ─────────────────────
st.markdown(
    "<p style='font-size:10px;font-weight:700;letter-spacing:0.1em;color:#9CA3AF;"
    "text-transform:uppercase;margin-bottom:12px;'>Fleet Health — Current Status</p>",
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

st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

# ── Per-compressor detail tabs ────────────────────────────────────────────────
st.markdown(
    "<p style='font-size:10px;font-weight:700;letter-spacing:0.1em;color:#9CA3AF;"
    "text-transform:uppercase;margin-bottom:4px;'>Detailed Diagnostics — Select Compressor</p>",
    unsafe_allow_html=True,
)
tab_a, tab_b, tab_c = st.tabs(["Compressor A", "Compressor B", "Compressor C"])

for tab, comp in [(tab_a, "A"), (tab_b, "B"), (tab_c, "C")]:
    with tab:
        health  = val(df, f"Compressor_{comp}_Health")
        alert   = latest_str(df, f"Compressor_{comp}_Alert", "offline").lower()
        a_color = ALERT_COLORS.get(alert, "#6B7280")
        disp    = COMP_STATUS_LABELS.get(alert, alert.upper())
        bear    = val(df, f"Compressor_{comp}_Bear_Score")
        vib     = val(df, f"Compressor_{comp}_Vib_Score")
        oil     = val(df, f"Compressor_{comp}_Oil_Score")
        cr      = val(df, f"Compressor_{comp}_Cr_Score")

        if comp == "A":
            st.info("Compressor A operates as standby (~16% of time). Scores apply to running hours only.")

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

        # Health trend (left)  +  Sub-score tiles 2×2 (right)
        trend_col, score_col = st.columns([3, 2])

        with trend_col:
            st.markdown(
                "<p style='font-size:10px;font-weight:700;letter-spacing:0.1em;"
                "color:#9CA3AF;text-transform:uppercase;'>Health Index — Trend Over Time</p>",
                unsafe_allow_html=True,
            )
            fig_hi = timeseries(
                df, [f"Compressor_{comp}_Health"], ["Health Index"],
                colors=[C_BLUE], y_label="Health Index (0–100)", height=260,
            )
            fig_hi.add_hline(y=70, line_dash="dot",  line_color=C_GREEN, annotation_text="Healthy ≥ 70")
            fig_hi.add_hline(y=55, line_dash="dash", line_color=C_AMBER, annotation_text="Degraded ≥ 55")
            fig_hi.add_hrect(y0=0,   y1=55,  fillcolor=C_RED,   opacity=0.05, layer="below", line_width=0)
            fig_hi.add_hrect(y0=55,  y1=70,  fillcolor=C_AMBER, opacity=0.05, layer="below", line_width=0)
            fig_hi.add_hrect(y0=70,  y1=110, fillcolor=C_GREEN, opacity=0.04, layer="below", line_width=0)
            st.plotly_chart(fig_hi, use_container_width=True, key=f"hi_ts_{comp}")

        with score_col:
            st.markdown(
                "<p style='font-size:10px;font-weight:700;letter-spacing:0.1em;"
                "color:#9CA3AF;text-transform:uppercase;'>Sub-scores (Latest)</p>",
                unsafe_allow_html=True,
            )
            sc1, sc2 = st.columns(2)
            sc1.markdown(kpi_tile("Bearing",    f"{bear:.1f}" if not pd.isna(bear) else "N/A", "Weight 35%", C_GREEN if bear >= 70 else (C_AMBER if bear >= 55 else C_RED)),  unsafe_allow_html=True)
            sc2.markdown(kpi_tile("Vibration",  f"{vib:.1f}"  if not pd.isna(vib)  else "N/A", "Weight 25%", C_GREEN if vib  >= 70 else (C_AMBER if vib  >= 55 else C_RED)),  unsafe_allow_html=True)
            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
            sc3, sc4 = st.columns(2)
            sc3.markdown(kpi_tile("Oil System", f"{oil:.1f}"  if not pd.isna(oil)  else "N/A", "Weight 25%", C_GREEN if oil  >= 70 else (C_AMBER if oil  >= 55 else C_RED)),  unsafe_allow_html=True)
            sc4.markdown(kpi_tile("Comp. Ratio",f"{cr:.1f}"   if not pd.isna(cr)   else "N/A", "Weight 15%", C_GREEN if cr   >= 70 else (C_AMBER if cr   >= 55 else C_RED)),  unsafe_allow_html=True)

        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

        # Raw sensor data
        st.markdown(section_title(f"Raw Sensor Data — Compressor {comp}", "Sourced directly from plant historian"), unsafe_allow_html=True)

        sensor_map = {
            "Motor Current":        f"Compressor {comp} Motor Current",
            "Hottest Bearing Temp": f"Compressor {comp} Hottest Bearing Temperature",
            "Oil Filter dP":        f"Compressor {comp} Oil Filter dP",
            "Oil Pressure":         f"Compressor {comp} Oil Pressure",
            "Oil Temperature":      f"Compressor {comp} Oil Temperature",
            "Motor DE Vibration":   f"Compressor {comp} Motor DE Vibration",
            "Frame DE Vibration":   f"Compressor {comp} Frame DE Vibration",
            "Interstage Vibration": f"Compressor {comp} Interstage Cooler Vibration",
        }

        s_cols = st.columns(4)
        for j, (label, raw_col) in enumerate(sensor_map.items()):
            v = val(df_raw, raw_col)
            s_cols[j % 4].markdown(
                kpi_tile(label, f"{v:.2f}" if not pd.isna(v) else "N/A", raw_col, C_BLUE),
                unsafe_allow_html=True,
            )
            if j == 3:
                st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
                s_cols = st.columns(4)

        avail_raw_cols = [c for c in sensor_map.values() if c in df_raw.columns]
        avail_labels   = [lb for lb, c in sensor_map.items() if c in df_raw.columns]
        if avail_raw_cols:
            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
            st.plotly_chart(
                timeseries(df_raw, avail_raw_cols[:3], avail_labels[:3],
                           title=f"Compressor {comp} — Key Sensor Trends", height=260),
                use_container_width=True, key=f"sensor_ts_{comp}",
            )
        else:
            st.warning("Raw sensor data not available — requires raw Combined_Data.csv.")

with st.expander("Health Index Methodology"):
    st.markdown("""
| Sub-score | Weight | Source signals |
|---|---|---|
| Bearing Score | 35% | Hottest bearing temperature, trend deviation |
| Vibration Score | 25% | Motor DE, Frame DE, Interstage vibration |
| Oil System Score | 25% | Oil filter dP, oil pressure, oil temperature |
| Compression Ratio | 15% | 1st stage H₂ compression ratio |

| Health Index | Status |
|---|---|
| ≥ 70 | Healthy |
| 55 – 69 | Degraded |
| < 55 | Critical — action required |
    """)
