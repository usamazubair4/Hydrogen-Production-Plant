import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

st.set_page_config(page_title="Optimisation | SMR Dashboard",
                   page_icon=":material/tune:", layout="wide")

from utils.data_loader import load_enriched, val, C_BLUE, C_GREEN, C_AMBER, C_RED
from utils.charts import timeseries
from utils.components import nav_sidebar, kpi_tile, alert_banner, section_title

df_full = load_enriched()
df = nav_sidebar(df_full)

st.markdown("<h1 style='font-size:26px;font-weight:700;color:#111827;margin-bottom:4px;'>Optimisation</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#9CA3AF;font-size:13px;margin-bottom:20px;'>Post-ML KPIs, steam reduction opportunity analysis, and efficiency benchmarking.</p>", unsafe_allow_html=True)

# ── Steam reduction opportunity ───────────────────────────────────────────────
steam_opp = val(df, "Steam Reduction Opportunity", 0)
eff_gap   = val(df, "Efficiency Gap to Design (BTU/SCF)")
steam_idx = val(df, "Steam Cost Index")
hts_util  = val(df, "HTS Catalyst Utilization (%)")
co_head   = val(df, "CO Spec Headroom (Predicted)")

opp_color = C_GREEN if steam_opp else "#6B7280"
opp_msg   = ("Steam reduction opportunity exists — CO spec margin allows safe S/C reduction"
             if steam_opp
             else "No steam reduction opportunity — S/C ratio near minimum safe level (2.7)")
st.markdown(alert_banner(opp_msg, opp_color), unsafe_allow_html=True)

# ── Optimisation KPI tiles ────────────────────────────────────────────────────
o1, o2, o3, o4, o5 = st.columns(5)
o1.markdown(kpi_tile("CO Spec Margin",    f"{co_head:.2f} ppm",   "ppm below 10 ppm spec limit", C_GREEN), unsafe_allow_html=True)
o2.markdown(kpi_tile("Efficiency Gap",    f"{eff_gap:.1f}",        "BTU/SCF  ·  vs 285 design",  C_AMBER), unsafe_allow_html=True)
o3.markdown(kpi_tile("Steam Cost Index",  f"{steam_idx:.1f}",     "S/C / 2.7 × 100",             C_BLUE),  unsafe_allow_html=True)
o4.markdown(kpi_tile("HTS Catalyst Util.",f"{hts_util:.1f}%",     "Zero — CO GC offline",        C_AMBER), unsafe_allow_html=True)
o5.markdown(kpi_tile("Steam Reduction",   "Yes" if steam_opp else "No", "CO < 8 ppm & S/C > 3.0", opp_color), unsafe_allow_html=True)

st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

# ── Yield & efficiency KPIs ───────────────────────────────────────────────────
st.markdown(section_title("Yield & Efficiency"), unsafe_allow_html=True)
y1, y2, y3, y4 = st.columns(4)
y1.markdown(kpi_tile("H₂/NG Yield Ratio",     f"{val(df,'H2/NG Yield Ratio (SCF/SCF)'):.3f}",  "SCF/SCF",              C_GREEN), unsafe_allow_html=True)
y2.markdown(kpi_tile("Carbon Efficiency",      f"{val(df,'Carbon Efficiency (%)'):.1f}%",        "% C→H₂",               C_GREEN), unsafe_allow_html=True)
y3.markdown(kpi_tile("H₂ Lost to Purge",       f"{val(df,'H2 Lost to Purge (MSCFH)'):.0f} MSCFH","Recoverable",          C_AMBER), unsafe_allow_html=True)
y4.markdown(kpi_tile("Production Value Index", f"{val(df,'Production Value Index (%)'):.1f}%",   "Rate × PSA Recovery",  C_BLUE),  unsafe_allow_html=True)

st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
y5, y6, y7, y8 = st.columns(4)
y5.markdown(kpi_tile("Steam Efficiency Index", f"{val(df,'Steam Efficiency Index'):.0f}",        "Recovery / S/C × 100", C_BLUE),  unsafe_allow_html=True)
y6.markdown(kpi_tile("Reformer Severity",      f"{val(df,'Reformer Severity Index'):.0f}",       "Temp × Rate / 100",    C_AMBER), unsafe_allow_html=True)
y7.markdown(kpi_tile("S/C Excess over Min",    f"{val(df,'S/C Excess over Coking Min'):.3f}",    "Safety margin over 2.7",C_GREEN), unsafe_allow_html=True)
y8.markdown(kpi_tile("PSA Recovery",           f"{val(df,'PSA Recovery (%)'):.1f}%",             "H₂ captured",          C_GREEN), unsafe_allow_html=True)

st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

# ── Time series charts ─────────────────────────────────────────────────────────
col_l, col_r = st.columns(2)
with col_l:
    st.markdown("<p style='font-size:10px;font-weight:700;letter-spacing:0.1em;color:#9CA3AF;text-transform:uppercase;'>Efficiency vs Design</p>", unsafe_allow_html=True)
    fig_eff = timeseries(
        df,
        ["Gross Efficiency", "Net Efficiency", "Efficiency Gap to Design (BTU/SCF)"],
        ["Gross SHC", "Net SHC", "SHC Gap to Design"],
        colors=[C_BLUE, C_GREEN, C_AMBER], y_label="BTU/SCF", height=290,
    )
    fig_eff.add_hline(y=285, line_dash="dash", line_color="#9CA3AF", annotation_text="285 BTU/SCF design")
    st.plotly_chart(fig_eff, use_container_width=True, key="eff_gap_ts")

with col_r:
    st.markdown("<p style='font-size:10px;font-weight:700;letter-spacing:0.1em;color:#9CA3AF;text-transform:uppercase;'>Steam vs CO Spec Margin</p>", unsafe_allow_html=True)
    fig_sc = timeseries(
        df,
        ["S/C Ratio (Steam-to-Carbon)", "S/C Excess over Coking Min"],
        ["S/C Ratio", "S/C Excess over Min (2.7)"],
        colors=[C_BLUE, C_AMBER], y_label="mol/mol", height=290,
    )
    fig_sc.add_hline(y=2.7, line_dash="dash", line_color=C_RED, annotation_text="Coking minimum 2.7")
    st.plotly_chart(fig_sc, use_container_width=True, key="sc_opt_ts")

col_l2, col_r2 = st.columns(2)
with col_l2:
    st.markdown("<p style='font-size:10px;font-weight:700;letter-spacing:0.1em;color:#9CA3AF;text-transform:uppercase;'>Yield KPIs</p>", unsafe_allow_html=True)
    st.plotly_chart(
        timeseries(df,
            ["H2/NG Yield Ratio (SCF/SCF)", "Carbon Efficiency (%)"],
            ["H₂/NG Yield Ratio", "Carbon Efficiency (%)"],
            colors=[C_BLUE, C_GREEN], height=270),
        use_container_width=True, key="yield_opt_ts",
    )
with col_r2:
    st.markdown("<p style='font-size:10px;font-weight:700;letter-spacing:0.1em;color:#9CA3AF;text-transform:uppercase;'>Production Value Index</p>", unsafe_allow_html=True)
    st.plotly_chart(
        timeseries(df,
            ["Production Value Index (%)", "Plant Rate", "PSA Recovery (%)"],
            ["Production Value Index (%)", "Plant Rate (%)", "PSA Recovery (%)"],
            colors=[C_BLUE, C_AMBER, C_GREEN], height=270),
        use_container_width=True, key="pvi_ts",
    )

st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
st.markdown("<p style='font-size:10px;font-weight:700;letter-spacing:0.1em;color:#9CA3AF;text-transform:uppercase;'>CO Spec Margin — Operating Buffer to 10 ppm Spec Limit</p>", unsafe_allow_html=True)
fig_head = timeseries(
    df,
    ["CO Spec Headroom (Predicted)", "CO Spec Headroom (Measured)"],
    ["Spec Margin (Predicted ppm)", "Spec Margin (Measured ppm)"],
    colors=[C_BLUE, C_AMBER], y_label="ppm below 10 ppm CO spec", height=270,
)
fig_head.add_hline(y=0, line_dash="dash", line_color=C_RED, annotation_text="Spec boundary")
fig_head.add_hrect(y0=-5, y1=0, fillcolor=C_RED,   opacity=0.06, layer="below", line_width=0)
fig_head.add_hrect(y0=0,  y1=5,  fillcolor=C_AMBER, opacity=0.05, layer="below", line_width=0)
fig_head.add_hrect(y0=5,  y1=15, fillcolor=C_GREEN, opacity=0.04, layer="below", line_width=0)
st.plotly_chart(fig_head, use_container_width=True, key="headroom_opt_ts")

with st.expander("KPI Definitions"):
    st.markdown("""
| KPI | Formula | Interpretation |
|---|---|---|
| CO Spec Margin | 10 − CO_Predicted | ppm below 10 ppm spec limit. Positive = within spec. |
| Efficiency Gap | Gross Efficiency − 285 BTU/SCF | Distance from design point. Target: minimise. |
| Steam Cost Index | S/C / 2.7 × 100 | 100 = minimum safe steam. Higher = excess steam cost. |
| Steam Reduction Opportunity | 1 if CO < 8 ppm AND S/C > 3.0 | Actionable flag for operators. |
| H₂/NG Yield Ratio | PSA H₂ / NG feed (MSCFH) | Higher = better reforming conversion. |
| Carbon Efficiency | PSA H₂ / (NG × 3.8) × 100 | % of available carbon converted to H₂. |
| Production Value Index | Plant Rate × PSA Recovery | Composite throughput-quality metric. |
| Steam Efficiency Index | PSA Recovery / S/C × 100 | Recovery achieved per unit of steam. |
    """)
