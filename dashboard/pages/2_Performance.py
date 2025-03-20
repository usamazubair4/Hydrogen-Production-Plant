import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Performance | SMR Dashboard",
                   page_icon=":material/analytics:", layout="wide")

from utils.data_loader import load_enriched, val, C_BLUE, C_GREEN, C_AMBER, C_RED
from utils.charts import timeseries, TEMPLATE
from utils.components import nav_sidebar, kpi_tile, section_title

df_full = load_enriched()
df = nav_sidebar(df_full)

st.markdown("<h1 style='font-size:26px;font-weight:700;color:#111827;margin-bottom:4px;'>Performance</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#9CA3AF;font-size:13px;margin-bottom:20px;'>Efficiency, process parameters, material balance, and analyzer readings.</p>", unsafe_allow_html=True)

tab_eff, tab_proc, tab_matbal, tab_vents = st.tabs(
    ["Efficiency", "Process Parameters", "Material Balance", "Analyzers & Vents"]
)

_DIV = "<div style='height:{h}px;'></div>"
_HR  = "<hr style='border:none;border-top:1px solid #E5E7EB;margin:8px 0 20px;'>"


# ── TAB 1: Efficiency ─────────────────────────────────────────────────────────
with tab_eff:
    st.markdown(_DIV.format(h=12), unsafe_allow_html=True)

    # SHC tiles + trend side by side
    shc_tiles, shc_chart = st.columns([2, 3])
    with shc_tiles:
        e1, e2 = st.columns(2)
        e1.markdown(kpi_tile("Gross Sp. Heat Cons.", f"{val(df,'Gross Efficiency'):.1f}",    "BTU/SCF  ·  Primary reformer",    C_BLUE),  unsafe_allow_html=True)
        e2.markdown(kpi_tile("Net Sp. Heat Cons.",   f"{val(df,'Net Efficiency'):.1f}",      "BTU/SCF  ·  Incl. fuel credit",   C_BLUE),  unsafe_allow_html=True)
        st.markdown(_DIV.format(h=8), unsafe_allow_html=True)
        e3, e4 = st.columns(2)
        e3.markdown(kpi_tile("Burner Sp. Heat Cons.",f"{val(df,'Burner Efficiency'):.1f}",   "BTU/SCF  ·  Combustion",          C_BLUE),  unsafe_allow_html=True)
        e4.markdown(kpi_tile("Sp. Heat Cons. Gap",   f"{val(df,'Efficiency Gap to Design (BTU/SCF)'):.1f}", "BTU/SCF  ·  vs 285 design", C_AMBER), unsafe_allow_html=True)
    with shc_chart:
        st.plotly_chart(
            timeseries(df,
                ["Gross Efficiency", "Net Efficiency", "Burner Efficiency"],
                ["Gross Sp. Heat Cons.", "Net Sp. Heat Cons.", "Burner Sp. Heat Cons."],
                colors=[C_BLUE, C_GREEN, C_AMBER],
                y_label="BTU/SCF — Specific Heat Consumption", height=260),
            use_container_width=True, key="eff_ts",
        )

    st.markdown(_HR, unsafe_allow_html=True)

    # Yield & Carbon KPIs
    st.markdown(section_title("Yield & Carbon KPIs"), unsafe_allow_html=True)
    y1, y2, y3, y4 = st.columns(4)
    y1.markdown(kpi_tile("H₂/NG Yield Ratio",    f"{val(df,'H2/NG Yield Ratio (SCF/SCF)'):.3f}",   "SCF H₂ / SCF NG",     C_GREEN), unsafe_allow_html=True)
    y2.markdown(kpi_tile("Carbon Efficiency",     f"{val(df,'Carbon Efficiency (%)'):.1f}%",         "% C→H₂",              C_GREEN), unsafe_allow_html=True)
    y3.markdown(kpi_tile("Steam Efficiency Index",f"{val(df,'Steam Efficiency Index'):.0f}",         "Recovery / S/C × 100", C_BLUE),  unsafe_allow_html=True)
    y4.markdown(kpi_tile("H₂ Lost to Purge",      f"{val(df,'H2 Lost to Purge (MSCFH)'):.0f} MSCFH","Recoverable tail gas",  C_AMBER), unsafe_allow_html=True)

    st.markdown(_DIV.format(h=8), unsafe_allow_html=True)
    yl, yr = st.columns(2)
    with yl:
        st.plotly_chart(
            timeseries(df,
                ["H2/NG Yield Ratio (SCF/SCF)", "Carbon Efficiency (%)"],
                ["H₂/NG Yield Ratio", "Carbon Efficiency (%)"],
                colors=[C_BLUE, C_AMBER], height=260),
            use_container_width=True, key="yield_ts",
        )
    with yr:
        st.plotly_chart(
            timeseries(df,
                ["Steam Efficiency Index", "Carbon Efficiency (%)"],
                ["Steam Efficiency Index", "Carbon Efficiency (%)"],
                colors=[C_BLUE, C_GREEN], height=260),
            use_container_width=True, key="steam_eff_ts",
        )


# ── TAB 2: Process Parameters ─────────────────────────────────────────────────
with tab_proc:
    st.markdown(_DIV.format(h=12), unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns(4)
    p1.markdown(kpi_tile("Tube Outlet Temp",    f"{val(df,'Tube Outlet Temperature'):.1f} °F",                                        "Reformer exit temp",         C_RED),   unsafe_allow_html=True)
    p2.markdown(kpi_tile("Reformer DP",         f"{val(df,'Reformer Differential Pressure (Reformer DP)'):.2f} psid",                 "Catalyst bed ΔP",            C_AMBER), unsafe_allow_html=True)
    p3.markdown(kpi_tile("Excess O₂ Flue Gas",  f"{val(df,'Excess O2 in Flue Gas'):.2f}%",                                           "Combustion efficiency",       C_BLUE),  unsafe_allow_html=True)
    p4.markdown(kpi_tile("HTS Shift dT",        f"{val(df,'Shift dT (HTS Temperature Difference)'):.1f} °F",                         "WGS exothermic heat rise",    C_AMBER), unsafe_allow_html=True)

    st.markdown(_DIV.format(h=8), unsafe_allow_html=True)
    p5, p6, p7, p8 = st.columns(4)
    p5.markdown(kpi_tile("S/C Ratio",           f"{val(df,'S/C Ratio (Steam-to-Carbon)'):.3f}",                                       "Min safe: 2.7",               C_BLUE),  unsafe_allow_html=True)
    p6.markdown(kpi_tile("S/C Excess over Min", f"{val(df,'S/C Excess over Coking Min'):.3f}",                                        "Coking safety margin",        C_GREEN), unsafe_allow_html=True)
    p7.markdown(kpi_tile("PGB Pressure",        f"{val(df,'Purge Gas Buffer Vessel Pressure'):.3f} psig",                             "Purge gas header",            C_BLUE),  unsafe_allow_html=True)
    p8.markdown(kpi_tile("Reformer Severity",   f"{val(df,'Reformer Severity Index'):.0f}",                                           "Tube Temp × Rate / 100",      C_AMBER), unsafe_allow_html=True)

    st.markdown(_DIV.format(h=16), unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(
            timeseries(df,
                ["Tube Outlet Temperature", "Shift dT (HTS Temperature Difference)"],
                ["Tube Outlet Temp (°F)", "HTS Shift dT (°F)"],
                colors=[C_RED, C_AMBER], y_label="°F", height=280),
            use_container_width=True, key="temp_ts",
        )
    with col_r:
        st.plotly_chart(
            timeseries(df,
                ["S/C Ratio (Steam-to-Carbon)", "Excess O2 in Flue Gas"],
                ["S/C Ratio", "Excess O₂ (%)"],
                colors=[C_BLUE, C_GREEN], height=280),
            use_container_width=True, key="sc_ts",
        )
    st.plotly_chart(
        timeseries(df,
            ["Reformer Differential Pressure (Reformer DP)", "Purge Gas Buffer Vessel Pressure"],
            ["Reformer DP (psid)", "PGB Pressure (psig)"],
            colors=[C_AMBER, C_BLUE], y_label="psid / psig", height=260),
        use_container_width=True, key="pres_ts",
    )


# ── TAB 3: Material Balance ───────────────────────────────────────────────────
with tab_matbal:
    st.markdown(_DIV.format(h=8), unsafe_allow_html=True)
    st.caption("Values represent absolute % deviation between flow meters. Lower = better meter agreement.")

    # Smooth Steam Balance before display (RFG Agreement excluded — instrument data unreliable)
    df_mb = df.copy()
    _SMOOTH_COL = "Steam Balance (Material Balance)"
    if _SMOOTH_COL in df_mb.columns:
        _s = pd.to_numeric(df_mb[_SMOOTH_COL], errors="coerce")
        _q1, _q3 = _s.quantile(0.05), _s.quantile(0.95)
        _iqr = _q3 - _q1
        _s = _s.clip(lower=max(0.0, _q1 - 2.0 * _iqr), upper=_q3 + 3.0 * _iqr)
        df_mb[_SMOOTH_COL] = _s.rolling(7, min_periods=1, center=True).median()

    mat_cols = [
        ("NG Check",      "NG Check (Material Balance)"),
        ("Steam Balance", "Steam Balance (Material Balance)"),
        ("HC/H₂ Balance", "Hydrcarbon/Recycle H2 (HC/H2) Balance (Material Balance)"),
        ("Mix Tee",       "Mix Tee Balance (Material Balance)"),
        ("Burner Balance","Burner Balance (Material Balance)"),
        ("PSA Balance",   "PSA Balance (Material Balance)"),
        ("H₂ Balance",    "Hydrogen Balance (Material Balance)"),
    ]

    latest_vals  = [val(df_mb, c) for _, c in mat_cols]
    colors_tiles = [C_GREEN if v < 2 else (C_AMBER if v < 5 else C_RED) for v in latest_vals]

    # Tiles: rows of 4 (7 metrics → row of 4 + row of 3)
    for i, (label, _) in enumerate(mat_cols):
        if i % 4 == 0:
            if i > 0:
                st.markdown(_DIV.format(h=8), unsafe_allow_html=True)
            row_cols = st.columns(4)
        row_cols[i % 4].markdown(
            kpi_tile(label, f"{latest_vals[i]:.2f}%", "", colors_tiles[i]),
            unsafe_allow_html=True,
        )

    st.markdown(_DIV.format(h=20), unsafe_allow_html=True)

    # ── Heatmap: deviation across time for all balances ──────────────────────
    st.markdown(
        "<p style='font-size:10px;font-weight:700;letter-spacing:0.1em;color:#9CA3AF;"
        "text-transform:uppercase;'>Balance Deviation Heatmap — Over Time</p>",
        unsafe_allow_html=True,
    )

    _hm_cols   = [c for _, c in mat_cols]
    _hm_labels = [l for l, _ in mat_cols]

    df_hm = (
        df_mb.set_index("Timestamp")[_hm_cols]
        .apply(pd.to_numeric, errors="coerce")
        .resample("D").mean()
        .clip(lower=0)
    )
    # Cap display at 15% so color scale is readable
    df_hm = df_hm.clip(upper=15)

    _z    = df_hm[_hm_cols].T.values.tolist()
    _x    = df_hm.index.strftime("%b %d").tolist()

    _colorscale = [
        [0.00, "#15A34A"],   # 0%  — green
        [0.13, "#D97706"],   # 2%  — amber  (2/15)
        [0.33, "#DC2626"],   # 5%  — red    (5/15)
        [1.00, "#7F1D1D"],   # 15% — dark red
    ]

    fig_hm = go.Figure(go.Heatmap(
        x=_x,
        y=_hm_labels,
        z=_z,
        colorscale=_colorscale,
        zmin=0, zmax=15,
        colorbar=dict(
            title="% deviation",
            tickvals=[0, 2, 5, 10, 15],
            ticktext=["0%", "2% (OK)", "5% (Caution)", "10%", "15%"],
            thickness=14, len=0.9,
        ),
        hovertemplate="%{y}<br>%{x}: %{z:.2f}%<extra></extra>",
        xgap=1, ygap=1,
    ))
    fig_hm.update_layout(
        template=TEMPLATE,
        height=340,
        margin=dict(l=120, r=120, t=20, b=60),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#F9FAFB",
        xaxis=dict(showgrid=False, tickangle=-35),
        yaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig_hm, use_container_width=True, key="matbal_heatmap")

    with st.expander("Balance notes"):
        st.markdown(
            "- **RFG Agreement** excluded — systematic instrument errors (values > 100%).\n"
            "- **HC Balance** excluded — no recorded values in current dataset.\n"
            "- **Coker Agreement** excluded — instrument data unreliable (values in millions %).\n"
            "- **NG Balance** excluded — no recorded values in current dataset.\n"
            "- Steam Balance spikes removed via IQR-capping + 7-period rolling median.\n"
            "- Heatmap capped at 15% for readability. Green < 2% · Amber 2–5% · Red > 5%."
        )


# ── TAB 4: Analyzers & Vents ─────────────────────────────────────────────────
with tab_vents:
    st.markdown(_DIV.format(h=12), unsafe_allow_html=True)

    # ── Gas Analyzers ──────────────────────────────────────────────────────────
    st.markdown(section_title("Gas Analyzers", "CO Slip GC is offline in current dataset — values read zero"), unsafe_allow_html=True)

    az_t1, az_t2, az_t3, az_chart = st.columns([1, 1, 1, 2])
    az_t1.markdown(kpi_tile("CO Slip GC",      f"{val(df,'CO Slip (Syngas GC)'):.3f}%",     "Syngas GC — offline", "#9CA3AF"), unsafe_allow_html=True)
    az_t2.markdown(kpi_tile("Methane Slip GC", f"{val(df,'Methane Slip (Syngas GC)'):.3f}%","Unreacted CH₄",       C_AMBER),  unsafe_allow_html=True)
    az_t3.markdown(kpi_tile("CH₄ Syngas GC",   f"{val(df,'CH4  Syngas GC'):.3f}%",          "Feed methane fraction",C_BLUE),   unsafe_allow_html=True)
    with az_chart:
        st.plotly_chart(
            timeseries(df,
                ["Methane Slip (Syngas GC)", "CH4  Syngas GC"],
                ["Methane Slip (%)", "CH₄ Syngas GC (%)"],
                colors=[C_AMBER, C_BLUE], y_label="%", height=160),
            use_container_width=True, key="analyzer_ts",
        )

    st.markdown(_HR, unsafe_allow_html=True)

    # ── Vent Controllers ───────────────────────────────────────────────────────
    st.markdown(section_title("Vent Controllers", "% valve opening — 0% closed · 100% fully open"), unsafe_allow_html=True)

    v1, v2, v3, v4, v5 = st.columns(5)
    v1.markdown(kpi_tile("Purge Gas Vent", f"{val(df,'Purge Gas Vent'):.1f}%",          "% open", C_BLUE),  unsafe_allow_html=True)
    v2.markdown(kpi_tile("Midplant Vent",  f"{val(df,'Midplant Vent'):.1f}%",           "% open", C_BLUE),  unsafe_allow_html=True)
    v3.markdown(kpi_tile("PSA Vent",       f"{val(df,'PSA Vent (SMR PSA Vent)'):.1f}%", "% open", C_AMBER), unsafe_allow_html=True)
    v4.markdown(kpi_tile("Product Vent",   f"{val(df,'Product Vent'):.1f}%",            "% open", C_BLUE),  unsafe_allow_html=True)
    v5.markdown(kpi_tile("Steam Vent",     f"{val(df,'Steam Vent'):.1f}%",              "% open", C_BLUE),  unsafe_allow_html=True)

    st.markdown(_DIV.format(h=8), unsafe_allow_html=True)
    st.plotly_chart(
        timeseries(df,
            ["Purge Gas Vent", "Midplant Vent", "PSA Vent (SMR PSA Vent)", "Product Vent", "Steam Vent"],
            ["Purge Gas", "Midplant", "PSA", "Product", "Steam"],
            y_label="% open", height=280),
        use_container_width=True, key="vent_ts",
    )
