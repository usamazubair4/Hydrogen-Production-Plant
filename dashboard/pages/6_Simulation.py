import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Simulation | SMR Dashboard",
                   page_icon=":material/model_training:", layout="wide")

from utils.data_loader import load_enriched, val, C_BLUE, C_GREEN, C_AMBER, C_RED
from utils.charts import timeseries
from utils.components import nav_sidebar, kpi_tile, alert_banner, section_title

df_full = load_enriched()
df = nav_sidebar(df_full)

st.markdown("<h1 style='font-size:26px;font-weight:700;color:#111827;margin-bottom:4px;'>Simulation</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#9CA3AF;font-size:13px;margin-bottom:20px;'>What-if scenario analysis using a linear sensitivity model fitted to historical plant data. Adjust parameters to explore their impact on CO and efficiency.</p>", unsafe_allow_html=True)

# ── Sensitivity model ─────────────────────────────────────────────────────────
@st.cache_resource
def build_sensitivity_model(_df: pd.DataFrame):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    features = [
        "S/C Ratio (Steam-to-Carbon)",
        "Tube Outlet Temperature",
        "Plant Rate",
        "hts_outlet_temp_c",
    ]
    target = "CO_Predicted_ppm"
    data = _df[features + [target]].dropna()
    if len(data) < 50:
        return None, None, features, {}, {}

    X = data[features]
    y = data[target]
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = Ridge(alpha=10.0)
    model.fit(X_s, y)

    means = X.mean().to_dict()
    stds  = X.std().to_dict()
    return model, scaler, features, means, stds

model, scaler, features, feat_mean, feat_std = build_sensitivity_model(df_full)

# ── What-If Simulator: sliders (left) | Current + Simulated rows (right) ─────
ctrl_col, results_col = st.columns([1, 1])

co_sim    = 0.0
sc_delta  = 0.0
steam_chg = 0.0

with ctrl_col:
    st.markdown(section_title("What-If Controls", "Adjust parameters to explore CO response"), unsafe_allow_html=True)
    if model is None:
        st.warning("Insufficient data to build sensitivity model.")
    else:
        sc_def   = float(feat_mean.get("S/C Ratio (Steam-to-Carbon)", 3.2))
        tmp_def  = float(feat_mean.get("Tube Outlet Temperature", 1550.0))
        rate_def = float(feat_mean.get("Plant Rate", 85.0))
        hts_def  = float(feat_mean.get("hts_outlet_temp_c", 410.0))

        sc_val   = st.slider("S/C Ratio (Steam-to-Carbon)", min_value=2.7, max_value=4.5,
                              value=sc_def, step=0.05,
                              help="Steam-to-carbon molar ratio. Below 2.7 risks carbon deposition.")
        tmp_val  = st.slider("Tube Outlet Temperature (°F)", min_value=1350, max_value=1700,
                              value=int(tmp_def), step=5,
                              help="Reformer tube outlet temperature. Higher temp improves conversion.")
        rate_val = st.slider("Plant Rate (%)", min_value=50, max_value=110,
                              value=int(rate_def), step=1,
                              help="H₂ production rate as % of 47 MMSCFD design.")
        hts_val  = st.slider("HTS Outlet Temperature (°C)", min_value=350, max_value=470,
                              value=int(hts_def), step=5,
                              help="High-Temperature Shift reactor outlet temperature.")

        X_sim   = np.array([[sc_val, float(tmp_val), float(rate_val), float(hts_val)]])
        co_sim  = max(0.0, float(model.predict(scaler.transform(X_sim))[0]))

        current_sc = val(df, "S/C Ratio (Steam-to-Carbon)")
        sc_delta   = sc_val - current_sc
        steam_chg  = sc_delta / current_sc * 100.0

with results_col:
    _cur_co    = val(df, "CO_Predicted_ppm")
    _cur_color = C_GREEN if _cur_co < 5 else (C_AMBER if _cur_co < 10 else C_RED)

    # ── Row 1: Current Operating Conditions ──────────────────────────────────
    st.markdown(
        section_title("Current Operating Conditions",
                       "Latest values from selected period"),
        unsafe_allow_html=True,
    )
    cc1, cc2 = st.columns(2)
    cc1.markdown(kpi_tile("S/C Ratio",        f"{val(df,'S/C Ratio (Steam-to-Carbon)'):.3f}", "Min safe: 2.7",  C_BLUE),     unsafe_allow_html=True)
    cc2.markdown(kpi_tile("Tube Outlet Temp", f"{val(df,'Tube Outlet Temperature'):.1f} °F",  "Reformer exit",  C_AMBER),    unsafe_allow_html=True)
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    cc3, cc4 = st.columns(2)
    cc3.markdown(kpi_tile("Plant Rate",       f"{val(df,'Plant Rate'):.1f}%",                 "vs 47 MMSCFD",   C_BLUE),     unsafe_allow_html=True)
    cc4.markdown(kpi_tile("CO Predicted",     f"{_cur_co:.2f} ppm",                           "Spec: 10 ppm",   _cur_color), unsafe_allow_html=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # ── Row 2: Simulated Results ──────────────────────────────────────────────
    st.markdown(
        section_title("Simulated Results",
                       "Based on slider settings — move sliders to update"),
        unsafe_allow_html=True,
    )
    if model is not None:
        sim_color  = C_GREEN if co_sim < 5 else (C_AMBER if co_sim < 10 else C_RED)
        sim_status = "Within spec" if co_sim < 10 else "EXCEEDS SPEC"

        sc1, sc2 = st.columns(2)
        sc1.markdown(kpi_tile("Simulated CO",   f"{co_sim:.2f} ppm",    sim_status,              sim_color),                             unsafe_allow_html=True)
        sc2.markdown(kpi_tile("CO Spec Margin", f"{10-co_sim:.2f} ppm", "ppm below 10 ppm spec", C_GREEN if co_sim < 10 else C_RED),     unsafe_allow_html=True)
        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        sc3, sc4 = st.columns(2)
        sc3.markdown(kpi_tile("S/C Change",     f"{sc_delta:+.3f}",     "from current S/C",       C_GREEN if sc_delta < 0 else C_AMBER),  unsafe_allow_html=True)
        sc4.markdown(kpi_tile("Steam Change",   f"{steam_chg:+.1f}%",   "vs current steam use",   C_GREEN if steam_chg < 0 else C_AMBER), unsafe_allow_html=True)

        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        if co_sim >= 10:
            st.markdown(alert_banner("These parameters exceed the 10 ppm CO spec — not recommended for operation.", C_RED), unsafe_allow_html=True)
        elif co_sim >= 5:
            st.markdown(alert_banner("CO in amber zone (5–10 ppm). Proceed with caution.", C_AMBER), unsafe_allow_html=True)
        else:
            st.markdown(alert_banner("CO within safe operating range (< 5 ppm). Green status.", C_GREEN), unsafe_allow_html=True)
    else:
        st.info("Insufficient data to run simulation.")

st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

# ── Operating envelope ────────────────────────────────────────────────────────
st.markdown(section_title("Historical Operating Envelope", "CO vs S/C ratio — coloured by alert level"), unsafe_allow_html=True)

env_l, env_r = st.columns(2)

with env_l:
    env_df = df_full[["S/C Ratio (Steam-to-Carbon)", "CO_Predicted_ppm", "CO_Alert_Level"]].dropna()
    if not env_df.empty:
        color_map = {
            "green": (C_GREEN, "In Specification"),
            "amber": (C_AMBER, "Elevated — Caution"),
            "red":   (C_RED,   "Off-Specification"),
        }
        fig_env = go.Figure()
        for level, (color, legend_name) in color_map.items():
            subset = env_df[env_df["CO_Alert_Level"] == level]
            if not subset.empty:
                fig_env.add_trace(go.Scatter(
                    x=subset["S/C Ratio (Steam-to-Carbon)"],
                    y=subset["CO_Predicted_ppm"],
                    mode="markers",
                    marker=dict(color=color, size=3, opacity=0.6),
                    name=legend_name,
                ))
        # Add spec line
        sc_range = [env_df["S/C Ratio (Steam-to-Carbon)"].min(), env_df["S/C Ratio (Steam-to-Carbon)"].max()]
        fig_env.add_hline(y=10, line_dash="dash", line_color=C_RED, annotation_text="Spec 10 ppm")
        fig_env.add_hline(y=5,  line_dash="dot",  line_color=C_AMBER, annotation_text="Amber 5 ppm")
        fig_env.add_vline(x=2.7, line_dash="dash", line_color=C_RED, annotation_text="Min S/C 2.7")
        fig_env.update_layout(
            template="plotly_white", height=320,
            xaxis_title="S/C Ratio", yaxis_title="CO Predicted (ppm)",
            paper_bgcolor="#FFFFFF", plot_bgcolor="#F9FAFB",
            legend=dict(orientation="h", y=1.08),
        )
        fig_env.update_xaxes(gridcolor="#E5E7EB")
        fig_env.update_yaxes(gridcolor="#E5E7EB")
        st.plotly_chart(fig_env, use_container_width=True, key="env_scatter")
    else:
        st.info("Insufficient data for operating envelope.")

with env_r:
    # Feature sensitivity (correlation with CO)
    st.markdown("<p style='font-size:10px;font-weight:700;letter-spacing:0.1em;color:#9CA3AF;text-transform:uppercase;margin-bottom:8px;'>Feature Correlation with CO (predicted)</p>", unsafe_allow_html=True)
    if model is not None and len(feat_std) > 0:
        coeffs = model.coef_
        feat_labels = ["S/C Ratio", "Tube Outlet Temp", "Plant Rate", "HTS Outlet Temp"]
        colors_coeff = [C_RED if c > 0 else C_GREEN for c in coeffs]
        fig_coeff = go.Figure(go.Bar(
            x=feat_labels, y=coeffs, marker_color=colors_coeff,
            text=[f"{c:+.3f}" for c in coeffs], textposition="outside",
            marker_line_color="#FFFFFF", marker_line_width=1,
        ))
        fig_coeff.update_layout(
            template="plotly_white", height=320,
            yaxis_title="Standardised coefficient (impact on CO ppm)",
            paper_bgcolor="#FFFFFF", plot_bgcolor="#F9FAFB",
            margin=dict(l=40, r=20, t=30, b=50),
            annotations=[dict(
                text="Red = increases CO   |   Green = reduces CO",
                x=0.5, xref="paper", y=-0.22, yref="paper",
                showarrow=False, font=dict(size=11, color="#9CA3AF"),
            )],
        )
        fig_coeff.update_xaxes(gridcolor="#E5E7EB")
        fig_coeff.update_yaxes(gridcolor="#E5E7EB")
        st.plotly_chart(fig_coeff, use_container_width=True, key="coeff_bar")
    else:
        st.info("Sensitivity coefficients unavailable.")

st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

# ── CO vs Tube Temp scatter ───────────────────────────────────────────────────
st.markdown(section_title("CO vs Tube Outlet Temperature", "Operating history coloured by S/C ratio"), unsafe_allow_html=True)
temp_df = df_full[["Tube Outlet Temperature", "CO_Predicted_ppm", "S/C Ratio (Steam-to-Carbon)"]].dropna()
if not temp_df.empty:
    fig_temp = go.Figure(go.Scatter(
        x=temp_df["Tube Outlet Temperature"],
        y=temp_df["CO_Predicted_ppm"],
        mode="markers",
        marker=dict(
            color=temp_df["S/C Ratio (Steam-to-Carbon)"],
            colorscale="RdYlGn",
            size=3,
            opacity=0.6,
            colorbar=dict(title="S/C Ratio", thickness=14),
        ),
    ))
    fig_temp.add_hline(y=10, line_dash="dash", line_color=C_RED, annotation_text="Spec 10 ppm")
    fig_temp.update_layout(
        template="plotly_white", height=300,
        xaxis_title="Tube Outlet Temperature (°F)",
        yaxis_title="CO Predicted (ppm)",
        paper_bgcolor="#FFFFFF", plot_bgcolor="#F9FAFB",
    )
    fig_temp.update_xaxes(gridcolor="#E5E7EB")
    fig_temp.update_yaxes(gridcolor="#E5E7EB")
    st.plotly_chart(fig_temp, use_container_width=True, key="temp_scatter")

with st.expander("Simulation Methodology"):
    st.markdown("""
**Model:** Ridge regression (α = 10) fitted to full historical enriched dataset.
**Target:** CO Predicted ppm (from the full ML pipeline ensemble model output stored in CSV).
**Features:** S/C Ratio, Tube Outlet Temperature, Plant Rate, HTS Outlet Temperature.
**Standardisation:** All features standardised to zero mean, unit variance before fitting.
**Coefficients:** Represent direction and magnitude of each feature's influence on CO.
A positive coefficient means increasing that variable increases predicted CO.
**Limitation:** Linear approximation — does not capture non-linear interactions present in the full XGBoost ensemble. Use for directional guidance only.
    """)
