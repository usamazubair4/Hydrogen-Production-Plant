import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta

st.set_page_config(page_title="Maintenance | SMR Dashboard",
                   page_icon=":material/calendar_month:", layout="wide")

from utils.data_loader import load_enriched, val, latest_str, ALERT_COLORS, C_BLUE, C_GREEN, C_AMBER, C_RED
from utils.charts import timeseries
from utils.components import nav_sidebar, kpi_tile, alert_banner, section_title

df_full = load_enriched()
df = nav_sidebar(df_full)

st.markdown("<h1 style='font-size:26px;font-weight:700;color:#111827;margin-bottom:4px;'>Maintenance Scheduling</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#9CA3AF;font-size:13px;margin-bottom:20px;'>Predictive maintenance analysis using health index trend projection. Identifies optimal inspection windows before threshold crossing.</p>", unsafe_allow_html=True)


# ── Helper: project days to threshold ────────────────────────────────────────
def days_to_threshold(comp: str, threshold: float) -> tuple[float | None, float, float]:
    """Returns (days_until_threshold, current_health, slope_per_day)."""
    hi_col = f"Compressor_{comp}_Health"
    if hi_col not in df_full.columns:
        return None, float("nan"), 0.0

    data = df_full[["Timestamp", hi_col]].dropna()
    data = data.copy()
    data[hi_col] = pd.to_numeric(data[hi_col], errors="coerce")
    data = data.dropna().tail(2000)

    if len(data) < 20:
        return None, float("nan"), 0.0

    x_days = (data["Timestamp"] - data["Timestamp"].iloc[0]).dt.total_seconds() / 86400
    y = data[hi_col].astype(float).values
    slope, intercept = np.polyfit(x_days.values, y, 1)

    current_x    = x_days.iloc[-1]
    current_hi   = float(y[-1])

    if slope >= 0 or current_hi <= threshold:
        return None, current_hi, slope

    days_ahead = (threshold - current_hi) / slope
    return max(0.0, days_ahead), current_hi, slope


# ── Fleet summary ─────────────────────────────────────────────────────────────
compressors = ["A", "B", "C"]
projections: dict[str, dict] = {}

for comp in compressors:
    days_amber, hi,  slope_a = days_to_threshold(comp, 70)
    days_red,   _,   slope_r = days_to_threshold(comp, 55)
    st_c  = latest_str(df_full, f"Compressor_{comp}_Alert", "offline").lower()
    projections[comp] = {
        "health":      hi,
        "status":      st_c,
        "days_amber":  days_amber,
        "days_red":    days_red,
        "slope":       slope_a,
    }

fc = st.columns(3)
for i, comp in enumerate(compressors):
    p = projections[comp]
    hi     = p["health"]
    st_c   = p["status"]
    a_c    = ALERT_COLORS.get(st_c, "#6B7280")
    d_amb  = p["days_amber"]
    d_red  = p["days_red"]
    slope  = p["slope"]

    trend_txt = (f"{slope*30:+.1f} pts/month" if abs(slope) > 0.001 else "Stable")
    amb_txt   = f"{d_amb:.0f} days" if d_amb is not None else "Not projected"
    red_txt   = f"{d_red:.0f} days" if d_red is not None else "Not projected"

    fc[i].markdown(
        f"<div style='background:#FFFFFF;border:1px solid #E5E7EB;border-radius:10px;"
        f"padding:18px 20px;box-shadow:0 1px 3px rgba(0,0,0,0.06);'>"
        f"<div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;'>"
        f"<div><p style='margin:0;font-size:10px;font-weight:700;letter-spacing:0.09em;"
        f"color:#9CA3AF;text-transform:uppercase;'>Compressor {comp}</p>"
        f"<p style='margin:4px 0 0;font-size:22px;font-weight:700;color:#111827;'>"
        f"{'N/A' if pd.isna(hi) else f'{hi:.0f}'}<span style='font-size:12px;color:#9CA3AF;'>/100</span></p></div>"
        f"<span style='font-size:11px;font-weight:700;color:{a_c};background:{a_c}15;"
        f"border-radius:20px;padding:3px 10px;text-transform:uppercase;'>{st_c}</span></div>"
        f"<div style='background:#F3F4F6;border-radius:3px;height:5px;margin-bottom:14px;'>"
        f"<div style='width:{max(0,min(100,hi if not pd.isna(hi) else 0)):.0f}%;background:{a_c};"
        f"height:5px;border-radius:3px;'></div></div>"
        f"<table style='width:100%;font-size:12px;border-collapse:collapse;'>"
        f"<tr><td style='color:#6B7280;padding:3px 0;'>30-day trend</td>"
        f"<td style='text-align:right;font-weight:600;color:{'#DC2626' if slope < -0.1 else ('#D97706' if slope < 0 else '#15A34A')};'>{trend_txt}</td></tr>"
        f"<tr><td style='color:#6B7280;padding:3px 0;'>Days to Degraded Zone (70)</td>"
        f"<td style='text-align:right;font-weight:600;color:{C_AMBER};'>{amb_txt}</td></tr>"
        f"<tr><td style='color:#6B7280;padding:3px 0;'>Days to Critical Zone (55)</td>"
        f"<td style='text-align:right;font-weight:600;color:{C_RED};'>{red_txt}</td></tr>"
        f"</table></div>",
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

# ── Health trend + projection ─────────────────────────────────────────────────
st.markdown(section_title("Fleet Health Trend & Projection", "Dashed lines show linear projection from recent trend"), unsafe_allow_html=True)

fig_proj = go.Figure()
colors_comp = {"A": C_BLUE, "B": C_GREEN, "C": C_AMBER}
last_ts = df_full["Timestamp"].dropna().max()

for comp in compressors:
    hi_col = f"Compressor_{comp}_Health"
    if hi_col not in df_full.columns:
        continue
    data = df_full[["Timestamp", hi_col]].dropna().tail(2000)
    data[hi_col] = pd.to_numeric(data[hi_col], errors="coerce")
    data = data.dropna()
    c = colors_comp[comp]

    # Actual
    fig_proj.add_trace(go.Scatter(
        x=data["Timestamp"], y=data[hi_col],
        name=f"Compressor {comp}",
        line=dict(color=c, width=2),
        mode="lines",
    ))

    # Projection (60 days forward)
    p = projections[comp]
    if p["slope"] < 0:
        proj_days = 60
        proj_x = [last_ts + timedelta(days=i) for i in range(proj_days + 1)]
        proj_y = [p["health"] + p["slope"] * i for i in range(proj_days + 1)]
        proj_y_clamped = [max(0, y) for y in proj_y]
        fig_proj.add_trace(go.Scatter(
            x=proj_x, y=proj_y_clamped,
            name=f"Compressor {comp} (projected)",
            line=dict(color=c, width=1.5, dash="dash"),
            mode="lines",
            showlegend=True,
            opacity=0.6,
        ))

fig_proj.add_hline(y=70, line_dash="dot",  line_color=C_GREEN, annotation_text="Healthy ≥ 70")
fig_proj.add_hline(y=55, line_dash="dash", line_color=C_AMBER, annotation_text="Degraded ≥ 55")
fig_proj.add_hrect(y0=0,   y1=55,  fillcolor=C_RED,   opacity=0.04, layer="below", line_width=0)
fig_proj.add_hrect(y0=55,  y1=70,  fillcolor=C_AMBER, opacity=0.04, layer="below", line_width=0)
fig_proj.add_hrect(y0=70,  y1=110, fillcolor=C_GREEN, opacity=0.03, layer="below", line_width=0)
fig_proj.update_layout(
    template="plotly_white", height=340,
    yaxis_title="Health Index (0–100)",
    margin=dict(l=40, r=20, t=30, b=30),
    legend=dict(orientation="h", y=1.08),
    hovermode="x unified",
    paper_bgcolor="#FFFFFF", plot_bgcolor="#F9FAFB",
)
fig_proj.update_xaxes(gridcolor="#E5E7EB")
fig_proj.update_yaxes(gridcolor="#E5E7EB")
st.plotly_chart(fig_proj, use_container_width=True, key="health_proj")

st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

# ── Priority matrix ───────────────────────────────────────────────────────────
priority_col, schedule_col = st.columns([1, 1])

with priority_col:
    st.markdown(section_title("Maintenance Priority Matrix", "Urgency (time to threshold) vs criticality (health decline rate)"), unsafe_allow_html=True)

    matrix_data = []
    for comp in compressors:
        p = projections[comp]
        urgency   = (1 / p["days_amber"]) * 100 if p["days_amber"] and p["days_amber"] > 0 else 0
        criticality = abs(p["slope"]) * 30
        matrix_data.append({
            "Compressor": f"Comp {comp}",
            "Urgency":    urgency,
            "Criticality": criticality,
            "Health":     p["health"],
            "Status":     p["status"],
        })

    mdf = pd.DataFrame(matrix_data)
    status_colors = {
        "green": C_GREEN, "amber": C_AMBER, "red": C_RED, "offline": "#6B7280"
    }

    fig_matrix = go.Figure()
    for _, row in mdf.iterrows():
        color = status_colors.get(row["Status"], "#6B7280")
        fig_matrix.add_trace(go.Scatter(
            x=[row["Urgency"]], y=[row["Criticality"]],
            mode="markers+text",
            marker=dict(size=40, color=color, opacity=0.85, line=dict(color="#FFFFFF", width=2)),
            text=[row["Compressor"]],
            textposition="middle center",
            textfont=dict(color="#FFFFFF", size=12, family="Arial"),
            name=row["Compressor"],
            showlegend=False,
        ))

    fig_matrix.add_vrect(x0=50, x1=200, fillcolor=C_RED,   opacity=0.05, layer="below", line_width=0)
    fig_matrix.add_hrect(y0=2,  y1=20,  fillcolor=C_AMBER, opacity=0.05, layer="below", line_width=0)
    fig_matrix.update_layout(
        template="plotly_white", height=320,
        xaxis_title="Urgency (1/days to amber threshold × 100)",
        yaxis_title="Criticality (health pts decline / 30 days)",
        paper_bgcolor="#FFFFFF", plot_bgcolor="#F9FAFB",
        annotations=[
            dict(x=0.02, y=0.98, xref="paper", yref="paper", text="Low priority",
                 showarrow=False, font=dict(size=10, color="#9CA3AF")),
            dict(x=0.98, y=0.98, xref="paper", yref="paper", text="Critical",
                 showarrow=False, font=dict(size=10, color=C_RED), xanchor="right"),
        ],
        margin=dict(l=50, r=20, t=30, b=50),
    )
    fig_matrix.update_xaxes(gridcolor="#E5E7EB")
    fig_matrix.update_yaxes(gridcolor="#E5E7EB")
    st.plotly_chart(fig_matrix, use_container_width=True, key="priority_matrix")

with schedule_col:
    st.markdown(section_title("Recommended Actions", "Based on health index trends and threshold projections"), unsafe_allow_html=True)

    actions = []
    today = pd.Timestamp.now().normalize()
    for comp in compressors:
        p = projections[comp]
        hi = p["health"]
        d_amb = p["days_amber"]
        d_red = p["days_red"]
        st_c  = p["status"]

        if st_c == "red" or (not pd.isna(hi) and hi < 55):
            priority = "Critical"
            p_color  = C_RED
            action   = "Immediate inspection required. Schedule maintenance within 7 days."
            due_date = (today + timedelta(days=7)).strftime("%d %b %Y")
        elif st_c == "amber" or (d_amb is not None and d_amb < 14):
            priority = "High"
            p_color  = C_AMBER
            action   = f"Inspection recommended. Projected to reach amber in {d_amb:.0f} days." if d_amb else "Monitor closely. Health in amber zone."
            due_date = (today + timedelta(days=7 if d_amb is None else max(1, int(d_amb * 0.6)))).strftime("%d %b %Y")
        elif d_amb is not None and d_amb < 45:
            priority = "Planned"
            p_color  = C_BLUE
            action   = f"Schedule inspection. Projected amber threshold in {d_amb:.0f} days."
            due_date = (today + timedelta(days=int(d_amb * 0.7))).strftime("%d %b %Y")
        else:
            priority = "Routine"
            p_color  = C_GREEN
            action   = "No immediate action required. Follow standard PM schedule."
            due_date = "Standard PM cycle"

        actions.append({
            "Compressor": f"Compressor {comp}",
            "Priority":   priority,
            "Due By":     due_date,
            "Action":     action,
            "_color":     p_color,
        })

    for act in actions:
        c = act["_color"]
        st.markdown(
            f"<div style='background:#FFFFFF;border:1px solid #E5E7EB;border-left:4px solid {c};"
            f"border-radius:0 6px 6px 0;padding:12px 14px;margin-bottom:10px;'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;'>"
            f"<span style='font-size:13px;font-weight:700;color:#111827;'>{act['Compressor']}</span>"
            f"<span style='font-size:11px;font-weight:700;color:{c};background:{c}15;"
            f"padding:2px 10px;border-radius:20px;'>{act['Priority']}</span></div>"
            f"<p style='margin:0;font-size:12px;color:#374151;'>{act['Action']}</p>"
            f"<p style='margin:4px 0 0;font-size:11px;color:#9CA3AF;'>Due: {act['Due By']}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

# ── Maintenance schedule Gantt ────────────────────────────────────────────────
st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
st.markdown(section_title("Maintenance Window Timeline", "Projected inspection and overhaul windows"), unsafe_allow_html=True)

gantt_rows = []
today = pd.Timestamp.now().normalize()
for comp in compressors:
    p = projections[comp]
    st_c = p["status"]
    d_amb = p["days_amber"]

    if st_c == "red" or (not pd.isna(p["health"]) and p["health"] < 55):
        start_days, dur, label = 0, 3, "Immediate Inspection"
        bar_color = C_RED
    elif st_c == "amber" or (d_amb is not None and d_amb < 14):
        start_days, dur, label = max(1, int((d_amb or 5) * 0.5)), 2, "Planned Inspection"
        bar_color = C_AMBER
    elif d_amb is not None:
        start_days, dur, label = int(d_amb * 0.65), 3, "Scheduled Inspection"
        bar_color = C_BLUE
    else:
        start_days, dur, label = 90, 2, "Routine PM"
        bar_color = C_GREEN

    s = today + timedelta(days=start_days)
    e = s + timedelta(days=dur)
    gantt_rows.append(dict(comp=f"Compressor {comp}", start=s, end=e, label=label, color=bar_color))

fig_gantt = go.Figure()
for row in gantt_rows:
    fig_gantt.add_trace(go.Bar(
        x=[(row["end"] - row["start"]).days],
        y=[row["comp"]],
        base=[(row["start"] - today).days],
        orientation="h",
        marker_color=row["color"],
        marker_line_color="#FFFFFF",
        marker_line_width=1,
        name=row["label"],
        text=row["label"],
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(color="#FFFFFF", size=11),
        showlegend=False,
    ))

fig_gantt.add_vline(x=0, line_color="#374151", line_width=1.5, annotation_text="Today")
fig_gantt.update_layout(
    template="plotly_white", height=220,
    barmode="overlay",
    xaxis=dict(title="Days from today", gridcolor="#E5E7EB"),
    yaxis=dict(autorange="reversed"),
    paper_bgcolor="#FFFFFF", plot_bgcolor="#F9FAFB",
    margin=dict(l=140, r=20, t=20, b=40),
)
st.plotly_chart(fig_gantt, use_container_width=True, key="gantt_chart")

with st.expander("Projection Methodology"):
    st.markdown("""
**Trend method:** Ordinary least squares linear fit on the most recent 2,000 data points
of each compressor's Health Index.

**Projection:** The slope (health points/day) is extrapolated forward to find the day
the index is projected to cross the **Degraded Zone (70)** and **Critical Zone (55)** thresholds.

**Limitations:**
- Linear projection does not capture step-change events (maintenance events, part replacements)
- If the slope is positive (improving), no threshold crossing is projected
- For compressors with insufficient data, projections are not shown

**Recommended action intervals:**
| Urgency | Condition | Action |
|---|---|---|
| Critical | Health < 55 (Critical Zone) | Inspect within 7 days |
| High | Health 55–70 (Degraded Zone) OR ≤ 14 days to Degraded Zone | Inspect within days × 0.6 |
| Planned | 14–45 days to Degraded Zone | Schedule inspection at days × 0.7 |
| Routine | > 45 days to Degraded Zone or flat/improving trend | Follow standard PM programme |
    """)
