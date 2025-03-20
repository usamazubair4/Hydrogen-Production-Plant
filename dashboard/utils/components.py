import streamlit as st
import pandas as pd

_CSS = """
<style>
/* Hide default Streamlit multipage auto-nav */
[data-testid="stSidebarNav"] { display: none; }

/* Metric → tile cards */
div[data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    padding: 16px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.07);
}
div[data-testid="stMetricLabel"] p {
    font-size: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 0.09em !important;
    color: #9CA3AF !important;
    text-transform: uppercase !important;
}
div[data-testid="stMetricValue"] {
    font-size: 22px !important;
    font-weight: 700 !important;
    color: #111827 !important;
}

/* Nav link active state */
[data-testid="stPageLink-active"] {
    background: #EFF6FF;
    border-radius: 6px;
}

/* Tabs cleaner */
button[data-baseweb="tab"] { font-size: 13px; font-weight: 600; }

/* Equal-height columns within the same row */
[data-testid="stHorizontalBlock"] { align-items: stretch !important; }

/* Tighter main padding */
.main .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

/* Dataframe header */
.stDataFrame thead th {
    font-size: 11px; font-weight: 700;
    letter-spacing: 0.07em; text-transform: uppercase;
}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def nav_sidebar(df_full: pd.DataFrame) -> pd.DataFrame:
    """Render professional sidebar navigation and date filter. Returns filtered df."""
    from utils.data_loader import sidebar_filters
    inject_css()
    with st.sidebar:
        st.markdown(
            "<div style='padding:4px 0 8px 0;'>"
            "<p style='margin:0;font-size:10px;font-weight:700;letter-spacing:0.12em;"
            "color:#9CA3AF;text-transform:uppercase;'>SMR Plant</p>"
            "<p style='margin:2px 0 0;font-size:20px;font-weight:700;color:#111827;"
            "line-height:1.2;'>Dashboard</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.page_link("app.py",                             label="Overview",           icon=":material/dashboard:")
        st.page_link("pages/2_Performance.py",             label="Performance",        icon=":material/analytics:")
        st.page_link("pages/3_Quality_Prediction.py",      label="Quality Prediction", icon=":material/science:")
        st.page_link("pages/4_Reliability.py",             label="Reliability",        icon=":material/engineering:")
        st.page_link("pages/5_Optimisation.py",            label="Optimisation",       icon=":material/tune:")
        st.page_link("pages/6_Simulation.py",              label="Simulation",         icon=":material/model_training:")
        st.page_link("pages/7_Maintenance_Scheduling.py",  label="Maintenance",        icon=":material/calendar_month:")
    return sidebar_filters(df_full)


def kpi_tile(label: str, value: str, subtitle: str = "", accent: str = "#2563EB",
             alert_text: str = "") -> str:
    sub = (f"<p style='margin:4px 0 0;font-size:11px;color:#9CA3AF;'>{subtitle}</p>"
           if subtitle else "")
    badge = (f"<p style='margin:6px 0 0;font-size:10px;font-weight:700;color:{accent};"
             f"text-transform:uppercase;letter-spacing:0.05em;'>{alert_text}</p>"
             if alert_text else "")
    return (
        f"<div style='background:#FFFFFF;border:1px solid #E5E7EB;"
        f"border-top:3px solid {accent};border-radius:8px;"
        f"padding:16px 18px;box-shadow:0 1px 3px rgba(0,0,0,0.06);"
        f"min-height:108px;height:100%;display:flex;flex-direction:column;justify-content:flex-start;'>"
        f"<p style='margin:0;font-size:10px;font-weight:700;letter-spacing:0.09em;"
        f"color:#9CA3AF;text-transform:uppercase;'>{label}</p>"
        f"<p style='margin:6px 0 0;font-size:26px;font-weight:700;color:#111827;"
        f"line-height:1.1;'>{value}</p>{sub}{badge}</div>"
    )


def health_tile(comp: str, health: float, status_key: str, status_color: str,
                bear: float, vib: float, oil: float, cr: float) -> str:
    """Compressor health summary card — anomaly detection intentionally excluded."""
    from utils.data_loader import COMP_STATUS_LABELS

    display_status = COMP_STATUS_LABELS.get(status_key.lower(), status_key.upper())

    def bar(v: float) -> str:
        pct = max(0, min(100, v if not pd.isna(v) else 0))
        c = "#15A34A" if pct >= 70 else ("#D97706" if pct >= 55 else "#DC2626")
        return (
            f"<div style='display:flex;align-items:center;gap:8px;padding:3px 0;'>"
            f"<div style='flex:1;background:#F3F4F6;border-radius:3px;height:5px;'>"
            f"<div style='width:{pct:.0f}%;background:{c};height:5px;border-radius:3px;'></div>"
            f"</div><span style='font-size:12px;font-weight:600;color:#374151;"
            f"width:24px;text-align:right;'>{pct:.0f}</span></div>"
        )

    hi     = health if not pd.isna(health) else 0
    hi_pct = max(0, min(100, hi))
    hi_c   = "#15A34A" if hi >= 70 else ("#D97706" if hi >= 55 else "#DC2626")

    return (
        f"<div style='background:#FFFFFF;border:1px solid #E5E7EB;border-radius:10px;"
        f"padding:18px 20px;box-shadow:0 1px 3px rgba(0,0,0,0.06);height:100%;'>"

        f"<div style='display:flex;justify-content:space-between;align-items:flex-start;"
        f"margin-bottom:12px;'>"
        f"<div><p style='margin:0;font-size:10px;font-weight:700;letter-spacing:0.09em;"
        f"color:#9CA3AF;text-transform:uppercase;'>Compressor {comp}</p>"
        f"<p style='margin:2px 0 0;font-size:22px;font-weight:700;color:#111827;'>"
        f"{hi:.0f}<span style='font-size:13px;color:#9CA3AF;'>/100</span></p></div>"
        f"<span style='font-size:11px;font-weight:700;color:{status_color};"
        f"background:{status_color}15;border-radius:20px;padding:3px 10px;"
        f"text-transform:uppercase;letter-spacing:0.05em;'>{display_status}</span></div>"

        f"<div style='background:#F3F4F6;border-radius:4px;height:6px;margin-bottom:14px;'>"
        f"<div style='width:{hi_pct:.0f}%;background:{hi_c};height:6px;border-radius:4px;'></div></div>"

        f"<p style='margin:0 0 4px;font-size:10px;font-weight:700;letter-spacing:0.08em;"
        f"color:#9CA3AF;text-transform:uppercase;'>Sub-scores</p>"
        f"<div style='font-size:11px;color:#6B7280;margin-bottom:2px;'>Bearing (35%)</div>{bar(bear)}"
        f"<div style='font-size:11px;color:#6B7280;margin-top:6px;margin-bottom:2px;'>Vibration (25%)</div>{bar(vib)}"
        f"<div style='font-size:11px;color:#6B7280;margin-top:6px;margin-bottom:2px;'>Oil System (25%)</div>{bar(oil)}"
        f"<div style='font-size:11px;color:#6B7280;margin-top:6px;margin-bottom:2px;'>Comp. Ratio (15%)</div>{bar(cr)}"
        f"</div>"
    )


def alert_banner(text: str, color: str) -> str:
    return (
        f"<div style='background:{color}15;border-left:4px solid {color};"
        f"border-radius:4px;padding:12px 18px;margin-bottom:16px;'>"
        f"<span style='font-size:14px;font-weight:600;color:{color};'>{text}</span>"
        f"</div>"
    )


def section_title(title: str, subtitle: str = "") -> str:
    sub = (f"<p style='margin:2px 0 0;font-size:12px;color:#9CA3AF;'>{subtitle}</p>"
           if subtitle else "")
    return (
        f"<div style='margin:8px 0 16px;'>"
        f"<p style='margin:0;font-size:15px;font-weight:700;color:#111827;"
        f"letter-spacing:-0.01em;'>{title}</p>{sub}</div>"
    )
