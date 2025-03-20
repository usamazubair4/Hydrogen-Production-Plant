import plotly.graph_objects as go
import pandas as pd

TEMPLATE = "plotly_white"
C_BLUE   = "#2563EB"
C_GREEN  = "#15A34A"
C_AMBER  = "#D97706"
C_RED    = "#DC2626"
C_GREY   = "#9CA3AF"

_Z_GREEN = "#DCFCE7"
_Z_AMBER = "#FEF9C3"
_Z_RED   = "#FEE2E2"

_FONT = "Inter, Arial, sans-serif"


def timeseries(df: pd.DataFrame, cols: list[str], labels: list[str] = None,
               colors: list[str] = None, title: str = "", y_label: str = "",
               height: int = 320) -> go.Figure:
    labels = labels or cols
    colors = colors or [C_BLUE, C_AMBER, C_GREEN, C_RED]
    fig = go.Figure()
    for i, col in enumerate(cols):
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        fig.add_trace(go.Scatter(
            x=df["Timestamp"], y=s,
            name=labels[i],
            line=dict(color=colors[i % len(colors)], width=2),
            mode="lines",
        ))
    fig.update_layout(
        template=TEMPLATE, title=title, height=height,
        yaxis_title=y_label,
        margin=dict(l=40, r=20, t=40, b=30),
        legend=dict(orientation="h", y=1.08),
        hovermode="x unified",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F9FAFB",
        font=dict(family=_FONT),
    )
    fig.update_xaxes(gridcolor="#E5E7EB", linecolor="#E5E7EB")
    fig.update_yaxes(gridcolor="#E5E7EB", linecolor="#E5E7EB")
    return fig


def co_gauge(value: float, height: int = 300) -> go.Figure:
    """
    Semi-circle CO gauge. The arc occupies the top 78% of the figure via
    domain; status text sits in the bottom 22% as layout annotations.
    """
    if pd.isna(value) or value != value:
        value = 0.0
    value = float(max(0.0, min(30.0, value)))

    if value < 5:
        c = C_GREEN
        status = "In Specification"
        sub    = "< 5 ppm  ·  Normal operating range"
    elif value < 10:
        c = C_AMBER
        status = "Elevated — Caution"
        sub    = "5 – 10 ppm  ·  Monitor closely"
    else:
        c = C_RED
        status = "Off-Specification — Alarm"
        sub    = f"≥ 10 ppm  ·  {value - 10:.2f} ppm above spec"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        # Reserve bottom 22% for status annotations
        domain={"x": [0, 1], "y": [0.20, 1.0]},
        number={
            "suffix": " ppm",
            "font": {"size": 38, "color": c, "family": "Arial Black, Arial, sans-serif"},
            "valueformat": ".2f",
        },
        gauge={
            "axis": {
                "range": [0, 25],
                "tickwidth": 1, "dtick": 5,
                "tickcolor": "#D1D5DB",
                "tickfont": {"size": 10, "color": "#9CA3AF"},
            },
            "bar": {"color": c, "thickness": 0.30, "line": {"width": 0}},
            "bgcolor": "#F8FAFC",
            "borderwidth": 2,
            "bordercolor": "#E5E7EB",
            "steps": [
                {"range": [0,  5],  "color": _Z_GREEN},
                {"range": [5,  10], "color": _Z_AMBER},
                {"range": [10, 25], "color": _Z_RED},
            ],
            "threshold": {
                "line": {"color": "#991B1B", "width": 4},
                "thickness": 0.90,
                "value": 10,
            },
        },
    ))

    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="#FFFFFF",
        font=dict(family=_FONT),
        annotations=[
            dict(
                text=f"<b>{status}</b>",
                x=0.5, y=0.155,
                xref="paper", yref="paper",
                xanchor="center", yanchor="middle",
                showarrow=False,
                font=dict(size=13, color=c, family=_FONT),
            ),
            dict(
                text=sub,
                x=0.5, y=0.055,
                xref="paper", yref="paper",
                xanchor="center", yanchor="middle",
                showarrow=False,
                font=dict(size=10, color="#9CA3AF", family=_FONT),
            ),
        ],
    )
    return fig


def health_gauge(value: float, label: str = "", height: int = 260) -> go.Figure:
    """
    Semi-circle health index gauge (0–100).
    Arc occupies top 78%; status text in bottom 22% via annotations.
    """
    if pd.isna(value) or value != value:
        value = 0.0
        c = C_GREY; status = "No Data"; sub = ""
    elif value >= 70:
        c = C_GREEN; status = "Healthy"; sub = "Health Index ≥ 70"
    elif value >= 55:
        c = C_AMBER; status = "Degraded"; sub = "Health Index 55 – 69  ·  Monitor"
    else:
        c = C_RED;   status = "Critical"; sub = "Health Index < 55  ·  Action required"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={"x": [0, 1], "y": [0.18, 1.0]},
        number={
            "font": {"size": 40, "color": c, "family": "Arial Black, Arial, sans-serif"},
            "valueformat": ".0f",
            "suffix": "/100",
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickvals": [0, 25, 55, 70, 100],
                "ticktext":  ["0", "25", "55", "70", "100"],
                "tickcolor": "#D1D5DB",
                "tickfont": {"size": 10, "color": "#9CA3AF"},
            },
            "bar": {"color": c, "thickness": 0.30, "line": {"width": 0}},
            "bgcolor": "#F8FAFC",
            "borderwidth": 2,
            "bordercolor": "#E5E7EB",
            "steps": [
                {"range": [0,  55],  "color": _Z_RED},
                {"range": [55, 70],  "color": _Z_AMBER},
                {"range": [70, 100], "color": _Z_GREEN},
            ],
            "threshold": {
                "line": {"color": "#374151", "width": 3},
                "thickness": 0.80,
                "value": value,
            },
        },
        title={
            "text": f"<span style='font-size:11px;color:#9CA3AF'>{label}</span>",
            "font": {"size": 12, "color": "#6B7280", "family": _FONT},
        },
    ))

    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="#FFFFFF",
        font=dict(family=_FONT),
        annotations=[
            dict(
                text=f"<b>{status}</b>",
                x=0.5, y=0.12,
                xref="paper", yref="paper",
                xanchor="center", yanchor="middle",
                showarrow=False,
                font=dict(size=13, color=c, family=_FONT),
            ),
            dict(
                text=sub,
                x=0.5, y=0.03,
                xref="paper", yref="paper",
                xanchor="center", yanchor="middle",
                showarrow=False,
                font=dict(size=10, color="#9CA3AF", family=_FONT),
            ),
        ],
    )
    return fig


def subscore_bar(scores: dict[str, float], height: int = 220) -> go.Figure:
    labels = list(scores.keys())
    values = [v if not pd.isna(v) else 0 for v in scores.values()]
    colors = [C_GREEN if v >= 70 else (C_AMBER if v >= 55 else C_RED) for v in values]
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors,
        text=[f"{v:.0f}" for v in values], textposition="outside",
        marker_line_color="#FFFFFF", marker_line_width=1,
    ))
    fig.update_layout(
        template=TEMPLATE, height=height, yaxis=dict(range=[0, 115]),
        margin=dict(l=20, r=20, t=20, b=30),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#F9FAFB",
        font=dict(family=_FONT),
    )
    fig.update_xaxes(gridcolor="#E5E7EB")
    fig.update_yaxes(gridcolor="#E5E7EB")
    return fig


def bar_chart(labels: list, values: list, title: str = "", color: str = C_BLUE,
              y_label: str = "", height: int = 280) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=color,
        marker_line_color="#FFFFFF", marker_line_width=1,
    ))
    fig.update_layout(
        template=TEMPLATE, title=title, height=height,
        yaxis_title=y_label, margin=dict(l=40, r=20, t=40, b=60),
        xaxis_tickangle=-30, paper_bgcolor="#FFFFFF", plot_bgcolor="#F9FAFB",
        font=dict(family=_FONT),
    )
    fig.update_xaxes(gridcolor="#E5E7EB")
    fig.update_yaxes(gridcolor="#E5E7EB")
    return fig
