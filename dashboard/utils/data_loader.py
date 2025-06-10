import pandas as pd
import streamlit as st
from pathlib import Path

ROOT         = Path(__file__).resolve().parents[2]
ENRICHED_CSV = ROOT / "Combined_Data_with_KPIs.csv"
if not ENRICHED_CSV.exists():
    ENRICHED_CSV = ROOT / "sample_data" / "Combined_Data_with_KPIs.csv"
RAW_CSV      = ROOT / "Combined_Data.csv"

# ── Design parameters ─────────────────────────────────────────────────────────
DESIGN_MMSCFD   = 47.0   # correct design capacity
_LEGACY_MMSCFD  = 45.0   # denominator used when computing Plant Rate in existing CSV

# ── Colour palette ────────────────────────────────────────────────────────────
ALERT_COLORS = {"red": "#DC2626", "amber": "#D97706", "green": "#15A34A"}
C_BLUE   = "#2563EB"
C_GREEN  = "#15A34A"
C_AMBER  = "#D97706"
C_RED    = "#DC2626"
C_CARD   = "#FFFFFF"
C_BORDER = "#E5E7EB"
C_TEXT   = "#111827"
C_MUTED  = "#6B7280"

# ── Status display labels (replaces raw "green" / "amber" / "red") ────────────
CO_STATUS_LABELS = {
    "green": "IN SPECIFICATION",
    "amber": "ELEVATED — CAUTION",
    "red":   "OFF-SPECIFICATION — ALARM",
    "unknown": "STATUS UNKNOWN",
}
CO_STATUS_SUBTITLE = {
    "green": "Predicted CO within normal range (< 5 ppm)",
    "amber": "Predicted CO between 5 – 10 ppm — monitor closely",
    "red":   "Predicted CO exceeds 10 ppm specification limit",
    "unknown": "",
}
COMP_STATUS_LABELS = {
    "green":   "HEALTHY",
    "amber":   "DEGRADED",
    "red":     "CRITICAL",
    "offline": "STANDBY",
}


@st.cache_data(ttl=300, show_spinner="Loading plant data...")
def load_enriched() -> pd.DataFrame:
    df = pd.read_csv(ENRICHED_CSV, encoding="latin-1", low_memory=False)
    ts_col = "Timestamp" if "Timestamp" in df.columns else df.columns[0]
    df["Timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # Correct Plant Rate from legacy 45 MMSCFD denominator to 47 MMSCFD
    if "Plant Rate" in df.columns:
        df["Plant Rate"] = (
            pd.to_numeric(df["Plant Rate"], errors="coerce") * (_LEGACY_MMSCFD / DESIGN_MMSCFD)
        )

    # PSA Recovery: stored as fraction (0.0–1.0) → convert to %
    if "PSA Recovery" in df.columns:
        df["PSA Recovery (%)"] = pd.to_numeric(df["PSA Recovery"], errors="coerce") * 100

    return df


@st.cache_data(ttl=300, show_spinner="Loading sensor data...")
def load_raw_sensors() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV, encoding="latin-1", low_memory=False)
    ts_col = df.columns[0]
    df = df.iloc[2:].reset_index(drop=True)
    df = df.rename(columns={ts_col: "Timestamp"})
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    for c in df.columns:
        if c != "Timestamp":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("Timestamp").reset_index(drop=True)


def val(df: pd.DataFrame, col: str, default=float("nan")) -> float:
    if col not in df.columns:
        return default
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(s.iloc[-1]) if len(s) else default


def series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def latest_str(df: pd.DataFrame, col: str, default: str = "—") -> str:
    if col not in df.columns:
        return default
    s = df[col].dropna().astype(str)
    return s.iloc[-1] if len(s) else default


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Date Range")
    ts = df["Timestamp"].dropna()
    min_d = ts.min().date()
    max_d = ts.max().date()
    start = st.sidebar.date_input("From", value=min_d, min_value=min_d, max_value=max_d)
    end   = st.sidebar.date_input("To",   value=max_d, min_value=min_d, max_value=max_d)
    mask = (df["Timestamp"].dt.date >= start) & (df["Timestamp"].dt.date <= end)
    return df[mask].reset_index(drop=True)
