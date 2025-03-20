"""KPI formula helpers for post-ingestion processing."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

KPI_UNITS: dict[str, str] = {
    "Gross Efficiency": "BTU/SCF",
    "Net Efficiency": "BTU/SCF",
    "Burner Efficiency": "BTU/SCF",
    "NG Check (Material Balance)": "%",
    "Steam Balance (Material Balance)": "%",
    "RFG Agreement (Material Balance)": "%",
    "Hydrocarbon (HC) Balance (Material Balance)": "%",
    "Hydrcarbon/Recycle H2 (HC/H2) Balance (Material Balance)": "%",
    "Mix Tee Balance (Material Balance)": "%",
    "Burner Balance (Material Balance)": "%",
    "PSA Balance (Material Balance)": "%",
    "Coker Agreement (Material Balance)": "%",
    "Hydrogen Balance (Material Balance)": "%",
    "Plant Rate": "%",
    "Hydrotreater Outlet Temperature": "°F",
    "PSA Recovery": "%",
    "Shift dT (HTS Temperature Difference)": "°F",
    "S/C Ratio (Steam-to-Carbon)": "mol/mol",
    "Excess O2 in Flue Gas": "%",
    "Tube Outlet Temperature": "°F",
    "CO Slip (Syngas GC)": "%",
    "Methane Slip (Syngas GC)": "%",
    "CO in Product": "ppm",
    "Reformer Differential Pressure (Reformer DP)": "psid",
    "Purge Gas Buffer Vessel Pressure": "psig",
    "Purge Gas Vent": "%",
    "Midplant Vent": "%",
    "PSA Vent (SMR PSA Vent)": "%",
    "Product Vent": "%",
    "Steam Vent": "%",
    "NG Balance": "%",
    "S/C/OUT.CV": "mol/mol",
    "SMR PSA Vent": "%",
    "CH4  Syngas GC": "%",
    "Hydrotreater Out A": "°F",
    # ── Optimization KPIs ─────────────────────────────────────────────────────
    "H2/NG Yield Ratio":            "SCF H2/SCF NG",
    "H2 Lost to Purge":             "MSCFH",
    "CO Spec Headroom (Measured)":  "ppm",
    "S/C Excess over Coking Min":   "mol/mol",
    "Carbon Efficiency":            "%",
    "Production Value Index":       "%",
    "Steam Efficiency Index":       "index",
    "Reformer Severity Index":      "°F·%",
}

COLUMN_ALIASES: dict[str, list[str]] = {
    "ng check meter": [
        "us51700110fy0499ai1pvcv",
        "ng_check_meter",
        "ng_check",
    ],
    "bl rfg": [
        "us51700112fi5201ai1pvcv",
        "bl_rfg",
    ],
    "rfg": [
        "us51700170ai_0272sai1pvcv",
        "us51700170ai_0272sai1pvcv_feed",
        "rfg",
    ],
    "rfg hhv": [
        "us51700170ai_0272sai1pvcv",
        "rfg_hhv",
    ],
    "rfg to trim fuel": [
        "us51700110pic0564pid1outcv",
        "rfg_to_trim_fuel",
        "rfg_to_trim",
    ],
    "rfg to feed": [
        "us51700110fic0562pid1outcv",
        "rfg_to_feed",
    ],
    "bl h2 to citgo": [
        "us51700120ft013120fi0131cvcv",
        "bl_h2_to_citgo",
        "bl_h2",
    ],
    "bl coker ii h2": [
        "us51700118fic0515pid1pvcv",
        "bl_coker_ii_h2",
        "bl_coker_h2",
    ],
    "bl steam": [
        "us51700183ft061083fi0610cvcv",
        "bl_steam",
    ],
    "pgb flow": [
        "us51700111fic0070pid1pvcv",
        "pgb_flow",
        "purge_gas_buffer_vessel_flow",
    ],
    "trim fuel flow to reformer": [
        "us51700111fic0150pid1pvcv",
        "trim_fuel_flow_to_reformer",
        "trim_fuel",
    ],
    "feed gas to mixing tee": [
        "us51700110fic0120pid1pvcv",
        "feed_gas_to_mixing_tee",
        "feed_gas",
    ],
    "steam to mixing tee": [
        "us51700110fic0220pid1pvcv",
        "steam_to_mixing_tee",
        "steam_to_mix",
    ],
    "flow out of flue gas steam drum": [
        "us51700111fic0627ai1pvcv",
        "fg_steam",
    ],
    "flow out of pgb steam drum": [
        "us51700111fic0607ai1pvcv",
        "pgb_steam",
    ],
    "hydrogenation steam flow": [
        "us51700110fic0579ai1pvcv",
        "hydro_steam",
    ],
    "bl ng from citgo": [
        "us51700143fi148ai1pvcv",
        "bl_ng_from_citgo",
        "bl_ng",
    ],
    "bl ng (calc)": [
        "us51700143fi148ai1pvcvviabbcalc",
        "bl_ng_kscfh",
    ],
    "ng flow to compressor": [
        "us51700110fic0553pid1pvcv",
        "ng_flow_to_compressor",
    ],
    "rfg feed flow": [
        "us51700110fic0562ai1pvcv",
        "rfg_feed_flow",
    ],
    "psa product h2 flow": [
        "us51700118fi0121ai1pvcv",
        "psa_product_h2_flow",
        "psa_h2",
    ],
    "syngas to psa": [
        "us51700112fic0050pid1pvcv",
        "syngas_to_psa",
    ],
    "gc h2 in syngas": [
        "us51700170ai_0273aai1pvcv",
        "gc_h2_in_syngas",
        "gc_h2_syn_gas",
    ],
    "hts outlet temperature": [
        "us51700112ti0633_salm1pvcv",
        "hts_outlet_temperature",
    ],
    "hts inlet temperature": [
        "us51700111tic0060pid1pvcv",
        "hts_inlet_temperature",
    ],
    "excess o2 analyzer": [
        "us51700111aic0090pid1pvcv",
    ],
    "tube outlet thermocouple": [
        "us51700111tic0140pid1pvcv",
    ],
    "co in syngas (mass spec)": [
        "us51700170ai_0275dai1pvcv",
    ],
    "ch4 in syngas (mass spec)": [
        "us51700170ai_0275fai1pvcv",
    ],
    "co in product": [
        "us51700170ai0190dai1pvcv",
    ],
    "reformer tube-side dp": [
        "us51700110pdi_0020ai1pvcv",
    ],
    "pgb pressure pv": [
        "us51700118pic0210pid1pvcv",
    ],
    "pgb pressure sp": [
        "us51700118pic0210pid1spcv",
    ],
    "purge gas vent": [
        "us51700118pic0210pid1outcv",
    ],
    "midplant vent pressure controller": [
        "us51700112pic0100pid1outcv",
    ],
    "smr psa vent pressure controller": [
        "us51700118pic0125apid1outcv",
    ],
    "product vent pressure controller": [
        "us51700120pic0070bpid1outcv",
    ],
    "steam vent pressure controller": [
        "us51700183pic0020bpid1outcv",
    ],
    "sc_check_div3_outcv": [
        "us517001sc_checkdiv3outcv",
    ],
    "co analyzer product stream": [
        "us51700170ai0190dai1pvcv",
    ],
}


def normalize_name(name: Any) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(name)).strip("_")


def find_column(columns: pd.Index, key: str) -> str | None:
    key_norm = normalize_name(key)
    available = {normalize_name(col): col for col in columns}
    if key_norm in available:
        return available[key_norm]
    for alias in COLUMN_ALIASES.get(key_norm, []):
        if alias in available:
            return available[alias]
    for col_norm, actual in available.items():
        if key_norm in col_norm or any(alias in col_norm for alias in COLUMN_ALIASES.get(key_norm, [])):
            return actual
    return None


def get_value(row: pd.Series, key: str) -> Any:
    if key in row.index:
        return row[key]
    col = find_column(row.index, key)
    if col is not None:
        return row[col]
    return pd.NA


def safe_float(value: Any) -> float:
    if pd.isna(value):
        return math.nan
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return math.nan
    if isinstance(value, str):
        if value.strip().lower() in {"", "bad", "none", "nan"}:
            return math.nan
        cleaned = value.strip().replace("−", "-").replace(",", "")
        try:
            return float(cleaned)
        except ValueError:
            return math.nan
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return float(value)


def bool_bad(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() == "bad"


def safe_divide(numerator: float, denominator: float, default: float = math.nan) -> float:
    if denominator == 0 or math.isnan(denominator):
        return default
    return numerator / denominator


def selected_row(df: pd.DataFrame, row_selector: str = "last") -> pd.Series:
    if df.empty:
        raise ValueError("DataFrame is empty")
    if row_selector == "last":
        return df.iloc[-1]
    if row_selector == "first":
        return df.iloc[0]
    if row_selector == "mean":
        return df.mean(numeric_only=True)
    if row_selector.isdigit():
        return df.iloc[int(row_selector)]
    raise ValueError(f"Unsupported row_selector: {row_selector}")


def direct_tag(row: pd.Series, tag: str) -> float:
    return safe_float(get_value(row, tag))


def gross_efficiency(row: pd.Series) -> float:
    ng_check = safe_float(get_value(row, "PLANT_01:10FY0499/aI1/PV.CV"))
    bl_rfg = safe_float(get_value(row, "PLANT_01:12FI5201/AI1/PV.CV"))
    rfg = get_value(row, "PLANT_01:70AI_0272S/AI1/PV.CV")
    rfg_to_trim = safe_float(get_value(row, "PLANT_01:10PIC0564/PID1/OUT.CV"))
    rfg_to_feed = safe_float(get_value(row, "PLANT_01:10FIC0562/PID1/OUT.CV"))
    bl_h2 = safe_float(get_value(row, "PLANT_01:20FT0131/20FI0131.CV"))
    bl_coker_h2 = safe_float(get_value(row, "PLANT_01:18FIC0515/PID1/PV.CV"))
    if math.isnan(ng_check) or math.isnan(bl_rfg) or math.isnan(bl_h2) or math.isnan(bl_coker_h2):
        return math.nan
    rfg_term = 0.0
    if (rfg_to_trim + rfg_to_feed) > 0:
        if bool_bad(rfg):
            rfg_term = bl_rfg * 300.0
        else:
            rfg_term = bl_rfg * safe_float(rfg)
    denominator = bl_h2 - bl_coker_h2 * 24.0
    return safe_divide(ng_check * 24.0 * 1050.0 + rfg_term, denominator)


def net_efficiency(row: pd.Series) -> float:
    ng_check = safe_float(get_value(row, "PLANT_01:10FY0499/aI1/PV.CV"))
    bl_rfg = safe_float(get_value(row, "PLANT_01:12FI5201/AI1/PV.CV"))
    rfg_hhv = get_value(row, "PLANT_01:70AI_0272S/AI1/PV.CV")
    rfg_to_trim = safe_float(get_value(row, "PLANT_01:10PIC0564/PID1/OUT.CV"))
    rfg_to_feed = safe_float(get_value(row, "PLANT_01:10FIC0562/PID1/OUT.CV"))
    bl_steam = safe_float(get_value(row, "PLANT_01:83FT0610/83FI0610.CV"))
    bl_h2 = safe_float(get_value(row, "PLANT_01:20FT0131/20FI0131.CV"))
    bl_coker_h2 = safe_float(get_value(row, "PLANT_01:18FIC0515/PID1/PV.CV"))
    if math.isnan(ng_check) or math.isnan(bl_rfg) or math.isnan(bl_steam) or math.isnan(bl_h2) or math.isnan(bl_coker_h2):
        return math.nan
    rfg_term = 0.0
    if (rfg_to_trim + rfg_to_feed) > 0:
        if bool_bad(rfg_hhv):
            rfg_term = bl_rfg * 300.0
        else:
            rfg_term = bl_rfg * safe_float(rfg_hhv)
    numerator = ng_check * 24.0 * 1050.0 + rfg_term - bl_steam * 1.366 * 1000.0
    denominator = bl_h2 - bl_coker_h2 * 24.0
    result = safe_divide(numerator, denominator)
    # Clamp to physically plausible range; extreme values indicate near-zero denominator
    if math.isnan(result) or result < -500.0 or result > 1000.0:
        return math.nan
    return result


def burner_efficiency(row: pd.Series) -> float:
    pgb_flow = safe_float(get_value(row, "PLANT_01:11FIC0070/PID1/PV.CV"))
    trim_fuel = safe_float(get_value(row, "PLANT_01:11FIC0150/PID1/PV.CV"))
    feed_gas = safe_float(get_value(row, "PLANT_01:10FIC0120/PID1/PV.CV"))
    steam_to_mix = safe_float(get_value(row, "PLANT_01:10FIC0220/PID1/PV.CV"))
    if math.isnan(pgb_flow) or math.isnan(trim_fuel) or math.isnan(feed_gas) or math.isnan(steam_to_mix):
        return math.nan
    numerator = pgb_flow * 281.4 + trim_fuel * 1050.0
    denominator = feed_gas + steam_to_mix / 18.0 * 379.48
    return safe_divide(numerator, denominator)


def ng_check_balance(row: pd.Series) -> float:
    ng_check = safe_float(get_value(row, "PLANT_01:10FY0499/aI1/PV.CV"))
    bl_ng = get_value(row, "PLANT_01:43FI148/AI1/PV.CV")
    bl_ng_value = safe_float(bl_ng)
    if bool_bad(bl_ng) or math.isnan(ng_check) or math.isnan(bl_ng_value) or bl_ng_value == 0:
        return math.nan
    return abs((ng_check * 24.0 / 1000.0 - bl_ng_value) / bl_ng_value) * 100.0


def steam_balance(row: pd.Series) -> float:
    fg_steam = safe_float(get_value(row, "PLANT_01:11FI0627/AI1/PV.CV"))
    pgb_steam = safe_float(get_value(row, "PLANT_01:11FI0607/AI1/PV.CV"))
    hydro_steam = safe_float(get_value(row, "PLANT_01:10FI0579/AI1/PV.CV"))
    bl_steam = safe_float(get_value(row, "PLANT_01:83FT0610/83FI0610.CV"))
    steam_to_mix = safe_float(get_value(row, "PLANT_01:10FIC0220/PID1/PV.CV"))
    denom = fg_steam + pgb_steam + hydro_steam
    if math.isnan(denom) or denom == 0:
        return math.nan
    numerator = fg_steam + pgb_steam + hydro_steam - bl_steam / 24.0 - steam_to_mix
    return abs(numerator / denom) * 100.0


def rfg_agreement(row: pd.Series) -> float:
    bl_rfg = safe_float(get_value(row, "PLANT_01:12FI5201/AI1/PV.CV"))
    rfg_floboss = safe_float(get_value(row, "PLANT_01:10FT0092/10FI0092.CV"))
    if math.isnan(bl_rfg) or math.isnan(rfg_floboss) or bl_rfg == 0:
        return math.nan
    return abs((round(bl_rfg * 1000.0, 0) - rfg_floboss) / (bl_rfg * 1000.0)) * 100.0


def hydrocarbon_balance(row: pd.Series) -> float:
    ng_check = safe_float(get_value(row, "PLANT_01:10FY0499/aI1/PV.CV"))
    bl_rfg = safe_float(get_value(row, "PLANT_01:12FI5201/AI1/PV.CV"))
    ng_comp = safe_float(get_value(row, "PLANT_01:10FIC0553/PID1/PV.CV"))
    rfg_feed = safe_float(get_value(row, "PLANT_01:10FIC0562/AI1/PV.CV"))
    trim = safe_float(get_value(row, "PLANT_01:11FIC0150/PID1/PV.CV"))
    bl_ng = safe_float(get_value(row, "PLANT_01:43FI148/AI1/PV.CV"))
    rfg_to_trim = safe_float(get_value(row, "PLANT_01:10PIC0564/PID1/OUT.CV"))
    rfg_to_feed = safe_float(get_value(row, "PLANT_01:10FIC0562/PID1/OUT.CV"))
    if math.isnan(ng_check) or math.isnan(bl_rfg) or math.isnan(ng_comp) or math.isnan(trim) or math.isnan(bl_ng):
        return math.nan
    rfg_adj = bl_rfg / 24.0 if (rfg_to_trim + rfg_to_feed) > 0 else 0.0
    rfg_feed_adjusted = rfg_feed if rfg_to_feed > 0 else 0.0
    denom = bl_ng / 0.024 + rfg_adj
    if math.isnan(denom) or denom == 0:
        return math.nan
    numerator = ng_check + rfg_adj - ng_comp - rfg_feed_adjusted - trim
    return abs(numerator / denom) * 100.0


def hc_recycle_balance(row: pd.Series) -> float:
    ng_check = safe_float(get_value(row, "PLANT_01:10FY0499/aI1/PV.CV"))
    bl_rfg = safe_float(get_value(row, "PLANT_01:12FI5201/AI1/PV.CV"))
    h2_recycle = safe_float(get_value(row, "PLANT_01:12FIC0642/PID1/PV.CV"))
    trim = safe_float(get_value(row, "PLANT_01:11FIC0150/PID1/PV.CV"))
    feed_gas = safe_float(get_value(row, "PLANT_01:10FIC0120/PID1/PV.CV"))
    bl_ng = safe_float(get_value(row, "PLANT_01:43FI148/AI1/PV.CV"))
    if math.isnan(ng_check) or math.isnan(bl_rfg) or math.isnan(h2_recycle) or math.isnan(trim) or math.isnan(feed_gas) or math.isnan(bl_ng):
        return math.nan
    numerator = ng_check + bl_rfg / 24.0 + h2_recycle - trim - feed_gas
    denom = bl_ng / 0.024 + bl_rfg / 24.0 + h2_recycle
    return abs(numerator / denom) * 100.0 if denom != 0 else math.nan


def mix_tee_balance(row: pd.Series) -> float:
    ng_comp = safe_float(get_value(row, "PLANT_01:10FIC0553/PID1/PV.CV"))
    rfg_feed = safe_float(get_value(row, "PLANT_01:10FIC0562/AI1/PV.CV"))
    h2_recycle = safe_float(get_value(row, "PLANT_01:12FIC0642/PID1/PV.CV"))
    feed_gas = safe_float(get_value(row, "PLANT_01:10FIC0120/PID1/PV.CV"))
    if math.isnan(ng_comp) or math.isnan(h2_recycle) or math.isnan(feed_gas):
        return math.nan
    rfg_feed_adjusted = rfg_feed if rfg_feed > 0 else 0.0
    denom = ng_comp + rfg_feed_adjusted + h2_recycle
    if denom == 0:
        return math.nan
    numerator = ng_comp + rfg_feed_adjusted + h2_recycle - feed_gas
    return abs(numerator / denom) * 100.0


def burner_balance(row: pd.Series) -> float:
    ng_check = safe_float(get_value(row, "PLANT_01:10FY0499/aI1/PV.CV"))
    bl_rfg = safe_float(get_value(row, "PLANT_01:12FI5201/AI1/PV.CV"))
    h2_recycle = safe_float(get_value(row, "PLANT_01:12FIC0642/PID1/PV.CV"))
    trim = safe_float(get_value(row, "PLANT_01:11FIC0150/PID1/PV.CV"))
    feed_gas = safe_float(get_value(row, "PLANT_01:10FIC0120/PID1/PV.CV"))
    denom = ng_check + bl_rfg + h2_recycle
    if math.isnan(denom) or denom == 0:
        return math.nan
    numerator = ng_check + bl_rfg + h2_recycle - trim - feed_gas
    return abs(numerator / denom) * 100.0


def psa_balance(row: pd.Series) -> float:
    syngas_psa = safe_float(get_value(row, "PLANT_01:12FIC0050/PID1/PV.CV"))
    pgb_flow = safe_float(get_value(row, "PLANT_01:11FIC0070/PID1/PV.CV"))
    psa_h2 = safe_float(get_value(row, "PLANT_01:18FI0121/AI1/PV.CV"))
    if math.isnan(syngas_psa) or syngas_psa == 0 or math.isnan(pgb_flow) or math.isnan(psa_h2):
        return math.nan
    return abs((syngas_psa - pgb_flow - psa_h2) / syngas_psa) * 100.0


def coker_agreement(row: pd.Series) -> float:
    bl_coker_flow = safe_float(get_value(row, "PLANT_01:43FI180/AI1/PV.CV"))
    bl_coker_billing = safe_float(get_value(row, "PLANT_01:18FIC0515/PID1/PV.CV"))
    if math.isnan(bl_coker_flow) or bl_coker_flow == 0 or math.isnan(bl_coker_billing):
        return math.nan
    return abs((bl_coker_flow / 24.0 - bl_coker_billing) / (bl_coker_flow / 24.0)) * 100.0


def hydrogen_balance(row: pd.Series) -> float:
    psa_h2 = safe_float(get_value(row, "PLANT_01:18FI0121/AI1/PV.CV"))
    bl_coker_h2 = safe_float(get_value(row, "PLANT_01:18FIC0515/PID1/PV.CV"))
    bl_h2 = safe_float(get_value(row, "PLANT_01:20FT0131/20FI0131.CV"))
    h2_recycle = safe_float(get_value(row, "PLANT_01:12FIC0642/PID1/PV.CV"))
    denom = psa_h2 + bl_coker_h2
    if math.isnan(denom) or denom == 0 or math.isnan(psa_h2) or math.isnan(bl_coker_h2):
        return math.nan
    numerator = psa_h2 + bl_coker_h2 - bl_h2 / 24.0 - h2_recycle
    return abs(numerator / denom) * 100.0


def plant_rate(row: pd.Series) -> float:
    psa_h2_flow = safe_float(get_value(row, "PLANT_01:18FI0121/AI1/PV.CV"))
    if math.isnan(psa_h2_flow):
        return math.nan
    return psa_h2_flow * 24.0 / 1000.0 / 45.0 * 100.0


def hydrotreater_outlet_temperature(row: pd.Series) -> float:
    a = safe_float(get_value(row, "PLANT_01:10TI0090A_S/ALM1/PV.CV"))
    b = safe_float(get_value(row, "PLANT_01:10TI0090B_S/ALM1/PV.CV"))
    if math.isnan(a) and math.isnan(b):
        return math.nan
    return max(x for x in (a, b) if not math.isnan(x))


def psa_recovery(row: pd.Series) -> float:
    psa_h2 = safe_float(get_value(row, "PLANT_01:18FI0121/AI1/PV.CV"))
    syngas_psa = safe_float(get_value(row, "PLANT_01:12FIC0050/PID1/PV.CV"))
    gc_h2 = safe_float(get_value(row, "PLANT_01:70AI_0273A/AI1/PV.CV"))
    denominator = syngas_psa * gc_h2 / 100.0
    return safe_divide(psa_h2, denominator)


def shift_dt(row: pd.Series) -> float:
    outlet = safe_float(get_value(row, "PLANT_01:12TI0633_S/ALM1/PV.CV"))
    inlet = safe_float(get_value(row, "PLANT_01:11TIC0060/PID1/PV.CV"))
    if math.isnan(outlet) or math.isnan(inlet):
        return math.nan
    return outlet - inlet


def sc_ratio(row: pd.Series) -> float:
    steam_to_mix = safe_float(get_value(row, "PLANT_01:10FIC0220/PID1/PV.CV"))
    feed_gas = safe_float(get_value(row, "PLANT_01:10FIC0120/PID1/PV.CV"))
    h2_recycle = safe_float(get_value(row, "PLANT_01:12FIC0642/PID1/PV.CV"))
    if math.isnan(steam_to_mix) or math.isnan(feed_gas) or math.isnan(h2_recycle):
        return math.nan
    numerator = steam_to_mix * 1000.0 / 24.0 / 18.0152 * 379.48 / 1000.0 * 24.0
    return safe_divide(numerator, (feed_gas - h2_recycle) * 1.01)


def ng_balance(row: pd.Series) -> float:
    feed_gas = safe_float(get_value(row, "PLANT_01:10FIC0120/PID1/PV.CV"))
    trim_fuel = safe_float(get_value(row, "PLANT_01:11FIC0150/PID1/PV.CV"))
    h2_recycle = safe_float(get_value(row, "PLANT_01:12FIC0642/PID1/PV.CV"))
    bl_ng_kscfh = safe_float(get_value(row, "PLANT_01:43FI148/AI1/PV.CV  [via BB calc]"))
    if math.isnan(feed_gas) or math.isnan(trim_fuel) or math.isnan(h2_recycle) or math.isnan(bl_ng_kscfh) or bl_ng_kscfh == 0:
        return math.nan
    return abs((feed_gas + trim_fuel - h2_recycle - bl_ng_kscfh) / bl_ng_kscfh) * 100.0


# ── Optimization KPIs ────────────────────────────────────────────────────────
# Design / constraint reference values
_CO_SPEC_LIMIT      = 10.0    # ppm — product spec limit
_SC_MIN_NO_COKING   = 2.7     # mol/mol — minimum S/C to prevent catalyst coking
_H2_DESIGN_RATE     = 45.0    # MSCFH — 100% plant rate reference


def h2_ng_yield_ratio(row: pd.Series) -> float:
    """SCF H2 produced per SCF of NG feed — core conversion efficiency metric."""
    psa_h2  = safe_float(get_value(row, "PLANT_01:18FI0121/AI1/PV.CV"))   # MSCFH
    ng_feed = safe_float(get_value(row, "PLANT_01:10FY0499/aI1/PV.CV"))   # MSCFH
    return safe_divide(psa_h2, ng_feed)


def h2_lost_to_purge(row: pd.Series) -> float:
    """H2 entering PSA but leaving in purge gas instead of product — MSCFH."""
    syngas_psa = safe_float(get_value(row, "PLANT_01:12FIC0050/PID1/PV.CV"))  # MSCFH syngas
    psa_h2     = safe_float(get_value(row, "PLANT_01:18FI0121/AI1/PV.CV"))    # MSCFH product
    gc_h2      = safe_float(get_value(row, "PLANT_01:70AI_0273A/AI1/PV.CV"))  # % H2 in syngas
    if math.isnan(syngas_psa) or math.isnan(psa_h2) or math.isnan(gc_h2):
        return math.nan
    h2_to_psa = syngas_psa * gc_h2 / 100.0
    return max(0.0, h2_to_psa - psa_h2)


def co_spec_headroom_measured(row: pd.Series) -> float:
    """ppm of CO margin remaining to spec limit (10 ppm). Negative = over spec."""
    co_measured = safe_float(get_value(row, "PLANT_01:70AI0190D/AI1/PV.CV"))
    if math.isnan(co_measured):
        return math.nan
    return _CO_SPEC_LIMIT - co_measured


def sc_excess_over_minimum(row: pd.Series) -> float:
    """Current S/C minus coking-prevention minimum (2.7). Positive = excess steam."""
    steam_to_mix = safe_float(get_value(row, "PLANT_01:10FIC0220/PID1/PV.CV"))
    feed_gas     = safe_float(get_value(row, "PLANT_01:10FIC0120/PID1/PV.CV"))
    h2_recycle   = safe_float(get_value(row, "PLANT_01:12FIC0642/PID1/PV.CV"))
    if math.isnan(steam_to_mix) or math.isnan(feed_gas) or math.isnan(h2_recycle):
        return math.nan
    numerator = steam_to_mix * 1000.0 / 24.0 / 18.0152 * 379.48 / 1000.0 * 24.0
    sc = safe_divide(numerator, (feed_gas - h2_recycle) * 1.01)
    return sc - _SC_MIN_NO_COKING


def carbon_efficiency(row: pd.Series) -> float:
    """H2 produced as % of theoretical maximum from NG feed (CH4+2H2O->CO2+4H2).
    Typical range 65-85%; higher = better conversion and less feed waste."""
    psa_h2  = safe_float(get_value(row, "PLANT_01:18FI0121/AI1/PV.CV"))
    ng_feed = safe_float(get_value(row, "PLANT_01:10FY0499/aI1/PV.CV"))
    if math.isnan(psa_h2) or math.isnan(ng_feed) or ng_feed == 0:
        return math.nan
    # Theoretical: 1 SCF CH4 → 4 SCF H2; NG ~95% CH4 → effective ratio ≈ 3.8
    theoretical_h2 = ng_feed * 3.8
    return safe_divide(psa_h2, theoretical_h2) * 100.0


def production_value_index(row: pd.Series) -> float:
    """Plant Rate × PSA Recovery / 100. Composite throughput-quality metric (0-110)."""
    psa_h2     = safe_float(get_value(row, "PLANT_01:18FI0121/AI1/PV.CV"))
    syngas_psa = safe_float(get_value(row, "PLANT_01:12FIC0050/PID1/PV.CV"))
    gc_h2      = safe_float(get_value(row, "PLANT_01:70AI_0273A/AI1/PV.CV"))
    if math.isnan(psa_h2):
        return math.nan
    plant_rate_val = psa_h2 * 24.0 / 1000.0 / _H2_DESIGN_RATE * 100.0
    denom_rec = syngas_psa * gc_h2 / 100.0 if not (math.isnan(syngas_psa) or math.isnan(gc_h2)) else math.nan
    psa_rec = safe_divide(psa_h2, denom_rec) * 100.0 if not math.isnan(denom_rec) else math.nan
    if math.isnan(psa_rec):
        return math.nan
    return plant_rate_val * psa_rec / 100.0


def steam_efficiency_index(row: pd.Series) -> float:
    """PSA Recovery / S/C Ratio × 100. Higher = more H2 recovered per unit of steam.
    Directly captures the S/C vs recovery trade-off on a single index."""
    steam_to_mix = safe_float(get_value(row, "PLANT_01:10FIC0220/PID1/PV.CV"))
    feed_gas     = safe_float(get_value(row, "PLANT_01:10FIC0120/PID1/PV.CV"))
    h2_recycle   = safe_float(get_value(row, "PLANT_01:12FIC0642/PID1/PV.CV"))
    psa_h2       = safe_float(get_value(row, "PLANT_01:18FI0121/AI1/PV.CV"))
    syngas_psa   = safe_float(get_value(row, "PLANT_01:12FIC0050/PID1/PV.CV"))
    gc_h2        = safe_float(get_value(row, "PLANT_01:70AI_0273A/AI1/PV.CV"))
    if any(math.isnan(v) for v in [steam_to_mix, feed_gas, h2_recycle, psa_h2, syngas_psa, gc_h2]):
        return math.nan
    numerator_sc = steam_to_mix * 1000.0 / 24.0 / 18.0152 * 379.48 / 1000.0 * 24.0
    sc = safe_divide(numerator_sc, (feed_gas - h2_recycle) * 1.01)
    psa_rec = safe_divide(psa_h2, syngas_psa * gc_h2 / 100.0) * 100.0
    if math.isnan(sc) or sc == 0:
        return math.nan
    return safe_divide(psa_rec, sc) * 100.0


def reformer_severity_index(row: pd.Series) -> float:
    """Tube Outlet Temperature × Plant Rate / 100. Tracks how hard reformer is being
    pushed — rising index = accelerated catalyst ageing and tube stress."""
    tube_temp  = safe_float(get_value(row, "PLANT_01:11TIC0140/PID1/PV.CV"))
    psa_h2     = safe_float(get_value(row, "PLANT_01:18FI0121/AI1/PV.CV"))
    if math.isnan(tube_temp) or math.isnan(psa_h2):
        return math.nan
    plant_rate_val = psa_h2 * 24.0 / 1000.0 / _H2_DESIGN_RATE * 100.0
    return tube_temp * plant_rate_val / 100.0


KPI_FUNCTIONS: dict[str, callable] = {
    "Gross Efficiency": gross_efficiency,
    "Net Efficiency": net_efficiency,
    "Burner Efficiency": burner_efficiency,
    "NG Check (Material Balance)": ng_check_balance,
    "Steam Balance (Material Balance)": steam_balance,
    "RFG Agreement (Material Balance)": rfg_agreement,
    "Hydrocarbon (HC) Balance (Material Balance)": hydrocarbon_balance,
    "Hydrcarbon/Recycle H2 (HC/H2) Balance (Material Balance)": hc_recycle_balance,
    "Mix Tee Balance (Material Balance)": mix_tee_balance,
    "Burner Balance (Material Balance)": burner_balance,
    "PSA Balance (Material Balance)": psa_balance,
    "Coker Agreement (Material Balance)": coker_agreement,
    "Hydrogen Balance (Material Balance)": hydrogen_balance,
    "Plant Rate": plant_rate,
    "Hydrotreater Outlet Temperature": hydrotreater_outlet_temperature,
    "PSA Recovery": psa_recovery,
    "Shift dT (HTS Temperature Difference)": shift_dt,
    "S/C Ratio (Steam-to-Carbon)": sc_ratio,
    "Excess O2 in Flue Gas": lambda row: direct_tag(row, "PLANT_01:11AIC0090/PID1/PV.CV"),
    "Tube Outlet Temperature": lambda row: direct_tag(row, "PLANT_01:11TIC0140/PID1/PV.CV"),
    "CO Slip (Syngas GC)": lambda row: direct_tag(row, "PLANT_01:70AI_0275D/AI1/PV.CV"),
    "Methane Slip (Syngas GC)": lambda row: direct_tag(row, "PLANT_01:70AI_0275F/AI1/PV.CV"),
    "CO in Product": lambda row: direct_tag(row, "PLANT_01:70AI0190D/AI1/PV.CV"),
    "Reformer Differential Pressure (Reformer DP)": lambda row: direct_tag(row, "PLANT_01:10PDI_0020/AI1/PV.CV"),
    "Purge Gas Buffer Vessel Pressure": lambda row: direct_tag(row, "PLANT_01:18PIC0210/PID1/PV.CV"),
    "Purge Gas Vent": lambda row: direct_tag(row, "PLANT_01:18PIC0210/PID1/OUT.CV"),
    "Midplant Vent": lambda row: direct_tag(row, "PLANT_01:12PIC0100/PID1/OUT.CV"),
    "PSA Vent (SMR PSA Vent)": lambda row: direct_tag(row, "PLANT_01:18PIC0125A/PID1/OUT.CV"),
    "Product Vent": lambda row: direct_tag(row, "PLANT_01:20PIC0070B/PID1/OUT.CV"),
    "Steam Vent": lambda row: direct_tag(row, "PLANT_01:83PIC0020B/PID1/OUT.CV"),
    "NG Balance": ng_balance,
    "S/C/OUT.CV": lambda row: direct_tag(row, "PLANT_01:SC_CHECK/DIV3/OUT.CV"),
    "SMR PSA Vent": lambda row: direct_tag(row, "PLANT_01:18PIC0125A/PID1/OUT.CV"),
    "CH4  Syngas GC": lambda row: direct_tag(row, "PLANT_01:70AI_0273F/AI1/PV.CV"),
    "Hydrotreater Out A": lambda row: direct_tag(row, "PLANT_01:10TI0090A_S/ALM1/PV.CV"),
    # ── Optimization KPIs ─────────────────────────────────────────────────────
    "H2/NG Yield Ratio":           h2_ng_yield_ratio,
    "H2 Lost to Purge":            h2_lost_to_purge,
    "CO Spec Headroom (Measured)": co_spec_headroom_measured,
    "S/C Excess over Coking Min":  sc_excess_over_minimum,
    "Carbon Efficiency":           carbon_efficiency,
    "Production Value Index":      production_value_index,
    "Steam Efficiency Index":      steam_efficiency_index,
    "Reformer Severity Index":     reformer_severity_index,
}


def calculate_kpis(df: pd.DataFrame, row_selector: str = "last") -> dict[str, float]:
    if df.empty:
        return {}
    row = selected_row(df, row_selector)
    return {name: float(value) if not math.isnan(value) else math.nan for name, value in ((name, func(row)) for name, func in KPI_FUNCTIONS.items())}


def add_kpi_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the dataframe with KPI columns appended."""
    new_df = df.copy()
    if "Timestamp" in new_df.columns:
        new_df["date"] = pd.to_datetime(new_df["Timestamp"], errors="coerce").dt.date

    for kpi_name, kpi_func in KPI_FUNCTIONS.items():
        new_df[kpi_name] = new_df.apply(kpi_func, axis=1)
    return new_df


def save_dataframe_with_kpis(df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """Add KPI columns to the dataframe and save the updated dataframe to CSV."""
    updated_df = add_kpi_columns(df)
    updated_df.to_csv(output_path, index=False, encoding="latin-1")
    return updated_df


def kpis_to_dataframe(kpis: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([kpis])


def save_kpis(kpis: dict[str, float], output_path: str) -> None:
    df = kpis_to_dataframe(kpis)
    df.to_csv(output_path, index=False)


def get_kpi_units(kpis: dict[str, float]) -> dict[str, str]:
    return {name: KPI_UNITS.get(name, "") for name in kpis}


def format_kpis(kpis: dict[str, float]) -> dict[str, dict[str, Any]]:
    return {name: {"value": value, "unit": KPI_UNITS.get(name, "")} for name, value in kpis.items()}
