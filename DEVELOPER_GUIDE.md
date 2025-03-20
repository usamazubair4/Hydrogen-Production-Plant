# Developer Guide — Lemont Plant ML Pipeline

**Plant:** Lemont Plant Steam Methane Reformer (SMR) — Metheson Gas  
**Client:** Octopus Digital

---

## Overview

Eight-stage pipeline: raw DCS export → enriched CSV with KPIs, ML CO predictions, compressor health scores, and optimisation KPIs.

```
Combined_Data.csv  (raw DCS export — 1,126 columns, 3-row header)
        │
        ▼  Stage 1 — Data_Ingestion.py
        │  load_and_clean_plant_data()
        │  Promotes row-2 DCS tags to column headers; drops metadata rows
        ▼
        │  Stage 2 — kpi_formulas.py
        │  add_kpi_columns()
        │  36 engineering KPIs + 8 optimisation KPIs  (row-by-row apply)
        ▼
        │  Stage 3 — co_product_model.py
        │  COPredictor.load()  or  train_model()
        │  Ridge / RF / XGBoost ensemble → saved as co_predictor.pkl
        ▼
        │  Stage 4 — feature_engineering.py + co_product_model.py
        │  compute_physics_features() → appended to df
        │  predictor.predict()        → CO_Predicted_ppm, CO_Alert_Level
        ▼
        │  Stage 5 — compressor_reliability.py
        │  CompressorHealthMonitor.load()  or  train_monitor()
        │  score_dataframe() → 24 Compressor_X_* columns
        ▼
        │  Stage 6 — optimization.py
        │  compute_opt_kpis()     → 5 post-ML optimisation KPI columns
        │  OptimizationAnalyzer   → trade-off curves + envelope + recs
        ▼
        │  Stage 7 — Save
        ▼
Combined_Data_with_KPIs.csv  (1,206 columns)
        │
        ▼  Stage 8 — Console report (latest row real-time snapshot)
```

---

## Module Reference

| File | Role |
|---|---|
| `main.py` | CLI entry point — wires all 8 stages |
| `Data_Ingestion.py` | Reads and cleans the raw DCS CSV |
| `kpi_formulas.py` | 44 engineering + optimisation KPIs, tag resolution helpers |
| `feature_engineering.py` | First-principles WGS thermodynamic features; ML feature matrix builder |
| `co_product_model.py` | `COPredictor` — Ridge/RF/XGB ensemble; SHAP analysis; real-time prediction |
| `compressor_reliability.py` | `CompressorHealthMonitor` — composite health index + Isolation Forest per compressor |
| `compressor_eda.py` | Standalone EDA: 8 diagnostic plots for compressor sensor data |
| `optimization.py` | `OptimizationAnalyzer` — post-ML KPIs, trade-off curves, 2D envelope, recommendations |

Saved artefacts:

| File | Contents |
|---|---|
| `co_predictor.pkl` | Fitted `COPredictor` (Ridge pipeline + feature names) |
| `compressor_monitor.pkl` | Fitted `CompressorHealthMonitor` (Isolation Forest × 3 + scalers) |
| `model_plots/` | Training plots (CO model SHAP, compressor health timeseries, etc.) |
| `eda_plots/` | Exploratory plots from compressor EDA |

---

## Running the Pipeline

```bash
# Standard run — loads saved models if present, trains on first run
uv run python main.py

# Force retrain of both CO model and compressor monitor
uv run python main.py --train

# KPI calculation only (skip all ML)
uv run python main.py --no-prediction

# Skip compressor reliability scoring
uv run python main.py --no-reliability

# Skip optimisation analysis
uv run python main.py --no-optimisation

# Custom file paths
uv run python main.py --input path/to/raw.csv --output path/to/out.csv

# Custom model paths
uv run python main.py --model co_predictor.pkl --comp-model compressor_monitor.pkl
```

### Python environment

The project uses `uv` to manage the virtual environment (`.venv/`). All dependencies are installed there. Always run via `uv run python` rather than bare `python`.

```bash
uv run python --version   # 3.12
uv pip install -r requirements.txt
```

---

## Raw CSV Structure — Critical Detail

The plant DCS export has **three header rows** before the data:

| File row (1-indexed) | pandas index after `pd.read_csv` | Content |
|---|---|---|
| 1 | columns (headers) | **Friendly display names** — e.g. `"Compressor A Motor Current"` |
| 2 | iloc[0] | Units descriptions — e.g. `"Amps"` |
| 3 | iloc[1] | **DCS tag paths** — e.g. `"US517001:12TI0633_S/ALM1/PV.CV"` |
| 4+ | iloc[2]+ | Actual time-series data |

`load_and_clean_plant_data()` promotes the **DCS tag row** (iloc[1]) to column headers and drops rows 0–1:

```python
raw_df.columns = raw_df.iloc[1]   # DCS tags become column names
cleaned_df = raw_df.iloc[2:].reset_index(drop=True)
```

**Consequence:** the enriched DataFrame and output CSV use DCS tag strings as column names for raw sensor columns. KPI columns use the friendly names registered in `KPI_FUNCTIONS`.

**Important for compressor reliability:** the `CompressorHealthMonitor` expects friendly column names (e.g. `"Compressor A Motor Current"`) which only exist when the raw CSV is loaded directly with `pd.read_csv()` (friendly names in file row 1). The training and scoring steps always load the raw CSV directly — never the enriched CSV — for this reason. See Stage 5 section.

---

## Stage 1 — Data Ingestion (`Data_Ingestion.py`)

```python
from Data_Ingestion import load_and_clean_plant_data
df = load_and_clean_plant_data("Combined_Data.csv")
# Returns: DataFrame with DCS tag column names, Timestamp as datetime64
```

Steps:
1. Read with `encoding='latin-1'` (special characters in tag names).
2. Promote iloc[1] (DCS tag row) to column headers.
3. Drop iloc[0] and iloc[1]; reset index.
4. Rename first column to `Timestamp`.
5. Parse `Timestamp` with `pd.to_datetime(..., errors='coerce')`.

---

## Stage 2 — KPI Calculation (`kpi_formulas.py`)

### Tag resolution

| Helper | Purpose |
|---|---|
| `normalize_name(s)` | Lowercase alphanumeric + underscores |
| `COLUMN_ALIASES` | Dict: canonical key → list of alternate normalized tag names |
| `find_column(columns, key)` | Exact → alias → substring match; returns `None` if unresolved |
| `get_value(row, key)` | Calls `find_column`; returns cell value or `pd.NA` |
| `direct_tag(row, tag)` | Single-tag shortcut used by simple pass-through KPIs |

### Data quality helpers

| Helper | Behaviour |
|---|---|
| `safe_float(v)` | `float(v)` or `nan` for `None`, `pd.NA`, `""`, `"bad"` |
| `bool_bad(v)` | `True` if value is the DCS bad-quality string `"bad"` |
| `safe_divide(n, d)` | `n/d`; `nan` if `d == 0` or either is `nan` |

### KPIs — engineering (original 36)

**Efficiency (3)**

| KPI | Unit | Formula summary |
|---|---|---|
| Gross Efficiency | BTU/SCF | `(NG × 24 × 1050 + RFG_term) / (H2_bl − Coker_H2 × 24)` |
| Net Efficiency | BTU/SCF | Same numerator minus `bl_steam × 1.366 × 1000`; **clamped to [−500, 1000] — returns NaN outside this range** (near-zero denominator guard) |
| Burner Efficiency | BTU/SCF | `(PGB × 281.4 + TrimFuel × 1050) / (FeedGas + Steam/18 × 379.48)` |

**Material balance (10, all in %)**
NG Check, Steam Balance, RFG Agreement, HC Balance, HC/H2 Balance, Mix Tee Balance, Burner Balance, PSA Balance, Coker Agreement, Hydrogen Balance.

**Process parameters (23)**
Plant Rate (%), Hydrotreater Outlet Temp (°F), PSA Recovery (fraction 0–1 — **display as ×100 on dashboard**), Shift dT (°F), S/C Ratio (mol/mol), Excess O2 (%), Tube Outlet Temp (°F), CO Slip (%), Methane Slip (%), CH4 Syngas GC (%), CO in Product (ppm), Reformer DP (psid), PGB Pressure (psig), 5 vent controllers (%), NG Balance (%), S/C Out CV, plus date.

### KPIs — optimisation (8 new)

Added to `KPI_FUNCTIONS` and `KPI_UNITS`; computed in Stage 2 alongside original KPIs.

| KPI | Unit | Formula |
|---|---|---|
| H2/NG Yield Ratio | SCF/SCF | PSA H2 / NG feed (both MSCFH) |
| H2 Lost to Purge | MSCFH | syngas × %H2 − PSA H2 |
| CO Spec Headroom (Measured) | ppm | 10.0 − CO_in_Product |
| S/C Excess over Coking Min | mol/mol | S/C − 2.7 |
| Carbon Efficiency | % | PSA H2 / (NG feed × 3.8) × 100 |
| Production Value Index | % | Plant Rate × PSA Recovery |
| Steam Efficiency Index | index | PSA Recovery / S/C × 100 |
| Reformer Severity Index | °F·% | Tube Temp × Plant Rate / 100 |

Constants used: `_CO_SPEC_LIMIT = 10.0`, `_SC_MIN_NO_COKING = 2.7`, `_H2_DESIGN_RATE = 45.0` (MMSCFD).

### Adding a new KPI

```python
# 1. Write the function in kpi_formulas.py
def my_kpi(row: pd.Series) -> float:
    a = safe_float(get_value(row, "US517001:TAG/PV.CV"))
    if math.isnan(a):
        return math.nan
    return a * 1.5

# 2. Register
KPI_FUNCTIONS["My KPI"] = my_kpi

# 3. Add unit
KPI_UNITS["My KPI"] = "unit"
```

The column will automatically appear in the output CSV on next run.

---

## Stage 3 — CO Predictor (`co_product_model.py`)

`COPredictor` wraps three sklearn pipelines (Ridge, RandomForest, XGBoost) in a voting ensemble.

```python
from co_product_model import COPredictor, main as train_model

# Load saved model
predictor = COPredictor.load("co_predictor.pkl")

# Force retrain
predictor = train_model("Combined_Data_with_KPIs.csv")

# Real-time single-row prediction
result = predictor.predict_realtime(sensor_dict)
# result keys: co_ppm_predicted, co_ppm_low, co_ppm_high, alert_level, confidence, top_features
```

Alert thresholds: amber ≥ 5 ppm, red ≥ 10 ppm (CO product spec limit).

Training uses the KPI-enriched CSV as input; the model is saved to `co_predictor.pkl`.

---

## Stage 4 — Feature Engineering + Batch Prediction (`feature_engineering.py`)

### Physics features

`compute_physics_features(df)` appends five first-principles thermodynamic features and returns a new DataFrame. These are computed from the HTS outlet temperature DCS tag and syngas flow tags.

| Feature | Formula / source |
|---|---|
| `hts_outlet_temp_c` | HTS outlet °F → °C |
| `hts_k_eq` | WGS equilibrium constant: `exp(4577.8/T_K − 4.33)` [Moe 1962] |
| `approx_eq_co_pct` | Quadratic solve of WGS equilibrium for CO% at current T |
| `approach_to_eq` | CO Slip % / approx_eq_co_pct — **zero when CO Slip GC has no data** |
| `psa_space_vel_proxy` | Syngas flow / PSA H2 flow |

**Data gap note:** `approach_to_eq` is all zeros in this dataset because the syngas CO GC tag (`US517001:70AI_0275D/AI1/PV.CV`) records no data. `HTS Catalyst Utilization (%)` (= approach_to_eq × 100) inherits this gap and will also be zero until the analyzer starts recording.

### Batch prediction in main.py

`run_batch_prediction()` calls `compute_physics_features()` first, **persists the physics feature columns to the enriched DataFrame**, then calls `predictor.predict(X)` and appends `CO_Predicted_ppm` and `CO_Alert_Level`.

---

## Stage 5 — Compressor Reliability (`compressor_reliability.py`)

### Sensor map

`SENSOR_MAP` defines 26 friendly-named sensor columns per compressor (A, B, C). Examples:

```python
SENSOR_MAP["A"] = {
    "current":      "Compressor A Motor Current",
    "vib_motor_de": "Compressor A Motor DE Vibration",
    "bear_hot":     "Compressor A Hottest Bearing Temperature",
    "oil_dp":       "Compressor A Oil Filter dP",
    "cr_h2_1":      "Compressor A 1st Stage H2 Compression Ratio",
    ...
}
```

These friendly names exist **only** when the raw CSV is loaded with `pd.read_csv()` (file row 1 = friendly headers). They are absent in the enriched DataFrame which uses DCS tag strings.

### Training

```python
from compressor_reliability import main as train_monitor
monitor = train_monitor(Path("Combined_Data.csv"), Path("compressor_monitor.pkl"))
```

`load_data_for_training()` does a two-pass read:
1. `nrows=2` to detect which friendly columns are present.
2. Full read with `usecols` limited to those columns.

**Always pass the raw `Combined_Data.csv` path for training**, not the enriched CSV. `main.py` Stage 5 was fixed to do this:

```python
monitor = get_comp_monitor(comp_model_path, input_path, force_train=args.train)
#                                           ^^^^^^^^^^^ raw CSV path, not enriched df
```

### Scoring

```python
monitor = CompressorHealthMonitor.load("compressor_monitor.pkl")
df_scored = monitor.score_dataframe(df_with_friendly_cols)
```

`score_dataframe()` appends 8 columns per compressor (24 total):

| Column pattern | Type | Description |
|---|---|---|
| `Compressor_X_Health` | float 0–100 | Composite health index |
| `Compressor_X_Alert` | str | `green` / `amber` / `red` / `offline` |
| `Compressor_X_Bear_Score` | float 0–100 | Bearing thermal sub-score (weight 35%) |
| `Compressor_X_Vib_Score` | float 0–100 | Vibration sub-score (weight 25%) |
| `Compressor_X_Oil_Score` | float 0–100 | Oil system sub-score (weight 25%) |
| `Compressor_X_Cr_Score` | float 0–100 | Compression efficiency sub-score (weight 15%) |
| `Compressor_X_Anomaly` | int 0/1 | Isolation Forest anomaly flag |
| `Compressor_X_Anomaly_Score` | float −1 to 0 | IF score; closer to −1 = more anomalous |

Running thresholds (motor current): A > 100 A, B > 400 A, C > 200 A. Rows below threshold → `offline`, health columns → NaN.

**Compressor A** runs only ~15.7% of rows (mostly standby). Its health columns are sparse by design — not a data quality issue.

### Pickle deserialisation note

The `IFWrapper` class is defined inside `compressor_reliability.py`. When the pkl is loaded from a different `__main__` script, inject it first:

```python
import compressor_reliability as _cr, __main__
__main__.IFWrapper = _cr.IFWrapper
__main__.CompressorHealthMonitor = _cr.CompressorHealthMonitor
monitor = CompressorHealthMonitor.load("compressor_monitor.pkl")
```

---

## Stage 6 — Optimisation (`optimization.py`)

### Post-ML KPIs

`compute_opt_kpis(df)` requires `CO_Predicted_ppm` to be present and appends:

| Column | Formula |
|---|---|
| CO Spec Headroom (Predicted) | `10.0 − CO_Predicted_ppm` |
| HTS Catalyst Utilization (%) | `approach_to_eq × 100` — zero when CO Slip GC has no data |
| Efficiency Gap to Design (BTU/SCF) | `Gross Efficiency − 285.0` |
| Steam Cost Index | `(S/C / 2.7) × 100` — 100 = minimum safe steam |
| Steam Reduction Opportunity | `1` if `CO_pred < 8.0` AND `S/C > 3.0`, else `0` |

**Steam Reduction Opportunity is 0 in this dataset** because S/C is always 2.88–2.95, never exceeding the 3.0 threshold. This is correct — the plant is already operating near minimum safe steam.

### Trade-off analysis

```python
from optimization import OptimizationAnalyzer
analyzer = OptimizationAnalyzer(predictor)
analyzer.run_analysis(df)
```

Generates three plots saved to `model_plots/optimization/`:
- `01_tradeoff_curves.png` — CO vs S/C, HTS temp, plant rate, PSA recovery
- `02_operating_envelope.png` — 2D Plant Rate × S/C heat map coloured by predicted CO
- `03_opt_kpi_trends.png` — time series of optimisation KPIs

Also prints a ranked recommendation list to stdout.

---

## Stage 7 — Output CSV

### Structure

```
Combined_Data_with_KPIs.csv  — 1,206 columns after full pipeline run
```

| Column group | Count | Description |
|---|---|---|
| Raw DCS tag columns | ~1,080 | Original sensor data; column names are DCS tag strings |
| Original KPIs | 36 | Stage 2 engineering KPIs |
| Optimisation KPIs | 8 | Stage 2 new opt KPIs |
| CO_Predicted_ppm | 1 | Stage 4 ML output |
| CO_Alert_Level | 1 | Stage 4 alert string |
| Physics features | 5 | Stage 4: hts_k_eq, approx_eq_co_pct, approach_to_eq, psa_space_vel_proxy, hts_outlet_temp_c |
| Compressor health | 24 | Stage 5: 8 columns × 3 compressors |
| Post-ML opt KPIs | 5 | Stage 6: CO Spec Headroom (Predicted), HTS Catalyst Utilization, Efficiency Gap, Steam Cost Index, Steam Reduction Opportunity |

### Known column quirks

| Column | Quirk |
|---|---|
| PSA Recovery | Stored as fraction (0.726–0.803); multiply by 100 for display as % |
| CO in Product | Occasional small negative values (−0.1 ppm) from sensor noise; clip to 0 on display |
| CO Slip (Syngas GC) | All zeros in this dataset — Mass Spec CO GC not recording |
| approach_to_eq | All zeros — derived from CO Slip; will populate when GC data arrives |
| Net Efficiency | Clamped to [−500, 1000] BTU/SCF; returns NaN outside range to prevent blowup when denominator approaches zero |

---

## Patch Script (`_patch_enriched_csv.py`)

Used to add new columns to an existing enriched CSV **without re-running the full 10-minute KPI stage**. Run whenever new KPI functions, physics features, or model outputs need to be added to an already-generated CSV.

```bash
uv run python _patch_enriched_csv.py
```

Steps:
1. Load enriched CSV.
2. Apply any new `kpi_formulas` functions (skips columns already present).
3. Recompute physics features.
4. Load `compressor_monitor.pkl`; score from raw CSV; merge by row index.
5. Run `compute_opt_kpis()`.
6. Save updated CSV.

---

## Diagnostic Script (`_quick_diag.py`)

Checks all 46 dashboard-critical parameters for data coverage.

```bash
uv run python _quick_diag.py
```

Prints per-parameter: valid %, min, mean, max, status (OK / SPARSE / CONSTANT / MISSING).

---

## Common Issues

| Problem | Likely cause | Fix |
|---|---|---|
| Compressor health all `offline` | Scoring called on enriched CSV (DCS tag column names); friendly names not found | Always score from raw CSV or inject friendly-named columns first |
| `AttributeError: Can't get attribute 'IFWrapper' on __main__` | Pkl loaded from a script that isn't `compressor_reliability.py` | Inject `__main__.IFWrapper = _cr.IFWrapper` before calling `.load()` |
| Net Efficiency = −170 billion | Near-zero denominator (`bl_h2 − bl_coker_h2 × 24` ≈ 0) | Fixed in kpi_formulas.py with clamp guard; re-run pipeline or patch script |
| approach_to_eq = 0 | CO Slip GC tag `US517001:70AI_0275D/AI1/PV.CV` has no data | Data gap; not a code bug |
| All KPI columns NaN | Tag names in export don't match expected format | Add aliases to `COLUMN_ALIASES` in kpi_formulas.py |
| `UnicodeDecodeError` on read | Wrong encoding | Always use `encoding='latin-1'` |
| `Timestamp` all NaT | Format change in DCS export | Update `pd.to_datetime` format in `Data_Ingestion.py` |
| Stage 2 takes 10+ minutes | Row-by-row `df.apply()` on 20k rows × 44 KPIs | Expected; use patch script to avoid re-running when adding new columns |

---

## Architecture Notes

### Column naming convention

Raw sensor data → DCS tag strings (e.g. `US517001:12TI0633_S/ALM1/PV.CV`)  
KPI outputs → friendly names (e.g. `Gross Efficiency`)  
ML outputs → prefixed (e.g. `CO_Predicted_ppm`, `Compressor_A_Health`)

### Two-CSV design for compressor scoring

The enriched pipeline (Data_Ingestion → enriched df) loses friendly column names because Data_Ingestion promotes the DCS tag row (file row 3) as headers. The compressor reliability module needs friendly names to find sensor columns via `SENSOR_MAP`. The solution: always feed the raw CSV directly to `CompressorHealthMonitor` for both training and scoring; extract just the scored columns and merge into the enriched df.

### Row-by-row vs vectorized

`df.apply(kpi_func, axis=1)` is used throughout Stage 2 for simplicity and debuggability. For datasets significantly larger than 20k rows, vectorizing the most-used KPIs (Gross Efficiency, PSA Recovery, Plant Rate) with direct column arithmetic would reduce Stage 2 from ~10 minutes to ~30 seconds.

### Model persistence format

Both models use `joblib.dump` / `joblib.load`. The `IFWrapper` and `CompressorHealthMonitor` classes are pickled with a reference to their defining module. If the pkl is loaded from a non-`compressor_reliability` `__main__`, the class reference breaks. The injection pattern (`__main__.IFWrapper = _cr.IFWrapper`) resolves this without restructuring the modules.

### Dashboard parameter availability

After a full pipeline run (or patch script), the enriched CSV contains all parameters needed for all 6 dashboard pages. Parameters that currently show constant values due to data gaps (CO Slip GC, approach_to_eq, HTS Catalyst Utilization, Steam Reduction Opportunity) will auto-populate when the corresponding DCS tags start recording — no code changes required.
