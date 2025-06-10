# Sample Data

`Combined_Data_with_KPIs.csv` is a **fully synthetic** dataset generated to demonstrate the dashboard without exposing real plant data.

- 8,640 rows — 30 days of 5-minute process data
- 81 columns — matching the exact schema produced by the ML pipeline
- Values are statistically realistic (correct units, ranges, autocorrelation) but contain no real operational information

To use real historian data, place your `Combined_Data_with_KPIs.csv` in the project root. The dashboard automatically prefers the root file over this sample.
