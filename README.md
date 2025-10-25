# MackathonChal2 — GHG Explorer

Local Streamlit app to explore greenhouse gas (GHG) emissions by industry sector, state, and year.

Features
- Top N emitting industry sectors (horizontal bar)
- Emissions by state (horizontal bar)
- Yearly trend with 3-year rolling average
- Download aggregated CSVs

Quick start (Linux)
1. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. Run the Streamlit app (development):
```bash
streamlit run src/streamlit_app.py
```
Open the URL printed by Streamlit (usually http://localhost:8501).

Using the app
- Upload a CSV or enter a local path (the repo includes `testdata/data.csv`).
- Expected columns (case-insensitive):  
  `facility_name`, `state`, `industry_sector`, `total_ghg_emissions_tonnes`, `latitude`, `longitude`, `reporting_year`
- The app auto-detects and normalizes common variants of these columns.

Project layout (important files)
- src/streamlit_app.py — main Streamlit UI
- src/model/data_process.py — file reading helpers (returns None on read errors)
- src/model/analysis.py — aggregation helpers (top sectors, by-state totals, yearly trend)
- testdata/data.csv — sample data used by default
- requirements.txt — dependencies (pandas, streamlit, plotly)

Notes & tips
- If preprocess_data returns None for a local path, check path and file permissions.
- For large datasets, consider increasing available memory or pre-aggregating data.
- For production, run behind a proper web server; Streamlit is intended for local/dev use.

Troubleshooting
- "Import 'pandas' could not be resolved": make sure the venv is activated and dependencies installed.
- If charts look clipped, resize the browser window or use the "wide" layout toggle in Streamlit.

License / Next steps
- Add LICENSE, tests, and CI as needed.
- Replace Plotly with Altair/Matplotlib if you prefer other visualization libs.