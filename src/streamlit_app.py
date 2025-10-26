import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path
import time

from model.data_process import (
    preprocess_data,
    create_random_forest,
    random_forest_predict_emissions,
    create_linear_regression,
    linear_predict_emissions,
    evaluate_model
)
from model.analysis import (
    top_sectors,
    emissions_by_state,
    yearly_trend,
    ensure_columns,
    EM_COL,
    SECTOR_COL,
    STATE_COL,
    YEAR_COL,
)

st.set_page_config(page_title="GHG Explorer", layout="wide")
st.title("GHG Explorer — Top sectors, choropleth, trends, and predictive models")
st.markdown(
    "Upload a CSV or use the bundled sample (`testdata/data.csv`).\n\n"
    "Required columns (any reasonable variant will be mapped): `facility_name`, `state` (abbr or full name), `industry_sector`, `total_ghg_emissions_tonnes`, `latitude`, `longitude`, `reporting_year`.\n\n"
    "The app will auto-train or load saved ML models (Random Forest and Linear Regression) on the loaded dataset and allow single-row predictions."
)

# Move uploader and basic controls to the sidebar for clearer main layout
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"]) 
path = st.sidebar.text_input("Or local path (optional)", value="testdata/data.csv")
top_n = st.sidebar.slider("Top N sectors", min_value=3, max_value=20, value=10)
st.sidebar.markdown("---")
st.sidebar.write("Tip: For the map, two-letter US state codes (e.g. 'CA', 'NY') work best — full state names are also accepted and will be mapped where possible.")
st.sidebar.write("Filter the dataset by year, sector, or state using the controls below. Use the Preview to confirm your uploaded file's columns and a small sample.")

df = None
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
elif path:
    df = preprocess_data(path)
    if df is None:
        st.warning("Could not read the file at the given path. Upload a CSV instead or check the path/permissions.")

if df is None:
    st.info("Awaiting CSV upload or valid path.")
    st.stop()

# normalize canonical column names and ensure numeric types
df = ensure_columns(df)
if YEAR_COL in df.columns:
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
if EM_COL in df.columns:
    df[EM_COL] = pd.to_numeric(df[EM_COL], errors="coerce").fillna(0)

# filters (in sidebar)
available_years = []
if YEAR_COL in df.columns:
    available_years = sorted(pd.unique(df[YEAR_COL].dropna()).astype(int).tolist())

st.sidebar.markdown("### Filters")
years_selected = st.sidebar.multiselect("Years", options=available_years, default=[y for y in [2021, 2022, 2023] if y in available_years])
# sectors for multiselect: take top candidates from data
try:
    sector_choices = top_sectors(df, top_n=50)[SECTOR_COL].tolist()
except Exception:
    sector_choices = []
sectors_selected = st.sidebar.multiselect("Sectors (choose to filter)", options=sector_choices, default=None)
# state choices
try:
    state_choices = emissions_by_state(df)[STATE_COL].tolist()
except Exception:
    state_choices = []
states_selected = st.sidebar.multiselect("States (2-letter) — optional", options=state_choices, default=None)

# apply filters
filtered_df = df.copy()
if years_selected:
    filtered_df = filtered_df[filtered_df[YEAR_COL].isin(years_selected)]
if sectors_selected:
    # some rows contain comma-separated sector lists; match any selected sector substring
    mask = pd.Series(False, index=filtered_df.index)
    for s in sectors_selected:
        mask = mask | filtered_df[SECTOR_COL].astype(str).str.contains(s, na=False)
    filtered_df = filtered_df[mask]
if states_selected:
    # use the canonical state column; allow matching either abbrev or full name
    filtered_df = filtered_df[filtered_df[STATE_COL].astype(str).isin(states_selected)]

# compact preview inside an expander so main charts stay prominent
with st.expander("Preview (first 10 rows)", expanded=False):
    st.dataframe(filtered_df.head(10))

# compute aggregates
sectors_df = top_sectors(filtered_df, top_n=top_n)
states_mapped_df = emissions_by_state(filtered_df)  # mapped (abbr) df; full-table view in attrs
# attrs['table_view'] is stored as a list-of-dicts for JSON safety; convert back to DataFrame for display
states_table_df = pd.DataFrame(states_mapped_df.attrs.get("table_view", []))
trend_df, year_col = yearly_trend(filtered_df)

# helper formatting
def fmt_int(v):
    try:
        return f"{int(round(v)):,.0f}"
    except Exception:
        return str(v)

# Make the Top sectors chart full-width to avoid being scrunched by the map
st.subheader(f"Top {top_n} Emitting Sectors")
if sectors_df.empty:
    st.warning("Could not detect sector/emissions columns.")
else:
    # compute percent of filtered total
    total_all = sectors_df[EM_COL].sum()
    sectors_df["pct_of_total"] = sectors_df[EM_COL] / total_all * 100
    sectors_df = sectors_df.sort_values(by=EM_COL, ascending=True)
    fig = px.bar(
        sectors_df,
        x=EM_COL, y=SECTOR_COL, orientation="h",
        color=EM_COL, color_continuous_scale="Turbo",
        labels={SECTOR_COL: "Industry Sector", EM_COL: "Total GHG (tonnes)"},
        hover_data={EM_COL: ':.0f', "pct_of_total": ':.2f'},
    )
    fig.update_layout(template="plotly_white", margin=dict(l=220, r=20, t=50, b=50), height=700, showlegend=False)
    # show absolute formatted and percent text
    fig.update_traces(text=sectors_df[EM_COL].map(fmt_int).values, textposition="outside", cliponaxis=False)
    st.plotly_chart(fig, use_container_width=True)
    st.download_button("Download top sectors CSV", sectors_df.to_csv(index=False), "top_sectors.csv", "text/csv")

# Place the map below the sectors chart, full width, with an expander for the table
st.subheader("Emissions by State — US Choropleth")
if states_mapped_df.empty:
    st.warning("Could not detect state/emissions columns or no mappable state codes.")
    if not states_table_df.empty:
        with st.expander("Tabular totals by raw state values"):
            st.dataframe(states_table_df.head(50))
else:
    # add a toggle for log scale (helpful when a few states dominate)
    log_scale = st.checkbox("Use log color scale (helps with skew)", value=False)
    color_vals = np.log1p(states_mapped_df[EM_COL]) if log_scale else states_mapped_df[EM_COL]
    plot_df = states_mapped_df.copy()
    plot_df["_color_val"] = color_vals
    fig_map = px.choropleth(
        plot_df,
        locations=STATE_COL,
        locationmode="USA-states",
        color="_color_val",
        hover_name=STATE_COL,
        hover_data={EM_COL: ':.0f'},
        color_continuous_scale="Cividis",
        scope="usa",
        labels={EM_COL: "Total GHG (tonnes)"},
        title="Total GHG by State (hover for details)"
    )
    if log_scale:
        fig_map.update_coloraxes(colorbar_title="log(tonnes+1)")
    fig_map.update_layout(template="plotly_white", margin=dict(l=0, r=0, t=50, b=0), height=600)
    st.plotly_chart(fig_map, use_container_width=True)
    st.download_button("Download emissions by state CSV", states_mapped_df.to_csv(index=False), "emissions_by_state.csv", "text/csv")
    with st.expander("Show top states table"):
        st.dataframe(states_mapped_df.head(50))

# Yearly trend: line + rolling average
st.markdown("---")
st.subheader("Yearly Trend")
if trend_df.empty:
    st.warning("Could not detect year/emissions columns.")
else:
    # allow user to drill into a sector or view overall
    sector_for_trend = st.selectbox("Show trend for (All or select a sector)", options=["All"] + sector_choices)
    trend_plot_df = trend_df.copy()
    if sector_for_trend != "All":
        # compute yearly totals for the selected sector
        mask = df[SECTOR_COL].astype(str).str.contains(sector_for_trend, na=False)
        trend_plot_df = df[mask].groupby(YEAR_COL, dropna=True)[EM_COL].sum().reset_index().sort_values(by=YEAR_COL)

    trend_plot_df = trend_plot_df.sort_values(by=year_col).reset_index(drop=True)
    trend_plot_df["rolling3"] = trend_plot_df[EM_COL].rolling(window=3, min_periods=1).mean()
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=trend_plot_df[year_col], y=trend_plot_df[EM_COL],
        mode="lines+markers", name="Annual total",
        line=dict(color="#1f77b4", width=2), marker=dict(size=8),
        hovertemplate="%{x}: %{y:,.0f} tonnes"
    ))
    fig_trend.add_trace(go.Scatter(
        x=trend_plot_df[year_col], y=trend_plot_df["rolling3"],
        mode="lines", name="3-yr rolling avg",
        line=dict(color="#ff7f0e", width=3, dash="dash"),
        hovertemplate="%{x}: %{y:,.0f} tonnes"
    ))
    fig_trend.update_layout(template="plotly_white", margin=dict(l=60, r=20, t=50, b=40),
                            xaxis_title="Year", yaxis_title="Total GHG (tonnes)",
                            yaxis=dict(tickformat=",.0f"))
    st.plotly_chart(fig_trend, use_container_width=True)
    st.download_button("Download yearly trend CSV", trend_plot_df.to_csv(index=False), "yearly_trend.csv", "text/csv")

# --- Predictive model UI -------------------------------------------------
st.markdown("---")
st.subheader("Predict Emissions — ML Model")
with st.expander("Train or view model (uses state, sector, year)", expanded=True):
    st.write(
        "This section auto-loads saved models from `outputs/` if present. If no saved model is found, the app will train models on the currently loaded dataset at startup and save them to `outputs/` for reuse.\n\n"
        "Predictions are illustrative — they show model outputs given the dataset and are not a substitute for a validated emissions model."
    )
    # Ensure session_state keys exist for both models
    if 'model_rf' not in st.session_state:
        st.session_state['model_rf'] = None
        st.session_state['mse_rf'] = None
        st.session_state['r2_rf'] = None
    if 'model_lr' not in st.session_state:
        st.session_state['model_lr'] = None
        st.session_state['mse_lr'] = None
        st.session_state['r2_lr'] = None

    def find_saved_model(directory: str = "outputs", prefix: str = None):
        """Return Path to the newest model file in outputs matching optional prefix if present (pkl, joblib, sav)."""
        p = Path(directory)
        if not p.exists() or not p.is_dir():
            return None
        # candidate extensions
        exts = ("*.pkl", "*.joblib", "*.sav", "*.pickle")
        files = []
        for e in exts:
            files.extend(p.glob(e))
        if not files:
            return None
        if prefix:
            files = [f for f in files if f.name.startswith(prefix)]
            if not files:
                return None
        # return the most recently modified
        files = sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)
        return files[0]

    # Handle Random Forest and Linear Regression models separately
    # RF
    saved_rf = find_saved_model("outputs", prefix="model_rf_")
    if saved_rf and st.session_state.get('model_rf') is None:
        try:
            with st.spinner(f"Loading saved Random Forest model from {saved_rf.name}..."):
                st.session_state['model_rf'] = joblib.load(saved_rf)
            # compute metrics on current dataset if possible
            try:
                mse_rf, r2_rf = evaluate_model(st.session_state['model_rf'], df)
                st.session_state['mse_rf'] = mse_rf
                st.session_state['r2_rf'] = r2_rf
            except Exception:
                st.session_state['mse_rf'] = None
                st.session_state['r2_rf'] = None
            st.info(f"Loaded Random Forest model: {saved_rf.name}")
        except Exception as e:
            st.warning(f"Found RF model at {saved_rf} but failed to load: {e}")
    elif saved_rf is None and st.session_state.get('model_rf') is None:
        # train RF at startup
        try:
            with st.spinner("No saved Random Forest found — training Random Forest (this may take a minute)..."):
                m_rf, mse_rf, r2_rf = create_random_forest(df)
            st.session_state['model_rf'] = m_rf
            st.session_state['mse_rf'] = mse_rf
            st.session_state['r2_rf'] = r2_rf
            outdir = Path("outputs")
            outdir.mkdir(parents=True, exist_ok=True)
            fname_rf = outdir / f"model_rf_{int(time.time())}.joblib"
            try:
                joblib.dump(m_rf, fname_rf)
                st.info(f"Trained Random Forest and saved to {fname_rf.name}")
            except Exception as e:
                st.warning(f"Trained Random Forest but failed to save to disk: {e}")
        except Exception as e:
            st.error(f"Automatic Random Forest training failed: {e}")

    # LR
    saved_lr = find_saved_model("outputs", prefix="model_lr_")
    if saved_lr and st.session_state.get('model_lr') is None:
        try:
            with st.spinner(f"Loading saved Linear Regression model from {saved_lr.name}..."):
                st.session_state['model_lr'] = joblib.load(saved_lr)
            # compute metrics on current dataset if possible
            try:
                mse_lr, r2_lr = evaluate_model(st.session_state['model_lr'], df)
                st.session_state['mse_lr'] = mse_lr
                st.session_state['r2_lr'] = r2_lr
            except Exception:
                st.session_state['mse_lr'] = None
                st.session_state['r2_lr'] = None
            st.info(f"Loaded Linear Regression model: {saved_lr.name}")
        except Exception as e:
            st.warning(f"Found LR model at {saved_lr} but failed to load: {e}")
    elif saved_lr is None and st.session_state.get('model_lr') is None:
        # train LR at startup
        try:
            with st.spinner("No saved Linear Regression found — training Linear Regression (fast)..."):
                m_lr, mse_lr, r2_lr = create_linear_regression(df)
            st.session_state['model_lr'] = m_lr
            st.session_state['mse_lr'] = mse_lr
            st.session_state['r2_lr'] = r2_lr
            outdir = Path("outputs")
            outdir.mkdir(parents=True, exist_ok=True)
            fname_lr = outdir / f"model_lr_{int(time.time())}.joblib"
            try:
                joblib.dump(m_lr, fname_lr)
                st.info(f"Trained Linear Regression and saved to {fname_lr.name}")
            except Exception as e:
                st.warning(f"Trained Linear Regression but failed to save to disk: {e}")
        except Exception as e:
            st.error(f"Automatic Linear Regression training failed: {e}")

    # provide a single Clear button to reset in-session models AND delete saved artifacts
    col_clear = st.columns([1])[0]
    if col_clear.button("Clear cache: delete saved models and retrain"):
        # Clear in-memory cached models/metrics
        st.session_state['model_rf'] = None
        st.session_state['mse_rf'] = None
        st.session_state['r2_rf'] = None
        st.session_state['model_lr'] = None
        st.session_state['mse_lr'] = None
        st.session_state['r2_lr'] = None

        # Delete saved model files from outputs/
        outdir = Path("outputs")
        deleted = 0
        if outdir.exists() and outdir.is_dir():
            patterns = [
                "model_rf_*.joblib", "model_lr_*.joblib",
                "model_rf_*.pkl", "model_lr_*.pkl",
                "model_rf_*.pickle", "model_lr_*.pickle",
                "model_rf_*.sav", "model_lr_*.sav",
            ]
            for pat in patterns:
                for f in outdir.glob(pat):
                    try:
                        f.unlink(missing_ok=True)
                        deleted += 1
                    except Exception:
                        # if a specific file can't be deleted, continue with others
                        pass
        st.info(f"Cleared cached models (deleted {deleted} file(s) in outputs). Retraining will run on reload.")
        # Force an immediate rerun so the auto-train logic executes now
        try:
            st.rerun()
        except Exception:
            # Fallback for older Streamlit versions
            st.experimental_rerun()

    # show metrics for available models
    if st.session_state.get('model_rf') is not None:
        try:
            if st.session_state.get('r2_rf') is not None:
                st.metric("Random Forest Test R²", f"{st.session_state['r2_rf']:.3f}")
            if st.session_state.get('mse_rf') is not None:
                st.write(f"Random Forest Test MSE: {st.session_state['mse_rf']:.2f}")
        except Exception:
            st.write("Random Forest model available in session.")
    if st.session_state.get('model_lr') is not None:
        try:
            if st.session_state.get('r2_lr') is not None:
                st.metric("Linear Regression Test R²", f"{st.session_state['r2_lr']:.3f}")
            if st.session_state.get('mse_lr') is not None:
                st.write(f"Linear Regression Test MSE: {st.session_state['mse_lr']:.2f}")
        except Exception:
            st.write("Linear Regression model available in session.")

    # user inputs for prediction
    st.markdown("**Single prediction**")
    # provide helpful defaults from the loaded dataset
    default_state = state_choices[0] if state_choices else ""
    default_sector = sector_choices[0] if sector_choices else ""
    default_year = available_years[0] if available_years else 2024

    # Combo-like selectbox for state: allow choosing from detected states or typing a custom value
    state_options = ["Custom"] + state_choices if state_choices else ["Custom"]
    state_default_idx = 1 if (state_choices and default_state in state_choices) else 0
    state_sel = st.selectbox("State (choose or type custom)", options=state_options, index=state_default_idx)
    if state_sel == "Custom":
        state_input = st.text_input("State (2-letter or full name)", value=(default_state if default_state and default_state not in state_choices else ""))
    else:
        state_input = state_sel

    # Combo-like selectbox for sector: choose from detected sectors or type custom
    sector_options = ["Custom"] + sector_choices if sector_choices else ["Custom"]
    sector_default_idx = 1 if (sector_choices and default_sector in sector_choices) else 0
    sector_sel = st.selectbox("Industry sector (choose or type custom)", options=sector_options, index=sector_default_idx)
    if sector_sel == "Custom":
        sector_input = st.text_input("Industry sector", value=(default_sector if default_sector and default_sector not in sector_choices else ""))
    else:
        sector_input = sector_sel

    year_input = st.number_input("Reporting year", min_value=1900, max_value=2100, value=int(default_year))

    st.write("Predictions from both models (if available) will be shown below. Click 'Run predictions' to compute outputs for the chosen inputs.")

    if st.button("Run predictions"):
        preds = []
        # Random Forest prediction
        if st.session_state.get('model_rf') is not None:
            try:
                pr = random_forest_predict_emissions(st.session_state['model_rf'], state_input, sector_input, int(year_input), df)
                preds.append(("Random Forest", pr))
            except Exception as e:
                preds.append(("Random Forest", f"Error: {e}"))
        else:
            preds.append(("Random Forest", "Model not available"))

        # Linear Regression prediction
        if st.session_state.get('model_lr') is not None:
            try:
                pl = linear_predict_emissions(st.session_state['model_lr'], state_input, sector_input, int(year_input), df)
                preds.append(("Linear Regression", pl))
            except Exception as e:
                preds.append(("Linear Regression", f"Error: {e}"))
        else:
            preds.append(("Linear Regression", "Model not available"))

        # display results side-by-side when both present
        cols = st.columns(len(preds))
        for (name, val), col in zip(preds, cols):
            if isinstance(val, (int, float, np.floating, np.integer)):
                col.metric(name, f"{val:,.0f} tonnes")
            else:
                col.write(f"{name}: {val}")
        st.write("Note: predictions are illustrative and depend on how the models were trained on the current dataset.")

# Emissions change 2021 -> 2023 (if years present)
st.markdown("---")
st.subheader("Change 2021 → 2023")
if 2021 in available_years and 2023 in available_years:
    aggby = st.selectbox("Group change by", options=["Overall", "State", "Sector"], index=0)
    if aggby == "Overall":
        tot21 = df[df[YEAR_COL] == 2021][EM_COL].sum()
        tot23 = df[df[YEAR_COL] == 2023][EM_COL].sum()
        delta = tot23 - tot21
        pct = (delta / tot21 * 100) if tot21 else np.nan
        st.metric("Total change 2021 → 2023 (tonnes)", fmt_int(delta), delta=fmt_int(delta))
        st.write(f"Percent change: {pct:.2f}%")
    else:
        if aggby == "State":
            g21 = df[df[YEAR_COL] == 2021].groupby(STATE_COL, dropna=True)[EM_COL].sum()
            g23 = df[df[YEAR_COL] == 2023].groupby(STATE_COL, dropna=True)[EM_COL].sum()
        else:
            g21 = df[df[YEAR_COL] == 2021].groupby(SECTOR_COL, dropna=True)[EM_COL].sum()
            g23 = df[df[YEAR_COL] == 2023].groupby(SECTOR_COL, dropna=True)[EM_COL].sum()
        change = (g23 - g21).fillna(0).sort_values(ascending=False).head(10).reset_index().rename(columns={0: EM_COL})
        change.columns = [aggby, "change"]
        fig_ch = px.bar(change, x="change", y=aggby, orientation="h", labels={"change": "Change (tonnes)"})
        fig_ch.update_layout(template="plotly_white", height=500, margin=dict(l=200))
        st.plotly_chart(fig_ch, use_container_width=True)
else:
    st.info("Both 2021 and 2023 must be present in the dataset to compute the change."
            " Use the year filters or upload a dataset covering those years.")

# Outlier detection by facility
st.markdown("---")
st.subheader("Identify Outlier Facilities")
with st.expander("Outlier detection settings (z-score or IQR)"):
    outlier_method = st.selectbox("Method", options=["Z-score", "IQR"], index=0)
    if outlier_method == "Z-score":
        z_thresh = st.slider("Z-score threshold", min_value=1.0, max_value=5.0, value=3.0)
    else:
        iqr_mult = st.slider("IQR multiplier (k)", min_value=1.0, max_value=5.0, value=1.5)
    top_fac_n = st.number_input("Show top N outlier facilities", min_value=5, max_value=200, value=20)

fac_agg = filtered_df.groupby("facility_name", dropna=True, as_index=False)[[EM_COL, "latitude", "longitude"]].agg({EM_COL: "sum", "latitude": "first", "longitude": "first"})
fac_agg = fac_agg.sort_values(by=EM_COL, ascending=False).reset_index(drop=True)
if fac_agg.empty:
    st.info("No facility-level data available for outlier detection (missing facility name or emissions).")
else:
    if outlier_method == "Z-score":
        mu = fac_agg[EM_COL].mean()
        sigma = fac_agg[EM_COL].std(ddof=0) if fac_agg[EM_COL].std(ddof=0) != 0 else 1.0
        fac_agg["z"] = (fac_agg[EM_COL] - mu) / sigma
        outliers = fac_agg[fac_agg["z"].abs() >= z_thresh].sort_values(by="z", ascending=False)
    else:
        q1 = fac_agg[EM_COL].quantile(0.25)
        q3 = fac_agg[EM_COL].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_mult * iqr
        upper = q3 + iqr_mult * iqr
        outliers = fac_agg[(fac_agg[EM_COL] < lower) | (fac_agg[EM_COL] > upper)].sort_values(by=EM_COL, ascending=False)

    st.write(f"Found {len(outliers)} outlier facilities (matching current filters)")
    if not outliers.empty:
        st.dataframe(outliers.head(top_fac_n))
        # map outliers
        has_geo = outliers["latitude"].notna().any() and outliers["longitude"].notna().any()
        if has_geo:
            map_df = outliers.dropna(subset=["latitude", "longitude"]) 
            fig_map_out = px.scatter_geo(map_df, lat="latitude", lon="longitude",
                                         hover_name="facility_name", size=EM_COL,
                                         projection="natural earth", scope="usa",
                                         title="Outlier Facilities (size ~ emissions)")
            fig_map_out.update_layout(template="plotly_white", height=600)
            st.plotly_chart(fig_map_out, use_container_width=True)
        else:
            st.info("Outliers found but no latitude/longitude available to map them.")

# BEGIN Inserted: Drivers of Emissions analysis & visuals
st.markdown("---")
st.subheader("Drivers of Emissions — contribution, Pareto, and top contributors")

# Sector contribution and Pareto
if not sectors_df.empty:
    sec = sectors_df[[SECTOR_COL, EM_COL]].copy()
    sec = sec.sort_values(by=EM_COL, ascending=False).reset_index(drop=True)
    sec["pct"] = sec[EM_COL] / sec[EM_COL].sum() * 100
    sec["cumulative_pct"] = sec["pct"].cumsum()

    top10_share = sec.head(10)[EM_COL].sum() / sec[EM_COL].sum() * 100

    st.metric("Top 10 sectors share", f"{top10_share:.1f}%")
    st.write("Top sectors (by total)")
    st.dataframe(sec.head(10).assign(**{EM_COL: sec[EM_COL].map(fmt_int)}))

    # Pareto chart: bars + cumulative line
    pareto = go.Figure()
    pareto.add_trace(go.Bar(x=sec[SECTOR_COL], y=sec[EM_COL], name="Emissions", marker_color=px.colors.sequential.Turbo))
    pareto.add_trace(go.Scatter(x=sec[SECTOR_COL], y=sec["cumulative_pct"], name="Cumulative %", yaxis="y2", mode="lines+markers", line=dict(color="#ff7f0e")))
    pareto.update_layout(
        title="Sector Pareto — contributions and cumulative share",
        xaxis_tickangle=-45,
        yaxis=dict(title="Total GHG (tonnes)", tickformat=","),
        yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 110]),
        template="plotly_white",
        margin=dict(l=80, r=80, t=50, b=160),
        height=520,
    )
    st.plotly_chart(pareto, use_container_width=True)

# State contribution and Pareto
states_for_pareto = states_mapped_df.copy()
if not states_for_pareto.empty:
    st.write("Top states by contribution")
    st.dataframe(states_for_pareto.head(10).assign(**{EM_COL: states_for_pareto[EM_COL].map(fmt_int)}))

    s = states_for_pareto[[STATE_COL, EM_COL]].sort_values(by=EM_COL, ascending=False).reset_index(drop=True)
    s["pct"] = s[EM_COL] / s[EM_COL].sum() * 100
    s["cumulative_pct"] = s["pct"].cumsum()

    pareto_states = go.Figure()
    pareto_states.add_trace(go.Bar(x=s[STATE_COL], y=s[EM_COL], name="Emissions", marker_color=px.colors.sequential.Cividis))
    pareto_states.add_trace(go.Scatter(x=s[STATE_COL], y=s["cumulative_pct"], name="Cumulative %", yaxis="y2", mode="lines+markers", line=dict(color="#2ca02c")))
    pareto_states.update_layout(
        title="State Pareto — contributions and cumulative share",
        xaxis_tickangle=-45,
        yaxis=dict(title="Total GHG (tonnes)"),
        yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 110]),
        template="plotly_white",
        margin=dict(l=80, r=80, t=50, b=150),
        height=480,
    )
    st.plotly_chart(pareto_states, use_container_width=True)

# Top state x sector contributors and treemap / heatmap
ss = filtered_df.groupby([STATE_COL, SECTOR_COL], dropna=True)[EM_COL].sum().reset_index()
if not ss.empty:
    top_pairs = ss.sort_values(by=EM_COL, ascending=False).head(50).reset_index(drop=True)
    st.subheader("Top State × Sector contributors")
    st.write("Top combinations by total emissions")
    st.dataframe(top_pairs.head(20).assign(**{EM_COL: top_pairs[EM_COL].map(fmt_int)}))

    # Treemap for composition
    treemap = px.treemap(top_pairs, path=[STATE_COL, SECTOR_COL], values=EM_COL,
                         color=EM_COL, color_continuous_scale="Viridis",
                         title="Treemap: State → Sector contributions")
    treemap.update_layout(margin=dict(t=50, l=10, r=10, b=10), height=600)
    st.plotly_chart(treemap, use_container_width=True)

    # Pivot heatmap for a compact view (states x sectors)
    pivot = top_pairs.pivot_table(index=STATE_COL, columns=SECTOR_COL, values=EM_COL, fill_value=0)
    # keep only top N states/sectors for readability
    top_state_idx = pivot.sum(axis=1).sort_values(ascending=False).head(12).index
    top_sector_idx = pivot.sum(axis=0).sort_values(ascending=False).head(12).index
    pivot_small = pivot.loc[top_state_idx, top_sector_idx]
    if not pivot_small.empty:
        hm = px.imshow(pivot_small, labels=dict(x="Sector", y="State", color="Emissions"),
                       color_continuous_scale="Cividis", aspect="auto",
                       title="Heatmap: emissions by State (rows) × Sector (cols)")
        hm.update_layout(margin=dict(l=80, r=20, t=50, b=120), height=600)
        st.plotly_chart(hm, use_container_width=True)
# END Inserted: Drivers of Emissions analysis & visuals