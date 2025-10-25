import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from model.data_process import preprocess_data
from model.analysis import top_sectors, emissions_by_state, yearly_trend, EM_COL, SECTOR_COL, STATE_COL, YEAR_COL

st.set_page_config(page_title="GHG Explorer", layout="wide")
st.title("GHG Explorer — Top sectors, state choropleth, and yearly trends")
st.markdown("Upload a CSV or use the bundled sample (testdata/data.csv). Columns expected: facility_name, state, industry_sector, total_ghg_emissions_tonnes, latitude, longitude, reporting_year")

col1, col2 = st.columns([1, 3])
with col1:
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    path = st.text_input("Or local path (optional)", value="testdata/data.csv")
    top_n = st.slider("Top N sectors", min_value=3, max_value=20, value=10)
    st.markdown("---")
    st.write("Tip: For the map, two-letter US state codes (e.g. 'CA', 'NY') work best. Full state names will be mapped automatically where possible.")

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

# preview
st.subheader("Preview")
st.dataframe(df.head(10))

# compute aggregates
sectors_df = top_sectors(df, top_n=top_n)
states_mapped_df = emissions_by_state(df)  # mapped (abbr) df; full-table view in attrs
states_table_df = states_mapped_df.attrs.get("table_view", pd.DataFrame())
trend_df, year_col = yearly_trend(df)

# helper formatting
def fmt_int(v):
    try:
        return f"{int(round(v)):,.0f}"
    except Exception:
        return str(v)

# Top sectors: horizontal bar, big, with values
st.subheader(f"Top {top_n} Emitting Sectors")
if sectors_df.empty:
    st.warning("Could not detect sector/emissions columns.")
else:
    fig = px.bar(
        sectors_df.sort_values(by=EM_COL, ascending=True),
        x=EM_COL, y=SECTOR_COL, orientation="h",
        color=EM_COL, color_continuous_scale="Turbo",
        labels={SECTOR_COL: "Industry Sector", EM_COL: "Total GHG (tonnes)"},
        hover_data={EM_COL: ':.0f'},
    )
    fig.update_layout(template="plotly_white", margin=dict(l=220, r=20, t=50, b=50), height=600, showlegend=False)
    fig.update_traces(text=sectors_df[EM_COL].map(fmt_int).values, textposition="outside", cliponaxis=False)
    st.plotly_chart(fig, use_container_width=True)
    st.download_button("Download top sectors CSV", sectors_df.to_csv(index=False), "top_sectors.csv", "text/csv")

# Emissions by state: choropleth
st.subheader("Emissions by State — US Choropleth")
if states_mapped_df.empty:
    st.warning("Could not detect state/emissions columns or no mappable state codes.")
    if not states_table_df.empty:
        st.write("Tabular totals by raw state values:")
        st.dataframe(states_table_df.head(50))
else:
    # ensure 'state' column contains two-letter codes
    locations = states_mapped_df[STATE_COL]
    fig_map = px.choropleth(
        states_mapped_df,
        locations=STATE_COL,
        locationmode="USA-states",
        color=EM_COL,
        hover_name=STATE_COL,
        hover_data={EM_COL: ':.0f'},
        color_continuous_scale="Cividis",
        scope="usa",
        labels={EM_COL: "Total GHG (tonnes)"},
        title="Total GHG by State (hover for details)"
    )
    fig_map.update_layout(template="plotly_white", margin=dict(l=0, r=0, t=50, b=0), height=600)
    st.plotly_chart(fig_map, use_container_width=True)
    st.download_button("Download emissions by state CSV", states_mapped_df.to_csv(index=False), "emissions_by_state.csv", "text/csv")
    # also show top states table to the side
    with st.expander("Show top states table"):
        st.dataframe(states_mapped_df.head(50))

# Yearly trend: line + rolling average
st.markdown("---")
st.subheader("Yearly Trend")
if trend_df.empty:
    st.warning("Could not detect year/emissions columns.")
else:
    trend_df = trend_df.sort_values(by=year_col).reset_index(drop=True)
    trend_df["rolling3"] = trend_df[EM_COL].rolling(window=3, min_periods=1).mean()
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=trend_df[year_col], y=trend_df[EM_COL],
        mode="lines+markers", name="Annual total",
        line=dict(color="#1f77b4", width=2), marker=dict(size=8),
        hovertemplate="%{x}: %{y:,.0f} tonnes"
    ))
    fig_trend.add_trace(go.Scatter(
        x=trend_df[year_col], y=trend_df["rolling3"],
        mode="lines", name="3-yr rolling avg",
        line=dict(color="#ff7f0e", width=3, dash="dash"),
        hovertemplate="%{x}: %{y:,.0f} tonnes"
    ))
    fig_trend.update_layout(template="plotly_white", margin=dict(l=60, r=20, t=50, b=40),
                            xaxis_title="Year", yaxis_title="Total GHG (tonnes)",
                            yaxis=dict(tickformat=",.0f"))
    st.plotly_chart(fig_trend, use_container_width=True)
    st.download_button("Download yearly trend CSV", trend_df.to_csv(index=False), "yearly_trend.csv", "text/csv")