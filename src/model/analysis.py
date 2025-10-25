import pandas as pd
from typing import Tuple

# canonical names used across the app
STATE_COL = "state"
SECTOR_COL = "industry_sector"
EM_COL = "total_ghg_emissions_tonnes"
YEAR_COL = "reporting_year"

# mapping full state names -> USPS two-letter codes
US_STATE_TO_ABBR = {
    "Alabama": "AL","Alaska": "AK","Arizona": "AZ","Arkansas": "AR","California": "CA",
    "Colorado": "CO","Connecticut": "CT","Delaware": "DE","District of Columbia": "DC","Florida": "FL",
    "Georgia": "GA","Hawaii": "HI","Idaho": "ID","Illinois": "IL","Indiana": "IN",
    "Iowa": "IA","Kansas": "KS","Kentucky": "KY","Louisiana": "LA","Maine": "ME",
    "Maryland": "MD","Massachusetts": "MA","Michigan": "MI","Minnesota": "MN","Mississippi": "MS",
    "Missouri": "MO","Montana": "MT","Nebraska": "NE","Nevada": "NV","New Hampshire": "NH",
    "New Jersey": "NJ","New Mexico": "NM","New York": "NY","North Carolina": "NC","North Dakota": "ND",
    "Ohio": "OH","Oklahoma": "OK","Oregon": "OR","Pennsylvania": "PA","Rhode Island": "RI",
    "South Carolina": "SC","South Dakota": "SD","Tennessee": "TN","Texas": "TX","Utah": "UT",
    "Vermont": "VT","Virginia": "VA","Washington": "WA","West Virginia": "WV","Wisconsin": "WI",
    "Wyoming": "WY"
}
# also allow common abbreviations to pass through
ABBREV_SET = set(US_STATE_TO_ABBR.values())

def _to_state_abbr(val: str) -> str:
    if pd.isna(val):
        return None
    s = str(val).strip()
    if not s:
        return None
    # already abbrev?
    if len(s) == 2:
        return s.upper()
    # try title case match for full name
    key = s.title()
    if key in US_STATE_TO_ABBR:
        return US_STATE_TO_ABBR[key]
    # try upper (handles "DISTRICT OF COLUMBIA")
    key2 = s.upper()
    for fullname, abbr in US_STATE_TO_ABBR.items():
        if fullname.upper() == key2:
            return abbr
    return None

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to canonical ones used across the app.
    Does not modify values, only column names.
    """
    rename = {}
    # candidate lists for heuristics
    for canonical, candidates in {
        STATE_COL: ["state", "State", "STATE", "state_name", "StateName"],
        SECTOR_COL: ["industry_sector", "industry sector", "sector", "industry", "Industry"],
        EM_COL: ["total_ghg_emissions_tonnes", "total_ghg", "emissions", "ghg", "emission", "total_ghg_emissions"],
        YEAR_COL: ["reporting_year", "reporting year", "year", "Year", "report_year"]
    }.items():
        for cand in candidates:
            if cand in df.columns:
                rename[cand] = canonical
                break
    if rename:
        df = df.rename(columns=rename)
    return df

def _ensure_numeric_and_clean(df: pd.DataFrame, col: str) -> pd.Series:
    s = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return s

def top_sectors(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    df = ensure_columns(df)
    if SECTOR_COL not in df.columns or EM_COL not in df.columns:
        return pd.DataFrame(columns=[SECTOR_COL, EM_COL])
    df[EM_COL] = _ensure_numeric_and_clean(df, EM_COL)
    out = df.groupby(SECTOR_COL, dropna=True)[EM_COL].sum().reset_index()
    out = out.sort_values(by=EM_COL, ascending=False).head(top_n)
    return out

def emissions_by_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: state (US 2-letter code when possible) and total_ghg_emissions_tonnes.
    If a value can't be mapped to a two-letter code it will be dropped for the map, but kept in tabular output
    (with state left as original value).
    """
    df = ensure_columns(df)
    if STATE_COL not in df.columns or EM_COL not in df.columns:
        return pd.DataFrame(columns=[STATE_COL, EM_COL])
    df[EM_COL] = _ensure_numeric_and_clean(df, EM_COL)
    # preserve original state values
    df["_state_raw"] = df[STATE_COL].astype(str)
    # attempt to map to abbreviations
    df["_state_abbr"] = df["_state_raw"].apply(_to_state_abbr)
    # group by abbrev where available, otherwise by raw value
    # prefer abbrev grouping for map-friendly rows
    map_rows = df[df["_state_abbr"].notna()]
    table_rows = df.copy()
    mapped = map_rows.groupby("_state_abbr", dropna=True)[EM_COL].sum().reset_index().rename(columns={"_state_abbr": STATE_COL})
    # also produce a table grouped by raw values for completeness
    table = table_rows.groupby("_state_raw", dropna=True)[EM_COL].sum().reset_index().rename(columns={"_state_raw": STATE_COL})
    # attach both results in a dict-like struct by returning a dict? keep simple: return mapped (abbrs) with an attribute
    mapped = mapped.sort_values(by=EM_COL, ascending=False)
    # to keep backwards compatibility, return mapped (abbr) first; callers can use table if needed
    mapped.attrs["table_view"] = table
    return mapped

def yearly_trend(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    df = ensure_columns(df)
    if YEAR_COL not in df.columns or EM_COL not in df.columns:
        return pd.DataFrame(columns=[YEAR_COL, EM_COL]), YEAR_COL
    # ensure year is numeric-ish
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    df[EM_COL] = _ensure_numeric_and_clean(df, EM_COL)
    out = df.groupby(YEAR_COL, dropna=True)[EM_COL].sum().reset_index().sort_values(by=YEAR_COL)
    return out, YEAR_COL