import urllib.request
from urllib.parse import urlencode, urljoin
import json
import pandas as pd
import datetime
from .constants_elexon import Query, DataTypes
from .elexon_utils import iso_utc, each_day
from collections.abc import Callable
from pathlib import Path

def elexon_api(query: str, query_params: dict = {}) -> pd.DataFrame:
    """Call elxon api for specific query"""

    url = f"https://data.elexon.co.uk/{query}"
    if len(query_params):
        url = urljoin(url, "?" + urlencode(query_params))
    hdr = {'Cache-Control': 'no-cache'}

    print(url)
    req = urllib.request.Request(url, headers=hdr)
    response = urllib.request.urlopen(req)
    data = json.loads(response.read().decode("utf-8"))
    if isinstance(data, list):
        df = pd.DataFrame(data) 
    elif isinstance(data, dict):
    # If it's a dict of lists, this is fine, but you may want to check the structure
        if "data" in data:
            # If the data is nested under a key, extract it
            data = data["data"]
        df = pd.DataFrame.from_dict(data)
    else:
        raise ValueError("Unexpected data format from API")

    return df

def forecasted_supply(date_from: str, date_to:str) -> pd.DataFrame:
    """Query the forecasted supply between two published dates"""

    df = elexon_api(
        Query.PREDICTED_DATA.value,
        {
            "publishTime": str(date_to),
            }
        )

    # # Demand DataFrame
    # df_demand = df[["publishTime", "startTime", "indicatedDemand"]].copy()
    # df_demand.rename(columns={"indicatedDemand": "value"}, inplace=True)
    # df_demand["value"] = df_demand["value"].abs()  # Ensure demand is positive

    # Supply DataFrame
    df_supply = df[["publishTime", "startTime", "indicatedGeneration"]].copy()
    df_supply.rename(columns={"indicatedGeneration": "value"}, inplace=True)

    return df_supply

def actual_supply(date_from: datetime.date, date_to:datetime.date) -> pd.DataFrame:
    """Query the actual supply between two published dates"""

    df = elexon_api(
        Query.ACTUAL_SUPPLY.value,
        {
            "publishDateTimeFrom": str(date_from),
            "publishDateTimeTo": str(date_to),
            }
    )
    df_agg = df.groupby(["publishTime", "startTime"], as_index=False)["generation"].sum()
    df_agg.rename(columns={"generation": "value"}, inplace=True)

    return df_agg

def forecast_demand(date_from: datetime.date, date_to:datetime.date) -> pd.DataFrame:
    """Query the forecast demand between two Settlement dates"""

    df = elexon_api(
        Query.PREDICTED_DEMAND.value,
        {
            "settlementDateFrom": str(date_from),
            "settlementDateTo": str(date_to),
            }
        )

    df = df[["publishTime", "startTime", "nationalDemand"]]
    df.rename(columns={"nationalDemand": "value"}, inplace=True)

    return df

def actual_demand(date_from: datetime.date, date_to:datetime.date) -> pd.DataFrame:
    """Query the actual demand between two published dates"""

    df = elexon_api(
        Query.ACTUAL_DEMAND.value,
        {
            "settlementDateFrom": str(date_from),
            "settlementDateTo": str(date_to),
            }
        )

    df = df[["publishTime", "startTime", "initialTransmissionSystemDemandOutturn"]]
    df.rename(columns={"initialTransmissionSystemDemandOutturn": "value"}, inplace=True)

    return df


def fetch_eod_window(query_fn: Callable, pub_day: datetime.date,
                     window_start_hhmm="22:00:00", window_end_hhmm="23:59:59") -> pd.DataFrame:
    start_dt = datetime.datetime.combine(pub_day, datetime.time.fromisoformat(window_start_hhmm), tzinfo=datetime.timezone.utc)
    end_dt   = datetime.datetime.combine(pub_day, datetime.time.fromisoformat(window_end_hhmm), tzinfo=datetime.timezone.utc)
    df = query_fn(iso_utc(start_dt), iso_utc(end_dt))
    return df


def get_supply_data(target_day: datetime.date, query_fn: Callable) -> pd.DataFrame:
    # Get publications on the previous day, EOD
    pub_day = target_day - datetime.timedelta(days=1)
    df = fetch_eod_window(query_fn, pub_day)

    if df.empty:
        # widen the window as a fallback
        df = fetch_eod_window(query_fn, pub_day, "18:00:00", "23:59:59")
        if df.empty:
            return df

    # Keep only forecast rows whose startTime falls on the target day
    df = df[pd.to_datetime(df["startTime"], utc=True).dt.date == target_day]

    # Deduplicate per startTime: keep the last (latest publish)
    df = (df.sort_values("publishTime")
            .groupby("startTime", as_index=False)
            .tail(1))

    return df

def get_demand_data(day: datetime.date, query_fn: Callable) -> pd.DataFrame:
    df = query_fn(str(day), str(day))
    if df.empty:
        return df

    for col in ("publishTime","startTime"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    # Keep the last publication per startTime
    df = (df.sort_values("publishTime")
            .groupby("startTime", as_index=False)
            .tail(1))

    return df


def backfill_two_years(today_utc: datetime.date, function_pair: tuple[Callable, Callable], data_type: DataTypes) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = today_utc - datetime.timedelta(days=31)
    all_supply, all_demand = [], []

    for d in each_day(start, today_utc):
        # Day-ahead forecast for day d (published on d-1 EOD)
        supply = get_supply_data(d, function_pair[0])
        if not supply.empty:
            supply["data_type"] = data_type.value
            supply["horizon_hours"] = 24 if data_type == DataTypes.FORECAST else 0
            all_supply.append(supply)

        # Actual demand for day d (latest for each 30-min)
        demand = get_demand_data(d, function_pair[1])
        if not demand.empty:
            demand["data_type"] = data_type.value
            demand["horizon_hours"] = 24 if data_type == DataTypes.FORECAST else 0
            all_demand.append(demand)

    supply = pd.concat(all_supply, ignore_index=True) if all_supply else pd.DataFrame()
    demand = pd.concat(all_demand, ignore_index=True) if all_demand else pd.DataFrame()

    return supply, demand

    # Optional: left join forecast→actual on startTime for training pairs
    if not fc.empty and not ac.empty:
        pairs = (fc.merge(ac[["startTime","value"]]
                          .rename(columns={"value":"actual_value"}),
                          on="startTime", how="inner")
                   .rename(columns={"value":"forecast_value"}))
        return pairs, fc, ac
    return pd.DataFrame(), fc, ac

def test_fetch():
    forecast_functions = (forecasted_supply, forecast_demand, ) 
    actual_functions = (actual_supply, actual_demand)
    today_utc = datetime.date.today()
    start = today_utc - datetime.timedelta(days=7)

    supply, demand = backfill_two_years(start, actual_functions, DataTypes.FORECAST)

    save_joined(supply, outdir="data/test_out/supply/actual")
    save_joined(demand, outdir="data/test_out/demand/actual")



def save_joined(pairs: pd.DataFrame, outdir="data/out"):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    if not pairs.empty:
        pairs.to_parquet(Path(outdir)/"data.parquet", index=False)
        pairs.to_csv(Path(outdir)/"data.csv", index=False)
    else:
        print("No data to save")


if __name__ == "__main__":
    test_fetch()