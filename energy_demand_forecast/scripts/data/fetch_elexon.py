import urllib.request
from urllib.parse import urlencode, urljoin
import json
import pandas as pd
import datetime
from .constants_elexon import Query, DataTypes
from .elexon_utils import iso_utc, each_day
from collections.abc import Callable
from pathlib import Path
import logging
from typing import Optional

def elexon_api(query: str, query_params: dict) -> pd.DataFrame:
    """Call elxon api for specific query"""
    url = f"https://data.elexon.co.uk/{query}"
    print(query_params)
    if len(query_params):
        url = url + "?" + urlencode(query_params)
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
    print(f"Querying actual supply from {date_from} to {date_to}")
    df = elexon_api(
        Query.ACTUAL_SUPPLY.value,
        {
            "settlementDateFrom": str(date_from),
            "settlementDateTo": str(date_to),
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
            "publishTime": str(date_to),
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


def get_predicted_data(target_day: datetime.date, query_fn: Callable) -> pd.DataFrame:
    # Get publications on the previous day, EOD
    pub_day = target_day - datetime.timedelta(days=1)
    df = fetch_eod_window(query_fn, pub_day)

    if df.empty:
        # widen the window as a fallback
        df = fetch_eod_window(query_fn, pub_day, "18:00:00", "23:59:59")
        if df.empty:
            return df
        
    df = df[pd.to_datetime(df["startTime"], utc=True).dt.date == target_day]

    df = (df.sort_values("publishTime")
            .groupby("startTime", as_index=False)
            .tail(1))

    return df

def get_actual_data(day: datetime.date, query_fn: Callable) -> pd.DataFrame:
    df = query_fn(str(day), str(day))
    if df.empty:
        return df

    for col in ("publishTime","startTime"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")


    df = (df.sort_values("publishTime")
            .groupby("startTime", as_index=False)
            .tail(1))

    return df


def backfill_two_years(today_utc: datetime.date, days: Optional[int] = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    start = today_utc - datetime.timedelta(days=days or 365*2)
    all_supply_forecast, all_demand_forecast = [], []
    all_supply_actual, all_demand_actual = [], []

    for d in each_day(start, today_utc):

        forecast_demand_data = get_predicted_data(d, forecast_demand)
        actual_demand_data = get_actual_data(d, actual_demand)

        forecasted_supply_data = get_predicted_data(d, forecasted_supply)
        actual_supply_data = get_actual_data(d, actual_supply)
            
        all_supply_forecast.append(forecasted_supply_data)
        all_demand_forecast.append(forecast_demand_data)    
        all_supply_actual.append(actual_supply_data)
        all_demand_actual.append(actual_demand_data)


    all_supply_forecast = pd.concat(all_supply_forecast, ignore_index=True) if all_supply_forecast else pd.DataFrame()
    all_demand_forecast = pd.concat(all_demand_forecast, ignore_index=True) if all_demand_forecast else pd.DataFrame()
    all_supply_actual = pd.concat(all_supply_actual, ignore_index=True) if all_supply_actual else pd.DataFrame()
    all_demand_actual = pd.concat(all_demand_actual, ignore_index=True) if all_demand_actual else pd.DataFrame()

    print(f"Fetched {len(all_supply_forecast)} forecast supply rows, {len(all_demand_forecast)} forecast demand rows")
    print(f"Fetched {len(all_supply_actual)} actual supply rows, {len(all_demand_actual)} actual demand rows")

    return all_supply_forecast, all_demand_forecast, all_supply_actual, all_demand_actual


def test_fetch():

    today_utc = datetime.date(2025,11,10)
    start = today_utc - datetime.timedelta(days=7)
    all_supply_forecast, all_demand_forecast, all_supply_actual, all_demand_actual = backfill_two_years(start, days=31)
    save_joined(all_supply_actual, outdir="data/test_out/supply/actual")
    save_joined(all_demand_actual, outdir="data/test_out/demand/actual")
    save_joined(all_supply_forecast, outdir="data/test_out/supply/forecast")
    save_joined(all_demand_forecast, outdir="data/test_out/demand/forecast")



def save_joined(pairs: pd.DataFrame, outdir="data/out"):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    if not pairs.empty:
        pairs.to_parquet(Path(outdir)/"data.parquet", index=False)
        pairs.to_csv(Path(outdir)/"data.csv", index=False)
    else:
        print("No data to save")


if __name__ == "__main__":
    test_fetch()