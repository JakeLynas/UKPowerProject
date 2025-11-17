from ..data.fetch_elexon import backfill_two_years, save_joined
from ..data.preprocessing_data import combine_series_data, run_preprocess_combined
import datetime
from pathlib import Path

def get_data():
    today_utc = datetime.date(2025, 11, 10)
    all_supply_forecast, all_demand_forecast, all_supply_actual, all_demand_actual = backfill_two_years(today_utc)

    save_joined(all_supply_actual, outdir="data/out/supply/actual")
    save_joined(all_demand_actual, outdir="data/out/demand/actual")
    save_joined(all_supply_forecast, outdir="data/out/supply/forecast")
    save_joined(all_demand_forecast, outdir="data/out/demand/forecast")

def prepare_data():
    #  We do not need a demand and supply model on a national scale as they will mirror each other closely.
    combined = combine_series_data("data/out", series_list=("demand",))
    print(combined.head())
    print(combined["series"].value_counts())
    print(combined["kind"].value_counts())

    # Save for easy reuse
    out_path = Path("data/processed/combined")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(f"{out_path}.parquet", index=False)
    combined.to_csv(f"{out_path}.csv", index=False)

    run_preprocess_combined()


if __name__ == "__main__":
    get_data()
    prepare_data()


#  Next is run some dundamnetal analysis to decide what features to use.