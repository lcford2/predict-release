import argparse
import glob
import datetime
from typing import Callable
import pandas as pd
import numpy as np

def read_multiple_files_to_df(files: list, reader: Callable = pd.read_json) -> pd.DataFrame:
    dfs = [reader(file) for file in files]
    sizes = [df.shape[0] for df in dfs]
    res = []
    for file, size in zip(files, sizes):
        res.extend([file.split("/")[-1].split(".")[0]]*size)
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df["Reservoir"] = res
    return df

def load_data(location: str) -> pd.DataFrame:
    if location == "upper_col":
        data = pd.read_pickle("../upper_colorado_data/hydrodata_data/req_upper_col_data.pickle")
    elif location == "pnw":
        files = glob.glob(
            "../pnw_data/dam_data/*_data/*.json"
        )
        data = read_multiple_files_to_df(files)
    elif location == "lower_col":
        data = pd.read_pickle("../lower_col_data/lower_col_dam_data.pickle")
    elif location == "missouri":
        files = glob.glob(
            "../missouri_data/hydromet_data/*.json"
        )
        data = read_multiple_files_to_df(files)
    return data

def prep_data(df: pd.DataFrame, location: str) -> pd.DataFrame:
    if location == "upper_col":
        df[["storage_pre", "release_pre"]] = df.groupby(df.index.get_level_values(0))[
            ["storage", "release"]].shift(1)
        tmp = df.groupby(df.index.get_level_values(0))[
            ["storage_pre", "release_pre", "inflow"]].rolling(7).mean()
        tmp.index = df.index
        df[["storage_roll7", "release_roll7", "inflow_roll7"]] = tmp
    return df

def get_max_date_span(in_df: pd.DataFrame) -> tuple:
    df = pd.DataFrame()
    df["date"] = pd.to_datetime(in_df.index.get_level_values(1))
    df["mask"] = 1
    df.loc[df["date"] - datetime.timedelta(days=1) == df["date"].shift(), "mask"] = 0
    df["mask"] = df["mask"].cumsum()
    spans = df.loc[df["mask"] == df["mask"].value_counts().idxmax(), "date"]
    return (spans.min(), spans.max())

def get_max_res_date_spans(in_df: pd.DataFrame) -> pd.DataFrame:
    reservoirs = in_df.index.get_level_values(0).unique()
    spans = {r:{} for r in reservoirs}
    idx = pd.IndexSlice
    for res in reservoirs:
        span = get_max_date_span(
            in_df.loc[idx[res,:],:]
        )
        spans[res]["min"] = span[0]
        spans[res]["max"] = span[1]
    spans = pd.DataFrame.from_dict(spans).T
    spans["delta"] = spans["max"] - spans["min"]
    return spans.sort_values(by="delta")  

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--location", "-L", action="store", choices=["upper_col", "lower_col", "missouri", "pnw"],
                        help="Indicate which basin to load data for.")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    location = args.location
    data = load_data(location)
    data = prep_data(data, location)
    spans = get_max_res_date_spans(data)
    from IPython import embed as II
    II()

    