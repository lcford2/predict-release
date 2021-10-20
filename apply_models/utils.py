import argparse
import glob
import datetime
import pathlib
import pandas as pd
import numpy as np
from typing import Callable, Optional, Union
from IPython import embed as II
try:
    import cudf as cd
    import cupy as cp
    USE_GPU = True
except ImportError as e:
    USE_GPU = False


def read_multiple_files_to_df(files: list, reader: Callable = pd.read_csv, 
                              reader_args: Optional[dict] = None) -> pd.DataFrame:
    reader_args = {} if not reader_args else reader_args
    dfs = [reader(file, **reader_args) for file in files]
    sizes = [df.shape[0] for df in dfs]
    res = []
    for file, size in zip(files, sizes):
        fpath = pathlib.Path(file)
        res.extend([fpath.stem]*size)
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df["Reservoir"] = res
    return df

def load_data(location: str, use_gpu: bool=False) -> pd.DataFrame:
    if use_gpu:
        reader = cd.read_csv
    else:
        reader = pd.read_csv
    reader_args = {"index_col":0}

    if location == "upper_col":
        data = reader("../upper_colorado_data/hydrodata_data/req_upper_col_data.csv", **reader_args)
    elif location == "pnw":
        files = glob.glob(
            "../pnw_data/dam_data/*_data/*.csv"
        )
        data = read_multiple_files_to_df(files, reader=reader, reader_args=reader_args)
    elif location == "lower_col":
        data = reader("../lower_col_data/lower_col_dam_data.csv", **reader_args)
    elif location == "missouri":
        files = glob.glob(
            "../missouri_data/hydromet_data/*.csv"
        )
        data = read_multiple_files_to_df(files, reader=reader, reader_args=reader_args)
    return data

def set_proper_index(df: pd.DataFrame, location: str, use_gpu=False) -> pd.DataFrame:
    if use_gpu:
        mindex = cd.MultiIndex.from_frame
    else:
        mindex = pd.MultiIndex.from_frame
    if location == "upper_col":
        index = mindex(df.loc[:,["site_name", "datetime"]])
        df.index = index
        df = df.drop(["site_name", "datetime"], axis=1)
    else:
        raise NotImplementedError(f"Cannot set index for location {location}")
    return df

def prep_data(df: pd.DataFrame, location: str, use_gpu=False) -> pd.DataFrame:
    nan = cp.nan if use_gpu else np.nan
    if location == "upper_col":
        shifted = df.groupby(df.index.get_level_values(0))[
            ["storage", "release"]].shift(1)
        df["release_pre"] = shifted["release"]
        df["storage_pre"] = shifted["storage"]
        df = df.fillna(nan)
        tmp = df.groupby(df.index.get_level_values(0))[
            ["storage_pre", "release_pre", "inflow"]].rolling(7).mean()
        tmp.index = df.index
        df[["storage_roll7", "release_roll7", "inflow_roll7"]] = tmp
    else:
        raise NotImplementedError(f"Cannot prep data for location {location}")
    return df

def get_max_date_span(in_df: pd.DataFrame, use_gpu: bool =False) -> tuple:
    if use_gpu:
        df = cd.DataFrame()
        dates = cd.to_datetime(in_df.index.get_level_values(1))
    else:
        df = pd.DataFrame()
        dates = pd.to_datetime(in_df.index.get_level_values(1))
    df["date"] = dates
    df["mask"] = 1
    df.loc[df["date"] - datetime.timedelta(days=1) == df["date"].shift(), "mask"] = 0
    df["mask"] = df["mask"].cumsum()
    spans = df.loc[df["mask"] == df["mask"].value_counts().idxmax(), "date"]
    return (spans.min(), spans.max())

def get_max_res_date_spans(in_df: pd.DataFrame, use_gpu: bool=False) -> pd.DataFrame:
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

    parser.add_argument("location", action="store", choices=["upper_col", "lower_col", "missouri", "pnw"],
                        help="Indicate which basin to load data for.")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    location = args.location
    USE_GPU = False
    data = load_data(location, use_gpu=USE_GPU)
    data = set_proper_index(data, location, use_gpu=USE_GPU)
    data = prep_data(data, location, use_gpu=USE_GPU)
    spans = get_max_res_date_spans(data, use_gpu=USE_GPU)
    II()

    