import argparse
import glob
import datetime
import pathlib
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from typing import Callable, Optional, Union, Iterable
from datetime import timedelta
from IPython import embed as II
try:
    import cudf as cd
    import cupy as cp
    USE_GPU = True
except ImportError as e:
    USE_GPU = False


DATA_LOCS = {
    "upper_col":{
        "ready":"../upper_colorado_data/model_ready_data/upper_col_data.csv",
        "raw":"../upper_colorado_data/hydrodata_data/req_upper_col_data.csv",
    },
    "pnw":{
        "ready":"../pnw_data/model_ready_data/pnw_data.csv",
        "raw":"../pnw_data/dam_data/*_data/*.csv",
    },
    "lower_col":{
        "ready":"../lower_col_data/model_ready_data/lower_col_data.csv",
        "raw":"../lower_col_data/lower_col_dam_data.csv",
    },
    "missouri":{
        "ready":"../missouri_data/model_ready_data/missouri_data.csv",
        "raw":"../missouri_data/hydromet_data/*.csv",
    }
}


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
    raw_reader_args = {"index_col":0}
    ready_reader_args = {"index_col":[0,1]}

    if location in ["upper_col", "lower_col"]:
        ready_path = pathlib.Path(DATA_LOCS[location]["ready"])
        if ready_path.exists():
            data = reader(ready_path.as_posix(), **ready_reader_args)
            needs_format = False
        else:
            data = reader(DATA_LOCS[location]["raw"], **raw_reader_args)
            needs_format = True
    elif location in ["pnw", "missouri"]:
        ready_path = pathlib.Path(DATA_LOCS[location]["ready"])
        if ready_path.exists():
            data = reader(ready_path.as_posix(), **ready_reader_args)
            needs_format = False
        else:
            files = glob.glob(
                DATA_LOCS[location]["raw"]
            )
            data = read_multiple_files_to_df(files, reader=reader)
            needs_format = True
    else:
        raise NotImplementedError(f"No data available for location {location}")
    return data, needs_format

def get_valid_entries(df: pd.DataFrame, location: str) -> pd.DataFrame:
    if location == "upper_col":
        good = ~df.isna()
        # get entries that have the required data in some form
        return df.loc[(good["release volume"] | good["total release"]) & (good["storage"])]
    elif location == "pnw":
        # only possible ways data is available
        df = df.loc[:, ["Reservoir","DateTime","Inflow_cfs", "Release_cfs", "Storage_acft"]]
        # so we can just filter out rows that do not have all of this information
        return df.loc[(~df.isna()).any(axis=1)]
    elif location == "missouri":
        df = df.loc[:, ["Reservoir", "DATE", "IN", "QD", "AF"]]
        return df.loc[(~df.isna()).any(axis=1)]
    elif location == "lower_col":
        # hoover -> davis -> parker -> 
        keepers = ["Hoover_rel_cfs", "Davis_rel_cfs", "Parker_rel_cfs", 
                   "Hoover_stor_KAF", "Davis_stor_KAF", "Parker_stor_KAF",
                   "Hoover_inflow_cfs" ]
        df = df.loc[:,keepers]
        df["Davis_inflow_cfs"] = df["Hoover_rel_cfs"]
        df["Parker_inflow_cfs"] = df["Davis_rel_cfs"]
        df = df.melt(ignore_index=False)
        df = df.reset_index()
        df[["Reservoir", "ResVar", "Unit"]] = df["variable"].str.split("_", expand=True)
        df.loc[df["Unit"] == "KAF", "value"] *= 1000
        df = df.drop(["variable", "Unit"], axis=1)
        df = df.pivot(index=["Reservoir", "Day"], columns=["ResVar"])
        df.columns = df.columns.droplevel(0)
        return df.reset_index()
    else:
        raise NotImplementedError(f"Cannot get valid entries for location {location}")

def rename_columns(df: pd.DataFrame, location: str) -> pd.DataFrame:
    if location == "upper_col":
        return df
    elif location == "pnw":
        columns = {"Reservoir":"site_name", "DateTime":"datetime", "Inflow_cfs":"inflow", 
                   "Release_cfs":"release", "Storage_acft":"storage"}
        return df.rename(columns=columns)
    elif location == "missouri":
        columns = {"Reservoir":"site_name", "DATE":"datetime", "IN":"inflow",
                    "QD":"release","AF":"storage"}
        return df.rename(columns=columns)
    elif location == "lower_col":
        columns = {"Reservoir":"site_name", "Day":"datetime", "rel":"release", "stor":"storage"}
        return df.rename(columns=columns)
    else:
        raise NotImplementedError(f"Cannot rename columns for location {location}")

def set_proper_index(df: pd.DataFrame, location: str, use_gpu=False) -> pd.DataFrame:
    if use_gpu:
        mindex = cd.MultiIndex.from_frame
    else:
        mindex = pd.MultiIndex.from_frame
    if location in ["upper_col", "pnw", "missouri", "lower_col"]:
        dt_values = pd.to_datetime(df.loc[:, "datetime"])
        if location == "pnw":
            # some are marked as being recorded at the end of the previous day (i.e. hour = 23)
            # (e.g. records for 2000-10-23 00:00:00 and 2000-10-23 23:00:00 but none for 2000-10-24)
            # so I need to move the day to the next one for the span checker to work properly
            # dt_values = dt_values.apply(lambda x: x+timedelta(hours=1) if x.hour == 23 else x)
            # the above does not check if there are entries the next day that are the same
            # so while this is slower, it will not replace data that exists
            dt_values = move_dt_one_hour(dt_values)
        df.loc[:,"datetime"] = dt_values
        index = mindex(df.loc[:,["site_name", "datetime"]])
        df.index = index
        df = df.drop(["site_name", "datetime"], axis=1)
    else:
        raise NotImplementedError(f"Cannot set index for location {location}")
    return df

def move_dt_one_hour(dt_values):
    ret_vals = dt_values.values
    size = dt_values.size
    for i, x in enumerate(dt_values):
        if x.hour == 23:
            new_x = x+timedelta(hours=1)
            if i < size - 1:
                if new_x != ret_vals[i+1]:
                    ret_vals[i] = new_x
    return ret_vals

def prep_data(df: pd.DataFrame, location: str, use_gpu=False) -> pd.DataFrame:
    nan = cp.nan if use_gpu else np.nan
    if location in ["upper_col", "pnw", "missouri", "lower_col"]:
        shifted = df.groupby(df.index.get_level_values(0))[
            ["storage", "release"]].shift(1)
        df["release_pre"] = shifted["release"]
        df["storage_pre"] = shifted["storage"]
        tmp = df.groupby(df.index.get_level_values(0))[
            ["storage_pre", "release_pre", "inflow"]].rolling(7).mean()
        tmp.index = tmp.index.droplevel(0)
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

def trim_data_to_span(in_df: pd.DataFrame, spans: pd.DataFrame, min_yrs: int=5) -> pd.DataFrame:
    cut_off = min_yrs * 365.25
    trimmed_spans = spans[spans["delta"].dt.days >= cut_off]
    out_dfs = []
    idx = pd.IndexSlice
    for res, row in trimmed_spans.iterrows():
        min_date = row["min"]
        max_date = row["max"]
        my_df = in_df.loc[idx[res,:],:]
        my_df = my_df.loc[(my_df.index.get_level_values(1) >= min_date) &
                          (my_df.index.get_level_values(1) <= max_date)]
        out_dfs.append(my_df)
    out_df = pd.concat(out_dfs, axis=0, ignore_index=False)
    return out_df

def standardize_variables(in_df: pd.DataFrame) -> pd.DataFrame:
    means = in_df.groupby(in_df.index.get_level_values(0)).mean()
    stds = in_df.groupby(in_df.index.get_level_values(0)).std()
    idx = pd.IndexSlice
    out_dfs = []
    for res in means.index:
        out_dfs.append(
            (in_df.loc[idx[res,:],:] - means.loc[res]) / stds.loc[res]
        )
    return pd.concat(out_dfs, axis=0, ignore_index=False), means, stds

def make_meta_data(df: pd.DataFrame, means: pd.DataFrame, loc: str) -> pd.DataFrame:
    rts = means["storage"] / (means["release"] * 24 * 3600 / 43560)
    corrs = {}
    idx = pd.IndexSlice
    for res in means.index:
        corrs[res] = df.loc[idx[res,:],"release"].corr(df.loc[idx[res,:], "inflow"])
    corrs = pd.Series(corrs)

    max_sto = df.groupby(df.index.get_level_values(0))["storage"].max()
    return pd.DataFrame({"rts":rts, "rel_inf_corr":corrs, "max_sto":max_sto})

def find_res_group(meta_row: pd.Series) -> str:
    if meta_row.rts > 31:
        return "high_rt"
    else:
        if meta_row.rel_inf_corr >= 0.95 and meta_row.max_sto < 10:
            return "ror"
        else:
            return "low_rt"

def get_model_ready_data(args):
    location = args.location
    USE_GPU = args.use_gpu
    data, needs_format = load_data(location, use_gpu=USE_GPU)
    if needs_format:
        data = get_valid_entries(data, location)
        data = rename_columns(data, location)
        data = set_proper_index(data, location, use_gpu=USE_GPU)
        # data = prep_data(data, location, use_gpu=USE_GPU)
        spans = get_max_res_date_spans(data, use_gpu=USE_GPU)
        trimmed_data = trim_data_to_span(data, spans)
        trimmed_data = prep_data(trimmed_data, location, use_gpu=USE_GPU)
        trimmed_data = trimmed_data.loc[:, ["release","release_pre", "storage", "storage_pre", "inflow",
                                            "release_roll7", "inflow_roll7", "storage_roll7"]]
        trimmed_data = trimmed_data[~trimmed_data.isna().any(axis=1)]
        trimmed_data[["storage", "storage_roll7", "storage_pre"]] *= (1/1000) # 1000 acre-ft
        trimmed_data[["release", "release_pre", "release_roll7","inflow","inflow_roll7"]] *= (43560 / 24 / 3600 / 1000) # cfs to 1000 acre ft per day
        trimmed_data["storage_x_inflow"] = trimmed_data["storage_pre"] * trimmed_data["inflow"]
        trimmed_data.to_csv(DATA_LOCS[location]["ready"])
        std_data, means, std = standardize_variables(trimmed_data)
        meta = make_meta_data(trimmed_data, means, location)
        meta["group"] = meta.apply(find_res_group, axis=1)
        return trimmed_data, std_data, means, std, meta
    else:
        std_data, means, std = standardize_variables(data) 
        meta = make_meta_data(data, means, location)
        meta["group"] = meta.apply(find_res_group, axis=1)
        return data, std_data, means, std, meta

def get_group_res(meta: pd.DataFrame) -> tuple:
    high_rt_res = list(meta[meta["group"] == "high_rt"].index)
    low_rt_res = list(meta[meta["group"] == "low_rt"].index)
    ror_res = list(meta[meta["group"] == "ror"].index)
    return (high_rt_res, low_rt_res, ror_res)

def filter_group_data(df: pd.DataFrame, resers: Iterable) -> pd.DataFrame:
    return df[df.index.get_level_values(0).isin(resers)]

def calc_metrics(eval_data: pd.DataFrame) -> pd.DataFrame:
    metrics = pd.DataFrame(index=eval_data.index.get_level_values(0).unique(), 
                           columns=["r2_score", "rmse"])
    idx = pd.IndexSlice
    for res in metrics.index:
        score = r2_score(
            eval_data.loc[idx[res,:],"actual"],
            eval_data.loc[idx[res,:],"modeled"]
        )
        rmse = np.sqrt(mean_squared_error(
            eval_data.loc[idx[res,:],"actual"],
            eval_data.loc[idx[res,:],"modeled"]
        ))
        metrics.loc[res,:] = [score, rmse]
    return metrics

def combine_res_meta():
    metas = []
    for location in DATA_LOCS.keys():
        metas.append(pd.read_pickle(f"./basin_output/{location}_meta.pickle"))
    meta = pd.concat(metas, axis=0, ignore_index=False)
    return meta

if __name__ == "__main__":
    print("This file is not designed to be ran on it is own.")
    print("It simply provides utilities for other scripts.")