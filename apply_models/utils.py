import argparse
import glob
import datetime
import pathlib
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
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
    return data

def get_valid_entries(df: pd.DataFrame, location: str) -> pd.DataFrame:
    if location == "upper_col":
        good = ~df.isna()
        return df.loc[(good["release volume"] | good["total release"]) & (good["storage"])]
    else:
        raise NotImplementedError(f"Cannot get valid entries for location {location}")

def set_proper_index(df: pd.DataFrame, location: str, use_gpu=False) -> pd.DataFrame:
    if use_gpu:
        mindex = cd.MultiIndex.from_frame
    else:
        mindex = pd.MultiIndex.from_frame
    if location == "upper_col":
        df.loc[:, "datetime"] = pd.to_datetime(df.loc[:, "datetime"])
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
    stds = in_df.groupby(in_df.index.get_level_values(0)).mean()
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
        if meta_row.rel_inf_corr >= 0.95 and meta_row.max_sto < 10_000:
            return "ror"
        else:
            return "low_rt"

def apply_high_rt_model(std_df: pd.DataFrame):
    from high_rt_model import vec_find_leaf, get_params, run_model
    leaves = vec_find_leaf(std_df["release_pre"])
    params = np.array([get_params(l) for l in leaves])
    X = std_df.copy()
    X["sto_diff"] = X["storage_pre"] - X["storage_roll7"]
    X = X[["sto_diff", "storage_x_inflow", "release_pre", "release_roll7", "inflow", "inflow_roll7"]]
    X = X.values
    Y = (X*params).sum(axis=1)
    return pd.Series(Y, index=std_df.index)

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
    data = get_valid_entries(data, location)
    data = set_proper_index(data, location, use_gpu=USE_GPU)
    data = prep_data(data, location, use_gpu=USE_GPU)
    spans = get_max_res_date_spans(data, use_gpu=USE_GPU)
    trimmed_data = trim_data_to_span(data, spans)
    trimmed_data = trimmed_data.loc[:, ["release","release_pre", "storage", "storage_pre", "inflow",
                                        "release_roll7", "inflow_roll7", "storage_roll7"]]
    trimmed_data = trimmed_data[~trimmed_data.isna().any(axis=1)]
    trimmed_data[["storage", "storage_roll7", "storage_pre"]] *= (1/1000) # 1000 acre-ft
    trimmed_data[["release", "release_pre", "release_roll7","inflow","inflow_roll7"]] *= (43560 / 24 / 3600 / 1000) # cfs to 1000 acre ft per day
    trimmed_data["storage_x_inflow"] = trimmed_data["storage_pre"] * trimmed_data["inflow"]
    std_data, means, std = standardize_variables(trimmed_data)
    meta = make_meta_data(trimmed_data, means, location)
    meta["group"] = meta.apply(find_res_group, axis=1)
    
    high_rt_res = meta[meta["group"] == "high_rt"].index
    high_rt_std_data = std_data[std_data.index.get_level_values(0).isin(high_rt_res)]
    high_rt_data = trimmed_data[trimmed_data.index.get_level_values(0).isin(high_rt_res)]
    modeled_release = apply_high_rt_model(high_rt_std_data)
    modeled_release_act = (modeled_release.unstack().T * std.loc[high_rt_res, "release"] +
                             means.loc[high_rt_res, "release"]).T.stack()
    eval_data = pd.DataFrame(
        {
            "actual":trimmed_data["release"],
            "modeled":modeled_release_act
        }
    )
    eval_data = eval_data.loc[eval_data.index.get_level_values(0) != "CRYSTAL",:]
    metrics = pd.DataFrame(index=eval_data.index.get_level_values(0).unique(), columns=["r2_score", "rmse"])
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

    II() 