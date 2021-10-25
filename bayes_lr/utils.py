import sys
import pathlib
import pickle
import pandas as pd
import numpy as np
from time import perf_counter as timer

def time_func_print(func):
    def wrapper(*args, **kwargs):
        output, time = time_function(func, *args, **kwargs)
        print(f"Time to run '{func.__name__}': {time:.5f} seconds")
        return output
    return wrapper

def time_function(func, *args, **kwargs):
    time1 = timer()
    output = func(*args, **kwargs)
    time2 = timer()
    return output, time2-time1

def time_func_stats(func, N=10):
    def wrapper(*args, **kwargs):
        times = np.zeros(N, dtype=np.float64)
        for i in range(N):
            output, time = time_function(func, *args, **kwargs)
            times[i] = time
        avg, std = times.mean(), times.std()
        print(f"{func.__name__} called {N} times: Avg. +- 1 SD = {avg:.5f} +- {std:.5f} seconds")
        return output
    return wrapper

@time_func_print
def read_basin_data(basin: str) -> pd.DataFrame:
    data_locs = {
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
    if basin not in data_locs:
        raise NotImplementedError(f"No data available for basin {basin}")
    
    fpath = pathlib.Path(data_locs[basin]["ready"])
    if fpath.exists():
        return pd.read_csv(fpath, index_col=[0,1], converters={1:pd.to_datetime})
    else:
        print(f"Please run ../apply_models/apply_models.py {basin} to generate model ready data for this basin.")
        sys.exit()

def get_basin_meta_data(basin: str):
    with open(f"../apply_models/basin_output/{basin}_meta.pickle", "rb") as f:
        meta = pickle.load(f)
    return meta

def prep_data(df):
    grouper = df.index.get_level_values(0)
    std_data = df.groupby(grouper).apply(lambda x: (x - x.mean()) / x.std())
    means = df.groupby(grouper).mean()
    std = df.groupby(grouper).std()
    X = std_data[[
        "release_pre", "storage", "storage_pre", "inflow", "release_roll7", 
        "inflow_roll7", "storage_roll7", "storage_x_inflow"
    ]]
    y = std_data["release"]
    return X, y, means, std

if __name__ == "__main__":
    read_basin_data("upper_col") 