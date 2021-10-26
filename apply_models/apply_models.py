import pathlib
import argparse
import json
import pickle
import pandas as pd
import numpy as np
from time import perf_counter as timer
from numba import jit
from IPython import embed as II
from utils import (get_model_ready_data, get_group_res, 
                   filter_group_data, calc_metrics)
import cupy as cp
import cudf as cd   

def load_params(file: str="./model_params.json", use_gpu: bool=False) -> dict:
    with open(file, "r") as f:
        params = json.load(f)
    if use_gpu:
        for key in params.keys():
            params[key]["params"] = cp.array(params[key]["params"])
        params["high_rt"]["leaves"] = cp.array(params["high_rt"]["leaves"])
    else:
        params["high_rt"]["leaves"] = np.array(params["high_rt"]["leaves"])
    return params

def get_low_rt_params(params: dict) -> list:
    return params["low_rt"]["params"]

def get_ror_params(params: dict) -> list:
    return params["ror"]["params"]

def apply_high_rt_model(std_df: pd.DataFrame, params: dict, use_gpu:bool=False) -> pd.Series:
    leaves = params["high_rt"]["leaves"]
    my_params = params["high_rt"]["params"]
    if use_gpu:
        # coefs = cp.array([my_params[get_high_rt_params(i)] for i in std_df["release_pre"].values_host])
        coef_index = cp.searchsorted(leaves, std_df["release_pre"].values)
        coefs = my_params[coef_index]
    else:
        # coefs = np.array([my_params[get_high_rt_params(i)] for i in std_df["release_pre"]])
        coef_index = np.searchsorted(leaves, std_df["release_pre"].values)
        coefs = np.take(my_params, coef_index, axis=0)
    X = std_df.copy()
    X["sto_diff"] = X["storage_pre"] - X["storage_roll7"]
    X = X[["sto_diff", "storage_x_inflow", "release_pre", "release_roll7",
             "inflow", "inflow_roll7"]]
    X = X.values
    # N = 1000
    # time1 = timer()
    # for N in range(N):
    Y = (X*coefs).sum(axis=1)
    # time2 = timer()
    # avg_time = (time2 - time1) / N
    # print(f"Avg Time for comp.: {avg_time}")
    if use_gpu:
        return cd.Series(Y, index=std_df.index)
    else:
        return pd.Series(Y, index=std_df.index)

def apply_low_rt_model(std_df: pd.DataFrame, params: dict, group: str, use_gpu:bool=False) -> pd.Series:
    if group == "ror":
        coefs = get_ror_params(params)
    elif group == "low_rt":
        coefs = get_low_rt_params(params)
      
    X = std_df.copy()
    X["const"] = 1
    X = X[["const", "inflow", "storage_pre", "release_pre", "storage_roll7", "inflow_roll7", 
            "release_roll7", "storage_x_inflow"]]
    X = X.values
    Y = X @ coefs
    if use_gpu:
        return cd.Series(Y, index=std_df.index)
    else:
        return pd.Series(Y, index=std_df.index)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("location", action="store", choices=["upper_col", "lower_col", "missouri", "pnw"],
                        help="Indicate which basin to load data for.")
    parser.add_argument("--use_gpu", "-GPU", action="store_true", default=False,
                        help="Flag to indicate process should be performed on GPU") 
    return parser.parse_args()

def main(args):
    use_gpu = args.use_gpu
    params = load_params(use_gpu=use_gpu)
    act_data, std_data, means, stds, meta = get_model_ready_data(args)
    res_groups = get_group_res(meta) # h, l, ror 
    group_output = {}
    for label, resers in zip(["high_rt", "low_rt", "ror"], res_groups):
        nresers = len(resers)
        if nresers == 0:
            continue
        model_data = filter_group_data(std_data, resers)
        actual_data = filter_group_data(act_data, resers)
        if label == "high_rt":
            modeled_release = apply_high_rt_model(model_data, params, use_gpu=use_gpu)
        else:
            modeled_release = apply_low_rt_model(model_data, params, label, use_gpu=use_gpu)

        if use_gpu:
            modeled_release = modeled_release.reset_index().pivot(
                index="datetime", columns="site_name"
            )
            modeled_release.columns = modeled_release.columns.droplevel(0)

            if nresers > 1:
                modeled_release_act = modeled_release * stds.loc[resers, "release"] + means.loc[resers, "release"]
            else:
                modeled_release_act = modeled_release * stds.loc[resers[0], "release"] + means.loc[resers[0], "release"]
            
            modeled_release_act = modeled_release_act.stack()
            modeled_release_act.index = cd.MultiIndex.from_tuples(
                zip(
                    modeled_release_act.index.get_level_values(1).values_host,
                    modeled_release_act.index.get_level_values(0).values_host
                )
            )
        else:
            modeled_release_act = (modeled_release.unstack().T * stds.loc[resers, "release"] +
                                means.loc[resers, "release"]).T.stack()

        if use_gpu:
            df_construct = cd.DataFrame
        else:
            df_construct = pd.DataFrame

        eval_data = df_construct(
            {
                "actual":actual_data["release"],
                "modeled":modeled_release_act
            }
        )
        metrics = calc_metrics(eval_data, use_gpu=use_gpu)
        if use_gpu:
            print(metrics.to_pandas().to_markdown(floatfmt="0.3f"))
        else:
            print(metrics.to_markdown(floatfmt="0.3f"))

        group_output[label] = {
            "eval_data":eval_data, "metrics":metrics
        }
    with open(f"basin_output_no_ints/{args.location}.pickle", "wb") as f:
        pickle.dump(group_output, f)
    
    with open(f"basin_output_no_ints/{args.location}_meta.pickle", "wb") as f:
        pickle.dump(meta, f)


if __name__ == "__main__":
    main(parse_args())