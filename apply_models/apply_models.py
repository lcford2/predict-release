import pathlib
import argparse
import json
import pandas as pd
import numpy as np
from IPython import embed as II
from utils import (get_model_ready_data, get_group_res, 
                   filter_group_data, calc_metrics)


def load_params(file: str="./model_params.json") -> dict:
    with open(file, "r") as f:
        params = json.load(f)
    params["high_rt"]["leaves"] = np.array(params["high_rt"]["leaves"])
    return params

def get_low_rt_params(params: dict) -> list:
    return params["low_rt"]["params"]

def get_ror_params(params: dict) -> list:
    return params["ror"]["params"]

def get_high_rt_params(rel_pre: float, params: dict) -> list:
    # leaves are already sorted from smallest to largest, so no problem here.
    leaf = np.searchsorted(params["high_rt"]["leaves"], rel_pre)
    return params["high_rt"]["params"][leaf]

def apply_high_rt_model(std_df: pd.DataFrame, params: dict) -> pd.Series:
    coefs = np.array([get_high_rt_params(i, params) for i in std_df["release_pre"]])
    X = std_df.copy()
    X["sto_diff"] = X["storage_pre"] - X["storage_roll7"]
    X = X[["sto_diff", "storage_x_inflow", "release_pre", "release_roll7",
             "inflow", "inflow_roll7"]]
    X = X.values
    Y = (X*coefs).sum(axis=1)
    return pd.Series(Y, index=std_df.index)

def apply_low_rt_model(std_df: pd.DataFrame, params: dict, group: str) -> pd.Series:
    if group == "ror":
        coefs = get_ror_params(params)
    elif group == "low_rt":
        coefs = get_low_rt_params(params)
      
    X = std_df.copy()
    X["const"] = 1
    X = X[["const", "inflow", "storage_pre", "storage_roll7", "inflow_roll7", 
            "storage_x_inflow"]]
    X = X.values
    Y = X @ coefs
    return pd.Series(Y, index=std_df.index)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("location", action="store", choices=["upper_col", "lower_col", "missouri", "pnw"],
                        help="Indicate which basin to load data for.")
    parser.add_argument("--use_gpu", "-GPU", action="store_true", default=False,
                        help="Flag to indicate process should be performed on GPU") 
    return parser.parse_args()

def main(args):
    # params = load_params()
    act_data, std_data, means, stds, meta = get_model_ready_data(args)
    import sys
    sys.exit()
    res_groups = get_group_res(meta) # h, l, ror

    group_output = {}
    for label, resers in zip(["high_rt", "low_rt", "ror"], res_groups):
        model_data = filter_group_data(std_data, resers)
        actual_data = filter_group_data(act_data, resers)
        if label == "high_rt":
            modeled_release = apply_high_rt_model(model_data, params)
        else:
            modeled_release = apply_low_rt_model(model_data, params, label)
        
        modeled_release_act = (modeled_release.unstack().T * stds.loc[resers, "release"] +
                                means.loc[resers, "release"]).T.stack()
        eval_data = pd.DataFrame(
            {
                "actual":actual_data["release"],
                "modeled":modeled_release_act
            }
        )
        metrics = calc_metrics(eval_data)
        group_output[label] = {
            "eval_data":eval_data, "metrics":metrics
        }
    II() 

if __name__ == "__main__":
    main(parse_args())