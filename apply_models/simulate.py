import sys
import argparse
import json
import pickle
import glob
import pandas as pd
import numpy as np
import multiprocessing as mp
from datetime import datetime, timedelta
from time import perf_counter as timer
from IPython import embed as II
from utils import (get_model_ready_data, get_group_res,
                   filter_group_data, calc_metrics)
from apply_models import (load_params, get_low_rt_params, get_ror_params)
from timing_function import time_function


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("location", action="store", choices=["upper_col", "lower_col", "missouri",
                                                             "pnw", "tva", "all", "colorado"],
                        help="Indicate which basin to load data for.")
    parser.add_argument("--bsp", "-B", action="store_true",
                        help="Flag to use basin specific parameters for simulation.")
    parser.add_argument("--ints", "-I", action="store_true",
                        help="Use time varying intercepts or not.")
    parser.add_argument("--purp_grp", "-P", action="store_true",
                        help="Use purpose based grouping instead of RT based.")
    # parser.add_argument("--use_gpu", "-GPU", action="store_true", default=False,
                        # help="Flag to indicate process should be performed on GPU") 
    return parser.parse_args()

def simulate_res(X, dates, porder, means, std, params, use_ints=False, leaves=None):
    # X should be actual data, will standardize as we go.
    high_rt = leaves is not None
    out_dates = dates[6:]
    end = out_dates.size
    months = out_dates.astype('datetime64[M]').astype(int) % 12 + 1
    storage_out = np.empty(shape=out_dates.size, dtype=np.float64)
    release_out = np.empty(shape=out_dates.size, dtype=np.float64)
    # x_order = ["release_pre", "storage_pre", "inflow", "release_roll7", 
    #            "storage_roll7", "inflow_roll7", "storage_x_inflow"]

    for i in range(end):
        j = i + 6
        data = X[j]
        rel_roll = X[i:j, 0].mean()
        sto_roll = X[i:j, 1].mean()
        inf_roll = X[i:j, 2].mean()
        sto_x_inf = data[1] * data[2]
        data[3] = rel_roll
        data[4] = sto_roll
        data[5] = inf_roll
        data[6] = sto_x_inf

        month = months[i]

        std_data = (data - means) / std
        if high_rt:
            mparams = params[np.searchsorted(leaves, std_data[0])]
            mint = mparams[month + 5] if use_ints else 0.0 #! hard coded number of variables here (5)
            mparams = mparams[:6] #! same deal here (6 comes from the above 5 + 1)
            mdata = np.array(
                [std_data[1] - std_data[4], *std_data[porder]]
            )
        else:
            mparams = params
            #! similar to above, the 7 and 8 are hardcoded and will need to be dealt with later
            mint = mparams[month + 7] if use_ints else 0.0
            mparams = mparams[:8]
            mdata = np.array(
                [1, *std_data[porder]]
            )

        rel_t = mdata @ mparams + mint
        rel_t_act = rel_t * std[0] + means[0]
        if rel_t_act < 0:
            rel_t_act = 0
            rel_t = (rel_t_act - means[0]) / std[0]
        sto_t_1 = data[1]
        inf_t = data[2]
        sto_t_act = sto_t_1 + inf_t - rel_t_act
        sto_t_act = max(sto_t_act, 0)
        storage_out[i] = sto_t_act
        release_out[i] = rel_t_act
        if i+1 != end:
            X[j+1,0] = rel_t_act
            X[j+1,1] = sto_t_act
    return pd.DataFrame.from_records(zip(out_dates, release_out, storage_out),
                                     columns=["datetime", "release", "storage"])

def simul(res_args):
    res, req_args = res_args
    act_data, meta, params, means, stds, x_order, high_rt_porder, low_rt_porder, use_ints, basin = req_args
    group = meta.loc[res, "group"]
    if group == "high_rt":
        porder = high_rt_porder
        if basin:
            my_params = params[basin][group]["params"]
            leaves = params[basin][group]["leaves"]
        else: 
            my_params = params[group]["params"]
            leaves = params[group]["leaves"]
    else:
        porder = low_rt_porder
        if basin:
            my_params = params[basin][group]["params"]
        else:
            my_params = params[group]["params"]
        leaves = None
    idx = pd.IndexSlice
    X = act_data.loc[idx[res,:]]
    dates = pd.to_datetime(X.index).values
    # X will not have sto_diff or const in it yet, so we trim those off
    X = X[x_order].values
    my_means = means.loc[res, x_order].values
    my_std = stds.loc[res, x_order].values
    return simulate_res(X, dates, porder, my_means, my_std, my_params, use_ints, leaves)

def main(args):
    use_gpu = False
    args.use_gpu = use_gpu
    int_mod = "" if args.ints else "_no_ints"
    if args.bsp:
        if args.purp_grp:
            files = glob.glob(f"./basin_params/*{int_mod}_purpose.json")
        else:
            files = glob.glob(f"./basin_params/*{int_mod}.json")
        params = {i.split("/")[-1].split(".")[0]: load_params(i, use_gpu=False) for i in files}
        for key, value in list(params.items()):
            if key.split("_")[-1] == "purpose":
                params["_".join(key.split("_")[:-1])] = params[key]
                del params[key]
    else:
        params = load_params(use_gpu=False)
    act_data, std_data, means, stds, meta = get_model_ready_data(args)
    index = pd.MultiIndex.from_tuples(zip(act_data.index.get_level_values(0),
                                    pd.to_datetime(act_data.index.get_level_values(1))))
    act_data.index = index
    purposes = pd.read_csv("../nid_data/col_purposes.csv", index_col=0, squeeze=True)
    res_groups = get_group_res(meta)

    if args.purp_grp:
        meta["group"] = purposes

    output = {}
    high_rt_order = ["sto_diff", "storage_x_inflow", "release_pre", 
                     "release_roll7", "inflow", "inflow_roll7"]
    low_rt_order  = ["const", "inflow", "storage_pre", "release_pre", "storage_roll7",
                     "inflow_roll7", "release_roll7", "storage_x_inflow"]
    x_order = ["release_pre", "storage_pre", "inflow", "release_roll7", 
               "storage_roll7", "inflow_roll7", "storage_x_inflow"]

    high_rt_porder = [x_order.index(i) for i in high_rt_order[1:]]
    low_rt_porder = [x_order.index(i) for i in low_rt_order[1:]]

    # meta["group"] = "high_rt"

    req_args = [act_data, meta, params, means, stds, x_order, high_rt_porder, low_rt_porder, args.ints]
    if args.bsp:
        res_args = [(res, [*req_args, meta.loc[res, "basin"]]) for res in means.index]
    else:
        res_args = [(res, [*req_args, False]) for res in means.index]

    with mp.Pool(min(mp.cpu_count(), means.index.size)) as p:
        out_vals = p.map(simul, res_args)
    
    for i, res in enumerate(means.index):
        out_vals[i]["site_name"] = res
        out_vals[i]["basin"] = meta.loc[res, "basin"]
    output = pd.concat(out_vals, ignore_index=True)
    output = output.set_index(["site_name", "datetime"])
    output["release_act"] = act_data["release"]
    output["storage_act"] = act_data["storage"]
    storage_metrics = calc_metrics(output, act_name="storage_act", mod_name="storage")
    release_metrics = calc_metrics(output, act_name="release_act", mod_name="release")
    bsp_flag = "bsp_" if args.bsp else ""
    pg_flag = "_purposes" if args.purp_grp else ""
    with open(f"./simulation_output/{args.location}_{bsp_flag}results{int_mod}{pg_flag}.pickle", "wb") as f:
        pickle.dump({"output":output, 
                     "metrics":{
                         "storage":storage_metrics,
                         "release":release_metrics
        }}, f)



if __name__ == "__main__":
    main(parse_args())
