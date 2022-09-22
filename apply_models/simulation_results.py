import pickle
import glob
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np 
import pandas as pd
import argparse
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime, timedelta
from plot_helpers import determine_grid_size, abline
from IPython import embed as II

sns.set_context("paper")
plt.style.use("ggplot")
style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


BASIN_NAMES = {"upper_col":"Upper Colorado", "lower_col":"Lower Colorado",
               "pnw":"Pacific Northwest","missouri":"Missouri","tva":"TVA",
               "colorado":"Colorado"}


def read_results(basin,bsp):
    modifier = "bsp_" if bsp else ""
    with open(f"./simulation_output/{basin}_{modifier}results.pickle", "rb") as f:
        data = pickle.load(f)
    return data

# def r2_score(y_act, y_mod):
#     ss_res = sum((i-j)**2 for i,j in zip(y_act, y_mod))
#     y_act_mean = np.mean(y_act)
#     ss_tot = sum((i-y_act_mean)**2 for i in y_act)
#     if ss_tot == 0:
#         return 0
#     else:
#         return 1 - (ss_res)/(ss_tot)

# def mean_squared_error(y_act, y_mod, squared=True):
#     mse = np.mean([(i-j)**2 for i,j in zip(y_act, y_mod)])
#     if squared:
#         return mse
#     else:
#         return np.sqrt(mse)


def check_first_N_days(output, N=30):
    delta = timedelta(days=N)
    res = output.index.get_level_values(0).unique()
    metrics = []
    idx = pd.IndexSlice
    for res in res:
        my_df = output.loc[idx[res,:]]
        initial_date = my_df.index[0]
        cutoff = initial_date + delta
        my_df = my_df[my_df.index < cutoff]
        rel_score = r2_score(my_df["release_act"], my_df["release"])
        sto_score = r2_score(my_df["storage_act"], my_df["storage"])
        rel_rmse = mean_squared_error(my_df["release_act"], my_df["release"], squared=False)
        sto_rmse = mean_squared_error(my_df["storage_act"], my_df["storage"], squared=False)
        metrics.append([res, "release", rel_score, rel_rmse])
        metrics.append([res, "storage", sto_score, sto_rmse])

    metrics = pd.DataFrame.from_records(metrics, columns=["res", "var", "r2_score", "rmse"])

    return metrics

def plot_time_series(output, var: str="release", N: int=-1):
    res = output.index.get_level_values(0).unique()
    gs = determine_grid_size(res.size, col_bias=False)
    fig, axes = plt.subplots(*gs, figsize=(20,8.7))
    axes = axes.flatten()
    fig.patch.set_alpha(0.0)

    idx = pd.IndexSlice
    for res, ax in zip(res, axes):
        my_df = output.loc[idx[res,:]]
        if N != -1:
            my_df = my_df[my_df.index < my_df.index[0] + timedelta(days=N)]
        ax.plot(my_df.index, my_df[var])
        ax.plot(my_df.index, my_df[f"{var}_act"])
        ax.set_title(res)
        ax.tick_params(axis="x", which="major", pad=2)
    
    plt.subplots_adjust(
        top=0.94,
        bottom=0.06,
        left=0.035,
        right=0.98,
        hspace=0.4,
        wspace=0.1
    )   
    plt.show()


def plot_one2one_split(output, meta:dict, var:str="release", N: int=-1):
    idx = pd.IndexSlice
    basins = output["basin"].unique()
    for basin in basins:
        bdf = output[output["basin"] == basin]
        mean_release = bdf.groupby(bdf.index.get_level_values(0))["release_act"].mean()
        mean_release = mean_release.sort_values()
        res = mean_release.index
        # res = bdf.index.get_level_values(0).unique()
        gs = determine_grid_size(res.size, col_bias=False)
        fig, axes = plt.subplots(*gs, figsize=(20,8.7))
        axes = axes.flatten()
        fig.patch.set_alpha(0.0)
        basin_meta = meta[basin]
        for group, ax in zip(res, axes):
            my_df = output.loc[idx[group,:]]
            if N != -1:
                my_df = my_df[my_df.index < my_df.index[0] + timedelta(days=N)]
            ax.scatter(my_df[f"{var}_act"], my_df[var])
            abline(0, 1, ax=ax, c=style_colors[1])
            ax.set_title(f"{group} ({basin_meta.loc[group, 'group']})")
            # ax.tick_params(axis="x", which="major", pad=2)
        left_over = axes.size - res.size
        if left_over > 0:
            for ax in axes[-left_over:]:
                ax.set_axis_off()
        
        fig.suptitle(BASIN_NAMES[basin], fontsize=18) 
        fig.text(0.5, 0.02, f"Actual {var.capitalize()}", ha="center", fontsize=18)
        fig.text(0.02, 0.5, f"Modeled {var.capitalize()}", ha="center", va="center", rotation=90, fontsize=18)
        # fig.tight_layout()
        if basin == "tva":
            hspace = 0.63
        elif basin == "missouri":
            hspace = 0.5
        else:
            hspace = 0.4
    
        plt.subplots_adjust(
            top=0.92,
            bottom=0.08,
            left=0.05,
            right=0.98,
            hspace=hspace,
            wspace=0.1
        )   
        plt.show()
    

def plot_one2one(output, var: str="release", grouper: str="res", N: int=-1):
    if grouper == "res":
        groups = output.index.get_level_values(0).unique()
    elif grouper == "basin":
        groups = output["basin"].unique()

    idx = pd.IndexSlice
    gs = determine_grid_size(groups.size, col_bias=False)
    fig, axes = plt.subplots(*gs, figsize=(20,8.7))
    axes = axes.flatten()
    fig.patch.set_alpha(0.0)
    for group, ax in zip(groups, axes):
        if grouper == "res":
            my_df = output.loc[idx[group,:]]
            if N != -1:
                my_df = my_df[my_df.index < my_df.index[0] + timedelta(days=N)]
        elif grouper == "basin":
            my_df = output.loc[output["basin"] == group]
        ax.scatter(my_df[f"{var}_act"], my_df[var])
        abline(0, 1, ax=ax, c=style_colors[1])
        ax.set_title(group)
        # ax.tick_params(axis="x", which="major", pad=2)
    
    plt.subplots_adjust(
        top=0.94,
        bottom=0.06,
        left=0.04,
        right=0.98,
        hspace=0.5,
        wspace=0.15
    )   
    plt.show()

def load_meta_data():
    files = glob.glob("./basin_output_no_ints/*meta.pickle")
    return {
        "_".join(i.split("/")[-1].split("_")[:-1]): pd.read_pickle(i)
        for i in files
    }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("location", action="store", choices=["upper_col", "lower_col",
                                                             "pnw", "tva", "missouri",
                                                             "all", "colorado"],
                        help="Indicate which basin to load data for.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    basin = args.location
    results = read_results(basin,True)
    output = results["output"]
    meta = load_meta_data()
    meta["colorado"] = meta["upper_col"].append(meta["lower_col"])
    # PLOT_time_series(output, var="release", N=-1)
    plot_one2one_split(output, meta, var="release", N=-1)
