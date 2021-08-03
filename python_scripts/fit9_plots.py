import sys
import calendar
import pickle
import pathlib
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as GS
import matplotlib.lines as mlines
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from pandasgui import show
from itertools import product
from IPython import embed as II

# import my helper functions
from helper_functions import read_tva_data
from analysis_plots import (format_dict, CASCADE, determine_grid_size,
                            find_plot_functions)
from plot_helpers import abline

# setup plotting environment
plt.style.use("ggplot")
sns.set_context("talk")
style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# paths
RESULTS_DIR = pathlib.Path("../results")

TREE_LVOUT = {
    "baseline":[], # baseline
    "100":["Douglas", "Hiwassee", "Cherokee"], # < 100 days
    "100-150":["BlueRidge", "Fontana", "Nottely"], # 100 - 150 days
    "150-200":["Norris", "Chatuge"], # 150 - 200 days
    "200":["TimsFord", "SHolston", "Watauga"], # > 200 days
    "order":["100", "100-150", "150-200", "200"]
}

SIMP_LVOUT = {
    "baseline":[], # baseline
    "3-7":["Nikajack", "FtPatrick", "Wilson"], # 3-7 days
    "7-15":["Pickwick", "Chikamauga", "Wheeler", "FtLoudoun", "MeltonH", "Guntersville", "Apalachia"], # 7 - 15 days
    "15-30":["WattsBar", "Kentucky", "Ocoee1", "Boone"], # 15-30 days
    "order":["3-7", "7-15", "15-30"]
}

LABEL_MAP = {
    "scores":   dict(label="NSE", units=""),
    "rmse_rel": dict(label="RMSE", units="[1000 acre-ft/day]"),
    "rmse_sto": dict(label="RMSE", units="[1000 acre-ft/day]"),
    "rmse_pct": dict(label="RMSE", units="[%]")
}

def load_results():
    """Load results for plotting. 

    :return: Two dataframes, one with results from treed model and one from simple model
    :rtype: pd.DataFrame, pd.DataFrame
    """
    file = "fit9_results.pickle"
    treed_path = RESULTS_DIR / "synthesis" / "treed_model" / file
    simple_path = RESULTS_DIR / "synthesis" / "simple_model" / file
    with open(treed_path.as_posix(), "rb") as f:
        treed_data = pickle.load(f)
    with open(simple_path.as_posix(), "rb") as f:
        simple_data = pickle.load(f)

    return treed_data, simple_data

def combine_data_scores(treed_data, simple_data):
    """Combine data from two different models into a usable dictionary

    :param treed_data: DataFrame with results from the treed model runs
    :type treed_data: pd.DataFrame
    :param simple_data: DataFrame with results from the simple model runs
    :type simple_data: pd.DataFrame
    :return: Dictionary with combined train and test data for the treed 
    model and the simple model, along with scores and rmse for both models.
    :rtype: dict
    """
    tree_preds = treed_data["pred"]
    simp_preds = simple_data["pred"]

    tree_fit = treed_data["fitted"]
    simp_fit = simple_data["fitted"]

    tree = tree_preds.append(tree_fit)
    simp = simp_preds.append(simp_fit)

    tree_groups = [k for k,v in treed_data.items() if "res_scores" in v]
    simp_groups = [k for k,v in simple_data.items() if "res_scores" in v]

    bias_keys = [i for i in tree.index if "bias" in i]
    bias_flag = len(bias_keys) != 0

    if bias_flag:
        tree_bias = pd.DataFrame(index=treed_data[tree_groups[0]]["res_scores"].index, columns=tree_groups)
        simp_bias = pd.DataFrame(index=simple_data[simp_groups[0]]["res_scores"].index, columns=simp_groups)
        tree_bias_map = pd.DataFrame(index=treed_data[tree_groups[0]]["res_scores"].index, columns=tree_groups)
        simp_bias_map = pd.DataFrame(index=simple_data[simp_groups[0]]["res_scores"].index, columns=simp_groups)

        for group in tree_groups:
            tree_p_bias = tree_preds.loc["p_bias", group]      
            tree_bias.loc[tree_p_bias.index, group] = tree_p_bias
            tree_bias_map.loc[tree_p_bias.index, group] = "p"
            tree_f_bias = tree_fit.loc["f_bias", group]
            tree_bias.loc[tree_f_bias.index, group] = tree_f_bias
            tree_bias_map.loc[tree_f_bias.index, group] = "f"

        for group in simp_groups:
            simp_p_bias = simp_preds.loc["p_bias", group]      
            simp_bias.loc[simp_p_bias.index, group] = simp_p_bias
            simp_bias_map.loc[simp_p_bias.index, group] = "p"
            simp_f_bias = simp_fit.loc["f_bias", group]
            simp_bias.loc[simp_f_bias.index, group] = simp_f_bias
            simp_bias_map.loc[simp_f_bias.index, group] = "f"

        tree = tree.drop(bias_keys)
        simp = simp.drop(bias_keys)
    
    if "baseline" in treed_data.keys():
        blt = "baseline"
        bls = blt
    elif "0" in treed_data.keys():
        blt = "0"
        bls = blt
    else:
        blt = tree_groups[0]
        bls = simp_groups[0]

    tree_scores = pd.DataFrame(index=treed_data[blt]["res_scores"].index, columns=tree_groups)
    tree_rmse = pd.DataFrame(index=treed_data[blt]["res_scores"].index, columns=tree_groups)

    simp_scores = pd.DataFrame(index=simple_data[bls]["res_scores"].index, columns=simp_groups)
    simp_rmse = pd.DataFrame(index=simple_data[bls]["res_scores"].index, columns=simp_groups)

    for group in tree_groups:
        tree_scores.loc[:, group] = treed_data[group]["res_scores"]["NSE"]
        tree_rmse.loc[:, group] = treed_data[group]["res_scores"]["RMSE"]

    for group in simp_groups:
        simp_scores.loc[:, group] = simple_data[group]["res_scores"]["NSE"]
        simp_rmse.loc[:, group] = simple_data[group]["res_scores"]["RMSE"]

    output = {
        "tree": tree, "simp": simp,
        "tree_scores": tree_scores,
        "tree_rmse": tree_rmse,
        "simp_scores": simp_scores,
        "simp_rmse":simp_rmse
    }
    if bias_flag:
        output["bias_data"] = {
            "tree_bias":tree_bias,
            "tree_map":tree_bias_map,
            "simp_bias":simp_bias,
            "simp_map":simp_bias_map
        }
    return output

def get_model_data(tree, simp):
    tree_train = tree["train_data"]
    simp_train = simp["train_data"]
    tree_test = tree["test_data"]
    simp_test = simp["test_data"]

    tree_train["res_group"] = "upstream"
    simp_train["res_group"] = "downstream"
    
    tree_test["res_group"] = "upstream"
    simp_test["res_group"] = "downstream"

    train = tree_train.append(simp_train).sort_index()
    test = tree_test.append(simp_test).sort_index()

    return test, train

def get_quant_scores(tree, simp):
    tree_scores = tree.get("quant_scores")
    simp_scores = simp.get("quant_scores")
    tree_scores["res_group"] = "upstream"
    simp_scores["res_group"] = "downstream"

    return tree_scores.append(simp_scores)

def get_monthly_bias(tree, simp):
    tree_key = tree["fitted"].keys()[0]
    simp_key = simp["fitted"].keys()[0]

    tree_bias = pd.DataFrame(columns=["fitted", "preds"], index=range(1,13))
    simp_bias = pd.DataFrame(columns=["fitted", "preds"], index=range(1,13))

    f_bias = tree["fitted"][tree_key]["f_bias_month"]
    p_bias = tree["pred"][tree_key]["p_bias_month"]
    tree_bias["fitted"] = f_bias
    tree_bias["preds"] = p_bias
    
    f_bias = simp["fitted"][simp_key]["f_bias_month"]
    p_bias = simp["pred"][simp_key]["p_bias_month"]
    simp_bias["fitted"] = f_bias
    simp_bias["preds"] = p_bias

    return tree_bias, simp_bias


def normalize_values(df, divtype="res"):
    release_means = pd.read_pickle(
        "../pickles/tva_dam_mean_daily_release_1000acftpermonth.pickle"
    )
    if divtype == "res":
        df = df.T.divide(release_means).dropna(axis=1).T * 100 
    elif divtype == "month":
        II()
    else:
        II()
    return df

def parse_args(plot_functions):
    parser = argparse.ArgumentParser(description="Plot results from leave out runs.")
    parser.add_argument("-p", "--plot_func", help="What visualization to plot.", choices=plot_functions.keys(),
                        default=None)
    parser.add_argument("--pmod", choices=["hist", "1to1"], default="1to1",
                        help="Modifier for the specific plot type chosen. Alters the data visualization.")
    # parser.add_argument("-d", "--data_set", choices=["one", "some", "incr", "corr", "fit9"],
                        # help="Specify what data set should be used for plots")
    parser.add_argument("-m", "--metric", choices=["scores", "rmse"], default="scores",
                        help="Specify what metric should be plotted")
    parser.add_argument("--relative", action="store_true", default=False,
                        help="Plot values as percentages relative to means. Only works for physical quantities")
    parser.add_argument("-S", "--sort_by", choices=["CASCADE", "RT", "MStL"], default="RT",
                        help="Specify how the reservoirs should be sorted.")
    args = parser.parse_args()
    return args


def plot_quants(tree,simp,args):
    test, train = get_model_data(tree, simp)
    quant_scores = get_quant_scores(tree, simp)
    sns.set_context("notebook")
    fig, axes = plt.subplots(4, 3, figsize=(20,8.7))
    flat_axes = axes.flatten()
    fig.patch.set_alpha(0.0) 

    indices = product(
        ["upstream","downstream"],
        ["train", "test"],
        range(3)
    )

    qlabs = ["Lower 1/3", "Middle 1/3", "Upper 1/3"]
    
    for ax, index in zip(flat_axes, indices):
        res_type, group, q = index
        if group == "train":
            data = train[
                (train["bin"] == q) & (train["res_group"] == res_type)][
                    ["actual", "model"]
                ]
        else:
            data = test[
                (test["bin"] == q) & (test["res_group"] == res_type)][
                    ["actual", "model"]
                ]
        if args.pmod == "hist":
            sns.histplot(
                data, kde=True, ax=ax
            ) 
        else:
            ax.scatter(data["actual"], data["model"])
            abline(0,1,ax=ax,c="b")

        if ax in axes[0]:
            ax.set_title(f"{qlabs[q]}")
        
        if ax not in axes[:,0]:
            ax.set_ylabel("")
        else:
            if args.pmod == "hist":
                ylabel = ax.get_ylabel()
                ax.set_ylabel(f"{ylabel} [{group.capitalize()}]")
            else:
                ax.set_ylabel(f"Mod. Release [{group.capitalize()}]")
        
        if ax != flat_axes[0]:
            try:
                ax.get_legend().remove()
            except AttributeError as e:
                # if there is no legend there we cannot remove it
                # so we can catch this and ignore it.
                pass

        ylim = ax.get_ylim()
        xlim = ax.get_xlim()

        score = quant_scores[quant_scores["res_group"] == res_type].loc[q, group]
        # text_x = 0.8*(xlim[1] - xlim[0])
        # text_y = 0.8*(ylim[1] - ylim[0])
        # ax.text(text_x, text_y, f"NSE={score:.3f}", ha="center", va="center")

    fig.text(0.02, 3/4, "Upstream", va="center", rotation=90, fontsize=14, c="#555555")
    fig.text(0.02, 1/4, "Downstream", va="center", rotation=90, fontsize=14, c="#555555")
    if args.pmod == "hist":
        fig.text(0.5, 0.01, "Release [1000 acre-ft/day]", ha="center", c="#555555")
    else:
        fig.text(0.5, 0.01, "Actual Release [1000 acre-ft/day]", ha="center", c="#555555")

    plt.subplots_adjust(
        top=0.956,
        bottom=0.063,
        left=0.08,
        right=0.988,
        hspace=0.286,
        wspace=0.121
    )
    print(quant_scores)
    plt.show()

def plot_monthly_bias(tree, simp, args):
    tree_bias, simp_bias = get_monthly_bias(tree, simp)
    
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 8.7),
                            sharex=True)
    axes = axes.flatten()
    fig.patch.set_alpha(0.0)

    rename = {"fitted":"Training","preds":"Testing"}
    tree_bias = tree_bias.rename(columns=rename)
    simp_bias = simp_bias.rename(columns=rename) 

    if args.relative:
        tva = read_tva_data(just_load=True)
        rel = tva["Release"].unstack()

        test_tree_rel = rel[tree["0.3"]["test_res"]]
        train_tree_rel = rel[tree["0.3"]["train_res"]]
        test_simp_rel = rel[simp["0.6"]["test_res"]]
        train_simp_rel = rel[simp["0.6"]["train_res"]]

        test_tree_means = test_tree_rel.groupby(test_tree_rel.index.month).mean().mean(axis=1)
        train_tree_means = train_tree_rel.groupby(train_tree_rel.index.month).mean().mean(axis=1)
        test_simp_means = test_simp_rel.groupby(test_simp_rel.index.month).mean().mean(axis=1)
        train_simp_means = train_simp_rel.groupby(train_simp_rel.index.month).mean().mean(axis=1)

        tree_bias_norm = (tree_bias[["Training"]].T / train_tree_means * 100).T
        tree_bias_norm["Testing"] = tree_bias["Testing"] /  test_tree_means * 100
        simp_bias_norm = (simp_bias[["Training"]].T / train_simp_means * 100).T
        simp_bias_norm["Testing"] = simp_bias["Testing"] /  test_simp_means * 100

        tree_bias = tree_bias_norm
        simp_bias = simp_bias_norm

    tree_bias.plot.bar(ax=axes[0])
    simp_bias.plot.bar(ax=axes[1])

    axes[1].set_xticklabels(calendar.month_abbr[1:], rotation=0, ha="center")

    axes[0].set_title("Upstream")
    axes[1].set_title("Downstream")

    axes[0].set_ylabel("")
    axes[1].set_ylabel("")

    axes[1].get_legend().remove()

    if args.relative:
        units = r"% of mean"
    else:
        units = "1000 acre-ft/day"
    fig.text(
        0.02, 0.5, f"Release Bias [{units}]",
        ha="left", va="center", rotation=90
    )

    plt.subplots_adjust(
        top=0.937,
        bottom=0.078,
        left=0.075,
        right=0.987,
        hspace=0.196,
        wspace=0.2
    )
    plt.show()

def main(namespace):
    plot_functions = find_plot_functions(namespace)
    args = parse_args(plot_functions)

    tree, simp = load_results()

    if not args.plot_func:
        II()
        sys.exit()

    
    globals()[plot_functions[args.plot_func]](tree,simp,args)

if __name__ == "__main__":
    namespace = dir()
    main(namespace)
