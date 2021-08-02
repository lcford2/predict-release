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
from IPython import embed as II

# import my helper functions
from helper_functions import read_tva_data
from analysis_plots import (format_dict, CASCADE, determine_grid_size,
                            find_plot_functions)

# setup plotting environment
plt.style.use("ggplot")
sns.set_context("talk")

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

def load_results(ftype="one"):
    """Load results for plotting. 

    :param ftype: what results to load [one, incr, some], defaults to "one"
    :type ftype: str, optional
    :return: Two dataframes, one with results from treed model and one from simple model
    :rtype: pd.DataFrame, pd.DataFrame
    """
    if ftype == "one":
        file = "leave_one_out.pickle"
    elif ftype == "incr":
        file = "leave_incremental_out.pickle"
    elif ftype == "corr":
        file = "leave_corr_out.pickle"
    elif ftype == "fit9":
        file = "fit9_results.pickle"
    else:
        file = "leave_some_out.pickle"
    treed_path = RESULTS_DIR / "synthesis" / "treed_model" / file
    simple_path = RESULTS_DIR / "synthesis" / "simple_model" / file
    with open(treed_path.as_posix(), "rb") as f:
        treed_data = pickle.load(f)
    with open(simple_path.as_posix(), "rb") as f:
        simple_data = pickle.load(f)

    return treed_data, simple_data

def combine_data_scores(treed_data, simple_data):
    """Combine the score data frames from each model into a preds and fitted df

    :param treed_data: Dictionary containing results from LOO treed model runs
    :param simple_data: Dictionary containing results from LOO simple model runs
    :returns: preds, fitt - Both are dataframes containing scores for respective modeling periods
    """

    tree_preds = treed_data["pred"]
    simp_preds = simple_data["pred"]
    preds = tree_preds.join(simp_preds)

    tree_fitt = treed_data["fitted"]
    simp_fitt = simple_data["fitted"]
    fitt = tree_fitt.join(simp_fitt)

    return preds, fitt

def combine_data_scores_some(treed_data, simple_data):
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

def normalize_rmse(df):
    release_means = pd.read_pickle(
        "../pickles/tva_dam_mean_daily_release_1000acftpermonth.pickle"
    )
    df = df.T.divide(release_means).dropna(axis=1).T * 100 
    return df

def plot_score_bars(preds, fitt, sort_by="CASCADE"):
    if sort_by == "CASCADE":
        sort_by = CASCADE
    elif sort_by == "RT":
        rt = pd.read_pickle("../pickles/tva_res_times.pickle")
        msl = pd.read
        rt = rt.sort_values()
        sort_by = rt.index
    elif sort_by == "MStL":
        msl = pd.read_pickle("../pickles/tva_mean_st_level.pickle")
        msl = msl.sort_values()
        sort_by = msl.index

    p_scores = preds.loc[f"p_act_score", sort_by]
    f_scores = fitt.loc[f"f_act_score", sort_by]
    p_rmse = preds.loc[f"p_act_rmse", sort_by]
    f_rmse = fitt.loc[f"f_act_rmse", sort_by]
    scores = pd.DataFrame({"Fitted":f_scores, "Predicted":p_scores})
    rmse = pd.DataFrame({"Fitted":f_rmse, "Predicted":p_rmse})

    fig, axes = plt.subplots(2,1, sharex=True, figsize=(20,8.7))
    axes = axes.flatten()
    fig.patch.set_alpha(0.0)

    scores.plot.bar(ax=axes[0], width=0.8)
    rmse.plot.bar(ax=axes[1], width=0.8)

    axes[0].set_ylabel("NSE")
    axes[1].set_ylabel("RMSE")
    axes[0].get_legend().remove()

    axes[1].set_xticklabels(sort_by, rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    plt.show()


def explore_coefs(treed_data, simple_data):
    treed_res = [k for k,v in treed_data.items() if "coefs" in v]
    simple_res = [k for k,v in simple_data.items() if "coefs" in v]

    tree_coef_index = pd.MultiIndex.from_product(
        [treed_res, pd.DataFrame(treed_data[treed_res[0]]["coefs"]).index])
    simple_coef_index = pd.MultiIndex.from_product(
        [simple_res, pd.DataFrame(simple_data[simple_res[0]]["coefs"]).index])

    tree_coefs = pd.DataFrame(index=tree_coef_index, columns = treed_data[treed_res[0]]["coefs"].keys())
    simple_coefs = pd.DataFrame(index=simple_coef_index, columns = simple_data[simple_res[0]]["coefs"].keys())

    idx = pd.IndexSlice
    for res in treed_res:
         r_coefs = pd.DataFrame(treed_data[res]["coefs"])
         tree_coefs.loc[idx[res,:],:] = r_coefs

    for res in simple_res:
         r_coefs = pd.DataFrame(simple_data[res]["coefs"])
         simple_coefs.loc[idx[res,:],:] = r_coefs

    II()

def plot_res_scores_some(args):
    sort_by = args.sort_by
    metric = args.metric
    treed_data, simple_data = load_results(ftype=args.data_set)
    combined_data = combine_data_scores_some(treed_data, simple_data)
    if sort_by == "CASCADE":
        sort_by = CASCADE
    elif sort_by == "RT":
        rt = pd.read_pickle("../pickles/tva_res_times.pickle")
        rt = rt.sort_values()
        sort_by = rt.index
    elif sort_by == "MStL":
        msl = pd.read_pickle("../pickles/tva_mean_st_level.pickle")
        msl = msl.sort_values()
        sort_by = msl.index

    tree = combined_data.get(f"tree_{metric}")
    simp = combined_data.get(f"simp_{metric}")

    if not isinstance(tree, pd.DataFrame):
        print("Metric not available. Please choose either 'scores' or 'rmse'.")
        sys.exit()

    tree_plot = pd.DataFrame(index=tree.index, columns=["Baseline", "LO"])
    simp_plot = pd.DataFrame(index=simp.index, columns=["Baseline", "LO"])


    for key, rgroup in TREE_LVOUT.items():
        if key == "order":
            continue
        tree_plot.loc[rgroup, "LO"] = tree.loc[rgroup, key]

    for key, rgroup in SIMP_LVOUT.items():
        if key == "order":
            continue
        simp_plot.loc[rgroup, "LO"] = simp.loc[rgroup, key]

    tree_plot["Baseline"] = tree["baseline"]
    simp_plot["Baseline"] = simp["baseline"]


    simp_plot.loc[["Wilbur", "Ocoee3"], "LO"] = simp_plot.loc[["Wilbur", "Ocoee3"], "Baseline"]

    style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, axes = plt.subplots(2,1, figsize=(20,8.7))
    fig.patch.set_alpha(0.0)
    axes = axes.flatten()

    if metric == "rmse":
        tree_plot = normalize_rmse(tree_plot)
        simp_plot = normalize_rmse(simp_plot)
        label_info = LABEL_MAP["rmse_pct"]
    else:
        label_info = LABEL_MAP["scores"]
    
    ylabel = " ".join([label_info["label"], label_info["units"]])
    
    for ax, df, label, lvout in zip(axes, [tree_plot, simp_plot], ["Upstream", "Downstream"], [TREE_LVOUT, SIMP_LVOUT]):
        df.plot.bar(ax=ax, width=0.8)
        ax.set_title(label)
        label_colors = []

        tlabels = [i for i in sort_by if i in df.index]
        possible_colors = style_colors[1:len(lvout["order"])+1]

        colors = []
        for t in tlabels:
            for i, group in enumerate(lvout["order"]):
                if t in lvout[group]:
                    colors.append(possible_colors[i])

        ax.set_xticklabels(tlabels, rotation=30, ha="right", rotation_mode="anchor")
        if label == "Downstream":
            colors = [style_colors[0], style_colors[0]] + colors
        for xtick, color in zip(ax.get_xticklabels(), colors):
            xtick.set_color(color)

        #handles, labels = ax.get_legend_handles_labels()
        ax.legend(loc="lower right", ncol=len(lvout["order"]) + 1)
        ax.set_ylabel(ylabel)

    plt.tight_layout()
    plt.show()

def plot_res_scores_incremental(args):
    metric = args.metric
    treed_data, simple_data = load_results(ftype=args.data_set)
    combined_data = combine_data_scores_some(treed_data, simple_data)

    tree = combined_data.get(f"tree_{metric}")
    simp = combined_data.get(f"simp_{metric}")

    if metric == "rmse":
        tree = normalize_rmse(tree)
        simp = normalize_rmse(simp)
        label_info = LABEL_MAP["rmse_pct"]
        fmt = "0.0f"
    else:
        label_info = LABEL_MAP["scores"]
        fmt = "0.2f"
    
    label = " ".join([label_info["label"], label_info["units"]])


    if not isinstance(tree, pd.DataFrame):
        print("Metric not available. Please choose either 'scores' or 'rmse'.")
        sys.exit()

    tree_order = ['BlueRidge', 'Chatuge', 'Cherokee', 'Douglas', 'Fontana', 'Hiwassee',
                  'Norris', 'Nottely', 'SHolston', 'TimsFord', 'Watauga']
    simp_order = ['Apalachia', 'Boone', 'Chikamauga', 'FtLoudoun', 'FtPatrick',
                  'Guntersville', 'Kentucky', 'MeltonH', 'Nikajack', 'Ocoee1', 'Ocoee3',
                  'Pickwick', 'WattsBar', 'Wheeler', 'Wilbur', 'Wilson']

    tree_lvout = {str(i):j for i, j in enumerate(["None"] + tree_order)}
    tree = tree.rename(columns=tree_lvout)
    tree = tree.astype(float)

    simp_lvout = {str(i):j for i, j in enumerate(["None"] + simp_order)}
    simp = simp.rename(columns=simp_lvout)
    simp = simp.astype(float)

    tree_columns = {i: round(float(i), 2) for i in tree.columns}
    simp_columns = {i: round(float(i), 2) for i in simp.columns}
    tree = tree.rename(columns=tree_columns)
    simp = simp.rename(columns=simp_columns)

    fig = plt.figure(figsize=(20,8.7))
    fig.patch.set_alpha(0.0)
    gs = GS.GridSpec(ncols=2, nrows=1, figure=fig, width_ratios=[20, 1])
    ax = fig.add_subplot(gs[0,0])
    cbar_ax = fig.add_subplot(gs[0,1])

    sns.heatmap(tree, ax=ax, cbar_ax=cbar_ax, annot=True, fmt=fmt)
    ax.set_xticklabels(tree.columns, rotation=45, ha="right", rotation_mode="anchor")
    cbar_ax.set_ylabel(label)
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(20,8.7))
    fig.patch.set_alpha(0.0)
    gs = GS.GridSpec(ncols=2, nrows=1, figure=fig, width_ratios=[20, 1])
    ax = fig.add_subplot(gs[0,0])
    cbar_ax = fig.add_subplot(gs[0,1])

    sns.heatmap(simp, ax=ax, cbar_ax=cbar_ax, annot=True, fmt=fmt)
    ax.set_xticklabels(simp.columns, rotation=45, ha="right", rotation_mode="anchor")
    cbar_ax.set_ylabel(label)
    plt.tight_layout()
    plt.show()

def plot_results_corr(args):
    metric = args.metric
    treed_data, simple_data = load_results(ftype=args.data_set)
    combined_data = combine_data_scores_some(treed_data, simple_data)
    tree_map = combined_data["bias_data"]["tree_map"]
    simp_map = combined_data["bias_data"]["simp_map"]

    tree_bias = combined_data["bias_data"]["tree_bias"]
    simp_bias = combined_data["bias_data"]["simp_bias"]

    tree_order = tree_map.applymap(lambda x: 1 if x == "p" else 0).sum(axis=1).sort_values().index
    simp_order = simp_map.applymap(lambda x: 1 if x == "p" else 0).sum(axis=1).sort_values().index

    tree_scores = combined_data["tree_scores"]
    simp_scores = combined_data["simp_scores"]

    tree_scores = tree_scores.astype(float)
    simp_scores = simp_scores.astype(float)

    tree_scores = tree_scores.loc[tree_order]
    simp_scores = simp_scores.loc[simp_order]

    def get_sign_string(x):
        if abs(x) < 0.1:
            return "  "
        else:
            return f" {np.sign(x):+.0f}"[:2]

    # tree_bias_sign = np.sign(tree_bias).applymap(lambda x: f" {x:+d}"[:2])
    # simp_bias_sign = np.sign(simp_bias).applymap(lambda x: f" {x:+d}"[:2])
    tree_bias_sign = tree_bias.applymap(get_sign_string)
    simp_bias_sign = simp_bias.applymap(get_sign_string)

    tree_annot = tree_map.applymap(
        lambda x: x.upper()) + tree_bias_sign + tree_scores.applymap(lambda x: f"{x:.3f}")
    tree_annot = tree_annot.loc[tree_order]

    
    simp_annot = simp_map.applymap(
        lambda x: x.upper()) + simp_bias_sign + simp_scores.applymap(lambda x: f"{x:.3f}")
    simp_annot = simp_annot.loc[simp_order]
    
    fig = plt.figure(figsize=(20,8.7))
    fig.patch.set_alpha(0.0)
    gs = GS.GridSpec(ncols=2, nrows=1, figure=fig, width_ratios=[20, 1])
    ax = fig.add_subplot(gs[0,0])
    cbar_ax = fig.add_subplot(gs[0,1])
    sns.heatmap(tree_scores, annot=tree_annot, fmt="s", ax=ax, cbar_ax=cbar_ax)
    cbar_ax.set_ylabel("NSE")
    plt.show()

    fig = plt.figure(figsize=(20,8.7))
    fig.patch.set_alpha(0.0)
    gs = GS.GridSpec(ncols=2, nrows=1, figure=fig, width_ratios=[20, 1])
    ax = fig.add_subplot(gs[0,0])
    cbar_ax = fig.add_subplot(gs[0,1])
    sns.heatmap(simp_scores, annot=simp_annot, fmt="s", ax=ax, cbar_ax=cbar_ax)
    cbar_ax.set_ylabel("NSE")
    plt.show()



def parse_args(plot_functions):
    parser = argparse.ArgumentParser(description="Plot results from leave out runs.")
    parser.add_argument("-p", "--plot_func", help="What visualization to plot.", choices=plot_functions.keys(),
                        default=None)
    parser.add_argument("-d", "--data_set", choices=["one", "some", "incr", "corr", "fit9"],
                        help="Specify what data set should be used for plots")
    parser.add_argument("-m", "--metric", choices=["scores", "rmse"], default="scores",
                        help="Specify what metric should be plotted")
    parser.add_argument("-S", "--sort_by", choices=["CASCADE", "RT", "MStL"], default="RT",
                        help="Specify how the reservoirs should be sorted.")
    args = parser.parse_args()
    return args

def infer_data_set(args):
    pfunc = args.plot_func 
    if not pfunc:
        raise ValueError("Must provide at least one of the following command line \
                          arguments: 'plot_func', 'data_set'.")
    pfunc_split = pfunc.split("_")
    poss_type = pfunc_split[-1]
    if poss_type == "incremental":
        args.data_set = "incr"
    elif poss_type == "some":
        args.data_set = "some"
    elif poss_type == "corr":
        args.data_set = "corr"
    else:
        args.data_set = "one"
    return args


def main(namespace):
    plot_functions = find_plot_functions(namespace)
    args = parse_args(plot_functions)
    if not args.data_set:
        args = infer_data_set(args)

    if not args.plot_func:
        II()
    
    globals()[plot_functions[args.plot_func]](args)

if __name__ == "__main__":
    namespace = dir()
    main(namespace)
