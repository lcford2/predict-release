import sys
import calendar
import pickle
import pathlib
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
from analysis_plots import (format_dict, CASCADE, determine_grid_size)

# setup plotting environment
plt.style.use("ggplot")
sns.set_context("talk")

# paths
RESULTS_DIR = pathlib.Path("../results")

def load_results(ftype="one"):
    """TODO describe function

    :param ftype:
    :returns:

    """
    if ftype == "one":
        file = "leave_one_out.pickle"
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

    II()

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



    
def main():
    treed_data, simple_data = load_results(ftype="some")
    # preds, fitt = combine_data_scores(treed_data, simple_data)
    combine_data_scores_some(treed_data, simple_data)
#    plot_score_bars(preds, fitt, sort_by="RT")
    #explore_coefs(treed_data, simple_data)

if __name__ == "__main__":
    main()
