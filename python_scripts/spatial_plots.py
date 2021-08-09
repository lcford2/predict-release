import sys
import calendar
import pathlib
import pickle
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as GS
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from itertools import product
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from IPython import embed as II

# import my functions
from helper_functions import read_tva_data, calc_bias
from analysis_plots import format_dict, CASCADE
from plot_helpers import (abline, combine_legends, determine_grid_size, 
                          find_plot_functions)

# setup plotting environment
plt.style.use("ggplot")
sns.set_context("notebook")
# context can be paper (0.8), talk (1.2) or poster (1.4)

# get style colors as a list
style_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

RESULTS_DIR = pathlib.Path("../results")

def load_results(args):
    if args.thirds:
        pickle_file = "thirds_model/quant_slopes.pickle"
    else:
        pickle_file = "spatial_model/res_slopes.pickle"
    file = RESULTS_DIR / "synthesis" / pickle_file
    with open(file.as_posix(), "rb") as f:
        results = pickle.load(f)
    return results

def plot_coef_dist(results, args):
    coefs = results["coefs"]
    grid = determine_grid_size(coefs.index.size)

    fig, axes = plt.subplots(*grid, figsize=(20, 8.7), sharey=True)
    axes = axes.flatten()
    fig.patch.set_alpha(0.0)

    for ax, coef in zip(axes, coefs.index):
        sns.histplot(
            coefs.loc[coef], kde=True, ax=ax
        )
        ax.set_title(coef)
    plt.tight_layout() 
    plt.show()
    
def plot_quants(results, args):
    rel = results["y"]
    reservoirs = rel.index.get_level_values(1).unique()
    quant_index = pd.MultiIndex.from_product(
        [reservoirs, range(3)],
        names=["res", "bin"]
    )
    quant_metrics = pd.DataFrame(index=quant_index, columns=[
        "rmse", "score", "bias", "corr", "var_m", "var_a", "mean_a"
    ])

    idx = pd.IndexSlice
    for res in reservoirs:
        y_a = rel.loc[idx[:,res],"actual"]
        y_m = rel.loc[idx[:,res],"model"]
        q_bins = pd.qcut(y_a, 3, labels=False)
        for i in range(3):
            y_a_b = y_a[q_bins == i]
            y_m_b = y_m[q_bins == i]
            rmse = np.sqrt(mean_squared_error(y_a_b, y_m_b))
            score = r2_score(y_a_b, y_m_b)
            bias = calc_bias(y_a_b, y_m_b)
            corr = pearsonr(y_a_b, y_m_b)[0]
            var_m = y_m_b.var()
            var_a = y_a_b.var()
            mean = y_a_b.mean()
            quant_metrics.loc[(res, i)] = [
                rmse, score, bias, corr, var_m, var_a, mean 
            ]
    
    if args.relative:
        quant_metrics["rmse"] = quant_metrics["rmse"] / quant_metrics["mean_a"]
        quant_metrics["bias"] = quant_metrics["bias"] / quant_metrics["mean_a"]
    
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(20, 8.7), sharex=True)
    axes = axes.flatten()
    fig.patch.set_alpha(0.0)

    for ax, (metric, qbin) in zip(axes, product(["score", "corr", "rmse", "bias"], range(3))):
        qm = quant_metrics.loc[idx[:,qbin], metric]
        ax.bar(range(reservoirs.size), qm)
        ax.set_xticks(range(reservoirs.size))
        ax.set_xticklabels(reservoirs, rotation=45, ha="right")
        ax.set_ylabel(metric)

        if ax in axes[:3]:
            ax.set_title(f"{qbin+1}/3")
    
    plt.show()

def plot_coefs(results, args):
    coefs = results["coefs"]
    groups = coefs.columns
    xlabs = coefs.index
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 8.7))
    fig.patch.set_alpha(0.0)

    coefs.plot.bar(ax=ax, width=0.8)
    ax.set_xticks(range(xlabs.size))
    ax.set_xticklabels(xlabs, rotation=45, ha="right")

    plt.tight_layout()
    plt.show()
    

def parse_args(plot_functions):
    parser = argparse.ArgumentParser(description="Plot results from spatial model runs.")
    parser.add_argument("-p", "--plot_func", dest="plot_func", choices=plot_functions.keys(),
                        help="Choose the visualization that should be plotted. If not provided, launch an IPython shell",
                        default=None)
    parser.add_argument("--relative", action="store_true", default=False,
                        help="Plot values as percentages relative to their means.")
    parser.add_argument("-3", "--thirds", dest="thirds", action="store_true",
                        help="Plot data from thirds model instead of spatial model")
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    namespace = dir()
    plot_functions = find_plot_functions(namespace)
    args = parse_args(plot_functions)

    results = load_results(args)

    if not args.plot_func:
        II()
    else:
        globals()[plot_functions[args.plot_func]](results, args)