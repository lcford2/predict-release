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

# import my functions
from helper_functions import read_tva_data
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
    file = RESULTS_DIR / "synthesis" / "spatial_model" / "res_slopes.pickle"
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
            coefs.loc[coef], kde=True
        )
        ax.set_title(coef)
    
    plt.show()

def parse_args(plot_functions):
    parser = argparse.ArgumentParser(description="Plot results from spatial model runs.")
    parser.add_argument("-p", "--plot_func", dest="plot_func", choices=plot_functions.keys(),
                        help="Choose the visualization that should be plotted. If not provided, launch an IPython shell",
                        default=None)
    
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