import sys
import pickle
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot
from IPython import embed as II
from plot_helpers import determine_grid_size
from utils.helper_functions import swap_index_levels

plt.style.use("ggplot")
sns.set_context("talk")

def load_simul_results(basin):
    if basin == "TVA":
        with open("../results/simul_model/"
                  "sto_and_rel_results.pickle", "rb") as f:
            results = pickle.load(f)
    else:
        raise NotImplementedError(f"Basin {basin} not implemented.")
    return results

def residual_cf(data, storage=False, splitres=False,
                train=True, stdzd=False, plot_func=plot_acf):
    if train:
        data = data["train"]
        set_label = "Train"
    else:
        data = data["test"]
        set_label = "Test"

    if stdzd:
        means = data.groupby(data.index.get_level_values(0)).mean()
        stds = data.groupby(data.index.get_level_values(0)).std()
        data["Storage_act"] = ((data["Storage_act"].unstack().T - means["Storage_act"])
                               / stds["Storage_act"]).T.stack()
        data["Storage_simul"] = ((data["Storage_simul"].unstack().T - means["Storage_act"])
                               / stds["Storage_act"]).T.stack()
        data["Release_act"] = ((data["Release_act"].unstack().T - means["Release_act"])
                               / stds["Release_act"]).T.stack()
        data["Release_simul"] = ((data["Release_simul"].unstack().T - means["Release_act"])
                                 / stds["Release_act"]).T.stack()
        stdzd_label = "Standardized"
    else:
        stdzd_label = "Normal Space"

    if storage:
        resids = data["Storage_simul"] - data["Storage_act"]
        title = "Storage Residual"
    else:
        resids = data["Release_simul"] - data["Release_act"]
        title = "Release Residual"

    if plot_func == plot_pacf:
        title = " ".join([title, "Partial Autocorrelation"])
        plot_args = {}
        subplot_args = {"sharex":True, "sharey":True}
    elif plot_func == plot_acf:
        title = " ".join([title, "Autocorrelation"])
        plot_args = {}
        subplot_args = {"sharex":True, "sharey":True}
    elif plot_func == qqplot:
        title = " ".join([title, "Q-Q Plot"])
        plot_args = {"line":"q", "markerfacecolor":"b", "markeredgecolor":"b"}
        subplot_args = {"sharex":True, "sharey":False}


    if splitres:
        sns.set_context("paper")
        reservoirs = data.index.get_level_values(0).unique()
        gs = determine_grid_size(reservoirs.size)
        fig, axes = plt.subplots(*gs, **subplot_args)
        fig.patch.set_alpha(0.0)
        axes = axes.flatten()
        idx = pd.IndexSlice
        for res, ax in zip(reservoirs, axes):
            res_resid = resids.loc[idx[res,:]]
            plot_func(res_resid, ax=ax, **plot_args)
            ax.set_title(res)
            ax.set_xlabel("")
            ax.set_ylabel("")

        left_over = axes.size - reservoirs.size
        for ax in axes[-left_over:]:
            ax.set_axis_off()
        fig.suptitle(f"{set_label} {title} [{stdzd_label}]")
    else:
        fig = plot_func(resids, **plot_args)
        ax = plt.gca()
        ax.set_title(title)

    if plot_func == qqplot:
        fig.text(0.02, 0.5, "Sample Quantiles", ha="center", va="center", rotation=90)
        fig.text(0.5, 0.02, "Theoretical Quantiles", ha="center", va="center")
        plt.subplots_adjust(
            top=0.938,
            bottom=0.058,
            left=0.052,
            right=0.993,
            hspace=0.202,
            wspace=0.278
        )
    plt.show()



if __name__ == "__main__":
    try:
        basin = sys.argv[1]
        if len(sys.argv) == 3:
            storage = True if sys.argv[2] == "s" else False
        else:
            storage = False
    except IndexError as e:
        print("Usage:")
        print("python simul_residual_analyis.py <basin_name> [s]")
        print("e.g., python simul_residual_analysis.py TVA")
        sys.exit()

    data = load_simul_results(basin)
    data["train"] = swap_index_levels(data["train"]).sort_index()
    data["test"] = swap_index_levels(data["test"]).sort_index()
    residual_cf(data,
                storage=storage,
                train=True,
                splitres=True,
                stdzd=True,
                plot_func=plot_pacf)
