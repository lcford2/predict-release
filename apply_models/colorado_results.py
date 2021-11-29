import glob
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime, timedelta
from plot_helpers import determine_grid_size, abline

from IPython import embed as II


sns.set_context("talk")
plt.style.use("ggplot")
style_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

BASIN_NAMES = {"upper_col":"Upper Colorado", "lower_col":"Lower Colorado",
               "pnw":"Pacific Northwest","missouri":"Missouri","tva":"TVA",
               "colorado":"Colorado"}

METRIC_TITLES = {"r2_score":"NSE", "rmse":"RMSE"}

def load_data():
    with open("./simulation_output/colorado_bsp_results.pickle", "rb") as f:
        norm_results = pickle.load(f)

    with open("./simulation_output/colorado_bsp_results_low_rt.pickle", "rb") as f:
        lrt_results = pickle.load(f)

    with open("./simulation_output/colorado_bsp_results_high_rt.pickle", "rb") as f:
        hrt_results = pickle.load(f)
    with open("./simulation_output/colorado_bsp_results_purposes.pickle", "rb") as f:
        purp_results = pickle.load(f)
    return norm_results, lrt_results, hrt_results, purp_results

def get_metrics(norm, lrt, hrt, purp, var="release", metric="r2_score"):
    normmetrics = norm["metrics"][var]
    lrtmetrics = lrt["metrics"][var]
    hrtmetrics = hrt["metrics"][var]
    prpmetrics = purp["metrics"][var]

    metrics = pd.DataFrame({
        "Normal":normmetrics[metric],
        "HRT":hrtmetrics[metric],
        "LRT":lrtmetrics[metric],
        "P-GRP":prpmetrics[metric]
    })
    return metrics

def fix_metric_index(metrics):
    metrics.index = metrics.index.str.title()
    metrics.index = [i if i != "Smith & Morehouse Reservoir"
                        else " ".join(i.split()[:-1])
                        for i in metrics.index]
    return metrics

def plot_metric_bars(data, var="release", metric="r2_score"):
    output = data[0]["output"][["release_act", "storage_act"]]
    output = output.rename(columns={"storage_act":"storage", "release_act":"release"})
    means = output.groupby(output.index.get_level_values(0)).mean()
    if metric == "both":
        nse = get_metrics(*data, var=var, metric="r2_score")
        nse = nse.dropna()
        nse = nse.sort_values(by="Normal")
        means = means.loc[nse.index,:]
        nse = fix_metric_index(nse)
        rmse = get_metrics(*data, var=var, metric="rmse")
        rmse = fix_metric_index(rmse)
        rmse = rmse.loc[nse.index]

        fig, axes = plt.subplots(2,1, figsize=(19,11), sharex=True)
        axes = axes.flatten()

        ax = axes[0]
        nse.plot.bar(ax=ax, width=0.8)
        ax.legend(loc="best")
        ax.set_ylabel(f"Simulation {var.capitalize()} {METRIC_TITLES['r2_score']}",
                    fontsize=16)
        ax.set_title(BASIN_NAMES["colorado"], fontsize=18)
        ax.axhline(nse["Normal"].median(), ls="--", c=style_colors[0])
        ax.axhline(nse["HRT"].median(), ls="--", c=style_colors[1])
        ax.axhline(nse["LRT"].median(), ls="--", c=style_colors[2])
        ax.axhline(nse["P-GRP"].median(), ls="--", c=style_colors[3])
        ax.set_ylim(-1.1, 1.1)

        ax = axes[1]
        # rmse = rmse.subtract(means[var].values, axis=0)
        # for col in rmse.columns:
        #     rmse[col] = rmse[col].divide(means[var].values) * 100

        rmse.plot.bar(ax=ax, width=0.8)
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(rmse.index, rotation=45, ha="right")
        ax.set_ylabel(f"Simulation {var.capitalize()} {METRIC_TITLES['rmse']}",
                    fontsize=16)
        ax.axhline(rmse["Normal"].median(), ls="--", c=style_colors[0])
        ax.axhline(rmse["HRT"].median(), ls="--", c=style_colors[1])
        ax.axhline(rmse["LRT"].median(), ls="--", c=style_colors[2])
        ax.axhline(rmse["P-GRP"].median(), ls="--", c=style_colors[3])
        ax.get_legend().remove()
        fig.align_ylabels()
    else:
        metrics = get_metrics(*data, var=var, metric=metric)
        metrics = metrics.dropna()
        metrics = fix_metric_index(metrics)
        metrics = metrics.sort_values(by="Normal")

        fig, ax = plt.subplots(1,1, figsize=(19,11))
        metrics.plot.bar(ax=ax, width=0.8)
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(metrics.index, rotation=45, ha="right")
        ax.legend(loc="best")
        ax.set_ylabel(f"Simulation {var.capitalize()} {METRIC_TITLES[metric]}",
                    fontsize=18)
        ax.set_title(BASIN_NAMES["colorado"])
        ax.axhline(metrics["Normal"].median(), ls="--", c=style_colors[0])
        ax.axhline(metrics["HRT"].median(), ls="--", c=style_colors[1])
        ax.axhline(metrics["LRT"].median(), ls="--", c=style_colors[2])
        ax.axhline(metrics["P-GRP"].median(), ls="--", c=style_colors[3])

        print(metrics.describe())

    plt.tight_layout()
    plt.show()

def plot_res_ts(data, var="release"):
    metrics = get_metrics(*data, var=var, metric="r2_score")
    metrics = metrics.dropna().sort_values(by="Normal")
    norm, lrt, hrt, purp = [
        i["output"][[var, f"{var}_act"]] for i in data]
    act = norm[f"{var}_act"].unstack().T.drop(["SANTA ROSA ", "DILLON RESERVOIR"], axis=1)
    norm = norm[var].unstack().T.drop(["SANTA ROSA ", "DILLON RESERVOIR"], axis=1)
    lrt = lrt[var].unstack().T.drop(["SANTA ROSA ", "DILLON RESERVOIR"], axis=1)
    hrt = hrt[var].unstack().T.drop(["SANTA ROSA ", "DILLON RESERVOIR"], axis=1)
    purp = purp[var].unstack().T

    resers = metrics.index
    idxs = [list(range(i,i+4)) for i in range(0, resers.size, 4)]

    for idx in idxs:
    # gs = determine_grid_size(purp.shape[1])
        fig, axes = plt.subplots(2,2, figsize=(19,11))
        axes = axes.flatten()
        reses = [resers[i] for i in filter(lambda x: x < resers.size, idx)]
        for res, ax in zip(reses, axes):
            ax.plot(act[res].dropna().index, act[res].dropna(), label="Actual")
            ax.plot(norm[res].dropna().index, norm[res].dropna(), label="Normal")
            ax.plot(hrt[res].dropna().index, hrt[res].dropna(), label="HRT")
            ax.plot(lrt[res].dropna().index, lrt[res].dropna(), label="LRT")
            ax.plot(purp[res].dropna().index, purp[res].dropna(), label="P-GRP")
            ax.set_title(res)
            ax.tick_params(axis="x", labelrotation=0)
        axes[0].legend(loc="best", ncol=2, prop={"size":12})
        fig.text(0.02, 0.5, var.capitalize(), ha="center", va="center", rotation=90,
                 fontsize=20)

        left_over = axes.size - len(reses)
        if left_over > 0:
            for ax in axes[-left_over:]:
                ax.set_axis_off()

        plt.subplots_adjust(
            top=0.965,
            bottom=0.05,
            left=0.079,
            right=0.985,
            hspace=0.17,
            wspace=0.1
        )
        plt.show()

    
if __name__ == "__main__":
    data = load_data()
    # plot_metric_bars(data, var="storage", metric="both")
    plot_res_ts(data, var="release")

# LOST CREEK = 4633
# DEER CREEK = 1980
