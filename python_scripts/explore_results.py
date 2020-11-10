import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import pandas as pd
import scaleogram as scg
import numpy as np
from datetime import timedelta
from statsmodels.graphics.regressionplots import abline_plot
from IPython import embed as II
import calendar

# setup plotting style and sizing
plt.style.use("seaborn-paper")
sns.set_context("talk")

metric_titles = {
    "score": "R-Squared",
    "const": "Intercept",
    "Inflow": "Inflow",
    "Release": "Release",
    "Storage": "Storage",
    "PrevStorage": "Previous Day's Storage",
    "PrevInflow": "Previous Day's Inflow"
}

# read results
results_dir = pathlib.Path("storage_m1_results")
with open(results_dir / "simple_regression_results.pickle", "rb") as f:
    results = pickle.load(f)

def split_results(results_dict):
    values_dict, preds_dict, metrics_dict = {}, {}, {}
    for key, value in results_dict.items():
        values_dict[key] = value["fittedvalues"]
        preds_dict[key] = value["preds"]
        metrics_dict[key] = {
            "score":value["score"],
            # "adj_score":value["adj_score"],
            "pred_score":value["pred_score"],
            "const": value["params"]["const"],
            "Inflow": value["params"]["Inflow"],
            "Release": value["params"]["Release"],
            "Storage": value["params"]["Storage"],
            # "PrevStorage": value["params"]["PrevStorage"],
            # "PrevInflow": value["params"]["PrevInflow"]
        }
    values_df = pd.DataFrame.from_dict(values_dict).T
    preds_df = pd.DataFrame.from_dict(preds_dict).T
    metrics_df = pd.DataFrame.from_dict(metrics_dict).T
    values_df = values_df.sort_index()
    preds_df = preds_df.sort_index()
    metrics_df = metrics_df.sort_index()
    return values_df, preds_df, metrics_df

def plot_metrics(metrics):
    fig, axes = plt.subplots(7, 1, sharex=True)
    axes = axes.flatten()
    plots = ["score", "const", "Inflow", "Release", "Storage", "PrevStorage", "PrevInflow"]
    titles = ["R-Squared", "Intercept", "Inflow",
              "Release", "Storage", "PrevStorage", "PrevInflow"]
    ylabels = ["Value", "Fitted Value",
               "Fitted Value", "Fitted Value", "Fitted Value",
               "Fitted Value", "Fitted Value"]
    metrics = metrics[metrics.index.year >= 1991]
    for ax, plot, title, ylabel in zip(axes, plots, titles, ylabels):
        metrics[plot].plot(ax=ax)
        ax.tick_params(axis="x", which="minor", bottom=False)
        if plot == "score":
            ax.set_yticks([-2,-1,0,1])
        # ax.get_legend().remove()
        ax.set_title(title)
        # ax.set_ylabel(ylabel)
    # fig.align_ylabels()
    plt.show()

def plot_scaleogram(metrics, key="score"):
    fig, ax = plt.subplots(1, 1, figsize=(20, 15))

    t0 = metrics.index[0]
    scg.cws(metrics[key], scales=np.arange(1, 365*3), 
            ylabel="Period [Days]", xlabel="Date",
            coikw=dict(alpha=0.5), ax=ax)
    xticks = list(ax.get_xticks())
    new_labels = [t0 + timedelta(days=i) for i in xticks]
    new_labels = [i.strftime("%Y-%m") for i in new_labels]
    ax.set_xticklabels(new_labels)
    ax.set_title(metric_titles[key])
    plt.show()

def plot_reservoir_fit(value_df, resid=True, versus=True):
    sns.set_context("paper")
    df = pd.read_pickle("../pickles/tva_dam_data.pickle")
    df = df["Release"].unstack().loc[value_df.index, value_df.columns]
    # II()
    fig, axes = plt.subplots(4,7)
    axes = axes.flatten()
    for ax, column in zip(axes, value_df.columns):
        if resid:
            plt_df = (df[column] - value_df[column])#/df[column] * 100
            plt_df.plot(ax=ax)
        else:
            if versus:
                ax.scatter(df[column], value_df[column])
                abline_plot(0,1,ax=ax,c="r",linestyle="--")
            else:
                df[column].plot(ax=ax)
                value_df[column].plot(ax=ax, alpha=0.6)
        ax.set_title(column)
    
    axes[-1].axis("off")
    plt.show()


def plot_reservoir_fit_preds(value_df, preds_df):
    sns.set_context("paper")
    df = pd.read_pickle("../pickles/tva_dam_data.pickle")
    df = df["Release"].unstack().loc[value_df.index, value_df.columns]
    # II()
    fig, axes = plt.subplots(4, 7)
    axes = axes.flatten()
    for ax, column in zip(axes, value_df.columns):
        # df[column].plot(ax=ax)
        value_df[column].plot(ax=ax, alpha=1)
        preds_df[column].plot(ax=ax, alpha=0.6)
        ax.set_title(column)

    axes[-1].axis("off")
    plt.show()


def plot_monthly_metrics(metrics, key="score"):
    ylabels = {
        "score": "Value",
        "const": "Fitted Value",
        "Inflow": "Fitted Value",
        "Release": "Fitted Value",
        "Storage": "Fitted Value",
        "PrevInflow": "Fitted Value",
        "PrevStorage": "Fitted Value"
    }
    if key == "all":
        plot_metrics = metrics.groupby(metrics.index.month).mean()
        fig, axes = plt.subplots(3, 2, sharex="col")
        axes = axes.flatten()
        keys = ["score", "Storage", "Release", "Inflow", "PrevStorage", "PrevInflow"]
        for ax, metric in zip(axes, keys):
            plot_metrics[metric].plot.bar(ax=ax, width=0.9)
            ax.set_title(metric_titles[metric])
            ax.set_ylabel(ylabels[metric])
            # ax.set_xticks(list(map(int, list(np.linspace(1,366,num=25)))))
            ax.set_xticklabels(calendar.month_abbr[1:], ha="right", rotation=45)
        fig.align_ylabels()
        plt.show()
    else:
        plot_metrics = metrics[key].groupby(metrics.index.month).mean()
        fig, ax = plt.subplots(1,1)
        plot_metrics.plot.bar(ax=ax,width=0.9)
        ax.set_title(metric_titles[key])
        ax.set_ylabel(ylabels[key])
        ax.set_xticklabels(calendar.month_abbr[1:], ha="right", rotation=45)
        plt.show()
    

if __name__ == "__main__":
    values_df, preds_df, metrics_df = split_results(results)
    II()
    # plot_reservoir_fit_preds(values_df, preds_df)
    # plot_reservoir_fit(values_df, resid=False, versus=True)
    # plot_metrics(metrics_df)
    # plot_scaleogram(metrics_df, key="PrevInflow")
    # plot_monthly_metrics(metrics_df, key="all")
