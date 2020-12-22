import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import pandas as pd
import scaleogram as scg
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from datetime import timedelta
from statsmodels.graphics.regressionplots import abline_plot
from IPython import embed as II
import calendar

# setup plotting style and sizing
plt.style.use("seaborn-paper")
sns.set_context("talk")

metric_titles = {
    "score": "R-Squared",
    "pred_score": "R-Squared",
    "const": "Intercept",
    "Inflow": "Inflow",
    "Release": "Release",
    "Storage": "Storage",
    "PrevStorage": "Previous Day's Storage",
    "PrevInflow": "Previous Day's Inflow",
    "spearmanr": "Spearman r",
    "pearsonr": "Pearson r",
    "bias": r"Bias [$ft^3/day$]",
    "RelBias": "Relative Bias"
}

# read results
results_dir = pathlib.Path("one_day_results")
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
            "spearmanr": value["spearmanr"][0],
            "pearsonr": value["pearsonr"][0],
            "bias": value["bias"]
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

def plot_scaleogram(metrics, key="score", ax=None):
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(20, 15))
        show = True
    else:
        show = False

    t0 = metrics.index[0]
    scg.cws(metrics[key], scales=np.arange(1, 365*3), 
            ylabel="Period [Days]", xlabel="Date",
            coikw=dict(alpha=0.5), ax=ax)
    xticks = list(ax.get_xticks())
    new_labels = [t0 + timedelta(days=i) for i in xticks]
    new_labels = [i.strftime("%Y-%m") for i in new_labels]
    ax.set_xticklabels(new_labels)
    ax.set_title(metric_titles[key])

    if show:
        plt.show()

def plot_reservoir_fit(value_df, resid=True, versus=True):
    sns.set_context("paper")
    df = pd.read_pickle("../pickles/tva_dam_data.pickle")
    df = df["Release"].unstack().loc[value_df.index, value_df.columns] * 86400
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
                df[column].plot(ax=ax, label="Actual")
                value_df[column].plot(ax=ax, alpha=0.6, label="Modeled")
        ax.set_title(column)
    
    axes[-1].axis("off")
    handles, labels = axes[-2].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="center")
    plt.show()


def plot_reservoir_fit_preds(value_df, preds_df, versus=False):
    sns.set_context("paper")
    df = pd.read_pickle("../pickles/tva_dam_data.pickle")
    df = df["Release"].unstack().loc[value_df.index, value_df.columns]
    # II()
    fig, axes = plt.subplots(4, 7)
    axes = axes.flatten()
    for ax, column in zip(axes, value_df.columns):
        # df[column].plot(ax=ax)
        if versus:
            ax.scatter(value_df[column], preds_df[column])
            abline_plot(0,1,ax=ax,c="r",linestyle="--")
        else:
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
        "PrevStorage": "Fitted Value",
        "spearmanr": "Spearman r",
        "pearsonr": "Pearson r",
        "bias": "Bias"
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

def plot_pred_diagnostics(metrics_df, diags=True, hist=False, cumulative=False, quan=None):
    if diags:
        metrics = ["pred_score", "pearsonr", "bias"]
    else:
        metrics = ["Inflow", "Storage", "Release", "const"]
    if hist:
        sharex=False
    else:
        sharex=True
    if quan:
        total_rel = metrics_df["TotalInflow"]
        lower_quan = total_rel.quantile(quan)
        upper_quan = total_rel.quantile(1 - quan)
        lower_index = total_rel[total_rel <= lower_quan].index
        upper_index = total_rel[total_rel >= upper_quan].index
        rem_index = total_rel[(total_rel > lower_quan) &
                            (total_rel < upper_quan)].index
        regime = []
        for i in metrics_df.index:
            if i in lower_index:
                regime.append("Low Flow")
            elif i in upper_index:
                regime.append("High Flow")
            else:
                regime.append("Normal Flow")
        metrics_df["Flow Regime"] = regime
        fig, axes = plt.subplots(len(metrics), 3, sharex=sharex, sharey="row")
    else:    
        fig, axes = plt.subplots(len(metrics), 1, sharex=sharex)
        axes = axes.flatten()
    for ax, metric in zip(axes, metrics):
        if hist:
            if quan:
                metrics_df = relative_bias(metrics_df, quan=quan)
                if metric == "bias":
                    metric = "RelBias"
                for i, (axis, flow) in enumerate(zip(ax, ["Low Flow", "Normal Flow", "High Flow"])):
                    plot_metric = metrics_df[metrics_df["Flow Regime"] == flow][metric]
                    
                    sns.histplot(plot_metric, ax=axis, stat="probability", cumulative=cumulative, element="bars", linewidth=0.0)
                    axis.set_xlabel(metric_titles[metric])
                    if metric in ["pred_score", "Inflow"]:
                        axis.set_title(flow)
                    if diags:
                        axis.set_xlim(-1.1, 1.1)
                    else:
                        xlims = axis.get_xlim()
                        xmax = max([abs(xlims[0]), abs(xlims[1])])
                        axis.set_xlim(-xmax, xmax)
                    axis.axvline(plot_metric.median(),
                                 c="r", linestyle="--")
            else:
                sns.histplot(metrics_df[metric], ax=ax, stat="probability", cumulative=cumulative)
                ax.set_xlabel(metric_titles[metric])    
        else:
            metrics_df[metric].plot(ax=ax)
            ax.set_ylabel(metric_titles[metric])    
            ax.set_title(metric_titles[metric])
    
    fig.align_ylabels()
    plt.show()
    # fig, axes = plt.subplots(3, 1, sharex=True)
    # axes = axes.flatten()
    # for ax, metric in zip(axes, metrics):
    #     plot_scaleogram(metrics_df, key=metric, ax=ax)
    #     # ax.set_ylabel(metric_titles[metric])
    #     # ax.set_title(metric_titles[metric])

    # fig.align_ylabels()
    # plt.show()

def flow_regime_analysis(values_df, preds_df, quan=0.1):
    total_rel = values_df.sum(axis=1)
    lower_quan = total_rel.quantile(quan)
    upper_quan = total_rel.quantile(1-quan)
    lower_index = total_rel[total_rel <= lower_quan].index
    upper_index = total_rel[total_rel >= upper_quan].index
    rem_index = total_rel[(total_rel > lower_quan) & (total_rel < upper_quan)].index
    # scores
    lower_score = r2_score(values_df.loc[lower_index].values.flatten(), 
                             preds_df.loc[lower_index].values.flatten())
    upper_score = r2_score(values_df.loc[upper_index].values.flatten(),
                             preds_df.loc[upper_index].values.flatten())
    rem_score = r2_score(values_df.loc[rem_index].values.flatten(),
                             preds_df.loc[rem_index].values.flatten())
                             
    # pearson correlations
    lower_corr = pearsonr(values_df.loc[lower_index].values.flatten(),
                             preds_df.loc[lower_index].values.flatten())
    upper_corr = pearsonr(values_df.loc[upper_index].values.flatten(),
                             preds_df.loc[upper_index].values.flatten())
    rem_corr = pearsonr(values_df.loc[rem_index].values.flatten(),
                         preds_df.loc[rem_index].values.flatten())
    
    # bias
    def calc_bias(true, pred, relative=False):
        if relative:
            return (np.mean(pred) - np.mean(true)) / np.mean(true)
        else:
            return np.mean(pred) - np.mean(true)

    lower_bias = calc_bias(values_df.loc[lower_index].values.flatten(),
                          preds_df.loc[lower_index].values.flatten())
    upper_bias = calc_bias(values_df.loc[upper_index].values.flatten(),
                          preds_df.loc[upper_index].values.flatten())
    rem_bias = calc_bias(values_df.loc[rem_index].values.flatten(),
                        preds_df.loc[rem_index].values.flatten())

    lower_bias_rel = calc_bias(values_df.loc[lower_index].values.flatten(),
                           preds_df.loc[lower_index].values.flatten(), relative=True)
    upper_bias_rel = calc_bias(values_df.loc[upper_index].values.flatten(),
                           preds_df.loc[upper_index].values.flatten(), relative=True)
    rem_bias_rel = calc_bias(values_df.loc[rem_index].values.flatten(),
                         preds_df.loc[rem_index].values.flatten(), relative=True)

    results = pd.DataFrame(0.0, index=["Upper", "Middle", "Lower"], columns=["Score", "Corr", "Bias", "RelativeBias"])
    
    results.loc["Upper", "Score"] = upper_score
    results.loc["Upper", "Corr"] = upper_corr[0]
    results.loc["Upper", "Bias"] = upper_bias
    results.loc["Upper", "RelativeBias"] = upper_bias_rel

    results.loc["Lower", "Score"] = lower_score
    results.loc["Lower", "Corr"] = lower_corr[0]
    results.loc["Lower", "Bias"] = lower_bias
    results.loc["Lower", "RelativeBias"] = lower_bias_rel
    
    results.loc["Middle", "Score"] = rem_score
    results.loc["Middle", "Corr"] = rem_corr[0]
    results.loc["Middle", "Bias"] = rem_bias
    results.loc["Middle", "RelativeBias"] = rem_bias_rel

    return results

def non_exceedance_plot(Z, ylabel=None, xlabel=None, ax=None):
    if not ylabel:
        ylabel = "Non-Exceedance Probability"    
    N = len(Z)
    Z.sort()
    x = np.arange(1, N+1)
    y = x/N
    if not ax:
        ax = plt.gca()
        show = True
    else:
        show = False
    ax.plot(Z, y)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if show:
        plt.show()

def preds_diag_NE_plot(metrics_df):
    fig, axes = plt.subplots(3,1)
    axes = axes.flatten()
    metrics = ["pred_score", "pearsonr", "bias"]
    ylabels = [" ", "", " "]
    for ax, metric, ylabel in zip(axes, metrics, ylabels):
        non_exceedance_plot(metrics_df[metric].values, ylabel=ylabel, xlabel=metric_titles[metric], ax=ax)
    plt.show()

def paired_dens(metrics_df):
    params = ["const", "Release", "Storage", "Inflow"]
    g = sns.PairGrid(metrics_df[params], diag_sharey=False)
    g.map_upper(sns.scatterplot)
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.histplot)
    plt.show()


def relative_bias(metrics_df, quan=0.33):
    dam_data = pd.read_pickle("../pickles/tva_dam_data.pickle")
    release = dam_data["Release"].unstack() * 86400
    total_release = release.sum(axis=1)
    total_inflow = metrics_df["TotalInflow"]
    lower_quan = total_inflow.quantile(quan)
    upper_quan = total_inflow.quantile(1 - quan)
    lower_index = total_inflow[total_inflow <= lower_quan].index
    upper_index = total_inflow[total_inflow >= upper_quan].index
    rem_index = total_inflow[(total_inflow > lower_quan) &
                            (total_inflow < upper_quan)].index
    lower_release = total_release.loc[lower_index].mean()
    upper_release = total_release.loc[upper_index].mean()
    rem_release = total_release.loc[rem_index].mean()

    rel_bias = []
    for i, row in metrics_df.iterrows():
        if i in lower_index:
            rel_bias.append(row["bias"] / lower_release)
        elif i in upper_index:
            rel_bias.append(row["bias"] / upper_release)
        else:
            rel_bias.append(row["bias"] / rem_release)
    metrics_df["RelBias"] = rel_bias
    return metrics_df

if __name__ == "__main__":
    values_df, preds_df, metrics_df = split_results(results)
    dam_data = pd.read_pickle("../pickles/tva_dam_data.pickle")
    inflow = dam_data["Net Inflow"].unstack()
    metrics_df["TotalInflow"] = inflow.sum(axis=1) * 86400
    flows = flow_regime_analysis(values_df, preds_df, quan=0.33)
    # II()
    # plot_reservoir_fit_preds(values_df, preds_df, versus=True)
    # plot_reservoir_fit(values_df, resid=False, versus=True)
    # plot_monthly_metrics(metrics_df, key="spearmanr")
    # plot_scaleogram(metrics_df, key="PrevInflow")
    # plot_monthly_metrics(metrics_df, key="all")
    # plot_pred_diagnostics(metrics_df, diags=True, hist=True, cumulative=True, quan=0.3333)
    # preds_diag_NE_plot(metrics_df)
    paired_dens(metrics_df)
    # II()
