import glob
import re
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from IPython import embed as II


def find_depth(file):
    mat = re.search(r"TD(.*)_RT_MS", file)
    return mat.groups(1)[0]


def load_data(drop_keys=None, add_data=None):
    files = glob.glob("../results/tclr_model/all/TD*_exhaustive/results.pickle")
    data = {}
    for file in files:
        d = find_depth(file)
        data[d] = pd.read_pickle(file)["simmed_data"]
    if drop_keys:
        for k in drop_keys:
            del data[k]
    if add_data:
        for key, file in add_data.items():
            data[key] = pd.read_pickle(file)["simmed_data"]
    return data


def make_bin(x):
    if x <= 1/3:
        return 1
    elif x <= 2/3:
        return 2
    else:
        return 3


def get_res_bins(df):
    df["pct"] = df.groupby("site_name")["actual"].rank(pct=True)
    df["bin"] = df["pct"].apply(make_bin)
    return df

def calc_binned_scores(df, metric="NSE"):
    if metric == "NSE":
        mfunc = r2_score
    elif metric == "RMSE":
        mfunc = partial(mean_squared_error, squared=False)
    else:
        raise ValueError(f"Metric '{metric}' is not implemented.")

    df = get_res_bins(df)
    scores = df.groupby(["site_name", "bin"]).apply(
        lambda x: mfunc(x["actual"], x["model"])
    )
    return scores


def plot_binned_scores(data):
    binned_scores = {i: calc_binned_scores(j) for i, j in data.items()}
    bs_df = []
    for key, df in binned_scores.items():
        df = pd.DataFrame({"NSE": df})
        df["depth"] = int(key)
        bs_df.append(df)
    bs_df = pd.concat(bs_df).reset_index()

    sns.catplot(
        data=bs_df,
        x="depth",
        y="NSE",
        hue="bin",
        kind="box",
        whis=(0.05, 0.95),
        showfliers=False,
    )
    plt.show()


def make_binned_score_tables(data):
    binned_scores = {i: calc_binned_scores(j) for i, j in data.items()}
    bs_df = []
    for key, df in binned_scores.items():
        df = pd.DataFrame({"NSE": df})
        df["depth"] = int(key)
        bs_df.append(df)
    bs_df = pd.concat(bs_df).reset_index()

    for b in range(1, 4):
        df = bs_df[bs_df["bin"] == b]
        print(f"Bin: {b}")
        print(df.groupby("depth")["NSE"].describe().to_markdown(floatfmt="0.2f"))


def plot_scatter_perf(data):
    concat_dfs = []
    for depth, df in data.items():
        df["depth"] = depth
        concat_dfs.append(df)
    df = pd.concat(concat_dfs)

    ndepths = len(data)
    fg = sns.relplot(
        data=df,
        x="actual",
        y="model",
        hue="depth",
        kind="scatter",
        alpha=0.8,
        # palette=sns.color_palette("Spectral", ndepths)
    )

    fg.ax.axline([0, 0], [1, 1], c="r", linestyle="--")
    fg.ax.set_xlabel("Actual Release [1000 acre-ft/day]")
    fg.ax.set_ylabel("Modeled Release [1000 acre-ft/day]")
    fg.fig.patch.set_alpha(0.0)
    fg.ax.patch.set_alpha(0.0)
    plt.show()


def plot_parallel_coords(data):
    concat_dfs = []
    for depth, df in data.items():
        df["depth"] = depth
        concat_dfs.append(df)
    df = pd.concat(concat_dfs)
    df = df.groupby(["site_name", "depth"]).apply(
        lambda x: r2_score(x["actual"], x["model"])
    )
    df = df.groupby("depth").describe()
    df = df.drop("count", axis=1)
    df = df.reset_index()

    fig, ax = plt.subplots(1, 1)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    pd.plotting.parallel_coordinates(
        df,
        "depth",
        colormap=sns.color_palette("viridis", df.shape[0], as_cmap=True),
        ax=ax,
        lw=4
    )
    ax.set_ylabel("NSE")
    ax.legend(loc="lower left")

    plt.show()


def plot_parallel_coords_bin(data, metric="NSE"):
    binned_scores = {i: calc_binned_scores(j, metric=metric) for i, j in data.items()}
    bs_df = []
    for key, df in binned_scores.items():
        df = pd.DataFrame({metric: df})
        df["depth"] = key
        bs_df.append(df)
    bs_df = pd.concat(bs_df).reset_index()
    if metric == "RMSE":
        act_df = data[list(data.keys())[0]]
        means = act_df.groupby(["site_name", "bin"])["actual"].mean().abs()
        divisors = [means.loc[i, j] for i, j in bs_df[["site_name", "bin"]].values]
        bs_df["RMSE"] = bs_df["RMSE"] / divisors * 100
        # act = data[list(data.keys())[0]]["actual"]
        # means = act.groupby("site_name").mean()
        # divisors = [means[i] for i in bs_df["site_name"]]
        # bs_df["RMSE"] = bs_df["RMSE"] / divisors * 100

    # bs_df = bs_df.reset_index()

    df = bs_df.groupby(["bin", "depth"]).describe()
    # df = bs_df.groupby(["bin", "depth"])[metric].quantile([0.25, 0.5, 0.75, 0.9])

    df.columns = df.columns.droplevel(0)
    df = df.drop("count", axis=1)
    # df = df.drop(["mean", "std"], axis=1)
    df = df.reset_index()
    # df = df.rename(columns={"level_2":"Quantile"})
    # df = df.pivot(columns="Quantile", index=["bin", "depth"])
    # df.columns = df.columns.droplevel(0)
    # df = df.reset_index()

    fig, axes = plt.subplots(3, 1, sharex=True)
    fig.patch.set_alpha(0.0)

    titles = ["Below ", "", "Above "]

    for b, ax in zip(range(1, 4), axes):
        ax.patch.set_alpha(0.0)
        pdf = df.loc[df["bin"] == b, :].drop("bin", axis=1)
        pd.plotting.parallel_coordinates(
            pdf,
            "depth",
            colormap=sns.color_palette("viridis", df.shape[0], as_cmap=True),
            ax=ax,
            lw=2,
        )
        if metric == "RMSE":
            ax.set_ylabel(f"n{metric} [%]")
        else:
            ax.set_ylabel(metric)

        ax.legend(loc="lower right")
        ax.set_title(f"{titles[b - 1]}Normal Flow")

    fig.align_ylabels()

    plt.show()


def plot_binned_bar_plots(data, metric="NSE"):
    binned_scores = {i: calc_binned_scores(j, metric=metric) for i, j in data.items()}

    bs_df = []
    for key, df in binned_scores.items():
        df = pd.DataFrame({metric: df})
        df["depth"] = key
        bs_df.append(df)
    bs_df = pd.concat(bs_df).reset_index()
    if metric == "RMSE":
        act_df = data[list(data.keys())[0]]
        means = act_df.groupby(["site_name", "bin"])["actual"].mean().abs()
        divisors = [means.loc[i, j] for i, j in bs_df[["site_name", "bin"]].values]
        bs_df["RMSE"] = bs_df["RMSE"] / divisors * 100
        # act = data[list(data.keys())[0]]["actual"]
        # means = act.groupby("site_name").mean()
        # divisors = [means[i] for i in bs_df["site_name"]]
        # bs_df["RMSE"] = bs_df["RMSE"] / divisors * 100

    df = bs_df.groupby(["bin", "depth"]).describe()
    # df = bs_df.groupby(["bin", "depth"])[metric].quantile([0.25, 0.5, 0.75, 0.9])

    df.columns = df.columns.droplevel(0)
    df = df.drop("count", axis=1)
    # df = df.drop(["mean", "std"], axis=1)
    df = df.reset_index()
    # df = df.rename(columns={"level_2":"Quantile"})
    # df = df.pivot(columns="Quantile", index=["bin", "depth"])
    # df.columns = df.columns.droplevel(0)
    # df = df.reset_index()

    fig, axes = plt.subplots(3, 3, sharex=True, sharey="row")
    fig.patch.set_alpha(0.0)

    titles = ["Minimum", "Median", "Max"]
    bins = [1, 2, 3]
    btitles = ["Below Normal", "Normal", "Above Normal"]
    ax_titles = [[f"{t} [{b}]" for b in btitles] for t in titles]
    cols = ["min", "50%", "max"]

    for col, ax_row, ax_title in zip(cols, axes, ax_titles):
        pdf = df[["bin", "depth", col]]
        pdf = pdf.pivot(columns="depth", index="bin")
        pdf.columns = pdf.columns.droplevel(0)
        for b, ax, title in zip(bins, ax_row, ax_title):
            pdf.loc[b, :].plot.bar(ax=ax, legend=False)
            ax.set_xticklabels(pdf.columns, rotation=0)
            # if col == "min" and b == 3:
            #     ax.legend(loc="best", title="Model")
            # else:
            #     ax.get_legend().remove()
            ax.set_title(title)
            ax.set_xlabel("Model")
            if b == 1:
                ax.set_ylabel(metric)
    fig.align_ylabels()

    plt.show()


def plot_joint_scatter(data):
    concat_dfs = []
    for depth, df in data.items():
        df["depth"] = depth
        concat_dfs.append(df)
        if depth[:4] == "PLRT":
            plrtkey = depth
    df = pd.concat(concat_dfs)
    df = df.reset_index()
    # df["actual"] = np.log(df["actual"], out=np.zeros_like(df["actual"]), where=(df["actual"] != 0))
    # df["model"] = np.log(df["model"], out=np.zeros_like(df["model"]), where=(df["model"] != 0))

    plrt_idx = df[df["depth"] == plrtkey].index
    trm_idx = df[df["depth"] == "TRM"].index
    plrt_act = df.loc[plrt_idx, "actual"]
    plrt_mod = df.loc[plrt_idx, "model"].values.reshape(plrt_act.size, 1)
    plrt_x = np.array([np.ones_like(plrt_act), plrt_act])
    trm_act = df.loc[trm_idx, "actual"]
    trm_mod = df.loc[trm_idx, "model"].values.reshape(trm_act.size, 1)
    trm_x = np.array([np.ones_like(trm_act), trm_act])

    bf_plrt = np.linalg.inv(plrt_x @ plrt_x.T) @ (plrt_x @ plrt_mod)
    bf_trm = np.linalg.inv(trm_x @ trm_x.T) @ (trm_x @ trm_mod)

    act_space = np.linspace(df["actual"].min(), df["actual"].max(), 1000)
    act_x = np.array([np.ones_like(act_space), act_space]).T
    x_plrt = act_x @ bf_plrt
    x_trm = act_x @ bf_trm

    # fg = sns.jointplot(
    #     data=df,
    #     x="actual",
    #     y="model",
    #     hue="depth",
    #     # kind="scatter",
    #     alpha=0.8,
    # )

    fg = sns.relplot(
        data=df,
        x="actual",
        y="model",
        hue="depth",
        kind="scatter",
        alpha=0.8,
        facet_kws={
            "legend_out": False
        }
        # palette=sns.color_palette("Spectral", ndepths)
    )
    style_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fg.ax.axline([0, 0], [1, 1], lw=2, c="r", linestyle="--", label="1:1 Line")
    fg.ax.plot(act_space, x_plrt, lw=2, c=style_colors[4], linestyle="--", label=f"{plrtkey} LBF")
    fg.ax.plot(act_space, x_trm, lw=2, c=style_colors[5], linestyle="--", label="TRM LBF")
    fg.ax.set_xlabel("Actual Release [1000 acre-ft/day]")
    fg.ax.set_ylabel("Modeled Release [1000 acre-ft/day]")
    fg.ax.legend(loc="best")
    # fg.set_axis_labels(
    #     xlabel="Actual Release [1000 acre-ft/day]",
    #     ylabel="Modeled Release [1000 acre-ft/day]",
    # )
    fg.figure.patch.set_alpha(0.0)
    # for ax in fg.figure.get_axes():
    fg.ax.patch.set_alpha(0.0)
    fg.ax.set_ylim(-10, 800)
    plt.show()


def determine_text_ax_loc(ax, px, py):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x = xlim[0] + px * (xlim[1] - xlim[0])
    y = ylim[0] + py * (ylim[1] - ylim[0])
    return x, y

def plot_binned_scatter(data):
    for key in data.keys():
        if key[:4] == "PLRT":
            plrtkey = key

    plrt = get_res_bins(data[plrtkey])
    trm = get_res_bins(data["TRM"])

    fig, axes = plt.subplots(3, 2)
    # axes = axes.flatten();
    fig.patch.set_alpha(0.0)
    bins = [1, 2, 3]
    for ax_row, b in zip(axes, bins):
        ax_row[0].patch.set_alpha(0.0)
        ax_row[1].patch.set_alpha(0.0)
        plrt_df = plrt[plrt["bin"] == b]
        trm_df = trm[trm["bin"] == b]
        ax_row[0].scatter(plrt_df["actual"], plrt_df["model"], label=plrtkey)
        ax_row[1].scatter(trm_df["actual"], trm_df["model"], label="TRM")
        plrt_score = r2_score(plrt_df["actual"], plrt_df["model"])
        trm_score = r2_score(trm_df["actual"], trm_df["model"])
        ax_row[0].text(
            *determine_text_ax_loc(ax_row[0], 0.10, 0.85),
            f"NSE = {plrt_score:.3f}"
        )
        ax_row[1].text(
            *determine_text_ax_loc(ax_row[1], 0.10, 0.85),
            f"NSE = {trm_score:.3f}"
        )
        ax_row[0].axline([0, 0], [1, 1], lw=2, c="r", linestyle="--", label="1:1 Line")
        ax_row[1].axline([0, 0], [1, 1], lw=2, c="r", linestyle="--", label="1:1 Line")
        ax_row[0].set_title(f"{plrtkey} [Bin: {b}]")
        ax_row[1].set_title(f"TRM [Bin: {b}]")
        # if b == 3:
            # ax_row[0].legend(loc="best")
            # ax_row[0].set_xlabel("Actual Release [1000 acre-ft/day]")

    fig.text(0.02, 0.5, "Modeled Release [1000 acre-ft/day]", rotation=90, ha="center", va="center")
    fig.text(0.5, 0.02, "Actual Release [1000 acre-ft/day]", rotation=0, ha="center", va="center")
    plt.subplots_adjust(
        top=0.933,
        bottom=0.065,
        left=0.051,
        right=0.946,
        hspace=0.248,
        wspace=0.068
    )


    plt.show()

def plot_seasonal_performance(data, lines=False):
    scores = []
    for key, df in data.items():
        df["month"] = df.index.get_level_values("datetime").month
        score = df.groupby(["site_name", "month"]).apply(
            lambda x: r2_score(x["actual"], x["model"])
        )
        score = pd.DataFrame({"NSE": score})
        score["model"] = key
        scores.append(score)

    df = pd.concat(scores).reset_index()
    if not lines:
        fg = sns.catplot(
            data=df,
            x="month",
            y="NSE",
            hue="model",
            kind="box",
            showfliers=False,
            # hue_order=["TRM", "1", "2", "3", "4", "5"],
            hue_order=["TRM", "4" ],
            # whis=(0.1, 0.9)
            legend_out=False
        )
        fg.figure.patch.set_alpha(0.0)
        # fg.ax.patch.set_alpha(0.0)
    else:
        p = [0.1, 0.25, 0.5, 0.75, 0.9]
        quants = df.groupby(["month", "model"])["NSE"].quantile(p)
        quants = quants.reset_index().rename(columns={"level_2": "percentile"})

        style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        c = [style_colors[i] for i in range(len(p))]
        fig, ax = plt.subplots(1, 1)
        for percentile, color in zip(p, c):
            plrt_df = quants.loc[
                (quants["percentile"] == percentile) & (quants["model"] == "4"), :
            ]
            trm_df = quants.loc[
                (quants["percentile"] == percentile) & (quants["model"] == "TRM"), :
            ]
            ax.plot(plrt_df["month"], plrt_df["NSE"], label="PLRT4", marker="o", c=color)
            ax.plot(trm_df["month"], trm_df["NSE"], label="TRM", marker="s", c=color)
        ax.set_ylabel("NSE")
        ax.set_xlabel("Month")

        leglines = [mlines.Line2D([], [], c=color) for color in c]
        legmarkers = [plt.scatter([], [], marker="o", c=style_colors[0]),
                      plt.scatter([], [], marker="s", c=style_colors[0])]
        labels = [*p, "PLRT4", "TRM"]
        handles = [*leglines, *legmarkers]
        ax.legend(handles, labels, loc="best")
        # ax.patch.set_alpha(0.0)
        fig.patch.set_alpha(0.0)

    plt.show()


if __name__ == "__main__":
    plt.style.use("seaborn")
    sns.set_context("notebook")
    data = load_data(
        drop_keys=["1", "2", "3", "5", "6", "7", "8", "9", "10"],
        add_data={
            "TRM": "../results/three_reg_model/all/stovars_three_model/results.pickle"
        },
    )
    # plot_binned_scores(data)
    # make_binned_score_tables(data)
    # plot_scatter_perf(data)
    # plot_parallel_coords(data)
    # plot_parallel_coords_bin(data, metric="NSE")
    # plot_binned_bar_plots(data, metric="NSE")
    # plot_scatter_perf({"PLRT":data["3"], "TRM":data["TRM"]})
    # import sys
    # d = sys.argv[1]
    # plot_joint_scatter({f"PLRT{d}": data[d], "TRM": data["TRM"]})
    # plot_binned_scatter({f"PLRT{d}": data[d], "TRM": data["TRM"]})
    plot_seasonal_performance(data, lines=False)
