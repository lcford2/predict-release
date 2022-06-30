import calendar
from matplotlib.style.core import _remove_blacklisted_style_params
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from functools import partial
from IPython import embed as II


def load_data():
    folder = "../results/tclr_model/all/TD{depth}_RT_MS_exhaustive"
    out = {}
    for depth in [1, 4]:
        fdir = folder.format(depth=depth)
        file = "/".join([fdir, "results.pickle"])
        out[depth] = pd.read_pickle(file)
    out["TRM"] = pd.read_pickle(
        "../results/three_reg_model/all/stovars_three_model/results.pickle"
    )
    return out


def norm_mean_squared_error(act, mod, **kwargs):
    mean = np.mean(act)
    return mean_squared_error(act, mod, **kwargs) / mean


def calc_performance(data, res=False, season=False, metric="nse"):
    output = {}
    if metric == "nse":
        mfunc = r2_score
    elif metric == "rmse":
        mfunc = partial(mean_squared_error, squared=False)
    elif metric == "nrmse":
        mfunc = partial(norm_mean_squared_error, squared=False)
    else:
        raise ValueError(f"Metric {metric} not implemented.")

    for depth, results in data.items():
        df = results["simmed_data"]
        if res and season:
            scores = df.groupby(
                [df.index.get_level_values(1).month,
                 df.index.get_level_values(0)]
            ).apply(
                lambda x: mfunc(x["actual"], x["model"])
            )
        elif season:
            scores = df.groupby(df.index.get_level_values(1).month).apply(
                lambda x: mfunc(x["actual"], x["model"]))
        elif res:
            scores = df.groupby(df.index.get_level_values(0)).apply(
                lambda x: mfunc(x["actual"], x["model"])
            )
        else:
            scores = mfunc(df["actual"], df["model"])
        output[depth] = scores
    return pd.DataFrame(output)


if __name__ == "__main__":
    plt.style.use("ggplot")
    sns.set_context("talk")
    data = load_data()
    # sperf = calc_performance(data, res=True, season=True, metric="nrmse")
    best = "TRM"
    # sperf = sperf[best]
    rmse = calc_performance(data, res=True, season=True, metric="rmse")
    rmse = rmse[[best]]
    df = data[best]["simmed_data"]
    means = df.groupby([
        df.index.get_level_values(0),
        df.index.get_level_values(1).month
    ])["actual"].mean()
    means = means.abs()

    means = means.reset_index().set_index(["datetime", "site_name"])

    rmse["mean"] = means
    rmse = rmse.reset_index().rename(columns={
        best: "RMSE",
        "datetime": "Month",
    })

    seasons = {
        1: "Winter",
        2: "Winter",
        3: "Spring",
        4: "Spring",
        5: "Spring",
        6: "Summer",
        7: "Summer",
        8: "Summer",
        9: "Autumn",
        10: "Autumn",
        11: "Autumn",
        12: "Winter"
    }
    smap = ["Winter", "Spring", "Summer", "Autumn"]

    def get_season_month(season, month):
        if season == "Winter" or season == 1:
            return [12, 1, 2].index(month)
        elif season == "Spring" or season == 2:
            return [3, 4, 5].index(month)
        elif season == "Summer" or season == 3:
            return [6, 7, 8].index(month)
        elif season == "Autumn" or season == 4:
            return [9, 10, 11].index(month)

    rmse["Season"] = rmse["Month"].apply(lambda x: smap.index(seasons.get(x, "")) + 1)
    # rmse["Season"] = rmse["Month"].apply(lambda x: seasons.get(x, ""))
    rmse["smonth"] = rmse.apply(lambda x: get_season_month(x["Season"], x["Month"]), axis=1)
    # rmse.plot.scatter(x="mean", y="RMSE", c="Month", cmap="viridis")
    fg = sns.relplot(
        data=rmse,
        x="mean",
        y="RMSE",
        # col="Month",
        # col_wrap=4,
        hue="Season",
        kind="scatter",
        hue_order=range(1,13),
        # palette=sns.color_palette("RdBu", 12),
        palette="Spectral",
        alpha=0.8,
        facet_kws={"legend_out": False}

    )
    fg.set_xlabels("Mean Release [1000 acre-ft/day]")
    fg.set_ylabels("RMSE [1000 acre-ft/day]")
    handles, labels = fg.ax.get_legend_handles_labels()
    # fg.ax.legend(handles, [calendar.month_abbr[i] for i in range(1, 13)], title="Month", ncol=4)
    fg.ax.legend(handles, smap, title="Season", ncol=4)
    fg.fig.patch.set_alpha(0.0)
    # fg.ax.patch.set_alpha(0.0)

    rmse["Month"] = rmse["Month"].apply(lambda x: calendar.month_abbr[x])
    # fg = sns.catplot(
    #     data=rmse,
    #     x="RMSE",
    #     y="Month",
    #     kind="strip",
    #     s="mean",
    #     # hue="mean",
    #     legend=False
    # )

    # style_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # max_value = rmse["mean"].max()
    # min_value = rmse["mean"].min()
    # col_space = np.linspace(min_value, max_value, num=4)
    # max_size = 30
    # size_ratio = max_size / max_value

    # max_p_size = -float("inf")
    # min_p_size = float("inf")
    # for child in fg.ax.get_children():
    #     try:
    #         pmin = child.get_sizes().min()
    #         pmax = child.get_sizes().max()
    #         if pmin < min_p_size:
    #             min_p_size = pmin
    #         if pmax > min_p_size:
    #             max_p_size = pmax
    #     except AttributeError:
    #         pass

    # size_space = np.linspace(min_p_size, max_p_size, num=4)
    # color = fg.ax.get_children()[1].get_facecolors().tolist()

    # leg_markers = [
    #     plt.scatter([], [], s=i, c=[color[0]], alpha=0.9)
    #     for i in size_space
    # ]

    # leg_labels = [f"{i:.1f}" for i in col_space]

    # fg.ax.legend(leg_markers, leg_labels, title="Mean Release [1000 acre-ft/day]",
    #              ncol=4)
    # fg.ax.set_xlabel("RMSE [1000 acre-ft/day]")
    # fg.fig.patch.set_alpha(0.0)
    plt.show()
