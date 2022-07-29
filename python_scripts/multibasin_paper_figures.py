import calendar
import json
import os
import pathlib
import pickle
import socket
from datetime import datetime

hostname = socket.gethostname()
if hostname == "CCEE-DT-094":
    os.environ[
        "PROJ_LIB"
    ] = r"C:\\Users\\lcford2\\AppData\\Local\\Continuum\\anaconda3\\envs\\sry-env\\Library\\share"
elif hostname == "inspiron-laptop":
    os.environ[
        "PROJ_LIB"
    ] = r"C:\\Users\\lcford\\miniconda3\\envs\\sry-env\\Library\\share"

import sys

import geopandas as gpd
import matplotlib.gridspec as GS
import matplotlib.lines as mlines
import matplotlib.patches as mpatch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython import embed as II
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from mpl_toolkits.basemap import Basemap
from scipy.stats import boxcox
from sklearn.metrics import mean_squared_error, r2_score
from utils.helper_functions import (
    ColorInterpolator,
    linear_scale_values,
    make_bin_label_map,
)

if hostname == "CCEE-DT-094":
    GIS_DIR = pathlib.Path("G:/My Drive/PHD/GIS")
elif hostname == "inspiron13":
    GIS_DIR = pathlib.Path("/home/lford/data/GIS")


def load_pickle(file):
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data


def write_pickle(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def read_results():
    path_format = "../results/tclr_model_drop_res_sto_diff_pers_testing_minsamples/all/TD{}_RT_MS_exhaustive/results.pickle".format
    models = {
        # "TD0": path_format(0),
        "TD1": path_format(1),
        "TD2": path_format(2),
        "TD3": path_format(3),
        "TD4": path_format(4),
        "TD5": path_format(5),
        "TD6": path_format(6),
        "TD7": path_format(7),
        "TD8": path_format(8),
        "TD9": path_format(9),
        # "TRM": "../results/three_reg_model/all/stovars_three_model/results.pickle"
    }

    return {i: load_pickle(j) for i, j in models.items()}


def select_results(results, get_item):
    return {i: j[get_item] for i, j in results.items()}


def mean_absolute_scaled_error(yact, ymod):
    error = (ymod - yact).abs().mean()
    yact = np.array(yact)
    lagerror = np.absolute(yact[1:] - yact[:-1]).mean()
    return error / lagerror


def trmse(act, mod):
    tact = boxcox(act, 0.3)
    tmod = boxcox(mod, 0.3)
    return mean_squared_error(tact, tmod, squared=False)


def get_r2score(df, grouper=None):
    if grouper:
        return pd.DataFrame(
            {
                "NSE": df.groupby(grouper).apply(
                    lambda x: r2_score(x["actual"], x["model"])
                )
            }
        )
    else:
        return r2_score(df["actual"], df["model"])


def get_rmse(df, grouper=None):
    if grouper:
        return pd.DataFrame(
            {
                "RMSE": df.groupby(grouper).apply(
                    lambda x: mean_squared_error(x["actual"], x["model"], squared=False)
                )
            }
        )
    else:
        return mean_squared_error(df["actual"], df["model"], squared=False)


def get_nrmse(df, grouper=None):
    normer = "mean"
    if grouper:
        return pd.DataFrame(
            {
                "nRMSE": df.groupby(grouper).apply(
                    lambda x: mean_squared_error(x["actual"], x["model"], squared=False)
                    / getattr(x["actual"], normer)()
                )
            }
        )
    else:
        return (
            mean_squared_error(df["actual"], df["model"], squared=False)
            / getattr(df["actual"], normer)()
        )


def get_mase(df, grouper=None):
    if grouper:
        return pd.DataFrame(
            {
                "MASE": df.groupby(grouper).apply(
                    lambda x: mean_absolute_scaled_error(x["actual"], x["model"])
                )
            }
        )
    else:
        return mean_absolute_scaled_error(df["actual"], df["model"])


def get_ntrmse(df, grouper=None):
    normer = "mean"
    if grouper:
        return pd.DataFrame(
            {
                "nTRMSE": df.groupby(grouper).apply(
                    lambda x: mean_squared_error(x["actual"], x["model"], squared=False)
                    / np.mean(boxcox(x["actual"], 0.3))
                )
            }
        )
    else:
        return mean_squared_error(df["actual"], df["model"], squared=False) / np.mean(
            boxcox(df["actual"], 0.3)
        )


def get_model_scores(model_dfs, metric="NSE", grouper=None):
    if metric == "NSE":
        return {i: get_r2score(j, grouper) for i, j in model_dfs.items()}
    elif metric == "RMSE":
        return {i: get_rmse(j, grouper) for i, j in model_dfs.items()}
    elif metric == "MASE":
        return {i: get_mase(j, grouper) for i, j in model_dfs.items()}
    elif metric == "nRMSE":
        return {i: get_nrmse(j, grouper) for i, j in model_dfs.items()}


def combine_dict_to_df(dct, colname):
    output = []
    for key, df in dct.items():
        df[colname] = key
        output.append(df)
    return pd.concat(output)


def get_name_replacements():
    with open("../pnw_data/dam_names.json", "r") as f:
        pnw = json.load(f)

    with open("../missouri_data/dam_names.json", "r") as f:
        missouri = json.load(f)

    tva = {}
    with open("./actual_names.csv", "r") as f:
        for line in f.readlines():
            line = line.strip("\n\r")
            key, value = line.split(",")
            tva[key] = value
    return pnw | tva | missouri


def make_bin(df, res=True):
    def bins(x):
        if x <= 1 / 3:
            return 1
        elif x <= 2 / 3:
            return 2
        else:
            return 3

    if res:
        df["pct"] = df.groupby("site_name")["actual"].rank(pct=True)
    else:
        df["pct"] = df["actual"].rank(pct=True)

    df["bin"] = df["pct"].apply(bins)
    return df


def plot_performance_boxplots(results):
    # calculating the mean and std of the performance across basins
    # rank each of these values (1 being best and n being worst)
    # calculate per depth means of these for the mean and the std
    # add the two together and take the smallest number as the best model for that metric
    # nRMSE - TD2 is best
    # NSE - TD3 is best
    # MASE - TD2 and TD3 are tied
    # RMSE - TD4 is best
    metric = "NSE"
    train = select_results(results, "train_data")
    test = select_results(results, "test_data")
    simul = select_results(results, "simmed_data")
    train_score = get_model_scores(train, metric=metric, grouper="site_name")
    test_score = get_model_scores(test, metric=metric, grouper="site_name")
    simul_score = get_model_scores(simul, metric=metric, grouper="site_name")

    train_score = combine_dict_to_df(train_score, "Model").reset_index()
    test_score = combine_dict_to_df(test_score, "Model").reset_index()
    simul_score = combine_dict_to_df(simul_score, "Model").reset_index()

    scores = combine_dict_to_df(
        {"Train": train_score, "Test": test_score, "Simulation": simul_score}, "Data Set"
    )
    from tclr_model import get_basin_meta_data

    meta = get_basin_meta_data("all")
    scores = scores.set_index("site_name")
    scores[["rts", "max_sto"]] = meta[["rts", "max_sto"]]

    rbasins = pd.read_pickle("../pickles/res_basin_map.pickle")
    rename = {
        "upper_col": "colorado",
        "lower_col": "colorado",
        "pnw": "columbia",
        "tva": "tennessee",
    }
    rbasins = rbasins.replace(rename)
    rbasins = rbasins.str.capitalize()

    scores["basin"] = rbasins
    # scores = combine_dict_to_df({"Simulation": simul_score}, "Data Set")
    sim_scores = scores[scores["Data Set"] == "Simulation"]
    ss_mean = sim_scores.groupby(["basin", "Model"]).mean()[metric]
    ss_std = sim_scores.groupby(["basin", "Model"]).std()[metric]
    if metric == "NSE":
        ascending = False
    else:
        ascending = True
    ss_mean_rank = ss_mean.unstack().T.rank(ascending=ascending)
    ss_std_rank = ss_std.unstack().T.rank(ascending=ascending)
    perf = ss_mean_rank.mean(axis=1) + ss_std_rank.mean(axis=1)

    fg = sns.catplot(
        data=scores,
        y=metric,
        hue="Data Set",
        # x="Data Set",
        x="Model",
        # hue_order=["TD1", "TD2", "TD3", "TD4", "TD5", "TD6"],
        hue_order=["Train", "Test", "Simulation"],
        legend_out=False,
        kind="box",
        showfliers=False,
        whis=(0.01, 0.99),
    )
    fg.set_xlabels("")
    fg.ax.legend(loc="best")
    plt.show()


def plot_variable_correlations():
    from tclr_model import get_basin_meta_data, read_basin_data

    df = read_basin_data("all")

    inf_corr = {}
    inflow = df.loc[:, ["release", "inflow"]]
    lags = range(0, 15)
    for i in lags:
        inflow[i] = inflow["inflow"].groupby("site_name").shift(i)

    inf_corr = inflow.groupby("site_name").corr()["release"].unstack()
    inf_corr = inf_corr.drop(["release", "inflow"], axis=1)
    inf_corr_mean = inf_corr.median()
    inf_corr_quants = inf_corr.quantile([0.25, 0.75])
    inf_corr_errbar = (inf_corr_quants - inf_corr_mean).abs()

    rbasins = pd.read_pickle("../pickles/res_basin_map.pickle")
    rename = {
        "upper_col": "colorado",
        "lower_col": "colorado",
        "pnw": "columbia",
        "tva": "tennessee",
    }
    rbasins = rbasins.replace(rename)
    rbasins = rbasins.str.capitalize()
    inf_corr["basin"] = rbasins
    # inf_corr["basin"] = inf_corr["basin"].replace(rename)
    # inf_corr["basin"] = inf_corr["basin"].str.capitalize()

    inf_corr = inf_corr.melt(id_vars=["basin"], var_name="Lag", value_name="Correlation")

    st = df.loc[:, ["release", "storage_pre", "storage_roll7", "storage_x_inflow"]]
    st["sto_diff"] = df["storage_pre"] - df["storage_roll7"]

    drop_res = [
        "Causey",
        "Lost Creek",
        "Echo",
        "Smith & Morehouse Reservoir",
        "Jordanelle",
        "Deer Creek",
        "Hyrum",
        "Santa Rosa ",
    ]
    drop_res = [i.upper() for i in drop_res]

    st = st[~st.index.get_level_values(0).isin(drop_res)]
    stcorr = st.groupby("site_name").corr()["release"].unstack()
    stcorr = stcorr.drop("release", axis=1)

    # sns.set_context("notebook")
    fig = plt.figure()
    fig.set_alpha(0.0)
    gs = GS.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[10, 0.5])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])]
    ex_ax = fig.add_subplot(gs[:, 1])
    ex_ax.patch.set_alpha(0.0)
    ex_x = np.random.normal(0, 1, 10000)
    sns.boxplot(y=ex_x, showfliers=False, ax=ex_ax)
    ex_ax.tick_params(
        axis="both", bottom=False, top=False, left=False, right=False, labelleft=False
    )
    # ex_ax.text(-0.5, 0.680, r"75th %ile")
    # ex_ax.text(-0.5, -0.665, r"25th %ile")

    basins = ["Columbia", "Missouri", "Colorado", "Tennessee"]
    colors = sns.color_palette("tab10")
    ax = axes[0]

    axes[0].axhline(0, c="k", linewidth=2, zorder=1, linestyle="--")
    axes[1].axvline(0, c="k", linewidth=2, zorder=1, linestyle="--")

    sns.boxplot(
        data=inf_corr,
        x="Lag",
        y="Correlation",
        hue="basin",
        palette="tab10",
        whis=(0.05, 0.95),
        ax=ax,
        showfliers=False,
    )
    axes[0].legend(loc="lower right", ncol=4)

    stcorr["basin"] = rbasins

    sns.boxplot(
        data=stcorr.melt(id_vars=["basin"]),
        y="variable",
        x="value",
        hue="basin",
        palette="tab10",
        ax=axes[-1],
        whis=(0.05, 0.95),
        showfliers=False,
    )
    axes[1].legend(loc="lower right", ncol=4)

    axes[0].set_ylabel("$r(R_t, I_L)$")
    axes[0].set_xlabel("Lag $L$ [days]")

    axes[-1].set_xlabel(r"Pearson's $r$ with Release")
    axes[-1].set_ylabel("")
    axes[-1].set_yticklabels(
        [
            r"$S_{t-1}$",
            r"$\bar{S}_{t-1}^7$",
            r"$S_{t-1} \times I_{t}$",
            r"$S_{t-1} - \bar{S}_{t-1}^7$",
        ]
    )

    style_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    handles = [
        mpatch.Patch(facecolor=style_colors[0]),
        mlines.Line2D([], [], linewidth=1.5, color="k"),
    ]
    labels = [r"Median Pearson's $r$", "25 - 75 quantiles"]
    plt.show()


def make_map(ax=None, coords=None, other_bound=None):
    if not ax:
        ax = plt.gca()

    if coords:
        west, south, east, north = coords
    else:
        west, south, east, north = (-127.441406, 24.207069, -66.093750, 49.382373)
    m = Basemap(
        # projection="merc",
        epsg=3857,
        resolution="c",
        llcrnrlon=west,
        llcrnrlat=south,
        urcrnrlon=east,
        urcrnrlat=north,
        ax=ax,
    )
    parallels = np.arange(0.0, 81, 10.0)
    meridians = np.arange(10.0, 351.0, 20.0)
    pvals = m.drawparallels(parallels, linewidth=0.0, labels=[1, 1, 1, 1])
    mvals = m.drawmeridians(meridians, linewidth=0.0, labels=[1, 1, 1, 1])
    xticks = [i[1][0].get_position()[0] for i in mvals.values()]
    yticks = []
    for i in pvals.values():
        try:
            yticks.append(i[1][0].get_position()[1])
        except IndexError:
            pass

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.tick_params(
        axis="both",
        direction="in",
        left=True,
        right=True,
        top=True,
        bottom=True,
        labelleft=False,
        labelright=False,
        labeltop=False,
        labelbottom=False,
        zorder=10,
    )

    states_path = GIS_DIR / "cb_2017_us_state_500k"

    mbound = m.drawmapboundary(fill_color="white")
    states = m.readshapefile(states_path.as_posix(), "states")

    rivers = [
        (
            GIS_DIR / "NHDPlus" / "trimmed_flowlines" / "NHDPlusLC_trimmed_flowlines_noz"
        ).as_posix(),
        (
            GIS_DIR / "NHDPlus" / "trimmed_flowlines" / "NHDPlusUC_trimmed_flowlines_noz"
        ).as_posix(),
        (
            GIS_DIR / "NHDPlus" / "trimmed_flowlines" / "NHDPlusTN_trimmed_flowlines_noz"
        ).as_posix(),
        (
            GIS_DIR / "NHDPlus" / "trimmed_flowlines" / "NHDPlusML_trimmed_flowlines_noz"
        ).as_posix(),
        (
            GIS_DIR / "NHDPlus" / "trimmed_flowlines" / "NHDPlusMU_trimmed_flowlines_noz"
        ).as_posix(),
        (
            GIS_DIR / "NHDPlus" / "trimmed_flowlines" / "NHDPlusPN_trimmed_flowlines_noz"
        ).as_posix(),
    ]

    for i, r in enumerate(rivers):
        river_lines = m.readshapefile(
            r, f"river_{i}", color="b", linewidth=0.5, default_encoding="latin-1"
        )
        river_lines[4].set_alpha(1.0)

    if other_bound:
        for b, c in other_bound:
            bound = m.readshapefile(
                b,
                "bound",
                # color="#FF3BC6"
                color=c,
            )
            # bound[4].set_facecolor("#FF3BC6")
            bound[4].set_facecolor(c)
            bound[4].set_alpha(0.5)
            bound[4].set_zorder(2)

    return m


def make_maps(axes, coords=None, other_bound=None):
    if coords:
        west, south, east, north = coords
    else:
        west, south, east, north = (-127.441406, 23.807069, -66.093750, 49.392373)

    # parallels = np.arange(0.0, 81, 10.0)
    # meridians = np.arange(-180.0, 181.0, 20.0)
    parallels = np.arange(south + 5, north - 5, 10.0)
    meridians = np.arange(west + 10, east - 10, 20.0)

    states_path = GIS_DIR / "cb_2017_us_state_500k.shp"
    states = gpd.read_file(states_path.as_posix())

    rivers = [
        (
            GIS_DIR / "NHDPlus" / "trimmed_flowlines" / "NHDPlusLC_trimmed_flowlines_noz"
        ).as_posix(),
        (
            GIS_DIR / "NHDPlus" / "trimmed_flowlines" / "NHDPlusUC_trimmed_flowlines_noz"
        ).as_posix(),
        (
            GIS_DIR / "NHDPlus" / "trimmed_flowlines" / "NHDPlusTN_trimmed_flowlines_noz"
        ).as_posix(),
        (
            GIS_DIR / "NHDPlus" / "trimmed_flowlines" / "NHDPlusML_trimmed_flowlines_noz"
        ).as_posix(),
        (
            GIS_DIR / "NHDPlus" / "trimmed_flowlines" / "NHDPlusMU_trimmed_flowlines_noz"
        ).as_posix(),
        (
            GIS_DIR / "NHDPlus" / "trimmed_flowlines" / "NHDPlusPN_trimmed_flowlines_noz"
        ).as_posix(),
    ]

    river_gdfs = [gpd.read_file(river + ".shp") for river in rivers]

    other_gdfs = []
    if other_bound:
        for b, c in other_bound:
            other_gdfs.append((gpd.read_file(b + ".shp"), c))

    label_map = {
        0: dict(labelleft=True, labelright=False, labeltop=True, labelbottom=False),
        1: dict(labelleft=False, labelright=True, labeltop=True, labelbottom=False),
        2: dict(labelleft=True, labelright=False, labeltop=False, labelbottom=False),
        3: dict(labelleft=False, labelright=True, labeltop=False, labelbottom=False),
        4: dict(labelleft=True, labelright=False, labeltop=False, labelbottom=True),
        5: dict(labelleft=False, labelright=True, labeltop=False, labelbottom=True),
    }
    for i, ax in enumerate(axes):
        states.plot(ax=ax, edgecolor="k", facecolor="None")
        for gdf in river_gdfs:
            gdf.plot(ax=ax, color="b", linewidth=0.3, alpha=1.0)
        for gdf, c in other_gdfs:
            gdf.plot(ax=ax, facecolor=c, alpha=0.5, zorder=2)

        ax.set_ylim(south, north)
        ax.set_xlim(west, east)
        ax.patch.set_alpha(0.0)
        ax.grid(False)
        ax.tick_params(
            axis="both",
            which="major",
            direction="in",
            bottom=True,
            top=True,
            left=True,
            right=True,
            **label_map[i],
        )
        ax.set_yticks(parallels)
        ax.set_yticklabels([f"{i:.0f}$^\circ$N" for i in parallels], fontsize=10)
        ax.set_xticks(meridians)
        ax.set_xticklabels([f"{abs(i):.0f}$^\circ$W" for i in meridians], fontsize=10)
        ax.set_frame_on(True)
        for spine in ax.spines.values():
            spine.set_edgecolor("black")


def plot_res_locs():
    res_locs = pd.read_csv("../geo_data/reservoirs.csv")
    res_locs = res_locs.set_index("site_name")

    drop_res = [
        "Causey",
        "Lost Creek",
        "Echo",
        "Smith & Morehouse Reservoir",
        "Jordanelle",
        "Deer Creek",
        "Hyrum",
        "Santa Rosa ",
    ]
    drop_res = [i.upper() for i in drop_res]
    res_locs = res_locs.drop(drop_res)

    with open("../geo_data/extents.json", "r") as f:
        coords = json.load(f)
    color_map = sns.color_palette("Set2")
    basins = [
        ((GIS_DIR / "columbia_shp" / "Shape" / "WBDHU2").as_posix(), color_map[0]),
        ((GIS_DIR / "missouri_shp" / "Shape" / "WBDHU2").as_posix(), color_map[1]),
        #   (GIS_DIR/"lowercol_shp"/"Shape"/"WBDHU2").as_posix(),
        #   (GIS_DIR/"uppercol_shp"/"Shape"/"WBDHU2").as_posix(),
        ((GIS_DIR / "colorado_shp" / "Shape" / "WBDHU2").as_posix(), color_map[2]),
        ((GIS_DIR / "tennessee_shp" / "Shape" / "WBDHU2").as_posix(), color_map[3]),
    ]

    fig, ax = plt.subplots(1, 1)
    m = make_map(ax, other_bound=basins)
    x, y = m(res_locs.long, res_locs.lat)
    max_size = 600
    min_size = 50
    size_var = "max_sto"
    max_value = res_locs[size_var].max()
    min_value = res_locs[size_var].min()

    ratio = (max_size - min_size) / (max_value - min_value)
    sizes = [min_size + i * ratio for i in res_locs[size_var]]
    markers = ax.scatter(x, y, marker="v", edgecolor="k", s=sizes, zorder=4)
    marker_color = markers.get_facecolor()

    river_line = mlines.Line2D([], [], color="b", alpha=1, linewidth=0.5)
    river_basins = [mpatch.Patch(facecolor=color_map[i], alpha=0.5) for i in range(4)]
    size_legend_sizes = np.linspace(min_size, max_size, 4)
    # size_legend_labels = [(i-min_size) / ratio for i in size_legend_sizes]
    size_legend_labels = np.linspace(min_value, max_value, 4)

    size_markers = [
        plt.scatter([], [], s=i, edgecolors="k", c=marker_color, marker="v")
        for i in size_legend_sizes
    ]

    size_legend = plt.legend(
        size_markers,
        [
            f"{round(size_legend_labels[0]*1000, -2):.0f}",
            *[f"{round(i / 1000, 0):,.0f} million" for i in size_legend_labels[1:]],
        ],
        title="Maximum Storage [acre-feet]",
        loc="lower left",
        ncol=4,
    )
    hydro_legend = plt.legend(
        [river_line, *river_basins],
        [
            "Major River Flowlines",
            "Columbia HUC2",
            "Missouri HUC2",
            "Colorado HUC2",
            "Tennessee HUC2",
        ],
        loc="lower right",
    )
    ax.add_artist(size_legend)
    ax.add_artist(hydro_legend)

    plt.show()


def plot_res_perf_map(results):
    train = select_results(results, "train_data")
    test = select_results(results, "test_data")
    simul = select_results(results, "simmed_data")
    train_score = get_model_scores(train, grouper="site_name")
    test_score = get_model_scores(test, grouper="site_name")
    simul_score = get_model_scores(simul, grouper="site_name")

    train_score = combine_dict_to_df(train_score, "Model").reset_index()
    test_score = combine_dict_to_df(test_score, "Model").reset_index()
    simul_score = combine_dict_to_df(simul_score, "Model").reset_index()

    res_locs = pd.read_csv("../geo_data/reservoirs.csv")
    res_locs = res_locs.set_index("site_name")
    score = simul_score[simul_score["Model"] == "TD4"]
    score = score.set_index("site_name")["NSE"]

    res_locs["NSE"] = score

    rbasins = pd.read_pickle("../pickles/res_basin_map.pickle")
    rename = {"upper_col": "colorado", "lower_col": "colorado"}
    res_locs["basin"] = rbasins
    res_locs["basin"] = res_locs["basin"].replace(rename)

    max_size = 200

    drop_res = [
        "Causey",
        "Lost Creek",
        "Echo",
        "Smith & Morehouse Reservoir",
        "Jordanelle",
        "Deer Creek",
        "Hyrum",
        "Santa Rosa ",
    ]
    drop_res = [i.upper() for i in drop_res]

    with open("../geo_data/extents.json", "r") as f:
        coords = json.load(f)

    II()
    sys.exit()

    fig, axes = plt.subplots(2, 2)

    make_map(axes[0, 0], coords["Columbia"])
    # [(GIS_DIR/"columbia_shp"/"Shape"/"WBDHU2").as_posix()])
    make_map(axes[0, 1], coords["Missouri"])
    # [(GIS_DIR/"missouri_shp"/"Shape"/"WBDHU2").as_posix()])
    make_map(axes[1, 0], coords["Colorado"])
    # [(GIS_DIR/"lowercol_shp"/"Shape"/"WBDHU2").as_posix(),
    # (GIS_DIR/"uppercol_shp"/"Shape"/"WBDHU2").as_posix()])
    make_map(axes[1, 1], coords["Tennessee"])
    # [(GIS_DIR/"tennessee_shp"/"Shape"/"WBDHU2").as_posix()])

    basins = ["pnw", "missouri", "colorado", "tva"]
    titles = ["Columbia", "Missouri", "Colorado", "Tennessee"]
    for ax, basin, title in zip(axes.flatten(), basins, titles):
        pdf = res_locs[res_locs["basin"] == basin]
        size = pdf["NSE"] * max_size
        ax.scatter(pdf["long"], pdf["lat"], s=size, edgecolor="k")
        ax.set_title(title)

    plt.show()


def plot_seasonal_performance(results):
    train = select_results(results, "train_data")
    test = select_results(results, "test_data")
    simul = select_results(results, "simmed_data")

    train_score = {
        i: j.groupby(
            [
                j.index.get_level_values("site_name"),
                j.index.get_level_values("datetime").month,
            ]
        ).apply(lambda x: mean_squared_error(x["actual"], x["model"], squared=False))
        for i, j in train.items()
    }
    test_score = {
        i: j.groupby(
            [
                j.index.get_level_values("site_name"),
                j.index.get_level_values("datetime").month,
            ]
        ).apply(lambda x: mean_squared_error(x["actual"], x["model"], squared=False))
        for i, j in test.items()
    }
    simul_score = {
        i: j.groupby(
            [
                j.index.get_level_values("site_name"),
                j.index.get_level_values("datetime").month,
            ]
        ).apply(lambda x: mean_squared_error(x["actual"], x["model"], squared=False))
        for i, j in simul.items()
    }

    train_dfs = []
    test_dfs = []
    simul_dfs = []

    for i, j in train_score.items():
        j = pd.DataFrame({"RMSE": j})
        j["Model"] = i
        train_dfs.append(j)

        test_score[i] = pd.DataFrame({"RMSE": test_score[i]})
        test_score[i]["Model"] = i
        test_dfs.append(test_score[i])

        simul_score[i] = pd.DataFrame({"RMSE": simul_score[i]})
        simul_score[i]["Model"] = i
        simul_dfs.append(simul_score[i])

    train_df = pd.concat(train_dfs).reset_index()
    test_df = pd.concat(test_dfs).reset_index()
    simul_df = pd.concat(simul_dfs).reset_index()

    # pdf = simul_df.groupby(["Model", "datetime"]).quantile(0.5).reset_index()

    simul_score = {
        i: pd.DataFrame(
            {
                "RMSE": j.groupby(j.index.get_level_values("datetime").month).apply(
                    lambda x: mean_squared_error(x["actual"], x["model"], squared=False)
                )
            }
        )
        for i, j in simul.items()
    }
    simul_df = combine_dict_to_df(simul_score, "Model").reset_index()

    ax = sns.barplot(
        data=simul_df,
        x="datetime",
        y="RMSE",
        hue="Model",
        palette="tab10",
        errwidth=0,
        alpha=0.6,
    )

    means = simul_df.groupby("Model").mean()["RMSE"]
    colors = sns.color_palette("tab10")

    for i, mv in enumerate(means):
        ax.axhline(mv, color=colors[i])

    ax.set_ylabel("Simulated RMSE [1000 acre-ft/day]")
    ax.set_xticklabels(calendar.month_abbr[1:])
    ax.set_xlabel("")
    ax.legend(loc="best", ncol=4, title="")

    plt.show()


def plot_upper_lower_perf(results):
    from tclr_model import read_basin_data

    df = read_basin_data("all")

    train = select_results(results, "train_data")
    test = select_results(results, "test_data")
    simul = select_results(results, "simmed_data")

    train_inflow = df.loc[train["TD1"].index, "inflow"]
    test_inflow = df.loc[test["TD1"].index, "inflow"]
    simul_inflow = df.loc[simul["TD1"].index, "inflow"]

    # train_pct = {i: j.groupby("site_name")["actual"].rank(pct=True)
    #              for i, j in train.items()}
    # test_pct = {i: j.groupby("site_name")["actual"].rank(pct=True)
    #              for i, j in test.items()}
    # simul_pct = {i: j.groupby("site_name")["actual"].rank(pct=True)
    #              for i, j in simul.items()}

    train_pct = train_inflow.groupby("site_name").rank(pct=True)
    test_pct = test_inflow.groupby("site_name").rank(pct=True)
    simul_pct = simul_inflow.groupby("site_name").rank(pct=True)

    train_trimmed = {}
    test_trimmed = {}
    simul_trimmed = {}

    for d in train.keys():
        df = train[d]
        df["pct"] = train_pct
        df = df[(df["pct"] >= 0.95) | (df["pct"] <= 0.05)]
        df["label"] = ""
        df.loc[df["pct"] <= 0.05, "label"] = "Lowest 5% Inflow"
        df.loc[df["pct"] >= 0.95, "label"] = "Highest 5% Inflow"

        train_trimmed[d] = df

        df = test[d]
        df["pct"] = test_pct
        df = df[(df["pct"] >= 0.95) | (df["pct"] <= 0.05)]
        df["label"] = ""
        df.loc[df["pct"] <= 0.05, "label"] = "Lowest 5% Inflow"
        df.loc[df["pct"] >= 0.95, "label"] = "Highest 5% Inflow"

        test_trimmed[d] = df

        df = simul[d]
        df["pct"] = simul_pct
        df = df[(df["pct"] >= 0.95) | (df["pct"] <= 0.05)]
        df["label"] = ""
        df.loc[df["pct"] <= 0.05, "label"] = "Lowest 5% Inflow"
        df.loc[df["pct"] >= 0.95, "label"] = "Highest 5% Inflow"

        simul_trimmed[d] = df

    df = combine_dict_to_df(simul_trimmed, "Depth")
    scores = pd.DataFrame(
        {
            "NSE": df.groupby(["Depth", "label"]).apply(
                lambda x: r2_score(x["actual"], x["model"])
            )
        }
    )

    scores = scores.reset_index()

    ax = sns.barplot(data=scores, y="NSE", x="Depth", hue="label",)
    ax.set_xlabel("")
    ax.legend(loc="best")
    plt.show()


def plot_perf_vs_datalength(results):
    simul = select_results(results, "simmed_data")
    simul_score = get_model_scores(simul, grouper="site_name")
    simul_score = combine_dict_to_df(simul_score, "Model").reset_index()

    ntrees = simul_score["Model"].unique().size
    counts = simul["TD4"].index.get_level_values("site_name").value_counts()
    train_counts = counts / 0.2 * 0.8

    simul_score = simul_score.set_index("site_name")
    simul_score["counts"] = train_counts / 365

    sns.scatterplot(data=simul_score, x="counts", y="NSE", hue="Model")
    plt.show()


def plot_grid_search_results(ds="simul", metric="NSE"):
    import glob
    import re

    files = glob.glob(
        "../results/tclr_model_testing/all/TD?_MSS0.??_RT_MS_exhaustive_new_hoover/results.pickle"
    )
    td_mss_pat = re.compile("TD(\d)_MSS(\d\.\d\d)")
    matches = [re.search(td_mss_pat, i) for i in files]
    td_mss = [i.groups() for i in matches]
    results = {}
    for key, file in zip(td_mss, files):
        with open(file, "rb") as f:
            results[key] = pickle.load(f)
    for key in td_mss:
        if key[1] == "0.00":
            del results[key]

    train_data = select_results(results, "train_data")
    test_data = select_results(results, "test_data")
    simmed_data = select_results(results, "simmed_data")

    train_scores = get_model_scores(train_data, metric=metric, grouper="site_name")
    test_scores = get_model_scores(test_data, metric=metric, grouper="site_name")
    simmed_scores = get_model_scores(simmed_data, metric=metric, grouper="site_name")

    train_scores_records = []
    for key, values in train_scores.items():
        for res, value in values[metric].items():
            train_scores_records.append([int(key[0]), float(key[1]), res, value])
    train_scores = pd.DataFrame.from_records(
        train_scores_records, columns=["TD", "MSS", "Reservoir", metric]
    )
    test_scores_records = []
    for key, values in test_scores.items():
        for res, value in values[metric].items():
            test_scores_records.append([int(key[0]), float(key[1]), res, value])
    test_scores = pd.DataFrame.from_records(
        test_scores_records, columns=["TD", "MSS", "Reservoir", metric]
    )
    simmed_scores_records = []
    for key, values in simmed_scores.items():
        for res, value in values[metric].items():
            simmed_scores_records.append([int(key[0]), float(key[1]), res, value])
    simmed_scores = pd.DataFrame.from_records(
        simmed_scores_records, columns=["TD", "MSS", "Reservoir", metric]
    )

    # train_agg = train_scores.groupby(["TD", "MSS"])[[metric]].mean()
    # test_agg = test_scores.groupby(["TD", "MSS"])[[metric]].mean()
    # simmed_agg = simmed_scores.groupby(["TD", "MSS"])[[metric]].mean()

    # train_agg["Data Set"] = "Train"
    # test_agg["Data Set"] = "Test"
    # simmed_agg["Data Set"] = "Simmed"
    # df = pd.concat([train_agg, test_agg, simmed_agg])
    if ds == "simul":
        df = simmed_scores
    elif ds == "test":
        df = test_scores
    else:
        df = train_scores

    fg = sns.catplot(
        data=df,
        hue="TD",
        x="MSS",
        y=metric,
        kind="box",
        palette="tab10",
        whis=(0.1, 0.9),
        showfliers=False,
        legend=False
        # ci=None,
    )
    fg.ax.legend(loc="best", ncol=4, title="Max Depth")
    # fg.ax.set_xticklabels(fg.ax.get_xticklabels(), rotation=45, ha="right")
    plt.show()


def plot_data_assim_results(metric="NSE"):
    import glob
    import re

    files = glob.glob(
        "../results/tclr_model_testing/all/TD?_*_MSS0.??_RT_MS_exhaustive_new_hoover/results.pickle"
    )
    td_mss_assim_pat = re.compile("TD(\d)_(.*)_MSS(\d\.\d\d)")
    matches = [re.search(td_mss_assim_pat, i) for i in files]
    td_mss_assim = [i.groups() for i in matches]
    results = {}
    for key, file in zip(td_mss_assim, files):
        with open(file, "rb") as f:
            results[key] = pickle.load(f)
    train_data = select_results(results, "train_data")
    test_data = select_results(results, "test_data")
    simmed_data = select_results(results, "simmed_data")

    train_scores = get_model_scores(train_data, metric=metric, grouper="site_name")
    test_scores = get_model_scores(test_data, metric=metric, grouper="site_name")
    simmed_scores = get_model_scores(simmed_data, metric=metric, grouper="site_name")

    columns = ["TD", "Assim", "MSS", "Reservoir", metric]
    train_scores_records = []
    for key, values in train_scores.items():
        for res, value in values[metric].items():
            train_scores_records.append([int(key[0]), key[1], float(key[2]), res, value])
    train_scores = pd.DataFrame.from_records(train_scores_records, columns=columns)
    test_scores_records = []
    for key, values in test_scores.items():
        for res, value in values[metric].items():
            test_scores_records.append([int(key[0]), key[1], float(key[2]), res, value])
    test_scores = pd.DataFrame.from_records(test_scores_records, columns=columns)
    simmed_scores_records = []
    for key, values in simmed_scores.items():
        for res, value in values[metric].items():
            simmed_scores_records.append([int(key[0]), key[1], float(key[2]), res, value])
    simmed_scores = pd.DataFrame.from_records(simmed_scores_records, columns=columns)

    # simmed_scores[metric] = np.log(simmed_scores[metric])

    fg = sns.catplot(
        data=simmed_scores,
        hue="TD",
        x="Assim",
        y=metric,
        kind="box",
        order=["daily", "weekly", "monthly", "seasonally", "semi-annually"],
        whis=(0.1, 0.9),
        showfliers=False,
        legend_out=False,
    )
    strip = sns.stripplot(
        data=simmed_scores,
        hue="TD",
        x="Assim",
        y=metric,
        order=["daily", "weekly", "monthly", "seasonally", "semi-annually"],
        ax=fg.ax,
        dodge=True,
        jitter=0.2,
    )
    for col in strip.collections:
        col.set_color("#808080")
        col.set_alpha(0.5)
        col.set_edgecolors("k")
        col.set_linewidth(1)
    fg.ax.set_xticklabels(["Daily", "Weekly", "Monthly", "Seasonally", "Semi-annually"])
    fg.ax.set_xlabel("Assimilation Frequency")

    fg.ax.set_xticks([], minor=True)
    fg.despine(left=False, right=False, top=False, bottom=False)

    handles, labels = fg.ax.get_legend_handles_labels()
    handles = handles[:2]
    labels = ["TD2-MSS0.20", "TD5-MSS0.01"]
    fg.ax.legend(handles, labels, loc="best")

    # fg.ax.set_yscale("log")
    II()

    plt.show()


def plot_assim_score_ridgelines(metric="NSE"):
    import glob
    import re

    files = glob.glob(
        "../results/tclr_model_testing/all/TD?_*_MSS0.??_RT_MS_exhaustive_new_hoover/results.pickle"
    )
    td_mss_assim_pat = re.compile("TD(\d)_(.*)_MSS(\d\.\d\d)")
    matches = [re.search(td_mss_assim_pat, i) for i in files]
    td_mss_assim = [i.groups() for i in matches]
    results = {}
    for key, file in zip(td_mss_assim, files):
        with open(file, "rb") as f:
            results[key] = pickle.load(f)
    train_data = select_results(results, "train_data")
    test_data = select_results(results, "test_data")
    simmed_data = select_results(results, "simmed_data")

    train_scores = get_model_scores(train_data, metric=metric, grouper="site_name")
    test_scores = get_model_scores(test_data, metric=metric, grouper="site_name")
    simmed_scores = get_model_scores(simmed_data, metric=metric, grouper="site_name")

    columns = ["TD", "Assim", "MSS", "Reservoir", metric]
    train_scores_records = []
    for key, values in train_scores.items():
        for res, value in values[metric].items():
            train_scores_records.append([int(key[0]), key[1], float(key[2]), res, value])
    train_scores = pd.DataFrame.from_records(train_scores_records, columns=columns)
    test_scores_records = []
    for key, values in test_scores.items():
        for res, value in values[metric].items():
            test_scores_records.append([int(key[0]), key[1], float(key[2]), res, value])
    test_scores = pd.DataFrame.from_records(test_scores_records, columns=columns)
    simmed_scores_records = []
    for key, values in simmed_scores.items():
        for res, value in values[metric].items():
            simmed_scores_records.append([int(key[0]), key[1], float(key[2]), res, value])
    simmed_scores = pd.DataFrame.from_records(simmed_scores_records, columns=columns)
    fg = sns.FacetGrid(
        simmed_scores,
        row="Assim",
        hue="TD",
        sharey=False,  # ) aspect=15, height=0.5,)
        row_order=["daily", "weekly", "monthly", "seasonally", "semi-annually"],
    )
    fg.fig.patch.set_alpha(0.0)
    for ax in fg.axes.flatten():
        ax.patch.set_alpha(0.0)
        ax.grid(False)

    fg.map(
        sns.kdeplot,
        metric,
        bw_adjust=0.5,
        clip_on=False,
        fill=True,
        alpha=1,
        linewidth=1.5,
    )
    fg.map(sns.kdeplot, metric, clip_on=False, color="k", lw=2, bw_adjust=0.5)

    fg.figure.subplots_adjust(hspace=-0.25)
    fg.set_titles("")
    fg.set(yticks=[], ylabel="")
    row_order = ["daily", "weekly", "monthly", "seasonally", "semi-annually"]
    for ax, label in zip(fg.axes.flatten(), row_order):
        ax.set_ylabel(label.capitalize(), rotation=0)

    fg.axes.flatten()[0].legend(loc="best")
    fg.fig.align_ylabels()


def plot_best_and_worst_reservoirs(metric="NSE"):
    files = [
        "../results/tclr_model_testing/all/TD2_MSS0.20_RT_MS_exhaustive_new_hoover/results.pickle",
        "../results/tclr_model_testing/all/TD5_MSS0.01_RT_MS_exhaustive_new_hoover/results.pickle",
    ]
    keys = [(2, 0.2), (5, 0.01)]
    results = {}
    for k, f in zip(keys, files):
        with open(f, "rb") as f:
            results[k] = pickle.load(f)

    train_data = select_results(results, "train_data")
    test_data = select_results(results, "test_data")
    simmed_data = select_results(results, "simmed_data")

    train_scores = get_model_scores(train_data, metric=metric, grouper="site_name")
    test_scores = get_model_scores(test_data, metric=metric, grouper="site_name")
    simmed_scores = get_model_scores(simmed_data, metric=metric, grouper="site_name")

    train_top_20 = {
        key: df.sort_values(by=metric).tail(20) for key, df in train_scores.items()
    }
    test_top_20 = {
        key: df.sort_values(by=metric).tail(20) for key, df in test_scores.items()
    }
    simmed_top_20 = {
        key: df.sort_values(by=metric).tail(20) for key, df in simmed_scores.items()
    }
    train_btm_20 = {
        key: df.sort_values(by=metric).head(20) for key, df in train_scores.items()
    }
    test_btm_20 = {
        key: df.sort_values(by=metric).head(20) for key, df in test_scores.items()
    }
    simmed_btm_20 = {
        key: df.sort_values(by=metric).head(20) for key, df in simmed_scores.items()
    }
    records = []

    for key in keys:
        for res, value in train_top_20[key]["NSE"].items():
            records.append((key, "Train", "Best 20", res, value))
        for res, value in test_top_20[key]["NSE"].items():
            records.append((key, "Test", "Best 20", res, value))
        for res, value in simmed_top_20[key]["NSE"].items():
            records.append((key, "Simmed", "Best 20", res, value))
        for res, value in train_btm_20[key]["NSE"].items():
            records.append((key, "Train", "Worst 20", res, value))
        for res, value in test_btm_20[key]["NSE"].items():
            records.append((key, "Test", "Worst 20", res, value))
        for res, value in simmed_btm_20[key]["NSE"].items():
            records.append((key, "Simmed", "Worst 20", res, value))
    df = pd.DataFrame.from_records(
        records, columns=["Model", "Data Set", "group", "Reservoir", metric]
    )

    fg = sns.catplot(
        data=df,
        x="Data Set",
        y=metric,
        row="group",
        hue="Model",
        kind="box",
        whis=(0.1, 0.9),
        showfliers=True,
        sharey=False,
        legend=False,
    )

    axes = fg.axes.flatten()
    axes[0].set_title("20 Best Performing Reservoirs")
    axes[1].set_title("20 Worst Performing Reservoirs")

    handles, labels = axes[0].get_legend_handles_labels()
    labels = ["TD2-MSS0.20", "TD5-MSS0.01"]
    axes[1].legend(handles, labels, loc="best")

    axes[1].set_xlabel("")
    plt.show()


def get_top_btm_res_characteristic(metric="NSE", count=20):
    from find_basin_groups import prep_seasonalities
    from tclr_model import get_basin_meta_data, read_basin_data

    files = [
        "../results/tclr_model_testing/all/TD2_MSS0.20_RT_MS_exhaustive_new_hoover/results.pickle",
        "../results/tclr_model_testing/all/TD5_MSS0.01_RT_MS_exhaustive_new_hoover/results.pickle",
    ]
    keys = [(2, 0.2), (5, 0.01)]
    results = {}
    for k, f in zip(keys, files):
        with open(f, "rb") as f:
            results[k] = pickle.load(f)

    # train_data = select_results(results, "train_data")
    # test_data = select_results(results, "test_data")
    simmed_data = select_results(results, "simmed_data")

    # train_scores = get_model_scores(train_data, metric=metric, grouper="site_name")
    # test_scores =  get_model_scores(test_data, metric=metric, grouper="site_name")
    simmed_scores = get_model_scores(simmed_data, metric=metric, grouper="site_name")

    df = read_basin_data("all")
    meta = get_basin_meta_data("all")

    drop_res = [
        "Causey",
        "Lost Creek",
        "Echo",
        "Smith & Morehouse Reservoir",
        "Jordanelle",
        "Deer Creek",
        "Hyrum",
        "Santa Rosa ",
        "MCPHEE",
    ]
    drop_res = [i.upper() for i in drop_res]
    meta = meta.drop(drop_res)
    df = df[~df.index.get_level_values(0).isin(drop_res)]
    mmeans = df.groupby(
        [df.index.get_level_values(0), df.index.get_level_values(1).month]
    ).mean()
    seasonalities = prep_seasonalities(mmeans)

    cv = (
        df.groupby(df.index.get_level_values(0))["release"].std()
        / df.groupby(df.index.get_level_values(0))["release"].mean()
    )

    top_ssn_res = seasonalities["SI_rel"].sort_values().tail(count).index
    top_ssn_sto_res = seasonalities["SI_sto"].sort_values().tail(count).index
    top_sto_res = meta["max_sto"].sort_values().tail(count).index
    top_rel_res = (
        df.groupby(df.index.get_level_values(0))["release"]
        .mean()
        .sort_values()
        .tail(count)
        .index
    )
    top_cv_res = cv.sort_values().tail(count).index
    top_rts_res = meta["rts"].sort_values().tail(count).index

    btm_ssn_res = seasonalities["SI_rel"].sort_values().head(count).index
    btm_ssn_sto_res = seasonalities["SI_sto"].sort_values().head(count).index
    btm_sto_res = meta["max_sto"].sort_values().head(count).index
    btm_rel_res = (
        df.groupby(df.index.get_level_values(0))["release"]
        .mean()
        .sort_values()
        .head(count)
        .index
    )
    btm_cv_res = cv.sort_values().head(count).index
    btm_rts_res = meta["rts"].sort_values().head(count).index

    top_ssn_perf = {key: df.loc[top_ssn_res] for key, df in simmed_scores.items()}
    top_ssn_sto_perf = {key: df.loc[top_ssn_sto_res] for key, df in simmed_scores.items()}
    top_sto_perf = {key: df.loc[top_sto_res] for key, df in simmed_scores.items()}
    top_rel_perf = {key: df.loc[top_rel_res] for key, df in simmed_scores.items()}
    top_rts_perf = {key: df.loc[top_rts_res] for key, df in simmed_scores.items()}
    top_cv_perf = {key: df.loc[top_cv_res] for key, df in simmed_scores.items()}
    btm_ssn_perf = {key: df.loc[btm_ssn_res] for key, df in simmed_scores.items()}
    btm_ssn_sto_perf = {key: df.loc[btm_ssn_sto_res] for key, df in simmed_scores.items()}
    btm_sto_perf = {key: df.loc[btm_sto_res] for key, df in simmed_scores.items()}
    btm_rel_perf = {key: df.loc[btm_rel_res] for key, df in simmed_scores.items()}
    btm_rts_perf = {key: df.loc[btm_rts_res] for key, df in simmed_scores.items()}
    btm_cv_perf = {key: df.loc[btm_cv_res] for key, df in simmed_scores.items()}
    records = []
    for key in simmed_scores.keys():
        for res, score in top_ssn_perf[key][metric].items():
            records.append((key, f"Top {count}", "Release Seasonality", res, score))
        for res, score in top_ssn_sto_perf[key][metric].items():
            records.append((key, f"Top {count}", "Storage Seasonality", res, score))
        for res, score in top_sto_perf[key][metric].items():
            records.append((key, f"Top {count}", "Max Storage", res, score))
        for res, score in top_rel_perf[key][metric].items():
            records.append((key, f"Top {count}", "Mean Release", res, score))
        for res, score in top_rts_perf[key][metric].items():
            records.append((key, f"Top {count}", "Residence Time", res, score))
        for res, score in top_cv_perf[key][metric].items():
            records.append((key, f"Top {count}", r"Release $CV$", res, score))
        for res, score in btm_ssn_perf[key][metric].items():
            records.append((key, f"Bottom {count}", "Release Seasonality", res, score))
        for res, score in btm_ssn_sto_perf[key][metric].items():
            records.append((key, f"Bottom {count}", "Storage Seasonality", res, score))
        for res, score in btm_sto_perf[key][metric].items():
            records.append((key, f"Bottom {count}", "Max Storage", res, score))
        for res, score in btm_rel_perf[key][metric].items():
            records.append((key, f"Bottom {count}", "Mean Release", res, score))
        for res, score in btm_rts_perf[key][metric].items():
            records.append((key, f"Bottom {count}", "Residence Time", res, score))
        for res, score in btm_cv_perf[key][metric].items():
            records.append((key, f"Bottom {count}", r"Release $CV$", res, score))
    perf_df = pd.DataFrame.from_records(
        records, columns=["Model", "group", "Characteristic", "Reservoir", metric]
    )
    return perf_df


def plot_top_characteristic_res(metric="NSE", count=20):
    char_df = get_top_btm_res_characteristic(metric, count)
    fg = sns.catplot(
        data=char_df,
        row="Model",
        x="Characteristic",
        y=metric,
        hue="group",
        kind="strip",
        # whis=(0.1, 0.9),
        order=[
            "Release Seasonality",
            "Storage Seasonality",
            "Max Storage",
            "Mean Release",
            r"Release $CV$",
            "Residence Time",
        ],
        legend=False,
    )
    axes = fg.axes.flatten()
    axes[0].set_title("TD 2 - MSS 0.20")
    axes[1].set_title("TD 5 - MSS 0.01")
    axes[1].legend(loc="best")
    plt.show()


def get_res_characteristic(metric="NSE"):
    from find_basin_groups import prep_seasonalities
    from tclr_model import get_basin_meta_data, read_basin_data

    files = [
        "../results/tclr_model_testing/all/TD2_MSS0.20_RT_MS_exhaustive_new_hoover/results.pickle",
        "../results/tclr_model_testing/all/TD5_MSS0.01_RT_MS_exhaustive_new_hoover/results.pickle",
    ]
    keys = [(2, 0.2), (5, 0.01)]
    results = {}
    for k, f in zip(keys, files):
        with open(f, "rb") as f:
            results[k] = pickle.load(f)

    # train_data = select_results(results, "train_data")
    # test_data = select_results(results, "test_data")
    simmed_data = select_results(results, "simmed_data")

    # train_scores = get_model_scores(train_data, metric=metric, grouper="site_name")
    # test_scores =  get_model_scores(test_data, metric=metric, grouper="site_name")
    simmed_scores = get_model_scores(simmed_data, metric=metric, grouper="site_name")

    df = read_basin_data("all")
    meta = get_basin_meta_data("all")

    drop_res = [
        "Causey",
        "Lost Creek",
        "Echo",
        "Smith & Morehouse Reservoir",
        "Jordanelle",
        "Deer Creek",
        "Hyrum",
        "Santa Rosa ",
        "MCPHEE",
    ]
    drop_res = [i.upper() for i in drop_res]
    meta = meta.drop(drop_res)
    df = df[~df.index.get_level_values(0).isin(drop_res)]
    mmeans = df.groupby(
        [df.index.get_level_values(0), df.index.get_level_values(1).month]
    ).mean()
    seasonalities = prep_seasonalities(mmeans)

    cv = (
        df.groupby(df.index.get_level_values(0))["release"].std()
        / df.groupby(df.index.get_level_values(0))["release"].mean()
    )

    char_df = pd.DataFrame(
        index=seasonalities.index,
        columns=[
            "Release Seasonality",
            "Storage Seasonality",
            "Maximum Storage",
            "Mean Release",
            r"Release $CV$",
            "Residence Time",
        ],
    )
    char_df["Release Seasonality"] = seasonalities["SI_rel"]
    char_df["Storage Seasonality"] = seasonalities["SI_sto"]
    char_df["Maximum Storage"] = meta["max_sto"]
    char_df["Mean Release"] = df.groupby(df.index.get_level_values(0))["release"].mean()
    char_df[r"Release $CV$"] = cv
    char_df["Residence Time"] = meta["rts"]
    for key, value in simmed_scores.items():
        df_key = f"TD{key[0]}-MSS{key[1]:.2f}"
        char_df[df_key] = value
    return char_df


def plot_top_characteristic_res_scatter(metric="NSE"):
    char_df = get_res_characteristic(metric)
    fig, axes = plt.subplots(3, 2, sharey=True, sharex=False)
    axes = axes.flatten()
    pvars = [
        "Release Seasonality",
        "Storage Seasonality",
        "Maximum Storage",
        "Mean Release",
        r"Release $CV$",
        "Residence Time",
    ]
    markers = [
        "o",
        "s",
    ]
    # colors = sns.color_palette("Tab2", 2)
    for i, var in enumerate(pvars):
        axes[i].scatter(char_df[var], char_df["TD2-MSS0.20"], label="TD2-MSS0.20")
        axes[i].scatter(char_df[var], char_df["TD5-MSS0.01"], label="TD5-MSS0.01")
        axes[i].set_xlabel(var)
        axes[i].set_ylabel(metric)

    axes[0].legend(loc="best")
    fig.align_ylabels()
    plt.show()


def plot_characteristic_res_line_plot(metric="NSE"):
    char_df = get_res_characteristic(metric)
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(19, 10))
    # axes = axes.flatten()
    fig.patch.set_alpha(0.0)

    pvars = [
        "Release Seasonality",
        "Storage Seasonality",
        "Maximum Storage",
        "Mean Release",
        r"Release $CV$",
        "Residence Time",
    ]
    zeros = np.zeros_like(char_df["TD2-MSS0.20"].values)
    max_size = 500
    min_size = 20

    # size_df = char_df.copy()
    # for column in pvars:
    #     size_df[column] = linear_scale_values(size_df[column], min_size, max_size)

    # mdf = size_df.melt(id_vars=["TD2-MSS0.20", "TD5-MSS0.01"], var_name="Characteristic").melt(
    #     id_vars=["Characteristic", "value"], var_name="Model", value_name=metric)

    # sns.stripplot(
    #     data=mdf,
    #     y="Characteristic",
    #     x=metric,
    #     hue="Model",
    #     size="value",
    # )

    colors = sns.color_palette("tab10", 2)
    for i, var in enumerate(pvars):
        y = np.zeros_like(char_df["TD2-MSS0.20"]) + i
        axes.scatter(
            char_df["TD2-MSS0.20"],
            y,
            color=colors[0],
            s=linear_scale_values(char_df[var], min_size, max_size),
            label="TD2-MSS0.20",
        )
        axes.scatter(
            char_df["TD5-MSS0.01"],
            y,
            color=colors[1],
            s=linear_scale_values(char_df[var], min_size, max_size),
            label="TD5-MSS0.01",
        )
        # axes.set_xlabel(var)
        axes.set_xlabel(metric)

    axes.set_yticks(range(len(pvars)))
    axes.set_yticklabels(pvars)
    handles, labels = axes.get_legend_handles_labels()
    axes.legend(handles[:2], labels[:2], loc="best")
    fig.align_ylabels()
    plt.show()


def plot_res_characteristic_bin_performance(metric="NSE", nbins=3):
    char_df = get_res_characteristic(metric)
    pvars = [
        "Release Seasonality",
        "Storage Seasonality",
        "Maximum Storage",
        "Mean Release",
        r"Release $CV$",
        "Residence Time",
    ]
    for var in pvars:
        char_df[var] = pd.qcut(char_df[var], nbins, labels=False) + 1

    m1_df = pd.DataFrame(index=range(1, nbins + 1), columns=pvars)
    m2_df = pd.DataFrame(index=range(1, nbins + 1), columns=pvars)

    for var in pvars:
        m1_df[var] = char_df.groupby(var)["TD2-MSS0.20"].mean()
        m2_df[var] = char_df.groupby(var)["TD5-MSS0.01"].mean()

    m1_df["Model"] = "TD2-MSS0.20"
    m2_df["Model"] = "TD5-MSS0.01"
    m1_df = (
        m1_df.reset_index()
        .rename(columns={"index": "Bin"})
        .melt(id_vars=["Model", "Bin"])
    )
    m2_df = (
        m2_df.reset_index()
        .rename(columns={"index": "Bin"})
        .melt(id_vars=["Model", "Bin"])
    )

    df = pd.concat([m1_df, m2_df])
    label_map = make_bin_label_map(nbins, start_index=1)
    df["Bin"] = df["Bin"].replace(label_map)
    fg = sns.catplot(
        data=df,
        x="Bin",
        y="value",
        hue="Model",
        col="variable",
        col_wrap=2,
        kind="bar",
        legend_out=False,
    )
    fg.set_titles("{col_name}")
    fg.set_ylabels(metric)
    fg.set_xlabels("Characteristic Percentile")

    plt.show()


def plot_res_characteristic_map(metric="NSE"):
    char_df = get_res_characteristic(metric)
    pvars = [
        "Release Seasonality",
        "Storage Seasonality",
        "Maximum Storage",
        "Mean Release",
        r"Release $CV$",
        "Residence Time",
    ]

    # fig, axes = plt.subplots(3, 2, sharex=True, sharey=True)
    # axes = axes.flatten()
    fig = plt.figure()
    gs = GS.GridSpec(3, 3, height_ratios=[1, 1, 1], width_ratios=[10, 10, 0.5])
    fig.patch.set_alpha(0.0)
    #
    #  [0,0], [0,1], [0,2
    #
    #
    #
    #
    axes = [fig.add_subplot(gs[0, 0])]
    for pair in [(0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]:
        axes.append(fig.add_subplot(gs[pair], sharex=axes[0], sharey=axes[0]))
    axes = np.array(axes)
    cbar_ax = fig.add_subplot(gs[:, 2])
    res_locs = pd.read_csv("../geo_data/reservoirs.csv")
    res_locs = res_locs.set_index("site_name")

    drop_res = [
        "Causey",
        "Lost Creek",
        "Echo",
        "Smith & Morehouse Reservoir",
        "Jordanelle",
        "Deer Creek",
        "Hyrum",
        "Santa Rosa ",
    ]
    drop_res = [i.upper() for i in drop_res]
    res_locs = res_locs.drop(drop_res)

    # with open("../geo_data/extents.json", "r") as f:
    #     coords = json.load(f)
    color_map = sns.color_palette("Set2")
    basins = [
        ((GIS_DIR / "columbia_shp" / "Shape" / "WBDHU2").as_posix(), color_map[0]),
        ((GIS_DIR / "missouri_shp" / "Shape" / "WBDHU2").as_posix(), color_map[1]),
        #   (GIS_DIR/"lowercol_shp"/"Shape"/"WBDHU2").as_posix(),
        #   (GIS_DIR/"uppercol_shp"/"Shape"/"WBDHU2").as_posix(),
        ((GIS_DIR / "colorado_shp" / "Shape" / "WBDHU2").as_posix(), color_map[2]),
        ((GIS_DIR / "tennessee_shp" / "Shape" / "WBDHU2").as_posix(), color_map[3]),
    ]

    # maps = [make_map(i, other_bound=basins) for i in axes]
    units = ["", "", " [1000 acre-ft]", " [1000 acre-ft/day]", "", " [Days]"]
    make_maps(axes, other_bound=basins)
    model_key = "TD2-MSS0.20"
    min_color = "#FFFFFF"
    max_color = "#EF2727"
    min_value = char_df[model_key].min()
    max_value = char_df[model_key].max()
    # * use custom colormap
    # color_map = ColorInterpolator("#FFFFFF", "#EF2727", char_df[model_key].min(), char_df[model_key].max())
    # color_map = ColorInterpolator(min_color, max_color, min_value, max_value)
    # colors = [color_map(v) for v in char_df.loc[res_locs.index, model_key]]
    # norm = Normalize(vmin=min_value, vmax=max_value)
    norm = LogNorm(vmin=min_value, vmax=max_value)
    cmap = "inferno"
    if metric == "nRMSE":
        # want bright colors to be high performing reserviors
        color_map = get_cmap(cmap).reversed()
    else:
        color_map = get_cmap(cmap)
    colors = [color_map(norm(i)) for i in char_df.loc[res_locs.index, model_key]]
    for ax, var, unit in zip(axes, pvars, units):
        max_size = 400
        min_size = 20
        values = char_df.loc[res_locs.index, var]
        size = np.array(linear_scale_values(values, min_size, max_size))
        ax.scatter(
            res_locs["long"],
            res_locs["lat"],
            facecolor=colors,
            s=size,
            edgecolor="k",
            zorder=4,
            linewidths=1,
        )
        legend_scores = np.linspace(values.min(), values.max(), 4)
        legend_sizes = linear_scale_values(legend_scores, min_size, max_size)
        legend_markers = [
            plt.scatter([], [], s=i, edgecolors="k", c="#ef2727", alpha=1, linewidths=1)
            for i in legend_sizes
        ]
        leg_kwargs = dict(
            ncol=4,
            frameon=True,
            handlelength=1,
            loc="lower right",
            borderpad=1,
            scatterpoints=1,
            handletextpad=1,
            labelspacing=1,
            markerfirst=False,
            title=var,
            fontsize=6,
        )
        if var == pvars[2]:
            legend_labels = [f"{round(i, -2):.0f}" for i in legend_scores]
        elif var == pvars[3]:
            legend_labels = [f"{round(i, -1):.0f}" for i in legend_scores]
        elif var == pvars[5]:
            legend_labels = [f"{i:.0f}" for i in legend_scores]
        else:
            legend_labels = [f"{i:.2f}" for i in legend_scores]
        ax.legend(
            legend_markers,
            legend_labels,
            ncol=4,
            loc="lower left",
            title=f"{var}{unit}",
            fontsize=8,
            handlelength=1,
            columnspacing=1,
            title_fontsize=8,
            handletextpad=1,
            borderpad=1,
            labelspacing=1,
        )
    # * use the comment out lines if creating custom colormap
    # cmap = LinearSegmentedColormap.from_list("mcmap", [min_color, max_color])
    # norm = Normalize(vmin=min_value, vmax=max_value)
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=color_map), cax=cbar_ax)
    cbar.outline.set_edgecolor("k")
    cbar.outline.set_linewidth(1)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(metric, fontsize=14)
    plt.show()


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) > 0:
        metric = args[0]
    else:
        metric = "NSE"
    plt.style.use("seaborn-notebook")
    # plt.style.use(["science", "nature"])
    sns.set_context("notebook", font_scale=1.4)
    # results = read_results()
    # plot_res_perf_map(results)
    # plot_seasonal_performance(results)
    # plot_upper_lower_perf(results)
    # plot_perf_vs_datalength(results)

    # * FIGURE 1
    # plot_res_locs()
    # * FIGURE 2
    # plot_variable_correlations()
    # * FIGURE 3
    # plot_performance_boxplots(results)

    # plot_grid_search_results(ds="simul", metric=metric)
    # plot_data_assim_results(metric)
    # plot_best_and_worst_reservoirs(metric)
    # plot_top_characteristic_res(metric, 10)
    # plot_top_characteristic_res_scatter(metric)
    # plot_characteristic_res_line_plot(metric)
    # plot_res_characteristic_bin_performance(metric, 5)
    plot_res_characteristic_map(metric)
