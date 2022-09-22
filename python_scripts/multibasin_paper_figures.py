import calendar
import glob
import json
import os
import pathlib
import pickle
import re
import socket
import sys
from functools import partial
from time import perf_counter as timer

from joblib import Parallel, delayed

hostname = socket.gethostname()
if hostname == "CCEE-DT-094":
    os.environ["PROJ_LIB"] = (
        "C:\\Users\\lcford2\\AppData\\Local\\Continuum\\"
        "anaconda3\\envs\\sry-env\\Library\\share"
    )
elif hostname == "inspiron-laptop":
    os.environ[
        "PROJ_LIB"
    ] = r"C:\\Users\\lcford\\miniconda3\\envs\\sry-env\\Library\\share"

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
from matplotlib.colors import ListedColormap, LogNorm, Normalize
from matplotlib.transforms import Bbox
from mpl_toolkits.basemap import Basemap
from scipy.stats import boxcox
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils.helper_functions import linear_scale_values, make_bin_label_map

if hostname == "CCEE-DT-094":
    GIS_DIR = pathlib.Path("G:/My Drive/PHD/GIS")
elif hostname == "inspiron13":
    GIS_DIR = pathlib.Path("/home/lford/data/GIS")

CHAR_VARS = [
    "Release Seasonality",
    "Storage Seasonality",
    "Maximum Storage",
    "Mean Release",
    r"Release $CV$",
    "Residence Time",
]


def load_pickle(file):
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data


def write_pickle(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def read_results():
    path_format = (
        "../results/tclr_model_drop_res_sto_diff_pers_testing_minsamples/"
        + "all/TD{}_RT_MS_exhaustive/results.pickle".format
    )
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


def total_absolute_error(yact, ymod):
    return (ymod - yact).abs().sum()


def bias(yact, ymod):
    return np.mean(ymod) - np.mean(yact)


def pbias(yact, ymod):
    return (np.mean(ymod) - np.mean(yact)) / np.mean(yact)


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


def get_pbias(df, grouper=None):
    if grouper:
        return pd.DataFrame(
            {"PBIAS": df.groupby(grouper).apply(lambda x: pbias(x["actual"], x["model"]))}
        )
    else:
        return pbias(["actual"], df["model"])


def get_model_scores(model_dfs, metric="NSE", grouper=None):
    if metric == "NSE":
        return {i: get_r2score(j, grouper) for i, j in model_dfs.items()}
    elif metric == "RMSE":
        return {i: get_rmse(j, grouper) for i, j in model_dfs.items()}
    elif metric == "MASE":
        return {i: get_mase(j, grouper) for i, j in model_dfs.items()}
    elif metric == "nRMSE":
        return {i: get_nrmse(j, grouper) for i, j in model_dfs.items()}
    elif metric == "PBIAS":
        return {i: get_pbias(j, grouper) for i, j in model_dfs.items()}


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
    sns.despine(ax=ex_ax, left=True, bottom=True)
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
        whis=(5.0, 95.0),
        # whis=(0.01, 0.99),
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
        # whis=(0.01, 0.99),
        whis=(5.0, 95.0),
        showfliers=False,
    )
    axes[1].legend(loc="lower right", ncol=4)

    axes[0].set_ylabel("$r(R_t, NI_L)$")
    axes[0].set_xlabel("Lag $L$ [days]")

    axes[-1].set_xlabel(r"Pearson's $r$ with Release")
    axes[-1].set_ylabel("")
    axes[-1].set_yticklabels(
        [
            r"$S_{t-1}$",
            r"$\bar{S}_{t-1}^7$",
            r"$S_{t-1} \times NI_{t}$",
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
        ax.set_yticklabels([r"{:.0f}$^\circ$N".format(i) for i in parallels], fontsize=10)
        ax.set_xticks(meridians)
        ax.set_xticklabels(
            [r"{:.0f}$^\circ$W".format(abs(i)) for i in meridians], fontsize=10
        )
        ax.set_frame_on(True)
        for spine in ax.spines.values():
            spine.set_edgecolor("black")


def make_basin_map(ax, basin_info):
    river_gdfs = [gpd.read_file(i + ".shp") for i in basin_info["rivers"]]
    bound_gdf = gpd.read_file(basin_info["shp"] + ".shp")
    west, south, east, north = basin_info["extents"]
    pstep = int(np.ceil((north - south) / 4))
    mstep = int(np.ceil((east - west) / 3))
    # pstep = 2
    # mstep = 4
    parallels = np.arange(south + pstep / 2, north - pstep / 2, pstep)
    meridians = np.arange(west + mstep / 2, east - mstep / 2, mstep)

    states_path = GIS_DIR / "cb_2017_us_state_500k.shp"
    states = gpd.read_file(states_path.as_posix())

    label_map = {
        "labelleft": True,
        "labelright": False,
        "labeltop": False,
        "labelbottom": True,
    }

    states.plot(ax=ax, edgecolor="k", facecolor="None", linewidth=0.5)
    for river in river_gdfs:
        river.plot(ax=ax, color="b", linewidth=0.3, alpha=1.0)
    bound_gdf.plot(ax=ax, facecolor=basin_info["color"], alpha=0.5, zorder=2)

    ax.set_ylim(south, north)
    ax.set_xlim(west, east)
    ax.patch.set_alpha(0.0)
    ax.grid = False
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        bottom=True,
        top=True,
        left=True,
        right=True,
        length=6,
        width=1,
        **label_map,
    )

    ax.set_yticks(parallels)
    ax.set_yticklabels([r"{:.0f}$^\circ$N".format(i) for i in parallels], fontsize=16)
    ax.set_xticks(meridians)
    ax.set_xticklabels(
        [r"{:.0f}$^\circ$W".format(abs(i)) for i in meridians], fontsize=16
    )
    ax.set_frame_on(True)
    for spine in ax.spines.values():
        spine.set_edgecolor("black")


def plot_res_locs(colors=None):
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

    fig = plt.figure()
    ax = fig.add_subplot()
    m = make_map(ax, other_bound=basins)
    x, y = m(res_locs.long, res_locs.lat)
    # max_size = 600
    # min_size = 50
    # size_var = "max_sto"
    # max_value = res_locs[size_var].max()
    # min_value = res_locs[size_var].min()

    # ratio = (max_size - min_size) / (max_value - min_value)
    # sizes = [min_size + i * ratio for i in res_locs[size_var]]
    # markers = ax.scatter(x, y, marker="v", edgecolor="k", s=sizes, zorder=4)
    markers = ax.scatter(x, y, marker="v", edgecolor="k", s=150, zorder=4)
    marker_color = markers.get_facecolor()

    river_line = mlines.Line2D([], [], color="b", alpha=1, linewidth=0.5)
    river_basins = [mpatch.Patch(facecolor=color_map[i], alpha=0.5) for i in range(4)]
    # size_legend_sizes = np.linspace(min_size, max_size, 4)
    # # size_legend_labels = [(i-min_size) / ratio for i in size_legend_sizes]
    # size_legend_labels = np.linspace(min_value, max_value, 4)

    # size_markers = [
    #     plt.scatter([], [], s=i, edgecolors="k", c=marker_color, marker="v")
    #     for i in size_legend_sizes
    # ]

    # size_legend = plt.legend(
    #     size_markers,
    #     [
    #         f"{round(size_legend_labels[0]*1000, -2):.0f}",
    #         *[f"{round(i / 1000, 0):,.0f} million" for i in size_legend_labels[1:]],
    #     ],
    #     title="Maximum Storage [acre-feet]",
    #     loc="lower left",
    #     ncol=4,
    # )
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
    # ax.add_artist(size_legend)
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

    ax = sns.barplot(
        data=scores,
        y="NSE",
        x="Depth",
        hue="label",
    )
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
    files = glob.glob(
        "../results/tclr_model_testing/all/TD?_MSS0.??_RT_MS_exhaustive_new_hoover/results.pickle"
    )
    td_mss_pat = re.compile(r"TD(\d)_MSS(\d\.\d\d)")
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
        whis=(5, 95),
        showfliers=False,
        legend=False
        # ci=None,
    )
    fg.ax.legend(loc="best", ncol=4, title="Max Depth")
    # fg.ax.set_xticklabels(fg.ax.get_xticklabels(), rotation=45, ha="right")
    # if metric == "nRMSE":
    #     fg.ax.set_yscale("log")
    plt.show()


def determine_assim_large_change_reservoirs(scores):
    diff_pairs = [
        ("semi-annually", "seasonally"),
        ("seasonally", "monthly"),
        ("monthly", "weekly"),
        ("weekly", "daily"),
    ]
    output = {}
    models = [("4", "0.10"), ("5", "0.01")]
    for m in models:
        for pair in diff_pairs:
            key1 = (m[0], pair[0], m[1])
            key2 = (m[0], pair[1], m[1])
            diff = scores[key1] - scores[key2]
            # diff_mean = diff.abs().mean()
            outdf = scores[key2].copy()
            outdf["diff"] = diff
            outdf["pdiff"] = diff / scores[key1]
            # outdf = outdf[outdf["diff"].abs().values > threshold]
            output[key2] = outdf
    return output


def determine_assim_improvement(scores):
    output = {}
    models = [("4", "0.10"), ("5", "0.01")]
    assims = ["semi-annually", "seasonally", "monthly", "weekly", "daily"]

    for m in models:
        for assim in assims:
            key1 = (m[0], "never", m[1])
            key2 = (m[0], assim, m[1])
            outdf = scores[key2].copy()
            metric = outdf.columns[0]
            outdf["diff"] = scores[key1] - scores[key2]
            outdf["pdiff"] = outdf["diff"] / scores[key1][metric] * 100
            output[key2] = outdf
    return output


def stripplot(
    data,
    x,
    y,
    hue=None,
    order=None,
    hue_order=None,
    jitter=True,
    dodge=True,
    palette=None,
    size=None,
    ax=None,
    colors=None,
    **kwgs,
):

    df = data.copy()
    df = df.rename(columns={x: "x", y: "y", hue: "hue", size: "size"})

    order = order if order else df["x"].unique()
    hue_order = hue_order if hue_order else df["hue"].unique()

    xticks = list(range(df["x"].unique().size))
    nhue = df["hue"].unique().size
    width = 0.8 / nhue
    if dodge:
        adjs = linear_scale_values(range(nhue), -0.4 + width / 2, 0.4 - width / 2)
    else:
        adjs = [0.0 for i in range(nhue)]

    if jitter is True:
        jlim = 0.1
    else:
        jlim = float(jitter)
    if dodge:
        jlim /= nhue

    jlim *= width
    jitterer = partial(np.random.uniform, low=-jlim, high=+jlim)
    njitter = df.shape[0] / nhue / len(order)
    jitter = jitterer(size=int(njitter))

    for i, hue in enumerate(hue_order):
        x_adj = [x + adjs[i] for x in xticks]

        pdf = df[df["hue"] == hue]
        for j, var in enumerate(order):
            vdf = pdf[pdf["x"] == var]
            if colors:
                color = vdf[colors]
            else:
                color = palette[i]
            ax.scatter(x_adj[j] + jitter, vdf["y"], color=color, s=vdf["size"], **kwgs)
    return ax


def plot_data_assim_results(metric="NSE"):
    files = glob.glob(
        "../results/tclr_model_testing/all/TD4_*_MSS0.??_RT_MS_exhaustive_new_hoover/results.pickle"
    ) + glob.glob(
        "../results/tclr_model_testing/all/TD5_*_MSS0.??_RT_MS_exhaustive_new_hoover/results.pickle"
    )
    td_mss_assim_pat = re.compile(r"TD(\d)_(.*)_MSS(\d\.\d\d)")
    matches = [re.search(td_mss_assim_pat, i) for i in files]
    td_mss_assim = [i.groups() for i in matches]
    results = {}
    for key, file in zip(td_mss_assim, files):
        with open(file, "rb") as f:
            results[key] = pickle.load(f)
    with open(
        "../results/tclr_model_testing/all/TD4_MSS0.10_RT_MS_exhaustive_new_hoover/results.pickle",
        "rb",
    ) as f:
        results[("4", "never", "0.10")] = pickle.load(f)
    with open(
        "../results/tclr_model_testing/all/TD5_MSS0.01_RT_MS_exhaustive_new_hoover/results.pickle",
        "rb",
    ) as f:
        results[("5", "never", "0.01")] = pickle.load(f)

    simmed_data = select_results(results, "simmed_data")

    simmed_scores = get_model_scores(simmed_data, metric=metric, grouper="site_name")

    improvements = determine_assim_improvement(simmed_scores)
    # del simmed_scores[("4", "never", "0.10")]
    # del simmed_scores[("5", "never", "0.01")]
    simmed_large_diffs = determine_assim_large_change_reservoirs(simmed_scores)

    columns = ["TD", "Assim", "MSS", "Reservoir", metric]
    simmed_scores_records = []
    for key, values in simmed_scores.items():
        for res, value in values[metric].items():
            simmed_scores_records.append([int(key[0]), key[1], float(key[2]), res, value])
    simmed_scores = pd.DataFrame.from_records(simmed_scores_records, columns=columns)

    simmed_diff_records = []
    for key, values in simmed_large_diffs.items():
        for i, row in values.iterrows():
            simmed_diff_records.append(
                [
                    int(key[0]),
                    key[1],
                    float(key[2]),
                    i,
                    row[metric],
                    row["diff"],
                    row["pdiff"],
                ]
            )
    simmed_diffs = pd.DataFrame.from_records(
        simmed_diff_records, columns=columns + ["diff", "pdiff"]
    )

    simmed_improve_records = []
    for key, values in improvements.items():
        for i, row in values.iterrows():
            simmed_improve_records.append(
                [
                    int(key[0]),
                    key[1],
                    float(key[2]),
                    i,
                    row[metric],
                    row["diff"],
                    row["pdiff"],
                ]
            )
    simmed_improves = pd.DataFrame.from_records(
        simmed_improve_records, columns=columns + ["diff", "pdiff"]
    )

    simmed_scores = simmed_scores.set_index(["TD", "Assim", "MSS", "Reservoir"])
    simmed_diffs = simmed_diffs.set_index(["TD", "Assim", "MSS", "Reservoir"])
    simmed_improves = simmed_improves.set_index(["TD", "Assim", "MSS", "Reservoir"])

    simmed_scores["pdiff_improve"] = simmed_improves["pdiff"]
    simmed_scores["pdiff_improve"] = simmed_scores["pdiff_improve"].fillna(0.0)

    min_size = 20
    max_size = 100
    simmed_scores["size"] = linear_scale_values(
        simmed_scores["pdiff_improve"], min_size, max_size
    )

    simmed_scores["pdiff_diff"] = simmed_diffs["pdiff"]
    simmed_scores["color"] = simmed_scores["pdiff_diff"].apply(
        lambda x: "#00FF00" if x > 0.5 else "#808080"
    )

    simmed_scores = simmed_scores.reset_index()

    # simmed_scores_for_strip = simmed_scores.merge(simmed_diffs, indicator=True, how="outer").query(
    #     '_merge=="left_only"').drop("_merge", axis=1)

    II()
    sys.exit()
    fg = sns.catplot(
        data=simmed_scores,
        hue="TD",
        x="Assim",
        y=metric,
        kind="box",
        order=["daily", "weekly", "monthly", "seasonally", "semi-annually", "never"],
        whis=(5, 95),
        showfliers=False,
        legend_out=False,
        palette="colorblind",
    )
    # stripplot(
    #     data=simmed_scores,
    #     hue="TD",
    #     x="Assim",
    #     y=metric,
    #     order=["daily", "weekly", "monthly", "seasonally", "semi-annually"],
    #     ax=fg.ax,
    #     dodge=True,
    #     jitter=0.8,
    #     # palette=["#808080", "#808080"],
    #     colors="color",
    #     edgecolor="k",
    #     alpha=0.5,
    #     linewidth=1,
    #     size="size",
    # )

    fg.ax.set_xticklabels(
        ["Daily", "Weekly", "Monthly", "Seasonally", "Semi-annually", "Never"]
    )
    fg.ax.set_xlabel("Assimilation Frequency")

    fg.ax.set_xticks([], minor=True)
    fg.despine(left=False, right=False, top=False, bottom=False)

    handles, labels = fg.ax.get_legend_handles_labels()
    handles = handles[:2]
    labels = ["TD4-MSS0.10", "TD5-MSS0.01"]
    fg.ax.legend(handles, labels, loc="best")
    sns.despine(ax=fg.ax)
    # fg.ax.set_ylabel(f"Percent Improvement in RMSE")

    # fg.ax.set_yscale("log")
    # fg.ax.set_yticks([0.1, 1.0])
    # fg.ax.set_yticklabels(["0.1", "1.0"])

    # size_leg_values = np.linspace(
    #     simmed_scores["pdiff_improve"].min(), simmed_scores["pdiff_improve"].max(), 4
    # )
    # size_leg_sizes = linear_scale_values(size_leg_values, min_size, max_size)
    # size_leg_markers = [
    #     plt.scatter(
    #         [0],
    #         [0],
    #         s=i,
    #         edgecolors="k",
    #         c="#808080",
    #         alpha=0.5,
    #         linewidths=1,
    #     )
    #     for i in size_leg_sizes
    # ]

    # size_leg_labels = [f"{i:.0%}" for i in size_leg_values]
    # fg.ax.legend(
    #     size_leg_markers,
    #     size_leg_labels,
    #     ncol=4,
    #     loc="lower left",
    #     # title="",
    #     fontsize=8,
    #     handlelength=1,
    #     columnspacing=1,
    #     title_fontsize=8,
    #     handletextpad=1,
    #     borderpad=1,
    #     labelspacing=1,
    # )
    plt.show()


def plot_assim_score_ridgelines(metric="NSE"):
    files = glob.glob(
        "../results/tclr_model_testing/all/TD?_*_MSS0.??_RT_MS_exhaustive_new_hoover/results.pickle"
    )
    td_mss_assim_pat = re.compile(r"TD(\d)_(.*)_MSS(\d\.\d\d)")
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
        "../results/tclr_model_testing/all/TD4_MSS0.10_RT_MS_exhaustive_new_hoover/results.pickle",
        "../results/tclr_model_testing/all/TD5_MSS0.01_RT_MS_exhaustive_new_hoover/results.pickle",
    ]
    keys = [(4, 0.1), (5, 0.01)]
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

    ascending = True
    if metric == "nRMSE":
        ascending = False

    train_top_20 = {
        key: df.sort_values(by=metric, ascending=ascending).tail(20)
        for key, df in train_scores.items()
    }
    test_top_20 = {
        key: df.sort_values(by=metric, ascending=ascending).tail(20)
        for key, df in test_scores.items()
    }
    simmed_top_20 = {
        key: df.sort_values(by=metric, ascending=ascending).tail(20)
        for key, df in simmed_scores.items()
    }
    train_btm_20 = {
        key: df.sort_values(by=metric, ascending=ascending).head(20)
        for key, df in train_scores.items()
    }
    test_btm_20 = {
        key: df.sort_values(by=metric, ascending=ascending).head(20)
        for key, df in test_scores.items()
    }
    simmed_btm_20 = {
        key: df.sort_values(by=metric, ascending=ascending).head(20)
        for key, df in simmed_scores.items()
    }
    records = []

    for key in keys:
        for res, value in train_top_20[key][metric].items():
            records.append((key, "Train", "Best 20", res, value))
        for res, value in test_top_20[key][metric].items():
            records.append((key, "Test", "Best 20", res, value))
        for res, value in simmed_top_20[key][metric].items():
            records.append((key, "Simmed", "Best 20", res, value))
        for res, value in train_btm_20[key][metric].items():
            records.append((key, "Train", "Worst 20", res, value))
        for res, value in test_btm_20[key][metric].items():
            records.append((key, "Test", "Worst 20", res, value))
        for res, value in simmed_btm_20[key][metric].items():
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
        kind="violin",
        # whis=(0.1, 0.9),
        # showfliers=True,
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


def plot_two_best_models(metric="NSE"):
    files = [
        "../results/tclr_model_testing/all/TD4_MSS0.10_RT_MS_exhaustive_new_hoover/results.pickle",
        "../results/tclr_model_testing/all/TD5_MSS0.01_RT_MS_exhaustive_new_hoover/results.pickle",
    ]
    keys = [(4, 0.1), (5, 0.01)]
    key_map = {i: f"TD{i[0]}-MSS{i[1]:.2f}" for i in keys}
    results = {}
    for k, f in zip(keys, files):
        with open(f, "rb") as f:
            results[k] = pickle.load(f)

    simmed_data = select_results(results, "simmed_data")

    simmed_nrmse = get_model_scores(simmed_data, metric="nRMSE", grouper="site_name")
    simmed_nse = get_model_scores(simmed_data, metric="NSE", grouper="site_name")

    dfs = []
    for key, label in key_map.items():
        df1 = simmed_nrmse[key]
        # kdf["metric"] = "nRMSE"
        df1 = df1.set_index(
            pd.MultiIndex.from_product(([label], df1.index), names=["model", "site_name"])
        )

        df2 = simmed_nse[key]
        df2 = df2.set_index(
            pd.MultiIndex.from_product(([label], df2.index), names=["model", "site_name"])
        )
        df1["NSE"] = df2["NSE"]
        # kdf["model"] = label
        # # kdf["metric"] = "NSE"
        # # kdf = kdf.rename(columns={"NSE": "value"})
        dfs.append(df1)
    sdf = pd.concat(dfs).reset_index()
    II()
    sys.exit()
    sdf = simmed_scores[keys[0]]
    sdf[keys[1]] = simmed_scores[keys[1]]
    sdf = sdf.rename(columns=key_map)
    sdf = sdf.reset_index().melt(id_vars=["site_name"])

    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(19, 10))
    # plt.scatter(sdf[keys[0]], sdf[keys[1]])

    # ax.set_xlabel(keys[0])
    # ax.set_ylabel(keys[1])

    fg = sns.catplot(data=sdf, x="value", y="variable", kind="box", whis=(5, 95))

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


def get_res_characteristic(metric=None):
    from find_basin_groups import prep_seasonalities
    from tclr_model import get_basin_meta_data, read_basin_data

    files = [
        "../results/tclr_model_testing/all/TD4_MSS0.10_RT_MS_exhaustive_new_hoover/results.pickle",
        "../results/tclr_model_testing/all/TD5_MSS0.01_RT_MS_exhaustive_new_hoover/results.pickle",
    ]
    keys = [(4, 0.1), (5, 0.01)]
    results = {}
    for k, f in zip(keys, files):
        with open(f, "rb") as f:
            results[k] = pickle.load(f)

    # train_data = select_results(results, "train_data")
    # test_data = select_results(results, "test_data")
    simmed_data = select_results(results, "simmed_data")

    # train_scores = get_model_scores(train_data, metric=metric, grouper="site_name")
    # test_scores =  get_model_scores(test_data, metric=metric, grouper="site_name")
    if metric:
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
    if metric:
        for key, value in simmed_scores.items():
            df_key = f"TD{key[0]}-MSS{key[1]:.2f}"
            char_df[df_key] = value
    return char_df


def plot_top_characteristic_res_scatter(metric="NSE"):
    char_df = get_res_characteristic(metric)
    fig, axes = plt.subplots(3, 2, sharey=True, sharex=False)
    axes = axes.flatten()
    markers = [
        "o",
        "s",
    ]
    # colors = sns.color_palette("Tab2", 2)
    for i, var in enumerate(CHAR_VARS):
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
    fig.patch.set_alpha(0.0)

    zeros = np.zeros_like(char_df["TD2-MSS0.20"].values)
    max_size = 500
    min_size = 20

    colors = sns.color_palette("tab10", 2)
    for i, var in enumerate(CHAR_VARS):
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

    axes.set_yticks(range(len(CHAR_VARS)))
    axes.set_yticklabels(CHAR_VARS)
    handles, labels = axes.get_legend_handles_labels()
    axes.legend(handles[:2], labels[:2], loc="best")
    fig.align_ylabels()
    plt.show()


def plot_res_characteristic_bin_performance(metric="NSE", nbins=3):
    char_df = get_res_characteristic(metric)
    cuts = {}
    for var in CHAR_VARS:
        values, labels = pd.qcut(char_df[var], nbins, labels=False, retbins=True)
        char_df[var] = values + 1
        cuts[var] = labels

    label_map = make_bin_label_map(nbins, start_index=1)
    df = char_df.melt(
        id_vars=["TD4-MSS0.10", "TD5-MSS0.01"], value_name="bin", ignore_index=False
    ).melt(
        id_vars=["variable", "bin"],
        var_name="model",
        value_name=metric,
        ignore_index=False,
    )
    df["bin"] = df["bin"].replace(label_map)

    fg = sns.catplot(
        data=df,
        x="bin",
        y=metric,
        hue="model",
        col="variable",
        col_wrap=2,
        kind="bar",
        legend_out=False,
        errorbar=None,
        # errwidth=1,
        # capsize=0.1,
        palette="colorblind",
        order=[label_map[i] for i in range(1, nbins + 1)],
    )
    for ax in fg.axes:
        ax.grid(False)
        ax.patch.set_alpha(0.0)
        ax.set_axis_on()
        ax.spines["bottom"].set_color("black")
        ax.spines["left"].set_color("black")
        ax.tick_params(axis="both", color="black", labelcolor="black")

    fg.set_titles("{col_name}")
    fg.set_ylabels(metric, color="black")
    fg.set_xlabels("Attribute Percentile", color="black")
    handles, labels = fg.axes[0].get_legend_handles_labels()
    fg.axes[0].legend(handles, labels, frameon=False)
    plt.show()


def plot_res_characteristic_scatter_performance(metric="NSE"):
    char_df = get_res_characteristic(metric)

    df = char_df.melt(id_vars=["TD4-MSS0.10", "TD5-MSS0.01"], ignore_index=False).melt(
        id_vars=["variable", "value"],
        var_name="model",
        value_name=metric,
        ignore_index=False,
    )

    fg = sns.relplot(
        data=df,
        x="value",
        y=metric,
        hue="model",
        col="variable",
        col_wrap=2,
        kind="scatter",
        palette="colorblind",
    )
    for ax in fg.axes:
        ax.grid(False)
        ax.patch.set_alpha(0.0)
        ax.set_axis_on()
        ax.spines["bottom"].set_color("black")
        ax.spines["left"].set_color("black")
        ax.tick_params(axis="both", color="black", labelcolor="black")

    fg.set_titles("{col_name}")
    fg.set_ylabels(metric, color="black")
    fg.set_xlabels("Attribute Percentile", color="black")
    handles, labels = fg.axes[0].get_legend_handles_labels()
    fg.axes[0].legend(handles, labels, frameon=False)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    # II()
    plt.show()


def plot_res_characteristic_dist():
    char_df = get_res_characteristic()
    df = char_df.melt(var_name="Characteristic")

    fg = sns.displot(
        data=df,
        x="value",
        col="Characteristic",
        col_wrap=2,
        kind="hist",
        # legend_out=False,
        palette="colorblind",
        facet_kws={"subplot_kws": {"sharex": False, "sharey": False}},
    )

    # fg.set_titles("{col_name}")
    # fg.set_ylabels(metric, color="black")
    # fg.set_xlabels("Attribute Percentile", color="black")
    # handles, labels = fg.axes[0].get_legend_handles_labels()
    # fg.axes[0].legend(handles, labels, frameon=False)
    plt.show()


def plot_res_characteristic_map(metric="NSE"):
    char_df = get_res_characteristic(metric)

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
    for ax, var, unit in zip(axes, CHAR_VARS, units):
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
        if var == CHAR_VARS[2]:
            legend_labels = [f"{round(i, -2):.0f}" for i in legend_scores]
        elif var == CHAR_VARS[3]:
            legend_labels = [f"{round(i, -1):.0f}" for i in legend_scores]
        elif var == CHAR_VARS[5]:
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


def plot_res_characteristic_split_map():
    char_df = get_res_characteristic()
    rbasins = pd.read_pickle("../pickles/res_basin_map.pickle")
    rename = {
        "upper_col": "colorado",
        "lower_col": "colorado",
        "pnw": "columbia",
        "tva": "tennessee",
    }
    rbasins = rbasins.replace(rename)
    rbasins = rbasins.str.capitalize()
    char_df["basin"] = rbasins

    res_locs = pd.read_csv("../geo_data/reservoirs.csv")
    res_locs = res_locs.set_index("site_name")
    char_df["x"] = res_locs["long"]
    char_df["y"] = res_locs["lat"]

    for var in CHAR_VARS:
        char_df[var] = char_df[var].rank(pct=True)

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
    basin_color_map = sns.color_palette("Set2")
    basin_info = {
        "Columbia": {
            "extents": coords["Columbia"],
            "shp": (GIS_DIR / "columbia_shp" / "Shape" / "WBDHU2").as_posix(),
            "color": basin_color_map[0],
            "rivers": [
                (
                    GIS_DIR
                    / "NHDPlus"
                    / "trimmed_flowlines"
                    / "NHDPlusPN_trimmed_flowlines_noz"
                ).as_posix()
            ],
        },
        "Missouri": {
            "extents": coords["Missouri"],
            "shp": (GIS_DIR / "missouri_shp" / "Shape" / "WBDHU2").as_posix(),
            "color": basin_color_map[1],
            "rivers": [
                (
                    GIS_DIR
                    / "NHDPlus"
                    / "trimmed_flowlines"
                    / "NHDPlusMU_trimmed_flowlines_noz"
                ).as_posix(),
                (
                    GIS_DIR
                    / "NHDPlus"
                    / "trimmed_flowlines"
                    / "NHDPlusML_trimmed_flowlines_noz"
                ).as_posix(),
            ],
        },
        "Colorado": {
            "extents": coords["Colorado"],
            "shp": (GIS_DIR / "colorado_shp" / "Shape" / "WBDHU2").as_posix(),
            "color": basin_color_map[2],
            "rivers": [
                (
                    GIS_DIR
                    / "NHDPlus"
                    / "trimmed_flowlines"
                    / "NHDPlusUC_trimmed_flowlines_noz"
                ).as_posix(),
                (
                    GIS_DIR
                    / "NHDPlus"
                    / "trimmed_flowlines"
                    / "NHDPlusLC_trimmed_flowlines_noz"
                ).as_posix(),
            ],
        },
        "Tennessee": {
            "extents": coords["Tennessee"],
            "shp": (GIS_DIR / "tennessee_shp" / "Shape" / "WBDHU2").as_posix(),
            "color": basin_color_map[3],
            "rivers": [
                (
                    GIS_DIR
                    / "NHDPlus"
                    / "trimmed_flowlines"
                    / "NHDPlusTN_trimmed_flowlines_noz"
                ).as_posix()
            ],
        },
    }

    Parallel(n_jobs=len(CHAR_VARS), verbose=11)(
        delayed(char_split_map_single_var)(var, char_df, basin_info, True, False)
        for var in CHAR_VARS
    )


def char_split_map_single_var(
    var, char_df, basin_info, save=False, add_legend=False, legend_labels=None
):
    vmin = char_df[var].min()
    vmax = char_df[var].max()
    norm = Normalize(vmin=vmin, vmax=vmax)
    # cmap = "inferno"
    # color_map = get_cmap(cmap)
    colors = ("#DAFF47", "#EDA200", "#D24E71", "#91008D", "#001889")
    color_map = ListedColormap(colors, "qual_inferno")
    fig = plt.figure(figsize=(19, 10))
    # if add_legend:
    #     gs = GS.GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 0.4])
    # else:
    gs = GS.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
    fig.patch.set_alpha(0.0)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]

    positions = [
        Bbox(
            [
                [0.24340625350138054, 0.5379881391965984],
                [0.4408909297545936, 0.9728096676737158],
            ]
        ),
        Bbox(
            [
                [0.5086472454653168, 0.5379881391965984],
                [0.8070555712787091, 0.9728096676737158],
            ]
        ),
        Bbox(
            [
                [0.2452846166142043, 0.05158330535974043],
                [0.43901256664176985, 0.48640483383685795],
            ]
        ),
        Bbox(
            [
                [0.47861855895825467, 0.05158330535974043],
                [0.8370842577857711, 0.48640483383685795],
            ]
        ),
    ]

    for ax, pos, (basin, binfo) in zip(axes, positions, basin_info.items()):
        make_basin_map(ax, binfo)
        ax.set_position(pos)
        bdf = char_df[char_df["basin"] == basin]
        ax.scatter(
            bdf["x"],
            bdf["y"],
            edgecolor="k",
            linewidths=0.5,
            facecolor=[color_map(norm(i)) for i in bdf[var].values],
            zorder=4,
            s=250,
        )

    if add_legend:
        # leg_ax = fig.add_subplot(gs[:, 2])
        leg_fig = plt.figure()
        leg_ax = leg_fig.add_subplot()
        leg_ax.set_axis_off()
        handles = [
            mpatch.Patch(facecolor=c, edgecolor="k", linewidth=0.5) for c in colors
        ]
        leg_ax.legend(handles, legend_labels, loc="center", frameon=False)

    # mng = plt.get_current_fig_manager()
    # mng.window.showMaximized()
    if save:
        plt.savefig(
            f"C:\\Users\\lcford2\\Dropbox\\PHD\\multibasin_model_figures\\new_paper_figures\\split_char_map_{var}_larger.png",
            dpi=400,
        )
    return fig, axes


def plot_data_assim_scatter(metric="NSE"):
    files = glob.glob(
        "../results/tclr_model_testing/all/TD?_*_MSS0.??_RT_MS_exhaustive_new_hoover/results.pickle"
    )
    td_mss_assim_pat = re.compile(r"TD(\d)_(.*)_MSS(\d\.\d\d)")
    matches = [re.search(td_mss_assim_pat, i) for i in files]
    td_mss_assim = [i.groups() for i in matches]
    results = {}
    for key, file in zip(td_mss_assim, files):
        with open(file, "rb") as f:
            results[key] = pickle.load(f)
    simmed_data = select_results(results, "simmed_data")

    simmed_scores = get_model_scores(simmed_data, metric=metric, grouper="site_name")

    fig, axes = plt.subplots(1, 2)
    axes = axes.flatten()

    ya = "daily"
    xa = "weekly"
    y1_key = ("5", ya, "0.01")
    x1_key = ("5", xa, "0.01")
    y2_key = ("2", ya, "0.20")
    x2_key = ("2", xa, "0.20")
    ax = axes[0]
    ax.scatter(simmed_scores[x1_key], simmed_scores[y1_key], label="TD5-MSS0.01")
    ax.scatter(simmed_scores[x2_key], simmed_scores[y2_key], label="TD2-MSS0.20")
    ax.set_ylabel(f"{ya.capitalize()} Assim. Performance ({metric})")
    ax.set_xlabel(f"{xa.capitalize()} Assim. Performance ({metric})")

    ya = "monthly"
    xa = "semi-annually"
    y1_key = ("5", ya, "0.01")
    x1_key = ("5", xa, "0.01")
    y2_key = ("2", ya, "0.20")
    x2_key = ("2", xa, "0.20")
    ax = axes[1]
    ax.scatter(simmed_scores[x1_key], simmed_scores[y1_key], label="TD5-MSS0.01")
    ax.scatter(simmed_scores[x2_key], simmed_scores[y2_key], label="TD2-MSS0.20")
    ax.set_ylabel(f"{ya.capitalize()} Assim. Performance ({metric})")
    ax.set_xlabel(f"{xa.capitalize()} Assim. Performance ({metric})")

    xmin = 100000
    ymin = 100000
    xmax = -100000
    ymax = -100000
    for ax in axes:
        xlim = ax.get_xlim()
        if xlim[0] < xmin:
            xmin = xlim[0]
        if xlim[1] > xmax:
            xmax = xlim[1]
        ylim = ax.get_ylim()
        if ylim[0] < ymin:
            ymin = ylim[0]
        if ylim[1] > ymax:
            ymax = ylim[1]

    for ax in axes:
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))
        # ax.set_xscale("log")
        # ax.set_yscale("log")

    plt.show()


def plot_reservoir_metric(metric, ds="simul"):
    files = glob.glob(
        "../results/tclr_model_testing/all/TD?_*_MSS0.??_RT_MS_exhaustive_new_hoover/results.pickle"
    )
    td_mss_assim_pat = re.compile(r"TD(\d)_(.*)_MSS(\d\.\d\d)")
    matches = [re.search(td_mss_assim_pat, i) for i in files]
    td_mss_assim = [i.groups() for i in matches]
    results = {}
    for key, file in zip(td_mss_assim, files):
        with open(file, "rb") as f:
            results[key] = pickle.load(f)
    simmed_data = select_results(results, "simmed_data")

    simmed_scores = get_model_scores(simmed_data, metric=metric, grouper="site_name")

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

    df["model"] = ["M1" if i == 5 else "M2" for i in df["TD"]]
    df = df.sort_values(by="model")

    fg = sns.catplot(
        data=df,
        hue="model",
        x="Reservoir",
        y=metric,
        kind="bar",
        palette="tab10",
        # whis=(5, 95),
        # showfliers=False,
        legend=False
        # ci=None,
    )
    # fg.ax.legend(loc="best", ncol=4, title="Max Depth")
    # fg.ax.set_xticklabels(fg.ax.get_xticklabels(), rotation=45, ha="right")
    # if metric == "nRMSE":
    #     fg.ax.set_yscale("log")
    plt.show()


def plot_assim_metric_scatter():
    file_templates = [
        "../results/tclr_model_testing/all/TD4_*_MSS0.10_RT_MS_exhaustive_new_hoover/results.pickle",
        "../results/tclr_model_testing/all/TD5_*_MSS0.01_RT_MS_exhaustive_new_hoover/results.pickle",
    ]
    keys = [(4, 0.1), (5, 0.01)]
    results = {}
    td_mss_assim_pat = re.compile(r"TD(\d)_(.*)_MSS(\d\.\d\d)")
    for ft in file_templates:
        files = glob.glob(ft)
        matches = [re.search(td_mss_assim_pat, i) for i in files]
        td_mss_assim = [i.groups() for i in matches]
        for file, tma in zip(files, td_mss_assim):
            with open(file, "rb") as f:
                results[tma] = pickle.load(f)

    simmed_data = select_results(results, "simmed_data")

    simmed_pbias = get_model_scores(simmed_data, metric="PBIAS", grouper="site_name")
    simmed_nrmse = get_model_scores(simmed_data, metric="nRMSE", grouper="site_name")

    columns = ["TD", "Assim", "MSS", "Reservoir"]

    simmed_pbias_records = []
    for key, values in simmed_pbias.items():
        for res, value in values["PBIAS"].items():
            simmed_pbias_records.append([int(key[0]), key[1], float(key[2]), res, value])
    simmed_pbias = pd.DataFrame.from_records(
        simmed_pbias_records, columns=columns + ["pbias"]
    )
    simmed_nrmse_records = []
    for key, values in simmed_nrmse.items():
        for res, value in values["nRMSE"].items():
            simmed_nrmse_records.append([int(key[0]), key[1], float(key[2]), res, value])
    simmed_nrmse = pd.DataFrame.from_records(
        simmed_nrmse_records, columns=columns + ["nrmse"]
    )

    df = pd.merge(simmed_pbias, simmed_nrmse)
    df["model"] = ["M1" if i == 5 else "M2" for i in df["TD"]]

    sns.jointplot(
        data=df,
        x="pbias",
        y="nrmse",
        hue="Assim",
        hue_order=["daily", "weekly", "monthly", "seasonally", "semi-annually"],
        # joint_kws=dict(style="model"),
        kind="scatter",
    )
    plt.show()


def plot_basin_performance(metric="nRMSE"):
    char_df = get_res_characteristic(metric)
    res_basin_map = pd.read_pickle("../pickles/res_basin_map.pickle")

    rename = {
        "upper_col": "colorado",
        "lower_col": "colorado",
        "pnw": "columbia",
        "tva": "tennessee",
    }
    res_basin_map = res_basin_map.replace(rename)
    res_basin_map = res_basin_map.str.capitalize()

    char_df["basin"] = res_basin_map
    II()
    df = char_df[["TD4-MSS0.10", "TD5-MSS0.01", "basin"]]
    df = df.melt(id_vars=["basin"], var_name="model", value_name=metric)

    sns.catplot(
        data=df,
        x="basin",
        y=metric,
        hue="model",
        palette="colorblind",
        kind="box",
    )
    plt.show()


def plot_assim_lines():
    files = glob.glob(
        "../results/tclr_model_testing/all/TD4_*_MSS0.??_RT_MS_exhaustive_new_hoover/results.pickle"
    ) + glob.glob(
        "../results/tclr_model_testing/all/TD5_*_MSS0.??_RT_MS_exhaustive_new_hoover/results.pickle"
    )
    td_mss_assim_pat = re.compile(r"TD(\d)_(.*)_MSS(\d\.\d\d)")
    matches = [re.search(td_mss_assim_pat, i) for i in files]
    td_mss_assim = [i.groups() for i in matches]
    results = {}
    for key, file in zip(td_mss_assim, files):
        with open(file, "rb") as f:
            results[key] = pickle.load(f)
    with open(
        "../results/tclr_model_testing/all/TD4_MSS0.10_RT_MS_exhaustive_new_hoover/results.pickle",
        "rb",
    ) as f:
        results[("4", "never", "0.10")] = pickle.load(f)
    with open(
        "../results/tclr_model_testing/all/TD5_MSS0.01_RT_MS_exhaustive_new_hoover/results.pickle",
        "rb",
    ) as f:
        results[("5", "never", "0.01")] = pickle.load(f)

    simmed_data = select_results(results, "simmed_data")

    keys = list(simmed_data.keys())

    resers = simmed_data[keys[0]].index.get_level_values(0).unique()

    for res in resers:
        df = pd.DataFrame()
        for assim in [
            "daily",
            "weekly",
            "monthly",
            "seasonally",
            "semi-annually",
            "never",
        ]:
            mdf = simmed_data[("4", assim, "0.10")]
            mdf = mdf.loc[pd.IndexSlice[res, :]]
            if df.empty:
                df["actual"] = mdf["actual"]
                df[assim] = mdf["model"]
            else:
                df[assim] = mdf["model"]
        # df = df.reset_index().melt(
        #     id_vars=["datetime"], var_name="assim", value_name="release"
        # )
        # fig, ax = plt.subplots(1, 1, figsize=(38, 20))
        # ax = sns.lineplot(
        #     data=df,
        #     x="datetime",
        #     y="release",
        #     estimator=None,
        #     errorbar=None,
        #     hue="assim",
        #     ax=ax,
        #     palette="tab10",
        #     alpha=0.8,
        # )
        # ax.set_title(res)
        fig, axes = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(38, 20))
        axes = axes.flatten()
        for ax, assim in zip(
            axes, ["never", "semi-annually", "seasonally", "monthly", "weekly", "daily"]
        ):
            ax.plot(df.index, df["actual"], label="Actual")
            ax.plot(df.index, df[assim], label=assim.title())
            ax.set_title(assim.title())
            ax.legend(loc="best")
        fig.suptitle(res)

        plt.tight_layout()
        plt.savefig(f"../figures/release_timeseries_split/{res}.png")
        plt.close()


def plot_data_assim_scatter():
    files = glob.glob(
        "../results/tclr_model_testing/all/TD4_*_MSS0.??_RT_MS_exhaustive_new_hoover/results.pickle"
    ) + glob.glob(
        "../results/tclr_model_testing/all/TD5_*_MSS0.??_RT_MS_exhaustive_new_hoover/results.pickle"
    )
    td_mss_assim_pat = re.compile(r"TD(\d)_(.*)_MSS(\d\.\d\d)")
    matches = [re.search(td_mss_assim_pat, i) for i in files]
    td_mss_assim = [i.groups() for i in matches]
    td_mss_assim = [i for i in td_mss_assim if "yearly" not in i]
    results = {}
    for key, file in zip(td_mss_assim, files):
        with open(file, "rb") as f:
            results[key] = pickle.load(f)
    with open(
        "../results/tclr_model_testing/all/TD4_MSS0.10_RT_MS_exhaustive_new_hoover/results.pickle",
        "rb",
    ) as f:
        results[("4", "never", "0.10")] = pickle.load(f)
    with open(
        "../results/tclr_model_testing/all/TD5_MSS0.01_RT_MS_exhaustive_new_hoover/results.pickle",
        "rb",
    ) as f:
        results[("5", "never", "0.01")] = pickle.load(f)

    simmed_data = select_results(results, "simmed_data")

    simmed_nse = get_model_scores(simmed_data, metric="NSE", grouper="site_name")
    simmed_nrmse = get_model_scores(simmed_data, metric="nRMSE", grouper="site_name")

    columns = ["TD", "Assim", "MSS", "Reservoir", "Metric", "Value"]
    simmed_scores_records = []
    for key, values in simmed_nse.items():
        for res, value in values["NSE"].items():
            simmed_scores_records.append(
                [int(key[0]), key[1], float(key[2]), res, "NSE", value]
            )
    for key, values in simmed_nrmse.items():
        for res, value in values["nRMSE"].items():
            simmed_scores_records.append(
                [int(key[0]), key[1], float(key[2]), res, "nRMSE", value]
            )

    simmed_scores = pd.DataFrame.from_records(simmed_scores_records, columns=columns)
    simmed_scores = simmed_scores.pivot(
        index=["TD", "Assim", "MSS", "Reservoir"], columns=["Metric"]
    )
    simmed_scores.columns = simmed_scores.columns.droplevel(0)
    simmed_scores = simmed_scores.reset_index()
    simmed_scores = simmed_scores[simmed_scores["TD"] == 4]

    assim_codes = {
        "never": 0,
        "semi-annually": 1,
        "seasonally": 2,
        "monthly": 3,
        "weekly": 4,
        "daily": 5,
    }
    simmed_scores["Assim"] = simmed_scores["Assim"].replace(assim_codes)
    # simmed_scores = simmed_scores[simmed_scores["Assim"].isin(["never", "daily"])]
    II()
    sys.exit()

    ax = sns.scatterplot(
        data=simmed_scores,
        # style="TD",
        hue="Assim",
        y="NSE",
        x="nRMSE",
        hue_order=["never", "semi-annually", "seasonally", "monthly", "weekly", "daily"],
        alpha=0.6,
        palette="viridis",
    )
    # ax.set_xscale("log")
    plt.show()
    # II()
    # sys.exit()


def get_unique_trees():
    with open(
        "../results/tclr_model_testing/all/TD4_MSS0.10_RT_MS_exhaustive_new_hoover/results.pickle",
        "rb",
    ) as f:
        results = pickle.load(f)

    groups = results["groups"]
    train_data = results["train_data"]
    train_data["Group"] = groups

    df = train_data.reset_index()

    value_counts = df.groupby(["site_name", df["datetime"].dt.month])[
        "Group"
    ].value_counts()  # [no-member]
    value_counts.name = "Count"
    value_counts = value_counts.reset_index().set_index(
        ["site_name", "datetime", "Group"]
    )

    group_error = df.groupby(["site_name", df["datetime"].dt.month, "Group"]).apply(
        lambda x: total_absolute_error(x["actual"], x["model"])
    )
    value_counts["error"] = group_error
    value_counts = value_counts.reset_index()

    single_group_res = []
    resers = value_counts["site_name"].unique()

    for res in resers:
        rgroups = value_counts[value_counts["site_name"] == res]["Group"]
        rgroups = rgroups.unique()
        if len(rgroups) == 1:
            single_group_res.append((res, rgroups[0]))
    drop_res = [i[0] for i in single_group_res]

    value_counts = value_counts[~value_counts["site_name"].isin(drop_res)]

    plot_res = value_counts["site_name"].unique()
    name_replacements = get_name_replacements()
    rbasins = pd.read_pickle("../pickles/res_basin_map.pickle")

    group_names = {
        7: "High Inflow Storage Maintenance (7)",
        3: "Low Steady Release (3)",
        4: "Storage Build-Up (4)",
        5: "Storage and Release Maintenance (5)",
        6: "Release Trend Persistence (6)",
    }

    style_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for res in plot_res:
        basin = rbasins[res]
        print_basin = " ".join(basin.split("_")).title()
        print_res = name_replacements.get(res, res).title()

        rdf = value_counts[value_counts["site_name"] == res]
        counts = rdf.pivot(index=["datetime"], columns=["Group"], values=["Count"])
        counts = counts.divide(counts.sum(axis=1).values, axis=0) * 100
        counts.columns = counts.columns.droplevel(0)
        counts = counts.fillna(0.0)

        error = rdf.pivot(index=["datetime"], columns=["Group"], values=["error"])
        error = error.divide(error.sum(axis=1).values, axis=0) * 100
        error.columns = error.columns.droplevel(0)
        error = error.fillna(0.0)

        fig = plt.figure(figsize=(19, 10))
        ax = fig.add_subplot()

        # rdf.plot.bar(stacked=True, ax=ax)
        width = 0.45
        x = range(1, 13)
        xleft = [i - (width / 2 + 0.01) for i in x]
        xright = [i + (width / 2 + 0.01) for i in x]
        bottom_count = 0
        bottom_error = 0
        for i, group in enumerate(error.columns):
            if i == 0:
                ax.bar(
                    xleft,
                    counts[group],
                    label=group_names[group],
                    width=width,
                    color=style_colors[i],
                )
                ax.bar(
                    xright, error[group], width=width, color=style_colors[i], hatch="//"
                )
                bottom_count = counts[group]
                bottom_error = error[group]
            else:
                ax.bar(
                    xleft,
                    counts[group],
                    bottom=bottom_count,
                    label=group_names[group],
                    width=width,
                    color=style_colors[i],
                )
                ax.bar(
                    xright,
                    error[group],
                    bottom=bottom_error,
                    width=width,
                    color=style_colors[i],
                    hatch="//",
                )
                bottom_count += counts[group]
                bottom_error += error[group]
        ax.set_xticks(x)
        ax.set_xticklabels(calendar.month_abbr[1:])
        ax.set_ylim(ax.get_ylim()[0], 115)
        ax.set_ylabel("Group Percent")
        ax.set_xlabel("Month")
        ax.set_title(f"{print_basin} - {print_res}")
        ax.legend(loc="upper left", title=None, prop={"size": 16})

        print(f"Saving figure for {print_res}")
        plt.subplots_adjust(
            top=0.942, bottom=0.141, left=0.066, right=0.985, hspace=0.2, wspace=0.2
        )
        plt.savefig(f"../figures/monthly_tree_breakdown_terror/{rbasins[res]}_{res}.png")
        # plt.show()
        plt.close()
        # cont = input("Continue? [Y/n] ")
        # if cont.lower() == "n":
        #     break


def get_operating_groups():
    files = glob.glob(
        "../results/tclr_model_testing/all/TD4_*_MSS0.??_RT_MS_exhaustive_new_hoover/results.pickle"
    )

    td_mss_assim_pat = re.compile(r"TD(\d)_(.*)_MSS(\d\.\d\d)")
    matches = [re.search(td_mss_assim_pat, i) for i in files]
    td_mss_assim = [i.groups() for i in matches]
    td_mss_assim = [i for i in td_mss_assim if "yearly" not in i]

    results = {}
    for key, file in zip(td_mss_assim, files):
        with open(file, "rb") as f:
            results[key] = pickle.load(f)
    with open(
        "../results/tclr_model_testing/all/TD4_MSS0.10_RT_MS_exhaustive_new_hoover/results.pickle",
        "rb",
    ) as f:
        results[("4", "never", "0.10")] = pickle.load(f)

    groups = select_results(results, "groups")
    columns = ["Res", "Date", "TD", "Assim", "MSS", "Group"]
    records = []
    for key, value in groups.items():
        td, assim, mss = key
        for (res, dt), group in value.items():
            records.append([res, dt, td, assim, mss, group])
    df = pd.DataFrame.from_records(records, columns=columns)
    value_counts = df.groupby(["Res", df["Date"].dt.month, "Assim"])[
        "Group"
    ].value_counts()  # [no-member]
    value_counts.name = "Count"
    value_counts = value_counts.reset_index()
    value_counts = value_counts[value_counts["Assim"] == "never"]
    value_counts = value_counts.drop("Assim", axis=1)

    op_codes = [[1], [2], [3, 5, 7], [4, 5, 7], [4, 6, 7]]
    op_names = {
        0: "Small ROR",
        1: "Large ROR",
        2: "Small St. Dam",
        3: "Medium St. Dam",
        4: "Large St. Dam",
    }
    op_mods = value_counts.groupby("Res")["Group"].unique()
    op_mods = op_mods.apply(np.sort).apply(list)

    # there are three reservoirs that only end up in two groups
    # but could theoretically end up in three
    # AGR, SILVER JACK, and WCR all are [5, 7] but have
    # max storages that less than 630.9, which means they could
    # be in group 3 as well.
    # * I am replacing them here but this only works for this
    # specific instance

    op_mods = op_mods.apply(lambda x: "".join(map(str, x)))
    op_mods = op_mods.replace({"57": "357"})
    op_mods = op_mods.apply(lambda x: list(map(int, list(x))))

    op_mod_ids = pd.DataFrame(op_mods.apply(op_codes.index))
    op_mod_ids["Group Name"] = op_mod_ids["Group"].apply(op_names.get)

    rbasins = pd.read_pickle("../pickles/res_basin_map.pickle")
    rename = {
        "upper_col": "colorado",
        "lower_col": "colorado",
        "pnw": "columbia",
        "tva": "tennessee",
    }
    rbasins = rbasins.replace(rename)
    rbasins = rbasins.str.capitalize()
    op_mod_ids["basin"] = rbasins
    return op_mod_ids


def plot_operation_group_map():
    op_mod_ids = get_operating_groups()
    res_locs = pd.read_csv("../geo_data/reservoirs.csv")
    res_locs = res_locs.set_index("site_name")
    op_mod_ids["x"] = res_locs["long"]
    op_mod_ids["y"] = res_locs["lat"]

    with open("../geo_data/extents.json", "r") as f:
        coords = json.load(f)
    basin_color_map = sns.color_palette("Set2")
    basin_info = {
        "Columbia": {
            "extents": coords["Columbia"],
            "shp": (GIS_DIR / "columbia_shp" / "Shape" / "WBDHU2").as_posix(),
            "color": basin_color_map[0],
            "rivers": [
                (
                    GIS_DIR
                    / "NHDPlus"
                    / "trimmed_flowlines"
                    / "NHDPlusPN_trimmed_flowlines_noz"
                ).as_posix()
            ],
        },
        "Missouri": {
            "extents": coords["Missouri"],
            "shp": (GIS_DIR / "missouri_shp" / "Shape" / "WBDHU2").as_posix(),
            "color": basin_color_map[1],
            "rivers": [
                (
                    GIS_DIR
                    / "NHDPlus"
                    / "trimmed_flowlines"
                    / "NHDPlusMU_trimmed_flowlines_noz"
                ).as_posix(),
                (
                    GIS_DIR
                    / "NHDPlus"
                    / "trimmed_flowlines"
                    / "NHDPlusML_trimmed_flowlines_noz"
                ).as_posix(),
            ],
        },
        "Colorado": {
            "extents": coords["Colorado"],
            "shp": (GIS_DIR / "colorado_shp" / "Shape" / "WBDHU2").as_posix(),
            "color": basin_color_map[2],
            "rivers": [
                (
                    GIS_DIR
                    / "NHDPlus"
                    / "trimmed_flowlines"
                    / "NHDPlusUC_trimmed_flowlines_noz"
                ).as_posix(),
                (
                    GIS_DIR
                    / "NHDPlus"
                    / "trimmed_flowlines"
                    / "NHDPlusLC_trimmed_flowlines_noz"
                ).as_posix(),
            ],
        },
        "Tennessee": {
            "extents": coords["Tennessee"],
            "shp": (GIS_DIR / "tennessee_shp" / "Shape" / "WBDHU2").as_posix(),
            "color": basin_color_map[3],
            "rivers": [
                (
                    GIS_DIR
                    / "NHDPlus"
                    / "trimmed_flowlines"
                    / "NHDPlusTN_trimmed_flowlines_noz"
                ).as_posix()
            ],
        },
    }
    fig, axes = char_split_map_single_var(
        "Group",
        op_mod_ids,
        basin_info,
        add_legend=True,
        legend_labels=list(op_names.values()),
    )
    plt.show()


def plot_error_by_variable():
    op_mod_ids = get_operating_groups()
    result_file = (
        "../results/tclr_model_testing/all/"
        + "TD4_MSS0.10_RT_MS_exhaustive_new_hoover/results.pickle"
    )
    data_file = (
        "../results/tclr_model_testing/all/"
        + "TD4_MSS0.10_RT_MS_exhaustive_new_hoover/model_data.pickle"
    )
    with open(result_file, "rb") as f:
        results = pickle.load(f)
    with open(data_file, "rb") as f:
        data = pickle.load(f)

    results = results["train_data"]
    results["op_group"] = [
        op_mod_ids.loc[i, "Group Name"]
        for i in results.index.get_level_values("site_name")
    ]

    train_data = data["xtrain"]
    means = data["means"]
    std = data["std"]

    train_data_act = train_data.copy()
    for col in train_data_act.columns:
        if col in means.columns:
            train_data_act[col] = (
                (train_data_act[col].unstack().T * std[col]) + means[col]
            ).T.stack()

    results["actual_std"] = (
        (results["actual"].unstack().T - means["release"]) / std["release"]
    ).T.stack()
    results["model_std"] = (
        (results["model"].unstack().T - means["release"]) / std["release"]
    ).T.stack()

    # results["error"] = (results["model"] - results["actual"]).abs()
    # results["error_std"] = (results["model_std"] - results["actual_std"]).abs()
    results["error"] = results["model"] - results["actual"]
    results["error_std"] = results["model_std"] - results["actual_std"]

    vars = train_data.columns.drop(["const", "rts", "max_sto"])

    # fig, axes = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(19, 10))
    # axes = axes.flatten()

    # for ax, var in zip(axes, vars):
    var_map = {
        "storage_pre": "Current Storage",
        "release_pre": "Previous Release",
        "inflow": "Net Inflow",
        "sto_diff": "Weekly Storage Difference",
        "release_roll7": "Weekly Mean Release",
        "inflow_roll7": "Weekly Mean Inflow",
        "storage_x_inflow": "Storage-Inflow Interaction",
    }
    for var in vars:
        x = train_data_act[var]
        y = results["error"]
        color = results["op_group"]
        df = pd.DataFrame({var: x, "Error": y, "color": color})
        # sns.scatterplot(data=df, x=var, y="Error", hue="color", ax=ax)
        ax = sns.scatterplot(
            data=df,
            x=var,
            y="Error",
            hue="color",
            hue_order=[
                "Small ROR",
                "Large ROR",
                "Small St. Dam",
                "Medium St. Dam",
                "Large St. Dam",
            ],
        )
        ax.legend(loc="upper right")
        ax.set_xlabel(var_map[var])
        ax.set_ylabel(r"$|y_{mod} - y_{obs}|$")

        plt.show()


def get_unique_paths():
    df = pd.read_pickle(
        "../results/tclr_model_testing/all/TD4_MSS0.10_RT_MS_exhaustive_new_hoover/train_paths.pickle"
    )
    df = df.apply(lambda x: "->".join(str(i) for i in x))
    uniq_paths = list(df.unique())
    uniq_paths.sort(key=len)

    path_ids = df.apply(uniq_paths.index)
    path_ids = path_ids.reset_index()

    path_counts = path_ids.groupby(
        [path_ids["site_name"], path_ids["datetime"].dt.month]
    )["path"].value_counts()
    path_counts.name = "count"
    path_counts = path_counts.reset_index()

    single_group_res = []
    resers = path_counts["site_name"].unique()
    for res in resers:
        rgroups = path_counts[path_counts["site_name"] == res]["count"]
        rgroups = rgroups.unique()
        if len(rgroups) == 1:
            single_group_res.append((res, rgroups[0]))
    drop_res = [i[0] for i in single_group_res]

    path_counts = path_counts[~path_counts["site_name"].isin(drop_res)]

    plot_res = path_counts["site_name"].unique()
    name_replacements = get_name_replacements()
    rbasins = pd.read_pickle("../pickles/res_basin_map.pickle")

    figManager = plt.get_current_fig_manager()
    for res in plot_res:
        basin = rbasins[res]
        print_basin = " ".join(basin.split("_")).title()
        print_res = name_replacements.get(res, res).title()

        rdf = path_counts[path_counts["site_name"] == res]
        rdf = rdf.pivot(index=["datetime"], columns=["path"], values=["count"])
        rdf = rdf.divide(rdf.sum(axis=1).values, axis=0) * 100
        rdf.columns = rdf.columns.droplevel(0)

        fig, ax = plt.subplots(1, 1, figsize=(19, 10))

        rdf.plot.bar(stacked=True, ax=ax)
        ax.set_ylabel("Path Percent")
        ax.set_xlabel("Month")
        ax.set_title(f"{print_basin} - {print_res}")
        ax.legend(loc="upper left", title=None)

        print(f"Saving figure for {print_res}")
        figManager.window.showMaximized()
        plt.tight_layout()
        plt.savefig(f"../figures/monthly_tree_breakdown_paths/{rbasins[res]}_{res}.png")
        plt.close()

        # plt.show()


def plot_interannual_group_variability():
    with open(
        "../results/tclr_model_testing/all/TD4_MSS0.10_RT_MS_exhaustive_new_hoover/results.pickle",
        "rb",
    ) as f:
        results = pickle.load(f)

    groups = results["groups"]
    groups = groups.reset_index().rename(columns={0: "group"})

    # remove reservoirs that do not have any group variability
    gvar = groups.groupby("site_name")["group"].var()
    var_res = gvar[gvar > 0].index

    groups = groups.loc[groups["site_name"].isin(var_res), :]

    groups["year"] = groups["datetime"].dt.year
    groups["month"] = groups["datetime"].dt.month

    # name groups 3, 4, 5, 6, 7
    group_names = {
        7: "High Inflow Storage Maintenance (7)",
        3: "Low Steady Release (3)",
        4: "Storage Build-Up (4)",
        5: "Storage and Release Maintenance (5)",
        6: "Release Trend Persistence (6)",
    }

    counts = groups.groupby(["site_name", "year", "month"])["group"].value_counts()
    # counts = groups.groupby(["site_name", "year"])["group"].value_counts()
    counts.name = "count"

    # plot_res = counts.index.get_level_values("site_name").unique()
    rbasins = pd.read_pickle("../pickles/res_basin_map.pickle")
    renames = get_name_replacements()
    basin_name_map = {
        "lower_col": "Lower Colorado",
        "upper_col": "Upper Colorado",
        "missouri": "Missouri",
        "pnw": "Columbia",
        "tva": "Tennessee",
    }

    idx = pd.IndexSlice
    for res in var_res:
        df = counts.loc[idx[res, :, :, :]]
        basin = rbasins[res]
        print_basin = basin_name_map[basin]
        pname = renames.get(res, res).title()

        df = (
            df.reset_index()
            .sort_values(by=["year", "month", "group"])
            .set_index(["year", "month", "group"])["count"]
        )
        df = df.unstack()
        # get the percentage of each group in each year
        df = df.divide(df.sum(axis=1), axis=0) * 100
        df = df.fillna(0.0)
        df.columns = df.columns.values
        df = df.reset_index().melt(id_vars=["year", "month"], var_name="group")
        sns.barplot(data=df, x="month", y="value", hue="group", estimator=np.mean, errorbar=("ci", 90), capsize=0.15, errwidth=2)

        # quants = df.groupby("month").quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        # quants.columns = quants.columns.values
        # quants = quants.reset_index().rename(columns={"level_1": "quantile"})
        # quants = quants.melt(id_vars=["month", "quantile"], value_vars=[5, 7])

        II()
        sys.exit()

        # drop the first and last year as they are incomplete
        df = df.drop(df.index.min())
        df = df.drop(df.index.max())
        df = df.rename(columns=group_names)
        fig = plt.figure(figsize=(19, 10))
        ax = fig.add_subplot()
        df.plot.bar(stacked=True, ax=ax)
        ax.set_title(f"{pname} - {print_basin}")
        ax.legend(loc="upper left", title="", prop={"size": 14})
        ax.set_ylim(ax.get_ylim()[0], 115)
        ax.set_ylabel("Group Percentage")
        ax.set_xlabel("Year")
        plt.subplots_adjust(
            top=0.942, bottom=0.141, left=0.066, right=0.985, hspace=0.2, wspace=0.2
        )
        plt.savefig(
            f"../figures/interannual_group_variability_new/{basin}_{res}.png", dpi=400
        )

        plt.close()


def plot_interannual_seasonal_group_variability():
    sns.set_context("paper")
    with open(
        "../results/tclr_model_testing/all/TD4_MSS0.10_RT_MS_exhaustive_new_hoover/results.pickle",
        "rb",
    ) as f:
        results = pickle.load(f)

    groups = results["groups"]
    groups = groups.reset_index().rename(columns={0: "group"})

    # remove reservoirs that do not have any group variability
    gvar = groups.groupby("site_name")["group"].var()
    var_res = gvar[gvar > 0].index

    groups = groups[groups["site_name"].isin(var_res)]

    groups["year"] = groups["datetime"].dt.year
    groups["month"] = groups["datetime"].dt.month

    # name groups 3, 4, 5, 6, 7
    group_names = {
        7: "High Inflow Storage Maintenance (7)",
        3: "Low Steady Release (3)",
        4: "Storage Build-Up (4)",
        5: "Storage and Release Maintenance (5)",
        6: "Release Trend Persistence (6)",
    }

    counts = groups.groupby(["site_name", "year", "month"])["group"].value_counts()
    # counts = groups.groupby(["site_name", "year"])["group"].value_counts()
    counts.name = "count"

    plot_res = counts.index.get_level_values("site_name").unique()

    idx = pd.IndexSlice
    # for res in plot_res:
    res = "Hoover"

    df = counts.loc[idx[res, :, :, :]]
    index_names = df.index.names
    df = df.reset_index().sort_values(by=index_names).set_index(index_names)["count"]
    df = df.unstack()

    # get the percentage of each group in each year
    df = df.divide(df.sum(axis=1), axis=0) * 100

    # drop the first and last year as they are incomplete
    min_year = df.index.get_level_values("year").min()
    max_year = df.index.get_level_values("year").max()
    df = df[~df.index.get_level_values("year").isin([min_year, max_year])]
    df = df.rename(columns=group_names)
    II()
    sys.exit()
    # years = df.index.get_level_values("year").unique().sort_values()
    # # years = years[:3]
    # nyears = len(years)
    # fig, axes = plt.subplots(1, nyears, sharey=True)

    # xmin = 0.05
    # xmax = 0.95
    # ymin = 0.05
    # ymax = 0.95
    # xwidth = (xmax - xmin) / nyears
    # for i, (year, ax) in enumerate(zip(years, axes)):
    #     xstart = xwidth * i + xmin
    #     xstop = xwidth * (i+1) + xmin
    #     bbox = Bbox.from_extents(xstart, ymin, xstop, ymax)
    #     ax.set_position(bbox)
    #     pdf = df.loc[idx[year, :]]
    #     # pdf.plot.barh(stacked=True, ax=ax)
    #     pdf.T.plot(ax=ax)
    #     ax.set_xticks([0])
    #     ax.set_xticklabels([year], rotation=90)
    #     handles, labels = ax.get_legend_handles_labels()
    #     ax.get_legend().remove()

    # plt.show()


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) > 0:
        metric = args[0]
    else:
        metric = "NSE"
    plt.style.use("tableau-colorblind10")
    # plt.style.use(["science", "nature"])
    sns.set_context("talk", font_scale=1.1)
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
    # plot_grid_search_results(ds="simul", metric=metric)
    # * FIGURE 4
    # * This is the optimal tree for TD4 MSS0.10
    # * FIGURE 5 - NOT INCLUDING AT THE MOMENT
    # plot_best_and_worst_reservoirs(metric)
    # * FIGURE 5
    # plot_res_characteristic_bin_performance(metric, 5)
    # plot_res_characteristic_scatter_performance(metric)
    # * FIGURE 6
    # * this is the attribute maps
    # * FIGURE 7
    # plot_data_assim_results(metric)
    # plot_data_assim_scatter()

    # plot_top_characteristic_res(metric, 10)
    # plot_top_characteristic_res_scatter(metric)
    # plot_characteristic_res_line_plot(metric)
    # plot_res_characteristic_map(metric)
    # plot_res_characteristic_split_map()
    # plot_data_assim_scatter(metric)
    # plot_reservoir_metric(metric, "test")
    # plot_assim_metric_scatter()
    # plot_basin_performance(metric)
    # plot_assim_lines()

    # get_unique_trees()
    # get_unique_paths()
    # get_operating_groups()
    plot_interannual_group_variability()
    # plot_interannual_seasonal_group_variability()

    # plot_error_by_variable()
