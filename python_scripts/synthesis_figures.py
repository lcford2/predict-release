import calendar
import glob
import json
import os
import pathlib
import pickle
import re
import socket
import sys

import geopandas as gpd
import inflect
import matplotlib.gridspec as GS
import matplotlib.lines as mlines
import matplotlib.patches as mpatch
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
import seaborn as sns
from IPython import embed as II
from joblib import Parallel, delayed
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.colors import ListedColormap, LogNorm, Normalize
from matplotlib.transforms import Bbox
from scipy.signal import cwt, ricker
from scipy.stats import zscore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tclr_model import read_basin_data

hostname = socket.gethostname()
if hostname == "CCEE-DT-094":
    os.environ["PROJ_LIB"] = (
        "C:\\Users\\lcford2\\AppData\\Local\\Continuum\\"
        "anaconda3\\envs\\sry-env\\Library\\share"
    )
    GIS_DIR = pathlib.Path("G:/My Drive/PHD/GIS")
    HOME = "C:\\Users\\lcford2"
elif hostname == "inspiron-laptop":
    os.environ[
        "PROJ_LIB"
    ] = r"C:\\Users\\lcford\\miniconda3\\envs\\sry-env\\Library\\share"
    GIS_DIR = pathlib.Path("/home/lford/data/GIS")
    HOME = "~"


from mpl_toolkits.basemap import Basemap

CHAR_VARS = [
    "Release Seasonality",
    "Storage Seasonality",
    "Maximum Storage",
    "Mean Release",
    r"Release $CV$",
    "Residence Time",
]

OP_NAMES = {
    0: "Small ROR",
    1: "Large ROR",
    2: "Small St. Dam",
    3: "Medium St. Dam",
    4: "Large St. Dam",
}

OP_MODES = {
    3: "Low Steady Release",
    4: "Storage Build Up",
    5: "St. & Rel. Maint.",
    6: "Rel. Trend Pers.",
    7: "High Inf. St. Maint.",
}

RESULT_FILE = (
    "../results/tclr_model_testing/all/"
    + "TD4_MSS0.10_RT_MS_exhaustive_new_hoover/results.pickle"
)

DATA_FILE = (
    "../results/tclr_model_testing/all/"
    + "TD4_MSS0.10_RT_MS_exhaustive_new_hoover/model_data.pickle"
)

INFLECT_ENGINE = inflect.engine()


def load_pickle(file):
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data


def write_pickle(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def select_results(results, get_item):
    return {i: j[get_item] for i, j in results.items()}


def total_absolute_error(yact, ymod):
    return (ymod - yact).abs().sum()


def bias(yact, ymod):
    return np.mean(ymod) - np.mean(yact)


def pbias(yact, ymod):
    return (np.mean(ymod) - np.mean(yact)) / np.mean(yact)


def get_r2score(df, grouper=None):
    if grouper:
        return pd.DataFrame(
            {
                "NSE": df.groupby(grouper).apply(
                    lambda x: r2_score(x["actual"], x["model"])
                )
            }
        )

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

    return (
        mean_squared_error(df["actual"], df["model"], squared=False)
        / getattr(df["actual"], normer)()
    )


def get_pbias(df, grouper=None):
    if grouper:
        return pd.DataFrame(
            {"PBIAS": df.groupby(grouper).apply(lambda x: pbias(x["actual"], x["model"]))}
        )

    return pbias(["actual"], df["model"])


def get_model_scores(model_dfs, metric="NSE", grouper=None):
    if metric == "NSE":
        return {i: get_r2score(j, grouper) for i, j in model_dfs.items()}
    elif metric == "RMSE":
        return {i: get_rmse(j, grouper) for i, j in model_dfs.items()}
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

    m.drawmapboundary(fill_color="white")
    m.readshapefile(states_path.as_posix(), "states")

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
    ax.scatter(x, y, marker="v", edgecolor="k", s=150, zorder=4)

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
    fig_file = (
        "C:\\Users\\lcford2\\Dropbox\\PHD\\multibasin_model_figures"
        + f"\\new_paper_figures\\split_char_map_{var}_larger.png"
    )
    if save:
        plt.savefig(fig_file, dpi=400)
    return fig, axes


def get_unique_trees():
    results = load_pickle(RESULT_FILE)

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
    rbasins = pd.load_pickle("../pickles/res_basin_map.pickle")

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
        "../results/tclr_model_testing/all/TD4_*_MSS0.??_RT_MS_exhaustive_new_hoover/"
        + "results.pickle"
    )

    td_mss_assim_pat = re.compile(r"TD(\d)_(.*)_MSS(\d\.\d\d)")
    matches = [re.search(td_mss_assim_pat, i) for i in files]
    td_mss_assim = [i.groups() for i in matches]
    td_mss_assim = [i for i in td_mss_assim if "yearly" not in i]

    results = {}
    for key, file in zip(td_mss_assim, files):
        with open(file, "rb") as f:
            results[key] = pickle.load(f)
    results[("4", "never", "0.10")] = load_pickle(RESULT_FILE)

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
    op_mod_ids["Group Name"] = op_mod_ids["Group"].apply(OP_NAMES.get)

    rbasins = pd.load_pickle("../pickles/res_basin_map.pickle")
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
        legend_labels=list(OP_NAMES.values()),
    )
    plt.show()


def plot_error_by_variable():
    op_mod_ids = get_operating_groups()
    results = load_pickle(RESULT_FILE)
    data = load_pickle(DATA_FILE)

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
    df = pd.load_pickle(
        "../results/tclr_model_testing/all/TD4_MSS0.10_RT_MS_exhaustive_new_hoover/"
        + "train_paths.pickle"
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
    rbasins = pd.load_pickle("../pickles/res_basin_map.pickle")

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


def plot_intraannual_modes():
    sns.set_context("notebook")
    results = load_pickle(RESULT_FILE)
    df = results["groups"]
    df.name = "group"

    counts = df.groupby(["site_name", df.index.get_level_values(1).month]).value_counts()
    counts.name = "count"
    counts = counts.to_frame().reset_index()

    single_group_res = []
    resers = counts["site_name"].unique()
    for res in resers:
        rgroups = counts[counts["site_name"] == res]["count"]
        rgroups = rgroups.unique()
        if len(rgroups) == 1:
            single_group_res.append((res, rgroups[0]))
    drop_res = [i[0] for i in single_group_res]

    counts = counts[~counts["site_name"].isin(drop_res)]

    # counts["group"] = counts["group"].apply(OP_MODES.get)

    plot_res = counts["site_name"].unique()
    name_replacements = get_name_replacements()
    rbasins = load_pickle("../pickles/res_basin_map.pickle")

    # load inflow data
    res_ts = read_basin_data("all")
    inflow = res_ts["inflow"]
    monthly_flow_records = []
    for res in inflow.index.get_level_values("site_name").unique():
        # daily flow records summed to get monthly flow records
        # units were per day, so now they are per month
        rminf = inflow.loc[pd.IndexSlice[res, :]].resample("MS").sum()
        rminf.index = pd.MultiIndex.from_product(
            [[res], rminf.index], names=["site_name", "datetime"]
        )
        monthly_flow_records.append(rminf)

    inflow = pd.concat(monthly_flow_records)
    inflow = inflow.groupby(
        ["site_name", inflow.index.get_level_values("datetime").month]
    ).mean()

    for res in plot_res:
        basin = rbasins[res]
        print_basin = " ".join(basin.split("_")).title()
        print_res = name_replacements.get(res, res).title()

        rdf = counts[counts["site_name"] == res]
        rdf = rdf.pivot(index=["datetime"], columns=["group"], values=["count"])
        rdf = rdf.divide(rdf.sum(axis=1).values, axis=0) * 100
        rdf.columns = rdf.columns.droplevel(0)

        rinf = inflow.loc[pd.IndexSlice[res, :]]

        fig = plt.figure(figsize=(19, 10))

        gs = GS.GridSpec(2, 1, height_ratios=[1, 4])

        inf_ax = fig.add_subplot(gs[0, 0])
        ax = fig.add_subplot(gs[1, 0], sharex=inf_ax)

        rinf.plot.bar(ax=inf_ax)
        inf_ax.set_title(f"{print_basin} - {print_res}")
        inf_ax.set_ylabel("Avg. Net Inflow\n[1000 acre-ft / month]")

        rdf.plot.bar(stacked=True, ax=ax)
        ax.set_ylabel("Operational Mode Likelihood")
        ax.set_xlabel("Month")
        handles, labels = ax.get_legend_handles_labels()
        labels = [OP_MODES.get(int(i)) for i in labels]
        ax.legend(handles, labels, loc="upper left", title=None, ncol=len(handles))
        ax.set_ylim(0, 110)
        ax.set_xticklabels(calendar.month_abbr[1:], rotation=0)

        print(f"Saving figure for {print_res}")
        fig.align_ylabels()
        plt.tight_layout()
        plt.savefig(f"../figures/monthly_tree_breakdown_winflow/{rbasins[res]}_{res}.png")
        plt.close()

        # plt.show()
        # sys.exit()


def plot_interannual_group_variability():
    results = load_pickle(RESULT_FILE)

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
    rbasins = pd.load_pickle("../pickles/res_basin_map.pickle")
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

        # quants = df.groupby("month").quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        # quants.columns = quants.columns.values
        # quants = quants.reset_index().rename(columns={"level_1": "quantile"})
        # quants = quants.melt(id_vars=["month", "quantile"], value_vars=[5, 7])

        # drop the first and last year as they are incomplete
        fig = plt.figure(figsize=(19, 10))
        ax = fig.add_subplot()
        sns.barplot(
            data=df,
            x="month",
            y="value",
            hue="group",
            estimator=np.mean,
            errorbar=("ci", 95),
            capsize=0.15,
            errwidth=2,
            ax=ax,
        )
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


def plot_transition_diagnostics():
    results = load_pickle(RESULT_FILE)
    data = load_pickle(DATA_FILE)
    ydata = results["train_data"]
    ydata["groups"] = results["groups"]

    ydata["group_shift"] = ydata.groupby("site_name")["groups"].shift(-1)
    xdata = data["xtrain"]
    xdata = xdata.drop(["const", "rts", "max_sto"], axis=1)
    for col in xdata.columns:
        ydata[col] = xdata[col]

    transitions = ydata.loc[ydata["groups"] != ydata["group_shift"], :].dropna()
    transitions["t_type"] = transitions.apply(
        lambda x: (int(x["groups"]), int(x["group_shift"])), axis=1
    )

    # counts = transitions.groupby([
    #     transitions.index.get_level_values(0),
    #     transitions.index.get_level_values(1).month
    # ])["t_type"].value_counts()

    # counts = transitions.groupby(
    #     transitions.index.get_level_values(0),
    # )["t_type"].value_counts()

    res = "Hoover"
    df = transitions.loc[pd.IndexSlice[res, :], ["t_type"]]
    df["from"] = df["t_type"].apply(lambda x: x[0])
    df["to"] = df["t_type"].apply(lambda x: x[1])
    df["count"] = 1
    counts = df.groupby(["from", "to"])["count"].sum()
    counts = counts.fillna(0.0).unstack()
    # rows are FROM, columns are TO
    percents = counts.divide(counts.sum(axis=1), axis=0) * 100
    percents.columns = [OP_MODES.get(i) for i in percents.columns]
    percents.index = [OP_MODES.get(i) for i in percents.index]
    ax = sns.heatmap(percents.T, annot=True, fmt=".0f")
    ax.tick_params(axis="x", labelrotation=0)

    plt.show()


def plot_shift_probabilities():
    results = load_pickle(RESULT_FILE)
    data = load_pickle(DATA_FILE)
    ydata = results["train_data"]
    ydata = ydata.sort_index()
    ydata["groups"] = results["groups"]

    ydata["group_shift"] = ydata.groupby("site_name")["groups"].shift(-1)
    xdata = data["xtrain"]
    xdata = xdata.drop(["const", "rts", "max_sto"], axis=1)
    for col in xdata.columns:
        ydata[col] = xdata[col]

    next_count = ydata.groupby(["site_name", "groups"])["group_shift"].value_counts()

    rbasins = pd.read_pickle("../pickles/res_basin_map.pickle")
    renames = get_name_replacements()
    basin_name_map = {
        "lower_col": "Lower Colorado",
        "upper_col": "Upper Colorado",
        "missouri": "Missouri",
        "pnw": "Columbia",
        "tva": "Tennessee",
    }
    resers = next_count.index.get_level_values("site_name").unique()
    for res in resers:
        basin = rbasins[res]
        print_basin = basin_name_map[basin]
        pname = renames.get(res, res).title()

        df = next_count.loc[pd.IndexSlice[res, :, :]].unstack()

        percents = df.divide(df.sum(axis=1), axis=0) * 100
        percents.columns = [OP_MODES.get(i) for i in percents.columns]
        percents.index = [OP_MODES.get(i) for i in percents.index]
        annot = []
        for row in percents.values:
            annot.append([f"{i:.0f} %" for i in row])

        fig = plt.figure(figsize=(19, 10))
        ax = fig.add_subplot()
        sns.heatmap(percents, annot=annot, fmt="s", ax=ax)
        # ax = sns.heatmap(percents, annot=True, fmt=".0f")
        ax.tick_params(axis="x", labelrotation=0)
        ax.set_ylabel("Current Op. Mode")
        ax.set_xlabel("Next Op. Mode")
        ax.set_title(f"{pname} - {print_basin}")
        plt.subplots_adjust(
            top=0.94, bottom=0.105, left=0.163, right=0.985, hspace=0.2, wspace=0.2
        )
        plt.savefig(f"../figures/transition_probs/{basin}_{res}.png", dpi=400)
        plt.close()


def rename_transistion_columns(columns):
    out = []
    for col in columns:
        start = col[0]
        to = col[1]
        start_text = OP_MODES[int(start)]
        to_text = OP_MODES[int(to)]
        out.append(f"{start_text} -> \n{to_text}")
    return out


def plot_lagged_shift_probabilities():
    results = load_pickle(RESULT_FILE)
    data = load_pickle(DATA_FILE)
    ydata = results["train_data"]
    ydata = ydata.sort_index()
    ydata["groups"] = results["groups"]

    NLAGS = 30

    for i in range(1, NLAGS + 1):
        ydata[f"group_shift{i}"] = ydata.groupby("site_name")["groups"].shift(-i)

    xdata = data["xtrain"]
    xdata = xdata.drop(["const", "rts", "max_sto"], axis=1)
    for col in xdata.columns:
        ydata[col] = xdata[col]

    counts = pd.DataFrame()
    for i in range(1, NLAGS + 1):
        if counts.empty:
            temp = ydata.groupby(["site_name", "groups"])[
                f"group_shift{i}"
            ].value_counts()
            temp.name = "lag1_count"
            counts = temp.to_frame()
        else:
            counts[f"lag{i}_count"] = ydata.groupby(["site_name", "groups"])[
                f"group_shift{i}"
            ].value_counts()

    rbasins = pd.read_pickle("../pickles/res_basin_map.pickle")
    renames = get_name_replacements()
    basin_name_map = {
        "lower_col": "Lower Colorado",
        "upper_col": "Upper Colorado",
        "missouri": "Missouri",
        "pnw": "Columbia",
        "tva": "Tennessee",
    }
    resers = counts.index.get_level_values("site_name").unique()
    mean_variance = {}
    max_lag = {}

    for res in resers:
        basin = rbasins[res]
        print_basin = basin_name_map[basin]
        pname = renames.get(res, res).title()

        rcounts = counts.loc[pd.IndexSlice[res, :, :], :]

        df = rcounts.unstack().T
        totals = df.groupby(df.index.get_level_values(0)).sum()

        for lag in totals.index:
            df.loc[pd.IndexSlice[lag, :], :] /= totals.loc[lag]

        df *= 100
        df.columns = df.columns.droplevel(0)

        if df.columns.size > 1:
            res_var = {}
            res_maxidx = {}
            for column in df.columns:
                unstacked = df[column].unstack()
                # drop columns that only contain NA
                unstacked = unstacked.dropna(axis=1, how="all")
                res_var[column] = unstacked.var().mean()
                maxidx = unstacked.idxmax(skipna=True)
                maxidx = maxidx.apply(lambda x: int(re.findall(r"\d\d?", x)[0]))
                res_maxidx[column] = maxidx
            mean_variance[res] = res_var
            max_lag[res] = res_maxidx

        df.columns = [OP_MODES.get(int(i)) for i in df.columns]
        # percents.index = [OP_MODES.get(i) for i in percents.index]
        # annot = []
        # for row in percents.values:
        #     annot.append([f"{i:.0f} %" for i in row])

        if df.columns.size > 1:
            fig, axes = plt.subplots(df.columns.size, 1, figsize=(19, 10), sharex=True)
            axes = axes.flatten()

            for ax, col in zip(axes, df.columns):
                pdf = df[col].unstack()
                pdf.index = [int(re.findall(r"\d\d?", i)[0]) for i in pdf.index]
                # pdf.plot.bar(ax=ax, stacked=False)
                pdf.plot(ax=ax)
                ax.set_xlabel("Lag")
                ax.set_title(col)
                if ax == axes[0]:
                    handles, labels = ax.get_legend_handles_labels()
                    labels = [OP_MODES.get(int(float(i))) for i in labels]
                    ax.legend(handles, labels)
                else:
                    ax.get_legend().remove()
            plt.subplots_adjust(
                top=0.90, bottom=0.113, left=0.072, right=0.984, hspace=0.198, wspace=0.2
            )
            fig.text(0.02, 0.5, "Group Probability [%]", va="center", rotation=90)
            fig.suptitle(f"{pname} - {print_basin}")
            plt.savefig(
                f"../figures/lagged_transition_probs/lines/{basin}_{res}.png", dpi=400
            )
            plt.close()

    vardf = pd.DataFrame.from_dict(mean_variance)
    # where variance is 0 the groups are constant through time
    # so I want to drop them
    vardf = vardf.replace({0.0: np.nan}).dropna(axis=1, how="all")
    # now drop the rows of groups that dont vary
    vardf = vardf.dropna(axis=0, how="all").sort_index().T
    vardf["mean"] = vardf.mean(axis=1)
    vardf = vardf.sort_values(by="mean")
    vardf.to_pickle(
        "../results/synthesis_paper/transition_analysis/lagged_transition_probs_variance.pickle"
    )


@numba.jit(nopython=True)
def count_values(values, groups):
    # for this function, each value in values must exit in groups
    counts = [0 for i in groups]
    for value in values:
        idx = groups.index(value)
        counts[idx] += 1
    return counts


@numba.jit(nopython=True)
def sort_two_lists(a, b):
    a_sort = sorted(a)
    sorted_idx = [a.index(i) for i in a_sort]
    b_sort = [b[i] for i in sorted_idx]
    return a_sort, b_sort


@numba.jit(nopython=True)
def cross_reference_lists(master, key, values):
    # return values in the order that the keys
    # in key show up in master
    return [values[key.index(i)] for i in master]


@numba.jit(nopython=True)
def calc_rolling_props(values, groups, window=7):
    size = values.size
    output = [[0.0 for i in groups]] * (size - window)
    for i in range(0, size - window):
        cvals = values[i : i + window]
        counts = count_values(cvals, groups)
        props = [float(i) / window for i in counts]
        output[i] = props
    return output


def get_tick_years(index, ax):
    nticks = len(ax.get_xticks()) - 1
    start_year = index.min().year + 1
    stop_year = index.max().year
    nyears = stop_year - start_year
    tick_years = np.arange(start_year, stop_year, nyears // nticks)

    ticks = [
        np.where(index == pd.Timestamp(year=i, day=1, month=1))[0][0] for i in tick_years
    ]
    return tick_years, ticks


def plot_rolling_group_frequencies():
    results = load_pickle(RESULT_FILE)
    data = load_pickle(DATA_FILE)
    ydata = results["train_data"]
    ydata = ydata.sort_index()
    ydata["groups"] = results["groups"]

    xdata = data["xtrain"]
    xdata = xdata.drop(["const", "rts", "max_sto"], axis=1)
    for col in xdata.columns:
        ydata[col] = xdata[col]

    rbasins = pd.read_pickle("../pickles/res_basin_map.pickle")
    renames = get_name_replacements()
    basin_name_map = {
        "lower_col": "Lower Colorado",
        "upper_col": "Upper Colorado",
        "missouri": "Missouri",
        "pnw": "Columbia",
        "tva": "Tennessee",
    }
    resers = ydata.index.get_level_values("site_name").unique()

    WINDOW = 7
    for res in resers:
        try:
            basin = rbasins[res]
            print_basin = basin_name_map[basin]
            pname = renames.get(res, res).title()

            df = ydata.loc[pd.IndexSlice[res, :], ["groups"]]
            if df.groups.var() == 0:
                continue
            groups = numba.typed.List(df["groups"].unique())
            rolling_props = calc_rolling_props(df["groups"].values, groups, WINDOW)
            rolling_props = pd.DataFrame.from_records(
                rolling_props,
                columns=groups,
                index=df.index.get_level_values("datetime")[WINDOW:],
            )
            widths = np.arange(1, 366, 7)
            cwts = [cwt(rolling_props[i], ricker, widths) for i in groups]

            gs = GS.GridSpec(
                len(cwts), 2, width_ratios=[20, 1], height_ratios=[1 for i in cwts]
            )
            fig = plt.figure(figsize=(19, 10))
            leg_ax = fig.add_subplot(gs[:, 1])

            vmin = float("inf")
            vmax = -float("inf")
            for mat in cwts:
                cmin = abs(mat).min()
                cmax = abs(mat).max()
                if cmin < vmin:
                    vmin = cmin
                if cmax > vmax:
                    vmax = cmax
            cwt_axes = []
            for i, mat in enumerate(cwts):
                if i > 0:
                    ax = fig.add_subplot(gs[i, 0], sharex=cwt_axes[0])
                else:
                    ax = fig.add_subplot(gs[i, 0])
                cwt_axes.append(ax)
                plt.imshow(
                    mat,
                    extent=[0, df.shape[0], max(widths), min(widths)],
                    aspect="auto",
                    vmax=vmax,
                    vmin=vmin,
                )
                index = df.index.get_level_values("datetime")
                tick_years, ticks = get_tick_years(index, ax)
                tick_labels = [str(i) for i in tick_years]
                ax.set_xticks(ticks)
                if i == (len(cwts) - 1):
                    ax.set_xticklabels(tick_labels)
                else:
                    ax.set_xticklabels(["" for i in tick_labels])
                ax.set_title(OP_MODES[groups[i]])

            norm = Normalize(vmin=vmin, vmax=vmax)
            color_map = plt.get_cmap("viridis")
            cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=color_map), cax=leg_ax)
            cbar.outline.set_edgecolor("k")
            cbar.outline.set_linewidth(1)
            cbar.ax.tick_params(labelsize=12)
            cbar.set_label("Spectral Power", fontsize=14)
            fig.text(0.02, 0.5, "Days", rotation=90, va="center")
            plt.subplots_adjust(
                top=0.910, bottom=0.07, left=0.08, right=0.952, hspace=0.359, wspace=0.043
            )
            fig.suptitle(f"{pname} - {print_basin}")

            folder = f"../figures/group_wavelet/{INFLECT_ENGINE.number_to_words(WINDOW)}_day_window/"
            fullpath = os.path.abspath(folder)
            if not os.path.exists(fullpath):
                os.makedirs(fullpath)
            plt.savefig(os.path.join(folder, f"{basin}_{res}.png"))
            plt.close()
        except:
            continue


def plot_all_res_mode_probabilities():
    sns.set_context("paper")
    results = load_pickle(RESULT_FILE)

    df = results["groups"]
    df.name = "mode"

    counts = df.groupby(["site_name", df.index.get_level_values(1).month]).value_counts()

    counts.name = "count"
    counts = counts.to_frame().reset_index()

    cvar = counts.groupby("site_name")["mode"].var()
    single_res = cvar[cvar == 0].index

    counts = counts[~counts["site_name"].isin(single_res)]

    modes = counts["mode"].unique()

    counts = counts.set_index(["site_name", "datetime", "mode"])

    probs = counts.groupby(["site_name", "datetime"]).apply(
        lambda x: x["count"] / x["count"].sum() * 100
    )

    probs.index = probs.index.droplevel([0, 1])
    rbasins = load_pickle("../pickles/res_basin_map.pickle")
    rename = {
        "upper_col": "colorado",
        "lower_col": "colorado",
        "pnw": "columbia",
        "tva": "tennessee",
    }
    rbasins = rbasins.replace(rename)
    rbasins = rbasins.str.capitalize()

    idx = pd.IndexSlice
    name_replacements = get_name_replacements()

    for mode in modes:
        pdf = probs.loc[idx[:, :, mode]].unstack()
        pdf["basin"] = rbasins
        pdf = pdf.reset_index().sort_values(by=["basin", "site_name"])
        pdf = pdf.set_index("site_name").drop("basin", axis=1)
        pdf = pdf.fillna(0.0)

        pdf.columns = calendar.month_abbr[1:]
        pdf.index = [name_replacements.get(i, i).title() for i in pdf.index]

        gs = GS.GridSpec(1, 2, width_ratios=[20, 1])
        fig = plt.figure(figsize=(19, 10))
        ax = fig.add_subplot(gs[0, 0])
        cbar_ax = fig.add_subplot(gs[0, 1])

        sns.heatmap(pdf, vmin=0, vmax=100, annot=False, fmt=".0f", ax=ax, cbar_ax=cbar_ax)

        ax.set_title(OP_MODES.get(mode))
        ax.set_yticks(np.arange(pdf.shape[0]) + 0.5)
        ax.set_yticklabels(pdf.index)

        cbar_ax.set_ylabel(f"Likelihood of operating in {OP_MODES.get(mode)}")

        plt.subplots_adjust(
            top=0.967, bottom=0.038, left=0.1, right=0.5, hspace=0.2, wspace=0.017
        )
        plt.savefig(f"../figures/mode_res_monthly_prob_heatmaps/mode{mode}.png")
        plt.close()


def determine_similar_operating_months_across_reservoirs_corr():
    results = load_pickle(RESULT_FILE)
    groups = results["groups"]

    months = list(range(1, 13))

    var = groups.groupby("site_name").var()
    resers = var[var != 0].index

    similar_res = {r: {m: ("", 0.0) for m in months} for r in resers}

    for month in months:
        mdf = groups[groups.index.get_level_values("datetime").month == month]
        for res in resers:
            max_corr = 0.0
            best_res = ""
            for comp_res in resers:
                if comp_res == res:
                    continue
                rdf = mdf.loc[pd.IndexSlice[res, :]]
                cdf = mdf.loc[pd.IndexSlice[comp_res, :]]
                cmpdf = pd.DataFrame.from_dict({res: rdf, comp_res: cdf})
                cmpdf = cmpdf.dropna()
                corr = cmpdf.corr(method="spearman").loc[res, comp_res]
                if corr > max_corr:
                    best_res = comp_res
                    max_corr = corr

            similar_res[res][month] = (best_res, max_corr)
    records = []
    for res, scores in similar_res.items():
        for month, (cres, corr) in scores.items():
            records.append([res, month, cres, corr])

    df = pd.DataFrame.from_records(
        records, columns=["site_name", "month", "similar_res", "corr"]
    )

    df.to_pickle("../results/synthesis_paper/similar_res/sim_res.pickle")


def determine_transition_similarities(df, res1, res2):
    df[f"{res1}_shift"] = df[res1].shift(-1)
    df[f"{res2}_shift"] = df[res2].shift(-1)
    df[f"{res1}_trans"] = df[res1] != df[f"{res1}_shift"]
    df[f"{res2}_trans"] = df[res2] != df[f"{res2}_shift"]

    res1_tcounts = (
        df[df[f"{res1}_trans"]]
        .groupby(res1)[f"{res1}_shift"]
        .apply(lambda x: x.value_counts() / x.shape[0])
    )
    res2_tcounts = (
        df[df[f"{res2}_trans"]]
        .groupby(res2)[f"{res2}_shift"]
        .apply(lambda x: x.value_counts() / x.shape[0])
    )
    res1_tcounts.index = pd.MultiIndex.from_tuples(
        [(int(i), int(j)) for i, j in res1_tcounts.index]
    )
    res2_tcounts.index = pd.MultiIndex.from_tuples(
        [(int(i), int(j)) for i, j in res2_tcounts.index]
    )
    return 1 - (res1_tcounts - res2_tcounts).abs().mean()


def determine_similar_operating_reservoirs():
    results = load_pickle(RESULT_FILE)
    groups = results["groups"]

    var = groups.groupby("site_name").var()
    resers = var[var != 0].index

    similar_res = {r: {} for r in resers}

    for res in resers:
        osims = []
        tsims = []
        rdf = groups.loc[pd.IndexSlice[res, :]]
        rgroups = list(rdf.unique())
        if len(rgroups) == 1:
            continue
        for comp_res in resers:
            if comp_res == res:
                continue
            cdf = groups.loc[pd.IndexSlice[comp_res, :]]
            cgroups = list(cdf.unique())
            if sorted(rgroups) != sorted(cgroups):
                continue
            cmpdf = pd.DataFrame.from_dict({res: rdf, comp_res: cdf})
            cmpdf = cmpdf.dropna()
            if cmpdf.empty:
                continue
            cmpdf["same"] = cmpdf[res] == cmpdf[comp_res]
            overall_similarity = cmpdf["same"].sum() / cmpdf.shape[0]
            transition_similarity = determine_transition_similarities(
                cmpdf, res, comp_res
            )
            osims.append((comp_res, overall_similarity))
            tsims.append((comp_res, transition_similarity))

        osims.sort(key=lambda x: x[1], reverse=True)
        tsims.sort(key=lambda x: x[1], reverse=True)
        similar_res[res] = {"overall_sim": osims, "transition_sim": tsims}

    osim_records, tsim_records = [], []
    for res, sims in similar_res.items():
        for cres, osim in sims["overall_sim"]:
            osim_records.append([res, cres, osim])
        for cres, tsim in sims["transition_sim"]:
            tsim_records.append([res, cres, tsim])

    osim = pd.DataFrame.from_records(
        osim_records, columns=["site_name", "comp_site_name", "sim"]
    )
    tsim = pd.DataFrame.from_records(
        tsim_records, columns=["site_name", "comp_site_name", "sim"]
    )

    osim_max = osim.groupby("site_name")["sim"].idxmax()
    osim_max.name = "max_idx"
    osim_max = osim_max.to_frame().reset_index()
    tsim_max = tsim.groupby("site_name")["sim"].idxmax()
    tsim_max.name = "max_idx"
    tsim_max = tsim_max.to_frame().reset_index()

    osim_max["comp_site_name"] = osim.loc[osim_max["max_idx"], "comp_site_name"].values
    osim_max["sim"] = osim.loc[osim_max["max_idx"], "sim"].values

    tsim_max["comp_site_name"] = tsim.loc[tsim_max["max_idx"], "comp_site_name"].values
    tsim_max["sim"] = tsim.loc[tsim_max["max_idx"], "sim"].values

    rbasins = load_pickle("../pickles/res_basin_map.pickle")
    rename = {
        "upper_col": "colorado",
        "lower_col": "colorado",
        "pnw": "columbia",
        "tva": "tennessee",
    }
    rbasins = rbasins.replace(rename)
    rbasins = rbasins.str.capitalize()

    osim_max["basin"] = [rbasins[i] for i in osim_max["site_name"]]
    osim_max["comp_basin"] = [rbasins[i] for i in osim_max["comp_site_name"]]

    tsim_max["basin"] = [rbasins[i] for i in tsim_max["site_name"]]
    tsim_max["comp_basin"] = [rbasins[i] for i in tsim_max["comp_site_name"]]

    osim_max["same_basin"] = osim_max["basin"] == osim_max["comp_basin"]
    tsim_max["same_basin"] = tsim_max["basin"] == tsim_max["comp_basin"]
    II()


def make_multicolored_line_plot(df, x, y, c, colors, ax=None, **kwargs):
    if not ax:
        ax = plt.gca()

    df["shift"] = df[c].shift()
    df["mask"] = 1
    df.loc[df[c] == df["shift"], "mask"] = 0
    df["mask"] = df["mask"].cumsum()

    plot_segments = df["mask"].unique()

    groups = sorted(list(df[c].unique()))

    for ps in plot_segments:
        pdf = df[df["mask"] == ps]
        cgroup = pdf[c].values[0]
        if ps != plot_segments[0]:
            pre_group = df.loc[[pdf.index[0] - 1]]
            pdf = pd.concat([pre_group, pdf])
        color = colors[groups.index(cgroup)]
        ax.plot(pdf[x], pdf[y], c=color, **kwargs)


def plot_res_group_colored_timeseries():
    arg_res = sys.argv[1:]

    sns.set_context("notebook")
    results = load_pickle(RESULT_FILE)
    data = load_pickle(DATA_FILE)
    xtrain = data["xtrain"]
    means = data["means"]
    std = data["std"]

    ydata = results["train_data"]
    groups = results["groups"]
    vars = ["storage_pre", "inflow"]

    df = xtrain.loc[:, vars]

    df["storage_pre"] = (
        df["storage_pre"].unstack().T * std["storage_pre"] + means["storage_pre"]
    ).T.stack()
    df["inflow"] = (df["inflow"].unstack().T * std["inflow"] + means["inflow"]).T.stack()
    df["release"] = ydata["actual"]
    df["modeled_release"] = ydata["model"]
    df["groups"] = groups

    rbasins = load_pickle("../pickles/res_basin_map.pickle")
    rename = {
        "upper_col": "colorado",
        "lower_col": "colorado",
        "pnw": "columbia",
        "tva": "tennessee",
    }
    rbasins = rbasins.replace(rename)
    rbasins = rbasins.str.capitalize()
    name_replacements = get_name_replacements()

    resers = df.index.get_level_values("site_name").unique()

    style_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    parallel = True
    if arg_res:
        resers = arg_res
        parallel = False

    if parallel:
        Parallel(n_jobs=-1, verbose=11)(
            delayed(parallel_body_colored_group_plots)(
                df,
                res,
                name_replacements.get(res, res).title(),
                rbasins[res],
                " ".join(rbasins[res].split("_")).title(),
                style_colors,
            )
            for res in resers
        )
    else:
        for res in resers:
            parallel_body_colored_group_plots(
                df,
                res,
                name_replacements.get(res, res).title(),
                rbasins[res],
                " ".join(rbasins[res].split("_")).title(),
                style_colors,
                show=True,
                save=True,
            )


def parallel_body_colored_group_plots(
    df,
    res,
    print_res,
    basin,
    print_basin,
    style_colors,
    ptype="scatter",
    show=False,
    save=True,
):
    pdf = df.loc[pd.IndexSlice[res, :]]
    rgroups = sorted(list(pdf["groups"].unique()))
    if len(rgroups) == 1:
        return

    plot_resid = False
    if plot_resid:
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(19, 10))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(19, 10))
        axes = axes.flatten()

    pdf = pdf.reset_index()

    for var in ["storage_pre", "inflow"]:
        inf_scores = zscore(pdf[var])
        pdf.loc[inf_scores.abs() > 3, var] = np.nan
        pdf[var] = pdf[var].interpolate()

    pdf["residual"] = pdf["modeled_release"] - pdf["release"]

    if ptype == "scatter":
        colors = [style_colors[rgroups.index(i)] for i in pdf["groups"]]
        axes[0].scatter(pdf.index, pdf["storage_pre"], c=colors, s=10)
        axes[1].scatter(pdf.index, pdf["inflow"], c=colors, s=10)
        axes[2].scatter(pdf.index, pdf["release"], c=colors, s=10)
        if plot_resid:
            axes[3].scatter(
                pdf.index,
                pdf["modeled_release"] - pdf["release"],
                c=colors,
                s=10,
            )
    else:
        make_multicolored_line_plot(
            pdf, "datetime", "storage_pre", "groups", style_colors, ax=axes[0]
        )
        make_multicolored_line_plot(
            pdf, "datetime", "inflow", "groups", style_colors, ax=axes[1]
        )
        make_multicolored_line_plot(
            pdf, "datetime", "release", "groups", style_colors, ax=axes[2]
        )
        if plot_resid:
            make_multicolored_line_plot(
                pdf, "datetime", "residual", "groups", style_colors, ax=axes[3]
            )

    axes[0].set_ylabel("Storage [TAF]")
    axes[1].set_ylabel("Inflow [TAF/day]")
    axes[2].set_ylabel("Release [TAF/day]")
    bottom_ax = axes[2]
    if plot_resid:
        axes[3].set_ylabel("Residual [TAF/day]")
        bottom_ax = axes[3]

    index = pdf["datetime"]
    tick_years, ticks = get_tick_years(index, bottom_ax)
    tick_labels = [str(i) for i in tick_years]

    bottom_ax.set_xticks(ticks)
    bottom_ax.set_xticklabels(tick_labels)

    if ptype == "scatter":
        handles = [
            plt.scatter([], [], color=style_colors[i], alpha=1)
            for i in range(len(rgroups))
        ]
    else:
        handles = [
            mlines.Line2D([], [], color=style_colors[i], alpha=1, linewidth=1)
            for i in range(len(rgroups))
        ]
    # labels = [OP_MODES.get(i) for i in rgroups]
    labels = rgroups
    bottom_ax.legend(handles, labels, loc="upper right")
    fig.align_ylabels()
    fig.suptitle(f"{print_res} - {print_basin}")
    plt.subplots_adjust(
        top=0.945, bottom=0.045, left=0.045, right=0.991, hspace=0.126, wspace=0.2
    )

    if save:
        plt.savefig(
            f"{HOME}/Dropbox/PHD/multibasin_model_figures/new_paper_figures/"
            f"group_colored_timeseries/{basin}_{res}.png",
            dpi=450,
        )
    if show:
        plt.show()
    plt.close()


def enso_correlation():
    nino34 = pd.read_csv(
        "../csv/nino34.tsv", delim_whitespace=True, index_col=0, header=0
    )
    nino34 = nino34.stack()
    nino34 = nino34.reset_index().rename(columns={"level_1": "month", 0: "nino34"})
    nino34["day"] = 1
    nino34["datetime"] = pd.to_datetime(nino34[["year", "month", "day"]])
    nino34 = nino34.set_index("datetime").drop(["year", "month", "day"], axis=1)
    nino34 = nino34.replace({-99.99: np.nan}).dropna()

    renames = {
        "nino34": "Lag 0",
    }
    for lag in range(1, 7):
        nino34[f"nino34_lag{lag}"] = nino34["nino34"].shift(lag)
        renames[f"nino34_lag{lag}"] = f"Lag {lag}"

    nino34 = nino34.dropna()

    results = load_pickle(RESULT_FILE)
    groups = results["groups"]
    groups.name = "group"

    II()
    sys.exit()

    resers = groups.index.get_level_values(0).unique()

    rbasins = load_pickle("../pickles/res_basin_map.pickle")
    name_replacements = get_name_replacements()

    for res in resers:
        print_res = name_replacements.get(res, res).title()
        basin = rbasins[res]
        print_basin = " ".join(rbasins[res].split("_")).title()
        df = groups.loc[pd.IndexSlice[res, :]].to_frame()
        df["value"] = 1

        df = df.pivot(columns="group", values="value")
        if df.columns.size == 1:
            continue

        df = df.resample("MS").sum()

        orig_cols = df.columns
        df = pd.concat([df, nino34], axis=1)
        df = df.dropna()

        corr = df.corr(method="spearman")

        corr = corr[orig_cols].drop(orig_cols, axis=0)
        corr = corr.rename(index=renames, columns=OP_MODES)

        fig = plt.figure(figsize=(38, 21))
        axes = fig.add_subplot()

        sns.heatmap(
            corr.T, annot=True, ax=axes, cbar_kws={"label": "Spearman Correlation"}
        )
        axes.tick_params(axis="both", labelrotation=0)
        axes.set_xlabel("Nino3.4")
        axes.set_title(f"{print_res} - {print_basin}")

        plt.subplots_adjust(
            top=0.942, bottom=0.102, left=0.102, right=0.985, hspace=0.2, wspace=0.2
        )

        plt.savefig(
            f"../figures/enso_corr_heatmaps_jpg/{basin}_{res}.jpeg",
            dpi=400,
            format="jpeg",
            pil_kwargs={"quality": 85},
        )
        plt.close()
        # plt.show()
        # sys.exit()


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) > 0:
        metric = args[0]
    else:
        metric = "NSE"
    plt.style.use("tableau-colorblind10")
    sns.set_context("talk", font_scale=1.1)

    # get_unique_trees()
    # get_unique_paths()
    # get_operating_groups()
    # plot_interannual_group_variability()
    # plot_interannual_seasonal_group_variability()
    # plot_error_by_variable()
    # plot_transition_diagnostics()
    # plot_shift_probabilities()
    # plot_lagged_shift_probabilities()
    # plot_rolling_group_frequencies()
    # plot_intraannual_modes()
    # plot_all_res_mode_probabilities()
    # determine_similar_operating_months_across_reservoirs()
    # determine_similar_operating_reservoirs()
    # plot_res_group_colored_timeseries()
    enso_correlation()
