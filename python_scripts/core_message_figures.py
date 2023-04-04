import calendar
import datetime
import json
import os
import pathlib
import re
from itertools import product

import geopandas as gpd
import matplotlib.gridspec as GS
import matplotlib.lines as mlines
import matplotlib.patches as mpatch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython import embed as II
from matplotlib.transforms import Bbox
from utils.helper_functions import (
    get_julian_day_for_month_starts,
    get_n_median_index,
    idxquantile,
    read_pickle,
    write_pickle,
    load_rbasins,
)

GIS_DIR = pathlib.Path(os.path.expanduser("~/data/GIS"))

RESULT_FILE = (
    "../results/tclr_model_testing/all/"
    + "TD4_MSS0.10_RT_MS_exhaustive_new_hoover/results.pickle"
)

DATA_FILE = (
    "../results/tclr_model_testing/all/"
    + "TD4_MSS0.10_RT_MS_exhaustive_new_hoover/model_data.pickle"
)


def determine_core_reservoirs():
    data = read_pickle("../pickles/basin_group_res_dict.pickle")
    groups = ["small_st_dam", "medium_st_dam", "large_st_dam"]
    slots = [[b, g, ""] for b, g in product(data.keys(), groups)]

    for i, (basin, group, res) in enumerate(slots):
        res_list = data[basin][group]
        if len(res_list) == 0:
            res = None
        elif len(res_list) == 1:
            res = res_list[0]
        else:
            for j, iterres in enumerate(res_list):
                print(f"{j}: {iterres}")
            response = input(f"Choose reservoir for {group} in {basin}: ")
            res = res_list[int(response)]
        slots[i] = [basin, group, res]
    write_pickle("../pickles/selected_basin_group_res.pickle", slots)


def get_core_reservoirs():
    return read_pickle("../pickles/selected_basin_group_res.pickle")


def make_core_res_df(core_res):
    columns = ["basin", "group", "name", "pretty_name"]
    records = []
    for basin, group, res in core_res:
        if res:
            res, pres = res
            pres = pres.title()
            records.append([basin, group, res, pres])

    return pd.DataFrame.from_records(records, columns=columns)


def plot_core_reservoirs_map(
    res_locs, core_res, basin_info, save=False, add_legend=False, legend_labels=None
):
    fig = plt.figure(figsize=(19, 10))
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
    colors = [
        "#D62728",
        "#2CA02C",
        "#FF7F0E",
        "#1F77BF"
    ]

    color_map = {
        j: colors[i]
        for i, j in enumerate(["small_st_dam", "medium_st_dam", "large_st_dam"])
    }

    res_locs["basin"] = load_rbasins()
    non_core_res = res_locs[~res_locs["core"]]
    for ax, pos, (basin, binfo) in zip(axes, positions, basin_info.items()):
        make_basin_map(ax, binfo)
        ax.set_position(pos)
        bdf = core_res[core_res["basin"] == basin]
        core_colors = [color_map[i] for i in bdf["group"]]
        ax.scatter(
            bdf["x"],
            bdf["y"],
            edgecolor="k",
            marker="v",
            facecolor=core_colors,
            linewidths=0.5,
            zorder=10,
            s=250,
        )
        other_res = non_core_res[non_core_res["basin"] == basin]
        ax.scatter(
            other_res["long"],
            other_res["lat"],
            edgecolor="k",
            marker="v",
            facecolor=colors[-1],
            linewidths=0.5,
            zorder=9,
            s=250,
            alpha=0.5
        )

    if add_legend:
        leg_fig = plt.figure()
        leg_ax = leg_fig.add_subplot()
        leg_ax.set_axis_off()
        handles = [
            mpatch.Patch(facecolor=binfo["color"], edgecolor="k", linewidth=0.5)
            for binfo in basin_info.values()
        ]
        handles.extend(
            plt.scatter([], [], facecolor=c, marker="v", s=250, edgecolor="k", linewidth=0.5)
            for c in colors[:3]
        )
        handles.append(
            plt.scatter([], [], facecolor=colors[-1], s=250, marker="v", edgecolor="k", linewidth=0.5, alpha=0.5)
        )

        legend_labels.extend(basin_info.keys())
        # handles = handles[3:] + handles[:3]
        labels = legend_labels[3:] + legend_labels[:3]
        labels.append("Other Dams")

        leg_ax.legend(
            handles, labels, loc="center", frameon=False, prop={"size": 18}, ncol=4
        )

    return fig, axes


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


def make_core_reservoirs_split_map(core_res):
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
    core_res["x"] = [res_locs.loc[name, "long"] for name in core_res["name"]]
    core_res["y"] = [res_locs.loc[name, "lat"] for name in core_res["name"]]
    res_locs["core"] = [name in core_res["name"].values for name in res_locs.index.get_level_values(0)]

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
    fig, axes = plot_core_reservoirs_map(
        res_locs,
        core_res,
        basin_info,
        save=False,
        add_legend=True,
        legend_labels=["Small St. Dam", "Medium St. Dam", "Large St. Dam"],
    )
    plt.show()


def plot_core_res_seasonal_group_percentages(core_res):
    water_year = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    water_year_months = [calendar.month_abbr[i] for i in water_year]
    results = read_pickle(RESULT_FILE)
    groups = results["groups"]
    core_res = core_res.sort_values(by="group", ascending=False)

    resers = core_res["name"]
    groups = groups.loc[pd.IndexSlice[resers, :]]

    counts = groups.groupby(
        [groups.index.get_level_values(0), groups.index.get_level_values(1).month]
    ).value_counts()
    counts.index.names = ["site_name", "datetime", "group"]
    counts.name = "count"
    counts = counts.reset_index()
    # counts["group"] = counts["group"].replace(group_map)

    # fig, axes = plt.subplots(5, 2, sharex=True, sharey=True)
    # axes = axes.flatten()
    fig, all_axes = plt.subplots(3, 4, sharex=False, sharey=True)
    #           -------------------------
    # sm st dam | Col | Mis | Ten |     |
    #           -------------------------
    # md st dam | Col | Mis | Ten | PNW |
    #           -------------------------
    # lg st dam | Col | Mis | Ten |     |
    #           -------------------------
    all_axes = list(all_axes.flatten())
    axes = all_axes[:3] + all_axes[4:11]
    other_axes = [all_axes[3], all_axes[11]]
    for ax in other_axes:
        ax.axis("off")

    # swap 5 and 6
    resers = list(resers)
    resers_5 = resers[5]
    resers[5] = resers[6]
    resers[6] = resers_5

    legend_ax = [0, 3, 7]
    for i, (res, ax) in enumerate(zip(resers, axes)):
        rdf = counts[counts["site_name"] == res]
        rdf = rdf.drop("site_name", axis=1).pivot(index="datetime", columns="group")
        rdf.columns = rdf.columns.droplevel(0)
        rdf = rdf.apply(lambda x: x / x.sum() * 100, axis=1)
        rdf = rdf.loc[water_year]
        rdf.plot.bar(stacked=True, ax=ax, width=0.7)
        # nodes = list(rdf.columns)
        # high_inflow = 7
        # nodes.remove(high_inflow)
        # low_release = min(nodes)
        # normal = max(nodes)
        # rdf[high_inflow] /= rdf[normal]
        # rdf[low_release] /= rdf[normal]
        # rdf = rdf.drop(normal, axis=1)
        # rdf.plot.bar(ax=ax, width=0.7)

        core_res_row = core_res[core_res["name"] == res]
        pretty_name = core_res_row["pretty_name"].values[0]
        basin = core_res_row["basin"].values[0]
        group = core_res_row["group"].values[0]
        group = re.sub("st", "st.", group)
        pretty_group = " ".join(group.split("_")).title()

        # title = f"{pretty_name} - {basin} - {pretty_group}"
        title = pretty_name
        ax.set_title(title, fontsize=18)

        ax.tick_params(axis="x", labelrotation=0)

        handles, labels = ax.get_legend_handles_labels()
        # group_map = {
        #     "3": "4",
        #     "4": "5",
        #     "5": "6",
        #     "6": "7",
        #     "7": "3"
        # }
        group_map = {
            "3": "Low Release",
            "4": "Low Release",
            "5": "Normal Operation",
            "6": "Normal Operation",
            "7": "High Inflow"
        }
        labels = [group_map.get(i) for i in labels]
        # if i in legend_ax:
        #     ax.legend(
        #         handles, labels, title="", ncol=3, loc="lower left", prop={"size": 10}
        #     )
        # else:
        ax.get_legend().remove()
        ax.set_xticklabels(
            [i[0] for i in water_year_months],
            fontsize=14,
        )
        ax.set_xlabel("")
    
    other_axes[0].legend(
        handles, labels, title="Operational Mode", ncol=1, loc="center", prop={"size": 14}
    )

    fig.text(
        0.02,
        0.5,
        "Monthly Group Occurence Probability [%]",
        rotation=90,
        ha="center",
        va="center",
        fontsize=16,
    )

    plt.subplots_adjust(
        top=0.938, bottom=0.121, left=0.096, right=0.903, hspace=0.27, wspace=0.048
    )

    plt.show()
    # plt.savefig("../figures/agu_2022_figures/seasonal_group_percentages.png", dpi=600)


def plot_median_inflow_year_time_series(core_res):
    core_res = core_res.sort_values(by="group", ascending=False)
    results = read_pickle(RESULT_FILE)
    data = read_pickle(DATA_FILE)

    train_data = results["train_data"]

    inflow = data["xtrain"]["inflow"].unstack().T
    means = data["means"]["inflow"]
    std = data["std"]["inflow"]

    inflow = inflow * std + means

    res_years = {}
    for res in core_res["name"]:
        res_inflow = inflow[res]
        res_inflow = res_inflow.dropna()
        water_year = res_inflow.index
        water_year = water_year.year.where(water_year.month < 10, water_year.year + 1)

        # year_inflow = res_inflow.resample("Y").sum()
        # year_inflow.index = year_inflow.index.year
        year_inflow = res_inflow.groupby(water_year).sum()
        year_inflow = year_inflow.drop(year_inflow.index[0])
        year_inflow = year_inflow.drop(year_inflow.index[-1])
        # median_indices = get_n_median_index(year_inflow, 3)
        quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
        median_indices = [idxquantile(year_inflow, q) for q in quantiles]
        res_years[res] = median_indices

    # fig, axes = plt.subplots(5, 2, sharex=True, sharey=False)
    # axes = axes.flatten()
    fig, all_axes = plt.subplots(3, 4, sharex=False, sharey=False)
    #           -------------------------
    # sm st dam | Col | Mis | Ten |     |
    #           -------------------------
    # md st dam | Col | Mis | Ten | PNW |
    #           -------------------------
    # lg st dam | Col | Mis | Ten |     |
    #           -------------------------
    all_axes = list(all_axes.flatten())
    axes = all_axes[:3] + all_axes[4:11]
    other_axes = [all_axes[3], all_axes[11]]
    for ax in other_axes:
        ax.axis("off")

    # swap 5 and 6
    resers = list(res_years.keys())
    resers_5 = resers[5]
    resers[5] = resers[6]
    resers[6] = resers_5
    colors = sns.color_palette("colorblind", 3)
    for ax, res in zip(axes, resers):
        years = res_years[res]
        res_results = train_data.loc[pd.IndexSlice[res, :]]
        water_year = res_results.index
        water_year = water_year.year.where(water_year.month < 10, water_year.year + 1)
        new_index = [
            datetime.datetime(j, i.month, i.day)
            for i, j in zip(res_results.index, water_year)
        ]
        res_results.index = new_index
        years = years[1:-1]
        for year, color in zip(years, colors):
            year_data = res_results.loc[res_results.index.year == year]
            year_data.index = range(year_data.shape[0])

            ax.plot(
                year_data.index,
                year_data["actual"].values,
                color=color,
            )
            ax.plot(
                year_data.index,
                year_data["model"].values,
                color=color,
                linestyle="--",
            )
        core_res_row = core_res[core_res["name"] == res]
        pretty_name = core_res_row["pretty_name"].values[0]
        basin = core_res_row["basin"].values[0]
        group = core_res_row["group"].values[0]
        group = re.sub("st", "st.", group)
        pretty_group = " ".join(group.split("_")).title()

        title = f"{pretty_name} - {basin} - {pretty_group}"
        ax.set_title(title)

        ticks = get_julian_day_for_month_starts(2022, 10)
        water_year_months = [(i + 10 - 1) % 12 + 1 for i in range(12)]
        tick_labels = [calendar.month_abbr[i][0] for i in water_year_months]
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels)

    fig.text(
        0.13, 0.5, "Release [1000 acre-ft / day]", rotation=90, va="center", ha="center"
    )
    fig.text(0.5, 0.02, "Day of Water Year", ha="center")

    plt.subplots_adjust(
        top=0.961, bottom=0.065, left=0.15, right=0.85, hspace=0.322, wspace=0.067
    )

    leg_fig = plt.figure()
    leg_ax = leg_fig.add_subplot()
    leg_ax.set_axis_off()
    handles = [mlines.Line2D([], [], color=c) for c in colors]
    handles.extend(
        [
            mlines.Line2D([], [], color="k"),
            mlines.Line2D([], [], color="k", linestyle="--"),
        ]
    )

    year_labels = ["Min.", r"25%ile", r"50%ile", r"75%ile", "Max."]
    year_labels = year_labels[1:-1] 
    labels = year_labels + ["Observed", "Predicted"]
    order = [0,5,1,6,2,3,4]
    order = [0,4,1,4,2]
    leg_ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order], 
        loc="center", 
        prop={"size": 20}, 
        ncol=3,
        frameon=False,
    )

    plt.show()


if __name__ == "__main__":
    plt.style.use("seaborn-colorblind")
    sns.set_context("notebook", font_scale=1.1)
    core_res = get_core_reservoirs()
    core_res = make_core_res_df(core_res)
    # make_core_reservoirs_split_map(core_res)
    plot_core_res_seasonal_group_percentages(core_res)
    # plot_median_inflow_year_time_series(core_res)
