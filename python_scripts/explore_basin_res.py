import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import embed as II

from multi_basin_tree import read_basin_data, get_basin_meta_data
from plot_helpers import determine_grid_size


plt.style.use("ggplot")
sns.set_context("notebook")


def parse_args():
    args = sys.argv[1:]
    if len(args) > 0:
        basin = args[0].lower()
    else:
        raise IndexError("Must provide basin as first command line argument.")
    return basin


def load_actual_res_names(basin):
    file = f"../{basin}_data/dam_names.json"

    with open(file, "r") as f:
        names = json.load(f)

    return names


def get_date_spans(data):
    min_dates = data.groupby(data.index.get_level_values(0)).apply(
        lambda x: x.index.get_level_values(1).min()
    )
    max_dates = data.groupby(data.index.get_level_values(0)).apply(
        lambda x: x.index.get_level_values(1).max()
    )
    spans = pd.DataFrame({"start": min_dates, "stop": max_dates})
    spans["span"] = spans["stop"] - spans["start"]
    return spans


def plot_timeline(spans):
    import plotly.express as px
    sorted_spans = spans.sort_values(by="span")
    fig = px.timeline(
        sorted_spans,
        x_start="start",
        x_end="stop",
        y=sorted_spans.index
    )
    fig.update_yaxes(autorange="reversed")
    fig.show()


def plot_lag_corrs(data, meta, add_lags=7):
    phys_vars = ["release", "storage", "inflow"]
    df = data.loc[:, phys_vars]
    res_grouper = df.index.get_level_values(0)
    for var in phys_vars:
        for lag in range(1, add_lags+1):
            new_var = f"{var}_L{lag}"
            df[new_var] = df.groupby(res_grouper)[var].shift(lag)
    df = df.dropna()
    corrs = df.groupby(df.index.get_level_values(0)).corr()["release"].unstack()

    grid_size = determine_grid_size(corrs.index.size)
    fig, axes = plt.subplots(*grid_size, sharex=True, sharey=True)
    flat_axes = axes.flatten()
    x = range(add_lags+1)
    markers = ["o", "s", "D"]
    for res, ax in zip(corrs.index, flat_axes):
        for j, var in enumerate(phys_vars):
            pdf = corrs.loc[res, [i for i in corrs.columns if var in i]]
            ax.plot(x, pdf.values, marker=markers[j], label=var)
        rt = meta.loc[res, "rts"]
        ax.set_title(f"{res} [{rt:.0f}]")
        ax.set_xticks(x)
        if ax in axes[-1, :]:
            ax.set_xticklabels(x)
            ax.set_xlabel("Lag")
        if ax in axes[:, 0]:
            ax.set_ylabel("Corr with Release")
        if ax == flat_axes[0]:
            handles, labels = ax.get_legend_handles_labels()

    left_over = flat_axes.size - corrs.index.size
    if left_over > 0:
        for i in range(1, left_over + 1):
            flat_axes[-i].set_axis_off()
        flat_axes[-1].legend(handles, labels, loc="center", prop={"size": 16})
    else:
        flat_axes[0].legend(handles, labels, loc="best")

    plt.show()

def plot_interaction_terms(data, meta, aggregate=False):
    new_terms = [
        ("release_pre", "storage_pre"),
        ("release_pre", "inflow"),
        ("release_pre", "release_L2"),
        ("storage_pre", "storage_pre"),
        ("release_pre", "release_pre"),
        ("inflow", "inflow")
    ]
    term_map = {
        "release_pre_x_storage_pre": r"$R_{t-1} \times S_{t-1}$",
        "release_pre_x_inflow": r"$R_{t-1} \times I_{t}$",
        "release_pre_x_release_L2": r"$R_{t-1} \times R_{t-2}$",
        "storage_pre_x_storage_pre": r"$S_{t-1}^2$",
        "release_pre_x_release_pre": r"$R_{t-1}^2$",
        "inflow_x_inflow": r"$I_{t}^2$",
        "storage_x_inflow": r"$S_{t-1} \times I_{t}$"
    }
    data["release_L2"] = data.groupby(
        data.index.get_level_values(0)
    )["release_pre"].shift(1)
    data = data.dropna()
    interactions = {}
    for t1, t2 in new_terms:
        name = f"{t1}_x_{t2}"
        interactions[name] = data[t1] * data[t2]
    int_df = pd.DataFrame.from_dict(interactions)
    int_df["release"] = data["release"]
    int_df["storage_x_inflow"] = data["storage_x_inflow"]
    int_df = int_df[["storage_x_inflow"]+list(int_df.columns[:-1])]
    corrs = int_df.groupby(
        int_df.index.get_level_values(0)
    ).corr()["release"].unstack().drop("release", axis=1)

    x = range(corrs.columns.size)
    int_vars = corrs.columns
    if aggregate:
        sns.set_context("talk")
        corrs = corrs.melt(ignore_index=False).reset_index()
        ax = sns.boxplot(data=corrs, x="variable", y="value")
        ax.set_xticks(x)
        ax.set_xticklabels([term_map.get(i) for i in int_vars],
                           ha="center")
        ax.set_ylabel("Corr with release")
        ax.set_xlabel("")
    else:
        grid_size = determine_grid_size(corrs.index.size)
        fig, axes = plt.subplots(*grid_size, sharex=True, sharey=True)
        flat_axes = axes.flatten()
        for res, ax in zip(corrs.index, flat_axes):
            ax.bar(x, corrs.loc[res, :])
            rt = meta.loc[res, "rts"]
            ax.set_title(f"{res} [{rt:.0f}]")
            ax.set_xticks(x)
            if ax in axes[-1, :]:
                ax.set_xticklabels([term_map.get(i) for i in corrs.columns],
                                   ha="center", rotation=90, fontsize=16)
            if ax in axes[:, 0]:
                ax.set_ylabel("Corr with Release")

        left_over = flat_axes.size - corrs.index.size
        if left_over > 0:
            for i in range(1, left_over + 1):
                flat_axes[-i].set_axis_off()

    plt.show()


def main():
    basin = parse_args()
    data = read_basin_data(basin)
    meta = get_basin_meta_data(basin)
    if basin == "pnw" or basin == "missouri":
        names = load_actual_res_names(basin)
        meta = meta.rename(index=names)
        data_res = [names.get(i) for i in data.index.get_level_values(0)]
        new_index = pd.MultiIndex.from_tuples(zip(data_res, data.index.get_level_values(1)),
                                              names=["site_name", "date"])
        data.index = new_index
    # spans = get_date_spans(data)
    # plot_timeline(spans)
    # plot_lag_corrs(data, meta, add_lags=14)
    plot_interaction_terms(data, meta, True)



if __name__ == "__main__":
    main()
