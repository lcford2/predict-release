import pathlib
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import defaultdict
from IPython import embed as II

VAR_MAP = {
    "rts": "Residence Time [days]",
    "release_pre": "Prev. Release [1000 acre-ft/day]",
    "storage_pre": "Prev. Storage [1000 acre-ft]",
    "inflow": "Net Inflow [1000 acre-ft/day]",
    "release_roll7": "Rel. Weekly Mean [1000 acre-ft/day]",
    "storage_roll7": "Sto. Weekly Mean [1000 acre-ft]",
    "inflow_roll7": "Inf. Weekly Mean [1000 acre-ft/day]",
    "storage_x_inflow": "St. x Inf.",
    "max_sto": "Maximum Storage [1000 acre-ft]"
}
VAR_MAP_SHORT = {
    "rts": "RT",
    "release_pre": "Rel Pre",
    "storage_pre": "Sto Pre",
    "inflow": "Net Inf",
    "release_roll7": "Rel Roll7",
    "storage_roll7": "Sto Roll7",
    "inflow_roll7": "Inf Roll7",
    "storage_x_inflow": "Sto x Inf",
    "max_sto": "Sto Max"
}

def load_data():
    path = "../results/tclr_spatial_eval/all/"
    dirfmt = "TD3_RT_MS_0.75_{seed}"
    output = {}
    for seed in range(1000):
        file = pathlib.Path(path) / \
               dirfmt.format(seed=seed) / \
               "tree.dot"
        output[seed] = nx.Graph(
            nx.drawing.nx_agraph.read_dot(
                file.as_posix())).to_directed()
    return output

def parse_graph(tree):
    output = {}
    for n in tree.nodes:
        node = tree.nodes[n]
        label = node["label"]
        split = label.split("\n")[-1]
        try:
            var, thresh = split.split(" > ")
            output[n] = (var, thresh)
        except ValueError as e:
            # leaf node
            pass
    return output

def trees_to_frames(parsed):
    node_vars = {str(i):[np.nan for _ in range(1000)] for i in range(13)}
    node_thresh = {str(i):[np.nan for _ in range(1000)] for i in range(13)}
    for s, p in parsed.items():
        for node, (var, thresh) in p.items():
            node_vars[node][int(s)] = var
            node_thresh[node][int(s)] = float(thresh)
    node_vars = pd.DataFrame(node_vars).dropna(how="all", axis=1)
    node_thresh = pd.DataFrame(node_thresh).dropna(how="all", axis=1)
    return node_vars, node_thresh

def get_variable_counts(node_vars):
    var_counts = {}
    for col in node_vars.columns:
        var_counts[col] = node_vars[col].value_counts()
    var_counts = pd.DataFrame(var_counts)
    return var_counts

def combine_var_thresh(node_vars, node_thresh):
    node_vars = node_vars.reset_index().rename(columns={"index":"seed"}).melt(
        id_vars=["seed"], var_name="node", value_name="var")
    node_thresh = node_thresh.reset_index().rename(columns={"index":"seed"}).melt(
        id_vars=["seed"], var_name="node", value_name="thresh")

    node_vars = node_vars.set_index(["seed", "node"])
    node_thresh = node_thresh.set_index(["seed", "node"])
    node_vars["thresh"] = node_thresh["thresh"]
    node_vars = node_vars.reset_index().dropna()
    return node_vars

def plot_thresh_dist(df, node=0, show=False, save=False):
    sns.set_context("notebook")
    pdf = df[df["node"] == str(node)]
    pvars = pdf["var"].unique()
    pvar_counts = pdf["var"].value_counts()
    nvars = pvars.size
    fig, axes = plt.subplots(
        1, nvars,
        sharex=False, sharey=False,
        figsize=(16,9)
    )
    axes = axes.flatten()
    for var, ax in zip(pvars, axes):
        sns.histplot(
            data=pdf[pdf["var"] == var],
            x="thresh",
            stat="count",
            kde=False,
            legend=False,
            ax=ax
        )
        ax.set_title(f"{VAR_MAP_SHORT.get(var, var)} (N={pvar_counts.get(var)})")
        # ax.set_title(f"{var} (N={pvar_counts.get(var)})")
        ax.set_xlabel("Threshold")
    fig.suptitle(f"Node {node}")

    figmgr = plt.get_current_fig_manager()
    figmgr.window.showMaximized()
    if nvars > 5:
        wspace = 0.35 + 0.05 * (nvars - 5)
    else:
        wspace = 0.35
    plt.subplots_adjust(
        top=0.926,
        bottom=0.065,
        left=0.042,
        right=0.988,
        hspace=0.2,
        wspace=wspace
    )
    if save:
        if isinstance(save, str):
            filename = save
        else:
            dirname = "/home/lford/Documents/move_to_drive/03_30_2022_meeting"
            file = f"var_thresh_hist_node_{node}"
            filename = f"{dirname}/{file}.png"
        plt.savefig(filename)
    if show:
        plt.show()

def main():
    plt.style.use("ggplot")
    text_color = "black"
    mpl.rcParams["text.color"] = text_color
    mpl.rcParams["axes.labelcolor"] = text_color
    mpl.rcParams["xtick.color"] = text_color
    mpl.rcParams["ytick.color"] = text_color

    sns.set_context("talk")

    trees = load_data()
    parsed = {s:parse_graph(t) for s, t in trees.items()}
    node_vars, node_thresh = trees_to_frames(parsed)
    var_counts = get_variable_counts(node_vars)
    comb = combine_var_thresh(node_vars, node_thresh)
    nodes = comb["node"].unique()
    for node in nodes:
        plot_thresh_dist(comb, node=node, show=False, save=True)

if __name__ == "__main__":
    main()
