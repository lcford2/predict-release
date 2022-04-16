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

TREE_STRUCT = {
    2:{
        "0": ["1", "8"],
        "1": ["2", "5"],
        "8": ["9", "12"]
    }
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

def load_coefs():
    path = "../results/tclr_spatial_eval/all/"
    dirfmt = "TD3_RT_MS_0.75_{seed}"
    output = {}
    for seed in range(1000):
        file = pathlib.Path(path) / \
               dirfmt.format(seed=seed) / \
               "random_effects.csv"
        output[seed] = pd.read_csv(file.as_posix(), index_col=0)
    return output

def parse_graph(tree):
    output = {}
    for n in tree.nodes:
        node = tree.nodes[n]
        label = node["label"]
        split = label.split("\n")[-1]
        try:
            var, thresh = split.split(" &le; ")
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

def determine_tree_structure(node_vars):
    structs = [
        set([str(i) for i in [0,1,2,5,8,9]]), # 133
        set([str(i) for i in [0,1,2,6,7,10]]), # 885
        set([str(i) for i in [0,1,2,5,8,9,12]]) # 1
    ]
    tree_structs = []
    for i, row in node_vars.iterrows():
        nodes = set(row.dropna().index.tolist())
        for j, strct in enumerate(structs):
            if strct == nodes:
                tree_structs.append(j)
                break
    return tree_structs

def get_variable_counts(node_vars):
    var_counts = {}
    for col in node_vars.columns:
        var_counts[col] = node_vars[col].value_counts()
    var_counts = pd.DataFrame(var_counts)
    return var_counts

def combine_var_thresh(node_vars, node_thresh):
    node_vars = node_vars.reset_index().rename(columns={"index":"seed"}).melt(
        id_vars=["struct", "seed"], var_name="node", value_name="var")
    node_thresh = node_thresh.reset_index().rename(columns={"index":"seed"}).melt(
        id_vars=["seed"], var_name="node", value_name="thresh")

    node_vars = node_vars.set_index(["seed", "node"])
    node_thresh = node_thresh.set_index(["seed", "node"])
    node_vars["thresh"] = node_thresh["thresh"]
    node_vars = node_vars.reset_index().dropna()
    return node_vars

def plot_thresh_dist(df, node=0, struct=None, var=None, show=False, save=False):
    if struct is None:
        struct = "all"
    sns.set_context("notebook")
    pdf = df[df["node"] == str(node)]
    if not var is None:
        if isinstance(var, list):
            pvars = var
        else:
            pvars = [var]
        nvars = len(pvars)
    else:
        pvars = pdf["var"].unique()
        nvars = pvars.size

    pvar_counts = pdf["var"].value_counts()
    fig, axes = plt.subplots(
        1, nvars,
        sharex=False, sharey=False,
        figsize=(16,9)
    )
    try:
        axes = axes.flatten()
    except AttributeError:
        axes = [axes]
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
            file = f"var_thresh_hist_node_{struct}_{node}"
            filename = f"{dirname}/{file}.png"
        plt.savefig(filename)
    if show:
        plt.show()
    plt.close()

def plot_all_thresh_dist(comb):
    for struct in [None, 0, 1, 2]:
        if struct is None:
            pdf = comb
        else:
            pdf = comb[comb["struct"] == struct]
        nodes = pdf["node"].unique()
        for node in nodes:
            plot_thresh_dist(
                pdf,
                node=node,
                struct=struct,
                show=False,
                save=True
            )

def cond_prob_splits(comb, struct=2):
    df = comb[comb["struct"] == struct]
    ts = TREE_STRUCT[struct]
    nodes = list(ts.keys())
    output = {}
    for node in nodes:
        node_df = df[df["node"] == node]
        nvars = node_df["var"].unique().tolist()
        for child in ts[node]:
            child_df = df[df["node"] == child]
            output[f"N{node}C{child}"] = {}
            for var in nvars:
                vdf = node_df[node_df["var"] == var]
                seeds = vdf["seed"]
                counts = child_df[child_df["seed"].isin(seeds)]["var"].value_counts()
                output[f"N{node}C{child}"][var] = counts

    mi, records = [],[]
    for key, value in output.items():
        for key2, record in value.items():
            mi.append((key, key2))
            records.append(record)

    return pd.DataFrame.from_records(records, index=pd.MultiIndex.from_tuples(mi))

def cond_prob_heatmaps(cps):
    idx = pd.IndexSlice
    slicers = cps.index.get_level_values(0).unique()
    for slicer in slicers:
        pdf = cps.loc[idx[slicer, :], :].dropna(axis=1, how="all")
        pdf.index = pdf.index.droplevel(0)
        pdf = pdf.T
        ppdf = pdf / pdf.sum()
        annots = [
            [f"{pdf.values[i,j]:.0f}\n{ppdf.values[i,j]:.0%}"
             for j in range(pdf.shape[1])] for i in range(pdf.shape[0])
        ]
        ax = sns.heatmap(pdf, annot=annots, fmt="s",
                         cbar_kws={"label": "Number of Occurences"})
        plt.gcf().patch.set_alpha(0.0)
        ax.set_xlabel(f"Node {slicer[1]} Variable")
        ax.set_ylabel(f"Node {''.join(slicer[3:])} Variable")
        ax.set_yticklabels(
            [VAR_MAP_SHORT[i.get_text()] for i in ax.get_yticklabels()]
        )
        ax.set_xticklabels(
            [VAR_MAP_SHORT[i.get_text()] for i in ax.get_xticklabels()],
            rotation=45,
            ha="right"
        )
        plt.show()

def main():
    plt.style.use("ggplot")
    text_color = "black"
    mpl.rcParams["text.color"] = text_color
    mpl.rcParams["axes.labelcolor"] = text_color
    mpl.rcParams["xtick.color"] = text_color
    mpl.rcParams["ytick.color"] = text_color

    sns.set_context("paper")

    trees = load_data()
    parsed = {s:parse_graph(t) for s, t in trees.items()}
    node_vars, node_thresh = trees_to_frames(parsed)
    node_vars["struct"] = determine_tree_structure(node_vars)
    var_counts = get_variable_counts(node_vars)
    comb = combine_var_thresh(node_vars, node_thresh)
    s2 = comb[comb["struct"] == 2]
    coefs = load_coefs()
    coefs = {i: coefs.get(i) for i in s2["seed"].unique()}
    mi = pd.MultiIndex.from_product(
        (coefs.keys(), coefs[0].index), names=["seed", "variable"]
    )
    cdf = pd.DataFrame(index=mi, columns=range(1, 9))
    idx = pd.IndexSlice
    for key, coef in coefs.items():
        cdf.loc[idx[key, coef.index], :] = coef.values

    rts_seeds = s2[(s2["node"] == "0") & (s2["var"] == "rts")]["seed"]
    relpre_seeds = s2[(s2["node"] == "0") & (s2["var"] == "release_pre")]["seed"]

    cdf = cdf.reset_index().melt(id_vars=["seed", "variable"], var_name="node", value_name="fitted_value")
    cdf = cdf[cdf["seed"].isin(rts_seeds)]
    sns.displot(data=cdf, x="fitted_value", row="variable", hue="node",
                facet_kws={"sharex":False, "sharey":False})
    plt.show()

    # fig, axes = plt.subplots(8, cdf["variable"].unique().size, sharex=False, sharey=True)

    # for i, axes_i in zip(range(1,9), axes):
    #     node = i
    #     for coef, ax in zip(cdf["variable"].unique(), axes_i):
    #         pdf = cdf.loc[(cdf["node"] == node) & (cdf["variable"] == coef)]
    #         sns.histplot(ax=ax, data=pdf, x="fitted_value")
    #         if node == 1:
    #             ax.set_title(coef)
    #         if ax == axes_i[0]:
    #             ax.set_ylabel(f"Count [Node: {node}]")
    #         if node != 8:
    #             ax.set_xlabel("")
    # plt.show()

    # cps = cond_prob_splits(comb)
    # cond_prob_heatmaps(cps)
    # plot_thresh_dict(
    #     comb[comb["struct"] == 2],
    #     node=9,
    #     struct=2,
    #     var="rts",
    #     show=True
    # )

if __name__ == "__main__":
    main()
