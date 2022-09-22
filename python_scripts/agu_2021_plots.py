import pickle
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from IPython import embed as II
from utils.helper_functions import read_tva_data
from plot_helpers import determine_grid_size

plt.style.use("ggplot")
sns.set_context("talk")
colors = sns.color_palette()

ncsu_colors = [
    "#990000",
    "#D14905",
    "#FAC800",
    "#6F7D1C",
    "#008473",
    "#427E93",
    "#4156A1",
]
my_ncsu_colors = ["#ea1500", "#84a0dc"]
bg = "#F2F2F2"
colors = ncsu_colors[::-1]

TVA = read_tva_data(just_load=True)
TREE_RES = [
    "BlueRidge",
    "Chatuge",
    "Cherokee",
    "Douglas",
    "Fontana",
    "Hiwassee",
    "Norris",
    "Nottely",
    "SHolston",
    "TimsFord",
    "Watauga",
]
SIMP_RES = [
    "Apalachia",
    "Boone",
    "Chikamauga",
    "FtLoudoun",
    "FtPatrick",
    "Guntersville",
    "Kentucky",
    "MeltonH",
    "Nikajack",
    "Ocoee1",
    "Ocoee3",
    "Pickwick",
    "WattsBar",
    "Wheeler",
    "Wilbur",
    "Wilson",
]


def load_tree_data():
    with open(
        "../results/agu_2021_runs/tree_model_temporal_validation/results.pickle", "rb"
    ) as f:
        results = pickle.load(f)
    return results


def load_simple_data():
    with open(
        "../results/agu_2021_runs/simple_model_temporal_validation/results.pickle", "rb"
    ) as f:
        results = pickle.load(f)
    return results


def load_simple_all_data():
    with open(
        "../results/agu_2021_runs/simple_model_temporal_validation_all_res/results.pickle",
        "rb",
    ) as f:
        results = pickle.load(f)
    return results


def load_spatial_data():
    with open(
        "../results/agu_2021_runs/simple_model_spatial_validation_all_res_lvoneout/results.pickle",
        "rb",
    ) as f:
        results = pickle.load(f)
    return results


def combine_tree_simple_release(tree, simple):
    train_data = tree["data"]["train_data"]
    train_data = train_data.append(simple["data"]["y_results"]).sort_index()

    pred_data = tree["data"]["test_p_data"]
    pred_data = pred_data.append(simple["data"]["y_pred"]).sort_index()

    fc_data = tree["data"]["test_f_data"]
    fc_data = fc_data.append(simple["data"]["y_forc"]).sort_index()
    test_data = combine_pred_fc_data(pred_data, fc_data)
    test_data = test_data.dropna()
    return train_data, test_data


def combine_pred_fc_data(pred_data, fc_data):
    return pd.DataFrame(
        {
            "actual": pred_data["actual"],
            "preds": pred_data["model"],
            "fc": fc_data["model"],
        }
    )


def combine_tree_simple_storage(tree, simple):
    stor = tree["data"]["forecasted"]["Storage_act"]
    stor = stor.append(simple["data"]["forecasted"]["Storage_act"]).sort_index()
    stor = pd.DataFrame({"model": stor, "actual": TVA.loc[stor.index, "Storage"]})
    return stor


def calc_grouped_metric(data, grouper, act_var="actual", mod_var="model", metric="nse"):
    if metric == "nse":
        values = data.groupby(grouper).apply(lambda x: r2_score(x[act_var], x[mod_var]))
    else:
        values = data.groupby(grouper).apply(
            lambda x: mean_squared_error(x[act_var], x[mod_var], squared=False)
        )
    return values


def get_release_means():
    return TVA.groupby(TVA.index.get_level_values(1))["Release"].mean()


def get_storage_means():
    return TVA.groupby(TVA.index.get_level_values(1))["Storage"].mean()


def plot_score_bars_w_simul(test_data, simul_data, storage=False):
    res_grouper = test_data.index.get_level_values(1)
    # resers = res_grouper.unique()
    resers = np.array([*TREE_RES, *SIMP_RES])
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(19, 10))
    fig.patch.set_alpha(0.0)
    axes = axes.flatten()

    nse_fc = calc_grouped_metric(test_data, res_grouper, mod_var="fc")

    if storage:
        simvar = "Storage"
    else:
        simvar = "Release"

    nse_simul = calc_grouped_metric(
        simul_data,
        simul_data.index.get_level_values(1),
        act_var=f"{simvar}_act",
        mod_var=f"{simvar}_simul",
        metric="nse",
    )

    if storage:
        means = get_release_means()
    else:
        means = get_storage_means()

    rmse_fc = (
        calc_grouped_metric(test_data, res_grouper, mod_var="fc", metric="rmse")
        / means
        * 100
    )

    rmse_simul = (
        calc_grouped_metric(
            simul_data,
            simul_data.index.get_level_values(1),
            act_var=f"{simvar}_act",
            mod_var=f"{simvar}_simul",
            metric="rmse",
        )
        / means
        * 100
    )


    ax1 = axes[0]
    # ax2 = ax1.twiny()

    width = 0.4
    y1 = [i - width / 2 for i in range(resers.size)]
    y2 = [i + width / 2 for i in range(resers.size)]

    act_names = {}
    with open("./actual_names.csv", "r") as f:
        for line in f.readlines():
            line = line.strip("\n\r")
            mname, aname = line.split(",")
            act_names[mname] = aname
    labels = [act_names.get(i) for i in resers]

    # ax1.barh(y2, nse_pred.loc[resers], height=width, color=my_ncsu_colors[0], label="P")
    # ax1.barh(y1, nse_fc.loc[resers], height=width, color=my_ncsu_colors[1], label="S")
    ax1.bar(y2, nse_simul.loc[resers], width=width, label="Simul. Fit")
    ax1.bar(y1, nse_fc.loc[resers], width=width, label="LR Fit")
    if storage:
        ax1.set_ylabel("Storage NSE")
    else:
        ax1.set_ylabel("Release NSE")
    ax1.grid(visible=True, axis="y")

    ax1 = axes[1]
    # ax2 = ax1.twiny()

    # ax1.barh(y2, rmse_pred.loc[resers], height=width, label="Predicted")
    # ax1.barh(y1, rmse_fc.loc[resers], height=width, label="Simulated")
    ax1.bar(y2, rmse_simul.loc[resers], width=width, label="Simul. Fit")
    ax1.bar(y1, rmse_fc.loc[resers], width=width, label="LR Fit")
    if storage:
        ax1.set_ylabel("Storage RMSE [%]")
    else:
        ax1.set_ylabel("Release RMSE [%]")
    ax1.grid(visible=True, axis="y")
    ax1.legend(loc="best")
    ax1.set_xticks(range(resers.size))
    ax1.set_xticklabels(labels, rotation=45, ha="right")

    plt.tight_layout()
    fig.align_ylabels()
    # plt.savefig("/home/lford/projects/predict-release/agu_figures/nse_rmse_temp_val_bars_new.png", dpi=400, facecolor="#FFFFFFFF")
    plt.show()


def plot_score_bars(test_data):
    res_grouper = test_data.index.get_level_values(1)
    # resers = res_grouper.unique()
    resers = np.array([*TREE_RES, *SIMP_RES])
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(19, 10))
    fig.patch.set_alpha(0.0)
    axes = axes.flatten()

    nse_fc = calc_grouped_metric(test_data, res_grouper, mod_var="fc")

    nse_pred = calc_grouped_metric(test_data, res_grouper, mod_var="pred")

    release_means = get_release_means()

    rmse_fc = (
        calc_grouped_metric(test_data, res_grouper, mod_var="fc", metric="rmse")
        / release_means
        * 100
    )

    rmse_pred = (
        calc_grouped_metric(test_data, res_grouper, mod_var="pred", metric="rmse")
        / release_means
        * 100
    )

    ax1 = axes[0]
    # ax2 = ax1.twiny()

    width = 0.4
    y1 = [i - width / 2 for i in range(resers.size)]
    y2 = [i + width / 2 for i in range(resers.size)]

    act_names = {}
    with open("./actual_names.csv", "r") as f:
        for line in f.readlines():
            line = line.strip("\n\r")
            mname, aname = line.split(",")
            act_names[mname] = aname
    labels = [act_names.get(i) for i in resers]

    # ax1.barh(y2, nse_pred.loc[resers], height=width, color=my_ncsu_colors[0], label="P")
    # ax1.barh(y1, nse_fc.loc[resers], height=width, color=my_ncsu_colors[1], label="S")
    ax1.bar(y2, nse_pred.loc[resers], width=width, label="P")
    ax1.bar(y1, nse_fc.loc[resers], width=width, label="S")
    ax1.set_ylabel("Release NSE")
    ax1.grid(visible=True, axis="y")

    ax1 = axes[1]
    # ax2 = ax1.twiny()

    # ax1.barh(y2, rmse_pred.loc[resers], height=width, label="Predicted")
    # ax1.barh(y1, rmse_fc.loc[resers], height=width, label="Simulated")
    ax1.bar(y2, rmse_pred.loc[resers], width=width, label="Predicted")
    ax1.bar(y1, rmse_fc.loc[resers], width=width, label="Simulated")
    ax1.set_ylabel("Release RMSE [%]")
    ax1.grid(visible=True, axis="y")
    ax1.legend(loc="best")
    ax1.set_xticks(range(resers.size))
    ax1.set_xticklabels(labels, rotation=45, ha="right")

    plt.tight_layout()
    # plt.savefig("/home/lford/projects/predict-release/agu_figures/nse_rmse_temp_val_bars_new.png", dpi=400, facecolor="#FFFFFFFF")
    plt.show()


def plot_time_series(test_data):
    res_grouper = test_data.index.get_level_values(1)
    # resers = res_grouper.unique()
    resers = np.array([*TREE_RES, *SIMP_RES])
    gs = determine_grid_size(resers.size)
    fig, axes = plt.subplots(*gs, sharex=True)
    axes = axes.flatten()

    idx = pd.IndexSlice
    for r, ax in zip(resers, axes):
        rdf = test_data.loc[idx[:, r], :]
        x = rdf.index.get_level_values(0)
        ax.plot(x, rdf["actual"], color=colors[0], label="Observed")
        ax.plot(x, rdf["preds"], color=colors[1], label="Predicted")
        ax.plot(x, rdf["fc"], color=colors[2], label="Simulated")
        ax.set_title(r)
        handles, labels = ax.get_legend_handles_labels()

    axes[-1].legend(handles, labels, loc="best")
    axes[-1].set_axis_off()

    plt.show()


def plot_coef_bars(coefs):
    coefs = coefs.loc[
        [
            "Release_pre",
            "Storage_pre",
            "Net Inflow",
            "Release_roll7",
            "Storage_roll7",
            "Inflow_roll7",
            "Storage_Inflow_interaction",
        ]
    ]

    coefs = coefs.rename(
        columns={
            "ComboFlow-RunOfRiver": "Run of River",
            "ComboFlow-StorageDam": "LRT",
            "NaturalFlow-StorageDam": "HRT",
        }
    )
    title_map = {
        "Run of River": "Run of River",
        "LRT": "Low Residence Time",
        "HRT": "High Residence Time",
    }
    labels = [
        "Previous Release",
        "Previous Storage",
        "Current Inflow",
        "Weekly Mean Release",
        "Weekly Mean Storage",
        "Weekly Mean Inflow",
        "Storage Inflow Interaction",
    ]
    labels = ["\n".join(i.split()) for i in labels]

    fig, axes = plt.subplots(3, 1, sharex=True)
    fig.patch.set_alpha(0.0)
    axes = axes.flatten()
    x = range(coefs.index.size)
    for ax, group in zip(axes, coefs.columns):
        ax.set_prop_cycle(c=my_ncsu_colors)
        ax.bar(x, coefs[group])
        ax.set_title(title_map[group])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)

    fig.text(0.02, 0.5, "Fitted Coefficient", ha="center", va="center", rotation=90)
    plt.show()


def plot_spatial_validation(orig_data, spatial_data):
    spatial_scores = pd.concat([i["fc_res_scores"] for i in spatial_data.values()])
    orig_scores = pd.DataFrame(columns=["NSE", "RMSE"], index=spatial_scores.index)
    orig_scores["NSE"] = orig_data.groupby(orig_data.index.get_level_values(1)).apply(
        lambda x: r2_score(x["actual"], x["fc"])
    )

    orig_scores["RMSE"] = orig_data.groupby(orig_data.index.get_level_values(1)).apply(
        lambda x: mean_squared_error(x["actual"], x["fc"], squared=False)
    )

    II()


if __name__ == "__main__":
    tree = load_tree_data()
    simple = load_simple_data()
    simple_all = load_simple_all_data()
    spatial_data = load_spatial_data()
    sa_test_rel = combine_pred_fc_data(
        simple_all["data"]["y_pred"], simple_all["data"]["y_forc"]
    ).dropna()
    storage_results = simple_all["data"]["forecasted"][["Storage", "Storage_act"]]
    storage_results = storage_results.rename(
        columns={"Storage": "fc", "Storage_act": "actual"}
    )
    simul_results = pd.read_pickle(
        "../results/agu_2021_runs/simul_model/simul_test_rel_sto_output.pickle"
    )
    # II()
    # sys.exit()
    # train_rel, test_rel = combine_tree_simple_release(tree, simple)
    # sto_data = combine_tree_simple_storage(tree, simple)
    # plot_score_bars(sa_test_rel)
    # plot_time_series(test_rel)
    # plot_coef_bars(simple_all["coefs"])
    # plot_spatial_validation(sa_test_rel, spatial_data)
    plot_score_bars_w_simul(sa_test_rel, simul_results, storage=False)
    # plot_score_bars_w_simul(storage_results, simul_results, storage=True)
