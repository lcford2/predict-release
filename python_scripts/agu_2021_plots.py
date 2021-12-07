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

plt.style.use("seaborn-paper")
sns.set_context("paper")
colors = sns.color_palette()

TVA = read_tva_data(just_load=True)
TREE_RES = ['BlueRidge', 'Chatuge', 'Cherokee', 'Douglas', 'Fontana', 'Hiwassee',
       'Norris', 'Nottely', 'SHolston', 'TimsFord', 'Watauga']
SIMP_RES = ['Apalachia', 'Boone', 'Chikamauga', 'FtLoudoun', 'FtPatrick',
       'Guntersville', 'Kentucky', 'MeltonH', 'Nikajack', 'Ocoee1', 'Ocoee3',
       'Pickwick', 'WattsBar', 'Wheeler', 'Wilbur', 'Wilson']

def load_tree_data():
    with open("../results/agu_2021_runs/tree_model_temporal_validation/results.pickle", "rb") as f:
        results = pickle.load(f)
    return results

def load_simple_data():
    with open("../results/agu_2021_runs/simple_model_temporal_validation/results.pickle", "rb") as f:
        results = pickle.load(f)
    return results

def combine_tree_simple_release(tree, simple):
    train_data = tree["data"]["train_data"]
    train_data = train_data.append(simple["data"]["y_results"]).sort_index()

    pred_data = tree["data"]["test_p_data"]
    pred_data = pred_data.append(simple["data"]["y_pred"]).sort_index()

    fc_data = tree["data"]["test_f_data"]
    fc_data = fc_data.append(simple["data"]["y_forc"]).sort_index()
    test_data = pd.DataFrame({"actual":pred_data["actual"], "preds":pred_data["model"], "fc":fc_data["model"]})
    test_data = test_data.dropna()
    return train_data, test_data

def combine_tree_simple_storage(tree, simple):
    stor = tree["data"]["forecasted"]["Storage_act"]
    stor = stor.append(simple["data"]["forecasted"]["Storage_act"]).sort_index()
    stor = pd.DataFrame({"model":stor, "actual":TVA.loc[stor.index, "Storage"]})
    return stor

def calc_grouped_metric(data, grouper, act_var="actual", mod_var="model", metric="nse"):
    if metric == "nse":
        values = data.groupby(grouper).apply(lambda x: r2_score(x[act_var], x[mod_var]))
    else:
        values = data.groupby(grouper).apply(lambda x: mean_squared_error(x[act_var], x[mod_var], squared=False))
    return values

def get_release_means():
    return TVA.groupby(TVA.index.get_level_values(1))["Release"].mean()

def plot_score_bars(test_data):
    res_grouper = test_data.index.get_level_values(1)
    # resers = res_grouper.unique()
    resers = np.array([*TREE_RES, *SIMP_RES])
    fig, axes = plt.subplots(1, 2, sharey=True)
    axes = axes.flatten()

    nse_fc = calc_grouped_metric(
        test_data,
        res_grouper,
        mod_var="fc"
    )

    nse_pred = calc_grouped_metric(
        test_data,
        res_grouper,
        mod_var="preds"
    )

    release_means = get_release_means()

    rmse_fc = calc_grouped_metric(
        test_data,
        res_grouper,
        mod_var="fc",
        metric="rmse"
    ) / release_means * 100

    rmse_pred = calc_grouped_metric(
        test_data,
        res_grouper,
        mod_var="preds",
        metric="rmse"
    ) / release_means * 100

    ax1 = axes[0]
    # ax2 = ax1.twiny()

    width = 0.4
    y1 = [i - width/2 for i in range(resers.size)]
    y2 = [i + width/2 for i in range(resers.size)]

    ax1.barh(y1, nse_pred.loc[resers], height=width, color=colors[0], label="P")
    ax1.barh(y2, nse_fc.loc[resers], height=width, color=colors[1], label="S")
    ax1.set_yticks(range(resers.size))
    ax1.set_yticklabels(resers)
    ax1.set_xlabel("Release NSE")
    ax1.grid(visible=True, axis="x")

    ax1 = axes[1]
    # ax2 = ax1.twiny()

    ax1.barh(y1, rmse_pred.loc[resers], height=width, color=colors[0], label="Predicted")
    ax1.barh(y2, rmse_fc.loc[resers], height=width, color=colors[1], label="Simulated")
    ax1.set_xlabel("Release RMSE [%]")
    ax1.grid(visible=True, axis="x")
    ax1.legend(loc="best")

    plt.tight_layout()
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
        rdf = test_data.loc[idx[:,r],:]
        x = rdf.index.get_level_values(0)
        ax.plot(x, rdf["actual"], color=colors[0], label="Observed")
        ax.plot(x, rdf["preds"], color=colors[1], label="Predicted")
        ax.plot(x, rdf["fc"], color=colors[2], label="Simulated")
        ax.set_title(r)
        handles, labels = ax.get_legend_handles_labels()

    axes[-1].legend(handles, labels, loc="best")
    axes[-1].set_axis_off()

    plt.show()



if __name__ == "__main__":
    tree = load_tree_data()
    simple = load_simple_data()
    train_rel, test_rel = combine_tree_simple_release(tree, simple)
    sto_data = combine_tree_simple_storage(tree, simple)
    # plot_score_bars(test_rel)
    plot_time_series(test_rel)
