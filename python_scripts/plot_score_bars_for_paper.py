import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from IPython import embed as II

def load_tree_data():
    return pd.read_pickle("../results/synthesis/treed_model/upstream_basic_td3_roll7_simple_tree_month_coefs/results.pickle")

def load_simp_data():
    return pd.read_pickle("../results/synthesis/simple_model/all_res_time_fit/NaturalOnly-RunOfRiver_filter_ComboFlow_SIx_pre_std_swapped_res_roll7_wrel.pickle")

def combine_data(tree, simp):
    tree_data = pd.DataFrame(
        {
            "model":tree["data"]["fitted_rel"].stack(),
            "actual":tree["data"]["y_train_rel_act"],
        }
    )
    simp_data = simp["data"]["y_results"].drop("bin", axis=1)
    return tree_data.append(simp_data).sort_index()

def calculate_metrics(data):
    metrics = pd.DataFrame(
        columns=["NSE", "RMSE"],
        index=data.index.get_level_values(1).unique() # reservoirs
    )
    idx = pd.IndexSlice
    for res in metrics.index:
        res_data = data.loc[idx[:,res],:]
        nse = r2_score(res_data["actual"], res_data["model"])
        rmse = np.sqrt(mean_squared_error(res_data["actual"], res_data["model"]))
        rmse = rmse / res_data["actual"].mean() * 100
        metrics.loc[res] = [nse, rmse]
    return metrics

def add_groups(metrics):
    groups = pd.read_pickle("../pickles/tva_groups.pickle")    
    idx = pd.IndexSlice
    groups = groups.loc[idx[groups.index[-1][0],:]]
    groups.index = groups.index.get_level_values(1)
    metrics["Group"] = groups
    return metrics

def change_names(metrics):
    import csv
    with open("actual_names.csv", "r") as f:
        reader = csv.reader(f)
        act_names = {i:j for i,j in reader} 
    return metrics.rename(index=act_names)

def plot_metrics(metrics, orient="v"):
    # axes = metrics.sort_index(ascending=False).plot.barh(
    #     subplots=True, layout=(1,2), sharex=False, sharey=True,
    #     color="Group", width=0.6
    # )
    # print(metrics)
    metrics["Reservoir"] = metrics.index
    metrics = metrics.melt(id_vars=["Reservoir", "Group"], var_name="Metric")
    metrics = metrics.sort_values(by=["Group","value"])
    if orient == "v":
        args = {"x":"value", "y":"Reservoir","sharex":False, "sharey":True}
    else:
        args = {"y":"value", "x":"Reservoir", "sharex":True, "sharey":False,
                "col_wrap":1}
    fg = sns.catplot(data=metrics, col="Metric", **args, palette="mako",
                     hue="Group", kind="bar", dodge=False)
    axes = fg.axes.flatten()
    for ax in axes:
        # ax.get_legend().remove()
        ax.set_title("")

    if orient == "v":
        axes[0].set_xlabel("NSE")
        axes[1].set_xlabel(r"RMSE [% of $ \bar{D}_r $]")
    else:
        axes[0].set_ylabel("NSE")
        axes[1].set_ylabel(r"RMSE [% of $ \bar{D}_r $]")
        axes[1].set_xlabel("")
        fg.set_xticklabels(rotation=45, ha="right")
    fg.legend.remove()
    axes[1].legend(loc="best", title=fg._hue_var)
    # plt.tight_layout()
    fg.tight_layout()
    plt.show()


if __name__ == "__main__":
    plt.style.use("ggplot")
    sns.set_context("talk")
    style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    tree = load_tree_data()
    simp = load_simp_data()
    combined = combine_data(tree, simp)

    metrics = calculate_metrics(combined)
    metrics = add_groups(metrics)
    metrics = change_names(metrics)

    import sys
    orient = sys.argv[1] if len(sys.argv) > 1 else "v"
    plot_metrics(metrics, orient)
    