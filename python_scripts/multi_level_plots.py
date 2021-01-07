import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from sklearn.metrics import r2_score
from IPython import embed as II
import pathlib
import glob
import argparse

plt.style.use("ggplot")
sns.set_context("talk")

results_dir = pathlib.Path("../results")
multi_level_dir = results_dir / "multi-level-results"

def load_results(args):
    file = args.results_pickle
    if not file:
        file = query_file(args)
    with open(multi_level_dir/file, "rb") as f:
        data = pickle.load(f)
    return data

def parse_args(plot_functions):
    parser = argparse.ArgumentParser(
        description="Plot results for multi-level models."
    )
    parser.add_argument("-f", "--file", dest="results_pickle", default=None,
                        help="File name for results pickle. Expected to be in ../results/mixed-level-results"
                        )
    parser.add_argument("-p", "--plot_func", choices=plot_functions.keys(), default=None, dest="plot_func",
                        help="Provide the name of the desired plot. If none is provided an IPython interpreter will be opened.")
    parser.add_argument("-g", "--group_labels", action="store_true", dest="group_labels",
                        help="Flag to include group labels on plots. Can act differently on different plots.")
    parser.add_argument("--res_names", dest="res_names", nargs="+", default=None, 
                        help="Specify reservoir names to only plot those reservoirs for certian plots.")
    parser.add_argument("-F", "--forecasted", dest="forecasted", action="store_true", 
                        help="Use forecasted data instead of simple prediction.")
    parser.add_argument("-A", "--all_res", dest="all_res", action="store_true",
                        help="Look at results for all reservoirs, not just ones used to fit model.")
    args = parser.parse_args()
    return args

def query_file(args):
    if args.all_res:
        files = glob.glob((multi_level_dir/"*all_res.pickle").as_posix())
    else:
        files = [i for i in glob.glob((multi_level_dir/"*.pickle").as_posix()) if "all_res" not in i]
    for i, file in enumerate(files):
        print(f"[{i}] {file}")
    selection = input("Enter the number of the file you wish to use: ")
    if not selection.isnumeric():
        raise ValueError("Must provide an integer corresponding to your selection.")
    elif int(selection) > len(files):
        raise ValueError("Number provided is not valid.")
    else:
        file = pathlib.Path(files[int(selection)])
        return file.name

def find_plot_functions(namespace):
    plot_functions = filter(lambda x: x[:4] == "plot", namespace)
    plot_name_dict = {"_".join(i.split("_")[1:]):i for i in plot_functions}
    return plot_name_dict

def plot_reservoir_score(data, args):
    if args.forecasted:
        if args.all_res:
            modeled = data["Release_act"].unstack()
            actual = data["Release_act_obs"].unstack()
            groupnames = data["compositegroup"].unstack().values[0,:]
        else:
            modeled = data["data"]["forecasted"]["Release_act"].unstack()
            actual = data["data"]["y_test_act"].unstack()
            groupnames = data["data"]["X_test"]["compositegroup"].unstack().values[0, :]
    else:
        modeled = data["data"]["predicted_act"].unstack()
        actual = data["data"]["y_test_act"].unstack()
        groupnames = data["data"]["X_test"]["compositegroup"].unstack().values[0, :]

    scores = pd.DataFrame(index=modeled.columns, columns=["Score", "GroupName"], dtype="float64")
    for i, index in enumerate(scores.index):
        score = r2_score(actual[index], modeled[index])
        scores.loc[index, "Score"] = score
        scores.loc[index, "GroupName"] = groupnames[i]
    scores = scores.sort_values(by=["GroupName", "Score"])
    groupmeans = scores.groupby("GroupName").mean()
    fig, ax = plt.subplots(1,1)
    scores["Score"].plot.bar(ax=ax, width=0.8)
    ticks = ax.get_xticks()
    if getattr(args, "group_labels", None):
        for tick, name in zip(ticks, scores["GroupName"].values.tolist()):
            ax.text(tick, 0.1, name, rotation=90, va="bottom", ha="center")
    else:
        groupmeans = scores.groupby("GroupName").mean()
        groups = groupmeans.index.tolist()
        groupmeans = groupmeans.values.flatten()
        groupcount = scores.groupby("GroupName").count().cumsum().values.flatten() / modeled.shape[1]
        last = 0
        modif = 0.4 / modeled.shape[1]
        for i, mean in enumerate(groupmeans):
            ax.axhline(mean, last + modif, groupcount[i]+modif, c="b")
            group = groups[i]
            ax.text((last+modif + groupcount[i])*modeled.shape[1]/2, mean, group, ha="center", va="bottom")
            last = groupcount[i]
    ax.set_ylabel("NSE")
    plt.subplots_adjust(
        top=0.88,
        bottom=0.205,
        left=0.11,
        right=0.9,
        hspace=0.2,
        wspace=0.2
    )
    plt.show()


def determine_grid_size(N):
    if N <= 3:
        return (N,1)
    else:
        poss_1 = [(i, N//i) for i in range(2, int(N**0.5) + 1) if N % i == 0]
        poss_2 = [(i, (N+1) // i) for i in range(2, int((N+1)**0.5) + 1) if (N+1) % i == 0]
        poss = poss_1 + poss_2
        min_index = np.argmin([sum(i) for i in poss])
        return poss[min_index]

def plot_reservoir_TS(data, args):
    if args.forecasted:
        if args.all_res:
            modeled = data["Release_act"].unstack()
            actual = data["Release_act_obs"].unstack()
            groupnames = data["compositegroup"].unstack().values[0, :]
        else:
            modeled = data["data"]["forecasted"]["Release_act"].unstack()
            actual = data["data"]["y_test_act"].unstack()
            groupnames = data["data"]["X_test"]["compositegroup"].unstack(
            ).values[0, :]
    else:
        modeled = data["data"]["predicted_act"].unstack()
        actual = data["data"]["y_test_act"].unstack()
        groupnames = data["data"]["X_test"]["compositegroup"].unstack(
        ).values[0, :]

    if args.res_names:
        res_names = np.array(args.res_names)
    else:
        res_names = modeled.columns

    grid_size = determine_grid_size(res_names.size)
    if res_names.size > 4:
        sns.set_context("paper")
    fig, axes = plt.subplots(*grid_size)
    axes = axes.flatten()
    for ax, col in zip(axes, res_names):
        actual[col].plot(ax=ax)
        modeled[col].plot(ax=ax)
        ax.set_title(col)
    
    if res_names.size <= 6:
        handles, labels = axes[0].get_legend_handles_labels()
        labels = ["Observed", "Modeled"]
        axes[0].legend(handles, labels, loc="best")
    left_over = axes.size - res_names.size
    if left_over > 0:
        for ax in axes[-left_over:]:
            ax.set_axis_off()
    # plt.subplots_adjust(

    # )
    plt.show()

if __name__ == "__main__":
    namespace = dir()
    plot_functions = find_plot_functions(namespace)
    args = parse_args(plot_functions=plot_functions)
    data = load_results(args)
    if args.plot_func:
        globals()[plot_functions[args.plot_func]](data, args)
    else:
        II()
