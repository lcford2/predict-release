# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
import calendar
import pathlib
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.helper_functions import read_tva_data
from plot_helpers import determine_grid_size
from simple_model_simul import (
    combine_columns,
    change_group_names,
    group_names,
    normalize_rmse,
    get_scores,
    find_res_groups,
    get_simulated_release,
)

sns.set_context("paper")
plt.style.use("ggplot")


TFSIZE = 16
LFSIZE = 14

def plot_coefs(coefs_old, coefs_train, coefs_test):
    fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
    axes = axes.flatten()
    coef_order = [
        "NaturalFlow-StorageDam",
        "ComboFlow-StorageDam",
        "ComboFlow-RunOfRiver",
    ]
    coef_map = {
        "NaturalFlow-StorageDam": "High RT",
        "ComboFlow-StorageDam": "Low RT",
        "ComboFlow-RunOfRiver": "Run of River",
    }

    for ax, coef in zip(axes, coef_order):
        label = coef_map[coef]
        width = 0.3
        xlabels = coefs_test.index
        x = [i for i in range(xlabels.size)]
        xleft = [i - width for i in x]
        xright = [i + width for i in x]
        ax.bar(xleft, coefs_old[coef], width=width, label="Old Coefs")
        ax.bar(x, coefs_train[coef], width=width, label="Train Coefs")
        ax.bar(xright, coefs_test[coef], width=width, label="Test Coefs")
        if ax == axes[0]:
            ax.legend(loc="best")
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=45, ha="right")
        ax.set_title(label)
    plt.show()


def load_results(norm=False):
    results_dir = pathlib.Path("../results/agu_2021_runs/")
    modif = "_norm" if norm else ""
    old_results = pd.read_pickle(
        results_dir / "simple_model_temporal_validation_all_res/results.pickle"
    )
    train_results = pd.read_pickle(
        results_dir / f"simul_model/train_sto_and_rel_results{modif}.pickle"
    )
    test_results = pd.read_pickle(
        results_dir / f"simul_model/test_sto_and_rel_results{modif}.pickle"
    )
    return old_results, train_results, test_results


def get_coefs(old, train, test):
    return (old["coefs"], train["new_coefs"], test["new_coefs"])


def add_interaction(df):
    df["Storage_Inflow_interaction"] = df.loc[:, "Storage_pre"].mul(
        df.loc[:, "Net Inflow"]
    )
    return df


def select_columns(df):
    df = df.loc[
        :,
        [
            "Release",
            "Storage",
            "Storage_pre",
            "Net Inflow",
            "Release_pre",
            "Storage_Inflow_interaction",
            "Release_roll7",
            "Storage_roll7",
            "Inflow_roll7",
        ],
    ]
    return df


def get_means_std(df):
    means = df.groupby(df.index.get_level_values(1)).mean()
    std = df.groupby(df.index.get_level_values(1)).std()
    return means, std


def change_groups(df, names, col, to_group):
    idx = pd.IndexSlice
    for name in names:
        df.loc[idx[:, name], col] = to_group
    return df


def add_month_dummies(df):
    month_arrays = {i: [0 for i in range(df.shape[0])] for i in calendar.month_abbr[1:]}
    for i, date in enumerate(df.index.get_level_values(0)):
        abbr = calendar.month_abbr[date.month]
        month_arrays[abbr][i] = 1

    for key, array in month_arrays.items():
        df[key] = array
    return df


def split_train_test(df, split_date):
    train_df = df.loc[df.index.get_level_values(0) < split_date]
    test_df = df.loc[df.index.get_level_values(0) >= split_date - timedelta(days=7)]
    return train_df, test_df


def prep_model_df(groups):
    df = TVA.copy()
    for_groups = df.loc[:, groups]
    df = add_interaction(df)
    df = select_columns(df)
    means, std = get_means_std(df)
    df[groups] = for_groups
    df = change_group_names(df, groups, group_names)
    df = change_groups(
        df, ["Douglas", "Cherokee", "Hiwassee"], "NaturalOnly", "NaturalFlow"
    )
    df = add_month_dummies(df)
    df = combine_columns(df, groups, "compositegroup")
    train_df, test_df = split_train_test(df, datetime(2010, 1, 1))
    return train_df, test_df, means, std


def plot_res_scores(actual, model, title, groups):
    scores = get_scores(actual, model)
    scores = normalize_rmse(scores, actual)
    scores["Group"] = groups

    scores = scores.sort_values(by=["Group", "NSE"])

    fig, axes = plt.subplots(2, 1, sharex=True)
    fig.patch.set_alpha(0.0)
    ax1, ax2 = axes.flatten()

    sns.barplot(
        x=scores.index, y=scores["NSE"], hue=scores["Group"], ax=ax1, dodge=False
    )
    ax1.set_title("NSE", fontsize=LFSIZE)
    ax1.set_ylabel("NSE", fontsize=LFSIZE)
    ax1.get_legend().remove()

    sns.barplot(
        x=scores.index, y=scores["nRMSE"], hue=scores["Group"], ax=ax2, dodge=False
    )
    ax2.set_title("nRMSE", fontsize=LFSIZE)
    ax2.set_ylabel("nRMSE [%]", fontsize=LFSIZE)
    ax2.set_xticks(range(scores.index.size))
    ax2.set_xticklabels(scores.index, rotation=45, ha="right")
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, loc="best", prop={"size": 14})

    fig.suptitle(title, fontsize=TFSIZE)
    fig.align_ylabels()
    plt.show()


def plot_time_series(actual, model, title, groups):
    res = actual.columns
    gs = determine_grid_size(res.size)
    fig, axes = plt.subplots(*gs, figsize=(19, 10))
    axes = axes.flatten()
    fig.patch.set_alpha(0.0)

    for res, ax in zip(res, axes):
        ax.plot(actual.index, actual[res], label="Actual")
        ax.plot(actual.index, model[res], label="Simul")
        ax.set_title(f"{res} [{groups[res]}]")
        handles, labels = ax.get_legend_handles_labels()

    axes[-1].set_axis_off()
    axes[-1].legend(handles, labels, loc="center", prop={"size": 16})

    fig.text(
        0.02, 0.5, f"{title}", va="center", ha="center", rotation=90, fontsize=TFSIZE
    )
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    TVA = read_tva_data(just_load=False)

    old_results, train_results, test_results = load_results(norm=False)
    old_coefs, train_coefs, test_coefs = get_coefs(
        old_results, train_results, test_results
    )
    old_coefs = old_coefs.loc[train_coefs.index]
    old_coefs = old_coefs.rename(index={"Storage_Inflow_interaction": "SxI"})
    train_coefs = train_coefs.rename(index={"Storage_Inflow_interaction": "SxI"})
    test_coefs = test_coefs.rename(index={"Storage_Inflow_interaction": "SxI"})

    train, test, means, std = prep_model_df(["NaturalOnly", "RunOfRiver"])

    res_groups = find_res_groups(train)
    group_map = {
        "ComboFlow-RunOfRiver": "ROR",
        "ComboFlow-StorageDam": "HRT",
        "NaturalFlow-StorageDam": "LRT",
    }

    train_rel, train_sto = get_simulated_release(
        train, train_coefs, res_groups, means, std
    )
    test_rel, test_sto = get_simulated_release(test, test_coefs, res_groups, means, std)

    test_train_rel, test_train_sto = get_simulated_release(
        test, train_coefs, res_groups, means, std
    )

    act_train_rel = train["Release"].unstack().loc[train_rel.index]
    act_train_sto = train["Storage"].unstack().loc[train_rel.index]

    act_test_rel = test["Release"].unstack().loc[test_rel.index]
    act_test_sto = test["Storage"].unstack().loc[test_rel.index]

    train_output = pd.DataFrame({
        "Release_act":act_test_rel.stack(),
        "Storage_act":act_test_sto.stack(),
        "Release_simul":test_train_rel.stack(),
        "Storage_simul":test_train_sto.stack()
    })
    from IPython import embed as II
    II()
    sys.exit()

    # train_rel_scores = get_scores(act_train_rel, train_rel)
    # train_sto_scores = get_scores(act_train_sto, train_sto)
    # test_rel_scores = get_scores(act_test_rel, test_rel)
    # test_sto_scores = get_scores(act_test_sto, test_sto)
    # test_train_rel_scores = get_scores(act_test_rel, test_train_rel)
    # test_train_sto_scores = get_scores(act_test_sto, test_train_sto)

    # plot_coefs(old_coefs, train_coefs, test_coefs)
    # plot_res_scores(act_test_rel, test_train_rel,
    #                 "Simulated Release Performance", res_groups)
    # plot_res_scores(act_test_sto, test_train_sto,
    #                 "Simulated Storage Performance", res_groups)

    # res_groups = res_groups.apply(group_map.get)
    # plot_time_series(act_test_rel, test_train_rel, "Release", res_groups)
    # plot_time_series(act_test_sto, test_train_sto, "Storage", res_groups)
