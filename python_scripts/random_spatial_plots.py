import pathlib
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import embed as II


def load_data():
    path = "../results/tclr_spatial_eval/all/"
    dirfmt = "TD3_RT_MS_0.75_{seed}"
    train, test, simmed = {}, {}, {}
    for seed in range(1000):
        file = pathlib.Path(path) / dirfmt.format(seed=seed) / "results.pickle"
        results = pd.read_pickle(file)
        train[seed] = results["f_res_scores"].dropna()
        test[seed] = results["p_res_scores"].dropna()
        simmed[seed] = results["s_res_scores"].dropna()
    return train, test, simmed


def plot_overall_boxes(train, test, simmed, metric="NSE"):
    train_values, test_values, simmed_values = [], [], []
    for seed in range(1000):
        train_values.extend(train[seed][metric].values)
        test_values.extend(test[seed][metric].values)
        simmed_values.extend(simmed[seed][metric].values)
    train_df = pd.DataFrame({metric: train_values})
    train_df["Reservoir Set"] = "Training"
    test_df = pd.DataFrame({metric: test_values})
    test_df["Reservoir Set"] = "Testing"
    simmed_df = pd.DataFrame({metric: simmed_values})
    simmed_df["Reservoir Set"] = "Simmed"

    df = pd.concat([train_df, test_df, simmed_df])
    fg = sns.catplot(data=df, x="Reservoir Set", y=metric, kind="box")
    fg.figure.patch.set_alpha(0.0)
    plt.show()


def plot_res_perf(data, title, metric="NSE"):
    basins = pd.read_pickle("../pickles/res_basin_map.pickle")
    df = pd.concat([data[i][[metric]] for i in range(1000)])
    df["basin"] = basins
    df = df.reset_index()
    df = df.sort_values(by=["basin", metric])
    sns.set_context("paper")
    fg = sns.catplot(
        data=df,
        y=metric,
        x="site_name",
        kind="box",
        hue="basin",
        dodge=False,
        legend_out=False,
        ci=None,
    )
    fg.figure.suptitle(title)
    fg.set_xticklabels(rotation=45, ha="right")
    fg.figure.patch.set_alpha(0.0)
    plt.show()


def main():
    plt.style.use("ggplot")
    text_color = "black"
    mpl.rcParams["text.color"] = text_color
    mpl.rcParams["axes.labelcolor"] = text_color
    mpl.rcParams["xtick.color"] = text_color
    mpl.rcParams["ytick.color"] = text_color

    sns.set_context("talk")
    train, test, simmed = load_data()
    plot_overall_boxes(train, test, simmed)
    # plot_res_perf(simmed, title="Simulation")


if __name__ == "__main__":
    main()
