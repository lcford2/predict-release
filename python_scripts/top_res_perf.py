import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
import argparse

plt.style.use("ggplot")
sns.set_context("talk")


def load_mss_results():
    msses = [
        "0.00", "0.01", "0.02", "0.03", "0.04",
        "0.05", "0.06", "0.07", "0.08", "0.09",
        # "0.10", "0.11", "0.15", "0.20"
        "0.10", "0.15", "0.20"
    ]
    files = [
        f"../results/tclr_model_testing/all/TD4_MSS{i}_RT_MS_exhaustive_new_hoover/results.pickle" 
        for i in msses
    ]
    results = {i: pd.read_pickle(f) for i, f in zip(msses, files)}
    return results


def select_res_results(res, results):
    output = {}
    for mss, df in results.items():
        mdf = df["simmed_data"]
        rdf = mdf[mdf.index.get_level_values(0).isin(res)]
        output[mss] = rdf
    return output


def calc_res_score(res_results, metric="NSE"):
    if metric == "NSE":
        metric_func = r2_score
    else:
        metric_func = lambda x, y: mean_squared_error(x, y, squared=False)

    return {
        mss: df.groupby("site_name").apply(lambda x: metric_func(x["actual"], x["model"])) for mss, df in res_results.items()
    }


def plot_score_boxes(res_scores, title, score_metric):
    scores = pd.DataFrame.from_dict(res_scores).reset_index().melt(
        id_vars="site_name", value_name="score", var_name="mss"
    )
    fg = sns.catplot(
        data=scores,
        x="mss",
        y="score",
        kind="box",
        legend_out=False,
        height=10,
        aspect=19/10,
    )
    ax = fg.ax
    ax.set_title(title)
    ax.set_xlabel("MSS")
    ax.set_ylabel(score_metric)
    plt.show()  


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "top_res",
        choices=["release_cv", "storage_cv", "release", "storage"],
        help="Define what reservoirs to show the plot for."
    )
    parser.add_argument(
        "metric",
        choices=["NSE", "RMSE"],
        help="What metric to plot."
    )
    return parser.parse_args()


if __name__ == "__main__":
    title_map = {
        "release_cv": r"Top 20 Release $CV$ Reservoirs",
        "storage_cv": r"Top 20 Storage $CV$ Reservoirs",
        "release": r"Top 20 Release Reservoirs",
        "storage": r"Top 20 Storage Reservoirs"
    }
    args = parse_args()
    top_res = pd.read_pickle("../pickles/top_resers.pickle")
    results = load_mss_results()
    res_results = select_res_results(top_res[args.top_res], results)
    res_scores = calc_res_score(res_results, metric=args.metric)
    plot_score_boxes(res_scores, title_map[args.top_res], args.metric)
