import pandas as pd
import glob
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

from IPython import embed as II


def load_results():
    files = glob.glob("../results/tclr_model/all/TD4_*_RT_MS_Nelder-Mead")
    # files.append("../results/tclr_model/all/TD4_RT_MS_Nelder-Mead")
    results = {}
    for file in files:
        name = file.split("_")[2]
        results[name] = pd.read_pickle(f"{file}/results.pickle")["simmed_data"]
    results["never"] = pd.read_pickle(
        "../results/tclr_model/all/TD4_RT_MS_Nelder-Mead/results.pickle"
    )["simmed_data"]
    return results


def merge_results(results):
    output = pd.DataFrame()
    for name, df in results.items():
        df["assim"] = name
        output = pd.concat([output, df])
    return output


if __name__ == "__main__":
    plt.style.use("ggplot")
    sns.set_context("talk")
    results = load_results()
    results = merge_results(results)

    rmse = results.groupby(["assim", "site_name"]).apply(
        lambda x: mean_squared_error(x["actual"], x["model"], squared=False))
    nse = results.groupby(["assim", "site_name"]).apply(
        lambda x: r2_score(x["actual"], x["model"]))

    nse = nse.reset_index().rename(columns={0: "NSE"})
    rmse = rmse.reset_index().rename(columns={0: "RMSE"})

    order=[
        "never",
        "yearly",
        "semi-annually",
        "seasonally",
        "monthly",
        "weekly",
        "daily"
    ]
    nse["assim"] = nse["assim"].astype("category")
    rmse["assim"] = rmse["assim"].astype("category")
    nse["assim"] = nse["assim"].cat.set_categories(order)
    rmse["assim"] = rmse["assim"].cat.set_categories(order)
    nse = nse.sort_values("assim")
    rmse = rmse.sort_values("assim")

    # ax = nse.plot.box(by="assim", whis=(0.05, 0.95))
    # ax = ax[0]

    fg = sns.catplot(
        data=nse,
        x="assim",
        y="NSE",
        kind="box",
        showfliers=False,
    )
    sns.despine()
    ax = fg.ax
    ax.set_xlabel("Release and Storage Assimilation Frequency")
    ax.set_xticklabels([i.get_text().capitalize() for i in ax.get_xticklabels()])
    ax.set_title("")
    ax.set_ylabel("NSE")
    # II()
    plt.show()
