from functools import partial

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from IPython import embed as II

def load_plrt_results(depth=2):
    path = f"../results/tclr_model/tva/TD{depth}_RT_MS/results.pickle"
    return pd.read_pickle(path)

def load_lr_results():
    path = f"../results/tclr_model/tva/TD0_RT_MS/results.pickle"
    return pd.read_pickle(path)

def load_simple_results():
    path = "../results/synthesis/simple_model/no_ints_model/results.pickle"
    return pd.read_pickle(path)

def load_tree_results():
    path = "../results/synthesis/treed_model/TD3_tree_all_res/results.pickle"
    return pd.read_pickle(path)

def load_async_model(just_tree=True):
    if just_tree:
        tree = load_tree_results()
        return {"tree": tree}
    else:
        simp = load_simple_results()
        tree = load_tree_results()
        return {"simp": simp, "tree": tree}

def calc_res_scores(df, metric="NSE"):
    if metric == "NSE":
        mfunc = r2_score
    elif metric == "RMSE":
        mfunc = partial(mean_squared_error, squared=False)
        # def mfunc(act, mod):
        #     return mean_squared_error(act, mod, squared=False)
    else:
        raise ValueError(f"Metric {metric} not implemented")

    return pd.DataFrame({metric: df.groupby(df.index.get_level_values(1)).apply(
        lambda x: mfunc(x["actual"], x["model"])
    )})

def calc_async_scores(async_data):
    data_names = {
        "tree": {
            "train": "train_data",
            "test": "test_p_data",
            "simmed": "test_f_data"
        },
        "simp": {
            "train": "y_results",
            "test": "y_pred",
            "simmed": "y_forc"
        }
    }
    scores = []
    for key, value in async_data.items():
        train_data = value["data"][data_names[key]["train"]]
        train_nse = calc_res_scores(train_data, metric="NSE")
        train_nse["group"] = key
        train_nse["mset"] = "train"
        train_nse = train_nse.reset_index().rename(
            columns={"index": "reservoir"}
        )

        test_data = value["data"][data_names[key]["test"]]
        test_nse = calc_res_scores(test_data, metric="NSE")
        test_nse["group"] = key
        test_nse["mset"] = "test"
        test_nse = test_nse.reset_index().rename(
            columns={"index": "reservoir"}
        )

        simmed_data = value["data"][data_names[key]["simmed"]]
        simmed_nse = calc_res_scores(simmed_data, metric="NSE")
        simmed_nse["group"] = key
        simmed_nse["mset"] = "simmed"
        simmed_nse = simmed_nse.reset_index().rename(
            columns={"index": "reservoir"}
        )

        scores.append(train_nse)
        scores.append(test_nse)
        scores.append(simmed_nse)

    return pd.concat(scores)

def prep_plrt_or_lr_data(data):
   train = data["f_res_scores"]
   test = data["p_res_scores"]
   simmed = data["s_res_scores"]

   train["mset"] = "train"
   test["mset"] = "test"
   simmed["mset"] = "simmed"

   scores = pd.concat([train, test, simmed])
   scores = scores.reset_index().rename(
       columns={"index": "reservoir"}
   )
   return scores


def plot_score_bars(lr_scores, async_scores, plrt_scores, mset="simmed", metric="NSE"):
    lr_df = lr_scores.loc[
        lr_scores["mset"] == mset,
        ["reservoir", metric]
    ]
    lr_df["model"] = "LR"
    async_df = async_scores.loc[
        async_scores["mset"] == mset,
        ["reservoir", metric]
    ]
    async_df["model"] = "ASYNC"
    plrt_df = plrt_scores.loc[
        plrt_scores["mset"] == mset,
        ["reservoir", metric]
    ]
    plrt_df["model"] = "PLRT"

    pdf = pd.concat([lr_df, async_df, plrt_df])
    pdf = pdf.melt(id_vars=["reservoir", "model"])
    pdf = pdf.sort_values(by=["model", "value"])
    fg = sns.catplot(
        data=pdf,
        x="reservoir",
        hue="model",
        y="value",
        kind="bar",
        legend_out=False,
        hue_order=["LR", "ASYNC", "PLRT"]
    )
    fg.ax.set_xticklabels(fg.ax.get_xticklabels(), rotation=45, ha="right")
    fg.ax.set_ylabel(metric)
    fg.ax.set_xlabel("")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


    fg.ax.axhline(lr_df[metric].mean(), c=colors[0])
    fg.ax.axhline(async_df[metric].mean(), c=colors[1])
    fg.ax.axhline(plrt_df[metric].mean(), c=colors[2])

    plt.show()


def main():
    # load and prep async data
    async_data = load_async_model()
    async_scores = calc_async_scores(async_data)

    # load and prep plrt data
    plrt_data = load_plrt_results(depth=2)
    plrt_scores = prep_plrt_or_lr_data(plrt_data)

    # load and prep lr data
    lr_data = load_lr_results()
    lr_scores = prep_plrt_or_lr_data(lr_data)

    plot_score_bars(lr_scores, async_scores, plrt_scores, mset="simmed", metric="NSE")


if __name__ == "__main__":
    plt.style.use("ggplot")
    text_color = "black"
    mpl.rcParams["text.color"] = text_color
    mpl.rcParams["axes.labelcolor"] = text_color
    mpl.rcParams["xtick.color"] = text_color
    mpl.rcParams["ytick.color"] = text_color

    sns.set_context("talk")
    main()
