import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from IPython import embed as II
import sys

MAX_DEPTH_LOAD = 7


def load_all_data(get_value=None, postpend=None, treevars=False):
    base_path = "../results/basin_eval/all/tclr_model_no_ints_all_res_"

    data = {}
    for base_id in range(0, MAX_DEPTH_LOAD):
        d_id = f"D{base_id}"
        d_path = f"{base_path}{base_id}"
        if postpend:
            d_id += f"_{postpend}"
            d_path += f"_{postpend}"
        if treevars:
            d_path += "_RT_MS"
        res = pd.read_pickle(f"{d_path}/results.pickle")
        if get_value:
            if isinstance(get_value, list):
                data_val = res.get(get_value[0])
                for val in get_value[1:]:
                    data_val = data_val.get(val)
                data[d_id] = data_val
            else:
                data[d_id] = res.get(get_value)
        else:
            data[d_id] = res["data"]["test_data"]

    return data


def load_spatial_val_data(
    depth, get_value=None, postpend=None, treevars=False, basin_split=True
):
    base_path = "../results/basin_eval/all/tclr_model_spatial_no_ints_all_res_"

    data = {}
    d_id = f"D{depth}"
    d_path = f"{base_path}{depth}"
    if postpend:
        d_id += f"_{postpend}"
        d_path += f"_{postpend}"
    if treevars:
        d_path += "_RT_MS"
    train_props = [f"{i/100:.2f}" for i in range(25, 91, 5)]
    for tp in train_props:
        if basin_split:
            c_path = "_".join([d_path, tp, "all-basin-split"])
        else:
            c_path = "_".join([d_path, tp])
        res = pd.read_pickle(f"{c_path}/results.pickle")
        if get_value:
            if isinstance(get_value, list):
                data_val = res.get(get_value[0])
                for val in get_value[1:]:
                    data_val = data_val.get(val)
                data[tp] = data_val
            else:
                data[tp] = res.get(get_value)
        else:
            data[tp] = res

    return data


def calc_scores(data):
    scores = {}
    for d_id, test_data in data.items():
        res_grouper = test_data.index.get_level_values(0)
        d_scores = pd.DataFrame(index=res_grouper.unique(), columns=["NSE", "RMSE"])
        d_scores["NSE"] = test_data.groupby(res_grouper).apply(
            lambda x: r2_score(x["actual"], x["model"])
        )
        d_scores["RMSE"] = test_data.groupby(res_grouper).apply(
            lambda x: mean_squared_error(x["actual"], x["model"], squared=False)
        )
        scores[d_id] = d_scores
    return scores


def plot_overall_assim_scores(treevars=False):
    postpends = [None, "daily", "weekly", "monthly", "seasonally"]
    data = [
        load_all_data(get_value="s_act_score", postpend=i, treevars=treevars)
        for i in postpends
    ]
    records = []
    columns = [f"D{i}" for i in range(0, MAX_DEPTH_LOAD)]
    for i, pdata in enumerate(data):
        subrecords = []
        if i == 0:
            for column in columns:
                subrecords.append(pdata[column])
        else:
            for column in columns:
                subrecords.append(pdata["_".join([column, postpends[i]])])
        records.append(subrecords)

    index = ["Never"] + [i.capitalize() for i in postpends[1:]]
    df = pd.DataFrame.from_records(records, index=index, columns=columns)
    df = df.loc[["Never", "Seasonally", "Monthly", "Weekly", "Daily"], :]

    ax = df.T.plot.bar(ylabel="Overall NSE", rot=0)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, ncol=3, loc="upper left")

    plt.show()


def plot_res_assim_scores_box(treevars=False):
    postpends = [None, "daily", "weekly", "monthly", "seasonally"]
    data = [
        load_all_data(get_value="simmed_res_scores", postpend=i, treevars=treevars)
        for i in postpends
    ]
    records = []
    columns = [f"D{i}" for i in range(0, MAX_DEPTH_LOAD)]
    for i, pdata in enumerate(data):
        if i == 0:
            assim = "Never"
        else:
            assim = postpends[i].capitalize()
        for column in columns:
            if i != 0:
                column = f"{column}_{postpends[i]}"
            df = pdata[column]["NSE"]
            df = df.reset_index().melt(id_vars=["site_name"])
            df["assim"] = assim
            df["depth"] = int(column.split("_")[0][1:])
            records.append(df)

    df = pd.concat(records).reset_index()
    assim_map = {"Never": 0, "Seasonally": 1, "Monthly": 2, "Weekly": 3, "Daily": 4}
    df["assim_map"] = df["assim"].apply(assim_map.get)
    df = df.sort_values(by=["assim_map", "depth"])
    fg = sns.catplot(
        data=df, x="depth", y="value", hue="assim", kind="box", legend_out=False
    )
    ax = fg.ax
    ax.set_ylabel("NSE")
    ax.set_xlabel("Tree Depth")
    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    ax.legend(handles, labels, ncol=3, loc="lower right")
    fg.figure.patch.set_alpha(0.0)
    plt.show()


def plot_res_assim_scores_box_thirds(treevars=False):
    postpends = [None, "weekly", "monthly", "seasonally"]
    data = [
        load_all_data(get_value=["data", "simmed_data"], postpend=i, treevars=treevars)
        for i in postpends
    ]

    columns = [f"D{i}" for i in range(0, MAX_DEPTH_LOAD)]
    output = pd.DataFrame()
    for i, pdata in enumerate(data):
        if i == 0:
            assim = "Never"
        else:
            assim = postpends[i].capitalize()
        for column in columns:
            if i != 0:
                column = f"{column}_{postpends[i]}"
            df = pdata[column]
            df = df.rename(columns={"model": column})
            if output.empty:
                output = df
            else:
                output[column] = df[column]

    resers = output.index.get_level_values(0).unique()
    thirds_index = pd.MultiIndex.from_product([resers, [1, 2, 3]])
    scores = pd.DataFrame(index=thirds_index, columns=output.columns[1:])

    # II()
    # sys.exit()
    idx = pd.IndexSlice
    for res in resers:
        rdf = output.loc[idx[res, :], :].copy()
        try:
            bins = pd.qcut(rdf["actual"], 3, labels=False)
        except ValueError:
            print(f"Not enough unique values for {res}")
            continue
        rdf["bin"] = bins
        for col in scores.columns:
            try:
                cscore = rdf.groupby("bin").apply(lambda x: r2_score(x["actual"], x[col]))
                scores.loc[idx[res, [1, 2, 3]], col] = cscore.values
            except ValueError:
                print(f"NaN values for {res} and {col}")

    scores = (
        scores.reset_index()
        .rename(columns={"level_1": "bin"})
        .melt(id_vars=["site_name", "bin"])
    )
    scores[["depth", "assim"]] = scores["variable"].str.split("_", expand=True)
    scores["assim"] = scores["assim"].fillna("never")

    sns.catplot(
        data=scores,
        x="depth",
        y="value",
        hue="assim",
        row="bin",
        kind="box",
        sharey=False,
    )
    plt.show()


def plot_assim_scores_box_thirds(treevars=False):
    postpends = [None, "daily", "weekly", "monthly", "seasonally"]
    data = [
        load_all_data(get_value=["data", "simmed_data"], postpend=i, treevars=treevars)
        for i in postpends
    ]

    columns = [f"D{i}" for i in range(0, MAX_DEPTH_LOAD)]
    output = pd.DataFrame()
    for i, pdata in enumerate(data):
        for column in columns:
            if i != 0:
                column = f"{column}_{postpends[i]}"
            df = pdata[column]
            df = df.rename(columns={"model": column})
            if output.empty:
                output = df
            else:
                output[column] = df[column]

    scores = pd.DataFrame(index=[1, 2, 3], columns=output.columns[1:])

    ranks = output.groupby(output.index.get_level_values(0))["actual"].rank(pct=True)

    def make_bins(x):
        if x < 1 / 3:
            return 1
        elif x < 2 / 3:
            return 2
        else:
            return 3

    bins = ranks.apply(make_bins)

    # bins = pd.qcut(output["actual"], 3, labels=False)

    output["bin"] = bins
    for col in scores.columns:
        try:
            cscore = output.groupby("bin").apply(lambda x: r2_score(x["actual"], x[col]))
            scores[col] = cscore.values
        except ValueError:
            print(f"NaN values for {col}")

    scores = scores.reset_index().rename(columns={"index": "bin"}).melt(id_vars=["bin"])
    scores[["depth", "assim"]] = scores["variable"].str.split("_", expand=True)
    scores["assim"] = scores["assim"].fillna("never")
    scores["assim"] = scores["assim"].str.capitalize()
    assim_map = {"Never": 0, "Seasonally": 1, "Monthly": 2, "Weekly": 3, "Daily": 4}
    scores["assim_map"] = scores["assim"].apply(assim_map.get)
    scores = scores.sort_values(by=["assim_map", "depth"])

    fg = sns.catplot(
        data=scores,
        x="depth",
        y="value",
        hue="assim",
        row="bin",
        kind="bar",
        sharey=False,
        legend_out=False,
    )
    fg.set_ylabels("NSE")
    ax = fg.axes.flatten()[0]
    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    ax.legend(handles, labels, ncol=3, loc="lower right")
    fg.figure.patch.set_alpha(0.0)

    plt.show()


def plot_spatial_val_scores(depth, basin_split=False):
    data = load_spatial_val_data(depth, treevars=True, basin_split=basin_split)
    f_scores = {i: j["f_act_score"] for i, j in data.items()}
    p_scores = {i: j["p_act_score"] for i, j in data.items()}
    s_scores = {i: j["s_act_score"] for i, j in data.items()}
    df = pd.DataFrame(
        {"Fitted NSE": f_scores, "Predicted NSE": p_scores, "Simulated NSE": s_scores}
    )
    title = "Basin Specific Split" if basin_split else "Overall Split"
    fig, ax = plt.subplots(1, 1)
    fig.patch.set_alpha(0.0)
    df.plot.bar(
        width=0.8,
        title=title,
        rot=0,
        ax=ax
    )
    ax.set_ylabel("NSE")
    ax.set_xlabel("Training Proportion")
    ax.legend(loc="lower left")
    plt.show()


def plot_spatial_val_scores_boxes(depth, basin_split=False):
    data = load_spatial_val_data(depth, treevars=True, basin_split=basin_split)
    f_scores = {i: j["train_res_scores"].dropna() for i, j in data.items()}
    p_scores = {i: j["test_res_scores"].dropna() for i, j in data.items()}
    s_scores = {i: j["simmed_res_scores"].dropna() for i, j in data.items()}

    score_values = []

    METRIC = "NSE"
    for key in f_scores.keys():
        df = f_scores[key][METRIC].reset_index()
        df["tp"] = key
        df["mset"] = "Fitted"
        score_values.append(df)

        df = p_scores[key][METRIC].reset_index()
        df["tp"] = key
        df["mset"] = "Predicted"
        score_values.append(df)

        df = s_scores[key][METRIC].reset_index()
        df["tp"] = key
        df["mset"] = "Simulated"
        score_values.append(df)

    scores = pd.concat(score_values)
    fg = sns.catplot(
        data=scores,
        x="tp",
        y="NSE",
        hue="mset",
        kind="box",
        legend_out=False
    )
    fg.figure.patch.set_alpha(0.0)
    ax = fg.ax
    # ax.set_ylabel("NSE")
    ax.set_xlabel("Train Proportion")
    ax.legend(loc="lower right", title="")
    ax.set_title("Basin Specific Splits" if basin_split else "Overall Splits")
    plt.show()


if __name__ == "__main__":
    plt.style.use("ggplot")
    text_color = "black"
    mpl.rcParams["text.color"] = text_color
    mpl.rcParams["axes.labelcolor"] = text_color
    mpl.rcParams["xtick.color"] = text_color
    mpl.rcParams["ytick.color"] = text_color

    sns.set_context("talk")
    # plot_overall_assim_scores(treevars=True)
    # plot_res_assim_scores_box(treevars=True)
    # plot_res_assim_scores_box_thirds(treevars=True)
    # plot_assim_scores_box_thirds(treevars=True)
    # compare_daily_predict(treevars=True)
    plot_spatial_val_scores(3, basin_split=True)
    # plot_spatial_val_scores_boxes(3, basin_split=True)
