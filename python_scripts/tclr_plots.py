import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from IPython import embed as II


def load_all_data(get_value=None, postpend=None):
    base_path = "../results/basin_eval/all/tclr_model_no_ints_all_res_"

    data = {}
    for base_id in range(0, 11):
        d_id = f"D{base_id}"
        d_path = f"{base_path}{base_id}"
        if postpend:
            d_id += f"_{postpend}"
            d_path += f"_{postpend}"
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


def calc_scores(data):
    scores = {}
    for d_id, test_data in data.items():
        res_grouper = test_data.index.get_level_values(0)
        d_scores = pd.DataFrame(
            index=res_grouper.unique(),
            columns=["NSE", "RMSE"])
        d_scores["NSE"] = test_data.groupby(res_grouper).apply(
            lambda x: r2_score(x["actual"], x["model"])
        )
        d_scores["RMSE"] = test_data.groupby(res_grouper).apply(
            lambda x: mean_squared_error(x["actual"], x["model"], squared=False)
        )
        scores[d_id] = d_scores
    return scores


def plot_overall_assim_scores():
    postpends = [None, "weekly", "monthly", "seasonally"]
    data = [load_all_data(get_value="s_act_score", postpend=i)
            for i in postpends]
    records = []
    columns = [f"D{i}" for i in range(0, 11)]
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
    df = df.loc[[
        "Never", "Seasonally", "Monthly", "Weekly"
    ], :]

    df.T.plot.bar(ylabel="Overall NSE", rot=0)

    plt.show()


def plot_res_assim_scores_box():
    postpends = [None, "weekly", "monthly", "seasonally"]
    data = [load_all_data(get_value="simmed_res_scores", postpend=i)
            for i in postpends]
    records = []
    columns = [f"D{i}" for i in range(0, 11)]
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
    assim_map = {
        "Never": 0,
        "Seasonally": 1,
        "Monthly": 2,
        "Weekly": 3
    }
    df["assim_map"] = df["assim"].apply(assim_map.get)
    df = df.sort_values(by=["assim_map", "depth"])
    fg = sns.catplot(data=df, x="depth", y="value", hue="assim", kind="box",
                     legend_out=False)
    ax = fg.ax
    ax.set_ylabel("NSE")
    ax.set_xlabel("Tree Depth")
    ax.get_legend().set_title("")
    plt.show()


def plot_res_assim_scores_box_thirds():
    postpends = [None, "weekly", "monthly", "seasonally"]
    data = [load_all_data(get_value=["data", "simmed_data"], postpend=i)
            for i in postpends]

    columns = [f"D{i}" for i in range(0, 11)]
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

    scores = scores.reset_index().rename(columns={"level_1": "bin"}).melt(
        id_vars=["site_name", "bin"])
    scores[["depth", "assim"]] = scores["variable"].str.split("_", expand=True)
    scores["assim"] = scores["assim"].fillna("never")

    sns.catplot(
        data=scores,
        x="depth",
        y="value",
        hue="assim",
        row="bin",
        kind="box",
        sharey=False
    )
    plt.show()


def plot_assim_scores_box_thirds():
    postpends = [None, "weekly", "monthly", "seasonally"]
    data = [load_all_data(get_value=["data", "simmed_data"], postpend=i)
            for i in postpends]

    columns = [f"D{i}" for i in range(0, 11)]
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

    idx = pd.IndexSlice
    bins = pd.qcut(output["actual"], 3, labels=False)
    output["bin"] = bins
    for col in scores.columns:
        try:
            cscore = output.groupby("bin").apply(lambda x: r2_score(x["actual"], x[col]))
            scores[col] = cscore.values
        except ValueError:
            print(f"NaN values for {col}")

    scores = scores.reset_index().rename(columns={"index": "bin"}).melt(
        id_vars=["bin"])
    scores[["depth", "assim"]] = scores["variable"].str.split("_", expand=True)
    scores["assim"] = scores["assim"].fillna("never")

    sns.catplot(
        data=scores,
        x="depth",
        y="value",
        hue="assim",
        row="bin",
        kind="bar",
        sharey=False
    )
    plt.show()



if __name__ == "__main__":
    plt.style.use("ggplot")
    sns.set_context("talk")
    # plot_overall_assim_scores()
    # plot_res_assim_scores_box()
    # plot_res_assim_scores_box_thirds()
    plot_assim_scores_box_thirds()
