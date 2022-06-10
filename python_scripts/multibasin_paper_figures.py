import socket
import pickle
import pathlib
import json
from datetime import datetime
import calendar
import os
os.environ["PROJ_LIB"] = r"C:\\Users\\lcford2\AppData\\Local\\Continuum\\anaconda3\\envs\\sry-env\\Library\\share"

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.gridspec as GS
import matplotlib.lines as mlines
import matplotlib.patches as mpatch
from mpl_toolkits.basemap import Basemap
import seaborn as sns
import geopandas as gpd
from IPython import embed as II
from simulate_reservoir import simulate_storage


hostname = socket.gethostname()
if hostname == "CCEE-DT-094":
    GIS_DIR = pathlib.Path("G:/My Drive/PHD/GIS")
else:
    GIS_DIR = pathlib.Path("~/data/GIS")

def load_pickle(file):
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data

def write_pickle(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f)

def read_results():
    models = {
        "TD0": "../results/tclr_model_drop_res_sto_diff/all/TD0_RT_MS_exhaustive/results.pickle",
        "TD1": "../results/tclr_model_drop_res_sto_diff/all/TD1_RT_MS_exhaustive/results.pickle",
        "TD2": "../results/tclr_model_drop_res_sto_diff/all/TD2_RT_MS_exhaustive/results.pickle",
        "TD3": "../results/tclr_model_drop_res_sto_diff/all/TD3_RT_MS_exhaustive/results.pickle",
        "TD4": "../results/tclr_model_drop_res_sto_diff/all/TD4_RT_MS_exhaustive/results.pickle",
        "TD5": "../results/tclr_model_drop_res_sto_diff/all/TD5_RT_MS_exhaustive/results.pickle",
        "TD6": "../results/tclr_model_drop_res_sto_diff/all/TD6_RT_MS_exhaustive/results.pickle",
        # "TRM": "../results/three_reg_model/all/stovars_three_model/results.pickle"
    }

    return {i: load_pickle(j) for i, j in models.items()}

def select_results(results, get_item):
    return {i: j[get_item] for i, j in results.items()}

def mean_absolute_scaled_error(yact, ymod):
    error = (ymod - yact).abs().mean()
    yact = np.array(yact)
    lagerror = np.absolute(yact[1:] - yact[:-1]).mean()
    return error / lagerror

def get_r2score(df, grouper=None):
    if grouper:
        return pd.DataFrame(
            {"NSE": df.groupby(grouper).apply(
                lambda x: r2_score(x["actual"], x["model"])
            )}
        )
    else:
        return r2_score(df["actual"], df["model"])

def get_rmse(df, grouper=None):
    if grouper:
        return pd.DataFrame(
            {"RMSE": df.groupby(grouper).apply(
                lambda x: mean_squared_error(x["actual"], x["model"], squared=False)
            )}
        )
    else:
        return mean_squared_error(df["actual"], df["model"], squared=False)

def get_nrmse(df, grouper=None):
    normer = "mean"
    if grouper:
        return pd.DataFrame(
            {"nRMSE": df.groupby(grouper).apply(
                lambda x: mean_squared_error(x["actual"], x["model"], squared=False) / getattr(x["actual"], normer)())
            }
        )
    else:
        return mean_squared_error(df["actual"], df["model"], squared=False) / getattr(df["actual"], normer)()

def get_mase(df, grouper=None):
    if grouper:
        return pd.DataFrame(
            {"MASE": df.groupby(grouper).apply(
                lambda x: mean_absolute_scaled_error(x["actual"], x["model"])
            )}
        )
    else:
        return mean_absolute_scaled_error(df["actual"], df["model"])

def get_model_scores(model_dfs, metric="NSE", grouper=None):
    if metric == "NSE":
        return {i: get_r2score(j, grouper) for i, j in model_dfs.items()}
    elif metric == "RMSE":
        return {i: get_rmse(j, grouper) for i, j in model_dfs.items()}
    elif metric == "MASE":
        return {i: get_mase(j, grouper) for i, j in model_dfs.items()}
    elif metric == "nRMSE":
        return {i: get_nrmse(j, grouper) for i, j in model_dfs.items()}

def combine_dict_to_df(dct, colname):
    output = []
    for key, df in dct.items():
        df[colname] = key
        output.append(df)
    return pd.concat(output)

def get_name_replacements():
    with open("../pnw_data/dam_names.json", "r") as f:
        pnw = json.load(f)

    with open("../missouri_data/dam_names.json", "r") as f:
        missouri = json.load(f)

    tva = {}
    with open("./actual_names.csv", "r") as f:
        for line in f.readlines():
            line = line.strip("\n\r")
            key, value = line.split(",")
            tva[key] = value
    return pnw | tva | missouri

def make_bin(df, res=True):
    def bins(x):
        if x <= 1/3:
            return 1
        elif x <= 2/3:
            return 2
        else:
            return 3
    if res:
        df["pct"] = df.groupby("site_name")["actual"].rank(pct=True)
    else:
        df["pct"] = df["actual"].rank(pct=True)

    df["bin"] = df["pct"].apply(bins)
    return df

def plot_performance_boxplots(results):
    metric = "nRMSE"
    train = select_results(results, "train_data")
    test = select_results(results, "test_data")
    simul = select_results(results, "simmed_data")
    train_score = get_model_scores(train, metric=metric, grouper="site_name")
    test_score = get_model_scores(test, metric=metric, grouper="site_name")
    simul_score = get_model_scores(simul, metric=metric, grouper="site_name")

    train_score = combine_dict_to_df(train_score, "Model").reset_index()
    test_score = combine_dict_to_df(test_score, "Model").reset_index()
    simul_score = combine_dict_to_df(simul_score, "Model").reset_index()

    scores = combine_dict_to_df(
        {"Train": train_score, "Test": test_score, "Simulation": simul_score},
        "Data Set"
    )

    # scores = combine_dict_to_df({"Simulation": simul_score}, "Data Set")
    # II()

    fg = sns.catplot(
        data=scores,
        y=metric,
        hue="Data Set",
        # x="Data Set",
        x="Model",
        # hue_order=["TD1", "TD2", "TD3", "TD4", "TD5", "TD6"],
        hue_order=["Train", "Test", "Simulation"],
        legend_out=False,
        kind="boxen",
        showfliers=False,
        # whis=(0.01, 0.99)
    )
    fg.set_xlabels("")
    fg.ax.legend(loc="best")
    plt.show()

def plot_storage_performance_boxplots(results):
    from tclr_model import read_basin_data
    df = read_basin_data("all")

    metric = "nRMSE"
    train = select_results(results, "train_data")
    test = select_results(results, "test_data")
    simul = select_results(results, "simmed_data")

    train_storage = df.loc[train["TD0"].index, "storage"]
    test_storage = df.loc[test["TD0"].index, "storage"]
    simul_storage = df.loc[simul["TD0"].index, "storage"]

    train_storage_pre = df.loc[train["TD0"].index, "storage_pre"]
    test_storage_pre = df.loc[test["TD0"].index, "storage_pre"]
    simul_storage_pre = df.loc[simul["TD0"].index, "storage_pre"]

    train_storage_init = train_storage_pre.groupby("site_name").apply(lambda x: x.head(1))
    test_storage_init = test_storage_pre.groupby("site_name").apply(lambda x: x.head(1))
    simul_storage_init = simul_storage_pre.groupby("site_name").apply(lambda x: x.head(1))


    train_storage_init.index = train_storage_init.index.get_level_values(0)
    test_storage_init.index = test_storage_init.index.get_level_values(0)
    simul_storage_init.index = simul_storage_init.index.get_level_values(0)

    train_storage_sim = {}
    test_storage_sim = {}
    simul_storage_sim = {}
    for key in simul.keys():
        # rdf = train[key]
        # resers = rdf.index.get_level_values("site_name").unique()
        # storage = []
        # indexes = []
        # idx = pd.IndexSlice
        # for res in resers:
        #     rindex = rdf.loc[idx[res, :], :].index
        #     rst = simulate_storage(
        #         train_storage_init[res],
        #         df.loc[rindex, "inflow"],
        #         rdf.loc[rindex, "model"]
        #     )
        #     storage.extend(rst)
        #     indexes.extend(rindex)
        # train_storage_sim[key] = pd.DataFrame(
        #     {"model": storage},
        #     index=pd.MultiIndex.from_tuples(indexes)
        # )

        # rdf = test[key]
        # resers = rdf.index.get_level_values("site_name").unique()
        # storage = []
        # indexes = []
        # idx = pd.IndexSlice
        # for res in resers:
        #     rindex = rdf.loc[idx[res, :], :].index
        #     rst = simulate_storage(
        #         test_storage_init[res],
        #         df.loc[rindex, "inflow"],
        #         rdf.loc[rindex, "model"]
        #     )
        #     storage.extend(rst)
        #     indexes.extend(rindex)
        # test_storage_sim[key] = pd.DataFrame(
        #     {"model": storage},
        #     index=pd.MultiIndex.from_tuples(indexes)
        # )

        rdf = simul[key]
        resers = rdf.index.get_level_values("site_name").unique()
        storage = []
        indexes = []
        idx = pd.IndexSlice
        for res in resers:
            rindex = rdf.loc[idx[res, :], :].index
            rst = simulate_storage(
                simul_storage_init[res],
                df.loc[rindex, "inflow"],
                rdf.loc[rindex, "model"]
            )
            storage.extend(rst)
            indexes.extend(rindex)
        simul_storage_sim[key] = pd.DataFrame(
            {"model": storage},
            index=pd.MultiIndex.from_tuples(indexes, names=["site_name", "datetime"])
        )


    for key, rdf in simul_storage_sim.items():
        rdf["actual"] = simul_storage
    # train_score = get_model_scores(train, metric=metric, grouper="site_name")
    # test_score = get_model_scores(test, metric=metric, grouper="site_name")
    simul_score = get_model_scores(simul_storage_sim, metric=metric, grouper="site_name")

    # train_score = combine_dict_to_df(train_score, "Model").reset_index()
    # test_score = combine_dict_to_df(test_score, "Model").reset_index()
    simul_score = combine_dict_to_df(simul_score, "Model").reset_index()

    # scores = combine_dict_to_df(
        # {"Train": train_score, "Test": test_score, "Simulation": simul_score},
        # "Data Set"
    # )

    # scores = combine_dict_to_df({"Simulation": simul_score}, "Data Set")

    fg = sns.catplot(
        data=simul_score,
        y=metric,
        # hue="Data Set",
        # x="Data Set",
        x="Model",
        # hue_order=["TD1", "TD2", "TD3", "TD4", "TD5", "TD6"],
        # hue_order=["Train", "Test", "Simulation"],
        legend_out=False,
        kind="boxen",
        showfliers=False,
        # whis=(0.01, 0.99)
    )
    fg.set_xlabels("")
    # fg.ax.legend(loc="best")
    plt.show()

def plot_third_performance_boxplots(results):
    train = select_results(results, "train_data")
    test = select_results(results, "test_data")
    simul = select_results(results, "simmed_data")

    train = {i: make_bin(j, res=True).reset_index() for i, j in train.items()}
    test = {i: make_bin(j, res=True).reset_index() for i, j in test.items()}
    simul = {i: make_bin(j, res=True).reset_index() for i, j in simul.items()}

    train_score = get_model_scores(train, metric="RMSE", grouper=["site_name", "bin"])
    test_score = get_model_scores(test, metric="RMSE", grouper=["site_name", "bin"])
    simul_score = get_model_scores(simul, metric="RMSE", grouper=["site_name", "bin"])

    train_score = combine_dict_to_df(train_score, "Model").reset_index()
    test_score = combine_dict_to_df(test_score, "Model").reset_index()
    simul_score = combine_dict_to_df(simul_score, "Model").reset_index()

    scores = combine_dict_to_df(
        {"Train": train_score, "Test": test_score, "Simulation": simul_score},
        "Data Set"
    )

    means = train["TRM"].groupby("bin")["actual"].mean().values

    fg = sns.catplot(
        data=scores,
        y="RMSE",
        hue="Model",
        x="Data Set",
        hue_order=["TD1", "TD2", "TD3", "TD4", "TD5", "TD6"],
        row="bin",
        legend_out=False,
        kind="box",
        sharey=False
    )

    fg.set_ylabels("RMSE [1000 acre-ft/day]")
    fg.set_xlabels("")

    axes = fg.axes.flatten()
    for m, ax in zip(means, axes):
        ax.axhline(m, c="r", linestyle="--")

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].get_legend().remove()

    line = mlines.Line2D([], [], c="r", linestyle="--")
    handles.append(line)
    labels.append("Mean Release")
    axes[0].legend(handles, labels, ncol=3, loc="upper left")
    plt.show()


def plot_variable_correlations():
    # sns.set_context("notebook")
    from tclr_model import read_basin_data
    from statsmodels.tsa.stattools import acf
    df = read_basin_data("all")

    rel = df["release"].unstack()
    res = rel.index
    racf = {}
    nlag = 30
    for r in res:
        racf[r] = acf(rel.loc[r].dropna(), nlags=nlag)
    racf = pd.DataFrame(racf)
    racf_std = racf.std(axis=1)
    racf_quants = racf.quantile([0.25, 0.75], axis=1)
    racf_mean = racf.median(axis=1)
    racf_errbar = (racf_quants - racf_mean).abs()


    inf_corr = {}
    inflow = df.loc[:, ["release", "inflow"]]
    for i in range(31):
        inflow[i] = inflow["inflow"].groupby("site_name").shift(i)

    inf_corr = inflow.groupby("site_name").corr()["release"].unstack()
    inf_corr = inf_corr.drop(["release", "inflow"], axis=1)
    inf_corr_mean = inf_corr.median()
    inf_corr_quants = inf_corr.quantile([0.25, 0.75])
    inf_corr_errbar = (inf_corr_quants - inf_corr_mean).abs()

    st = df.loc[:,["release", "storage_pre", "storage_roll7", "storage_x_inflow"]]
    st["sto_diff"] = df["storage_pre"] - df["storage_roll7"]
    # std_st = st.groupby("site_name").apply(lambda x: (x - x.mean()) / x.std())

    stcorr = st.groupby("site_name").corr()["release"].unstack()
    stcorr = stcorr.drop("release", axis=1)

    # fig, axes = plt.subplots(1, 2)
    # axes = axes.flatten()
    fig = plt.figure()
    gs = GS.GridSpec(2, 2, width_ratios=[1.5,1], height_ratios=[1,1])
    axes = [
        fig.add_subplot(gs[0,0]),
        fig.add_subplot(gs[1,0]),
        fig.add_subplot(gs[:,1])
    ]

    lag_cors = pd.DataFrame({"Release": racf_mean, "Inflow": inf_corr_mean})

    x = range(nlag+1)
    width = 0.8
    x_left = [i-width/2 for i in x]
    x_right = [i+width/2 for i in x]

    axes[0].bar(
        x,
        racf_mean,
        width=width,
        yerr=racf_errbar.values,
        error_kw={
            "elinewidth": 1.5,
        }
    )
    axes[1].bar(
        x,
        inf_corr_mean,
        width=width,
        yerr=inf_corr_errbar.values,
        error_kw={
            "elinewidth": 1.5,
        }
    )
    # racf_mean.plot.bar(
    #     # yerr=racf_std,
    #     yerr=racf_errbar.values,
    #     width=0.7,
    #     error_kw={
    #         "elinewidth": 1.5,
    #     },
    #     ax=axes[0]
    # )

    sns.boxplot(
        data=stcorr.melt(),
        x="variable",
        y="value",
        ax=axes[2],
        whis=(0.05, 0.95)
    )

    axes[0].set_ylabel("$r(R_t, R_L)$")
    axes[1].set_ylabel("$r(R_t, I_L)$")
    axes[1].set_xlabel("Lag $L$ [days]")
    axes[0].set_xticks(x)
    axes[1].set_xticks(x)
    axes[0].set_xticklabels([])
    axes[1].set_xticklabels(
        range(nlag+1), rotation=0,
    )

    axes[2].set_ylabel(r"Pearson's $r$ with Release")
    axes[2].set_xlabel("")
    axes[2].set_xticklabels(
        [
            r"$S_{t-1}$",
            r"$\bar{S}_{t-1}^7$",
            r"$S_{t-1} \times I_{t}$",
            r"$S_{t-1} - \bar{S}_{t-1}^7$",
    ])

    style_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    handles = [
        mpatch.Patch(facecolor=style_colors[0]),
        mlines.Line2D([], [], linewidth=1.5, color="k")
    ]
    labels = [r"Median Pearson's $r$", "25 - 75 quantiles"]
    axes[1].legend(handles, labels, loc="best")
    plt.show()


def plot_variable_correlations_new():
    from tclr_model import read_basin_data, get_basin_meta_data
    df = read_basin_data("all")

    inf_corr = {}
    inflow = df.loc[:, ["release", "inflow"]]
    lags = range(-14, 15)
    for i in lags:
        inflow[i] = inflow["inflow"].groupby("site_name").shift(i)

    inf_corr = inflow.groupby("site_name").corr()["release"].unstack()
    inf_corr = inf_corr.drop(["release", "inflow"], axis=1)
    inf_corr_mean = inf_corr.median()
    inf_corr_quants = inf_corr.quantile([0.25, 0.75])
    inf_corr_errbar = (inf_corr_quants - inf_corr_mean).abs()

    rbasins = pd.read_pickle("../pickles/res_basin_map.pickle")
    rename = {"upper_col": "colorado", "lower_col": "colorado", "pnw": "columbia", "tva": "tennessee"}
    inf_corr["basin"] = rbasins
    inf_corr["basin"] = inf_corr["basin"].replace(rename)
    inf_corr["basin"] = inf_corr["basin"].str.capitalize()

    # inf_corr = inf_corr.melt(id_vars=["basin"], var_name="Lag", value_name="Correlation")

    st = df.loc[:,["release", "storage_pre", "storage_roll7", "storage_x_inflow"]]
    st["sto_diff"] = df["storage_pre"] - df["storage_roll7"]

    drop_res = ["Causey", "Lost Creek", "Echo", "Smith & Morehouse Reservoir",
                "Jordanelle", "Deer Creek", "Hyrum", "Santa Rosa "]
    drop_res = [i.upper() for i in drop_res]

    st = st[~st.index.get_level_values(0).isin(drop_res)]
    stcorr = st.groupby("site_name").corr()["release"].unstack()
    stcorr = stcorr.drop("release", axis=1)
    II()
    sys.exit()

    sns.set_context("notebook")
    fig = plt.figure()
    gs = GS.GridSpec(2, 1, height_ratios=[3, 1])
    axes = [
        fig.add_subplot(gs[0]),
        fig.add_subplot(gs[1])
    ]

    basins = ["Columbia", "Missouri", "Colorado", "Tennessee"]
    colors = sns.color_palette("tab10")
    ax = axes[0]

    sns.boxplot(
        data=inf_corr,
        x="Lag",
        y="Correlation",
        hue="basin",
        palette="tab10",
        whis=(0.05, 0.95),
        ax=ax
    )
    axes[0].legend(loc="best", ncol=4)

    sns.boxplot(
        data=stcorr.melt(),
        y="variable",
        x="value",
        ax=axes[-1],
        whis=(0.05, 0.95)
    )

    axes[0].set_ylabel("$r(R_t, I_L)$")
    axes[0].set_xlabel("Lag $L$ [days]")

    axes[-1].set_xlabel(r"Pearson's $r$ with Release")
    axes[-1].set_ylabel("")
    axes[-1].set_yticklabels(
        [
            r"$S_{t-1}$",
            r"$\bar{S}_{t-1}^7$",
            r"$S_{t-1} \times I_{t}$",
            r"$S_{t-1} - \bar{S}_{t-1}^7$",
    ])

    style_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    handles = [
        mpatch.Patch(facecolor=style_colors[0]),
        mlines.Line2D([], [], linewidth=1.5, color="k")
    ]
    labels = [r"Median Pearson's $r$", "25 - 75 quantiles"]
    plt.show()

def make_map(ax=None, coords=None, other_bound=None):
    if not ax:
        ax = plt.gca()

    if coords:
        west, south, east, north = coords
    else:
        west, south, east, north = (
            -127.441406, 24.207069, -66.093750, 49.382373
        )
    m = Basemap(
        # projection="merc",
        epsg=3857,
        resolution="c",
        llcrnrlon=west,
        llcrnrlat=south,
        urcrnrlon=east,
        urcrnrlat=north,
        ax=ax
    )
    parallels = np.arange(0.0, 81, 10.0)
    meridians = np.arange(10.0, 351.0, 20.0)
    pvals = m.drawparallels(parallels, linewidth=0.0, labels=[1,1,1,1])
    mvals = m.drawmeridians(meridians, linewidth=0.0, labels=[1,1,1,1])
    xticks = [i[1][0].get_position()[0] for i in mvals.values()]
    yticks = []
    for i in pvals.values():
        try:
            yticks.append(i[1][0].get_position()[1])
        except IndexError:
            pass

    ax.set_xticks(xticks) 
    ax.set_yticks(yticks)
    ax.tick_params(
        axis="both",
        direction="in",
        left=True,
        right=True,
        top=True,
        bottom=True,
        labelleft=False,
        labelright=False,
        labeltop=False,
        labelbottom=False,
        zorder=10
    )

    states_path = GIS_DIR / "cb_2017_us_state_500k"

    mbound = m.drawmapboundary(fill_color="white")
    states = m.readshapefile(states_path.as_posix(), "states")

    rivers = [
        (GIS_DIR / "NHDPlus" / "trimmed_flowlines" / "NHDPlusLC_trimmed_flowlines_noz").as_posix(),
        (GIS_DIR / "NHDPlus" / "trimmed_flowlines" / "NHDPlusUC_trimmed_flowlines_noz").as_posix(),
        (GIS_DIR / "NHDPlus" / "trimmed_flowlines" / "NHDPlusTN_trimmed_flowlines_noz").as_posix(),
        (GIS_DIR / "NHDPlus" / "trimmed_flowlines" / "NHDPlusML_trimmed_flowlines_noz").as_posix(),
        (GIS_DIR / "NHDPlus" / "trimmed_flowlines" / "NHDPlusMU_trimmed_flowlines_noz").as_posix(),
        (GIS_DIR / "NHDPlus" / "trimmed_flowlines" / "NHDPlusPN_trimmed_flowlines_noz").as_posix()
    ]

    for i, r in enumerate(rivers):
        river_lines = m.readshapefile(r, f"river_{i}", color="b", linewidth=0.5, default_encoding="latin-1")
        river_lines[4].set_alpha(1.0)

    if other_bound:
        for b, c in other_bound:
            bound = m.readshapefile(
                b,
                "bound",
                # color="#FF3BC6"
                color=c
            )
            # bound[4].set_facecolor("#FF3BC6")
            bound[4].set_facecolor(c)
            bound[4].set_alpha(0.5)
            bound[4].set_zorder(2)

    return m


def plot_res_locs():
    res_locs = pd.read_csv("../geo_data/reservoirs.csv")
    res_locs = res_locs.set_index("site_name")

    drop_res = ["Causey", "Lost Creek", "Echo", "Smith & Morehouse Reservoir",
                "Jordanelle", "Deer Creek", "Hyrum", "Santa Rosa "]
    drop_res = [i.upper() for i in drop_res]
    res_locs = res_locs.drop(drop_res)
    
    with open("../geo_data/extents.json", "r") as f:
        coords = json.load(f)
    color_map = sns.color_palette("Set2")
    basins = [((GIS_DIR/"columbia_shp"/"Shape"/"WBDHU2").as_posix(), color_map[0]),
              ((GIS_DIR/"missouri_shp"/"Shape"/"WBDHU2").as_posix(), color_map[1]),
            #   (GIS_DIR/"lowercol_shp"/"Shape"/"WBDHU2").as_posix(),
            #   (GIS_DIR/"uppercol_shp"/"Shape"/"WBDHU2").as_posix(),
              ((GIS_DIR/"colorado_shp"/"Shape"/"WBDHU2").as_posix(), color_map[2]),
              ((GIS_DIR/"tennessee_shp"/"Shape"/"WBDHU2").as_posix(), color_map[3])]


    fig, ax = plt.subplots(1, 1)
    m = make_map(ax, other_bound=basins)
    x, y = m(res_locs.long, res_locs.lat)
    max_size = 600
    min_size = 50
    size_var = "max_sto"
    max_value = res_locs[size_var].max()
    min_value = res_locs[size_var].min()
    
    ratio = (max_size - min_size) / (max_value - min_value)
    sizes = [min_size + i * ratio for i in res_locs[size_var]]
    markers = ax.scatter(x, y, marker="v", edgecolor="k", s=sizes, zorder=4)
    marker_color = markers.get_facecolor()
    
    river_line = mlines.Line2D(
        [], [], color="b", alpha=1, linewidth=0.5
    )
    river_basins = [
        mpatch.Patch(facecolor=color_map[i], alpha=0.5) for i in range(4)
    ]
    size_legend_sizes = np.linspace(min_size, max_size, 4)
    # size_legend_labels = [(i-min_size) / ratio for i in size_legend_sizes]
    size_legend_labels = np.linspace(min_value, max_value, 4)

    size_markers = [
        plt.scatter([], [], s=i, edgecolors="k", c=marker_color, marker="v")
        for i in size_legend_sizes
    ]

    size_legend = plt.legend(
        size_markers,
        [
            f"{round(size_legend_labels[0]*1000, -2):.0f}",
            *[f"{round(i / 1000, 0):,.0f} million" for i in size_legend_labels[1:]]
        ],
        title="Maximum Storage [acre-feet]",
        loc="lower left",
        ncol=4,
    )
    hydro_legend = plt.legend(
        [river_line, *river_basins],
        [
            "Major River Flowlines",
            "Columbia HUC2",
            "Missouri HUC2",
            "Colorado HUC2",
            "Tennessee HUC2",
        ],
        loc="lower right"
    )
    ax.add_artist(size_legend)
    ax.add_artist(hydro_legend)

    plt.show()

def plot_res_perf_map(results):
    train = select_results(results, "train_data")
    test = select_results(results, "test_data")
    simul = select_results(results, "simmed_data")
    train_score = get_model_scores(train, grouper="site_name")
    test_score = get_model_scores(test, grouper="site_name")
    simul_score = get_model_scores(simul, grouper="site_name")

    train_score = combine_dict_to_df(train_score, "Model").reset_index()
    test_score = combine_dict_to_df(test_score, "Model").reset_index()
    simul_score = combine_dict_to_df(simul_score, "Model").reset_index()

    res_locs = pd.read_csv("../geo_data/reservoirs.csv")
    res_locs = res_locs.set_index("site_name")
    score = simul_score[simul_score["Model"] == "TD4"]
    score = score.set_index("site_name")["NSE"]

    res_locs["NSE"] = score

    rbasins = pd.read_pickle("../pickles/res_basin_map.pickle")
    rename = {"upper_col": "colorado", "lower_col": "colorado"}
    res_locs["basin"] = rbasins
    res_locs["basin"] = res_locs["basin"].replace(rename)

    max_size = 200

    drop_res = ["Causey", "Lost Creek", "Echo", "Smith & Morehouse Reservoir",
                "Jordanelle", "Deer Creek", "Hyrum", "Santa Rosa "]
    drop_res = [i.upper() for i in drop_res]


    with open("../geo_data/extents.json", "r") as f:
        coords = json.load(f)

    II()
    sys.exit()


    fig, axes = plt.subplots(2, 2)

    make_map(axes[0,0], coords["Columbia"])
             # [(GIS_DIR/"columbia_shp"/"Shape"/"WBDHU2").as_posix()])
    make_map(axes[0,1], coords["Missouri"])
             # [(GIS_DIR/"missouri_shp"/"Shape"/"WBDHU2").as_posix()])
    make_map(axes[1,0], coords["Colorado"])
             # [(GIS_DIR/"lowercol_shp"/"Shape"/"WBDHU2").as_posix(),
              # (GIS_DIR/"uppercol_shp"/"Shape"/"WBDHU2").as_posix()])
    make_map(axes[1,1], coords["Tennessee"])
             # [(GIS_DIR/"tennessee_shp"/"Shape"/"WBDHU2").as_posix()])

    basins = ["pnw", "missouri", "colorado", "tva"]
    titles = ["Columbia", "Missouri", "Colorado", "Tennessee"]
    for ax, basin, title in zip(axes.flatten(), basins, titles):
        pdf = res_locs[res_locs["basin"] == basin]
        size = pdf["NSE"] * max_size
        ax.scatter(pdf["long"], pdf["lat"], s=size, edgecolor="k")
        ax.set_title(title)

    plt.show()

def plot_flood_analysis(results):
    train_data = select_results(results, "train_data")
    dates = {
        "tennessee": ["04-01-2003", "07-01-2003"],
        "colorado": ["05-01-1983", "08-01-1983"],
        "columbia": ["01-01-1996", "04-01-1996"],
        "missouri": ["06-01-1993", "09-01-1993"]
    }

    rbasins = pd.read_pickle("../pickles/res_basin_map.pickle")
    rename = {"upper_col": "colorado", "lower_col": "colorado"}
    rbasins = rbasins.replace(rename)

    resers = {
        "tennessee": ["Apalachia", "Hiwassee", "WattsBar", "Chikamauga"],
        "colorado": ["Lake Powell".upper()],
        "columbia": rbasins[rbasins == "pnw"].index,
        "missouri": rbasins[rbasins == "missouri"].index
    }

    idx = pd.IndexSlice
    df = train_data["TD3"]
    dfs = {b: df.loc[idx[r, :], :] for b, r in resers.items()}

    trimmed_dfs = {}

    for b, (start, stop) in dates.items():
        bdf = dfs[b]
        start = datetime.strptime(start, "%m-%d-%Y")
        stop = datetime.strptime(stop, "%m-%d-%Y")
        trimmed = bdf.loc[
            (bdf.index.get_level_values("datetime") >= start) &
            (bdf.index.get_level_values("datetime") < stop)
        ]
        trimmed_dfs[b] = trimmed

    floodops = combine_dict_to_df(trimmed_dfs, "Basin")
    floodops = floodops.drop("bin", axis=1).reset_index()
    floodops = floodops.melt(id_vars=["site_name", "datetime", "Basin"])
    fg = sns.displot(
        data=floodops,
        x="value",
        hue="variable",
        col="Basin",
        col_wrap=2,
        kind="ecdf",
        facet_kws={
            "sharex":False,
            "legend_out": False
    })
    plt.show()


def plot_overall_ecdf(results):
    train = select_results(results, "train_data")
    test = select_results(results, "test_data")
    simul = select_results(results, "simmed_data")
    train = train["TD3"]
    test = test["TD3"]
    simul = simul["TD3"]

    df = combine_dict_to_df({
        "train": train,
        "test": test,
        "simul": simul
    }, "dataset")

    df = df.drop("bin", axis=1).reset_index()
    quant, bins = pd.qcut(df["actual"], 100, labels=False, retbins=True)
    df["bin"] = quant

    mase = df.groupby("bin").apply(
        lambda x: mean_absolute_scaled_error(x["actual"], x["model"]))

    II()
    # df = df.melt(id_vars=["site_name", "datetime", "dataset"])

    # sns.displot(
    #     data=df,
    #     x="value",
    #     hue="variable",
    #     col="dataset",
    #     kind="ecdf",
    #     facet_kws={
    #         "sharex":True,
    #         "legend_out": False
    #     }
    # )

    # plt.show()

def plot_seasonal_performance(results):
    train = select_results(results, "train_data")
    test = select_results(results, "test_data")
    simul = select_results(results, "simmed_data")

    train_score = {i: j.groupby(
        [j.index.get_level_values("site_name"),
         j.index.get_level_values("datetime").month]
    ).apply(lambda x: mean_squared_error(x["actual"], x["model"], squared=False)) for i, j in train.items()}
    test_score = {i: j.groupby(
        [j.index.get_level_values("site_name"),
         j.index.get_level_values("datetime").month]
    ).apply(lambda x: mean_squared_error(x["actual"], x["model"], squared=False)) for i, j in test.items()}
    simul_score = {i: j.groupby(
        [j.index.get_level_values("site_name"),
         j.index.get_level_values("datetime").month]
    ).apply(lambda x: mean_squared_error(x["actual"], x["model"], squared=False)) for i, j in simul.items()}


    train_dfs = []
    test_dfs = []
    simul_dfs = []

    for i, j in train_score.items():
        j = pd.DataFrame({"RMSE": j})
        j["Model"] = i
        train_dfs.append(j)

        test_score[i] = pd.DataFrame({"RMSE": test_score[i]})
        test_score[i]["Model"] = i
        test_dfs.append(test_score[i])

        simul_score[i] = pd.DataFrame({"RMSE": simul_score[i]})
        simul_score[i]["Model"] = i
        simul_dfs.append(simul_score[i])

    train_df = pd.concat(train_dfs).reset_index()
    test_df = pd.concat(test_dfs).reset_index()
    simul_df = pd.concat(simul_dfs).reset_index()


    # pdf = simul_df.groupby(["Model", "datetime"]).quantile(0.5).reset_index()

    simul_score = {i: pd.DataFrame({"RMSE": j.groupby(j.index.get_level_values("datetime").month).apply(
        lambda x: mean_squared_error(x["actual"], x["model"], squared=False)
    )}) for i, j in simul.items()}
    simul_df = combine_dict_to_df(simul_score, "Model").reset_index()

    ax = sns.barplot(
        data=simul_df,
        x="datetime",
        y="RMSE",
        hue="Model",
        palette="tab10",
        errwidth=0,
        alpha=0.6
    )

    means = simul_df.groupby("Model").mean()["RMSE"]
    colors = sns.color_palette("tab10")

    for i, mv in enumerate(means):
        ax.axhline(mv, color=colors[i])

    ax.set_ylabel("Simulated RMSE [1000 acre-ft/day]")
    ax.set_xticklabels(calendar.month_abbr[1:])
    ax.set_xlabel("")
    ax.legend(loc="best", ncol=4, title="")

    plt.show()

def plot_upper_lower_perf(results):
    from tclr_model import read_basin_data
    df = read_basin_data("all")

    train = select_results(results, "train_data")
    test = select_results(results, "test_data")
    simul = select_results(results, "simmed_data")

    train_inflow = df.loc[train["TD0"].index, "inflow"]
    test_inflow = df.loc[test["TD0"].index, "inflow"]
    simul_inflow = df.loc[simul["TD0"].index, "inflow"]

    # train_pct = {i: j.groupby("site_name")["actual"].rank(pct=True)
    #              for i, j in train.items()}
    # test_pct = {i: j.groupby("site_name")["actual"].rank(pct=True)
    #              for i, j in test.items()}
    # simul_pct = {i: j.groupby("site_name")["actual"].rank(pct=True)
    #              for i, j in simul.items()}

    train_pct = train_inflow.groupby("site_name").rank(pct=True)
    test_pct = test_inflow.groupby("site_name").rank(pct=True)
    simul_pct = simul_inflow.groupby("site_name").rank(pct=True)

    train_trimmed = {}
    test_trimmed = {}
    simul_trimmed = {}

    for d in train.keys():
        df = train[d]
        df["pct"] = train_pct
        df = df[(df["pct"] >= 0.95) | (df["pct"] <= 0.05)]
        df["label"] = ""
        df.loc[df["pct"] <= 0.05, "label"] = "Lowest 5% Inflow"
        df.loc[df["pct"] >= 0.95, "label"] = "Highest 5% Inflow"

        train_trimmed[d] = df

        df = test[d]
        df["pct"] = test_pct
        df = df[(df["pct"] >= 0.95) | (df["pct"] <= 0.05)]
        df["label"] = ""
        df.loc[df["pct"] <= 0.05, "label"] = "Lowest 5% Inflow"
        df.loc[df["pct"] >= 0.95, "label"] = "Highest 5% Inflow"

        test_trimmed[d] = df

        df = simul[d]
        df["pct"] = simul_pct
        df = df[(df["pct"] >= 0.95) | (df["pct"] <= 0.05)]
        df["label"] = ""
        df.loc[df["pct"] <= 0.05, "label"] = "Lowest 5% Inflow"
        df.loc[df["pct"] >= 0.95, "label"] = "Highest 5% Inflow"

        simul_trimmed[d] = df

    df = combine_dict_to_df(simul_trimmed, "Depth")
    scores = pd.DataFrame({"NSE": df.groupby(["Depth", "label"]).apply(
        lambda x: r2_score(x["actual"], x["model"])
    )})

    scores = scores.reset_index()

    ax = sns.barplot(
        data=scores,
        y="NSE",
        x="Depth",
        hue="label",
    )
    ax.set_xlabel("")
    ax.legend(loc="best")
    plt.show()

def plot_constinency_analysis(results):
    simul = select_results(results, "simmed_data")

    df = simul["TD3"]
    means = df.groupby("site_name")["actual"].mean()
    resers = df.index.get_level_values("site_name").unique()
    idx = pd.IndexSlice
    const = []
    thresh = 0.05
    window = 10
    for res in resers:
        rdf = df.loc[idx[res, :], :]
        rconst = rdf["actual"].rolling(window).std() / means.loc[res]
        const.append(rconst[rconst < thresh])

    const = pd.concat(const)

    indexes = []
    for s, d in const.index:
        date_range = pd.date_range(
             d - np.timedelta64(window - 1, "D"), d
        )
        for dr in date_range:
            indexes.append((s, dr))

    indexes = list(set(indexes))

    cdf = df.loc[indexes, :]
    cdf = cdf.sort_index()

    II()

    # scores = pd.DataFrame({"NSE": cdf.groupby("site_name").apply(
    #     lambda x: r2_score(x["actual"], x["model"])
    # )})

    rbasins = pd.read_pickle("../pickles/res_basin_map.pickle")
    # scores["basin"] = rbasins
    # scores = scores.reset_index()
    # scores = scores.sort_values(by=["basin", "NSE"])
    # II()
    # scores.plot.bar()
    # scores = scores.melt(id_vars=["site_name", "basin"])
    # sns.barplot(
    #     data=scores,
    #     x="site_name",
    #     y="NSE",
    #     hue="basin"
    #     # hue="variable",
    # )
    # plt.show()

    # import sys
    # sys.exit()


    name_replacements = get_name_replacements()

    sds = cdf.groupby("site_name").rolling(window - 1).std()
    sds["diff"] = sds["model"] - sds["actual"]
    sds.index = sds.index.droplevel(0)

    sds_means = pd.DataFrame({
        "Observed": sds["actual"].groupby("site_name").mean(),
        "Modeled": sds["model"].groupby("site_name").mean(),
        "basin": rbasins
    })
    sds_means["Observed"] = sds_means["Observed"] / means
    sds_means["Modeled"] = sds_means["Modeled"] / means

    sds_means = sds_means.reset_index()
    sds_means["site_name"] = sds_means["site_name"].replace(name_replacements).str.title()
    sds_means = sds_means.melt(id_vars=["site_name", "basin"])

    sds_means = sds_means.sort_values(by=["basin", "value"]).dropna()
    ax = sns.barplot(
        data=sds_means,
        x="site_name",
        y="value",
        hue="variable"
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_xlabel("")
    ax.set_ylabel("Coefficient of Variation")
    ax.legend(loc="best")

    plt.show()

    sds_means["basin"] = sds_means["basin"].replace(
        {
            "lower_col":"Colorado",
            "upper_col":"Colorado",
            "pnw": "Columbia",
            "tva": "Tennessee"
        }
    )
    sds_means["basin"] = sds_means["basin"].str.capitalize()

    ax = sns.boxplot(
        data=sds_means,
        x="basin",
        y="value",
        hue="variable"
    )
    ax.legend(loc="best")
    ax.set_xlabel("")
    ax.set_ylabel("Coefficient of Variation")
    plt.show()

def plot_perf_vs_datalength(results):
    simul = select_results(results, "simmed_data")
    simul_score = get_model_scores(simul, grouper="site_name")
    simul_score = combine_dict_to_df(simul_score, "Model").reset_index()

    ntrees = simul_score["Model"].unique().size
    counts = simul["TD0"].index.get_level_values("site_name").value_counts()
    train_counts = counts / 0.2 * 0.8

    simul_score = simul_score.set_index("site_name")
    simul_score["counts"] = train_counts  / 365

    sns.scatterplot(
        data=simul_score,
        x="counts",
        y="NSE",
        hue="Model"
    )
    plt.show()


if __name__ == "__main__":
    plt.style.use("seaborn")
    sns.set_context("talk")
    results = read_results()
    # plot_performance_boxplots(results)
    plot_storage_performance_boxplots(results)
    # plot_third_performance_boxplots(results)
    # plot_res_perf_map(results)
    # plot_res_locs()
    # plot_variable_correlations_new()
    # plot_flood_analysis(results)
    # plot_overall_ecdf(results)
    # plot_seasonal_performance(results)
    # plot_upper_lower_perf(results)
    # plot_constinency_analysis(results)
    # plot_perf_vs_datalength(results)
    # II()
