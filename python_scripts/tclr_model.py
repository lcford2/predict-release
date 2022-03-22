import argparse
import calendar
import copy
import glob
import pathlib
import pickle
import subprocess
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from IPython import embed as II
from sklearn.metrics import mean_squared_error, r2_score
from tclr import TreeComboLR
from utils.timing_function import time_function


@time_function
def read_basin_data(basin: str) -> pd.DataFrame:
    data_locs = {
        "upper_col": {
            "ready": "../upper_colorado_data/model_ready_data/upper_col_data_net_inflow.csv",
            "raw": "../upper_colorado_data/hydrodata_data/req_upper_col_data.csv",
        },
        "pnw": {
            "ready": "../pnw_data/model_ready_data/pnw_data_net_inflow.csv",
            "raw": "../pnw_data/dam_data/*_data/*.csv",
        },
        "lower_col": {
            "ready": "../lower_col_data/model_ready_data/lower_col_data_net_inflow.csv",
            "raw": "../lower_col_data/lower_col_dam_data.csv",
        },
        "missouri": {
            "ready": "../missouri_data/model_ready_data/missouri_data_net_inflow.csv",
            "raw": "../missouri_data/hydromet_data/*.csv",
        },
        "tva": {"ready": "../csv/tva_model_ready_data.csv"},
    }

    if basin == "colorado":
        lfpath = data_locs["lower_col"]["ready"]
        ldf = pd.read_csv(lfpath)
        ldf["datetime"] = pd.to_datetime(ldf["datetime"])
        ldf = ldf.set_index(["site_name", "datetime"])

        ufpath = data_locs["upper_col"]["ready"]
        udf = pd.read_csv(ufpath)
        udf["datetime"] = pd.to_datetime(udf["datetime"])
        udf = udf.set_index(["site_name", "datetime"])

        df = ldf.append(udf)
        df = df.sort_index()
        df = df.dropna()
    elif basin in data_locs:
        fpath = pathlib.Path(data_locs[basin]["ready"])
        df = pd.read_csv(fpath)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index(["site_name", "datetime"])
        df = df.sort_index()
        df = df.dropna()
    elif basin == "all":
        df = pd.DataFrame()
        for b in data_locs.keys():
            fpath = data_locs[b]["ready"]
            bdf = pd.read_csv(fpath)
            bdf["datetime"] = pd.to_datetime(bdf["datetime"])
            bdf = bdf.set_index(["site_name", "datetime"])
            bdf = bdf.sort_index()
            bdf = bdf.dropna()
            if df.empty:
                df = bdf
            else:
                df = pd.concat([df, bdf])
    else:
        raise NotImplementedError(f"No data available for basin {basin}")
    return df


def get_basin_meta_data(basin: str):
    if basin == "tva":
        files = ["../pickles/tva_res_meta.pickle"]
    elif basin == "colorado":
        files = [
            "../apply_models/basin_output_no_ints/upper_col_meta.pickle",
            "../apply_models/basin_output_no_ints/lower_col_meta.pickle",
        ]
    elif basin == "all":
        files = glob.glob("../apply_models/basin_output_no_ints/*_meta.pickle")
    else:
        files = [f"../apply_models/basin_output_no_ints/{basin}_meta.pickle"]

    meta = pd.DataFrame()
    for file in files:
        fmeta = pd.read_pickle(file)
        meta = fmeta if meta.empty else meta.append(fmeta)
    return meta


def get_max_date_span(in_df):
    df = pd.DataFrame()
    df["date"] = in_df.index.get_level_values(1)
    df["mask"] = 1
    df.loc[df["date"] - np.timedelta64(1, "D") == df["date"].shift(), "mask"] = 0
    df["mask"] = df["mask"].cumsum()
    span = df.loc[df["mask"] == df["mask"].value_counts().idxmax(), "date"]
    return (span.min(), span.max())


@time_function
def prep_data(df):
    grouper = df.index.get_level_values(0)
    std_data = df.groupby(grouper).apply(lambda x: (x - x.mean()) / x.std())
    means = df.groupby(grouper).mean()
    std = df.groupby(grouper).std()
    columns = [
        "release_pre",
        "storage",
        "storage_pre",
        "inflow",
        "release_roll7",
        "inflow_roll7",
        "storage_roll7",
        "storage_x_inflow",
    ]
    X = std_data.loc[:, columns]
    y = std_data["release"]
    return X, y, means, std


@time_function
def split_train_test_dt(index, date, level=None, keep=0):
    # cannot check truthy here because level can be integer 0, which is equivalent to False
    if level is not None:
        test = index[index.get_level_values(level) >= date - timedelta(days=keep)]
        train = index[index.get_level_values(level) < date]
    else:
        test = index[index >= date]
        train = index[index < date]
    return train, test


def split_train_test_res(index, test_res):
    train = index[~index.get_level_values(1).isin(test_res)]
    test = index[index.get_level_values(1).isin(test_res)]
    return train, test


def get_params_and_groups(X, tree):
    # use the tree to get what leaves correpond with each entry
    # in the X matrix
    params, leaves = tree.apply(X)
    # make those leaves into a pandas series for the ml model
    groups = pd.Series(leaves, index=X.index)
    return params, groups


def split_train_test_index_by_res(df, prop=0.8):
    resers = df.index.get_level_values(0).unique()
    train_index = []
    test_index = []
    idx = pd.IndexSlice
    for res in resers:
        rdf = df.loc[idx[res, :], :].index
        size = len(rdf)
        train_size = int(prop * size)
        train_index.extend(rdf.values[:train_size])
        test_index.extend(rdf.values[train_size:])
    return train_index, test_index


def pipeline(args):
    basin = args.basin
    month_intercepts = args.month_ints
    max_depth = args.max_depth

    df = read_basin_data(basin)
    meta = get_basin_meta_data(basin)
    meta = meta.drop("MCPHEE")

    lower_bounds = df.groupby(df.index.get_level_values(0)).min()
    upper_bounds = df.groupby(df.index.get_level_values(0)).max()

    use_all = True
    if use_all:
        reservoirs = meta.index
    else:
        reservoirs = meta[meta["group"] == "high_rt"].index

    idx = pd.IndexSlice
    spans = {res: get_max_date_span(df.loc[idx[res,:], :]) for res in reservoirs}
    
    trimmed = []
    for res, span in spans.items():
        rdf = df.loc[idx[res, :], :]
        rindex = rdf.index.get_level_values(1)
        trimmed.append(rdf[
            (rindex >= span[0]) &
            (rindex <= span[1])
        ])
    df = pd.concat(trimmed).sort_index()

    res_grouper = df.index.get_level_values(0)
    df = df.loc[res_grouper.isin(reservoirs)]
    # need to set it again after we trimmed the data set
    res_grouper = df.index.get_level_values(0)

    X, y, means, std = prep_data(df)
    X["sto_diff"] = X["storage_pre"] - X["storage_roll7"]
    X["const"] = 1

    # Setup monthly intercepts if using
    if month_intercepts:
        month_arrays = {i: [] for i in calendar.month_abbr[1:]}
        for date in X.index.get_level_values(1):
            for key in month_arrays.keys():
                if calendar.month_abbr[date.month] == key:
                    month_arrays[key].append(1)
                else:
                    month_arrays[key].append(0)
        for key, array in month_arrays.items():
            X[key] = array

    # set exogenous variables
    X_vars = [
        "storage_pre",
        "release_pre",
        "inflow",
        "storage_roll7",
        "release_roll7",
        "inflow_roll7",
        "storage_x_inflow",
    ]

    X_vars_tree = copy.copy(X_vars)

    # no constants for tree to split on
    if month_intercepts:
        X_vars.extend(calendar.month_abbr[1:])
    else:
        X_vars.insert(0, "const")

    # train_index = X[X.index.get_level_values(1) < datetime(2010, 1, 1)].index
    # test_index = X[X.index.get_level_values(1) >= datetime(2010, 1, 1)].index
    train_index, test_index = split_train_test_index_by_res(X, prop=0.8)

    X_train = X.loc[train_index, X_vars]
    y_train = y.loc[train_index]
    X_test = X.loc[test_index, X_vars]
    y_test = y.loc[test_index]

    df["const"] = 1
    X_test_act = df.loc[test_index, X_vars]
    y_test_act = df.loc[test_index, "release"]

    add_tree_vars = ["rts", "max_sto"]
    for tree_var in add_tree_vars:
        train_values = [meta.loc[i, tree_var] for i in X_train.index.get_level_values(0)]
        test_values = [meta.loc[i, tree_var] for i in X_test.index.get_level_values(0)]
        X_train[tree_var] = train_values
        X_test[tree_var] = test_values
        X_test_act[tree_var] = test_values

    X_vars_tree = [*X_vars, *add_tree_vars]

    train_res = res_grouper.unique()
    test_res = train_res

    make_dot = False
    if max_depth > 0:
        make_dot = True
        model = TreeComboLR(
            X_train,
            y_train,
            max_depth=max_depth,
            feature_names=X_vars_tree,
            response_name="release",
            tree_vars=X_vars_tree,
            reg_vars=X_vars,
        )

        model.grow_tree()

        params, groups = get_params_and_groups(X_train, model)
        groups_uniq = groups.unique()
        groups_uniq.sort()
        group_map = {j: i + 1 for i, j in enumerate(groups_uniq)}
        groups = groups.apply(group_map.get)
        groups_list = list(groups)
        coefs = pd.DataFrame(
            {i: params[groups_list.index(i)] for i in group_map.values()},
            index=group_map.values(),
            columns=X_vars,
        )

        fitted = model.predict()
        preds = model.predict(X_test)
        simuled = simulate_tclr_model(
            model,
            "model",
            X_test_act,
            means,
            std,
            X_vars_tree,
            X_vars,
            lower_bounds,
            upper_bounds,
            args.assim,
            pd.Series(preds, index=X_test.index),
            X_test
        )
        simuled = simuled[["release", "storage"]].dropna()
    else:
        X_train = X_train[X_vars]
        X_test = X_test[X_vars]
        X_test_act = X_test_act[X_vars]
        beta = np.linalg.inv(X_train.T @ X_train) @ (X_train.T @ y_train)
        coefs = pd.Series(beta, index=X_vars)
        fitted = X_train @ beta
        preds = X_test @ beta
        simuled = simulate_tclr_model(
            beta, 
            "beta",
            X_test_act,
            means,
            std,
            X_vars,
            X_vars,
            lower_bounds,
            upper_bounds,
            args.assim,
            pd.Series(preds, index=X_test.index),
            X_test
        )
        simuled = simuled[["release", "storage"]].dropna()

    fitted = pd.Series(fitted, index=X_train.index)
    preds = pd.Series(preds, index=X_test.index)
    simmed = simuled["release"]
    

    y_train_act = (
        y_train.unstack().T * std.loc[train_res, "release"]
        + means.loc[train_res, "release"]
    ).T.stack()
    y_test_act = (
        y_test.unstack().T * std.loc[test_res, "release"] + means.loc[test_res, "release"]
    ).T.stack()

    fitted_act = (
        fitted.unstack().T * std.loc[train_res, "release"]
        + means.loc[train_res, "release"]
    ).T.stack()
    preds_act = (
        preds.unstack().T * std.loc[test_res, "release"] + means.loc[test_res, "release"]
    ).T.stack()

        
    y_test_sim = y_test_act.loc[simmed.index]

    f_act_score = r2_score(y_train_act, fitted_act)
    f_act_rmse = np.sqrt(mean_squared_error(y_train_act, fitted_act))
    p_act_score = r2_score(y_test_act, preds_act)
    p_act_rmse = np.sqrt(mean_squared_error(y_test_act, preds_act))
    s_act_score = r2_score(y_test_sim, simmed)
    s_act_rmse = np.sqrt(mean_squared_error(y_test_sim, simmed))

    train_res_grouper = y_train_act.index.get_level_values(0)
    test_res_grouper = y_test_act.index.get_level_values(0)
    train_time_grouper = y_train_act.index.get_level_values(1)
    test_time_grouper = y_test_act.index.get_level_values(1)

    y_train_mean = y_train_act.groupby(train_res_grouper).mean()
    y_test_mean = y_test_act.groupby(test_res_grouper).mean()
    fmean = fitted_act.groupby(train_res_grouper).mean()
    pmean = preds_act.groupby(test_res_grouper).mean()
    smean = simmed.groupby(simmed.index.get_level_values(0)).mean()

    f_bias = fmean - y_train_mean
    f_bias_month = (
        fitted_act.groupby(train_time_grouper.month).mean()
        - y_train_act.groupby(train_time_grouper.month).mean()
    )
    p_bias = pmean - y_test_mean
    p_bias_month = (
        preds_act.groupby(test_time_grouper.month).mean()
        - y_test_act.groupby(test_time_grouper.month).mean()
    )
    s_bias = smean - y_test_mean
    s_bias_month = (
        simmed.groupby(simmed.index.get_level_values(1).month).mean()
        - y_test_sim.groupby(simmed.index.get_level_values(1).month).mean()
    )

    results = {
        "f_act_score": f_act_score,
        "f_act_rmse": f_act_rmse,
        "f_bias": f_bias,
        "f_bias_month": f_bias_month,
        "p_act_score": p_act_score,
        "p_act_rmse": p_act_rmse,
        "p_bias": p_bias,
        "p_bias_month": p_bias_month,
        "s_act_score": s_act_score,
        "s_act_rmse": s_act_rmse,
        "s_bias": s_bias,
        "s_bias_month": s_bias_month,
        "coefs": coefs,
    }

    train_data = pd.DataFrame(dict(actual=y_train_act, model=fitted_act))
    test_data = pd.DataFrame(dict(actual=y_test_act, model=preds_act))
    simmed_data = pd.DataFrame(dict(actual=y_test_sim, model=simmed))

    train_res_scores = pd.DataFrame(index=reservoirs, columns=["NSE", "RMSE"])
    test_res_scores = pd.DataFrame(index=reservoirs, columns=["NSE", "RMSE"])
    simmed_res_scores = pd.DataFrame(index=reservoirs, columns=["NSE", "RMSE"])

    train_res_scores["NSE"] = train_data.groupby(train_res_grouper).apply(
        lambda x: r2_score(x["actual"], x["model"])
    )
    train_res_scores["RMSE"] = train_data.groupby(train_res_grouper).apply(
        lambda x: mean_squared_error(x["actual"], x["model"], squared=False)
    )

    results["train_res_scores"] = train_res_scores

    test_res_scores["NSE"] = test_data.groupby(test_res_grouper).apply(
        lambda x: r2_score(x["actual"], x["model"])
    )
    test_res_scores["RMSE"] = test_data.groupby(test_res_grouper).apply(
        lambda x: mean_squared_error(x["actual"], x["model"], squared=False)
    )

    results["test_res_scores"] = test_res_scores
    simmed_res_grouper = simmed_data.index.get_level_values(0)
    simmed_res_scores["NSE"] = simmed_data.groupby(simmed_res_grouper).apply(
        lambda x: r2_score(x["actual"], x["model"])
    )
    simmed_res_scores["RMSE"] = simmed_data.groupby(simmed_res_grouper).apply(
        lambda x: mean_squared_error(x["actual"], x["model"], squared=False)
    )

    results["simmed_res_scores"] = simmed_res_scores

    # print(test_res_scores.to_markdown(floatfmt="0.3f"))
    # print(simmed_res_scores.to_markdown(floatfmt="0.3f"))

    train_quant, train_bins = pd.qcut(train_data["actual"], 3, labels=False, retbins=True)
    quant_scores = pd.DataFrame(index=[0, 1, 2], columns=["NSE", "RMSE"])
    train_data["bin"] = train_quant

    quant_scores["NSE"] = train_data.groupby("bin").apply(
        lambda x: r2_score(x["actual"], x["model"])
    )
    quant_scores["RMSE"] = train_data.groupby("bin").apply(
        lambda x: mean_squared_error(x["actual"], x["model"], squared=False)
    )

    test_quant, test_bins = pd.qcut(test_data["actual"], 3, labels=False, retbins=True)
    quant_scores = pd.DataFrame(index=[0, 1, 2], columns=["NSE", "RMSE"])
    test_data["bin"] = test_quant

    quant_scores["NSE"] = test_data.groupby("bin").apply(
        lambda x: r2_score(x["actual"], x["model"])
    )
    quant_scores["RMSE"] = test_data.groupby("bin").apply(
        lambda x: mean_squared_error(x["actual"], x["model"], squared=False)
    )

    # setup output parameters
    foldername = "tclr_model"
    int_mod = "" if month_intercepts else "_no_ints"
    all_mod = "_all_res" if use_all else ""
    assim_mod = f"_{args.assim}" if args.assim else ""
    foldername = foldername + int_mod + all_mod + f"_{max_depth}" + assim_mod + "_RT_MS"
    folderpath = pathlib.Path("..", "results", "basin_eval", basin, foldername)
    # check if the directory exists and handle it
    if folderpath.is_dir():
        # response = input(f"{folderpath} already exists. Are you sure you want to overwrite its contents? [y/N] ")
        response = "y"
        if response[0].lower() != "y":
            folderpath = pathlib.Path(
                "..",
                "results",
                "basin_eval",
                basin,
                "_".join([foldername, datetime.today().strftime("%Y%m%d_%H%M")]),
            )
            print(f"Saving at {folderpath} instead.")
            folderpath.mkdir()
    else:
        folderpath.mkdir(parents=True)

    # export tree to graphviz file so it can be converted nicely
    # rotate_tree = True if max_depth > 3 else e
    if make_dot:
        model.to_graphviz(folderpath / "tree.dot")

    # setup output container for modeling information
    X_train["storage_pre"] = X["storage_pre"]
    output = dict(
        **results,
        quant_scores=quant_scores,
        data=dict(
            X_train=X_train,
            test_data=test_data,
            train_data=train_data,
            simmed_data=simmed_data,  # groups=groups,
        ),
    )
    # write the output dict to a pickle file
    with open((folderpath / "results.pickle").as_posix(), "wb") as f:
        pickle.dump(output, f, protocol=4)

    # write the random effects to a csv file for easy access
    coefs.to_csv((folderpath / "random_effects.csv").as_posix())
    dot_command = [
        "dot",
        "-Tpng",
        (folderpath / "tree.dot").as_posix(),
        "-o",
        (folderpath / "tree.png").as_posix(),
    ]
    # subprocess.call(dot_command)


def simulate_tclr_model(
    model,
    model_or_beta,
    X_act,
    means,
    std,
    X_vars,
    reg_vars,
    lower_bounds,
    upper_bounds,
    assim,
    preds,
    X_test
):

    # I need to keep track of actual storage and release outputs
    # as well as rolling weekly mean storage and release outputs
    track_df = pd.DataFrame(
        columns=["release", "storage"] + list(X_act.columns), index=X_act.index
    )
    track_df[["const", "inflow", "inflow_roll7"]] = X_act[
        ["const", "inflow", "inflow_roll7"]
    ]

    # setup initial tracking information as well as find the start dates
    resers = track_df.index.get_level_values(0).unique()
    start_dates = {}
    idx = pd.IndexSlice
    for res in resers:
        rdf = X_act.loc[idx[res, :], :]
        # rolling 7 day means so we need 7 days of actual values
        first_seven = rdf.index.get_level_values(1).values[:7]
        track_df.loc[idx[res, first_seven], ["release_pre", "storage_pre"]] = X_act.loc[
            idx[res, first_seven], ["release_pre", "storage_pre"]
        ]
        start_dates[res] = first_seven[-1]

    # find the initial rolling release and storage values
    init_rolling = (
        track_df.groupby(track_df.index.get_level_values(0))[
            ["storage_pre", "release_pre"]
        ]
        .rolling(7)
        .mean()
        .dropna()
    )
    # get a duplicate reservoir index after the above command
    init_rolling.index = init_rolling.index.droplevel(0)

    # add init rolling values to track df
    track_df.loc[
        init_rolling.index, ["storage_roll7", "release_roll7"]
    ] = init_rolling.values

    # since all reservoirs have different temporal spans
    # we have to iterate through each reservoir independently
    from joblib import Parallel, delayed
   
    parallel = True 
    if parallel:
        outputdfs = Parallel(n_jobs=-1, verbose=11)(
            delayed(simul_reservoir)(
                res,
                model,
                model_or_beta,
                X_act,
                means,
                std,
                X_vars,
                reg_vars,
                lower_bounds,
                upper_bounds,
                assim,
                preds,
                X_test
            ) for res in resers
        )
    else:
        outputdfs = []
        for res in resers:
            outputdfs.append(
                simul_reservoir(
                    res,
                    model,
                    model_or_beta,
                    X_act,
                    means,
                    std,
                    X_vars,
                    reg_vars,
                    lower_bounds,
                    upper_bounds,
                    assim,
                    preds,
                    X_test
                )
            )
    return pd.concat(outputdfs)


def simul_reservoir(
    res,
    model,
    model_or_beta,
    track_df,
    means,
    std,
    X_loc_vars,
    reg_vars,
    lower_bounds,
    upper_bounds,
    assim=None,
    preds=None,
    X_test=None
):
    idx = pd.IndexSlice
    rdf = track_df.loc[idx[res, :], :].copy(deep=True)
    # cut off initial 6 values as we do not have a roling release for those
    dates = list(rdf.index.get_level_values(1).values)[6:]
    # need to identify the end date so we can know to stop adding values to track_df
    end_date = dates[-1]

    if assim == "weekly":
        assim_shift = 7
    elif assim == "monthly":
        assim_shift = 30
    elif assim == "seasonally":
        assim_shift = 90
    elif assim == "daily":
        assim_shift = 1

    start_date = dates[0]
    if "const" in reg_vars:
        cindex = reg_vars.index("const")
        reg_vars = reg_vars[:cindex] + reg_vars[cindex + 1:]
  
    for date in dates:
        # get values for today
        X_r = rdf.loc[idx[res, date], X_loc_vars]
        # add the interaction term
        X_r["storage_x_inflow"] = X_r["storage_pre"] * X_r["inflow"]
        # grab actual rt and max_sto values if they exist
        try:
            rts, max_sto = X_r[["rts", "max_sto"]]
            res_vars = True
        except KeyError:
            res_vars = False
        # standardize the values
        X_r = (X_r - means.loc[res, reg_vars]) / std.loc[res, reg_vars]
        # and the constant
        X_r["const"] = 1
        if res_vars:
            X_r["max_sto"] = max_sto
            X_r["rts"] = rts

        # reshape to a 2 d row vector
        if model_or_beta == "model":
            release = model.predict(X_r[model.feats].values.reshape(1, X_r.size))[0]
        else:
            release = X_r[X_loc_vars] @ model
        
        # if abs(release - preds.loc[idx[res, date]]) > 0.000001:
        #     print(res, date, release, preds.loc[idx[res, date]])
        # if date - np.timedelta64(7, "D"):
        #     sys.exit()
        # else:
        #     II()
        # get release back to actual space
        release_act = release * std.loc[res, "release"] + means.loc[res, "release"]
        # calculate storage from mass balance
        storage = (
            rdf.loc[idx[res, date], "storage_pre"]
            + rdf.loc[idx[res, date], "inflow"]
            - release_act
        )
        # keep storage and release within bounds
        if storage > upper_bounds.loc[res, "storage"]:
            storage = upper_bounds.loc[res, "storage"]
        elif storage < lower_bounds.loc[res, "storage"]:
            storage = lower_bounds.loc[res, "storage"]

        if release_act > upper_bounds.loc[res, "release"]:
            release_act = upper_bounds.loc[res, "release"]
        elif release_act < lower_bounds.loc[res, "release"]:
            release_act = lower_bounds.loc[res, "release"]

        # store calculated values
        rdf.loc[idx[res, date], "storage"] = storage
        rdf.loc[idx[res, date], "release"] = release_act

        # if we are not at the last day, store values needed for tomorrow
        if date != end_date:
            tomorrow = date + np.timedelta64(1, "D")
            prev_seven = pd.date_range(tomorrow - np.timedelta64(6, "D"), tomorrow)

            if assim:
                offset = ((date - start_date) / np.timedelta64(1, "D")) % assim_shift
                if offset == 0:
                    rdf.loc[idx[res, tomorrow], "storage_pre"] = track_df.loc[
                        idx[res, tomorrow], "storage_pre"
                    ]
                    rdf.loc[idx[res, tomorrow], "release_pre"] = track_df.loc[
                        idx[res, tomorrow], "release_pre"
                    ]
                else:
                    rdf.loc[idx[res, tomorrow], "storage_pre"] = storage
                    rdf.loc[idx[res, tomorrow], "release_pre"] = release_act
            else:
                rdf.loc[idx[res, tomorrow], "storage_pre"] = storage
                rdf.loc[idx[res, tomorrow], "release_pre"] = release_act
            # here we already updated the _pre values we can use the same logic to get rolling values
            rdf.loc[idx[res, tomorrow], "storage_roll7"] = rdf.loc[
                idx[res, prev_seven], "storage_pre"
            ].mean()
            rdf.loc[idx[res, tomorrow], "release_roll7"] = rdf.loc[
                idx[res, prev_seven], "release_pre"
            ].mean()

    return rdf


def parse_args(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Fit and test a TCLR model to specified data"
    )
    parser.add_argument(
        "basin",
        default=None,
        choices=["tva", "colorado", "pnw", "missouri", "all"],
        help="What basin should be modeled",
    )
    parser.add_argument(
        "-d",
        "--max_depth",
        dest="max_depth",
        default=3,
        type=int,
        help="How deep should the tree be allowed to go? (default=3)",
    )
    parser.add_argument(
        "-m",
        "--month_ints",
        dest="month_ints",
        action="store_true",
        default=False,
        help="Should monthly varying intercepts be included"
        "as regression variables? (default=False)",
    )
    parser.add_argument(
        "--assim",
        default=None,
        choices=("weekly", "monthly", "seasonally", "daily"),
        help="Frequency at which to assimilate observed storage and release values",
    )
    if arg_list:
        return parser.parse_args(arg_list)
    else:
        return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipeline(args)
