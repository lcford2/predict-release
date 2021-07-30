import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLMParams
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.tree import (DecisionTreeRegressor,
                          plot_tree, export_graphviz)
from sklearn.ensemble import RandomForestRegressor
# import dtreeviz.trees as dtrees
from helper_functions import (read_tva_data, scale_multi_level_df,
                              read_all_res_data, find_max_date_range)
from simple_model import (change_group_names, combine_columns,
                          predict_mixedLM, forecast_mixedLM, print_coef_table,
                          fit_release_and_storage, forecast_mixedLM_new)
from timing_function import time_function
from time import perf_counter as timer
from copy import deepcopy
from datetime import timedelta, datetime
from collections import defaultdict
from itertools import combinations
from IPython import embed as II
import calendar
import sys
import pickle
import pathlib
import subprocess
import psutil

# group name map to numeric variables
group_names = {
    "RunOfRiver": {0: "StorageDam", 1: "RunOfRiver"},
    "NaturalOnly": {0: "ComboFlow", 1: "NaturalFlow"},
    "PrimaryType": {0: "FloodControl",
                    1: "Hydropower",
                    2: "Irrigation",
                    3: "Navigation",
                    4: "WaterSupply"}
}

# flop the group name dict for filtering
inverse_groupnames = {key: {label: idx for idx, label in value.items()}
                      for key, value in group_names.items()}

@time_function
def prep_data(df, groups, filter_groups=None, scaler=None, timelevel="all"):
    # store the groups so we can add them back after scaling
    for_groups = df.loc[:, groups]
    # create an interactoin term between 
    df["Storage_Inflow_interaction"] = df.loc[:, "Storage_pre"].mul(df.loc[:, "Net Inflow"])

    change_names = ["Douglas", "Cherokee", "Hiwassee"]
    for ch_name in change_names:
        size = df[df.index.get_level_values(1) == ch_name].shape[0]
        df.loc[df.index.get_level_values(1) == ch_name, "NaturalOnly"] = [
            1 for i in range(size)]

    if filter_groups:
        for key, value in filter_groups.items():
            df = df.loc[df[key] == inverse_groupnames[key][value], :]

    if scaler == "mine":
        scaled_df, means, std = scale_multi_level_df(df, timelevel)
        X = scaled_df.loc[:, ["Storage", "Storage_pre", "Net Inflow", "Release_pre",
                                "Storage_Inflow_interaction",
                                "Release_roll7", "Inflow_roll7", "Storage_roll7",
                                "Release_7", "Storage_7"
                                #"Storage_roll14", "Release_roll7", "Inflow_roll14"
                                ]]
        y = scaled_df.loc[:, "Release"]

    else:
        X = df.loc[:, ["Storage_pre", "Net Inflow", "Release_pre",
                    "Storage_Inflow_interaction",
                    "Release_roll7", "Inflow_roll7", "Storage_roll7",
                    #"Storage_roll14", "Release_roll7", "Inflow_roll14"
                   ]]

        y = df.loc[:,"Release"]
        # give means of 0 and std of 1 so I do not have to write conditionals throughout
        # for unscaled products
        means = pd.DataFrame(0, index=y.unstack().index, columns=y.unstack().columns)
        std = pd.DataFrame(1, index=y.unstack().index, columns=y.unstack().columns)

    # add back the groups 
    X[groups] = for_groups

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

@time_function
def tree_model(X, y, tree_type="decision", **tree_args):
    if tree_type == "decision":
        tree = DecisionTreeRegressor(**tree_args)
    elif tree_type == "ensemble":
        def_args = dict(
            max_depth=4,
            min_samples_leaf=100,
            bootstrap=True,
            n_jobs=-1,
        )
        for key, value in tree_args.items():
            def_args[key] = value
        tree = RandomForestRegressor(**def_args)
    else:
        raise ValueError("Invalid tree_type provided. Must be either 'Decision' or 'Extra'.")
    
    tree.fit(X, y)
    return tree

def get_leaves_and_groups(X, tree):
    # use the tree to get what leaves correpond with each entry 
    # in the X matrix
    leaves = tree.apply(X)
    # make those leaves into a pandas series for the ml model
    if leaves.ndim == 2:
        groups = pd.DataFrame(leaves, columns=range(
            1, leaves.shape[1] + 1), index=X.index)
    else:
        groups = pd.Series(leaves, index=X.index)
    return leaves, groups


# @time_function
def sub_tree_multi_level_model(X, y, tree=None, groups=None, my_id=None):
    if tree:
        leaves, groups = get_leaves_and_groups(X, tree)
    elif not groups:
        raise ValueError("Must provide either a tree or groups.")
    else:
        groups = groups[my_id]
        
    mexog = pd.DataFrame(np.ones((y.size, 1)), index=y.index,  columns=["const"])
    free = MixedLMParams.from_components(fe_params=np.ones(mexog.shape[1]),
                                         cov_re=np.eye(X.shape[1]))
    try:
        X = X.drop(["NaturalOnly", "RunOfRiver"], axis=1)
    except KeyError as e:
        pass
    md = sm.MixedLM(y, mexog, groups=groups, exog_re=X)
    mdf = md.fit()
    return mdf


def predict_from_sub_tree_model(X, y, tree, ml_model, forecast=False, means=None, std=None, timelevel="all", actual_inflow=None):
    # means and std required if forecast is True
    if forecast:
        leaves, groups = get_leaves_and_groups(X.drop("Storage_pre_act", axis=1), tree)
    else:
        leaves, groups = get_leaves_and_groups(X, tree)


    mexog = pd.DataFrame(np.ones((y.size, 1)), index=y.index, columns=["const"])
    exog_re = deepcopy(X)
    exog_re["group"] = groups
    try:
        exog_re = exog_re.drop(["NaturalOnly", "RunOfRiver"], axis=1)
    except KeyError as e:
        pass

    if forecast:
        if not isinstance(means, pd.DataFrame) or not isinstance(std, pd.DataFrame):
            raise ValueError("If forecast=True, means and std must be provided.")
 
        preds = forecast_mixedLM(
                ml_model.params,
                ml_model.random_effects,
                mexog,
                exog_re,
                means, 
                std,
                "group",
                actual_inflow,
                timelevel,
                tree=True
            )
    else:
        preds = predict_mixedLM(
                ml_model.params,
                ml_model.random_effects,
                mexog,
                exog_re,
                "group"
            )
    
    return preds, groups

def pipeline():
    df = read_tva_data()
    groups = ["NaturalOnly", "RunOfRiver"]
    # filter groups {group to filter by: attribute of entries that should be included}
    filter_groups={"NaturalOnly":"NaturalFlow"}
    # filter_groups={"RunOfRiver":"StorageDam"}
    # filter_groups = {}
    # get the X matrix, and y vector along with means and std.
    # note: if no scaler is provided, means and std will be 0 and 1 respectively.
    timelevel="all"
    
    X,y,means,std = prep_data(df, groups, filter_groups, scaler="mine", timelevel=timelevel)
    X["rel_diff"] = X["Release_pre"] - X["Release_roll7"]
    X["sto_diff"] = X["Storage_pre"] - X["Storage_roll7"]
    X["inf_diff"] = X["Net Inflow"] - X["Inflow_roll7"]
    # train_index, test_index = split_train_test_dt(y.index, date=datetime(2010, 1, 1), level=0, keep=8)
    reservoirs = X.index.get_level_values(1).unique()
    #rts = pd.read_pickle("../pickles/tva_res_times.pickle")
    #X["RT"] = [rts.loc[i] for i in X.index.get_level_values(1)]
    #lvoneout_results = {}
    lvout_rt_results = {}

    lvout_sets = [
        [], # baseline
        ["Douglas", "Hiwassee", "Cherokee"], # < 100 days
        ["BlueRidge", "Fontana", "Nottely"], # 100 - 150 days
        ["Norris", "Chatuge"], # 150 - 200 days
        ["TimsFord", "SHolston", "Watauga"] # > 200 days
    ]

    lvout_labels = [
        "baseline",
        "100",
        "100-150",
        "150-200",
        "200"
    ]

    #for res in reservoirs:
    #for lvout_set, label in zip(lvout_sets, lvout_labels):
    from simple_model import filter_on_corr
    rts = pd.read_pickle("../pickles/tva_res_times.pickle")
    corrs = pd.read_pickle("../pickles/tva_release_corrs.pickle")

    exclude = defaultdict(list)
    # for level in np.arange(0.1, 0.7, 0.1)[::-1]:
    for level in [0.3]:
        for lvout_res, label in zip(lvout_sets, lvout_labels):
            leave_out = filter_on_corr(level, lvout_res, corrs, rts)
            exclude[level].extend(leave_out)
    
    # for i, res in enumerate(reservoirs):
        # label = str(i)
        # lvout_set = reservoirs[:i]
    for level, lvout_set in exclude.items():
        label = str(round(level,2))
        train_index, test_index = split_train_test_res(y.index, lvout_set)
        print(f"Solving model run {label}")

        # set exogenous variables
        X_vars = ["Storage_pre", "Release_pre", "Net Inflow",
                "Storage_Inflow_interaction",
                "Inflow_roll7", "Release_roll7", "Storage_roll7",
                #"Storage_7", "Release_7",
                #   "NaturalOnly", "RunOfRiver"
                ]

        X_vars = ["sto_diff", "Storage_Inflow_interaction",
                "Release_pre", "Release_roll7",
                "Net Inflow", "Inflow_roll7"]
    
        # split into training and testing sets
        X_train = X.loc[train_index,X_vars]
        X_test = X.loc[test_index,X_vars]
        X_test_all = X.loc[:,X_vars]
        y_train_rel = y.loc[train_index]
        y_test_rel = y.loc[test_index]
        y_train_sto = X.loc[train_index, "Storage"]
        y_test_sto = X.loc[test_index, "Storage"]

        train_res = X_train.index.get_level_values(1).unique()
        test_res = X_test.index.get_level_values(1).unique()

        # fit the decision tree model
        max_depth=3
        #, splitter="best" - for decision_tree
        tree = tree_model(X_train, y_train_rel, tree_type="decision", max_depth=max_depth,
                        random_state=37)
        leaves, groups = get_leaves_and_groups(X_train, tree)

        # # fit the sub_tree ml model
        # X_train = X_train.drop(["NaturalOnly", "RunOfRiver"], axis=1)

        ml_model = sub_tree_multi_level_model(X_train, y_train_rel, tree)
        
        coefs = ml_model.random_effects
        fitted = ml_model.fittedvalues

        y_train_rel_act = (y_train_rel.unstack() *
                        std.loc[train_res,"Release"] + means.loc[train_res,"Release"]).stack()
        y_train_sto_act = (y_train_sto.unstack() *
                        std.loc[train_res,"Storage"] + means.loc[train_res,"Storage"]).stack()
        y_test_rel_act = (y_test_rel.unstack() *
                        std.loc[test_res,"Release"] + means.loc[test_res,"Release"]).stack()
        y_test_sto_act = (y_test_sto.unstack() *
                        std.loc[test_res,"Storage"] + means.loc[test_res,"Storage"]).stack()

        fitted_act = (fitted.unstack() *
                          std.loc[train_res, "Release"] + means.loc[train_res, "Release"]).stack()

        f_act_score = r2_score(y_train_rel_act, fitted_act)
        f_norm_score = r2_score(y_train_rel, fitted)
        f_act_rmse = np.sqrt(mean_squared_error(y_train_rel_act, fitted_act))
        f_norm_rmse = np.sqrt(mean_squared_error(y_train_rel, fitted))
        
        y_train_mean = y_train_rel_act.groupby(
            y_train_rel_act.index.get_level_values(1)
        ).mean()
        y_test_mean = y_test_rel_act.groupby(
            y_test_rel_act.index.get_level_values(1)
        ).mean()
        fmean = fitted_act.groupby(
            fitted_act.index.get_level_values(1)
        ).mean()

        f_bias = fmean - y_train_mean
        f_bias_month = fitted_act.groupby(
            fitted_act.index.get_level_values(0).month
            ).mean() - y_train_rel_act.groupby(
                y_train_rel_act.index.get_level_values(0).month).mean()

        if label == "0":
            p_act_score = f_act_score
            p_norm_score = f_norm_score
            p_act_rmse = f_act_rmse
            p_norm_rmse = f_norm_rmse
            p_act_score_all = f_act_score
            p_act_rmse_all = f_act_rmse
            p_bias = f_bias
            p_bias_month = f_bias_month
        else:
            preds, pgroups = predict_from_sub_tree_model(X_test, y_test_rel, tree, ml_model)
            preds_all, pgroups_all = predict_from_sub_tree_model(
                X_test_all, y, tree, ml_model
            )

            preds_act = (preds.unstack() *
                         std.loc[test_res, "Release"] + means.loc[test_res, "Release"]).stack()
            preds_act_all = (preds_all.unstack() *
                         std.loc[:, "Release"] + means.loc[:, "Release"]).stack()
            
            preds_mean = preds_act.groupby(
                preds_act.index.get_level_values(1)
            ).mean()
            
            preds_mean_all = preds_act_all.groupby(
                preds_act_all.index.get_level_values(1)
            ).mean()
            
            p_act_score = r2_score(y_test_rel_act, preds_act)
            p_norm_score = r2_score(y_test_rel, preds)
            p_act_rmse = np.sqrt(mean_squared_error(y_test_rel_act, preds_act))
            p_norm_rmse = np.sqrt(mean_squared_error(y_test_rel, preds))
            
            y_act = (y.unstack() * std["Release"] + means["Release"]).stack()
            p_act_score_all = r2_score(
                (y.unstack() * std["Release"] + means["Release"]).stack(),
                preds_act_all
            )
            p_act_rmse_all = np.sqrt(mean_squared_error(
                (y.unstack() * std["Release"] + means["Release"]).stack(),
                preds_act_all
            ))
            p_bias = preds_mean - y_test_mean
            p_bias_month = preds_act.groupby(
                preds_act.index.get_level_values(0).month  
                ).mean() - y_test_rel_act.groupby(
                    y_test_rel_act.index.get_level_values(0).month).mean()


        lvout_rt_results[label] = {
            "pred":{
                "p_act_score": p_act_score,
                "p_act_rmse": p_act_rmse,
                "p_act_score_all":p_act_score_all,
                "p_act_rmse_all":p_act_rmse_all,
                "p_bias":p_bias,
                "p_bias_month":p_bias_month
            },
            "fitted": {
                "f_act_score": f_act_score,
                "f_act_rmse": f_act_rmse,
                "f_bias": f_bias,
                "f_bias_month": f_bias_month
            },
            "coefs":coefs,
            "test_res":test_res,
            "train_res":train_res
        }

        fitted_act = fitted_act.unstack()
        y_train_act = y_train_rel_act.unstack()

        res_scores = pd.DataFrame(index=reservoirs, columns=["NSE", "RMSE"])

        if label != "0":
            preds_act = preds_act.unstack()
            y_test_act = y_test_rel_act.unstack()

        for res in reservoirs:
            try:
                ya = y_train_act[res]
                ym = fitted_act[res]
            except KeyError as e:
                ya = y_test_act[res]
                ym = preds_act[res]
            res_scores.loc[res, "NSE"] = r2_score(ya, ym)
            res_scores.loc[res, "RMSE"] = np.sqrt(mean_squared_error(ya, ym))

        lvout_rt_results[label]["res_scores"] = res_scores

    try:
        pred_df = pd.DataFrame({k:v["pred"] for k,v in lvout_rt_results.items()})
        fitt_df = pd.DataFrame({k:v["fitted"] for k,v in lvout_rt_results.items()})

        lvout_rt_results["pred"] = pred_df
        lvout_rt_results["fitted"] = fitt_df
        train_data = pd.DataFrame(dict(actual=y_train_rel_act, model=fitted_act.stack()))
        test_data = pd.DataFrame(dict(actual=y_test_rel_act, model=preds_act.stack()))
        # with open("../results/synthesis/treed_model/leave_corr_out.pickle", "wb") as f:
        #    pickle.dump(lvout_rt_results, f)
        II()
        sys.exit()
    except:
        II()
        sys.exit()

    # coefs = {g: np.mean(fit_results["params"][g], axis=0)
    #          for g in groups.unique()}
    # coefs = pd.DataFrame(coefs, index=X_train.columns)

    # test_leaves, test_groups = get_leaves_and_groups(X_test, tree)

    # predict and forecast from the sub_tree model
    # preds, pgroups = predict_from_sub_tree_model(X_test, y_test, tree, ml_model)
    # pred_result = fit_release_and_storage(y_test_rel, y_test_sto, X_test, test_groups, means, std,
                                        #   init_values=coefs, niters=0)

    # idx = pd.IndexSlice
    # X_test.loc[idx[datetime(2010, 1, 1), resers],"Storage_pre_act"] = df.loc[idx[datetime(2010, 1, 1), resers],"Storage_pre"]
    
    # forecasted, fgroups = predict_from_sub_tree_model(X_test, y_test, tree, ml_model, 
    #                                         #  forecast=True, means=test_means, std=test_sd,
    #                                         forecast=True, means=means,std=std,
    #                                         timelevel=timelevel, actual_inflow=actual_inflow_test)
    X_test["group"] = test_groups
    forecasted = forecast_mixedLM_new(coefs, X_test, means, std, "group", actual_inflow_test)
    forecasted = forecasted[forecasted.index.get_level_values(0).year >= 2010]


    # get all variables back to original space
    preds_rel = pred_result["f_rel_act"][0].unstack()
    preds_sto = pred_result["f_sto_act"][0].unstack()
    fc_rel = forecasted.loc[:, "Release_act"].unstack()
    fc_sto = forecasted.loc[:, "Storage_act"].unstack()
    # fitted = fitted.unstack()
    # y_train = y_train.unstack()
    # y_test = y_test.unstack()
    # II()
    # try:
    #     y_test.columns = y_test.columns.get_level_values(1)
    #     y_train.columns = y_train.columns.get_level_values(1)
    # except IndexError as e:
    #     pass
    
    # train_means = means
    # test_means = means
    # train_sd = std
    # test_sd = std
    # if timelevel == "all":
    #     preds_act = (preds * test_sd["Release"] + test_means["Release"]).stack()
    #     fitted_act = (fitted * train_sd["Release"] + train_means["Release"]).stack()
    #     y_train_act = (y_train * train_sd["Release"] + train_means["Release"]).stack()
    #     y_test_act = (y_test * test_sd["Release"] + test_means["Release"]).stack()
    # else:
    #     rel_means_test = test_means["Release"].unstack()
    #     rel_std_test = test_sd["Release"].unstack()
    #     rel_means_train = train_means["Release"].unstack()
    #     rel_std_train = train_sd["Release"].unstack()
    #     preds_act = deepcopy(preds)
    #     fitted_act = deepcopy(fitted)
    #     y_train_act = deepcopy(y_train)
    #     y_test_act = deepcopy(y_test)
    #     extent = 13 if timelevel == "month" else 4
    #     for tl in range(1,extent):
    #         ploc = getattr(preds.index, timelevel) == tl
    #         floc = getattr(fitted.index, timelevel) == tl
    #         preds_act.loc[ploc] = preds_act.loc[ploc] * rel_std_test.loc[tl] + rel_means_test.loc[tl]
    #         fitted_act.loc[floc] = fitted_act.loc[floc] * rel_std_train.loc[tl] + rel_means_train.loc[tl]
    #         y_train_act.loc[floc] = y_train_act.loc[floc] * rel_std_train.loc[tl] + rel_means_train.loc[tl]
    #         y_test_act.loc[ploc] = y_test_act.loc[ploc] * rel_std_test.loc[tl] + rel_means_test.loc[tl]
    #     preds_act = preds_act.stack()
    #     fitted_act = fitted_act.stack()
    #     y_train_act = y_train_act.stack()
    #     y_test_act = y_test_act.stack()
    # forecasted_act = forecasted["Release_act"]


    # report scores for the current model run
    try:
        preds_score_rel = r2_score(y_test_rel_act, preds_rel.stack())
        preds_score_sto = r2_score(y_test_sto_act, preds_sto.stack())
    except ValueError as e:
        II()
    y_test_rel_act = y_test_rel_act.loc[fc_rel.index]
    y_test_sto_act = y_test_sto_act.loc[fc_sto.index]

    fit_score_rel = r2_score(y_train_rel_act, fitted_rel)
    fit_score_sto = r2_score(y_train_sto_act, fitted_sto)
    try:
        fc_score_rel = r2_score(y_test_rel_act, fc_rel.stack())
        fc_score_sto = r2_score(y_test_sto_act, fc_sto.stack())
    except ValueError as e:
        fc_score_rel = np.nan
        fc_score_sto = np.nan

    score_strings = [f"{ftype:<12} = {score:.3f}" for ftype, score in zip(
                    ["Fit", "Forecast", "Preds"], [fit_score_rel, fc_score_rel, preds_score_rel])]
    print("Release Scores:")
    print("\n".join(score_strings))

    score_strings = [f"{ftype:<12} = {score:.3f}" for ftype, score in zip(
                    ["Fit", "Forecast", "Preds"], [fit_score_sto, fc_score_sto, preds_score_sto])]
    print("Storage Scores:")
    print("\n".join(score_strings))

    # setup output parameters
    if timelevel == "all":
        prepend = ""
    else:
        prepend = f"{timelevel}_"
    foldername = f"{prepend}upstream_basic_td{max_depth:d}_roll7"
    folderpath = pathlib.Path("..", "results", "treed_ml_model_dual_fit", foldername)
    
    # check if the directory exists and handle it
    if folderpath.is_dir():
        # response = input(f"{folderpath} already exists. Are you sure you want to overwrite its contents? [y/N] ")
        response = "y"
        if response[0].lower() != "y":
            folderpath = pathlib.Path(
                "..", "results", "treed_ml_model", 
                "_".join([foldername, datetime.today().strftime("%Y%m%d_%H%M")]))
            print(f"Saving at {folderpath} instead.")
            folderpath.mkdir()
    else:
        folderpath.mkdir()
    
    # export tree to graphviz file so it can be converted nicely
    # rotate_tree = True if max_depth > 3 else False

    export_graphviz(tree, out_file=(folderpath / "tree.dot").as_posix(),
                    feature_names=X_vars, filled=True, proportion=True, rounded=True,
                    special_characters=True)

    # setup output container for modeling information
    output = dict(
        # re_coefs=ml_model.random_effects,
        # fe_coefs=ml_model.params,
        # cov_re=ml_model.cov_re,
        coefs=coefs,
        data=dict(
            X_test=X_test,
            # y_test=y_test,
            X_train=X_train,
            # y_train=y_train,
            fitted_rel=fitted_rel,
            fitted_sto=fitted_sto,
            y_test_rel_act=y_test_rel_act,
            y_train_rel_act=y_train_rel_act,
            y_test_sto_act=y_test_sto_act,
            y_train_sto_act=y_train_sto_act,
            predicted_act_rel=preds_rel.stack(),
            predicted_act_sto=preds_sto.stack(),
            groups=test_groups,
            forecasted=forecasted[["Release", "Storage",
                                   "Release_act", "Storage_act"]]
        )
    )
    # write the output dict to a pickle file
    with open((folderpath / "results.pickle").as_posix(), "wb") as f:
        pickle.dump(output, f, protocol=4)
    
    # write the random effects to a csv file for easy access
    # pd.DataFrame(ml_model.random_effects).to_csv(
    #     (folderpath / "random_effects.csv").as_posix())
    coefs.to_csv((folderpath/"random_effects.csv").as_posix())

if __name__ == "__main__":
    pipeline()
    # mem_usage = psutil.Process().memory_info().peak_wset
    # print(f"Max Memory Used: {mem_usage/1000/1000:.4f} MB")
    # from resource import getrusage, RUSAGE_SELF
    # print(getrusage(RUSAGE_SELF).ru_maxrss)
