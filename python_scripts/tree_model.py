import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLMParams
from sklearn.metrics import r2_score
from sklearn.tree import (DecisionTreeRegressor,
                          plot_tree, export_graphviz)
from sklearn.ensemble import RandomForestRegressor
# import dtreeviz.trees as dtrees
from helper_functions import (read_tva_data, scale_multi_level_df,
                              read_all_res_data, find_max_date_range)
from simple_model import (change_group_names, combine_columns,
                          predict_mixedLM, forecast_mixedLM, print_coef_table)
from timing_function import time_function
from time import perf_counter as timer
from copy import deepcopy
from datetime import timedelta, datetime
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
        X = scaled_df.loc[:, ["Storage_pre", "Net Inflow", "Release_pre",
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
    train_index, test_index = split_train_test_dt(y.index, date=datetime(2010, 1, 1), level=0, keep=8)
    # train_sto = df.loc[train_index,"Storage"]
    # test_sto = df.loc[test_index,"Storage"]
    # train_sto_scaled, train_sto_mean, train_sto_sd = scale_multi_level_df(train_sto)
    # test_sto_scaled, test_sto_mean, test_sto_sd = scale_multi_level_df(test_sto)

    
    actual_inflow_train = df["Net Inflow"].loc[train_index]
    actual_inflow_test = df["Net Inflow"].loc[test_index]
    # II()
    # sys.exit()
    # set exogenous variables
    X_vars = ["Storage_pre", "Release_pre", "Net Inflow",
              "Storage_Inflow_interaction",
              "Inflow_roll7", "Release_roll7", #"Storage_roll7",
              #"Storage_7", "Release_7",
            #   "NaturalOnly", "RunOfRiver"
              ]

    X = X.loc[:, X_vars]

    reservoirs = X.index.get_level_values(1).unique()
    print("These reservoirs are being modeled:")
    print("\n".join(reservoirs))
    # response = input("Is this what you want? [Y/n] ") or "y"
    # if response.lower() != "y":
    #     sys.exit()

  
    # split into training and testing sets
    X_train = X.loc[train_index,:]
    X_test = X.loc[test_index,:]
    y_train = y.loc[train_index]
    y_test = y.loc[test_index]

    # X_train_scaled, X_train_means, X_train_sd = scale_multi_level_df(X_train)
    # X_test_scaled, X_test_means, X_test_sd = scale_multi_level_df(X_test)
    # y_train_scaled, y_train_means, y_train_sd = scale_multi_level_df(y_train)
    # y_test_scaled, y_test_means, y_test_sd = scale_multi_level_df(y_test)
    
    # train_means = X_train_means.join(y_train_means)
    # train_sd = X_train_sd.join(y_train_sd)
    # test_means = X_test_means.join(y_test_means)
    # test_sd = X_test_sd.join(y_test_sd)
    # test_means = test_means.join(test_sto_mean)
    # train_means = train_means.join(train_sto_mean)
    # test_sd = test_sd.join(test_sto_sd)
    # train_sd = train_sd.join(train_sto_sd)
    # II()
    # sys.exit()

    # X_train = X_train_scaled
    # X_test = X_test_scaled
    # y_train = y_train_scaled
    # y_test = y_test_scaled

    # fit the decision tree model
    max_depth=3
    #, splitter="best" - for decision_tree
    tree = tree_model(X_train, y_train, tree_type="decision", max_depth=max_depth,
                      random_state=37)
    leaves, groups = get_leaves_and_groups(X_train, tree)

    # fit the sub_tree ml model
    # X_train = X_train.drop(["NaturalOnly", "RunOfRiver"], axis=1)
    ml_model = sub_tree_multi_level_model(X_train, y_train, tree)

    fitted = ml_model.fittedvalues

    # predict and forecast from the sub_tree model

    preds, pgroups = predict_from_sub_tree_model(X_test, y_test, tree, ml_model)
    forecasted, fgroups = predict_from_sub_tree_model(X_test, y_test, tree, ml_model, 
                                            #  forecast=True, means=test_means, std=test_sd,
                                            forecast=True, means=means,std=std,
                                            timelevel=timelevel, actual_inflow=actual_inflow_test)
    forecasted = forecasted[forecasted.index.get_level_values(0).year >= 2010]

    # get all variables back to original space
    preds = preds.unstack()
    fitted = fitted.unstack()
    y_train = y_train.unstack()
    y_test = y_test.unstack()
    # II()
    try:
        y_test.columns = y_test.columns.get_level_values(1)
        y_train.columns = y_train.columns.get_level_values(1)
    except IndexError as e:
        pass
    
    train_means = means
    test_means = means
    train_sd = std
    test_sd = std
    if timelevel == "all":
        preds_act = (preds * test_sd["Release"] + test_means["Release"]).stack()
        fitted_act = (fitted * train_sd["Release"] + train_means["Release"]).stack()
        y_train_act = (y_train * train_sd["Release"] + train_means["Release"]).stack()
        y_test_act = (y_test * test_sd["Release"] + test_means["Release"]).stack()
    else:
        rel_means_test = test_means["Release"].unstack()
        rel_std_test = test_sd["Release"].unstack()
        rel_means_train = train_means["Release"].unstack()
        rel_std_train = train_sd["Release"].unstack()
        preds_act = deepcopy(preds)
        fitted_act = deepcopy(fitted)
        y_train_act = deepcopy(y_train)
        y_test_act = deepcopy(y_test)
        extent = 13 if timelevel == "month" else 4
        for tl in range(1,extent):
            ploc = getattr(preds.index, timelevel) == tl
            floc = getattr(fitted.index, timelevel) == tl
            preds_act.loc[ploc] = preds_act.loc[ploc] * rel_std_test.loc[tl] + rel_means_test.loc[tl]
            fitted_act.loc[floc] = fitted_act.loc[floc] * rel_std_train.loc[tl] + rel_means_train.loc[tl]
            y_train_act.loc[floc] = y_train_act.loc[floc] * rel_std_train.loc[tl] + rel_means_train.loc[tl]
            y_test_act.loc[ploc] = y_test_act.loc[ploc] * rel_std_test.loc[tl] + rel_means_test.loc[tl]
        preds_act = preds_act.stack()
        fitted_act = fitted_act.stack()
        y_train_act = y_train_act.stack()
        y_test_act = y_test_act.stack()
    forecasted_act = forecasted["Release_act"]


    # report scores for the current model run
    try:
        preds_score = r2_score(y_test_act, preds_act)
    except ValueError as e:
        II()
    y_test_act = y_test_act.loc[forecasted_act.index]
    fit_score = r2_score(y_train_act, fitted_act)
    try:
        forecasted_score = r2_score(y_test_act, forecasted_act)
    except ValueError as e:
        forecasted_score = np.nan

    score_strings = [f"{ftype:<12} = {score:.3f}" for ftype, score in zip(
                    ["Fit", "Forecast", "Preds"], [fit_score, forecasted_score, preds_score])]
    print("Scores:")
    print("\n".join(score_strings))

    # setup output parameters
    if timelevel == "all":
        prepend = ""
    else:
        prepend = f"{timelevel}_"
    foldername = f"{prepend}upstream_basic_td{max_depth:d}_roll7_new"
    folderpath = pathlib.Path("..", "results", "treed_ml_model", foldername)
    
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
        re_coefs=ml_model.random_effects,
        fe_coefs=ml_model.params,
        cov_re=ml_model.cov_re,
        data=dict(
            X_test=X_test,
            y_test=y_test,
            X_train=X_train,
            y_train=y_train,
            fitted=fitted_act,
            y_test_act=y_test_act,
            y_train_act=y_train_act,
            predicted_act=preds_act,
            groups=fgroups,
            forecasted=forecasted[["Release", "Storage",
                                   "Release_act", "Storage_act"]]
        )
    )
    # write the output dict to a pickle file
    with open((folderpath / "results.pickle").as_posix(), "wb") as f:
        pickle.dump(output, f, protocol=4)
    
    # write the random effects to a csv file for easy access
    pd.DataFrame(ml_model.random_effects).to_csv(
        (folderpath / "random_effects.csv").as_posix())

if __name__ == "__main__":
    pipeline()
    mem_usage = psutil.Process().memory_info().peak_wset
    print(f"Max Memory Used: {mem_usage/1000/1000:.4f} MB")
