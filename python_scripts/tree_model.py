# import base packages for fs ops, calendar, serialization, and data management
import calendar
import sys
import pickle
import pathlib
import subprocess
import psutil
import pandas as pd
import numpy as np
from copy import deepcopy
from datetime import timedelta, datetime
from collections import defaultdict
from itertools import combinations
from IPython import embed as II

# import modeling packages
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLMParams
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.tree import (DecisionTreeRegressor,
                          plot_tree, export_graphviz)
from sklearn.ensemble import RandomForestRegressor

# import my functions for this project
from utils.helper_functions import (read_tva_data, scale_multi_level_df,
                              read_all_res_data, find_max_date_range)
from simple_model import (change_group_names, combine_columns,
                          predict_mixedLM, forecast_mixedLM, print_coef_table,
                          fit_release_and_storage, forecast_mixedLM_new)

# timing functions
from utils.timing_function import time_function
from time import perf_counter as timer

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
    df["sto_diff_pre"] = df["Storage_pre"] - df["Storage_roll7"]
    

    # these reservoirs are high RT res, original grouping
    # was based on inflow types so they need to be swapped
    change_names = ["Douglas", "Cherokee", "Hiwassee"]
    for ch_name in change_names:
        size = df[df.index.get_level_values(1) == ch_name].shape[0]
        df.loc[df.index.get_level_values(1) == ch_name, "NaturalOnly"] = [
            1 for i in range(size)]

    # if we only want to model a single group
    if filter_groups:
        for key, value in filter_groups.items():
            df = df.loc[df[key] == inverse_groupnames[key][value], :]

    # standardize variables x = (X - mu) / SD
    if scaler == "mine":
        scaled_df, means, std = scale_multi_level_df(df, timelevel)
        X = scaled_df.loc[:, [
            "Storage", "Storage_pre", "Net Inflow", "Release_pre",
            "Storage_Inflow_interaction",
            "Release_roll7", "Inflow_roll7", "Storage_roll7",
            "Release_7", "Storage_7", "sto_diff_pre"
            ]
        ]
                        
        y = scaled_df.loc[:, "Release"]

    else:
        X = df.loc[:, [
            "Storage_pre", "Net Inflow", "Release_pre",
            "Storage_Inflow_interaction",
            "Release_roll7", "Inflow_roll7", "Storage_roll7",
            ]
        ]

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
    # II()
    leaves = tree.apply(X)
    # make those leaves into a pandas series for the ml model
    if leaves.ndim == 2:
        groups = pd.DataFrame(leaves, columns=range(
            1, leaves.shape[1] + 1), index=X.index)
    else:
        groups = pd.Series(leaves, index=X.index)
    return leaves, groups


# @time_function
def sub_tree_multi_level_model(X, y, tree_vars, tree=None, groups=None, my_id=None):
    if tree:
        X_tree = X.loc[:,tree_vars]
        leaves, groups = get_leaves_and_groups(X_tree, tree)
    elif not groups:
        raise ValueError("Must provide either a tree or groups.")
    else:
        groups = groups[my_id]
    if "st_frac" in X.columns:
        X = X.drop("st_frac", axis=1)
    # setup constants
    mexog = pd.DataFrame(np.ones((y.size, 1)), index=y.index,  columns=["const"])
    X["const"] = mexog["const"]
    # setup covariance structure for ml model
    free = MixedLMParams.from_components(fe_params=np.ones(mexog.shape[1]),
                                         cov_re=np.eye(X.shape[1]))
    
    # we do not want these columns, but an error is thrown 
    # if we drop and they do not exist
    try:
        X = X.drop(["NaturalOnly", "RunOfRiver"], axis=1)
    except KeyError as e:
        pass
    # setup model and return fitted instance
    md = sm.MixedLM(y, mexog, groups=groups, exog_re=X)
    return md.fit()


def predict_from_sub_tree_model(X, y, tree, ml_model, forecast=False, means=None, std=None, timelevel="all", actual_inflow=None, tree_vars=None):
    # wrap prediction for sub tree model
    # means and std required if forecast is True
    # first thing to do is get the proper groupings for the exogogenous (independent) variables
    if tree_vars == None:
        tree_vars = X.columns
    if forecast:
        leaves, groups = get_leaves_and_groups(X.drop("Storage_pre_act", axis=1)[tree_vars], tree)
    else:
        leaves, groups = get_leaves_and_groups(X[tree_vars], tree)

    # create exog dataframe for predictoin
    mexog = pd.DataFrame(np.ones((y.size, 1)), index=y.index, columns=["const"])
    exog_re = deepcopy(X)
    exog_re["group"] = groups
    exog_re["const"] = mexog["const"]
    try:
        exog_re = exog_re.drop(["NaturalOnly", "RunOfRiver"], axis=1)
    except KeyError as e:
        pass

    if forecast:
        # Cannot forecast withour standardization variables because storage must be calculated
        # using the mass balance in normal space
        if not isinstance(means, pd.DataFrame) or not isinstance(std, pd.DataFrame):
            raise ValueError("If forecast=True, means and std must be provided.")
        # step through time for forecast release, then calculate storage based on 
        # mass balance, and back transform storage for use in the next time step
        preds = forecast_mixedLM_new(
                # ml_model.params,
                # ml_model.random_effects,
                pd.DataFrame(ml_model.random_effects),
                # mexog,
                exog_re,
                means, 
                std,
                "group",
                actual_inflow,
                timelevel,
                tree=True
            )
    else:
        # in this case, we are directly providing all the independent variable values
        # this is essentially a day ahead release prediction beacuse we assume all past data is known
        # in the forcast function we only seed the model with initial values
        preds = predict_mixedLM(
                ml_model.params,
                ml_model.random_effects,
                mexog,
                exog_re,
                "group"
            )
    
    return preds, groups

def pipeline():
    # get prepped data set for modeling
    df = read_tva_data()
    groups = ["NaturalOnly", "RunOfRiver"]
    # filter groups {group to filter by: attribute of entries that should be included}
    filter_groups={"NaturalOnly":"NaturalFlow"}
    # filter_groups={"RunOfRiver":"StorageDam"}
    # filter_groups = {}
    # get the X matrix, and y vector along with means and std.
    # note: if no scaler is provided, means and std will be 0 and 1 respectively.
    timelevel="all"
    st_max = df.groupby(df.index.get_level_values(1))["Storage"].max()

    st_frac = (df["Storage"].unstack() / st_max).stack()
    
    #* I want to try the tree using storage fraction as well. 
    #* So previous to standardization, Take the current storage and divide it by the maximum storage.
    #* then fit the tree with just that variable as an unstandardized value. 
    #* this actually would probably work best if we could fit the tree splits and the parameters at the same time. 
    # split data sets and standardize variables
    X,y,means,std = prep_data(df, groups, filter_groups, scaler="mine", timelevel=timelevel)
    # calculate difference between storage pre and rolling 
    X["sto_diff"] = X["Storage_pre"] - X["Storage_roll7"]
    X["st_frac"] = st_frac
    
    # if we want a training and testing set that is split temporally
    # train_index, test_index = split_train_test_dt(y.index, date=datetime(2010, 1, 1), level=0, keep=8)
    reservoirs = X.index.get_level_values(1).unique()


    # old x_vars 
    X_vars = [
        "Storage_pre", "Release_pre", "Net Inflow",
        "Storage_Inflow_interaction",
        "Inflow_roll7", "Release_roll7", "Storage_roll7",
    ] 
    # new x_vars
    X_vars = [
        "sto_diff", "Storage_Inflow_interaction",
        "Release_pre", "Release_roll7",
        "Net Inflow", "Inflow_roll7"
    ]
    # for fitting tree
    X_vars_tree = [
        "sto_diff", "Storage_Inflow_interaction",
         "Release_pre", "Net Inflow", "Inflow_roll7"
        # "Net Inflow", "Inflow_roll7"
    ]

    
    # split into training and testing sets
    split_date = datetime(2010, 1, 1)
    test_index = X.loc[X.index.get_level_values(0) >= split_date - timedelta(days=8)].index
    train_index = X.loc[X.index.get_level_values(0) < split_date].index
    X_train = X.loc[train_index,X_vars]
    X_train_tree = X.loc[train_index, X_vars_tree]
    X_test = X.loc[test_index,X_vars]
    X_test_tree = X.loc[test_index,X_vars_tree]
    # X_test_all = X.loc[:,X_vars]
    y_train_rel = y.loc[train_index]
    y_test_rel = y.loc[test_index]
    # y_test_sto = X.loc[test_index, "Storage"]

    # train_res = X_train.index.get_level_values(1).unique()
    # test_res = X_test.index.get_level_values(1).unique()

    # fit the decision tree model
    max_depth=3
    #, splitter="best" - for decision_tree
    tree = tree_model(X_train_tree, y_train_rel, tree_type="decision", max_depth=max_depth,
                    random_state=37)
    leaves, groups = get_leaves_and_groups(X_train_tree, tree)

    month_arrays = {i:[0 for i in range(X_train.shape[0])] for i in calendar.month_abbr[1:]}

    for i, date in enumerate(X_train.index.get_level_values(0)):
        abbr = calendar.month_abbr[date.month]            
        month_arrays[abbr][i] = 1   

    for key, value in month_arrays.items():
        X_train[key] = value                      
    
    X_train["st_frac"] = st_frac
    ml_model = sub_tree_multi_level_model(X_train, y_train_rel, X_vars_tree, tree)

    coefs = pd.DataFrame(ml_model.random_effects)
    fitted = ml_model.fittedvalues
    
    month_arrays = {i:[0 for i in range(X_test.shape[0])] for i in calendar.month_abbr[1:]}
    
    for i, date in enumerate(X_test.index.get_level_values(0)):
        abbr = calendar.month_abbr[date.month]            
        month_arrays[abbr][i] = 1   

    for key, value in month_arrays.items():
        X_test[key] = value                      
    # coefs = {g: np.mean(fit_results["params"][g], axis=0)
    #          for g in groups.unique()}
    # coefs = pd.DataFrame(coefs, index=X_train.columns)

    test_leaves, test_groups = get_leaves_and_groups(X_test_tree, tree)

    # predict and forecast from the sub_tree model
    X_test["Storage_pre_act"] = df.loc[test_index, "Storage_pre"]
    preds, pgroups = predict_from_sub_tree_model(X_test, y_test_rel, tree, ml_model, tree_vars=X_vars_tree)

    idx = pd.IndexSlice
    actual_inflow_test = df.loc[test_index, "Net Inflow"] 
    X_test["Storage_pre"] = X.loc[test_index, "Storage_pre"]
    forecasted, fgroups = predict_from_sub_tree_model(X_test, y_test_rel, tree, ml_model, 
                                            forecast=True, means=means,std=std,
                                            timelevel=timelevel, actual_inflow=actual_inflow_test,
                                            tree_vars=X_vars_tree)
    forecasted = forecasted.dropna()
    
    # X_test["group"] = test_groups
    # forecasted = forecast_mixedLM_new(coefs, X_test, means, std, "group", actual_inflow_test)
    # forecasted = forecasted[forecasted.index.get_level_values(0).year >= 2010]

    y_train_rel_act = (y_train_rel.unstack() *
                    std.loc[:,"Release"] + means.loc[:,"Release"]).stack()
    y_test_rel_act = (y_test_rel.unstack() *
                    std.loc[:,"Release"] + means.loc[:,"Release"]).stack()
    fitted_act = (fitted.unstack() *
                      std.loc[:, "Release"] + means.loc[:, "Release"]).stack()
    preds_act = (preds.unstack() *
                      std.loc[:, "Release"] + means.loc[:, "Release"]).stack()
    forecasted_act = forecasted["Release_act"]

    train_score = r2_score(y_train_rel_act, fitted_act)
    train_rmse = mean_squared_error(y_train_rel_act, fitted_act, squared=False)
    test_score = r2_score(y_test_rel_act, preds_act)
    test_rmse = mean_squared_error(y_test_rel_act, preds_act, squared=False)
    fc_index = forecasted_act.index
    forecasted_score = r2_score(y_test_rel_act.loc[fc_index], forecasted_act)
    forecasted_rmse = mean_squared_error(y_test_rel_act.loc[fc_index], forecasted_act, squared=False)
    
    y_train_mean = y_train_rel_act.groupby(
        y_train_rel_act.index.get_level_values(1)
    ).mean()
    y_test_mean = y_test_rel_act.groupby(
        y_test_rel_act.index.get_level_values(1)
    ).mean()
    fmean = fitted_act.groupby(
        fitted_act.index.get_level_values(1)
    ).mean()
    pmean = preds_act.groupby(
        preds_act.index.get_level_values(1)
    ).mean()
    fcmean = forecasted_act.groupby(
        forecasted_act.index.get_level_values(1)
    ).mean()

    f_bias = fmean - y_train_mean
    f_bias_month = fitted_act.groupby(
        fitted_act.index.get_level_values(0).month
        ).mean() - y_train_rel_act.groupby(
            y_train_rel_act.index.get_level_values(0).month).mean()
    p_bias = pmean - y_test_mean
    p_bias_month = preds_act.groupby(
        preds_act.index.get_level_values(0).month
        ).mean() - y_test_rel_act.groupby(
            y_test_rel_act.index.get_level_values(0).month).mean()
    fc_bias = fcmean - y_test_mean
    fc_bias_month = forecasted_act.groupby(
        forecasted_act.index.get_level_values(0).month
        ).mean() - y_test_rel_act.groupby(
            y_test_rel_act.index.get_level_values(0).month).mean()

    results = {
        "f_act_score": train_score,
        "f_act_rmse": train_rmse,
        "f_bias": f_bias,
        "f_bias_month": f_bias_month,

        "p_act_score": test_score,
        "p_act_rmse": test_rmse,
        "p_bias": p_bias,
        "p_bias_month": p_bias_month,
        
        "fc_act_score": forecasted_score,
        "fc_act_rmse": forecasted_rmse,
        "fc_bias": fc_bias,
        "fc_bias_month": fc_bias_month,
        
        "coefs":coefs,
    }

    fitted_act = fitted_act.unstack()
    preds_act = preds_act.unstack()
    forecasted_act = forecasted_act.unstack()
    y_train_act = y_train_rel_act.unstack()

    res_scores = pd.DataFrame(index=reservoirs, columns=["NSE", "RMSE"])
    for res in reservoirs:
        ya = y_train_act[res]
        ym = fitted_act[res]
        res_scores.loc[res, "NSE"] = r2_score(ya, ym)
        res_scores.loc[res, "RMSE"] = np.sqrt(mean_squared_error(ya, ym))

    results["res_scores"] = res_scores

    train_data = pd.DataFrame(dict(actual=y_train_rel_act, model=fitted_act.stack()))
    test_p_data = pd.DataFrame(dict(actual=y_test_rel_act, model=preds_act.stack()))
    test_f_data = pd.DataFrame(dict(actual=y_test_rel_act, model=forecasted_act.stack())).dropna()
    train_quant, train_bins = pd.qcut(train_data["actual"], 3, labels=False, retbins=True)
    test_quant, test_bins = pd.qcut(test_p_data["actual"], 3, labels=False, retbins=True)

    train_quant_scores = pd.DataFrame(index=[0,1,2], columns=["NSE", "RMSE"])
    train_data["bin"] = train_quant
    for q in [0,1,2]:
        score = r2_score(
            train_data[train_data["bin"] == q]["actual"],
            train_data[train_data["bin"] == q]["model"]
        )
        rmse = np.sqrt(mean_squared_error(
            train_data[train_data["bin"] == q]["actual"],
            train_data[train_data["bin"] == q]["model"]
        ))
        train_quant_scores.loc[q] = [score, rmse]

    test_p_quant_scores = pd.DataFrame(index=[0,1,2], columns=["NSE", "RMSE"])
    test_p_data["bin"] = test_quant
    for q in [0,1,2]:
        score = r2_score(
            test_p_data[test_p_data["bin"] == q]["actual"],
            test_p_data[test_p_data["bin"] == q]["model"]
        )
        rmse = np.sqrt(mean_squared_error(
            test_p_data[test_p_data["bin"] == q]["actual"],
            test_p_data[test_p_data["bin"] == q]["model"]
        ))
        test_p_quant_scores.loc[q] = [score, rmse]

    test_f_quant_scores = pd.DataFrame(index=[0,1,2], columns=["NSE", "RMSE"])
    test_f_data["bin"] = test_quant
    for q in [0,1,2]:
        score = r2_score(
            test_f_data[test_f_data["bin"] == q]["actual"],
            test_f_data[test_f_data["bin"] == q]["model"]
        )
        rmse = np.sqrt(mean_squared_error(
            test_f_data[test_f_data["bin"] == q]["actual"],
            test_f_data[test_f_data["bin"] == q]["model"]
        ))
        test_f_quant_scores.loc[q] = [score, rmse]
    
    
    print(train_quant_scores.to_markdown(floatfmt="0.3f"))
    print(test_p_quant_scores.to_markdown(floatfmt="0.3f"))
    print(test_f_quant_scores.to_markdown(floatfmt="0.3f"))

    # setup output parameters
    if timelevel == "all":
        prepend = ""
    else:
        prepend = f"{timelevel}_"
    # foldername = f"{prepend}upstream_basic_td{max_depth:d}_roll7_simple_tree_month_coefs_st_frac"
    # folderpath = pathlib.Path("..", "results", "treed_ml_model_dual_fit", foldername)
    # folderpath = pathlib.Path("..", "results", "synthesis", "treed_model", foldername)
    # check if the directory exists and handle it
    # if folderpath.is_dir():
    #     # response = input(f"{folderpath} already exists. Are you sure you want to overwrite its contents? [y/N] ")
    #     response = "y"
    #     if response[0].lower() != "y":
    #         folderpath = pathlib.Path(
    #             "..", "results", "treed_ml_model", 
    #             "_".join([foldername, datetime.today().strftime("%Y%m%d_%H%M")]))
    #         print(f"Saving at {folderpath} instead.")
    #         folderpath.mkdir()
    # else:
    #     folderpath.mkdir()

    output_dir = pathlib.Path(f"../results/agu_2021_runs/tree_model_temporal_validation")
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # export tree to graphviz file so it can be converted nicely
    # rotate_tree = True if max_depth > 3 else False

    export_graphviz(tree, out_file=(output_dir / "tree.dot").as_posix(),
                    feature_names=X_vars_tree, filled=True, proportion=True, rounded=True,
                    special_characters=True)

    # setup output container for modeling information
    X_train["Storage_pre"] = X["Storage_pre"]
    output = dict(
        **results,
        data=dict(
            X_test=X_test,
            X_train=X_train,
            fitted_rel=fitted_act,
            y_train_rel_act=y_train_rel_act,
            groups=groups,
            train_quant_scores=train_quant_scores,
            test_p_quant_scores=test_p_quant_scores,
            test_f_quant_scores=test_f_quant_scores,
            train_data=train_data,
            test_p_data=test_p_data,
            test_f_data=test_f_data,
            forecasted=forecasted[["Release", "Storage",
                                   "Release_act", "Storage_act"]]
        )
    )
    # write the output dict to a pickle file
    with open((output_dir / "results.pickle").as_posix(), "wb") as f:
        pickle.dump(output, f, protocol=4)
    
    # write the random effects to a csv file for easy access
    # pd.DataFrame(ml_model.random_effects).to_csv(
    #     (folderpath / "random_effects.csv").as_posix())
    coefs.to_csv((output_dir/"random_effects.csv").as_posix())

if __name__ == "__main__":
    pipeline()
    # mem_usage = psutil.Process().memory_info().peak_wset
    # print(f"Max Memory Used: {mem_usage/1000/1000:.4f} MB")
    # from resource import getrusage, RUSAGE_SELF
    # print(getrusage(RUSAGE_SELF).ru_maxrss)
