import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLMParams
from sklearn.metrics import r2_score
from sklearn.tree import (DecisionTreeRegressor,
                          plot_tree, export_graphviz)
from sklearn.ensemble import RandomForestRegressor
# import dtreeviz.trees as dtrees
from utils.helper_functions import (read_tva_data, scale_multi_level_df,
                              read_all_res_data, find_max_date_range)
from simple_model import (change_group_names, combine_columns,
                          predict_mixedLM, forecast_mixedLM, print_coef_table)
from utils.timing_function import time_function
from time import perf_counter as timer
from copy import deepcopy
from datetime import timedelta, datetime
from IPython import embed as II
from mpi4py import MPI
import calendar
import sys
import pickle
import pathlib
import subprocess
import warnings
warnings.simplefilter("ignore")

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

# @time_function
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
                                #"Storage_roll14", "Release_roll14", "Inflow_roll14"
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

# @time_function
def split_train_test_dt(index, date, level=None):
    # cannot check truthy here because level can be integer 0, which is equivalent to False
    if level is not None:
        test = index[index.get_level_values(level) >= date]
        train = index[index.get_level_values(level) < date]
    else:
        test = index[index >= date]
        train = index[index < date]
    return train, test

# @time_function
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

def get_tree_breaks(X, tree):
    trees = tree.estimators_
    breaks = []
    for clf in trees:
        iterator = zip(
            clf.tree_.feature,
            clf.tree_.threshold,
            clf.tree_.children_left,
            clf.tree_.children_right
        )
        for i, (feat, thresh, left, right) in enumerate(iterator):
            if left != -1 and right != -1:
                breaks.append([
                    i,
                    X.columns[feat],
                    thresh,
                    left,
                    right
                ])
    breaks = pd.DataFrame(breaks, 
        columns=["Node", "Param", "Threshold", "Left", "Right"])
    return breaks


# @time_function
def sub_tree_multi_level_model(X, y, tree=None, groups=None, my_id=None):
    if tree:
        leaves, groups = get_leaves_and_groups(X, tree)
    else:
        groups = groups[my_id]

        
    mexog = pd.DataFrame(np.ones((y.size, 1)), index=y.index,  columns=["const"])
    free = MixedLMParams.from_components(fe_params=np.ones(mexog.shape[1]),
                                         cov_re=np.eye(X.shape[1]))
    md = sm.MixedLM(y, mexog, groups=groups, exog_re=X)
    mdf = md.fit(disp=False)# free=free)
    return mdf


def predict_from_sub_tree_model(X, y, tree, ml_model, forecast=False, means=None, std=None, timelevel="all"):
    # means and std required if forecast is True
    leaves, groups = get_leaves_and_groups(X, tree)

    mexog = pd.DataFrame(np.ones((y.size, 1)), index=y.index, columns=["const"])
    exog_re = deepcopy(X)
    exog_re["group"] = groups

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
                timelevel
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
    # get the X matrix, and y vector along with means and std.
    # note: if no scaler is provided, means and std will be 0 and 1 respectively.
    timelevel="all"
    X,y,means,std = prep_data(df, groups, filter_groups, scaler="mine", timelevel=timelevel)
    train_index, test_index = split_train_test_dt(y.index, date=datetime(2010, 1, 1), level=0)

    # set exogenous variables
    X_vars = ["Storage_pre", "Release_pre", "Net Inflow",
              "Storage_Inflow_interaction",
              "Inflow_roll7", "Storage_roll7", "Release_roll7",
              #"Storage_7", "Release_7"
              ]
    X = X.loc[:, X_vars]
  
    # split into training and testing sets
    X_train = X.loc[train_index,:]
    X_test = X.loc[test_index,:]
    y_train = y.loc[train_index]
    y_test = y.loc[test_index]

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    # fit the decision tree model
    max_depth=3
    #, splitter="best" - for decision_tree
    size = 100
    tree = tree_model(X_train, y_train, tree_type="ensemble", max_depth=max_depth,
                      random_state=37, n_estimators=size, oob_score=False)
    leaves, groups = get_leaves_and_groups(X_train, tree)
    tree_breaks = get_tree_breaks(X, tree)

    my_results = {}
    for i in range(rank, size, nprocs):
        my_results[i] = sub_tree_multi_level_model(
            X_train, y_train, groups=groups, my_id=i+1
            )
    # lag
    # lag no_free
    # lag oob
    # lag oob no_free
    # roll
    # roll no_free
    # roll oob
    # roll oob no_free
    all_results = comm.gather(my_results, root=0) 
    if rank == 0:
        results = {}
        for item in all_results:
            for key, value in item.items():
                results[key] = value.random_effects
        outfile = "../results/treed_ml_model/sensitivity/rf_results_roll_no_free.pickle"
        with open(outfile, "wb") as f:
            pickle.dump(results, f, protocol=4)
        with open("../results/treed_ml_model/sensitivity/rf_results_roll_no_fre_breaks.pickle", "wb") as f:
            pickle.dump(tree_breaks, f, protocol=4)
        sys.exit()
    else:
        sys.exit()
    # fit the sub_tree ml model
    # ml_model = sub_tree_multi_level_model(X_train, y_train, tree)
    fitted = ml_model.fittedvalues

    # predict and forecast from the sub_tree model
    preds, pgroups = predict_from_sub_tree_model(X_test, y_test, tree, ml_model)
    forecasted, fgroups = predict_from_sub_tree_model(X_test, y_test, tree, ml_model, 
                                             forecast=True, means=means, std=std,
                                            timelevel=timelevel)
    
    # get all variables back to original space
    preds = preds.unstack()
    fitted = fitted.unstack()
    y_train = y_train.unstack()
    y_test = y_test.unstack()
    if timelevel == "all":
        preds_act = (preds * std["Release"] + means["Release"]).stack()
        fitted_act = (fitted * std["Release"] + means["Release"]).stack()
        y_train_act = (y_train * std["Release"] + means["Release"]).stack()
        y_test_act = (y_test * std["Release"] + means["Release"]).stack()
    else:
        rel_means = means["Release"].unstack()
        rel_std = std["Release"].unstack()
        preds_act = deepcopy(preds)
        fitted_act = deepcopy(fitted)
        y_train_act = deepcopy(y_train)
        y_test_act = deepcopy(y_test)
        extent = 13 if timelevel == "month" else 4
        for tl in range(1,extent):
            ploc = getattr(preds.index, timelevel) == tl
            floc = getattr(fitted.index, timelevel) == tl
            preds_act.loc[ploc] = preds_act.loc[ploc] * rel_std.loc[tl] + rel_means.loc[tl]
            fitted_act.loc[floc] = fitted_act.loc[floc] * rel_std.loc[tl] + rel_means.loc[tl]
            y_train_act.loc[floc] = y_train_act.loc[floc] * rel_std.loc[tl] + rel_means.loc[tl]
            y_test_act.loc[ploc] = y_test_act.loc[ploc] * rel_std.loc[tl] + rel_means.loc[tl]
        preds_act = preds_act.stack()
        fitted_act = fitted_act.stack()
        y_train_act = y_train_act.stack()
        y_test_act = y_test_act.stack()
    forecasted_act = forecasted["Release_act"]

    # report scores for the current model run
    fit_score = r2_score(y_train_act, fitted_act)
    forecasted_score = r2_score(y_test_act, forecasted_act)
    preds_score = r2_score(y_test_act, preds_act)
    score_strings = [f"{ftype:<12} = {score:.3f}" for ftype, score in zip(
                    ["Fit", "Forecast", "Preds"], [fit_score, forecasted_score, preds_score])]
    print("Scores:")
    print("\n".join(score_strings))

    # setup output parameters
    if timelevel == "all":
        prepend = ""
    else:
        prepend = f"{timelevel}_"
    foldername = f"{prepend}upstream_basic_td{max_depth:d}_roll7"
    folderpath = pathlib.Path("..", "results", "treed_ml_model", foldername)

    # check if the directory exists and handle it
    if folderpath.is_dir():
        response = input(f"{folderpath} already exists. Are you sure you want to overwrite its contents? [y/N] ")
        if response[0].lower() != "y":
            folderpath = pathlib.Path(
                "..", "results", "treed_ml_model", 
                foldername + datetime.today().strftime("%Y%m%d_%H%M"))
            print(f"Saving at {folderpath} instead.")
            folderpath.mkdir()
    else:
        folderpath.mkdir()
    
    # export tree to graphviz file so it can be converted nicely
    export_graphviz(tree, out_file=(folderpath / "tree.dot").as_posix(),
                    feature_names=X_vars, filled=True, proportion=True, rounded=True,
                    special_characters=True)

    # setup output container for modeling information
    output = dict(
        re_coefs=ml_model.random_effects,
        fe_coefs=ml_model.params,
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
