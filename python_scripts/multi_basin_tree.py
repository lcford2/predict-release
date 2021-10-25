import pickle
import pathlib
import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLMParams
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.tree import (DecisionTreeRegressor, export_graphviz)
from sklearn.ensemble import RandomForestRegressor
from simple_model import (predict_mixedLM, forecast_mixedLM)
from utils.timing_function import time_function
from time import perf_counter as timer
from copy import deepcopy
from datetime import timedelta, datetime
from IPython import embed as II


def read_basin_data(basin: str) -> pd.DataFrame:
    data_locs = {
        "upper_col":{
            "ready":"../upper_colorado_data/model_ready_data/upper_col_data.csv",
            "raw":"../upper_colorado_data/hydrodata_data/req_upper_col_data.csv",
        },
        "pnw":{
            "ready":"../pnw_data/model_ready_data/pnw_data.csv",
            "raw":"../pnw_data/dam_data/*_data/*.csv",
        },
        "lower_col":{
            "ready":"../lower_col_data/model_ready_data/lower_col_data.csv",
            "raw":"../lower_col_data/lower_col_dam_data.csv",
        },
        "missouri":{
            "ready":"../missouri_data/model_ready_data/missouri_data.csv",
            "raw":"../missouri_data/hydromet_data/*.csv",
        }
    }
    if basin not in data_locs:
        raise NotImplementedError(f"No data available for basin {basin}")
    
    fpath = pathlib.Path(data_locs[basin]["ready"])
    if fpath.exists():
        return pd.read_csv(fpath, index_col=[0,1], converters={1:pd.to_datetime})
    else:
        print(f"Please run ../apply_models/apply_models.py {basin} to generate model ready data for this basin.")
        sys.exit()

def get_basin_meta_data(basin: str):
    with open(f"../apply_models/basin_output/{basin}_meta.pickle", "rb") as f:
        meta = pickle.load(f)
    return meta  

@time_function
def prep_data(df):
    grouper = df.index.get_level_values(0)
    std_data = df.groupby(grouper).apply(lambda x: (x - x.mean()) / x.std())
    means = df.groupby(grouper).mean()
    std = df.groupby(grouper).std()
    X = std_data[[
        "release_pre", "storage", "storage_pre", "inflow", "release_roll7", 
        "inflow_roll7", "storage_roll7", "storage_x_inflow"
    ]]
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
def sub_tree_multi_level_model(X, y, tree=None, groups=None, my_id=None, drop_rel_roll: bool=False):
    if tree:
        if drop_rel_roll and "release_roll7" in X.columns:
            leaves, groups = get_leaves_and_groups(X.drop("release_roll7", axis=1), tree)
        else:
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
        leaves, groups = get_leaves_and_groups(X.drop("storage_pre_act", axis=1), tree)
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
    basin = "lower_col"
    df = read_basin_data(basin) 
    meta = get_basin_meta_data(basin)
    
    reservoirs = meta[meta["group"] == "high_rt"].index

    res_grouper = df.index.get_level_values(0)
    df = df.loc[res_grouper.isin(reservoirs)]
    # need to set it again after we trimmed the data set
    res_grouper = df.index.get_level_values(0)
    time_grouper = df.index.get_level_values(1)

    X,y,means,std = prep_data(df)
    X["sto_diff"] = X["storage_pre"] - X["storage_roll7"]

    # set exogenous variables
    X_vars = ["sto_diff",  "storage_x_inflow",
            "release_pre", "release_roll7",
            "inflow", "inflow_roll7"]

    # X_vars_tree = ["sto_diff", "storage_x_inflow",
    #         "release_pre",
    #         "inflow", "inflow_roll7"]
    X_vars_tree = X_vars

    train_index = X.index
    X_train = X.loc[train_index,X_vars]
    X_train_tree = X.loc[train_index, X_vars_tree]
    y_train_rel = y.loc[train_index]

    train_res = res_grouper.unique()

    # fit the decision tree model
    max_depth=3
    tree = tree_model(X_train_tree, y_train_rel, tree_type="decision", max_depth=max_depth,
                    random_state=37)
    leaves, groups = get_leaves_and_groups(X_train_tree, tree)
    ml_model = sub_tree_multi_level_model(X_train, y_train_rel, tree, drop_rel_roll=False)
    
    coefs = pd.DataFrame(ml_model.random_effects)
    fitted = ml_model.fittedvalues
    y_train_rel_act = (y_train_rel.unstack().T *
                    std.loc[train_res,"release"] + means.loc[train_res,"release"]).T.stack()

    fitted_act = (fitted.unstack().T *
                      std.loc[train_res, "release"] + means.loc[train_res, "release"]).T.stack()

    f_act_score = r2_score(y_train_rel_act, fitted_act)
    f_act_rmse = np.sqrt(mean_squared_error(y_train_rel_act, fitted_act))
    
    y_train_mean = y_train_rel_act.groupby(res_grouper).mean()
    fmean = fitted_act.groupby(res_grouper).mean()

    f_bias = fmean - y_train_mean
    f_bias_month = fitted_act.groupby(time_grouper.month).mean() - \
        y_train_rel_act.groupby(time_grouper.month).mean()

    results = {
        "f_act_score": f_act_score,
        "f_act_rmse": f_act_rmse,
        "f_bias": f_bias,
        "f_bias_month": f_bias_month,
        "coefs":coefs,
    }

    # fitted_act = fitted_act.unstack()
    # y_train_act = y_train_rel_act.unstack()

    train_data = pd.DataFrame(dict(actual=y_train_rel_act, model=fitted_act))
    
    res_scores = pd.DataFrame(index=reservoirs, columns=["NSE", "RMSE"])

    res_scores["NSE"] = train_data.groupby(res_grouper).apply(lambda x: r2_score(x["actual"], x["model"]))
    res_scores["RMSE"] = train_data.groupby(res_grouper).apply(lambda x: mean_squared_error(x["actual"], x["model"], squared=False))
    # for res in reservoirs:
    #     ya = y_train_act[res]
    #     ym = fitted_act[res]
    #     res_scores.loc[res, "NSE"] = r2_score(ya, ym)
    #     res_scores.loc[res, "RMSE"] = np.sqrt(mean_squared_error(ya, ym))

    results["res_scores"] = res_scores
    print(res_scores.to_markdown(floatfmt="0.3f"))

    train_quant, train_bins = pd.qcut(train_data["actual"], 3, labels=False, retbins=True)
    quant_scores = pd.DataFrame(index=[0,1,2], columns=["NSE", "RMSE"])
    train_data["bin"] = train_quant

    quant_scores["NSE"] = train_data.groupby("bin").apply(lambda x: r2_score(x["actual"], x["model"]))
    quant_scores["RMSE"] = train_data.groupby("bin").apply(lambda x: mean_squared_error(x["actual"], x["model"], squared=False))
    # for q in [0,1,2]:
    #     score = r2_score(
    #         train_data[train_data["bin"] == q]["actual"],
    #         train_data[train_data["bin"] == q]["model"]
    #     )
    #     rmse = np.sqrt(mean_squared_error(
    #         train_data[train_data["bin"] == q]["actual"],
    #         train_data[train_data["bin"] == q]["model"]
    #     ))
    #     quant_scores.loc[q] = [score, rmse]

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
    # X_test["group"] = test_groups
    # forecasted = forecast_mixedLM_new(coefs, X_test, means, std, "group", actual_inflow_test)
    # forecasted = forecasted[forecasted.index.get_level_values(0).year >= 2010]


    # get all variables back to original space
    # preds_rel = pred_result["f_rel_act"][0].unstack()
    # preds_sto = pred_result["f_sto_act"][0].unstack()
    # fc_rel = forecasted.loc[:, "Release_act"].unstack()
    # fc_sto = forecasted.loc[:, "Storage_act"].unstack()
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
    # try:
    #     preds_score_rel = r2_score(y_test_rel_act, preds_rel.stack())
    #     preds_score_sto = r2_score(y_test_sto_act, preds_sto.stack())
    # except ValueError as e:
    #     II()
    # y_test_rel_act = y_test_rel_act.loc[fc_rel.index]
    # y_test_sto_act = y_test_sto_act.loc[fc_sto.index]

    # fit_score_rel = r2_score(y_train_rel_act, fitted_rel)
    # fit_score_sto = r2_score(y_train_sto_act, fitted_sto)
    # try:
    #     fc_score_rel = r2_score(y_test_rel_act, fc_rel.stack())
    #     fc_score_sto = r2_score(y_test_sto_act, fc_sto.stack())
    # except ValueError as e:
    #     fc_score_rel = np.nan
    #     fc_score_sto = np.nan

    # score_strings = [f"{ftype:<12} = {score:.3f}" for ftype, score in zip(
    #                 ["Fit", "Forecast", "Preds"], [fit_score_rel, fc_score_rel, preds_score_rel])]
    # print("Release Scores:")
    # print("\n".join(score_strings))

    # score_strings = [f"{ftype:<12} = {score:.3f}" for ftype, score in zip(
    #                 ["Fit", "Forecast", "Preds"], [fit_score_sto, fc_score_sto, preds_score_sto])]
    # print("Storage Scores:")
    # print("\n".join(score_strings))

    # setup output parameters
    foldername = "treed_model"
    folderpath = pathlib.Path("..", "results", "basin_eval", basin, foldername)
    # check if the directory exists and handle it
    if folderpath.is_dir():
        # response = input(f"{folderpath} already exists. Are you sure you want to overwrite its contents? [y/N] ")
        response = "y"
        if response[0].lower() != "y":
            folderpath = pathlib.Path(
                "..", "results", "basin_eval", basin, 
                "_".join([foldername, datetime.today().strftime("%Y%m%d_%H%M")]))
            print(f"Saving at {folderpath} instead.")
            folderpath.mkdir()
    else:
        folderpath.mkdir(parents=True)
    
    # export tree to graphviz file so it can be converted nicely
    # rotate_tree = True if max_depth > 3 else False

    export_graphviz(tree, out_file=(folderpath / "tree.dot").as_posix(),
                    feature_names=X_vars_tree, filled=True, proportion=True, rounded=True,
                    special_characters=True)

    # setup output container for modeling information
    X_train["storage_pre"] = X["storage_pre"]
    output = dict(
        # re_coefs=ml_model.random_effects,
        # fe_coefs=ml_model.params,
        # cov_re=ml_model.cov_re,
        # coefs=coefs,
        **results,
        data=dict(
            # X_test=X_test,
            # y_test=y_test,
            X_train=X_train,
            # y_train=y_train,
            fitted_rel=fitted_act,
            # fitted_sto=fitted_sto,
            # y_test_rel_act=y_test_rel_act,
            y_train_rel_act=y_train_rel_act,
            # y_test_sto_act=y_test_sto_act,
            # y_train_sto_act=y_train_sto_act,
            # predicted_act_rel=preds_rel.stack(),
            # predicted_act_sto=preds_sto.stack(),
            # groups=test_groups,
            groups=groups,
            quant_scores=quant_scores
            # forecasted=forecasted[["Release", "Storage",
                                #    "Release_act", "Storage_act"]]
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
