import sys
import pickle
import calendar
import pathlib
from time import perf_counter as timer
from multiprocessing import Pool, cpu_count
from datetime import timedelta, datetime
from collections import defaultdict
from itertools import combinations, product

import pandas as pd
import numpy as np
import ctypes as ct
from numpy.ctypeslib import ndpointer
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLMParams
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import xgboost as xgb

from utils.helper_functions import (read_tva_data, scale_multi_level_df,
                              read_all_res_data, find_max_date_range)
from utils.timing_function import time_function

from IPython import embed as II

group_names = {
    "RunOfRiver": {0: "StorageDam", 1: "RunOfRiver"},
    "NaturalOnly": {0: "ComboFlow", 1: "NaturalFlow"},
    "PrimaryType": {0: "FloodControl",
                    1: "Hydropower",
                    2: "Irrigation",
                    3: "Navigation",
                    4: "WaterSupply"}
}

inverse_groupnames = {key: {label: idx for idx, label in value.items()} for key, value in group_names.items()}

def filter_on_corr(level, rgroup, corrs, rts):
    combos = combinations(rgroup, 2)

    leave_out = []

    for res1, res2 in combos:
        rrcorr = corrs.loc[res1, res2]
        if rrcorr > level:
            r1_rt = rts.loc[res1]
            r2_rt = rts.loc[res2]
            if r1_rt > r2_rt:
                if res1 not in leave_out:
                    leave_out.append(res1)
            else:
                if res2 not in leave_out:
                    leave_out.append(res2)
    return leave_out

def load_library():
    lib = ct.CDLL("../c_res_simul/lib/libres_simul.so")

    lib.res_simul.argtypes = [
            # ct.c_double,
            ndpointer(ct.c_double, ndim=1, flags="C_CONTIGUOUS"), # intercepts
            ndpointer(ct.c_double, ndim=1, flags="C_CONTIGUOUS"), # coefs
            ndpointer(ct.c_double, ndim=1, flags="C_CONTIGUOUS"), # prev_rel
            ndpointer(ct.c_double, ndim=1, flags="C_CONTIGUOUS"), # prev_sto
            ndpointer(ct.c_double, ndim=1, flags="C_CONTIGUOUS"), # inflow
            ct.c_double,                                          # rel_mean
            ct.c_double,                                          # rel_std
            ct.c_double,                                          # sto_mean
            ct.c_double,                                          # sto_std
            ct.c_double,                                          # inf_mean
            ct.c_double,                                          # inf_std
            ct.c_double,                                          # sxi_mean
            ct.c_double,                                          # sxi_std
            ct.c_int,                                             # ts_len
            ct.c_double,                                          # rel_max
            ct.c_double,                                          # rel_min
            ct.c_double,                                          # sto_max
            ct.c_double,                                          # sto_min
            ct.c_double,                                          # inf_max
            ct.c_double,                                          # inf_min
            ct.c_double,                                          # sxi_max
            ct.c_double,                                          # sxi_min
            ct.c_int,                                             # std_or_norm
            ndpointer(ct.c_double, ndim=1, flags="C_CONTIGUOUS"), # rel_out
            ndpointer(ct.c_double, ndim=1, flags="C_CONTIGUOUS")  # sto_out
    ]
    lib.res_simul.restype = None
    lib.standardize.argtypes = [ct.c_double, ct.c_double, ct.c_double]
    lib.standardize.restype = ct.c_double
    lib.unstandardize.argtypes = [ct.c_double, ct.c_double, ct.c_double]
    lib.unstandardize.restype = ct.c_double
    lib.normalize.argtypes = [ct.c_double, ct.c_double, ct.c_double]
    lib.normalize.restype = ct.c_double
    lib.unnormalize.argtypes = [ct.c_double, ct.c_double, ct.c_double]
    lib.unnormalize.restype = ct.c_double
    return lib

def simul_res(res_exog, coefs, month_coefs, means, std, std_or_norm=0):
    # intercept = coefs["const"]
    # re_coefs = coefs[[
    #     "Release_pre", "Storage_pre", "Net Inflow",
    #     "Storage_Inflow_interaction",
    #     "Release_roll7", "Storage_roll7", "Inflow_roll7"
    # ]].values
    intercept = coefs[0]
    re_coefs  = coefs[1:-12]
    # month_coefs = coefs[-12:]

    intercepts = np.array([intercept + month_coefs[i-1]
                           for i in res_exog.index.get_level_values(0).month],
                          dtype="float64")

    ts_len = res_exog.shape[0] - 6
    rel_out = np.zeros(ts_len, dtype="float64")
    sto_out = np.zeros(ts_len, dtype="float64")
    lib = load_library()
    simul_func = lib.res_simul
    simul_func(intercepts, re_coefs,
                res_exog["Release_pre"].values,
                res_exog["Storage_pre"].values,
                res_exog["Net Inflow"].values,
                means["Release"],
                means["Storage"],
                means["Net Inflow"],
                means["Storage_Inflow_interaction"],
                std["Release"],
                std["Storage"],
                std["Net Inflow"],
                std["Storage_Inflow_interaction"],
                ts_len,
                res_exog["Release_pre"].max(),
                res_exog["Release_pre"].min(),
                res_exog["Storage_pre"].max(),
                res_exog["Storage_pre"].min(),
                res_exog["Net Inflow"].max(),
                res_exog["Net Inflow"].min(),
                res_exog["Storage_Inflow_interaction"].max(),
                res_exog["Storage_Inflow_interaction"].min(),
                std_or_norm,
                rel_out, sto_out)
    return rel_out, sto_out

def single_res_error(coefs, res_exog, act_rel, means, std, lib):
    rel_out, sto_out = simul_res(res_exog, coefs, means, std, lib)
    return mean_squared_error(act_rel, rel_out)

def std_array(arr, arr_mean, arr_std):
    return (arr - arr_mean) / arr_std

def unstd_array(arr, arr_mean, arr_std):
    return arr * arr_std + arr_mean

def norm_array(arr, arr_min, arr_max):
    return (arr - arr_min) / (arr_max - arr_min)

def unnorm_array(arr, arr_min, arr_max):
    return (arr * (arr_max - arr_min)) + arr_min

def multi_res_error(coefs, month_coefs, X, act_rel, act_sto, means, std, coef_index, rel_and_sto=False, std_or_norm=0):
    error = 0
    idx = pd.IndexSlice

    for res, index in coef_index.items():
        res_coefs = coefs[index[0]:index[1]]
        res_month_coefs = month_coefs[int(index[0]/8)*12:int(index[1]/8)*12]
        res_exog = X.loc[idx[:,res],:]
        res_means = means.loc[res]
        res_std = std.loc[res]
        rel_out, sto_out = simul_res(res_exog,
                                     res_coefs,
                                     res_month_coefs,
                                     res_means,
                                     res_std,
                                     # lib,
                                     # simul_func,
                                     std_or_norm)
        if np.isnan(rel_out).sum() > 0 or np.isnan(sto_out).sum():
            # double error when number are not real.
            error += error
        else:
            if std_or_norm == 0:
                error += mean_squared_error(
                    std_array(
                        act_rel.loc[idx[:,res]].values[6:],
                        res_means["Release_pre"],
                        res_std["Release_pre"]
                    ),
                    std_array(
                        rel_out,
                        res_means["Release_pre"],
                        res_std["Release_pre"]
                    )
                )
            else:
                error += mean_squared_error(
                    norm_array(
                        act_rel.loc[idx[:,res]].values[6:],
                        res_exog["Release_pre"].min(),
                        res_exog["Release_pre"].max()
                    ),
                    norm_array(
                        rel_out,
                        res_exog["Release_pre"].min(),
                        res_exog["Release_pre"].max()
                    )
                )

            if rel_and_sto:
                if std_or_norm == 0:
                    error += mean_squared_error(
                        std_array(
                            act_sto.loc[idx[:,res]].values[6:],
                            res_means["Storage_pre"],
                            res_std["Storage_pre"]
                        ),
                        std_array(
                            sto_out,
                            res_means["Storage_pre"],
                            res_std["Storage_pre"]
                        )
                    )
                else:
                    error += mean_squared_error(
                        norm_array(
                            act_sto.loc[idx[:,res]].values[6:],
                            res_exog["Storage_pre"].min(),
                            res_exog["Storage_pre"].max()
                        ),
                        norm_array(
                            sto_out,
                            res_exog["Storage_pre"].min(),
                            res_exog["Storage_pre"].max()
                        )
                    )
    return error


def filter_groups_out(df, filter_groups):
    for key, value in filter_groups.items():
        df = df.loc[df[key] == inverse_groupnames[key][value],:]
    return df

def my_minimize(args):
    func = args[0]
    init_params = args[1]
    return minimize(func, init_params, args=args[2:], method="Nelder-Mead",
                    options={"maxiter":5000, "maxfev":10000})

def fit_simul_res(df, simul_func, groups, filter_groups=None, scaler="mine"):
    for_groups = df.loc[:,groups]
    fraction_names = ["Fraction_Storage",
                      "Fraction_Net Inflow"]
    fractions = df.loc[:, fraction_names]
    if filter_groups:
        df = filter_groups_out(df, filter_groups)

    df = df.copy()

    df["Storage_Inflow_interaction"] = df.loc[:,"Storage_pre"].mul(
        df.loc[:,"Net Inflow"])

    scaled_df, means, std = scale_multi_level_df(df)
    X_scaled = df.loc[:, ["Storage_pre", "Net Inflow", "Release_pre",
                                "Storage_Inflow_interaction",
                                "Release_roll7", "Storage_roll7",
                                "Inflow_roll7"]
                            ]
    y_scaled = df.loc[:, "Release"]

    X_scaled[groups] = for_groups
    # X_scaled[fraction_names] = fractions

    X_scaled = change_group_names(X_scaled, groups, group_names)

    # these reservoirs exhibit different characteristics than their group may suggest.
    change_names = ["Douglas", "Cherokee", "Hiwassee"]
    for ch_name in change_names:
        size = X_scaled[X_scaled.index.get_level_values(1) == ch_name].shape[0]
        X_scaled.loc[X_scaled.index.get_level_values(1) == ch_name, "NaturalOnly"] = ["NaturalFlow" for i in range(size)]

    X_scaled = combine_columns(X_scaled, groups, "compositegroup")

    #* this introduces a intercept that varies monthly and between groups
    month_arrays = {i:[0 for i in range(X_scaled.shape[0])] for i in calendar.month_abbr[1:]}
    for i, date in enumerate(X_scaled.index.get_level_values(0)):
        abbr = calendar.month_abbr[date.month]
        month_arrays[abbr][i] = 1

    for key, array in month_arrays.items():
        X_scaled[key] = array

    split_date = datetime(2010,1,1)
    reservoirs = X_scaled.index.get_level_values(1).unique()

    train_index = X_scaled.index

    X_test = X_scaled.loc[X_scaled.index.get_level_values(0) >= split_date - timedelta(days=7)]
    X_train = X_scaled.loc[X_scaled.index.get_level_values(0) < split_date]

    y_test = y_scaled.loc[y_scaled.index.get_level_values(0) >= split_date - timedelta(days=7)]
    y_train = y_scaled.loc[y_scaled.index.get_level_values(0) < split_date]

    N_time_train = X_train.index.get_level_values(0).unique().shape[0]
    N_time_test = X_test.index.get_level_values(0).unique().shape[0]

    # train model
    # Instead of adding a constant, I want to create dummy variable that accounts for season differences
    exog_terms = [
        "Net Inflow", "Storage_pre", "Release_pre", "Storage_Inflow_interaction"
    ]

    old_results = pd.read_pickle("../results/agu_2021_runs/simple_model_temporal_validation_all_res/results.pickle")
    init_params = old_results["coefs"]
    groups = X_scaled.groupby(X_scaled.index.get_level_values(1))["compositegroup"].unique().apply(
        lambda x: x[0]
    )

    test = False

    if test:
        exog_df = X_test.copy()
        label = "test"
    else:
        exog_df = X_train.copy()
        label = "train"

    idx = pd.IndexSlice
    coef_order = [
        "const",
        "Release_pre", "Storage_pre", "Net Inflow",
        "Storage_Inflow_interaction",
        "Release_roll7", "Storage_roll7", "Inflow_roll7",
        # *calendar.month_abbr[1:]
    ]
    nprm = len(coef_order)
    coef_map = {
        'NaturalFlow-StorageDam': 0,
        'ComboFlow-StorageDam': 1,
        'ComboFlow-RunOfRiver': 2
    }
    coef_index = {res:(coef_map[g]*nprm,(coef_map[g]+1)*nprm) for res, g in groups.items()}

    all_res_coefs = [
        *init_params.loc[coef_order,"NaturalFlow-StorageDam"].values,
        *init_params.loc[coef_order,"ComboFlow-StorageDam"].values,
        *init_params.loc[coef_order,"ComboFlow-RunOfRiver"].values
    ]
    all_res_month_coefs = [
        *init_params.loc[calendar.month_abbr[1:],"NaturalFlow-StorageDam"].values,
        *init_params.loc[calendar.month_abbr[1:],"ComboFlow-StorageDam"].values,
        *init_params.loc[calendar.month_abbr[1:],"ComboFlow-RunOfRiver"].values,
    ]

    # result = minimize(single_res_error, res_coefs.values, args=(
    #     res_exog, df.loc[res_exog.index, "Release"].values[6:],
    #     res_means, res_std, lib)
    # )

    std_or_norm = 0
    sn_label = "std" if std_or_norm == 0 else "norm"

    N_trials = 26
    n_param = len(all_res_coefs)

    prange = 1
    trial_coefs = [
        np.random.rand(n_param) * prange * np.random.choice([-1,1], n_param)
        for i in range(N_trials)
    ]

    arg_sets = [
        (
            multi_res_error,
            trial_coefs[i],
            all_res_month_coefs,
            exog_df.loc[:,exog_terms],
            df.loc[exog_df.index, "Release"],
            df.loc[exog_df.index, "Storage"],
            means,
            std,
            # lib,
            # simul_func,
            coef_index,
            True,
            std_or_norm
        ) for i in range(N_trials)
    ]

    with Pool(processes=N_trials) as pool:
        pool_results = pool.map_async(my_minimize, arg_sets)
        print(pool_results)
        results = pool_results.get()


    file = "../results/simul_model/multi_trial_nelder-mead.pickle"
    try:
        with open(file, "rb") as f:
            old_results = pickle.load(f)
    except FileNotFoundError as e:
        old_results = []

    old_results.extend(results)
    with open(file, "rb") as f:
        pickle.dump(old_results, f)

    sys.exit()
    # result = minimize(multi_res_error, all_res_coefs, args=(
    #     exog_df.loc[:,exog_terms], df.loc[exog_df.index, "Release"],
    #     df.loc[exog_df.index, "Storage"],
    #     means, std, lib, coef_index, True, std_or_norm
    # ))

    new_coefs = {}
    for group, i in coef_map.items():
        new_coefs[group] = result.x[i*nprm:(i+1)*nprm]
    new_coefs = pd.DataFrame(new_coefs, index=coef_order)

    rel_output, sto_output = get_simulated_release(
        exog_df.loc[:, exog_terms],
        new_coefs,
        month_coefs,
        groups,
        means,
        std,
        # lib,
        simul_func,
        std_or_norm
    )
    act_release = df["Release"].unstack().loc[rel_output.index]
    act_storage = df["Storage"].unstack().loc[sto_output.index]
    rel_scores = get_scores(act_release, rel_output)
    sto_scores = get_scores(act_storage, sto_output)
    # print("Release Train Scores")
    # print(rel_scores.mean())
    # print("Storage Train Scores")
    # print(sto_scores.mean())

    train_out = pd.DataFrame({
        "Release_act":act_release.stack(),
        "Storage_act":act_storage.stack(),
        "Release_simul":rel_output.stack(),
        "Storage_simul":sto_output.stack()
    })

    exog_df = X_test.copy()
    rel_output, sto_output = get_simulated_release(
        exog_df.loc[:, exog_terms],
        new_coefs,
        month_coefs,
        groups,
        means,
        std,
        lib,
        std_or_norm
    )
    act_release = df["Release"].unstack().loc[rel_output.index]
    act_storage = df["Storage"].unstack().loc[sto_output.index]
    rel_scores = get_scores(act_release, rel_output)
    sto_scores = get_scores(act_storage, sto_output)
    # print("Release Test Scores")
    # print(rel_scores.mean())
    # print("Storage Test Scores")
    # print(sto_scores.mean())

    test_out = pd.DataFrame({
        "Release_act":act_release.stack(),
        "Storage_act":act_storage.stack(),
        "Release_simul":rel_output.stack(),
        "Storage_simul":sto_output.stack()
    })
    output = {"new_coefs":new_coefs, "results":result,
              "train":train_out, "test":test_out}

    with open(f"../results/simul_model/sto_and_rel_results.pickle", "wb") as f:
        pickle.dump(output, f)

def get_simulated_release(exog, coefs, month_coefs, groups, means, std, lib=None, std_or_norm=0):
    if lib == None:
        lib = load_library()
    resers = exog.index.get_level_values(1).unique()
    rel_output = pd.DataFrame()
    sto_output = pd.DataFrame()
    idx = pd.IndexSlice
    for res in resers:
        res_exog = exog.loc[idx[:,res],:]
        res_coef = coefs[groups.loc[res]].values
        res_month_coef = month_coefs[groups.loc[res]].values
        res_mean = means.loc[res]
        res_std = std.loc[res]
        res_rel, res_sto = simul_res(
            res_exog, res_coef, res_month_coef, res_mean, res_std, std_or_norm
        )
        rel_output[res] = pd.Series(
            res_rel,
            index=res_exog.index.get_level_values(0).values[6:]
        )
        sto_output[res] = pd.Series(
            res_sto,
            index=res_exog.index.get_level_values(0).values[6:]
        )
    # rel_output = pd.concat(rel_output)
    # sto_output = pd.concat(sto_output)
    return rel_output, sto_output

def find_res_groups(df):
    return df.groupby(df.index.get_level_values(1))["compositegroup"].unique().apply(
        lambda x: x[0]
    )

def get_scores(actual, model):
    scores = pd.DataFrame.from_records([
            (
                r2_score(actual[i], model[i]),
                mean_squared_error(actual[i], model[i], squared=False)
            )
            for i in actual.columns],
            index=actual.columns,
            columns=["NSE", "RMSE"]
    )
    return scores

def normalize_rmse(scores, actual):
    means = actual.mean()
    scores["nRMSE"] = (scores["RMSE"] / means) * 100
    return scores

def boosted_training(df, groups, filter_groups=None, scaler="mine"):
    for_groups = df.loc[:, groups]
    df.loc[:,"Storage_Inflow_interaction"] = df.loc[:,"Storage_pre"].mul(
        df.loc[:,"Net Inflow"])
    df.loc[:,"Storage_Release_interaction"] = df.loc[:,"Storage_pre"].mul(
        df.loc[:,"Release_pre"])
    df.loc[:,"Release_Inflow_interaction"] = df.loc[:,"Release_pre"].mul(
        df.loc[:,"Net Inflow"])
    
    scaled_df, means, std = scale_multi_level_df(df)
    X_scaled = scaled_df.loc[:, ["Storage_pre", "Net Inflow", "Release_pre",
                                    "Storage_Inflow_interaction",
                                    "Storage_Release_interaction",
                                    "Release_Inflow_interaction",
                                    "Release_roll7", "Release_roll14", "Storage_roll7",
                                    "Storage_roll14", "Inflow_roll7", "Inflow_roll14"] 
                            ]
    y_scaled = scaled_df.loc[:, "Release"]

    X_scaled[groups] = for_groups
    X_scaled = change_group_names(X_scaled, groups, group_names)
    
    # these reservoirs exhibit different characteristics than their group may suggest.
    change_names = ["Douglas", "Cherokee", "Hiwassee"]
    for ch_name in change_names:
        size = X_scaled[X_scaled.index.get_level_values(1) == ch_name].shape[0]
        X_scaled.loc[X_scaled.index.get_level_values(1) == ch_name, "NaturalOnly"] = ["NaturalFlow" for i in range(size)]

    X_scaled = combine_columns(X_scaled, groups, "compositegroup") 
    

    #* this introduces a intercept that varies monthly and between groups
    month_arrays = {i:[0 for i in range(X_scaled.shape[0])] for i in calendar.month_abbr[1:]}
    for i, date in enumerate(X_scaled.index.get_level_values(0)):
        abbr = calendar.month_abbr[date.month]
        month_arrays[abbr][i] = 1

    for key, array in month_arrays.items():
        X_scaled[key] = array
    
    split_date = datetime(2010,1,1)
    reservoirs = X_scaled.index.get_level_values(1).unique()

    X_test = X_scaled.loc[X_scaled.index.get_level_values(0) >= split_date - timedelta(days=8)]
    X_train = X_scaled.loc[X_scaled.index.get_level_values(0) < split_date]

    y_test = y_scaled.loc[y_scaled.index.get_level_values(0) >= split_date - timedelta(days=8)]
    y_train = y_scaled.loc[y_scaled.index.get_level_values(0) < split_date]

    N_time_train = X_train.index.get_level_values(0).unique().shape[0]
    N_time_test = X_test.index.get_level_values(0).unique().shape[0]

    group_dummies_test = pd.get_dummies(X_test["compositegroup"])
    group_dummies_train = pd.get_dummies(X_train["compositegroup"])

    X_test[group_dummies_test.columns] = group_dummies_test
    X_train[group_dummies_train.columns] = group_dummies_train

    name_map = {
        "ComboFlow-RunOfRiver":"ROR",
        "ComboFlow-StorageDam":"LRT",
        "NaturalFlow-StorageDam":"HRT"
    }
    X_test = X_test.rename(columns=name_map)
    X_train = X_train.rename(columns=name_map)

    # train model
    interaction_terms = ["Storage_Inflow_interaction"]
        
    exog_terms = [
        "Net Inflow", "Storage_pre", "Release_pre",
        "Storage_roll7",  "Inflow_roll7", "Release_roll7",
    ]


    dtrain = xgb.DMatrix(X_train.loc[:, exog_terms + interaction_terms],
                         label=y_train)
    dtest = xgb.DMatrix(X_test.loc[:, exog_terms + interaction_terms],
                         label=y_test)

    best_train_params = {
        "max_depth": 4,
        "eta": 0.2,
        "gamma": 6,
        "subsample": 1,
        "lambda": 1,
        "alpha": 1,
        "objective":"reg:squarederror",
        "nthread":cpu_count(),
        "eval_metric":"rmse",
        "tree_method":"hist"
    }
    best_train_num_round = 350

    best_test_params = {
        "max_depth": 3,
        "eta": 0.4,
        "gamma": 6,
        "subsample": 1,
        "lambda": 2,
        "alpha": 1,
        "objective":"reg:squarederror",
        "nthread":cpu_count(),
        "eval_metric":"rmse",
        "tree_method":"hist"
    }
    best_test_num_round = 150

    bst_train = xgb.train(best_train_params, dtrain, best_train_num_round)
    bst_test = xgb.train(best_test_params, dtrain, best_test_num_round)

    train_train_preds = bst_train.predict(dtrain)
    train_test_preds = bst_train.predict(dtest)
    test_train_preds = bst_test.predict(dtrain)
    test_test_preds = bst_test.predict(dtest)

    train_out = pd.DataFrame({"Release_act":y_train.values,
                              "Release_trprms":train_train_preds,
                              "Release_teprms":test_train_preds},
                             index=y_train.index)

    test_out = pd.DataFrame({"Release_act":y_test.values,
                             "Release_trprms":train_test_preds,
                             "Release_teprms":test_test_preds},
                             index=y_test.index)
    for col in train_out.columns:
        train_out[col] = (train_out[col].unstack() * std["Release"] + means["Release"]).stack()
        test_out[col] = (test_out[col].unstack() * std["Release"] + means["Release"]).stack()

    trtr_scores = get_scores(train_out["Release_act"].unstack(), train_out["Release_trprms"].unstack())
    trte_scores = get_scores(train_out["Release_act"].unstack(), train_out["Release_teprms"].unstack())
    tetr_scores = get_scores(test_out["Release_act"].unstack(), test_out["Release_trprms"].unstack())
    tete_scores = get_scores(test_out["Release_act"].unstack(), test_out["Release_teprms"].unstack())

    output = {
        "train":{
            "params":best_train_params,
            "num_round":best_train_num_round,
            "results":train_out,
        },
        "test":{
            "params":best_test_params,
            "num_round":best_test_num_round,
            "results":test_out
        },
        "scores":{
            "trtr":trtr_scores,
            "trte":trte_scores,
            "tetr":tetr_scores,
            "tete":tete_scores,
        }
    }
    with open("../results/xgboost/best_params_results.pickle", "wb") as f:
        pickle.dump(output, f)

    depths = np.arange(3,9)
    etas = np.arange(0,0.7,0.1)
    gammas = np.arange(0,11,2)
    subsamples = [0.5, 0.75, 1]
    lambdas = [1,1.5,2]
    alphas = [0,0.5,1]
    num_rounds = np.arange(50,400,50)

    grid = list(product(depths, etas, gammas,
                        subsamples, lambdas, alphas,
                        num_rounds))

    II()

    # grid_size = len(grid)
    # chunk_size = int(grid_size / 12)
    # output = []
    # outputpath = pathlib.Path("../results/xgboost/grid_search.pickle")
    # time1 = timer()
    # for i in range(9,12):
    #     print(f"\nWorking on Chunk {i+1} of 12")
    #     print(f"Elapsed Time: {timer() - time1:.3f} seconds")
    #     for sparams in grid[i*chunk_size:(i+1)*chunk_size]:
    #         param = {
    #             "max_depth": sparams[0],
    #             "eta": sparams[1],
    #             "gamma":sparams[2],
    #             "subsample":sparams[3],
    #             "lambda":sparams[4],
    #             "alpha":sparams[5],
    #             "objective":"reg:squarederror",
    #             "nthread":cpu_count(),
    #             "eval_metric":"rmse",
    #             "tree_method":"gpu_hist"
    #         }
    #         num_round = sparams[6]
    #         try:
                # bst = xgb.train(param, dtrain, num_round)
                # training = bst.predict(dtrain)
                # bst = xgb.train(param, dtest, num_round)
                # testing = bst.predict(dtest)
                # train_score = r2_score(y_train, training)
                # test_score = r2_score(y_test, testing)
                # output.append(
                #    (train_score, test_score)
                # )
    #         except:
    #             output.append((np.nan, np.nan))

    #     if outputpath.exists():
    #         with open(outputpath.as_posix(), "rb") as f:
    #             results = pickle.load(f)
    #     else:
    #         results = {"grid":grid, "results":[]}

    #     results["results"].extend(output)

    #     with open(outputpath.as_posix(), "wb") as f:
    #         pickle.dump({"grid":grid,"results":output}, f)

def change_group_names(df, groups, names):
    for group in groups:
        df[group] = [names[group][i] for i in df[group]]
    return df

def combine_columns(df, columns, new_name, sep="-"):
    df[new_name] = df[columns[0]].str.cat(df[columns[1:]], sep=sep)
    return df


LIB = load_library()
SIMUL_FUNC = LIB.res_simul

if __name__ == "__main__":
    df = read_tva_data()
    boosted_training(df, groups = ["NaturalOnly","RunOfRiver"])
