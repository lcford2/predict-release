import sys
import pickle
import calendar
import pathlib
from time import perf_counter as timer
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
import ctypes as ct
from numpy.ctypeslib import ndpointer
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLMParams
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from utils.helper_functions import (read_tva_data, scale_multi_level_df,
                              read_all_res_data, find_max_date_range)
from utils.timing_function import time_function
from datetime import timedelta, datetime
from collections import defaultdict
from itertools import combinations
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

def simul_res(res_exog, coefs, means, std, std_or_norm=0):
    # intercept = coefs["const"]
    # re_coefs = coefs[[
    #     "Release_pre", "Storage_pre", "Net Inflow",
    #     "Storage_Inflow_interaction",
    #     "Release_roll7", "Storage_roll7", "Inflow_roll7"
    # ]].values
    intercept = coefs[0]
    re_coefs  = coefs[1:-12]
    month_coefs = coefs[-12:]

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

def multi_res_error(coefs, X, act_rel, act_sto, means, std, coef_index, rel_and_sto=False, std_or_norm=0):
    error = 0
    idx = pd.IndexSlice

    for res, index in coef_index.items():
        res_coefs = coefs[index[0]:index[1]]
        res_exog = X.loc[idx[:,res],:]
        res_means = means.loc[res]
        res_std = std.loc[res]
        rel_out, sto_out = simul_res(res_exog,
                                     res_coefs,
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
    return minimize(func, init_params, args=args[2:])

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
        *calendar.month_abbr[1:]
    ]

    coef_map = {
        'NaturalFlow-StorageDam': 0,
        'ComboFlow-StorageDam': 1,
        'ComboFlow-RunOfRiver': 2
    }
    coef_index = {res:(coef_map[g]*20,(coef_map[g]+1)*20) for res, g in groups.items()}

    all_res_coefs = [
        *init_params.loc[coef_order,"NaturalFlow-StorageDam"].values,
        *init_params.loc[coef_order,"ComboFlow-StorageDam"].values,
        *init_params.loc[coef_order,"ComboFlow-RunOfRiver"].values
    ]

    # result = minimize(single_res_error, res_coefs.values, args=(
    #     res_exog, df.loc[res_exog.index, "Release"].values[6:],
    #     res_means, res_std, lib)
    # )

    std_or_norm = 0
    sn_label = "std" if std_or_norm == 0 else "norm"

    N_trials = 28
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

    with open("../results/simul_model/multi_trial.pickle", "rb") as f:
        old_results = pickle.load(f)

    old_results.extend(results)
    with open("../results/simul_model/multi_trial.pickle", "wb") as f:
        pickle.dump(old_results, f)

    sys.exit()
    # result = minimize(multi_res_error, all_res_coefs, args=(
    #     exog_df.loc[:,exog_terms], df.loc[exog_df.index, "Release"],
    #     df.loc[exog_df.index, "Storage"],
    #     means, std, lib, coef_index, True, std_or_norm
    # ))

    new_coefs = {}
    for group, i in coef_map.items():
        new_coefs[group] = result.x[i*20:(i+1)*20]
    new_coefs = pd.DataFrame(new_coefs, index=coef_order)

    rel_output, sto_output = get_simulated_release(
        exog_df.loc[:, exog_terms],
        new_coefs,
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

def get_simulated_release(exog, coefs, groups, means, std, lib=None, std_or_norm=0):
    if lib == None:
        lib = load_library()
    resers = exog.index.get_level_values(1).unique()
    rel_output = pd.DataFrame()
    sto_output = pd.DataFrame()
    idx = pd.IndexSlice
    for res in resers:
        res_exog = exog.loc[idx[:,res],:]
        res_coef = coefs[groups.loc[res]].values
        res_mean = means.loc[res]
        res_std = std.loc[res]
        res_rel, res_sto = simul_res(
            res_exog, res_coef, res_mean, res_std, std_or_norm
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

def scaled_MixedEffects(df, groups, filter_groups=None, scaler="mine"):
    filename = "-".join(groups)
    for_groups = df.loc[:,groups]
    fraction_names = ["Fraction_Storage",
                      "Fraction_Net Inflow"]
    fractions = df.loc[:, fraction_names]
    if filter_groups:
        filename += "_filter"
        for key, value in filter_groups.items():
            df = df.loc[df[key] == inverse_groupnames[key][value],:]
            filename += f"_{value}"
    
    df = df.copy()

    df["Storage_Inflow_interaction"] = df.loc[:,"Storage_pre"].mul(
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
    X_scaled[fraction_names] = fractions

    X_scaled = change_group_names(X_scaled, groups, group_names)
    
    # these reservoirs exhibit different characteristics than their group may suggest.
    change_names = ["Douglas", "Cherokee", "Hiwassee"]
    for ch_name in change_names:
        size = X_scaled[X_scaled.index.get_level_values(1) == ch_name].shape[0]
        X_scaled.loc[X_scaled.index.get_level_values(1) == ch_name, "NaturalOnly"] = ["NaturalFlow" for i in range(size)]

    X_scaled = combine_columns(X_scaled, groups, "compositegroup") 
    
    # X_scaled = X_scaled[~X_scaled.index.get_level_values(1).isin(change_names)]   
    # y_scaled = y_scaled[~y_scaled.index.get_level_values(1).isin(change_names)]
    # means = means[~means.index.isin(change_names)]
    # std = std[~std.index.isin(change_names)]
    

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

    X_test = X_scaled.loc[X_scaled.index.get_level_values(0) >= split_date - timedelta(days=8)]
    X_train = X_scaled.loc[X_scaled.index.get_level_values(0) < split_date]

    y_test = y_scaled.loc[y_scaled.index.get_level_values(0) >= split_date - timedelta(days=8)]
    y_train = y_scaled.loc[y_scaled.index.get_level_values(0) < split_date]

    N_time_train = X_train.index.get_level_values(0).unique().shape[0]
    N_time_test = X_test.index.get_level_values(0).unique().shape[0]

    # train model
    # Instead of adding a constant, I want to create dummy variable that accounts for season differences
    exog = X_train
    exog.loc[:,"const"] = 1.0
    
    groups = exog["compositegroup"]
    
    interaction_terms = ["Storage_Inflow_interaction"]
        
    exog_terms = [
        "const", "Net Inflow", "Storage_pre", "Release_pre",
        "Storage_roll7",  "Inflow_roll7", "Release_roll7"
    ]

    exog_re = exog.loc[:,exog_terms + interaction_terms + calendar.month_abbr[1:]]

    mexog = exog.loc[:,["const"]]

    actual_inflow_test = df["Net Inflow"].loc[df.index.get_level_values(
        0) >= split_date - timedelta(days=8)]

    #* fit the model 
    free = MixedLMParams.from_components(fe_params=np.ones(mexog.shape[1]),
                                        cov_re=np.eye(exog_re.shape[1]))
    md = sm.MixedLM(y_train, mexog, groups=groups, exog_re=exog_re)
    mdf = md.fit(free=free)
    
    #* extract the fitted values (i.e., training set dependent variable predictions)
    fitted = mdf.fittedvalues
    #* tranform variables back to normal space for evaluation
    fitted_act = (fitted.unstack() * 
                std.loc[:, "Release"] + means.loc[:, "Release"]).stack()
    y_train_act = (y_train.unstack() *
                   std.loc[:, "Release"] + means.loc[:, "Release"]).stack()
    y_test_act = (y_test.unstack() *
                   std.loc[:, "Release"] + means.loc[:, "Release"]).stack()

    #* get the NSE and RMSE for the training set
    f_act_score = r2_score(y_train_act, fitted_act)
    f_act_rmse = np.sqrt(mean_squared_error(y_train_act, fitted_act))

    #* extract fitted parameters
    fe_coefs = mdf.params
    re_coefs = mdf.random_effects

    #* calculate total and monthly bias for fitted values
    y_train_mean = y_train_act.groupby(
        y_train_act.index.get_level_values(1)
    ).mean()
    y_test_mean = y_test_act.groupby(
        y_test_act.index.get_level_values(1)
    ).mean()
    fmean = fitted_act.groupby(
        fitted_act.index.get_level_values(1)
    ).mean()

    f_bias = fmean - y_train_mean
    f_bias_month = fitted_act.groupby(
        fitted_act.index.get_level_values(0).month
        ).mean() - y_train_act.groupby(
            y_train_act.index.get_level_values(0).month).mean()

    #* predict on testing data
    X_test["const"] = 1
    exog = X_test
    groups = exog["compositegroup"]
    exog_re = exog.loc[:, exog_terms + interaction_terms + 
                        calendar.month_abbr[1:] + ["compositegroup"]]
    mexog = exog[["const"]]
        
    preds = predict_mixedLM(fe_coefs, re_coefs, mexog, exog_re, "compositegroup")

    #* simulate/forecast reservoirs
    coefs = pd.DataFrame(mdf.random_effects)
    idx = pd.IndexSlice
    #* provide storage_pre_act 
    exog_re.loc[idx[datetime(2010,1,1),:], "Storage_pre_act"] = df.loc[idx[datetime(2010,1,1),:], "Storage_pre"]
    forecasted = forecast_mixedLM_new(coefs, exog_re, means, std, "compositegroup", actual_inflow_test)


    preds_act = (preds.unstack() *
                    std.loc[:, "Release"] + means.loc[:, "Release"]).stack()
    y_test_act = (y_test.unstack() *
                    std.loc[:, "Release"] + means.loc[:, "Release"]).stack()
    y_act = y_train_act.append(y_test_act).sort_index()
    forecasted = forecasted[forecasted.index.get_level_values(0).year >= 2010]
    forecasted_act = forecasted.loc[:, "Release_act"]
    forecast_score_rel = r2_score(y_test_act.loc[forecasted_act.index], 
                              forecasted_act)

    preds_mean = preds_act.groupby(
        preds_act.index.get_level_values(1)
    ).mean()

    p_bias = preds_mean - y_test_mean 
    p_act_score = r2_score(y_test_act, preds_act)
    p_norm_score = r2_score(y_test, preds)
    p_act_rmse = np.sqrt(mean_squared_error(y_test_act, preds_act))
    p_norm_rmse = np.sqrt(mean_squared_error(y_test, preds))

    p_bias_month = preds_act.groupby(
        preds_act.index.get_level_values(0).month  
        ).mean() - y_test_act.groupby(
            y_test_act.index.get_level_values(0).month).mean()


    fitted_act = fitted_act.unstack()
    y_train_act = y_train_act.unstack()
    y_test_act = y_test_act.unstack()
    preds_act = preds_act.unstack()
    forecasted_act = forecasted_act.unstack()

    res_scores = pd.DataFrame(index=reservoirs, columns=["NSE", "RMSE"])
    for res in reservoirs:
        try:
            y = y_train_act[res]
            ym = fitted_act[res]
        except KeyError as e:
            y = y_test_act[res]
            ym = preds_act[res]
        res_scores.loc[res, "NSE"] = r2_score(y, ym)
        res_scores.loc[res, "RMSE"] = np.sqrt(mean_squared_error(y, ym))



    coefs = pd.DataFrame(mdf.random_effects)
    train_data = pd.DataFrame(dict(actual=y_train_act.stack(), model=fitted_act.stack()))
    test_p_data = pd.DataFrame(dict(actual=y_test_act.stack(), model=preds_act.stack()))
    test_f_data = pd.DataFrame(dict(actual=y_test_act.stack(), model=forecasted_act.stack())).dropna()
    
    train_quant, train_bins = pd.qcut(train_data["actual"], 3, labels=False, retbins=True)
    test_quant, test_bins = pd.qcut(test_p_data["actual"], 3, labels=False, retbins=True)

    train_data["bin"] = train_quant
    test_p_data["bin"] = test_quant
    test_f_data["bin"] = test_quant

    train_quant_scores = pd.DataFrame(index=[0,1,2], columns=["NSE", "RMSE"])
    for q in [0,1,2]:
        score = r2_score(
            train_data[train_data["bin"] == q]["actual"],
            train_data[train_data["bin"] == q]["model"],
        )
        rmse = np.sqrt(mean_squared_error(
            train_data[train_data["bin"] == q]["actual"],
            train_data[train_data["bin"] == q]["model"],
        ))
        train_quant_scores.loc[q] = [score, rmse]
    train_quant_table = train_quant_scores.to_markdown(tablefmt="github", floatfmt=".3f")
    print(train_quant_table) 

    test_p_quant_scores = pd.DataFrame(index=[0,1,2], columns=["NSE", "RMSE"])
    for q in [0,1,2]:
        score = r2_score(
            test_p_data[test_p_data["bin"] == q]["actual"],
            test_p_data[test_p_data["bin"] == q]["model"],
        )
        rmse = np.sqrt(mean_squared_error(
            test_p_data[test_p_data["bin"] == q]["actual"],
            test_p_data[test_p_data["bin"] == q]["model"],
        ))
        test_p_quant_scores.loc[q] = [score, rmse]
    test_p_quant_table = test_p_quant_scores.to_markdown(tablefmt="github", floatfmt=".3f")
    print(test_p_quant_table) 

    test_f_quant_scores = pd.DataFrame(index=[0,1,2], columns=["NSE", "RMSE"])
    for q in [0,1,2]:
        score = r2_score(
            test_f_data[test_f_data["bin"] == q]["actual"],
            test_f_data[test_f_data["bin"] == q]["model"],
        )
        rmse = np.sqrt(mean_squared_error(
            test_f_data[test_f_data["bin"] == q]["actual"],
            test_f_data[test_f_data["bin"] == q]["model"],
        ))
        test_f_quant_scores.loc[q] = [score, rmse]
    test_f_quant_table = test_f_quant_scores.to_markdown(tablefmt="github", floatfmt=".3f")
    print(test_f_quant_table) 

    output = dict(
        coefs=coefs,
        f_act_score=f_act_score,
        f_act_rmse=f_act_rmse,
        f_bias=f_bias,
        f_bias_month=f_bias_month,
        data=dict(
            X_test=X_test,
            y_test=y_test,
            X_train=X_train,
            y_train=y_train,
            y_results=train_data,
            y_pred=test_p_data,
            y_forc=test_f_data,
            forecasted=forecasted[["Release", "Storage", "Release_act", "Storage_act"]]
        )
    )

    output_dir = pathlib.Path(f"../results/agu_2021_runs/simple_model_temporal_validation_all_res")
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    with open(output_dir / "results.pickle", "wb") as f:
        pickle.dump(output, f, protocol=4)

@time_function
def predict_mixedLM(fe_coefs, re_coefs, exog_fe, exog_re, group_col):
    output_df = pd.Series(index=exog_re.index, dtype="float64")
    groups = exog_re[group_col].unique()
    re_keys = re_coefs[groups[0]].keys()
    fe_keys = exog_fe.columns
    for group in groups:
        re_df = exog_re.loc[exog_re[group_col] == group, re_keys]
        index = re_df.index
        fe_df = exog_fe.loc[index, fe_keys]
        output_df.loc[index] = re_df.dot(re_coefs[group]) + fe_df.dot(fe_coefs[fe_keys])
    return output_df

@time_function
def forecast_mixedLM_new(coefs, exog, means, std, group_col, actual_inflow, timelevel="all", tree=False):
    # create output data frame
    output_df = pd.DataFrame(index=exog.index, 
                             columns=list(exog.columns) + ["Storage_act", "Release_act"],
                             dtype="float64")
    # get important indexers
    groups = exog[group_col].unique()
    re_keys = coefs[groups[0]].keys()

    # define date ranges
    start_date = exog.index.get_level_values(0)[0] + timedelta(days=8)
    end_date = exog.index.get_level_values(0)[-1]
    pre_dates = pd.date_range(start=exog.index.get_level_values(0)[0], end=start_date)
    # columns that need to be calculated every time step
    calc_columns = ["Release_pre", "Storage_pre", "Storage_pre_act",
                    "Release_roll7", "Storage_roll7",
                    "Storage_act", "Release_act"]
    # define where to stop adding columns to output df
    if "Storage_roll7" in exog.columns:
        stop_it = 5
    else:
        stop_it = 4    
    
    # setup output df by adding pre_date information
    idx = pd.IndexSlice
    for col in calc_columns[:stop_it]:
        output_df.loc[idx[pre_dates,:], col] = exog.loc[idx[pre_dates,:], col]

    for col in output_df.columns:
        if col not in calc_columns:
            output_df[col] = exog.loc[:,col]

    # actual iteration dates
    dates = pd.date_range(start=start_date, end=end_date)
    date = dates[0]
    output_df[group_col] = exog[group_col]
    resers = exog.index.get_level_values(1).unique()

    # determine how to get parameters if it is a tree or simple model
    if tree:
        coeffs = pd.DataFrame(coefs).T
    else:
        coeffs = pd.DataFrame(index=resers, columns=re_keys, dtype="float64")
        for res in resers:
            coeffs.loc[res, re_keys] = coefs[exog.loc[(date, res), group_col]]

    # extract stdztion info for convience
    sto_means = means["Storage"]
    inf_means = means["Net Inflow"]
    rel_means = means["Release"]
    sto_std = std["Storage"]
    inf_std = std["Net Inflow"]
    rel_std = std["Release"]
    
    # it is possible to have time varying means, though they provide no help
    if timelevel != "all":
        sto_means = sto_means.unstack()
        inf_means = inf_means.unstack()
        rel_means = rel_means.unstack()
        sto_std = sto_std.unstack()
        inf_std = inf_std.unstack()
        rel_std = rel_std.unstack() 
   
    my_coefs = pd.DataFrame(index=resers, columns=re_keys, dtype="float64")

    for date in dates:
        #* extract todays data
        dexog = output_df.loc[idx[date,:]][re_keys]
        #* get parameters for todays data
        if tree:
            for res in resers:
                param_group = exog.loc[idx[date,res],group_col]
                my_coefs.loc[res,re_keys] = coeffs.loc[param_group,re_keys]
        else:
            my_coefs = coeffs

        # calculate todays standardized release
        rel = dexog.mul(my_coefs).sum(axis=1)

        if timelevel != "all":
            tl = getattr(date, timelevel)
            rel_act = rel*rel_std.loc[tl] + rel_means.loc[tl]
            # stor_pre_act = output_df.loc[idx[date,:]]["Storage_pre"]*sto_p_std.loc[tl] + sto_p_means.loc[tl]
            inflow_act = output_df.loc[idx[date,:]]["Net Inflow"]*inf_std.loc[tl] + inf_means.loc[tl]
            # inflow_act = actual_inflow.loc[idx[date,:]]
        else:
            #* back transform values needed for mass balance
            rel_act = rel*rel_std+rel_means
            # stor_pre_act = output_df.loc[idx[date, :]
            #                              ]["Storage_pre"] * sto_std + sto_means
            # stor_pre_act = output_df.loc[idx[date, :]
            #                              ]["Storage_pre_act"]
            inflow_act = output_df.loc[idx[date, :]
                                       ]["Net Inflow"] * inf_std + inf_means
            # inflow_act = actual_inflow.loc[idx[date,:]]
        stor_pre_act = output_df.loc[idx[date, :]]["Storage_pre_act"]

        # inflow_act.index = inflow_act.index.get_level_values(1)
        #* perform mass balance to get current storage
        storage_act = stor_pre_act + inflow_act - rel_act
            
        if timelevel != "all":
            tl = getattr(date, timelevel)
            storage = (storage_act - sto_means.loc[tl])/sto_std.loc[tl]
        else:
            storage = (storage_act-sto_means)/sto_std

        # store standardized and actual release and storage
        output_df.loc[idx[date,:],"Release"] = rel.values
        output_df.loc[idx[date,:],"Storage"] = storage.values
        output_df.loc[idx[date,:],"Release_act"] = rel_act.values
        output_df.loc[idx[date,:],"Storage_act"] = storage_act.values

        # calculate rolling means for next time step
        tmrw = date+timedelta(days=1)
        tmp = output_df.groupby(
            output_df.index.get_level_values(1))[
                ["Storage_act", "Release_act"]].rolling(
                    7, min_periods=1).mean()
        tmp.index = tmp.index.droplevel(0)
        tmp = tmp.sort_index()
        tmp = tmp.loc[idx[date,:]]
        tmp.loc[:,"Storage_act"] = (tmp.loc[:,"Storage_act"] - sto_means)/sto_std
        tmp.loc[:,"Release_act"] = (tmp.loc[:,"Release_act"] - rel_means)/rel_std
    
        month_len = calendar.monthrange(date.year, date.month)[1]
        # as long as there is another time step to simulate, set it up
        if date != end_date:
            # if date.timetuple().tm_yday in (365, 366):
            # if date.weekday() == 6:# or date.day == int(month_len//2):
            # output_df.loc[idx[tmrw,:],"Release_pre"] = exog.loc[idx[tmrw,:], "Release_pre"]
            #* add calculated release values
            output_df.loc[idx[tmrw,:], "Release_pre"] = rel.values
                # output_df.loc[idx[tmrw,:],"Storage_pre"] = exog.loc[idx[tmrw,:], "Storage_pre"]
            # else:
                # output_df.loc[idx[tmrw,:],"Release_pre"] = rel.values
            #* add calculated storage values (standardized)
            output_df.loc[idx[tmrw,:],"Storage_pre"] = storage.values
            #* add calculated storage values (normal space)
            output_df.loc[idx[tmrw,:],"Storage_pre_act"] = storage_act.values
            #* add rolling terms
            output_df.loc[idx[tmrw, :], [
                "Storage_roll7", "Release_roll7"]] = tmp.values       

    try:
        output_df = output_df.drop(calendar.month_abbr[1:], axis=1)
    except KeyError as e:
        pass

    return output_df

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
    # forecast_mixedLM_other_res(groups=["NaturalOnly", "RunOfRiver"])
    # scaled_MixedEffects(df, groups = ["NaturalOnly","RunOfRiver"])
                            # filter_groups={"NaturalOnly": "NaturalFlow"})
                        # filter_groups={"NaturalOnly":"ComboFlow"})
    fit_simul_res(df, SIMUL_FUNC, groups=["NaturalOnly", "RunOfRiver"])
