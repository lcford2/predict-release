import sys
import pickle
import calendar
import pathlib
from time import perf_counter as timer
import pandas as pd
import numpy as np
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

@time_function
def scaledOLS(df, scaler="mine"):
    if scaler == "mine":
        scaled_df, means, std = scale_multi_level_df(df)
        X_scaled = scaled_df.loc[:, ["Storage_pre", "Net Inflow", "Release_pre"]]
        y_scaled = scaled_df.loc[:, "Release"]
    else:
        scaler = StandardScaler()
        X = df.loc[:,["Storage_pre", "Net Inflow", "Release_pre"]]
        y = df.loc[:,"Release"]

        x_scaler = scaler.fit(X)
        X_scaled = x_scaler.transform(X)

        y_scaler = scaler.fit(y.values.reshape(-1,1))
        y_scaled = y_scaler.transform(y.values.reshape(-1, 1))

        X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
        y_scaled = pd.Series(y_scaled.reshape(-1), index=y.index)

    X_test = X_scaled[X_scaled.index.get_level_values(0).year >= 2010]
    X_train = X_scaled[X_scaled.index.get_level_values(0).year < 2010]

    y_test = y_scaled[y_scaled.index.get_level_values(0).year >= 2010]
    y_train = y_scaled[y_scaled.index.get_level_values(0).year < 2010]

    model = sm.OLS(y_train, sm.add_constant(X_train))
    fit = model.fit()

    preds = pd.DataFrame(fit.predict(sm.add_constant(X_test)), columns=["Release"])
    fitted = pd.DataFrame(fit.fittedvalues, columns=["Release"])
    y_test = pd.DataFrame(y_test, columns=["Release"])
    y_train = pd.DataFrame(y_train, columns=["Release"])

    if scaler == "mine":
        idx = pd.IndexSlice
        for index in means.index:
            mu, sigma = means.loc[index, "Release"], std.loc[index, "Release"]
            preds.loc[idx[:,index],:] = (preds.loc[idx[:,index],:] * sigma) + mu
            fitted.loc[idx[:,index],:] = (fitted.loc[idx[:,index],:] * sigma) + mu
            y_train.loc[idx[:,index],:] = (y_train.loc[idx[:,index],:] * sigma) + mu
            y_test.loc[idx[:,index],:] = (y_test.loc[idx[:,index],:] * sigma) + mu
    else:
        fitted = y_scaler.inverse_transform(fit.fittedvalues)
        preds = y_scaler.inverse_transform(preds)
        y_train = y_scaler.inverse_transform(y_train)
        y_test = y_scaler.inverse_transform(y_test)

    train_set = fitted.join(y_train.rename(columns={"Release": "Train"}))
    test_set = preds.join(y_test.rename(columns={"Release": "Test"}))
    train_set.rename(columns={"Release": "Fitted"}, inplace=True)
    test_set.rename(columns={"Release": "Preds"}, inplace=True)

    scaled_OLS_results = {
        "means":means,
        "std":std,
        "fit":fit,
        "model":model,
        "test_set":test_set,
        "train_set":train_set,
        "scaled_df":scaled_df
    }

    with open("../results/simple_model_results/scaled_OLS_results.pickle", "wb") as f:
        pickle.dump(scaled_OLS_results, f)

    print(f"Train Score : {r2_score(y_train, fitted):.4f}")
    print(f"Test Score  : {r2_score(y_test, preds):.4f}")


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

def scaled_MixedEffects(df, groups, filter_groups=None, scaler="mine"):
    filename = "-".join(groups)
    for_groups = df.loc[:,groups]
    fraction_names = ["Fraction_Storage",
                    #   "Fraction_Release", 
                      "Fraction_Net Inflow"]
    fractions = df.loc[:, fraction_names]
    if filter_groups:
        filename += "_filter"
        for key, value in filter_groups.items():
            df = df.loc[df[key] == inverse_groupnames[key][value],:]
            # y_scaled = y_scaled.loc[X_scaled.index]
            filename += f"_{value}"
    
    df = df.copy()

    df["Storage_Inflow_interaction"] = df.loc[:,"Storage_pre"].mul(
        df.loc[:,"Net Inflow"])
    df.loc[:,"Storage_Release_interaction"] = df.loc[:,"Storage_pre"].mul(
        df.loc[:,"Release_pre"])
    df.loc[:,"Release_Inflow_interaction"] = df.loc[:,"Release_pre"].mul(
        df.loc[:,"Net Inflow"])
    # sys.exit()
    # extra_lag_terms = np.ravel([[f"Storage_{i}pre", f"Release_{i}pre"] for i in range(2,8)]).tolist()
    # extra_lag_terms = [f"Release_{i}pre" for i in range(2,8)]
    if scaler == "mine":
        scaled_df, means, std = scale_multi_level_df(df)
        X_scaled = scaled_df.loc[:, ["Storage_pre", "Net Inflow", "Release_pre",
                                     "Storage_Inflow_interaction",
                                     "Storage_Release_interaction",
                                     "Release_Inflow_interaction",
                                     "Release_roll7", "Release_roll14", "Storage_roll7",
                                     "Storage_roll14", "Inflow_roll7", "Inflow_roll14"] 
                                ]
        y_scaled = scaled_df.loc[:, "Release"]
        y_scaled_sto = scaled_df.loc[:, "Storage"]
    else:
        scaler = StandardScaler()
        X = df.loc[:,["Storage_pre", "Net Inflow", "Release_pre"] + 
                  extra_lag_terms + ["Storage_roll7", "Release_roll7", "Inflow_roll7"]
                  ]
        y = df.loc[:,"Release"]

        x_scaler = scaler.fit(X)
        X_scaled = x_scaler.transform(X)

        y_scaler = scaler.fit(y.values.reshape(-1,1))
        y_scaled = y_scaler.transform(y.values.reshape(-1, 1))

        X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
        y_scaled = pd.Series(y_scaled.reshape(-1), index=y.index)

    # X_scaled = df.loc[X_scaled.index, X_scaled.columns]
    # y_scaled = df.loc[y_scaled.index, y_scaled.name]
    # means = pd.DataFrame(0,index=means.index,columns=means.columns)
    # std = pd.DataFrame(1,index=std.index,columns=std.columns)
    # II()
    # sys.exit()
    
    X_scaled[groups] = for_groups
    X_scaled[fraction_names] = fractions

    X_scaled = change_group_names(X_scaled, groups, group_names)
    
    # these reservoirs exhibit different characteristics than their group may suggest.
    change_names = ["Douglas", "Cherokee", "Hiwassee"]
    for ch_name in change_names:
        size = X_scaled[X_scaled.index.get_level_values(1) == ch_name].shape[0]
        X_scaled.loc[X_scaled.index.get_level_values(1) == ch_name, "NaturalOnly"] = ["NaturalFlow" for i in range(size)]

    X_scaled = combine_columns(X_scaled, groups, "compositegroup") 
    # II()
    X_scaled = X_scaled[~X_scaled.index.get_level_values(1).isin(change_names)]   
    y_scaled = y_scaled[~y_scaled.index.get_level_values(1).isin(change_names)]
    y_scaled_sto = y_scaled_sto[~y_scaled_sto.index.get_level_values(1).isin(change_names)]
    means = means[~means.index.isin(change_names)]
    std = std[~std.index.isin(change_names)]
    
    #* This lets me group by month 
    # X_scaled["compositegroup"] = [calendar.month_abbr[i.month]
    #                      for i in X_scaled.index.get_level_values(0)]

    #* this introduces a intercept that varies monthly and between groups
    month_arrays = {i:[0 for i in range(X_scaled.shape[0])] for i in calendar.month_abbr[1:]}
    for i, date in enumerate(X_scaled.index.get_level_values(0)):
        abbr = calendar.month_abbr[date.month]
        month_arrays[abbr][i] = 1
    # for date in X_scaled.index.get_level_values(0):
    #     for key in month_arrays.keys():
    #         if calendar.month_abbr[date.month] == key:
    #             month_arrays[key].append(1)
    #         else:
    #             month_arrays[key].append(0)

    for key, array in month_arrays.items():
        X_scaled[key] = array
    
    split_date = datetime(2010,1,1)
    reservoirs = X_scaled.index.get_level_values(1).unique()

    # test_index = X_scaled.index[X_scaled.index.get_level_values(1).isin(lvout_res)]
    train_index = X_scaled.index
    # test_res = lvout_res
    train_res = train_index.get_level_values(1).unique()


    X_test = X_scaled.loc[X_scaled.index.get_level_values(0) >= split_date - timedelta(days=8)]
    X_train = X_scaled.loc[X_scaled.index.get_level_values(0) < split_date]

    y_test = y_scaled.loc[y_scaled.index.get_level_values(0) >= split_date - timedelta(days=8)]
    y_train = y_scaled.loc[y_scaled.index.get_level_values(0) < split_date]
    y_test_sto = y_scaled_sto.loc[y_scaled_sto.index.get_level_values(0) >= split_date - timedelta(days=8)]
    y_train_sto = y_scaled_sto.loc[y_scaled_sto.index.get_level_values(0) < split_date]

    # X_train = X_scaled.loc[train_index]
    # X_test = X_scaled.loc[test_index]
    # y_train = y_scaled.loc[train_index]
    # y_test = y_scaled.loc[test_index]

    N_time_train = X_train.index.get_level_values(0).unique().shape[0]
    N_time_test = X_test.index.get_level_values(0).unique().shape[0]

    # train model
    # Instead of adding a constant, I want to create dummy variable that accounts for season differences
    exog = X_train
    exog["const"] = 1.0
    
    groups = exog["compositegroup"]
    # Storage Release Interactions are near Useless
    # Release Inflow Interaction does not provide a lot either
    # Storage Inflow interaction seems to matter for ComboFlow-StorageDam reservoirs.
    interaction_terms = ["Storage_Inflow_interaction"]
        
        # exog_terms = [
        #     "Storage_pre", "Net Inflow", "Release_pre", 
        #     ] + extra_lag_terms


    exog_terms = [
        "const", "Net Inflow", "Storage_pre", "Release_pre",
        "Storage_roll7",  "Inflow_roll7", "Release_roll7"
    ]

        # exog_terms = [
        #     "const", "Net Inflow", "sto_diff", "Release_pre",
        #     "Release_roll7", "Inflow_roll7"
        # ]

    exog_re = exog.loc[:,exog_terms + interaction_terms + calendar.month_abbr[1:]]

    mexog = exog.loc[:,["const"]]

        # actual_inflow_train = df["Net Inflow"].loc[df.index.get_level_values(
        #     0) < split_date]
        # actual_inflow_test = df["Net Inflow"].loc[df.index.get_level_values(
        #     0) >= split_date - timedelta(days=8)]
        
    free = MixedLMParams.from_components(fe_params=np.ones(mexog.shape[1]),
                                        cov_re=np.eye(exog_re.shape[1]))
    md = sm.MixedLM(y_train, mexog, groups=groups, exog_re=exog_re)
    mdf = md.fit(free=free)
    
    fitted = mdf.fittedvalues
    fitted_act = (fitted.unstack() * 
                std.loc[train_res, "Release"] + means.loc[train_res, "Release"]).stack()
    y_train_act = (y_train.unstack() *
                   std.loc[train_res, "Release"] + means.loc[train_res, "Release"]).stack()
    y_test_act = (y_test.unstack() *
                   std.loc[:, "Release"] + means.loc[:, "Release"]).stack()


    f_act_score = r2_score(y_train_act, fitted_act)
    f_act_rmse = np.sqrt(mean_squared_error(y_train_act, fitted_act))

    fe_coefs = mdf.params
    re_coefs = mdf.random_effects

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

    # if label == "0":
    #     p_act_score = f_act_score
    #     p_norm_score = f_norm_score
    #     p_act_rmse = f_act_rmse
    #     p_norm_rmse = f_norm_rmse
    #     p_act_score_all = f_act_score
    #     p_act_rmse_all = f_act_rmse
    #     p_bias = f_bias
    #     p_bias_month = f_bias_month
    # else:
    X_test["const"] = 1
    X_all = X_train.append(X_test).sort_index()
    exog = X_test
    groups = exog["compositegroup"]
    exog_re = exog.loc[:, exog_terms + interaction_terms + 
                        calendar.month_abbr[1:] + ["compositegroup"]]
    mexog = exog[["const"]]
        
    preds = predict_mixedLM(fe_coefs, re_coefs, mexog, exog_re, "compositegroup")

    exog = X_all
    groups = exog["compositegroup"] 
    exog_re = exog.loc[:, exog_terms + interaction_terms +
                    calendar.month_abbr[1:] + ["compositegroup"]]
    mexog = exog[["const"]]
    preds_all = predict_mixedLM(fe_coefs, re_coefs, mexog, exog_re, "compositegroup")

    preds_act = (preds.unstack() *
                    std.loc[:, "Release"] + means.loc[:, "Release"]).stack()
    preds_act_all = (preds_all.unstack() *
                    std.loc[:, "Release"] + means.loc[:, "Release"]).stack()
    y_test_act = (y_test.unstack() *
                    std.loc[:, "Release"] + means.loc[:, "Release"]).stack()
    y_act = y_train_act.append(y_test_act).sort_index()

    preds_mean = preds_act.groupby(
        preds_act.index.get_level_values(1)
    ).mean()

    preds_mean_all = preds_act_all.groupby(
        preds_act_all.index.get_level_values(1)
    ).mean()

    p_act_score_all = r2_score(y_act, preds_act_all)
    p_act_rmse_all = np.sqrt(mean_squared_error(y_act, preds_act_all))
    p_bias = preds_mean - y_test_mean 
    p_act_score = r2_score(y_test_act, preds_act)
    p_norm_score = r2_score(y_test, preds)
    p_act_rmse = np.sqrt(mean_squared_error(y_test_act, preds_act))
    p_norm_rmse = np.sqrt(mean_squared_error(y_test, preds))

    p_bias_month = preds_act.groupby(
        preds_act.index.get_level_values(0).month  
        ).mean() - y_test_act.groupby(
            y_test_act.index.get_level_values(0).month).mean()

    # lvout_rt_results[label] = {
    #     "pred": {
    #         "p_act_score": p_act_score,
    #         "p_act_rmse": p_act_rmse,
    #         "p_act_score_all":p_act_score_all,
    #         "p_act_rmse_all":p_act_rmse_all,
    #         "p_bias":p_bias,
    #         "p_bias_month":p_bias_month
    #     },
    #     "fitted":{
    #         "f_act_score": f_act_score,
    #         "f_act_rmse": f_act_rmse,
    #         "f_bias": f_bias,
    #         "f_bias_month":f_bias_month
    #     },
    #     "coefs":re_coefs,
    #     "test_res":test_res,
    #     "train_res":train_res
    # }

    fitted_act = fitted_act.unstack()
    y_train_act = y_train_act.unstack()
    y_test_act = y_test_act.unstack()

    # if label != "0":
    preds_act = preds_act.unstack()
    # y_test_act = y_test_act.unstack()

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


    # pred_df = pd.DataFrame({k:v["pred"] for k,v in lvout_rt_results.items()})
    # fitt_df = pd.DataFrame({k:v["fitted"] for k,v in lvout_rt_results.items()})

    coefs = pd.DataFrame(mdf.random_effects)
    train_data = pd.DataFrame(dict(actual=y_train_act.stack(), model=fitted_act.stack()))

    test_data = pd.DataFrame(dict(actual=y_test_act.stack(), model=preds_act.stack()))
    
    train_quant, train_bins = pd.qcut(train_data["actual"], 3, labels=False, retbins=True)
    test_quant, test_bins = pd.qcut(test_data["actual"], 3, labels=False, retbins=True)

    train_data["bin"] = train_quant
    test_data["bin"] = test_quant

    quant_scores = pd.DataFrame(index=[0,1,2], columns=["NSE", "RMSE"])
    
    for q in [0,1,2]:
        score = r2_score(
            train_data[train_data["bin"] == q]["actual"],
            train_data[train_data["bin"] == q]["model"],
        )
        rmse = np.sqrt(mean_squared_error(
            train_data[train_data["bin"] == q]["actual"],
            train_data[train_data["bin"] == q]["model"],
        ))
        quant_scores.loc[q] = [score, rmse]

    quant_table = quant_scores.to_markdown(tablefmt="github", floatfmt=".3f")
    print(quant_table) 


    # train_score_rel = r2_score(y_train_rel_act, fitted_rel)
    # train_score_sto = r2_score(y_train_sto_act, fitted_sto)

    # # fe_coefs = mdf.params
    # # re_coefs = mdf.random_effects
    # coefs = {g:np.mean(fit_results["params"][g], axis=0) for g in groups.unique()}
    # coefs = pd.DataFrame(coefs, index=exog_re.columns)

    # print(pd.DataFrame(coefs))
    
    # # test on unseen data
    # exog = sm.add_constant(X_test)
    # groups = exog["compositegroup"]


    # exog_re = exog.loc[:,exog_terms + interaction_terms + calendar.month_abbr[1:] + ["compositegroup"]]
    # mexog = exog[["const"]]
    # resers = exog_re.index.get_level_values(1).unique()
    # idx = pd.IndexSlice
    # exog_re.loc[idx[datetime(2010,1,1),resers], "Storage_pre_act"] = df.loc[idx[datetime(2010,1,1),resers], "Storage_pre"]
  
    # pred_result = fit_release_and_storage(y_test, y_test_sto, exog_re.drop(
    #     ["compositegroup", "Storage_pre_act"], axis=1), groups, means, std, init_values=coefs, niters=0)
    # predicted_rel = pred_result["f_rel_act"][0]
    # predicted_sto = pred_result["f_sto_act"][0]
    
    # # predicted = predict_mixedLM(fe_coefs, re_coefs, mexog, exog_re, "compositegroup")

  
    # # forecasted = forecast_mixedLM(fe_coefs, re_coefs, mexog, exog_re, means, std, "compositegroup", actual_inflow_test)
    # forecasted = forecast_mixedLM_new(coefs, exog_re, means, std, "compositegroup", actual_inflow_test)
    # # predicted_act = (predicted.unstack() * std["Release"] + means["Release"]).stack()
    # test_score_rel = r2_score(y_test_rel_act, predicted_rel)
    # test_score_sto = r2_score(y_test_sto_act, predicted_sto)

    
    # forecasted = forecasted[forecasted.index.get_level_values(0).year >= 2010]
    # forecast_score_rel = r2_score(y_test_rel_act.loc[forecasted["Release_act"].index], 
    #                           forecasted["Release_act"])
    # forecast_score_sto = r2_score(y_test_sto_act.loc[forecasted["Storage_act"].index], 
    #                           forecasted["Storage_act"])
    # # if forecast_score < 0:
    # #     II()
    # # print(mdf.summary())
    # # print(f"N Time Train: {N_time_train}")
    # # print(f"N Time Test : {N_time_test}")
    # print("Release Scores")
    # print(f"Train Score : {train_score_rel:.4f}")
    # print(f"Test Score  : {test_score_rel:.4f}")
    # print(f"Forecast Score  : {forecast_score_rel:.4f}")
    # print("Storage Scores")
    # print(f"Train Score : {train_score_sto:.4f}")
    # print(f"Test Score  : {test_score_sto:.4f}")
    # print(f"Forecast Score  : {forecast_score_sto:.4f}")
    # # print("\nGroup Sizes:")
    # # for (label, array) in zip(md.group_labels, md.exog_re_li):
    # #     length = array.shape[0]
    # #     nres = int(length/N_time_train)
    # #     print(f"\t{label}: Length = {length}, # Res = {nres}")
    # # print("\nGroup Coefficients:")
    # # print_coef_table(re_coefs)
    # # II()
    
    output = dict(
        coefs=coefs,
        # fe_coefs=fe_coefs,
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
            # fitted_rel=fitted_act,
            # fitted_sto=fitted_sto,
            # y_test_rel_act=y_test_rel_act,
            # y_train_rel_act=y_train_rel_act,
            # y_test_sto_act=y_test_sto_act,
            # y_train_sto_act=y_train_sto_act,
            # predicted_act_rel=predicted_rel,
            # predicted_act_sto=predicted_sto,
            quant_scores=quant_scores
            # forecasted=forecasted[["Release", "Storage", "Release_act", "Storage_act"]]
        )
    )

    output_dir = pathlib.Path(f"../results/agu_2021_runs/simple_model_temporal_validation")
    if not output_dir.exists():
        pathlib.mkdir(output_dir, parents=True)

    with open(output_dir / "results.pickle", "wb") as f:
        pickle.dump(output, f, protocol=4)

def mass_balance(sto_pre, rel, inf):
        return sto_pre + inf - rel
    
def get_release(parms, exog):
    return np.dot(exog, parms)

def loss_function(parms, exog, y_rel, y_sto, means, std):
    rel = get_release(parms, exog)
    rel_act = rel * std[0] + means[0]
    inf_act = exog["Net Inflow"] * std[2] + means[2]
    sto_p_act = exog["Storage_pre"] * std[1] + means[1]
    sto_act = mass_balance(sto_p_act, rel_act, inf_act)
    sto = (sto_act - means[1])/std[1]
    rel_error = rel - y_rel
    sto_error = sto - y_sto
    error = np.power(rel_error, 2).mean() + np.power(sto_error, 2).mean()
    return error

@time_function
def fit_release_and_storage(y_rel, y_sto, exog, groups, means, std, init_values=None, niters=10):
    group_names = groups.unique()
    data = {g:{} for g in group_names}
    for group in group_names:
        gindex = groups[groups == group].index
        data[group]["index"] = gindex
        data[group]["y_rel"] = y_rel.loc[gindex]
        data[group]["y_sto"] = y_sto.loc[gindex]
        data[group]["X"] = exog.loc[gindex,:]
        if not isinstance(init_values, pd.DataFrame):
            gparms = [np.random.choice([1,-1]) * np.random.rand() for p in exog.columns]
        else:
            gparms = init_values[group]
        data[group]["init_params"] = gparms
        data[group]["params"] = gparms
    
    params = {g:[] for g in group_names}
    fitted_rel_values = {g:[] for g in group_names}
    fitted_rel_act = {g:[] for g in group_names}
    fitted_sto_act = {g:[] for g in group_names}

    for group in group_names:
        gindex = data[group]["index"]
        rel_means = [means.loc[res, "Release"] for res in gindex.get_level_values(1)]
        rel_std = [std.loc[res, "Release"] for res in gindex.get_level_values(1)]
        sto_means = [means.loc[res, "Storage"] for res in gindex.get_level_values(1)]
        sto_std = [std.loc[res, "Storage"] for res in gindex.get_level_values(1)]
        inf_means = [means.loc[res, "Net Inflow"] for res in gindex.get_level_values(1)]
        inf_std = [std.loc[res, "Net Inflow"] for res in gindex.get_level_values(1)]
            
        gmeans = np.array([rel_means, sto_means, inf_means])
        gstd = np.array([rel_std, sto_std, inf_std])

        # parms = data[group]["params"]
        exog = data[group]["X"]
        y_rel = data[group]["y_rel"]
        y_sto = data[group]["y_sto"]

        if niters > 0:
            for i in range(niters):
                parms = [np.random.choice([1, -1]) * np.random.rand()
                            for p in exog.columns]      
                results = minimize(loss_function, parms, args=(exog,y_rel,y_sto,gmeans,gstd))
                new_parms = results.x
                params[group].append(new_parms)
                rel = get_release(new_parms, exog)
                fitted_rel_values[group].append(rel)
                rel_act = rel * gstd[0] + gmeans[0]
                inf_act = exog["Net Inflow"] * gstd[2] + gmeans[2]
                sto_p_act = exog["Storage_pre"] * gstd[1] + gmeans[1]
                fitted_rel_act[group].append(rel_act)
                fitted_sto_act[group].append(mass_balance(sto_p_act, rel_act, inf_act))
        else:
            parms = data[group]["init_params"]
            params[group].append(parms)
            rel = get_release(parms, exog)
            fitted_rel_values[group].append(rel)
            rel_act = rel * gstd[0] + gmeans[0]
            inf_act = exog["Net Inflow"] * gstd[2] + gmeans[2]
            sto_p_act = exog["Storage_pre"] * gstd[1] + gmeans[1]
            fitted_rel_act[group].append(rel_act)
            fitted_sto_act[group].append(
                mass_balance(sto_p_act, rel_act, inf_act))

    nloop = 1 if niters == 0 else niters

    for group in group_names:
        gindex = data[group]["index"]
        for i in range(nloop):
            fitted_rel_act[group][i] = pd.Series(fitted_rel_act[group][i], index=gindex)
            fitted_sto_act[group][i] = pd.Series(fitted_sto_act[group][i], index=gindex)
            fitted_rel_values[group][i] = pd.Series(fitted_rel_values[group][i], index=gindex)
    output = {
        "params": params,
        "f_rel_act":[pd.concat([fitted_rel_act[g][i] for g in group_names]).sort_index() for i in range(nloop)],
        "f_rel_val":[pd.concat([fitted_rel_values[g][i] for g in group_names]).sort_index() for i in range(nloop)],
        "f_sto_act":[pd.concat([fitted_sto_act[g][i] for g in group_names]).sort_index() for i in range(nloop)]
    }
    
    return output

def print_coef_table(coefs):
    rows = list(coefs.keys())
    columns = list(coefs[rows[0]].keys())
    longest = max(len(i) for i in rows)
    column_lengths = [longest + 2] + [len(i) + 2 for i in columns]
    columns.insert(0, " ")
    print(" ".join(f"{i:{j}}" for i,j in zip(columns, column_lengths)))
    for row in rows:
        to_print = [coefs[row][i] for i in columns[1:]]
        formatted = " ".join(
            f"{i:.3f}".center(j) for i, j in zip(to_print, column_lengths[1:]))
        print("".join([f"{row:{longest + 2}}", formatted]))    

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
    #* It is legitimately crazy how much slower this is than what is above.
    # for i, row in exog_re.iterrows():
    #     group = row["compositegroup"]
    #     coefs = re_coefs[group]
    #     keys = list(coefs.keys())
    #     value = (row[keys] * coefs).sum() + (exog_fe.loc[i] * fe_coefs[exog_fe.columns]).sum()
    #     output_df[i] = value
    return output_df

@time_function
def fit_release_storage_stepping(y_rel, y_sto, exog, groups, means, std, init_values=None, niters=10):
    # create output data frame
    output_df = pd.DataFrame(index=exog.index, 
                             columns=list(exog.columns) + ["Storage_act", "Release_act", "Storage_pre_act"],
                             dtype="float64")
    # get important indexers
    group_names = groups.unique()
    if not init_values:
        init_values = [
            [np.random.choice([1,-1])*np.random.rand() for i in range(group_names.size)]
            for j in range(exog.columns.size)
        ]

    coefs = pd.DataFrame(init_values, columns=group_names, index=exog.columns)
    keys = coefs[group_names[0]].keys()

    start_date = exog.index.get_level_values(0)[0] + timedelta(days=8)
    end_date = exog.index.get_level_values(0)[-1]
    pre_dates = pd.date_range(start=exog.index.get_level_values(0)[0], end=start_date)
    calc_columns = ["Release_pre", "Storage_pre", "Storage_pre_act", "Storage_Inflow_interaction",
                    "Release_roll7", "Storage_roll7",
                    "Storage_act", "Release_act"]
    const_keys = [i for i in keys if i not in calc_columns]

    idx = pd.IndexSlice
    exog["Storage_pre_act"] = (exog["Storage_pre"].unstack() * std["Storage"] + means["Storage"]).stack()
    exog.loc[idx[pre_dates,:],"Storage_act"] = (y_sto.loc[idx[pre_dates,:]].unstack() * std["Storage"] + means["Storage"]).stack()
    exog.loc[idx[pre_dates,:],"Release_act"] = (y_rel.loc[idx[pre_dates,:]].unstack() * std["Release"] + means["Release"]).stack()

    for col in output_df.columns:
        if col in calc_columns:
            output_df.loc[idx[pre_dates,:], col] = exog.loc[idx[pre_dates,:], col]
        else:
            output_df.loc[:,col] = exog.loc[:,col]


    dates = pd.date_range(start=start_date, end=end_date)
    date = dates[0]
    output_df["group"] = groups
    resers = exog.index.get_level_values(1).unique()
    nres = resers.size

    def get_release(exog, parms):
        return np.dot(exog, parms)

    def stepping_mass_balance(coefs, dates, exog_const, exog_updte, keys, gmeans, gstd):
        idx = pd.IndexSlice
        for date in dates:
            dexog = exog_const.loc[idx[date,:]].join(exog_updte.loc[idx[date,:]])[keys]
            rel = get_release(dexog, coefs)
            rel_act = rel * gstd["Release"] + gmeans["Release"]
            inf_act = exog_const.loc[idx[date,:],"Net Inflow"].values * gstd["Net Inflow"] + gmeans["Net Inflow"]
            sto_pre_act = exog_updte.loc[idx[date,:],"Storage_pre_act"]
            sto_act = sto_pre_act.values + inf_act - rel_act
            sto = (sto_act - gmeans["Storage"])/gstd["Storage"]

            # update values we have so far
            exog_updte.loc[idx[date,:],"Release"] = rel
            exog_updte.loc[idx[date,:],"Storage"] = sto.values
            exog_updte.loc[idx[date,:],"Release_act"] = rel_act.values
            exog_updte.loc[idx[date,:],"Storage_act"] = sto_act.values
            
            roll_keys = ["Storage_act", "Release_act"]
            roll_dates = pd.date_range(date - timedelta(days=6), date)
            roll_means = exog_updte.loc[idx[roll_dates,:], roll_keys].unstack().mean().unstack()

            tmrw = date + timedelta(days=1)
            if date != end_date:
                # get the new interaction term
                tmrw_inf_act = exog_const.loc[idx[tmrw,:],"Net Inflow"].values * gstd["Net Inflow"] + gmeans["Net Inflow"]
                sto_x_inf_act = sto_act * tmrw_inf_act
                sto_x_inf = (sto_x_inf_act - gmeans["Storage_Inflow_interaction"]) / gstd["Storage_Inflow_interaction"]
                # update values for tomorrows model
                exog_updte.loc[idx[tmrw,:],"Release_pre"] = rel
                exog_updte.loc[idx[tmrw,:],"Storage_pre"] = sto.values
                exog_updte.loc[idx[tmrw,:],"Storage_pre_act"] = sto_act.values
                exog_updte.loc[idx[tmrw,:],"Storage_Inflow_interaction"] = sto_x_inf.values
                exog_updte.loc[idx[tmrw,:],"Release_roll7"] = (
                    (roll_means.loc["Release_act"] - gmeans["Release"]) / gstd["Release"]).values
                exog_updte.loc[idx[tmrw,:],"Storage_roll7"] = (
                    (roll_means.loc["Storage_act"] - gmeans["Storage"]) / gstd["Storage"]).values
        return exog_updte

    def loss_function(params, rel, sto, mb_args):
        mb_out = stepping_mass_balance(params, **mb_args)
        rel_error = mb_out["Release"] - rel
        sto_error = mb_out["Storage"] - sto
        return np.power(rel_error, 2).sum() + np.power(sto_error, 2).sum() 

    g_results = {}
    # for group in group_names[:1]:
    # try to setup as much data as possible to speed up comp
    group = "ComboFlow-StorageDam"
    gindex = groups[groups == group].index
    gmeans = means.loc[gindex.get_level_values(1).unique(),["Release", "Storage", "Net Inflow", "Storage_Inflow_interaction"]]
    gstd = std.loc[gindex.get_level_values(1).unique(),["Release", "Storage", "Net Inflow", "Storage_Inflow_interaction"]]
    gcoefs = coefs[group]

    exog_const = exog.loc[gindex, const_keys]
    exog_updte = output_df.loc[gindex,calc_columns]

    g_y_rel = y_rel.loc[gindex]
    g_y_sto = y_sto.loc[gindex]
    methods = [
        "Nelder-Mead","BFGS","trust-ncg","trust-krylov","Newton-GC"
    ]
    method_results = {}
    for method in methods:
        results = minimize(loss_function, gcoefs.values, method=method, args=(
            g_y_rel, g_y_sto, {
                "dates":dates, "exog_const":exog_const, 
                "exog_updte":exog_updte, "keys":keys, 
                "gmeans":gmeans, "gstd":gstd
            })
        )
        # g_results[group] = results
        method_results[method] = results
    
    with open("./method_results.pickle", "wb") as f:
        pickle.dump(method_results, f)
        
    II()
    sys.exit()
    
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

    start_date = exog.index.get_level_values(0)[0] + timedelta(days=8)
    end_date = exog.index.get_level_values(0)[-1]
    pre_dates = pd.date_range(start=exog.index.get_level_values(0)[0], end=start_date)
    calc_columns = ["Release_pre", "Storage_pre", "Storage_pre_act",
                    "Release_roll7", "Storage_roll7",
                    "Storage_act", "Release_act"]
    if "Storage_roll7" in exog.columns:
        stop_it = 5
    else:
        stop_it = 4    
    idx = pd.IndexSlice
    for col in calc_columns[:stop_it]:
        output_df.loc[idx[pre_dates,:], col] = exog.loc[idx[pre_dates,:], col]
    # output_df.loc[idx[start_date, :],
    #               "Release_pre"] = exog.loc[idx[start_date, :], "Release_pre"]
    # output_df.loc[idx[start_date, :], "Storage_pre"] = exog.loc[idx[start_date, :], "Storage_pre"]
    # output_df.loc[:, "Storage_pre"] = exog.loc[:, "Storage_pre"]

    for col in output_df.columns:
        if col not in calc_columns:
            output_df[col] = exog.loc[:,col]

    dates = pd.date_range(start=start_date, end=end_date)
    date = dates[0]
    output_df[group_col] = exog[group_col]
    resers = exog.index.get_level_values(1).unique()
    nres = resers.size

    if tree:
        coeffs = pd.DataFrame(coefs).T
    else:
        coeffs = pd.DataFrame(index=resers, columns=re_keys, dtype="float64")
        for res in resers:
            coeffs.loc[res, re_keys] = coefs[exog.loc[(date, res), group_col]]

    sto_means = means["Storage"]
    sto_p_means = means["Storage_pre"]
    inf_means = means["Net Inflow"]
    rel_means = means["Release"]
    sto_std = std["Storage"]
    sto_p_std = std["Storage_pre"]
    inf_std = std["Net Inflow"]
    rel_std = std["Release"]
    
    if timelevel != "all":
        sto_means = sto_means.unstack()
        inf_means = inf_means.unstack()
        rel_means = rel_means.unstack()
        sto_std = sto_std.unstack()
        inf_std = inf_std.unstack()
        rel_std = rel_std.unstack()

    my_coefs = pd.DataFrame(index=resers, columns=re_keys, dtype="float64")
    # II()
    # sys.exit()
    for date in dates:
        dexog = output_df.loc[idx[date,:]][re_keys]
        if tree:
            for res in resers:
                param_group = dexog.loc[idx[date,res],group_col]
                my_coefs.loc[res,re_keys] = coeffs.loc[param_group,re_keys]
        else:
            my_coefs = coeffs

        rel = dexog.mul(my_coefs).sum(axis=1)

        if timelevel != "all":
            tl = getattr(date, timelevel)
            rel_act = rel*rel_std.loc[tl] + rel_means.loc[tl]
            # stor_pre_act = output_df.loc[idx[date,:]]["Storage_pre"]*sto_p_std.loc[tl] + sto_p_means.loc[tl]
            inflow_act = output_df.loc[idx[date,:]]["Net Inflow"]*inf_std.loc[tl] + inf_means.loc[tl]
            # inflow_act = actual_inflow.loc[idx[date,:]]
        else:
            rel_act = rel*rel_std+rel_means
            # stor_pre_act = output_df.loc[idx[date, :]
            #                              ]["Storage_pre"] * sto_p_std + sto_p_means
            # stor_pre_act = output_df.loc[idx[date, :]
            #                              ]["Storage_pre_act"]
            inflow_act = output_df.loc[idx[date, :]
                                       ]["Net Inflow"] * inf_std + inf_means
            # inflow_act = actual_inflow.loc[idx[date,:]]
        stor_pre_act = output_df.loc[idx[date, :]]["Storage_pre_act"]

        # inflow_act.index = inflow_act.index.get_level_values(1)
        storage_act = stor_pre_act + inflow_act - rel_act
            
        if timelevel != "all":
            tl = getattr(date, timelevel)
            storage = (storage_act - sto_means.loc[tl])/sto_std.loc[tl]
        else:
            storage = (storage_act-sto_means)/sto_std

        output_df.loc[idx[date,:],"Release"] = rel.values
        output_df.loc[idx[date,:],"Storage"] = storage.values
        output_df.loc[idx[date,:],"Release_act"] = rel_act.values
        output_df.loc[idx[date,:],"Storage_act"] = storage_act.values

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
        if date != end_date:
            # if date.timetuple().tm_yday in (365, 366):
            # if date.weekday() == 6:# or date.day == int(month_len//2):
            # output_df.loc[idx[tmrw,:],"Release_pre"] = exog.loc[idx[tmrw,:], "Release_pre"]
            output_df.loc[idx[tmrw,:], "Release_pre"] = rel.values
                # output_df.loc[idx[tmrw,:],"Storage_pre"] = exog.loc[idx[tmrw,:], "Storage_pre"]
            # else:
                # output_df.loc[idx[tmrw,:],"Release_pre"] = rel.values
            output_df.loc[idx[tmrw,:],"Storage_pre"] = storage.values
            output_df.loc[idx[tmrw,:],"Storage_pre_act"] = storage_act.values
            output_df.loc[idx[tmrw, :], [
                "Storage_roll7", "Release_roll7"]] = tmp.values       

    try:
        output_df = output_df.drop(calendar.month_abbr[1:], axis=1)
    except KeyError as e:
        pass

    return output_df


@time_function
def forecast_mixedLM(fe_coefs, re_coefs, exog_fe, exog_re, means, std, group_col, actual_inflow, timelevel="all", tree=False):
    # create output data frame
    output_df = pd.DataFrame(index=exog_re.index, 
                             columns=list(exog_re.columns) + ["Storage_act", "Release_act"],
                             dtype="float64")
    # get important indexers
    groups = exog_re[group_col].unique()
    re_keys = re_coefs[groups[0]].keys()
    fe_keys = exog_fe.columns
    start_date = exog_re.index.get_level_values(0)[0] + timedelta(days=8)
    end_date = exog_re.index.get_level_values(0)[-1]
    pre_dates = pd.date_range(start=exog_re.index.get_level_values(0)[0], end=start_date)
    calc_columns = ["Release_pre", "Storage_pre", "Storage_pre_act",
                    "Release_roll7", "Storage_roll7",
                    "Storage_act", "Release_act"]
    if "Storage_roll7" in exog_re.columns:
        stop_it = 5
    else:
        stop_it = 4    
    idx = pd.IndexSlice
    for col in calc_columns[:stop_it]:
        output_df.loc[idx[pre_dates,:], col] = exog_re.loc[idx[pre_dates,:], col]
    # output_df.loc[idx[start_date, :],
    #               "Release_pre"] = exog_re.loc[idx[start_date, :], "Release_pre"]
    # output_df.loc[idx[start_date, :], "Storage_pre"] = exog_re.loc[idx[start_date, :], "Storage_pre"]
    # output_df.loc[:, "Storage_pre"] = exog_re.loc[:, "Storage_pre"]

    for col in output_df.columns:
        if col not in calc_columns:
            output_df[col] = exog_re.loc[:,col]

    dates = pd.date_range(start=start_date, end=end_date)
    date = dates[0]
    output_df[group_col] = exog_re[group_col]
    resers = exog_re.index.get_level_values(1).unique()
    nres = resers.size

    if tree:
        coeffs = pd.DataFrame(re_coefs).T
    else:
        coeffs = pd.DataFrame(index=resers, columns=re_keys, dtype="float64")
        for res in resers:
            coeffs.loc[res, re_keys] = re_coefs[exog_re.loc[(
                date, res), group_col]]

    sto_means = means["Storage"]
    sto_p_means = means["Storage_pre"]
    inf_means = means["Net Inflow"]
    rel_means = means["Release"]
    sto_std = std["Storage"]
    sto_p_std = std["Storage_pre"]
    inf_std = std["Net Inflow"]
    rel_std = std["Release"]
    
    if timelevel != "all":
        sto_means = sto_means.unstack()
        inf_means = inf_means.unstack()
        rel_means = rel_means.unstack()
        sto_std = sto_std.unstack()
        inf_std = inf_std.unstack()
        rel_std = rel_std.unstack()

    my_coefs = pd.DataFrame(index=resers, columns=re_keys, dtype="float64")
    # II()
    # sys.exit()
    for date in dates:
        exog = output_df.loc[idx[date,:]][re_keys]
        fe_exog = exog_fe.loc[idx[date,:]][fe_keys]
        if tree:
            for res in resers:
                param_group = exog_re.loc[idx[date,res],group_col]
                my_coefs.loc[res,re_keys] = coeffs.loc[param_group,re_keys]
        else:
            my_coefs = coeffs

        rel = exog.mul(my_coefs).sum(axis=1) + fe_exog.dot(fe_coefs[fe_keys])

        if timelevel != "all":
            tl = getattr(date, timelevel)
            rel_act = rel*rel_std.loc[tl] + rel_means.loc[tl]
            # stor_pre_act = output_df.loc[idx[date,:]]["Storage_pre"]*sto_p_std.loc[tl] + sto_p_means.loc[tl]
            inflow_act = output_df.loc[idx[date,:]]["Net Inflow"]*inf_std.loc[tl] + inf_means.loc[tl]
            # inflow_act = actual_inflow.loc[idx[date,:]]
        else:
            rel_act = rel*rel_std+rel_means
            # stor_pre_act = output_df.loc[idx[date, :]
            #                              ]["Storage_pre"] * sto_p_std + sto_p_means
            # stor_pre_act = output_df.loc[idx[date, :]
            #                              ]["Storage_pre_act"]
            inflow_act = output_df.loc[idx[date, :]
                                       ]["Net Inflow"] * inf_std + inf_means
            # inflow_act = actual_inflow.loc[idx[date,:]]
        stor_pre_act = output_df.loc[idx[date, :]]["Storage_pre_act"]

        # inflow_act.index = inflow_act.index.get_level_values(1)
        storage_act = stor_pre_act + inflow_act - rel_act
            
        if timelevel != "all":
            tl = getattr(date, timelevel)
            storage = (storage_act - sto_means.loc[tl])/sto_std.loc[tl]
        else:
            storage = (storage_act-sto_means)/sto_std
        # if date != end_date:
        #     storage = output_df.loc[idx[date+timedelta(days=1),:],"Storage_pre"]

        output_df.loc[idx[date,:],"Release"] = rel.values
        output_df.loc[idx[date,:],"Storage"] = storage.values
        output_df.loc[idx[date,:],"Release_act"] = rel_act.values
        output_df.loc[idx[date,:],"Storage_act"] = storage_act.values

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
        if date != end_date:
            # if date.timetuple().tm_yday in (365, 366):
            # if date.weekday() == 6:# or date.day == int(month_len//2):
            output_df.loc[idx[tmrw,:],"Release_pre"] = exog_re.loc[idx[tmrw,:], "Release_pre"]
                # output_df.loc[idx[tmrw,:],"Storage_pre"] = exog_re.loc[idx[tmrw,:], "Storage_pre"]
            # else:
                # output_df.loc[idx[tmrw,:],"Release_pre"] = rel.values
            output_df.loc[idx[tmrw,:],"Storage_pre"] = storage.values
            output_df.loc[idx[tmrw,:],"Storage_pre_act"] = storage_act.values
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

@time_function
def forecast_mixedLM_other_res(groups, unseen=True):
    df = read_all_res_data()
    date_range = find_max_date_range(file="date_res.csv")
    df = df[df.index.get_level_values(0).isin(date_range)]
    filename = "-".join(groups)
    
    for_groups = df.loc[:, groups]
    
    scaled_df, means, std = scale_multi_level_df(df)

    df.loc[:,"Storage_act"] = (df.loc[:,"Storage"].unstack() * std["Storage"] + means["Storage"]).stack()
    df.loc[:,"Release_act"] = (df.loc[:,"Release"].unstack() * std["Release"] + means["Release"]).stack()

    X_scaled = scaled_df.loc[:, ["Storage_pre", "Net Inflow", "Release_pre"]]
    y_scaled = scaled_df.loc[:, "Release"]
    
    X_scaled[groups] = for_groups

    X_scaled = change_group_names(X_scaled, groups, group_names)
    X_scaled = combine_columns(X_scaled, groups, "compositegroup")

    if unseen:
        X_scaled = X_scaled.loc[X_scaled.index.get_level_values(0).year >= 2010]

    exog = sm.add_constant(X_scaled)
    groups = exog["compositegroup"]

    exog_re = exog[["Storage_pre", "Net Inflow",
                    "Release_pre", "compositegroup"]]                    
    mexog = exog[["const"]]

    with open(f"../results/multi-level-results/{filename}.pickle", "rb") as f:
        model_results = pickle.load(f)
    
    re_coefs = model_results["re_coefs"]
    fe_coefs = model_results["fe_coefs"]
    
    output_df = forecast_mixedLM(fe_coefs, re_coefs, mexog, exog_re, means, std)
    output_df["Release_act_obs"] = df["Release_act"]
    output_df["Storage_act_obs"] = df["Storage_act"]
    if unseen:
        filename += "-unseen"
    output_df.to_pickle(f"../results/multi-level-results/{filename}-all_res.pickle", protocol=4)
    

if __name__ == "__main__":
    df = read_tva_data()
    # forecast_mixedLM_other_res(groups=["NaturalOnly", "RunOfRiver"])
    scaled_MixedEffects(df, groups = ["NaturalOnly","RunOfRiver"],
                            # filter_groups={"NaturalOnly": "NaturalFlow"})
                        filter_groups={"NaturalOnly":"ComboFlow"})
    # Recorded Forecast Scores:
    # NaturalOnly, PrimaryType :             0.9643
    # NaturalOnly, RunOfRiver :              0.9646
    # NaturalOnly, RunOfRiver, PrimaryType : 0.9640
    # NaturalOnly :                          0.9656
