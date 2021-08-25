import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLMParams
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import pickle
from utils.helper_functions import (read_tva_data, scale_multi_level_df, 
                              read_all_res_data, find_max_date_range)
from simple_model import (change_group_names, combine_columns,
                          predict_mixedLM, forecast_mixedLM, print_coef_table)
from utils.timing_function import time_function
from time import perf_counter as timer
from datetime import timedelta
from IPython import embed as II
import calendar
import sys
import warnings
warnings.simplefilter("ignore")

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


def model(df, groups, filter_groups=None, scaler="mine"):
    filename = "-".join(groups)
    for_groups = df.loc[:, groups]
    fraction_names = ["Fraction_Storage",
                      "Fraction_Net Inflow"]
    fractions = df.loc[:, fraction_names]

    df["Storage_Inflow_interaction"] = df.loc[:, "Storage_pre"].mul(
        df.loc[:, "Net Inflow"])
    df["Storage_Release_interaction"] = df.loc[:, "Storage_pre"].mul(
        df.loc[:, "Release_pre"])
    df["Release_Inflow_interaction"] = df.loc[:, "Release_pre"].mul(
        df.loc[:, "Net Inflow"])

    change_names = ["Douglas", "Cherokee", "Hiwassee"]
    for ch_name in change_names:
        size = df[df.index.get_level_values(1) == ch_name].shape[0]
        df.loc[df.index.get_level_values(1) == ch_name, "NaturalOnly"] = [
            1 for i in range(size)]

    if filter_groups:
        filename += "_filter"
        for key, value in filter_groups.items():
            df = df.loc[df[key] == inverse_groupnames[key][value], :]
            filename += f"_{value}"

    if scaler == "mine":
        scaled_df, means, std = scale_multi_level_df(df)
        X_scaled = scaled_df.loc[:, ["Storage_pre", "Net Inflow", "Release_pre",
                                     "Storage_Inflow_interaction",
                                     "Storage_Release_interaction",
                                     "Release_Inflow_interaction"]
                                 ]
        y_scaled = scaled_df.loc[:, "Release"]
    else:
        scaler = StandardScaler()
        X = df.loc[:, ["Storage_pre", "Net Inflow", "Release_pre"]]
        y = df.loc[:, "Release"]

        x_scaler = scaler.fit(X)
        X_scaled = x_scaler.transform(X)

        y_scaler = scaler.fit(y.values.reshape(-1, 1))
        y_scaled = y_scaler.transform(y.values.reshape(-1, 1))

        X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
        y_scaled = pd.Series(y_scaled.reshape(-1), index=y.index)

    X_scaled[groups] = for_groups
    X_scaled[fraction_names] = fractions

    X_scaled = change_group_names(X_scaled, groups, group_names)

    X_scaled = combine_columns(X_scaled, groups, "compositegroup")

    #* This lets me group by month
    X_scaled["compositegroup"] = [calendar.month_abbr[i.month]
                                  for i in X_scaled.index.get_level_values(0)]

    #* this introduces a intercept that varies monthly and between groups
    # month_arrays = {i:[] for i in calendar.month_abbr[1:]}
    # for date in X_scaled.index.get_level_values(0):
    #     for key in month_arrays.keys():
    #         if calendar.month_abbr[date.month] == key:
    #             month_arrays[key].append(1)
    #         else:
    #             month_arrays[key].append(0)

    # for key, array in month_arrays.items():
    #     X_scaled[key] = array

    X_test = X_scaled.loc[X_scaled.index.get_level_values(0).year >= 2010]
    X_train = X_scaled.loc[X_scaled.index.get_level_values(0).year < 2010]

    y_test = y_scaled.loc[y_scaled.index.get_level_values(0).year >= 2010]
    y_train = y_scaled.loc[y_scaled.index.get_level_values(0).year < 2010]

    N_time_train = X_train.index.get_level_values(0).unique().shape[0]
    N_time_test = X_test.index.get_level_values(0).unique().shape[0]

    # train model
    # Instead of adding a constant, I want to create dummy variable that accounts for season differences
    exog = sm.add_constant(X_train)

    groups = exog["compositegroup"]
    # Storage Release Interactions are near Useless
    # Release Inflow Interaction does not provide a lot either
    # Storage Inflow interaction seems to matter for ComboFlow-StorageDam reservoirs.
    interaction_terms = ["Storage_Inflow_interaction"]

    exog_terms = [
         "Storage_pre", "Net Inflow"
    ]

    # + calendar.month_abbr[1:]]
    exog_re = exog[exog_terms + interaction_terms]

    mexog = exog[["const"]]
    
    orders = [
        (1,0,0),
        (0,0,1),
        (1,0,1),
        (1,1,1),
        (2,0,0)
    ]
    sorders = [
        (1,0,0,7),
        (0,0,1,7),
        (1,1,0,7),
        (1,1,1,7),
    ]

    results = {}

    for res in X_train.index.get_level_values(1).unique():
        res_train_x = X_train[X_train.index.get_level_values(1) == res]
        res_train_x.index = res_train_x.index.get_level_values(0)
        res_train_y = y_train[y_train.index.get_level_values(1) == res].unstack()

        res_test_x = X_test[X_test.index.get_level_values(1) == res]
        res_test_x.index = res_test_x.index.get_level_values(0)
        res_test_y = y_test[y_test.index.get_level_values(1) == res].unstack()

        exog = res_train_x[exog_terms + interaction_terms]
        exog_test = res_test_x[exog_terms + interaction_terms]
        

        best = ((2,1,2), (2,1,1,7))
        order, sorder = best

        # for order in orders:
        #     for sorder in sorders:
        print(f"Solving SARIMAX ({order}) x ({sorder}) for {res}")
        md = SARIMAX(res_train_y, exog=exog, order=order,
                    seasonal_order=sorder, freq=res_train_y.index.inferred_freq)
        fit = md.fit(maxiter=100, disp=False, low_memory=True)
        # results[(order,sorder)] = fit
        # print(f"SARIMAX ({order}) x ({sorder}) BIC={fit.bic}")
        
        fitted = fit.fittedvalues * std.loc[res, "Release"] + means.loc[res, "Release"]
        y_train_act = res_train_y * std.loc[res, "Release"] + means.loc[res, "Release"]
        y_test_act = res_test_y * std.loc[res, "Release"] + means.loc[res, "Release"]
    
        train_score = r2_score(y_train_act, fitted)
        
        forecasted = fit.forecast(y_test_act.index[-1], exog=exog_test)
        forecasted = forecasted * std.loc[res, "Release"] + means.loc[res, "Release"]
        forecasted_score = r2_score(y_test_act, forecasted)
        print(f"Scores for above model: train={train_score:.3f}, test={forecasted_score:.3f}")

        params = fit.params
        results[res] = {
            "fc_score":forecasted_score,
            "train_score":train_score,
            "fitted":fitted,
            "forecasted":forecasted, 
            "train_y":y_train_act,
            "test_y":y_test_act,
            "params":params
        }
    try:
        with open("../results/ts_results/sarimax_single_res_results.pickle", "wb") as f:
            pickle.dump(results, f, protocol=4)
    except Error as e:
        II()

    # # test on unseen data
    # exog = sm.add_constant(X_test)
    # groups = exog["compositegroup"]

    # exog_re = exog[exog_terms + interaction_terms + ["compositegroup"]]
    # mexog = exog[["const"]]

    # predicted = predict_mixedLM(
    #     fe_coefs, re_coefs, mexog, exog_re, "compositegroup")
    # forecasted = forecast_mixedLM(
    #     fe_coefs, re_coefs, mexog, exog_re, means, std, "compositegroup")
    # predicted_act = (predicted.unstack() *
    #                  std["Release"] + means["Release"]).stack()
    # test_score = r2_score(y_test_act, predicted_act)
    # forecast_score = r2_score(y_test_act, forecasted["Release_act"])
    # # print(mdf.summary())
    # # print(f"N Time Train: {N_time_train}")
    # # print(f"N Time Test : {N_time_test}")
    # print(f"Train Score : {train_score:.4f}")
    # print(f"Test Score  : {test_score:.4f}")
    # print(f"Forecast Score  : {forecast_score:.4f}")

    # # print("\nGroup Sizes:")
    # # for (label, array) in zip(md.group_labels, md.exog_re_li):
    # #     length = array.shape[0]
    # #     nres = int(length/N_time_train)
    # #     print(f"\t{label}: Length = {length}, # Res = {nres}")
    # print("\nGroup Coefficients:")
    # print_coef_table(re_coefs)
    # # II()

    # output = dict(
    #     re_coefs=re_coefs,
    #     fe_coefs=fe_coefs,
    #     data=dict(
    #         X_test=X_test,
    #         y_test=y_test,
    #         X_train=X_train,
    #         y_train=y_train,
    #         fitted=fitted,
    #         y_test_act=y_test_act,
    #         y_train_act=y_train_act,
    #         predicted_act=predicted_act,
    #         forecasted=forecasted[["Release", "Storage",
    #                                "Release_act", "Storage_act"]]
    #     )
    # )

    # with open(f"../results/tvs_results/upstream/{filename}_SIx_pre_std_swapped_res_daily_help.pickle", "wb") as f:
    #     pickle.dump(output, f, protocol=4)


if __name__ == "__main__":
    df = read_tva_data()
    model(df, groups=["NaturalOnly", "RunOfRiver"],
                    filter_groups={"NaturalOnly": "NaturalFlow"})
