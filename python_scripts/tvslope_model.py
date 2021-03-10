import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLMParams
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import pickle
from helper_functions import (read_tva_data, scale_multi_level_df, 
                              read_all_res_data, find_max_date_range)
from simple_model import (change_group_names, combine_columns,
                          predict_mixedLM, forecast_mixedLM, print_coef_table)
from timing_function import time_function
from time import perf_counter as timer
from datetime import timedelta
from IPython import embed as II
import calendar
import sys

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

def scaled_tv_slope(df, groups, filter_groups=None, scaler="mine"):
    filename = "-".join(groups)
    for_groups = df.loc[:,groups]
    fraction_names = ["Fraction_Storage",
                      "Fraction_Net Inflow"]
    fractions = df.loc[:, fraction_names]

    df["Storage_Inflow_interaction"] = df.loc[:,"Storage_pre"].mul(
        df.loc[:,"Net Inflow"])
    df["Storage_Release_interaction"] = df.loc[:,"Storage_pre"].mul(
        df.loc[:,"Release_pre"])
    df["Release_Inflow_interaction"] = df.loc[:,"Release_pre"].mul(
        df.loc[:,"Net Inflow"])

    change_names = ["Douglas", "Cherokee", "Hiwassee"]
    for ch_name in change_names:
        size = df[df.index.get_level_values(1) == ch_name].shape[0]
        df.loc[df.index.get_level_values(1) == ch_name, "NaturalOnly"] = [1 for i in range(size)]

    if filter_groups:
        filename += "_filter"
        for key, value in filter_groups.items():
            df = df.loc[df[key] == inverse_groupnames[key][value],:]
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
        X = df.loc[:,["Storage_pre", "Net Inflow", "Release_pre"]]
        y = df.loc[:,"Release"]

        x_scaler = scaler.fit(X)
        X_scaled = x_scaler.transform(X)

        y_scaler = scaler.fit(y.values.reshape(-1,1))
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
        "const", "Storage_pre", "Net Inflow", "Release_pre", 
        ]

    exog_re = exog[exog_terms + interaction_terms]# + calendar.month_abbr[1:]]

    mexog = exog[["const"]]

    result_params = {}

    for res in X_train.index.get_level_values(1).unique():
        mexog_res = mexog[mexog.index.get_level_values(1) == res]
        exog_re_res = exog_re[exog_re.index.get_level_values(1) == res]    
        free = MixedLMParams.from_components(fe_params=np.ones(mexog_res.shape[1]),
                                         cov_re=np.eye(exog_re_res.shape[1]))
        y_train_res = y_train[y_train.index.get_level_values(1) == res]
        groups_res = groups[groups.index.get_level_values(1) == res]
        md = sm.MixedLM(y_train_res, mexog_res, groups=groups_res, exog_re=exog_re_res)
        fit_time_1 = timer()
        mdf = md.fit(free=free)
        fit_time_2 = timer()
        fe_coefs = mdf.params
        re_coefs = mdf.random_effects
        result_params[res] = {"fe":fe_coefs,"re":re_coefs}
    
    II()

    trans_time_1 = timer()
    fitted = (mdf.fittedvalues.unstack() * std["Release"] + means["Release"]).stack()
    y_train_act = (y_train.unstack() * std["Release"] + means["Release"]).stack()
    y_test_act = (y_test.unstack() * std["Release"] + means["Release"]).stack()
    trans_time_2 = timer()
    
    train_score = r2_score(y_train_act, fitted)

    fe_coefs = mdf.params
    re_coefs = mdf.random_effects

    # test on unseen data
    exog = sm.add_constant(X_test)
    groups = exog["compositegroup"]

    exog_re = exog[exog_terms + interaction_terms + ["compositegroup"]]
    mexog = exog[["const"]]

    predicted = predict_mixedLM(fe_coefs, re_coefs, mexog, exog_re, "compositegroup")
    forecasted = forecast_mixedLM(fe_coefs, re_coefs, mexog, exog_re, means, std, "compositegroup")
    predicted_act = (predicted.unstack() * std["Release"] + means["Release"]).stack()
    test_score = r2_score(y_test_act, predicted_act)
    forecast_score = r2_score(y_test_act, forecasted["Release_act"])
    # print(mdf.summary())
    # print(f"N Time Train: {N_time_train}")
    # print(f"N Time Test : {N_time_test}")
    print(f"Train Score : {train_score:.4f}")
    print(f"Test Score  : {test_score:.4f}")
    print(f"Forecast Score  : {forecast_score:.4f}")

    # print("\nGroup Sizes:")
    # for (label, array) in zip(md.group_labels, md.exog_re_li):
    #     length = array.shape[0]
    #     nres = int(length/N_time_train)
    #     print(f"\t{label}: Length = {length}, # Res = {nres}")
    print("\nGroup Coefficients:")
    print_coef_table(re_coefs)
    # II()
    
    output = dict(
        re_coefs=re_coefs,
        fe_coefs=fe_coefs,
        data=dict(
            X_test=X_test,
            y_test=y_test,
            X_train=X_train,
            y_train=y_train,
            fitted=fitted,
            y_test_act=y_test_act,
            y_train_act=y_train_act,
            predicted_act=predicted_act,
            forecasted=forecasted[["Release", "Storage", "Release_act", "Storage_act"]]
        )
    )

    with open(f"../results/tvs_results/upstream/{filename}_SIx_pre_std_swapped_res_daily_help.pickle", "wb") as f:
        pickle.dump(output, f, protocol=4)

if __name__ == "__main__":
    df = read_tva_data()
    scaled_tv_slope(df, groups = ["NaturalOnly","RunOfRiver"],
                   filter_groups={"NaturalOnly":"NaturalFlow"})
