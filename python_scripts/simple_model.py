import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLMParams
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import pickle
from helper_functions import (read_tva_data, scale_multi_level_df, 
                              read_all_res_data, find_max_date_range)
from timing_function import time_function
from time import perf_counter as timer
from datetime import timedelta, datetime
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
    
    
    df["Storage_Inflow_interaction"] = df["Storage_pre"].mul(
        df["Net Inflow"])
    df["Storage_Release_interaction"] = df["Storage_pre"].mul(
        df["Release_pre"])
    df["Release_Inflow_interaction"] = df["Release_pre"].mul(
        df["Net Inflow"])

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
    
    X_scaled[groups] = for_groups
    X_scaled[fraction_names] = fractions

    X_scaled = change_group_names(X_scaled, groups, group_names)
    
    # these reservoirs exhibit different characteristics than their group may suggest.
    change_names = ["Douglas", "Cherokee", "Hiwassee"]
    for ch_name in change_names:
        size = X_scaled[X_scaled.index.get_level_values(1) == ch_name].shape[0]
        X_scaled.loc[X_scaled.index.get_level_values(1) == ch_name, "NaturalOnly"] = ["NaturalFlow" for i in range(size)]

    X_scaled = combine_columns(X_scaled, groups, "compositegroup")

    

    #* This lets me group by month 
    # X_scaled["compositegroup"] = [calendar.month_abbr[i.month]
    #                      for i in X_scaled.index.get_level_values(0)]

    #* this introduces a intercept that varies monthly and between groups
    month_arrays = {i:[] for i in calendar.month_abbr[1:]}
    for date in X_scaled.index.get_level_values(0):
        for key in month_arrays.keys():
            if calendar.month_abbr[date.month] == key:
                month_arrays[key].append(1)
            else:
                month_arrays[key].append(0)

    for key, array in month_arrays.items():
        X_scaled[key] = array
    
    split_date = datetime(2010,1,1)
    X_test = X_scaled.loc[X_scaled.index.get_level_values(0) >= split_date - timedelta(days=8)]
    X_train = X_scaled.loc[X_scaled.index.get_level_values(0) < split_date]

    y_test = y_scaled.loc[y_scaled.index.get_level_values(0) >= split_date - timedelta(days=8)]
    y_train = y_scaled.loc[y_scaled.index.get_level_values(0) < split_date]
    
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
    
    # exog_terms = [
    #     "Storage_pre", "Net Inflow", "Release_pre", 
    #     ] + extra_lag_terms

    exog_terms = [
        "const", "Net Inflow", "Storage_pre", "Release_pre",
        "Release_roll7", "Storage_roll7",  "Inflow_roll7"
    ]

    exog_re = exog[exog_terms + interaction_terms + calendar.month_abbr[1:]]

    mexog = exog[["const"]]

    
    free = MixedLMParams.from_components(fe_params=np.ones(mexog.shape[1]),
                                         cov_re=np.eye(exog_re.shape[1]))
    md = sm.MixedLM(y_train, mexog, groups=groups, exog_re=exog_re)
    fit_time_1 = timer()
    mdf = md.fit(free=free)
    fit_time_2 = timer()

    trans_time_1 = timer()
    fitted = (mdf.fittedvalues.unstack() * std["Release"] + means["Release"]).stack()
    y_train_act = (y_train.unstack() * std["Release"] + means["Release"]).stack()
    y_test_act = (y_test.unstack() * std["Release"] + means["Release"]).stack()
    trans_time_2 = timer()
    
    train_score = r2_score(y_train_act, fitted)

    fe_coefs = mdf.params
    re_coefs = mdf.random_effects

    print(pd.DataFrame(re_coefs))
    
    # test on unseen data
    exog = sm.add_constant(X_test)
    groups = exog["compositegroup"]

    exog_re = exog[exog_terms + interaction_terms + calendar.month_abbr[1:] + ["compositegroup"]]
    mexog = exog[["const"]]

    predicted = predict_mixedLM(fe_coefs, re_coefs, mexog, exog_re, "compositegroup")
    forecasted = forecast_mixedLM(fe_coefs, re_coefs, mexog, exog_re, means, std, "compositegroup")
    predicted_act = (predicted.unstack() * std["Release"] + means["Release"]).stack()
    test_score = r2_score(y_test_act, predicted_act)
    forecast_score = r2_score(y_test_act.loc[forecasted["Release_act"].index], 
                              forecasted["Release_act"])
    # print(mdf.summary())
    # print(f"N Time Train: {N_time_train}")
    # print(f"N Time Test : {N_time_test}")
    print(f"Train Score : {train_score:.4f}")
    print(f"Test Score  : {test_score:.4f}")
    print(f"Forecast Score  : {forecast_score:.4f}")

    print("\nGroup Sizes:")
    for (label, array) in zip(md.group_labels, md.exog_re_li):
        length = array.shape[0]
        nres = int(length/N_time_train)
        print(f"\t{label}: Length = {length}, # Res = {nres}")
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

    with open(f"../results/multi-level-results/for_graps/{filename}_SIx_pre_std_swapped_res_roll7.pickle", "wb") as f:
        pickle.dump(output, f, protocol=4)
    

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
def forecast_mixedLM(fe_coefs, re_coefs, exog_fe, exog_re, means, std, group_col, timelevel="all", tree=False):
    # create output data frame
    output_df = pd.DataFrame(index=exog_re.index, 
                             columns=list(exog_re.columns) + ["Storage_act", "Release_act"],
                             dtype="float64")
    # get important indexers
    groups = exog_re[group_col].unique()
    re_keys = re_coefs[groups[0]].keys()
    fe_keys = exog_fe.columns
    start_date = exog_re.index.get_level_values(0)[0] + timedelta(days=7)
    end_date = exog_re.index.get_level_values(0)[-1]
    pre_dates = pd.date_range(start=exog_re.index.get_level_values(0)[0], end=start_date)
    calc_columns = ["Release_pre", "Storage_pre",
                    "Release_roll7", "Storage_roll7",
                    "Storage_act", "Release_act"]
                    
    idx = pd.IndexSlice
    for col in calc_columns[:3]:
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
            stor_pre_act = output_df.loc[idx[date,:]]["Storage_pre"]*sto_p_std.loc[tl] + sto_p_means.loc[tl]
            inflow_act = output_df.loc[idx[date,:]]["Net Inflow"]*inf_std.loc[tl] + inf_means.loc[tl]
        else:
            rel_act = rel*rel_std+rel_means
            stor_pre_act = output_df.loc[idx[date, :]
                                         ]["Storage_pre"] * sto_p_std + sto_p_means
            inflow_act = output_df.loc[idx[date, :]
                                       ]["Net Inflow"] * inf_std + inf_means
        
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
            output_df.loc[idx[tmrw, :], [
                "Storage_roll7", "Release_roll7"]] = tmp.values

    try:
        output_df = output_df.drop(calendar.month_abbr[1:], axis=1)
    except KeyError as e:
        pass

    return output_df.dropna()

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
