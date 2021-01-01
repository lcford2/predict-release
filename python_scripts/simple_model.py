import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLMParams
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import pickle
from helper_functions import (read_tva_data, scale_multi_level_df)
from timing_function import time_function
from time import perf_counter as timer
from datetime import timedelta
from IPython import embed as II
import sys

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

def scaled_MixedEffects(df, groups, scaler="mine"):
    filename = "-".join(groups)
    for_groups = df.loc[:,groups]
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
    
    X_scaled[groups] = for_groups
    names = {
        "RunOfRiver": {0: "StorageDam", 1: "RunOfRiver"},
        "NaturalOnly": {0: "ComboFlow", 1: "NaturalFlow"},
        "PrimaryType": {0: "FloodControl",
                        1: "Hydropower",
                        2: "Irrigation",
                        3: "Navigation",
                        4: "WaterSupply"}
    }
    X_scaled = change_group_names(X_scaled, groups, names)
    X_scaled = combine_columns(X_scaled, groups, "compositegroup")

    # X_scaled["NaturalOnly"] = ["Natural" if i == 1 else "Both" for i in X_scaled["NaturalOnly"]]
    
    X_test = X_scaled[X_scaled.index.get_level_values(0).year >= 2010]
    X_train = X_scaled[X_scaled.index.get_level_values(0).year < 2010]

    y_test = y_scaled[y_scaled.index.get_level_values(0).year >= 2010]
    y_train = y_scaled[y_scaled.index.get_level_values(0).year < 2010]
    
    N_time_train = X_train.index.get_level_values(0).unique().shape[0]
    N_time_test = X_test.index.get_level_values(0).unique().shape[0]

    # train model
    exog = sm.add_constant(X_train)
    groups = exog["compositegroup"]

    exog_re = exog[["Storage_pre", "Net Inflow", "Release_pre"]]
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
    
    # test on unseen data
    exog = sm.add_constant(X_test)
    groups = exog["compositegroup"]

    exog_re = exog[["Storage_pre", "Net Inflow", "Release_pre", "compositegroup"]]
    mexog = exog[["const"]]

    predicted = predict_mixedLM(fe_coefs, re_coefs, mexog, exog_re)
    forecasted = forecast_mixedLM(fe_coefs, re_coefs, mexog, exog_re, means, std)
    predicted_act = (predicted.unstack() * std["Release"] + means["Release"]).stack()
    test_score = r2_score(y_test_act, predicted_act)
    print(mdf.summary())
    # print(f"N Time Train: {N_time_train}")
    # print(f"N Time Test : {N_time_test}")
    print(f"Train Score : {train_score:.4f}")
    print(f"Test Score  : {test_score:.4f}")

    print("\nGroup Sizes:")
    for (label, array) in zip(md.group_labels, md.exog_re_li):
        length = array.shape[0]
        nres = int(length/N_time_train)
        print(f"\t{label}: Length = {length}, # Res = {nres}")
    print("\nGroup Coefficients:")
    print_coef_table(re_coefs)
    
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

    with open(f"../results/multi-level-results/{filename}.pickle", "wb") as f:
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
def predict_mixedLM(fe_coefs, re_coefs, exog_fe, exog_re):
    output_df = pd.Series(index=exog_re.index, dtype="float64")
    groups = exog_re["compositegroup"].unique()
    re_keys = re_coefs[groups[0]].keys()
    fe_keys = exog_fe.columns
    for group in groups:
        re_df = exog_re.loc[exog_re["compositegroup"] == group, re_keys]
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
def forecast_mixedLM(fe_coefs, re_coefs, exog_fe, exog_re, means, std):
    # create output data frame
    output_df = pd.DataFrame(index=exog_re.index, 
                             columns=[
                                 "Release", "Storage", 
                                 "Release_act", "Storage_act", 
                                 "Release_pre", "Storage_pre", "Net Inflow",
                                 "compositegroup"
                                 ],
                             dtype="float64")
    # get important indexers
    groups = exog_re["compositegroup"].unique()
    re_keys = re_coefs[groups[0]].keys()
    fe_keys = exog_fe.columns
    start_date = exog_re.index.get_level_values(0)[0]
    end_date = exog_re.index.get_level_values(0)[-1]
    idx = pd.IndexSlice
    output_df.loc[idx[start_date, :],
                  "Release_pre"] = exog_re.loc[idx[start_date, :], "Release_pre"]
    output_df.loc[idx[start_date, :], "Storage_pre"] = exog_re.loc[idx[start_date, :], "Storage_pre"]
    output_df["Net Inflow"] = exog_re["Net Inflow"]
    output_df["compositegroup"] = exog_re["compositegroup"]
    resers = exog_re.index.get_level_values(1).unique()
    coeffs = pd.DataFrame(index=resers, columns=re_keys, dtype="float64")
    nres = resers.size
    
    dates = pd.date_range(start=start_date, end=end_date)
    date = dates[0]
    for res in resers:
        coeffs.loc[res, re_keys] = re_coefs[exog_re.loc[(date,res),"compositegroup"]]

    for date in dates:
        exog = output_df.loc[idx[date,:]][re_keys]
        fe_exog = exog_fe.loc[idx[date,:]][fe_keys]
        
        rel = exog.mul(coeffs).sum(axis=1) + fe_exog.dot(fe_coefs[fe_keys])

        rel_act = rel*std["Release"]+means["Release"]
        stor_pre_act = output_df.loc[idx[date,:]]["Storage_pre"]*std["Storage"]+means["Storage"]
        inflow_act = output_df.loc[idx[date,:]]["Net Inflow"]*std["Net Inflow"]+means["Net Inflow"]
        
        storage_act = stor_pre_act + inflow_act - rel_act
        storage = (storage_act-means["Storage"])/std["Storage"]
        
        output_df.loc[idx[date,:],"Release"] = rel.values
        output_df.loc[idx[date,:],"Storage"] = storage.values
        output_df.loc[idx[date,:],"Release_act"] = rel_act.values
        output_df.loc[idx[date,:],"Storage_act"] = storage_act.values
        if date != end_date:
            output_df.loc[idx[date+timedelta(days=1),:],"Release_pre"] = rel.values
            output_df.loc[idx[date+timedelta(days=1),:],"Storage_pre"] = storage.values

    return output_df
    

def change_group_names(df, groups, names):
    for group in groups:
        df[group] = [names[group][i] for i in df[group]]
    return df

def combine_columns(df, columns, new_name, sep="-"):
    df[new_name] = df[columns[0]].str.cat(df[columns[1:]], sep=sep)
    return df
    

if __name__ == "__main__":
    df = read_tva_data()
    scaled_MixedEffects(df, groups = ["NaturalOnly", "PrimaryType"])
