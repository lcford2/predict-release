import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import pickle
from helper_functions import (read_tva_data, scale_multi_level_df)
from timing_function import time_function
from IPython import embed as II

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
    
    X_test = X_scaled[X_scaled.index.get_level_values(0).year >= 2010]
    X_train = X_scaled[X_scaled.index.get_level_values(0).year < 2010]

    y_test = y_scaled[y_scaled.index.get_level_values(0).year >= 2010]
    y_train = y_scaled[y_scaled.index.get_level_values(0).year < 2010]

    # model = sm.MixedLM(
    #     y_train,
    #     sm.add_constant(X_train),
    #     groups
    # )
    model = sm.OLS(y_train, X_train)
    fit = model.fit()
    print(fit.summary())
    # II()
    # fit = model.fit()
    # II()

if __name__ == "__main__":
    df = read_tva_data()
    II()
    # scaled_MixedEffects(df, groups = ["NaturalOnly"])

    # scaledOLS(df)
