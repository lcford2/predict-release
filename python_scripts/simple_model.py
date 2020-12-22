import pandas as pd
import numpy as np
import scipy
from IPython import embed as II
import pathlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from datetime import timedelta, datetime
import seaborn as sns
from math import ceil
import pickle
import sys

pickles = pathlib.Path("..", "pickles")

df = pd.read_pickle(pickles / "tva_dam_data.pickle")

# create a time series of previous days storage for all reservoirs
df["Storage_pre"] = df.groupby(df.index.get_level_values(1))["Storage"].shift(1)
df["Release_pre"] = df.groupby(df.index.get_level_values(1))["Release"].shift(1)

#* Information about data record
#* There is missing data from 1982 to 1990-10-16
#* after then all of the data through 2015 is present.
#* if we just include data from 1991 till the end we still have
#* like 250000 data points (27 for each day for 25 years)
#* I am going to trim the data sets down to start in 1990-10-16,
#* then I will start fitting regressions starting in 1991-01-01.

start_date = datetime(1990, 10, 16)
df = df[df.index.get_level_values(0) >= start_date]

# pull out and get all variables to correct units
df["Storage"] = df["Storage"] * 86400 * 1000  # 1000 second-ft-day to ft3
df["Storage_pre"] = df["Storage_pre"] * \
    86400 * 1000  # 1000 second-ft-day to ft3
df["Net Inflow"] = df["Net Inflow"] * 86400  # cfs to ft3/day
df["Release"] = df["Release"] * 86400  # cfs to ft3/day
df["Release_pre"] = df["Release_pre"] * 86400  # cfs to ft3/day

#* Storage should be box cox transformed with a lambda of 0.3219
st_bxcx, st_lmbda = scipy.stats.boxcox(df["Storage"])
df["Storage_bxcx"] = st_bxcx
df["Storage_pre_bxcx"] = scipy.stats.boxcox(df["Storage_pre"], lmbda=st_lmbda)
df["Storage_ln"] = np.log(df["Storage"])
df["Storage_pre_ln"] = np.log(df["Storage_pre"])

#* Release has one added to it to make all values positive
#* then it is transformed with a lmbda of 0.1030
df["Release_bxcx"], rel_lmbda = scipy.stats.boxcox(df["Release"]+1)
df["Release_pre_bxcx"] = scipy.stats.boxcox(df["Release_pre"] + 1, lmbda=rel_lmbda)
df["Release_ln"] = np.log(df["Release"] + 1)
df["Release_pre_ln"] = np.log(df["Release_pre"] + 1)

#* Net Inflow is made positive by adding the minimum value and 1 
#* it is then log-transformed because the box cox value is approx 0
min_inflow = df["Net Inflow"].min()
df["Net_Inflow_ln"] = np.log(df["Net Inflow"]+1+abs(min_inflow))


def logOLS(df):
    X = sm.add_constant(df[["Storage_pre_ln", "Net_Inflow_ln", "Release_pre_ln"]])
    y = df["Release_ln"]

    X_test = X[X.index.get_level_values(0).year >= 2010]
    X_train = X[X.index.get_level_values(0).year < 2010]

    y_test = y[y.index.get_level_values(0).year >= 2010]
    y_train = y[y.index.get_level_values(0).year < 2010]

    model = sm.OLS(y_train, X_train)
    fit = model.fit()

    preds = fit.predict(X_test)

    y_fit_back = np.exp(fit.fittedvalues) - 1
    y_pred_back = np.exp(preds) - 1

    y_act_train = np.exp(y_train) - 1
    y_act_test = np.exp(y_test) - 1

    print(f"Train Score : {r2_score(y_act_train, y_fit_back):.4f}")
    print(f"Test Score  : {r2_score(y_act_test, y_pred_back):.4f}")

def scale_df(df):
    grouper = df.index.get_level_values(1)
    means = df.groupby(grouper).mean()
    std = df.groupby(grouper).std()
    scaled_df = pd.DataFrame(df.values, index=df.index, columns=df.columns)
    idx = pd.IndexSlice
    for index in means.index:
        scaled_df.loc[idx[:,index],:] = (scaled_df.loc[idx[:,index],:] - means.loc[index]) / std.loc[index]
    return scaled_df, means, std

def prep_single_res_data(df):
    means = df.mean(axis=1)
    

def scaledOLS(df, scaler="mine"):
    if scaler == "mine":
        scaled_df, means, std = scale_df(df)
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
    # scaled_df = scaled_df.rename(columns={"Net Inflow":"NetInflow"})
    # train_df = scaled_df[scaled_df.index.get_level_values(0).year < 2010]
    # test_df = scaled_df[scaled_df.index.get_level_values(0).year >= 2010]

    # model = smf.ols("Release ~ Storage + Release_pre + NetInflow", data=train_df)
    fit = model.fit()
 
    # preds = pd.DataFrame(fit.predict(test_df), columns=["Release"])
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

    # anova = sm.stats.anova_lm(fit, typ=2)
    # print(anova)
    scaled_OLS_results = {
        "means":means,
        "std":std,
        "fit":fit,
        "model":model,
        "test_set":test_set,
        "train_set":train_set,
        "scaled_df":scaled_df
        # "anova":anova
    }
    with open("./simple_model_results/scaled_OLS_results.pickle", "wb") as f:
        pickle.dump(scaled_OLS_results, f)

    print(f"Train Score : {r2_score(y_train, fitted):.4f}")
    print(f"Test Score  : {r2_score(y_test, preds):.4f}")
    # II()

scaledOLS(df)
# scale_df(df)
