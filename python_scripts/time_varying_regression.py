import pandas as pd
import numpy as np
import scipy
from IPython import embed as II
import pathlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import r2_score
from datetime import timedelta, datetime
import seaborn as sns
from math import ceil
import pickle 
import sys
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

pickles = pathlib.Path("..", "pickles")

df = pd.read_pickle(pickles / "tva_dam_data.pickle")

# create a time series of previous days storage for all reservoirs
# df["Storage_pre"] = df.groupby(df.index.get_level_values(1))["Storage"].shift(1)
# drop the instances where there was no preceding values
# (i.e. the first day of record for each reservoir because there is no previous storage)
storage = df["Storage"].unstack()
inflow = df["Inflow"].unstack()
release = df["Release"].unstack()

storage_trimmed = storage.dropna()
inflow_trimmed = inflow.dropna()
release_trimmed = release.dropna()

#* Information about data record
#* There is missing data from 1982 to 1990-10-16
#* after then all of the data through 2015 is present. 
#* if we just include data from 1991 till the end we still have
#* like 250000 data points (27 for each day for 25 years)
#* I am going to trim the data sets down to start in 1990-10-16,
#* then I will start fitting regressions starting in 1991-01-01.

storage_trimmed = storage_trimmed[storage_trimmed.index >= datetime(1990, 10, 16)]
inflow_trimmed = inflow_trimmed[inflow_trimmed.index >= datetime(1990, 10, 16)]
release_trimmed = release_trimmed[release_trimmed.index >= datetime(1990, 10, 16)]

initial_date = datetime(1991, 1, 1)
ndays = (storage_trimmed.index[-1] - initial_date).days
delta30 = timedelta(days=30)
delta1 = timedelta(days=1)
regression_index = list(storage_trimmed[storage_trimmed.index >= initial_date].index)
reservoirs = storage_trimmed.columns
results = {}
export_results = {}
daily_mean_release = pd.read_pickle(pickles/"tva_daily_mean_release.pickle")
daily_mean_storage = pd.read_pickle(pickles/"storage_daily_means.pickle")
daily_mean_inflow = pd.read_pickle(pickles/"inflow_daily_means.pickle")
storage_windowed_mean = pd.read_pickle(pickles/"storage_windowed_means.pickle")
release_windowed_mean = pd.read_pickle(pickles/"release_windowed_means.pickle")
inflow_windowed_mean = pd.read_pickle(pickles/"inflow_windowed_means.pickle")


def make_day_window(dayofyear, windowsize=30):
    if dayofyear - windowsize <= 0:
        remaining = abs(dayofyear - windowsize)
        first = list(range(365-remaining + 1, 366))
        last = list(range(1, dayofyear+1))
        return first + last
    else:
        return list(range(dayofyear - windowsize + 1, dayofyear + 1))


def get_windowed_averages(data, windowsize=30):
    new_df = pd.DataFrame()
    for index in data.index:
        dayofyear = index.dayofyear
        window = make_day_window(dayofyear-1, windowsize)
        windowed_mean = data[data.index.dayofyear.isin(window)].mean()
        new_df[index] = windowed_mean
    return new_df


# N = release_trimmed.index.shape[0]
N = ndays
n_proc = ceil(N/size)
regress_dates = storage_trimmed[storage_trimmed.index >= initial_date].index
mine = regress_dates[rank*n_proc:(rank+1) * n_proc]

for date in mine:
    endog = release_trimmed.loc[date, :]
    my_year = date.year
    day_of_year = date.timetuple().tm_yday
    endog = endog/daily_mean_release.loc[day_of_year]
    exog_inflow = pd.DataFrame()
    exog_release = pd.DataFrame()
    exog_storage = pd.DataFrame()
    for year in range(1991, 2016):
        if year != my_year:
            end_index = datetime(year, 1, 1) + timedelta(day_of_year - 2)
            start_index = end_index - delta30
            year_inflow = inflow_trimmed[
                (inflow_trimmed.index >= start_index) & (inflow_trimmed.index < end_index)].mean()
            # this gets the 30 day average storage
            # year_storage = storage_trimmed[
            #     (storage_trimmed.index >= start_index) & (storage_trimmed.index < end_index)].mean()
            year_storage = storage_trimmed.loc[end_index]
            year_release = release_trimmed[
                (release_trimmed.index >= start_index) & (release_trimmed.index < end_index)].mean()
            exog_inflow[f"Inflow{year}"] = year_inflow
            exog_storage[f"Storage{year}"] = year_storage
            exog_release[f"Release{year}"] = year_release

    exog_inflow = exog_inflow.mean(axis=1)/inflow_windowed_mean[date]
    # exog_storage = exog_storage.mean(axis=1)/storage_windowed_mean[date]
    exog_storage = exog_storage.mean(axis=1)/daily_mean_storage.loc[(date - delta1).timetuple().tm_yday]
    exog_release = exog_release.mean(axis=1)/release_windowed_mean[date]
    
    exog = pd.DataFrame([exog_inflow, exog_release, exog_storage], 
            index=["Inflow", "Release", "Storage"]).T

    exog = sm.add_constant(exog)
    model = sm.OLS(endog, exog)
    fit = model.fit()

    pred_end_index = date - delta1
    pred_start_index = pred_end_index - delta30
    exog_inflow = inflow_trimmed[
        (inflow_trimmed.index >= pred_start_index) & (inflow_trimmed.index < pred_end_index)].mean()
    exog_storage = storage_trimmed.loc[pred_end_index]
    exog_release = release_trimmed[
        (release_trimmed.index >= pred_start_index) & (release_trimmed.index < pred_end_index)].mean()

    exog_inflow = exog_inflow / inflow_windowed_mean[date]
    exog_storage = exog_storage / daily_mean_storage.loc[(date - delta1).timetuple().tm_yday]
    exog_release = exog_release / release_windowed_mean[date]
    
    exog = pd.DataFrame([exog_inflow, exog_release, exog_storage],
                        index=["Inflow", "Release", "Storage"]).T
    exog = sm.add_constant(exog)
    preds = fit.predict(exog)
    preds_score = r2_score(endog * daily_mean_release.loc[day_of_year], preds * daily_mean_release.loc[day_of_year])

    results[date] = fit    
    export_results[date] = {
        "score": r2_score(endog * daily_mean_release.loc[day_of_year], fit.fittedvalues * daily_mean_release.loc[day_of_year]),
        "pred_score": preds_score,
        "preds": preds * daily_mean_release.loc[day_of_year],
        "params":fit.params,
        "fittedvalues": fit.fittedvalues * daily_mean_release.loc[day_of_year],
        "adj_score":fit.rsquared_adj
    }


results = comm.gather(results, root=0)
export_results = comm.gather(export_results, root=0)

if rank == 0:
    from collections import ChainMap
    # gather gives you a list of whatever is on eachprocessor
    # i want to merge the dictionaries and this is a simple way to do it
    export_results = dict(ChainMap(*export_results))
    with open("./storage_m1_results/simple_regression_results.pickle", "wb") as f:
        pickle.dump(export_results, f)
    results = dict(ChainMap(*results))
    with open("./storage_m1_results/regression_results.pickle", "wb") as f:
        pickle.dump(results, f)
