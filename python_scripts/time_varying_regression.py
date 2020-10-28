import pandas as pd
import numpy as np
import scipy
from IPython import embed as II
import pathlib
import matplotlib.pyplot as plt
from statsmodels.regression.mixed_linear_model import MixedLM, MixedLMParams
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.metrics import r2_score
from datetime import timedelta, datetime
import seaborn as sns
from statsmodels.tsa.api import VAR
import pickle 

pickles = pathlib.Path("..", "pickles")

df = pd.read_pickle(pickles / "tva_dam_data.pickle")
# df = df[df.index.get_level_values(0).year >= year]

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
regression_index = list(storage_trimmed[storage_trimmed.index >= initial_date].index)
reservoirs = storage_trimmed.columns
results = {}
export_results = {}
daily_mean_release = pd.read_pickle(pickles/"tva_daily_mean_release.pickle")

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
II()
sys.exit()
# II()
# this needs to be parallelized. transfer to bezier to do. 
#* scratch that, put on github then just pull to bezier
for date in release_trimmed.index:
    endog = release_trimmed.loc[date, :]
    my_year = date.year
    day_of_year = date.timetuple().tm_yday
    exog_inflow = pd.DataFrame()
    exog_release = pd.DataFrame()
    exog_storage = pd.DataFrame()
    for year in range(1991, 2016):
        if year != my_year:
            end_index = datetime(year, 1, 1) + timedelta(day_of_year - 1)
            start_index = end_index - delta30
            year_inflow = inflow_trimmed[
                (inflow_trimmed.index >= start_index) & (inflow_trimmed.index < end_index)].mean()
            year_storage = storage_trimmed[
                (storage_trimmed.index >= start_index) & (storage_trimmed.index < end_index)].mean()
            year_release = release_trimmed[
                (release_trimmed.index >= start_index) & (release_trimmed.index < end_index)].mean()
            exog_inflow[f"Inflow{year}"] = year_inflow
            exog_storage[f"Storage{year}"] = year_storage
            exog_release[f"Release{year}"] = year_release

    exog_inflow = exog_inflow.mean(axis=1)
    exog_storage = exog_storage.mean(axis=1)
    exog_release = exog_release.mean(axis=1)

    exog = pd.DataFrame([exog_inflow, exog_release, exog_storage], index=["Inflow", "Release", "Storage"]).T

    exog = sm.add_constant(exog)
    model = sm.OLS(endog, exog)
    fit = model.fit()
    results[date] = fit
    export_results[date] = {
        "score":fit.rsquared,
        "params":fit.params,
        "fittedvalues":fit.fittedvalues,
        "adj_score":fit.rsquared_adj
    }


with open("./regression_result_pickles/simple_regression_results.pickle", "wb") as f:
    pickle.dump(export_results, f)

with open("./regression_result_pickles/regression_results.pickle", "wb") as f:
    pickle.dump(results, f)