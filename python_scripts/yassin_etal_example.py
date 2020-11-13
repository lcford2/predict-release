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

pickles = pathlib.Path("..", "pickles")

df = pd.read_pickle(pickles / "tva_dam_data.pickle")

# create a time series of previous days storage for all reservoirs
# df["Storage_pre"] = df.groupby(df.index.get_level_values(1))["Storage"].shift(1)
# drop the instances where there was no preceding values
# (i.e. the first day of record for each reservoir because there is no previous storage)
storage = df["Storage"].unstack() * 86400 * 1000 # 1000 second-ft-day to ft3
inflow = df["Net Inflow"].unstack() * 86400 # cfs to ft3/day
release = df["Release"].unstack() * 86400 # cfs to ft3/day 

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

storage_trimmed = storage_trimmed[storage_trimmed.index >= datetime(
    1990, 10, 16)]
inflow_trimmed = inflow_trimmed[inflow_trimmed.index >= datetime(1990, 10, 16)]
release_trimmed = release_trimmed[release_trimmed.index >= datetime(
    1990, 10, 16)]

# all reservoirs are non-irrigation reservoirs
initial_date = datetime(1991, 1, 1)
ndays = (storage_trimmed.index[-1] - initial_date).days
delta1 = timedelta(days=1)
regression_index = list(
    storage_trimmed[storage_trimmed.index >= initial_date].index)
reservoirs = storage_trimmed.columns
results = {}
export_results = {}
# daily means for all reservoirs
daily_mean_release = pd.read_pickle(pickles / "tva_daily_mean_release.pickle")
daily_mean_storage = pd.read_pickle(pickles / "storage_daily_means.pickle")
# daily_mean_inflow = pd.read_pickle(pickles / "inflow_daily_means.pickle")
daily_mean_inflow = inflow_trimmed.groupby(inflow_trimmed.index.dayofyear).mean()
max_storage = pd.read_pickle(pickles / "max_storage.pickle")
max_inflow = inflow_trimmed.max()
max_release = release_trimmed.max()

Sdead = 0.1 * max_storage
grouping = "month"
Sn = storage_trimmed.groupby(getattr(storage_trimmed.index, grouping)).quantile(0.45)
Sm = storage_trimmed.groupby(getattr(storage_trimmed.index, grouping)).quantile(0.85)
Sc = storage_trimmed.groupby(getattr(storage_trimmed.index, grouping)).quantile(0.1)


Qmc = release_trimmed.quantile(0.99)
Qn = release_trimmed.groupby(getattr(release_trimmed.index, grouping)).quantile(0.45)
Qm = release_trimmed.groupby(getattr(release_trimmed.index, grouping)).quantile(0.85)
Qc = release_trimmed.groupby(getattr(release_trimmed.index, grouping)).quantile(0.1)

new_release = pd.DataFrame(0, index=regression_index, columns=storage_trimmed.columns)

for date in regression_index:
    date_1 = date - delta1
    dayofyear = date.timetuple().tm_yday
    month = date.month
    dayofyear_1 = (date - delta1).timetuple().tm_yday
    if grouping == "month":
        sloc = month
        qloc = month
    elif grouping == "dayofyear":
        sloc = dayofyear_1
        qloc = dayofyear

    for res in storage_trimmed.columns:
        Smin = Sdead[res]
        Smax = Sm.loc[sloc,res]
        Snorm = Sn.loc[sloc,res]
        Scrit = Sc.loc[sloc,res]
        S = storage_trimmed.loc[date_1,res]
        Qmax = Qm.loc[qloc, res]
        Qnorm = Qn.loc[qloc, res]
        Qcrit = Qc.loc[qloc, res]
        Qmax_chan = Qmc[res]
        I = inflow_trimmed.loc[date, res]
        if S <= Smin:
            Q = 0
        elif S <= Scrit:
            Q = min([Qcrit, S - Smin])
        elif S <= Snorm:
            Q = Qcrit + (Qnorm - Qcrit)*(S - Snorm)/(Smax - Scrit)
        elif S <= Smax:
            if I <= Qnorm:
                Q = Qnorm + (Qmax - Qnorm)*(S-Snorm)/(Smax-Snorm)
            else:
                Q = Qnorm + max([I - Qnorm, Qmax - Qnorm])*(S - Snorm)/(Smax - Snorm)
        elif S > Smax:
            Q = min([max([S - Smax, Qmax]), Qmax_chan])
        new_release.loc[date, res] = Q

scores = {}
for col in new_release.columns:
    scores[col] = r2_score(release_trimmed.loc[new_release.index, col], new_release[col])

scores = pd.Series(scores)

II()
