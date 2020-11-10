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
daily_mean_inflow = pd.read_pickle(pickles / "inflow_daily_means.pickle")
max_storage = pd.read_pickle(pickles / "max_storage.pickle")
max_inflow = inflow_trimmed.max()
max_release = release_trimmed.max()

st_lagged = storage_trimmed.shift(1)/max_storage
normal_inflow = inflow_trimmed / max_release
normal_release = release_trimmed / max_release

leaveout = st_lagged.index[0]
cal_range = [leaveout + timedelta(days=1), datetime(2011, 12, 31)]
val_range = [datetime(2012, 1, 1), datetime(2015, 12, 31)]

endog_cal = normal_release.loc[cal_range[0]:cal_range[1]].unstack()
endog_val = normal_release.loc[val_range[0]:val_range[1]].unstack()

st_exog_cal = st_lagged.loc[cal_range[0]:cal_range[1]].unstack()
st_exog_val = st_lagged.loc[val_range[0]:val_range[1]].unstack()

inf_exog_cal = normal_inflow.loc[cal_range[0]:cal_range[1]].unstack()
inf_exog_val = normal_inflow.loc[val_range[0]:val_range[1]].unstack()

exog_cal = pd.DataFrame([st_exog_cal, inf_exog_cal], index=["PrevStorage", "Inflow"]).T
exog_val = pd.DataFrame([st_exog_val, inf_exog_val], index=["PrevStorage", "Inflow"]).T

exog_cal = sm.add_constant(exog_cal)
exog_val = sm.add_constant(exog_val)

model = sm.OLS(endog_cal, exog_cal)
fit = model.fit()
fitted = fit.fittedvalues.unstack().T * max_release
preds = fit.predict(exog_val)
preds = preds.unstack().T * max_release

endog_cal_act = endog_cal.unstack().T * max_release
endog_val_act = endog_val.unstack().T * max_release

II()
