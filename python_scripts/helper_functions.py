import numpy as np
import pandas as pd
import pathlib
from datetime import datetime

def scale_multi_level_df(df):
    grouper = df.index.get_level_values(1)
    means = df.groupby(grouper).mean()
    std = df.groupby(grouper).std()
    scaled_df = pd.DataFrame(df.values, index=df.index, columns=df.columns)
    idx = pd.IndexSlice
    for index in means.index:
        scaled_df.loc[idx[:,index],:] = (scaled_df.loc[idx[:,index],:] - means.loc[index]) / std.loc[index]
    return scaled_df, means, std

def prep_single_res_data(df, unit_change):
    keep_keys = []
    for key, conv in unit_change.items():
        df[key] = df[key] * conv
        new_key = "_".join(key.split("_")[:-1])
        df = df.rename(columns={key:new_key})
        keep_keys.append(new_key)
   
    df = df[keep_keys]
    means = df.mean()
    std = df.std()
    normed = (df - means) / std
    normed["release_pre"] = normed["release"].shift(1)
    normed["storage_pre"] = normed["storage"].shift(1)
    normed = normed.loc[normed.index[1:]]
    return normed, means, std

def read_tva_data():
    pickles = pathlib.Path("..", "pickles")
    df = pd.read_pickle(pickles / "tva_dam_data.pickle")

    # create a time series of previous days storage for all reservoirs
    df["Storage_pre"] = df.groupby(df.index.get_level_values(1))["Storage"].shift(1)
    df["Release_pre"] = df.groupby(df.index.get_level_values(1))["Release"].shift(1)

    #* Information about data record
    # There is missing data from 1982 to 1990-10-16
    # after then all of the data through 2015 is present.
    # if we just include data from 1991 till the end we still have
    # like 250000 data points (27 for each day for 25 years)

    start_date = datetime(1990, 10, 16)
    # trim data frame
    df = df[df.index.get_level_values(0) >= start_date]

    # get all variables to similar units for Mass Balance
    df["Storage"] = df["Storage"] * 86400 * 1000  # 1000 second-ft-day to ft3
    df["Storage_pre"] = df["Storage_pre"] * \
        86400 * 1000  # 1000 second-ft-day to ft3
    df["Net Inflow"] = df["Net Inflow"] * 86400  # cfs to ft3/day
    df["Release"] = df["Release"] * 86400  # cfs to ft3/day
    df["Release_pre"] = df["Release_pre"] * 86400  # cfs to ft3/day
    return df