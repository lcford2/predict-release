import numpy as np
import pandas as pd
import pathlib
from datetime import datetime
from timing_function import time_function
from IPython import embed as II

tva_res = ['BlueRidge', 'Chikamauga', 'Guntersville', 'Hiwassee', 
           'Nikajack', 'Norris', 'Ocoee1', 'Pickwick', 'Wheeler', 
           'Wilson', 'Cherokee', 'WattsBar', 'Nottely', 'Chatuge',
           'Apalachia', 'Douglas', 'Ocoee3', 'FtLoudoun', 'Kentucky',
           'Fontana', 'Watauga', 'SHolston', 'Boone', 'FtPatrick', 
           'MeltonH', 'TimsFord', 'Wilbur']

acf_res = ['Woodruff', 'Buford', 'George', 'West']


# @time_function
def scale_multi_level_df(df, timelevel="all"):
    if isinstance(df, pd.DataFrame):
        columns = df.columns
    else:
        columns = [df.name]
    if timelevel == "all":
        grouper = df.index.get_level_values(1)
        means = df.groupby(grouper).mean()
        std = df.groupby(grouper).std()
        scaled_df = pd.DataFrame(df.values, index=df.index, columns=columns)
        idx = pd.IndexSlice
        for index in means.index:
            scaled_df.loc[idx[:, index], :] = (
                scaled_df.loc[idx[:, index], :] - means.loc[index]) / std.loc[index]
    else:
        scaled_df = pd.DataFrame(df.values, index=df.index, columns=columns)
        df = df.reset_index().rename(
            columns={"level_0": "Date", "level_1": "Reservoir"})
        
        df["TimeGroup"] = getattr(df["Date"].dt, timelevel)
        means = df.groupby(["TimeGroup", "Reservoir"]).mean()
        std = df.groupby(["TimeGroup", "Reservoir"]).std()

        # check memoized scaled file
        idx = pd.IndexSlice
        for index in means.index:
            scaled_index = idx[
                getattr(scaled_df.index.get_level_values(0), timelevel) == index[0],
                index[1]
            ]
            scaled_df.loc[scaled_index, :] = (scaled_df.loc[scaled_index, :] -
                                              means.loc[index]) / std.loc[index]

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

def find_max_date_range(file="date_res.csv"):
    # This function finds the longest date range in the data set
    # that has the same information. 
    # It could be update in the future to find the longest date range
    # that has a specific set of information (such as certain reservoirs)
    # this could be done by filtering the date set then performing all of this
    date_res = pd.read_csv(file, usecols=[0, 1], header=None, 
                           index_col=0, names=["Date", "NRes"])
    date_res.index = pd.to_datetime(date_res.index)
    nres = date_res["NRes"].max()
    date_res["streak_start"] = date_res["NRes"].ne(date_res["NRes"].shift())
    date_res["streak_id"] = date_res["streak_start"].cumsum()
    date_res["streak_count"] = date_res.groupby("streak_id").cumcount() + 1
    end = date_res["streak_count"].idxmax()
    sid = date_res.loc[end, "streak_id"]
    start = date_res[(date_res["streak_id"] == sid) & (date_res["streak_start"])].index[0]
    date_range = pd.date_range(start=start, end=end)
    return date_range


def read_tva_data(just_load=False):
    pickles = pathlib.Path("..", "pickles")
    df = pd.read_pickle(pickles / "tva_dam_data.pickle")
    fractions = pd.read_pickle(pickles / "tva_fractions.pickle")
    for column in fractions.columns:
        df[column] = [fractions.loc[i, column] for i in df.index.get_level_values(1)]
    
    start_date = datetime(1990, 10, 16)
    # trim data frame
    df = df[df.index.get_level_values(0) >= start_date]

    # get all variables to similar units for Mass Balance
    df.loc[:,"Storage"] *= 86400 * 1000  # 1000 second-ft-day to ft3
    # df.loc[:,"Storage"] = df.loc[:,"Storage"] / 43560 / 1000 # ft3 to 1000 acre ft
    # df.loc[:,"Storage_pre"] = df.loc[:,"Storage_pre"] * \
        # 86400 * 1000  # 1000 second-ft-day to ft3
    df.loc[:,"Net Inflow"] *= 86400  # cfs to ft3/day
    # df.loc[:,"Net Inflow"] = df.loc[:,"Net Inflow"] / 43560 / 1000 # ft3/day to 1000 acre-ft/day
    df.loc[:,"Release"] *= 86400  # cfs to ft3/day
    # df.loc[:,"Release"] = df.loc[:,"Release"] / 43560 / 1000  # ft3/day to 1000 acre-ft/day
    # df.loc[:,"Release_pre"] = df.loc[:,"Release_pre"] * 86400  # cfs to ft3/day

    # maybe make these values 1000 acre ft to help solver
    df.loc[:,"Storage"] = df.loc[:,"Storage"] / 43560 / 1000
    df.loc[:,"Release"] = df.loc[:,"Release"] / 43560 / 1000
    df.loc[:,"Net Inflow"] = df.loc[:,"Net Inflow"] / 43560 / 1000
 
    df[["Storage_pre", "Release_pre"]] = df.groupby(df.index.get_level_values(1))[
        ["Storage", "Release"]].shift(1)

    # create a time series of previous days storage for all reservoirs
    if not just_load:
        df[["Storage_7", "Release_7"]] = df.groupby(df.index.get_level_values(1))[
            ["Storage", "Release"]].shift(7)
        
        tmp = df.groupby(df.index.get_level_values(1))[
            ["Storage_pre", "Release_pre", "Net Inflow"]].rolling(7, min_periods=1).mean()
        tmp.index = tmp.index.droplevel(0)
        tmp = tmp.sort_index()
        df[["Storage_roll7", "Release_roll7", "Inflow_roll7"]] = tmp

        tmp = df.groupby(df.index.get_level_values(1))[
            ["Storage_pre", "Release_pre", "Net Inflow"]].rolling(14, min_periods=1).mean()
        tmp.index = tmp.index.droplevel(0)
        tmp = tmp.sort_index()
        df[["Storage_roll14", "Release_roll14", "Inflow_roll14"]] = tmp

    #* Information about data record
    # There is missing data from 1982 to 1990-10-16
    # after then all of the data through 2015 is present.
    # if we just include data from 1991 till the end we still have
    # like 250000 data points (27 for each day for 25 years)

    df = df.dropna()
    return df

def read_all_res_data():
    pickles = pathlib.Path("..", "pickles")
    df = pd.read_pickle(pickles / "all_res_data.pickle")
    df = df.drop("Inflow", axis=1)

    # create a time series of previous days storage for all reservoirs
    df["Storage_pre"] = df.groupby(df.index.get_level_values(1))[
        "Storage"].shift(1)
    df["Release_pre"] = df.groupby(df.index.get_level_values(1))[
        "Release"].shift(1)

    df = df.dropna()

    # TVA Data Units:
    # Storage is in 1000 second-ft-day
    # Inflow and release are in CFS

    # ACF Data Units:
    # Storage is in ac-ft
    # Inflow and release are in CFS

    # get all variables to similar units for Mass Balance
    # These variables are all CFS regardless of which basin they are in
    df["Net Inflow"] = df["Net Inflow"] * 86400  # cfs to ft3/day
    df["Release"] = df["Release"] * 86400  # cfs to ft3/day
    df["Release_pre"] = df["Release_pre"] * 86400  # cfs to ft3/day

    # TVA Storage is a specific unit
    tva_indexer = df.index.get_level_values(1).isin(tva_res)
    df.loc[tva_indexer,"Storage"] = df[tva_indexer]["Storage"] * 86400 * 1000  # 1000 second-ft-day to ft3
    df.loc[tva_indexer,"Storage_pre"] = df[tva_indexer]["Storage_pre"] * 86400 * 1000  # 1000 second-ft-day to ft3

    # ACF Storage is a specific unit
    acf_indexer = df.index.get_level_values(1).isin(acf_res)
    df.loc[acf_indexer,"Storage"] = df[acf_indexer]["Storage"]* 86400  # acre-ft to ft3
    df.loc[acf_indexer,"Storage_pre"] = df[acf_indexer]["Storage_pre"]* 86400  # acre-ft to ft3
    return df

def flatten_2d_list(lst):
    return [item for sublist in lst for item in sublist]

def calc_bias(y_a, y_m):
    return np.mean(y_m) - np.mean(y_a)