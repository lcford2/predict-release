import pickle
from IPython import embed as II
from scipy.interpolate import interp1d
import pandas as pd


with open("./usgs_data/prepped_data/res_inflow.pickle", "rb") as f:
    flows = pickle.load(f)


dfs = {i:pd.read_csv(f"./acoe_data/csv/{i}.csv") for i in flows.keys()}

storage_curves = {i:pd.read_csv(f"./acoe_data/meta_data/storage_curves/{i}_storage_elevation_area.csv")[["elev", "storage"]] for i in flows.keys()}

storage_interps = {i:interp1d(j["elev"], j["storage"]) for i,j in storage_curves.items()}

for res, df in dfs.items():
    df = df[["Elevation (ft)", "Inflow (cfs)", "Outflow (cfs)", "Date-Time"]]
    df = df.rename(columns = {"Elevation (ft)":"elev", "Inflow (cfs)":"inflow", "Outflow (cfs)":"release", "Date-Time":"datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")
    daily = df[["elev"]].resample("D").last()
    daily[["release_acoe", "inflow_acoe"]] = df[["release", "inflow"]].resample("D").mean()
    daily["storage"] = daily["elev"].apply(storage_interps[res])
    daily["release"] = flows[res]["release"]
    daily["inflow"] = flows[res]["inflow"]
    daily["sto_delta"] = daily["storage"].diff(periods=1)
    dfs[res] = daily

dfs_ready = []

for res, df in dfs.items():
    df["site_name"] = res
    df = df.reset_index()
    index = pd.MultiIndex.from_tuples(zip(df["site_name"], df["datetime"]))
    df.index = index
    df = df.drop(["datetime", "site_name"], axis=1)
    dfs_ready.append(df)

output = pd.concat(dfs_ready, axis=0, ignore_index=False)
output.to_csv("./acoe_data/csv/prepped/missouri_data.csv")