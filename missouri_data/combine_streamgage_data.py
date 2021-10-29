import pandas as pd
import json
import re
import sys
import pickle
from IPython import embed as II

with open("./acoe_data/meta_data/res_sgs.json", "r") as f:
    sgs = json.load(f)

def find_skip_rows(file):
    with open(file, "r") as f:
        lines = f.readlines()
    pattern = r"5s\t15s\t20d\t14n\t10s"
    break_line = 0
    for i, line in enumerate(lines):
        if re.search(pattern, line):
            break_line = i
            break
    return i+1

def load_prep_data(file):
    skiprows = find_skip_rows(file)
    df = pd.read_csv(file,
                    delim_whitespace=True, 
                    skiprows=skiprows, 
                    names=["agency", "site", "datetime", station, "flag"]
    )
    df = df.drop(["agency", "site", "flag"], axis=1)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")
    df = df.replace("Ice", 0)
    df = df.replace("Rat", 0)
    try:
        df = df.astype(float)
    except ValueError as e:
        print(file)
        II()
    return df

output = {}

for res, values in sgs.items():
    upstream_values = []
    for station in values["upstream"]:
        file = f"./usgs_data/streamgages/{station}.tsv"
        df = load_prep_data(file)
        upstream_values.append(df)
    upstream = pd.concat(upstream_values,ignore_index=False, axis=1)
    inflow = upstream.sum(axis=1)
    
    downstream_values = []
    for station in values["downstream"]:
        if station[-1] == "-":
            minus = True
            station = station[:-1]
        else:
            minus = False
        file = f"./usgs_data/streamgages/{station}.tsv"
        df = load_prep_data(file)
        if minus:
            df *= -1
        downstream_values.append(df)
    if downstream_values:
        downstream = pd.concat(downstream_values, ignore_index=False, axis=1)
        release = downstream.sum(axis=1)
    else:
        release = pd.Series(0, index=inflow.index)
    output[res] = pd.DataFrame({"inflow":inflow, "release":release})

with open("./usgs_data/prepped_data/res_inflow.pickle", "wb") as f:
    pickle.dump(output, f)
    
