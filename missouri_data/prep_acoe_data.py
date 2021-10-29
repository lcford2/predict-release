import pandas as pd
import glob


def load_rename_resample(file):
    df = pd.read_csv(file)
    df = df.dropna(axis=1, how="all")
    df = df.rename(columns={"Date-Time":"datetime", "Elevation (ft)":"elev",
        "Inflow (cfs)":"inflow", "Outflow (cfs)":"release"})
    df = df.loc[:,["datetime", "elev", "inflow", "release"]]
    df.loc[:,"datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")
    daily = df[["elev"]].resample("D").last()
    daily[["inflow", "release"]] = df[["inflow", "release"]].resample("D").mean()
    return daily

if __name__ == "__main__":
    files = glob.glob("./acoe_data/csv/*.csv")
    from IPython import embed as II
    II()

