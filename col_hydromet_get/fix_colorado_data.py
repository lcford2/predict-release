import pandas as pd
import sys
from IPython import embed as II


def load_data():
    return pd.read_pickle("./data/colorado_trimmed.pickle")


def make_storage_pre(df):
    df["storage_pre"] = df.groupby(
        df.index.get_level_values(0)
    )["storage"].shift(1)
    return df


def make_calc_timeseries(df):
    df = make_storage_pre(df)
    df["my_inflow_volume"] = df["storage"] - df["storage_pre"] + df["release volume"]
    df["my_inflow"] = df["my_inflow_volume"] * 43560 / 3600 / 24
    df["my_release_volume"] = df["storage_pre"] - df["storage"] + df["inflow volume"]
    df["my_release"] = df["my_release_volume"] * 43560 / 3600 / 24
    return df


def select_good_rows(df):
    good_rows = df.isna().query(
        "((inflow == False |`inflow volume` == False)|my_inflow == False) & \
         ((`release volume` == False | `total release` == False)|my_release == False)"
    ).index
    return df.loc[good_rows, :]


def fill_na(df):
    for i, row in df.isna().iterrows():
        if row["inflow"]:
            df.loc[i, "inflow"] = df.loc[i, "my_inflow"]

        if row["inflow volume"]:
            df.loc[i, "inflow volume"] = df.loc[i, "my_inflow_volume"]

        if row["total release"]:
            df.loc[i, "total release"] = df.loc[i, "my_release"]

        if row["release volume"]:
            df.loc[i, "release volume"] = df.loc[i, "my_release_volume"]
    return df


def main():
    df = load_data()
    df = make_calc_timeseries(df)
    df = select_good_rows(df)
    df = fill_na(df)
    II()

if __name__ == "__main__":
    main()
