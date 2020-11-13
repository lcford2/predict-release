import pandas as pd
import numpy as np
from IPython import embed as II


def combine_data():
    storage_file = "../pickles/observed_storage.pickle"
    inflow_file = "../pickles/observed_uncont_inflow.pickle"
    release_file = "../pickles/observed_total.pickle"
    net_inflow_file = "../pickles/observed_net_inflow.pickle"

    storage = pd.read_pickle(storage_file)
    inflow = pd.read_pickle(inflow_file)
    release = pd.read_pickle(release_file)
    net_inflow = pd.read_pickle(net_inflow_file)

    storage = storage.drop("RacoonMt", axis=1)
    release = release.drop("RacoonMt", axis=1)

    storage = storage.stack()
    release = release.stack()
    inflow = inflow.stack()
    net_inflow = net_inflow.stack()

    df = pd.DataFrame([storage, release, inflow, net_inflow], index=["Storage", "Release", "Inflow", "Net Inflow"]).T
    df = df.dropna()
    df.to_pickle("../pickles/trimmed_TVA_all.pickle")

if __name__ == "__main__":
    response = input("This script can take a long time to run. Are you sure you want to run this (y or n)? ")
    if response.upper() == "Y":
        combine_data()
