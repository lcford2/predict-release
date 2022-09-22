import ctypes as ct
import numpy as np
import pandas as pd
from datetime import datetime
from numpy.ctypeslib import ndpointer
from time import perf_counter as timer
from calendar import month_abbr
from IPython import embed as II


def read_tva():
    tva = pd.read_pickle("~/projects/predict-release/pickles/tva_dam_data.pickle")
    start = datetime(1990, 10, 16)
    tva = tva[tva.index.get_level_values(0) >= start]
    tva.loc[:, "Storage"] *= (86400 / 43560)
    tva.loc[:, "Net Inflow"] *= (86400 / 43560 / 1000)
    tva.loc[:, "Release"] *= (86400 / 43560 / 1000)

    tva[["Storage_pre", "Release_pre"]] = tva.groupby(
        tva.index.get_level_values(1))[["Storage", "Release"]].shift(1)

    tva["Storage_Inflow_interaction"] = tva["Storage_pre"] * tva["Net Inflow"]
    return tva.dropna()

def load_library():
    lib = ct.CDLL("./libres_simul.so")

    lib.res_simul.argtypes = [
            # ct.c_double,
            ndpointer(ct.c_double, ndim=1, flags="C_CONTIGUOUS"),
            ndpointer(ct.c_double, ndim=1, flags="C_CONTIGUOUS"),
            ndpointer(ct.c_double, ndim=1, flags="C_CONTIGUOUS"),
            ndpointer(ct.c_double, ndim=1, flags="C_CONTIGUOUS"),
            ndpointer(ct.c_double, ndim=1, flags="C_CONTIGUOUS"),
            ct.c_double,
            ct.c_double,
            ct.c_double,
            ct.c_double,
            ct.c_double,
            ct.c_double,
            ct.c_double,
            ct.c_double,
            ct.c_int,
            ndpointer(ct.c_double, ndim=1, flags="C_CONTIGUOUS"),
            ndpointer(ct.c_double, ndim=1, flags="C_CONTIGUOUS")
    ]

    lib.res_simul.restype = None
    return lib

def initial_example(lib):
    intercept = 0.0
    re_coefs = np.array([1, -1, 1, 0, 1, -1, 1], dtype="float64")

    ts_len = 10000

    prev_rel = np.arange(ts_len+6, dtype="float64")
    prev_sto = np.arange(ts_len+6, dtype="float64")
    inflow   = np.arange(ts_len+6, dtype="float64")

    rel_mean = prev_rel.mean()
    sto_mean = prev_sto.mean()
    inf_mean = inflow.mean()
    sxi_mean = (prev_sto * inflow).mean()

    rel_std = prev_rel.std()
    sto_std = prev_sto.std()
    inf_std = inflow.std()
    sxi_std = (prev_sto * inflow).std()

    rel_out = np.zeros(ts_len, dtype="float64")
    sto_out = np.zeros(ts_len, dtype="float64")

    time1 = timer()
    lib.res_simul(intercept, re_coefs,
                prev_rel, prev_sto, inflow,
                rel_mean, sto_mean, inf_mean, sxi_mean,
                rel_std, sto_std, inf_std, sxi_std,
                ts_len,
                rel_out, sto_out)
    time2 = timer()

    print(f"Time to simul res: {time2 - time1:.5f}")

def test_tva_res(lib):
    tva = read_tva()
    params = pd.read_pickle("./params.pickle")
    groups = pd.read_pickle("./groups.pickle")
    res = "Kentucky"
    group = groups.loc[res]
    coefs = params[group]

    intercept = coefs["const"]
    re_coefs = coefs[[
        "Release_pre", "Storage_pre", "Net Inflow",
        "Storage_Inflow_interaction",
        "Release_roll7", "Storage_roll7", "Inflow_roll7"
    ]].values

    idx = pd.IndexSlice
    df = tva.loc[idx[:,res],:]
    means = df.mean()
    std = df.std()
    intercepts = np.array([intercept + coefs[month_abbr[i]]
                           for i in df.index.get_level_values(0).month],
                          dtype="float64")

    ts_len = df.shape[0] - 6
    rel_out = np.zeros(ts_len, dtype="float64")
    sto_out = np.zeros(ts_len, dtype="float64")

    time1 = timer()
    lib.res_simul(intercepts, re_coefs,
                  df["Release_pre"].values,
                  df["Storage_pre"].values,
                  df["Net Inflow"].values,
                  means["Release"],
                  means["Storage"],
                  means["Net Inflow"],
                  means["Storage_Inflow_interaction"],
                  std["Release"],
                  std["Storage"],
                  std["Net Inflow"],
                  std["Storage_Inflow_interaction"],
                  ts_len, rel_out, sto_out)
    time2 = timer()
    print(f"Time to simul res: {time2 - time1:.5f}")
    II()

if __name__ == "__main__":
    lib = load_library()
    test_tva_res(lib)
