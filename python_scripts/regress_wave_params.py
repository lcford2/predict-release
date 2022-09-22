import sys
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from IPython import embed as II

PICKLE_DIR = pathlib.Path("G:/My Drive/PHD/SRY_curves/data/pickles")
RESULTS_DIR = pathlib.Path("G:/My Drive/PHD/SRY_curves/data/results")

def load_rts():
    return pd.read_pickle(PICKLE_DIR / "tva_res_times.pickle")

def load_mstl():
    return pd.read_pickle(PICKLE_DIR / "tva_mean_st_level.pickle")

def load_mrel():
    return pd.read_pickle(PICKLE_DIR / "release_longterm_means.pickle") * 86400 / 43560 / 1000

def load_wave_results():
    return pd.read_pickle(RESULTS_DIR / "synthesis" / "spatial_model" / "sin_results.pickle")

def fit_wave_params(x_var = "RT"):
    func_map = {
        "RT": load_rts,
        "MStL": load_mstl,
        "MRel": load_mrel
    }

    data_loader = func_map.get(x_var)

    if data_loader:
        X = data_loader().astype(float)
    else:
        print(f"Provided `x_var` ({x_var}) is not valid. Please choose from RT, MStL, or MRel.")
        sys.exit()

    wave_params = load_wave_results().astype(float)
    print(wave_params.describe())
    params = ["b0", "b1", "omega"]
    models = {}
    wave_params[x_var] = X
    for param in params:
        md = smf.ols(f"{param} ~ {x_var}", data=wave_params)
        mdf = md.fit()
        models[param] = mdf

    for param, mdf in models.items():
        print(f"\n{param}\n")
        print(mdf.summary())
    
def get_corrs():
    rts = load_rts().astype(float)
    mstl = load_mstl().astype(float)
    mrel = load_mrel().astype(float)
    wave = load_wave_results().astype(float)
    wave["rts"] = rts
    wave["mstl"] = mstl
    wave["mrel"] = mrel
    print(wave.corr().to_markdown(floatfmt="0.2f", tablefmt="github"))

if __name__ == "__main__":
    fit_wave_params(x_var = "MRel")
    # get_corrs()