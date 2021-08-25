import sys
import pickle
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
from IPython import embed as II

from utils.helper_functions import read_tva_data

tva = read_tva_data(just_load=True)
PICKLE_DIR = pathlib.Path("G:/My Drive/PHD/SRY_curves/data/pickles")
RESULTS_DIR = pathlib.Path("G:/My Drive/PHD/SRY_curves/data/results")

rts = pd.read_pickle(PICKLE_DIR / "tva_res_times.pickle")
mstl = pd.read_pickle(PICKLE_DIR / "tva_mean_st_level.pickle")

with open(RESULTS_DIR / "results" / "synthesis" / "spatial_model" / "res_slopes.pickle", "rb") as f:
    sp_results = pickle.load(f)

res_cats = tva[["RunOfRiver", "NaturalOnly", "PrimaryType"]].groupby(tva.index.get_level_values(1)).mean()

res_cats["rts"] = rts
res_cats["mstl"] = mstl

coefs = sp_results["coefs"].T.rename(columns={"Net Inflow":"Inflow"})

data = coefs.join(res_cats)

models = {}
params = coefs.columns[:8]
labels = []
for param in params:
    formula = f"{param} ~ C(RunOfRiver) + C(NaturalOnly) + C(PrimaryType) + rts + mstl"
    md = smf.ols(formula=formula, data=data)
    mdf = md.fit()
    models[param] = (md,mdf)
    labels = mdf.params.index

cols = pd.MultiIndex.from_product([params, ["coef", "pval"]])
coefs = pd.DataFrame(index=labels, columns=cols)
idx = pd.IndexSlice
for param, ident in cols:
    if ident == "coef":
        coefs.loc[:,idx[param, ident]] = models[param][1].params
    else:
        coefs.loc[:,idx[param, ident]] = models[param][1].pvalues
II()