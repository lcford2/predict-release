import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed as II


def abline(intercept, slope, ax=None, **kwargs):
    if not ax:
        ax = plt.gca()
    x_values = np.array(ax.get_xlim())
    y_values = intercept + slope * x_values
    ax.plot(x_values, y_values, "--", **kwargs)

mb = pd.read_csv("../model_val/model_val_output/mass_balance_vars.out",
                delim_whitespace=True, header=None)
inf = pd.read_csv("../model_val/model_val_output/res_inflow_breakdown.out",
                delim_whitespace=True, header=None)
 
mb = mb.drop(1, axis=1)
inf = inf.drop(1, axis=1)

mb = mb.rename(columns={0: "res", 2: "inf", 3: "st_pre", 4: "def", 5: "spill", 6: "rel",
                        7: "evap", 8: "st_cur", 9: "chk", 10: "stflag", 11: "lbound", 12: "ubound"})

inf = inf.rename(columns={0: "res", 2: "uncnt_inf", 3: "cnt_inf"})

obs_netinf = pd.read_pickle("../../../pickles/observed_net_inflow.pickle")
obs_unct   = pd.read_pickle("../../../pickles/new_observed_uncont_inflow.pickle")
obs_rel    = pd.read_pickle("../../../pickles/observed_turbine.pickle")
obs_totrel = pd.read_pickle("../../../pickles/observed_total.pickle")

dates = pd.date_range("2010-01-01", "2015-12-31")

II()
