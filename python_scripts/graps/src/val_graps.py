import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import embed as II
import pickle

plt.style.use("ggplot")
sns.set_context("talk")

def abline(intercept, slope, ax=None, **kwargs):
    if not ax:
        ax = plt.gca()
    x_values = np.array(ax.get_xlim())
    y_values = intercept + slope * x_values
    ax.plot(x_values, y_values, "--", **kwargs)

mb = pd.read_csv("../forecast_period/forecast_period_output/mass_balance_vars.out",
                delim_whitespace=True, header=None)
inf = pd.read_csv("../forecast_period/forecast_period_output/res_inflow_breakdown.out",
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

tva = pd.read_pickle("../../../pickles/tva_dam_data.pickle")
tva["Storage_pre"] = tva.groupby(tva.index.get_level_values(1))["Storage"].shift(1)
tva["Storage_pre"] *= 86400 / 43560
tva["Net Inflow"] = tva["Net Inflow"] / 43560 / 1000 * 86400


dates = pd.date_range("2010-01-01", "2015-12-31")

with open("../../../results/treed_ml_model/upstream_basic_td3_roll7_new/results.pickle", "rb") as f:
    us_results = pickle.load(f)

with open("../../../results/multi-level-results/for_graps/NaturalOnly-RunOfRiver_filter_ComboFlow_SIx_pre_std_swapped_res_roll7.pickle", "rb") as f:
    ds_results = pickle.load(f)

idx = pd.IndexSlice
ds_fc = ds_results["data"]["forecasted"]
us_fc = us_results["data"]["forecasted"]

# wat_fc_rel = fc.loc[idx[dates, "Watauga"], "Release_act"] / 43560 / 1000
# wat_fc_sto = fc.loc[idx[dates, "Watauga"], "Storage_act"] / 43560 / 1000


def make_my_sto(res):
    res_tva = tva.loc[idx[dates, res],:]
    res_tva.index = res_tva.index.get_level_values(0)
    try:
        fc_rel = us_fc.loc[idx[dates, res], "Release_act"] / 43560 / 1000
    except KeyError as e:
        fc_rel = ds_fc.loc[idx[dates, res], "Release_act"] / 43560 / 1000
    fc_rel.index = fc_rel.index.get_level_values(0)


    # fc_sto = fc.loc[idx[dates, res], "Storage_act"] / 43560 / 1000
    my_sto = []
    for i, date in enumerate(dates):
        if i == 0:
            sto_i = res_tva.loc[date, "Storage_pre"] + res_tva.loc[date, "Net Inflow"] - fc_rel.loc[date]
        else:
            sto_i = my_sto[i - 1] + res_tva.loc[date, "Net Inflow"] - fc_rel.loc[date]
        my_sto.append(sto_i)
    return np.array(my_sto)


# II()
# sys.exit()
bad = ["Wilbur", "Boone", "Douglas", "FtLoudoun", "MeltonH", "WattsBar", 
            "Hiwassee", "Apalachia", "Ocoee3", "Chikamauga", "Nikajack", "Guntersville", "Wheeler", "Wilson"]
upstream = ['BlueRidge', 'Chatuge', 'Fontana', 
                'Norris', 'Nottely', 'SHolston', 'TimsFord', 'Watauga']

for res in upstream:
    fig, axes = plt.subplots(2,1,sharex=True,sharey=True,figsize=(20,8.7))
    fig.patch.set_alpha(0.0)
    axes = axes.flatten()
    ax1, ax2 = axes

    ax1.set_title(res)
    ax2.set_title("Post-calculated storage")

    # res = "Fontana"
    my_sto = make_my_sto(res)
    st_cur = mb.loc[mb["res"] == res, "st_cur"]
    fc_sto = us_fc.loc[idx[dates,res], "Storage_act"] / 43560 / 1000

    ax1.plot(dates, st_cur, label="GRAPS")
    ax1.plot(dates, fc_sto, label="FC")

    ax2.plot(dates, st_cur, label="GRAPS")
    ax2.plot(dates, my_sto, label="Post Calc.")

    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")

    ax2.set_xlabel("Date")

    ax1.set_ylabel("Storage [1000 acre-ft]")
    ax2.set_ylabel("Storage [1000 acre-ft]")

    plt.show()
