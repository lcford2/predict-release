import pandas as pd
from IPython import embed as II

ds_res_path = "../results/multi-level-results/for_graps/NaturalOnly-RunOfRiver_filter_ComboFlow_SIx_pre_std_swapped_res_roll7.pickle"

us_res_path = "../results/treed_ml_model/upstream_basic_td3_roll7_new/results.pickle"

ds_res_data = pd.read_pickle(ds_res_path)
us_res_data = pd.read_pickle(us_res_path)

ds_fc = ds_res_data["data"]["forecasted"]["Release_act"].unstack()
us_fc = us_res_data["data"]["forecasted"]["Release_act"].unstack()

ds_fc[us_fc.columns] = us_fc

order = ['Watauga', 'Wilbur', 'SHolston', 'Boone',
         'FtPatrick', 'Cherokee', 'Douglas', 'FtLoudoun',
         'Fontana', 'Norris', 'MeltonH', 'WattsBar',
         'Chatuge', 'Nottely', 'Hiwassee', 'Apalachia',
         'BlueRidge', 'Ocoee3', 'Ocoee1', 'Chikamauga',
         'RacoonMt', 'Nikajack', 'Guntersville', 'TimsFord',
         'Wheeler', 'Wilson', 'Pickwick', 'Kentucky']

rcmt_mean = 3311 * 86400 / 43560 / 1000 # cfs to ft3/day to 1000 acre-ft /day
rcmt_median = 3261 * 86400 / 43560 / 1000

ds_fc = ds_fc / 43560 / 1000 # ft3 / day tp 1000 acre-ft/day

ds_fc["RacoonMt"] = rcmt_median

write_order = ds_fc[order].T
write_array = write_order.values.flatten()
write_string = "\n".join(map(str, write_array))
with open("./graps/forecast_period/decisionvar_details.dat", "w") as f:
    f.write(write_string)
