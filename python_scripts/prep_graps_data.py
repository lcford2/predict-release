import pandas as pd
import datetime
import re
from helper_functions import read_tva_data
from IPython import embed as II


DS_RES_PATH = "../results/multi-level-results/for_graps/NaturalOnly-RunOfRiver_filter_ComboFlow_SIx_pre_std_swapped_res_roll7.pickle"
US_RES_PATH = "../results/treed_ml_model/upstream_basic_td3_roll7_new/results.pickle"

def read_results():
    ds_res_data = pd.read_pickle(DS_RES_PATH)
    us_res_data = pd.read_pickle(US_RES_PATH)
    # for using forecasted data
    ds_fc = ds_res_data["data"]["forecasted"]["Release_act"].unstack()
    us_fc = us_res_data["data"]["forecasted"]["Release_act"].unstack()
    ds_fc[us_fc.columns] = us_fc
    return ds_fc

def read_actual(column=None):
    df = read_tva_data(just_load=True)
    if column:
        df = df.loc[:,column]
        df = df.unstack()
    return df

def trim_dfs(dfs, start_date, end_date):
    dates = pd.date_range(start_date, end_date)
    for i, df in enumerate(dfs):
        dfs[i] = df.loc[dates,:]
    return dfs

def get_rcmt_data():
    return pd.read_pickle("../pickles/rcmt_data.pickle")

def write_dec_vars(release, order, path):
    write_order = release[order].T
    write_array = write_order.values.flatten()
    write_string = "\n".join(map(str, write_array))
    with open(f"{path}/decisionvar_details.dat", "w") as f:
        f.write(write_string)

def change_initial_storage(stdf, date, path):
    with open(f"{path}/reservoir_details.dat", "r") as f:
        res_dets = f.readlines()
    
    edit_positions = dict()
    for i, line in enumerate(res_dets):
        m = re.match(r".*Reservoir", line)
        if m:
            edit_positions[m.group(0).split()[0]] = i + 3
    pre_r_vals = stdf.loc[date]

    for name, i in edit_positions.items():
        values = res_dets[i].split()
        values[-1] = f"{pre_r_vals[name]:.3f}"
        res_dets[i] = "\t".join(values) + "\n"
    
    with open(f"{path}/reservoir_details.dat", "w") as f:
        f.writelines(res_dets)

def change_ntime(ntime, path):
    with open(f"{path}/input.dat", "r") as f:
        lines = f.readlines()
    fline = lines[0].split()
    fline[0] = str(ntime)
    lines[0] = "\t".join(fline) + "\n"
    with open(f"{path}/input.dat", "w") as f:
        f.writelines(lines)

def write_inflow_files(start, end, path):
    inf = pd.read_pickle("../pickles/new_observed_uncont_inflow.pickle")
    inf["RacoonMt"] = 0.0
    inf, = trim_dfs([inf], start, end)
    inf *= 86400 / 43560 / 1000
    file_map = pd.read_csv("./inflow_file_map.csv",header=0,index_col=0)
    for reservoir in inf.columns:
        file = file_map.loc[reservoir,"Inflow_File"]
        flows = inf[reservoir].values
        with open(f"{path}/InflowData/{file}", "w") as f:
            string = "\t".join(map(str, flows)) + "\n"
            f.write(string)

def write_path_dat(start, path):
    import calendar
    start_month = int(start.split("-")[1])
    start_abbr = calendar.month_abbr[start_month].lower()
    data_path = path.split("/")[-1]
    input_path = f"../{data_path}/"
    if start_month == 1:
        output_folder = f"{data_path}_output"
    else:
        output_folder = f"{data_path}_output_{start_abbr}"
    output_path = f"../{data_path}/{output_folder}/"
    # remove leading zeros
    start_date_line = " ".join([f"{int(i)}" for i in start.split("-")])
    with open(f"./graps/path.dat", "w") as f:
        f.write(f"{input_path}\n")
        f.write(f"{output_path}\n")
        f.write(f"{start_date_line}\n")
    

def main():
    start_date = "2010-01-01"
    end_date = "2015-12-31"

    order = ['Watauga', 'Wilbur', 'SHolston', 'Boone',
             'FtPatrick', 'Cherokee', 'Douglas', 'FtLoudoun',
             'Fontana', 'Norris', 'MeltonH', 'WattsBar',
             'Chatuge', 'Nottely', 'Hiwassee', 'Apalachia',
             'BlueRidge', 'Ocoee3', 'Ocoee1', 'Chikamauga',
             'RacoonMt', 'Nikajack', 'Guntersville', 'TimsFord',
             'Wheeler', 'Wilson', 'Pickwick', 'Kentucky', "RcMt_intake"]

    release = read_results() # ft3 / day
    tva_dat = read_actual()
    storage = tva_dat["Storage_pre"].unstack()
    # release = tva_dat["Release"].unstack()
    rcmt = get_rcmt_data() # cfs
    storage["RacoonMt"] = rcmt["Sto"].shift(1)
    release["RacoonMt"] = rcmt["TurbQ"] * 86400 # cfs to cf/day
    release["RcMt_intake"] = rcmt["Canal"] * 86400 # cfs to cf/day

    # trim dfs to my date range
    release, storage, rcmt = trim_dfs(
        [release, storage, rcmt], start_date, end_date)

    release = release / 43560 / 1000 # ft3 / day tp 1000 acre-ft/day
    storage = storage / 43560 / 1000 # ft3 to 1000 acre-ft

    path = "./graps/forecast_period"
    change_initial_storage(storage, start_date, path)
    write_dec_vars(release, order, path)
    change_ntime(release.shape[0], path)
    write_inflow_files(start_date, end_date, path)
    write_path_dat(start_date, path)

if __name__ == "__main__":
    main()
