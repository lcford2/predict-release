import pandas as pd
import json
from IPython import embed as II


def load_res_data():
    res_data_files = [
        "../upper_colorado_data/model_ready_data/upper_col_data_net_inflow.csv",
        "../pnw_data/model_ready_data/pnw_data_net_inflow.csv",
        "../lower_col_data/model_ready_data/lower_col_data_net_inflow.csv",
        "../missouri_data/model_ready_data/missouri_data_net_inflow.csv",
        "../csv/tva_model_ready_data.csv",
    ]

    df = pd.concat([pd.read_csv(i) for i in res_data_files])
    return df


def load_nid():
    return pd.read_csv('./all_dams_data.csv', low_memory=False)


def get_name_replacements():
    with open("../pnw_data/dam_names.json", "r") as f:
        pnw = json.load(f)

    with open("../missouri_data/dam_names.json", "r") as f:
        missouri = json.load(f)

    tva = {}
    with open("../python_scripts/actual_names.csv", "r") as f:
        for line in f.readlines():
            line = line.strip("\n\r")
            key, value = line.split(",")
            tva[key] = value
    return pnw | tva | missouri


def get_basins():
    rbasins = pd.read_pickle("../pickles/res_basin_map.pickle")
    rename = {"upper_col": "colorado", "lower_col": "colorado", "pnw": "columbia", "tva": "tennessee"}
    rbasins = rbasins.replace(rename)
    rbasins = rbasins.str.capitalize()
    return rbasins

def get_res_nid_indexes(df, nid, rbasins):
    resers = df["site_name"].unique().tolist()
    indices = {}
    nofind = []
    for res in resers:
        matches = nid[nid["Dam_Name"].str.contains(res, na=False, case=False)]
        if matches.shape[0] == 1:
            question = f"Is index {matches.index[0]} correct? (y/n) "
            rtype = "yn"
        else:
            question = f"What index is correct for {res} in {rbasins.loc[res, 'basin']}? "
            rtype = "num"
        print(matches)
        resp = input(question)
        if rtype == "num":
            try:
                resp = int(resp)
                indices[res] = resp
            except ValueError as e:
                nofind.append(res)
        elif rtype == "yn":
            if resp.lower() == "y":
                indices[res] = matches.index[0]
            else:
                nofind.append(res)

    with open("nofind.txt", "w") as f:
        f.writelines([f"{i}\n" for i in nofind])
    with open("found.csv", "w") as f:
        f.writelines([f"{i},{j}\n" for i, j in indices.items()])  


def main():
    df = load_res_data()
    nid = load_nid()
    renames = get_name_replacements()
    rbasins = get_basins()
    rbasins = pd.DataFrame({"site_name": r, "basin": rbasins[r]} for r in df["site_name"].unique())
    df["site_name"] = df["site_name"].replace(renames)
    rbasins["site_name"] = rbasins["site_name"].replace(renames)
    rbasins = rbasins.set_index("site_name")
    # get_res_nid_indexes(df, nid, rbasins)
    II()
    

if __name__ == "__main__":
    main()
