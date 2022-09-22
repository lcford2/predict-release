import glob
import json
from datetime import datetime
import pandas as pd
import numpy as np
from IPython import embed as II

ORDER = ["site_name", "elev_mp", "elev_fc", "sto_mp", "sto_fc",
        "elev", "elev_delta", "storage", "inflow", 
        "release", "mp_sto_%", "fc_sto", "fc_sto_%"]

with open("./res_name_swap.json", "r") as f:
    names = json.load(f)

def main():
    files = glob.glob("./acoe_data/pdf_to_csv_fixed/*.csv")
    
    output = []
    for file in files:
        date = file.split("/")[-1].split(".")[0]
        df = pd.read_csv(file, index_col=0)
        df = df.replace(-999999.0, np.nan)
        df = df.dropna(axis=0, thresh=3)
        if df.empty:
            print(f"No data for {date}")
            continue
        try:
            df.columns = ORDER
        except ValueError as e:
            if df.shape[1] == 11:
                df = df.rename(columns={j:i for i,j in enumerate(df.columns)})
                df["site_name"] = df[0].str.slice(0,-14)
                try:
                    df[["elev_mp", "elev_fc"]] = df[0].str.slice(-13,).str.split(expand=True)
                except ValueError as e:
                    print("Value 2")
                    II()
                    sys.exit()
                df = df[["site_name", "elev_mp", "elev_fc", *range(1, 11)]]
                df.columns = ORDER
            else:
                II()
                sys.exit()
        df["datetime"] = pd.to_datetime(date, format="%Y_%m_%d")
        df = df[df["site_name"] != "System Totals"]
        df["site_name"] = df["site_name"].apply(lambda x: names.get(x, x))
        df = df.set_index(["site_name", "datetime"])

        small_df = df.loc[:, ["storage", "inflow", "release"]]
        small_df = small_df.dropna(how="all")

        output.append(small_df)

    big_df = pd.concat(output, axis=0, ignore_index=False) 
    II()

if __name__ == "__main__":
    main()
