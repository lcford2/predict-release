import glob
import pandas as pd
import numpy as np
import multiprocessing as mp
from IPython import embed as II
from locale import atof, setlocale, LC_NUMERIC

setlocale(LC_NUMERIC, "")

ORDER = ["site_name", "elev_mp", "elev_fc", "sto_mp", "sto_fc",
        "elev", "elev_delta", "storage", "inflow", 
        "release", "mp_sto_%", "fc_sto", "fc_sto_%"]

def my_atof(x):
    try:
        return atof(x)
    except (ValueError, AttributeError) as e:
        return "-999999"

def fix_df(df):
    odf = df.copy()
    II()
    df = df[df.isna().sum(axis=1) < 2]
    if df.dropna(axis=1, how="all").dropna().size > 0:
        df = df.dropna(axis=1, how="all").dropna()
    df = df.reset_index().drop("index", axis=1)
    columns = df.columns
    output = pd.DataFrame()
    for i, col in enumerate(columns):
        ser = df[col]
        output_size = output.shape[1]
        try:
            split = ser.str.split(expand=True)
            split_size = split.shape[1]
        except AttributeError as e:
            output[ORDER[output_size]] = ser
            continue
        if i == 0:
            if split_size == 5:
                name_split = ser.str.split()
                name_mp_fc = []
                for i in name_split:
                    if len(i) == 3:
                        name_mp_fc.append(i)
                    else:
                        size = len(i)
                        name = " ".join(i[:size-2])
                        name_mp_fc.append([name] + i[-2:])
                output[["site_name", "elev_mp", "elev_fc"]] = name_mp_fc
            else:
                output["site_name"] = ser
        else:
            split = split.dropna(axis=1, thresh=split.shape[0] - int(0.95 * split.shape[0]))
            for col in split.columns:
                # basically if there is only one value in the columne (e.g. "M") we dont want it
                if split[col].value_counts().shape[0] == 1:
                    split = split.drop(col, axis=1)
            split_size = split.shape[1]
            if split_size > 1:
                my_cols = ORDER[output_size:output_size + split_size]
                output[my_cols] = split
            else:
                output[output_size] = ser
    output = output.replace("--", "-999999")
    output = output.replace("M", "-999999")
    convert_columns = output.columns[1:] 
    for col in convert_columns:
        if output[col].dtype == "object":
            output[col] = output[col].apply(my_atof)
    return output

def load_fix_store(file):
    try:
        df = pd.read_csv(file)
    except pd.errors.EmptyDataError as e:
        print(f"No data in file {file}")
        return
    new_df = fix_df(df)
    filename = file.split("/")[-1]
    ofile = f"./acoe_data/pdf_to_csv_fixed/{filename}"
    new_df.to_csv(ofile)

def main():
    files = glob.glob("./acoe_data/pdf_to_csv/*.csv")
    pool = mp.Pool(mp.cpu_count())
    # results = pool.map(load_fix_store, files)
    files = ["./acoe_data/pdf_to_csv/2015_03_27.csv"]
    for file in files:
        load_fix_store(file)
    pool.close()

if __name__ == "__main__":
    main() 
