from numpy import number
import pandas as pd
import re
import io
from calendar import monthrange

def load_data(file):
    with open(file, "r") as f:
        data = f.readlines()
    return data

def strip_empty_lines(data):
    # first and last line are html tags
    data = data[1:-1]
    # get rid of blank lines
    return list(filter(lambda x: x != "\n", data))

def split_tables(data):
    patt = re.compile("BUREAU OF RECLAMATION")
    # each table starts with a header that matches the above patter
    starts = [i for i, j in enumerate(data) if re.search(patt, j)]
    # since the whitespace is stripped, the end of the table will be one line above the start of the next one
    ends = [i - 1 for i in starts[1:]]+[len(data)]
    return [data[s + 5:e + 1] for s, e in zip(starts, ends)]

def get_headers():
    columns1 = [
        "Day", "GlenCanyon_rel_cfs", "GlenCanyon_cum_rel_AF", "Hoover_elev", "Hoover_rel_cfs", "Hoover_rel_AF",
        "Hoover_cum_rel_AF", "Hoover_gen_MWh", "Hoover_gen_rate_KWh/AF", "Davis_elev",
        "Davis_rel_cfs", "Davis_rel_AF", "Davis_cum_rel_AF", "Davis_gen_MWh", "MWD_diver_AF",
        "MWD_cum_diver_AF", "Parker_elev", "Parker_rel_cfs", "Parker_rel_AF",
        "Parker_cum_rel_AF", "Parker_gen_MWh", "CAP_diver_AF", "CAP_cum_diver_AF"
    ]
    columns2 = [
        "Day", "Stor_total", "Stor_avail", "Hoover_elev", "Hoover_stor_KAF", "Hoover_rel_cfs",
        "Davis_elev", "Davis_stor_KAF", "Davis_rel_cfs",
        "Parker_elev", "Parker_stor_KAF", "Parker_rel_cfs",
        "SenWash_elev", "SenWash_stor_KAF", "SenWash_pump_cfs", "SenWash_rel_cfs",
    ]
    columns3 = [
        "Day", "GlenCanyon_rel_cfs", "Hoover_rel_cfs", "Hoover_stor_KAF", "Hoover_del_stor_KAF",
        "Hoover_pump_cfs", "Hoover_evap_cfs", "Hoover_cbs_cfs", "Hoover_loss_cfs", "Hoover_cum_loss_AF", 
        "Hoover_inflow_cfs"
    ]
    return columns1, columns2, columns3

def trim_columns(df):
    keep = [
        "GlenCanyon_rel_cfs", "Hoover_rel_cfs", "Davis_rel_cfs", "MWD_diver_AF", "Parker_rel_cfs", "CAP_diver_AF",
        "Hoover_stor_KAF", "Davis_stor_KAF", "Parker_stor_KAF", "SenWash_stor_KAF", "SenWash_rel_cfs",
        "Hoover_inflow_cfs"
    ]
    return df.loc[:, keep]

def number_days_in_month(year, month):
    return monthrange(year, month)[1]

def make_dataframe(table, columns, month, year):
    # table begins with a line of -----
    start_patt = re.compile("-+")
    # to match end of table, i am looking for the line that starts with the last day of the month
    # this will obviously match anything from 28 to 31, so when i get the index I make sure the grab the very last one. 
    # end_patt = re.compile("^ *( 28 )|( 29 )|( 30 )|( 31 )")

    start = [i for i,j in enumerate(table) if re.search(start_patt, j)][0]
    end = start + number_days_in_month(year, month)
    table = table[start+1:end+1]
    return pd.read_csv(io.StringIO("".join(table)), delim_whitespace=True, names=columns, index_col=False)

def fix_df_dates(df, year, month):
    try:
        df["Day"] = pd.to_datetime(
            [f"{i:02}-{month:02}-{year}" for i in df["Day"]]
        , format="%d-%m-%Y")
        df = df.set_index("Day")
    except ValueError as e:
        from IPython import embed as II
        II()
    return df

def combine_dfs(dfs):
    df = dfs[0]
    for i in dfs[1:]:
        df[i.columns] = i
    return df


def main():
    big_df = pd.DataFrame()
    for year in range(2000, 2021):
        for month in range(1,13):
            file_name = f"{month:02}_{year}.out"
            file = f"./raw_data/{file_name}"
            data = load_data(file)
            data = strip_empty_lines(data)
            tables = split_tables(data)
            tables = tables[:2] + tables[-1:]
            headers = get_headers()
            dfs = [make_dataframe(table, header, month, year) for table, header in zip(tables, headers)]
            dfs = [fix_df_dates(df, year, month) for df in dfs]
            df = combine_dfs(dfs)
            big_df = df if big_df.empty else big_df.append(df)
    from IPython import embed as II
    II()

if __name__ == "__main__":
    main()
