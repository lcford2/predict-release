import inspect
import json
import pandas as pd
from dateutil.parser import parse as date_parser
from IPython import embed as II

## Source https://www.usbr.gov/pn/hydromet/arcread.html
with open("./pac_nw_hydromet_stations.json", "r") as f:
    stations = json.load(f)
stations = {value: key for key, value in stations.items()}
with open("./column_meaning.json", "r") as f:
    column_meaning = json.load(f)


def format_url(location, start_date, end_date):
    codes = "&".join([f"pcode={i}" for i in column_meaning.keys()])
    url = """https://www.usbr.gov/pn-bin/
            daily.pl?station={station:s}&format=html&
            year={syear:d}&month={smonth:d}&day={sday:d}&
            year={eyear:d}&month={emonth:d}&day={eday:d}&
            &{codes:s}
        """
    
    if len(location) > 5:
        location = stations.get(location, None)
        if not location:
            raise KeyError("Provided location not valid.")
    else:
        if location not in set(stations.values()):
            raise KeyError("Provided location not valid.")
    location = location.lower()
    sdate = date_parser(start_date)
    edate = date_parser(end_date)
    url = url.format(
        station=location,
        syear=sdate.year,
        smonth=sdate.month,
        sday=sdate.day,
        eyear=edate.year,
        emonth=edate.month,
        eday=edate.day,
        codes=codes
    )
    url = inspect.cleandoc(url).replace("\n", "")
    return url

def rename_columns(columns):
    new_columns = {}
    for column in columns:
        split = column.split("_")
        if len(split) == 2:
            meaning = column_meaning.get(split[1])
            if meaning:
                rename = meaning["rename"]
                units = meaning["units"]
                new_name = "_".join((rename, units))
                new_columns[column] = new_name
    return new_columns

def get_df_from_url(url):
    tables = pd.read_html(url)
    df = tables[0]
    df = df.set_index("DateTime")
    new_columns = rename_columns(df.columns)
    df = df.rename(columns=new_columns)
    df = df.dropna(axis=1, how="all")
    return df


url = format_url("PAL",
                 "2010-01-01", "2019-12-31")
df = get_df_from_url(url)
df.to_json("./dams/palisades.json")
II()
