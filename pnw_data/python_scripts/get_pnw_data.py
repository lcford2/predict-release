import datetime
import pandas as pd
import time

locs = pd.read_csv("pnw_locs.csv", header=None, names=["loc_id", "Desc."])

# units
# ID - Computed Average Reservoir Inflow, cfs
# QD - Average Stream Discharge, cfs
# AF - Reservoir Water Storage, acre-feet

start_date = datetime.datetime(1990, 1, 1)
end_date = datetime.datetime(2020, 12, 31)

# format url
url = "https://www.usbr.gov/pn-bin/daily.pl?station={station}&format=html&year=1990&month=1&day=1&year=2020&month=12&day=31&pcode=id&pcode=qd&pcode=af&pcode=gain"

# locations 
dams = [
    "AGA", "AND", "ARK", "BEU", "BUL", "CCR",
    "CLS", "CSC", "ELD", "EMI", "EMM", "GCL",
    "GLI", "HEN", "HPD", "HYA", "ISL", "JCK",
    "MAN", "MIL", "MIN", "PHL", "RIR", "SCO",
    "THF", "WAR", "WAS", "WLD", "WOD",
]


res = [
    "AMF", "CRA", "DRW", "HAY", "HGH", "MCK",
    "OCH", "PAL", "POT", "PRV", "PRV", "UNY",
    "WIC"
]

dam_locs = locs[locs.loc_id.isin(res+dams)]

for i, row in dam_locs.iterrows():
    dam = row.loc_id
    description = row["Desc."]
    
    dam_url = url.format(station=dam.lower())
    tables = pd.read_html(dam_url)
    df = tables[0]
    df = df.dropna(how="all")
    dcolumns = df.dropna(how="all", axis=1).columns
    print(f"Station: {dam} has data for {','.join(dcolumns)}")
    if df.empty:
        print(f"Station: {dam}, Desc: {description} does not have all data.")
    else:
        df = df.rename(columns=
            {
                f"{dam.lower()}_{vid}":var for vid, var in {
                    "id":"Inflow_cfs", "qd":"Release_cfs", "af":"Storage_acft", "gain":"Gain_cfs"
                }.items()
            }
        )
        df.to_json(f"dam_data/{dam}_new.json")
    time.sleep(0.1)