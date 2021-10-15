from os import error
import pandas as pd
from IPython import embed as II
import requests
import json
import time
import re

from requests.api import request

sample_url = "http://www.usbr.gov/gp-bin/webarccsv.pl?parameter=PUER%20FB&syer=2018&smnth=1&sdy=1&eyer=2018&emnth=1&edy=24&format=2"

# parameter=<station_id>
# %20 is a space and it separates stations (parameters) and variables
# FB is a variable, I care about AF for storage in acre-feet, IN for Inflow in CFS, and QD for discharge in cfs
# syer=<start_year>
# smnth=<start_month>
# sdy=<start_day>
# for end the same as above but with e as the beginning
# format=#
# 1 = tsv
# 2 = csv
# 3 = html
# 8 csv with data masks for bad values
# 998877 and 0 are blank.

with open("./data_keys.json", "r") as f:
    data_keys = json.load(f)


locs = pd.read_csv("./hydromet_stations.csv")
locs = locs[locs["Name"].str.contains("(DAM)|(RESERVOIR)")]
parameters = {
    "AF":"Storage_acft", "IN":"Inflow_cfs", "QD":"Release_cfs"
}

discovery_url = "https://www.usbr.gov/gp-bin/arccsv.pl?{site},+"
# request_url = "https://www.usbr.gov/gp-bin/webarccsv.pl?parameter={site}%20{parameters}&syer=1990&smnth=1&sdy=1&eyer=2020&emnth=12&edy=31&format=10"
request_url = "https://www.usbr.gov/gp-bin/arcread.pl?st={site}&by=1990&bm=1&bd=1&ey=2020&em=12&ed=31&pa={parameters}&json=1"

error_patt = re.compile("Software error")

for i, row in locs.iterrows():
    station = row["ID"]
    url = discovery_url.format(site=station)
    results = requests.get(url)
    content = results.content.decode()
    if re.search(error_patt, content):
        continue
    
    fields = content.strip("\n").split(",")
    parms = [i for i in fields if i in data_keys.keys()]
    url = request_url.format(
        site=station,
        parameters="&pa=".join(parms)
    )
    print(f"Getting {', '.join(parms)} for {station}")

    results = requests.get(url)
    try:
        data = json.loads(results.content)
    except json.decoder.JSONDecodeError as e:
        print(f"Error getting data for {station}")
        continue

    df = pd.DataFrame(data["SITE"]["DATA"])
    df.to_json(f"./hydromet_data/{station}.json")
    time.sleep(0.1)
