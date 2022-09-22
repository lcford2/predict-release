import json
import re
from collections import defaultdict
import pandas as pd


DATA_REQUEST_URL = "https://www.usbr.gov/uc/water/hydrodata/\
                    reservoir_data/{site_id}/csv/{var_id}.csv".replace(" ", "")


def load_meta():
    with open("./metadata/var_map.json", "r") as f:
        var_map = json.load(f)

    with open("./metadata/site_map.json", "r") as f:
        site_map = json.load(f)

    return var_map, site_map


def get_var_ids(var_map: dict, varnames: list) -> list():
    return [var_map[i]["id"] for i in varnames]


def get_site_with_vars(site_map: dict, req_var_ids: list) -> dict:
    output = {}
    for res, info in site_map.items():
        vids = info["vars"]
        has_vars = True
        for i in req_var_ids:
            r = re.compile(i)
            matches = list(filter(r.match, vids))
            if len(matches) == 0:
                has_vars = False
                break
        if has_vars:
            output[res] = info
    return output


def retrieve_data(avail_site):
    output = defaultdict(list)
    bad_requests = []
    for site, info in avail_site.items():
        for vid in info["vars"]:
            try:
                df = pd.read_csv(
                    DATA_REQUEST_URL.format(
                        site_id=info["id"],
                        var_id=vid
                    ))
                output[site].append(df)
            except Exception:
                print("ERROR: {site} {vid}")
                bad_requests.append((site, vid))
    return output, bad_requests


def main():
    var_map, site_map = load_meta()
    lookup_vars = [
        "inflow",
        "inflow volume",
        "total release",
        "release volume",
        "storage"
    ]
    lookup_ids = get_var_ids(var_map, lookup_vars)
    match_ids = [
        f"({lookup_ids[0]}|{lookup_ids[1]})|\
          ({lookup_ids[2]}|{lookup_ids[3]})".replace(" ", ""),
        lookup_ids[4]
    ]
    avail_site = get_site_with_vars(site_map, match_ids)
    data, bad_requests = retrieve_data(avail_site)
    from IPython import embed as II
    II()


if __name__ == "__main__":
    main()
