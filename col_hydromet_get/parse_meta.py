import pandas as pd
import json


def read_meta():
    return pd.read_csv("./reservoir_meta.csv")


def map_site_and_var_name_id(meta):
    sites = {}
    var_names = {}
    for i, row in meta.iterrows():
        sid = row["site_id"]
        sname = row["site_metadata.site_common_name"]
        vid = row["datatype_id"]
        vname = row["datatype_metadata.datatype_common_name"]
        vunit = row["datatype_metadata.unit_common_name"]
        sites[sname] = sid
        var_names[vname] = {"id": vid, "unit": vunit}
    return sites, var_names


def get_available_vars(meta, site_map):
    output = {}
    for sname, sid in site_map.items():
        svars = meta.loc[meta["site_id"] == sid, "datatype_id"]
        output[sname] = {
            "id": sid,
            # convert int64 to str for json serialization
            "vars": list(svars.astype(str).values)
        }
    return output


def write_pretty_json(dct, file):
    with open(file, "w") as f:
        json.dump(dct, f, indent=4, sort_keys=True)


def main():
    df = read_meta()
    site_map, var_map = map_site_and_var_name_id(df)
    site_vars = get_available_vars(df, site_map)
    write_pretty_json(var_map, "./metadata/var_map.json")
    write_pretty_json(site_vars, "./metadata/site_map.json")


if __name__ == "__main__":
    main()
