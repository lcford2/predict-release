import json
import glob
import subprocess

files = glob.glob("./dam_data/*data/*.json")
names = [i.split("/")[-1].split(".")[0] for i in files]

with open("dam_abbrs.json", "r") as f:
    meta = json.load(f)

for name in names:
    if name not in meta.keys():
        # print(f"{name} is not in dam_abbrs.json")
        output = subprocess.run([
            "grep", name, "pnw_locs.csv"
        ], capture_output=True)
        print(output.stdout)