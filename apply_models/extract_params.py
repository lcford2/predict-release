import pickle
import json
import sys
import os
import pandas as pd

basin = sys.argv[1]

tree = f"../results/basin_eval/{basin}/treed_model/results.pickle"
simp = f"../results/basin_eval/{basin}/simple_model/results.pickle"

output = {}
if os.path.exists(tree):
    with open(tree, "rb") as f:
        results = pickle.load(f)

    params = results["coefs"]
    groups = results["data"]["groups"]
    rel_pre = results["data"]["X_train"]["release_pre"]
    tree_breaks = pd.DataFrame({"release_pre":rel_pre, "groups":groups})
    tree_breaks = tree_breaks.groupby("groups").max().values.flatten()[:-1]
    output["high_rt"] = {
        "leaves": list(tree_breaks),
        "params": params.T.values.tolist(),
    }
else:
    output["high_rt"] = {"leaves":[], "params":[[]]}

if os.path.exists(simp):
    with open(simp, "rb") as f:
        results = pickle.load(f)

    params = results["coefs"]
    for column in ["ror", "low_rt"]:
        if column in params.columns:
            output[column] = {"params":params[column].values.tolist()}
        else:
            output[column] = {"params":[]}

with open(f"./basin_params/{basin}.json", "w") as f:
    json.dump(output, f, indent=4, separators=(",", ": "))
