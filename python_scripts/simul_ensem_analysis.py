import sys
import pathlib
import pickle
import calendar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")
sns.set_context("talk")

def load_results():
    with open("../results/simul_model/multi_trial.pickle", "rb") as f:
        results = pickle.load(f)

    return results

def extract_final_error(results):
    return [i.fun for i in results]

def extract_final_params(results):
    return [i.x for i in results]

def format_params(params):
    group_order = ["HRT", "LRT", "ROR"]
    param_order = [
        "const",
        "Release_pre",
        "Storage_pre",
        "Net Inflow",
        "Storage_Inflow_Interaction",
        "Release_roll7",
        "Storage_roll7",
        "Inflow_roll7",
        *calendar.month_abbr[1:]
    ]

    nprms = len(param_order)
    coefs = {}

    for i, group in enumerate(group_order):
        coefs[group] = params[i*nprms:(i+1)*nprms]

    return pd.DataFrame(coefs, index=param_order)

def get_param_metric(params, metric="mean"):
    return getattr(np.array(params), metric)(axis=0)

def melt_and_merge_params(params):
    new_params = []
    for i, p in enumerate(params):
        np = p.reset_index().rename(columns={"index":"param"}).melt(
                id_vars=["param"], var_name="group"
        )
        np["trial"] = i
        new_params.append(np)
    return pd.concat(new_params)


def main():
    results = load_results()
    final_error = extract_final_error(results)
    final_params = extract_final_params(results)
    formatted_params = [format_params(i) for i in final_params]
    from IPython import embed as II
    II()

if __name__ == "__main__":
    main()
