import sys
import pathlib
import pickle
import calendar
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from check_simul import plot_res_scores, plot_time_series

plt.style.use("ggplot")
sns.set_context("talk")

def load_results(method="nelder-mead"):
    if method == "nelder-mead":
       with open("../results/simul_model/multi_trial_nelder-mead_group_specific.pickle", "rb") as f:
            results = pickle.load(f)
    else:
        with open("../results/simul_model/multi_trial.pickle", "rb") as f:
            results = pickle.load(f)
    with open("../results/simul_model/best_ensem_results_group_specific.pickle", "rb") as f:
        ts_results = pickle.load(f)
    return results, ts_results

def extract_final_error(results):
    return [i.fun for i in results]

def extract_final_params(results):
    return [i.x for i in results]

def format_params(params, months=False):
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
    ]
    if months:
        param_order.extend(calendar.month_abbr[1:])

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze and plot results from simulation optimization with random starting parameters"
    )
    parser.add_argument("-m", "--method", action="store", default="nelder-mead",
                        choices=["nelder-mead", "bfgs"], dest="method",
                        help="The optimization method used for the results to be analyzed.")
    parser.add_argument("--months", action="store_true", default=False, dest="months",
                        help="Indicates if monthly intercepts were fit or not.")
    parser.add_argument("-D", "--data-set", action="store", default="test",
                        choices=["train", "test"], dest="data_set",
                        help="Indicate which data set should be plotted.")
    parser.add_argument("-V", "--var", action="store", default="release",
                        choices=["release", "storage"], dest="var",
                        help="What variable should be plotted.")
    return parser.parse_args()

def main():
    args = parse_args()
    results, ts_results = load_results(method=args.method)
    res_groups = pd.read_pickle("../pickles/tva_res_groups.pickle")
    # final_error = extract_final_error(results)
    # final_params = extract_final_params(results)
    # formatted_params = [format_params(i, months=args.months) for i in final_params]
    sns.set_context("paper")
    from IPython import embed as II
    II()
    # sys.exit()
    # plot_time_series(
    #     ts_results[args.data_set][f"{args.var.capitalize()}_act"].unstack(),
    #     ts_results[args.data_set][f"{args.var.capitalize()}_simul"].unstack(),
    #     f"Simulated {args.var.capitalize()}({args.data_set.capitalize()}ing Set:Best Params)",
    #     res_groups
    # )
    # plot_res_scores(
    #     ts_results[args.data_set][f"{args.var.capitalize()}_act"].unstack(),
    #     ts_results[args.data_set][f"{args.var.capitalize()}_simul"].unstack(),
    #     f"{args.var.capitalize()} Scores",
    #     res_groups)

if __name__ == "__main__":
    main()
