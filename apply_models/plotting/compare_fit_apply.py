import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Patch as mpatch
from matplotlib.lines import Line2D as mline
import seaborn as sns

plt.style.use("seaborn")
sns.set_context("talk")
style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def load_apply_results():
    results = {
        i:pd.read_pickle(f"../basin_output_no_ints/{i}.pickle") for i in [
            "lower_col", "upper_col", "missouri", "pnw"
        ]
    }
    return results

def get_res_metrics(results):
    output = pd.DataFrame(columns=["site_name", "basin", "group", "metric", "value"])
    for basin, values in results.items():
        for group, data in values.items():
            metrics = data["metrics"].reset_index().rename(
                columns={"index":"site_name"}
            )
            try:
                # need to convert from cudf to pddf if needed
                metrics = metrics.to_pandas()
            except AttributeError as e:
                pass
            metrics = metrics.melt(id_vars=["site_name"], var_name="metric")
            metrics["basin"] = basin
            metrics["group"] = group
            output = pd.concat([output, metrics])
    return output

def get_mean_release(results):
    output = pd.Series(dtype=np.float64)
    for basin, values in results.items():
        for group, data in values.items():
            act = data["eval_data"]["actual"]
            means = act.groupby(act.index.get_level_values(0)).mean()
            try:
                # need to convert from cudf to pddf if needed
                means = means.to_pandas()
            except AttributeError as e:
                pass
            output = output.append(means)
    return output

def get_fit_metrics(copy_df):
    tree = {
        i:pd.read_pickle(f"../../results/basin_eval/{i}/treed_model/results.pickle")
        for i in ["upper_col", "lower_col", "tva", "missouri", "pnw"]
    }
    simp = {}
    for i in ["upper_col", "lower_col", "tva", "missouri", "pnw"]:
        try:
            df = pd.read_pickle(f"../../results/basin_eval/{i}/simple_model/results.pickle")
            simp[i] = df
        except FileNotFoundError as e:
            pass
    
    tree_metrics = {i:j["res_scores"] for i,j in tree.items()}
    simp_metrics = {i:j["res_scores"] for i,j in simp.items()}
    
    output = copy_df.copy()# = pd.DataFrame(columns=["site_name", "basin", "group", "metric", "value"])
    output = output.reset_index().drop("index", axis=1)
    for i, row in output.iterrows():
        name = row["site_name"]
        basin = row["basin"]
        group = row["group"]
        metric = row["metric"]
        print(name, basin, group, metric, row["value"])
        if metric == "r2_score":
            my_metric = "NSE"
        elif metric == "rmse":
            my_metric = "RMSE"
            
        if group == "high_rt":
            try:
                value = tree_metrics[basin].loc[name, my_metric]
            except KeyError as e:
                print(f"Tree no {', '.join([name, basin, group, metric])}")
        else:
            try:
                value = simp_metrics[basin].loc[name, my_metric]
            except KeyError as e:
                print(f"Simp no {', '.join([name, basin, group, metric])}")
        print(i, name, basin, group, metric, value)
        output.loc[i] = [name, basin, group, metric, value]

    return output

from IPython import embed as II

II()