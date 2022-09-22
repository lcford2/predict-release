import pickle
import pathlib
import numpy as np
import pymc3 as pm
from utils import read_basin_data, get_basin_meta_data, prep_data, summarize_trace
from tree_model import tree_model, get_leaves_and_groups
import matplotlib.pyplot as plt

def main():
    basin = "upper_col"
    df = read_basin_data(basin)
    meta = get_basin_meta_data(basin)
    
    reservoirs = meta[meta["group"] == "high_rt"].index

    res_grouper = df.index.get_level_values(0)
    df = df.loc[res_grouper.isin(reservoirs)]
    # need to set it again after we trimmed the data set
    res_grouper = df.index.get_level_values(0)
    time_grouper = df.index.get_level_values(1)

    X,y,means,std = prep_data(df)
    X["sto_diff"] = X["storage_pre"] - X["storage_roll7"]

    max_depth = 3
    random_state=37
    tree = tree_model(X, y, max_depth=max_depth, random_state=random_state)
    leaves, groups = get_leaves_and_groups(X, tree)

    group_indices = {
        g:groups[groups == g].index for g in groups.unique()
    }
    
    formula = "release ~ release_pre + sto_diff + inflow + release_roll7 + \
                inflow_roll7 + storage_x_inflow"
    outdir = pathlib.Path(f"./output/{basin}")
    if not outdir.exists():
        (outdir / "figures").mkdir(parents=True)

    grouped_results = {}
    summaries = {}
    for g, index in group_indices.items():
        data = X.loc[index]
        data["release"] = y.loc[index]
        with pm.Model() as normal_model:
            family = pm.glm.families.Normal()
            pm.GLM.from_formula(formula, data=data, family=family)
            normal_trace = pm.sample(draws=2000, tune=1000, chains=2, cores=1, return_inferencedata=False,
                                        discard_tuned_samples=True)
            axes = pm.plot_trace(normal_trace)
            plt.tight_layout()
            plt.savefig(outdir / "figures" / f"{g}_trace.png")
        grouped_results[g] = (normal_model, normal_trace)
        summaries[g] = summarize_trace(normal_trace)
    
    output = {"results":grouped_results, "summaries":summaries}
    with open(outdir /"fit_results.pickle", "wb") as f:
        pickle.dump(output, f)


if __name__ == "__main__":
    main()
