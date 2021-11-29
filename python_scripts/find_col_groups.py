import sys
import pickle
import pathlib
import pandas as pd
import numpy as np
import subprocess
from utils.timing_function import time_function
from time import perf_counter
from multi_basin_tree import (read_basin_data,
                              get_basin_meta_data,
                              prep_data)
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.tree import (DecisionTreeRegressor,
                          export_graphviz)
from IPython import embed as II

def main():
    time1 = perf_counter()
    basin = "colorado"
    df = read_basin_data(basin)
    meta = pd.read_csv("../group_res/colorado_meta.csv", index_col=0)

    reservoirs = meta.index

    df = df.loc[df.index.get_level_values(0).isin(reservoirs)]
    res_grouper = df.index.get_level_values(0)

    X,y,means,std = prep_data(df)
    purposes = pd.read_csv("../nid_data/col_purposes.csv",
                           index_col=0,
                           squeeze=True)
    x_cols = ["Primary_Purpose", "Multipurpose", "rts",
              "max_sto"]
    meta["Multipurpose"] = [0 if i < 2 else 1 for i in meta["Num_Purposes"]]

    print(f"Making records: {perf_counter() - time1:.4f} secs.")
    counts = df.groupby(res_grouper).count()["release"]
    x_records = []
    # for r in res_grouper:
    for r in reservoirs:
        x_records.append(meta.loc[r, x_cols].tolist())

    print(f"Making records frame: {perf_counter() - time1:.4f} secs.")
    x_frame = pd.DataFrame.from_records(x_records,
                                        #index=df.index,
                                        index=reservoirs,
                                        columns=x_cols)

    print(f"Making one hot data: {perf_counter() - time1:.4f} secs.")
    one_hot_data = pd.get_dummies(x_frame["Primary_Purpose"])
    x_frame[one_hot_data.columns] = one_hot_data
    x_frame = x_frame.drop("Primary_Purpose", axis=1)

    stdzed = False
    if not stdzed:
        y = df["release"]
    y = means["release"]

    std_mod = "_std" if stdzed else ""

    output_path = "../group_res/colorado_output/tdepth{depth}" + std_mod + "_no_inflow_means"
    # print(f"Fitting Tree: {perf_counter() - time1:.4f} secs.")
    max_depths = [3,4,5,6]

    for max_depth in max_depths:
        output_dir = pathlib.Path(output_path.format(depth=max_depth))
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        # fit tree
        tree = DecisionTreeRegressor(max_depth=max_depth,
                                     random_state=37)
        tree.fit(x_frame, y)
        # write tree to disk
        export_graphviz(tree, out_file=(output_dir / "tree.dot").as_posix(),
                        feature_names=x_frame.columns, filled=True, proportion=False,
                        rounded=True, rotate=True)
        with open(output_dir / "tree.pickle", "wb") as f:
            pickle.dump(tree, f)

        sp_output = subprocess.run([
            "dot", (output_dir / "tree.dot").as_posix(), "-Tpng",
            "-o",(output_dir / "tree.png").as_posix()
        ], capture_output=True)
        leaves = pd.Series(tree.apply(x_frame), index=x_frame.index)
        counts = leaves.groupby(leaves.index.get_level_values(0)).mean().value_counts()
        print(max_depth)
        print(counts)



if __name__ == "__main__":
    main()
