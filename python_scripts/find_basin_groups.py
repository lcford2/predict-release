import sys
import pickle
import pathlib
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import (DecisionTreeRegressor,
                          export_graphviz)
from sklearn.ensemble import ExtraTreesRegressor
from multi_basin_tree import (read_basin_data,
                              prep_data)
from simple_model_simul import norm_array, unnorm_array
from IPython import embed as II


plt.style.use("ggplot")
sns.set_context("talk")


def seasonality(mmeans):
    monthmids = [round(15+(365/12)*i, 2) for i in range(12)]
    sin = np.sin([i / 180 * np.pi for i in monthmids])
    cos = np.cos([i / 180 * np.pi for i in monthmids])
    S = np.sum([i*j for i, j in zip(sin, mmeans)])
    C = np.sum([i*j for i, j in zip(cos, mmeans)])
    Pr = np.sqrt(S*S + C*C)
    phi_r = np.arctan(S/C) * 180 / np.pi
    if C < 0:
        phi_out = phi_r + 180
    elif S < 0:
        phi_out = phi_r + 360
    else:
        phi_out = phi_r
    SI = Pr / np.sum(mmeans)
    return SI, phi_out


def prep_seasonalities(mmeans):
    mmeans_rel = mmeans["release"].unstack()
    mmeans_sto = mmeans["storage"].unstack()
    mmeans_inf = mmeans["inflow"].unstack()
    mmeans_sxi = mmeans["storage_x_inflow"].unstack()
    seasonalities = pd.DataFrame(index=mmeans_rel.index,
                                 columns=[
                                     "SI_rel", "PHI_rel",
                                     "SI_inf", "PHI_inf",
                                     "SI_sto", "PHI_sto",
                                     "SI_sxi", "PHI_sxi",
                                 ])

    for res in mmeans_rel.index:
        si_phi = seasonality(mmeans_rel.loc[res])
        seasonalities.loc[res, ["SI_rel", "PHI_rel"]] = si_phi
        si_phi = seasonality(mmeans_sto.loc[res])
        seasonalities.loc[res, ["SI_sto", "PHI_sto"]] = si_phi
        si_phi = seasonality(mmeans_inf.loc[res])
        seasonalities.loc[res, ["SI_inf", "PHI_inf"]] = si_phi
        si_phi = seasonality(mmeans_sxi.loc[res])
        seasonalities.loc[res, ["SI_sxi", "PHI_sxi"]] = si_phi

    return seasonalities.astype(np.float64)


def load_basin_meta_data(basin):
    return pd.read_csv(f"../group_res/{basin}_meta.csv", index_col=0)

def load_data(basin):
    df = pd.concat(read_basin_data(b.lower()) for b in basin)
    meta = pd.concat(load_basin_meta_data(b) for b in basin)
    reservoirs = meta.index
    df = df.loc[df.index.get_level_values(0).isin(reservoirs)]
    return df, meta


def make_multipurpose_col(meta):
    meta["Multipurpose"] = [0 if i < 2 else 1 for i in meta["Num_Purposes"]]
    return meta


def make_model_data(df, meta, stdzed=False, normed=False, single_rows=False, sr_var="Mean"):
    reservoirs = meta.index
    if stdzed:
        X, y, means, std = prep_data(df)
    elif normed:
        X = df.loc[:, df.columns != "release"]
        y = df["release"]
        y = norm_array(y, y.min(), y.max())
    else:
        X = df.loc[:, df.columns != "release"]
        y = df["release"]
    mmeans = df.groupby([df.index.get_level_values(0),
                         df.index.get_level_values(1).month]).mean()
    seasonalities = prep_seasonalities(mmeans)
    meta[seasonalities.columns] = seasonalities
    meta = make_multipurpose_col(meta)

    if single_rows:
        if stdzed:
            raise ValueError("stdzed must not be True if single_rows is True")
        x_cols = ["Primary_Purpose", "Multipurpose", "rts",
                  "max_sto"]

        x_records = []
        for r in reservoirs:
            x_records.append(meta.loc[r, x_cols].tolist())

        x_frame = pd.DataFrame.from_records(x_records,
                                            index=reservoirs,
                                            columns=x_cols)
        x_frame[seasonalities.columns] = seasonalities
        one_hot_data = pd.get_dummies(x_frame["Primary_Purpose"])
        x_frame[one_hot_data.columns] = one_hot_data
        x_frame = x_frame.drop("Primary_Purpose", axis=1)
        y = getattr(df.groupby(df.index.get_level_values(0))["release"], sr_var.lower())()
        y = y.loc[x_frame.index]
    else:
        x_frame = pd.DataFrame(index=X.index)
        reser_index = x_frame.index.get_level_values(0)
        rts = [meta.loc[i, "rts"]
               for i in reser_index]
        max_sto = [meta.loc[i, "max_sto"]
                   for i in reser_index]
        multip = [meta.loc[i, "Multipurpose"]
                  for i in reser_index]
        x_frame["rts"] = rts
        x_frame["max_sto"] = max_sto
        x_frame["multip"] = multip
        x_frame[seasonalities.columns] = [seasonalities.loc[i, :]
                                          for i in reser_index]
        x_frame["Primary_Purpose"] = [meta.loc[i, "Primary_Purpose"]
                                      for i in reser_index]
        one_hot_data = pd.get_dummies(x_frame["Primary_Purpose"])
        x_frame[one_hot_data.columns] = one_hot_data
        x_frame = x_frame.drop("Primary_Purpose", axis=1)

    return x_frame, y


def fit_multi_depth(x_frame, y, stdzed=False, max_depths=(3, 4, 5, 6)):
    std_mod = "_std" if stdzed else ""

    output_path = "../group_res/colorado_output/tdepth{depth}" + \
        std_mod + "_no_inflow_means"

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

        subprocess.run([
            "dot", (output_dir / "tree.dot").as_posix(), "-Tpng",
            "-o", (output_dir / "tree.png").as_posix()
        ])
        leaves = pd.Series(tree.apply(x_frame), index=x_frame.index)
        counts = leaves.groupby(
            leaves.index.get_level_values(0)).mean().value_counts()
        print(max_depth)
        print(counts)


def check_feature_importance(x_frame, y, **kwargs):
    rgr = ExtraTreesRegressor(**kwargs)
    rgr = rgr.fit(x_frame, y)
    ft_imp = {
        i: j for i, j in zip(x_frame.columns, rgr.feature_importances_)
    }
    return ft_imp


def plot_feature_importances(basin):
    df, meta = load_data(basin)
    args = [
        ("Standardized Response", True, False, False),
        ("0-1 Normalized Response", False, True, False),
        ("Regular Space Response", False, False, False)
    ]
    fig, axes = plt.subplots(3, 1, sharex=True)
    fig.patch.set_alpha(0.0)
    axes = axes.flatten()
    for argset, ax in zip(args, axes):
        title, stdzed, normed, single_rows = argset
        x_frame, y = make_model_data(df, meta,
                                     stdzed=stdzed,
                                     normed=normed,
                                     single_rows=single_rows)
        print(x_frame)
        ft_imp = check_feature_importance(x_frame, y, n_estimators=100)
        pd.Series(ft_imp).plot.bar(ax=ax)
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(x_frame.columns, rotation=45, ha="right")
        ax.set_title(title)

    fig.text(0.02, 0.5, "Feature Importances",
             ha="center",
             va="center",
             rotation=90)

    plt.tight_layout()
    plt.show()

def plot_single_row_feature_importances(basin):
    df, meta = load_data(basin)
    fig, ax = plt.subplots(1, 1)
    fig.patch.set_alpha(0.0)

    var = "Std"
    x_frame, y = make_model_data(df, meta,
                                 stdzed=False,
                                 normed=False,
                                 single_rows=True,
                                 sr_var=var)
    print(x_frame)
    ft_imp = check_feature_importance(x_frame, y, n_estimators=100)
    pd.Series(ft_imp).plot.bar(ax=ax)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(x_frame.columns, rotation=45, ha="right")
    ax.set_title(f"Normal Rows [{var}]")
    ax.set_ylabel("Feature Importances")

    plt.tight_layout()
    plt.show()



def main(basin):
    # plot_feature_importances(basin)
    plot_single_row_feature_importances(basin)


if __name__ == "__main__":
    basin = sys.argv[1:]
    main(basin)
