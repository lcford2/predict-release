import sys
import pickle
import calendar
import pathlib
import pandas as pd
import numpy as np
import scipy.optimize as spo
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from plot_helpers import determine_grid_size
from IPython import embed as II

plt.style.use("ggplot")
sns.set_context("notebook")
RESULTS_DIR = pathlib.Path("G:/My Drive/PHD/SRY_curves/data/results") 

def read_spatial_data():
    with open((RESULTS_DIR / "synthesis" / "spatial_model" / "res_slopes.pickle").as_posix(), "rb") as f:
        data = pickle.load(f)
    
    return data["coefs"]

def read_grouped_data():
    tree_path = RESULTS_DIR / "synthesis" / "treed_model" / "upstream_basic_td3_roll7_simple_tree_month_coefs" / "results.pickle"
    simp_path = RESULTS_DIR / "synthesis" / "simple_model" / "all_res_time_fit" / \
        "NaturalOnly-RunOfRiver_filter_ComboFlow_SIx_pre_std_swapped_res_roll7_wrel.pickle"

    with open(tree_path.as_posix(), "rb") as f:
        tree = pickle.load(f)

    with open(simp_path.as_posix(), "rb") as f:
        simp = pickle.load(f)
    
    tree_coefs = tree["coefs"].loc[calendar.month_abbr[1:]].T
    simp_coefs = simp["coefs"].loc[calendar.month_abbr[1:]].T
     
    return tree_coefs.append(simp_coefs).T


def sine(t, b0, b1, omega):
    return b0 + b1 * np.sin(t*omega)

def loss(params, t, y):
    b0, b1, omega = params
    y_hat = b0 + b1 * np.sin(t*omega)
    return np.mean(np.power(y_hat - y, 2))

def main(g_or_s="s"):
    coefs = read_spatial_data() if g_or_s == "s" else read_grouped_data()

    results = pd.DataFrame(index=coefs.columns, columns=["b0", "b1", "omega", "score"])

    months = calendar.month_abbr[1:]
    x = np.arange(12)
    for res in coefs.columns:
        y = coefs.loc[months, res]
        # popt, pcov = spo.curve_fit(sine, x, y)
        out = spo.minimize(loss, [0.01, 0.01, 0.5], args=(x, y))
        popt = out.x

        fitted = sine(x, *popt)
        score = r2_score(y, fitted)
        results.loc[res] = [popt[0], popt[1], popt[2], score]

    if g_or_s == "s":
        results.to_pickle(RESULTS_DIR / "synthesis" / "spatial_model" / "sin_results.pickle")
    else:
        results.to_pickle(RESULTS_DIR / "synthesis" / "sin_results_groups.pickle" )

    # results = results.sort_values(by="score")
    rename={
            3:"Leaf 1",
            4:"Leaf 2",
            6:"Leaf 3",
            7:"Leaf 4",
            10:"Leaf 5",
            11:"Leaf 6",
            13:"Leaf 7",
            14:"Leaf 8",
            "ComboFlow-RunOfRiver":"ROR",
            "ComboFlow-StorageDam":"LRT"
    }
    results = results.rename(index=rename)
    coefs = coefs.rename(columns=rename)

    results = results.loc[[
        "ROR",
        "LRT",
        "Leaf 1",
        "Leaf 2",
        "Leaf 3",
        "Leaf 4",
        "Leaf 5",
        "Leaf 6",
        "Leaf 7",
        "Leaf 8"
    ],:]
    grid_size = determine_grid_size(results.shape[0])
    fig, axes = plt.subplots(*grid_size, figsize=(20, 8.7), sharex=True, sharey=True)
    axes = axes.flatten()
    # fig.patch.set_alpha(0.0)

    for ax, res in zip(axes, results.index):
        y = coefs.loc[months, res] 
        fitted = sine(x, results.loc[res,"b0"], results.loc[res,"b1"], results.loc[res,"omega"])
        ax.plot(x, y, label="Original Model")
        ax.plot(x, fitted, label="Sine Wave")
        handles, labels = ax.get_legend_handles_labels()
        ax.set_title(res)
        ax.set_xticks(x)
        ax.set_xticklabels(months, rotation=90)#, ha="right")

    ax = axes[-1]
    if results.shape[0] % 2 == 1:
        ax.set_axis_off()
        ax.legend(handles, labels, loc="center", frameon=True, prop={"size":14})
    else:
        axes[0].legend(handles, labels, loc="best", frameon=True, prop={"size":12})
    
    fig.text(0.02, 0.5, "Monthly Intercept", va="center", ha="center", rotation=90,
            fontsize=14)
    
    plt.tight_layout()
    plt.show() 
   


if __name__ == "__main__":
    main(g_or_s = sys.argv[1])
    
