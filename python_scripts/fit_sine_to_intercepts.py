import pickle
import calendar
import pandas as pd
import numpy as np
import scipy.optimize as spo
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")
sns.set_context("paper")


def read_data():
    with open("../results/synthesis/spatial_model/res_slopes.pickle", "rb") as f:
        data = pickle.load(f)
    
    return data["coefs"]

def sine(t, b0, b1, omega):
    return b0 + b1 * np.sin(t*omega)

def loss(params, t, y):
    b0, b1, omega = params
    y_hat = b0 + b1 * np.sin(t*omega)
    return np.mean(np.power(y_hat - y, 2))

def main():
    coefs = read_data()
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

    fig, axes = plt.subplots(nrows=4, ncols=7, figsize=(20, 8.7), sharex=True, sharey=True)
    axes = axes.flatten()
    # fig.patch.set_alpha(0.0)

    for ax, res in zip(axes, coefs.columns):
        y = coefs.loc[months, res] 
        fitted = sine(x, results.loc[res,"b0"], results.loc[res,"b1"], results.loc[res,"omega"])
        ax.plot(x, y, label="Actual Coefs")
        ax.plot(x, fitted, label="Sin Coefs")
        handles, labels = ax.get_legend_handles_labels()
        ax.set_title(res)
        ax.set_xticks(x)
        ax.set_xticklabels(months, rotation=45, ha="right")
    
    ax = axes[-1]
    ax.set_axis_off()
    ax.legend(handles, labels, loc="center", frameon=True, prop={"size":14})
    plt.tight_layout()
    plt.show() 
   
    from IPython import embed as II


if __name__ == "__main__":
    main()
    
