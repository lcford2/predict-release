import calendar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t as tstat
from statsmodels.tsa.stattools import pacf, acf
from statsmodels.graphics.gofplots import qqplot
from sklearn.metrics import r2_score, mean_squared_error
from IPython import embed as II
from tclr_seasonal_performance import load_data

plt.style.use("seaborn")
sns.set_context("talk")


def corr_plots(df):
    resid = df["model"] - df["actual"]

    racf = acf(resid, nlags=30)
    rpacf = pacf(resid, nlags=30)

    fig, axes = plt.subplots(2, 1, sharex=True)
    fig.patch.set_alpha(0.0)
    ax1, ax2 = axes.flatten()

    ax1.bar(range(31), racf)
    ax2.bar(range(31), rpacf)
    ax1.set_ylabel("Residual ACF")
    ax2.set_ylabel("Residual PACF")
    ax2.set_xlabel("Lag")
    plt.show()


def plot_qqplot(df):
    resid = df["model"] - df["actual"]
    fig, ax = plt.subplots(1, 1)
    fig.patch.set_alpha(0.0)
    qqplot(resid, line="45",
           loc=resid.mean(), scale=resid.std(), ax=ax)
    plt.show()


def plot_dist(df):
    resid = df["model"] - df["actual"]
    fig, ax = plt.subplots(1, 1)
    fig.patch.set_alpha(0.0)
    resid.plot.kde()
    # plt.hist(resid, density=True, bins=200)
    # sns.ecdfplot(resid)
    # resid.plot.box(ax=ax)
    plt.show()


if __name__ == "__main__":
    data = load_data()
    import sys
    try:
        best = sys.argv[1]
    except IndexError:
        best = 4

    df = data[best]["train_data"]
    means = df.groupby("site_name")["actual"].mean()
    stds = df.groupby("site_name")["actual"].std()
    df["actual"] = ((df["actual"].unstack().T - means) / stds).T.stack()
    df["model"] = ((df["model"].unstack().T - means) / stds).T.stack()
    # corr_plots(df)
    plot_qqplot(df)
    # plot_dist(df)
    # II()

    # this takes too long
    # pd.plotting.autocorrelation_plot(resid[0], maxlags=30)
    # plt.acorr(resid[0], maxlags=30, normed=False)
