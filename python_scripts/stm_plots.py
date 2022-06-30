import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython import embed as II


def load_data():
    return pd.read_pickle(
        "../results/three_reg_model/all/stodiff_three_model_/results.pickle"
    )


def plot_simmed_scatter(data):
    df = data["simmed_data"]
    fg = sns.relplot(
        data=df,
        x="actual",
        y="model",
        alpha=0.6,
        palette=sns.color_palette("Spectral", 1)
    )

    fg.ax.axline([0, 0], [1, 1], c="r", linestyle="--")
    fg.ax.set_xlabel("Actual Release [1000 acre-ft/day]")
    fg.ax.set_ylabel("Modeled Release [1000 acre-ft/day]")
    fg.fig.patch.set_alpha(0.0)
    fg.ax.patch.set_alpha(0.0)
    plt.show()



if __name__ == "__main__":
    sns.set_context("talk")
    data = load_data()
    plot_simmed_scatter(data)
