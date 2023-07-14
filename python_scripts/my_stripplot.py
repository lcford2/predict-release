from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from utils.helper_functions import linear_scale_values


def stripplot(
    data,
    x,
    y,
    hue=None,
    order=None,
    hue_order=None,
    jitter=True,
    dodge=True,
    palette=None,
    size=None,
    ax=None,
    colors=None,
    **kwgs,
):

    df = data.copy()
    df = df.rename(columns={x: "x", y: "y", hue: "hue", size: "size"})

    order = order if order else df["x"].unique()
    hue_order = hue_order if hue_order else df["hue"].unique()

    xticks = list(range(df["x"].unique().size))
    nhue = df["hue"].unique().size
    width = 0.8 / nhue
    if dodge:
        adjs = linear_scale_values(range(nhue), -0.4 + width / 2, 0.4 - width / 2)
    else:
        adjs = [0.0 for i in range(nhue)]

    if jitter is True:
        jlim = 0.1
    else:
        jlim = float(jitter)
    if dodge:
        jlim /= nhue

    jlim *= width
    jitterer = partial(np.random.uniform, low=-jlim, high=+jlim)
    njitter = df.shape[0] / nhue / len(order)
    jitter = jitterer(size=int(njitter))

    if ax == None:
        ax = plt.gca()

    for i, hue in enumerate(hue_order):
        x_adj = [x + adjs[i] for x in xticks]

        pdf = df[df["hue"] == hue]
        for j, var in enumerate(order):
            vdf = pdf[pdf["x"] == var]
            if colors:
                color = vdf[colors]
            else:
                color = palette[i]
            ax.scatter(x_adj[j] + jitter, vdf["y"], color=color, s=vdf["size"], **kwgs)
    return ax
