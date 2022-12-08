#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import seaborn as sns
from functools import partial

# setup python
# plt.style.use(["science", "nature"])
# plt.style.use("tableau-colorblind10")
# plt.rcParams.update({"figure.dpi": 100})  # only for ieee
sns.set_context("poster")
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# generate random data
x = np.random.rand(1000) * 10
x.sort()

x1 = x[:500]
x2 = x[500:]
x_split = (x1.max() + x2.min()) / 2

m1 = 4
b1 = 2
m2 = 1

def line(m, b, x):
    return m * x + b

# functions for lines
fun1 = partial(line, m1, b1)

b2 = fun1(x_split) - m2 * x_split
fun2 = partial(line, m2, b2)

# get y values
y1 = fun1(x1) + np.random.normal(0, 1, 500)
y2 = fun2(x2) + np.random.normal(0, 1, 500)

# fit linear regression to X and Y
X = np.concatenate((x1, x2))
y = np.concatenate((y1, y2))
X = np.concatenate((np.ones(1000).reshape(-1, 1), X.reshape(-1, 1)), axis=1)
lr_fit = np.linalg.inv(X.T @ X) @ X.T @ y


# define figure size
ratio = 20 / 10 # width to height
height = 8
fig = plt.figure(figsize=(ratio * height, height), dpi=800)
ax = fig.add_subplot()

ax.scatter(x1, y1, c=colors[0], s=32, alpha=0.6)
ax.scatter(x2, y2, c=colors[0], s=32, alpha=0.6)

lw = 4

ax.axhline(y1.mean(), 0, x_split / 10, c=colors[1], label="Regression Tree", lw=lw)
ax.axhline(y2.mean(), x_split / 10, 1, c=colors[1], lw=lw)
ax.axvline(x_split, c="k")

x_actual_1 = np.linspace(0, x_split, 500)
x_actual_2 = np.linspace(x_split, 10, 500)

ax.plot(
    x_actual_1,
    [lr_fit[0] + lr_fit[1] * i for i in x_actual_1],
    c=colors[2],
    linestyle="-.",
    label="LR",
    linewidth=lw,
)
ax.plot(
    x_actual_2,
    [lr_fit[0] + lr_fit[1] * i for i in x_actual_2],
    c=colors[2],
    linestyle="-.",
    linewidth=lw,
)
ax.plot(
    x_actual_1, fun1(x_actual_1), c=colors[3], linestyle="--", label="PLRT", linewidth=lw
)
ax.plot(x_actual_2, fun2(x_actual_2), c=colors[3], linestyle="--", linewidth=lw)

ax.set_xlabel("X")
ax.set_ylabel("Y")

ax.legend(loc="upper left")
ax.set_xlim(0, 10)

# plt.show()
plt.savefig("../figures/agu_2022_figures/plrt_example_new.png", format="png", dpi=800)
