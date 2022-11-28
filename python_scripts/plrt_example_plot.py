#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots

plt.style.use(["science", "nature"])
# plt.rcParams.update({"figure.dpi": 100})  # only for ieee
sns.set_context("poster")
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

x = np.random.rand(1000) * 10
x.sort()

x1 = x[:500]
x2 = x[500:]

def fun1(x):
    return 3 * x + 2


def fun2(x):
    return x + 12


y1 = fun1(x1) + np.random.normal(0, 1, 500)
y2 = fun2(x2) + np.random.normal(0, 1, 500)

x_split = (x1.max() + x2.min()) / 2

ratio = 19/10
ratio = 12/10
height = 8
fig = plt.figure(figsize=(ratio * height, height), dpi=800)
ax = fig.add_subplot()

ax.scatter(x1, y1, c=colors[0], s=36)
ax.scatter(x2, y2, c=colors[0], s=36)

lw = 4

ax.axhline(y1.mean(), 0, x_split/10, c=colors[1], label="Regression Tree", lw=lw)
ax.axhline(y2.mean(), x_split/10, 1, c=colors[1], lw=lw)
ax.axvline(x_split, c="k")

x_actual_1 = np.linspace(0, x_split, 500)
x_actual_2 = np.linspace(x_split, 10, 500)

ax.plot(x_actual_1, fun1(x_actual_1), c=colors[2], linestyle="--", label="PLRT", linewidth=lw)
ax.plot(x_actual_2, fun2(x_actual_2), c=colors[2], linestyle="--", linewidth=lw)

ax.set_xlabel("X")
ax.set_ylabel("Y")

ax.legend(loc="upper left")
ax.set_xlim(0, 10)

# plt.show()
plt.savefig(
    "../figures/agu_2022_figures/plrt_example.eps", format="eps", dpi=800
)
