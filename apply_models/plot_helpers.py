import matplotlib.pyplot as plt
import numpy as np


def abline(intercept, slope, ax=None, **kwargs):
    """Generate an abline (y=mx+b). If `ax` is not provided, 
    will get current axes with plt.gca(). `kwargs` will be passed to 
    ax.plot.

    :param intercept: Intercept of line: b in y=mx+b
    :type intercept: float
    :param slope: Slope of line: m in y=mx+b`
    :type slope: float
    :param ax: Axes to draw line on, defaults to None
    :type ax: matplotlib.axes, optional
    """
    if not ax:
        ax = plt.gca()
    x_values = np.array(ax.get_xlim())
    y_values = intercept + slope * x_values
    ax.plot(x_values, y_values, "--", **kwargs)

def combine_legends(*axes):
    h, l = [], []
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        h.extend(handles)
        l.extend(labels)
    return h, l

def determine_grid_size(N, col_bias=True):
    if N <= 3:
        return (N,1)
    else:
        poss_1 = [(i, N//i) for i in range(2, int(N**0.5) + 1) if N % i == 0]
        poss_2 = [(i, (N+1) // i) for i in range(2, int((N+1)**0.5) + 1) if (N+1) % i == 0]
        poss = poss_1 + poss_2
        min_index = np.argmin([sum(i) for i in poss])
        if col_bias:
            return poss[min_index]
        else:
            return (poss[min_index][1], poss[min_index][0])

def find_plot_functions(namespace):
    """Parses the namespace for functions that start with "plot"

    :param namespace: list of items in the namespace, must be passed because it cannot be determined from within the function
    :type namespace: list
    :return: Dictionary of {simple name:function name} for available plotting functions. simple name is what will be provided by the user.
    :rtype: dict
    """
    plot_functions = filter(lambda x: x[:4] == "plot", namespace)
    plot_name_dict = {"_".join(i.split("_")[1:]):i for i in plot_functions}
    return plot_name_dict