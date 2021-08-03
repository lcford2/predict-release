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