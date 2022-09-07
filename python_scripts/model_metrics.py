import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def rmse(act, mod):
    return mean_squared_error(act, mod, squared=False)


def mean_absolute_scaled_error(yact, ymod):
    error = (ymod - yact).abs().mean()
    yact = np.array(yact)
    lagerror = np.absolute(yact[1:] - yact[:-1]).mean()
    return error / lagerror
