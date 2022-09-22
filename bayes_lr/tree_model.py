import numpy as np
import pandas as pd
from sklearn.tree import (DecisionTreeRegressor, export_graphviz)

def tree_model(X, y, **tree_args):
    tree = DecisionTreeRegressor(**tree_args)
    tree.fit(X,y)
    return tree

def get_leaves_and_groups(X,tree):
    leaves = tree.apply(X)
    if leaves.ndim == 2:
        groups = pd.DataFrame(leaves, columns=range(
            1, leaves.shape[1] + 1), index=X.index)
    else:
        groups = pd.Series(leaves, index=X.index)
    return leaves, groups
 