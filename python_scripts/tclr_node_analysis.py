import pandas as pd
import numpy as np
from IPython import embed as II
from tclr_seasonal_performance import load_data


if __name__ == "__main__":
    data = load_data()
    groups = data[4]["groups"].reset_index()
    groups = groups.rename(columns={0: "node"})
    node_counts = groups["node"].value_counts()
