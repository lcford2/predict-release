import os
import sys
import pickle
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
os.environ["PROJ_LIB"] = r"C:\\Users\\lcford2\AppData\\Local\\Continuum\\anaconda3\\envs\\sry-env\\Library\\share"
from mpl_toolkits.basemap import Basemap
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
import matplotlib.gridspec as GS
import geopandas as gpd
import numpy as np
from IPython import embed as II

GIS_DIR = pathlib.Path("G:\\My Drive\\ms_yr_1\\GIS")

OFFSETS = {
    "Apalachia":(-0.15, 0.1),
    "BlueRidge":(-0.12, -0.12),
    "Boone":(-0.1, -0.13),
    "Chatuge":(0.1, 0.1),
    "Cherokee":(-0.4, 0.02),
    "Chickamauga":(0.1, 0.1),
    "Douglas":(0.08, -0.1),
    "Fontana":(0.1, 0.04),
    "FtLoudoun":(0.1, -0.1),
    "FtPatrick":(-0.2,0.12),
    "Guntersville":(0.1, 0.1),
    "Hiwassee":(0.1, 0.06),
    "Kentucky":(0.1, 0.1),
    "MeltonH":(0.1, 0.1),
    "Nickajack":(-0.15, 0.1),
    "Norris":(-0.3, 0.02),
    "Nottely":(0.04, -0.1),
    "Ocoee1":(-0.3, -0.1),
    "Ocoee3":(-0.12, -0.12),
    "Pickwick":(0.1, 0.1),
    "RaccoonMt":(0.1, 0.1),
    "SHolston":(0.08, 0.1),
    "TimsFord":(0.1, 0.1),
    "WattsBar":(0.1, -0.1),
    "Watauga":(-0.12, -0.12),
    "Wheeler":(0.1, 0.1),
    "Wilson":(0.1, 0.1),
    "Wilbur":(0.1, 0.1)
}

def setup_map(ax, control_area=True, coords=None):
    if not coords:
        west, south, east, north = -90.62, 32.08, -80.94, 37.99
    else:
        west, south, east, north = coords
    m = Basemap(resolution="c",
                llcrnrlon=west,
                llcrnrlat=south,
                urcrnrlon=east,
                urcrnrlat=north,
                ax=ax)
    states_path = GIS_DIR / "cb_2017_us_state_500k"
    river_path = GIS_DIR / "TVA_RIVERS" / "combined"
    bound_path = GIS_DIR / "Bound_fixed" / "Region_06_boundary"
    control_area_path = GIS_DIR / "BA-Control_Areas-electricity" / "tva_control_area"

    mbound = m.drawmapboundary(fill_color='white')
    states = m.readshapefile(states_path.as_posix(), "states")
    streams = m.readshapefile(river_path.as_posix(), "streams", color="b", linewidth=1)
    bound = m.readshapefile(bound_path.as_posix(), "bound", color="#FF3BC6")
    if control_area:
        control_areas = m.readshapefile(control_area_path.as_posix(), "control_areas", 
                                    linewidth=1.2, color="#5d9732")
    bound_lines = bound[4]
    stream_lines = streams[4]
    bound_lines.set_facecolor("#FF3BC6")
    bound_lines.set_alpha(0.4)
    bound_lines.set_zorder(2)

    mid_lon = np.mean([west, east]) #-85.78
    mid_lat = np.mean([south, north]) #35.04
    # mid_lon = -85.78
    # mid_lat = 35.04
    ax.text(mid_lon - 0.8, mid_lat + 0.5, 'TN', fontsize=18,
            ha="center")  # TN
    ax.text(mid_lon + 1.1, mid_lat + 1.3, 'KY', fontsize=18,
            ha="center")  # KY
    ax.text(mid_lon - 3.4, mid_lat - 1.6, 'MS', fontsize=18,
            ha="center")  # MS
    ax.text(mid_lon - 1.3, mid_lat - 1.6, 'AL', fontsize=18,
            ha="center")  # AL
    ax.text(mid_lon + 1.6, mid_lat - 1.6, 'GA', fontsize=18,
            ha="center")  # GA
    ax.text(mid_lon + 3.7, mid_lat + 1.2, 'VA', fontsize=18,
            ha="center")  # VA
    ax.text(mid_lon + 4.1, mid_lat + 0.5, 'NC', fontsize=18,
            ha="center")  # NC
    ax.text(mid_lon + 4.1, mid_lat - 1.4, 'SC', fontsize=18,
            ha="center")  # SC

def get_res_groups():
    with open("../results/treed_ml_model/upstream_basic_td3_roll7_new/results.pickle", "rb") as f:
        tree = pickle.load(f)
    
    with open("../results/multi-level-results/for_graps/NaturalOnly-RunOfRiver_filter_ComboFlow_SIx_pre_std_swapped_res_roll7.pickle", "rb") as f:
        simp = pickle.load(f)

    upstream = tree["data"]["y_test"].columns
    downstream = simp["data"]["X_test"]["compositegroup"].unstack().drop_duplicates().T
    downstream = pd.Series(downstream.values[:,0], index=downstream.index)
    ror = downstream[downstream.str.contains("RunOfRiver")].index
    stdam = downstream[~downstream.str.contains("RunOfRiver")].index

    return list(upstream), list(stdam), list(ror)

def make_system_map():
    fig = plt.figure()
    spec = GS.GridSpec(ncols=1, nrows=1)#, height_ratios=[20, 1])
    ax = fig.add_subplot(spec[0])
    # leg_ax = fig.add_subplot(spec[1])
    # leg_ax.axis("off")
    west, south, east, north = -89.42, 33.58, -80.94, 37.49
    coords = [west, south, east, north]
    m = setup_map(ax, control_area=False, coords=coords)

    reservoirs = GIS_DIR.joinpath("kml_converted", "Hydroelectric-point.shp")

    res_drop = ["Great Falls Dam", "Ocoee #2 Dam",
                "Tellico Dam", "Nolichucky Dam",
                "Raccoon Mountain Pumped-Storage Plant"]

    res_loc = gpd.read_file(reservoirs)
    res_loc = res_loc[~res_loc["Name"].isin(res_drop)]

    res_name_map = {}
    with open("../csv/loc_name_map.csv", "r") as f:
        for line in f:
            line = line.strip("\r\n")
            old, new = line.split(",")
            res_name_map[old] = new

    res_loc["Name"] = [res_name_map[i] for i in res_loc["Name"]]
    colors = ["#00bce5", "#e55100", "#36e500"]

    res_groups = get_res_groups()
    for group in res_groups:
        if "Nikajack" in group:
            index = group.index("Nikajack")
            group.pop(index)
            group.insert(index, "Nickajack")
        if "Chikamauga" in group:
            index = group.index("Chikamauga")
            group.pop(index)
            group.insert(index, "Chickamauga")
    # II()
    for c, g, l in zip(colors, res_groups, ["High RT", "Low RT", "Run of River"]):
        ax.scatter(*zip(*[(row.geometry.x,row.geometry.y) for i, row in res_loc.iterrows() if row.Name in g]),
                facecolor=c, marker="*", s=380, alpha=1, zorder=4, edgecolor="k", label=l)
        
    for i, row in res_loc.iterrows():
        loc = row["geometry"]
        offset = OFFSETS[row["Name"]]
        x = loc.x + offset[0]
        y = loc.y + offset[1]
        name = row["Name"]
        ax.text(x, y, name, zorder=5)

    handles, labels = ax.get_legend_handles_labels()
    
    river_line = mlines.Line2D(
        [], [], color="b", alpha=1, linewidth=1)
    CA_line = mlines.Line2D(
        [], [], color="#5d9732", alpha=1, linewidth=1.2)
    river_basin = mpatches.Patch(facecolor="#FF3BC6", alpha=0.4)

    handles.extend([river_line, river_basin])#, CA_line])
    labels.extend(
        ["Tennessee River", "Tennessee River Basin"])#, "TVA Control Area"])

    # leg_ax.legend(handles, labels, loc="center", frameon=False, ncol=3)
    ax.legend(handles, labels, loc="best", prop={"size":14})

    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    make_system_map()
