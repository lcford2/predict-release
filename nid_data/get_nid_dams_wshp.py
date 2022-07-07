import pandas as pd
import geopandas as gpd
import pathlib
from IPython import embed as II
import matplotlib.pyplot as plt
style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

GIS_DIR = pathlib.Path("G:/My Drive/PHD/GIS/")
my_res = gpd.read_file("../geo_data/my_res_nid_combo")
min_height = my_res["NID_Height"].min()
min_storage = my_res["NID_Storag"].min()

states = gpd.read_file((GIS_DIR / "cb_2017_us_state_500k.shp").as_posix())

# basin = "tennessee"
basins = ["tennessee", "colorado", "missouri", "columbia"]
# basins = basins[:1]

hucs = []
buffs = []
nids = []
min_value = float("inf")
max_value = -float("inf")
min_size = 1
max_size = 500

# height_filter = 10
# storage_filter = 1000
# height_filter = 50
# storage_filter = 10000
height_filter = min_height
storage_filter = min_storage

for basin in basins:
    huc2 = GIS_DIR / f"{basin}_shp" / "Shape" / "WBDHU2.shp"
    huc_gdf = gpd.read_file(huc2.as_posix())
    huc_buffer = gpd.GeoDataFrame(huc_gdf.copy(), geometry=huc_gdf.buffer(0.02, resolution=32, cap_style=3, join_style=2))

    hucs.append(huc_gdf)
    buffs.append(huc_buffer)

    nid = pd.read_csv("./all_dams_data.csv", low_memory=False)
    nid_gdf = gpd.GeoDataFrame(nid, geometry=gpd.points_from_xy(nid["Longitude"], nid["Latitude"], crs="EPSG:4269"))
    huc_nid = gpd.tools.sjoin(nid_gdf, huc_buffer, how="inner", predicate="within")
    huc_nid = huc_nid[
        (huc_nid["NID_Height"] > height_filter) & (huc_nid["NID_Storage"] > storage_filter)
    ]

    nids.append(huc_nid)

    huc_min = huc_nid["Max_Storage"].min()
    huc_max = huc_nid["Max_Storage"].max()
    if huc_min < min_value:
        min_value = huc_min
    if huc_max > max_value:
        max_value = huc_max

II()
sys.exit()

ratio = (max_size - min_size) / (max_value - min_value)
# sizes = huc_nid["Max_Storage"] * ratio + min_size

fig, ax = plt.subplots(1, 1)
states.plot(ax=ax, facecolor="white", edgecolor="k")

for huc_buffer, huc_gdf, huc_nid in zip(buffs, hucs, nids):
    huc_buffer.plot(ax=ax, facecolor="none", edgecolor="r")
    huc_gdf.plot(ax=ax, facecolor="none", edgecolor="b")
    sizes = huc_nid["Max_Storage"] * ratio + min_size
    huc_nid.plot(ax=ax, markersize=sizes, facecolor=style_colors[2])

my_res = gpd.read_file("../geo_data/my_res_nid_combo")
# shp files truncate column names to 10 characters
sizes = my_res["Max_Storag"] * ratio + min_size
my_res.plot(ax=ax, markersize=sizes, facecolor="r", edgecolor="k")

ax.set_ylim(
    23.32979,
    53.36092
)
ax.set_xlim(
    -128.37323,
    -65.48857
)
plt.show()
