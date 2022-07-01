import pandas as pd
import geopandas as gpd
import pathlib
from IPython import embed as II
import matplotlib.pyplot as plt

GIS_DIR = pathlib.Path("G:/My Drive/PHD/GIS/")

states = gpd.read_file((GIS_DIR / "cb_2017_us_state_500k.shp").as_posix())

basin = "missouri"

huc2 = GIS_DIR / f"{basin}_shp" / "Shape" / "WBDHU2.shp"

huc_gdf = gpd.read_file(huc2.as_posix())
huc_buffer = gpd.GeoDataFrame(huc_gdf.copy(), geometry=huc_gdf.buffer(0.02 resolution=32, cap_style=3, join_style=2))

nid = pd.read_csv("./all_dams_data.csv", low_memory=False)
nid_gdf = gpd.GeoDataFrame(nid, geometry=gpd.points_from_xy(nid["Longitude"], nid["Latitude"], crs="EPSG:4269"))
huc_nid = gpd.tools.sjoin(nid_gdf, huc_buffer, how="inner", op="within")

min_size = 1
max_size = 600
min_value = huc_nid["Max_Storage"].min()
max_value = huc_nid["Max_Storage"].max()
print(huc_nid["Max_Storage"].sum())

ratio = (max_size - min_size) / (max_value - min_value)
sizes = huc_nid["Max_Storage"] * ratio + min_size

fig, ax = plt.subplots(1, 1)

states.plot(ax=ax, facecolor="white", edgecolor="k")
huc_buffer.plot(ax=ax, facecolor=(1, 1, 1, 0), edgecolor="r", alpha=1.0)
huc_gdf.plot(ax=ax, facecolor=(1, 1, 1, 0), edgecolor="b", alpha=1.0)
huc_nid.plot(ax=ax, markersize=sizes)

ax.set_ylim(33.95, 37.41)
ax.set_xlim(-88.92, -80.54)

plt.show()
