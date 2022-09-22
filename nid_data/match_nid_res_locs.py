import pandas as pd
import geopandas as gpd
import pathlib
from IPython import embed as II
import numpy as np
from scipy.spatial import KDTree

def ckdnearest(gdA, gdB):
    nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
    btree = KDTree(nB)
    dist, idx = btree.query(nA, k=1)
    result = pd.DataFrame({"dist":dist, "nid_idx": idx}, index=gdA.index)
    result = result[result["dist"] != np.inf]
    add_columns = gdB.drop(columns="geometry").columns
    result[add_columns] = gdB.loc[result["nid_idx"], add_columns].reset_index(drop=True)
    gdf = pd.concat([gdA, result], axis=1)
    # gdB_nearest = gdB.iloc[idx].drop(columns="geometry").reset_index(drop=True)
    # gdf = pd.concat(
    #     [
    #         gdA.reset_index(drop=True),
    #         gdB_nearest,
    #         pd.Series(dist, name="dist"),
    #         pd.Series(idx, name="")
    #     ], axis=1)
    return gdf

res = pd.read_csv("../geo_data/reservoirs.csv")
res_gdf = gpd.GeoDataFrame(res, geometry=gpd.points_from_xy(x=res["long"], y=res["lat"], crs="EPSG:4269"))

nid = pd.read_csv("./all_dams_data.csv", low_memory=False)
nid_gdf = gpd.GeoDataFrame(nid, geometry=gpd.points_from_xy(nid["Longitude"], nid["Latitude"], crs="EPSG:4269"))

result = ckdnearest(res_gdf, nid_gdf)
II()
