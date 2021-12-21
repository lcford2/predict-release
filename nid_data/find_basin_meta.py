import sys
import pandas as pd
from IPython import embed as II

basin = sys.argv[1]

if basin == "colorado":
    upper_col = pd.read_csv(
        "../upper_colorado_data/model_ready_data/upper_col_data.csv", index_col=[0, 1]
    )
    lower_col = pd.read_csv(
        "../lower_col_data/model_ready_data/lower_col_data.csv", index_col=[0, 1]
    )

    df = upper_col.append(lower_col)
    nid = pd.read_csv("./NID_ul_CO.csv")
elif basin == "TVA":
    df = pd.read_csv("../csv/tva_model_ready_data.csv", index_col=[0, 1])
    nid = pd.read_csv("./all_dams_data.csv")


nid["Dam_Name"] = nid["Dam_Name"].astype(str)
res = df.index.get_level_values(0)
resuniq = res.unique()
nid_sim = {}

for r in resuniq:
    nid_sim[r] = nid.loc[nid["Dam_Name"] == r.upper()]

singles = [j for i, j in filter(lambda x: x[1].shape[0] == 1, nid_sim.items())]
nonefnd = [i for i, j in filter(lambda x: x[1].empty, nid_sim.items())]
remain = {i: j for i, j in filter(lambda x: x[1].shape[0] > 1, nid_sim.items())}

# idxs = {}
# nofnd = []
# for r in resuniq:
#     print(nid.loc[nid["Dam_Name"].str.contains(r.upper())])
#     ans = input("Enter index of correct dam or 'none': ")
#     if ans == "none":
#         nofnd.append(r)
#     else:
#         idxs[r] = int(ans)
idxs = {
    "Apalachia": 14153,
    "Boone": 19369,
    "Chatuge": 14154,
    "Cherokee": 14426,
    "Douglas": 14438,
    "Fontana": 14157,
    "Guntersville": 11001,
    "Hiwassee": 12117,
    "Kentucky": 14032,
    "Norris": 13247,
    "Nottely": 13941,
    "Pickwick": 13246,
    "Watauga": 16464,
    "Wheeler": 11004,
    "Wilbur": 6854,
    "BlueRidge": 7727,
    "Chikamauga": 13249,
    "FtLoudoun": 14437,
    "FtPatrick": 22814,
    "MeltonH": 35855,
    "Nikajack": 44046,
    "Ocoee1": 6848,
    "Ocoee3": 14440,
    "SHolston": 19371,
    "TimsFord": 49809,
    "WattsBar": 14422,
    "Wilson": 7362,
}

bdf = pd.DataFrame({i: nid.loc[j] for i, j in idxs.items()})
bdf = bdf.T.reset_index().rename(columns={"index": "model_name"})
bdf.to_csv(f"./{basin}_nid.csv")

meta = bdf.loc[:,["model_name", "Dam_Name", "Primary_Purpose", "All_Purposes"]]
meta = meta.set_index("model_name")
meta["Num_Purposes"] = meta["All_Purposes"].str.split(",").apply(len)

means = df.groupby(df.index.get_level_values(0)).mean()
meta["rts"] = means["storage"] / means["inflow"]
meta["max_sto"] = df.groupby(df.index.get_level_values(0))["storage"].max()
meta["min_sto"] = df.groupby(df.index.get_level_values(0))["storage"].min()
mean_cols = ["release", "release_pre",
             "storage", "storage_pre",
             "inflow","release_roll7",
             "inflow_roll7", "storage_roll7",
             "storage_x_inflow"]
meta[mean_cols] = means[mean_cols]

meta.to_csv(f"../group_res/{basin}_meta.csv")
