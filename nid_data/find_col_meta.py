import pandas as pd
from IPython import embed as II

upper_col = pd.read_csv("../upper_colorado_data/model_ready_data/upper_col_data.csv",
                        index_col=[0,1])
lower_col = pd.read_csv("../lower_col_data/model_ready_data/lower_col_data.csv",
                        index_col=[0,1])
nid = pd.read_csv("./NID_ul_CO.csv")
nid["Dam_Name"] = nid["Dam_Name"].astype(str)

col = upper_col.append(lower_col)
res = col.index.get_level_values(0)
resuniq = res.unique()
II()
sys.exit()
nid_sim = {}

for r in resuniq:
    nid_sim[r] = nid.loc[nid["Dam_Name"] == r]

singles = [j for i,j in filter(lambda x: x[1].shape[0] == 1, nid_sim.items())]
nonefnd = [i for i,j in filter(lambda x: x[1].empty, nid_sim.items())]
remain  = {i:j for i,j in filter(lambda x: x[1].shape[0] > 1, nid_sim.items())}


idxs = {}
nofnd = []
for r in resuniq:
    print(nid.loc[nid["Dam_Name"].str.contains(r)])
    ans = input("Enter index of correct dam or 'none': ")
    if ans == "none":
        nofnd.append(r)
    else:
        idxs[r] = int(ans)
II()
