import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error


def load_all_data():
    base_path = "../results/basin_eval/all/tclr_model_no_ints_all_res_"

    all_data_paths = {
        "D0": "../results/basin_eval/all/tclr_model_no_ints_all_res_0",
        "D1": "../results/basin_eval/all/tclr_model_no_ints_all_res_1",
        "D2": "../results/basin_eval/all/tclr_model_no_ints_all_res_2",
        "D3": "../results/basin_eval/all/tclr_model_no_ints_all_res_3",
        "D4": "../results/basin_eval/all/tclr_model_no_ints_all_res_4",
        "D5": "../results/basin_eval/all/tclr_model_no_ints_all_res_5",
        "D6": "../results/basin_eval/all/tclr_model_no_ints_all_res_6"
    }

    data = {}
    for base_id in range(11):
        d_id = f"D{base_id}"
        d_path = f"{base_path}{base_id}"
        data[d_id] = pd.read_pickle(f"{d_path}/results.pickle")["data"]["test_data"]

    return data

def calc_scores(data):
    scores = {}
    for d_id, test_data in data.items():
        res_grouper = test_data.index.get_level_values(0)
        d_scores = pd.DataFrame(
            index=res_grouper.unique(),
            columns=["NSE", "RMSE"]
        )
        d_scores["NSE"] = test_data.groupby(res_grouper).apply(
            lambda x: r2_score(x["actual"], x["model"])
        )
        d_scores["RMSE"] = test_data.groupby(res_grouper).apply(
            lambda x: mean_squared_error(x["actual"], x["model"], squared=False)
        )
        scores[d_id] = d_scores
    return scores


def plot_scores(scores):
    scores = scores["D1"].sort_values(by="NSE")
    df = pd.DataFrame(
        {i: j["NSE"] for i, j in scores.items()}
    )


if __name__ == "__main__":
    data = load_all_data()
    scores = calc_scores(data)

    df = pd.DataFrame(
        {i: j["NSE"] for i, j in scores.items()}
    )
    df = df.sort_values(by="D0")

    from IPython import embed as II
    II()
