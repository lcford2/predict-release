import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score

plt.style.use("ggplot")
sns.set_context("talk")


def load_mss_results(depth):
    msses = [
        "0.00",
        "0.01",
        "0.02",
        "0.03",
        "0.04",
        "0.05",
        "0.06",
        "0.07",
        "0.08",
        "0.09",
        "0.10",
        "0.11",
        "0.15",
        "0.20",
    ]
    files = [
        f"../results/tclr_model_testing/all/TD{depth}_MSS{i}_RT_MS_exhaustive_new_hoover/results.pickle"
        for i in msses
    ]
    results = {i: pd.read_pickle(f) for i, f in zip(msses, files)}
    return results


def calc_scores(results):
    scores = {}
    for mss, res in results.items():
        df = res["simmed_data"]
        mscores = df.groupby("site_name").apply(
            lambda x: r2_score(x["actual"], x["model"])
        )
        mean = mscores.mean()
        std = mscores.std()
        median = mscores.median()
        scores[mss] = {"mean": mean, "std": std, "median": median}
    return pd.DataFrame(scores)


def plot_score_metrics(scores, depth):
    fig, ax = plt.subplots(1, 1, figsize=(19, 10))
    scores.index = scores.index.astype(float)
    ax.plot(scores.index, scores["mean"], label="Mean", marker="s")
    ax.plot(scores.index, scores["median"], label="Median", marker="o")
    ax.plot(scores.index, scores["std"], label="StD", marker="X")
    ax.set_ylabel("NSE")
    ax.set_xlabel("MSS")
    ax.set_title(f"TD{depth} - Varied MSS")
    ax.legend(loc="upper right")
    # plt.savefig("./figures/mss_line_plot_new.png", dpi=300)
    plt.show()


def rank_score_metrics(scores):
    rank = scores.copy()
    rank[["mean", "median"]] = rank[["mean", "median"]].rank(ascending=False)
    rank["std"] = rank["std"].rank(ascending=True)
    rank["score"] = rank.sum(axis=1)
    print(rank.sort_values("score").to_markdown(floatfmt=".0f"))


def main():
    depth = 4
    results = load_mss_results(depth)
    scores = calc_scores(results).T
    rank_score_metrics(scores)
    plot_score_metrics(scores, depth)


if __name__ == "__main__":
    main()
