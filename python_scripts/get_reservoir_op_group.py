import pandas as pd

META_FILE = "../geo_data/reservoir_meta.csv"


def load_meta():
    return pd.read_csv(META_FILE)


def apply_rules(meta):
    if meta["rts"] <= 9.553:
        if meta["max_sto"] <= 124.9:
            return "small_ror"
        else:
            return "large_ror"
    else:
        if meta["max_sto"] <= 630.9:
            return "small_st_dam"
        elif meta["max_sto"] <= 5196.9:
            return "medium_st_dam"
        else:
            return "large_st_dam"


if __name__ == "__main__":
    meta = load_meta()
    meta["op_group"] = meta.apply(apply_rules, axis=1)
    meta.to_csv(
        META_FILE,
        index=False,
    )
