import pandas as pd
import sys
import glob
import pathlib

def parse_args():
    args = sys.argv
    if len(args) == 1:
        raise ValueError("Must provide the name of the file as the first argument to this script.")
    elif args[1] == "-G":
        files = glob.glob(args[2])
    else:
        files = args[1:]
    return files

def clean_results(filename):
    try:
        results = pd.read_pickle(filename)
    except Exception as e:
        print(f"Failed to load {filename} with error '{e.args[1]}'", 
              file=sys.stderr)
        return False

    trees = list(results.keys())
    leafs = set(results[0].keys())
    # make sure all leafs are considered. 
    for tree in trees:
        leafs = leafs | set(results[tree].keys())

    leafs = list(leafs)

    params = list(results[trees[0]][leafs[0]].index)
    params = ["Tree", "Leaf"] + params

    df = pd.DataFrame(0, columns=params, index=[])

    for tree, leaf_dict in results.items():
        for leaf, series in leaf_dict.items():
            series["Tree"] = tree
            series["Leaf"] = leaf
            df = df.append(series, ignore_index=True)
    filepath = pathlib.Path(filename)
    clean_dir = filepath.parent.parent/"clean"
    split_name = filepath.name.split(".")
    clean_name = f"{split_name[0]}_cleaned.{split_name[1]}"
    df.to_pickle(clean_dir/clean_name)
    return True

if __name__ == "__main__":
    files = parse_args()
    for file in files:
        clean_results(file)