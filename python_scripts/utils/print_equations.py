import pathlib
import pickle
import sys
import calendar

args = sys.argv[1:]
if len(args) == 0:
    print("Indicate what model you want equations for [tree, simp]")
    sys.exit()
else:
    models = args

results_dir = pathlib.Path("G:/My Drive/PHD/SRY_curves/data/results")

model_map = {
    "tree": results_dir/"synthesis"/"treed_model"/"upstream_basic_td3_roll7_simple_tree"/"results.pickle",
    "simp": results_dir / "synthesis" / "simple_model" / "all_res_time_fit" / "NaturalOnly-RunOfRiver_filter_ComboFlow_SIx_pre_std_swapped_res_roll7.pickle"
}

fr"$$r_t =  1.996(s_{{t-1}} - \bar{{s}}_7) + 0.108(s_{{t-1}} \times i_t) + 0.622 r_{{t-1}} + 0.403 \bar{{r}}_7 - 0.086 i_t - 0.148 \bar{{i}}_7 + MI(t)$$"

symbol_map = {
    "Release":r"r_t",
    "Net Inflow":r"i_t",
    "Release_pre":r"r_{{t-1}}",
    "Storage_pre":r"s_{{t-1}}",
    "Storage_Inflow_interaction":r"(s_{{t-1}} \times i_t)",
    "Release_roll7":r"\bar{{r}}_7",
    "Storage_roll7":r"\bar{{s}}_7",
    "Inflow_roll7":r"\bar{{i}}_7",
    "sto_diff":r"(s_{{t-1}} - \bar{{s}}_7)",
    "const":""
}

def sign(x):
    return "+" if x >= 0 else "-"

for model in models:
    file = model_map.get(model)
    if file:
        with open(file, "rb") as f:
            data = pickle.load(f)
    else:
        print(f"Model {model} does not exist.")
        continue

    coefs = data["coefs"]
    if model == "simp":
        coefs = coefs.drop(calendar.month_abbr[1:], axis=0)
    
    for i, row in coefs.T.iterrows():
        symbols = list(map(symbol_map.get, row.index))
        numbers = [f"{sign(i)} {abs(i):.3f}" for i in row.values]
        rhs = " ".join(f"{n} {s}" for n, s in zip(numbers, symbols))
        if rhs[0] == "+":
            rhs = rhs[2:]
        elif rhs[0] == "-":
            rhs = rhs[0] + rhs[2:]        
        rhs += r" + \text{{MI}}(t)"
        equation = f"{symbol_map['Release']} = {rhs}"
        print(i, equation)

    
