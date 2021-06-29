import pandas as pd
import pickle
import sys
from IPython import embed as II

US_PATH = "../results/treed_ml_model_dual_fit/upstream_basic_td3_roll7/results.pickle"
DS_PATH = "../results/multi-level-results/for_graps_dual_fit/NaturalOnly-RunOfRiver_filter_ComboFlow_SIx_pre_std_swapped_res_roll7.pickle"

def merge_results(ds_path=None, us_path=None):
    if ds_path:
        ds_res_data = pd.read_pickle(ds_path)
    else:
        ds_res_data = pd.read_pickle(DS_PATH)
    if us_path:
        us_res_data = pd.read_pickle(us_path)
    else:
        us_res_data = pd.read_pickle(US_PATH)

    output = {"ds_output":{},
            "us_output":{},
            "groups":{},
            "data":{}}

    output["ds_output"] = {key:value for key, value in ds_res_data.items() if key != "data"}
    output["us_output"] = {key:value for key, value in us_res_data.items() if key != "data"}

    output["groups"]["us"] = us_res_data["data"]["groups"]
    output["groups"]["ds"] = ds_res_data["data"]["X_test"]["compositegroup"]

   
    output["data"]["y_train_rel_act"] = ds_res_data["data"]["y_train_rel_act"].unstack().join(
        us_res_data["data"]["y_train_rel_act"].unstack())

    output["data"]["y_test_rel_act"] = ds_res_data["data"]["y_test_rel_act"].unstack().join(
        us_res_data["data"]["y_test_rel_act"].unstack())

    output["data"]["predicted_act_rel"] = ds_res_data["data"]["predicted_act_rel"].unstack().join(
        us_res_data["data"]["predicted_act_rel"].unstack())
    
    output["data"]["y_train_sto_act"] = ds_res_data["data"]["y_train_sto_act"].unstack().join(
        us_res_data["data"]["y_train_sto_act"].unstack())

    output["data"]["y_test_sto_act"] = ds_res_data["data"]["y_test_sto_act"].unstack().join(
        us_res_data["data"]["y_test_sto_act"].unstack())

    output["data"]["predicted_act_sto"] = ds_res_data["data"]["predicted_act_sto"].unstack().join(
        us_res_data["data"]["predicted_act_sto"].unstack())
    

    output["data"]["forecasted"] = {
        "Release":ds_res_data["data"]["forecasted"]["Release_act"].unstack().join(
            us_res_data["data"]["forecasted"]["Release_act"].unstack()),
        "Storage":ds_res_data["data"]["forecasted"]["Storage_act"].unstack().join(
            us_res_data["data"]["forecasted"]["Storage_act"].unstack()),
        }

    output

    return output