import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import pandas as pd
import scaleogram as scg
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from datetime import timedelta, datetime
from statsmodels.graphics.regressionplots import abline_plot
from IPython import embed as II
import calendar
from explore_results import flow_regime_analysis, plot_reservoir_fit
from timing_function import *

# setup plotting style and sizing
plt.style.use("seaborn-paper")
sns.set_context("talk")

results_dir = pathlib.Path("./simple_model_results")
with open(results_dir / "scaled_OLS_results.pickle", "rb") as f:
    results = pickle.load(f)

means = results["means"]
std = results["std"]
fit = results["fit"]
model = results["model"]
test_set = results["test_set"]
train_set = results["train_set"]
scaled_df = results["scaled_df"]

values_df = test_set["Test"].unstack()
preds_df = test_set["Preds"].unstack()

thirds_metrics = flow_regime_analysis(values_df, preds_df, quan=0.333)

rmeans = means["Release"]
rstd = std["Release"]

values_scaled = (values_df - rmeans) / rstd
preds_scaled = (preds_df - rmeans) / rstd

@time_function
def predict_all(fit, df, start_date, means, std):
    params = fit.params
    b0 = params["const"]
    st_pre = params["Storage_pre"]
    net_inf = params["Net Inflow"]
    rel_pre = params["Release_pre"]
    end_date = df.index.get_level_values(0)[-1]
    pred_index = df[df.index.get_level_values(0) >= start_date].index
    forecasted = pd.DataFrame(0, index=pred_index, 
            columns=["Storage", "Release", "Release_pre", "Storage_pre"])
    idx = pd.IndexSlice
    forecasted.loc[idx[start_date,:],"Release_pre"] = df.loc[idx[start_date,:],"Release_pre"]
    forecasted.loc[idx[start_date,:],"Storage_pre"] = df.loc[idx[start_date,:],"Storage_pre"]
    for i in range((end_date - start_date).days+1):
        current_index = start_date + timedelta(days=i)
        net_inflow = df.loc[idx[current_index, :], "Net Inflow"].unstack().loc[current_index]
        prev_release = forecasted.loc[idx[current_index, :], "Release_pre"].unstack().loc[current_index]
        prev_storage = forecasted.loc[idx[current_index, :], "Storage_pre"].unstack().loc[current_index]
        release = b0 + st_pre * prev_storage + net_inf * net_inflow + rel_pre * prev_release
        #* This way does not preserve mass balance
        # storage = prev_storage + net_inflow - release
        #* This way DOES preserve mass balance
        storage = (prev_storage*std["Storage"]+means["Storage"]) + \
                  (net_inflow*std["Net Inflow"]+means["Net Inflow"]) - \
                  (release*std["Release"]+means["Release"])
        storage = (storage - means["Storage"])/std["Storage"]
        forecasted.loc[idx[current_index,:],"Storage"] = storage.values
        forecasted.loc[idx[current_index,:],"Release"] = release.values
        if current_index != end_date:
            forecasted.loc[idx[current_index+timedelta(days=1),:], "Storage_pre"] = storage.values
            forecasted.loc[idx[current_index+timedelta(days=1),:], "Release_pre"] = release.values

    return forecasted

@time_function
def predict_single_res(data, fit, means, std, start_date=None, horizon=None):
    if not start_date:
        start_date = data.index[0]
    if horizon:
        end_date = start_date + timedelta(days=horizon)
    else:
        end_date = data.index[-1]

    const = fit.params["const"]
    st_pre = fit.params["Storage_pre"]
    net_inf = fit.params["Net Inflow"]
    rel_pre = fit.params["Release_pre"]
    
    pred_index = data[(data.index >= start_date) & (data.index <= end_date)].index
    forecasted = pd.DataFrame(0, index=pred_index,
                              columns=["storage", "release", "release_pre", "storage_pre"])
    forecasted.loc[start_date,"release_pre"] = data.loc[start_date,"release_pre"]
    forecasted.loc[start_date,"storage_pre"] = data.loc[start_date,"storage_pre"]
    for i in range((end_date - start_date).days + 1):
        current_index = start_date + timedelta(days=i)
        net_inflow = data.loc[current_index, "avg_inflow"]
        prev_release = forecasted.loc[current_index, "release_pre"]
        prev_storage = forecasted.loc[current_index, "storage_pre"]
        # II()
        # sys.exit()
        release = const + st_pre * prev_storage + \
            net_inf * net_inflow + rel_pre * prev_release
        storage = (prev_storage * std["storage"] + means["storage"]) + \
                  (net_inflow * std["avg_inflow"] + means["avg_inflow"]) - \
                  (release * std["release"] + means["release"])
        storage = (storage - means["storage"]) / std["storage"]
        forecasted.loc[current_index, "storage"] = storage
        forecasted.loc[current_index, "release"] = release
        if current_index != end_date:
            forecasted.loc[current_index + timedelta(days=1), "storage_pre"] = storage
            forecasted.loc[current_index + timedelta(days=1), "release_pre"] = release
    return forecasted

if __name__ == "__main__":
    # forecasted = predict_all(fit, scaled_df, datetime(2010, 1, 1), means, std)
    # forecast_release = forecasted["Release"].unstack() * rstd + rmeans
    #0.9522364078627754
    # II()
    df = pd.read_json("../br_dam_data/dams/grand_coulee.json")
    remaining_and_units = {
        "avg_inflow_cfs":86400,
        "release_cfs":86400,
        "storage_acre-ft": 43560
    }  
    df_scale, df_mean, df_std = prep_single_res_data(df, unit_change=remaining_and_units)
    forecasted = predict_single_res(df_scale, fit, df_mean, df_std)
    II()