import sys
import pickle
import calendar
import pathlib
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLMParams
from sklearn.metrics import r2_score, mean_squared_error
from multi_basin_tree import read_basin_data, get_basin_meta_data, prep_data
from IPython import embed as II


def pipeline():
    basin = sys.argv[1]
    df = read_basin_data(basin)
    meta = get_basin_meta_data(basin)

    use_all = True
    if use_all:
        reservoirs = meta.index
    else:
        reservoirs = meta[meta["group"].isin(["low_rt", "ror"])].index

    reservoirs = reservoirs[~reservoirs.isin(["SANTA ROSA ", "DILLON RESERVOIR"])]

    if len(reservoirs) == 0:
        print("No reservoirs to model.")
        sys.exit()

    res_grouper = df.index.get_level_values(0)
    df = df.loc[res_grouper.isin(reservoirs)]
    # need to set it again after we trimmed the data set
    res_grouper = df.index.get_level_values(0)

    X,y,means,std = prep_data(df)
    groups = df.groupby(res_grouper).apply(lambda x: meta.loc[x.name, "group"])
    purposes = pd.read_csv("../nid_data/col_purposes.csv", index_col=0, squeeze=True)

    groups = purposes
    for res, group in groups.items():
        X.loc[res_grouper == res, "group"] = group

    mi = True
    output = scaled_MixedEffects(X,y,means,std,month_intercepts=mi)

    int_mod = "" if mi else "_no_ints"
    all_mod = "_all_res" if use_all else ""

    output_dir = pathlib.Path(f"../results/basin_eval/{basin}/simple_model{int_mod}{all_mod}_purposes")

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    with open(output_dir / "results.pickle", "wb") as f:
        pickle.dump(output, f, protocol=4)    

def scaled_MixedEffects(X,y,means,std,month_intercepts=False):
    if month_intercepts:
        #* this introduces a intercept that varies monthly and between groups
        month_arrays = {i:[] for i in calendar.month_abbr[1:]}
        for date in X.index.get_level_values(1):
            for key in month_arrays.keys():
                if calendar.month_abbr[date.month] == key:
                    month_arrays[key].append(1)
                else:
                    month_arrays[key].append(0)

        for key, array in month_arrays.items():
            X[key] = array
    
    
    res_grouper = X.index.get_level_values(0)
    time_grouper = X.index.get_level_values(1)

    reservoirs = res_grouper.unique()

    train_index = X.index
    train_res = train_index.get_level_values(0).unique()

    # X_test = X.loc[X.index.get_level_values(0) >= split_date - timedelta(days=8)]
    # X_train = X.loc[X.index.get_level_values(0) < split_date]

    # y_test = y_scaled.loc[y_scaled.index.get_level_values(0) >= split_date - timedelta(days=8)]
    # y_train = y_scaled.loc[y_scaled.index.get_level_values(0) < split_date]
    # y_test_sto = y_scaled_sto.loc[y_scaled_sto.index.get_level_values(0) >= split_date - timedelta(days=8)]
    # y_train_sto = y_scaled_sto.loc[y_scaled_sto.index.get_level_values(0) < split_date]

    X_train = X.loc[train_index]
    # X_test = X.loc[test_index]
    y_train = y.loc[train_index]
    # y_test = y_scaled.loc[test_index]

    # N_time_train = X_train.index.get_level_values(0).unique().shape[0]
    # N_time_test = X_test.index.get_level_values(0).unique().shape[0]

    # train model
    # Instead of adding a constant, I want to create dummy variable that accounts for season differences
    exog = X_train
    exog["const"] = 1.0
    
    groups = exog["group"]
    # Storage Release Interactions are near Useless
    # Release Inflow Interaction does not provide a lot either
    # Storage Inflow interaction seems to matter for ComboFlow-StorageDam reservoirs.
    interaction_terms = ["storage_x_inflow"]

    exog_terms = [
        "const", "inflow", "storage_pre", "release_pre",
        "storage_roll7",  "inflow_roll7", "release_roll7"
        , *interaction_terms
    ]
    
    if month_intercepts:
        exog_terms.extend(calendar.month_abbr[1:])

    exog_re = exog.loc[:,exog_terms]

    mexog = exog.loc[:,["const"]]
       
    free = MixedLMParams.from_components(fe_params=np.ones(mexog.shape[1]),
                                        cov_re=np.eye(exog_re.shape[1]))
    md = sm.MixedLM(y_train, mexog, groups=groups, exog_re=exog_re)
    mdf = md.fit(free=free)

    fitted = mdf.fittedvalues
    fitted_act = (fitted.unstack().T * 
                std.loc[train_res, "release"] + means.loc[train_res, "release"]).T.stack()
    y_train_act = (y_train.unstack().T *
                   std.loc[train_res, "release"] + means.loc[train_res, "release"]).T.stack()

    f_act_score = r2_score(y_train_act, fitted_act)
    f_act_rmse = np.sqrt(mean_squared_error(y_train_act, fitted_act))

    fe_coefs = mdf.params
    re_coefs = mdf.random_effects


    y_train_mean = y_train_act.groupby(res_grouper).mean()
    fmean = fitted_act.groupby(res_grouper).mean()

    f_bias = fmean - y_train_mean
    f_bias_month = fitted_act.groupby(time_grouper).mean() - \
        y_train_act.groupby(time_grouper).mean()

    fitted_act = fitted_act.unstack()
    y_train_act = y_train_act.unstack()


    # pred_df = pd.DataFrame({k:v["pred"] for k,v in lvout_rt_results.items()})
    # fitt_df = pd.DataFrame({k:v["fitted"] for k,v in lvout_rt_results.items()})

    coefs = pd.DataFrame(mdf.random_effects)

    train_data = pd.DataFrame(dict(actual=y_train_act.stack(), model=fitted_act.stack()))

    res_scores = pd.DataFrame(index=reservoirs, columns=["NSE", "RMSE"])
    res_scores["NSE"] = train_data.groupby(res_grouper).apply(
        lambda x: r2_score(x["actual"], x["model"]))
    res_scores["RMSE"] = train_data.groupby(res_grouper).apply(
        lambda x: mean_squared_error(x["actual"], x["model"], squared=False))
    
    train_quant, train_bins = pd.qcut(train_data["actual"], 3, labels=False, retbins=True)
    train_data["bin"] = train_quant
    
    quant_scores = pd.DataFrame(index=[0,1,2], columns=["NSE", "RMSE"])
    quant_scores["NSE"] = train_data.groupby("bin").apply(
        lambda x: r2_score(x["actual"], x["model"]))
    quant_scores["RMSE"] = train_data.groupby("bin").apply(
        lambda x: mean_squared_error(x["actual"], x["model"], squared=False))

    print(res_scores.to_markdown(floatfmt="0.3f")) 
    output = dict(
        coefs=coefs,
        # fe_coefs=fe_coefs,
        f_act_score=f_act_score,
        f_act_rmse=f_act_rmse,
        f_bias=f_bias,
        f_bias_month=f_bias_month,
        res_scores=res_scores,
        quant_scores=quant_scores,
        data=dict(
            # X_test=X_test,
            # y_test=y_test,
            X_train=X_train,
            y_train=y_train,
            fitted_rel=train_data["model"],
            y_train_act_rel=train_data["actual"],
            # fitted_rel=fitted_act,
            # fitted_sto=fitted_sto,
            # y_test_rel_act=y_test_rel_act,
            # y_train_rel_act=y_train_rel_act,
            # y_test_sto_act=y_test_sto_act,
            # y_train_sto_act=y_train_sto_act,
            # predicted_act_rel=predicted_rel,
            # predicted_act_sto=predicted_sto,
            # forecasted=forecasted[["Release", "Storage", "Release_act", "Storage_act"]]
        )
    )
    return output

if __name__ == "__main__":
    pipeline()
