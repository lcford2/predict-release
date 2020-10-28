import pandas as pd
import numpy as np
import scipy
from IPython import embed as II
import pathlib
import matplotlib.pyplot as plt
from statsmodels.regression.mixed_linear_model import MixedLM, MixedLMParams
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor
from datetime import timedelta, datetime
import seaborn as sns
import sys

args = sys.argv

if len(args) > 1:
    year = int(args[1])
else:
    year = 1970


pickles = pathlib.Path("..", "pickles")

df = pd.read_pickle(pickles / "tva_dam_data.pickle")
df = df[df.index.get_level_values(0).year >= year]
# create a time series of previous days storage for all reservoirs
# df["Storage_pre"] = df.groupby(df.index.get_level_values(1))["Storage"].shift(1)
# drop the instances where there was no preceding values 
# (i.e. the first day of record for each reservoir because there is no previous storage)
df = df.dropna()
II()

sys.exit()

# endog is the dependent variable
endog = df["Release"]

# the fixed effects are Previous Storage and Inflow
exog = df[["Storage_pre", "Inflow"]]

# the random effects are reservoir characteristics 
# RunOfRiver, NaturalOnly, and PrimaryType
groups = ["RunOfRiver", "NaturalOnly", "PrimaryType"]
exog_re = df[groups]


def standardize(x):
    x = np.array(x)
    xmu = x.mean()
    xsd = x.std()
    return (x-xmu)/xsd, xmu, xsd
    

def feat_select(df, Xnames, Yname):
    lr = LinearRegression()
    efs = EFS(lr, min_features=1, max_features=len(Xnames),
                scoring="r2", 
                print_progress=True,cv=5,n_jobs=-1)
    
    X = df[Xnames]
    Y = df[Yname]
    results = efs.fit(X, Y)
    return results

def normality_check(exog, endog):
    fig, axes = plt.subplots(2,3)
    ax11, ax12, ax13, ax21, ax22, ax23 = axes.flatten()
    qqplot(exog["Storage_pre"], line="45", ax=ax11)
    qqplot(exog["Inflow"], line="45", ax=ax12)
    qqplot(endog, line="45", ax=ax13)
    sns.histplot(exog["Storage_pre"], ax=ax21, kde=True)
    sns.histplot(exog["Inflow"], ax=ax22, kde=True)
    sns.histplot(endog, ax=ax23, kde=True)
    ax11.set_title("Storage")
    ax12.set_title("Inflow")
    ax13.set_title("Release")
    plt.show()

Xnames = ["Storage_pre", "Inflow", "RunOfRiver", "NaturalOnly", "PrimaryType"]
# results = feat_select(df, Xnames, "Release")
# feature selection indicates that run of river and primary type 
# are not good for TVA (likely because their variance is too low)
exog = df.loc[:,["Storage_pre", "Inflow", "NaturalOnly"]]




II()
