# import plotting librarys
import os # i need OS here so I can add PROJ_LIB to path for Basemap
import matplotlib.pyplot as plt
import matplotlib.gridspec as GS
os.environ["PROJ_LIB"] = r"C:\\Users\\lcford2\AppData\\Local\\Continuum\\anaconda3\\envs\\sry-env\\Library\\share"
from mpl_toolkits.basemap import Basemap
import seaborn as sns
# import data and math libraries
import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr
# import generic libraries
from IPython import embed as II
import pickle
import pathlib
import glob
import argparse
import calendar
# import my helper functions
from helper_functions import read_tva_data

# setup plotting environment
plt.style.use("ggplot")
sns.set_context("talk")

# indicate where certain data files are
results_dir = pathlib.Path("../results")
multi_level_dir = results_dir / "multi-level-results"
GIS_DIR = pathlib.Path("G:/My Drive/ms_yr_1/GIS")

def load_results(args):
    """Loads pickled results from modeled runs

    :param args: Namespace from argparse with all CLI args
    :type args: <argparse.Namespace>
    :return: Results object, either a dictionary or a pandas DataFrame
    :rtype: [dict, pandas.DataFrame]
    """
    if args.results_pickle:
        file = multi_level_dir / args.results_pickle
    elif args.filtered:
        files = ["NaturalOnly-RunOfRiver_filter_ComboFlow.pickle",
                 "NaturalOnly-RunOfRiver_filter_NaturalFlow.pickle"]
        files = [multi_level_dir/args.model_path/i for i in files]
        data = combine_filtered_data(files)
        II()
        return data
    else:
        file = query_file(args)
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data

def combine_filtered_data(data_files):
    data_dicts = []
    for file in data_files:
        with open(file, "rb") as f:
            data_dicts.append(pickle.load(f))
    tmp = data_dicts[0]
    re_coefs = tmp["re_coefs"]
    fe_coefs = tmp["fe_coefs"]
    data = tmp["data"]
    for data_dict in data_dicts[1:]:
        re_coefs.update(data_dict["re_coefs"])
        fe_coefs.update(data_dict["fe_coefs"])
        for key, value in data_dict["data"].items():
            data[key] = data[key].append(value).sort_index()
    return {"re_coefs":re_coefs, "fe_coefs":fe_coefs, "data":data}

def parse_args(plot_functions):
    """Argument parsing function for the CLI

    :param plot_functions: List of available plotting functions in the namespace
    :type plot_functions: list
    :return: Namespace with user supplied and default arguments
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Plot results for multi-level models."
    )
    parser.add_argument("-f", "--file", dest="results_pickle", default=None,
                        help="File name for results pickle. Expected to be in ../results/mixed-level-results"
                        )
    parser.add_argument("--model_path", dest="model_path", default="with_month_vars", 
                        help="Specify which directory to look for results pickles in.")
    parser.add_argument("-p", "--plot_func", choices=plot_functions.keys(), default=None, dest="plot_func",
                        help="Provide the name of the desired plot. If none is provided an IPython interpreter will be opened.")
    parser.add_argument("-g", "--group_labels", action="store_true", dest="group_labels",
                        help="Flag to include group labels on plots. Can act differently on different plots.")
    parser.add_argument("--res_names", dest="res_names", nargs="+", default=None, 
                        help="Specify reservoir names to only plot those reservoirs for certian plots.")
    parser.add_argument("-F", "--forecasted", dest="forecasted", action="store_true", 
                        help="Use forecasted data instead of simple prediction.")
    parser.add_argument("-A", "--all_res", dest="all_res", action="store_true",
                        help="Look at results for all reservoirs, not just ones used to fit model.")
    parser.add_argument("-M", "--metric", dest="metric", choices=["bias", "nse", "corr", "mae", "rmse"], default="nse",
                        help="Choose what metric should be plotted for certain plots.")
    parser.add_argument("--relative", action="store_true", 
                        help="Flag for computing relative error metrics instead of absolute.")
    parser.add_argument("--filtered", action="store_true",
                        help="Specify if you want to plot filtered results. e.g. Model runs that only consider natural or combo flows.")
    parser.add_argument("--subplot_type", default="scatter", choices=["scatter", "ratio_bars", "sep_bars"],
                        help="Specify what sort of plot you want. This works within certain plotting functions for further control.")
    parser.add_argument("--storage", action="store_true", 
                        help="Flag to plot storages instead of release. Only works with the --forecasted (-F) flag.")
    parser.add_argument("--error", action="store_true",
                        help="Plot error instead of actual values for time series plot.")
    args = parser.parse_args()
    return args

def query_file(args):
    """Used to ask the user what results file they want to use for plotting

    :param args: Arguments specified by user
    :type args: argparse.Namespace
    :raises ValueError: An integer must be provided for file selection.
    :raises ValueError: The integer must be one that is in the menu.
    :return: Path to user specified file
    :rtype: pathlib.Path
    """
    model_path = args.model_path
    if args.all_res:
        files = glob.glob((multi_level_dir/model_path/"*all_res.pickle").as_posix())
    else:
        files = [i for i in glob.glob((multi_level_dir/model_path/"*.pickle").as_posix()) if "all_res" not in i]
    if len(files) == 1:
        print(f"\nUsing data file: {files[0]}\n")
        file = pathlib.Path(files[0])
    else:
        for i, file in enumerate(files):
            print(f"[{i}] {file}")
        selection = input("Enter the number of the file you wish to use: ")
        if not selection.isnumeric():
            raise ValueError("Must provide an integer corresponding to your selection.")
        elif int(selection) > len(files):
            raise ValueError("Number provided is not valid.")
        else:
            file = pathlib.Path(files[int(selection)])
    return file

def find_plot_functions(namespace):
    """Parses the namespace for functions that start with "plot"

    :param namespace: list of items in the namespace, must be passed because it cannot be determined from within the function
    :type namespace: list
    :return: Dictionary of {simple name:function name} for available plotting functions. simple name is what will be provided by the user.
    :rtype: dict
    """
    plot_functions = filter(lambda x: x[:4] == "plot", namespace)
    plot_name_dict = {"_".join(i.split("_")[1:]):i for i in plot_functions}
    return plot_name_dict

def select_correct_data(data, args):
    """Pulls out modeled and actual time series as well as the groupnames from the
    data as certain arguments specify

    :param data: Object with results data
    :type data: [dict, pandas.DataFrame]
    :param args: CLI Arguments
    :type args: argparse.Namespace
    :return: Modeled DataFrame, Actual DataFrame, array of groupnames
    :rtype: [pandas.DataFrame, pandas.DataFrame, numpy.Array]
    """
    if args.storage:
        key = "Storage_act"
    else:
        key = "Release_act"
    if args.forecasted:
        if args.all_res:
            modeled = data["f{key}"].unstack()
            actual = data[f"{key}_obs"].unstack()
            groupnames = data["compositegroup"]
        else:
            modeled = data["data"]["forecasted"][f"{key}"].unstack()
            if key == "Storage_act":
                df = read_tva_data()
                actual = df["Storage"].loc[modeled.index].unstack()
            else:
                actual = data["data"]["y_test_act"].unstack()
            groupnames = data["data"]["X_test"]["compositegroup"]
    else:
        modeled = data["data"]["predicted_act"].unstack()
        actual = data["data"]["y_test_act"].unstack()
        groupnames = data["data"]["X_test"]["compositegroup"]

    return modeled, actual, groupnames

def bias(X, Y):
    """Calculate the bias between X and Y.
    This calculation is setup to provide a postive value for an overestimation,
    assuming X is the actual data and Y is the modeled data.
    
    :math:`bias = \overline{Y} - \overline{X}`

    :param X: Input array.
    :type X: array_like
    :param Y: Input array.
    :type Y: array_like
    """
    return np.mean(Y) - np.mean(X)

def metric_linker(metric_arg):
    linker = {
        "bias" : bias,
        "nse"  : r2_score,
        "corr" : lambda x,y: pearsonr(x,y)[0],
        "mae"  : mean_absolute_error,
        "rmse" : lambda x,y : np.sqrt(mean_squared_error(x,y))
    }
    return linker[metric_arg]

def plot_score_bars(data, args):
    modeled, actual, groupnames = select_correct_data(data, args)
    groupnames = groupnames.unstack().values[0, :]
    scores = pd.DataFrame(index=modeled.columns, columns=["Score", "GroupName"], dtype="float64")
    for i, index in enumerate(scores.index):
        score = r2_score(actual[index], modeled[index])
        scores.loc[index, "Score"] = score
        scores.loc[index, "GroupName"] = groupnames[i]
    
    scores = scores.sort_values(by=["GroupName", "Score"])
    # scores = pd.read_pickle("../pickles/std_ratio_st_inflow.pickle")

    order = ['Ocoee3', 'Wilbur', 'Douglas', 'Cherokee', 'Hiwassee', 'Boone',
             'Kentucky', 'Ocoee1', 'FtPatrick', 'Apalachia', 'MeltonH', 'WattsBar',
             'FtLoudoun', 'Wilson', 'Nikajack', 'Chikamauga', 'Guntersville',
             'Wheeler', 'Pickwick', 'BlueRidge', 'TimsFord', 'Watauga', 'Nottely',
             'Chatuge', 'SHolston', 'Norris', 'Fontana']
    scores = scores.loc[order]
    # groupmeans = scores.groupby("GroupName").mean()
    fig, ax = plt.subplots(1,1)
    scores["Score"].plot.bar(ax=ax, width=0.8)
    # scores.plot.bar(ax=ax, width=0.8)
    ticks = ax.get_xticks()
    if getattr(args, "group_labels", None):
        for tick, name in zip(ticks, scores["GroupName"].values.tolist()):
            ax.text(tick, 0.1, name, rotation=90, va="bottom", ha="center")
    # else:
    #     groupmeans = scores.groupby("GroupName").mean()
    #     groups = groupmeans.index.tolist()
    #     groupmeans = groupmeans.values.flatten()
    #     groupcount = scores.groupby("GroupName").count().cumsum().values.flatten() / modeled.shape[1]
    #     last = 0
    #     modif = 0.4 / modeled.shape[1]
    #     for i, mean in enumerate(groupmeans):
    #         ax.axhline(mean, last + modif, groupcount[i]+modif, c="b")
    #         group = groups[i]
    #         # ax.text((last+modif + groupcount[i])*modeled.shape[1]/2, mean, group, ha="center", va="bottom")
    #         last = groupcount[i]
    ax.set_ylabel("NSE")
    # ax.set_ylabel(r"$\sigma_{St.}/\sigma_{In.}$")
    # ax.set_ylim((0.0, 1.0433202579442722))
    plt.subplots_adjust(
        top=0.88,
        bottom=0.205,
        left=0.11,
        right=0.9,
        hspace=0.2,
        wspace=0.2
    )
    plt.show()


def plot_std(data, args):
    modeled, actual, groupnames = select_correct_data(data, args)
    
    data = read_tva_data()
    quarter_std = data.groupby([
        data.index.get_level_values(0).quarter, 
        data.index.get_level_values(1)
    ]).std()

    quarter_std = quarter_std.loc[:,["Storage", "Net Inflow"]]    
    quarter_std["Ratio"] = quarter_std["Storage"]/quarter_std["Net Inflow"]
    order = ['Ocoee3', 'Wilbur', 'Douglas', 'Cherokee', 'Hiwassee', 'Boone',
             'Kentucky', 'Ocoee1', 'FtPatrick', 'Apalachia', 'MeltonH', 'WattsBar',
             'FtLoudoun', 'Wilson', 'Nikajack', 'Chikamauga', 'Guntersville',
             'Wheeler', 'Pickwick', 'BlueRidge', 'TimsFord', 'Watauga', 'Nottely',
             'Chatuge', 'SHolston', 'Norris', 'Fontana']

    idx = pd.IndexSlice
    groupnames = groupnames.loc[idx[groupnames.index[0],:]]
    groupnames.index = groupnames.index.get_level_values(1)
    quarter_std["Group"] = [groupnames[i] for i in quarter_std.index.get_level_values(1)]
    names = groupnames.unique()

    titles = ["JFM", "AMJ", "JAS", "OND"]
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    axes = axes.flatten()

    plot_type = args.subplot_type

    for i, (ax, title) in enumerate(zip(axes, titles)):
        series = quarter_std.loc[
            quarter_std.index.get_level_values(0) == i+1, :]
        series.index = series.index.get_level_values(1)
        if plot_type == "scatter":
            for name in names:
                plt_series = series[series["Group"] == name]
                ax.scatter(plt_series["Storage"], plt_series["Net Inflow"], label=name)
        elif plot_type == "ratio_bars":
            series.loc[order,"Ratio"].plot.bar(ax=ax, width=0.8)
        elif plot_type == "sep_bars":
            series.loc[order, ["Storage", "Net Inflow"]
                       ].plot.bar(ax=ax, width=0.8)
            ax.get_legend().remove()
        ax.set_title(title)

    if plot_type in ["scatter", "sep_bars"]:
        axes[-2].legend(loc="best")
    
    bottom = 0.191
    if plot_type == "scatter":
        bottom = 0.121

    pos = dict(
        top=0.945,
        bottom=bottom,
        left=0.063,
        right=0.986,
        hspace=0.203,
        wspace=0.044
    )

    yoffset = 0.01
    xoffset = 0.01
    ypos = (pos["top"]+pos["bottom"])/2
    xpos = (pos["right"]+pos["left"])/2
    
    if plot_type == "scatter":
        fig.text(xpos, yoffset, r"$\sigma_{Storage}$", ha="center")
        fig.text(xoffset, ypos, r"$\sigma_{Inflow}$", rotation=90, va="center")
    elif plot_type == "ratio_bars":
        fig.text(
            xoffset, ypos, r"$\sigma_{Storage}/\sigma_{Inflow}$", rotation=90, va="center")
    elif plot_type == "sep_bars":
        fig.text(xoffset, ypos, r"$\sigma$", rotation=90, va="center")

    plt.subplots_adjust(**pos)
    plt.show()



def determine_grid_size(N):
    if N <= 3:
        return (N,1)
    else:
        poss_1 = [(i, N//i) for i in range(2, int(N**0.5) + 1) if N % i == 0]
        poss_2 = [(i, (N+1) // i) for i in range(2, int((N+1)**0.5) + 1) if (N+1) % i == 0]
        poss = poss_1 + poss_2
        min_index = np.argmin([sum(i) for i in poss])
        return poss[min_index]

def plot_month_coefs(data, args):
    # modeled, actual, groupnames = select_correct_data(data, args)
    re_coefs = pd.DataFrame(data["re_coefs"])
    try:
        re_coefs = re_coefs[calendar.month_abbr[1:]]
    except KeyError as e:
        re_coefs = re_coefs.T[calendar.month_abbr[1:]]
    else:
        print("No month coefficients to plot.")
        return

    format_dict = {
        "Storage_pre":{"marker":"o", "label":"Prev. St."},
        "Release_pre":{"marker":"s", "label":"Prev. Rel."},
        "Net Inflow":{"marker":"X", "label":"Inflow"},
        "NaturalOnly":{"marker":"^", "label":"Inflow Type"},
        "RunOfRiver":{"marker":"d", "label":"Storage Type"},
        "Fraction_Storage":{"marker":"o", "label":"St. Contrib."},
        "Fraction_Release":{"marker":"s", "label":"Rel. Contrib."},
        "Fraction_Net Inflow":{"marker":"X", "label":"Inf. Contrib."},
        "ComboFlow-RunOfRiver": {"marker": "o", "label": "Prev. St."},
        "ComboFlow-StorageDam": {"marker": "s", "label": "Prev. Rel."},
        "NaturalFlow-StorageDam": {"marker": "X", "label": "Inflow"},
    }

    fig, ax = plt.subplots(nrows=1, ncols=1)
    
    x = range(1,13)
    for key in re_coefs.index:
        formats = format_dict[key]
        y = re_coefs.loc[key]
        ax.plot(x, y, **formats)
    ax.set_xticks(list(x))
    ax.set_xticklabels(calendar.month_abbr[1:])
    ax.set_ylabel("Fitted Coefficients")
    ax.legend(loc="lower right")
    ylim = ax.get_ylim()
    if ylim[0] > -0.4:
        ax.set_ylim((-0.4, ylim[1]))
    plt.show()


def plot_time_series(data, args):
    modeled, actual, groupnames = select_correct_data(data, args)
    groupnames = groupnames.unstack().values[0, :]
    if args.res_names:
        res_names = np.array(args.res_names)
    else:
        res_names = modeled.columns

    grid_size = determine_grid_size(res_names.size)
    if res_names.size > 4:
        sns.set_context("paper")
    fig, axes = plt.subplots(*grid_size)
    axes = axes.flatten()
    error = args.error
    means = actual.mean()
    for ax, col in zip(axes, res_names):
        if error:
            error_series = (modeled[col] - actual[col]) / means[col]
            error_series.plot(ax=ax, label="Error")
        else:
            actual[col].plot(ax=ax, label="Observed")
            modeled[col].plot(ax=ax, label="Modeled")
        ax.set_title(col)
    handles, labels = axes[0].get_legend_handles_labels()
    if res_names.size <= 6:
        handles, labels = axes[0].get_legend_handles_labels()
        labels = ["Observed", "Modeled"]
        axes[0].legend(handles, labels, loc="best")
    else:
        axes[-1].legend(handles, labels, loc="center", prop={"size":18})
        axes[-1].set_axis_off()

    left_over = axes.size - res_names.size
    if left_over > 0:
        for ax in axes[-left_over:]:
            ax.set_axis_off()
    
    plt.show()

def join_test_train(data):
    y_test = data["data"]["y_test_act"]
    y_train = data["data"]["y_train_act"]
    y = y_train.append(y_test)
    return y.sort_index().unstack()

def plot_acf(data, args):
    from statsmodels.graphics.tsaplots import plot_acf
    modeled, actual, groupnames = select_correct_data(data, args)
    if not args.storage:
        actual = join_test_train(data)
    
    groupnames = groupnames.unstack().values[0, :]
    if args.res_names:
        res_names = np.array(args.res_names)
    else:
        res_names = modeled.columns

    grid_size = determine_grid_size(res_names.size)
    if res_names.size > 4:
        sns.set_context("paper")
    fig, axes = plt.subplots(*grid_size)
    axes = axes.flatten()
    lags = 30
    for ax, col in zip(axes, res_names):
        output = plot_acf(actual[col], ax=ax, lags=lags, use_vlines=True, 
                          title=col, zero=False, alpha=None)
        ax.set_xticks(range(0,lags+2,5))
    
    left_over = axes.size - res_names.size
    if left_over > 0:
        for ax in axes[-left_over:]:
            ax.set_axis_off()
    
    plt.show()
    

def plot_monthly_metrics(data, args):
    modeled, actual, groupnames = select_correct_data(data, args)
    groupnames = groupnames.unstack().values[0, :]
    columns = [i for i in calendar.month_abbr[1:]] + ["GroupName"]
    metrics = pd.DataFrame(index=modeled.columns, columns=columns, dtype="float64")
    metric_calc = metric_linker(args.metric)
    for i, index in enumerate(metrics.index):
        for j, month in enumerate(columns[:-1]):
            mod = modeled.loc[modeled.index.month == j + 1, index]
            act = actual.loc[actual.index.month == j + 1, index]
            value = metric_calc(act, mod)
            metrics.loc[index, month] = value
        metrics.loc[index, "GroupName"] = groupnames[i]
    
    if args.relative and args.metric not in ("nse", "corr"):
        rel_mets = (metrics.loc[:,columns[:-1]].T / actual.mean()).T * 100
        metrics.loc[:, columns[:-1]] = rel_mets
        label = f"% {args.metric.upper()}"
    else:
        label = args.metric.upper()
    # metrics = metrics.sort_values(by=["GroupName"])
    grouped = metrics.groupby("GroupName").mean()
    N = grouped.index.size
    # if N > 2:
    #     sns.set_context("paper")
    grid_size = determine_grid_size(N)

    fig, axes = plt.subplots(*grid_size)
    axes = axes.flatten()
    for ax, group in zip(axes, grouped.index):
        grouped.loc[group].plot.bar(ax=ax, width=0.8)
        ax.set_title(group)
        ax.set_ylabel(label)
    fig.align_ylabels()
    hspace = 0.3
    if N > 2:
        hspace = 0.5
    
    plt.subplots_adjust(
        top=0.88,
        bottom=0.11,
        left=0.11,
        right=0.9,
        hspace=hspace,
        wspace=0.2
    )
    plt.show()
    
    
def setup_map(ax, control_area=True, coords=None):
    if not coords:
        west, south, east, north = -90.62, 32.08, -80.94, 37.99
    else:
        west, south, east, north = coords
    m = Basemap(resolution="c",
                llcrnrlon=west,
                llcrnrlat=south,
                urcrnrlon=east,
                urcrnrlat=north,
                ax=ax)
    states_path = GIS_DIR / "cb_2017_us_state_500k"
    river_path = GIS_DIR / "TVA_RIVERS" / "combined"
    bound_path = GIS_DIR / "Bound_fixed" / "Region_06_boundary"
    control_area_path = GIS_DIR / "BA-Control_Areas-electricity" / "tva_control_area"

    mbound = m.drawmapboundary(fill_color='white')
    states = m.readshapefile(states_path.as_posix(), "states")
    streams = m.readshapefile(river_path.as_posix(),
                              "streams", color="b", linewidth=1)
    bound = m.readshapefile(bound_path.as_posix(), "bound", color="#FF3BC6")
    if control_area:
        control_areas = m.readshapefile(control_area_path.as_posix(), "control_areas",
                                        linewidth=1.2, color="#5d9732")
    bound_lines = bound[4]
    stream_lines = streams[4]
    bound_lines.set_facecolor("#FF3BC6")
    bound_lines.set_alpha(0.4)
    bound_lines.set_zorder(2)

    # mid_lon = np.mean([west, east]) #-85.78
    # mid_lat = np.mean([south, north]) #35.04
    mid_lon = -85.78
    mid_lat = 35.04
    ax.text(mid_lon - 0.8, mid_lat + 0.5, 'TN', fontsize=18,
            ha="center")  # TN
    ax.text(mid_lon + 1.1, mid_lat + 2.1, 'KY', fontsize=18,
            ha="center")  # KY
    ax.text(mid_lon - 3.4, mid_lat - 1.6, 'MS', fontsize=18,
            ha="center")  # MS
    ax.text(mid_lon - 1.3, mid_lat - 1.6, 'AL', fontsize=18,
            ha="center")  # AL
    ax.text(mid_lon + 1.6, mid_lat - 1.6, 'GA', fontsize=18,
            ha="center")  # GA
    ax.text(mid_lon + 3.7, mid_lat + 1.9, 'VA', fontsize=18,
            ha="center")  # VA
    ax.text(mid_lon + 4.1, mid_lat + 0.5, 'NC', fontsize=18,
            ha="center")  # NC
    ax.text(mid_lon + 4.1, mid_lat - 1.4, 'SC', fontsize=18,
            ha="center")  # SC


def plot_score_map(data, args):
    modeled, actual, groupnames = select_correct_data(data, args)
    groupnames = groupnames.unstack().values[0, :]
    fig = plt.figure()
    spec = GS.GridSpec(ncols=1, nrows=2, height_ratios=[20, 1])
    ax = fig.add_subplot(spec[0])
    leg_ax = fig.add_subplot(spec[1])
    leg_ax.axis("off")

    west, south, east, north = -89.62, 33.08, -80.94, 37.99
    coords = (west, south, east, north)

    m = setup_map(ax, control_area=False, coords=coords)
    reservoirs = GIS_DIR.joinpath("kml_converted", "Hydroelectric-point.shp")
    res_loc = gpd.read_file(reservoirs)
    res_drop = ["Great Falls Dam", "Ocoee #2 Dam",
                "Tellico Dam", "Nolichucky Dam", 
                "Raccoon Mountain Pumped-Storage Plant"]
    res_loc = res_loc[~res_loc["Name"].isin(res_drop)]

    res_name_map = {}
    with open("loc_name_map.csv", "r") as f:
        for line in f:
            line = line.strip("\r\n")
            old, new = line.split(",")
            res_name_map[old] = new
        
    res_loc["Name"] = [res_name_map[i] for i in res_loc["Name"]]

    name_conv = {}
    with open("name_conversion.txt", "r") as f:
        for line in f:
            name_t, name_act = line.split()
            name_conv[name_t.split("_")[0]] = name_act.split("_")[0]

    
    scores = pd.DataFrame(index=modeled.columns, columns=[
                          "Score", "GroupName"], dtype="float64")
    for i, index in enumerate(scores.index):
        score = r2_score(actual[index], modeled[index])
        scores.loc[index, "Score"] = score
        scores.loc[index, "GroupName"] = groupnames[i]

    sizes = np.array([scores.loc[name_conv[i], "Score"] for i in res_loc["Name"]])
    max_size = 800
    max_score = sizes.max()
    scale = max_size/max_score
    # plot_sizes = sizes * scale
    plot_sizes = np.power(max_size, sizes)
    ax.scatter(*zip(*[(i.x, i.y) for i in res_loc["geometry"]]),
               facecolor="#00bce5", marker="o", s=plot_sizes, alpha=1, zorder=4, edgecolor="k")
    
    legend_scores = np.linspace(sizes.min(), sizes.max(), 4)
    legend_sizes = np.power(800, legend_scores)
    legend_markers = [plt.scatter(
        [], [], s=i, edgecolors="k", c="#00bce5", alpha=1
    ) for i in legend_sizes]
    leg_kwargs = dict(ncol=4, frameon=True, handlelength=1, loc='center', borderpad=1,
                          scatterpoints=1, handletextpad=1, labelspacing=1, title="NSE")
    ax_legend = leg_ax.legend(legend_markers, [f"{i:.2f}" for i in legend_scores], **leg_kwargs)
    plt.show()

if __name__ == "__main__":
    namespace = dir()
    plot_functions = find_plot_functions(namespace)
    args = parse_args(plot_functions=plot_functions)
    data = load_results(args)
    if args.plot_func:
        globals()[plot_functions[args.plot_func]](data, args)
    else:
        II()
