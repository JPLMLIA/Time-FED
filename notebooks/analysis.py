#%%
%load_ext autoreload
%autoreload 2
%matplotlib inline

import logging
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from types import SimpleNamespace as SN

from utils import (
    load_weather,
    load_cn2
)

# Disable warnings
warnings.filterwarnings('ignore')

# Disable logger except for errors/exceptions
logger = logging.getLogger()
logger.setLevel(logging.ERROR)

#%% Commonly used arguments
dist_plot = {
    'kind'   : 'bar',
    'ylabel' : 'Count',
    'xlabel' : 'Distance',
    'logy'   : True,
    'figsize': (30, 10)
}

#%% Analysis functions

def protect(func):
    """
    Protects a function from raising exceptions to allow for graceful exiting
    """
    def _wrap(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f'Function {func.__name__}() raised an exception:\n{e}')
    return _wrap

@protect
def analyze(df, quantile=0.001, std=3):
    """
    Analyzes a provided dataframe with some generic statistics

    Assumptions:
    - The index is dtype DateTime

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to perform analysis on
    quantile : float
        The lower quantile to extract, the upper will be 1-this
    std : int
        The number of standard deviations away when determining outliers
    """
    stats = SN()

    # Some integrity checks
    assert df.index.is_all_dates
    df = df.sort_index()

    ## Time analysis
    stats.time   = time = SN()
    time.total   = df.index.size
    time.first   = df.index[0]
    time.last    = df.index[-1]
    time.cadence = df.index[1:] - df.index[:-1].values
    time.counts  = time.cadence.value_counts()
    time.mean    = time.cadence.mean()

    ## NaN analysis
    stats.df   = sndf = SN()
    nan_df     = df.isna()
    sndf.total = nan_df.sum()
    sndf.corr  = (nan_df * 1).corr(method='pearson')

    ### Per column analysis
    stats.columns = SN()
    for col in df.columns:
        series = df[col]
        vars(stats.columns)[col] = ref = SN()

        # Distance between nans
        nans = ref.nans = SN()
        nan_inds     = series[series.isna()].index
        nans.dist    = nan_inds[1:] - nan_inds[:-1].values
        nans.mean    = nans.dist.mean()
        nans.count   = nans.dist.value_counts()
        nans.total   = nan_inds.size
        nans.percent = (nans.total / df[col].size) * 100

        ## Value analysis
        vals = ref.values = SN()
        vals.mean = series.mean()

        # Outliers using std
        vals.stddevs = (series - series.mean()).abs()
        vals.std     = series.std()
        vals.filter  = vals.std * std # This is the value to use as the outlier filter

    return stats

#%% Load data
path = 'Weather Data/'

#df = load_r0(path, round=True)
#df = load_cn2(path, round=True)
df = load_weather(path, interpolate=False)

df

#%% Perform analysis
stats = analyze(df)
interpret(stats)

#%% Time
print(f'There are {stats.time.total} rows of timestamps starting from {stats.time.first} to {stats.time.last}.')
print(f'For this range, the average cadence was {stats.time.mean}. Below is a graph showing other differences in cadences discovered.\n')
ax = stats.time.counts.plot(title='Distance between timestamps', **dist_plot)

#%% Global NaNs
print(f'The total NaNs for each column are as follows:\n{stats.df.total}')
print(f'\nThe correlation of these nans between columns are provided in the following heatmap:\n')
ax = sns.heatmap(stats.df.corr, annot=True, vmin=-1, vmax=1, cmap='RdYlBu')
# Correlation matrix between features

#%% Column NaNs
def interpret_columns():
    for col in vars(stats.columns):
        ref = vars(stats.columns)[col]
        #print(list(vars(ref.nans).keys()))
        print(f'Column {col} is {ref.nans.percent:.2f}% NaNs ({ref.nans.total} / {stats.time.total}).')
        print(f'The average distance between NaNs is {ref.nans.mean}. Below is a graph showing the various differences in distances between NaNs.\n')

        if ref.nans.count.size > 150:
            print(f'The following graph may not be useful as there are >150 different distances between NaNs for this column.')
            print(f'As such, a preview of the series from which this plot is generated will be provided.\n {ref.nans.count}')
            print('\nThe graph is useful for gaining a feel on how often there\'s differing distances between NaNs.')

        ax = ref.nans.count.plot(title=f'Distance between NaNs for {col}', **dist_plot)
        plt.show()

        N = int(ref.values.std.filter / ref.values.std.std)
        print(f'Outlier values are discovered using two methods: standard deviation and quantiles')
        print(f'Using the standard deviation ({ref.values.std.std:.2f}), values above/below N={N} standard deviations away are flagged as outliers ({N}*{ref.values.std.std:.2f}={ref.values.std.filter:.2f} = the "filter").')
        print(f'The values above this filter are:\n{ref.values.std.above}')
        print(f'\nFirst 50 values, graphed:')
        ref.values.std.above.plot(**dist_plot)
        plt.show()

        print(f'\nThe values below this filter are:\n{ref.values.std.below}')
        print(f'\nFirst 50 values, graphed:')
        ref.values.std.below.plot(**dist_plot)
        plt.show()


        yield

col_analysis = interpret_columns()
# Distance between non-nans, see x-axis as ascending
#%% Execute this line to generate the analysis for the next column in the dataframe
next(col_analysis)

#%%
ref.values.std.below

# plot histograms then colour the 5 sigma


stats.columns.pressure.values
#%%
above = stats.columns.pressure.values.stddevs > stats.columns.pressure.values.filter
upper = (df.pressure.loc[above[above].index[0]], df.pressure.loc[above[above].index[-1]])
below = stats.columns.pressure.values.stddevs < stats.columns.pressure.values.filter
lower = (df.pressure.loc[above[below].index[0]], df.pressure.loc[above[below].index[-1]])
upper
lower
#%%
ax = df.pressure.hist(bins=100)
ax.axvspan(*lower, color='red', alpha=0.5)
ax.axvspan(*upper, color='red', alpha=0.5)
#plt.show()

dir(ax)
min, max = ax.get_xlim()
mean = df.pressure.mean()
std = df.pressure.std()
lower, upper = mean - 3 * std, mean + 3 * std
#%%
from matplotlib.patches import Patch
from matplotlib.ticker  import FormatStrFormatter

n_bins = 109
sigmas = {
    1: 'blue',
    2: 'yellow',
    3: 'red',
    4: 'green',
    5: 'purple'
}

legend = [Patch(facecolor=colour, label=f'{sigma}σ', alpha=.3, edgecolor='black') for sigma, colour in sigmas.items()]

for col in df:
    ticks = set()

    min  = df[col].min()
    max  = df[col].max()
    mean = df[col].mean()
    std  = df[col].std()
    ax   = df[col].hist(bins=n_bins, log=True, color='grey', figsize=(20, 10))
    lower, upper = zip(ax.get_xlim(), )

    # Highlight the sigma regions
    for sigma, colour in sigmas.items():
        lower_x2, upper_x1 = (mean - sigma * std, mean + sigma * std)
        lower_x1, upper_x2 = (mean - (sigma + 1) * std, mean + (sigma + 1) * std)
        if min < lower_x2:
            ax.axvspan(lower_x1, lower_x2, color=colour, alpha=0.3)
            ticks.add(lower_x1); ticks.add(lower_x2)
        if upper_x1 < max:
            ax.axvspan(upper_x1, upper_x2, color=colour, alpha=0.3)
            ticks.add(upper_x1); ticks.add(upper_x2)

    # Plot the stddev
    ax.axvspan(mean, mean, color='red')

    # Change the x-axis ticks to be meaningful
    ticks.add(mean)
    ax.set_xticks(list(ticks), minor=False)
    ax.xaxis.set_minor_formatter(FormatStrFormatter("%.2f"))
    ax.tick_params(which='minor', direction='inout', length=7, labeltop=True, labelbottom=False, top=True)
    ax.set_xticks([min, max], minor=True)
    #ax.text(mean, 0, 'mean', verticalalignment='center')

    # Set some texts
    ax.set_title(f'Histogram of {col}, bins={n_bins}')
    ax.set_ylabel('Value Count')
    ax.set_xlabel('Value')
    ax.legend(handles=legend, loc='upper right')
    plt.savefig(f'plots/5sigma_hist_{col}.png')
    plt.show()
#ax.text(lower[1] - (lower[1] - lower[0])/2, ax.get_ylim()[1]/2, 'boop')

#%%
