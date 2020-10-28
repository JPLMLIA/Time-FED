#%%
%load_ext autoreload
%autoreload 2
%matplotlib inline

import logging
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from matplotlib.patches import Patch
from matplotlib.ticker  import FormatStrFormatter
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

#%% Analysis functions
valCount = {
    'kind'   : 'bar',
    'ylabel' : 'Count',
    'logy'   : True,
    'figsize': (30, 10)
}
override = {
    'Cn2': {
        'logx': True
    }
}

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

def show(save=False):
    """
    Shows a plot, saves it if set
    """
    plt.tight_layout()
    if save:
        plt.savefig(save)
    plt.show()

def generate_histogram(col, df, n_bins=109, save=False):
    """
    Generates a histogram for the provided column and highlights the sigma
    regions.
    """
    sigmas = {
        1: 'red',
        2: 'orange',
        3: 'yellow',
        4: 'green',
        5: 'blue'
    }

    legend = [Patch(facecolor=colour, label=f'{sigma}σ', alpha=.3, edgecolor='black') for sigma, colour in sigmas.items()]

    ticks = set()

    # Gather stats
    min  = df[col].min()
    max  = df[col].max()
    mean = df[col].mean()
    std  = df[col].std()

    # Create the histogram
    ax = df[col].hist(bins=n_bins, log=True, color='grey', figsize=(30, 10))

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
    ax.axvspan(min, min, color='black')
    ax.axvspan(max, max, color='black')

    # Change the x-axis ticks to be meaningful
    ticks.add(mean)
    ax.set_xticks(list(ticks), minor=False)
    ax.xaxis.set_minor_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(which='minor', length=0, labeltop=True, labelbottom=False, top=True)
    ax.set_xticks([min, max], minor=True)
    ax.grid(axis='x', b=False)

    # Set some texts
    ax.set_title(f'Histogram of {col}, bins={n_bins}')
    ax.set_ylabel('Value Count')
    ax.set_xlabel('Value')
    ax.legend(handles=legend, loc='upper right')

    if save:
        save = f'{save}/5sigma_hist_{col}.png'
    show(save)

@protect
def analyze(df, drop=True, save=False, name=None):
    """
    Analyzes a provided dataframe with some generic statistics

    Assumptions:
    - The index is dtype DateTime

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to perform analysis on
    """
    if save and not name:
        print('The `save` parameter was provided but no name was given, please provide a name for this dataset (eg. "weather")')
        return

    # Special printing
    fprint = lambda s: print(f"{'='*80}\n{s} |\n{'-'*(len(s)+2)}")

    # Some integrity checks
    assert df.index.is_all_dates
    df = df.sort_index()

    # Drop rows that are entirely empty
    if drop:
        df = df.dropna(axis=0, how='all')

    ## Time analysis
    fprint('Time Analysis')

    first   = df.index[0]
    last    = df.index[-1]
    cadence = df.index[1:] - df.index[:-1].values - pd.Timedelta('1 minute')

    print(f'There are {df.index.size} rows of timestamps ranging from {first} to {last}.')
    print(f'For this range, the average cadence was {cadence.mean()}. Below is a graph showing other differences in cadences discovered.\n')

    ax = cadence.value_counts().sort_index().plot(rot=45, title='Distance between timestamps', xlabel='Distance', **valCount)
    show(f'{save}/{name}_time_cadence.png' if save else False)

    ## Value analysis
    fprint('Value Analysis')

    print(f'The total values for each column are as follows:\n{~df.isna().sum()}')
    print(f'\nThe correlation of these values between columns are provided in the following heatmap:\n')

    fig, ax = plt.subplots(figsize=(5, 5))
    ax = sns.heatmap(df.corr(method='pearson'), annot=True, vmin=-1, vmax=1, cmap='RdYlBu', ax=ax)
    ax.set_title('Correlation of values between columns{f" for {name}" if name else ""}')
    show(f'{save}/{name}_value_corrs.png' if save else False)

    ## NaN analysis
    fprint('NaN Analysis')

    nandf    = df.isna()
    nantotal = nandf.sum()
    nancorr  = (nandf * 1).corr(method='pearson')

    print(f'The total NaNs for each column are as follows:\n{nantotal}')
    if nancorr.any().any():
        print(f'\nThe correlation of the positions of these nans between columns are provided in the following heatmap:\n')

        fig, ax = plt.subplots(figsize=(5, 5))
        ax = sns.heatmap(nancorr, annot=True, vmin=-1, vmax=1, cmap='RdYlBu', ax=ax)
        ax.set_title(f'Correlation of NaN positions between columns{f" for {name}" if name else ""}')
        show(f'{save}/{name}_nan_corrs.png' if save else False)

    ### Per column analysis
    for col in df.columns:
        fprint(f'Analysis for Column {col}')
        series = df[col]

        # Distance between values
        valinds = series[~series.isna()].index
        cadence = valinds[1:] - valinds[:-1].values - pd.Timedelta('1 minute')
        count   = cadence.value_counts()
        percent = (valinds.size / series.size) * 100

        print(f'The column is {percent:.2f}% dense, {valinds.size} / {series.size}')
        print(f'The average distance between values is {cadence.mean()}. Below is a graph showing the various differences in distances between values:\n')

        ax = count.sort_index().iloc[:50].plot(title='Distance between Values', xlabel='Distance', rot=45, **valCount)
        show(f'{save}/{name}_dist_between_vals.png' if save else False)

        # Average Values
        print('\nThe following graph plots the daily average of each year:')

        fig, ax = plt.subplots(figsize=(30, 10))
        daily   = series.resample('1D').mean()
        years   = pd.date_range(daily.index[0], daily.index[-1], freq='1Y', normalize=True)
        for i, year in enumerate(years):
            if i == 0:
                data = daily[daily.index <= year]
            else:
                data = daily[(daily.index > years[i-1]) & (daily.index <= year)]

            # Reset the index to day of year to remove the year, and plot
            data.index = data.index.dayofyear
            ax = data.plot(label=year.year, ax=ax, alpha=1)
        else:
            ax.legend()
            ax.set_title(f'Daily Average Value for {col}')
            ax.set_xlabel('Day of Year')
            ax.set_ylabel('Value')
            show(f'{save}/{col}_daily_average.png' if save else False)

        # Outliers
        print('\nThe follow graph is a histogram of the given column with the standard deviation regions away from the mean highlighted to show outliers.')
        print('The red vertical line is the mean of the column, and each highlighted region is N standard deviations away.')
        print('The lower x-axis mark the boundaries of these sigma regions. The upper x-axis mark the min/max values for the series.')
        generate_histogram(col, df, save=save)

#%% Load data
path = 'Weather Data/'

#df = load_r0(path, round=True)
#df = load_cn2(path, round=True)
df = load_weather(path, interpolate=False)

df

#%% Perform analysis
analyze(df, save='plots/weather/', name='weather')

#%% Generate Histograms

for col in df:
    generate_histograms(col, save=True)

#%% TESTING
from matplotlib.text import Text
months = pd.date_range(start='1/1/1970', periods=12, freq=pd.offsets.MonthBegin(1))
ticks = [Text(months.dayofyear[i], 0, month) for i, month in enumerate(months.month_name())]

fig, ax = plt.subplots(figsize=(30, 10))
daily   = series.resample('1D').mean()
years   = pd.date_range(daily.index[0], daily.index[-1], freq='1Y', normalize=True)
for i, year in enumerate(years):
    if i == 0:
        data = daily[daily.index <= year]
    else:
        data = daily[(daily.index > years[i-1]) & (daily.index <= year)]

    # Reset the index to day of year to remove the year, and plot
    data.index = data.index.strftime('%m/%d')
    ax = data.plot(label=year.year, ax=ax, alpha=1)
else:
    ax.legend()
    ax.set_title(f'Daily Average Value for {col}')
    ax.set_xlabel('Day of Year')
    ax.set_ylabel('Value')
    plt.show()

#%% TESTING

import numpy as np
m = daily.copy()
m.index = m.index.dayofyear
m.values


m = m.sort_index()
z = np.polyfit(x=m.index.values, y=m.values, deg=1)
p = np.poly1d(z)
p
z
