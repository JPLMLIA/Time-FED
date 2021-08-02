#%%
%load_ext autoreload
%autoreload 2
%matplotlib inline

#%%

import matplotlib.pyplot as plt
import pandas  as pd
import seaborn as sns

from types import SimpleNamespace as sns

sns.set_context('talk')
sns.set_style('darkgrid')

#%%
# Load r0 data
old_scores = 'local/research/r0/scores.h5'
oscs = {
    'r0.cn2.weather.historical': sns(df=pd.read_hdf(old_scores, 'r0.cn2.weather.historical'), label='Cn2 + Weather + Historical'),
    'r0.cn2.weather'           : sns(df=pd.read_hdf(old_scores, 'r0.cn2.weather'), label='Cn2 + Weather'),
    'r0.weather.historical'    : sns(df=pd.read_hdf(old_scores, 'r0.weather.historical'), label='Weather + Historical'),
    'r0.weather'               : sns(df=pd.read_hdf(old_scores, 'r0.weather'), label='Weather')
}


new_scores = 'local/research/r0/verify/scores.h5'
nscs = {
    'r0.cn2.weather.historical': sns(df=pd.read_hdf(new_scores, 'r0.cn2.weather.historical'), label='Cn2 + Weather + Historical'),
    'r0.cn2.weather'           : sns(df=pd.read_hdf(new_scores, 'r0.cn2.weather'), label='Cn2 + Weather'),
    'r0.weather.historical'    : sns(df=pd.read_hdf(new_scores, 'r0.weather.historical'), label='Weather + Historical'),
    'r0.weather'               : sns(df=pd.read_hdf(new_scores, 'r0.weather'), label='Weather')
}

vars_file = 'local/research/r0/verify/vars.h5'
vars = {
    'r0.cn2.weather.historical': sns(df=pd.read_hdf(vars_file, 'r0.cn2.weather.historical'), label='Cn2 + Weather + Historical'),
    'r0.cn2.weather'           : sns(df=pd.read_hdf(vars_file, 'r0.cn2.weather'), label='Cn2 + Weather'),
    'r0.weather.historical'    : sns(df=pd.read_hdf(vars_file, 'r0.weather.historical'), label='Weather + Historical'),
    'r0.weather'               : sns(df=pd.read_hdf(vars_file, 'r0.weather'), label='Weather')
}

#%%
# Load weather data
weather_file = 'local/research/weather/weather_scores.h5'
wthr = {
    'temperature'      : sns(df=pd.read_hdf(weather_file, 'temperature'), label='Temperature'),
    'pressure'         : sns(df=pd.read_hdf(weather_file, 'pressure'), label='Pressure'),
    'relative_humidity': sns(df=pd.read_hdf(weather_file, 'relative_humidity'), label='Relative Humidity'),
    'wind_speed'       : sns(df=pd.read_hdf(weather_file, 'wind_speed'), label='Wind Speed')
}

#%%
# Load PWV data
weather_file = 'local/research/pwv/07_20_2021/scores.h5'
pwvs = {
    'pwv': sns(df=pd.read_hdf(weather_file, 'pwv'), label='Water Vapor')
}

#%%
# Remove keys
remove = ['r0.weather.historical', 'r0.weather']
for group in [oscs, nscs, vars]:
    for key in remove:
        if key in group:
            group.pop(key)
#%%
remove = ['temperature']
for group in [wthr]:
    for key in remove:
        if key in group:
            group.pop(key)

#%%

def score_plots(dfs, vars=None, multiple=False, count=False, save=False):
    """
    Creates score plots. For each metric, plot all dataframes in dfs together
    """
    plots = 3
    if vars is not None:
        plots += 1
        if count:
            plots += 1

    fig, axes = plt.subplots(plots, 1, figsize=(25, 5*plots))
    align = {0: 'RMSE', 1: 'MAPE', 2: 'R2'}

    for i, metric in align.items():
        ax = axes[i]

        if multiple:
            fmt = False
            for tag, _dfs in dfs.items():
                for key, data in _dfs.items():
                    if fmt:
                        ax.plot(data.df[metric], label=f'{tag} | {data.label}')
                    else:
                        ax.plot(data.df[metric], '--', label=f'{tag} | {data.label}')
                fmt = True
        else:
            for key, data in dfs.items():
                ax.plot(data.df[metric], label=data.label)

        ax.legend()
        ax.set_xticks(data.df.index)
        ax.set_title(f'{metric} Scores')
        ax.set_xlabel('Forecast (minutes)')
        ax.set_ylabel(metric)

    if vars is not None:
        ax = axes[3]
        for key, data in vars.items():
            ax.plot(data.df['var'], label=data.label)
        ax.legend()
        ax.set_xticks(data.df.index)
        ax.set_title(f"Variance of Target's Test Set")
        ax.set_xlabel('Forecast (minutes)')
        ax.set_ylabel('Variance of Target')

        if count:
            ax = axes[4]
            for key, data in vars.items():
                ax.plot(data.df['count'], label=data.label)
            ax.legend()
            ax.set_xticks(data.df.index)
            ax.set_title(f"Count of Test Set Rows")
            ax.set_xlabel('Forecast (minutes)')
            ax.set_ylabel('Count')

    plt.tight_layout()

    if save:
        plt.savefig(save)

#%%
# Old scores
score_plots(oscs, save='local/research/r0/verify/old_scores.png')

#%%
# New scores
score_plots(nscs, save='local/research/r0/verify/new_scores.png')

#%%
# Old & New on same plots
score_plots({'Old': oscs, 'New': nscs}, multiple=True, save='local/research/r0/verify/combined_scores.png')

#%%
# New with variance
score_plots(nscs, vars, count=True, save='local/research/r0/verify/new_with_vars.png')

#%%
# PWV scores
score_plots(pwvs, save='local/research/pwv/07_20_2021/scores.png')

#%%
# Weather scores
score_plots(wthr, save='local/research/weather/scores_no_temp.png')

#%%

def separate_plots(dfs, metric, save=False):
    """
    Creates separate plots for each key with the given metric
    """
    plots = len(dfs)
    fig, axes = plt.subplots(plots, 1, figsize=(25, 5*plots))

    for i, var in enumerate(dfs.items()):
        key, data = var
        ax = axes[i]
        ax.plot(data.df[metric], label=data.label)

        ax.set_xticks(data.df.index)
        ax.set_title(f'{metric} Score for {data.label}')
        ax.set_xlabel('Forecast (minutes)')
        ax.set_ylabel(metric)

    plt.tight_layout()
    if save:
        plt.savefig(save)

#%%
# Separate MAPE plots
separate_plots(wthr, 'MAPE', save='local/research/weather/mape_no_temp.png')

#%%
separate_plots(wthr, 'RMSE', save='local/research/weather/rmse.png')
