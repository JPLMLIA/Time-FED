#%%
# Imports
import pandas
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

#%%
#
def predict(data, i, m, t, kind):
    testing  = data.iloc[i:i+m]

    t1 = data.iloc[i-t : i]
    t2 = data.iloc[i+m : i+m+t]

    try:
        assert t1.isnull().mean() < .25
        assert t2.isnull().mean() < .25

        training = pd.concat([t1, t2])

        interp  = interp1d(x=training.index, y=training.values, kind=kind)
        predict = interp  (x=testing.index)

        diff = testing - predict

        return predict
    except Exception as e:
        print(e)

#%%
## SETUP
# import data
file = '/path/to/data.h5'
df = pd.read_hdf(file, 'merged')

# ndf is typically my 'new' df / secondary df; in this case, date subset
ndf = df.loc[('2018/12/15 20:00' <= df.index) & (df.index < '2018/12/16 00:00')]
ndf = ndf.reset_index() # Index needs to be integers

# Manually set these
block = 'night'                      # Used in filename
time  = '8pm to 12am on 12/15/2018'  # Subtitle for plot

# How to do interpolation predictions
m = 3
t = 45
method = 'linear'

#%%
# This is for creating scatter plots of true vs predicted

for c in ndf:
    if c in ['datetime']:
        continue
    if c not in ['r0']:
        continue

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    wins = pd.DataFrame(index=range(data.size))
    data = ndf[c]

    # Create interpolation predictions and average the results per row
    for i in range(t, data.shape[0]-m-t):
        pred = predict(data, i, m, t, method)
        wins[i] = np.nan
        wins[i][i:i+m] = pred
    mean = wins.mean(axis=1)

    ax.scatter(x=mean.index, y=mean.values, s=10, color='red' , alpha=1, label='Predicted')
    ax.scatter(x=data.index, y=data.values, s=10, color='blue', alpha=1, label='True')
    ax.set_title(f'Predicted vs True for {c}\n{time}\nm={m}, t={t}, method={method}')

    ax.legend()
    ax.set_xlabel('Time')
    ax.set_ylabel(c)

    # Reset xticks to be datetimes
    ticks = ax.get_xticks()
    ticks = [0.0] + list(ticks[1:-2]) + [ndf.index[-1]]
    ax.set_xticklabels(ndf.loc[ticks].datetime, rotation=45)

    plt.tight_layout()
    plt.savefig(f'local/plots/subs/{c}.{method}.{block}.png')

#%%
# This code creates the error plots
# `data` is the dataframe output of window_interpolate.py
def draw_error_plots(data, output, center='mean', error='std', subtitle=''):
    m = np.unique(data.index.get_level_values('m'))
    t = np.unique(data.index.get_level_values('t'))
    k = np.unique(data.index.get_level_values('kind'))

    ncols = len(k)
    nrows = len(m)

    for c in np.unique(data.columns.get_level_values(0)):
        fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10*ncols, 10*nrows))
        for i, _m in enumerate(m):
            for j, _k in enumerate(k):
                win = data[c, 'windows'][_m, 5, _k]
                ax = axes[i][j]
                ax.set_title(f'{c} | m={_m}, method={_k}, windows={win}\n{subtitle}')
                ax.set_xlabel('t (minutes)')
                ax.set_xticks(t)
                ax.set_ylabel('Error')
                try:
                    # ax.set_ylim(
                    #     ymin=min([0, data[c, center][_m].min()-data[c, error][_m].max()]),
                    #     ymax=data[c, center][_m].max()+data[c, error][_m].max()
                    # )
                    max = data[c, center][_m].max()+data[c, error][_m].max()
                    ax.set_ylim(
                        ymin=0,g
                        ymax=max
                    )
                    # ax.set_ylim(ymin=data[c, 'std'][_m].min(), ymax=data[c, 'std'][_m].max())
                    for _t in t:
                        color = 'blue'
                        if data[c, 'windows'][_m, _t, _k] != win:
                            color = 'orange'
                        mean = data[c, center][_m, _t, _k]
                        yerr = data[c, error ][_m, _t, _k]
                        ax.errorbar(x=_t, y=mean, yerr=yerr, fmt='-o', color=color, ecolor='red')
                    ax.hlines(0, *ax.get_xlim(), color='green', alpha=.5)
                except:
                    pass
        plt.tight_layout()
        plt.savefig(f'{output}.{c}.png')

#%%
# Manually set these; file is the output of window_interpolate.py
file   = 'local/data/fourday.results.h5'

# this dict is just for creating the filename and subtitles to differentiate the different plots
blocks = {
    'spring': '04-01',
    'summer': '07-01',
    'fall'  : '10-01',
    'winter': '12-01'
}
subtitle = '12pm to 12am PST'
for season, date in blocks.items():
    data = pd.read_hdf(file, season)
    draw_error_plots(data, f'local/plots/subs/percent/{season}.{date}', subtitle=f'{date} from {subtitle}')
