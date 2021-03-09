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
