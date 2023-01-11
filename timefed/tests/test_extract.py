import pandas as pd

from timefed.research.extract import roll

def generate(samples=None):
    if not samples:
        samples = np.random.randint(1e3, 1e5)

    df = pd.DataFrame({
        'feature_A': np.random.rand(samples),
        'feature_B': np.random.rand(samples),
        'feature_C': np.random.rand(samples),
        'label'    : np.random.choice([0]*100+[1], size=samples)
        },
        index = pd.date_range(start='2020-01-01', periods=samples, freq='1 min')
        # index = rng.choice(pd.date_range(start='2020-01-01', periods=samples*2, freq='1 min'), samples)
    )
    rng = np.random.default_rng()
    rrows = lambda n: rng.choice(samples, int(samples*np.random.uniform(.1, n)))
    # Randomly sprinkle some random percentage of NaNs into each column except label
    for feature in set(df.columns) - {'label', }:
        df[feature].iloc[rrows(.2)] = np.nan

    # Randomly drop some rows to simulate gaps
    df = df.drop(index=df.index[rrows(.3)])

    df.isna().sum() / df.shape[0] * 100 # Percentage of NaNs in each column

    df = df.sort_index()

    return df

def test_roll(df):
    windows, stats = roll(df, '20 min', required=['feature_A', 'label'], as_frames=False)
    assert stats.possible == 56980
    assert stats.valid == 257
    assert dict(stats.optional) == {'feature_B': 4, 'feature_C': 19}
    assert dict(stats.reasons) == {'wrong_size': 0, 'required_nans': 53964, 'too_large': 2759, 'not_ordered': 0}

    windows, stats = roll(df, '10 min', required=['feature_A', 'label'], as_frames=False)
    assert stats.possible == 56990
    assert stats.valid == 5378
    assert dict(stats.optional) == {'feature_B': 919, 'feature_C': 1729}
    assert dict(stats.reasons) == {'wrong_size': 0, 'required_nans': 44040, 'too_large': 7572, 'not_ordered': 0}


df = pd.read_hdf('timefed/research/local/tests/test.h5', 'roll_1')
test_roll(df)
