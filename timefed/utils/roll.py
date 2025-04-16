# Copyright 2025, by the California Institute of Technology. ALL RIGHTS RESERVED.
# United States Government Sponsorship acknowledged. Any commercial use must be
# negotiated with the Office of Technology Transfer at the California Institute of
# Technology.
#
# This software may be subject to U.S. export control laws. By accepting this software,
# the user agrees to comply with all applicable U.S. export laws and regulations. User
# has the responsibility to obtain export licenses, or other export authority as may be
# required before exporting such information to foreign countries or providing access
# to foreign persons.

import logging

import pandas as pd
import xarray as xr

from mlky import Sect
from mlky.utils.track import Track

Logger = logging.getLogger('timefed/utils/roll')


class Roll:
    zero = pd.Timedelta(0)

    def __init__(self, df, window, frequency=None, step=1, required=None, optional=[], method='groups'):
        """
        Performs some preprocessing for the roll function.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to operate on. The index must be a datetime.
        window : str
            The window size to extract. This must be a pandas.Timedelta compatible
            string, such as 5m or 2h.
        frequency : str, default=None
            The assumed frequency of the DataFrame. Must be a pandas.Timedelta
            compatible string. If not provided, will assume the most common frequency.
        step : int or str, default=1
            The step size between windows. If int, steps by index position. If str,
            uses pandas.Timedelta to step along the index.
        required : list, default=None
            These columns are required to be fully dense in each window.
        optional : list, default=[]
            These columns are optional. (NYI)
        method : 'groups', 'pandas', None
            How to format the .windows attribute
            None:
                Leave windows in index form tuples [(i, j), ...] such that i is the
                start of the window and j is the end on the index
            pandas:
                Convert windows to individual pandas DataFrames (view, not copy)
            groups:
                Does 'pandas' then copies the frames, adds a windowID column, then
                stacks them into a single DataFrame
            xarray:
                Converts windows to a 2D xarray object with dimensions
                (window index, window ID)

        Notes
        -----
        >>> windows = Roll(df, '60s')
        >>> windows.roll()
        20
        >>> windows.windows
        ...
        """
        self.df = df.copy()
        self.windows = []
        self.tuples  = []
        self.method  = method

        freqs = (df.index[1:] - df.index[:-1]).value_counts().sort_values(ascending=False)
        if self.zero in freqs:
            Logger.info('Duplicate timestamps were detected, windowing may return unexpected results')

        if frequency is None:
            frequency = freqs.index[0]
            Logger.info(f'Frequency not provided, selecting the most common frequency difference: {frequency}')

        self.freq = pd.Timedelta(frequency)

        if isinstance(step, str):
            self.offset = pd.Timedelta(step)
            self.step   = self.stepByTime

        elif isinstance(step, int):
            self.offset = step
            self.step   = self.stepByIndex

        self.delta = pd.Timedelta(window)
        self.size  = int(self.delta / self.freq)
        if self.size < 1:
            Logger.error(f'The window size is too short for the cadence of the data (min size 1): size = int(delta / frequency) = int({delta} / {frequency}) = {size}')

        if not required:
            required = list(df.columns)
        self.required = required

        if not optional:
            optional = set(df.columns) - set(self.required)
        self.optional = optional

        # Stats
        self.possible = 0
        self.valid    = 0
        self.reasons  = Sect(
            wrong_size  = 0,
            had_nans    = 0,
            had_gap     = 0,
            not_ordered = 0
        )


    def stepByTime(self, i):
        """
        Performs a step in time that respects an imperfectly sampled datetime index

        Parameters
        ----------
        i : int
            Index position to start from

        Returns
        -------
        i : int
            The next index position such that this index is still less than [i] + offset
        """
        k = self.df.index[i] + self.offset
        while self.df.index[i] < k:
            i += 1
        return i


    def stepByIndex(self, i):
        """
        Performs a step by integer index

        Parameters
        ----------
        i : int
            Step from this

        Returns
        -------
        int
            i + step offset
        """
        return i + self.offset


    def roll(self):
        """
        Rolls over a datetime index and calculates the possible valid windows that can
        be extracted. After executing, access the windows from the .windows attribute.

        Returns
        -------
        self.valid : int
            The number of valid windows available
        """
        Logger.info(f'Rolling using a delta: {self.delta}')

        total = self.df.shape[0] - self.size
        track = Track(total, step=10, print=Logger.info)

        i = 0
        while i <= total:
            self.possible += 1

            j = i
            k = j + self.size
            i = self.step(i)
            track(i)

            window = self.df.iloc[j:k]

            # This window was the wrong size (rare edge case)
            if window.shape[0] != self.size:
                self.reasons.wrong_size += 1
                continue

            # This window had NaNs in a required column
            if window[self.required].isna().any(axis=None):
                self.reasons.had_nans += 1
                continue

            diff = window.index[-1] - window.index[0]

            # Window too large
            if diff > self.delta:
                self.reasons.had_gap += 1
                continue
            # Timestamps are not in order causing a negative value or there's duplicates
            elif diff <= self.zero:
                self.reasons.not_ordered += 1
                continue

            self.valid += 1
            self.tuples.append((j, k))

        self.windows = self.tuples

        # Possibly convert the tuples to an object structure
        self.convert(self.method)

        Logger.debug(f'Valid windows: {self.valid}')
        return self.valid


    def convert(self, method):
        """
        Converts the tuples list into a more usable data object

        Parameters
        ----------
        method : 'groups', 'pandas', None
            How to format the .windows attribute
            None:
                Leave windows in index form tuples [(i, j), ...] such that i is the
                start of the window and j is the end on the index
            pandas:
                Convert windows to individual pandas DataFrames (view, not copy)
            groups:
                Does 'pandas' then copies the frames, adds a windowID column, then
                stacks them into a single DataFrame
            xarray:
                Converts windows to a 2D xarray object with dimensions
                (window index, window ID)
        """
        if method in ('pandas', 'groups'):
            return self.to_pandas(method == 'groups')

        elif method == 'xarray':
            return self.to_xarray()

    def to_pandas(self, groups=True):
        """
        Converts the windows attribute from tuples to a pandas DataFrame

        Parameters
        ----------
        groups : bool, default=True
            Copies the frames, adds a windowID column, then stacks them into a single DataFrame
        """
        Logger.info('Converting to pandas DataFrames')
        self.windows = [self.df.iloc[i:j] for i, j in self.tuples]

        if groups:
            Logger.info('Duplicating and stacking windows to create groups')
            for w, df in enumerate(self.windows):
                self.windows[w] = df.copy()
                self.windows[w]['windowID'] = w

            self.windows = pd.concat(self.windows)

    def to_xarray(self):
        """
        Converts the windows attribute from tuples to a 2D xarray Dataset
        """
        Logger.info('Converting to an xarray Dataset')
        ds = self.df.reset_index().to_xarray()
        windows = []
        for i, j in self.tuples:
            sel = ds.isel(index=slice(i, j))
            sel['index'] = range(sel.index.size)
            windows.append(sel)

        Logger.debug('Concatenating windows')
        self.windows = xr.concat(windows, 'windowID')
