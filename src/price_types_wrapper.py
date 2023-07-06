from datetime import timedelta, datetime
from typing import Union, Tuple

import pandas as pd
from gymnasium import spaces, ObservationWrapper
import numpy as np
from ml4trade.simulation_env import SimulationEnv

from src.prices_analysis import get_prices_optimums


WEEK = timedelta(days=7)
YEAR = timedelta(days=365)


class PriceTypeObsWrapper(ObservationWrapper):
    env: SimulationEnv

    def __init__(self, env, df: pd.DataFrame, grouping_period: timedelta, test_data_start: datetime,
                 use_future: bool = False):
        super().__init__(env)
        old_obs_len = self.observation_space.shape[0]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf] * old_obs_len + [0] * 6),
            high=np.array([np.inf] * old_obs_len + [1] * 6),
        )
        self.test_data_start = test_data_start
        self.use_future = use_future

        days_classified = get_prices_optimums(df)
        self.classes = days_classified[['date', 'cls']]
        self.period_unit: timedelta = YEAR if grouping_period >= YEAR else WEEK
        self.groupings, self.default_grouping = self._precalculate_groupings(days_classified, grouping_period)

    def _precalculate_groupings(self, days_classified: pd.DataFrame, grouping_period: timedelta) -> tuple:
        print('Precalculating groupings...')
        start_date = days_classified.head(1)['date'].item()
        end_date = days_classified.tail(1)['date'].item()
        period_start: datetime = start_date
        period_end: datetime = start_date + grouping_period

        groupings = {}
        while period_end <= end_date:
            cur_period = days_classified[days_classified['date'] <= period_end]
            cur_period = cur_period[cur_period['date'] >= period_start]
            key = self._get_key(period_end)
            groupings[key] = self._group_period(cur_period)

            period_start += self.period_unit
            period_end += self.period_unit

        default_grouping = self._group_period(
            days_classified[days_classified['date'] <= self.test_data_start]
        )
        return groupings, default_grouping

    def _get_key(self, _datetime: datetime) -> Union[Tuple[int, int], int]:
        if self.period_unit == WEEK:
            key = (_datetime.year, _datetime.isocalendar()[1])  # week nr
        else:  # YEAR
            key = _datetime.year
        return key

    def _get_classes_count(self, _datetime: datetime):
        group = self.groupings.get(self._get_key(_datetime), self.default_grouping)
        if self.period_unit == WEEK:
            prices_types = group[_datetime.isoweekday()]
        else:  # YEAR
            prices_types = group[(_datetime.month, _datetime.isoweekday())]
        return prices_types

    def _group_period(self, period: pd.DataFrame):
        if self.period_unit == WEEK:
            by = ['weekday']
        else:  # YEAR
            by = ['month', 'weekday']
        gb = period.groupby(by)
        res = {}
        for x in gb.groups:
            dct = dict(gb.get_group(x).value_counts(['cls'], normalize=True))
            dct = {k[0]: dct[k] for k in dct}
            res[x] = dct
        return res

    def observation(self, observation):
        classes = ['111', '212', '112', '211', '213', '312']

        tomorrow = self.env.new_clock_view().cur_datetime() + timedelta(days=1)

        if self.use_future:
            c = self.classes[self.classes['date'] == tomorrow.replace(hour=0)].head(1)['cls'].item()
            prices_types = {c: 1.0}
        else:
            prices_types = self._get_classes_count(tomorrow - self.period_unit)

        stats = np.zeros(len(classes))
        for i, cls in enumerate(classes):
            stats[i] = prices_types.get(cls, 0)

        return np.concatenate((observation, stats))
