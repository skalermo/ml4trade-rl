from datetime import timedelta, datetime
from typing import Union, Tuple

import gym
import pandas as pd
from gym import ObservationWrapper
import numpy as np
from ml4trade.simulation_env import SimulationEnv

from src.prices_analysis import get_prices_optimums


WEEK = timedelta(days=7)
YEAR = timedelta(days=365)


class PriceTypeObsWrapper(ObservationWrapper):
    env: SimulationEnv

    def __init__(self, env, df: pd.DataFrame, grouping_period: timedelta, test_data_start: datetime):
        super().__init__(env)
        old_obs_len = self.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf] * old_obs_len + [0] * 2),
            high=np.array([np.inf] * old_obs_len + [1] * 2),
        )
        self.test_data_start = test_data_start

        days_classified = get_prices_optimums(df)
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
        tomorrow = self.env.new_clock_view().cur_datetime() + timedelta(days=1)
        prices_types = self._get_classes_count(tomorrow - self.period_unit)
        return np.concatenate((observation, np.array([
            # prices_types.get(('p0',), 0),
            prices_types.get(('p1',), 0),
            prices_types.get(('p2',), 0),
        ])))
