from datetime import timedelta, datetime

import gym
import pandas as pd
from gym import ObservationWrapper
import numpy as np
from ml4trade.simulation_env import SimulationEnv

from src.prices_analysis import daily_prices_diff_analysis


def one_hot_encode(x: int, max_val: int) -> np.array:
    res = np.zeros(max_val)
    res[x - 1] = 1
    return res


class DateObsWrapper(ObservationWrapper):
    env: SimulationEnv

    def __init__(self, env):
        super().__init__(env)
        old_obs_len = self.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf] * old_obs_len + [0] * 19),
            high=np.array([np.inf] * old_obs_len + [1] * 19),
        )

    def observation(self, observation):
        cur_datetime = self.env.new_clock_view().cur_datetime()
        cur_month_encoded = one_hot_encode(cur_datetime.month, max_val=12)
        cur_day_of_week_encoded = one_hot_encode(cur_datetime.isoweekday(), max_val=7)
        return np.concatenate((observation, cur_month_encoded, cur_day_of_week_encoded))


class PriceTypeObsWrapper(ObservationWrapper):
    env: SimulationEnv

    def __init__(self, env, df: pd.DataFrame, test_data_start: datetime = None):
        super().__init__(env)
        old_obs_len = self.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf] * old_obs_len + [0] * 2),
            high=np.array([np.inf] * old_obs_len + [1] * 2),
        )
        self.prices = daily_prices_diff_analysis(df, test_data_start)
        # self.weekday_analyzed_prices = daily_prices_diff_analysis(df, test_data_start, group_by='weekday')
        # self.month_analyzed_prices = daily_prices_diff_analysis(df, test_data_start, group_by='month')

    def observation(self, observation):
        tomorrow = self.env.new_clock_view().cur_datetime() + timedelta(days=1)
        prices_types = self.prices[(tomorrow.month, tomorrow.isoweekday())]
        # weekday_types = self.weekday_analyzed_prices[tomorrow.isoweekday()]
        # month_types = self.month_analyzed_prices[tomorrow.month]

        return np.concatenate((observation, np.array([
            prices_types.get(('p1',), 0),
            prices_types.get(('p2',), 0),
            # weekday_types.get(('p1',), 0),
            # weekday_types.get(('p2',), 0),
            # month_types.get(('p1',), 0),
            # month_types.get(('p2',), 0),
        ])))


class FilterObsWrapper(ObservationWrapper):
    env: SimulationEnv

    def __init__(self, env, filter_out_idx: int):
        super().__init__(env)
        old_obs_len = self.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf] * (old_obs_len - 1)),
            high=np.array([np.inf] * (old_obs_len - 1)),
        )
        self.filter_out_idx = filter_out_idx

    def observation(self, observation):
        return observation[:self.filter_out_idx] + observation[self.filter_out_idx + 1:]
