from datetime import datetime
from typing import List

import pandas as pd
from gymnasium import spaces, ObservationWrapper
import numpy as np
from ml4trade.data_strategies import ImgwSolarDataStrategy, ImgwWindDataStrategy, PricesPlDataStrategy, ImgwDataStrategy
from ml4trade.domain.units import MW
from ml4trade.misc.norm_ds_wrapper import DataStrategyWrapper
from ml4trade.simulation_env import SimulationEnv


def one_hot_encode(x: int, max_val: int) -> np.array:
    res = np.zeros(max_val)
    res[x - 1] = 1
    return res


class DateObsWrapper(ObservationWrapper):
    env: SimulationEnv

    def __init__(self, env):
        super().__init__(env)
        old_obs_len = self.observation_space.shape[0]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf] * old_obs_len + [0] * 19),
            high=np.array([np.inf] * old_obs_len + [1] * 19),
        )

    def observation(self, observation):
        cur_datetime = self.env.new_clock_view().cur_datetime()
        cur_month_encoded = one_hot_encode(cur_datetime.month, max_val=12)
        cur_day_of_week_encoded = one_hot_encode(cur_datetime.isoweekday(), max_val=7)
        return np.concatenate((observation, cur_month_encoded, cur_day_of_week_encoded))


class FilterObsWrapper(ObservationWrapper):
    env: SimulationEnv

    def __init__(self, env, filter_out_idx: int):
        super().__init__(env)
        old_obs_len = self.observation_space.shape[0]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf] * (old_obs_len - 1)),
            high=np.array([np.inf] * (old_obs_len - 1)),
        )
        self.filter_out_idx = filter_out_idx

    def observation(self, observation):
        return observation[:self.filter_out_idx] + observation[self.filter_out_idx + 1:]


class WindWrapper(DataStrategyWrapper):
    def __init__(self, ds: ImgwWindDataStrategy, max_wind_speed: float):
        super().__init__(ds)
        self.max_wind_speed = max_wind_speed

    def observation(self, idx: int) -> List[float]:
        obs = self.ds.observation(idx)
        return self._minmax_scale(obs)

    def _minmax_scale(self, obs: list):
        return list(map(lambda x: x / self.max_wind_speed if x <= self.max_wind_speed else 0, obs))


class SolarWrapper(DataStrategyWrapper):
    def __init__(self, ds: ImgwSolarDataStrategy):
        super().__init__(ds)
        self.max_octant_value = 8

    def observation(self, idx: int) -> List[float]:
        obs = self.ds.observation(idx)
        return self._minmax_scale(obs)

    def _minmax_scale(self, obs: list):
        return list(map(lambda x: x / self.max_octant_value if x <= self.max_octant_value else 1, obs))


class PriceWrapper(DataStrategyWrapper):
    def __init__(self, ds: PricesPlDataStrategy):
        super().__init__(ds)

    def observation_size(self) -> int:
        return 24

    def observation(self, idx: int, _datetime: datetime = None) -> List[float]:
        obs = self.ds.observation(idx)
        # return self._minmax_scale_wrt(obs, self.max_price)
        # return self._log_scale(obs, price)
        obs = self._minmax_scale(obs)
        # obs = self.normalize(obs)
        # mi = int(np.argmin(obs)) / 24
        # ma = int(np.argmax(obs)) / 24
        # return obs + [mi, ma]
        return obs

    def _minmax_scale_wrt(self, obs: list, target: float):
        return list(map(lambda x: x / target, obs))

    @staticmethod
    def _minmax_scale(obs: list):
        min_val = min(obs)
        max_val = max(obs)
        return list(map(lambda x: (x - min_val) / (max_val - min_val), obs))

    def _log_scale(self, obs: list, prev_month_avg_price):
        return list(map(lambda x: np.log(x / prev_month_avg_price), obs))

    def normalize(self, obs: list):
        scaled = self._minmax_scale(obs)
        normalized = list(map(lambda x: x / sum(scaled), scaled))
        return normalized


class WeatherStatsWrapper(ImgwDataStrategy):
    def __init__(self, df: pd.DataFrame, window_size: int, max_solar_power: MW, solar_efficiency: float,
                 max_wind_power: MW, max_wind_speed: float, ref_power_MW: float):
        super().__init__(df, window_size, max_solar_power, solar_efficiency, max_wind_power, max_wind_speed)
        self.ref_power_MW = ref_power_MW

    def observation_size(self) -> int:
        return 24

    def observation(self, idx: int) -> List[float]:
        start_idx = idx + 24 - self.scheduling_hour
        end_idx = start_idx + 24
        power_prediction = [
            (self.imgwWindDataStrategy.process(idx) + self.imgwSolarDataStrategy.process(idx)) / self.ref_power_MW
            for idx in range(start_idx, end_idx)
        ]
        return power_prediction
