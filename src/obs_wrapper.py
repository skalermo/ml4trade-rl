import gym
from gym import ObservationWrapper
import numpy as np
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
        self.observation_space = gym.spaces.Box(
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
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf] * (old_obs_len - 1)),
            high=np.array([np.inf] * (old_obs_len - 1)),
        )
        self.filter_out_idx = filter_out_idx

    def observation(self, observation):
        return observation[:self.filter_out_idx] + observation[self.filter_out_idx + 1:]
