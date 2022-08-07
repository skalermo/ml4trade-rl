from datetime import timedelta
from typing import Union

from gym.core import RewardWrapper, Wrapper
from ml4trade.simulation_env import SimulationEnv


class RewardShapingEnv(RewardWrapper):
    env: SimulationEnv

    def __init__(self, env: Union[SimulationEnv, Wrapper], shaping_coef: int = 0):
        super().__init__(env)
        self.shaping_coef = shaping_coef

    def reward(self, reward):
        return reward + self._shaping_function() * self.shaping_coef

    @staticmethod
    def __battery_reward_function(battery_charge, max_at=0.5):
        return max(0, 1 - (battery_charge / max_at - 1) ** 2)

    def _shaping_function(self):
        cur_datetime = self.env.new_clock_view().cur_datetime()
        last_day_17 = cur_datetime.replace(hour=17) - timedelta(days=1)
        diff = cur_datetime - last_day_17
        idx = self.env.history._cur_tick_to_idx() - diff.seconds // 3600
        battery_at_17 = self.env.history[idx]['rel_battery']
        return self.__battery_reward_function(battery_at_17, max_at=0.9)
