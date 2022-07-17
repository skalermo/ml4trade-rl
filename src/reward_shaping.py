from gym.core import RewardWrapper
from ml4trade.simulation_env import SimulationEnv


class RewardShapingEnv(RewardWrapper):
    def __init__(self, env: SimulationEnv, shaping_coef: int = 0):
        super().__init__(env)
        self.shaping_coef = shaping_coef

    def reward(self, reward):
        return reward + self._shaping_function() * self.shaping_coef

    @staticmethod
    def __battery_reward_function(battery_charge, max_at=0.5):
        return max(0, 1 - (battery_charge / max_at - 1) ** 2)

    def _shaping_function(self):
        def hour_to_idx(h: int):
            return -10 - (24 - h) - 1

        battery_at_17 = self.env.history['battery'][hour_to_idx(17)]
        battery_at_20 = self.env.history['battery'][hour_to_idx(20)]
        return self.__battery_reward_function(battery_at_17, max_at=0.9) \
               # + self.__battery_reward_function(battery_at_20, max_at=0.15)
