from typing import Union

import gymnasium
import numpy as np
from gymnasium import Wrapper
from ml4trade.misc.norm_action_wrapper import ActionWrapper, AvgMonthPriceRetriever
from ml4trade.simulation_env import SimulationEnv


class DiscreteActionWrapper(ActionWrapper):
    def __init__(self, env: Union[SimulationEnv, Wrapper], ref_power_MW: float,
                 avg_month_price_retriever: AvgMonthPriceRetriever, bins: int = 100):
        super().__init__(env, ref_power_MW, avg_month_price_retriever)
        self._bins = bins
        self.action_space = gymnasium.spaces.MultiDiscrete([bins] * 96)
        multipliers = np.linspace(1, 4, bins // 2)
        self.multipliers = list(map(lambda x: 1 / x, reversed(multipliers))) + list(multipliers)

    def action(self, action):
        # 25% -> 400%
        new_action = np.array(list(map(lambda x: self.multipliers[x], action)))
        new_action[:48] *= self.ref_power_MW / 2
        price = self._avg_month_price_retriever.get_prev_month_avg_price(
            self.clock_view.cur_datetime()
        )
        new_action[48:] *= price
        return new_action

    def reverse_action(self, action):
        raise NotImplementedError
