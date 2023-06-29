from typing import Union

import numpy as np
from gym.core import RewardWrapper, Wrapper
from ml4trade.simulation_env import SimulationEnv


class RewardShapingEnv(RewardWrapper):
    env: SimulationEnv

    def __init__(self, env: Union[SimulationEnv, Wrapper], shaping_coef: float = 1.0):
        super().__init__(env)
        self.shaping_coef = shaping_coef

    def reward(self, reward):
        return reward + self._shaping_function() * self.shaping_coef

    def _shaping_function(self):
        end_idx = self.env.history._cur_tick_to_idx() - self.env.new_clock_view().cur_datetime().hour
        start_idx = end_idx - 24
        if start_idx < 0:
            return 0
        prev_day = self.env.history[start_idx:end_idx]
        prices = list(map(lambda x: x['price'], prev_day))
        sell_amounts = list(map(lambda x: x.get('scheduled_sell_amount', 0), prev_day))
        sell_thresholds = list(map(lambda x: x.get('scheduled_sell_threshold', 0), prev_day))
        unscheduled_buys = list(map(lambda x: x.get('unscheduled_buy_amount', 0), prev_day))

        max_price_idx = 9 + np.argmax(prices[9:])

        bonus_coefs = [0.438, 0.75, 0.859, 0.938, 0.984, 1.0, 0.984, 0.938, 0.859, 0.75, 0.438]
        bonuses = [0] * len(bonus_coefs)
        for i, bc in enumerate(bonus_coefs):
            idx = max_price_idx - len(bonus_coefs) // 2 + i
            if idx < 0 or idx >= 24:
                continue
            sold = sell_thresholds[idx] < prices[idx]
            unscheduled_buy_amount, bought = unscheduled_buys[idx]
            sold_amount = max(0, (sell_amounts[idx] - unscheduled_buy_amount * int(bought)))
            bonuses[i] = sold_amount * int(sold) * prices[idx] * bc

        return sum(bonuses)
