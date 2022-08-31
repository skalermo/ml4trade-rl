from typing import List

import numpy as np

from ml4trade.data_strategies import DataStrategy
from ml4trade.misc.norm_ds_wrapper import DataStrategyWrapper


class TemperatureDsWrapper(DataStrategyWrapper):
    col_idx = -1

    def __init__(self, ds: DataStrategy):
        super().__init__(ds)

    def observation(self, idx: int) -> List[float]:
        start_idx = idx + 24 - self.scheduling_hour
        end_idx = start_idx + 24
        avg_temp_tomorrow = np.average(self.df.iloc[start_idx:end_idx, self.col_idx])
        avg_temp_tomorrow /= 40  # scale down, assuming temperature range is [-40; 40]
        return [avg_temp_tomorrow]

    def observation_size(self) -> int:
        return 1
