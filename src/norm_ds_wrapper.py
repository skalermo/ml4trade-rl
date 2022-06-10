from typing import List, Union

from ml4trade.data_strategies import (
    DataStrategy,
    HouseholdEnergyConsumptionDataStrategy,
    ImgwWindDataStrategy,
    ImgwSolarDataStrategy,
    PricesPlDataStrategy,
    update_last_processed,
)


class DataStrategyWrapper(DataStrategy):
    def __init__(self, ds: DataStrategy):
        super().__init__()
        self.ds = ds

    @update_last_processed
    def process(self, idx: int) -> float:
        return self.ds.process(idx)

    def observation(self, idx: int) -> List[float]:
        return self.ds.observation(idx)

    def observation_size(self) -> int:
        return self.ds.observation_size()


class MarketWrapper(DataStrategyWrapper):
    def __init__(self, ds: PricesPlDataStrategy):
        super().__init__(ds)

    def observation(self, idx: int) -> List[float]:
        return []

    def observation_size(self) -> int:
        return 0


class ConsumptionWrapper(DataStrategyWrapper):
    def __init__(self, ds: HouseholdEnergyConsumptionDataStrategy):
        super().__init__(ds)

    def observation(self, idx: int) -> List[float]:
        obs = self.ds.observation(idx)
        return list(map(lambda x: x * 1000, obs))


class WeatherWrapper(DataStrategyWrapper):
    def __init__(self, ds: Union[ImgwSolarDataStrategy, ImgwWindDataStrategy]):
        super().__init__(ds)
        self.col_mean = ds.df.iloc[:, ds.col_idx].mean()
        self.col_std = ds.df.iloc[:, ds.col_idx].std()

    def observation(self, idx: int) -> List[float]:
        obs = self.ds.observation(idx)
        obs = list(map(lambda x: (x - self.col_mean) / self.col_std, obs))
        return obs
