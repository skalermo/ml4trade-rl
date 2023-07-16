import os
from typing import List, Dict, Callable
from pathlib import Path

import pandas as pd
from ml4trade.data_strategies import ImgwDataStrategy, HouseholdEnergyConsumptionDataStrategy, PricesPlDataStrategy, \
    imgw_col_ids, ImgwWindDataStrategy, ImgwSolarDataStrategy
from ml4trade.domain.units import *
from ml4trade.misc import (
    WeatherWrapper,
    ConsumptionWrapper,
    MarketWrapper,
)
from omegaconf import DictConfig

from src.obs_wrapper import WindWrapper, SolarWrapper, PriceWrapper


def get_weather_df() -> pd.DataFrame:
    weather_data_path = Path(__file__).parent.parent / '.data' / 'weather_unzipped_flattened'

    def _get_all_scv_filenames(path: str) -> List[str]:
        return [f for f in os.listdir(path) if f.endswith('.csv')]

    filenames = _get_all_scv_filenames(str(weather_data_path.absolute()))
    dfs = []
    for f in filenames:
        df = pd.read_csv(f'{weather_data_path}/{f}', header=None, encoding='cp1250',
                         names=imgw_col_ids.keys(), usecols=imgw_col_ids.values())
        dfs.append(df)
    weather_df: pd.DataFrame = pd.concat(dfs, axis=0, ignore_index=True)
    del dfs
    # 352200375 - station_code for Warszawa Okecie
    weather_df = weather_df.loc[weather_df['code'] == 352200375]
    weather_df.sort_values(by=['year', 'month', 'day', 'hour'], inplace=True)
    weather_df.fillna(method='bfill', inplace=True)
    return weather_df


def get_prices_df() -> pd.DataFrame:
    prices_pl_path = Path(__file__).parent.parent / '.data' / 'prices_pl.csv'
    prices_df: pd.DataFrame = pd.read_csv(prices_pl_path, header=0)
    prices_df.fillna(method='bfill', inplace=True)
    prices_df['index'] = pd.to_datetime(prices_df['index'])

    return prices_df


def get_data_strategies(cfg: DictConfig, weather_df: pd.DataFrame, prices_df: pd.DataFrame) -> Dict:
    weather_strat = ImgwDataStrategy(weather_df, window_size=24, window_direction='forward',
                                     max_solar_power=cfg.env.max_solar_power, solar_efficiency=cfg.env.solar_efficiency,
                                     max_wind_power=cfg.env.max_wind_power, max_wind_speed=cfg.env.max_wind_speed)
    weather_strat.imgwWindDataStrategy = WindWrapper(
    # weather_strat.imgwWindDataStrategy = WeatherWrapper(
        ImgwWindDataStrategy(weather_df, window_size=24,
                             max_wind_power=MW(cfg.env.max_wind_power), max_wind_speed=cfg.env.max_wind_speed,
                             window_direction='forward'),
        max_wind_speed=cfg.env.max_wind_speed,
    )
    weather_strat.imgwSolarDataStrategy = SolarWrapper(
    # weather_strat.imgwSolarDataStrategy = WeatherWrapper(
        ImgwSolarDataStrategy(weather_df, window_size=24,
                              max_solar_power=MW(cfg.env.max_solar_power), solar_efficiency=cfg.env.solar_efficiency,
                              window_direction='forward')
    )
    return {
        'production': weather_strat,
        'consumption': ConsumptionWrapper(HouseholdEnergyConsumptionDataStrategy(window_size=24,
                                                                                 household_number=cfg.env.households)),
        # 'market': MarketWrapper(PricesPlDataStrategy(prices_df)),
        'market': PriceWrapper(PricesPlDataStrategy(prices_df)),
    }


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func
