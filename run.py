import os
from typing import List
from datetime import datetime, time
import logging

from stable_baselines3 import A2C
from stable_baselines3.common import logger

import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
import quantstats as qs

from ml4trade.data_strategies import ImgwDataStrategy, HouseholdEnergyConsumptionDataStrategy, PricesPlDataStrategy, imgw_col_ids
from ml4trade.simulation_env import SimulationEnv
from ml4trade.units import *

from src.custom_policies import NormalizationPolicy


def get_all_scv_filenames(path: str) -> List[str]:
    return [f for f in os.listdir(path) if f.endswith('.csv')]


def setup_sim_env(cfg: DictConfig) -> (SimulationEnv, SimulationEnv):
    orig_cwd = hydra.utils.get_original_cwd()

    weather_data_path = f'{orig_cwd}/data/.data/weather_unzipped_flattened'
    filenames = get_all_scv_filenames(weather_data_path)
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

    prices_pl_path = f'{orig_cwd}/data/.data/prices_pl.csv'
    prices_df: pd.DataFrame = pd.read_csv(prices_pl_path, header=0)
    prices_df.fillna(method='bfill', inplace=True)

    data_strategies = {
        'production': ImgwDataStrategy(weather_df, window_size=24, window_direction='forward'),
        'consumption': HouseholdEnergyConsumptionDataStrategy(window_size=24),
        'market': PricesPlDataStrategy(prices_df)
    }

    env_train = SimulationEnv(
        data_strategies,
        start_datetime=datetime.fromisoformat(cfg.env.train_ep_start),
        end_datetime=datetime.fromisoformat(cfg.env.train_ep_end),
        scheduling_time=time.fromisoformat(cfg.env.scheduling_time),
        action_replacement_time=time.fromisoformat(cfg.env.action_time),
        prosumer_init_balance=Currency(cfg.env.init_balance),
        battery_capacity=MWh(cfg.env.bat_cap),
        battery_init_charge=MWh(cfg.env.bat_init_charge),
        battery_efficiency=cfg.env.bat_efficiency,
    )
    env_test = SimulationEnv(
        data_strategies,
        start_datetime=datetime.fromisoformat(cfg.env.test_ep_start),
        end_datetime=datetime.fromisoformat(cfg.env.test_ep_end),
        scheduling_time=time.fromisoformat(cfg.env.scheduling_time),
        action_replacement_time=time.fromisoformat(cfg.env.action_time),
        prosumer_init_balance=Currency(cfg.env.init_balance),
        battery_capacity=MWh(cfg.env.bat_cap),
        battery_init_charge=MWh(cfg.env.bat_init_charge),
        battery_efficiency=cfg.env.bat_efficiency,
    )
    return env_train, env_test


@hydra.main(config_path='conf', config_name='config', version_base='1.1')
def main(cfg: DictConfig) -> None:
    logging.info(OmegaConf.to_yaml(cfg))
    env_train, env_test = setup_sim_env(cfg)
    model = A2C(NormalizationPolicy, env_train,
                **cfg.agent, verbose=1)
    custom_logger = logger.configure('.', ['stdout', 'json'])
    orig_cwd = hydra.utils.get_original_cwd()
    model_name = f'a2c_{cfg.run.train_steps}.zip'
    model_path = f'{orig_cwd}/{model_name}'
    model.set_logger(custom_logger)
    print(model_path)
    if not os.path.exists(model_path):
        model.learn(total_timesteps=cfg.run.train_steps)
        model.save(model_name)
    else:
        print(f'Model {model_path} already exists. Skipping training...')
        model.load(model_path)

    obs = env_test.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env_test.step(action)

    qs.extend_pandas()
    net_worth = pd.Series(env_test.history['wallet_balance'], index=env_test.history['datetime'])
    returns = net_worth.pct_change().iloc[1:]
    qs.reports.full(returns)
    qs.reports.html(returns, output='a2c_quantstats.html', download_filename='a2c_quantstats.html')


if __name__ == '__main__':
    main()
