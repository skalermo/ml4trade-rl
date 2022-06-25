import os
import sys
from typing import List, Dict, Tuple
from datetime import datetime, time, timedelta
import logging

from stable_baselines3 import A2C, PPO
from stable_baselines3.common import logger
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
import quantstats as qs

from ml4trade.data_strategies import ImgwDataStrategy, HouseholdEnergyConsumptionDataStrategy, PricesPlDataStrategy, imgw_col_ids, ImgwWindDataStrategy, ImgwSolarDataStrategy
from ml4trade.simulation_env import SimulationEnv
from ml4trade.domain.units import *
from ml4trade.misc import IntervalWrapper, ActionWrapper, WeatherWrapper, ConsumptionWrapper, MarketWrapper


def get_all_scv_filenames(path: str) -> List[str]:
    return [f for f in os.listdir(path) if f.endswith('.csv')]


def setup_sim_env(cfg: DictConfig) -> Tuple[SimulationEnv, Dict]:
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
    prices_df['index'] = pd.to_datetime(prices_df['index'])
    avg_month_prices: Dict[Tuple[int, int], float] = prices_df.groupby(
        [prices_df['index'].dt.year.rename('year'), prices_df['index'].dt.month.rename('month')]
    )['Fixing I Price [PLN/MWh]'].mean().to_dict()

    weather_strat = ImgwDataStrategy(weather_df, window_size=24, window_direction='forward',
                                     max_solar_power=cfg.env.max_solar_power, solar_efficiency=cfg.env.solar_efficiency,
                                     max_wind_power=cfg.env.max_wind_power, max_wind_speed=cfg.env.max_wind_speed)
    weather_strat.imgwWindDataStrategy = WeatherWrapper(ImgwWindDataStrategy(weather_df, window_size=24, window_direction='forward'))
    weather_strat.imgwSolarDataStrategy = WeatherWrapper(ImgwSolarDataStrategy(weather_df, window_size=24, window_direction='forward'))
    data_strategies = {
        'production': weather_strat,
        'consumption': ConsumptionWrapper(
            HouseholdEnergyConsumptionDataStrategy(window_size=24)
        ),
        'market': MarketWrapper(PricesPlDataStrategy(prices_df)),
    }

    env = SimulationEnv(
        data_strategies,
        start_datetime=datetime.fromisoformat(cfg.env.start),
        end_datetime=datetime.fromisoformat(cfg.env.end),
        scheduling_time=time.fromisoformat(cfg.env.scheduling_time),
        action_replacement_time=time.fromisoformat(cfg.env.action_time),
        prosumer_init_balance=Currency(cfg.env.init_balance),
        battery_capacity=MWh(cfg.env.bat_cap),
        battery_init_charge=MWh(cfg.env.bat_init_charge),
        battery_efficiency=cfg.env.bat_efficiency,
    )
    return env, avg_month_prices


@hydra.main(config_path='conf', config_name='config', version_base='1.1')
def main(cfg: DictConfig) -> None:
    agent_name = 'ppo' if any('ppo' in arg for arg in sys.argv) else 'a2c'
    agent_class = {
        'ppo': PPO,
        'a2c': A2C,
    }[agent_name]
    logging.info(f'agent={agent_name}')
    logging.info(OmegaConf.to_yaml(cfg))
    env, avg_month_prices = setup_sim_env(cfg)
    env = IntervalWrapper(env, interval=timedelta(days=30 * 3), split_ratio=0.8)
    max_power = cfg.env.max_solar_power + cfg.env.max_wind_power
    env = ActionWrapper(env, avg_month_prices, max_power / 2, env.env._clock.view())

    model = agent_class('MlpPolicy', env,
                        **cfg.agent, verbose=1)
    custom_logger = logger.configure('.', ['stdout', 'json'])
    orig_cwd = hydra.utils.get_original_cwd()
    model_file = f'{agent_name}_{cfg.run.train_steps}.zip'
    model_path = f'{orig_cwd}/{model_file}'
    model.set_logger(custom_logger)
    if not os.path.exists(model_path):
        model.learn(total_timesteps=cfg.run.train_steps)
        model.save(model_file)
    else:
        print(f'Model {model_path} already exists. Skipping training...')
        model = agent_class.load(model_path, env)

    obs = env.set_to_test_and_reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)

    print(env.env.env._prosumer.wallet.balance)
    env.save_history()
    env.render_all()
    qs.extend_pandas()
    net_worth = pd.Series(env.history['wallet_balance'], index=env.history['datetime'])
    returns = net_worth.pct_change().iloc[1:]
    # qs.reports.full(returns)
    qs.reports.html(returns, output=f'{agent_name}_quantstats.html', download_filename=f'{agent_name}_quantstats.html')


if __name__ == '__main__':
    main()
