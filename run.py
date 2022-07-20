import logging
import os
import sys
from datetime import datetime, time, timedelta
from typing import List, Dict, Tuple

import hydra
import numpy as np
import pandas as pd
import quantstats as qs
from ml4trade.data_strategies import ImgwDataStrategy, HouseholdEnergyConsumptionDataStrategy, PricesPlDataStrategy, \
    imgw_col_ids, ImgwWindDataStrategy, ImgwSolarDataStrategy
from ml4trade.domain.units import *
from ml4trade.misc import IntervalWrapper, ActionWrapper, WeatherWrapper, ConsumptionWrapper, MarketWrapper
from ml4trade.simulation_env import SimulationEnv
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import A2C, PPO
from stable_baselines3.common import logger

from src.reward_shaping import RewardShapingEnv


def get_weather_df(original_cwd: str) -> pd.DataFrame:
    weather_data_path = f'{original_cwd}/data/.data/weather_unzipped_flattened'

    def _get_all_scv_filenames(path: str) -> List[str]:
        return [f for f in os.listdir(path) if f.endswith('.csv')]

    filenames = _get_all_scv_filenames(weather_data_path)
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


def get_prices_df(original_cwd: str) -> pd.DataFrame:
    prices_pl_path = f'{original_cwd}/data/.data/prices_pl.csv'
    prices_df: pd.DataFrame = pd.read_csv(prices_pl_path, header=0)
    prices_df.fillna(method='bfill', inplace=True)
    prices_df['index'] = pd.to_datetime(prices_df['index'])
    return prices_df


def get_data_strategies(cfg: DictConfig, weather_df: pd.DataFrame, prices_df: pd.DataFrame) -> Dict:
    weather_strat = ImgwDataStrategy(weather_df, window_size=24, window_direction='forward',
                                     max_solar_power=cfg.env.max_solar_power, solar_efficiency=cfg.env.solar_efficiency,
                                     max_wind_power=cfg.env.max_wind_power, max_wind_speed=cfg.env.max_wind_speed)
    weather_strat.imgwWindDataStrategy = WeatherWrapper(
        ImgwWindDataStrategy(weather_df, window_size=24,
                             max_wind_power=MW(cfg.env.max_wind_power), max_wind_speed=cfg.env.max_wind_speed,
                             window_direction='forward')
    )
    weather_strat.imgwSolarDataStrategy = WeatherWrapper(
        ImgwSolarDataStrategy(weather_df, window_size=24,
                              max_solar_power=MW(cfg.env.max_solar_power), solar_efficiency=cfg.env.solar_efficiency,
                              window_direction='forward')
    )
    return {
        'production': weather_strat,
        'consumption': ConsumptionWrapper(HouseholdEnergyConsumptionDataStrategy(window_size=24)),
        'market': MarketWrapper(PricesPlDataStrategy(prices_df)),
    }


def _parse_conf_interval(interval: Union[int, List[Tuple[int, int]]]) -> Union[timedelta, List[Tuple[timedelta, int]]]:
    if isinstance(interval, int):
        return timedelta(days=interval)
    return list(map(lambda x: (timedelta(days=x[0]), x[1]), interval))


def setup_sim_env(cfg: DictConfig):
    orig_cwd = hydra.utils.get_original_cwd()
    weather_df = get_weather_df(orig_cwd)
    prices_df = get_prices_df(orig_cwd)
    avg_month_prices: Dict[Tuple[int, int], float] = prices_df.groupby(
        [prices_df['index'].dt.year.rename('year'), prices_df['index'].dt.month.rename('month')]
    )['Fixing I Price [PLN/MWh]'].mean().to_dict()

    data_strategies = get_data_strategies(cfg, weather_df, prices_df)

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
    iw_env = IntervalWrapper(env, interval=_parse_conf_interval(cfg.run.interval), split_ratio=0.8)
    rs_env = RewardShapingEnv(iw_env, cfg.run.shaping_coef)
    max_power = cfg.env.max_solar_power + cfg.env.max_wind_power
    aw_env = ActionWrapper(rs_env, avg_month_prices, max_power / 2, env._clock.view())
    return aw_env


def evaluate_policy(model, env, n_eval_episodes: int = 5):
    episode_rewards = []
    episode_profits = []
    for i in range(n_eval_episodes):
        obs = env.set_to_test_and_reset()
        done = False
        ep_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(action)
            ep_reward += rewards
        episode_rewards.append(ep_reward)
        episode_profits.append(env.env.history['wallet_balance'][-1])
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_profit = np.mean(episode_profits)
    std_profit = np.std(episode_profits)
    return mean_reward, std_reward, mean_profit, std_profit


def quantstats_summary(env_history, agent_name):
    qs.extend_pandas()
    net_worth = pd.Series(env_history['wallet_balance'], index=env_history['datetime'])
    returns = net_worth.pct_change().iloc[1:]
    # qs.reports.full(returns)
    qs.reports.html(returns, output=f'{agent_name}_quantstats.html', download_filename=f'{agent_name}_quantstats.html')


@hydra.main(config_path='conf', config_name='config', version_base='1.1')
def main(cfg: DictConfig) -> None:
    logging.info(' '.join(sys.argv))
    agent_name = 'ppo' if any('ppo' in arg for arg in sys.argv) else 'a2c'
    agent_class = {
        'ppo': PPO,
        'a2c': A2C,
    }[agent_name]
    logging.info(f'agent={agent_name}')
    logging.info(OmegaConf.to_yaml(cfg))
    env = setup_sim_env(cfg)
    logging.info(f'action space: {env.action_space.shape}')
    logging.info(f'observation space: {env.observation_space.shape}')

    model = agent_class('MlpPolicy', env,
                        **cfg.agent, verbose=1)
    custom_logger = logger.configure('.', ['stdout', 'json', 'tensorboard'])
    orig_cwd = hydra.utils.get_original_cwd()
    model_file = f'{agent_name}_{cfg.run.train_steps}.zip'
    model_path = f'{orig_cwd}/{model_file}'
    if not os.path.exists(model_path):
        model.set_logger(custom_logger)
        model.learn(total_timesteps=cfg.run.train_steps, log_interval=10)
        model.save(model_file)
    else:
        print(f'Model {model_path} already exists. Skipping training...')
        model = agent_class.load(model_path, env)

    mean_reward, std_reward, mean_profit, std_profit = evaluate_policy(model, env, n_eval_episodes=5)
    logging.info(f'Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}')
    logging.info(f'Mean profit: {mean_profit:.2f} +/- {std_profit:.2f}')
    env.save_history()
    # for n in [2, 4, 30, 365]:
    #     save_path = f'last_{n}_days_plot.png'
    #     env.render_all(last_n_days=n, save_path=save_path)
    env.render_all()
    quantstats_summary(env.history, agent_name)


if __name__ == '__main__':
    main()
