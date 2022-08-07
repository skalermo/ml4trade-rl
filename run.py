import logging
import os
import sys
from datetime import datetime, time, timedelta
from typing import List, Tuple
import warnings

import hydra
from ml4trade.domain.units import *
from ml4trade.misc import (
    IntervalWrapper,
    ActionWrapper,
    AvgMonthPriceRetriever,
)
from ml4trade.simulation_env import SimulationEnv
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import A2C, PPO
from stable_baselines3.common import logger

from src.reward_shaping import RewardShapingEnv
from src.utils import get_weather_df, get_prices_df, get_data_strategies
from src.evaluation import evaluate_policy, quantstats_summary


def _parse_conf_interval(interval: Union[int, List[Tuple[int, int]]]) -> Union[timedelta, List[Tuple[timedelta, int]]]:
    if isinstance(interval, int):
        return timedelta(days=interval)
    return list(map(lambda x: (timedelta(days=x[0]), x[1]), interval))


def setup_sim_env(cfg: DictConfig, split_ratio: float = 0.8):
    orig_cwd = hydra.utils.get_original_cwd()
    weather_df = get_weather_df()
    prices_df = get_prices_df()

    data_strategies = get_data_strategies(cfg, weather_df, prices_df)
    avg_month_price_retriever = AvgMonthPriceRetriever(prices_df)

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
    iw_env = IntervalWrapper(
        env,
        interval=_parse_conf_interval(cfg.run.interval),
        split_ratio=split_ratio,
        randomly_set_battery=True,
    )
    test_env = SimulationEnv(
        data_strategies,
        start_datetime=iw_env.test_data_start,
        end_datetime=datetime.fromisoformat(cfg.env.end),
        scheduling_time=time.fromisoformat(cfg.env.scheduling_time),
        action_replacement_time=time.fromisoformat(cfg.env.action_time),
        prosumer_init_balance=Currency(cfg.env.init_balance),
        battery_capacity=MWh(cfg.env.bat_cap),
        battery_init_charge=MWh(cfg.env.bat_init_charge),
        battery_efficiency=cfg.env.bat_efficiency,
        start_tick=iw_env.test_data_start_tick,
    )
    max_power = cfg.env.max_solar_power + cfg.env.max_wind_power
    aw_env = ActionWrapper(iw_env, ref_power_MW=max_power / 2, avg_month_price_retriever=avg_month_price_retriever)
    test_aw_env = ActionWrapper(test_env, ref_power_MW=max_power / 2, avg_month_price_retriever=avg_month_price_retriever)
    # rs_env = RewardShapingEnv(aw_env, shaping_coef=cfg.run.shaping_coef)
    # test_rs_env = RewardShapingEnv(test_aw_env, shaping_coef=cfg.run.shaping_coef)
    return aw_env, test_aw_env


@hydra.main(config_path='conf', config_name='config', version_base='1.1')
def main(cfg: DictConfig) -> None:
    logging.info(' '.join(sys.argv))
    agent_name = 'a2c' if cfg.agent.get('use_rms_prop') is not None else 'ppo'
    agent_class = {
        'ppo': PPO,
        'a2c': A2C,
    }[agent_name]
    logging.info(f'agent={agent_name}')
    logging.info(OmegaConf.to_yaml(cfg))

    env, test_env = setup_sim_env(cfg, 0.8)
    logging.info(f'action space: {env.action_space.shape}')
    logging.info(f'observation space: {env.observation_space.shape}')

    model = agent_class('MlpPolicy', env,
                        **cfg.agent, verbose=1)

    orig_cwd = hydra.utils.get_original_cwd()
    model_file = f'{agent_name}_{cfg.run.train_steps}.zip'
    model_path = f'{orig_cwd}/{model_file}'

    if not os.path.exists(model_path):
        custom_logger = logger.configure('.', ['stdout', 'json', 'tensorboard'])
        model.set_logger(custom_logger)
        model.learn(total_timesteps=cfg.run.train_steps,
                    log_interval=max(1, 500 // cfg.agent.n_steps))
        model.save(model_file)
    else:
        print(f'Model {model_path} already exists. Skipping training...')
        model = agent_class.load(model_path, env)

    mean_reward, std_reward, mean_profit, std_profit = evaluate_policy(model, test_env, n_eval_episodes=5)
    logging.info(f'Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}')
    logging.info(f'Mean profit: {mean_profit:.2f} +/- {std_profit:.2f}')

    test_env.save_history()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for n in [2, 4, 30]:
            save_path = f'last_{n}_days_plot.png'
            test_env.render_all(last_n_days=n, n_days_offset=0, save_path=save_path)


if __name__ == '__main__':
    main()
