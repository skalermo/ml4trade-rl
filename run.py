import logging
import os
import sys
from datetime import datetime, time, timedelta
from pathlib import Path

import hydra
from ml4trade.domain.units import *
from ml4trade.misc import (
    IntervalWrapper,
    ActionWrapper,
    AvgMonthPriceRetriever,
)
from ml4trade.misc.norm_ds_wrapper import DummyWrapper
from ml4trade.simulation_env import SimulationEnv
from omegaconf import DictConfig, OmegaConf

import gymnasium
sys.modules["gym"] = gymnasium
from stable_baselines3 import A2C, PPO
from stable_baselines3.common import logger
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.evaluation import evaluate_policy
from src.obs_wrapper import FilterObsWrapper
from src.price_types_wrapper import PriceTypeObsWrapper
from src.utils import get_weather_df, get_prices_df, get_data_strategies


def setup_sim_env(cfg: DictConfig, split_ratio: float = 0.8, seed: int = None):
    weather_df = get_weather_df()
    prices_df = get_prices_df()

    data_strategies = get_data_strategies(cfg, weather_df, prices_df)
    data_strategies = {k: DummyWrapper(v) for k, v in data_strategies.items()}
    avg_month_price_retriever = AvgMonthPriceRetriever(prices_df)

    env = SimulationEnv(
        data_strategies,
        start_datetime=datetime.fromisoformat(cfg.env.start),
        # end_datetime=datetime.fromisoformat(cfg.env.end),
        end_datetime=datetime.fromisoformat(cfg.env.end) - timedelta(days=90 * 2),
        scheduling_time=time.fromisoformat(cfg.env.scheduling_time),
        action_replacement_time=time.fromisoformat(cfg.env.action_time),
        prosumer_init_balance=Currency(cfg.env.init_balance),
        battery_capacity=MWh(cfg.env.bat_cap),
        battery_init_charge=MWh(cfg.env.bat_init_charge),
        battery_efficiency=cfg.env.bat_efficiency,
    )
    iw_env = IntervalWrapper(
        env,
        interval=timedelta(days=cfg.run.interval),
        # split_ratio=split_ratio,
        split_ratio=1.0,
        randomly_set_battery=True,
    )
    eval_env = SimulationEnv(
        data_strategies,
        start_datetime=datetime.fromisoformat(cfg.env.end) - timedelta(days=90 * 2),
        # end_datetime=datetime.fromisoformat(cfg.env.end),
        end_datetime=datetime.fromisoformat(cfg.env.end) - timedelta(days=90),
        scheduling_time=time.fromisoformat(cfg.env.scheduling_time),
        action_replacement_time=time.fromisoformat(cfg.env.action_time),
        prosumer_init_balance=Currency(cfg.env.init_balance),
        battery_capacity=MWh(cfg.env.bat_cap),
        battery_init_charge=MWh(cfg.env.bat_init_charge),
        battery_efficiency=cfg.env.bat_efficiency,
        start_tick=iw_env.test_data_start_tick,
    )
    eval_iw_env = IntervalWrapper(
        eval_env,
        interval=timedelta(days=cfg.run.interval),
        # split_ratio=split_ratio,
        split_ratio=1.0,
        randomly_set_battery=True,
    )

    test_env = SimulationEnv(
        data_strategies,
        # start_datetime=iw_env.test_data_start,
        start_datetime=datetime.fromisoformat(cfg.env.end) - timedelta(days=90),
        end_datetime=datetime.fromisoformat(cfg.env.end),
        scheduling_time=time.fromisoformat(cfg.env.scheduling_time),
        action_replacement_time=time.fromisoformat(cfg.env.action_time),
        prosumer_init_balance=Currency(cfg.env.init_balance),
        battery_capacity=MWh(cfg.env.bat_cap),
        battery_init_charge=MWh(cfg.env.bat_init_charge),
        battery_efficiency=cfg.env.bat_efficiency,
        start_tick=eval_iw_env.test_data_start_tick,
    )
    test_iw_env = IntervalWrapper(
        test_env,
        interval=timedelta(days=cfg.run.interval),
        split_ratio=1.0,
        randomly_set_battery=True,
    )
    max_power = cfg.env.max_solar_power + cfg.env.max_wind_power
    aw_env = ActionWrapper(iw_env, ref_power_MW=max_power / 2, avg_month_price_retriever=avg_month_price_retriever)
    eval_aw_env = ActionWrapper(eval_iw_env, ref_power_MW=max_power / 2, avg_month_price_retriever=avg_month_price_retriever)
    test_aw_env = ActionWrapper(test_iw_env, ref_power_MW=max_power / 2, avg_month_price_retriever=avg_month_price_retriever)
    fow_env = FilterObsWrapper(aw_env, 0)
    eval_fow_env = FilterObsWrapper(eval_aw_env, 0)
    test_fow_env = FilterObsWrapper(test_aw_env, 0)
    pto_env = PriceTypeObsWrapper(fow_env, prices_df, timedelta(days=cfg.run.grouping_period), eval_aw_env.test_data_start)
    eval_pto = PriceTypeObsWrapper(eval_fow_env, prices_df, timedelta(days=cfg.run.grouping_period), eval_aw_env.test_data_start)
    test_pto = PriceTypeObsWrapper(test_fow_env, prices_df, timedelta(days=cfg.run.grouping_period), eval_aw_env.test_data_start)
    res_env = pto_env
    res_eval_env = eval_pto
    res_test_env = test_pto
    if seed is not None:
        res_env.reset(seed=seed)
        res_eval_env.reset(seed=seed)
        res_test_env.reset(seed=seed)
    return res_env, res_eval_env, res_test_env


@hydra.main(config_path='conf', config_name='config', version_base='1.1')
def main(cfg: DictConfig) -> None:
    orig_cwd = hydra.utils.get_original_cwd()

    agent_name = 'a2c' if cfg.agent.get('use_rms_prop') is not None else 'ppo'

    with open(Path(orig_cwd) / __file__, 'r') as f:
        logging.info(f.read())

    logging.info(' '.join(sys.argv))
    logging.info(f'agent={agent_name}')
    logging.info(OmegaConf.to_yaml(cfg))

    agent_class = {
        'ppo': PPO,
        'a2c': A2C,
    }[agent_name]

    seed = cfg.run.get('seed') or int(datetime.now().timestamp())
    env, eval_env, test_env = setup_sim_env(cfg, split_ratio=0.8, seed=seed)
    model = agent_class('MlpPolicy', env,
                        **cfg.agent, verbose=1, seed=seed)
    eval_callback = EvalCallback(Monitor(eval_env), best_model_save_path='.',
                                 log_path='.', eval_freq=cfg.run.eval_freq,
                                 n_eval_episodes=2, deterministic=True,
                                 render=False)

    model_file = f'{agent_name}_{cfg.run.train_steps}.zip'
    model_path = f'{orig_cwd}/{model_file}'

    logging.info(f'seed: {seed}')
    logging.info(f'action space: {env.action_space.shape}')
    logging.info(f'observation space: {env.observation_space.shape}')

    if not os.path.exists(model_path):
        custom_logger = logger.configure('.', ['stdout', 'json', 'tensorboard'])
        model.set_logger(custom_logger)
        model.learn(total_timesteps=cfg.run.train_steps,
                    log_interval=max(1, 500 // cfg.agent.n_steps),
                    callback=eval_callback)
        model.save(model_file)
        print('Training finished.')
        if cfg.run.train_steps >= cfg.run.eval_freq:
            print('Loading best model...')
            model = agent_class.load('best_model', env)
    else:
        print(f'Model {model_path} already exists. Skipping training...')
        model = agent_class.load(model_path, env)
    # model = agent_class.load(f'{orig_cwd}/best_model.zip', env)

    test_env.set_interval(timedelta(days=90))
    mean_reward, std_reward, mean_profit, std_profit = evaluate_policy(model, test_env, n_eval_episodes=3)
    logging.info(f'Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}')
    logging.info(f'Mean profit: {mean_profit:.2f} +/- {std_profit:.2f}')

    test_env.save_history()

    if cfg.run.render_all:
        for n in [2, 4, 10, 30]:
            save_path = f'last_{n}_days_plot.png'
            test_env.render_all(last_n_days=n, n_days_offset=0, save_path=save_path)


if __name__ == '__main__':
    main()
