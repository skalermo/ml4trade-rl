import logging
import os
import random
import sys
from datetime import datetime, time, timedelta
from pathlib import Path

import hydra
import numpy as np
import torch
from ml4trade.domain.units import *
from ml4trade.misc import (
    IntervalWrapper,
    ActionWrapper,
    AvgIntervalPriceRetriever,
)
from ml4trade.misc.norm_ds_wrapper import DummyWrapper
from ml4trade.simulation_env import SimulationEnv
from omegaconf import DictConfig, OmegaConf

from stable_baselines3 import A2C, PPO
from stable_baselines3.common import logger
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from src import es
from src.callbacks import ResampleCallback
from src.evaluation import evaluate_policy
from src.obs_wrapper import FilterObsWrapper
from src.price_types_wrapper import PriceTypeObsWrapper
from src.reward_shaping import RewardShapingEnv
from src.utils import get_weather_df, get_prices_df, get_data_strategies, linear_schedule
from src.custom_policy import CustomActorCriticPolicy, CustomMultiHeadPolicy


def setup_sim_env(cfg: DictConfig, split_ratio: float = 0.8, seed: int = None):
    weather_df = get_weather_df()
    prices_df = get_prices_df()

    data_strategies = get_data_strategies(cfg, weather_df, prices_df)
    data_strategies = {k: DummyWrapper(v) if k not in ('production', 'market') else v for k, v in data_strategies.items()}
    avg_interval_price_retriever = AvgIntervalPriceRetriever(prices_df, interval_days=cfg.run.aw_interval)
    max_power = cfg.env.max_solar_power + cfg.env.max_wind_power

    def create_train_env():
        env = SimulationEnv(
            data_strategies,
            start_datetime=datetime.fromisoformat(cfg.env.start),
            end_datetime=datetime.fromisoformat(cfg.env.end) - timedelta(days=90 * 2),
            scheduling_time=time.fromisoformat(cfg.env.scheduling_time),
            action_replacement_time=time.fromisoformat(cfg.env.action_time),
            prosumer_init_balance=Currency(cfg.env.init_balance),
            battery_capacity=MWh(cfg.env.bat_cap),
            battery_init_charge=MWh(cfg.env.bat_init_charge),
            battery_efficiency=cfg.env.bat_efficiency,
            use_reward_penalties=False,
        )
        iw_env = IntervalWrapper(
            env,
            interval=timedelta(days=cfg.run.train_ep_len),
            # split_ratio=split_ratio,
            split_ratio=1.0,
            randomly_set_battery=True,
        )
        aw_env = ActionWrapper(iw_env, ref_power_MW=max_power / 2, avg_interval_price_retriever=avg_interval_price_retriever)
        fow_env = FilterObsWrapper(aw_env, -2)
        if cfg.run.shaping_coef is not None:
            rs_env = RewardShapingEnv(fow_env, shaping_coef=cfg.run.shaping_coef)
            return rs_env
        return fow_env

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
        use_reward_penalties=False,
    )
    iw_env = IntervalWrapper(
        env,
        interval=timedelta(days=cfg.run.train_ep_len),
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
        use_reward_penalties=False,
    )
    eval_iw_env = IntervalWrapper(
        eval_env,
        interval=timedelta(days=90),
        # split_ratio=split_ratio,
        split_ratio=1.0,
        randomly_set_battery=False,
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
        use_reward_penalties=False,
    )
    test_iw_env = IntervalWrapper(
        test_env,
        interval=timedelta(days=90),
        split_ratio=1.0,
        randomly_set_battery=False,
    )
    aw_env = ActionWrapper(iw_env, ref_power_MW=max_power / 2, avg_interval_price_retriever=avg_interval_price_retriever)
    eval_aw_env = ActionWrapper(eval_iw_env, ref_power_MW=max_power / 2, avg_interval_price_retriever=avg_interval_price_retriever)
    test_aw_env = ActionWrapper(test_iw_env, ref_power_MW=max_power / 2, avg_interval_price_retriever=avg_interval_price_retriever)
    fow_env = FilterObsWrapper(aw_env, -2)
    eval_fow_env = FilterObsWrapper(eval_aw_env, -2)
    test_fow_env = FilterObsWrapper(test_aw_env, -2)
    # pto_env = PriceTypeObsWrapper(fow_env, prices_df, timedelta(days=cfg.run.grouping_period), eval_aw_env.test_data_start)
    # eval_pto = PriceTypeObsWrapper(eval_fow_env, prices_df, timedelta(days=cfg.run.grouping_period), eval_aw_env.test_data_start)
    # test_pto = PriceTypeObsWrapper(test_fow_env, prices_df, timedelta(days=cfg.run.grouping_period), eval_aw_env.test_data_start)
    # res_env = pto_env
    # res_eval_env = eval_pto
    # res_test_env = test_pto

    # res_env = fow_env
    res_env = VecMonitor(DummyVecEnv(
        [lambda: create_train_env()] * cfg.run.train_envs,
    ))
    res_eval_env = eval_fow_env
    res_test_env = test_fow_env
    if seed is not None:
        if isinstance(res_env, VecMonitor):
            res_env.seed(seed=seed)
        else:
            res_env.reset(seed=seed)
        res_eval_env.reset(seed=seed)
        res_test_env.reset(seed=seed)
    return res_env, res_eval_env, res_test_env


@hydra.main(config_path='conf', config_name='config', version_base='1.1')
def main(cfg: DictConfig) -> None:
    orig_cwd = hydra.utils.get_original_cwd()

    if cfg.agent.get('buffer_size') is not None:
        agent_name = 'ddpg'
    elif cfg.agent.get('use_rms_prop') is not None:
        agent_name = 'a2c'
    else:
        agent_name = 'ppo'

    with open(Path(orig_cwd) / __file__, 'r') as f:
        logging.info(f.read())
    with open(Path(orig_cwd) / 'src' / 'custom_policy.py', 'r') as f:
        logging.info(f.read())

    logging.info(' '.join(sys.argv))
    logging.info(f'agent={agent_name}')
    logging.info(OmegaConf.to_yaml(cfg))

    agent_class = {
        'ppo': PPO,
        'a2c': A2C,
    }[agent_name]

    seed = cfg.run.get('seed')
    if seed is None:
        seed = int(datetime.now().timestamp())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    env, eval_env, test_env = setup_sim_env(cfg, split_ratio=0.8, seed=seed)

    model = agent_class(
        # 'MlpPolicy', env,
        CustomActorCriticPolicy, env,
        # CustomMultiHeadPolicy, env,
        verbose=1, seed=seed,
        **cfg.agent,
        # policy_kwargs=dict(use_noise=True),
        # **{**cfg.agent, **dict(learning_rate=linear_schedule(cfg.agent.learning_rate))},
    )

    eval_callback = EvalCallback(Monitor(eval_env), best_model_save_path='.',
                                 log_path='.', eval_freq=cfg.run.eval_freq,
                                 n_eval_episodes=3, deterministic=True,
                                 render=False)

    model_file = f'{agent_name}_{cfg.run.train_steps}.zip'
    model_path = f'{orig_cwd}/{model_file}'

    logging.info(f'seed: {seed}')
    logging.info(f'action space: {env.action_space.shape}')
    logging.info(f'observation space: {env.observation_space.shape}')

    cbs = CallbackList([eval_callback, ResampleCallback()])

    # model = agent_class.load(f'{orig_cwd}/best_model.zip', env)
    if cfg.run.pretrain:
        es.pretrain(
            cfg, (env, eval_env, test_env), seed,
            **cfg.pretrain,
        )
        model.policy = es.Ml4tradeIndividual.from_params(
            torch.load(f'es_{cfg.pretrain.pop_size}_{cfg.pretrain.iterations}.zip')
        ).policy

    if not os.path.exists(model_path):
        custom_logger = logger.configure('.', ['stdout', 'json', 'tensorboard'])
        model.set_logger(custom_logger)
        try:
            model.learn(total_timesteps=cfg.run.train_steps,
                        log_interval=max(1, 500 // cfg.agent.get('n_steps', 500)),
                        callback=eval_callback, reset_num_timesteps=False)
                        # callback=cbs, reset_num_timesteps=False)
        except Exception as e:
            logging.info(e)
            exit()
        model.save(model_file)
        print('Training finished.')
        if cfg.run.train_steps >= cfg.run.eval_freq:
            print('Loading best model...')
            model = agent_class.load('best_model', env)
    else:
        print(f'Model {model_path} already exists. Skipping training...')
        model = agent_class.load(model_path, env)

    # test_env.set_interval(timedelta(days=90))
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
