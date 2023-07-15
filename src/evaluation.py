from typing import Union

import numpy as np
from gymnasium import Wrapper
from ml4trade.simulation_env import SimulationEnv


def evaluate_policy(model, env: Union[SimulationEnv, Wrapper], n_eval_episodes: int = 5):
    episode_rewards = []
    episode_profits = []
    for i in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
        episode_rewards.append(ep_reward)
        episode_profits.append(env.history[env.history._cur_tick_to_idx() - 1]['wallet_balance'])
        print(episode_profits[-1])
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_profit = np.mean(episode_profits)
    std_profit = np.std(episode_profits)
    return mean_reward, std_reward, mean_profit, std_profit
