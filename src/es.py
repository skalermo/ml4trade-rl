from typing import Dict

import torch
from evostrat import Individual
from evostrat import NormalPopulation, compute_centered_ranks
import gymnasium
import tqdm
from stable_baselines3 import PPO

from run import setup_sim_env
from src.custom_policy import CustomActorCriticPolicy

from src.evaluation import evaluate_policy


class Ml4tradeIndividual(Individual):
    env: gymnasium.Env = None

    def __init__(self):
        self.policy = self._new_policy()

    @staticmethod
    def _new_policy():
        action_space = Ml4tradeIndividual.env.action_space
        observation_space = Ml4tradeIndividual.env.observation_space
        return CustomActorCriticPolicy(observation_space, action_space, lambda x: 0.0)

    @staticmethod
    def from_params(params: Dict[str, torch.Tensor]):
        ind = Ml4tradeIndividual()
        ind.policy.mlp_extractor.policy_net.load_state_dict(
            {key: params[key] for key in ('0.weight', '0.bias', '2.weight', '2.bias')}
        )
        ind.policy.action_net.load_state_dict(
            {key: params[key] for key in ('weight', 'bias')}
        )
        return ind

    def fitness(self) -> float:
        obs, _ = self.env.reset()
        done = False
        r_tot = 0
        while not done:
            action = self.action(obs).squeeze().numpy()
            obs, r, _, done, _ = self.env.step(action)
            r_tot += r

        return r_tot

    def get_params(self) -> Dict[str, torch.Tensor]:
        action_net = self.policy.action_net
        policy_net = self.policy.mlp_extractor.policy_net
        return {
            **action_net.state_dict(),
            **policy_net.state_dict(),
        }

    def action(self, obs):
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            return self.policy._predict(x, deterministic=True)


def pretrain(
        cfg,
        seed,
        std: float = 0.1,
        lr: float = 0.01,
        iterations: int = 1000,
        pop_size: int = 16,
        eval_freq: int = 10,
):
    env, eval_env, test_env = setup_sim_env(cfg, split_ratio=0.8, seed=seed)
    Ml4tradeIndividual.env = env.envs[0]

    param_shapes = {k: v.shape for k, v in Ml4tradeIndividual().get_params().items()}
    population = NormalPopulation(param_shapes, Ml4tradeIndividual.from_params, std=std)

    optim = torch.optim.Adam(population.parameters(), lr=lr)
    pbar = tqdm.tqdm(range(1, iterations + 1))

    model = PPO(
        CustomActorCriticPolicy, env,
        verbose=1, seed=seed,
        **cfg.agent,
    )

    best_eval_reward = -float('inf')
    best_params = population.param_means

    for i in pbar:
        optim.zero_grad()
        with torch.multiprocessing.Pool() as pool:
            raw_fit = population.fitness_grads(pop_size, pool, compute_centered_ranks)
        optim.step()

        if i % eval_freq == 0:
            ind = Ml4tradeIndividual.from_params(population.param_means)
            model.policy = ind.policy
            mean_reward, std_reward, mean_profit, std_profit = evaluate_policy(model, eval_env, n_eval_episodes=1,
                                                                               silent=True)
            if mean_reward > best_eval_reward:
                best_eval_reward = mean_reward
                best_params = population.param_means

        pbar.set_description("fit avg: %0.3f, std: %0.3f" % (raw_fit.mean().item(), raw_fit.std().item()))

    best = Ml4tradeIndividual.from_params(best_params)
    model.policy = best.policy
    mean_reward, std_reward, mean_profit, std_profit = evaluate_policy(model, test_env, n_eval_episodes=1)
    test_env.save_history()

    if cfg.run.render_all:
        for n in [2, 4, 10, 30]:
            save_path = f'es_last_{n}_days_plot.png'
            test_env.render_all(last_n_days=n, n_days_offset=0, save_path=save_path)

    save_path = f'es_{pop_size}_{iterations}.zip'
    torch.save(best.get_params(), save_path)
