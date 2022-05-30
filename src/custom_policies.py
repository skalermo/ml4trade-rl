from typing import Tuple

import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy


class NormalizationPolicy(ActorCriticPolicy):
    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        normalized_obs = self._normalize_obs(obs)
        actions, values, log_prob = super().forward(normalized_obs, deterministic)
        return self._normalize_actions(actions), values, log_prob

    def _normalize_obs(self, obs: th.Tensor):
        print(f'Obs: {obs}')
        return obs

    def _normalize_actions(self, actions: th.Tensor):
        print(f'Actions: {actions}')
        return actions
