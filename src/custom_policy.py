from typing import Callable, Tuple, Optional

import numpy as np
from gymnasium import spaces
import torch as th
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.type_aliases import Schedule
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy

from src.noisy_layers import NoisyLinear


HIDDEN_SIZE = 64


class CustomNetwork(nn.Module):

    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = HIDDEN_SIZE,
        last_layer_dim_vf: int = HIDDEN_SIZE,
    ):
        super().__init__()

        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, last_layer_dim_pi),
            nn.ReLU(),
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, last_layer_dim_vf),
            nn.ReLU(),
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        self.use_noise = kwargs.pop('use_noise', False)
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)

    def _build(self, lr_schedule: Schedule) -> None:
        super(CustomActorCriticPolicy, self)._build(lr_schedule)
        if self.use_noise:
            self.action_net = NoisyLinear(self.action_net.in_features, self.action_net.out_features)
            self.value_net = NoisyLinear(self.value_net.in_features, self.value_net.out_features)

    def resample(self):
        for module in (self.action_net, self.value_net):
            if isinstance(module, NoisyLinear):
                module.resample()


class CustomMultiHeadPolicy(CustomActorCriticPolicy):
    def _build(self, lr_schedule: Schedule) -> None:
        super(CustomActorCriticPolicy, self)._build(lr_schedule)
        self.action_net = CustomMultiHead(self.action_net.in_features)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)

        distribution = []
        for h in self.action_net.heads:
            mean_actions = h(latent_pi)
            action_dist = DiagGaussianDistribution(1)
            distribution.append(
                action_dist.proba_distribution(mean_actions, th.tensor([0] * 1))
            )

        actions = th.stack([d.get_actions(deterministic=deterministic) for d in distribution])
        if not deterministic:
            mask = np.random.random((96, obs.shape[0]))
            threshold = 0.20
            mask[mask < threshold] = 0
            mask[mask >= threshold] = 1
            mask = th.tensor(mask)
            actions[mask == 0] = th.tensor([-20.0] * 1)

        log_probs = []
        for i, d in enumerate(distribution):
            log_probs.append(
                d.log_prob(actions[i, :, :])
            )

        log_probs = th.stack(log_probs)
        if not deterministic:
            log_probs[mask == 0] = 0.0
        log_prob = th.sum(log_probs, dim=0)

        actions = th.transpose(actions, 0, 1)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = []
        for h in self.action_net.heads:
            mean_actions = h(latent_pi)
            action_dist = DiagGaussianDistribution(1)
            distribution.append(
                action_dist.proba_distribution(mean_actions, th.tensor([0] * 1))
            )

        actions = th.stack(actions.split(1, dim=1))
        mask = (actions > -20)[:, :, 0]

        log_probs = []
        for i, d in enumerate(distribution):
            log_probs.append(
                d.log_prob(actions[i])
            )
        log_probs = th.stack(log_probs)
        log_probs[mask == 0] = 0.0
        log_prob = th.sum(log_probs, dim=0)

        values = self.value_net(latent_vf)
        entropy = th.sum(th.stack([d.entropy() for d in distribution]))
        return values, log_prob, entropy


class CustomMultiHead(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()

        self.heads = [nn.Linear(in_features, 1) for _ in range(96)]

    def forward(self, x):
        outs = [h(x) for h in self.heads]
        return th.concat(outs, dim=1)
