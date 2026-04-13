# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from rsl_rl.utils import resolve_nn_activation


class ActorCriticDrift(nn.Module):
    """
    Generative Actor-Critic for DriftPO.
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int] = [512, 512, 256],
        critic_hidden_dims: list[int] = [512, 512, 256],
        q_hidden_dims: list[int] | None = None,  # defaults to critic_hidden_dims
        activation: str = "elu",
        device=torch.device("cuda"),
        **kwargs,
    ):
        super().__init__()
        self.a_o_dim = num_actor_obs
        self.c_o_dim = num_critic_obs
        self.a_dim   = num_actions
        self.device  = device
        self.last_log_probs = None  # kept for RolloutStorage interface compatibility

        def make_mlp(input_dim: int, hidden_dims: list[int], output_dim: int) -> nn.Sequential:
            """Utility: build a fully-connected MLP with the shared activation."""
            layers: list[nn.Module] = []
            in_d = input_dim
            for h in hidden_dims:
                layers += [nn.Linear(in_d, h), resolve_nn_activation(activation)]
                in_d = h
            layers.append(nn.Linear(in_d, output_dim))
            return nn.Sequential(*layers)

        self.actor = make_mlp(
            input_dim=self.a_o_dim + self.a_dim,  # obs concatenated with noise
            hidden_dims=actor_hidden_dims,
            output_dim=self.a_dim,
        )

        self.critic = make_mlp(
            input_dim=self.c_o_dim,
            hidden_dims=critic_hidden_dims,
            output_dim=1,
        )

        q_dims = q_hidden_dims if q_hidden_dims is not None else critic_hidden_dims
        self.q_network = make_mlp(
            input_dim=self.c_o_dim + self.a_dim,  # obs concatenated with action
            hidden_dims=q_dims,
            output_dim=1,
        )
        self.q_target = copy.deepcopy(self.q_network)
        for p in self.q_target.parameters():
            p.requires_grad_(False)

        self.to(device)

    def reset(self, dones=None):
        pass  # no recurrent state

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Single-sample action for environment rollout (1-NFE).
        Returns: [B, D]
        """
        z = torch.randn(observations.shape[0], self.a_dim, device=self.device)
        return self.actor(torch.cat([observations, z], dim=-1))

    def act_multiple(self, observations: torch.Tensor, n_gen: int = 16) -> torch.Tensor:
        B = observations.shape[0]
        z       = torch.randn(B, n_gen, self.a_dim, device=self.device)
        obs_rep = observations.unsqueeze(1).expand(-1, n_gen, -1)  # [B, n_gen, obs_dim]
        return self.actor(torch.cat([obs_rep, z], dim=-1))         # [B, n_gen, D]

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        """Deterministic action (zero noise). For evaluation / deployment only."""
        z = torch.zeros(observations.shape[0], self.a_dim, device=self.device)
        return self.actor(torch.cat([observations, z], dim=-1))

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.critic(critic_observations)

    def q_value(self, critic_observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.q_network(torch.cat([critic_observations, actions], dim=-1))

    def actor_critic_parameters(self) -> list[nn.Parameter]:
        """Actor + V-Critic parameters for the main optimizer."""
        return list(self.actor.parameters()) + list(self.critic.parameters())

    def q_parameters(self) -> list[nn.Parameter]:
        """Q-Network parameters for the Q-optimizer."""
        return list(self.q_network.parameters())

    def q_target_network(self) -> nn.Module:
        return self.q_target

    @property
    def action_std(self) -> torch.Tensor:
        """
        The implicit action std comes from the noise distribution N(0,I).
        Return 0 here for logging compatibility with rsl_rl (which expects
        a scalar representing the explicit Gaussian sigma of a Normal policy).
        """
        return torch.tensor(0.0, device=self.device)

    def load_state_dict(self, state_dict, strict: bool = True):
        super().load_state_dict(state_dict, strict=strict)
        return True