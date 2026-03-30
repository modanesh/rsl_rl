# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.utils import resolve_nn_activation

class ActorCriticDrift(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims: list = [512, 512, 256],
        critic_hidden_dims: list = [512, 512, 256],
        activation="elu",
        device=torch.device("cuda"),
        **kwargs,
    ):
        super().__init__()
        activation_fn = resolve_nn_activation(activation)

        self.a_o_dim = num_actor_obs
        self.c_o_dim = num_critic_obs
        self.a_dim = num_actions
        self.device = device
        self.last_log_probs = None # Kept for compatibility with rsl_rl storage

        actor_layers = []
        actor_input_dim = self.a_o_dim + self.a_dim  # obs + noise
        actor_layers.append(nn.Linear(actor_input_dim, actor_hidden_dims[0]))
        actor_layers.append(activation_fn)
        for i in range(len(actor_hidden_dims) - 1):
            actor_layers.append(nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1]))
            actor_layers.append(activation_fn)
        actor_layers.append(nn.Linear(actor_hidden_dims[-1], self.a_dim))
        self.actor = nn.Sequential(*actor_layers)

        critic_layers = []
        critic_layers.append(nn.Linear(self.c_o_dim, critic_hidden_dims[0]))
        critic_layers.append(activation_fn)
        for i in range(len(critic_hidden_dims) - 1):
            critic_layers.append(nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]))
            critic_layers.append(activation_fn)
        critic_layers.append(nn.Linear(critic_hidden_dims[-1], 1))
        self.critic = nn.Sequential(*critic_layers)

        self.to(device)

    def reset(self, dones=None):
        pass

    def act(self, observations, **kwargs):
        """Standard rollout action (1-NFE)"""
        z = torch.randn(observations.shape[0], self.a_dim, device=self.device)
        action = self.actor(torch.cat([observations, z], dim=-1))
        return action

    def act_multiple(self, observations, n_gen=16):
        """Generates multiple actions per state for computing the Drift Field"""
        B = observations.shape[0]
        z = torch.randn(B, n_gen, self.a_dim, device=self.device)
        obs_rep = observations.unsqueeze(1).repeat(1, n_gen, 1)
        actions = self.actor(torch.cat([obs_rep, z], dim=-1))
        return actions

    def act_inference(self, observations):
        """Deterministic fallback for evaluation"""
        z = torch.zeros(observations.shape[0], self.a_dim, device=self.device)
        return self.actor(torch.cat([observations, z], dim=-1))

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)

    @property
    def action_std(self):
        return torch.tensor(0.0, device=self.device)

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True