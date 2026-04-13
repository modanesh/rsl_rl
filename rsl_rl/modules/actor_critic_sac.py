# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SAC Actor-Critic module.

Key design decisions (informed by https://araffin.github.io/post/sac-massive-sim/):
  - Squashed Gaussian actor (tanh): outputs actions in [-1, 1], then rescaled to env bounds.
  - Action bounds are SET EXPLICITLY via set_action_bounds(). Using Isaac Lab's default
    bounds of [-100, 100] will cause pathological behaviour: blog post for details.
  - Asymmetric actor-critic: actor sees `obs`, critics see `privileged_obs` (+ action).
  - Twin Q-networks (clipped double-Q) to reduce overestimation bias.
  - Separate optimizers for actor and critics (managed by SAC algorithm class).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from copy import deepcopy
from torch.distributions import Normal
from typing import Any

from rsl_rl.utils import resolve_nn_activation


class SACActorCritic(nn.Module):
    """Actor-Critic for SAC with twin Q-networks and squashed Gaussian actor."""

    is_recurrent: bool = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: tuple[int] | list[int] = (256, 256, 256),
        critic_hidden_dims: tuple[int] | list[int] = (256, 256, 256),
        activation: str = "elu",
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "SACActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs.keys()))
            )
        super().__init__()

        activation_fn = resolve_nn_activation(activation)
        self.num_actions = num_actions
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        actor_layers: list[nn.Module] = []
        actor_layers.append(nn.Linear(num_actor_obs, actor_hidden_dims[0]))
        actor_layers.append(activation_fn)
        for i in range(len(actor_hidden_dims)):
            if i == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[i], num_actions * 2))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1]))
                actor_layers.append(activation_fn)
        self.actor = nn.Sequential(*actor_layers)
        critic_input_dim = num_critic_obs + num_actions

        def _make_critic() -> nn.Sequential:
            layers: list[nn.Module] = []
            layers.append(nn.Linear(critic_input_dim, critic_hidden_dims[0]))
            layers.append(activation_fn)
            for i in range(len(critic_hidden_dims)):
                if i == len(critic_hidden_dims) - 1:
                    layers.append(nn.Linear(critic_hidden_dims[i], 1))
                else:
                    layers.append(nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]))
                    layers.append(activation_fn)
            return nn.Sequential(*layers)

        self.critic1 = _make_critic()
        self.critic2 = _make_critic()

        self.critic1_target = deepcopy(self.critic1)
        self.critic2_target = deepcopy(self.critic2)
        for p in self.critic1_target.parameters():
            p.requires_grad = False
        for p in self.critic2_target.parameters():
            p.requires_grad = False
        self.register_buffer("action_scale", torch.ones(num_actions))
        self.register_buffer("action_bias", torch.zeros(num_actions))

        # Disable argument validation for speed (same as rsl-rl PPO)
        Normal.set_default_validate_args(False)

        print(f"SAC Actor MLP:    {self.actor}")
        print(f"SAC Critic 1 MLP: {self.critic1}")
        print(f"SAC Critic 2 MLP: {self.critic2}")

    def set_action_bounds(self, action_low: torch.Tensor, action_high: torch.Tensor) -> None:
        """Set linear rescaling from tanh space [-1,1] to env action space.

        IMPORTANT: must be called before training starts.  Using Isaac Lab's
        default bounds of [-100, 100] will cause pathological exploration for
        SAC: see https://araffin.github.io/post/sac-massive-sim/ for details.

        Args:
            action_low:  Lower bound of env action space, shape (num_actions,).
            action_high: Upper bound of env action space, shape (num_actions,).
        """
        self.action_scale = ((action_high - action_low) / 2.0).to(self.action_scale.device)
        self.action_bias = ((action_high + action_low) / 2.0).to(self.action_bias.device)

        span = (action_high - action_low).max().item()
        if span > 10.0:
            print(
                f"[SACActorCritic WARNING] Action space span = {span:.1f}. "
                "Large bounds break SAC in Isaac Sim. Clip to PPO action percentiles. "
                "See https://araffin.github.io/post/sac-massive-sim/"
            )

    def reset(self, dones=None) -> None:  # noqa: D401
        pass

    def forward(self):  # noqa: D401
        raise NotImplementedError

    def _get_dist(self, obs: torch.Tensor) -> Normal:
        """Return the pre-squash Gaussian distribution N(mean, std)."""
        out = self.actor(obs)
        mean, log_std = out.chunk(2, dim=-1)
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)
        return Normal(mean, log_std.exp())

    def act(self, obs: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reparameterised stochastic action.

        Returns:
            action_env:   Action rescaled to env bounds,  shape (B, num_actions).
            log_prob:     Log-probability with tanh correction, shape (B, 1).
            action_tanh:  Tanh-squashed action in [-1, 1], shape (B, num_actions).
                          Store this in the replay buffer; critics operate in tanh space.
        """
        dist = self._get_dist(obs)
        z = dist.rsample()
        action_tanh = torch.tanh(z)
        log_prob = (dist.log_prob(z) - torch.log(1.0 - action_tanh.pow(2) + 1e-6)).sum(dim=-1, keepdim=True)
        action_env = action_tanh * self.action_scale + self.action_bias
        return action_env, log_prob, action_tanh

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action for evaluation / deployment (no sampling)."""
        out = self.actor(obs)
        mean, _ = out.chunk(2, dim=-1)
        action_tanh = torch.tanh(mean)
        return action_tanh * self.action_scale + self.action_bias

    def evaluate_actor(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Same as act() but intended for use inside the update loop (grads enabled)."""
        return self.act(obs)

    def evaluate_critics(
        self,
        critic_obs: torch.Tensor,
        action_tanh: torch.Tensor,
        use_target: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Q1 and Q2 values for the given (obs, action) pair.

        Args:
            critic_obs:   Privileged observations, shape (B, num_critic_obs).
            action_tanh:  Tanh-squashed actions in [-1, 1], shape (B, num_actions).
            use_target:   If True, query target critics (no gradient).

        Returns:
            q1, q2: Q-value tensors, each shape (B, 1).
        """
        x = torch.cat([critic_obs, action_tanh], dim=-1)
        if use_target:
            with torch.no_grad():
                return self.critic1_target(x), self.critic2_target(x)
        return self.critic1(x), self.critic2(x)

    def soft_update_target(self, tau: float) -> None:
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)

    @property
    def action_std(self) -> torch.Tensor:
        return torch.ones(self.num_actions, device=self.action_scale.device)

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        super().load_state_dict(state_dict, strict=strict)
        return True