# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


class ActorCriticEmbodiment(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        urdf_feature_dim=None,
        urdf_mode='concat',
        urdf_tower_ratio=0.5,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticEmbodiment.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.urdf_mode = urdf_mode
        self.urdf_feature_dim = urdf_feature_dim

        # Validate URDF configuration
        if urdf_mode != 'none' and urdf_feature_dim is None:
            raise ValueError(f"urdf_feature_dim required when urdf_mode='{urdf_mode}'")

        activation = resolve_nn_activation(activation)

        # ============================================================
        # MODE 1: CONCATENATION
        # ============================================================
        if urdf_mode == 'concat':
            print(f"[INFO - ActorCriticEmbodiment] Mode: Concatenation")
            print(f"    Actor input: obs({num_actor_obs}) + urdf({urdf_feature_dim}) = {num_actor_obs + urdf_feature_dim}")
            print(f"    Critic input: obs({num_critic_obs}) + urdf({urdf_feature_dim}) = {num_critic_obs + urdf_feature_dim}")

            mlp_input_dim_a = num_actor_obs + urdf_feature_dim
            mlp_input_dim_c = num_critic_obs + urdf_feature_dim

            # Build Actor network (standard MLP with concatenated input)
            actor_layers = []
            actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
            actor_layers.append(activation)

            for layer_index in range(len(actor_hidden_dims)):
                if layer_index == len(actor_hidden_dims) - 1:
                    actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
                else:
                    actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                    actor_layers.append(activation)

            self.actor = nn.Sequential(*actor_layers)

            # Build Critic network (standard MLP with concatenated input)
            critic_layers = []
            critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
            critic_layers.append(activation)

            for layer_index in range(len(critic_hidden_dims)):
                if layer_index == len(critic_hidden_dims) - 1:
                    critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
                else:
                    critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                    critic_layers.append(activation)

            self.critic = nn.Sequential(*critic_layers)

        # ============================================================
        # MODE 2: TOWER (SEPARATE PROCESSING + FUSION)
        # ============================================================
        elif urdf_mode == 'tower':
            print(f"[INFO - ActorCriticEmbodiment] Mode: Separate Towers")

            # Calculate tower dimensions (like Dreamer)
            self.actor_urdf_tower_dim = int(actor_hidden_dims[0] * urdf_tower_ratio)
            self.critic_urdf_tower_dim = int(critic_hidden_dims[0] * urdf_tower_ratio)

            print(f"    Actor Architecture:")
            print(f"      obs({num_actor_obs}) → obs_tower → {actor_hidden_dims[0]}")
            print(f"      urdf({urdf_feature_dim}) → urdf_tower → {self.actor_urdf_tower_dim}")
            print(f"      concat({actor_hidden_dims[0] + self.actor_urdf_tower_dim}) → fusion → {actor_hidden_dims[0]}")
            print(f"      fusion → {actor_hidden_dims[1:]} → actions({num_actions})")

            print(f"    Critic Architecture:")
            print(f"      obs({num_critic_obs}) → obs_tower → {critic_hidden_dims[0]}")
            print(f"      urdf({urdf_feature_dim}) → urdf_tower → {self.critic_urdf_tower_dim}")
            print(f"      concat({critic_hidden_dims[0] + self.critic_urdf_tower_dim}) → fusion → {critic_hidden_dims[0]}")
            print(f"      fusion → {critic_hidden_dims[1:]} → value(1)")

            # -------------------- ACTOR TOWER --------------------
            # Actor observation tower (first layer only)
            self.actor_obs_tower = nn.Sequential(
                nn.Linear(num_actor_obs, actor_hidden_dims[0]),
                activation
            )

            # Actor URDF tower
            self.actor_urdf_tower = nn.Sequential(
                nn.Linear(urdf_feature_dim, self.actor_urdf_tower_dim),
                activation
            )

            # Actor fusion layer (combines both towers)
            self.actor_fusion = nn.Sequential(
                nn.Linear(actor_hidden_dims[0] + self.actor_urdf_tower_dim, actor_hidden_dims[0]),
                activation
            )

            # Remaining layers AFTER fusion (FIXED: process ALL remaining layers)
            actor_remaining_layers = []
            for i in range(len(actor_hidden_dims) - 1):
                if i == len(actor_hidden_dims) - 2:
                    # Last layer to output
                    actor_remaining_layers.append(nn.Linear(actor_hidden_dims[i], num_actions))
                else:
                    # Intermediate layers
                    actor_remaining_layers.append(nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1]))
                    actor_remaining_layers.append(activation)

            self.actor_remaining = nn.Sequential(*actor_remaining_layers) if actor_remaining_layers else nn.Identity()

            # -------------------- CRITIC TOWER --------------------
            # Observation tower
            self.critic_obs_tower = nn.Sequential(
                nn.Linear(num_critic_obs, critic_hidden_dims[0]),
                activation
            )

            # Critic URDF tower
            self.critic_urdf_tower = nn.Sequential(
                nn.Linear(urdf_feature_dim, self.critic_urdf_tower_dim),
                activation
            )

            # Critic fusion layer (combines both towers)
            self.critic_fusion = nn.Sequential(
                nn.Linear(critic_hidden_dims[0] + self.critic_urdf_tower_dim, critic_hidden_dims[0]),
                activation
            )

            # Remaining layers AFTER fusion (FIXED: process ALL remaining layers)
            critic_remaining_layers = []
            for i in range(len(critic_hidden_dims) - 1):
                if i == len(critic_hidden_dims) - 2:
                    # Last layer to output
                    critic_remaining_layers.append(nn.Linear(critic_hidden_dims[i], 1))
                else:
                    # Intermediate layers
                    critic_remaining_layers.append(nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]))
                    critic_remaining_layers.append(activation)

            self.critic_remaining = nn.Sequential(*critic_remaining_layers) if critic_remaining_layers else nn.Identity()

        else:
            raise ValueError(f"Unknown urdf_mode: {urdf_mode}. Must be 'none', 'concat', or 'tower'")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _process_actor_input(self, observations, urdf_features):
        if urdf_features is None:
            raise ValueError("URDF features not provided to forward pass.")

        if self.urdf_mode == 'concat':
            combined_input = torch.cat([observations, urdf_features], dim=-1)
            return self.actor(combined_input)

        elif self.urdf_mode == 'tower':
            obs_embed = self.actor_obs_tower(observations)
            urdf_embed = self.actor_urdf_tower(urdf_features)
            combined = torch.cat([obs_embed, urdf_embed], dim=-1)
            fused = self.actor_fusion(combined)

            # Process through remaining layers
            return self.actor_remaining(fused)

    def _process_critic_input(self, observations, urdf_features):
        """Process critic input with URDF features from mini-batch."""
        if urdf_features is None:
            raise ValueError("URDF features not provided to forward pass.")

        if self.urdf_mode == 'concat':
            combined_input = torch.cat([observations, urdf_features], dim=-1)
            return self.critic(combined_input)

        elif self.urdf_mode == 'tower':
            obs_embed = self.critic_obs_tower(observations)
            urdf_embed = self.critic_urdf_tower(urdf_features)
            combined = torch.cat([obs_embed, urdf_embed], dim=-1)
            fused = self.critic_fusion(combined)

            # Process through remaining layers
            return self.critic_remaining(fused)

    def update_distribution(self, observations, urdf_features=None):
        """Update action distribution with URDF features."""
        mean = self._process_actor_input(observations, urdf_features)

        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)

        self.distribution = Normal(mean, std)

    def act(self, observations, urdf_features=None, **kwargs):
        """Sample actions with URDF features."""
        self.update_distribution(observations, urdf_features)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, urdf_features=None):
        """Get mean actions with URDF features."""
        return self._process_actor_input(observations, urdf_features)

    def evaluate(self, critic_observations, urdf_features=None, **kwargs):
        """Evaluate value with URDF features."""
        return self._process_critic_input(critic_observations, urdf_features)

    def load_state_dict(self, state_dict, strict=True):
        """Load model parameters."""
        super().load_state_dict(state_dict, strict=strict)
        return True
