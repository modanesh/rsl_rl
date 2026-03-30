# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain

from rsl_rl.modules import ActorCriticDrift
from rsl_rl.storage import RolloutStorage


def compute_drift_target(x_gen, y_target, advantages, tau=0.2, eta=1.0, adv_temp=1.0, repel_coef=0.1, max_act=3.0):
    """
    Computes the Advantage-Weighted Mean-Shift Drift Field.
    x_gen: [B, Ngen, D] - Generated samples to construct the field
    y_target: [B, 1, D] - The empirical actions taken in the environment
    advantages: [B, 1] - GAE advantages of those empirical actions
    """
    B, Ngen, D = x_gen.shape

    # With a single positive sample per state, the kernel normalizes to 1.0, so it cancels.
    # Using normalized form directly preserves the anti-symmetry property.
    attract_weight = (advantages / (advantages.abs().max() + 1e-8)).unsqueeze(-1)
    V_attract = attract_weight * (y_target - x_gen)

    # 2. Kernelized Repulsion V-(x)
    # Repels generated actions from each other to maintain multimodality
    diff_neg = x_gen.unsqueeze(1) - x_gen.unsqueeze(2)  # x_j - x_i (toward others)
    dist_neg = torch.norm(diff_neg, p=2, dim=-1)  # [B, Ngen, Ngen]

    mask = torch.eye(Ngen, device=x_gen.device).unsqueeze(0).bool()
    dist_neg.masked_fill_(mask, 1e6)

    logit_neg = -dist_neg / tau
    W_neg = torch.softmax(logit_neg, dim=-1)  # [B, Ngen, Ngen]

    V_repel = torch.sum(W_neg.unsqueeze(-1) * diff_neg, dim=2)  # [B, Ngen, D]

    # Total Drift Field
    V = V_attract - repel_coef * V_repel

    # RMS normalization: stabilizes regression targets when field magnitude is erratic
    rms = (V.detach().pow(2).mean(dim=[1, 2], keepdim=True) + 1e-8).sqrt()
    V = V / rms

    # Extract magnitudes for logging (using .item() to detach and store as floats)
    metrics = {
        "drift_attract_mag": V_attract.norm(dim=-1).mean().item(),
        "drift_repel_mag": V_repel.norm(dim=-1).mean().item(),
        "drift_field_rms": rms.mean().item()
    }

    return (x_gen + eta * V).clamp(-max_act, max_act).detach(), metrics


class DriftPO:
    policy: ActorCriticDrift

    def __init__(
            self,
            policy,
            num_learning_epochs=1,
            num_mini_batches=1,
            clip_param=0.2,
            gamma=0.998,
            lam=0.95,
            value_loss_coef=1.0,
            learning_rate=1e-3,
            max_grad_norm=1.0,
            use_clipped_value_loss=True,
            device="cpu",
            normalize_advantage_per_mini_batch=False,
            multi_gpu_cfg: dict | None = None,
            drift_tau: float = 0.2,
            drift_eta: float = 0.5,
            drift_ngen: int = 4,
            drift_batch_size: int = 512,
            **kwargs,
    ):
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.policy = policy
        self.policy.to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.storage: RolloutStorage = None
        self.transition = RolloutStorage.Transition()

        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        # Drift Hyperparameters
        self.drift_tau = drift_tau
        self.drift_eta = drift_eta
        self.drift_ngen = drift_ngen
        self.drift_batch_size = drift_batch_size
        self.max_act = 3.0
        self.rnd = None
        self.symmetry = None

    def init_storage(self, training_type, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape):
        self.storage = RolloutStorage(
            training_type, num_envs, num_transitions_per_env,
            actor_obs_shape, critic_obs_shape, actions_shape,
            None, self.device,
        )

    def act(self, obs, critic_obs):
        self.transition.actions = self.policy.act(obs).detach().clamp(-self.max_act, self.max_act)
        self.transition.values = self.policy.evaluate(critic_obs).detach()
        # Dummy tensors to satisfy RolloutStorage without breaking interface
        self.transition.actions_log_prob = torch.zeros(obs.shape[0], device=self.device)
        self.transition.action_mean = torch.zeros_like(self.transition.actions)
        self.transition.action_sigma = torch.zeros_like(self.transition.actions)

        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )
        self.storage.add_transitions(self.transition)
        self.transition.clear()

    def compute_returns(self, last_critic_obs):
        last_values = self.policy.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def update(self):
        batch_size = self.drift_batch_size

        # Initialize accumulators
        mean_value_loss = 0.0
        mean_actor_loss = 0.0
        mean_attract_mag = 0.0
        mean_repel_mag = 0.0
        mean_field_rms = 0.0
        total_batches = 0  # Track exact batch count for accurate averaging

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
                obs_batch, critic_obs_batch, actions_batch, target_values_batch,
                advantages_batch, returns_batch, *_
        ) in generator:

            original_batch_size = obs_batch.shape[0]
            n = original_batch_size // batch_size + int(original_batch_size % batch_size != 0)

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            self.optimizer.zero_grad()

            for i in range(n):
                slice_idx = slice(i * batch_size, min((i + 1) * batch_size, original_batch_size))

                obs_b = obs_batch[slice_idx]
                critic_obs_b = critic_obs_batch[slice_idx]
                actions_b = actions_batch[slice_idx]
                returns_b = returns_batch[slice_idx].unsqueeze(1) if returns_batch.dim() == 1 else returns_batch[slice_idx]
                adv_b = advantages_batch[slice_idx].unsqueeze(1)
                target_v_b = target_values_batch[slice_idx].unsqueeze(1) if target_values_batch.dim() == 1 else target_values_batch[slice_idx]

                # 1. Critic Loss
                value_b = self.policy.evaluate(critic_obs_b)
                if self.use_clipped_value_loss:
                    value_clipped = target_v_b + (value_b - target_v_b).clamp(-self.clip_param, self.clip_param)
                    value_losses = (value_b - returns_b).pow(2)
                    value_losses_clipped = (value_clipped - returns_b).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_b - value_b).pow(2).mean()

                # 2. Advantage-Weighted Drift Loss
                x_gen = self.policy.act_multiple(obs_b, n_gen=self.drift_ngen)  # [B, Ngen, D]
                y_target = actions_b.unsqueeze(1)  # [B, 1, D]

                # Unpack target and metrics
                target_a, drift_metrics = compute_drift_target(
                    x_gen=x_gen,
                    y_target=y_target,
                    advantages=adv_b,
                    tau=self.drift_tau,
                    eta=self.drift_eta,
                    max_act=self.max_act
                )

                actor_loss = torch.nn.functional.mse_loss(x_gen, target_a)

                # Backpropagation
                loss = actor_loss + self.value_loss_coef * value_loss
                loss *= (slice_idx.stop - slice_idx.start) / original_batch_size
                loss.backward()

                if self.is_multi_gpu:
                    self.reduce_parameters()

                # Accumulate metrics
                mean_value_loss += value_loss.item()
                mean_actor_loss += actor_loss.item()
                mean_attract_mag += drift_metrics["drift_attract_mag"]
                mean_repel_mag += drift_metrics["drift_repel_mag"]
                mean_field_rms += drift_metrics["drift_field_rms"]
                total_batches += 1

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

        # Average out by the actual number of sub-batches processed
        mean_value_loss /= total_batches
        mean_actor_loss /= total_batches
        mean_attract_mag /= total_batches
        mean_repel_mag /= total_batches
        mean_field_rms /= total_batches

        self.storage.clear()

        return {
            "value_function": mean_value_loss,
            "drift_actor_loss": mean_actor_loss, # Renamed slightly for clarity
            "drift_attract_mag": mean_attract_mag,
            "drift_repel_mag": mean_repel_mag,
            "drift_field_rms": mean_field_rms,
        }

    def reduce_parameters(self):
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        offset = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset: offset + numel].view_as(param.grad.data))
                offset += numel