# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from rsl_rl.modules import SACActorCritic
from rsl_rl.storage import ReplayBuffer


class SAC:
    """Soft Actor-Critic off-policy RL algorithm."""

    policy: SACActorCritic

    def __init__(
        self,
        policy: SACActorCritic,
        learning_rate: float = 3e-4,
        actor_learning_rate: float | None = None,   # defaults to learning_rate
        critic_learning_rate: float | None = None,  # defaults to learning_rate
        gamma: float = 0.99,
        tau: float = 0.005,
        ent_coef: float | str = "auto_0.006",
        target_entropy: float | str = "auto",
        gradient_steps: int = 1,
        batch_size: int = 256,
        policy_delay: int = 1,
        buffer_size: int = 1_000_000,
        learning_starts: int = 10_000,
        max_grad_norm: float = 1.0,
        device: str = "cpu",
        reward_scale: float = 1.0,
        # Unused (kept for config-dict compatibility with PPO)
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        **kwargs,
    ) -> None:
        if kwargs:
            print(
                "SAC.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs.keys()))
            )

        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.gradient_steps = gradient_steps
        self.batch_size = batch_size
        self.policy_delay = policy_delay
        self.max_grad_norm = max_grad_norm
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts

        self.policy = policy.to(device)

        actor_lr = actor_learning_rate if actor_learning_rate is not None else learning_rate
        critic_lr = critic_learning_rate if critic_learning_rate is not None else learning_rate

        self.actor_optimizer = optim.Adam(self.policy.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(
            list(self.policy.critic1.parameters()) + list(self.policy.critic2.parameters()),
            lr=critic_lr,
        )
        # Expose a single .optimizer for checkpoint compatibility with OnPolicyRunner
        self.optimizer = self.actor_optimizer
        self.learning_rate = learning_rate

        if isinstance(ent_coef, str) and ent_coef.startswith("auto"):
            self.auto_entropy_tuning = True
            parts = ent_coef.split("_")
            init_alpha = float(parts[1]) if len(parts) > 1 else 1.0
            self.log_ent_coef = torch.log(torch.tensor(init_alpha, dtype=torch.float32, device=device))
            self.log_ent_coef = nn.Parameter(self.log_ent_coef)
            self.ent_coef_optimizer = optim.Adam([self.log_ent_coef], lr=learning_rate)
            self.ent_coef: float = init_alpha
        else:
            self.auto_entropy_tuning = False
            self.ent_coef = float(ent_coef)

        if isinstance(target_entropy, str) and target_entropy == "auto":
            self.target_entropy: float = -float(self.policy.num_actions)
        else:
            self.target_entropy = float(target_entropy)

        self.storage: ReplayBuffer | None = None
        self._update_counter: int = 0   # counts critic updates (for policy_delay)
        self._current_action_tanh: torch.Tensor | None = None
        self._using_random_actions: bool = False

        self.reward_scale = reward_scale

        # No RND for SAC (set to None for runner compatibility)
        self.rnd = None

        self.min_ent_coef: float = kwargs.pop("min_ent_coef", 0.0)  # add param

    def init_storage(
        self,
        training_type: str,          # unused in SAC; kept for interface parity
        num_envs: int,
        num_transitions_per_env: int, # used to compute learning_starts if not overridden
        actor_obs_shape: list[int],
        critic_obs_shape: list[int],
        actions_shape: list[int],
    ) -> None:
        priv_shape = None if critic_obs_shape == actor_obs_shape else critic_obs_shape

        self.storage = ReplayBuffer(
            num_envs=num_envs,
            buffer_size=self.buffer_size,
            obs_shape=actor_obs_shape,
            privileged_obs_shape=priv_shape,
            actions_shape=actions_shape,
            device=self.device,
        )
        print(
            f"[SAC] Replay buffer: {self.buffer_size:,} transitions | "
            f"learning_starts={self.learning_starts:,} | "
            f"batch_size={self.batch_size} | "
            f"gradient_steps={self.gradient_steps} | "
            f"policy_delay={self.policy_delay} | "
            f"tau={self.tau} | "
            f"ent_coef={'auto' if self.auto_entropy_tuning else f'{self.ent_coef:.4f}'} | "
            f"target_entropy={self.target_entropy:.1f}"
        )

    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor) -> torch.Tensor:
        """Select action. Returns random actions during warm-up."""
        assert self.storage is not None, "Call init_storage() before act()."

        if self.storage.size < self.learning_starts:
            # Uniform random actions in tanh-space, then rescale
            self._using_random_actions = True
            self._current_action_tanh = torch.rand(
                obs.shape[0], self.policy.num_actions, device=self.device
            ) * 2.0 - 1.0
            action_env = self._current_action_tanh * self.policy.action_scale + self.policy.action_bias
        else:
            self._using_random_actions = False
            with torch.no_grad():
                action_env, _, self._current_action_tanh = self.policy.act(obs)

        return action_env

    def process_env_step(self, obs, privileged_obs, next_obs, next_privileged_obs, rewards, dones, infos):
        rewards = rewards * self.reward_scale

        actual_dones = dones.clone()
        if "time_outs" in infos:
            time_outs = infos["time_outs"].to(self.device)
            with torch.no_grad():
                # Bootstrap soft V(s_t) using current policy at the CURRENT state
                _, log_pi, action_pi = self.policy.act(obs)
                q1, q2 = self.policy.evaluate_critics(privileged_obs, action_pi, use_target=True)
                soft_v = (torch.min(q1, q2) - self.ent_coef * log_pi).squeeze(-1)
            rewards = rewards + self.gamma * soft_v * time_outs
            # done=1 so next_obs is masked out by (1 - done) in Bellman

        self.storage.add(
            obs=obs, next_obs=next_obs,
            action_tanh=self._current_action_tanh,
            reward=rewards, done=actual_dones,
            privileged_obs=privileged_obs if self.storage.privileged_observations is not None else None,
            next_privileged_obs=next_privileged_obs if self.storage.privileged_observations is not None else None,
        )
        self._current_action_tanh = None

    def compute_returns(self, last_critic_obs: torch.Tensor) -> None:
        """No-op — SAC does not use GAE returns.  Kept for interface parity."""
        pass

    def update(self) -> dict[str, float]:  # noqa: C901
        """Perform gradient_steps SAC updates from the replay buffer.

        Returns:
            loss_dict: Dictionary of mean losses for logging.
        """
        assert self.storage is not None
        assert self.storage.size >= self.batch_size

        mean_critic_loss: float = 0.0
        mean_actor_loss: float = 0.0
        mean_ent_coef_loss: float = 0.0
        mean_ent_coef: float = 0.0
        actor_updates: int = 0

        mean_q1_val: float = 0.0
        mean_target_q: float = 0.0
        mean_log_prob: float = 0.0
        mean_action_mag: float = 0.0

        ent_coef_t = torch.tensor(self.ent_coef, device=self.device)

        for _ in range(self.gradient_steps):
            (obs_batch, priv_obs_batch,
             next_obs_batch, next_priv_obs_batch,
             actions_batch, rewards_batch, dones_batch) = self.storage.sample(self.batch_size)

            with torch.no_grad():
                # Sample next actions from current policy
                _, next_log_prob, next_action_tanh = self.policy.act(next_obs_batch)

                # Min-clipped target Q
                q1_next, q2_next = self.policy.evaluate_critics(
                    next_priv_obs_batch, next_action_tanh, use_target=True
                )
                min_q_next = torch.min(q1_next, q2_next)

                target_q = rewards_batch + (1.0 - dones_batch) * self.gamma * (min_q_next - ent_coef_t * next_log_prob)

            q1, q2 = self.policy.evaluate_critics(priv_obs_batch, actions_batch, use_target=False)
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.policy.critic1.parameters()) + list(self.policy.critic2.parameters()),
                self.max_grad_norm,
            )
            self.critic_optimizer.step()

            mean_critic_loss += critic_loss.item()

            mean_q1_val += q1.mean().item()
            mean_target_q += target_q.mean().item()

            self._update_counter += 1

            if self._update_counter % self.policy_delay == 0:
                # Sample fresh actions from current policy (with gradients for actor)
                _, log_prob, action_tanh_pi = self.policy.evaluate_actor(obs_batch)

                # Actor loss: maximise Q − alpha.log pi  <>  minimise alpha.log pi − Q
                q1_pi, q2_pi = self.policy.evaluate_critics(
                    priv_obs_batch, action_tanh_pi, use_target=False
                )
                min_q_pi = torch.min(q1_pi, q2_pi)
                actor_loss = (ent_coef_t * log_prob - min_q_pi).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                mean_actor_loss += actor_loss.item()
                actor_updates += 1

                mean_log_prob += log_prob.mean().item()
                mean_action_mag += action_tanh_pi.abs().mean().item()  # Tracks saturation

                if self.auto_entropy_tuning:
                    # Loss: − alpha . (log pi + H*)  [detached log_prob]
                    ent_coef_loss = -(
                        self.log_ent_coef * (log_prob.detach() + self.target_entropy)
                    ).mean()
                    self.ent_coef_optimizer.zero_grad()
                    ent_coef_loss.backward()
                    self.ent_coef_optimizer.step()
                    # Update cached float value
                    self.ent_coef = self.log_ent_coef.exp().item()
                    if self.min_ent_coef > 0.0:
                        self.ent_coef = max(self.ent_coef, self.min_ent_coef)
                        # also clamp the parameter itself so the optimizer doesn't drift further
                        with torch.no_grad():
                            self.log_ent_coef.clamp_(min=math.log(self.min_ent_coef))
                    ent_coef_t = torch.tensor(self.ent_coef, device=self.device)
                    mean_ent_coef_loss += ent_coef_loss.item()
                self.policy.soft_update_target(self.tau)

            mean_ent_coef += self.ent_coef

        mean_critic_loss /= self.gradient_steps
        mean_ent_coef /= self.gradient_steps
        mean_q1_val /= self.gradient_steps
        mean_target_q /= self.gradient_steps

        if actor_updates > 0:
            mean_actor_loss /= actor_updates
            mean_ent_coef_loss /= actor_updates
            mean_log_prob /= actor_updates
            mean_action_mag /= actor_updates

        return {
            "critic": mean_critic_loss,
            "actor": mean_actor_loss,
            "ent_coef": mean_ent_coef,
            "ent_coef_loss": mean_ent_coef_loss,
            "Debug_Q_Mean": mean_q1_val,
            "Debug_Target_Q_Mean": mean_target_q,
            "Debug_Log_Prob": mean_log_prob,
            "Debug_Action_Magnitude": mean_action_mag,
        }