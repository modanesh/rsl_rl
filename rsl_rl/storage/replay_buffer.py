# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay buffer for SAC / off-policy training.

Design notes for massive parallel simulation (e.g. Isaac Sim with 4096 envs):
  - All environments contribute transitions simultaneously each step, so the
    effective insertion rate is num_envs transitions per env-step.
  - The buffer stores transitions as flat tensors on the training device to avoid
    repeated CPU↔GPU transfers.
  - Observations (actor) and privileged observations (critic) are stored separately,
    mirroring the asymmetric actor-critic setup used in rsl-rl.
  - Actions are stored in tanh-squashed space [-1, 1] (what the critic sees).
    Rescaling to env bounds is handled in SACActorCritic / OffPolicyRunner.
  - Rewards and dones after timeout-correction are stored (timeouts are handled
    by the runner before calling add(), consistent with the PPO bootstrapping
    approach in rsl-rl).
"""

from __future__ import annotations

import torch


class ReplayBuffer:
    """Circular replay buffer for off-policy RL with vectorised environments.

    Args:
        num_envs:               Number of parallel environments.
        buffer_size:            Maximum total number of transitions stored.
                                When full, oldest transitions are overwritten.
        obs_shape:              Shape of actor observations, e.g. (48,).
        privileged_obs_shape:   Shape of critic observations, or None to fall back to obs_shape.
        actions_shape:          Shape of the action vector, e.g. (12,).
        device:                 Torch device for all tensors.
    """

    def __init__(
        self,
        num_envs: int,
        buffer_size: int,
        obs_shape: list[int] | tuple[int, ...],
        privileged_obs_shape: list[int] | tuple[int, ...] | None,
        actions_shape: list[int] | tuple[int, ...],
        device: str = "cpu",
    ) -> None:
        self.num_envs = num_envs
        self.max_size = buffer_size
        self.device = device

        # Write pointer and current fill level
        self._ptr: int = 0
        self._size: int = 0

        # ------------------------------------------------------------------ #
        # Allocate tensors
        # ------------------------------------------------------------------ #
        self.observations = torch.zeros(buffer_size, *obs_shape, device=device)
        self.next_observations = torch.zeros(buffer_size, *obs_shape, device=device)

        if privileged_obs_shape is not None:
            self.privileged_observations = torch.zeros(buffer_size, *privileged_obs_shape, device=device)
            self.next_privileged_observations = torch.zeros(buffer_size, *privileged_obs_shape, device=device)
        else:
            # Share memory with actor observations (no privileged info)
            self.privileged_observations = None
            self.next_privileged_observations = None

        # Actions stored in tanh-squashed space [-1, 1]
        self.actions = torch.zeros(buffer_size, *actions_shape, device=device)
        self.rewards = torch.zeros(buffer_size, 1, device=device)
        # 1.0 for terminal transitions; 0.0 for non-terminal / timeout-corrected
        self.dones = torch.zeros(buffer_size, 1, device=device)

    # ---------------------------------------------------------------------- #
    # Writing
    # ---------------------------------------------------------------------- #

    def add(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action_tanh: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        privileged_obs: torch.Tensor | None = None,
        next_privileged_obs: torch.Tensor | None = None,
    ) -> None:
        """Insert a batch of transitions (one per parallel environment).

        Args:
            obs:                  Current observations,          shape (num_envs, *obs_shape).
            next_obs:             Next observations,             shape (num_envs, *obs_shape).
            action_tanh:          Tanh-squashed actions in [-1,1], shape (num_envs, *actions_shape).
            reward:               Scalar rewards,               shape (num_envs,) or (num_envs, 1).
            done:                 Terminal flags (bool/float), shape (num_envs,) or (num_envs, 1).
                                  Should already be timeout-corrected by the caller so that
                                  episodes ending due to time-limit have done=0.
            privileged_obs:       Critic observations,           shape (num_envs, *priv_shape) or None.
            next_privileged_obs:  Next critic observations,      shape (num_envs, *priv_shape) or None.
        """
        batch = obs.shape[0]
        assert batch == self.num_envs, (
            f"Expected batch size {self.num_envs} but got {batch}."
        )

        # Compute destination indices (wraps around at max_size)
        end = self._ptr + batch
        if end <= self.max_size:
            idx = slice(self._ptr, end)
            self.observations[idx] = obs
            self.next_observations[idx] = next_obs
            self.actions[idx] = action_tanh
            self.rewards[idx] = reward.view(-1, 1)
            self.dones[idx] = done.float().view(-1, 1)
            if self.privileged_observations is not None:
                self.privileged_observations[idx] = privileged_obs
                self.next_privileged_observations[idx] = next_privileged_obs
        else:
            # Wraparound: write in two chunks
            first = self.max_size - self._ptr
            # -- chunk 1
            idx1 = slice(self._ptr, self.max_size)
            self.observations[idx1] = obs[:first]
            self.next_observations[idx1] = next_obs[:first]
            self.actions[idx1] = action_tanh[:first]
            self.rewards[idx1] = reward[:first].view(-1, 1)
            self.dones[idx1] = done[:first].float().view(-1, 1)
            if self.privileged_observations is not None:
                self.privileged_observations[idx1] = privileged_obs[:first]
                self.next_privileged_observations[idx1] = next_privileged_obs[:first]
            # -- chunk 2
            rest = batch - first
            idx2 = slice(0, rest)
            self.observations[idx2] = obs[first:]
            self.next_observations[idx2] = next_obs[first:]
            self.actions[idx2] = action_tanh[first:]
            self.rewards[idx2] = reward[first:].view(-1, 1)
            self.dones[idx2] = done[first:].float().view(-1, 1)
            if self.privileged_observations is not None:
                self.privileged_observations[idx2] = privileged_obs[first:]
                self.next_privileged_observations[idx2] = next_privileged_obs[first:]

        self._ptr = end % self.max_size
        self._size = min(self._size + batch, self.max_size)

    # ---------------------------------------------------------------------- #
    # Reading
    # ---------------------------------------------------------------------- #

    def sample(
        self, batch_size: int
    ) -> tuple[
        torch.Tensor,   # obs
        torch.Tensor,   # privileged_obs
        torch.Tensor,   # next_obs
        torch.Tensor,   # next_privileged_obs
        torch.Tensor,   # actions (tanh-squashed)
        torch.Tensor,   # rewards
        torch.Tensor,   # dones
    ]:
        """Uniformly sample a mini-batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            7-tuple of tensors, all on self.device.
            If no privileged observations were stored, privileged_obs == obs
            and next_privileged_obs == next_obs.
        """
        assert self._size >= batch_size, (
            f"Replay buffer only has {self._size} transitions, "
            f"but batch_size={batch_size} was requested."
        )
        idx = torch.randint(0, self._size, (batch_size,), device=self.device)

        obs = self.observations[idx]
        next_obs = self.next_observations[idx]
        actions = self.actions[idx]
        rewards = self.rewards[idx]
        dones = self.dones[idx]

        if self.privileged_observations is not None:
            privileged_obs = self.privileged_observations[idx]
            next_privileged_obs = self.next_privileged_observations[idx]
        else:
            privileged_obs = obs
            next_privileged_obs = next_obs

        return obs, privileged_obs, next_obs, next_privileged_obs, actions, rewards, dones

    # ---------------------------------------------------------------------- #
    # Properties
    # ---------------------------------------------------------------------- #

    @property
    def size(self) -> int:
        """Current number of transitions stored."""
        return self._size

    def __len__(self) -> int:
        return self._size