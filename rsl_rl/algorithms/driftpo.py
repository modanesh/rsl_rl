# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rsl_rl.modules import ActorCriticDrift
from rsl_rl.storage import RolloutStorage


# ══════════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _kde_repulsion(x: torch.Tensor, h: float | None = None) -> torch.Tensor:
    B, Ngen, D = x.shape

    # diff[b,i,j] = a_i − a_j
    diff    = x.unsqueeze(2) - x.unsqueeze(1)    # [B, Ngen, Ngen, D]
    sq_dist = diff.pow(2).sum(dim=-1)             # [B, Ngen, Ngen]

    eye = torch.eye(Ngen, dtype=torch.bool, device=x.device).unsqueeze(0)

    if h is None:
        # Median heuristic: h² ≈ mean_pairwise_sq_dist / (2 ln Ngen)
        with torch.no_grad():
            n_pairs  = Ngen * (Ngen - 1)
            mean_sq  = sq_dist.masked_fill(eye, 0.0).sum(dim=[-2, -1]).mean() / n_pairs
            h_sq_val = (mean_sq / (2.0 * math.log(Ngen + 1))).clamp(min=1e-4)
    else:
        h_sq_val = x.new_tensor(h * h)

    # Gaussian kernel weights
    log_K = -sq_dist / (2.0 * h_sq_val)
    log_K = log_K.masked_fill(eye, float("-inf"))
    W     = torch.softmax(log_K, dim=-1)          # [B, Ngen, Ngen]

    # Σ_j w_ij (a_i − a_j)
    repulsion = (W.unsqueeze(-1) * diff).sum(dim=2)   # [B, Ngen, D]
    return repulsion


def compute_drift_vector_field(
    q_network:   nn.Module,          # should be q_target in normal use  [Fix 9]
    critic_obs:  torch.Tensor,       # [B, c_obs_dim]
    x_gen:       torch.Tensor,       # [B, Ngen, D]
    alpha:       float       = 1.0,
    h:           float | None = None,
    repel_coef:  float       = 0.1,
) -> tuple[torch.Tensor, dict]:
    B, Ngen, D = x_gen.shape

    a_var   = x_gen.detach().reshape(B * Ngen, D).requires_grad_(True)
    obs_rep = (critic_obs.detach()
                         .unsqueeze(1)
                         .expand(-1, Ngen, -1)
                         .reshape(B * Ngen, -1))

    q_vals = q_network(torch.cat([obs_rep, a_var], dim=-1))   # [B·Ngen, 1]

    q_grad = torch.autograd.grad(
        outputs      = q_vals.sum(),
        inputs       = a_var,
        create_graph = False,
        retain_graph = False,
    )[0].reshape(B, Ngen, D)

    V_attract_raw = q_grad / alpha

    V_repel_raw = _kde_repulsion(x_gen.detach(), h=h)

    V_raw = V_attract_raw + repel_coef * V_repel_raw         # [B, Ngen, D]

    rms_combined = V_raw.pow(2).mean(dim=[-2, -1], keepdim=True).sqrt()  # [B, 1, 1]
    V = V_raw / (rms_combined + 1e-8)

    # Diagnostic metrics  (all pre-normalisation; drift_field_rms now informative)
    with torch.no_grad():
        attract_mag = V_attract_raw.norm(dim=-1).mean().item()
        repel_mag   = V_repel_raw.norm(dim=-1).mean().item()
        ratio       = repel_mag / (attract_mag + 1e-8)

    metrics = {
        "drift_attract_mag":         attract_mag,
        "drift_repel_mag":           repel_mag,
        "drift_repel_attract_ratio": ratio,
        "drift_q_mean":              q_vals.detach().mean().item(),
        "drift_field_rms":           rms_combined.mean().item(),
    }

    return V.detach(), metrics


# ══════════════════════════════════════════════════════════════════════════════
#  DriftPO algorithm
# ══════════════════════════════════════════════════════════════════════════════

class DriftPO:
    policy: ActorCriticDrift

    def __init__(
        self,
        policy:                             ActorCriticDrift,
        num_learning_epochs:                int   = 1,
        num_mini_batches:                   int   = 1,
        clip_param:                         float = 0.2,
        gamma:                              float = 0.998,
        lam:                                float = 0.95,
        value_loss_coef:                    float = 1.0,
        learning_rate:                      float = 1e-3,
        max_grad_norm:                      float = 1.0,
        use_clipped_value_loss:             bool  = True,
        device:                             str   = "cpu",
        normalize_advantage_per_mini_batch: bool  = False,
        multi_gpu_cfg:                      dict | None = None,
        alpha:              float       = 1.0,
        drift_eta:          float       = 0.1,         # was 0.5  [Fix 11]
        drift_h:            float | None = None,        # None = median heuristic
        drift_ngen:         int         = 16,
        drift_batch_size:   int         = 256,
        drift_repel_coef:   float       = 0.1,
        num_q_updates:      int         = 4,
        q_lr:               float | None = None,
        max_act:            float       = 3.0,
        q_actor_coef:       float       = 1.0,
        q_target_tau:       float       = 0.005,
        q_gen_aug:          int         = 4,
        q_gen_aug_coef:     float       = 0.5,
        **kwargs,
    ):
        self.device = device

        self.is_multi_gpu    = multi_gpu_cfg is not None
        self.gpu_global_rank = multi_gpu_cfg["global_rank"] if multi_gpu_cfg else 0
        self.gpu_world_size  = multi_gpu_cfg["world_size"]  if multi_gpu_cfg else 1

        self.policy = policy
        self.policy.to(self.device)

        self.optimizer   = optim.Adam(policy.actor_critic_parameters(), lr=learning_rate)
        self.q_optimizer = optim.Adam(policy.q_parameters(),            lr=q_lr or learning_rate)

        self.storage: RolloutStorage = None
        self.transition = RolloutStorage.Transition()

        self.clip_param              = clip_param
        self.num_learning_epochs     = num_learning_epochs
        self.num_mini_batches        = num_mini_batches
        self.value_loss_coef         = value_loss_coef
        self.gamma                   = gamma
        self.lam                     = lam
        self.max_grad_norm           = max_grad_norm
        self.use_clipped_value_loss  = use_clipped_value_loss
        self.learning_rate           = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        self.alpha             = alpha
        self.drift_eta         = drift_eta
        self.drift_h           = drift_h
        self.drift_ngen        = drift_ngen
        self.drift_batch_size  = drift_batch_size
        self.drift_repel_coef  = drift_repel_coef
        self.num_q_updates     = num_q_updates
        self.max_act           = max_act

        # v2
        self.q_actor_coef    = q_actor_coef
        self.q_target_tau    = q_target_tau
        self.q_gen_aug       = q_gen_aug
        self.q_gen_aug_coef  = q_gen_aug_coef

        self.rnd      = None
        self.symmetry = None

    def init_storage(
        self, training_type, num_envs, num_transitions_per_env,
        actor_obs_shape, critic_obs_shape, actions_shape,
    ):
        self.storage = RolloutStorage(
            training_type, num_envs, num_transitions_per_env,
            actor_obs_shape, critic_obs_shape, actions_shape,
            None, self.device,
        )

    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor) -> torch.Tensor:
        self.transition.actions  = self.policy.act(obs).detach().clamp(-self.max_act, self.max_act)
        self.transition.values   = self.policy.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = torch.zeros(obs.shape[0], device=self.device)
        self.transition.action_mean      = torch.zeros_like(self.transition.actions)
        self.transition.action_sigma     = torch.zeros_like(self.transition.actions)
        self.transition.observations            = obs
        self.transition.privileged_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards: torch.Tensor, dones: torch.Tensor, infos: dict):
        self.transition.rewards = rewards.clone()
        self.transition.dones   = dones
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )
        self.storage.add_transitions(self.transition)
        self.transition.clear()

    def compute_returns(self, last_critic_obs: torch.Tensor):
        last_values = self.policy.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam,
            normalize_advantage=not self.normalize_advantage_per_mini_batch,
        )

    def update(self) -> dict:
        batch_size = self.drift_batch_size
        sum_q_loss = 0.0
        n_q        = 0

        for _ in range(self.num_q_updates):
            for (
                obs_batch_q,        # kept (was _obs) — needed for act_multiple
                critic_obs_batch,
                actions_batch,
                _tv, _adv, returns_batch, *_
            ) in self.storage.mini_batch_generator(self.num_mini_batches, 1):

                B_full = critic_obs_batch.shape[0]
                n_sub  = B_full // batch_size + int(B_full % batch_size != 0)
                self.q_optimizer.zero_grad()

                for i in range(n_sub):
                    sl     = slice(i * batch_size, min((i + 1) * batch_size, B_full))
                    weight = (sl.stop - sl.start) / B_full

                    obs_b        = obs_batch_q[sl]
                    critic_obs_b = critic_obs_batch[sl]
                    returns_b    = (returns_batch[sl].unsqueeze(1)
                                   if returns_batch.dim() == 1
                                   else returns_batch[sl])

                    q_pred = self.policy.q_value(critic_obs_b, actions_batch[sl])
                    q_rollout_loss = F.mse_loss(q_pred, returns_b)
                    with torch.no_grad():
                        n_aug     = self.q_gen_aug
                        B_sub     = obs_b.shape[0]
                        x_gen_aug = self.policy.act_multiple(obs_b, n_gen=n_aug)  # [B, n_aug, D]
                        a_aug     = x_gen_aug.reshape(B_sub * n_aug, self.policy.a_dim)
                        obs_aug   = (critic_obs_b.unsqueeze(1)
                                                 .expand(-1, n_aug, -1)
                                                 .reshape(B_sub * n_aug, -1))
                        # V(s) repeated for each generated action
                        v_aug_target = (self.policy.evaluate(critic_obs_b)  # [B, 1]
                                                   .expand(-1, n_aug)
                                                   .reshape(B_sub * n_aug, 1))

                    q_gen     = self.policy.q_value(obs_aug, a_aug)
                    q_gen_loss = F.mse_loss(q_gen, v_aug_target.detach())

                    # Combined Q loss
                    q_loss = q_rollout_loss + self.q_gen_aug_coef * q_gen_loss
                    (q_loss * weight).backward()

                    sum_q_loss += q_rollout_loss.item()   # log rollout loss only (comparable to v1)
                    n_q        += 1

                nn.utils.clip_grad_norm_(self.policy.q_parameters(), self.max_grad_norm)
                self.q_optimizer.step()

                tau = self.q_target_tau
                with torch.no_grad():
                    for p, tp in zip(self.policy.q_network.parameters(),
                                     self.policy.q_target.parameters()):
                        tp.data.lerp_(p.data, tau)

        sum_v_loss        = 0.0
        sum_actor_loss    = 0.0
        sum_q_actor_loss  = 0.0
        sum_attract_mag   = 0.0
        sum_repel_mag     = 0.0
        sum_ratio         = 0.0
        sum_q_mean        = 0.0
        sum_field_rms     = 0.0
        n_actor           = 0

        for (
            obs_batch, critic_obs_batch, _actions,
            target_values_batch, advantages_batch, returns_batch, *_
        ) in self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):

            B_full = obs_batch.shape[0]
            n_sub  = B_full // batch_size + int(B_full % batch_size != 0)

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (
                        (advantages_batch - advantages_batch.mean())
                        / (advantages_batch.std() + 1e-8)
                    )

            self.optimizer.zero_grad()

            for i in range(n_sub):
                sl     = slice(i * batch_size, min((i + 1) * batch_size, B_full))
                weight = (sl.stop - sl.start) / B_full

                obs_b        = obs_batch[sl]
                critic_obs_b = critic_obs_batch[sl]
                returns_b    = (returns_batch[sl].unsqueeze(1)
                                if returns_batch.dim() == 1
                                else returns_batch[sl])
                target_v_b   = (target_values_batch[sl].unsqueeze(1)
                                if target_values_batch.dim() == 1
                                else target_values_batch[sl])

                v_pred = self.policy.evaluate(critic_obs_b)
                if self.use_clipped_value_loss:
                    v_clipped = target_v_b + (v_pred - target_v_b).clamp(
                        -self.clip_param, self.clip_param
                    )
                    v_loss = torch.max(
                        (v_pred    - returns_b).pow(2),
                        (v_clipped - returns_b).pow(2),
                    ).mean()
                else:
                    v_loss = F.mse_loss(v_pred, returns_b)

                x_gen = self.policy.act_multiple(obs_b, n_gen=self.drift_ngen)

                V_field, drift_metrics = compute_drift_vector_field(
                    q_network  = self.policy.q_target,   # frozen target  [Fix 9]
                    critic_obs = critic_obs_b,
                    x_gen      = x_gen,
                    alpha      = self.alpha,
                    h          = self.drift_h,
                    repel_coef = self.drift_repel_coef,
                )

                target_a    = (x_gen.detach() + self.drift_eta * V_field).clamp(
                    -self.max_act, self.max_act
                )
                drift_loss  = F.mse_loss(x_gen, target_a)

                z_q  = torch.randn(obs_b.shape[0], self.policy.a_dim, device=self.device)
                a_q  = self.policy.actor(torch.cat([obs_b, z_q], dim=-1))   # differentiable
                q_actor_vals = self.policy.q_target(
                    torch.cat([critic_obs_b.detach(), a_q], dim=-1)
                )   # grad flows through a_q only; q_target params are frozen

                # Scale-invariant normalisation (offline DriftQL convention)
                lam_q        = 1.0 / (q_actor_vals.abs().mean().detach() + 1e-6)
                q_actor_loss = -lam_q * q_actor_vals.mean()

                actor_loss = drift_loss + self.q_actor_coef * q_actor_loss

                ((actor_loss + self.value_loss_coef * v_loss) * weight).backward()

                sum_v_loss       += v_loss.item()
                sum_actor_loss   += drift_loss.item()
                sum_q_actor_loss += q_actor_loss.item()
                sum_attract_mag  += drift_metrics["drift_attract_mag"]
                sum_repel_mag    += drift_metrics["drift_repel_mag"]
                sum_ratio        += drift_metrics["drift_repel_attract_ratio"]
                sum_q_mean       += drift_metrics["drift_q_mean"]
                sum_field_rms    += drift_metrics["drift_field_rms"]
                n_actor          += 1

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.policy.actor_critic_parameters(), self.max_grad_norm)
            self.optimizer.step()

        self.storage.clear()

        return {
            "q_loss":                    sum_q_loss        / max(n_q,     1),
            "value_function":            sum_v_loss        / max(n_actor, 1),
            "drift_actor_loss":          sum_actor_loss    / max(n_actor, 1),
            "drift_q_actor_loss":        sum_q_actor_loss  / max(n_actor, 1),
            "drift_attract_mag":         sum_attract_mag   / max(n_actor, 1),
            "drift_repel_mag":           sum_repel_mag     / max(n_actor, 1),
            "drift_repel_attract_ratio": sum_ratio         / max(n_actor, 1),
            "drift_q_mean":              sum_q_mean        / max(n_actor, 1),
            "drift_field_rms":           sum_field_rms     / max(n_actor, 1),
        }

    def broadcast_parameters(self):
        for param in self.policy.parameters():
            torch.distributed.broadcast(param.data, src=0)

    def reduce_parameters(self):
        grads = [p.grad.view(-1) for p in self.policy.parameters() if p.grad is not None]
        if not grads:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                n = param.numel()
                param.grad.data.copy_(all_grads[offset: offset + n].view_as(param.grad.data))
                offset += n