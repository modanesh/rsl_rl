# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
import torch
from collections import deque
from tqdm import trange

import rsl_rl
from rsl_rl.env import VecEnv
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.utils import store_code_state

from rsl_rl.modules import SACActorCritic
from rsl_rl.algorithms import SAC


class OffPolicyRunner:
    """Off-policy runner for training and evaluation with SAC."""

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        self.cfg = train_cfg
        self.alg_cfg = dict(train_cfg["algorithm"])
        self.policy_cfg = dict(train_cfg["policy"])
        self.device = device
        self.env = env

        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]

        if "critic" in extras.get("observations", {}):
            self.privileged_obs_type = "critic"
            num_privileged_obs = extras["observations"]["critic"].shape[1]
        else:
            self.privileged_obs_type = None
            num_privileged_obs = num_obs
        policy_class_name = self.policy_cfg.pop("class_name", "SACActorCritic")
        assert policy_class_name == "SACActorCritic", (
            f"OffPolicyRunner expects policy 'SACActorCritic', got '{policy_class_name}'."
        )

        self.policy = SACActorCritic(
            num_actor_obs=num_obs,
            num_critic_obs=num_privileged_obs,
            num_actions=self.env.num_actions,
            **self.policy_cfg,
        ).to(device)

        self._setup_action_bounds()

        alg_class_name = self.alg_cfg.pop("class_name", "SAC")
        assert alg_class_name == "SAC", (
            f"OffPolicyRunner expects algorithm 'SAC', got '{alg_class_name}'."
        )

        self.alg = SAC(policy=self.policy, device=device, **self.alg_cfg)

        self.num_steps_per_env: int = self.cfg["num_steps_per_env"]
        self.save_interval: int = self.cfg["save_interval"]
        self.empirical_normalization: bool = self.cfg["empirical_normalization"]

        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(device)
            self.privileged_obs_normalizer = EmpiricalNormalization(
                shape=[num_privileged_obs], until=1.0e8
            ).to(device)
        else:
            self.obs_normalizer = torch.nn.Identity().to(device)
            self.privileged_obs_normalizer = torch.nn.Identity().to(device)

        self.alg.init_storage(
            training_type="rl",
            num_envs=self.env.num_envs,
            num_transitions_per_env=self.num_steps_per_env,
            actor_obs_shape=[num_obs],
            critic_obs_shape=[num_privileged_obs],
            actions_shape=[self.env.num_actions],
        )
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps: int = 0
        self.tot_time: float = 0.0
        self.current_learning_iteration: int = 0
        self.git_status_repos = [rsl_rl.__file__]


    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if self.log_dir is not None and self.writer is None:
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()
            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg, suffix="sac")
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        # Randomise initial episode lengths
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Initial observations
        obs, extras = self.env.get_observations()
        privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
        obs = self.obs_normalizer(obs.to(self.device))
        privileged_obs = self.privileged_obs_normalizer(privileged_obs.to(self.device))

        self.train_mode()

        # Book-keeping
        ep_infos: list[dict] = []
        rewbuffer: deque = deque(maxlen=100)
        lenbuffer: deque = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        logger_type = self.cfg.get("logger", "tensorboard").lower()

        for it in trange(start_iter, tot_iter):
            start = time.time()

            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Select actions (random during warm-up, policy otherwise)
                    actions = self.alg.act(obs, privileged_obs)
                    if self.log_dir is not None:
                        if not hasattr(self, '_obs_log_buf'):
                            self._obs_log_buf = []
                        self._obs_log_buf.append(obs.detach().cpu())
                        if not hasattr(self, '_action_log_buf'):
                            self._action_log_buf = []
                        self._action_log_buf.append(actions.detach().cpu())

                    # Step environment
                    next_obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    next_obs = next_obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    # Normalise observations
                    next_obs_norm = self.obs_normalizer(next_obs)
                    if self.privileged_obs_type is not None:
                        next_privileged_obs_norm = self.privileged_obs_normalizer(
                            infos["observations"][self.privileged_obs_type].to(self.device)
                        )
                    else:
                        next_privileged_obs_norm = next_obs_norm

                    # Store transition (timeout-correction handled inside)
                    self.alg.process_env_step(
                        obs=obs,
                        privileged_obs=privileged_obs,
                        next_obs=next_obs_norm,
                        next_privileged_obs=next_privileged_obs_norm,
                        rewards=rewards,
                        dones=dones,
                        infos=infos,
                    )

                    # Logging uses RAW rewards so curves stay comparable to PPO
                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                    # Advance observation
                    obs = next_obs_norm
                    privileged_obs = next_privileged_obs_norm

            collection_time = time.time() - start
            start = time.time()

            loss_dict: dict[str, float] = {}
            if self.alg.storage.size >= self.alg.learning_starts:
                # Freeze normalizer stats exactly once when training begins
                if self.empirical_normalization and not getattr(self, '_normalizer_frozen', False):
                    self.obs_normalizer.eval()
                    self.privileged_obs_normalizer.eval()
                    self._normalizer_frozen = True
                    print("[INFO] Obs normalizer frozen for SAC training.")
                loss_dict = self.alg.update()

            learn_time = time.time() - start
            self.current_learning_iteration = it
            if self.log_dir is not None:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()

            # Save code state on first iteration
            if it == start_iter:
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Final checkpoint
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 38) -> None:
        """Log metrics to writer and print a summary string."""
        collection_size = self.num_steps_per_env * self.env.num_envs
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # Episode info
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"{f'{key}:':>{pad}} {value:.4f}\n"
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"{f'Mean episode {key}:':>{pad}} {value:.4f}\n"

        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"] + 1e-9))

        # Losses
        loss_dict = locs.get("loss_dict", {})
        for key, value in loss_dict.items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])

        # Buffer stats
        self.writer.add_scalar("SAC/buffer_size", self.alg.storage.size, locs["it"])
        self.writer.add_scalar("SAC/ent_coef", self.alg.ent_coef, locs["it"])
        self.writer.add_scalar("SAC/learning_rate", self.alg.learning_rate, locs["it"])

        # -- Per-dim action stats (SAC)
        if hasattr(self, '_action_log_buf') and self._action_log_buf:
            stacked = torch.cat(self._action_log_buf, dim=0)
            for i in range(stacked.shape[1]):
                self.writer.add_scalar(f"Actions/dim_{i:02d}_mean", stacked[:, i].mean().item(), locs["it"])
                self.writer.add_scalar(f"Actions/dim_{i:02d}_std", stacked[:, i].std().item(), locs["it"])
            self._action_log_buf.clear()

        # -- Obs stats
        if hasattr(self, '_obs_log_buf') and self._obs_log_buf:
            stacked = torch.cat(self._obs_log_buf, dim=0)
            self.writer.add_scalar("Obs/mean", stacked.mean().item(), locs["it"])
            self.writer.add_scalar("Obs/std", stacked.std().item(), locs["it"])
            self.writer.add_scalar("Obs/abs_max", stacked.abs().max().item(), locs["it"])
            for i in range(stacked.shape[1]):
                self.writer.add_scalar(f"ObsDim/dim_{i:03d}_mean", stacked[:, i].mean().item(), locs["it"])
                self.writer.add_scalar(f"ObsDim/dim_{i:03d}_std", stacked[:, i].std().item(), locs["it"])
            self._obs_log_buf.clear()

        # Performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection_time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # Training rewards
        logger_type = self.cfg.get("logger", "tensorboard").lower()
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if logger_type != "wandb":
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        # Console summary
        warm_up = self.alg.storage.size < self.alg.learning_starts
        str_header = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "
        log_string = (
            f"{'#' * width}\n"
            f"{str_header.center(width, ' ')}\n\n"
            f"{'Computation:':>{pad}} {fps:.0f} steps/s "
            f"(collection: {locs['collection_time']:.3f}s, learning: {locs['learn_time']:.3f}s)\n"
            f"{'Buffer size:':>{pad}} {self.alg.storage.size:,} / {self.alg.storage.max_size:,}"
            f"{' [WARM-UP]' if warm_up else ''}\n"
            f"{'Entropy coefficient:':>{pad}} {self.alg.ent_coef:.5f}\n"
        )
        for key, value in loss_dict.items():
            log_string += f"{f'Mean {key} loss:':>{pad}} {value:.4f}\n"

        if len(locs["rewbuffer"]) > 0:
            log_string += f"{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"
            log_string += f"{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"

        log_string += ep_string
        log_string += (
            f"{'-' * width}\n"
            f"{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"
            f"{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"
            f"{'Time elapsed:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(self.tot_time))}\n"
            f"{'ETA:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(self.tot_time / (locs['it'] - locs['start_iter'] + 1) * (locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])))}\n"
        )
        # print(log_string)  # uncomment to enable console output

    def save(self, path: str, infos=None) -> None:
        """Save policy, optimizers and normalizers to a checkpoint file."""
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "actor_optimizer_state_dict": self.alg.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.alg.critic_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.alg.auto_entropy_tuning:
            saved_dict["log_ent_coef"] = self.alg.log_ent_coef.data
            saved_dict["ent_coef_optimizer_state_dict"] = self.alg.ent_coef_optimizer.state_dict()
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["privileged_obs_norm_state_dict"] = self.privileged_obs_normalizer.state_dict()

        torch.save(saved_dict, path)

        logger_type = self.cfg.get("logger", "tensorboard").lower()
        if logger_type in ["neptune", "wandb"] and self.writer is not None:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True) -> dict | None:
        """Load a checkpoint.

        Args:
            path:           Path to the .pt file.
            load_optimizer: Whether to restore optimizer states (set False for fine-tuning).

        Returns:
            infos dict stored in the checkpoint, or None.
        """
        loaded_dict = torch.load(path, weights_only=False)
        resumed = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])

        if self.empirical_normalization and "obs_norm_state_dict" in loaded_dict:
            if resumed:
                self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])
            else:
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])

        if load_optimizer and resumed:
            self.alg.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
            self.alg.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
            if self.alg.auto_entropy_tuning and "log_ent_coef" in loaded_dict:
                self.alg.log_ent_coef.data = loaded_dict["log_ent_coef"]
                self.alg.ent_coef_optimizer.load_state_dict(loaded_dict["ent_coef_optimizer_state_dict"])
                self.alg.ent_coef = self.alg.log_ent_coef.exp().item()

        if resumed and "iter" in loaded_dict:
            self.current_learning_iteration = loaded_dict["iter"]

        return loaded_dict.get("infos")

    def get_inference_policy(self, device=None):
        """Return a callable that maps observations to deterministic actions."""
        self.eval_mode()
        if device is not None:
            self.alg.policy.to(device)
        if self.cfg.get("empirical_normalization", False):
            if device is not None:
                self.obs_normalizer.to(device)
            return lambda x: self.alg.policy.act_inference(self.obs_normalizer(x))  # noqa: E731
        return self.alg.policy.act_inference

    def train_mode(self) -> None:
        self.alg.policy.train()
        if self.empirical_normalization and not getattr(self, '_normalizer_frozen', False):
            self.obs_normalizer.train()
            self.privileged_obs_normalizer.train()

    def eval_mode(self) -> None:
        self.alg.policy.eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.privileged_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        self.git_status_repos.append(repo_file_path)

    def _setup_action_bounds(self) -> None:
        """Read action bounds from env.action_space and pass to policy.

        Isaac Lab defaults are [-100, 100] — this will break SAC.
        Override via train_sac.py --action_bound or by calling
        runner.policy.set_action_bounds(low, high) directly.
        """
        action_space = self.env.action_space
        low = torch.tensor(action_space.low, dtype=torch.float32, device=self.device)
        high = torch.tensor(action_space.high, dtype=torch.float32, device=self.device)
        self.policy.set_action_bounds(low, high)
        print(
            f"[OffPolicyRunner] Action bounds set: "
            f"low=[{low.min().item():.2f}, …, {low.max().item():.2f}]  "
            f"high=[{high.min().item():.2f}, …, {high.max().item():.2f}]"
        )