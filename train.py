"""CS100709: A2C baseline and autonomous PPO campaign for BipedalWalker HARDCORE.

Primary source
    CS100709_Reinforcement_Learning_Project.pdf, credited to Hongtu Zhou,
    Di Zhang and Hai Zhang (TiEV lab). Requirements: pp. 1-6; A2C theory:
    pp. 20-22; continuous-action example: pp. 31-40.
    Adaptations: four joint actions, detached TD targets, independent evaluation,
    Windows spawn workers, terminal-observation handling, portable persistence.
    Default behavior implements the handout's one-step variant, without reward shaping,
    observation normalization, replay memory, PPO, or an entropy bonus.
    Issue #1 adds optional bootstrapped n-step returns and gradient clipping:
    --return-steps 8 --max-grad-norm 0.5. These are experimental extensions,
    not proven improvements or literal copies of the handout's default update.
    Evaluation records finish/fall/time-limit, distance and original reward terms;
    diagnostic coordinates are never added to the policy observation or reward.
    For the user's subsequent 300-point objective, --algorithm ppo uses the cited
    Stable-Baselines3 2.4.1 implementation, GAE, clipped updates and internal
    observation normalization. Original rewards and the 1600-step task stay fixed.
    PPO requires: python -m pip install stable-baselines3==2.4.1
    Reuse existing dependencies; do not reinstall PyTorch. The exported Agent
    stores normalization statistics and CPU weights and needs no SB3 for inference.
    Issue #4 starts from the interrupted run's pinned best complete checkpoint.
    Compare ordinary Gaussian PPO with clipped action policy gradient (CAPG),
    using executed-action probabilities at saturation bounds. Both use GAE .98,
    no entropy bonus, frozen observation statistics and bounded terrain replay.
    Two training seeds per method precede budget allocation to the better method;
    three unproductive continuation chunks stop the experiment. These changes
    are hypotheses, not measured gains. The final holdout remains isolated.
    CAPG: https://proceedings.mlr.press/v80/fujita18a.html
    Original rewards and the 1600-step task are preserved; see README.md.

Windows setup (run in a Miniconda Prompt)
    conda create -n cs100709 -c conda-forge python=3.11 box2d-py=2.3.8 pip -y
    conda activate cs100709
    python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
    python -m pip install numpy==1.26.4 gymnasium==0.29.1 pygame==2.6.1 tensorboard==2.20.0 matplotlib ipython
    Install NumPy with pip in this fresh environment to avoid mixing Conda's BLAS
    runtime with the one used by the PyTorch wheel. Box2D is installed by conda;
    do not additionally install gymnasium[box2d],
    which requests a different Box2D distribution and can require a C++ compiler.
    The CUDA wheel can also run on CPU. A CPU-only alternative uses the index
    https://download.pytorch.org/whl/cpu instead. No torchvision is required.

Commands (keep this file named train.py)
    python train.py --mode check
    python train.py --mode benchmark --device cpu --num-envs 8
    python train.py --mode benchmark --device cuda --num-envs 8
    python train.py --mode train --device cpu --num-envs 8 --steps 1000000
    python train.py --mode train --resume runs/RUN/checkpoint.pt --steps 2000000
    python train.py --mode eval --model-path a2c.model --eval-seeds 1 3 5 7 9 --render
    python train.py --mode train --return-steps 8 --max-grad-norm 0.5 --log-dir runs/nstep --model-path runs/nstep/a2c.model
    python train.py --mode check --algorithm ppo
    python train.py --mode campaign --dry-run
    python train.py --mode campaign --campaign-dir runs/campaign-issue4-v5
    python train.py --mode train --algorithm ppo --steps 20000000 --log-dir runs/ppo --model-path runs/ppo/a2c.model
    tensorboard --logdir runs

Modes and outputs
    Train and campaign update weights on Windows. Check, eval, dry-run and
    benchmark never update weights. The CLI rejects training on the user's Mac.
    Campaign defaults: CPU, 16 environments, 1 thread, minibatches of 256;
    two 5M-step pilots, two independent-seed replications, and 5M continuations
    within a total 60M new-interaction budget. Stage budgets round down to complete rollouts. Reissue
    the identical campaign command to resume; no PyTorch reinstall is required.
    Benchmark times environment sampling plus forward/backward computation; it
    excludes optimizer steps, logging, evaluation and saving, so it is a throughput
    comparison rather than an estimate of total training time.
    Defaults target an i9-14900HX: CPU, 8 environments, 1 PyTorch thread, 64 samples
    per update. Steps and batch size must be multiples of num-envs. --steps is the
    cumulative budget, including steps already completed in a resumed run.
    Each run has an independent directory containing TensorBoard events,
    config.json, evaluations.jsonl, checkpoint.pt and its best a2c.model.
    --model-path is also updated with the best validation model (default a2c.model).
    Resume restores optimizer/RNG/counters, then starts new environment episodes;
    Box2D world state is not restored, so continuation is not bit-for-bit identical.
    Ctrl+C requests a stop at an update boundary, then evaluation and saving.
    One million steps is the handout's initial budget, not a success guarantee.

Teacher's interface (pp. 4-6)
    from train import *
    agent = torch.load('a2c.model')
    action = agent.select_action(state, eval=True)
    Exported objects contain CPU tensors and basic metadata. Importing train
    registers only Agent for PyTorch's weights-only loader. Use your own models.

Compatibility references (algorithm remains based on the handout)
    https://github.com/Farama-Foundation/Gymnasium/blob/v0.29.1/gymnasium/envs/__init__.py
    https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/
    https://docs.pytorch.org/docs/2.8/generated/torch.load.html
    https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html
    https://arxiv.org/abs/1602.01783
    https://arxiv.org/abs/1707.06347
    https://github.com/DLR-RM/stable-baselines3/tree/v2.4.1
"""


# Import dependencies without creating environments, files or worker processes.
import argparse
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import platform
import random
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from contextlib import contextmanager
from datetime import datetime
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ENV_ID = "BipedalWalker-v3"
STATE_DIM = 24
ACTION_DIM = 4
EPISODE_LIMIT = 1600
FORMAT_VERSION = 1
VALIDATION_SEEDS = [101, 103, 105, 107, 109]
EXAMPLE_SEEDS = [1, 3, 5, 7, 9]
CAMPAIGN_VALIDATION_SEEDS = list(range(21001, 21041, 2))
CAMPAIGN_SELECTION_SEEDS = list(range(22001, 22081, 2))
CAMPAIGN_CONFIRMATION_SEEDS = list(range(23001, 23201, 2))
TRAIN_SEED_MIN = 1_000_000
BASELINE_URL = "https://github.com/Catherine135/cs100709-bipedalwalker-a2c/releases/download/campaign-v4-interrupted-20261108/final-best-a2c.model"
BASELINE_SHA256 = "8fd9ed360a38a95f287959daf859a0d683697b0b6d0dd127f0341874213a95c5"
BASELINE_CHECKPOINT_SHA256 = "20fbb89e90df4b0d4b92c52fb090f017c45b3a11ac1e26cc6b5e6fbd7c1cb5e5"
__all__ = ["Actor", "Agent", "Critic", "Mish"]


# Preserve the handout's activation and separate policy/value networks.
class Mish(nn.Module):
    def forward(self, inputs):
        return inputs * torch.tanh(F.softplus(inputs))


# Map observations to independent Gaussian means and learned joint deviations.
class Actor(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_dim, activation=Mish, std_mode="diagonal"):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), activation(),
            nn.Linear(hidden_dim, hidden_dim), activation(),
            nn.Linear(hidden_dim, n_actions),
        )
        self.logstds = nn.Parameter(torch.full((n_actions,), 0.1))
        self.register_parameter("sde_logstds", nn.Parameter(torch.zeros(hidden_dim, n_actions)) if std_mode == "sde" else None)


    # Bound log standard deviations before exponentiating to avoid overflow.
    def forward(self, states):
        means = self.model(states)
        if self.sde_logstds is None:
            stds = self.logstds.clamp(math.log(1e-3), math.log(50.0)).exp()
        else:
            features = self.model[:4](states).detach()
            stds = (features.square() @ self.sde_logstds.exp().square() + 1e-6).sqrt()
        return torch.distributions.Normal(means, stds)


# Estimate one state value using a network separate from the policy.
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim, activation=Mish):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), activation(),
            nn.Linear(hidden_dim, hidden_dim), activation(),
            nn.Linear(hidden_dim, 1),
        )


    def forward(self, states):
        return self.model(states).squeeze(-1)


# Construct a fixed TD target, retaining bootstrap value after time truncation.
def td_targets(rewards, next_values, terminated, gamma):
    return (rewards + gamma * (~terminated.bool()).to(rewards.dtype) * next_values).detach()


# Accumulate bounded n-step returns independently in each time-major worker stream.
def rollout_targets(rewards, next_values, terminated, truncated, gamma, num_envs, return_steps):
    if num_envs < 1 or return_steps < 1 or rewards.numel() % num_envs:
        raise ValueError("Returns require complete vector steps and a positive horizon.")
    if not (rewards.shape == next_values.shape == terminated.shape == truncated.shape):
        raise ValueError("Return arrays must have matching shapes.")
    with torch.no_grad():
        r, v = (array.reshape(-1, num_envs) for array in (rewards, next_values))
        term, trunc = (array.bool().reshape(-1, num_envs) for array in (terminated, truncated))
        targets = torch.zeros_like(r)
        if return_steps >= len(r):
            tail = v[-1]
            for index in reversed(range(len(r))):
                bootstrap = torch.where(trunc[index], v[index], tail)
                tail = r[index] + gamma * (~term[index]) * bootstrap
                targets[index] = tail
            return targets.reshape(rewards.shape)
        for start in range(len(r)):
            active = torch.ones(num_envs, dtype=torch.bool, device=r.device)
            discount = 1.0
            end = min(start + return_steps, len(r))
            for index in range(start, end):
                targets[start] += discount * active * r[index]
                boundary = term[index] | trunc[index] | (index == end - 1)
                targets[start] += discount * gamma * active * boundary * (~term[index]) * v[index]
                active = active & ~boundary  # Never consume rewards from an autoreset episode.
                discount *= gamma
        return targets.reshape(rewards.shape)


# Copy tensor trees to CPU without serializing modules, optimizers or devices.
def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(f"Unsupported checkpoint value: {type(value).__name__}")


# Expose the required Agent interface and implement the handout's one-step update.
class Agent:
    def __init__(self, args=None):
        values = vars(args) if isinstance(args, argparse.Namespace) else dict(args or {})
        self.config = {
            "state_dim": int(values.get("state_dim", STATE_DIM)),
            "n_actions": int(values.get("n_actions", ACTION_DIM)),
            "hidden_dim": int(values.get("hidden_dim", 128)),
            "gamma": float(values.get("gamma", 0.99)),
            "actor_lr": float(values.get("actor_lr", 4e-4)),
            "critic_lr": float(values.get("critic_lr", 4e-3)),
            "return_steps": int(values.get("return_steps", 1)),
            "max_grad_norm": float(values.get("max_grad_norm", 0.0)),
            "activation": values.get("activation", "mish"),
            "std_mode": values.get("std_mode", "diagonal"),
            "observation_normalization": values.get("observation_normalization", None),
            "action_low": np.broadcast_to(values.get("action_low", -1.0), (ACTION_DIM,)).tolist(),
            "action_high": np.broadcast_to(values.get("action_high", 1.0), (ACTION_DIM,)).tolist(),
        }
        if self.config["state_dim"] != STATE_DIM or self.config["n_actions"] != ACTION_DIM:
            raise ValueError("This project requires 24 observations and 4 actions.")
        if self.config["hidden_dim"] < 1 or not 0 < self.config["gamma"] <= 1:
            raise ValueError("hidden_dim must be positive and gamma must be in (0, 1].")
        for key in ("actor_lr", "critic_lr"):
            if not math.isfinite(self.config[key]) or self.config[key] <= 0:
                raise ValueError(f"{key} must be finite and positive.")
        if self.config["return_steps"] < 1 or not math.isfinite(self.config["max_grad_norm"]) or self.config["max_grad_norm"] < 0:
            raise ValueError("return_steps must be positive and max_grad_norm finite and nonnegative.")
        self.action_low = np.asarray(self.config["action_low"], dtype=np.float32)
        self.action_high = np.asarray(self.config["action_high"], dtype=np.float32)
        if not (np.all(self.action_low == -1) and np.all(self.action_high == 1)):
            raise ValueError("BipedalWalker requires action bounds [-1, 1] on every joint.")
        if self.config["activation"] not in ("mish", "tanh"):
            raise ValueError("Unsupported activation.")
        activation = Mish if self.config["activation"] == "mish" else nn.Tanh
        if self.config["std_mode"] not in ("diagonal", "sde"):
            raise ValueError("Unsupported policy standard deviation mode.")
        self.actor = Actor(STATE_DIM, ACTION_DIM, self.config["hidden_dim"], activation, self.config["std_mode"])
        self.critic = Critic(STATE_DIM, self.config["hidden_dim"], activation)
        self.actor_optim = None
        self.critic_optim = None
        self.to(values.get("device", "cpu"))


    @property
    def device(self):
        return next(self.actor.parameters()).device


    # Move parameters and any existing Adam state together.
    def to(self, device):
        device = torch.device(device)
        self.actor.to(device)
        self.critic.to(device)
        for optimizer in (self.actor_optim, self.critic_optim):
            if optimizer is not None:
                for state in optimizer.state.values():
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor) and key != "step":
                            state[key] = value.to(device)
        return self


    def ensure_optimizers(self):
        if self.actor_optim is None:
            self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.config["actor_lr"])
            self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.config["critic_lr"])


    # Validate observations and preserve a batch dimension for network evaluation.
    def state_tensor(self, states):
        array = np.asarray(states, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != STATE_DIM or len(array) == 0:
            raise ValueError(f"Expected a nonempty (batch, {STATE_DIM}) observation array.")
        if not np.isfinite(array).all():
            raise ValueError("Observations contain NaN or infinity.")
        normalization = self.config["observation_normalization"]
        if normalization is not None:
            mean = np.asarray(normalization["mean"], dtype=np.float64)
            variance = np.asarray(normalization["var"], dtype=np.float64)
            array = np.clip((array - mean) / np.sqrt(variance + normalization["epsilon"]),
                            -normalization["clip_obs"], normalization["clip_obs"]).astype(np.float32)
        return torch.as_tensor(array, device=self.device)


    @torch.no_grad()
    def sample_actions(self, states, deterministic=False):
        inputs = self.state_tensor(states)
        raw = self.actor.model(inputs) if deterministic else self.actor(inputs).sample()
        raw = raw.cpu().numpy().astype(np.float32, copy=False)
        if not np.isfinite(raw).all():
            raise FloatingPointError("The policy produced a non-finite action.")
        return raw, np.clip(raw, self.action_low, self.action_high).astype(np.float32)


    # Return deterministic actions for the teacher, or both training action forms.
    def select_action(self, state, eval=False):
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (STATE_DIM,):
            raise ValueError(f"Expected one observation with shape ({STATE_DIM},).")
        raw, clipped = self.sample_actions(state[None, :], deterministic=eval)
        return clipped[0] if eval else (raw[0], clipped[0])


    # Compute independent Actor/Critic losses with no gradient through TD targets.
    def losses(self, batch):
        states = self.state_tensor(batch["states"])
        next_states = self.state_tensor(batch["next_states"])
        size = len(states)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        terminated = torch.as_tensor(batch["terminated"], dtype=torch.bool, device=self.device)
        if actions.shape != (size, ACTION_DIM) or rewards.shape != (size,) or terminated.shape != (size,):
            raise ValueError("Transition arrays have inconsistent batch shapes.")
        if len(next_states) != size or not torch.isfinite(actions).all() or not torch.isfinite(rewards).all():
            raise ValueError("Transition arrays contain invalid values or mismatched lengths.")
        with torch.no_grad():
            next_values = self.critic(next_states)
            if self.config["return_steps"] == 1:
                targets = td_targets(rewards, next_values, terminated, self.config["gamma"])
            else:
                truncated = torch.as_tensor(batch["truncated"], dtype=torch.bool, device=self.device)
                targets = rollout_targets(rewards, next_values, terminated, truncated,
                                          self.config["gamma"], int(batch["num_envs"]), self.config["return_steps"])
        values = self.critic(states)
        advantage = targets - values
        distributions = self.actor(states)
        log_probs = distributions.log_prob(actions).sum(dim=-1)
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = F.mse_loss(values, targets)
        if not torch.isfinite(actor_loss) or not torch.isfinite(critic_loss):
            raise FloatingPointError("A non-finite loss was detected before updating parameters.")
        diagnostics = {
            "loss/actor": actor_loss.detach(),
            "loss/critic": critic_loss.detach(),
            "policy/log_prob": log_probs.detach().mean(),
            "policy/entropy": distributions.entropy().detach().sum(dim=-1).mean(),
            "policy/std": distributions.stddev.detach().mean(),
            "value/advantage": advantage.detach().mean(),
            "value/td_abs_error": advantage.detach().abs().mean(),
            "policy/mean_abs": distributions.mean.detach().abs().mean(),
            "policy/action_clip_fraction": (actions.detach().abs() > 1).float().mean(),
        }
        return actor_loss, critic_loss, diagnostics


    # Check both gradients before either optimizer is allowed to change weights.
    def backward_batch(self, batch):
        self.actor.zero_grad(set_to_none=True)
        self.critic.zero_grad(set_to_none=True)
        actor_loss, critic_loss, diagnostics = self.losses(batch)
        actor_loss.backward()
        critic_loss.backward()
        for network in (self.actor, self.critic):
            for parameter in network.parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError("A missing or non-finite gradient was detected.")
        for name, network in (("actor", self.actor), ("critic", self.critic)):
            limit = self.config["max_grad_norm"] or math.inf
            norm = nn.utils.clip_grad_norm_(network.parameters(), limit, error_if_nonfinite=True)
            diagnostics["gradient/" + name + "_norm"] = norm.detach()
        return diagnostics


    # Update parameters only when explicitly invoked by the training mode.
    def update(self, batch):
        self.ensure_optimizers()
        diagnostics = self.backward_batch(batch)
        self.actor_optim.step()
        self.critic_optim.step()
        return {name: float(value.cpu()) for name, value in diagnostics.items()}


    # Serialize CPU weights and primitive metadata while retaining an Agent object.
    def __getstate__(self):
        return {
            "format_version": FORMAT_VERSION,
            "config": cpu_tree(self.config),
            "actor": cpu_tree(self.actor.state_dict()),
            "critic": cpu_tree(self.critic.state_dict()),
        }


    def __setstate__(self, state):
        if state.get("format_version") != FORMAT_VERSION:
            raise ValueError("Unsupported Agent file format.")
        with torch.random.fork_rng(devices=[]):
            self.__init__(state["config"])
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])


# Allow only this module's portable Agent class in the teacher's torch.load call.
if __name__ == "train":
    torch.serialization.add_safe_globals([Agent])


# Create exactly the environment used by the teacher, including its time limit.
def make_env(render_mode=None):
    if mp.current_process().name != "MainProcess":
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # Let the learner finish and save on Ctrl+C.
    if gym.__version__ != "0.29.1":
        raise RuntimeError("Install gymnasium==0.29.1 to preserve the tested vector API and time limit.")
    env = gym.make(ENV_ID, hardcore=True, render_mode=render_mode)
    if (env.observation_space.shape != (STATE_DIM,)
            or env.action_space.shape != (ACTION_DIM,)
            or not np.all(env.action_space.low == -1)
            or not np.all(env.action_space.high == 1)
            or env.spec.max_episode_steps != EPISODE_LIMIT
            or not env.unwrapped.hardcore):
        env.close()
        raise RuntimeError("The environment does not match the course evaluation contract.")
    return env


def make_vector_env(num_envs):
    factories = [make_env] * num_envs
    if num_envs == 1:
        return gym.vector.SyncVectorEnv(factories)
    return gym.vector.AsyncVectorEnv(factories, context="spawn", shared_memory=True, copy=True)


# Replace autoreset observations with the actual final observations for TD targets.
def transition_next_states(observations, terminated, truncated, infos):
    result = np.asarray(observations, dtype=np.float32).copy()
    ended = np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
    if ended.any():
        if "final_observation" not in infos:
            raise RuntimeError("Missing final_observation from the pinned vector environment.")
        present = np.asarray(infos.get("_final_observation", ended), dtype=bool)
        for index in np.flatnonzero(ended):
            if not present[index] or infos["final_observation"][index] is None:
                raise RuntimeError("An ended episode has no final observation.")
            result[index] = infos["final_observation"][index]
    if not np.isfinite(result).all():
        raise FloatingPointError("A transition has a non-finite next observation.")
    return result


# Collect on-policy transitions and keep episode accounting separate for each worker.
class RolloutCollector:
    def __init__(self, env, agent, seed):
        self.env = env
        self.agent = agent
        self.num_envs = env.num_envs
        self.states, _ = env.reset(seed=[seed + index for index in range(self.num_envs)])
        self.returns = np.zeros(self.num_envs, dtype=np.float64)
        self.lengths = np.zeros(self.num_envs, dtype=np.int64)


    def collect(self, size, stop=None):
        if size <= 0 or size % self.num_envs:
            raise ValueError("A rollout size must be a positive multiple of num-envs.")
        rows = {key: [] for key in ("states", "actions", "rewards", "next_states", "terminated", "truncated")}
        episodes = []
        for _ in range(size // self.num_envs):
            if stop is not None and stop.requested:
                break
            raw, executed = self.agent.sample_actions(self.states)
            observations, rewards, terminated, truncated, infos = self.env.step(executed)
            next_states = transition_next_states(observations, terminated, truncated, infos)
            for key, value in zip(rows, (self.states, raw, rewards, next_states, terminated, truncated)):
                rows[key].append(np.array(value, copy=True))
            self.returns += rewards
            self.lengths += 1
            for index in np.flatnonzero(terminated | truncated):
                episodes.append({"reward": float(self.returns[index]), "length": int(self.lengths[index])})
                self.returns[index] = 0
                self.lengths[index] = 0
            self.states = np.asarray(observations, dtype=np.float32).copy()
        if not rows["states"]:
            return None, episodes
        batch = {key: np.concatenate(value, axis=0) for key, value in rows.items()}
        batch["num_envs"] = self.num_envs
        return batch, episodes


# Observe progress and reproduce the pinned environment reward without changing it.
class WalkerDiagnostics:
    def __init__(self, env):
        from gymnasium.envs.box2d.bipedal_walker import (
            MOTORS_TORQUE,
            SCALE,
            TERRAIN_GRASS,
            TERRAIN_LENGTH,
            TERRAIN_STEP,
        )
        self.walker = env.unwrapped
        self.scale = SCALE
        self.torque = MOTORS_TORQUE
        self.finish_x = (TERRAIN_LENGTH - TERRAIN_GRASS) * TERRAIN_STEP
        self.start_x = self.previous_x = self.max_x = float(self.walker.hull.position.x)
        self.previous_angle = abs(float(self.walker.hull.angle))
        self.terms = {"forward": 0.0, "posture": 0.0, "motor": 0.0, "terminal_adjustment": 0.0}
        self.raw_return = 0.0
        self.action_total = self.clipped = self.steps = 0
        self.window_x = self.previous_x
        self.window_max_x = self.previous_x
        self.stalled_windows = self.complete_windows = 0


    def observe(self, raw_action, action, reward):
        x, angle = float(self.walker.hull.position.x), abs(float(self.walker.hull.angle))
        terms = {"forward": 130 * (x - self.previous_x) / self.scale,
                 "posture": -5 * (angle - self.previous_angle),
                 "motor": -0.00035 * self.torque * float(np.clip(np.abs(action), 0, 1).sum(dtype=np.float64))}
        failed = self.walker.game_over or x < 0
        terms["terminal_adjustment"] = -100 - sum(terms.values()) if failed else 0.0
        for key, value in terms.items():
            self.terms[key] += value
        self.raw_return += float(reward)
        self.action_total += float(np.abs(action).sum())
        self.clipped += int(np.count_nonzero(np.abs(raw_action) > 1))
        self.steps += 1
        self.previous_x, self.previous_angle = x, angle
        self.max_x = max(self.max_x, x)
        self.window_max_x = max(self.window_max_x, x)
        if self.steps % 100 == 0:
            self.stalled_windows += int(self.window_max_x - self.window_x < 0.1)
            self.complete_windows += 1
            self.window_x = self.window_max_x = x


    def result(self, terminated, truncated):
        failed = bool(self.walker.game_over or self.previous_x < 0)
        success = bool(terminated and not failed and self.previous_x > self.finish_x)
        reason = ("fall" if self.walker.game_over else "left_boundary" if self.previous_x < 0
                  else "finish" if success else "time_limit" if truncated else "other_termination")
        return {"terminated": bool(terminated), "truncated": bool(truncated), "success": success,
                "end_reason": reason, "x_displacement": self.previous_x - self.start_x,
                "max_x_displacement": self.max_x - self.start_x,
                "progress_fraction": (self.max_x - self.start_x) / (self.finish_x - self.start_x),
                "action_clip_fraction": self.clipped / (ACTION_DIM * max(1, self.steps)),
                "mean_abs_action": self.action_total / (ACTION_DIM * max(1, self.steps)),
                "stall_fraction": self.stalled_windows / max(1, self.complete_windows),
                "stall_window_steps": 100, "stall_distance_threshold": 0.1,
                "reward_terms": self.terms.copy(),
                "reward_reconstruction_error": self.raw_return - sum(self.terms.values())}


# Evaluate complete episodes using the original, undiscounted environment reward.
def evaluate_agent(agent, seeds, render=False, env=None):
    owned = env is None
    env = make_env("rgb_array" if render else None) if owned else env
    figure = None
    rewards, lengths, diagnostics = [], [], []
    try:
        if render:
            import matplotlib.pyplot as plt
            plt.ion()
            figure, axes = plt.subplots()
            artist = None
        for seed in seeds:
            state, _ = env.reset(seed=int(seed))
            tracker = WalkerDiagnostics(env)
            total = 0.0
            for step in range(1, EPISODE_LIMIT + 1):
                raw, executed = agent.sample_actions(np.asarray(state)[None, :], deterministic=True)
                action = executed[0]
                state, reward, terminated, truncated, _ = env.step(action)
                tracker.observe(raw[0], action, reward)
                total += float(reward)
                if render:
                    frame = env.render()
                    if artist is None:
                        artist = axes.imshow(frame)
                    else:
                        artist.set_data(frame)
                    axes.set_title(f"seed {seed} | step {step} | reward {total:.3f}")
                    figure.canvas.draw_idle()
                    plt.pause(0.001)
                if terminated or truncated:
                    break
            else:
                raise RuntimeError("Evaluation exceeded the course environment's time limit.")
            rewards.append(total)
            lengths.append(step)
            diagnostics.append(tracker.result(terminated, truncated))
    finally:
        if owned:
            env.close()
        if figure is not None:
            plt.close(figure)
    return {"seeds": list(seeds), "rewards": rewards, "lengths": lengths,
            "mean_reward": float(np.mean(rewards)), "std_reward": float(np.std(rewards)),
            "diagnostics": diagnostics,
            "success_rate": float(np.mean([item["success"] for item in diagnostics]))}


# Save atomically after closing the temporary file, including on Windows.
def atomic_save(value, destination):
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    os.close(handle)
    try:
        torch.save(value, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def agent_from_state(state):
    agent = Agent.__new__(Agent)
    agent.__setstate__(state)
    return agent


# Preserve random generators using only weights-only-compatible values.
def capture_rng():
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "numpy": {"name": numpy_state[0], "keys": numpy_state[1].tolist(),
                      "position": numpy_state[2], "has_gauss": numpy_state[3], "cached": numpy_state[4]},
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state["name"], np.asarray(numpy_state["keys"], dtype=np.uint32),
                         numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached"]))
    if torch.cuda.is_available() and len(state["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(state["cuda"])


# Defer Ctrl+C until the current batch update has completed.
class StopRequest:
    def __init__(self):
        self.requested = False
        self.previous = None


    def __enter__(self):
        self.previous = signal.signal(signal.SIGINT, self.handle)
        return self


    def handle(self, signum, frame):
        self.requested = True
        print("\nStop requested; finishing the current update, evaluation and checkpoint.", flush=True)


    def __exit__(self, *exception):
        signal.signal(signal.SIGINT, self.previous)


# Manage training, validation and resumable state at completed update boundaries.
class TrainManager:
    def __init__(self, args, agent, env, run_dir, writer, resume=None):
        self.args = args
        self.agent = agent
        self.run_dir = Path(run_dir)
        self.writer = writer
        self.steps = int(resume["steps"]) if resume else 0
        self.episodes = int(resume["episodes"]) if resume else 0
        self.updates = int(resume["updates"]) if resume else 0
        self.best_score = float(resume["best_score"]) if resume else -math.inf
        self.best_state = resume["best_agent"] if resume else None
        self.validation_seeds = list(resume["validation_seeds"]) if resume else list(args.validation_seeds)
        self.next_evaluation = (self.episodes // args.eval_freq + 1) * args.eval_freq
        self.next_save = (self.episodes // args.save_freq + 1) * args.save_freq
        self.last_eval_step = None
        if args.steps < self.steps or (args.steps - self.steps) % args.num_envs:
            raise ValueError("Remaining --steps must be nonnegative and divisible by --num-envs.")
        if resume:
            agent.ensure_optimizers()
            agent.actor_optim.load_state_dict(resume["actor_optimizer"])
            agent.critic_optim.load_state_dict(resume["critic_optimizer"])
            agent.to(args.device)
            restore_rng(resume["rng"])
        episode_seed = (args.seed + self.steps) % (2**32 - args.num_envs)
        self.collector = RolloutCollector(env, agent, episode_seed)
        self.initial_steps = self.steps
        self.active_seconds = 0.0
        if self.best_state is not None:
            self.export_best()


    # Keep a best model in the run directory and at the teacher's expected path.
    def export_best(self):
        best_agent = agent_from_state(self.best_state)
        for destination in {self.run_dir / "a2c.model", Path(self.args.model_path).resolve()}:
            atomic_save(best_agent, destination)


    def evaluate(self):
        result = evaluate_agent(self.agent, self.validation_seeds)
        result["steps"] = self.steps
        result["episodes"] = self.episodes
        for key in ("mean_reward", "std_reward"):
            self.writer.add_scalar("evaluation/" + key, result[key], self.steps)
        if "diagnostics" in result:
            self.writer.add_scalar("evaluation/success_rate", result["success_rate"], self.steps)
            for key in ("progress_fraction", "action_clip_fraction", "mean_abs_action"):
                self.writer.add_scalar("evaluation/" + key, np.mean([row[key] for row in result["diagnostics"]]), self.steps)
        with (self.run_dir / "evaluations.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps(result, allow_nan=False) + "\n")
        print(f"Evaluation at {self.steps} steps: {result['rewards']} | mean {result['mean_reward']:.3f}", flush=True)
        if result["mean_reward"] > self.best_score:
            self.best_score = result["mean_reward"]
            self.best_state = self.agent.__getstate__()
            self.export_best()
        self.last_eval_step = self.steps


    def save_checkpoint(self):
        self.agent.ensure_optimizers()
        state = {"format_version": FORMAT_VERSION, "kind": "a2c_training_checkpoint",
                 "agent": self.agent.__getstate__(),
                 "actor_optimizer": cpu_tree(self.agent.actor_optim.state_dict()),
                 "critic_optimizer": cpu_tree(self.agent.critic_optim.state_dict()),
                 "steps": self.steps, "episodes": self.episodes, "updates": self.updates,
                 "best_score": self.best_score, "best_agent": self.best_state,
                 "validation_seeds": self.validation_seeds, "config": vars(self.args).copy(),
                 "rng": cpu_tree(capture_rng())}
        atomic_save(state, self.run_dir / "checkpoint.pt")
        self.writer.flush()


    def run(self, stop):
        while self.steps < self.args.steps and not stop.requested:
            started = time.perf_counter()
            size = min(self.args.batch_size, self.args.steps - self.steps)
            batch, episodes = self.collector.collect(size, stop)
            if batch is None:
                break
            metrics = self.agent.update(batch)
            self.steps += len(batch["rewards"])
            self.updates += 1
            self.active_seconds += time.perf_counter() - started
            for episode in episodes:
                self.episodes += 1
                self.writer.add_scalar("train/episode_reward", episode["reward"], self.steps)
                self.writer.add_scalar("train/episode_length", episode["length"], self.steps)
            for name, value in metrics.items():
                self.writer.add_scalar(name, value, self.steps)
            throughput = (self.steps - self.initial_steps) / max(self.active_seconds, 1e-9)
            self.writer.add_scalar("performance/active_steps_per_second", throughput, self.steps)
            if self.updates % self.args.log_interval == 0:
                print(f"steps {self.steps}/{self.args.steps} | episodes {self.episodes} | {throughput:.1f} steps/s", flush=True)
            if self.episodes >= self.next_save:
                self.save_checkpoint()
                self.next_save = (self.episodes // self.args.save_freq + 1) * self.args.save_freq
            if self.episodes >= self.next_evaluation:
                self.evaluate()
                self.next_evaluation = (self.episodes // self.args.eval_freq + 1) * self.args.eval_freq
                self.save_checkpoint()
        self.save_checkpoint()
        try:
            if self.last_eval_step != self.steps:
                self.evaluate()
        finally:
            self.save_checkpoint()
        print(f"Saved {self.run_dir / 'checkpoint.pt'}\nBest validation mean: {self.best_score:.3f}", flush=True)


# Seed libraries and constrain worker threads before starting any subprocesses.
def configure_runtime(args):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(args.torch_threads)
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Check the NVIDIA driver and CUDA PyTorch wheel.")


# Report real runtime capabilities without assuming the target machine has CUDA.
def preflight(args):
    import Box2D
    import pygame
    result = {"platform": platform.platform(), "python": platform.python_version(),
              "numpy": np.__version__, "gymnasium": gym.__version__, "torch": str(torch.__version__),
              "box2d": str(Box2D.__version__), "pygame": pygame.version.ver,
              "device": args.device, "cuda_available": torch.cuda.is_available(),
              "torch_cuda": torch.version.cuda, "logical_cpus": os.cpu_count(),
              "num_envs": args.num_envs, "torch_threads": torch.get_num_threads()}
    if torch.cuda.is_available():
        result["gpu"] = torch.cuda.get_device_name(0)
    print(json.dumps(result, indent=2), flush=True)
    if sys.version_info[:2] != (3, 11) or np.__version__ != "1.26.4" or str(torch.__version__).split("+")[0] != "2.8.0":
        print("Runtime differs from the documented dependency baseline; run --mode check before training.", flush=True)
    return result


# Restore a checkpoint or create an Agent using dimensions read from the environment.
def run_training(args):
    from torch.utils.tensorboard import SummaryWriter
    resume = None
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=True)
        if resume.get("kind") != "a2c_training_checkpoint" or resume.get("format_version") != FORMAT_VERSION:
            raise ValueError("--resume must point to this script's training checkpoint.pt.")
        agent = agent_from_state(resume["agent"]).to(args.device)
        print("Restoring model hyperparameters and validation seeds from checkpoint; episodes restart.", flush=True)
        args.return_steps = agent.config["return_steps"]
        args.max_grad_norm = agent.config["max_grad_norm"]
        print(f"Restored return_steps={args.return_steps}, max_grad_norm={args.max_grad_norm}; command-line overrides do not change checkpoint learning settings.", flush=True)
        args.seed = int(resume["config"]["seed"])
    else:
        probe = make_env()
        try:
            values = vars(args).copy()
            values.update(state_dim=probe.observation_space.shape[0], n_actions=probe.action_space.shape[0],
                          action_low=probe.action_space.low.tolist(), action_high=probe.action_space.high.tolist())
            agent = Agent(values)
        finally:
            probe.close()
    log_root = Path(args.log_dir)
    log_root.mkdir(parents=True, exist_ok=True)
    prefix = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-seed{args.seed}-"
    run_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=log_root)).resolve()
    effective = {"arguments": vars(args), "agent": agent.config, "source": "CS100709 handout pp. 31-40"}
    (run_dir / "config.json").write_text(json.dumps(effective, indent=2), encoding="utf-8")
    env = make_vector_env(args.num_envs)
    writer = SummaryWriter(str(run_dir))
    try:
        manager = TrainManager(args, agent, env, run_dir, writer, resume)
        print(f"Training directory: {run_dir}", flush=True)
        with StopRequest() as stop:
            manager.run(stop)
    finally:
        env.close()
        writer.close()


# Compare sampling and differentiation throughput without taking optimizer steps.
def run_benchmark(args):
    agent = Agent(args)
    before = agent.__getstate__()
    env = make_vector_env(args.num_envs)
    try:
        collector = RolloutCollector(env, agent, args.seed)
        batch, _ = collector.collect(args.batch_size)
        agent.backward_batch(batch)
        if args.device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        samples = 0
        while time.perf_counter() - started < args.benchmark_seconds:
            batch, _ = collector.collect(args.batch_size)
            agent.backward_batch(batch)
            if args.device == "cuda":
                torch.cuda.synchronize()
            samples += len(batch["rewards"])
        seconds = time.perf_counter() - started
        assert_weights_equal(before, agent.__getstate__())
        print(json.dumps({"device": args.device, "num_envs": args.num_envs, "samples": samples,
                          "seconds": seconds, "steps_per_second": samples / seconds,
                          "optimizer_steps": 0, "weights_unchanged": True}, indent=2))
    finally:
        env.close()


# Compare state dictionaries exactly when checking non-training modes and reloads.
def assert_weights_equal(first, second):
    for network in ("actor", "critic"):
        if first[network].keys() != second[network].keys():
            raise AssertionError("Different parameter names.")
        for key in first[network]:
            torch.testing.assert_close(first[network][key], second[network][key], rtol=0, atol=0)


# Provide a tiny autoresetting environment for deterministic collector checks.
class CheckVectorEnv:
    num_envs = 2


    def reset(self, seed=None):
        self.turn = 0
        return np.zeros((2, STATE_DIM), dtype=np.float32), {}


    def step(self, actions):
        self.turn += 1
        observations = np.full((2, STATE_DIM), self.turn, dtype=np.float32)
        terminated = np.array([self.turn % 2 == 0, False])
        truncated = np.array([False, self.turn % 3 == 0])
        ended = terminated | truncated
        finals = np.empty(2, dtype=object)
        finals[:] = None
        for index in np.flatnonzero(ended):
            finals[index] = observations[index].copy()
            observations[index] = -100
        return observations, np.array([1.0, 2.0]), terminated, truncated, {
            "final_observation": finals, "_final_observation": ended}


# Check analytical values, gradient isolation and collectors without fitting a model.
def check_math_and_flow(agent):
    agent = Agent({**agent.config, "device": agent.device.type, "return_steps": 1})
    rewards = torch.tensor([1.0, 2.0, 3.0])
    next_values = torch.tensor([10.0, 20.0, 30.0], requires_grad=True)
    target = td_targets(rewards, next_values, torch.tensor([False, True, False]), 0.9)
    torch.testing.assert_close(target, torch.tensor([10.0, 2.0, 30.0]))
    assert not target.requires_grad
    normal = torch.distributions.Normal(torch.zeros(2, 4), torch.ones(2, 4))
    joint = normal.log_prob(torch.zeros(2, 4)).sum(-1)
    torch.testing.assert_close(joint, torch.full((2,), -2 * math.log(2 * math.pi)))
    rng = np.random.default_rng(73)
    batch = {"states": rng.normal(size=(3, STATE_DIM)).astype(np.float32),
             "actions": np.full((3, ACTION_DIM), 1.5, dtype=np.float32),
             "rewards": np.array([1, 2, 3], dtype=np.float32),
             "next_states": rng.normal(size=(3, STATE_DIM)).astype(np.float32),
             "terminated": np.array([False, True, False]),
             "truncated": np.array([False, False, True])}
    actor_loss, critic_loss, _ = agent.losses(batch)
    with torch.no_grad():
        inputs = agent.state_tensor(batch["states"])
        next_inputs = agent.state_tensor(batch["next_states"])
        expected_target = torch.tensor(batch["rewards"], device=agent.device) + agent.config["gamma"] * torch.tensor([1, 0, 1], device=agent.device) * agent.critic(next_inputs)
        advantage = expected_target - agent.critic(inputs)
        distribution = agent.actor(inputs)
        raw_actions = torch.tensor(batch["actions"], device=agent.device)
        expected_actor = -(distribution.log_prob(raw_actions).sum(-1) * advantage).mean()
        expected_critic = ((agent.critic(inputs) - expected_target) ** 2).mean()
    torch.testing.assert_close(actor_loss, expected_actor)
    torch.testing.assert_close(critic_loss, expected_critic)
    agent.actor.zero_grad(set_to_none=True)
    agent.critic.zero_grad(set_to_none=True)
    actor_loss.backward()
    assert all(parameter.grad is None for parameter in agent.critic.parameters())
    assert all(parameter.grad is not None for parameter in agent.actor.parameters())
    agent.actor.zero_grad(set_to_none=True)
    critic_loss.backward()
    assert all(parameter.grad is None for parameter in agent.actor.parameters())
    agent.backward_batch(batch)
    collector = RolloutCollector(CheckVectorEnv(), agent, seed=1)
    first, episodes = collector.collect(4)
    assert len(first["rewards"]) == 4 and episodes == [{"reward": 2.0, "length": 2}]
    assert np.all(first["next_states"][2] == 2)
    last, episodes = collector.collect(2)
    assert len(last["rewards"]) == 2 and episodes == [{"reward": 6.0, "length": 3}]
    assert np.all(last["next_states"][1] == 3) and last["truncated"][1]
    assert last["terminated"].sum() == 0
    print("PASS: analytical TD/losses, gradient isolation, raw actions, autoreset and partial batches.")


# Check multistep boundaries, clipping and reward diagnostics without optimizer updates.
def check_revision(agent):
    from types import SimpleNamespace

    rewards = torch.tensor([1., 10., 2., 20., 100., 30.], device=agent.device)
    next_values = torch.tensor([5., 50., 6., 60., 7., 70.], device=agent.device, requires_grad=True)
    terminated = torch.tensor([False, False, True, False, False, False], device=agent.device)
    truncated = torch.tensor([False, False, False, True, False, False], device=agent.device)
    target = rollout_targets(rewards, next_values, terminated, truncated, 0.5, 2, 3)
    torch.testing.assert_close(target, torch.tensor([2., 35., 2., 50., 103.5, 65.], device=agent.device))
    assert not target.requires_grad
    torch.testing.assert_close(rollout_targets(rewards, next_values, terminated, truncated, 0.5, 2, 1),
                               td_targets(rewards, next_values, terminated, 0.5))
    torch.testing.assert_close(rollout_targets(rewards, next_values, terminated, truncated, 0.5, 2, 20), target)
    torch.testing.assert_close(rollout_targets(rewards, next_values, terminated, truncated, 0.5, 2, 2), target)
    no_end = torch.zeros_like(terminated)
    shorter = rollout_targets(rewards, next_values, no_end, no_end, 0.5, 2, 2)
    torch.testing.assert_close(shorter, torch.tensor([3.5, 35., 53.75, 52.5, 103.5, 65.], device=agent.device))
    revised = Agent({**agent.config, "device": agent.device.type, "return_steps": 3, "max_grad_norm": 0.5})
    before = revised.__getstate__()
    batch, _ = RolloutCollector(CheckVectorEnv(), revised, 1).collect(6)
    batch["rewards"] *= 10000
    actor_loss, critic_loss, _ = revised.losses(batch)
    actor_loss.backward()
    assert all(p.grad is None for p in revised.critic.parameters())
    revised.actor.zero_grad(set_to_none=True)
    critic_loss.backward()
    assert all(p.grad is None for p in revised.actor.parameters())
    metrics = revised.backward_batch(batch)
    assert any(float(metrics[f"gradient/{name}_norm"]) > 0.5 for name in ("actor", "critic"))
    for network in (revised.actor, revised.critic):
        norm = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in network.parameters()]))
        assert float(norm) <= 0.50001
    assert_weights_equal(before, revised.__getstate__())

    # Legacy exports keep the old one-step behavior when new config keys are absent.
    legacy = agent.__getstate__()
    for key in ("return_steps", "max_grad_norm"):
        legacy["config"].pop(key)
    loaded = agent_from_state(legacy)
    assert loaded.config["return_steps"] == 1 and loaded.config["max_grad_norm"] == 0
    assert_weights_equal(legacy, loaded.__getstate__())

    # A time limit is not success; falling replaces the entire step reward with -100.
    walker = SimpleNamespace(hull=SimpleNamespace(position=SimpleNamespace(x=5.0), angle=0.0), game_over=False)
    tracker = WalkerDiagnostics(SimpleNamespace(unwrapped=walker))
    walker.hull.position.x = 6.0
    walker.hull.angle = 0.1
    raw, action = np.full(4, 2.0), np.ones(4)
    reward = 130 / 30 - 0.5 - 0.028 * 4
    tracker.observe(raw, action, reward)
    row = tracker.result(False, True)
    assert row["end_reason"] == "time_limit" and not row["success"]
    assert row["action_clip_fraction"] == 1 and row["x_displacement"] == 1
    assert abs(row["reward_reconstruction_error"]) < 1e-6
    walker.game_over = True
    tracker.observe(raw, action, -100)
    row = tracker.result(True, False)
    assert row["end_reason"] == "fall" and abs(row["reward_reconstruction_error"]) < 1e-6
    walker.game_over = False
    walker.hull.position.x = tracker.finish_x + 1
    tracker.observe(np.zeros(4), np.zeros(4), 130 * (walker.hull.position.x - 6) / 30)
    assert tracker.result(True, False)["success"]
    print("PASS: multistep terminal/truncation boundaries, partial rollouts, gradient clipping, legacy models and reward diagnostics.")


# Verify import, serialization and the exact teacher interface in a fresh process.
def check_fresh_process(agent, directory):
    model_path = Path(directory) / "untrained.model"
    atomic_save(agent, model_path)
    state = np.linspace(-0.25, 0.25, STATE_DIM, dtype=np.float32)
    expected = agent.select_action(state, eval=True)
    source = (
        "import json, sys, torch, numpy as np\n"
        "from train import *\n"
        "agent = torch.load(sys.argv[1])\n"
        "assert isinstance(agent, Agent)\n"
        "assert agent.device.type == 'cpu'\n"
        "state = np.linspace(-0.25, 0.25, 24, dtype=np.float32)\n"
        "print(json.dumps(agent.select_action(state, eval=True).tolist()))\n"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parent)
    result = subprocess.run([sys.executable, "-B", "-c", source, str(model_path)],
                            capture_output=True, text=True, check=False, env=environment, cwd=directory)
    if result.returncode:
        raise RuntimeError(f"Fresh-process model loading failed:\n{result.stderr}")
    actual = np.asarray(json.loads(result.stdout.strip().splitlines()[-1]), dtype=np.float32)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    loaded = torch.load(model_path)
    assert_weights_equal(agent.__getstate__(), loaded.__getstate__())
    print("PASS: fresh-process teacher import/load, CPU export and action equivalence.")


# Simulate update boundaries, interruption and resume while prohibiting optimizer steps.
def check_training_lifecycle(agent, directory):
    import contextlib
    from unittest.mock import patch

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from torch.utils.tensorboard import SummaryWriter

    root = Path(directory)
    run_dir = root / "first"
    run_dir.mkdir()
    args = parse_args(["--num-envs", "2", "--batch-size", "4", "--steps", "6",
                       "--eval-freq", "1", "--save-freq", "1", "--device", agent.device.type,
                       "--model-path", str(root / "submission.model")])
    checked_agent = agent_from_state(agent.__getstate__()).to(args.device)
    checked_agent.ensure_optimizers()
    for optimizer in (checked_agent.actor_optim, checked_agent.critic_optim):
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                optimizer.state[parameter] = {"step": torch.tensor(7.0),
                                              "exp_avg": torch.full_like(parameter, 0.25),
                                              "exp_avg_sq": torch.full_like(parameter, 0.5)}
    sizes, saved_updates = [], []
    scores = iter([1.0, 0.5, 0.25, 0.125])
    real_save = atomic_save

    # Use synthetic validation scores solely to verify best-model bookkeeping.
    def fake_evaluation(current_agent, seeds):
        score = next(scores)
        return {"seeds": list(seeds), "rewards": [score] * len(seeds),
                "lengths": [1] * len(seeds), "mean_reward": score, "std_reward": 0.0}

    def checked_save(value, path):
        if isinstance(value, dict) and value.get("kind") == "a2c_training_checkpoint":
            saved_updates.append(value["updates"])
            assert value["updates"] == len(sizes)
        real_save(value, path)

    def inspect_update(current_agent, batch):
        sizes.append(len(batch["rewards"]))
        _, _, diagnostics = current_agent.losses(batch)
        return {name: float(value.cpu()) for name, value in diagnostics.items()}

    with patch.object(torch.optim.Adam, "step", side_effect=AssertionError("Optimizer steps are forbidden in checks")), \
            patch.object(Agent, "update", inspect_update), \
            patch("train.evaluate_agent", fake_evaluation), patch("train.atomic_save", checked_save), \
            contextlib.redirect_stdout(io.StringIO()):
        writer = SummaryWriter(str(run_dir))
        try:
            manager = TrainManager(args, checked_agent, CheckVectorEnv(), run_dir, writer)
            manager.run(StopRequest())
        finally:
            writer.close()
        checkpoint = torch.load(run_dir / "checkpoint.pt", weights_only=True)
        assert sizes == [4, 2] and saved_updates
        assert (checkpoint["steps"], checkpoint["episodes"], checkpoint["updates"]) == (6, 2, 2)
        assert checkpoint["best_score"] == 1.0
        assert checkpoint["actor_optimizer"]["state"][0]["step"].item() == 7.0
        assert_weights_equal(agent.__getstate__(), checkpoint["agent"])
        events = EventAccumulator(str(run_dir)).Reload()
        assert len(events.Scalars("train/episode_reward")) == 2
        assert len(events.Scalars("evaluation/mean_reward")) == 2

        # Resume the saved optimizer state and retain the earlier best model.
        resumed_dir = root / "resumed"
        resumed_dir.mkdir()
        args.steps = 10
        writer = SummaryWriter(str(resumed_dir))
        resumed_agent = agent_from_state(checkpoint["agent"]).to(args.device)
        try:
            resumed = TrainManager(args, resumed_agent, CheckVectorEnv(), resumed_dir, writer, checkpoint)
            resumed.run(StopRequest())
        finally:
            writer.close()
        continued = torch.load(resumed_dir / "checkpoint.pt", weights_only=True)
        assert sizes == [4, 2, 4] and continued["steps"] == 10
        assert continued["updates"] == 3 and continued["best_score"] == 1.0
        assert_weights_equal(checkpoint["best_agent"], continued["best_agent"])
        torch.testing.assert_close(continued["actor_optimizer"]["state"][0]["exp_avg"],
                                   checkpoint["actor_optimizer"]["state"][0]["exp_avg"])

        # Request a stop during a simulated update and check the saved boundary.
        stopped_dir = root / "stopped"
        stopped_dir.mkdir()
        args.steps = 18
        stop = StopRequest()
        def interrupted_update(current_agent, batch):
            metrics = inspect_update(current_agent, batch)
            stop.requested = True
            return metrics
        writer = SummaryWriter(str(stopped_dir))
        try:
            stopped = TrainManager(args, agent_from_state(continued["agent"]).to(args.device),
                                   CheckVectorEnv(), stopped_dir, writer, continued)
            with patch.object(Agent, "update", interrupted_update):
                stopped.run(stop)
        finally:
            writer.close()
        interrupted = torch.load(stopped_dir / "checkpoint.pt", weights_only=True)
        assert interrupted["steps"] == 14 and interrupted["updates"] == 4
        assert_weights_equal(agent.__getstate__(), interrupted["agent"])
    print("PASS: mocked update boundaries, TensorBoard, best model, optimizer-state resume and interruption (0 optimizer steps).")


# Exercise real Box2D sampling, spawn workers and rendering with unchanged weights.
def run_checks(args):
    agent = Agent(args)
    before = agent.__getstate__()
    check_math_and_flow(agent)
    check_revision(agent)
    state = np.zeros(STATE_DIM, dtype=np.float32)
    first = agent.select_action(state, eval=True)
    np.testing.assert_array_equal(first, agent.select_action(state, eval=True))
    raw, action = agent.select_action(state)
    assert raw.shape == action.shape == (ACTION_DIM,) and action.dtype == np.float32
    assert np.all(action >= -1) and np.all(action <= 1)
    env = make_env("rgb_array")
    try:
        state, _ = env.reset(seed=1)
        for _ in range(16):
            state, _, terminated, truncated, _ = env.step(agent.select_action(state, eval=True))
            if terminated or truncated:
                state, _ = env.reset()
        frame = env.render()
        assert frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8
    finally:
        env.close()
    for count in (1, args.num_envs):
        vector = make_vector_env(count)
        try:
            collector = RolloutCollector(vector, agent, args.seed)
            batch, _ = collector.collect(2 * count)
            assert len(batch["rewards"]) == 2 * count
        finally:
            vector.close()
    print(f"PASS: real Box2D, RGB rendering, single environment and {args.num_envs} spawn workers.")
    with tempfile.TemporaryDirectory(prefix="a2c-check-") as directory:
        check_fresh_process(agent, directory)
        check_training_lifecycle(agent, directory)
    result = evaluate_agent(agent, EXAMPLE_SEEDS)
    assert len(result["rewards"]) == 5
    assert result["mean_reward"] == sum(result["rewards"]) / len(result["rewards"])
    assert all(abs(row["reward_reconstruction_error"]) < 1e-4 for row in result["diagnostics"])
    assert_weights_equal(before, agent.__getstate__())
    print("PASS: five complete evaluation episodes and unchanged network weights.")
    print("All checks passed. Optimizer steps: 0. No trained model was produced.")


# Import the optional, pinned PPO implementation only for PPO work.
def ppo_dependencies():
    import stable_baselines3 as sb3
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import (
        DummyVecEnv,
        SubprocVecEnv,
        VecNormalize,
    )
    if sb3.__version__ != "2.4.1":
        raise RuntimeError("PPO requires stable-baselines3==2.4.1; reuse the existing Python environment.")
    return sb3.PPO, BaseCallback, Monitor, DummyVecEnv, SubprocVecEnv, VecNormalize


# Compute the probability of executed actions, including atoms at the two clipping bounds.
def clipped_normal_log_prob(means, stds, actions):
    lower = torch.special.log_ndtr((-1.0 - means) / stds)
    upper = torch.special.log_ndtr((means - 1.0) / stds)
    interior = torch.distributions.Normal(means, stds).log_prob(actions.clamp(-1., 1.))
    return torch.where(actions <= -1., lower, torch.where(actions >= 1., upper, interior)).sum(dim=-1)


# Report mixed discrete/continuous entropy in the fixed action coordinates [-1, 1].
def clipped_normal_entropy(means, stds):
    a, b = (-1. - means) / stds, (1. - means) / stds
    log_left, log_right = torch.special.log_ndtr(a), torch.special.log_ndtr(-b)
    inside = (torch.special.ndtr(b) - torch.special.ndtr(a)).clamp(0., 1.)
    boundary = -(log_left.exp() * log_left + log_right.exp() * log_right)
    phi_a, phi_b = torch.exp(-.5 * a.square()) / math.sqrt(2 * math.pi), torch.exp(-.5 * b.square()) / math.sqrt(2 * math.pi)
    continuous = inside * (stds.log() + .5 * math.log(2 * math.pi) + .5) + .5 * (a * phi_a - b * phi_b)
    return (boundary + continuous).sum(dim=-1)


# Adapt SB3's distribution protocol lazily; portable teacher models still need no SB3.
def configure_ppo_distribution(model, capg=False):
    from stable_baselines3.common.distributions import DiagGaussianDistribution
    if model.use_sde:
        if capg:
            raise ValueError("CAPG requires independent diagonal Gaussian actions, not gSDE.")
        return

    class ClippedGaussian(DiagGaussianDistribution):
        def sample(self):
            return super().sample().clamp(-1., 1.)

        def mode(self):
            return super().mode().clamp(-1., 1.)

        def log_prob(self, actions):
            return clipped_normal_log_prob(self.distribution.mean, self.distribution.stddev, actions)

        def entropy(self):
            return clipped_normal_entropy(self.distribution.mean, self.distribution.stddev)

    # SB3 saves network/optimizer state, not this object; explicitly reconstruct it after every load.
    model.policy.action_dist = ClippedGaussian(ACTION_DIM) if capg else DiagGaussianDistribution(ACTION_DIM)


# Revisit a bounded sample of failed training terrains without changing observations or rewards.
class TerrainReplay(gym.Wrapper):
    def __init__(self, env, replay_fraction=0.25):
        super().__init__(env)
        self.replay_fraction = replay_fraction
        self.rng = random.Random(0)
        self.failures = {}
        self.restored = False
        self.terrain_seed = None


    def reset(self, *, seed=None, options=None):
        if seed is not None and not self.restored:
            self.rng.seed(int(seed))
            self.failures.clear()
        self.restored = False
        self.replayed = bool(self.failures and self.rng.random() < self.replay_fraction)
        if self.replayed:
            self.terrain_seed = self.rng.choice(list(self.failures))
            self.failures[self.terrain_seed] += 1
        else:
            self.terrain_seed = self.rng.randrange(TRAIN_SEED_MIN, 2**31)
        observation, info = self.env.reset(seed=self.terrain_seed, options=options)
        self.episode_return = self.motor_cost = 0.0
        self.episode_steps = 0
        return observation, info


    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.episode_return += float(reward)
        self.episode_steps += 1
        self.motor_cost += 0.028 * float(np.clip(np.abs(action), 0, 1).sum(dtype=np.float64))
        if terminated or truncated:
            from gymnasium.envs.box2d.bipedal_walker import TERRAIN_LENGTH, TERRAIN_GRASS, TERRAIN_STEP
            walker = self.env.unwrapped
            x = float(walker.hull.position.x)
            success = bool(terminated and not walker.game_over and x > (TERRAIN_LENGTH - TERRAIN_GRASS) * TERRAIN_STEP)
            retries = self.failures.get(self.terrain_seed, 0)
            if success or retries >= 3:
                self.failures.pop(self.terrain_seed, None)
            else:
                self.failures[self.terrain_seed] = retries
                if len(self.failures) > 128:
                    del self.failures[next(iter(self.failures))]
            info = dict(info)
            info["terrain"] = {"seed": self.terrain_seed, "replayed": self.replayed, "success": success,
                               "motor_cost": self.motor_cost, "reward": self.episode_return,
                               "end_reason": "finish" if success else "fall" if walker.game_over or x < 0 else "time_limit",
                               "steps": self.episode_steps, "queue_size": len(self.failures)}
        return observation, reward, terminated, truncated, info


    # Restore curriculum randomness and queued seeds, but restart the Box2D episode.
    def get_curriculum_state(self):
        return {"rng": self.rng.getstate(), "failures": dict(self.failures)}


    def set_curriculum_state(self, state):
        self.rng.setstate(state["rng"])
        self.failures = dict(state["failures"])
        self.restored = True


# Preserve raw environment rewards; normalize observations only inside the learner.
def make_ppo_env(replay_fraction=0.0):
    _, _, Monitor, _, _, _ = ppo_dependencies()
    env = make_env()
    return Monitor(TerrainReplay(env, replay_fraction) if replay_fraction else env)


def normalization_state(env):
    return {"mean": env.obs_rms.mean.tolist(), "var": env.obs_rms.var.tolist(),
            "count": float(env.obs_rms.count), "clip_obs": float(env.clip_obs), "epsilon": float(env.epsilon)}


# Export PPO weights into the portable Agent, so teacher evaluation needs no SB3.
def export_ppo_agent(model, env):
    with torch.random.fork_rng(devices=[]):
        agent = Agent({"hidden_dim": 256, "activation": "tanh", "gamma": float(model.gamma),
                       "std_mode": "sde" if model.use_sde else "diagonal",
                       "observation_normalization": normalization_state(env)})
    policy = model.policy
    for destination, features, head in ((agent.actor.model, policy.mlp_extractor.policy_net, policy.action_net),
                                         (agent.critic.model, policy.mlp_extractor.value_net, policy.value_net)):
        for index in (0, 2):
            destination[index].load_state_dict(cpu_tree(features[index].state_dict()))
        destination[4].load_state_dict(cpu_tree(head.state_dict()))
    with torch.no_grad():
        destination = agent.actor.sde_logstds if model.use_sde else agent.actor.logstds
        destination.copy_(policy.log_std.detach().cpu())
    return agent


# Save JSON through an atomic replacement, including campaign decisions and progress.
def atomic_json(value, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=destination.name, suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


# Hold an operating-system lock; a crashed process releases it automatically.
@contextmanager
def file_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"Another process is using {path}; do not start a second run.") from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


# Keep schedules above a nonzero floor, including the final completed rollout.
def linear_schedule(start, end):
    return lambda remaining: float(end + (start - end) * min(1.0, max(0.0, remaining)))


def restore_normalization(env, stats):
    env.obs_rms.mean = np.asarray(stats["mean"], dtype=np.float64)
    env.obs_rms.var = np.asarray(stats["var"], dtype=np.float64)
    env.obs_rms.count = float(stats["count"])
    env.clip_obs, env.epsilon = float(stats["clip_obs"]), float(stats["epsilon"])


# Import a portable policy without importing the last run's degraded parameters.
def warm_start_ppo(model, env, agent, reset_critic=False):
    if not isinstance(agent, Agent) or agent.config["activation"] != "tanh" or agent.config["hidden_dim"] != 256:
        raise ValueError("PPO warm start requires an exported two-layer 256-unit Tanh Agent.")
    stats = agent.config["observation_normalization"]
    if stats is None:
        raise ValueError("The PPO warm-start model must contain observation statistics.")
    restore_normalization(env, stats)
    pairs = [(agent.actor.model, model.policy.mlp_extractor.policy_net, model.policy.action_net)]
    if not reset_critic:
        if not math.isclose(agent.config["gamma"], model.gamma):
            raise ValueError("A changed gamma requires --reset-critic and a critic warm-up.")
        pairs.append((agent.critic.model, model.policy.mlp_extractor.value_net, model.policy.value_net))
    for source, features, head in pairs:
        for index in (0, 2):
            features[index].load_state_dict(source[index].state_dict())
        head.load_state_dict(source[4].state_dict())
    with torch.no_grad():
        if model.use_sde:
            if agent.actor.sde_logstds is not None:
                model.policy.log_std.copy_(agent.actor.sde_logstds.to(model.device))
            else:
                # Match average marginal noise on fixed normalized probes; do not sample evaluation terrains.
                probes = torch.cat((torch.zeros(1, STATE_DIM), torch.eye(STATE_DIM), -torch.eye(STATE_DIM))).to(model.device)
                energy = model.policy.mlp_extractor.policy_net(probes).square().sum(dim=1).mean().clamp_min(1e-6)
                deviations = agent.actor.logstds.exp().clamp(0.1, 0.4).to(model.device)
                model.policy.log_std.copy_((deviations / energy.sqrt()).log().expand_as(model.policy.log_std))
            model.policy.reset_noise(model.n_envs)
        else:
            if agent.actor.sde_logstds is not None:
                raise ValueError("Continue an SDE policy with --ppo-use-sde; diagonal conversion would discard its noise model.")
            model.policy.log_std.copy_(agent.actor.logstds.to(model.device))


# Freeze only the policy during value recalibration; separate value parameters remain trainable.
def set_ppo_actor_trainable(model, enabled):
    for module in (model.policy.mlp_extractor.policy_net, model.policy.action_net):
        module.requires_grad_(enabled)
    model.policy.log_std.requires_grad_(enabled)


def optimizer_step_count(model):
    counts = [int(state["step"].item()) for state in model.policy.optimizer.state.values() if "step" in state]
    return max(counts, default=0)


def target_passed(result, target=300.0, success=1.0):
    return result["mean_reward"] >= target and result["success_rate"] >= success


# Save complete updates, with the optimizer matching the exported best policy.
def save_ppo_checkpoint(model, env, args, best_state, best_score, confirmed, path, runtime=None):
    buffer = io.BytesIO()
    model.save(buffer)
    atomic_save({"kind": "ppo_training_checkpoint", "format_version": FORMAT_VERSION,
                 "model_zip": torch.from_numpy(np.frombuffer(buffer.getvalue(), dtype=np.uint8).copy()),
                 "normalization": normalization_state(env), "steps": int(model.num_timesteps),
                 "best_agent": best_state, "best_score": best_score, "confirmed": confirmed,
                 "config": vars(args).copy(), "rng": cpu_tree(capture_rng()),
                 "runtime": runtime or {}, "optimizer_steps": optimizer_step_count(model)}, path)


def read_ppo_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("kind") != "ppo_training_checkpoint" or state.get("format_version") != FORMAT_VERSION:
        raise ValueError("Expected a PPO checkpoint, not an A2C checkpoint or exported Agent.")
    return state


# Train in an isolated run directory, or resume that directory under its process lock.
def run_ppo_training(args):
    if args.run_dir:
        with file_lock(Path(args.run_dir) / "run.lock"):
            return _run_ppo_training(args)
    return _run_ppo_training(args)


def _run_ppo_training(args):
    PPO, BaseCallback, _, DummyVecEnv, SubprocVecEnv, VecNormalize = ppo_dependencies()
    class Finished(Exception):
        pass

    resume = read_ppo_checkpoint(args.resume) if args.resume else None
    initial = read_ppo_checkpoint(args.init_checkpoint) if args.init_checkpoint else None
    if resume:
        if args.steps < resume["steps"] or args.num_envs != resume["config"]["num_envs"]:
            raise ValueError("Resume needs a sufficient cumulative budget and the original number of environments.")
        for key in ("seed", "validation_seeds", "confirmation_seeds", "defer_confirmation", "ppo_use_sde",
                    "sde_sample_freq", "terrain_replay_fraction", "freeze_obs_stats", "entropy_start", "entropy_end", "target_success",
                    "ppo_capg", "ppo_gae_lambda", "ppo_learning_rate", "ppo_final_lr", "ppo_clip_range", "ppo_final_clip",
                    "ppo_target_kl", "ppo_epochs", "ppo_eval_steps", "target_reward"):
            if key in resume["config"]:
                setattr(args, key, resume["config"][key])
        if set(args.eval_seeds) & (set(args.validation_seeds) | set(args.confirmation_seeds)):
            raise ValueError("Evaluation seeds overlap the checkpoint's validation or confirmation seeds.")
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        if (run_dir / "config.json").exists() and not resume:
            raise ValueError("Run directory exists. Resume its checkpoint or use a new directory.")
    else:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("%Y%m%d-%H%M%S-ppo-"), dir=args.log_dir)).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    factories = [partial(make_ppo_env, replay_fraction=args.terrain_replay_fraction)] * args.num_envs
    vector = DummyVecEnv(factories) if args.num_envs == 1 else SubprocVecEnv(factories, start_method="spawn")
    env = VecNormalize(vector, norm_obs=True, norm_reward=False, clip_obs=10.0)
    started = time.monotonic()
    try:
        if resume or initial:
            source = resume or initial
            restore_normalization(env, source["normalization"])
            model = PPO.load(io.BytesIO(source["model_zip"].numpy().tobytes()), env=env, device=args.device)
            model.tensorboard_log = str(run_dir)
            if resume:
                args.ppo_capg = resume["config"].get("ppo_capg", False)
                args.ppo_gae_lambda = model.gae_lambda
                args.gamma, args.ppo_rollout_steps, args.ppo_batch_size = model.gamma, model.n_steps, model.batch_size
                restore_rng(resume["rng"])
                if args.terrain_replay_fraction:
                    states = resume.get("runtime", {}).get("terrain_states")
                    if states is None or len(states) != args.num_envs:
                        raise ValueError("Resume requires a curriculum state for each training worker.")
                    for index, state in enumerate(states):
                        env.env_method("set_curriculum_state", state, indices=index)
                for name in ("best-checkpoint.pt", "coverage-checkpoint.pt", "coverage.model"):
                    old = Path(args.resume).parent / name
                    if old.exists() and old.resolve() != (run_dir / name).resolve():
                        shutil.copy2(old, run_dir / name)
            else:
                if not math.isclose(model.gamma, args.gamma) or model.n_steps != args.ppo_rollout_steps or model.use_sde != args.ppo_use_sde:
                    raise ValueError("Full-checkpoint continuation must preserve gamma, rollout length and noise distribution.")
                model.num_timesteps = 0  # A new stage counts new interactions; optimizer moments are retained.
                model._episode_num = 0
                model._last_obs = None
                model.set_random_seed(args.seed)
                model.batch_size = args.ppo_batch_size
                model.learning_rate = linear_schedule(args.ppo_learning_rate, args.ppo_final_lr)
                model._setup_lr_schedule()
                model.clip_range = linear_schedule(args.ppo_clip_range, args.ppo_final_clip)
                model.target_kl = args.ppo_target_kl
                model.n_epochs = args.ppo_epochs
                model.sde_sample_freq = args.sde_sample_freq
                model.gae_lambda = args.ppo_gae_lambda
                model.rollout_buffer.gae_lambda = args.ppo_gae_lambda
        else:
            model = PPO("MlpPolicy", env, device=args.device, seed=args.seed,
                        n_steps=args.ppo_rollout_steps, batch_size=args.ppo_batch_size,
                        learning_rate=linear_schedule(args.ppo_learning_rate, args.ppo_final_lr),
                        n_epochs=args.ppo_epochs, gamma=args.gamma, gae_lambda=args.ppo_gae_lambda,
                        clip_range=linear_schedule(args.ppo_clip_range, args.ppo_final_clip),
                        ent_coef=args.entropy_start, vf_coef=0.5, max_grad_norm=0.5,
                        use_sde=args.ppo_use_sde, sde_sample_freq=args.sde_sample_freq,
                        target_kl=args.ppo_target_kl, normalize_advantage=True,
                        policy_kwargs={"activation_fn": nn.Tanh, "net_arch": {"pi": [256, 256], "vf": [256, 256]}},
                        tensorboard_log=str(run_dir), verbose=1)
            if args.warm_start:
                warm_start_ppo(model, env, torch.load(args.warm_start, map_location="cpu", weights_only=True), args.reset_critic)
        configure_ppo_distribution(model, args.ppo_capg)
        warmup_until = args.critic_warmup_rollouts * model.n_steps * args.num_envs
        if resume:
            warmup_until = resume.get("runtime", {}).get("warmup_until", 0)
        elif initial:
            warmup_until = max(0, initial.get("runtime", {}).get("warmup_until", 0) - initial["steps"])
        best_state = resume["best_agent"] if resume else None
        best_score = resume["best_score"] if resume else -math.inf
        best_result = resume.get("runtime", {}).get("best_validation") if resume else None
        coverage_result = resume.get("runtime", {}).get("coverage_validation") if resume else None
        env.training = not args.freeze_obs_stats and model.num_timesteps >= warmup_until
        confirmed = False
        validation_goal = False
        last_eval, next_eval = -1, int(model.num_timesteps)
        initial_steps = int(model.num_timesteps)
        effective = {"arguments": vars(args), "algorithm": "Stable-Baselines3 PPO 2.4.1",
                     "reward_normalization": False, "observation_normalization": True,
                     "ppo": {"n_steps": model.n_steps, "batch_size": model.batch_size,
                             "gamma": model.gamma, "gae_lambda": model.gae_lambda, "epochs": model.n_epochs,
                             "learning_rate_start": model.lr_schedule(1.0), "learning_rate_end": model.lr_schedule(0.0),
                             "clip_start": model.clip_range(1.0), "clip_end": model.clip_range(0.0),
                             "target_kl": model.target_kl, "warmup_until": warmup_until,
                             "capg": args.ppo_capg,
                             "use_sde": model.use_sde, "sde_sample_freq": model.sde_sample_freq,
                             "entropy_start": args.entropy_start, "entropy_end": args.entropy_end,
                             "freeze_obs_stats": args.freeze_obs_stats, "terrain_replay_fraction": args.terrain_replay_fraction}}
        atomic_json(effective, run_dir / "config.json")
        print(f"Training directory: {run_dir}\nPPO cumulative stage budget: {args.steps}; rollout size: {model.n_steps * args.num_envs}.", flush=True)

        # Validate only on development seeds during a campaign; holdout seeds stay unread.
        def boundary(force=False):
            nonlocal best_state, best_score, best_result, coverage_result, confirmed, last_eval, next_eval, validation_goal
            improved = coverage_improved = False
            if model.num_timesteps >= next_eval or (force and last_eval != model.num_timesteps):
                candidate = export_ppo_agent(model, env)
                result = evaluate_agent(candidate, args.validation_seeds)
                result.update(steps=int(model.num_timesteps), phase="validation")
                validation_goal = target_passed(result, args.target_reward, args.target_success)
                confirmation = None
                if validation_goal and not args.defer_confirmation:
                    confirmation = evaluate_agent(candidate, args.confirmation_seeds)
                    confirmation.update(steps=int(model.num_timesteps), phase="confirmation")
                    confirmed = target_passed(confirmation, args.target_reward, args.target_success)
                if best_state is None or result["mean_reward"] > best_score or confirmed:
                    best_state, best_score, best_result = candidate.__getstate__(), result["mean_reward"], result
                    improved = True
                # Keep a separate finish-rate leader within 25 raw points of the return leader.
                floor = best_score - 25.0
                previous_goal = coverage_result is not None and target_passed(coverage_result, args.target_reward, args.target_success)
                if (validation_goal or result["mean_reward"] >= floor) and (coverage_result is None
                        or (not previous_goal and coverage_result["mean_reward"] < floor)
                        or (validation_goal, result["success_rate"], result["mean_reward"]) >
                           (previous_goal, coverage_result["success_rate"], coverage_result["mean_reward"])):
                    coverage_result = result
                    coverage_improved = True
                with (run_dir / "evaluations.jsonl").open("a", encoding="utf-8") as output:
                    for row in (result, confirmation):
                        if row is not None:
                            output.write(json.dumps(row, allow_nan=False) + "\n")
                print(f"PPO validation at {model.num_timesteps}: mean={result['mean_reward']:.3f}, success={result['success_rate']:.2f}", flush=True)
                if hasattr(model, "_logger"):
                    model.logger.record("evaluation/mean_reward", result["mean_reward"])
                    model.logger.record("evaluation/success_rate", result["success_rate"])
                    model.logger.record("train/optimizer_steps_actual", optimizer_step_count(model))
                    model.logger.dump(step=model.num_timesteps)
                last_eval, next_eval = model.num_timesteps, model.num_timesteps + args.ppo_eval_steps
            runtime = {"warmup_until": warmup_until, "best_validation": best_result, "coverage_validation": coverage_result,
                       "terrain_states": env.env_method("get_curriculum_state") if args.terrain_replay_fraction else None}
            if best_state is not None:
                for path in {run_dir / "a2c.model", Path(args.model_path).resolve()}:
                    atomic_save(agent_from_state(best_state), path)
            if improved:
                save_ppo_checkpoint(model, env, args, best_state, best_score, confirmed, run_dir / "best-checkpoint.pt", runtime)
            if coverage_improved:
                atomic_save(candidate, run_dir / "coverage.model")
                coverage_runtime = {**runtime, "best_validation": coverage_result}
                save_ppo_checkpoint(model, env, args, candidate.__getstate__(), coverage_result["mean_reward"],
                                    confirmed, run_dir / "coverage-checkpoint.pt", coverage_runtime)
            save_ppo_checkpoint(model, env, args, best_state, best_score, confirmed, run_dir / "checkpoint.pt", runtime)
            atomic_json({"steps": int(model.num_timesteps), "optimizer_steps": optimizer_step_count(model),
                         "epochs_attempted": int(model._n_updates), "best_reward": best_score,
                         "pid": os.getpid(), "status": "running"}, run_dir / "progress.json")

        class SaveAtBoundary(BaseCallback):
            def _on_rollout_start(self):
                boundary()
                if stop.requested or (args.stop_file and Path(args.stop_file).exists()) or confirmed:
                    raise Finished
                set_ppo_actor_trainable(model, model.num_timesteps >= warmup_until)
                env.training = not args.freeze_obs_stats and model.num_timesteps >= warmup_until
                model.ent_coef = linear_schedule(args.entropy_start, args.entropy_end)(1 - model.num_timesteps / args.steps)

            def _on_step(self):
                for info in self.locals.get("infos", []):
                    if "episode" in info:
                        with (run_dir / "episodes.jsonl").open("a", encoding="utf-8") as output:
                            output.write(json.dumps({"steps": int(model.num_timesteps), **info["episode"],
                                                     "terrain": info.get("terrain")}) + "\n")
                return True

        with StopRequest() as stop:
            try:
                if args.steps > model.num_timesteps:
                    model.learn(total_timesteps=args.steps - model.num_timesteps,
                                reset_num_timesteps=False, callback=SaveAtBoundary(), tb_log_name="ppo")
            except Finished:
                pass
            boundary(force=True)
        interrupted = stop.requested or bool(args.stop_file and Path(args.stop_file).exists())
        result = {"status": "interrupted" if interrupted else "completed", "steps": int(model.num_timesteps),
                  "new_steps_this_process": int(model.num_timesteps) - initial_steps,
                  "optimizer_steps": optimizer_step_count(model), "epochs_attempted": int(model._n_updates),
                  "best_validation": best_result, "coverage_validation": coverage_result,
                  "wall_seconds_this_process": time.monotonic() - started}
        atomic_json(result, run_dir / "stage-result.json")
        if not args.defer_confirmation:
            evaluation = evaluate_agent(agent_from_state(best_state), args.eval_seeds)
            evaluation.update(steps=int(model.num_timesteps), validation_and_confirmation_passed=confirmed,
                              target_reward=args.target_reward, example_set_passed=target_passed(evaluation, args.target_reward))
            atomic_json(evaluation, run_dir / "final-evaluation.json")
            print(json.dumps(evaluation, indent=2), flush=True)
        print(f"Saved {run_dir / 'checkpoint.pt'} and best-checkpoint.pt. Status: {result['status']}", flush=True)
        return result
    finally:
        env.close()
        if "model" in locals() and hasattr(model, "_logger"):
            for output in model.logger.output_formats:
                output.close()


# Identify the exact source and baseline used by every campaign stage.
def file_sha256(path, normalize_newlines=False):
    data = Path(path).read_bytes()
    if normalize_newlines:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def campaign_spec(args):
    rollout = args.num_envs * args.ppo_rollout_steps
    return {"version": 3, "source_lf_sha256": file_sha256(__file__, True),
            "baseline_sha256": BASELINE_SHA256, "baseline_checkpoint_sha256": BASELINE_CHECKPOINT_SHA256,
            "budget": args.campaign_budget // rollout * rollout,
            "pilot_steps": args.pilot_steps // rollout * rollout, "chunk_steps": args.chunk_steps // rollout * rollout,
            "rollout": rollout, "num_envs": args.num_envs, "device": args.device,
            "torch_threads": args.torch_threads, "seed": args.seed, "replication_seed": args.seed + 1000,
            "validation_seeds": CAMPAIGN_VALIDATION_SEEDS, "selection_seeds": CAMPAIGN_SELECTION_SEEDS,
            "confirmation_seeds": list(args.confirmation_seeds), "example_seeds": list(args.eval_seeds),
            "target": args.target_reward, "target_success": args.target_success, "gamma": 0.999,
            "profiles": ["gaussian", "capg"], "learning_rate": [2e-5, 1e-5], "clip_range": [0.1, 0.1], "gae_lambda": .98,
            "batch_size": 256, "terrain_replay_fraction": 0.25, "sde_sample_freq": 4,
            "freeze_obs_stats": True, "coverage_entropy": [0.0, 0.0], "efficiency_entropy": [0.0, 0.0],
            "continuation_reward_tolerance": 25.0, "final_reward_tolerance": 0.0,
            "plateau_chunks": 3, "meaningful_reward_gain": 5.0, "meaningful_success_gain": .05}


# Fetch only the user's pinned release artifact, verifying its digest before loading it.
def campaign_baseline(root, supplied=None, checkpoint=False):
    name = "final-best-checkpoint.pt" if checkpoint else "a2c.model"
    digest = BASELINE_CHECKPOINT_SHA256 if checkpoint else BASELINE_SHA256
    url = BASELINE_URL.rsplit("/", 1)[0] + "/" + name if checkpoint else BASELINE_URL
    path = Path(root) / "baseline" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        if supplied:
            if file_sha256(supplied) != digest:
                raise ValueError("Baseline does not match the pinned Issue #4 release artifact.")
            shutil.copy2(supplied, path)
        else:
            temporary = path.with_suffix(".download")
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url, timeout=30) as response, temporary.open("wb") as output:
                        shutil.copyfileobj(response, output)
                    if file_sha256(temporary) != digest:
                        raise ValueError("Downloaded baseline SHA-256 mismatch.")
                    os.replace(temporary, path)
                    break
                except OSError:
                    temporary.unlink(missing_ok=True)
                    if attempt == 2:
                        raise
    if file_sha256(path) != digest:
        raise ValueError("Cached baseline SHA-256 mismatch; preserve it and inspect the file.")
    return path.resolve()


# Compare common development seeds; all scores remain original environment returns.
def evaluation_summary(result):
    diagnostics = result.get("diagnostics", [])
    progress = [row["progress_fraction"] for row in diagnostics]
    stalls = [row.get("stall_fraction", 0.0) for row in diagnostics]
    return {"mean_reward": float(result["mean_reward"]), "success_rate": float(result["success_rate"]),
            "mean_progress": float(np.mean(progress)) if progress else 0.0,
            "median_progress": float(np.median(progress)) if progress else 0.0,
            "stall_fraction": float(np.mean(stalls)) if stalls else 0.0,
            "mean_motor_cost": float(np.mean([-row.get("reward_terms", {}).get("motor", 0.) for row in diagnostics])) if diagnostics else 0.0,
            "timeout_rate": float(np.mean([row.get("end_reason") == "time_limit" for row in diagnostics])) if diagnostics else 0.0,
            "fall_rate": float(np.mean([row.get("end_reason") in ("fall", "left_boundary") for row in diagnostics])) if diagnostics else 0.0,
            "action_clip_fraction": float(np.mean([row.get("action_clip_fraction", 0.) for row in diagnostics])) if diagnostics else 0.0,
            "finish_mean_reward": float(np.mean([reward for reward, row in zip(result["rewards"], diagnostics) if row["success"]]))
                if any(row.get("success") for row in diagnostics) else None}


def development_score(candidate):
    validation, selection = candidate["validation"], candidate["score"]
    nv, ns = len(validation.get("seeds", CAMPAIGN_VALIDATION_SEEDS)), candidate.get("selection_count", len(CAMPAIGN_SELECTION_SEEDS))
    return {key: (nv * validation[key] + ns * selection[key]) / (nv + ns) for key in ("mean_reward", "success_rate")}


def development_passed(candidate, spec):
    return all(target_passed(candidate[key], spec["target"], spec["target_success"]) for key in ("score", "validation"))


# Retain policies that trade a small return reduction for more completed terrains.
def pareto_candidates(candidates):
    frontier = []
    for candidate in candidates:
        score = development_score(candidate)
        if any(all(development_score(other)[key] >= score[key] for key in score)
               and any(development_score(other)[key] > score[key] for key in score) for other in candidates):
            continue
        frontier.append(candidate)
    return frontier


def candidate_key(candidate, coverage=False):
    score = development_score(candidate)
    values = (score["success_rate"], score["mean_reward"]) if coverage else (score["mean_reward"], score["success_rate"])
    return (*values, candidate["score"].get("median_progress", 0.0))


def choose_candidate(candidates, baseline, spec, coverage=False, final=False):
    tolerance = spec["final_reward_tolerance"] if final else spec["continuation_reward_tolerance"]
    floor = development_score(baseline)["mean_reward"] - tolerance
    eligible = [c for c in candidates if development_score(c)["mean_reward"] >= floor]
    if not eligible:
        return None
    sufficient = [c for c in eligible if all(c[key]["success_rate"] >= spec["target_success"] for key in ("score", "validation"))]
    if coverage and sufficient:
        eligible, coverage = sufficient, False  # Once coverage is sufficient, maximize original return.
    return max(eligible, key=lambda c: (development_passed(c, spec), *candidate_key(c, coverage)))


def campaign_spent(manifest):
    return sum(stage.get("steps", 0) + stage.get("lost_steps_upper_bound", 0) for stage in manifest["stages"].values())


# Run one child under the current interpreter; stdout and stderr remain in its stage log.
def execute_campaign_stage(command, stage_dir, stop, stop_file):
    stage_dir = Path(stage_dir)
    with (stage_dir / "console.log").open("a", encoding="utf-8") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        atomic_json({"pid": process.pid, "started": datetime.now().isoformat(), "command": command}, stage_dir / "process.json")
        last_report = 0.0
        while process.poll() is None:
            if stop.requested:
                Path(stop_file).touch()
            if time.monotonic() - last_report >= 30:
                progress = stage_dir / "progress.json"
                if progress.exists():
                    print(f"{stage_dir.name}: {progress.read_text(encoding='utf-8')}", flush=True)
                last_report = time.monotonic()
            time.sleep(0.5)
        return process.returncode


# Keep Windows awake during the campaign without changing its permanent power settings.
@contextmanager
def campaign_awake():
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        yield
    finally:
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


# Persist stage intentions before launch, then resume only their own completed-update checkpoint.
class Campaign:
    def __init__(self, root, spec, baseline_path, manifest, stop):
        self.root, self.spec = Path(root), spec
        self.baseline_path, self.manifest, self.stop = Path(baseline_path), manifest, stop


    def save(self):
        self.manifest["charged_steps_upper_bound"] = campaign_spent(self.manifest)
        atomic_json(self.manifest, self.root / "campaign.json")


    def assess(self, model_path, output_path, seeds=None):
        output_path = Path(output_path)
        seeds = self.spec["selection_seeds"] if seeds is None else list(seeds)
        digest = file_sha256(model_path)
        if output_path.exists():
            saved = json.loads(output_path.read_text(encoding="utf-8"))
            if saved["model_sha256"] != digest or saved["evaluation"]["seeds"] != seeds:
                raise ValueError("A model changed after its selection evaluation.")
            return saved["evaluation"]
        agent = torch.load(model_path, map_location="cpu", weights_only=True)
        result = evaluate_agent(agent, seeds)
        atomic_json({"model_sha256": digest, "evaluation": result}, output_path)
        return result


    def stage(self, name, gamma, seed, requested_steps, parent=None, profile="gaussian"):
        spec = self.spec
        directory = self.root / "stages" / name
        existing = self.manifest["stages"].get(name)
        if existing and existing["status"] == "completed":
            return existing["candidate"]
        if self.stop.requested or (self.root / "STOP").exists():
            raise InterruptedError("Campaign stop requested.")
        if existing is None:
            allocation = min(requested_steps, spec["budget"] - campaign_spent(self.manifest))
            allocation = allocation // spec["rollout"] * spec["rollout"]
            if allocation < spec["rollout"]:
                return None
            existing = {"name": name, "gamma": gamma, "seed": seed, "allocation": allocation, "profile": profile,
                        "steps": 0, "lost_steps_upper_bound": 0, "status": "pending", "phase": "efficiency",
                        "parent_checkpoint": parent["checkpoint"] if parent else
                            str(self.baseline_path.with_name("final-best-checkpoint.pt")),
                        "lr_start": spec["learning_rate"][0], "lr_end": spec["learning_rate"][1],
                        "clip_start": 0.1, "clip_end": 0.1,
                        "entropy_start": 0.0, "entropy_end": 0.0}
            if parent:
                # Continuation never restarts at a larger learning rate after selecting an earlier best.
                previous = read_ppo_checkpoint(parent["checkpoint"])
                phase = min(1.0, previous["steps"] / max(1, previous["config"]["steps"]))
                config = previous["config"]
                previous_lr = max(config["ppo_final_lr"], config["ppo_learning_rate"] + phase * (config["ppo_final_lr"] - config["ppo_learning_rate"]))
                existing["lr_end"] = min(existing["lr_end"], previous_lr)
                existing["lr_start"] = previous_lr
                old_entropy = linear_schedule(config.get("entropy_start", .001), config.get("entropy_end", .001))(1 - phase)
                existing["entropy_start"] = min(existing["entropy_start"], old_entropy)
                existing["entropy_end"] = min(existing["entropy_end"], existing["entropy_start"])
            self.manifest["stages"][name] = existing
            self.save()
        directory.mkdir(parents=True, exist_ok=True)
        # A surviving child may outlive its parent; its own lock prevents duplicate training.
        with file_lock(directory / "run.lock"):
            checkpoint_path = directory / "checkpoint.pt"
            result_path = directory / "stage-result.json"
            finished = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else None
            if finished and finished["status"] == "completed":
                existing["steps"] = finished["steps"]
            else:
                checkpoint = read_ppo_checkpoint(checkpoint_path) if checkpoint_path.exists() else None
                existing["steps"] = checkpoint["steps"] if checkpoint else 0
                if existing["status"] == "running" and not finished:
                    # A hard crash may discard at most the in-flight rollout beyond this checkpoint.
                    remaining = existing["allocation"] - existing["steps"] - existing["lost_steps_upper_bound"]
                    existing["lost_steps_upper_bound"] += min(spec["rollout"], max(0, remaining))
                existing["status"] = "pending"
                self.save()
        if not finished or finished["status"] != "completed":
            goal = existing["allocation"] - existing["lost_steps_upper_bound"]
            if goal == 0 and not checkpoint_path.exists():
                raise RuntimeError("No complete checkpoint survived this stage. Start a new campaign; budget was reserved conservatively.")
            if checkpoint_path.exists() and existing["steps"] == goal:
                checkpoint = read_ppo_checkpoint(checkpoint_path)
                finished = {"status": "completed", "steps": goal,
                            "best_validation": checkpoint["runtime"]["best_validation"],
                            "coverage_validation": checkpoint["runtime"].get("coverage_validation")}
                atomic_json(finished, result_path)
        if not finished or finished["status"] != "completed":
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--mode", "train", "--algorithm", "ppo",
                       "--device", spec["device"], "--num-envs", str(spec["num_envs"]), "--torch-threads", str(spec["torch_threads"]),
                       "--run-dir", str(directory), "--model-path", str(directory / "a2c.model"), "--steps", str(goal),
                       "--seed", str(seed), "--gamma", str(gamma), "--ppo-rollout-steps", str(spec["rollout"] // spec["num_envs"]),
                       "--ppo-batch-size", str(spec["batch_size"]), "--ppo-eval-steps", str(max(spec["rollout"], 250000)),
                       "--ppo-learning-rate", str(existing["lr_start"]), "--ppo-final-lr", str(existing["lr_end"]),
                       "--ppo-clip-range", str(existing["clip_start"]), "--ppo-final-clip", str(existing["clip_end"]),
                       "--ppo-target-kl", "0.02", "--defer-confirmation", "--target-reward", str(spec["target"]),
                       "--ppo-gae-lambda", str(spec["gae_lambda"]),
                       "--target-success", str(spec["target_success"]), "--freeze-obs-stats",
                       "--terrain-replay-fraction", str(spec["terrain_replay_fraction"]),
                       "--entropy-start", str(existing["entropy_start"]), "--entropy-end", str(existing["entropy_end"]),
                       "--sde-sample-freq", str(spec["sde_sample_freq"]),
                       "--stop-file", str(self.root / "STOP"), "--validation-seeds", *map(str, spec["validation_seeds"]),
                       "--confirmation-seeds", *map(str, spec["confirmation_seeds"]), "--eval-seeds", *map(str, spec["example_seeds"])]
            if profile == "capg":
                command += ["--ppo-capg"]
            if checkpoint_path.exists():
                command += ["--resume", str(checkpoint_path)]
            elif existing["parent_checkpoint"]:
                command += ["--init-checkpoint", existing["parent_checkpoint"]]
            else:
                command += ["--warm-start", str(self.baseline_path)]
            existing["status"] = "running"
            existing["command"] = command
            self.save()
            # Remove an old interruption result only after it has been accounted for.
            result_path.unlink(missing_ok=True)
            print(f"Starting {name}: gamma={gamma}, seed={seed}, stage budget={goal}.", flush=True)
            code = execute_campaign_stage(command, directory, self.stop, self.root / "STOP")
            if not result_path.exists():
                raise RuntimeError(f"Stage {name} exited {code} without a complete result. See {directory / 'console.log'}.")
            finished = json.loads(result_path.read_text(encoding="utf-8"))
            existing["steps"] = finished["steps"]
            if finished["status"] == "interrupted" or code == 130:
                existing["status"] = "interrupted"
                self.save()
                raise InterruptedError("Stage saved at a complete update boundary.")
            if code != 0:
                raise RuntimeError(f"Stage {name} failed with exit code {code}.")
        if existing["steps"] + existing["lost_steps_upper_bound"] > existing["allocation"]:
            raise RuntimeError("Stage exceeded its reserved budget.")
        candidates = []
        for role, model_name, checkpoint_name, validation in (
                ("return", "a2c.model", "best-checkpoint.pt", finished["best_validation"]),
                ("coverage", "coverage.model", "coverage-checkpoint.pt", finished.get("coverage_validation"))):
            if validation is None:
                raise RuntimeError("Stage is missing its independently saved coverage leader.")
            result = self.assess(directory / model_name, directory / f"selection-{role}.json")
            candidates.append({"name": name, "role": role, "profile": profile, "gamma": gamma, "seed": seed,
                               "target": spec["target"], "score": evaluation_summary(result), "selection_count": len(result["seeds"]),
                               "model": str(directory / model_name), "checkpoint": str(directory / checkpoint_name), "validation": validation})
        candidate = max(candidates, key=candidate_key)
        existing.update(status="completed", candidate=candidate, candidates=candidates)
        self.save()
        return candidate


# Compare both training seeds before assigning additional budget to one method.
def rank_profiles(campaign, baseline):
    summaries = {}
    floor = development_score(baseline)["mean_reward"] - campaign.spec["continuation_reward_tolerance"]
    for profile in campaign.spec["profiles"]:
        seeds = []
        for prefix in ("pilot", "replicate"):
            stage = campaign.manifest["stages"][f"{prefix}-{profile}"]
            candidate = max(stage["candidates"], key=candidate_key)
            seeds.append(development_score(candidate))
        summaries[profile] = {"acceptable_seeds": sum(row["mean_reward"] >= floor for row in seeds),
                              "mean_reward": float(np.mean([row["mean_reward"] for row in seeds])),
                              "success_rate": float(np.mean([row["success_rate"] for row in seeds])), "seeds": seeds}
    winner = max(summaries, key=lambda p: (summaries[p]["acceptable_seeds"], summaries[p]["mean_reward"], summaries[p]["success_rate"]))
    return winner, summaries


# Stop funding a losing method after replication; continue both reward/coverage leaders of the winner.
def campaign_search(campaign, baseline):
    spec = campaign.spec
    candidates = [baseline]

    def collect(name, profile, seed, steps, parent=None):
        result = campaign.stage(name, spec["gamma"], seed, steps, parent=parent, profile=profile)
        if result is not None:
            candidates.extend(campaign.manifest["stages"][name]["candidates"])
        frontier = pareto_candidates(candidates)
        campaign.manifest["frontier"] = [{"name": c["name"], "role": c.get("role"), "profile": c["profile"],
                                           **development_score(c)} for c in frontier]
        campaign.save()
        return result

    for prefix, seed in (("pilot", spec["seed"]), ("replicate", spec["replication_seed"])):
        for profile in spec["profiles"]:
            collect(f"{prefix}-{profile}", profile, seed, spec["pilot_steps"])
    profile, summaries = rank_profiles(campaign, baseline)
    campaign.manifest["allocation_decision"] = {"profile": profile, "replicated_results": summaries}
    campaign.save()
    champion = choose_candidate(candidates, baseline, spec, coverage=True, final=True)
    if summaries[profile]["acceptable_seeds"] < 2:
        return champion, "no_replicated_admissible_profile"
    index = plateau = 0
    def progress_marks():
        eligible = [c for c in candidates if development_score(c)["mean_reward"] >=
                    development_score(baseline)["mean_reward"] - spec["continuation_reward_tolerance"]]
        return {key: max(development_score(c)[key] for c in eligible) for key in ("mean_reward", "success_rate")}
    marks = progress_marks()
    while True:
        champion = choose_candidate(candidates, baseline, spec, coverage=True, final=True)
        if development_passed(champion, spec):
            return champion, "development_target_reached"
        if plateau >= spec["plateau_chunks"]:
            return champion, "development_plateau"
        index += 1
        coverage = index % 2 == 0
        pool = pareto_candidates([c for c in candidates if c["profile"] == profile])
        parent = choose_candidate(pool, baseline, spec, coverage=coverage)
        name = f"continue-{index:03d}-{profile}-{'coverage' if coverage else 'return'}"
        result = collect(name, profile, spec["seed"] + 2000 + index, spec["chunk_steps"], parent=parent)
        if result is None:
            return champion, "budget_exhausted"
        current = progress_marks()
        improved = (current["mean_reward"] >= marks["mean_reward"] + spec["meaningful_reward_gain"] or
                    current["success_rate"] >= marks["success_rate"] + spec["meaningful_success_gain"] - 1e-9)
        plateau = 0 if improved else plateau + 1
        if improved:
            marks = current


# Produce real curves from each stage's original rewards, with no synthesized training data.
def campaign_plots(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for directory in sorted((Path(root) / "stages").glob("*")):
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), constrained_layout=True)
        for filename, axis, field in (("episodes.jsonl", axes[0], "r"), ("evaluations.jsonl", axes[1], "mean_reward")):
            path = directory / filename
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
            values = np.asarray([row[field] for row in rows], dtype=float)
            if len(values):
                axis.plot([row["steps"] for row in rows], values, alpha=0.35, linewidth=0.6)
                window = min(100, len(values)) if field == "r" else 1
                smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
                axis.plot([row["steps"] for row in rows][window - 1:], smoothed, linewidth=1)
            axis.set_ylabel("Raw episode return" if field == "r" else "Validation mean")
            axis.set_xlabel("Stage environment steps")
            axis.grid(alpha=0.2)
        fig.suptitle(directory.name)
        fig.savefig(directory / "reward-curves.png", dpi=140)
        plt.close(fig)


# Freeze the final choice before opening holdout results; a failed holdout never selects another model.
def finish_campaign(campaign, champion, reason):
    root, spec, manifest = campaign.root, campaign.spec, campaign.manifest
    locked_path = root / "final-choice.json"
    choice = {"candidate": champion, "model_sha256": file_sha256(champion["model"]), "reason": reason}
    if locked_path.exists():
        if json.loads(locked_path.read_text(encoding="utf-8")) != choice:
            raise ValueError("The final choice was already frozen; holdout results must not select another candidate.")
    else:
        atomic_json(choice, locked_path)
    shutil.copy2(champion["model"], root / "a2c.model")
    shutil.copy2(champion["checkpoint"], root / "final-best-checkpoint.pt")
    agent = torch.load(root / "a2c.model", map_location="cpu", weights_only=True)
    results = {}
    for name, seeds in (("confirmation", spec["confirmation_seeds"]), ("example", spec["example_seeds"])):
        path = root / f"{name}-evaluation.json"
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
        else:
            result = evaluate_agent(agent, seeds)
            result["model_sha256"] = choice["model_sha256"]
            atomic_json(result, path)
        if result["model_sha256"] != choice["model_sha256"] or result["seeds"] != seeds:
            raise ValueError("Final evaluation provenance does not match the frozen model and seeds.")
        results[name] = result
    with tempfile.TemporaryDirectory() as directory:
        check_fresh_process(agent, directory)
    # The old five example seeds are diagnostic; the 100 new terrains form the final test.
    assessed = [champion["score"], champion["validation"], results["confirmation"]]
    coverage_passed = all(result["success_rate"] >= spec["target_success"] for result in assessed)
    reward_passed = all(result["mean_reward"] >= spec["target"] for result in assessed)
    passed = coverage_passed and reward_passed
    summary = {"goal_reached": passed, "target_reward": spec["target"], "stop_reason": reason,
               "target_success_rate": spec["target_success"], "coverage_goal_reached": coverage_passed,
               "reward_goal_reached": reward_passed, "strict_300_all_finished": all(target_passed(r, spec["target"]) for r in assessed),
               "selected_candidate": champion, "charged_steps_upper_bound": campaign_spent(manifest),
               "completed_update_steps": sum(s["steps"] for s in manifest["stages"].values()),
               "lost_steps_upper_bound": sum(s["lost_steps_upper_bound"] for s in manifest["stages"].values()),
               "evaluations": results, "teacher_interface_verified": True}
    atomic_json(summary, root / "final-evaluation.json")
    submission = root / "submission"
    submission.mkdir(exist_ok=True)
    shutil.copy2(__file__, submission / "train.py")
    shutil.copy2(root / "a2c.model", submission / "a2c.model")
    with zipfile.ZipFile(root / "submission.zip", "w", zipfile.ZIP_DEFLATED) as output:
        for name in ("train.py", "a2c.model"):
            output.write(submission / name, name)
    campaign_plots(root)
    dependencies = subprocess.run([sys.executable, "-m", "pip", "freeze"], text=True, capture_output=True, check=True)
    (root / "dependencies.txt").write_text(dependencies.stdout, encoding="utf-8")
    lines = ["# Campaign results", "", f"Goal reached on specified seeds: **{passed}**", f"Stop reason: {reason}",
             f"Coverage goal (at least {spec['target_success']:.0%} in each group): {coverage_passed}",
             f"Raw-reward goal (at least {spec['target']} in each group): {reward_passed}",
             f"Strict 300 and all finished: {summary['strict_300_all_finished']}",
             f"Selected candidate: {champion['name']} / {champion['role']}", f"Source LF SHA-256: {spec['source_lf_sha256']}",
             f"Model SHA-256: {choice['model_sha256']}", f"Completed-update steps: {summary['completed_update_steps']}",
             f"Additional discarded-step upper bound: {summary['lost_steps_upper_bound']}",
             f"Charged budget: {summary['charged_steps_upper_bound']} / {spec['budget']}", "",
             "| Stage | Profile | Role | Seed | Stage steps | Selection mean | Finish rate | Motor cost |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for stage in manifest["stages"].values():
        for candidate in stage["candidates"]:
            score = candidate["score"]
            lines.append(f"| {stage['name']} | {candidate['profile']} | {candidate['role']} | {stage['seed']} | {stage['steps']} | {score['mean_reward']:.3f} | {score['success_rate']:.3f} | {score['mean_motor_cost']:.3f} |")
    for name, result in results.items():
        lines += ["", f"{name}: mean={result['mean_reward']:.3f}; success={result['success_rate']:.3f}",
                  f"Seeds: {result['seeds']}", f"Rewards: {result['rewards']}", f"Lengths: {result['lengths']}"]
    lines += ["", "Raw rewards; Hardcore; 1600-step limit. No score guarantee on unseen teacher seeds.",
              "PPO/GAE and observation normalization extend the course A2C example; references are in README.md.",
              "Training optimizer updates occurred on this Windows machine; local development checks did not train.",
              "Each stage contains its own latest and best complete checkpoint, console log, evaluations and reward curve."]
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest.update(status="completed", goal_reached=passed, stop_reason=reason,
                    selected_candidate=champion, finished=datetime.now().isoformat())
    campaign.save()
    with zipfile.ZipFile(root / "diagnostics.zip", "w", zipfile.ZIP_DEFLATED) as output:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix not in {".pt", ".model", ".zip", ".lock", ".tmp"} and "submission" not in path.parts:
                output.write(path, str(path.relative_to(root)))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Campaign completed. Goal reached: {passed}. Deliverables: {root}", flush=True)
    return summary


# One command owns pilots, replication, bounded continuation, final evaluation and packaging.
def run_campaign(args):
    spec = campaign_spec(args)
    if args.dry_run:
        print(json.dumps(spec, indent=2))
        return spec
    root = Path(args.campaign_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with file_lock(root / "campaign.lock"), campaign_awake(), StopRequest() as stop:
        path = root / "campaign.json"
        manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
            "spec": spec, "status": "created", "stages": {}, "started": datetime.now().isoformat(),
            "platform": platform.platform(), "python": platform.python_version(), "torch": str(torch.__version__),
            "gymnasium": gym.__version__, "numpy": np.__version__}
        if manifest["spec"] != spec:
            raise ValueError("Campaign source or protocol changed. Restore its exact source/options or use a new campaign directory.")
        required = ["final-evaluation.json", "submission.zip", "diagnostics.zip", "RESULTS.md", "dependencies.txt", "a2c.model", "final-best-checkpoint.pt"]
        if manifest["status"] == "completed" and all((root / name).is_file() for name in required):
            result = json.loads((root / "final-evaluation.json").read_text(encoding="utf-8"))
            print(json.dumps(result, indent=2))
            return result
        # Reissuing the same command explicitly resumes a previously requested stop.
        (root / "STOP").unlink(missing_ok=True)
        baseline_path = campaign_baseline(root, args.campaign_baseline)
        baseline_checkpoint = campaign_baseline(root, args.campaign_checkpoint, checkpoint=True)
        checkpoint_agent = agent_from_state(read_ppo_checkpoint(baseline_checkpoint)["best_agent"])
        assert_weights_equal(checkpoint_agent.__getstate__(), torch.load(baseline_path, map_location="cpu", weights_only=True).__getstate__())
        campaign = Campaign(root, spec, baseline_path, manifest, stop)
        manifest["status"] = "running"
        campaign.save()
        try:
            selection = campaign.assess(baseline_path, root / "baseline-selection.json")
            validation = campaign.assess(baseline_path, root / "baseline-validation.json", spec["validation_seeds"])
            baseline = {"name": "release-baseline", "role": "return", "profile": "gaussian", "gamma": .999,
                        "seed": None, "target": spec["target"], "model": str(baseline_path),
                        "checkpoint": str(baseline_checkpoint), "score": evaluation_summary(selection),
                        "selection_count": len(selection["seeds"]), "validation": validation}
            if (root / "final-choice.json").exists():
                frozen = json.loads((root / "final-choice.json").read_text(encoding="utf-8"))
                champion, reason = frozen["candidate"], frozen["reason"]
            else:
                champion, reason = campaign_search(campaign, baseline)
            if stop.requested:
                raise InterruptedError("Campaign interrupted before final evaluation.")
            return finish_campaign(campaign, champion, reason)
        except InterruptedError:
            manifest["status"] = "interrupted"
            campaign.save()
            print(f"Campaign interrupted safely. Resume with the same command: {root}", flush=True)
            return {"status": "interrupted"}
        except Exception as error:
            manifest.update(status="failed", error=repr(error))
            campaign.save()
            raise


# Verify PPO export and resume without ever calling learn or an optimizer step.
def check_ppo(args):
    import contextlib
    from unittest.mock import patch
    PPO, _, _, DummyVecEnv, SubprocVecEnv, VecNormalize = ppo_dependencies()
    factories = [partial(make_ppo_env, replay_fraction=.25)] * args.num_envs
    vector = DummyVecEnv(factories) if args.num_envs == 1 else SubprocVecEnv(factories, start_method="spawn")
    env = VecNormalize(vector, norm_obs=True, norm_reward=False)
    try:
        with patch.object(torch.optim.Adam, "step", side_effect=AssertionError("No optimizer steps in checks")):
            model = PPO("MlpPolicy", env, seed=args.seed, n_steps=8, batch_size=8, device=args.device,
                        policy_kwargs={"activation_fn": nn.Tanh, "net_arch": {"pi": [256, 256], "vf": [256, 256]}})
            observations = env.reset()
            for _ in range(4):
                action, _ = model.predict(observations, deterministic=True)
                observations, _, _, _ = env.step(action)
            terrain_states = env.env_method("get_curriculum_state")
            assert len(terrain_states) == args.num_envs
            for index, state in enumerate(terrain_states):
                env.env_method("set_curriculum_state", state, indices=index)
            exported = export_ppo_agent(model, env)
            state = env.get_original_obs()[0]
            expected, _ = model.predict(env.normalize_obs(state[None, :]), deterministic=True)
            np.testing.assert_allclose(exported.select_action(state, eval=True), expected[0], atol=1e-6)
            # A longer discount horizon gets a fresh critic while retaining exactly the same initial policy.
            warm = PPO("MlpPolicy", env, seed=args.seed + 1, gamma=0.999, n_steps=8, batch_size=8, device=args.device,
                       policy_kwargs={"activation_fn": nn.Tanh, "net_arch": {"pi": [256, 256], "vf": [256, 256]}})
            warm_start_ppo(warm, env, exported, reset_critic=True)
            warm_action, _ = warm.predict(env.normalize_obs(state[None, :]), deterministic=True)
            np.testing.assert_array_equal(warm_action, expected)
            assert not torch.equal(warm.policy.value_net.weight, model.policy.value_net.weight)
            set_ppo_actor_trainable(warm, False)
            inputs = torch.as_tensor(env.normalize_obs(state[None, :]), device=warm.device)
            warm.policy.predict_values(inputs).sum().backward()
            assert warm.policy.action_net.weight.grad is None and warm.policy.log_std.grad is None
            assert warm.policy.value_net.weight.grad is not None
            set_ppo_actor_trainable(warm, True)
            assert warm.policy.action_net.weight.requires_grad and warm.policy.log_std.requires_grad
            assert optimizer_step_count(warm) == 0
            try:
                warm_start_ppo(warm, env, exported, reset_critic=False)
                raise AssertionError("A changed gamma reused the old critic silently")
            except ValueError:
                pass
            with tempfile.TemporaryDirectory() as folder:
                check_fresh_process(exported, folder)
                destination = Path(folder) / "checkpoint.pt"
                for group in model.policy.optimizer.param_groups:
                    for parameter in group["params"]:
                        model.policy.optimizer.state[parameter] = {"step": torch.tensor(7.0),
                                                                   "exp_avg": torch.zeros_like(parameter),
                                                                   "exp_avg_sq": torch.ones_like(parameter)}
                model.num_timesteps = 32  # Synthetic counter, not collected training data.
                save_ppo_checkpoint(model, env, args, exported.__getstate__(), -100.0, False, destination)
                state = torch.load(destination, weights_only=True)
                restored = PPO.load(io.BytesIO(state["model_zip"].numpy().tobytes()), env=env, device=args.device)
                assert all(item["step"].item() == 7.0 for item in restored.policy.optimizer.state.values())
                actual, _ = restored.predict(observations, deterministic=True)
                expected, _ = model.predict(observations, deterministic=True)
                np.testing.assert_array_equal(actual, expected)
                # Exercise orchestration and boundary saves with a mocked learner, not training.
                test_args = parse_args(["--algorithm", "ppo", "--num-envs", "1", "--steps", "8",
                                        "--ppo-rollout-steps", "8", "--ppo-batch-size", "8",
                                        "--log-dir", str(Path(folder) / "runs"),
                                        "--model-path", str(Path(folder) / "candidate.model")])
                def fake_learn(current, total_timesteps, reset_num_timesteps, callback, tb_log_name):
                    callback.model = current
                    callback._on_rollout_start()
                    current.num_timesteps += total_timesteps
                    return current
                def fake_eval(current, seeds):
                    return {"seeds": list(seeds), "rewards": [0.] * len(seeds), "lengths": [1] * len(seeds),
                            "mean_reward": 0., "std_reward": 0., "success_rate": 0., "diagnostics": []}
                with patch.object(PPO, "learn", fake_learn), patch("train.evaluate_agent", fake_eval), \
                        contextlib.redirect_stdout(io.StringIO()):
                    run_ppo_training(test_args)
                    checkpoint_path = next((Path(folder) / "runs").rglob("checkpoint.pt"))
                    checkpoint = torch.load(checkpoint_path, weights_only=True)
                    assert checkpoint["steps"] == 8 and not checkpoint["confirmed"]
                    best_checkpoint = torch.load(checkpoint_path.with_name("best-checkpoint.pt"), weights_only=True)
                    assert best_checkpoint["steps"] == 0  # Equal scores keep the earlier complete best state.
                    test_args.resume = str(checkpoint_path)
                    test_args.steps = 16
                    run_ppo_training(test_args)
                saved = [torch.load(p, weights_only=True)["steps"] for p in (Path(folder) / "runs").rglob("checkpoint.pt")]
                assert sorted(saved) == [8, 16]
                init_args = parse_args(["--algorithm", "ppo", "--num-envs", "1", "--steps", "8",
                    "--ppo-rollout-steps", "8", "--ppo-batch-size", "8", "--init-checkpoint", str(destination),
                    "--ppo-capg", "--ppo-gae-lambda", ".98",
                    "--log-dir", str(Path(folder) / "continuation"), "--model-path", str(Path(folder) / "continuation.model")])
                def continuation_learn(current, total_timesteps, reset_num_timesteps, callback, tb_log_name):
                    assert current.num_timesteps == 0 and optimizer_step_count(current) == 7
                    assert math.isclose(current.lr_schedule(1.), init_args.ppo_learning_rate)
                    assert current.gae_lambda == current.rollout_buffer.gae_lambda == .98
                    assert type(current.policy.action_dist).__name__ == "ClippedGaussian"
                    return fake_learn(current, total_timesteps, reset_num_timesteps, callback, tb_log_name)
                with patch.object(PPO, "learn", continuation_learn), patch("train.evaluate_agent", fake_eval), \
                        contextlib.redirect_stdout(io.StringIO()):
                    continuation = run_ppo_training(init_args)
                    assert continuation["steps"] == 8 and continuation["optimizer_steps"] == 7
                capg_checkpoint = next(Path(init_args.log_dir).rglob("checkpoint.pt"))
                init_args.init_checkpoint = None
                init_args.resume = str(capg_checkpoint)
                init_args.ppo_capg, init_args.ppo_gae_lambda = False, .95  # Resume must restore the saved algorithm settings.
                init_args.ppo_learning_rate, init_args.ppo_final_lr = .123, .0123
                init_args.steps = 16
                def capg_resume_learn(current, total_timesteps, reset_num_timesteps, callback, tb_log_name):
                    assert current.num_timesteps == 8 and optimizer_step_count(current) == 7
                    assert current.gae_lambda == current.rollout_buffer.gae_lambda == .98
                    assert type(current.policy.action_dist).__name__ == "ClippedGaussian"
                    assert init_args.ppo_learning_rate == read_ppo_checkpoint(capg_checkpoint)["config"]["ppo_learning_rate"]
                    return fake_learn(current, total_timesteps, reset_num_timesteps, callback, tb_log_name)
                with patch.object(PPO, "learn", capg_resume_learn), patch("train.evaluate_agent", fake_eval), \
                        contextlib.redirect_stdout(io.StringIO()):
                    assert run_ppo_training(init_args)["steps"] == 16
                warm_path = Path(folder) / "warm.model"
                atomic_save(exported, warm_path)
                warm_args = parse_args(["--algorithm", "ppo", "--num-envs", "1", "--steps", "16",
                    "--ppo-rollout-steps", "8", "--ppo-batch-size", "8", "--gamma", "0.999",
                    "--warm-start", str(warm_path), "--reset-critic", "--critic-warmup-rollouts", "1",
                    "--log-dir", str(Path(folder) / "warmup"), "--model-path", str(Path(folder) / "warmup.model")])
                def warmup_learn(current, total_timesteps, reset_num_timesteps, callback, tb_log_name):
                    callback.model = current
                    callback._on_rollout_start()
                    assert not current.policy.action_net.weight.requires_grad and not current.get_env().training
                    current.num_timesteps += 8
                    callback._on_rollout_start()
                    assert current.policy.action_net.weight.requires_grad and current.get_env().training
                    current.num_timesteps += 8
                    return current
                with patch.object(PPO, "learn", warmup_learn), patch("train.evaluate_agent", fake_eval), \
                        contextlib.redirect_stdout(io.StringIO()):
                    run_ppo_training(warm_args)
                # Distinct reward and finish leaders must retain their own complete update counters.
                dual_args = parse_args(["--algorithm", "ppo", "--num-envs", "1", "--steps", "16",
                    "--ppo-rollout-steps", "8", "--ppo-batch-size", "8", "--ppo-eval-steps", "1",
                    "--warm-start", str(warm_path), "--freeze-obs-stats", "--terrain-replay-fraction", ".25",
                    "--ppo-use-sde", "--entropy-start", ".001", "--entropy-end", ".0002", "--defer-confirmation",
                    "--run-dir", str(Path(folder) / "dual"), "--model-path", str(Path(folder) / "dual.model")])
                evaluation_index = 0
                def dual_eval(current, seeds):
                    nonlocal evaluation_index
                    result = fake_eval(current, seeds)
                    result["mean_reward"], result["success_rate"] = [(170., 0.), (185., 0.), (180., .05)][min(evaluation_index, 2)]
                    evaluation_index += 1
                    return result
                def dual_learn(current, total_timesteps, reset_num_timesteps, callback, tb_log_name):
                    assert current.use_sde and not current.get_env().training
                    current.get_env().reset()
                    callback.model = current
                    while current.num_timesteps < dual_args.steps:
                        callback._on_rollout_start()
                        assert not current.get_env().training
                        assert .0002 <= current.ent_coef <= .001
                        current.num_timesteps += 8
                    return current
                with patch.object(PPO, "learn", dual_learn), patch("train.evaluate_agent", dual_eval), \
                        contextlib.redirect_stdout(io.StringIO()):
                    run_ppo_training(dual_args)
                    best = read_ppo_checkpoint(Path(dual_args.run_dir) / "best-checkpoint.pt")
                    coverage = read_ppo_checkpoint(Path(dual_args.run_dir) / "coverage-checkpoint.pt")
                    assert best["steps"] == 8 and best["best_score"] == 185.
                    assert coverage["steps"] == 16 and coverage["best_score"] == 180.
                    assert len(coverage["runtime"]["terrain_states"]) == 1
                    assert coverage["runtime"]["best_validation"]["success_rate"] == .05
                    np.testing.assert_array_equal(coverage["normalization"]["mean"], exported.config["observation_normalization"]["mean"])
                    dual_args.warm_start = None
                    dual_args.resume = str(Path(dual_args.run_dir) / "checkpoint.pt")
                    dual_args.steps = 24
                    run_ppo_training(dual_args)
                    assert read_ppo_checkpoint(dual_args.resume)["steps"] == 24
                for confirmation_passes in (False, True):
                    test_args.resume = None
                    test_args.steps = 8
                    test_args.ppo_eval_steps = 1
                    test_args.log_dir = str(Path(folder) / f"goal-{confirmation_passes}")
                    def goal_eval(current, seeds, passes=confirmation_passes):
                        score = 299.0 if seeds == test_args.confirmation_seeds and not passes else 310.0
                        return {"seeds": list(seeds), "rewards": [score] * len(seeds), "lengths": [1] * len(seeds),
                                "mean_reward": score, "std_reward": 0., "success_rate": 1., "diagnostics": []}
                    with patch.object(PPO, "learn", fake_learn), patch("train.evaluate_agent", goal_eval), \
                            contextlib.redirect_stdout(io.StringIO()):
                        run_ppo_training(test_args)
                    result_path = next(Path(test_args.log_dir).rglob("final-evaluation.json"))
                    result = json.loads(result_path.read_text())
                    assert result["validation_and_confirmation_passed"] == confirmation_passes
                    assert result["example_set_passed"]
                # Passing validation alone must not create an endless chain of zero-step campaign stages.
                test_args.defer_confirmation = True
                test_args.log_dir = str(Path(folder) / "validation-only-pass")
                with patch.object(PPO, "learn", fake_learn), patch("train.evaluate_agent", goal_eval), \
                        contextlib.redirect_stdout(io.StringIO()):
                    completed = run_ppo_training(test_args)
                    assert completed["steps"] == test_args.steps
    finally:
        env.close()
    print("PASS: PPO normalized-observation export, teacher load, mocked lifecycle/goal checks and checkpoint round trip. Optimizer steps: 0.")


# Verify CAPG boundary mass, gradients and SB3 rollout/resume integration without training.
def check_capg(args):
    from unittest.mock import patch
    means = torch.tensor([[.3, -.2, 1.4, -1.4]], dtype=torch.float64, requires_grad=True)
    stds = torch.full_like(means, .7)
    actions = torch.tensor([[-1., .1, 1., -4.]], dtype=torch.float64)
    normal = torch.distributions.Normal(means, stds)
    expected = normal.cdf(actions.new_tensor(-1.))[0, 0].log() + normal.log_prob(actions)[0, 1]
    expected += (1 - normal.cdf(actions.new_tensor(1.)))[0, 2].log() + normal.cdf(actions.new_tensor(-1.))[0, 3].log()
    actual = clipped_normal_log_prob(means, stds, actions)
    torch.testing.assert_close(actual[0], expected)
    actual.sum().backward()
    assert torch.isfinite(means.grad).all()
    torch.testing.assert_close(actual, clipped_normal_log_prob(means, stds, actions.clamp(-1., 1.)))
    grid = torch.linspace(-1., 1., 10001, dtype=torch.float64)[:, None]
    mass = torch.trapezoid(normal.log_prob(grid).exp(), grid[:, 0], dim=0)
    mass += normal.cdf(grid[0])[0] + 1 - normal.cdf(grid[-1])[0]
    torch.testing.assert_close(mass, torch.ones_like(mass), rtol=1e-7, atol=1e-7)

    # For reward clip(a), the exact derivative with respect to its Gaussian mean is P(-1<a<1).
    mu = torch.tensor(1.4, dtype=torch.float64, requires_grad=True)
    sigma = .7
    samples = mu.detach() + sigma * torch.randn(200000, generator=torch.Generator().manual_seed(23), dtype=torch.float64)
    clipped = samples.clamp(-1., 1.)
    log_prob = clipped_normal_log_prob(mu.expand(len(samples), 1), torch.full((len(samples), 1), sigma, dtype=torch.float64), clipped[:, None])
    gradient = torch.autograd.grad((log_prob * clipped).mean(), mu)[0]
    gaussian = torch.distributions.Normal(mu.detach(), sigma)
    exact = gaussian.cdf(torch.tensor(1.)) - gaussian.cdf(torch.tensor(-1.))
    assert abs(float(gradient - exact)) < .006
    a, b = (-1 - mu.detach()) / sigma, (1 - mu.detach()) / sigma
    log_phi = lambda value: -.5 * value.square() - .5 * math.log(2 * math.pi)
    score = torch.where(samples <= -1, -(log_phi(a) - torch.special.log_ndtr(a)).exp() / sigma,
                        torch.where(samples >= 1, (log_phi(b) - torch.special.log_ndtr(-b)).exp() / sigma,
                                    (samples - mu.detach()) / sigma**2))
    raw_estimator = clipped * (samples - mu.detach()) / sigma**2
    assert (clipped * score).var() < raw_estimator.var()
    extremes = torch.tensor([[-20., 0., 20.]], requires_grad=True)
    entropy = clipped_normal_entropy(extremes, torch.full_like(extremes, .1))
    entropy.backward()
    assert torch.isfinite(entropy).all() and torch.isfinite(extremes.grad).all()

    PPO, _, _, DummyVecEnv, _, VecNormalize = ppo_dependencies()
    env = VecNormalize(DummyVecEnv([make_ppo_env]), norm_obs=True, norm_reward=False)
    try:
        with tempfile.TemporaryDirectory() as folder, patch.object(torch.optim.Adam, "step", side_effect=AssertionError("No optimizer steps in checks")):
            model = PPO("MlpPolicy", env, seed=123, n_steps=8, batch_size=8,
                        policy_kwargs={"activation_fn": nn.Tanh, "net_arch": {"pi": [256, 256], "vf": [256, 256]}})
            with torch.no_grad():
                model.policy.action_net.bias.fill_(1.4)  # Make clipping common in this untrained test fixture.
            configure_ppo_distribution(model, True)
            _, callback = model._setup_learn(8, callback=lambda local, global_: True)
            assert model.collect_rollouts(env, callback, model.rollout_buffer, n_rollout_steps=8)
            data = next(model.rollout_buffer.get(8))
            assert bool(torch.any(data.actions == 1.)) and bool(torch.all(data.actions.abs() <= 1.))
            _, logp, _ = model.policy.evaluate_actions(data.observations, data.actions)
            torch.testing.assert_close(logp, data.old_log_prob, atol=1e-6, rtol=1e-6)
            exported = export_ppo_agent(model, env)
            state = env.get_original_obs()[0]
            prediction, _ = model.predict(env.normalize_obs(state), deterministic=True)
            np.testing.assert_allclose(exported.select_action(state, eval=True), prediction, atol=1e-6)
            check_fresh_process(exported, folder)
            options = parse_args(["--algorithm", "ppo", "--ppo-capg", "--num-envs", "1"])
            checkpoint = Path(folder) / "checkpoint.pt"
            save_ppo_checkpoint(model, env, options, exported.__getstate__(), 0., False, checkpoint)
            saved = read_ppo_checkpoint(checkpoint)
            assert saved["config"]["ppo_capg"]
            restored = PPO.load(io.BytesIO(saved["model_zip"].numpy().tobytes()), env=env, device="cpu")
            configure_ppo_distribution(restored, saved["config"]["ppo_capg"])
            _, restored_logp, _ = restored.policy.evaluate_actions(data.observations, data.actions)
            torch.testing.assert_close(restored_logp, logp)
            assert optimizer_step_count(restored) == 0
    finally:
        env.close()
    print("PASS: CAPG tail probabilities, unbiased-gradient example, reduced variance, entropy, real rollout log ratios and checkpoint/teacher load. Optimizer steps: 0.")


# Verify exploration exports and training-only terrain replay with zero optimizer steps.
def check_exploration(args):
    from types import SimpleNamespace
    from unittest.mock import patch
    PPO, _, _, DummyVecEnv, _, VecNormalize = ppo_dependencies()
    env = VecNormalize(DummyVecEnv([make_ppo_env]), norm_obs=True, norm_reward=False)
    try:
        with patch.object(torch.optim.Adam, "step", side_effect=AssertionError("No optimizer steps in checks")):
            policy_kwargs = {"activation_fn": nn.Tanh, "net_arch": {"pi": [256, 256], "vf": [256, 256]}}
            diagonal = PPO("MlpPolicy", env, seed=42, n_steps=8, batch_size=8, policy_kwargs=policy_kwargs)
            original = export_ppo_agent(diagonal, env)
            sde = PPO("MlpPolicy", env, seed=43, n_steps=8, batch_size=8, use_sde=True,
                      sde_sample_freq=4, policy_kwargs=policy_kwargs)
            warm_start_ppo(sde, env, original)
            exported = export_ppo_agent(sde, env)
            inputs = np.random.default_rng(123).normal(size=(12, STATE_DIM)).astype(np.float32)
            np.testing.assert_array_equal(original.sample_actions(inputs, True)[0], exported.sample_actions(inputs, True)[0])
            states = exported.state_tensor(inputs)
            expected = sde.policy.get_distribution(states).distribution
            actual = exported.actor(states)
            torch.testing.assert_close(actual.mean, expected.mean)
            torch.testing.assert_close(actual.variance, expected.variance)
            assert np.isfinite(exported.sample_actions(inputs)[1]).all()
            repeated1, _ = sde.predict(env.normalize_obs(inputs), deterministic=False)
            repeated2, _ = sde.predict(env.normalize_obs(inputs), deterministic=False)
            np.testing.assert_array_equal(repeated1, repeated2)  # Fixed SDE noise is reused until explicitly resampled.
            with tempfile.TemporaryDirectory() as directory:
                check_fresh_process(exported, directory)
            warm_start_ppo(sde, env, exported)
            torch.testing.assert_close(sde.policy.log_std.cpu(), exported.actor.sde_logstds)
            assert optimizer_step_count(sde) == 0
    finally:
        env.close()

    class FakeTerrain(gym.Env):
        observation_space = gym.spaces.Box(-10., 10., shape=(STATE_DIM,), dtype=np.float32)
        action_space = gym.spaces.Box(-1., 1., shape=(ACTION_DIM,), dtype=np.float32)
        def __init__(self):
            self.hull = SimpleNamespace(position=SimpleNamespace(x=0.0))
            self.game_over, self.finish = False, False
        def reset(self, *, seed=None, options=None):
            self.seed_used = seed
            return np.ones(STATE_DIM, dtype=np.float32), {"unchanged": True}
        def step(self, action):
            self.hull.position.x = 200. if self.finish else 20.
            return np.full(STATE_DIM, 2., dtype=np.float32), 7.25, self.finish, not self.finish, {"raw": True}

    terrain = TerrainReplay(FakeTerrain())
    terrain.reset(seed=123)
    replay_count = 0
    for _ in range(2000):
        observation, reward, terminated, truncated, info = terrain.step(np.ones(ACTION_DIM, dtype=np.float32))
        assert reward == 7.25 and np.all(observation == 2.) and not terminated and truncated and info["raw"]
        assert info["terrain"]["seed"] >= TRAIN_SEED_MIN and len(terrain.failures) <= 128
        replay_count += info["terrain"]["replayed"]
        terrain.reset()
    assert 350 < replay_count < 650
    saved = terrain.get_curriculum_state()
    restored = TerrainReplay(FakeTerrain())
    restored.set_curriculum_state(saved)
    restored.reset(seed=999)  # SB3's first reset must not erase a restored scheduler.
    terrain.reset()
    assert restored.terrain_seed == terrain.terrain_seed and restored.failures == terrain.failures
    forced = TerrainReplay(FakeTerrain(), 1.)
    forced.reset(seed=1)
    first = forced.terrain_seed
    for _ in range(4):
        forced.step(np.zeros(ACTION_DIM))
        forced.reset()
    assert first not in forced.failures and forced.terrain_seed != first  # Three retries, then retire a failure.
    forced.step(np.zeros(ACTION_DIM))
    forced.reset()
    recovered = forced.terrain_seed
    forced.env.finish = True
    _, _, _, _, info = forced.step(np.zeros(ACTION_DIM))
    assert info["terrain"]["success"] and recovered not in forced.failures
    print("PASS: gSDE marginal distribution, coherent sampling, portable export, bounded terrain replay and RNG restore. Optimizer steps: 0.")


# Exercise campaign routing, budget accounting and holdout isolation using synthetic outcomes.
def check_campaign():
    import contextlib
    from types import SimpleNamespace
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as directory, \
            patch.object(torch.optim.Adam, "step", side_effect=AssertionError("No optimizer steps in checks")):
        root = Path(directory)
        args = parse_args(["--algorithm", "ppo", "--num-envs", "1", "--steps", "16",
                           "--ppo-rollout-steps", "8", "--ppo-batch-size", "8", "--target-success", ".8"])
        args.pilot_steps, args.chunk_steps, args.campaign_budget = 16, 16, 160
        spec = campaign_spec(args)
        spec["batch_size"] = 8
        baseline_path = root / "a2c.model"
        atomic_save(Agent({"hidden_dim": 256, "activation": "tanh", "gamma": .999}), baseline_path)
        base_score = {"mean_reward": 100., "success_rate": 0., "mean_progress": .5,
                      "median_progress": .5, "stall_fraction": .1}
        baseline = {"name": "baseline", "role": "return", "profile": "gaussian", "gamma": .999, "seed": None,
                    "score": base_score, "model": str(baseline_path),
                    "checkpoint": str(root / "final-best-checkpoint.pt"), "validation": base_score}
        baseline_config = {**vars(args), "gamma": .999, "ppo_learning_rate": 2.5e-5, "ppo_final_lr": 2e-5}
        atomic_save({"kind": "ppo_training_checkpoint", "format_version": FORMAT_VERSION, "steps": 8,
                     "config": baseline_config}, baseline["checkpoint"])
        commands = []
        scores = {"pilot-gaussian": (184.85, 0.), "pilot-capg": (179.43, .05),
                  "replicate-gaussian": (150., 0.), "replicate-capg": (190., .05),
                  "continue-002-capg-coverage": (200., .15),
                  "continue-004-capg-coverage": (310., .9)}

        def result_for(name, seeds=None):
            reward, success = scores.get(name, (130., 0.))
            seeds = list(seeds or spec["selection_seeds"])
            return {"seeds": seeds, "rewards": [reward] * len(seeds), "lengths": [100] * len(seeds),
                    "mean_reward": reward, "std_reward": 0., "success_rate": success,
                    "diagnostics": [{"progress_fraction": .8, "stall_fraction": .1}] * len(seeds)}

        def fake_execute(command, stage_dir, stop, stop_file):
            current = parse_args(command[3:])
            commands.append(current)
            stage_dir = Path(stage_dir)
            agent = Agent({"hidden_dim": 256, "activation": "tanh", "gamma": current.gamma,
                           "std_mode": "sde" if current.ppo_use_sde else "diagonal"})
            for name in ("a2c.model", "coverage.model"):
                atomic_save(agent, stage_dir / name)
            result = result_for(stage_dir.name, spec["validation_seeds"])
            saved = {"kind": "ppo_training_checkpoint", "format_version": FORMAT_VERSION,
                     "steps": current.steps, "config": vars(current),
                     "runtime": {"warmup_until": 0, "best_validation": result, "coverage_validation": result}}
            for name in ("checkpoint.pt", "best-checkpoint.pt", "coverage-checkpoint.pt"):
                atomic_save(saved, stage_dir / name)
            atomic_json({"status": "completed", "steps": current.steps, "best_validation": result,
                         "coverage_validation": result}, stage_dir / "stage-result.json")
            return 0

        def fake_assess(current, model_path, output_path, seeds=None):
            return result_for(Path(model_path).parent.name, seeds)

        def make_campaign(name):
            destination = root / name
            destination.mkdir()
            return Campaign(destination, spec.copy(), baseline_path, {"spec": spec.copy(), "stages": {}}, SimpleNamespace(requested=False))

        with patch("train.execute_campaign_stage", fake_execute), patch.object(Campaign, "assess", fake_assess), \
                contextlib.redirect_stdout(io.StringIO()):
            campaign = make_campaign("success")
            champion, reason = campaign_search(campaign, baseline)
            assert reason == "development_target_reached" and champion["name"] == "continue-004-capg-coverage"
            assert len(commands) == 8 and campaign_spent(campaign.manifest) == 128
            assert commands[0].init_checkpoint and not commands[0].ppo_capg
            assert all(c.entropy_start == c.entropy_end == 0. for c in commands)
            assert all(c.ppo_gae_lambda == .98 for c in commands)
            assert campaign.manifest["allocation_decision"]["profile"] == "capg"
            assert commands[1].init_checkpoint and commands[1].ppo_capg and commands[1].freeze_obs_stats
            assert commands[2].seed == commands[3].seed == spec["replication_seed"]
            assert commands[6].init_checkpoint and commands[7].init_checkpoint
            assert commands[7].ppo_learning_rate <= commands[1].ppo_learning_rate
            repeated, _ = campaign_search(campaign, baseline)
            assert repeated == champion and len(commands) == 8
            # The old score-only ranking would discard this rare-finishing policy.
            first, second = [campaign.manifest["stages"][n]["candidate"] for n in ("pilot-gaussian", "pilot-capg")]
            assert len(pareto_candidates([first, second])) == 2
            assert choose_candidate([first, second], baseline, spec, coverage=True) == second
            assert choose_candidate([first, second], baseline, spec, coverage=False) == first
            poor = {**second, "score": {**second["score"], "mean_reward": -100.},
                    "validation": {**second["validation"], "mean_reward": -100.}}
            assert choose_candidate([poor], baseline, spec, coverage=True) is None

            calls = []
            def fake_final_eval(agent, seeds):
                calls.append(list(seeds))
                result = result_for(champion["name"], seeds)
                if list(seeds) == spec["confirmation_seeds"]:
                    result["mean_reward"] = 250.
                    result["rewards"] = [250.] * len(seeds)
                return result
            with patch("train.evaluate_agent", fake_final_eval), patch("train.check_fresh_process"), \
                    patch("train.campaign_plots"), patch("train.subprocess.run", return_value=SimpleNamespace(stdout="test dependencies\n")):
                final = finish_campaign(campaign, champion, reason)
                assert final["coverage_goal_reached"] and not final["reward_goal_reached"] and not final["goal_reached"]
                finish_campaign(campaign, champion, reason)
                assert len(calls) == 2
                try:
                    finish_campaign(campaign, baseline, reason)
                    raise AssertionError("Holdout allowed a different final model")
                except ValueError:
                    pass
                with zipfile.ZipFile(campaign.root / "submission.zip") as archive:
                    assert set(archive.namelist()) == {"train.py", "a2c.model"}

            # A plateau terminates after three unproductive continuations instead of repeatedly restarting an old best.
            scores.pop("continue-004-capg-coverage")
            stalled = make_campaign("stalled")
            _, reason = campaign_search(stalled, baseline)
            assert reason == "development_plateau" and campaign_spent(stalled.manifest) == 144
            budgeted = make_campaign("budgeted")
            budgeted.spec["budget"] = 80
            assert campaign_search(budgeted, baseline)[1] == "budget_exhausted"
            assert campaign_spent(budgeted.manifest) == 80
            saved_scores = dict(scores)
            scores.clear()
            scores.update({name: (-100., 0.) for name in saved_scores})
            assert campaign_search(make_campaign("regressed"), baseline)[1] == "no_replicated_admissible_profile"
            scores.clear()
            scores.update(saved_scores)
            assert all(s["steps"] + s["lost_steps_upper_bound"] <= s["allocation"] for s in stalled.manifest["stages"].values())

            # Recover a clean interruption without charging its unspent budget twice.
            interrupted = make_campaign("interrupted")
            first_attempt = True
            def interrupt_once(command, stage_dir, stop, stop_file):
                nonlocal first_attempt
                code = fake_execute(command, stage_dir, stop, stop_file)
                if first_attempt:
                    first_attempt = False
                    saved = read_ppo_checkpoint(Path(stage_dir) / "checkpoint.pt")
                    saved["steps"] = 8
                    atomic_save(saved, Path(stage_dir) / "checkpoint.pt")
                    atomic_json({"status": "interrupted", "steps": 8}, Path(stage_dir) / "stage-result.json")
                    return 130
                return code
            with patch("train.execute_campaign_stage", interrupt_once):
                try:
                    interrupted.stage("interrupt", .999, 543, 24)
                    raise AssertionError("Expected an interruption")
                except InterruptedError:
                    pass
                interrupted.stage("interrupt", .999, 543, 24)
            assert campaign_spent(interrupted.manifest) == 24 and commands[-1].resume

            # A hard crash can discard at most one rollout beyond the last complete checkpoint.
            crashed = make_campaign("crashed")
            row = dict(interrupted.manifest["stages"]["interrupt"])
            row.update(name="crash", status="running", steps=0, allocation=24)
            crashed.manifest["stages"]["crash"] = row
            stage_dir = crashed.root / "stages" / "crash"
            stage_dir.mkdir(parents=True)
            saved = read_ppo_checkpoint(Path(commands[-1].resume))
            saved["steps"] = 8
            atomic_save(saved, stage_dir / "checkpoint.pt")
            crashed.stage("crash", .999, 543, 24)
            assert crashed.manifest["stages"]["crash"]["lost_steps_upper_bound"] == 8
            assert commands[-1].resume and commands[-1].steps == 16 and campaign_spent(crashed.manifest) == 24
        with file_lock(root / "duplicate.lock"):
            try:
                with file_lock(root / "duplicate.lock"):
                    raise AssertionError("Duplicate process lock was accepted")
            except RuntimeError:
                pass
        with patch("train.run_training", side_effect=AssertionError("Mac training guard failed")), \
                patch("train.run_ppo_training", side_effect=AssertionError("Mac training guard failed")), \
                patch("train.sys.platform", "darwin"):
            for mode in ("train", "campaign"):
                try:
                    main(["--mode", mode])
                    raise AssertionError("Mac training was not rejected")
                except RuntimeError as error:
                    assert "disabled" in str(error)
    print("PASS: dual-profile replication, Pareto coverage/return routes, strict budget, clean/crash resume and frozen holdout. Optimizer steps: 0.")


# Parse explicit modes and keep training budgets consistent with vector stepping.
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="CS100709 document-based A2C; see this file's header for setup.")
    parser.add_argument("--mode", choices=("train", "eval", "check", "benchmark", "campaign"), default="train")
    parser.add_argument("--algorithm", choices=("a2c", "ppo"), default="a2c")
    parser.add_argument("--ppo-rollout-steps", type=int, default=2048)
    parser.add_argument("--ppo-batch-size", type=int, default=64)
    parser.add_argument("--ppo-eval-steps", type=int, default=100000)
    parser.add_argument("--ppo-learning-rate", type=float, default=1e-4)
    parser.add_argument("--ppo-final-lr", type=float, default=2.5e-5)
    parser.add_argument("--ppo-clip-range", type=float, default=0.2)
    parser.add_argument("--ppo-final-clip", type=float, default=0.1)
    parser.add_argument("--ppo-target-kl", type=float, default=0.02)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--ppo-use-sde", action="store_true", help="Use SB3 generalized state-dependent exploration during PPO collection.")
    parser.add_argument("--ppo-capg", action="store_true", help="Optimize the clipped-action distribution using CAPG probabilities.")
    parser.add_argument("--ppo-gae-lambda", type=float, default=.95)
    parser.add_argument("--sde-sample-freq", type=int, default=4)
    parser.add_argument("--entropy-start", type=float, default=.001)
    parser.add_argument("--entropy-end", type=float, default=.001)
    parser.add_argument("--freeze-obs-stats", action="store_true", help="Keep pretrained observation scaling fixed.")
    parser.add_argument("--terrain-replay-fraction", type=float, default=0.0)
    parser.add_argument("--warm-start", help="Import an exported PPO Agent into a new optimizer.")
    parser.add_argument("--init-checkpoint", help="Start a new stage from a complete PPO checkpoint, retaining optimizer moments.")
    parser.add_argument("--reset-critic", action="store_true")
    parser.add_argument("--critic-warmup-rollouts", type=int, default=0)
    parser.add_argument("--run-dir", help="Fixed PPO stage directory; use --resume when it already contains a run.")
    parser.add_argument("--stop-file", help="Stop PPO at the next complete update boundary when this file exists.")
    parser.add_argument("--defer-confirmation", action="store_true", help="Campaign stage: keep confirmation and example seeds unused.")
    parser.add_argument("--campaign-dir", default="runs/campaign-issue4-v5")
    parser.add_argument("--campaign-baseline", help="Optional local copy of the pinned Issue #4 model; otherwise download it.")
    parser.add_argument("--campaign-checkpoint", help="Optional local copy of the pinned Issue #4 full checkpoint.")
    parser.add_argument("--campaign-budget", type=int, default=60_000_000)
    parser.add_argument("--pilot-steps", type=int, default=5_000_000)
    parser.add_argument("--chunk-steps", type=int, default=5_000_000)
    parser.add_argument("--dry-run", action="store_true", help="Print the campaign protocol without training, downloading or writing files.")
    parser.add_argument("--target-reward", type=float, default=300.0)
    parser.add_argument("--target-success", type=float, default=1.0, help="Required finish fraction; campaign defaults to 0.8.")
    parser.add_argument("--confirmation-seeds", type=int, nargs="+", default=list(range(201, 241, 2)))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--return-steps", type=int, default=1, help="A2C return horizon; 1 preserves the handout baseline.")
    parser.add_argument("--max-grad-norm", type=float, default=0.0, help="Separate Actor/Critic gradient limit; 0 disables clipping.")
    parser.add_argument("--actor-lr", type=float, default=4e-4)
    parser.add_argument("--critic-lr", type=float, default=4e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=543)
    parser.add_argument("--eval-freq", type=int, default=10, help="Completed training episodes between validations.")
    parser.add_argument("--save-freq", type=int, default=1, help="Completed training episodes between checkpoints.")
    parser.add_argument("--log-interval", type=int, default=100, help="Batch updates between console progress reports.")
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=VALIDATION_SEEDS)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=EXAMPLE_SEEDS)
    parser.add_argument("--log-dir", default="runs")
    parser.add_argument("--model-path", default="a2c.model")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--render", action="store_true", help="Show RGB frames in evaluation mode.")
    parser.add_argument("--benchmark-seconds", type=float, default=20.0)
    args = parser.parse_args(argv)
    provided = sys.argv[1:] if argv is None else argv
    if args.mode == "campaign":
        args.algorithm = "ppo"
        if not any(x == "--num-envs" or x.startswith("--num-envs=") for x in provided):
            args.num_envs = 16
        args.ppo_batch_size = 256
        args.validation_seeds = CAMPAIGN_VALIDATION_SEEDS.copy()
        for flag, name, value in (("--confirmation-seeds", "confirmation_seeds", CAMPAIGN_CONFIRMATION_SEEDS.copy()),
                                  ("--target-success", "target_success", .8), ("--seed", "seed", 19873)):
            if not any(x == flag or x.startswith(flag + "=") for x in provided):
                setattr(args, name, value)
    for name in ("num_envs", "torch_threads", "steps", "batch_size", "hidden_dim", "eval_freq", "save_freq", "log_interval", "return_steps"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.batch_size % args.num_envs or args.steps % args.num_envs:
        parser.error("--steps and --batch-size must be multiples of --num-envs.")
    if not math.isfinite(args.max_grad_norm) or args.max_grad_norm < 0:
        parser.error("--max-grad-norm must be finite and nonnegative.")
    if not 0 < args.gamma <= 1 or not all(math.isfinite(x) and x > 0 for x in (args.actor_lr, args.critic_lr, args.benchmark_seconds)):
        parser.error("Learning rates/benchmark duration must be positive and finite; gamma must be in (0, 1].")
    if not 0 <= args.seed < 2**32 - args.num_envs or any(seed < 0 or seed >= 2**32 for seed in args.eval_seeds + args.validation_seeds):
        parser.error("Seeds must fit nonnegative 32-bit integers, with room for worker offsets.")
    if args.resume and args.mode != "train":
        parser.error("--resume is only valid in training mode.")
    if sum(bool(x) for x in (args.resume, args.warm_start, args.init_checkpoint)) > 1:
        parser.error("Choose only one of --resume, --warm-start and --init-checkpoint.")
    if any((args.warm_start, args.init_checkpoint, args.reset_critic, args.run_dir, args.defer_confirmation, args.stop_file)) and (args.mode != "train" or args.algorithm != "ppo"):
        parser.error("PPO initialization/stage options require --mode train --algorithm ppo.")
    if args.reset_critic and (not args.warm_start or args.critic_warmup_rollouts < 1):
        parser.error("--reset-critic requires --warm-start and positive --critic-warmup-rollouts.")
    if args.dry_run and args.mode != "campaign":
        parser.error("--dry-run is only valid with --mode campaign.")
    if args.render and args.mode != "eval":
        parser.error("--render is only valid in evaluation mode.")
    if args.ppo_capg and args.algorithm != "ppo":
        parser.error("--ppo-capg requires --algorithm ppo.")
    if args.algorithm == "ppo":
        if not 0 <= args.ppo_gae_lambda <= 1 or (args.ppo_capg and args.ppo_use_sde):
            parser.error("GAE lambda must be in [0, 1]; CAPG and gSDE cannot be combined.")
        if not (0 <= args.entropy_end <= args.entropy_start < 1 and 0 < args.target_success <= 1
                and 0 <= args.terrain_replay_fraction <= .5 and args.sde_sample_freq > 0):
            parser.error("Invalid entropy schedule, success threshold, terrain replay fraction (0..0.5) or SDE frequency.")
        if args.terrain_replay_fraction and any(s >= TRAIN_SEED_MIN for s in args.validation_seeds + args.confirmation_seeds + args.eval_seeds):
            parser.error("Evaluation seeds must stay below the training-terrain seed range.")
        if args.freeze_obs_stats and not any((args.warm_start, args.init_checkpoint, args.resume)) and args.mode == "train":
            parser.error("Frozen observation statistics require a pretrained model or checkpoint.")
        if args.critic_warmup_rollouts < 0 or args.ppo_epochs < 1:
            parser.error("Invalid PPO epoch/warm-up count.")
        if not all(math.isfinite(v) and v > 0 for v in (args.ppo_learning_rate, args.ppo_final_lr, args.ppo_target_kl)):
            parser.error("PPO learning rates and KL threshold must be positive and finite.")
        if not 0 < args.ppo_final_clip <= args.ppo_clip_range < 1 or args.ppo_final_lr > args.ppo_learning_rate:
            parser.error("PPO schedules must decay to positive floors; clip must be below 1.")
        if min(args.ppo_rollout_steps, args.ppo_eval_steps) < 1 or args.ppo_batch_size < 2:
            parser.error("PPO needs positive rollout/evaluation sizes and batch size >= 2.")
        if args.ppo_rollout_steps * args.num_envs % args.ppo_batch_size:
            parser.error("PPO rollout size must be a multiple of ppo-batch-size.")
        if not math.isfinite(args.target_reward) or any(s < 0 or s >= 2**32 for s in args.confirmation_seeds):
            parser.error("Invalid PPO target or confirmation seeds.")
        groups = [set(args.validation_seeds), set(args.confirmation_seeds), set(args.eval_seeds)]
        if any(len(values) != len(set(values)) for values in (args.validation_seeds, args.confirmation_seeds, args.eval_seeds)):
            parser.error("PPO evaluation seed lists must not contain duplicates.")
        if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
            parser.error("PPO validation, confirmation and final evaluation seeds must be disjoint.")
        if args.mode == "benchmark":
            parser.error("The A2C benchmark does not measure PPO throughput; use observed PPO training timing.")
    if args.mode == "campaign":
        rollout = args.num_envs * args.ppo_rollout_steps
        if min(args.pilot_steps, args.chunk_steps) < rollout or args.campaign_budget < 4 * args.pilot_steps:
            parser.error("Campaign budget must cover two pilots and both replications; each stage needs a full rollout.")
        if args.seed + 100000 >= 2**32 - args.num_envs:
            parser.error("Campaign seed needs room for replication and continuation offsets.")
        groups = [set(CAMPAIGN_VALIDATION_SEEDS), set(CAMPAIGN_SELECTION_SEEDS), set(args.confirmation_seeds), set(args.eval_seeds)]
        if any(groups[i] & groups[j] for i in range(4) for j in range(i)):
            parser.error("Campaign development, confirmation and example seeds must be disjoint.")
        if any(s >= TRAIN_SEED_MIN for group in groups for s in group):
            parser.error("Campaign evaluation seeds must stay below the training-terrain seed range.")
    return args


# Dispatch modes only after explicit invocation of the command-line entry point.
def main(argv=None):
    args = parse_args(argv)
    if args.mode == "campaign" and args.dry_run:
        run_campaign(args)
        return 0
    if sys.platform == "darwin" and args.mode in ("train", "campaign"):
        raise RuntimeError("Training is disabled on the user's Mac. Use --mode check or --mode eval; train on Windows.")
    if args.mode == "campaign" and os.name != "nt":
        raise RuntimeError("The autonomous campaign runs on the target Windows computer.")
    configure_runtime(args)
    preflight(args)
    if args.mode == "campaign":
        result = run_campaign(args)
        return 130 if result.get("status") == "interrupted" else 0
    if args.mode == "check":
        run_checks(args)
        if args.algorithm == "ppo":
            check_ppo(args)
            check_capg(args)
            check_exploration(args)
            check_campaign()
    elif args.mode == "benchmark":
        run_benchmark(args)
    elif args.mode == "eval":
        agent = torch.load(args.model_path, map_location="cpu", weights_only=True)
        if not isinstance(agent, Agent):
            raise TypeError("The model file must contain an Agent, not a checkpoint or state_dict.")
        agent.to(args.device)
        print(json.dumps(evaluate_agent(agent, args.eval_seeds, args.render), indent=2))
    else:
        if args.algorithm == "ppo":
            result = run_ppo_training(args)
            return 130 if result["status"] == "interrupted" else 0
        else:
            run_training(args)
    return 0


# Run the importable module so pickled classes always resolve to train.Agent.
if __name__ == "__main__":
    mp.freeze_support()
    from train import main as imported_main
    raise SystemExit(imported_main())
