"""CS100709: document-based, one-step A2C for BipedalWalker HARDCORE.

Primary source
    CS100709_Reinforcement_Learning_Project.pdf, credited to Hongtu Zhou,
    Di Zhang and Hai Zhang (TiEV lab). Requirements: pp. 1-6; A2C theory:
    pp. 20-22; continuous-action example: pp. 31-40.
    Adaptations: four joint actions, detached TD targets, independent evaluation,
    Windows spawn workers, terminal-observation handling, portable persistence.
    This implements the handout's one-step variant, without reward shaping,
    observation normalization, replay memory, PPO, or an entropy bonus.

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
    tensorboard --logdir runs

Modes and outputs
    Only train calls optimizer.step(). Check and benchmark never update weights.
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
"""


# Import dependencies without creating environments, files or worker processes.
import argparse
import io
import json
import math
import multiprocessing as mp
import os
import platform
import random
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime
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
__all__ = ["Actor", "Agent", "Critic", "Mish"]


# Preserve the handout's activation and separate policy/value networks.
class Mish(nn.Module):
    def forward(self, inputs):
        return inputs * torch.tanh(F.softplus(inputs))


# Map observations to independent Gaussian means and learned joint deviations.
class Actor(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), Mish(),
            nn.Linear(hidden_dim, hidden_dim), Mish(),
            nn.Linear(hidden_dim, n_actions),
        )
        self.logstds = nn.Parameter(torch.full((n_actions,), 0.1))


    # Bound log standard deviations before exponentiating to avoid overflow.
    def forward(self, states):
        means = self.model(states)
        stds = self.logstds.clamp(math.log(1e-3), math.log(50.0)).exp()
        return torch.distributions.Normal(means, stds)


# Estimate one state value using a network separate from the policy.
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), Mish(),
            nn.Linear(hidden_dim, hidden_dim), Mish(),
            nn.Linear(hidden_dim, 1),
        )


    def forward(self, states):
        return self.model(states).squeeze(-1)


# Construct a fixed TD target, retaining bootstrap value after time truncation.
def td_targets(rewards, next_values, terminated, gamma):
    return (rewards + gamma * (~terminated.bool()).to(rewards.dtype) * next_values).detach()


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
        self.action_low = np.asarray(self.config["action_low"], dtype=np.float32)
        self.action_high = np.asarray(self.config["action_high"], dtype=np.float32)
        if not (np.all(self.action_low == -1) and np.all(self.action_high == 1)):
            raise ValueError("BipedalWalker requires action bounds [-1, 1] on every joint.")
        self.actor = Actor(STATE_DIM, ACTION_DIM, self.config["hidden_dim"])
        self.critic = Critic(STATE_DIM, self.config["hidden_dim"])
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
            targets = td_targets(rewards, self.critic(next_states), terminated, self.config["gamma"])
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
        return {key: np.concatenate(value, axis=0) for key, value in rows.items()}, episodes


# Evaluate complete episodes using the original, undiscounted environment reward.
def evaluate_agent(agent, seeds, render=False, env=None):
    owned = env is None
    env = make_env("rgb_array" if render else None) if owned else env
    figure = None
    rewards, lengths = [], []
    try:
        if render:
            import matplotlib.pyplot as plt
            plt.ion()
            figure, axes = plt.subplots()
            artist = None
        for seed in seeds:
            state, _ = env.reset(seed=int(seed))
            total = 0.0
            for step in range(1, EPISODE_LIMIT + 1):
                action = agent.select_action(state, eval=True)
                state, reward, terminated, truncated, _ = env.step(action)
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
    finally:
        if owned:
            env.close()
        if figure is not None:
            plt.close(figure)
    return {"seeds": list(seeds), "rewards": rewards, "lengths": lengths,
            "mean_reward": float(np.mean(rewards)), "std_reward": float(np.std(rewards))}


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
    assert_weights_equal(before, agent.__getstate__())
    print("PASS: five complete evaluation episodes and unchanged network weights.")
    print("All checks passed. Optimizer steps: 0. No trained model was produced.")


# Parse explicit modes and keep training budgets consistent with vector stepping.
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="CS100709 document-based A2C; see this file's header for setup.")
    parser.add_argument("--mode", choices=("train", "eval", "check", "benchmark"), default="train")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
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
    for name in ("num_envs", "torch_threads", "steps", "batch_size", "hidden_dim", "eval_freq", "save_freq", "log_interval"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.batch_size % args.num_envs or args.steps % args.num_envs:
        parser.error("--steps and --batch-size must be multiples of --num-envs.")
    if not 0 < args.gamma <= 1 or not all(math.isfinite(x) and x > 0 for x in (args.actor_lr, args.critic_lr, args.benchmark_seconds)):
        parser.error("Learning rates/benchmark duration must be positive and finite; gamma must be in (0, 1].")
    if not 0 <= args.seed < 2**32 - args.num_envs or any(seed < 0 or seed >= 2**32 for seed in args.eval_seeds + args.validation_seeds):
        parser.error("Seeds must fit nonnegative 32-bit integers, with room for worker offsets.")
    if args.resume and args.mode != "train":
        parser.error("--resume is only valid in training mode.")
    if args.render and args.mode != "eval":
        parser.error("--render is only valid in evaluation mode.")
    return args


# Dispatch modes only after explicit invocation of the command-line entry point.
def main(argv=None):
    args = parse_args(argv)
    configure_runtime(args)
    preflight(args)
    if args.mode == "check":
        run_checks(args)
    elif args.mode == "benchmark":
        run_benchmark(args)
    elif args.mode == "eval":
        agent = torch.load(args.model_path, map_location="cpu", weights_only=True)
        if not isinstance(agent, Agent):
            raise TypeError("The model file must contain an Agent, not a checkpoint or state_dict.")
        agent.to(args.device)
        print(json.dumps(evaluate_agent(agent, args.eval_seeds, args.render), indent=2))
    else:
        run_training(args)
    return 0


# Run the importable module so pickled classes always resolve to train.Agent.
if __name__ == "__main__":
    mp.freeze_support()
    from train import main as imported_main
    raise SystemExit(imported_main())
