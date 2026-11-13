"""GIRAC optimization with nominal and adversarial Bellman targets."""

import copy
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from .networks import GainInvariantEncoder, GIRACActor, GIRACCritic


@dataclass(frozen=True)
class Training:
    episodes: int = 100
    horizon: int = 200
    warmup_steps: int = 2000
    update_every: int = 1
    batch_size: int = 128
    buffer_size: int = 100_000
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005
    noise: float = 0.3
    noise_decay: float = 0.995
    noise_floor: float = 0.01
    switch_episode: int = 75
    ramp_episodes: int = 50
    ramp_origin: str = "training_start"
    epsilon_max: float = 0.25
    alpha_start: float = 0.2
    alpha_max: float = 0.4
    pgd_steps: int = 5

    def __post_init__(self):
        positive_counts = (
            self.episodes, self.horizon, self.update_every, self.batch_size,
            self.buffer_size, self.ramp_episodes, self.pgd_steps,
        )
        if any(type(value) is not int or value < 1 for value in positive_counts):
            raise ValueError("Training counts must be positive integers.")
        if self.batch_size > self.buffer_size:
            raise ValueError("The replay buffer must hold at least one batch.")
        if type(self.switch_episode) is not int or self.switch_episode < 0:
            raise ValueError("switch_episode must be a nonnegative integer.")
        if type(self.warmup_steps) is not int or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be a nonnegative integer.")
        if self.ramp_origin not in {"training_start", "robust_phase"}:
            raise ValueError("ramp_origin must be training_start or robust_phase.")
        if not 0 <= self.alpha_start <= self.alpha_max <= 1:
            raise ValueError("Require 0 <= alpha_start <= alpha_max <= 1.")
        if not 0 <= self.epsilon_max < float("inf"):
            raise ValueError("epsilon_max must be finite and nonnegative.")
        if not 0 <= self.gamma <= 1 or not 0 < self.tau <= 1:
            raise ValueError("gamma and tau must be in their probability ranges.")
        if not 0 <= self.noise_floor <= self.noise < float("inf"):
            raise ValueError("Require 0 <= noise_floor <= noise < infinity.")
        if not 0 < self.noise_decay <= 1:
            raise ValueError("noise_decay must be in (0, 1].")
        if any(not 0 < rate < float("inf") for rate in (self.lr_actor, self.lr_critic)):
            raise ValueError("Learning rates must be finite and positive.")

    def schedule(self, episode):
        """Return the active PGD coefficient and actor weight for a zero-based episode."""
        if episode < self.switch_episode:
            return 0.0, 0.0
        elapsed = episode if self.ramp_origin == "training_start" else episode - self.switch_episode + 1
        fraction = min(elapsed / self.ramp_episodes, 1.0)
        return self.epsilon_max * fraction, self.alpha_start + (self.alpha_max - self.alpha_start) * fraction


@dataclass(frozen=True)
class Network:
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    ffn_dim: int = 128
    hidden_size: int = 128

    def __post_init__(self):
        widths = (self.d_model, self.n_heads, self.n_layers, self.ffn_dim, self.hidden_size)
        if any(type(value) is not int or value < 1 for value in widths):
            raise ValueError("Network dimensions must be positive integers.")
        if self.d_model % (2 * self.n_heads):
            raise ValueError("Each rotary attention head must have even width.")

    def encoder(self, dimensions, window):
        inputs, observations, outputs = dimensions
        return GainInvariantEncoder(
            observations, inputs, outputs, window, self.d_model,
            self.n_heads, self.n_layers, self.ffn_dim,
        )

    def actor(self, dimensions, window):
        return GIRACActor(self.encoder(dimensions, window), dimensions[0], self.hidden_size)


class GIRAC:
    def __init__(self, dimensions, window, network, training, device="cpu"):
        self.training = training
        self.device = torch.device(device)
        self.actor = network.actor(dimensions, window).to(self.device)
        self.nominal = GIRACCritic(network.encoder(dimensions, window), dimensions[0], network.hidden_size).to(self.device)
        self.robust = GIRACCritic(network.encoder(dimensions, window), dimensions[0], network.hidden_size).to(self.device)
        self.target_actor = copy.deepcopy(self.actor).requires_grad_(False)
        self.target_nominal = copy.deepcopy(self.nominal).requires_grad_(False)
        self.target_robust = copy.deepcopy(self.robust).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=training.lr_actor)
        self.nominal_optimizer = torch.optim.Adam(self.nominal.parameters(), lr=training.lr_critic)
        self.robust_optimizer = torch.optim.Adam(self.robust.parameters(), lr=training.lr_critic)
        self.replay = deque(maxlen=training.buffer_size)
        self.noise = training.noise

        inputs, observations, outputs = dimensions
        observation_end = observations * window
        output_start = (observations + inputs) * window
        output_end = output_start + outputs * window
        self.perturbable = torch.zeros(output_end + outputs, device=self.device)
        self.perturbable[:observation_end] = 1
        self.perturbable[output_start:output_end] = 1

    def action(self, state):
        action = predict_action(self.actor, state)
        action += np.random.normal(0, self.noise, size=action.shape)
        return np.clip(action, -1, 1)

    def perturb(self, next_states, epsilon):
        """Minimize the robust target value within a masked, scaled box."""
        if epsilon < 1e-6:
            return next_states
        states = next_states.detach()
        radius = (states.abs() + 0.1) * epsilon * self.perturbable
        step = epsilon / self.training.pgd_steps
        delta = torch.zeros_like(states)
        for _ in range(self.training.pgd_steps):
            delta.requires_grad_(True)
            candidate = states + delta
            value = self.target_robust(candidate, self.target_actor(candidate))
            gradient, = torch.autograd.grad(value.sum(), delta)
            delta = torch.clamp(delta.detach() - step * gradient.sign(), -radius, radius)
        return states + delta

    def update(self, episode):
        batch_size = self.training.batch_size
        if len(self.replay) < batch_size:
            return
        s, a, r, next_s, terminal = zip(*random.sample(self.replay, batch_size))
        s, a, r, next_s, terminal = (
            torch.as_tensor(np.asarray(value), dtype=torch.float32, device=self.device)
            for value in (s, a, r, next_s, terminal)
        )
        discount = self.training.gamma * (1 - terminal[:, None])
        reward = r[:, None]
        with torch.no_grad():
            nominal_target = reward + discount * self.target_nominal(next_s, self.target_actor(next_s))
        optimize(self.nominal_optimizer, F.mse_loss(self.nominal(s, a), nominal_target))

        epsilon, alpha = self.training.schedule(episode)
        robust_target = nominal_target
        if episode >= self.training.switch_episode:
            perturbed = self.perturb(next_s, epsilon)
            with torch.no_grad():
                robust_target = reward + discount * self.target_robust(perturbed, self.target_actor(perturbed))
        optimize(self.robust_optimizer, F.mse_loss(self.robust(s, a), robust_target))

        command = self.actor(s)
        objective = -(1 - alpha) * self.nominal(s, command).mean()
        if alpha > 0:
            objective -= alpha * self.robust(s, command).mean()
        optimize(self.actor_optimizer, objective)

        with torch.no_grad():
            tau = self.training.tau
            for target, online in (
                (self.target_actor, self.actor),
                (self.target_nominal, self.nominal),
                (self.target_robust, self.robust),
            ):
                for destination, source in zip(target.parameters(), online.parameters()):
                    destination.copy_(tau * source + (1 - tau) * destination)


def optimize(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


@torch.no_grad()
def predict_action(actor, state):
    device = next(actor.parameters()).device
    tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    return actor(tensor)[0].cpu().numpy()
