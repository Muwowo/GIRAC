"""Bounded control and the nonlinear numerical process."""

import numpy as np


def nonlinear_transition(controls, observations, outputs, applied, gains):
    """Advance the four-lag nonlinear numerical system by one step."""
    observation_gain, output_gain, input_gain, coupling_gain = gains
    o = observations[::-1]
    y = outputs[::-1, 0]
    u = applied * input_gain
    previous = controls[-1] * input_gain
    next_o = np.array([
        o[:, 0] @ np.array([0.40, 0.12, 0.08, 0.05])
        + 0.15 * np.tanh(u[0]) + 0.08 * np.tanh(previous[0]) + 0.10 * coupling_gain * o[0, 1],
        o[:, 1] @ np.array([0.38, 0.12, 0.08, 0.05])
        + 0.12 * coupling_gain * np.tanh(o[0, 0] * u[0])
        + 0.10 * np.tanh(u[1]) + 0.06 * np.tanh(previous[1]),
        o[:, 2] @ np.array([0.36, 0.12, 0.08, 0.05])
        + 0.10 * coupling_gain * np.tanh(o[0, 0] * o[0, 1])
        + 0.08 * np.tanh(u[0] + u[1]) + 0.06 * coupling_gain * o[0, 0] * np.tanh(u[1]),
    ])
    next_o = np.clip(observation_gain * next_o, -0.95, 0.95)
    next_y = output_gain * (
        y @ np.array([0.38, 0.12, 0.08, 0.05]) + next_o @ np.array([0.10, 0.08, 0.06])
        + o[0] @ np.array([0.06, 0.05, 0.04]) + 0.05 * np.tanh(u[0]) + 0.04 * np.tanh(u[1])
    )
    return next_o.astype(np.float32), np.array([np.clip(next_y, -2.5, 2.5)], dtype=np.float32)


class Environment:
    def __init__(self, config, condition, series, surrogate=None):
        self.config = config
        self.condition = condition
        self.series = series
        self.surrogate = surrogate
        self.reported = config.protocol == "reported"
        dtype = np.float64 if self.reported else np.float32
        self.target = condition.target.astype(dtype)
        self.u_min = config.u_min.astype(dtype)
        self.u_max = config.u_max.astype(dtype)
        self.du_max = config.du_max.astype(dtype)
        self.o_min = config.o_min.astype(dtype)
        self.o_max = config.o_max.astype(dtype)

    def reset(self, index):
        """Start at the supplied row under the configured window convention."""
        stop = index + int(self.reported)
        start = stop - self.config.window
        if start < 0 or index >= len(self.series):
            raise ValueError(f"Reset row {index} has no complete history.")
        inputs, observations, _ = self.config.dimensions
        history = self.series[start:stop].copy()
        self.u = history[:, :inputs]
        self.o = history[:, inputs:inputs + observations]
        self.y = history[:, inputs + observations:]
        if not self.reported:
            self.u = np.clip(self.u, self.u_min, self.u_max)
        self.steps = 0
        return self.state()

    def state(self):
        return np.concatenate((self.o.ravel(), self.u.ravel(), self.y.ravel(), self.target))

    def step(self, action):
        previous = self.u[-1]
        desired = self.u_min + (np.clip(action, -1, 1) + 1) * 0.5 * (self.u_max - self.u_min)
        requested_delta = desired - previous
        applied = np.clip(previous + np.clip(requested_delta, -self.du_max, self.du_max), self.u_min, self.u_max)

        if self.surrogate is None:
            next_o, next_y = nonlinear_transition(self.u, self.o, self.y, applied, self.condition.gains)
        else:
            next_o, next_y = self.surrogate(self.u, self.o, self.y, applied)
        if not np.isfinite(next_o).all() or not np.isfinite(next_y).all():
            raise RuntimeError(f"{self.condition.name}: nonfinite process state at step {self.steps + 1}.")

        error = next_y - self.target
        excursion = np.maximum(0, next_o - self.o_max) + np.maximum(0, self.o_min - next_o)
        movement = np.abs(applied - previous)
        squared_error = np.sum(error ** 2) if self.reported else float(np.sum(error.astype(float) ** 2))
        reward = -20 * squared_error - 2 * np.sum(movement) - 50 * np.sum(excursion)
        normalized_excursion = excursion / (self.o_max - self.o_min)
        headroom = np.minimum(applied - self.u_min, self.u_max - applied) / (self.u_max - self.u_min)
        info = {
            "output": next_y.copy(),
            "error": error.copy(),
            "desired": desired.copy(),
            "applied": applied.copy(),
            "violation": bool(np.any(normalized_excursion > 1e-8)),
            "max_normalized_excursion": float(normalized_excursion.max()),
            "rate_limit_demand": bool(np.any(np.abs(requested_delta) >= self.du_max - 1e-8)),
            "rate_limit_execution": bool(np.any(movement >= 0.98 * self.du_max)),
            "bound_occupancy": bool(np.any(headroom <= 0.01)),
            "normalized_input_movement": float(np.mean(movement / self.du_max)),
            "input_variation": float(np.sum(movement)),
        }
        self.u = np.concatenate((self.u[1:], applied[None]), axis=0)
        self.o = np.concatenate((self.o[1:], next_o[None]), axis=0)
        self.y = np.concatenate((self.y[1:], next_y[None]), axis=0)
        self.steps += 1
        return self.state(), float(reward), self.steps >= self.config.training.horizon, info
