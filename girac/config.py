"""Load process and experiment settings from a JSON file."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .agent import Network, Training


def vector(value, length, label):
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(length, array)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite and scalar or length {length}.")
    return array


@dataclass(frozen=True)
class Condition:
    name: str
    path: Path
    target: np.ndarray
    gains: tuple[float, ...] | None


@dataclass(frozen=True)
class Config:
    path: Path
    settings: dict
    backend: str
    protocol: str
    scaling: str
    dimensions: tuple[int, int, int]
    window: int
    conditions: tuple[Condition, ...]
    u_min: np.ndarray
    u_max: np.ndarray
    du_max: np.ndarray
    o_min: np.ndarray
    o_max: np.ndarray
    training: Training
    network: Network
    surrogate_seed: int
    surrogate_epochs: int
    eval_starts: int
    reset_index: int

    @classmethod
    def load(cls, path):
        path = path.resolve()
        settings = json.loads(path.read_text(encoding="utf-8"))
        backend = settings["backend"]
        protocol = settings.get("protocol", "heldout")
        scaling = settings.get("scaling", "identity")
        if backend not in {"exact_nns", "learned"}:
            raise ValueError("backend must be exact_nns or learned.")
        if protocol not in {"heldout", "reported"}:
            raise ValueError("protocol must be heldout or reported.")
        if scaling not in {"identity", "source_minmax"}:
            raise ValueError("scaling must be identity or source_minmax.")
        dimensions = tuple(settings["dimensions"])
        if len(dimensions) != 3 or any(type(n) is not int or n < 1 for n in dimensions):
            raise ValueError("dimensions must contain positive integer input, observation, and output counts.")
        inputs, observations, outputs = dimensions
        window = settings["window"]
        if type(window) is not int or window < 2:
            raise ValueError("window must be an integer of at least 2.")
        if backend == "exact_nns" and (dimensions != (2, 3, 1) or window != 4 or scaling != "identity"):
            raise ValueError("exact_nns requires dimensions [2, 3, 1], window 4, and identity scaling.")
        if protocol == "reported" and (backend != "learned" or scaling != "identity"):
            raise ValueError("The reported protocol expects condition CSVs in the original controller coordinates.")

        conditions = []
        for item in settings["conditions"]:
            name = item["name"]
            if not isinstance(name, str) or not name or name in {c.name for c in conditions}:
                raise ValueError("Each condition needs a unique nonempty name.")
            gains = None
            if backend == "exact_nns":
                gains = tuple(map(float, vector(item["gains"], 4, "gains")))
                if min(gains) <= 0:
                    raise ValueError("Exact NNS gains must be positive.")
            conditions.append(Condition(
                name, (path.parent / item["csv"]).resolve(),
                vector(item["target"], outputs, f"{name}.target"), gains,
            ))
        if not conditions:
            raise ValueError("The first condition must specify the source data.")

        bounds = settings["bounds"]
        u_min = vector(bounds["u_min"], inputs, "u_min")
        u_max = vector(bounds["u_max"], inputs, "u_max")
        du_max = vector(bounds["du_max"], inputs, "du_max")
        o_min = vector(bounds["o_min"], observations, "o_min")
        o_max = vector(bounds["o_max"], observations, "o_max")
        if np.any(u_min >= u_max) or np.any(o_min >= o_max) or np.any(du_max <= 0):
            raise ValueError("Require increasing envelopes and positive actuator rate limits.")

        surrogate_seed = settings.get("surrogate_seed", 17)
        surrogate_epochs = settings.get("surrogate_epochs", 200 if protocol == "reported" else 150)
        eval_starts = settings.get("evaluation_starts", 5)
        reset_index = settings.get("reset_index", 17)
        counts = (surrogate_seed, surrogate_epochs, eval_starts, reset_index)
        if any(type(value) is not int for value in counts):
            raise ValueError("Surrogate, seed, and evaluation counts must be integers.")
        if surrogate_seed < 0 or min(surrogate_epochs, eval_starts, reset_index + 1) < 1:
            raise ValueError("Surrogate, seed, and evaluation counts are out of range.")

        return cls(
            path, settings, backend, protocol, scaling, dimensions, window, tuple(conditions),
            u_min, u_max, du_max, o_min, o_max, Training(**settings.get("training", {})),
            Network(**settings.get("network", {})), surrogate_seed, surrogate_epochs, eval_starts, reset_index,
        )
