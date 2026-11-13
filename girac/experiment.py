"""Source policy fitting and frozen-policy evaluation."""

import csv
import hashlib
import json
import platform
import random
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from .agent import GIRAC, predict_action
from .data import Scale, read_series, split_series
from .device import cpu_state_dict, device_metadata, select_device
from .simulation import Environment
from .surrogate import Surrogate, fit_surrogate, prediction_metrics


def seed_run(seed):
    if not 0 <= seed < 2**32:
        raise ValueError("Use integer seeds in [0, 2**32).")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def write_csv(path, records):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@contextmanager
def new_output(destination):
    """Publish a complete directory once its commands finish successfully."""
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"{destination} already exists. Choose a new output directory.")
    with TemporaryDirectory(prefix=".girac-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "output"
        staging.mkdir()
        yield staging
        if destination.exists():
            raise FileExistsError(f"{destination} already exists.")
        staging.rename(destination)


def fit(config, seed, destination, device="auto"):
    device = select_device(device)
    seed_run(seed)
    source = config.conditions[0]
    series = read_series(source.path, config.dimensions)
    train, validation, _ = split_series(series, config.window, config.protocol)
    scale = Scale.from_source(train, config.scaling)
    train = scale.apply(train)
    dtype = np.float64 if config.protocol == "reported" else np.float32
    train = train.astype(dtype)
    with new_output(destination) as output:
        model = None
        if config.backend == "learned":
            valid = scale.apply(validation).astype(dtype) if validation is not None else None
            model, epochs = fit_surrogate(config, train, valid, device=device)
            torch.save({"otm": cpu_state_dict(model.otm), "ssm": cpu_state_dict(model.ssm)},
                       output / "source_surrogate.pt")
            write_csv(output / "surrogate_history.csv", epochs)

        seed_run(seed)
        env = Environment(config, source, train, model)
        policy = GIRAC(config.dimensions, config.window, config.network, config.training, device=device)
        history = train_policy(config, policy, env, seed)
        torch.save(cpu_state_dict(policy.actor), output / "actor.pt")
        torch.save(cpu_state_dict(policy.nominal), output / "nominal_critic.pt")
        torch.save(cpu_state_dict(policy.robust), output / "robust_critic.pt")
        write_csv(output / "training.csv", history)
        write_json(output / "run.json", {
            "configuration": config.settings,
            "training": asdict(config.training),
            "network": asdict(config.network),
            "policy_seed": seed,
            "source_condition": source.name,
            "source_csv_sha256": sha256(source.path),
            "source_train_rows": len(train),
            "source_validation_rows": len(validation) if validation is not None else 0,
            "scale": scale.to_dict(),
            "actor_sha256": sha256(output / "actor.pt"),
            "source_surrogate_sha256": sha256(output / "source_surrogate.pt") if model else None,
            "actor_parameters": sum(parameter.numel() for parameter in policy.actor.parameters()),
            "checkpoint_selection": "final_episode",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": str(torch.__version__),
            "torch_threads": torch.get_num_threads(),
            **device_metadata(device),
        })


def train_policy(config, policy, env, seed):
    history = []
    steps = 0
    updates = 0
    rng = np.random.default_rng(seed)
    training = config.training
    for episode in range(training.episodes):
        if config.protocol == "reported":
            index = np.random.randint(config.window, len(env.series) - 1)
        else:
            index = rng.integers(config.window, len(env.series))
        state = env.reset(int(index))
        episode_reward = 0.0
        abs_errors = []
        for _ in range(training.horizon):
            action = policy.action(state)
            next_state, reward, done, info = env.step(action)
            policy.replay.append((state, action, reward, next_state, done))
            steps += 1
            update_clock = steps - 1 if config.protocol == "reported" else steps
            if update_clock >= training.warmup_steps and update_clock % training.update_every == 0:
                if len(policy.replay) >= training.batch_size:
                    policy.update(episode)
                    updates += 1
            state = next_state
            episode_reward += reward
            abs_errors.append(np.abs(info["error"]).mean())
        policy.noise = max(policy.noise * training.noise_decay, training.noise_floor)
        epsilon, alpha = training.schedule(episode)
        history.append({
            "episode": episode + 1,
            "environment_steps": steps,
            "gradient_updates": updates,
            "return": episode_reward,
            "mae": float(np.mean(abs_errors)),
            "epsilon": epsilon,
            "alpha": alpha,
        })
        if (episode + 1) % 25 == 0 or episode + 1 == training.episodes:
            print(f"Seed {seed}, episode {episode + 1}/{training.episodes}, return {episode_reward:.3f}", flush=True)
    return history


def evaluate(config, run, destination, device="auto"):
    device = select_device(device)
    manifest = json.loads((run / "run.json").read_text(encoding="utf-8"))
    if manifest["configuration"] != config.settings:
        raise ValueError("Use the same configuration for training and evaluation.")
    source = config.conditions[0]
    if sha256(source.path) != manifest["source_csv_sha256"]:
        raise ValueError("The source CSV changed after fitting.")
    if sha256(run / "actor.pt") != manifest["actor_sha256"]:
        raise ValueError("The actor checkpoint hash differs from the run manifest.")

    seed_run(manifest["policy_seed"])
    scale = Scale.from_dict(manifest["scale"])
    actor = config.network.actor(config.dimensions, config.window)
    actor.load_state_dict(torch.load(run / "actor.pt", weights_only=True, map_location="cpu"))
    actor.to(device).eval()
    dtype = np.float64 if config.protocol == "reported" else np.float32
    rows = []
    traces = []
    surrogate_scores = []
    sources = []
    with new_output(destination) as output:
        for condition in config.conditions:
            series = scale.apply(read_series(condition.path, config.dimensions)).astype(dtype)
            train, validation, test = split_series(series, config.window, config.protocol)
            model = None
            if config.backend == "learned":
                if condition.name == source.name:
                    model = Surrogate.create(config.dimensions, device=device)
                    checkpoint = run / "source_surrogate.pt"
                    expected = manifest.get("source_surrogate_sha256")
                    if expected is not None and sha256(checkpoint) != expected:
                        raise ValueError("The source simulator hash differs from the run manifest.")
                    model.load_weights(torch.load(checkpoint, weights_only=True, map_location="cpu"))
                else:
                    model, _ = fit_surrogate(config, train, validation, device=device)
                if config.protocol == "heldout":
                    scores = prediction_metrics(model, test, config.dimensions, config.window, config.training.horizon)
                    surrogate_scores.extend({"condition": condition.name, **score} for score in scores)

            env = Environment(config, condition, test, model)
            test_offset = len(series) - len(test)
            if config.protocol == "reported":
                origins = [config.reset_index]
            else:
                origins = np.unique(np.linspace(config.window, len(test) - 1, config.eval_starts, dtype=int))
            for index in origins:
                row, trace = rollout(actor, env, int(index), test_offset)
                rows.append(row)
                traces.extend(trace)
            sources.append({
                "condition": condition.name,
                "csv_sha256": sha256(condition.path),
                "fit_rows": len(train) if model is not None else 0,
                "validation_rows": len(validation) if model is not None and validation is not None else 0,
                "evaluation_block_rows": len(test),
                "initial_rows": [int(index) + test_offset for index in origins],
            })
        write_csv(output / "metrics.csv", rows)
        write_csv(output / "trajectories.csv", traces)
        if surrogate_scores:
            write_csv(output / "surrogate_metrics.csv", surrogate_scores)
        write_json(output / "evaluation.json", {
            "policy_seed": manifest["policy_seed"],
            "actor_sha256": manifest["actor_sha256"],
            "conditions": sources,
            "protocol": config.protocol,
            **device_metadata(device),
        })
    return rows


def rollout(actor, env, index, offset=0):
    state = env.reset(index)
    records = []
    errors = []
    rewards = []
    common = {"condition": env.condition.name, "reset_index": index + offset}
    for step in range(1, env.config.training.horizon + 1):
        state, reward, done, info = env.step(predict_action(actor, state))
        errors.append(info["error"])
        rewards.append(reward)
        record = {"step": step, **common, "reward": reward}
        for name in (
            "violation", "max_normalized_excursion", "rate_limit_demand", "rate_limit_execution",
            "bound_occupancy", "normalized_input_movement", "input_variation",
        ):
            record[name] = info[name]
        for name, values in (
            ("y", info["output"]), ("target", env.target),
            ("error", info["error"]), ("u", info["applied"]), ("desired_u", info["desired"]),
        ):
            record.update({f"{name}{i}": float(value) for i, value in enumerate(values, 1)})
        records.append(record)
        if done:
            break
    residual = np.asarray(errors, dtype=float)
    summary = {
        **common,
        "steps": len(records),
        "return": float(np.sum(rewards)),
        "terminal_reward": rewards[-1],
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt((residual ** 2).mean())),
        "max_normalized_excursion": max(row["max_normalized_excursion"] for row in records),
        "total_variation": float(np.sum([row["input_variation"] for row in records])),
    }
    for field in ("violation", "rate_limit_demand", "rate_limit_execution", "bound_occupancy", "normalized_input_movement"):
        summary[field] = float(np.mean([row[field] for row in records]))
    summary.update({f"terminal_error{i}": float(value) for i, value in enumerate(residual[-1], 1)})
    return summary, records


def summarize_runs(per_seed_rows):
    """Average starts within each seed, then summarize across independent seeds."""
    conditions = list(dict.fromkeys(row["condition"] for rows in per_seed_rows for row in rows))
    fields = [key for key in per_seed_rows[0][0] if key not in {"condition", "reset_index", "steps"}]
    summary = []
    for condition in conditions:
        for metric in fields:
            values = [
                float(np.mean([row[metric] for row in rows if row["condition"] == condition]))
                for rows in per_seed_rows
            ]
            summary.append({
                "condition": condition, "metric": metric, "seeds": len(values),
                "mean": float(np.mean(values)),
                "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else "",
            })
    return summary
