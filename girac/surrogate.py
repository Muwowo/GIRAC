"""GRU observation transition and soft sensor models."""

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import series_parts


class ObservationModel(nn.Module):
    def __init__(self, inputs, observations):
        super().__init__()
        self.gru_u = nn.GRU(inputs, 16, batch_first=True)
        self.gru_o = nn.GRU(observations, 16, batch_first=True)
        self.l_u = nn.Linear(16, 64)
        self.l_o = nn.Linear(16, 64)
        self.l_on = nn.Linear(64, observations)

    def forward(self, controls, observations):
        u, _ = self.gru_u(controls)
        o, _ = self.gru_o(observations)
        return self.l_on(self.l_u(u[:, -1]) * self.l_o(o[:, -1]))


class OutputModel(nn.Module):
    def __init__(self, inputs, observations, outputs):
        super().__init__()
        self.gru_u = nn.GRU(inputs, 16, batch_first=True)
        self.gru_o = nn.GRU(observations, 16, batch_first=True)
        self.gru_y = nn.GRU(outputs, 8, batch_first=True)
        self.l_u = nn.Linear(16, 64)
        self.l_o = nn.Linear(16, 64)
        self.l_y = nn.Linear(8, 64)
        self.l_on = nn.Linear(64, outputs)

    def forward(self, controls, observations, outputs):
        u, _ = self.gru_u(controls)
        o, _ = self.gru_o(observations)
        y, _ = self.gru_y(outputs)
        return self.l_on(self.l_u(u[:, -1]) * self.l_o(o[:, -1]) * self.l_y(y[:, -1]))


@dataclass
class Surrogate:
    otm: ObservationModel
    ssm: OutputModel

    @classmethod
    def create(cls, dimensions, device="cpu"):
        return cls(ObservationModel(*dimensions[:2]).to(device), OutputModel(*dimensions).to(device))

    @property
    def device(self):
        return next(self.otm.parameters()).device

    def weights(self):
        return {"otm": self.otm.state_dict(), "ssm": self.ssm.state_dict()}

    def load_weights(self, weights):
        self.otm.load_state_dict(weights["otm"])
        self.ssm.load_state_dict(weights["ssm"])
        self.otm.eval()
        self.ssm.eval()

    def predict(self, applied_controls, observations, outputs):
        next_o = self.otm(applied_controls, observations)
        updated_o = torch.cat((observations[:, 1:], next_o[:, None]), dim=1)
        next_y = self.ssm(applied_controls, updated_o, outputs)
        return next_o, next_y

    @torch.no_grad()
    def __call__(self, controls, observations, outputs, applied):
        updated_u = np.concatenate((controls[1:], applied[None]), axis=0)
        device = self.device
        u, o, y = (
            torch.as_tensor(value, dtype=torch.float32, device=device).unsqueeze(0)
            for value in (updated_u, observations, outputs)
        )
        next_o, next_y = self.predict(u, o, y)
        return next_o[0].cpu().numpy(), next_y[0].cpu().numpy()


def windows(series, dimensions, length):
    u, o, y = series_parts(series, dimensions)
    past = np.arange(length, len(series))[:, None] + np.arange(-length, 0)
    next_index = past[:, -1] + 1
    return tuple(
        torch.as_tensor(value, dtype=torch.float32)
        for value in (u[past], u[past + 1], o[past], o[past + 1], y[past], o[next_index], y[next_index])
    )


def fit_surrogate(config, train, validation, device="cpu"):
    """Fit both modules and return the selected model and epoch records."""
    torch.manual_seed(config.surrogate_seed)
    model = Surrogate.create(config.dimensions, device=device)
    old_u, applied_u, o, next_o, y, o_label, y_label = windows(
        train, config.dimensions, config.window
    )
    if config.protocol == "reported":
        otm_records = fit_module(model.otm, (old_u, o), o_label, config.surrogate_epochs)
        ssm_records = fit_module(model.ssm, (applied_u, next_o, y), y_label, config.surrogate_epochs)
        history = [
            {"epoch": epoch, "otm_mse": o_loss, "ssm_mse": y_loss}
            for epoch, (o_loss, y_loss) in enumerate(zip(otm_records, ssm_records), 1)
        ]
        model.load_weights(model.weights())
        return model, history

    _, v_u, v_o, _, v_y, v_o_label, v_y_label = (
        tensor.to(device) for tensor in windows(validation, config.dimensions, config.window)
    )
    loader = DataLoader(
        TensorDataset(applied_u, o, next_o, y, o_label, y_label),
        batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(config.surrogate_seed),
    )
    optimizer = torch.optim.Adam([*model.otm.parameters(), *model.ssm.parameters()], lr=1e-3)
    history = []
    best_loss = float("inf")
    stale = 0
    for epoch in range(1, config.surrogate_epochs + 1):
        model.otm.train()
        model.ssm.train()
        losses = []
        for batch in loader:
            batch_u, batch_o, batch_next_o, batch_y, batch_o_label, batch_y_label = (
                tensor.to(device) for tensor in batch
            )
            o_loss = nn.functional.mse_loss(model.otm(batch_u, batch_o), batch_o_label)
            y_loss = nn.functional.mse_loss(model.ssm(batch_u, batch_next_o, batch_y), batch_y_label)
            loss = o_loss + y_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.otm.eval()
        model.ssm.eval()
        with torch.no_grad():
            pred_o, pred_y = model.predict(v_u, v_o, v_y)
            loss = float(nn.functional.mse_loss(pred_o, v_o_label) + nn.functional.mse_loss(pred_y, v_y_label))
        if not np.isfinite(loss):
            raise RuntimeError("The surrogate produced a nonfinite validation loss.")
        history.append({"epoch": epoch, "train_mse_sum": float(np.mean(losses)), "validation_mse_sum": loss})
        if loss < best_loss - 1e-8:
            best_loss = loss
            best_weights = copy.deepcopy(model.weights())
            stale = 0
        else:
            stale += 1
            if stale >= 25:
                break
    model.load_weights(best_weights)
    return model, history


def fit_module(module, inputs, target, epochs):
    loader = DataLoader(TensorDataset(*inputs, target), batch_size=64)
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)
    device = next(module.parameters()).device
    history = []
    for _ in range(epochs):
        losses = []
        for batch in loader:
            *x, y = (tensor.to(device) for tensor in batch)
            loss = nn.functional.mse_loss(module(*x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        mean = float(np.mean(losses))
        if not np.isfinite(mean):
            raise RuntimeError("The surrogate produced a nonfinite training loss.")
        history.append(mean)
    return history


@torch.no_grad()
def prediction_metrics(model, test, dimensions, length, max_horizon):
    """Score the held-out cascade under recorded inputs."""
    device = model.device
    _, u, o, _, y, observed_o, observed_y = (
        tensor.to(device) for tensor in windows(test, dimensions, length)
    )
    pred_o, pred_y = model.predict(u, o, y)
    records = [
        errors("observation", 1, pred_o, observed_o, o[:, -1]),
        errors("output", 1, pred_y, observed_y, y[:, -1]),
    ]

    u, o, y = (torch.as_tensor(v, dtype=torch.float32, device=device) for v in series_parts(test, dimensions))
    horizon = min(max_horizon, len(test) - length)
    origins = np.unique(np.linspace(length, len(test) - horizon, 20, dtype=int))
    past = torch.as_tensor(origins[:, None] + np.arange(-length, 0), device=device)
    u_win, o_win, y_win = u[past], o[past], y[past]
    persist_o, persist_y = o_win[:, -1], y_win[:, -1]
    for step in range(1, horizon + 1):
        index = torch.as_tensor(origins + step - 1, device=device)
        u_win = torch.cat((u_win[:, 1:], u[index, None]), dim=1)
        pred_o, pred_y = model.predict(u_win, o_win, y_win)
        o_win = torch.cat((o_win[:, 1:], pred_o[:, None]), dim=1)
        y_win = torch.cat((y_win[:, 1:], pred_y[:, None]), dim=1)
        if step in {1, 10, 25, 50, 100, horizon}:
            records.append(errors("recursive_observation", step, pred_o, o[index], persist_o))
            records.append(errors("recursive_output", step, pred_y, y[index], persist_y))
    return records


def errors(signal, horizon, prediction, actual, persistence):
    residual = prediction - actual
    baseline = persistence - actual
    return {
        "signal": signal, "horizon": horizon, "origins": len(actual),
        "mae": float(residual.abs().mean()), "rmse": float(residual.square().mean().sqrt()),
        "persistence_mae": float(baseline.abs().mean()), "persistence_rmse": float(baseline.square().mean().sqrt()),
    }
