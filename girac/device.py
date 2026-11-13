"""PyTorch device selection and portable checkpoints."""

import os

import torch


def select_device(requested="auto"):
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda:0"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable in this PyTorch installation.")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable in this PyTorch installation.")
    return device


def cpu_state_dict(module):
    weights = module.state_dict()
    for name, tensor in weights.items():
        weights[name] = tensor.cpu()
    return weights


def device_metadata(device):
    details = {"device": str(device)}
    if device.type == "cuda":
        details["device_name"] = torch.cuda.get_device_name(device)
        details["cuda_version"] = torch.version.cuda
    return details
