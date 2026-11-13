"""Command line entry points for the numerical and learned-process workflows."""

import argparse
from pathlib import Path

import torch

from .config import Config
from .device import select_device
from .experiment import evaluate, fit, summarize_runs, write_csv


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Gain-Invariant Robust Actor-Critic.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    for command, description in (
        ("fit", "Fit the source policy."),
        ("evaluate", "Evaluate a frozen policy."),
        ("run", "Fit and evaluate each requested seed."),
    ):
        subparser = subcommands.add_parser(command, help=description)
        subparser.add_argument("--config", required=True, type=Path)
        subparser.add_argument("--output", required=True, type=Path)
        subparser.add_argument("--device", choices=("auto", "cuda:0", "mps", "cpu"), default="auto",
                               help="Compute device (auto: CUDA device 0, then MPS, then CPU).")
        subparser.add_argument("--threads", type=int, default=1)
        if command == "fit":
            subparser.add_argument("--seed", type=int, default=17)
        elif command == "evaluate":
            subparser.add_argument("--run", required=True, type=Path)
        else:
            subparser.add_argument("--seeds", type=int, nargs="+", default=[17])
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive.")
    torch.set_num_threads(args.threads)
    try:
        config = Config.load(args.config)
        device = select_device(args.device)
        if args.command == "fit":
            fit(config, args.seed, args.output, device=device)
        elif args.command == "evaluate":
            evaluate(config, args.run, args.output, device=device)
        else:
            if len(set(args.seeds)) != len(args.seeds):
                parser.error("--seeds must be distinct.")
            args.output.mkdir(parents=True, exist_ok=False)
            rows = []
            for seed in args.seeds:
                run = args.output / f"seed_{seed}"
                fit(config, seed, run / "fit", device=device)
                rows.append(evaluate(config, run / "fit", run / "evaluation", device=device))
            write_csv(args.output / "summary.csv", summarize_runs(rows))
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"girac: {exc}\n")


if __name__ == "__main__":
    main()
