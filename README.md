# GIRAC

Gain-Invariant Robust Actor-Critic for zero-shot cross-condition process control. GIRAC trains in one operating condition and evaluates the frozen actor in other conditions. The actor uses normalized histories, first differences, and tracking features. A nominal critic and a robust critic guide training. The robust target uses projected gradient descent (PGD) on observation and output histories.

## Install

Use Python 3.11 or newer and create a virtual environment.

```sh
python -m venv .venv
source .venv/bin/activate
```

For an NVIDIA GPU, install a compatible CUDA build of PyTorch in this environment using the [PyTorch installation selector](https://pytorch.org/get-started/locally/). The RTX 5070 Ti requires a build with Blackwell support.

```sh
python -m pip install -e .
```

The CLI selects `cuda:0`, then `mps`, then `cpu`. Use `--device` to select a backend explicitly. Checkpoints can be loaded on any supported backend.

## Run the nonlinear numerical system

The included NNS records are synthetic. The equation-based configuration trains in WC1 and evaluates the final actor from held-out histories in WC1–WC3.

```sh
python -m girac run --config configs/nns_exact.json --output runs/nns --seeds 17
```

To reproduce the CPU reference in [`reference/nns_exact_summary.csv`](reference/nns_exact_summary.csv), use the full seed list and a fresh output directory:

```sh
python -m girac run --config configs/nns_exact.json --output runs/nns_cpu \
  --seeds 17 117 217 317 417 --device cpu
```

The CPU reference uses Python 3.13.9, NumPy 2.3.5, and PyTorch 2.12.1. Results can vary across backends. The paper's original GIRAC benchmark training used an NVIDIA GeForce RTX 5070 Ti. The CPU reference corresponds to the supplementary equation-based evaluation.

`configs/nns_reported.json` runs the original NNS fitted-simulator protocol. It refits the observation and output models from the included NNS series before training and evaluating fresh actors. The same `run` command accepts this configuration.

## Use your own records

Copy `configs/process_example.json` and set the CSV paths, channel counts, history length, targets, and limits. Each CSV holds one condition, in time order. The first condition is the training source.

- Name columns `u1,...,uM` for inputs, `o1,...,oN` for observations, and `y1,...,yP` for outputs. Set `dimensions` to `[M,N,P]`. An optional numeric `time` column must increase strictly.
- Align each input with the transition into the row in which it is recorded. Each chronological fit, validation, and test block must contain more rows than the history window.
- With `source_minmax`, specify targets and bounds on the scale fitted to the source training block. Divide slew limits by the input span. Use `identity` for records already in controller coordinates. CSV paths are relative to the configuration file.

```sh
python -m girac fit --config configs/process_example.json --output runs/process --seed 17
python -m girac evaluate --config configs/process_example.json \
  --run runs/process --output runs/process_eval
```

The generic interface fits observation and output models and evaluates the source actor without updating it on target conditions.

## Outputs

`run` writes `summary.csv`. Each fit saves model weights, the training curve, and a manifest. Evaluation writes `metrics.csv` and `trajectories.csv`, including tracking errors, rewards, and requested and applied inputs. The manifests record the configuration, seeds, data hashes, and device.

## License

[MuLan Permissive Software License, Version 2](LICENSE).
