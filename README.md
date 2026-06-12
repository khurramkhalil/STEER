# STEER: Self-Terminating Efficient Reasoning

This repository contains the official implementation of **STEER**, a framework designed to enhance recursive reasoning models (such as the Tiny Recursion Model, TRM) by strictly regularizing their halting behavior and internal stability.

## Motivation

Recursive reasoning models like TRM demonstrate that small networks can solve complex logic puzzles by iteratively refining their answers. However, unconstrained recursion can lead to instability and inefficient compute usage.

**STEER** introduces a triplet of regularization terms to guide ("steer") the reasoning process:
1.  **Validity Regularization (`epsilon_viol`)**: Enforces that the model's intermediate reasoning steps remain within valid bounds.
2.  **Stability Regularization (`epsilon_stab`)**: Penalizes erratic fluctuations in the latent state between recursive steps.
3.  **Halting Efficiency**: Encourages the model to terminate recursion as soon as a correct answer is confidently reached, saving compute.

By balancing these factors via `steer_lambda`, we achieve higher accuracy and better compute efficiency than baseline recursive models.

## Results

### Sudoku-Extreme Benchmark

**Grokking Regime** (1,000 puzzles, 50,000 epochs, no augmentation):

| Method | `steer_lambda` | Val Exact Acc ↑ | Train Exact Acc |
| :--- | :---: | :---: | :---: |
| TRM Baseline | 0.0 | 8.68% | 100.00% |
| STEER (low) | 0.01 | 7.82% | 98.10% |
| **STEER (mid)** | **0.1** | **17.27%** 🏆 | 97.38% |
| STEER (high) | 1.0 | 16.42% | 94.36% |

> STEER at `λ=0.1` achieves **+8.59pp** over the unregularized baseline, effectively doubling generalization capability on the small-data regime.

**Generalization Regime** (1,000,000 augmented puzzles, 50,000 epochs):

| Method | `steer_lambda` | Val Exact Acc ↑ | Val Cell Acc ↑ | Val LM Loss ↓ |
| :--- | :---: | :---: | :---: | :---: |
| TRM Baseline | 0.0 | 53.47% | 83.64% | 0.3858 |
| **STEER (high)** | **1.0** | **53.84%** | **83.78%** | **0.3799** |

> On augmented data, STEER maintains accuracy advantages and reduces validation loss, indicating higher quality reasoning trajectories.

## Requirements

The reproducible environment is the Docker image (see [Dockerfile](Dockerfile)),
built on `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`. For a bare install that
matches it:

- Python 3.11 (the base image's Python)
- CUDA 12.4

```bash
pip install --upgrade pip wheel setuptools
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install --no-cache-dir --no-build-isolation adam-atan2==0.0.3
wandb login YOUR-LOGIN
```

## Tests

The STEER regularizer is covered by a unit/integration test suite (constraint
signals, STL robustness semantics, gradient flow, and loss-head wiring):

```bash
pytest
```

## Dataset Preparation

Data generation remains consistent with the original TRM benchmark.

```bash
# Sudoku-Extreme (small, grokking regime)
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-noaug --subsample-size 1000 --num-aug 0

# Sudoku-Extreme (augmented, generalization regime)
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000
```

## Usage & Experiments

Each paper run is a named Hydra config in [config/experiment/](config/experiment/),
so there is no long command line to copy. The available experiments:

| Experiment | Regime | `steer_lambda` |
| :--- | :--- | :---: |
| `grok_baseline` | Small data (1k, no aug) | 0.0 |
| `grok_steer` | Small data (1k, no aug) | 0.1 |
| `aug_baseline` | Augmented (1M) | 0.0 |
| `aug_steer` | Augmented (1M) | 1.0 |

> `ema=True` is set in every experiment config and is **required** for good performance.

### Run locally (single node)

```bash
# builds the dataset if missing, then trains (1 GPU shown)
scripts/run_local.sh grok_steer 0 1
```

Or invoke Hydra directly (overrides on the CLI as usual):

```bash
torchrun --nproc_per_node=4 pretrain.py experiment=grok_steer seed=0
```

### Run on the NRP (Nautilus) cluster

```bash
# one experiment, one seed
scripts/launch_nrp.sh grok_steer 0          # exp, seed, [gpus], [image_tag]

# a whole table (baseline + STEER, multiple seeds)
scripts/reproduce.sh grok 3                  # grok_{baseline,steer} x seeds 0..2
```

The container image is built and pushed from Brev:

```bash
# on Brev (CPU box) -- builds and pushes khurramkhalil/steer:latest
docker build -t khurramkhalil/steer:latest . && docker push khurramkhalil/steer:latest
```

Datasets are built reproducibly (the builder is seeded via `--seed`). To build
them by hand, see **Dataset Preparation** above.

**Logging.** Runs log to Weights & Biases using the `project_name`/`run_name`
from the experiment config. To run without a W&B account, set
`WANDB_MODE=offline`. Key metrics are also printed to stdout every
`log_interval` steps.

## Key Hyperparameter Guidance

| Hyperparameter | Recommended Value | Notes |
| :--- | :---: | :--- |
| `steer_lambda` | 0.1 (small data) / 1.0 (large data) | Too low (0.01): no effect. Too high with tight ε: unstable. |
| `steer_epsilon_viol` | 0.1 | Loose tolerance. Tight (0.01) causes NCCL crashes in DDP. |
| `steer_epsilon_stab` | 0.01 | Fixed across all stable runs. |
| `ema` | `True` | **Required**. Runs without EMA degrade by ~30% absolute. |
| `global_batch_size` | 512 | Effective batch size across all GPUs. |

## Reference

If you build upon STEER, please cite our work (citation pending) and the original TRM paper:

```bibtex
@misc{jolicoeurmartineau2025morerecursivereasoningtiny,
      title={Less is More: Recursive Reasoning with Tiny Networks}, 
      author={Alexia Jolicoeur-Martineau},
      year={2025},
      eprint={2510.04871},
}
```
