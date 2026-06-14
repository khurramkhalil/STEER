# STEER: Self-Terminating Efficient Reasoning

This repository contains the official implementation of **STEER**, a framework designed to enhance recursive reasoning models (such as the Tiny Recursion Model, TRM) by strictly regularizing their halting behavior and internal stability.

## Motivation

Recursive reasoning models like TRM demonstrate that small networks can solve complex logic puzzles by iteratively refining their answers. However, unconstrained recursion can lead to instability and inefficient compute usage.

**STEER** introduces a triplet of regularization terms to guide ("steer") the reasoning process:
1.  **Validity Regularization (`epsilon_viol`)**: Enforces that the model's intermediate reasoning steps remain within valid bounds.
2.  **Stability Regularization (`epsilon_stab`)**: Penalizes erratic fluctuations in the latent state between recursive steps.
3.  **Halting Efficiency**: Encourages the model to terminate recursion as soon as a correct answer is confidently reached, saving compute.

By balancing these factors via `steer_lambda`, STEER aims to improve accuracy and
compute efficiency over baseline recursive models.

## Results

Verified, multi-seed results with the corrected STEER training path. (The
numbers previously reported here were produced when STEER was an inadvertent
no-op and have been retracted.) Reproduce with
[`scripts/collect_results.py`](scripts/collect_results.py).

### Sudoku-Extreme, grokking regime (1k puzzles, no aug, 3 seeds)

| Metric | Baseline (λ=0.0) | STEER (λ=0.1) | Δ | perm. p |
| :--- | :---: | :---: | :---: | :---: |
| Peak val-exact (early-stopping) | **24.7 ± 1.1%** | 22.9 ± 1.5% | −1.8pp | 0.20 |
| Final val-exact (end of training) | 13.6 ± 0.6% | 13.6 ± 6.5% | −0.1pp | 1.00 |

### Sudoku-Extreme, generalization regime (1M augmented puzzles, 3 seeds)

| Metric | Baseline (λ=0.0) | STEER (λ=1.0) | Δ | perm. p |
| :--- | :---: | :---: | :---: | :---: |
| Peak val-exact (early-stopping) | **57.0 ± 3.4%** | 48.1 ± 0.5% | −8.9pp | 0.10 |
| Final val-exact (end of training) | **54.8 ± 4.1%** | 28.3 ± 6.3% | −26.5pp | 0.10 |

**Finding (honest):** with the corrected (non-no-op) implementation, STEER as
formulated provides **no benefit and is net harmful** — neutral-to-slightly-worse
in the small-data regime, and clearly worse in the generalization regime (every
STEER seed below every baseline seed). The regularizer does reduce intermediate
constraint violations during training, but forcing intermediate reasoning steps
toward validity/stability degrades final solution quality. The original positive
claims were artifacts of a no-op STEER. See [experiments_log.md](experiments_log.md).

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

## Key Hyperparameters

| Hyperparameter | Default | Meaning |
| :--- | :---: | :--- |
| `steer_lambda` | `0.0` | Weight of the STEER regularizer (`0` disables it). |
| `steer_epsilon_viol` | `0.1` | Tolerance for the path-validity property. |
| `steer_epsilon_stab` | `0.01` | Convergence threshold for the stability property. |
| `ema` / `ema_rate` | `False` / `0.999` | EMA of weights for evaluation (enabled in the experiment configs). |
| `global_batch_size` | `512` | Effective batch size across all GPUs. |
| `seed` / `deterministic` | `0` / `False` | RNG seed; deterministic cuDNN/cuBLAS kernels. |

(Hyperparameter *recommendations* and sensitivity will be added with the Phase 2 / Phase 3 results.)

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
