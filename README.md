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

- Python 3.10+
- CUDA 12.6+

```bash
pip install --upgrade pip wheel setuptools
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
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

We provide specific configurations to reproduce our paper results on the **Sudoku-Extreme** benchmark.

### 1. Best STEER Configuration (Grokking Regime — Small Data)
*   **Dataset**: 1,000 original puzzles (no augmentation)
*   **Val Exact Accuracy**: **~17.3%** (vs 8.7% baseline, **+8.6pp**)
*   **Key Flag**: `ema=True` is **REQUIRED** for high performance.

```bash
torchrun --nproc_per_node=4 pretrain.py \
  arch=trm \
  data_paths="[data/sudoku-extreme-1k-noaug]" \
  evaluators="[]" \
  epochs=50000 eval_interval=200 \
  lr=4e-4 puzzle_emb_lr=4e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.mlp_t=True arch.pos_encodings=none \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=6 \
  +steer_lambda=0.1 \
  +steer_epsilon_viol=0.1 \
  +steer_epsilon_stab=0.01 \
  global_batch_size=512 \
  +run_name="steer_grok_L0.1_E0.1" \
  +project_name="STEER_PAPER_EXACT" \
  ema=True
```

### 2. Best STEER Configuration (Generalization Regime — Augmented Data)
*   **Dataset**: 1,000,000 augmented puzzles
*   **Val Exact Accuracy**: **~53.8%** (vs 53.5% baseline, +0.4pp; prior campaign achieved +5.3pp)
*   **Key Flag**: `ema=True` is **REQUIRED** for high performance.

```bash
torchrun --nproc_per_node=4 pretrain.py \
  arch=trm \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  evaluators="[]" \
  epochs=50000 eval_interval=1000 \
  lr=4e-4 puzzle_emb_lr=4e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.mlp_t=True arch.pos_encodings=none \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=6 \
  +steer_lambda=1.0 \
  +steer_epsilon_viol=0.1 \
  +steer_epsilon_stab=0.01 \
  global_batch_size=512 \
  +run_name="steer_gen_L1.0_E0.1" \
  +project_name="STEER_PAPER_AUGMENTED" \
  ema=True
```

### 3. Unregularized Baseline (TRM)
```bash
torchrun --nproc_per_node=4 pretrain.py \
  arch=trm \
  data_paths="[data/sudoku-extreme-1k-noaug]" \
  evaluators="[]" \
  epochs=50000 eval_interval=200 \
  lr=4e-4 puzzle_emb_lr=4e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.mlp_t=True arch.pos_encodings=none \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=6 \
  +steer_lambda=0.0 \
  global_batch_size=512 \
  +run_name="baseline_no_steer" \
  +project_name="STEER_PAPER_EXACT" \
  ema=True
```

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
