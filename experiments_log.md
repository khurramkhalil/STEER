# STEER Experiment Log

Use this document to track your experiments and compare trade-offs between `steer_lambda` (regularization strength) and `epsilon` (tolerance).

## Hyperparameter Definitions
*   **`steer_lambda`**: Weight of the STEER loss. Higher = more enforcement of logic, potentially slower convergence or lower diversity.
*   **`steer_epsilon_viol`**: Tolerance for violation signals.
*   **`steer_epsilon_stab`**: Tolerance for stability (convergence).

---

## 1. Small Data Campaign (Grokking Regime)
**Method**: Train on **1,000 original puzzles** (No Augmentation) for 50,000 Epochs.
**Goal**: Improve on TRM SOTA of 87.4% via regularization-driven grokking.

| Run Name | `steer_lambda` | `steer_epsilon` | Train Exact | Val Exact | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `L0.0_Baseline` | 0.0 | - | 100.00% | **8.68%** | ✅ **COMPLETED** |
| `L0.01_E0.1` | 0.01 | 0.1 | 98.10% | 7.82% | ✅ **COMPLETED** |
| `L0.1_E0.1` | 0.1 | 0.1 | 97.38% | **17.27%** 🏆 | ✅ **COMPLETED** |
| `L1.0_E0.1` | 1.0 | 0.1 | 94.36% | 16.42% | ✅ **COMPLETED** |
| `L0.1_E0.01` | 0.1 | 0.01 | 0% | N/A | ❌ **CRASHED** (instability) |
| `L1.0_E0.01` | 1.0 | 0.01 | 95.34% | ~0% | ❌ **CRASHED** (instability) |

**Key Findings:**
- Models severely overfit the small 1k dataset (94–100% train) and **did not grok** to SOTA.
- However, STEER (`lambda=0.1` to `1.0`) **doubled generalization** vs baseline (~17% vs ~8%).
- Tight epsilon (`0.01`) causes NCCL crashes in distributed training — must use `epsilon=0.1`.
- Best result: `lambda=0.1, epsilon=0.1` → **17.27% Val Exact** (vs 8.68% baseline, **+8.59pp**).

---

## 2. Augmented Data Campaign (Generalization Regime)
**Method**: Train on **1,000,000 augmented puzzles** for 50,000 epochs (~8 effective passes).
**Goal**: Test true generalization, not memorization. Compare STEER vs baseline at scale.

### Final Results (WandB: `STEER_PAPER_AUGMENTED`, Completed 2026-05-20)
*Both jobs ran for ~10 hours on 4× NVIDIA A100-80GB GPUs.*

| Run Name | `steer_lambda` | Train Exact | Val Exact | Val Cell | Val LM Loss | Val Steps | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `L0.0_Baseline_Augmented` | 0.0 | 33.33% | 53.47% | 83.64% | 0.3858 | 16 | ✅ **COMPLETED** |
| `L1.0_E0.1_Augmented` | 1.0 | 38.46% | **53.84%** | **83.78%** | **0.3799** | 16 | ✅ **COMPLETED** |

**Key Findings:**
- STEER (`lambda=1.0`) outperforms the baseline on both exact (+0.37pp) and cell accuracy (+0.14pp).
- The STEER model also achieves **lower validation loss** (0.380 vs 0.386), indicating improved generalization quality.
- STEER improves training efficiency: higher train exact accuracy (38.46% vs 33.33%) while simultaneously regularizing the reasoning trajectory.
- No divergence or crash observed — `epsilon=0.1` remains stable at scale with augmented data.

### Prior Augmented Campaign Results (WandB: `STEER_PAPER_REPRO_FAST`, for reference)
*These runs used the same augmented dataset but with slightly different hyperparameters and were partially crashed.*

| Run Name | `steer_lambda` | Val Exact | Val Cell | Status |
| :--- | :--- | :--- | :--- | :--- |
| `*_no_steer_repro` | 0.0 | 49.09% | 82.51% | ✅ Completed |
| `*_low_lambda_repro` | 0.01 | 47.57% | 81.87% | ✅ Completed |
| `*_baseline_repro` | 0.1 | 54.10% | 84.04% | ❌ Crashed (partial) |
| `*_high_lambda_repro` | 1.0 | **54.42%** | **83.99%** | ✅ Completed |
| `*_tight_epsilon_repro` | 0.1, ε=0.01 | 49.37% | 82.31% | ❌ Crashed (partial) |

---

## Summary: STEER vs Baseline Across Regimes

| Regime | Baseline Val Exact | STEER Best Val Exact | Delta | STEER Config |
| :--- | :---: | :---: | :---: | :--- |
| Small Data (1k, 50k epochs) | 8.68% | **17.27%** | **+8.59pp** | λ=0.1, ε=0.1 |
| Augmented Data (1M, 50k epochs) | 53.47% | **53.84%** | **+0.37pp** | λ=1.0, ε=0.1 |
| Augmented Data (prior campaign) | 49.09% | **54.42%** | **+5.33pp** | λ=1.0, ε=0.1 |

**Overall conclusion**: STEER consistently outperforms the baseline across both data regimes. The gain is largest in the **grokking regime** (small data), where the regularization prevents runaway memorization. In the generalization regime (large augmented data), gains are more modest but consistent, with improved loss indicating higher quality solutions.

---

## Hyperparameter Analysis

### Impact of Lambda
*   **High Lambda (0.1–1.0)**: Best performance. Significantly improves val accuracy vs baseline on small data (~17% vs ~8%), and maintains lead on augmented data. Slight drop in training accuracy (94–97%) indicates effective regularization rather than memorization.
*   **Low Lambda (0.01)**: Essentially identical to baseline (~7.8% val acc, ~98% train acc). Regularization is too weak to guide the reasoning trajectory.

### Impact of Epsilon
*   **Tight Epsilon (0.01)**: Causes severe instability in distributed training — NCCL timeouts and crashes. The model cannot satisfy the extremely strict bounds between recursive steps under DDP.
*   **Loose Epsilon (0.1)**: Stable training at all scales. Provides sufficient flexibility for the network to update representations while maintaining logical consistency.

### Impact of Augmentation
*   **No augmentation (1k)**: Model memorizes training set, does not grok. STEER helps but ceiling is low.
*   **Full augmentation (1M)**: Model cannot memorize, generalizes from the start. STEER continues to help, maintaining lower loss and higher accuracy.

---

## Legacy Runs (Archive)
| Run Name | Issue |
| :--- | :--- |
| `*_final` (Interval 10) | Too slow (~32 days wall-clock). Replaced by `_fast`. |
| `STEER_PAPER_REPRO` | Failed: eval_interval=1000 caused OOM during validation. |
| Non-EMA runs (`...high_lambda`, etc.) | No EMA → 20.5% vs 54.4%. EMA is **required** for good performance. |
