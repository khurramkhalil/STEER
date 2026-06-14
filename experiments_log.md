# STEER Experiment Log

## ⚠️ Prior campaigns retracted

All experiment numbers recorded before 2026-06-13 were produced while the STEER
regularizer was an inadvertent **no-op** (it never affected training — see the
fix in `models/losses.py` / `pretrain.py`). Those tables (the "+8.59pp" grokking
claim, the augmented campaigns, etc.) are **not valid** and have been removed.
Only verified, multi-seed results with the corrected code are kept below.

Reproduce any table with:

```bash
python scripts/collect_results.py --project <PROJECT> --baseline <arm> --steer <arm>
```

---

## Phase 2 — verified results (corrected STEER)

Setup: TRM (`arch=trm`, `mlp_t=True`, `pos_encodings=none`, H=3/L=6, L_layers=2),
`ema=True`, `global_batch_size=512`, 4×A100, test split subsampled to 1k for
cheap periodic eval. Datasets fixed across seeds (only training varies by seed).

### Grokking regime — 1k puzzles, no augmentation, 50k epochs (3 seeds)

W&B project `STEER_PAPER_EXACT`. Completed 2026-06-13.

| Metric | Baseline (λ=0.0) | STEER (λ=0.1) | Δ | perm. p |
| :--- | :---: | :---: | :---: | :---: |
| Peak val-exact (early-stopping) | 24.7 ± 1.1% | 22.9 ± 1.5% | −1.8pp | 0.20 |
| Final val-exact (end of training) | 13.6 ± 0.6% | 13.6 ± 6.5% | −0.1pp | 1.00 |

Per-seed peak / final val-exact:

| Seed | Baseline peak | Baseline final | STEER peak | STEER final |
| :---: | :---: | :---: | :---: | :---: |
| 0 | 25.8% | 13.0% | 22.4% | 17.7% |
| 1 | 24.7% | 13.8% | 21.7% | 16.9% |
| 2 | 23.7% | 14.1% | 24.6% | 6.1% |

**Findings:**
- STEER gives **no improvement** in the small-data regime: slightly worse on
  peak (not significant, p=0.20), identical on final, with much higher variance
  (one STEER seed collapsed to 6.1%).
- All models **overfit**: val-exact peaks ~25% near step ~10k, then declines to
  ~13% by step ~98k. STEER does not reliably mitigate this collapse.
- The regularizer *does* work mechanically during training (violations ↓,
  validity-rate ↑), but this does not translate into better generalization.

### Generalization regime — 1M augmented puzzles (3 seeds)

W&B project `STEER_PAPER_AUGMENTED`. **In progress** (Phase 2-A, the deciding
experiment for whether STEER helps where overfitting is not the dominant
failure).
