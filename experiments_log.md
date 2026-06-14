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

### Generalization regime — 1M augmented puzzles, 3 seeds

W&B project `STEER_PAPER_AUGMENTED`. Completed 2026-06-14.

| Metric | Baseline (λ=0.0) | STEER (λ=1.0) | Δ | perm. p |
| :--- | :---: | :---: | :---: | :---: |
| Peak val-exact (early-stopping) | 57.0 ± 3.4% | 48.1 ± 0.5% | −8.9pp | 0.10 |
| Final val-exact (end of training) | 54.8 ± 4.1% | 28.3 ± 6.3% | −26.5pp | 0.10 |

Per-seed peak / final val-exact:

| Seed | Baseline peak | Baseline final | STEER peak | STEER final |
| :---: | :---: | :---: | :---: | :---: |
| 0 | 54.6% | 52.3% | 47.8% | 32.1% |
| 1 | 60.9% | 59.5% | 47.8% | 31.8% |
| 2 | 55.5% | 52.7% | 48.6% | 21.1% |

**Findings:** STEER **actively harms** generalization — every STEER seed is below
every baseline seed (complete separation; p=0.10 is the floor for 3v3), −8.9pp
on peak and a large −26.5pp on final. The harm is much larger than in the
grokking regime.

---

## Overall conclusion

With the corrected (non-no-op) implementation, STEER as formulated provides **no
benefit and is net harmful**: neutral-to-slightly-worse in the small-data
regime, and clearly worse in the generalization regime. The regularizer does
reduce intermediate constraint violations during training, but forcing
intermediate reasoning steps toward validity/stability degrades the final
solution quality. The original positive claims were artifacts of a no-op STEER.
