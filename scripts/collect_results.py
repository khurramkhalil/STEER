#!/usr/bin/env python
"""Aggregate Phase 2 results from Weights & Biases.

Pulls per-seed validation accuracy for a baseline arm and a STEER arm, then
reports mean +/- std, the delta, and an exact permutation-test p-value (no
SciPy dependency). Runs are grouped by run-name prefix (e.g. ``grok_baseline``
vs ``grok_steer``), matching scripts/launch_nrp.sh naming (``<exp>_s<seed>``).

Requires: ``pip install wandb`` and a W&B login (``WANDB_API_KEY`` env or
``wandb login``). The entity defaults to ``$WANDB_ENTITY``.

Examples:
  python scripts/collect_results.py --project STEER_PAPER_EXACT \
      --baseline grok_baseline --steer grok_steer
  python scripts/collect_results.py --project STEER_PAPER_AUGMENTED \
      --baseline aug_baseline --steer aug_steer --metric all/exact_accuracy
"""
from __future__ import annotations

import argparse
import itertools
import os

import numpy as np


def best_value(run, metric: str, mode: str):
    vals = [
        row[metric]
        for row in run.scan_history(keys=[metric])
        if row.get(metric) is not None
    ]
    if not vals:
        return None
    return float(max(vals) if mode == "max" else vals[-1])


def permutation_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided exact permutation test on the difference of means."""
    observed = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    n = len(a)
    count = total = 0
    for combo in itertools.combinations(range(len(pooled)), n):
        mask = np.zeros(len(pooled), dtype=bool)
        mask[list(combo)] = True
        diff = abs(pooled[mask].mean() - pooled[~mask].mean())
        count += diff >= observed - 1e-12
        total += 1
    return count / total


def summarize(name, vals):
    arr = np.array(vals, dtype=float)
    print(f"  {name}: n={len(arr)}  mean={arr.mean():.4f}  std={arr.std(ddof=1) if len(arr) > 1 else 0:.4f}  values={[round(v, 4) for v in vals]}")
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--project", required=True)
    ap.add_argument("--metric", default="all/exact_accuracy")
    ap.add_argument("--baseline", default="grok_baseline")
    ap.add_argument("--steer", default="grok_steer")
    ap.add_argument("--mode", choices=["max", "last"], default="max",
                    help="best-over-training (max) or final (last) value")
    args = ap.parse_args()

    import wandb

    api = wandb.Api()
    path = f"{args.entity}/{args.project}" if args.entity else args.project
    runs = list(api.runs(path))
    print(f"Found {len(runs)} runs in {path}\n")

    groups = {args.baseline: [], args.steer: []}
    for run in runs:
        name = run.name or ""
        for arm in groups:
            if name.startswith(arm):
                v = best_value(run, args.metric, args.mode)
                if v is not None:
                    groups[arm].append(v)
                else:
                    print(f"  (warn) {name}: metric '{args.metric}' not found; "
                          f"available e.g. {list(run.summary.keys())[:8]}")

    print(f"Metric: {args.metric}  ({args.mode})\n")
    arrays = {}
    for arm, vals in groups.items():
        if vals:
            arrays[arm] = summarize(arm, vals)
        else:
            print(f"  {arm}: no runs found")

    if len(arrays) == 2 and all(len(a) > 0 for a in arrays.values()):
        b, s = arrays[args.baseline], arrays[args.steer]
        delta = s.mean() - b.mean()
        print(f"\n  delta (STEER - baseline): {delta:+.4f} ({100 * delta:+.2f} pp)")
        if len(b) > 1 and len(s) > 1:
            p = permutation_pvalue(b, s)
            print(f"  exact permutation p-value: {p:.4f}")


if __name__ == "__main__":
    main()
