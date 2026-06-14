#!/usr/bin/env python
"""Aggregate Phase 2 results from Weights & Biases.

For a baseline arm and a STEER arm, pulls per-seed validation accuracy and
reports, for both the *peak* (early-stopping) and *final* (end-of-training)
value: mean +/- std and an exact permutation-test p-value (no SciPy).

Runs are grouped by run-name prefix (``grok_baseline`` vs ``grok_steer``),
matching scripts/launch_nrp.sh naming (``<exp>_s<seed>``). When a name has
several W&B runs (e.g. re-launched after preemption), only the most recently
created one is used.

Requires: ``pip install wandb`` and a W&B login (``WANDB_API_KEY`` env or
``wandb login``). Entity defaults to ``$WANDB_ENTITY``.

Examples:
  python scripts/collect_results.py --project STEER_PAPER_EXACT \
      --baseline grok_baseline --steer grok_steer
  python scripts/collect_results.py --project STEER_PAPER_AUGMENTED \
      --baseline aug_baseline --steer aug_steer
"""
from __future__ import annotations

import argparse
import itertools
import os

import numpy as np


def latest_runs_by_name(api, path, prefixes):
    """Most recently created run per name, restricted to the given prefixes."""
    latest = {}
    for run in api.runs(path):
        name = run.name or ""
        if name.startswith(prefixes):
            if name not in latest or run.created_at > latest[name].created_at:
                latest[name] = run
    return latest


def peak_and_final(run, metric):
    vals = [
        row[metric] for row in run.scan_history(keys=[metric]) if row.get(metric) is not None
    ]
    if not vals:
        return None, None
    return float(max(vals)), float(vals[-1])


def permutation_pvalue(a, b):
    """Two-sided exact permutation test on the difference of means."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    observed = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    n, count, total = len(a), 0, 0
    for combo in itertools.combinations(range(len(pooled)), n):
        mask = np.zeros(len(pooled), dtype=bool)
        mask[list(combo)] = True
        count += abs(pooled[mask].mean() - pooled[~mask].mean()) >= observed - 1e-12
        total += 1
    return count / total


def report(label, baseline, steer):
    b, s = np.array(baseline, float) * 100, np.array(steer, float) * 100
    print(f"\n=== {label} val-exact (%) ===")
    print(f"  baseline: {b.mean():6.2f} +/- {b.std(ddof=1) if len(b) > 1 else 0:.2f}   {np.round(b, 1)}")
    print(f"  STEER:    {s.mean():6.2f} +/- {s.std(ddof=1) if len(s) > 1 else 0:.2f}   {np.round(s, 1)}")
    delta = s.mean() - b.mean()
    line = f"  delta (STEER - baseline): {delta:+.2f} pp"
    if len(b) > 1 and len(s) > 1:
        line += f"   exact permutation p = {permutation_pvalue(b, s):.3f}"
    print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--project", required=True)
    ap.add_argument("--metric", default="all.exact_accuracy")
    ap.add_argument("--baseline", default="grok_baseline")
    ap.add_argument("--steer", default="grok_steer")
    args = ap.parse_args()

    import wandb

    api = wandb.Api()
    path = f"{args.entity}/{args.project}" if args.entity else args.project
    runs = latest_runs_by_name(api, path, (args.baseline, args.steer))

    arms = {args.baseline: {"peak": [], "final": []}, args.steer: {"peak": [], "final": []}}
    print(f"{'run':<22}{'peak%':>9}{'final%':>9}")
    for name in sorted(runs):
        peak, final = peak_and_final(runs[name], args.metric)
        if peak is None:
            print(f"{name:<22}  (metric '{args.metric}' not found)")
            continue
        arm = args.baseline if name.startswith(args.baseline) else args.steer
        arms[arm]["peak"].append(peak)
        arms[arm]["final"].append(final)
        print(f"{name:<22}{peak * 100:>8.2f}{final * 100:>9.2f}")

    if arms[args.baseline]["peak"] and arms[args.steer]["peak"]:
        report("PEAK (early-stopping)", arms[args.baseline]["peak"], arms[args.steer]["peak"])
        report("FINAL (end of training)", arms[args.baseline]["final"], arms[args.steer]["final"])


if __name__ == "__main__":
    main()
