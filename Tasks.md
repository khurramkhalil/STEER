# STEER — Path to a Top-Tier ML Venue

Phased, checkable task list to take STEER from "working code" to "submittable
paper". Work top-down: later phases depend on earlier ones. Check items off as
they land; keep acceptance criteria honest.

Legend: `[ ]` todo · `[~]` in progress · `[x]` done

---

## Phase 0 — Code integrity & correctness  *(foundation — mostly done)*

Goal: the implementation does what the paper will claim it does.

- [x] STEER loss actually applies during training, with gradients (was a no-op).
- [x] Fix `NameError` (metrics) and trivially-satisfied convergence term.
- [x] Unit/integration test suite (`pytest`, 25 tests) incl. gradient-flow regression.
- [x] Pin dependencies; remove scratch files; organize repo.
- [ ] **Smoke test on the real model**: train ~100 steps with `steer_lambda>0` and
      confirm (a) `steer/loss` is nonzero and trends down, (b) all `steer/*`
      metrics log without error, (c) a run with `steer_lambda=0` matches the
      pre-change baseline behavior.
- [ ] **DDP metric aggregation**: verify `steer/*` (means) are reduced correctly
      across ranks/micro-batches in `pretrain.py` (task metrics are sums) — fix
      or document.
- [ ] **CI**: GitHub Actions running `pytest` on every push/PR.

**Exit criteria:** STEER provably trains; CI green; no known correctness bugs.

---

## Phase 1 — Reproducibility infrastructure

Goal: anyone (incl. reviewers) can reproduce a result with one command.

- [ ] Per-experiment config files in `config/experiments/*.yaml` (no long CLI in README).
- [ ] `scripts/reproduce_<table>.sh` for each results table; documented in README.
- [ ] Deterministic seeding path verified (torch, numpy, cuDNN); `--seed` exposed
      and honored across DDP ranks.
- [ ] Environment fully pinned & verified: `Dockerfile` builds and runs a smoke
      train; document exact torch/CUDA versions (no nightly).
- [ ] W&B offline mode supported; consistent project/run naming.
- [ ] Release trained checkpoints + a `scripts/download_checkpoints.sh`.

**Exit criteria:** clean-machine `bash scripts/reproduce_*.sh` reproduces a logged number.

---

## Phase 2 — Re-establish core results  *(CRITICAL — prior numbers are invalid)*

Goal: real, statistically-backed STEER-vs-baseline numbers, since the published
tables came from the no-op STEER.

- [ ] Re-run **grokking regime**: baseline vs STEER (λ=0.1), ≥3 seeds each.
- [ ] Re-run **augmented regime**: baseline vs STEER (λ=1.0), ≥3 seeds each.
- [ ] Report **mean ± std** and a **significance test** (paired bootstrap / t-test).
- [ ] Re-evaluate the "EMA required" claim with working STEER.
- [ ] Update `README.md` + `experiments_log.md` with verified numbers; delete or
      clearly retract the old (invalid) tables.

**Exit criteria:** every headline number has ≥3 seeds, a CI/std, and a p-value.

---

## Phase 3 — Ablations & mechanistic analysis

Goal: show *which* part of STEER matters and *why*, and back the "Efficient" claim.

- [ ] **Component ablation**: each STL term alone (validity / stability /
      convergence / monotonic-progress) and combinations.
- [ ] **λ sweep** with working STEER: {0, 0.01, 0.1, 1.0} × both regimes.
- [ ] **ε sensitivity** (`epsilon_viol`, `epsilon_stab`); **root-cause the tight-ε
      NCCL crash** (currently only a workaround), fix or bound it.
- [ ] **Halting / compute efficiency**: measure real steps-to-converge and
      `compute_savings`; this is the "Self-Terminating / Efficient" claim.
- [ ] **Trajectory plots**: violation / progress / stability vs reasoning step,
      STEER vs baseline.

**Exit criteria:** a reviewer can see the contribution of each component and the efficiency gain.

---

## Phase 4 — Generalization beyond Sudoku

Goal: STEER is a *framework*, not a Sudoku trick. Currently `SudokuSignals` is hardcoded.

- [ ] Refactor signals into a task-agnostic interface (e.g. `Signals` base class)
      so the STL/loss layer is task-independent.
- [ ] Add a **second benchmark**: Maze (`dataset/build_maze_dataset.py` exists) —
      define maze-specific validity/progress/stability signals.
- [ ] (Optional/stretch) ARC signals.
- [ ] Show STEER helps on ≥2 tasks with the same machinery.

**Exit criteria:** results on ≥2 distinct reasoning tasks via a shared STEER core.

---

## Phase 5 — Method formalization

Goal: the novelty (STL robustness as a reasoning regularizer) is rigorously stated.

- [ ] Formal definitions: the three properties, STL robustness semantics
      (G→min, F→max), and the hinge surrogate `relu(-ρ)` justification.
- [ ] State assumptions/limitations of the differentiable signal relaxations.
- [ ] Related work: STL in ML, ACT/adaptive halting, recursive reasoning (TRM/HRM),
      regularization for grokking/generalization.

**Exit criteria:** method section is self-contained and defensible.

---

## Phase 6 — Manuscript

Goal: the paper itself.

- [ ] Paper skeleton: abstract, intro, method, experiments, related work,
      limitations, conclusion.
- [ ] Figures: method/architecture diagram, trajectory plots, results tables w/ CIs,
      ablation table.
- [ ] **Limitations & broader impact** section (honest: task-specific signals,
      ε instability, gain size by regime).
- [ ] Reproducibility checklist (NeurIPS/ICML style).
- [ ] BibTeX: fill the TRM citation properly; add STEER entry.

**Exit criteria:** complete draft ready for internal review.

---

## Phase 7 — Release & submission polish

Goal: artifact + submission hygiene.

- [ ] Decide `TRM/` fate: git submodule vs vendored vs external dependency; document it.
- [ ] Data redistribution check for bundled `kaggle/` ARC-AGI JSON (license).
- [ ] End-to-end README quickstart verified on a fresh machine.
- [ ] Anonymize repo/manuscript if the venue is double-blind.
- [ ] Tag a release; attach checkpoints.

**Exit criteria:** submission-ready repo + camera-ready-able artifact.

---

### Current standing (2026-06-10)
- Phase 0: ~70% (smoke test, DDP metric check, CI remain).
- Phases 1–7: not started.
- **Blocking risk:** Phase 2 — published results are from the no-op STEER and
  must be regenerated before any claim is made.
