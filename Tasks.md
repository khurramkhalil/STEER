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
- [x] **Smoke test on the real model** (NRP cluster, 2-GPU, `khurramkhalil/steer:phase0`):
      confirmed `steer/loss` is nonzero (~21.7) so STEER actually contributes to
      the loss, all `steer/*` metrics log without error, and `prog`/`viol`/`rho`
      signals behave sensibly. NOTE: "trends down" is *not* observable in ~90
      steps under the 2000-step LR warmup (model has barely started) — that is a
      Phase 2 convergence question, not a plumbing check. The `steer_lambda=0`
      path is covered by unit tests (`test_steer_disabled_emits_no_steer_metrics`).
- [x] **DDP metric aggregation**: fixed — `steer/*` are per-batch means and are
      now normalized by `world_size` (not the example count). Verified empirically
      on 2 GPUs (values on a sane O(1-10) scale, not ~512x smaller). STEER is also
      gated to training mode so eval aggregation is unaffected.
- [x] **CI**: `.github/workflows/ci.yml` runs `pytest` on push/PR (activates once pushed).

**Exit criteria:** STEER provably trains; CI green; no known correctness bugs. ✅ MET

---

## Phase 1 — Reproducibility infrastructure  ✅ COMPLETE

Goal: anyone (incl. reviewers) can reproduce a result with one command.

- [x] Per-experiment Hydra configs in `config/experiment/*.yaml`
      (`grok_baseline`, `grok_steer`, `aug_baseline`, `aug_steer`) — no long CLI.
      Composition validated via `--cfg job` in the container.
- [x] Reproduce scripts: `scripts/launch_nrp.sh` (one exp/seed),
      `scripts/reproduce.sh` (a whole table x seeds), `scripts/run_local.sh`;
      documented in the README.
- [x] Deterministic seeding: `seed_everything` seeds python/numpy/torch(+cuda),
      `seed+rank` per rank; optional `deterministic=True` for cuDNN/cuBLAS; the
      dataset builder is seeded via `--seed` so datasets are reproducible.
- [x] Environment pinned & verified: image built on `torch 2.5.1+cu124`; triton
      left to torch (now 3.1.0, coherent — un-breaks `torch.compile`); README +
      requirements aligned; image rebuilt & pushed (`khurramkhalil/steer:phase1`,
      `:latest`).
- [x] W&B offline supported (`WANDB_MODE=offline`); consistent project/run naming
      from the experiment configs; stdout logging every `log_interval` steps.
- [~] Checkpoint release: `scripts/download_checkpoints.sh` scaffolded; the actual
      checkpoints are published after the Phase 2 runs exist (training already
      uploads to the HF Hub).

**Exit criteria:** one-command reproduction via `scripts/reproduce.sh <table>`. ✅ MET
(checkpoint publishing deferred to post-Phase-2, by necessity).

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

### Current standing (2026-06-12)
- Phase 0: ✅ COMPLETE. STEER verified to actually train on the real model
  (NRP 2-GPU smoke run); DDP metric aggregation fixed; CI added.
- Phase 1: ✅ COMPLETE. Per-experiment configs + reproduce scripts + seeding +
  coherent pinned image (`khurramkhalil/steer:phase1`). One-command reproduction.
- Build/run infra established: Brev builds & pushes `khurramkhalil/steer:*`;
  jobs run on the NRP `gp-engine-mizzou-dcps` namespace (see `deploy/k8s/` and
  `scripts/`).
- **NEXT: Phase 2** — re-establish core results. This is the blocking item:
  published tables came from the no-op STEER and must be regenerated (≥3 seeds,
  mean±std, significance) before any claim is made. Launch with
  `scripts/reproduce.sh grok 3` then `scripts/reproduce.sh aug 3`.
