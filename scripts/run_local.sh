#!/usr/bin/env bash
#
# Run a STEER experiment on the local node (single machine, 1+ GPUs). Builds the
# dataset if missing, then trains via the per-experiment Hydra config. Useful for
# debugging; the paper runs use scripts/launch_nrp.sh on the cluster.
#
# Usage:
#   scripts/run_local.sh <experiment> [seed] [gpus] [extra hydra overrides...]
#
# Examples:
#   scripts/run_local.sh grok_steer 0 1 epochs=50 eval_interval=50 +log_interval=10
set -euo pipefail

EXP="${1:?experiment required (grok_baseline|grok_steer|aug_baseline|aug_steer)}"
SEED="${2:-0}"
GPUS="${3:-1}"
shift $(( $# < 3 ? $# : 3 )) || true   # remaining args are extra Hydra overrides

case "$EXP" in
  grok_*) DATA_DIR="data/sudoku-extreme-1k-noaug";    SUB=1000; AUG=0 ;;
  aug_*)  DATA_DIR="data/sudoku-extreme-1k-aug-1000"; SUB=1000; AUG=1000 ;;
  *) echo "Unknown experiment: $EXP" >&2; exit 1 ;;
esac

if [ ! -d "$DATA_DIR" ]; then
  echo "Building dataset $DATA_DIR (seed=$SEED)..."
  python dataset/build_sudoku_dataset.py \
    --output-dir "$DATA_DIR" --subsample-size "$SUB" --num-aug "$AUG" --seed "$SEED"
fi

torchrun --nproc_per_node="$GPUS" pretrain.py experiment="$EXP" seed="$SEED" run_name="${EXP}_s${SEED}" "$@"
