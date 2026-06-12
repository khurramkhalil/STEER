#!/usr/bin/env bash
#
# Reproduce a STEER paper table by launching every seed of the relevant
# baseline and STEER experiments on the NRP cluster.
#
# Usage:
#   scripts/reproduce.sh <grok|aug> [num_seeds] [gpus] [image_tag]
#
# Examples:
#   scripts/reproduce.sh grok 3     # grok_baseline + grok_steer, seeds 0,1,2
#   scripts/reproduce.sh aug 3
set -euo pipefail

TABLE="${1:?table required: grok|aug}"
NUM_SEEDS="${2:-3}"
GPUS="${3:-4}"
TAG="${4:-latest}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$TABLE" in
  grok) EXPS=(grok_baseline grok_steer) ;;
  aug)  EXPS=(aug_baseline aug_steer) ;;
  *) echo "Unknown table: $TABLE (expected grok|aug)" >&2; exit 1 ;;
esac

for exp in "${EXPS[@]}"; do
  for ((s = 0; s < NUM_SEEDS; s++)); do
    "$HERE/launch_nrp.sh" "$exp" "$s" "$GPUS" "$TAG"
  done
done

echo "Launched ${#EXPS[@]} experiments x ${NUM_SEEDS} seeds for table '${TABLE}'."
