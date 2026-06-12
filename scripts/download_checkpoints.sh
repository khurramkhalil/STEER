#!/usr/bin/env bash
#
# Download released STEER checkpoints from the Hugging Face Hub.
#
# NOTE: checkpoints are released after the Phase 2 multi-seed runs complete
# (see Tasks.md). Until then this is a scaffold -- set HF_REPO to the published
# repo id once it exists. Training already uploads checkpoints to the Hub
# (see the HfApi upload path in pretrain.py).
#
# Usage:
#   HF_REPO=<org>/steer-checkpoints scripts/download_checkpoints.sh [dest_dir]
set -euo pipefail

HF_REPO="${HF_REPO:-}"
DEST="${1:-checkpoints}"

if [ -z "$HF_REPO" ]; then
  echo "HF_REPO is not set. Once checkpoints are published, run:" >&2
  echo "  HF_REPO=<org>/steer-checkpoints scripts/download_checkpoints.sh" >&2
  exit 1
fi

mkdir -p "$DEST"
python - "$HF_REPO" "$DEST" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id, dest = sys.argv[1], sys.argv[2]
path = snapshot_download(repo_id=repo_id, repo_type="model", local_dir=dest)
print(f"Downloaded {repo_id} -> {path}")
PY
