#!/usr/bin/env bash
#
# Launch a STEER paper experiment as a Job on the NRP (Nautilus) Kubernetes
# cluster. The Job builds the dataset (reproducibly, seeded) inside the
# container, then runs distributed training via a per-experiment Hydra config.
#
# Usage:
#   scripts/launch_nrp.sh <experiment> [seed] [gpus] [image_tag]
#
# Examples:
#   scripts/launch_nrp.sh grok_steer 0          # grokking STEER, seed 0, 4 GPUs
#   scripts/launch_nrp.sh grok_baseline 1 4     # baseline, seed 1
#   scripts/launch_nrp.sh aug_steer 2 4 phase0  # augmented STEER, pinned tag
#
# Experiments are defined in config/experiment/*.yaml.
set -euo pipefail

EXP="${1:?experiment name required (grok_baseline|grok_steer|aug_baseline|aug_steer)}"
SEED="${2:-0}"
GPUS="${3:-4}"
IMAGE_TAG="${4:-latest}"
IMAGE="khurramkhalil/steer:${IMAGE_TAG}"
NS="gp-engine-mizzou-dcps"

# Dataset build parameters per regime.
case "$EXP" in
  grok_*) DATA_DIR="data/sudoku-extreme-1k-noaug";    SUB=1000; AUG=0 ;;
  aug_*)  DATA_DIR="data/sudoku-extreme-1k-aug-1000"; SUB=1000; AUG=1000 ;;
  *) echo "Unknown experiment: $EXP" >&2; exit 1 ;;
esac

JOB="steer-${EXP//_/-}-s${SEED}"
RUN_NAME="${EXP}_s${SEED}"

echo "Launching $JOB  (exp=$EXP seed=$SEED gpus=$GPUS image=$IMAGE)"

cat <<YAML | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB}
  namespace: ${NS}
spec:
  backoffLimit: 0
  template:
    metadata:
      labels:
        app: ${JOB}
    spec:
      restartPolicy: Never
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: nvidia.com/gpu.product
                operator: In
                values:
                - NVIDIA-A100-SXM4-80GB
                - NVIDIA-A100-80GB-PCIe
      tolerations:
      - key: "nautilus.io/reservation"
        operator: "Equal"
        value: "mizzou"
        effect: "NoSchedule"
      containers:
      - name: steer-train
        image: ${IMAGE}
        imagePullPolicy: Always
        workingDir: /workspace/STEER
        command: ["/bin/bash", "-c"]
        args:
          - |
            set -e
            echo "Building dataset (seed=${SEED})..."
            python dataset/build_sudoku_dataset.py \
              --output-dir ${DATA_DIR} --subsample-size ${SUB} --num-aug ${AUG} --seed ${SEED}
            echo "Training experiment=${EXP} seed=${SEED}..."
            torchrun --nproc_per_node=${GPUS} pretrain.py \
              experiment=${EXP} seed=${SEED} run_name=${RUN_NAME}
        resources:
          requests:
            nvidia.com/a100: ${GPUS}
            memory: "32Gi"
            cpu: "16"
          limits:
            nvidia.com/a100: ${GPUS}
            memory: "32Gi"
            cpu: "16"
        env:
        - name: PYTHONUNBUFFERED
          value: "1"
        - name: NCCL_P2P_DISABLE
          value: "1"
        - name: NCCL_DEBUG
          value: "WARN"
        envFrom:
        - secretRef:
            name: steer-secrets
        volumeMounts:
          - mountPath: /dev/shm
            name: dshm
      volumes:
        - name: dshm
          emptyDir:
            medium: Memory
YAML

echo "Submitted. Follow logs with:"
echo "  kubectl logs -n ${NS} -l app=${JOB} -f"
