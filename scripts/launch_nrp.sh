#!/usr/bin/env bash
#
# Launch a STEER paper experiment as a Job on the NRP (Nautilus) Kubernetes
# cluster. Datasets and checkpoints live on a shared RWX PVC (steer-data) so
# runs survive pod eviction/preemption: the job auto-resumes from the latest
# resume checkpoint, and backoffLimit lets Kubernetes restart the pod.
#
# Prereqs (once): kubectl apply -f deploy/k8s/pvc.yaml
#
# Usage:
#   scripts/launch_nrp.sh <experiment> [seed] [gpus] [image_tag]
#
# Examples:
#   scripts/launch_nrp.sh grok_steer 0
#   scripts/launch_nrp.sh aug_steer 2 4 phase1
set -euo pipefail

EXP="${1:?experiment name required (grok_baseline|grok_steer|aug_baseline|aug_steer)}"
SEED="${2:-0}"
GPUS="${3:-4}"
IMAGE_TAG="${4:-latest}"
IMAGE="khurramkhalil/steer:${IMAGE_TAG}"
NS="gp-engine-mizzou-dcps"
TEST_SUB=1000   # subsample the test split so periodic eval is cheap

# Dataset build parameters per regime. The dataset uses a FIXED seed so all
# training seeds share the same train/test split (only training varies by seed).
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
  backoffLimit: 6          # restart on eviction/preemption; job auto-resumes
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
            # Build dataset once (idempotent: skip if already on the PVC).
            if [ ! -f ${DATA_DIR}/.done ]; then
              echo "Building dataset ${DATA_DIR}..."
              python dataset/build_sudoku_dataset.py \
                --output-dir ${DATA_DIR} --subsample-size ${SUB} \
                --test-subsample-size ${TEST_SUB} --num-aug ${AUG} --seed 0
            else
              echo "Dataset ${DATA_DIR} already present; skipping build."
            fi
            echo "Training experiment=${EXP} seed=${SEED} (auto-resume on)..."
            torchrun --nproc_per_node=${GPUS} pretrain.py \
              experiment=${EXP} seed=${SEED} run_name=${RUN_NAME} \
              checkpoint_every_eval=False
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
          - { name: steer-data, mountPath: /workspace/STEER/data, subPath: data }
          - { name: steer-data, mountPath: /workspace/STEER/checkpoints, subPath: checkpoints }
          - { name: dshm, mountPath: /dev/shm }
      volumes:
        - name: steer-data
          persistentVolumeClaim:
            claimName: steer-data
        - name: dshm
          emptyDir:
            medium: Memory
YAML

echo "Submitted. Follow logs with:"
echo "  kubectl logs -n ${NS} -l app=${JOB} -f"
