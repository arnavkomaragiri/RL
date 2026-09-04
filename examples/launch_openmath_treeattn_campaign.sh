#!/usr/bin/env bash

# Submit a resumable OpenMathInstruct-2 GRPO campaign that exercises the
# SingleController + TransferQueue + Megatron exact-call tree-attention path.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SOURCE_REPO=${SOURCE_REPO:-$(cd "${SCRIPT_DIR}/.." && pwd)}
CONTAINER_REPO=${CONTAINER_REPO:-/opt/nemo-rl}

CONTAINER=${CONTAINER:-}
MODEL_PATH=${MODEL_PATH:-}
HF_CACHE_ROOT=${HF_CACHE_ROOT:-}
SLURM_ACCOUNT=${SLURM_ACCOUNT:-}
SLURM_PARTITION=${SLURM_PARTITION:-batch}
WALLTIME=${WALLTIME:-04:00:00}
NUM_STAGES=${NUM_STAGES:-16}
INITIAL_DEPENDENCY=${SLURM_DEPENDENCY:-}
DEPENDENCY_TYPE=${SLURM_DEPENDENCY_TYPE:-afterany}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}
EXP_NAME=${EXP_NAME:-openmath-qwen25-1p5b-treeattn-500step-${RUN_ID}}
RUN_ROOT=${RUN_ROOT:-${SOURCE_REPO}/results/${EXP_NAME}}
SNAPSHOT_DIR=${SNAPSHOT_DIR:-${SOURCE_REPO}/code_snapshots/${EXP_NAME}}
RECIPE=${RECIPE:-examples/configs/recipes/llm/grpo-qwen2.5-math-1.5b-instruct-1n8g-megatron-sc-tq-treeattn-500step.yaml}
WANDB_RUN_ID=${WANDB_RUN_ID:-${EXP_NAME}}

if [[ ! "${NUM_STAGES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_STAGES must be a positive integer." >&2
    exit 2
fi
if [[ "${DEPENDENCY_TYPE}" != "afterok" && "${DEPENDENCY_TYPE}" != "afterany" ]]; then
    echo "SLURM_DEPENDENCY_TYPE must be afterok or afterany." >&2
    exit 2
fi
if [[ -n "${INITIAL_DEPENDENCY}" && ! "${INITIAL_DEPENDENCY}" =~ ^(afterok|afterany):[0-9]+$ ]]; then
    echo "SLURM_DEPENDENCY must have the form afterok:<job-id> or afterany:<job-id>." >&2
    exit 2
fi
for name in CONTAINER MODEL_PATH HF_CACHE_ROOT SLURM_ACCOUNT; do
    if [[ -z "${!name}" ]]; then
        echo "${name} is required." >&2
        exit 2
    fi
done
for path in "${CONTAINER}" "${MODEL_PATH}/config.json" "${SOURCE_REPO}/${RECIPE}"; do
    if [[ ! -r "${path}" ]]; then
        echo "Required file is not readable: ${path}" >&2
        exit 2
    fi
done

MODEL_PATH=$(realpath -e -- "${MODEL_PATH}")
mkdir -p \
    "${HF_CACHE_ROOT}" \
    "${RUN_ROOT}/slurm" \
    "${RUN_ROOT}/logs" \
    "${RUN_ROOT}/checkpoints" \
    "${RUN_ROOT}/megatron-checkpoints"
HF_CACHE_ROOT=$(realpath -e -- "${HF_CACHE_ROOT}")

if [[ ! -d "${SNAPSHOT_DIR}" ]]; then
    SNAPSHOT_DIR=$(bash "${SOURCE_REPO}/tools/code_snapshot.sh" "${EXP_NAME}")
    # New experiment files may not be tracked yet, while the snapshot helper
    # intentionally copies only git-known paths.
    mkdir -p "${SNAPSHOT_DIR}/$(dirname "${RECIPE}")"
    cp "${SOURCE_REPO}/${RECIPE}" "${SNAPSHOT_DIR}/${RECIPE}"
fi
SNAPSHOT_DIR=$(realpath -e -- "${SNAPSHOT_DIR}")
if [[ ! -r "${SNAPSHOT_DIR}/${RECIPE}" ]]; then
    echo "Recipe is missing from snapshot: ${SNAPSHOT_DIR}/${RECIPE}" >&2
    exit 2
fi

SHARED_MOUNT=$(findmnt -n -o TARGET --target "${SOURCE_REPO}")
MOUNTS=${MOUNTS:-${SHARED_MOUNT}:${SHARED_MOUNT},${SNAPSHOT_DIR}:${CONTAINER_REPO}}

echo "Campaign:       ${EXP_NAME}"
echo "Container:      ${CONTAINER}"
echo "Snapshot:       ${SNAPSHOT_DIR}"
echo "Recipe:         ${RECIPE}"
echo "Model:          ${MODEL_PATH}"
echo "HF cache:       ${HF_CACHE_ROOT}"
echo "Run root:       ${RUN_ROOT}"
echo "Stages:         ${NUM_STAGES} x ${WALLTIME} (${DEPENDENCY_TYPE})"
echo "W&B run ID:     ${WANDB_RUN_ID}"

dependency=${INITIAL_DEPENDENCY}
job_ids=()
for ((stage = 1; stage <= NUM_STAGES; stage++)); do
    printf -v stage_suffix 's%02d' "${stage}"
    read -r -d '' COMMAND <<EOF || true
set -euo pipefail
cd ${CONTAINER_REPO}
export RAY_TMPDIR=/tmp/ray-${RUN_ID}-${stage_suffix}
export NRL_MEGATRON_CHECKPOINT_DIR=${RUN_ROOT}/megatron-checkpoints
export HF_HOME=${HF_CACHE_ROOT}
export HF_DATASETS_CACHE=${HF_CACHE_ROOT}/datasets
export HF_HUB_OFFLINE=1
uv run python -u examples/run_grpo_single_controller.py \
    --config ${RECIPE} \
    policy.model_name=${MODEL_PATH} \
    policy.tokenizer.name=${MODEL_PATH} \
    logger.log_dir=${RUN_ROOT}/logs \
    logger.wandb.name=${EXP_NAME} \
    ++logger.wandb.id=${WANDB_RUN_ID} \
    ++logger.wandb.resume=allow \
    checkpointing.checkpoint_dir=${RUN_ROOT}/checkpoints
EOF

    sbatch_args=(
        --parsable
        --nodes=1
        --account="${SLURM_ACCOUNT}"
        --partition="${SLURM_PARTITION}"
        --time="${WALLTIME}"
        --job-name="${EXP_NAME}-${stage_suffix}"
        --gres=gpu:8
        --output="${RUN_ROOT}/slurm/%j.out"
        --error="${RUN_ROOT}/slurm/%j.err"
        --export=ALL
    )
    if [[ -n "${dependency}" ]]; then
        sbatch_args+=(--dependency="${dependency}")
    fi

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        printf 'Dry run stage %d: sbatch' "${stage}"
        printf ' %q' "${sbatch_args[@]}"
        printf ' %q\n' "${SOURCE_REPO}/ray.sub"
        continue
    fi

    job_id=$(
        COMMAND="${COMMAND}" \
        CONTAINER="${CONTAINER}" \
        MOUNTS="${MOUNTS}" \
        BASE_LOG_DIR="${RUN_ROOT}/slurm" \
        GPUS_PER_NODE=8 \
        sbatch "${sbatch_args[@]}" "${SOURCE_REPO}/ray.sub"
    )
    echo "Submitted stage ${stage}: ${job_id}"
    job_ids+=("${job_id}")
    dependency="${DEPENDENCY_TYPE}:${job_id}"
done

if (( ${#job_ids[@]} > 0 )); then
    echo "Jobs: ${job_ids[*]}"
    echo "Logs: ${RUN_ROOT}/slurm/<job-id>.out"
fi
