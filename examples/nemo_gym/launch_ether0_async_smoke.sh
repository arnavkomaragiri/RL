#!/usr/bin/env bash

# Submit an Ether0 async GRPO job on Slurm. Model assets and the raw dataset
# stay on shared storage; only the formatted training manifest is written
# under the run directory.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_LOCATION=${REPO_LOCATION:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
CONTAINER_REPO_LOCATION=${CONTAINER_REPO_LOCATION:-/opt/nemo-rl}

CONTAINER=${CONTAINER:-}
SLURM_ACCOUNT=${SLURM_ACCOUNT:-}
SLURM_PARTITION=${SLURM_PARTITION:-batch}
SLURM_COMMENT=${SLURM_COMMENT:-}
SLURM_DEPENDENCY=${SLURM_DEPENDENCY:-}
WALLTIME=${WALLTIME:-02:00:00}
NUM_NODES=${NUM_NODES:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NUM_GENERATION_NODES=${NUM_GENERATION_NODES:-2}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}
EXP_NAME=${EXP_NAME:-ether0-nemotron35-1x16-${RUN_ID}}
RUN_ROOT=${RUN_ROOT:-${REPO_LOCATION}/results/${EXP_NAME}}
RECIPE=${RECIPE:-examples/configs/recipes/llm/grpo-nemotron3.5-lightning-30ba3b-4n8g-megatron-async-gym-ether0-sc-tq-1x16-one-step.yaml}
WANDB_RUN_ID=${WANDB_RUN_ID:-${EXP_NAME}}
WANDB_RESUME=${WANDB_RESUME:-allow}

MODEL_PATH=${MODEL_PATH:-}
ETHER0_SOURCE_DATASET=${ETHER0_SOURCE_DATASET:-}
ETHER0_ASSET_ROOT=${ETHER0_ASSET_ROOT:-}
ETHER0_PROBLEM_TYPE=${ETHER0_PROBLEM_TYPE:-retro-synthesis}
ETHER0_LIMIT=${ETHER0_LIMIT:-1}
ETHER0_REMOTES_PYTHON=${ETHER0_REMOTES_PYTHON:-${ETHER0_ASSET_ROOT:+${ETHER0_ASSET_ROOT}/remotes-venv/bin/python}}
ETHER0_MOLTRANS_MODEL_PATH=${ETHER0_MOLTRANS_MODEL_PATH:-${ETHER0_ASSET_ROOT:+${ETHER0_ASSET_ROOT}/USPTO480k_model_step_400000.pt}}
ETHER0_REMOTES_RUNTIME_HOME=${ETHER0_REMOTES_RUNTIME_HOME:-${ETHER0_ASSET_ROOT:+${ETHER0_ASSET_ROOT}/runtime-home}}
ETHER0_NATIVE_LIBRARY_DIR=${ETHER0_NATIVE_LIBRARY_DIR:-${ETHER0_ASSET_ROOT:+${ETHER0_ASSET_ROOT}/native-libs}}

for name in CONTAINER SLURM_ACCOUNT MODEL_PATH ETHER0_SOURCE_DATASET ETHER0_ASSET_ROOT; do
    if [[ -z "${!name}" ]]; then
        echo "${name} is required." >&2
        exit 2
    fi
done
for path in \
    "${CONTAINER}" \
    "${MODEL_PATH}/config.json" \
    "${ETHER0_SOURCE_DATASET}" \
    "${ETHER0_REMOTES_PYTHON}" \
    "${ETHER0_MOLTRANS_MODEL_PATH}" \
    "${ETHER0_NATIVE_LIBRARY_DIR}/libXrender.so.1"; do
    if [[ ! -r "${path}" ]]; then
        echo "Required file is not readable: ${path}" >&2
        exit 2
    fi
done
if [[ ! -d "${ETHER0_REMOTES_RUNTIME_HOME}" ]]; then
    echo "Ether0 runtime home is not a directory: ${ETHER0_REMOTES_RUNTIME_HOME}" >&2
    exit 2
fi
if [[ ! -r "${REPO_LOCATION}/${RECIPE}" ]]; then
    echo "Recipe is not readable: ${REPO_LOCATION}/${RECIPE}" >&2
    exit 2
fi
if (( NUM_GENERATION_NODES < 1 || NUM_GENERATION_NODES >= NUM_NODES )); then
    echo "NUM_GENERATION_NODES must be between 1 and NUM_NODES-1." >&2
    exit 2
fi
if [[ -n "${SLURM_DEPENDENCY}" && ! "${SLURM_DEPENDENCY}" =~ ^(afterok|afterany):[0-9]+$ ]]; then
    echo "SLURM_DEPENDENCY must have the form afterok:<job-id> or afterany:<job-id>." >&2
    exit 2
fi
if [[ ! "${WANDB_RUN_ID}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "WANDB_RUN_ID must contain only letters, digits, '_', or '-'." >&2
    exit 2
fi
case "${WANDB_RESUME}" in
    allow | must | never | auto) ;;
    *)
        echo "WANDB_RESUME must be allow, must, never, or auto." >&2
        exit 2
        ;;
esac

MODEL_PATH=$(realpath -e -- "${MODEL_PATH}")
ETHER0_SOURCE_DATASET=$(realpath -e -- "${ETHER0_SOURCE_DATASET}")
ETHER0_ASSET_ROOT=$(realpath -e -- "${ETHER0_ASSET_ROOT}")
ETHER0_MOLTRANS_MODEL_PATH=$(realpath -e -- "${ETHER0_MOLTRANS_MODEL_PATH}")
ETHER0_REMOTES_RUNTIME_HOME=$(realpath -e -- "${ETHER0_REMOTES_RUNTIME_HOME}")
ETHER0_NATIVE_LIBRARY_DIR=$(realpath -e -- "${ETHER0_NATIVE_LIBRARY_DIR}")

mkdir -p \
    "${RUN_ROOT}/slurm" \
    "${RUN_ROOT}/logs" \
    "${RUN_ROOT}/checkpoints" \
    "${RUN_ROOT}/manifests" \
    "${RUN_ROOT}/nemo-gym-venvs" \
    "${RUN_ROOT}/megatron-checkpoints"
ETHER0_TRAIN_MANIFEST=${RUN_ROOT}/manifests/train.jsonl
if [[ ! -s "${ETHER0_TRAIN_MANIFEST}" ]]; then
    python3 "${REPO_LOCATION}/3rdparty/Gym-workspace/Gym/resources_servers/ether0/scripts/prepare_ether0.py" \
        --input-jsonl "${ETHER0_SOURCE_DATASET}" \
        --output "${ETHER0_TRAIN_MANIFEST}" \
        --problem-types "${ETHER0_PROBLEM_TYPE}" \
        --limit "${ETHER0_LIMIT}"
fi
if [[ $(wc -l < "${ETHER0_TRAIN_MANIFEST}") -ne "${ETHER0_LIMIT}" ]]; then
    echo "Expected exactly ${ETHER0_LIMIT} Ether0 rows in ${ETHER0_TRAIN_MANIFEST}." >&2
    exit 2
fi

export ETHER0_REMOTES_PYTHON ETHER0_MOLTRANS_MODEL_PATH ETHER0_REMOTES_RUNTIME_HOME ETHER0_TRAIN_MANIFEST
export ETHER0_NATIVE_LIBRARY_DIR
export LD_LIBRARY_PATH="${ETHER0_NATIVE_LIBRARY_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

SHARED_MOUNT=$(findmnt -n -o TARGET --target "${REPO_LOCATION}")
MOUNTS=${MOUNTS:-${SHARED_MOUNT}:${SHARED_MOUNT},${REPO_LOCATION}:${CONTAINER_REPO_LOCATION}}
read -r -d '' COMMAND <<EOF || true
set -euo pipefail
cd ${CONTAINER_REPO_LOCATION}
export GYM_ROOT=${CONTAINER_REPO_LOCATION}/3rdparty/Gym-workspace/Gym
export RAY_TMPDIR=/tmp/ray-${RUN_ID}
export NEMO_GYM_VENV_DIR=${RUN_ROOT}/nemo-gym-venvs
export NRL_MEGATRON_CHECKPOINT_DIR=${RUN_ROOT}/megatron-checkpoints
export ETHER0_REMOTES_PYTHON=${ETHER0_REMOTES_PYTHON}
export ETHER0_MOLTRANS_MODEL_PATH=${ETHER0_MOLTRANS_MODEL_PATH}
export ETHER0_REMOTES_RUNTIME_HOME=${ETHER0_REMOTES_RUNTIME_HOME}
export ETHER0_TRAIN_MANIFEST=${ETHER0_TRAIN_MANIFEST}
export ETHER0_NATIVE_LIBRARY_DIR=${ETHER0_NATIVE_LIBRARY_DIR}
export LD_LIBRARY_PATH=${ETHER0_NATIVE_LIBRARY_DIR}:\${LD_LIBRARY_PATH:-}
test -r "\${ETHER0_NATIVE_LIBRARY_DIR}/libXrender.so.1"
uv run python -u examples/run_grpo_single_controller.py \
    --config ${RECIPE} \
    policy.model_name=${MODEL_PATH} \
    cluster.num_nodes=${NUM_NODES} \
    policy.generation.colocated.resources.num_nodes=${NUM_GENERATION_NODES} \
    data.train.data_path=${ETHER0_TRAIN_MANIFEST} \
    data.validation.data_path=${ETHER0_TRAIN_MANIFEST} \
    logger.log_dir=${RUN_ROOT}/logs \
    logger.wandb.name=${EXP_NAME} \
    ++logger.wandb.id=${WANDB_RUN_ID} \
    ++logger.wandb.resume=${WANDB_RESUME} \
    checkpointing.checkpoint_dir=${RUN_ROOT}/checkpoints \
    ++env.nemo_gym.uv_venv_dir=${RUN_ROOT}/nemo-gym-venvs \
    ++env.nemo_gym.skip_venv_if_present=false
EOF

echo "Run name:          ${EXP_NAME}"
echo "Container:         ${CONTAINER}"
echo "Recipe:            ${RECIPE}"
echo "Model:             ${MODEL_PATH}"
echo "Nodes:             ${NUM_NODES} ($((NUM_NODES - NUM_GENERATION_NODES)) train / ${NUM_GENERATION_NODES} generation)"
echo "Ether0 task:       ${ETHER0_PROBLEM_TYPE}"
echo "Ether0 rows:       ${ETHER0_LIMIT}"
echo "Ether0 manifest:   ${ETHER0_TRAIN_MANIFEST}"
echo "Ether0 sidecar:    ${ETHER0_REMOTES_PYTHON}"
echo "Ether0 native lib: ${ETHER0_NATIVE_LIBRARY_DIR}"
echo "Run root:          ${RUN_ROOT}"
echo "W&B run ID:        ${WANDB_RUN_ID} (resume=${WANDB_RESUME})"
echo "Slurm dependency:  ${SLURM_DEPENDENCY:-none}"

SBATCH_ARGS=(
    --parsable
    --nodes="${NUM_NODES}"
    --account="${SLURM_ACCOUNT}"
    --partition="${SLURM_PARTITION}"
    --time="${WALLTIME}"
    --job-name="${EXP_NAME}"
    --gres="gpu:${GPUS_PER_NODE}"
    --output="${RUN_ROOT}/slurm/%j.out"
    --error="${RUN_ROOT}/slurm/%j.err"
    --export=ALL
)
if [[ -n "${SLURM_COMMENT}" ]]; then
    SBATCH_ARGS+=(--comment="${SLURM_COMMENT}")
fi
if [[ -n "${SLURM_DEPENDENCY}" ]]; then
    SBATCH_ARGS+=(--dependency="${SLURM_DEPENDENCY}")
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'Dry run: sbatch'
    printf ' %q' "${SBATCH_ARGS[@]}"
    printf ' %q\n' "${REPO_LOCATION}/ray.sub"
    exit 0
fi

JOB_ID=$(
    COMMAND="${COMMAND}" \
    CONTAINER="${CONTAINER}" \
    MOUNTS="${MOUNTS}" \
    BASE_LOG_DIR="${RUN_ROOT}/slurm" \
    GPUS_PER_NODE="${GPUS_PER_NODE}" \
    sbatch "${SBATCH_ARGS[@]}" "${REPO_LOCATION}/ray.sub"
)
echo "Submitted Slurm job ${JOB_ID}"
echo "Queue: squeue -j ${JOB_ID}"
echo "Logs:  ${RUN_ROOT}/slurm/${JOB_ID}.out"
