#!/usr/bin/env bash

# Submit the one-step BBH Harbor/OpenSandbox async GRPO diagnostic on Slurm.
# OpenSandbox secrets are inherited by sbatch and are never embedded in COMMAND.
# OPENSANDBOX_API_KEY_FILE must contain only the raw key value, for example:
#   osb_example_key_value
# Do not include OPENSANDBOX_API_KEY=, quotes, or other dotenv syntax.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_LOCATION=${REPO_LOCATION:-$(cd "${SCRIPT_DIR}/../.." && pwd)}

CONTAINER=${CONTAINER:-}
SLURM_ACCOUNT=${SLURM_ACCOUNT:-}
SLURM_PARTITION=${SLURM_PARTITION:-batch}
WALLTIME=${WALLTIME:-02:00:00}
NUM_NODES=${NUM_NODES:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NUM_GENERATION_NODES=${NUM_GENERATION_NODES:-2}
NUM_PROMPTS_PER_STEP=${NUM_PROMPTS_PER_STEP:-2}
NUM_GENERATIONS_PER_PROMPT=${NUM_GENERATIONS_PER_PROMPT:-2}
TRAIN_GLOBAL_BATCH_SIZE=$((NUM_PROMPTS_PER_STEP * NUM_GENERATIONS_PER_PROMPT))

RUN_ID=${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}
EXP_NAME=${EXP_NAME:-harbor-bbh-tokcap-${RUN_ID}}
RUN_ROOT=${RUN_ROOT:-${REPO_LOCATION}/results/${EXP_NAME}}
RECIPE=${RECIPE:-examples/configs/recipes/llm/grpo-qwen3-30ba3b-thinking-4n8g-megatron-cp2-r3-async-gym-harbor-bbh-smoke.yaml}
MODEL_PATH=${MODEL_PATH:-}

HARBOR_DATASET_PATH=${HARBOR_DATASET_PATH:-}
HARBOR_BENCHMARK_NAME=${HARBOR_BENCHMARK_NAME:-bbh-harbor-rl-v0}
OPENSANDBOX_PROTOCOL=${OPENSANDBOX_PROTOCOL:-http}
OPENSANDBOX_USE_SERVER_PROXY=${OPENSANDBOX_USE_SERVER_PROXY:-true}

if [[ -z "${OPENSANDBOX_API_KEY:-}" && -n "${OPENSANDBOX_API_KEY_FILE:-}" ]]; then
    if [[ ! -r "${OPENSANDBOX_API_KEY_FILE}" ]]; then
        echo "OPENSANDBOX_API_KEY_FILE is not readable: ${OPENSANDBOX_API_KEY_FILE}" >&2
        exit 2
    fi
    OPENSANDBOX_API_KEY=$(<"${OPENSANDBOX_API_KEY_FILE}")
fi

if [[ -z "${OPENSANDBOX_DOMAIN:-}" ]]; then
    echo "OPENSANDBOX_DOMAIN is required (hostname only; set OPENSANDBOX_PROTOCOL separately)." >&2
    exit 2
fi
if [[ -z "${OPENSANDBOX_API_KEY:-}" ]]; then
    echo "Set OPENSANDBOX_API_KEY or OPENSANDBOX_API_KEY_FILE." >&2
    exit 2
fi
if [[ -z "${CONTAINER}" ]]; then
    echo "CONTAINER is required and must point to a NeMo-RL squashfs image." >&2
    exit 2
fi
if [[ -z "${SLURM_ACCOUNT}" ]]; then
    echo "SLURM_ACCOUNT is required." >&2
    exit 2
fi
if [[ -z "${MODEL_PATH}" ]]; then
    echo "MODEL_PATH is required and must point to a Hugging Face checkpoint." >&2
    exit 2
fi
if [[ -z "${HARBOR_DATASET_PATH}" ]]; then
    echo "HARBOR_DATASET_PATH is required." >&2
    exit 2
fi
if [[ ! -r "${CONTAINER}" ]]; then
    echo "Container is not readable: ${CONTAINER}" >&2
    exit 2
fi
if [[ ! -d "${HARBOR_DATASET_PATH}" ]]; then
    echo "Harbor dataset is not a directory: ${HARBOR_DATASET_PATH}" >&2
    exit 2
fi
if [[ ! -f "${REPO_LOCATION}/${RECIPE}" ]]; then
    echo "Recipe does not exist: ${REPO_LOCATION}/${RECIPE}" >&2
    exit 2
fi
if [[ ! -r "${MODEL_PATH}/config.json" ]]; then
    echo "Model checkpoint is not readable: ${MODEL_PATH}" >&2
    exit 2
fi

case "${OPENSANDBOX_USE_SERVER_PROXY}" in
    true|false) ;;
    *)
        echo "OPENSANDBOX_USE_SERVER_PROXY must be true or false." >&2
        exit 2
        ;;
esac

if (( NUM_GENERATION_NODES < 1 || NUM_GENERATION_NODES >= NUM_NODES )); then
    echo "NUM_GENERATION_NODES must be between 1 and NUM_NODES-1." >&2
    exit 2
fi
TRAINING_GPUS=$(((NUM_NODES - NUM_GENERATION_NODES) * GPUS_PER_NODE))
if (( TRAINING_GPUS < 16 )); then
    echo "The TP2 x PP2 x CP4 training topology requires at least 16 training GPUs." >&2
    exit 2
fi

export HARBOR_DATASET_PATH HARBOR_BENCHMARK_NAME
export OPENSANDBOX_DOMAIN OPENSANDBOX_API_KEY
export OPENSANDBOX_PROTOCOL OPENSANDBOX_USE_SERVER_PROXY

SHARED_MOUNT=$(findmnt -n -o TARGET --target "${REPO_LOCATION}")
MOUNTS=${MOUNTS:-${SHARED_MOUNT}:${SHARED_MOUNT}}
mkdir -p "${RUN_ROOT}/slurm" "${RUN_ROOT}/logs" "${RUN_ROOT}/checkpoints"

read -r -d '' COMMAND <<EOF || true
set -euo pipefail
cd ${REPO_LOCATION}
export GYM_ROOT=${REPO_LOCATION}/3rdparty/Gym-workspace/Gym
export HARBOR_DATASET_PATH=${HARBOR_DATASET_PATH}
export HARBOR_BENCHMARK_NAME=${HARBOR_BENCHMARK_NAME}
export RAY_TMPDIR=/tmp/ray-${RUN_ID}
export NEMO_GYM_VENV_DIR=/tmp/nemo-gym-venvs-${RUN_ID}
uv run python -u examples/run_grpo.py \\
    --config ${RECIPE} \\
    policy.model_name=${MODEL_PATH} \\
    cluster.num_nodes=${NUM_NODES} \\
    policy.generation.colocated.resources.num_nodes=${NUM_GENERATION_NODES} \\
    grpo.num_prompts_per_step=${NUM_PROMPTS_PER_STEP} \\
    grpo.num_generations_per_prompt=${NUM_GENERATIONS_PER_PROMPT} \\
    policy.train_global_batch_size=${TRAIN_GLOBAL_BATCH_SIZE} \\
    policy.generation_batch_size=${TRAIN_GLOBAL_BATCH_SIZE} \\
    env.nemo_gym.num_samples_in_parallel=${TRAIN_GLOBAL_BATCH_SIZE} \\
    logger.log_dir=${RUN_ROOT}/logs \\
    logger.wandb.name=${EXP_NAME} \\
    checkpointing.checkpoint_dir=${RUN_ROOT}/checkpoints \\
    env.nemo_gym.skip_venv_if_present=false
EOF

echo "Run name:             ${EXP_NAME}"
echo "Container:            ${CONTAINER}"
echo "Recipe:               ${RECIPE}"
echo "Model:                ${MODEL_PATH}"
echo "Nodes/GPUs:            ${NUM_NODES} x ${GPUS_PER_NODE}"
echo "Train/generation:      $((NUM_NODES - NUM_GENERATION_NODES)) / ${NUM_GENERATION_NODES} nodes"
echo "Rollouts:              ${NUM_PROMPTS_PER_STEP} x ${NUM_GENERATIONS_PER_PROMPT} = ${TRAIN_GLOBAL_BATCH_SIZE}"
echo "Partition/account:     ${SLURM_PARTITION} / ${SLURM_ACCOUNT}"
echo "OpenSandbox endpoint:  ${OPENSANDBOX_PROTOCOL}://${OPENSANDBOX_DOMAIN}"
echo "OpenSandbox proxy:     ${OPENSANDBOX_USE_SERVER_PROXY}"
echo "OpenSandbox API key:   set (value suppressed)"
echo "Dataset:               ${HARBOR_DATASET_PATH}"
echo "Run root:              ${RUN_ROOT}"

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
