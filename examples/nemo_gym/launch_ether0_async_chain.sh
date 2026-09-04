#!/usr/bin/env bash

# Submit sequential fault-tolerant Ether0 async-RL stages. The shared run
# directory carries model, optimizer, and completed-TQ-group checkpoints.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LAUNCHER=${LAUNCHER:-${SCRIPT_DIR}/launch_ether0_async_smoke.sh}
NUM_STAGES=${NUM_STAGES:-9}
CAMPAIGN_ID=${CAMPAIGN_ID:-$(date -u +%Y%m%d-%H%M%S)}
EXP_NAME=${EXP_NAME:-ether0-nemotron35-age4-dp2-${CAMPAIGN_ID}}
REPO_LOCATION=${REPO_LOCATION:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
RUN_ROOT=${RUN_ROOT:-${REPO_LOCATION}/results/${EXP_NAME}}
WANDB_RUN_ID=${WANDB_RUN_ID:-${EXP_NAME}}
INITIAL_DEPENDENCY=${SLURM_DEPENDENCY:-}
DEPENDENCY_TYPE=${SLURM_DEPENDENCY_TYPE:-afterany}
RECIPE=${RECIPE:-examples/configs/recipes/llm/grpo-nemotron3.5-lightning-30ba3b-6n8g-megatron-async-gym-ether0-sc-tq-32x16-age4-r3-100step.yaml}
NUM_NODES=${NUM_NODES:-6}
NUM_GENERATION_NODES=${NUM_GENERATION_NODES:-2}
WALLTIME=${WALLTIME:-04:00:00}

if [[ ! "${NUM_STAGES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_STAGES must be a positive integer." >&2
    exit 2
fi
if [[ "${DEPENDENCY_TYPE}" != "afterok" && "${DEPENDENCY_TYPE}" != "afterany" ]]; then
    echo "SLURM_DEPENDENCY_TYPE must be afterok or afterany." >&2
    exit 2
fi

dependency=${INITIAL_DEPENDENCY}
job_ids=()
for ((stage = 1; stage <= NUM_STAGES; stage++)); do
    printf -v stage_suffix 's%02d' "${stage}"
    stage_output=$(
        RUN_ID="${CAMPAIGN_ID}-${stage_suffix}" \
        EXP_NAME="${EXP_NAME}" \
        RUN_ROOT="${RUN_ROOT}" \
        WANDB_RUN_ID="${WANDB_RUN_ID}" \
        WANDB_RESUME=allow \
        RECIPE="${RECIPE}" \
        ETHER0_LIMIT="${ETHER0_LIMIT:-3200}" \
        NUM_NODES="${NUM_NODES}" \
        NUM_GENERATION_NODES="${NUM_GENERATION_NODES}" \
        WALLTIME="${WALLTIME}" \
        SLURM_PARTITION="${SLURM_PARTITION:-batch}" \
        SLURM_DEPENDENCY="${dependency}" \
        "${LAUNCHER}"
    )
    printf '%s\n' "${stage_output}"
    job_id=$(awk '/^Submitted Slurm job / {print $4}' <<<"${stage_output}")
    if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
        echo "Could not parse stage ${stage} job ID from launcher output." >&2
        exit 1
    fi
    job_ids+=("${job_id}")
    dependency="${DEPENDENCY_TYPE}:${job_id}"
done

echo "Campaign: ${EXP_NAME}"
echo "Run root: ${RUN_ROOT}"
echo "Jobs: ${job_ids[*]}"
