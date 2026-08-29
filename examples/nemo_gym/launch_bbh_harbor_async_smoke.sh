#!/usr/bin/env bash

# Submit the one-step BBH Harbor/OpenSandbox async GRPO diagnostic on Slurm.
# OpenSandbox and rubric secrets are inherited by sbatch and never embedded in COMMAND.
# *_API_KEY_FILE must contain only the raw key value, for example:
#   osb_example_key_value
# Do not include a variable name, quotes, or other dotenv syntax.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_LOCATION=${REPO_LOCATION:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
CONTAINER_REPO_LOCATION=${CONTAINER_REPO_LOCATION:-/opt/nemo-rl}

CONTAINER=${CONTAINER:-}
SLURM_ACCOUNT=${SLURM_ACCOUNT:-}
SLURM_PARTITION=${SLURM_PARTITION:-batch}
SLURM_COMMENT=${SLURM_COMMENT:-}
SLURM_DEPENDENCY=${SLURM_DEPENDENCY:-}
HARBOR_AGENTIC_VERIFIER=${HARBOR_AGENTIC_VERIFIER:-false}
HARBOR_HONEYPOT_AUDIT=${HARBOR_HONEYPOT_AUDIT:-false}
HARBOR_POLICY_ALERTS=${HARBOR_POLICY_ALERTS:-true}
HARBOR_POLICY_TIMEOUT_S=${HARBOR_POLICY_TIMEOUT_S:-3600}
HARBOR_POLICY_ALERT_POLL_INTERVAL_S=${HARBOR_POLICY_ALERT_POLL_INTERVAL_S:-5}
HARBOR_POLICY_ALERT_RETRY_INTERVAL_S=${HARBOR_POLICY_ALERT_RETRY_INTERVAL_S:-15}
HARBOR_POLICY_RLIMIT_AS_MIB=${HARBOR_POLICY_RLIMIT_AS_MIB:-49152}
if [[ "${HARBOR_AGENTIC_VERIFIER}" == "true" ]]; then
    DEFAULT_WALLTIME=00:40:00
else
    DEFAULT_WALLTIME=00:20:00
fi
WALLTIME=${WALLTIME:-${DEFAULT_WALLTIME}}
NUM_NODES=${NUM_NODES:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NUM_GENERATION_NODES=${NUM_GENERATION_NODES:-2}
NUM_PROMPTS_PER_STEP=${NUM_PROMPTS_PER_STEP:-2}
NUM_GENERATIONS_PER_PROMPT=${NUM_GENERATIONS_PER_PROMPT:-2}
TRAIN_GLOBAL_BATCH_SIZE=$((NUM_PROMPTS_PER_STEP * NUM_GENERATIONS_PER_PROMPT))
MAX_PARALLEL_ENVS=${MAX_PARALLEL_ENVS:-1024}

RUN_ID=${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}
EXP_NAME=${EXP_NAME:-harbor-bbh-tokcap-${RUN_ID}}
RUN_ROOT=${RUN_ROOT:-${REPO_LOCATION}/results/${EXP_NAME}}
WANDB_RUN_ID=${WANDB_RUN_ID:-${EXP_NAME}}
WANDB_RESUME=${WANDB_RESUME:-allow}
HARBOR_JOBS_DIR=${HARBOR_JOBS_DIR:-${RUN_ROOT}/harbor-jobs}
NRL_MEGATRON_CHECKPOINT_DIR=${NRL_MEGATRON_CHECKPOINT_DIR:-${RUN_ROOT}/megatron-checkpoints}
NEMO_GYM_VENV_DIR=${NEMO_GYM_VENV_DIR:-${RUN_ROOT}/nemo-gym-venvs}
TOKEN_CAPTURE_DIAGNOSTICS=${TOKEN_CAPTURE_DIAGNOSTICS:-false}
TOKEN_LOGPROB_DIAGNOSTICS=${TOKEN_LOGPROB_DIAGNOSTICS:-${TOKEN_CAPTURE_DIAGNOSTICS}}
USE_TRANSFER_QUEUE=${USE_TRANSFER_QUEUE:-false}
USE_SINGLE_CONTROLLER=${USE_SINGLE_CONTROLLER:-false}
NEMO_GYM_TOKEN_CAPTURE_RETAIN_CONSUMED=${NEMO_GYM_TOKEN_CAPTURE_RETAIN_CONSUMED:-${TOKEN_CAPTURE_DIAGNOSTICS}}
NEMO_GYM_MODEL_CALL_DIAGNOSTICS=${NEMO_GYM_MODEL_CALL_DIAGNOSTICS:-${TOKEN_CAPTURE_DIAGNOSTICS}}
if [[ "${NEMO_GYM_TOKEN_CAPTURE_RETAIN_CONSUMED}" == "true" ]]; then
    NEMO_GYM_TOKEN_CAPTURE_DIR=${NEMO_GYM_TOKEN_CAPTURE_DIR:-${RUN_ROOT}/token-captures}
fi
if [[ "${NEMO_GYM_MODEL_CALL_DIAGNOSTICS}" == "true" ]]; then
    NEMO_GYM_MODEL_CALL_CAPTURE_DIR=${NEMO_GYM_MODEL_CALL_CAPTURE_DIR:-${RUN_ROOT}/model-call-captures}
fi
if [[ "${USE_TRANSFER_QUEUE}" == "true" ]]; then
    DEFAULT_RECIPE=examples/configs/recipes/llm/grpo-qwen3-30ba3b-thinking-4n8g-megatron-cp2-r3-async-gym-harbor-bbh-smoke-tq-simple.yaml
else
    DEFAULT_RECIPE=examples/configs/recipes/llm/grpo-qwen3-30ba3b-thinking-4n8g-megatron-cp2-r3-async-gym-harbor-bbh-smoke.yaml
fi
RECIPE=${RECIPE:-${DEFAULT_RECIPE}}
MODEL_PATH=${MODEL_PATH:-}

HARBOR_DATASET_PATH=${HARBOR_DATASET_PATH:-}
HARBOR_VALIDATION_DATASET_PATH=${HARBOR_VALIDATION_DATASET_PATH:-}
HARBOR_TRAIN_MANIFEST_SOURCE=${HARBOR_TRAIN_MANIFEST_SOURCE:-}
HARBOR_VALIDATION_MANIFEST_SOURCE=${HARBOR_VALIDATION_MANIFEST_SOURCE:-}
HARBOR_TRAIN_EXCLUDE_FILE=${HARBOR_TRAIN_EXCLUDE_FILE:-${REPO_LOCATION}/patches/capsule_shortcuts/training_mask.txt}
HARBOR_VALIDATION_EXCLUDE_FILE=${HARBOR_VALIDATION_EXCLUDE_FILE:-${REPO_LOCATION}/patches/capsule_shortcuts/validation_mask.txt}
HARBOR_BENCHMARK_NAME=${HARBOR_BENCHMARK_NAME:-bbh-harbor-rl-v0}
HARBOR_IMAGE_OVERRIDE=${HARBOR_IMAGE_OVERRIDE:-docker.io/akomaragiri89904/bixbench:opencode-1.18.18@sha256:94e0846fd06cd88e98271e90cad7d4942d318fb0fac07269493d1e5ad281b376}
HARBOR_SANDBOX_ENTRYPOINT=${HARBOR_SANDBOX_ENTRYPOINT:-'["tail","-f","/dev/null"]'}
HARBOR_TASK_DATA_SOURCE=${HARBOR_TASK_DATA_SOURCE:-}
HARBOR_TASK_DATA_HOST_PATH=${HARBOR_TASK_DATA_HOST_PATH:-}
# HARBOR_TASK_DATA_HOST_PATH identifies the per-task Harbor directory. Only
# the capsule payload beneath environment/data is exposed to the policy.
# Set this explicitly to an empty string only when the supplied host path
# already points directly at capsule files.
HARBOR_TASK_DATA_SUBPATH=${HARBOR_TASK_DATA_SUBPATH-environment/data}
HARBOR_TASK_DATA_MOUNT_PATH=${HARBOR_TASK_DATA_MOUNT_PATH:-/data}
HARBOR_TASK_DATA_COMPAT_PATH=${HARBOR_TASK_DATA_COMPAT_PATH:-/app/data}
HARBOR_ENABLE_EFS=${HARBOR_ENABLE_EFS:-false}
HARBOR_ENABLE_DEFAULT_S3_MOUNT=${HARBOR_ENABLE_DEFAULT_S3_MOUNT:-false}
HARBOR_EFS_ARTIFACT_TRANSFER=${HARBOR_EFS_ARTIFACT_TRANSFER:-false}
HARBOR_EFS_ARTIFACT_HOST_PATH=${HARBOR_EFS_ARTIFACT_HOST_PATH:-/mnt/efs/data/shared}
NEMO_GYM_USER=${NEMO_GYM_USER:-${USER:-$(id -un)}}
HARBOR_EFS_ARTIFACT_OWNER=${HARBOR_EFS_ARTIFACT_OWNER:-${NEMO_GYM_USER}}
HARBOR_EFS_ARTIFACT_BASE_SUB_PATH=${HARBOR_EFS_ARTIFACT_BASE_SUB_PATH:-nemo-gym-harbor-artifacts}
HARBOR_EFS_ARTIFACT_RUN=${HARBOR_EFS_ARTIFACT_RUN:-${EXP_NAME}}
HARBOR_EFS_ARTIFACT_SOURCE_ROOT=${HARBOR_EFS_ARTIFACT_HOST_PATH}/${HARBOR_EFS_ARTIFACT_OWNER}/${HARBOR_EFS_ARTIFACT_BASE_SUB_PATH}/${HARBOR_EFS_ARTIFACT_RUN}
EXPECTED_HARBOR_EFS_ARTIFACT_SOURCE_PATH=${HARBOR_EFS_ARTIFACT_SOURCE_ROOT}/'{context_id}'
HARBOR_EFS_ARTIFACT_SOURCE_PATH=${HARBOR_EFS_ARTIFACT_SOURCE_PATH:-${EXPECTED_HARBOR_EFS_ARTIFACT_SOURCE_PATH}}
HARBOR_EFS_ARTIFACT_MOUNT_PATH=${HARBOR_EFS_ARTIFACT_MOUNT_PATH:-/app}
HARBOR_EFS_ARTIFACT_TIMEOUT_S=${HARBOR_EFS_ARTIFACT_TIMEOUT_S:-600}
HARBOR_EFS_CLEANUP_TIMEOUT_S=${HARBOR_EFS_CLEANUP_TIMEOUT_S:-600}
HARBOR_EFS_CLEANUP_TTL_S=${HARBOR_EFS_CLEANUP_TTL_S:-900}
HARBOR_HONEYPOT_PATH=${HARBOR_HONEYPOT_PATH:-/mnt/s3-data}
HARBOR_ADDITIONAL_HONEYPOT_PATHS=${HARBOR_ADDITIONAL_HONEYPOT_PATHS:-}
if [[ "${HARBOR_HONEYPOT_AUDIT}" == "true" || "${HARBOR_EFS_ARTIFACT_TRANSFER}" == "true" ]]; then
    HARBOR_POLICY_WORKSPACE_EXCLUDES=${HARBOR_POLICY_WORKSPACE_EXCLUDES:-[data,.opencode]}
    HARBOR_ENVIRONMENT_UPLOAD_EXCLUDES=${HARBOR_ENVIRONMENT_UPLOAD_EXCLUDES:-[data]}
else
    HARBOR_POLICY_WORKSPACE_EXCLUDES=${HARBOR_POLICY_WORKSPACE_EXCLUDES:-[.opencode]}
    HARBOR_ENVIRONMENT_UPLOAD_EXCLUDES=${HARBOR_ENVIRONMENT_UPLOAD_EXCLUDES:-[]}
fi
HARBOR_SANDBOX_PROBE_COMMAND=${HARBOR_SANDBOX_PROBE_COMMAND:-}
if [[ -z "${HARBOR_SANDBOX_PROBE_COMMAND}" ]]; then
    if [[ "${HARBOR_AGENTIC_VERIFIER}" == "true" ]]; then
        HARBOR_SANDBOX_PROBE_COMMAND="printf nemo-gym-sandbox-ready"
    else
        HARBOR_SANDBOX_PROBE_COMMAND="python -c \"import bbh_mcp.server; print('nemo-gym-sandbox-ready')\""
    fi
fi
HARBOR_SANDBOX_PROBE_EXPECTED_STDOUT=${HARBOR_SANDBOX_PROBE_EXPECTED_STDOUT:-nemo-gym-sandbox-ready}
HARBOR_SANDBOX_PROBE_TIMEOUT_S=${HARBOR_SANDBOX_PROBE_TIMEOUT_S:-180}
HARBOR_SANDBOX_PROBE_DEADLINE_S=${HARBOR_SANDBOX_PROBE_DEADLINE_S:-240}
HARBOR_SANDBOX_PROBE_STABLE_COUNT=${HARBOR_SANDBOX_PROBE_STABLE_COUNT:-1}
HARBOR_SANDBOX_PROBE_STABLE_DELAY_S=${HARBOR_SANDBOX_PROBE_STABLE_DELAY_S:-0}
HARBOR_SANDBOX_VOLUMES=${HARBOR_SANDBOX_VOLUMES:-}
HARBOR_SANDBOX_PATH_COPIES=${HARBOR_SANDBOX_PATH_COPIES:-}
HARBOR_SANDBOX_PATH_SYMLINKS=${HARBOR_SANDBOX_PATH_SYMLINKS:-}
if [[ -n "${HARBOR_TASK_DATA_HOST_PATH}" ]]; then
    if [[ -n "${HARBOR_TASK_DATA_SOURCE}" || -n "${HARBOR_SANDBOX_PATH_COPIES}" ]]; then
        echo "HARBOR_TASK_DATA_HOST_PATH cannot be combined with sandbox-local data copies." >&2
        exit 2
    fi
    if [[ -n "${HARBOR_SANDBOX_VOLUMES}" ]]; then
        echo "HARBOR_TASK_DATA_HOST_PATH cannot be combined with HARBOR_SANDBOX_VOLUMES." >&2
        exit 2
    fi
    if [[ "${HARBOR_TASK_DATA_SUBPATH}" == /* || "/${HARBOR_TASK_DATA_SUBPATH}/" == *"/../"* ]]; then
        echo "HARBOR_TASK_DATA_SUBPATH must be a relative path without '..' components." >&2
        exit 2
    fi
    HARBOR_TASK_DATA_VOLUME_HOST_PATH=${HARBOR_TASK_DATA_HOST_PATH%/}
    if [[ -n "${HARBOR_TASK_DATA_SUBPATH}" ]]; then
        HARBOR_TASK_DATA_VOLUME_HOST_PATH=${HARBOR_TASK_DATA_VOLUME_HOST_PATH}/${HARBOR_TASK_DATA_SUBPATH#/}
    fi
    HARBOR_SANDBOX_VOLUMES='[{"name":"problem-data","host":{"path":"'"${HARBOR_TASK_DATA_VOLUME_HOST_PATH}"'"},"mountPath":"'"${HARBOR_TASK_DATA_MOUNT_PATH}"'","readOnly":true}]'
    if [[ "${HARBOR_TASK_DATA_MOUNT_PATH}" == "${HARBOR_TASK_DATA_COMPAT_PATH}" ]]; then
        echo "HARBOR_TASK_DATA_MOUNT_PATH and HARBOR_TASK_DATA_COMPAT_PATH must differ." >&2
        exit 2
    fi
    if [[ -z "${HARBOR_SANDBOX_PATH_SYMLINKS}" ]]; then
        HARBOR_SANDBOX_PATH_SYMLINKS='[{"source":"'"${HARBOR_TASK_DATA_MOUNT_PATH}"'","destination":"'"${HARBOR_TASK_DATA_COMPAT_PATH}"'"}]'
    fi
fi
HARBOR_SANDBOX_VOLUMES=${HARBOR_SANDBOX_VOLUMES:-[]}
if [[ -n "${HARBOR_TASK_DATA_SOURCE}" && -z "${HARBOR_SANDBOX_PATH_COPIES}" ]]; then
    HARBOR_SANDBOX_PATH_COPIES='[{"source":"'"${HARBOR_TASK_DATA_SOURCE}"'","destination":"/app/data"}]'
fi
HARBOR_SANDBOX_PATH_COPIES=${HARBOR_SANDBOX_PATH_COPIES:-[]}
HARBOR_SANDBOX_PATH_SYMLINKS=${HARBOR_SANDBOX_PATH_SYMLINKS:-[]}

validate_sandbox_template_json() {
    local variable_name=$1
    local value=$2
    python3 - "${variable_name}" "${value}" <<'PY'
import json
import re
import sys

variable_name, raw_value = sys.argv[1:]
try:
    value = json.loads(raw_value)
except json.JSONDecodeError as error:
    raise SystemExit(f"{variable_name} must be valid JSON: {error}") from error

supported = {"context_id", "environment_name", "session_id", "task_id", "task_name"}
pattern = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
unsupported = set()


def collect(item):
    if isinstance(item, str):
        unsupported.update(pattern.findall(item))
    elif isinstance(item, dict):
        for key, nested in item.items():
            collect(key)
            collect(nested)
    elif isinstance(item, list):
        for nested in item:
            collect(nested)


collect(value)
unsupported.difference_update(supported)
if unsupported:
    names = ", ".join(f"{{{name}}}" for name in sorted(unsupported))
    allowed = ", ".join(f"{{{name}}}" for name in sorted(supported))
    raise SystemExit(
        f"{variable_name} contains unsupported template placeholder(s): {names}. "
        f"Supported placeholders: {allowed}."
    )
PY
}

validate_sandbox_template_json HARBOR_SANDBOX_VOLUMES "${HARBOR_SANDBOX_VOLUMES}"
validate_sandbox_template_json HARBOR_SANDBOX_PATH_COPIES "${HARBOR_SANDBOX_PATH_COPIES}"
validate_sandbox_template_json HARBOR_SANDBOX_PATH_SYMLINKS "${HARBOR_SANDBOX_PATH_SYMLINKS}"

HARBOR_ADDITIONAL_HONEYPOT_PATHS=${HARBOR_ADDITIONAL_HONEYPOT_PATHS:-[]}
OPENSANDBOX_PROTOCOL=${OPENSANDBOX_PROTOCOL:-http}
OPENSANDBOX_USE_SERVER_PROXY=${OPENSANDBOX_USE_SERVER_PROXY:-true}
OPENSANDBOX_REQUEST_TIMEOUT_S=${OPENSANDBOX_REQUEST_TIMEOUT_S:-1200}
RUBRIC_MODEL=${RUBRIC_MODEL:-nvidia/zai-org/glm-5.2}
RUBRIC_MODEL_API_MODE=${RUBRIC_MODEL_API_MODE:-chat_completions}
RUBRIC_MODEL_API_BASE=${RUBRIC_MODEL_API_BASE:-https://inference-api.nvidia.com/v1}
RUBRIC_MODEL_API_KEY=${RUBRIC_MODEL_API_KEY:-${NVINF_API_KEY:-}}
RUBRIC_MODEL_API_KEY_FILE=${RUBRIC_MODEL_API_KEY_FILE:-${NVINF_API_KEY_FILE:-}}

if [[ -z "${OPENSANDBOX_API_KEY:-}" && -n "${OPENSANDBOX_API_KEY_FILE:-}" ]]; then
    if [[ ! -r "${OPENSANDBOX_API_KEY_FILE}" ]]; then
        echo "OPENSANDBOX_API_KEY_FILE is not readable: ${OPENSANDBOX_API_KEY_FILE}" >&2
        exit 2
    fi
    OPENSANDBOX_API_KEY=$(<"${OPENSANDBOX_API_KEY_FILE}")
fi

if [[ -z "${RUBRIC_MODEL_API_KEY}" && -n "${RUBRIC_MODEL_API_KEY_FILE}" ]]; then
    if [[ ! -r "${RUBRIC_MODEL_API_KEY_FILE}" ]]; then
        echo "RUBRIC_MODEL_API_KEY_FILE is not readable: ${RUBRIC_MODEL_API_KEY_FILE}" >&2
        exit 2
    fi
    RUBRIC_MODEL_API_KEY=$(<"${RUBRIC_MODEL_API_KEY_FILE}")
fi

if [[ -z "${OPENSANDBOX_DOMAIN:-}" ]]; then
    echo "OPENSANDBOX_DOMAIN is required (hostname only; set OPENSANDBOX_PROTOCOL separately)." >&2
    exit 2
fi
if [[ -z "${OPENSANDBOX_API_KEY:-}" ]]; then
    echo "Set OPENSANDBOX_API_KEY or OPENSANDBOX_API_KEY_FILE." >&2
    exit 2
fi
if [[ -n "${OPENSANDBOX_CA_BUNDLE:-}" && ! -r "${OPENSANDBOX_CA_BUNDLE}" ]]; then
    echo "OPENSANDBOX_CA_BUNDLE is not readable: ${OPENSANDBOX_CA_BUNDLE}" >&2
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
if [[ -n "${SLURM_DEPENDENCY}" && ! "${SLURM_DEPENDENCY}" =~ ^afterok:[0-9]+$ ]]; then
    echo "SLURM_DEPENDENCY must have the form afterok:<job-id>." >&2
    exit 2
fi
if [[ -z "${MODEL_PATH}" ]]; then
    echo "MODEL_PATH is required and must point to a Hugging Face checkpoint." >&2
    exit 2
fi
if [[ ! "${WANDB_RUN_ID}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "WANDB_RUN_ID must contain only letters, digits, '_', or '-'." >&2
    exit 2
fi
case "${WANDB_RESUME}" in
    allow|must|never|auto) ;;
    *)
        echo "WANDB_RESUME must be allow, must, never, or auto." >&2
        exit 2
        ;;
esac
if [[ -z "${HARBOR_DATASET_PATH}" ]]; then
    echo "HARBOR_DATASET_PATH is required." >&2
    exit 2
fi
if [[ -z "${HARBOR_VALIDATION_DATASET_PATH}" ]]; then
    HARBOR_VALIDATION_DATASET_PATH=${HARBOR_DATASET_PATH%/*}/val
    if [[ ! -d "${HARBOR_VALIDATION_DATASET_PATH}" ]]; then
        HARBOR_VALIDATION_DATASET_PATH=${HARBOR_DATASET_PATH}
    fi
fi
if [[ "${HARBOR_HONEYPOT_AUDIT}" == "true" && "${HARBOR_SANDBOX_VOLUMES}" == "[]" && "${HARBOR_SANDBOX_PATH_COPIES}" == "[]" ]]; then
    echo "Configure HARBOR_TASK_DATA_HOST_PATH, HARBOR_SANDBOX_VOLUMES, or a sandbox-local data copy when HARBOR_HONEYPOT_AUDIT=true." >&2
    exit 2
fi
if [[ -z "${RUBRIC_MODEL}" ]]; then
    echo "RUBRIC_MODEL is required by the BBH Harbor task environment." >&2
    exit 2
fi
if [[ -z "${RUBRIC_MODEL_API_BASE}" ]]; then
    echo "RUBRIC_MODEL_API_BASE is required by the BBH Harbor task environment." >&2
    exit 2
fi
if [[ -z "${RUBRIC_MODEL_API_KEY}" ]]; then
    echo "Set RUBRIC_MODEL_API_KEY, RUBRIC_MODEL_API_KEY_FILE, NVINF_API_KEY, or NVINF_API_KEY_FILE." >&2
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
if [[ ! -d "${HARBOR_VALIDATION_DATASET_PATH}" ]]; then
    echo "Harbor validation dataset is not a directory: ${HARBOR_VALIDATION_DATASET_PATH}" >&2
    exit 2
fi
if [[ -n "${HARBOR_TRAIN_MANIFEST_SOURCE}" && ! -r "${HARBOR_TRAIN_MANIFEST_SOURCE}" ]]; then
    echo "Harbor train manifest is not readable: ${HARBOR_TRAIN_MANIFEST_SOURCE}" >&2
    exit 2
fi
if [[ -n "${HARBOR_VALIDATION_MANIFEST_SOURCE}" && ! -r "${HARBOR_VALIDATION_MANIFEST_SOURCE}" ]]; then
    echo "Harbor validation manifest is not readable: ${HARBOR_VALIDATION_MANIFEST_SOURCE}" >&2
    exit 2
fi
if [[ -n "${HARBOR_TRAIN_EXCLUDE_FILE}" && ! -r "${HARBOR_TRAIN_EXCLUDE_FILE}" ]]; then
    echo "Harbor train exclusion file is not readable: ${HARBOR_TRAIN_EXCLUDE_FILE}" >&2
    exit 2
fi
if [[ -n "${HARBOR_VALIDATION_EXCLUDE_FILE}" && ! -r "${HARBOR_VALIDATION_EXCLUDE_FILE}" ]]; then
    echo "Harbor validation exclusion file is not readable: ${HARBOR_VALIDATION_EXCLUDE_FILE}" >&2
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
# The driver changes its working directory to CONTAINER_REPO_LOCATION. Resolve
# host-relative model paths before constructing COMMAND so they still address
# the shared Lustre mount inside the container.
MODEL_PATH=$(realpath -e -- "${MODEL_PATH}")

case "${OPENSANDBOX_USE_SERVER_PROXY}" in
    true|false) ;;
    *)
        echo "OPENSANDBOX_USE_SERVER_PROXY must be true or false." >&2
        exit 2
        ;;
esac
case "${NEMO_GYM_TOKEN_CAPTURE_RETAIN_CONSUMED}" in
    true|false) ;;
    *)
        echo "NEMO_GYM_TOKEN_CAPTURE_RETAIN_CONSUMED must be true or false." >&2
        exit 2
        ;;
esac
case "${NEMO_GYM_MODEL_CALL_DIAGNOSTICS}" in
    true|false) ;;
    *)
        echo "NEMO_GYM_MODEL_CALL_DIAGNOSTICS must be true or false." >&2
        exit 2
        ;;
esac
case "${USE_TRANSFER_QUEUE}" in
    true|false) ;;
    *)
        echo "USE_TRANSFER_QUEUE must be true or false." >&2
        exit 2
        ;;
esac
case "${USE_SINGLE_CONTROLLER}" in
    true|false) ;;
    *)
        echo "USE_SINGLE_CONTROLLER must be true or false." >&2
        exit 2
        ;;
esac
if [[ "${USE_SINGLE_CONTROLLER}" == "true" && "${USE_TRANSFER_QUEUE}" != "true" ]]; then
    echo "USE_SINGLE_CONTROLLER=true requires USE_TRANSFER_QUEUE=true." >&2
    exit 2
fi
case "${HARBOR_AGENTIC_VERIFIER}" in
    true|false) ;;
    *)
        echo "HARBOR_AGENTIC_VERIFIER must be true or false." >&2
        exit 2
        ;;
esac
case "${HARBOR_HONEYPOT_AUDIT}" in
    true|false) ;;
    *)
        echo "HARBOR_HONEYPOT_AUDIT must be true or false." >&2
        exit 2
        ;;
esac
case "${HARBOR_POLICY_ALERTS}" in
    true|false) ;;
    *)
        echo "HARBOR_POLICY_ALERTS must be true or false." >&2
        exit 2
        ;;
esac
if [[ ! "${HARBOR_POLICY_RLIMIT_AS_MIB}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "HARBOR_POLICY_RLIMIT_AS_MIB must be a non-negative integer; use 0 to disable it." >&2
    exit 2
fi
case "${HARBOR_ENABLE_EFS}" in
    true|false) ;;
    *)
        echo "HARBOR_ENABLE_EFS must be true or false." >&2
        exit 2
        ;;
esac
case "${HARBOR_ENABLE_DEFAULT_S3_MOUNT}" in
    true|false) ;;
    *)
        echo "HARBOR_ENABLE_DEFAULT_S3_MOUNT must be true or false." >&2
        exit 2
        ;;
esac
case "${HARBOR_EFS_ARTIFACT_TRANSFER}" in
    true|false) ;;
    *)
        echo "HARBOR_EFS_ARTIFACT_TRANSFER must be true or false." >&2
        exit 2
        ;;
esac
if [[ "${HARBOR_EFS_ARTIFACT_TRANSFER}" == "true" && "${HARBOR_ENABLE_EFS}" != "true" ]]; then
    echo "HARBOR_EFS_ARTIFACT_TRANSFER=true requires HARBOR_ENABLE_EFS=true." >&2
    exit 2
fi
if [[ "${HARBOR_EFS_ARTIFACT_TRANSFER}" == "true" && "${HARBOR_AGENTIC_VERIFIER}" != "true" ]]; then
    echo "HARBOR_EFS_ARTIFACT_TRANSFER=true requires HARBOR_AGENTIC_VERIFIER=true." >&2
    exit 2
fi
if [[ "${HARBOR_EFS_ARTIFACT_TRANSFER}" == "true" ]]; then
    for path_segment_name in HARBOR_EFS_ARTIFACT_OWNER HARBOR_EFS_ARTIFACT_BASE_SUB_PATH HARBOR_EFS_ARTIFACT_RUN; do
        path_segment=${!path_segment_name}
        if [[ ! "${path_segment}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ || "${path_segment}" == "." || "${path_segment}" == ".." ]]; then
            echo "${path_segment_name} must be one safe path segment containing only letters, digits, '.', '_', or '-'." >&2
            exit 2
        fi
    done
fi
if [[ "${HARBOR_EFS_ARTIFACT_TRANSFER}" == "true" && "${HARBOR_EFS_ARTIFACT_SOURCE_PATH}" != "${EXPECTED_HARBOR_EFS_ARTIFACT_SOURCE_PATH}" ]]; then
    echo "HARBOR_EFS_ARTIFACT_SOURCE_PATH must be context-scoped as ${EXPECTED_HARBOR_EFS_ARTIFACT_SOURCE_PATH}." >&2
    exit 2
fi
if [[ "${HARBOR_EFS_ARTIFACT_TRANSFER}" == "true" && "${HARBOR_EFS_ARTIFACT_HOST_PATH}" != /mnt/efs/* ]]; then
    echo "HARBOR_EFS_ARTIFACT_HOST_PATH must be under /mnt/efs." >&2
    exit 2
fi
if [[ "${HARBOR_EFS_ARTIFACT_TRANSFER}" == "true" && "${HARBOR_EFS_ARTIFACT_MOUNT_PATH}" != "/app" ]]; then
    echo "HARBOR_EFS_ARTIFACT_MOUNT_PATH must be /app for direct policy-to-verifier workspace sharing." >&2
    exit 2
fi
if [[ "${HARBOR_EFS_ARTIFACT_TRANSFER}" == "true" && -n "${HARBOR_TASK_DATA_HOST_PATH}" ]]; then
    case "${HARBOR_TASK_DATA_MOUNT_PATH}" in
        "${HARBOR_EFS_ARTIFACT_MOUNT_PATH}"|"${HARBOR_EFS_ARTIFACT_MOUNT_PATH}"/*)
            echo "Task data must be mounted outside the shared workspace; use /data with /app/data as a symlink." >&2
            exit 2
            ;;
    esac
fi
case "${RUBRIC_MODEL_API_MODE}" in
    chat_completions|responses) ;;
    *)
        echo "RUBRIC_MODEL_API_MODE must be chat_completions or responses." >&2
        exit 2
        ;;
esac
if [[ ! "${MAX_PARALLEL_ENVS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_PARALLEL_ENVS must be a positive integer." >&2
    exit 2
fi

GYM_CONFIG_PATH_ENTRIES=(
    responses_api_models/vllm_model/configs/vllm_model_for_training.yaml
    responses_api_agents/gym_harbor_agent/configs/harbor_agent_opencode_compaction.yaml
    nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml
    responses_api_agents/gym_harbor_agent/configs/harbor_agent.yaml
    responses_api_agents/gym_harbor_agent/configs/harbor_agent_opensandbox.yaml
)
if [[ "${HARBOR_AGENTIC_VERIFIER}" == "true" ]]; then
    GYM_CONFIG_PATH_ENTRIES+=(
        responses_api_agents/gym_harbor_agent/configs/harbor_agent_agentic_verifier.yaml
    )
fi
if [[ "${HARBOR_EFS_ARTIFACT_TRANSFER}" == "true" ]]; then
    GYM_CONFIG_PATH_ENTRIES+=(
        responses_api_agents/gym_harbor_agent/configs/harbor_agent_efs_artifact_transfer.yaml
    )
fi
if [[ "${HARBOR_POLICY_ALERTS}" == "true" ]]; then
    GYM_CONFIG_PATH_ENTRIES+=(
        responses_api_agents/gym_harbor_agent/configs/harbor_agent_policy_alerts.yaml
    )
fi
if (( HARBOR_POLICY_RLIMIT_AS_MIB > 0 )); then
    GYM_CONFIG_PATH_ENTRIES+=(
        responses_api_agents/gym_harbor_agent/configs/harbor_agent_policy_rlimits.yaml
    )
fi
if [[ "${HARBOR_HONEYPOT_AUDIT}" == "true" ]]; then
    GYM_CONFIG_PATH_ENTRIES+=(
        responses_api_agents/gym_harbor_agent/configs/harbor_agent_honeypot_audit.yaml
    )
fi

GYM_ROOT=${REPO_LOCATION}/3rdparty/Gym-workspace/Gym
for config_path in "${GYM_CONFIG_PATH_ENTRIES[@]}"; do
    if [[ ! -r "${REPO_LOCATION}/${config_path}" && ! -r "${GYM_ROOT}/${config_path}" ]]; then
        echo "Gym config is not present in the launch tree: ${config_path}" >&2
        exit 2
    fi
done
GYM_CONFIG_PATHS=$(IFS=,; printf '[%s]' "${GYM_CONFIG_PATH_ENTRIES[*]}")

if (( NUM_GENERATION_NODES < 1 || NUM_GENERATION_NODES >= NUM_NODES )); then
    echo "NUM_GENERATION_NODES must be between 1 and NUM_NODES-1." >&2
    exit 2
fi
TRAINING_GPUS=$(((NUM_NODES - NUM_GENERATION_NODES) * GPUS_PER_NODE))
if (( TRAINING_GPUS < 16 )); then
    echo "This recipe requires at least 16 training GPUs." >&2
    exit 2
fi

export HARBOR_DATASET_PATH HARBOR_BENCHMARK_NAME HARBOR_JOBS_DIR
export HARBOR_IMAGE_OVERRIDE HARBOR_SANDBOX_ENTRYPOINT
export HARBOR_SANDBOX_PROBE_COMMAND HARBOR_SANDBOX_PROBE_EXPECTED_STDOUT
export HARBOR_SANDBOX_PROBE_TIMEOUT_S HARBOR_SANDBOX_PROBE_DEADLINE_S
export HARBOR_SANDBOX_PROBE_STABLE_COUNT HARBOR_SANDBOX_PROBE_STABLE_DELAY_S
export HARBOR_SANDBOX_VOLUMES HARBOR_SANDBOX_PATH_COPIES HARBOR_SANDBOX_PATH_SYMLINKS
export HARBOR_ENVIRONMENT_UPLOAD_EXCLUDES
export HARBOR_AGENTIC_VERIFIER HARBOR_POLICY_WORKSPACE_EXCLUDES
export HARBOR_HONEYPOT_AUDIT HARBOR_TASK_DATA_SOURCE HARBOR_HONEYPOT_PATH
export HARBOR_POLICY_ALERTS HARBOR_POLICY_TIMEOUT_S
export HARBOR_POLICY_ALERT_POLL_INTERVAL_S HARBOR_POLICY_ALERT_RETRY_INTERVAL_S
export HARBOR_POLICY_RLIMIT_AS_MIB
export HARBOR_ENABLE_EFS HARBOR_ENABLE_DEFAULT_S3_MOUNT HARBOR_ADDITIONAL_HONEYPOT_PATHS
export NEMO_GYM_USER
export HARBOR_EFS_ARTIFACT_TRANSFER HARBOR_EFS_ARTIFACT_SOURCE_PATH
export HARBOR_EFS_ARTIFACT_OWNER HARBOR_EFS_ARTIFACT_RUN
export HARBOR_EFS_ARTIFACT_HOST_PATH HARBOR_EFS_ARTIFACT_BASE_SUB_PATH
export HARBOR_EFS_ARTIFACT_SOURCE_ROOT HARBOR_EFS_ARTIFACT_MOUNT_PATH
export HARBOR_EFS_ARTIFACT_TIMEOUT_S HARBOR_EFS_CLEANUP_TIMEOUT_S HARBOR_EFS_CLEANUP_TTL_S
export OPENSANDBOX_DOMAIN OPENSANDBOX_API_KEY
export OPENSANDBOX_PROTOCOL OPENSANDBOX_USE_SERVER_PROXY OPENSANDBOX_REQUEST_TIMEOUT_S
export NEMO_GYM_TOKEN_CAPTURE_RETAIN_CONSUMED
export NEMO_GYM_MODEL_CALL_DIAGNOSTICS
if [[ -n "${NEMO_GYM_TOKEN_CAPTURE_DIR:-}" ]]; then
    export NEMO_GYM_TOKEN_CAPTURE_DIR
else
    unset NEMO_GYM_TOKEN_CAPTURE_DIR
fi
if [[ -n "${NEMO_GYM_MODEL_CALL_CAPTURE_DIR:-}" ]]; then
    export NEMO_GYM_MODEL_CALL_CAPTURE_DIR
else
    unset NEMO_GYM_MODEL_CALL_CAPTURE_DIR
fi
if [[ -n "${OPENSANDBOX_CA_BUNDLE:-}" ]]; then
    export OPENSANDBOX_CA_BUNDLE
else
    unset OPENSANDBOX_CA_BUNDLE
fi
export RUBRIC_MODEL RUBRIC_MODEL_API_MODE RUBRIC_MODEL_API_BASE RUBRIC_MODEL_API_KEY

SHARED_MOUNT=$(findmnt -n -o TARGET --target "${REPO_LOCATION}")
MOUNTS=${MOUNTS:-${SHARED_MOUNT}:${SHARED_MOUNT},${REPO_LOCATION}:${CONTAINER_REPO_LOCATION}}
mkdir -p \
    "${REPO_LOCATION}/3rdparty/Gym-workspace/Gym/cache" \
    "${RUN_ROOT}/slurm" \
    "${RUN_ROOT}/logs" \
    "${RUN_ROOT}/checkpoints" \
    "${HARBOR_JOBS_DIR}" \
    "${NEMO_GYM_VENV_DIR}" \
    "${NRL_MEGATRON_CHECKPOINT_DIR}"
HARBOR_MANIFEST_DIR=${RUN_ROOT}/manifests
HARBOR_TRAIN_MANIFEST=${HARBOR_MANIFEST_DIR}/train.jsonl
HARBOR_VALIDATION_MANIFEST=${HARBOR_MANIFEST_DIR}/validation.jsonl
mkdir -p "${HARBOR_MANIFEST_DIR}"
TRAIN_MANIFEST_ARGS=(--output "${HARBOR_TRAIN_MANIFEST}")
if [[ -n "${HARBOR_TRAIN_MANIFEST_SOURCE}" ]]; then
    TRAIN_MANIFEST_ARGS+=(
        --input-manifest "${HARBOR_TRAIN_MANIFEST_SOURCE}"
        --reference-split "${HARBOR_DATASET_PATH}"
    )
else
    TRAIN_MANIFEST_ARGS+=(--split "${HARBOR_DATASET_PATH}")
fi
if [[ -n "${HARBOR_TRAIN_EXCLUDE_FILE}" ]]; then
    TRAIN_MANIFEST_ARGS+=(--exclude-file "${HARBOR_TRAIN_EXCLUDE_FILE}")
fi
python3 "${REPO_LOCATION}/patches/harbor_opencode/write_nemo_rl_manifest.py" \
    "${TRAIN_MANIFEST_ARGS[@]}"

VALIDATION_MANIFEST_ARGS=(--output "${HARBOR_VALIDATION_MANIFEST}")
if [[ -n "${HARBOR_VALIDATION_MANIFEST_SOURCE}" ]]; then
    VALIDATION_MANIFEST_ARGS+=(
        --input-manifest "${HARBOR_VALIDATION_MANIFEST_SOURCE}"
        --reference-split "${HARBOR_VALIDATION_DATASET_PATH}"
    )
else
    VALIDATION_MANIFEST_ARGS+=(--split "${HARBOR_VALIDATION_DATASET_PATH}")
fi
if [[ -n "${HARBOR_VALIDATION_EXCLUDE_FILE}" ]]; then
    VALIDATION_MANIFEST_ARGS+=(--exclude-file "${HARBOR_VALIDATION_EXCLUDE_FILE}")
fi
python3 "${REPO_LOCATION}/patches/harbor_opencode/write_nemo_rl_manifest.py" \
    "${VALIDATION_MANIFEST_ARGS[@]}"
if [[ -n "${NEMO_GYM_TOKEN_CAPTURE_DIR:-}" ]]; then
    mkdir -p "${NEMO_GYM_TOKEN_CAPTURE_DIR}"
fi
if [[ -n "${NEMO_GYM_MODEL_CALL_CAPTURE_DIR:-}" ]]; then
    mkdir -p "${NEMO_GYM_MODEL_CALL_CAPTURE_DIR}"
fi

if [[ "${USE_SINGLE_CONTROLLER}" == "true" ]]; then
    TRAIN_ENTRYPOINT=examples/run_grpo_single_controller.py
else
    TRAIN_ENTRYPOINT=examples/nemo_gym/run_grpo_nemo_gym.py
fi

read -r -d '' COMMAND <<EOF || true
set -euo pipefail
cd ${CONTAINER_REPO_LOCATION}
export GYM_ROOT=${CONTAINER_REPO_LOCATION}/3rdparty/Gym-workspace/Gym
export HARBOR_DATASET_PATH=${HARBOR_DATASET_PATH}
export HARBOR_BENCHMARK_NAME=${HARBOR_BENCHMARK_NAME}
export HARBOR_JOBS_DIR=${HARBOR_JOBS_DIR}
export RAY_TMPDIR=/tmp/ray-${RUN_ID}
export NEMO_GYM_VENV_DIR=${NEMO_GYM_VENV_DIR}
export NRL_MEGATRON_CHECKPOINT_DIR=${NRL_MEGATRON_CHECKPOINT_DIR}
uv run python -u ${TRAIN_ENTRYPOINT} \\
    --config ${RECIPE} \\
    policy.model_name=${MODEL_PATH} \\
    cluster.num_nodes=${NUM_NODES} \\
    policy.generation.colocated.resources.num_nodes=${NUM_GENERATION_NODES} \\
    grpo.num_prompts_per_step=${NUM_PROMPTS_PER_STEP} \\
    grpo.num_generations_per_prompt=${NUM_GENERATIONS_PER_PROMPT} \\
    policy.train_global_batch_size=${TRAIN_GLOBAL_BATCH_SIZE} \\
    policy.generation_batch_size=${TRAIN_GLOBAL_BATCH_SIZE} \\
    env.nemo_gym.num_samples_in_parallel=${MAX_PARALLEL_ENVS} \\
    data.train.data_path=${HARBOR_TRAIN_MANIFEST} \\
    data.validation.data_path=${HARBOR_VALIDATION_MANIFEST} \\
    "env.nemo_gym.config_paths=${GYM_CONFIG_PATHS}" \\
    logger.log_dir=${RUN_ROOT}/logs \\
    logger.wandb.name=${EXP_NAME} \\
    ++logger.wandb.id=${WANDB_RUN_ID} \\
    ++logger.wandb.resume=${WANDB_RESUME} \\
    ++logger.token_logprob_diagnostics.enabled=${TOKEN_LOGPROB_DIAGNOSTICS} \\
    checkpointing.checkpoint_dir=${RUN_ROOT}/checkpoints \\
    ++env.nemo_gym.skip_venv_if_present=false
EOF

echo "Run name:             ${EXP_NAME}"
echo "W&B run ID:           ${WANDB_RUN_ID} (resume=${WANDB_RESUME})"
echo "Container:            ${CONTAINER}"
echo "Container source:     ${REPO_LOCATION} -> ${CONTAINER_REPO_LOCATION}"
echo "Recipe:               ${RECIPE}"
echo "TransferQueue:         ${USE_TRANSFER_QUEUE}"
echo "SingleController:      ${USE_SINGLE_CONTROLLER}"
echo "Training entrypoint:   ${TRAIN_ENTRYPOINT}"
echo "Model:                ${MODEL_PATH}"
echo "Nodes/GPUs:            ${NUM_NODES} x ${GPUS_PER_NODE}"
echo "Train/generation:      $((NUM_NODES - NUM_GENERATION_NODES)) / ${NUM_GENERATION_NODES} nodes"
echo "Rollouts:              ${NUM_PROMPTS_PER_STEP} x ${NUM_GENERATIONS_PER_PROMPT} = ${TRAIN_GLOBAL_BATCH_SIZE}"
echo "Max parallel envs:     ${MAX_PARALLEL_ENVS}"
echo "Partition/account:     ${SLURM_PARTITION} / ${SLURM_ACCOUNT}"
echo "Slurm comment:         $([[ -n "${SLURM_COMMENT}" ]] && echo configured || echo none)"
echo "Slurm dependency:      ${SLURM_DEPENDENCY:-none}"
echo "OpenSandbox endpoint:  ${OPENSANDBOX_PROTOCOL}://${OPENSANDBOX_DOMAIN}"
echo "OpenSandbox proxy:     ${OPENSANDBOX_USE_SERVER_PROXY}"
echo "Honeypot audit:        ${HARBOR_HONEYPOT_AUDIT}"
echo "Policy alerts:         ${HARBOR_POLICY_ALERTS} (${HARBOR_POLICY_TIMEOUT_S}s deadline)"
if (( HARBOR_POLICY_RLIMIT_AS_MIB > 0 )); then
    echo "Policy RLIMIT_AS:      ${HARBOR_POLICY_RLIMIT_AS_MIB} MiB per process"
else
    echo "Policy RLIMIT_AS:      disabled"
fi
echo "Scoped EFS volume:     ${HARBOR_ENABLE_EFS}"
echo "Blanket S3 mount:       ${HARBOR_ENABLE_DEFAULT_S3_MOUNT}"
echo "EFS direct workspace:  ${HARBOR_EFS_ARTIFACT_TRANSFER}"
echo "EFS workspace scope:   ${HARBOR_EFS_ARTIFACT_OWNER}/${HARBOR_EFS_ARTIFACT_RUN}"
echo "EFS workspace source:  ${HARBOR_EFS_ARTIFACT_SOURCE_PATH}"
echo "EFS policy mount:      ${HARBOR_EFS_ARTIFACT_MOUNT_PATH} (read-write)"
echo "EFS verifier mount:    ${HARBOR_EFS_ARTIFACT_MOUNT_PATH} (read-only)"
echo "OpenSandbox timeout:   ${OPENSANDBOX_REQUEST_TIMEOUT_S}s/request"
echo "OpenSandbox API key:   set (value suppressed)"
echo "OpenSandbox CA bundle: ${OPENSANDBOX_CA_BUNDLE:-system trust store}"
echo "Rubric model:          ${RUBRIC_MODEL}"
echo "Rubric API mode:       ${RUBRIC_MODEL_API_MODE}"
echo "Rubric API base:       ${RUBRIC_MODEL_API_BASE}"
echo "Rubric API key:        set (value suppressed)"
echo "Dataset:               ${HARBOR_DATASET_PATH}"
echo "Validation dataset:    ${HARBOR_VALIDATION_DATASET_PATH}"
if [[ -n "${HARBOR_TASK_DATA_HOST_PATH}" ]]; then
    echo "Capsule data source:   ${HARBOR_TASK_DATA_VOLUME_HOST_PATH}"
fi
echo "Train exclusions:      ${HARBOR_TRAIN_EXCLUDE_FILE:-disabled}"
echo "Validation exclusions: ${HARBOR_VALIDATION_EXCLUDE_FILE:-disabled}"
echo "Train manifest:        ${HARBOR_TRAIN_MANIFEST}"
echo "Validation manifest:   ${HARBOR_VALIDATION_MANIFEST}"
echo "Agentic verifier:      ${HARBOR_AGENTIC_VERIFIER}"
echo "Verifier environment:  separate"
echo "Sandbox image:         ${HARBOR_IMAGE_OVERRIDE}"
echo "Sandbox prewarm:       ${HARBOR_SANDBOX_PROBE_COMMAND}"
if [[ "${HARBOR_SANDBOX_VOLUMES}" == "[]" ]]; then
    echo "Sandbox volumes:       none"
else
    echo "Sandbox volumes:       configured (value suppressed)"
fi
echo "Upload excludes:       ${HARBOR_ENVIRONMENT_UPLOAD_EXCLUDES}"
echo "Workspace excludes:    ${HARBOR_POLICY_WORKSPACE_EXCLUDES}"
echo "Capture diagnostics:   ${NEMO_GYM_TOKEN_CAPTURE_RETAIN_CONSUMED}"
echo "Capture directory:     ${NEMO_GYM_TOKEN_CAPTURE_DIR:-node-local temporary store}"
echo "Request diagnostics:   ${NEMO_GYM_MODEL_CALL_DIAGNOSTICS}"
echo "Request directory:     ${NEMO_GYM_MODEL_CALL_CAPTURE_DIR:-node-local temporary store}"
echo "Logprob diagnostics:   ${TOKEN_LOGPROB_DIAGNOSTICS}"
echo "Run root:              ${RUN_ROOT}"
echo "Harbor jobs:           ${HARBOR_JOBS_DIR}"
echo "Megatron cache:        ${NRL_MEGATRON_CHECKPOINT_DIR}"
echo "Gym venvs:             ${NEMO_GYM_VENV_DIR}"

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
