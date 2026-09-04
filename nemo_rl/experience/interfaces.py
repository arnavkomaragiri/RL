# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Any, Optional

from nemo_rl.data.interfaces import LLMMessageLogType, VLMMessageLogType
from nemo_rl.data.packed_rollouts import TreeAttentionLayout

NEMO_GYM_TASK_INDEX_KEY = "_ng_task_index"
NEMO_GYM_ROLLOUT_INDEX_KEY = "_ng_rollout_index"
NEXT_NEMO_GYM_TASK_INDEX_KEY = "next_ng_task_index"


@dataclass(frozen=True)
class ExactCallTreeDiagnostics:
    """Small counters retained after one rollout's exact calls are compacted."""

    input_call_count: int
    input_token_count: int
    page_fork_count: int
    page_shared_token_count: int
    cross_replica_rollout: int
    replica_count: int
    baseline_attention_pairs: int
    min_generation_weight_version: int | None = None
    max_generation_weight_version: int | None = None


@dataclass(frozen=True)
class ExactCallTreePayload:
    """Owned compact representation of one rollout's captured model calls."""

    edge_message_log: LLMMessageLogType | VLMMessageLogType
    unique_message_log: LLMMessageLogType | VLMMessageLogType
    layout: TreeAttentionLayout
    diagnostics: ExactCallTreeDiagnostics


@dataclass
class Completion:
    """A single generated completion for one prompt."""

    message_log: LLMMessageLogType | VLMMessageLogType
    env_extras: Optional[dict[str, Any]]
    truncated: bool
    reward: float
    training_message_logs: Optional[list[LLMMessageLogType | VLMMessageLogType]] = None
    exact_call_tree: Optional[ExactCallTreePayload] = None


@dataclass
class PromptGroupRecord:
    """All completions for a single prompt, with prompt-level metadata."""

    prompt_idx: int
    prompt: LLMMessageLogType | VLMMessageLogType
    extra_env_info: Optional[dict[str, Any]]
    metadata: dict[str, Any]
    completions: list["Completion"]
    rollout_metrics: dict[str, Any]
