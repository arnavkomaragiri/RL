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

"""Producer-side payload helpers for the async-RL TQ path."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from nemo_rl.data.packed_rollouts import (
    PACKED_ATTENTION_SEGMENT_LENGTHS,
    TREE_ATTENTION_EDGE_LENGTHS,
    TREE_ATTENTION_EDGE_SOURCE_INDICES,
    TREE_ATTENTION_EDGE_TARGET_IDS,
    TREE_ATTENTION_LAYOUTS,
)
from nemo_rl.data_plane.codec import pack_jagged_fields
from nemo_rl.data_plane.column_io import (
    TOKEN_ALIGNED_FIELDS,
    TREE_EDGE_ALIGNED_FIELDS,
    TREE_EDGE_SHIFTED_FIELDS,
    TREE_EDGE_UNSHIFTED_FIELDS,
)
from nemo_rl.data_plane.schema import ROUTED_EXPERTS_FIELD
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.interfaces import PromptGroupRecord


def record_to_train_batch(
    record: PromptGroupRecord,
    *,
    pad_value_dict: Mapping[str, int],
) -> BatchedDataDict[Any]:
    """Convert one prompt group's record into a packed BatchedDataDict of N rows.

    Args:
        record: Rollout's PromptGroupRecord with N completions to flatten into rows.
        pad_value_dict: Field-name → pad value used by batched_message_log_to_flat_message.

    Returns:
        BatchedDataDict with input_ids, input_lengths, generation_logprobs, token_mask,
        sample_mask, prompt_ids_for_adv, total_reward, and optional routed_experts.
    """
    # Lazy imports: grpo and llm_message_utils transitively pull
    # experience.rollouts, so importing at module top risks a cycle.
    from nemo_rl.algorithms.grpo import (
        _apply_exact_nemo_gym_call_trees,
        _flatten_tree_model_inputs,
        _use_exact_nemo_gym_call_sequences,
        add_grpo_token_loss_masks_and_generation_logprobs,
        extract_initial_prompt_messages,
    )
    from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
    from nemo_rl.experience.rollouts import backfill_missing_routed_experts

    completions = record.completions
    n = len(completions)
    assert n > 0, "PromptGroupRecord has no completions"

    original_message_logs = [c.message_log for c in completions]
    rollout_batch = BatchedDataDict[Any]({"message_log": original_message_logs})
    exact_call_logs = [c.training_message_logs for c in completions]
    exact_call_trees = [c.exact_call_tree for c in completions]
    if any(tree is not None for tree in exact_call_trees):
        if any(tree is None for tree in exact_call_trees):
            raise ValueError(
                "precompacted exact-call metadata must be present for every "
                "completion in a prompt group"
            )
        if any(call_logs is not None for call_logs in exact_call_logs):
            raise ValueError(
                "a completion cannot carry both raw and precompacted exact-call metadata"
            )
        _apply_exact_nemo_gym_call_trees(
            rollout_batch,
            [tree for tree in exact_call_trees if tree is not None],
        )
    elif any(call_logs is not None for call_logs in exact_call_logs):
        if any(call_logs is None for call_logs in exact_call_logs):
            raise ValueError(
                "exact NeMo-Gym call metadata must be present for every completion "
                "in a prompt group"
            )
        rollout_batch["training_message_logs"] = [
            call_logs for call_logs in exact_call_logs if call_logs is not None
        ]
        _use_exact_nemo_gym_call_sequences(rollout_batch)

    message_logs = rollout_batch["message_log"]
    prompt_token_count = sum(len(m["token_ids"]) for m in record.prompt)
    prompt_lengths = torch.full((n,), prompt_token_count, dtype=torch.long)

    # Must precede the prompt extraction: it reuses the same message dicts, so
    # backfilling here also covers the prompt flatten below. Doing it only inside
    # add_grpo_token_loss_masks_and_generation_logprobs would be too late.
    backfill_missing_routed_experts(message_logs)

    prompt_message_logs = extract_initial_prompt_messages(
        original_message_logs, prompt_lengths
    )
    prompt_flat, _ = batched_message_log_to_flat_message(
        prompt_message_logs,
        pad_value_dict=dict(pad_value_dict),  # type: ignore
    )

    add_grpo_token_loss_masks_and_generation_logprobs(message_logs)
    flat, input_lengths = batched_message_log_to_flat_message(
        message_logs,  # type: ignore
        pad_value_dict=dict(pad_value_dict),  # type: ignore
    )

    model_flat = flat
    model_input_lengths = input_lengths
    if TREE_ATTENTION_LAYOUTS in rollout_batch:
        model_flat, model_input_lengths = _flatten_tree_model_inputs(
            rollout_batch,
            flat,
            input_lengths,
            pad_token_id=int(pad_value_dict.get("token_ids", 0)),
            make_sequence_length_divisible_by=1,
        )

    total_reward = torch.tensor(
        [float(c.reward) for c in completions], dtype=torch.float32
    )
    sample_mask = torch.ones(n, dtype=torch.float32)

    train_data: dict[str, Any] = {
        "input_ids": model_flat["token_ids"],
        "input_lengths": model_input_lengths,
        "generation_logprobs": flat["generation_logprobs"],
        "token_mask": flat["token_loss_mask"],
        "sample_mask": sample_mask,
        "prompt_ids_for_adv": prompt_flat["token_ids"],
        "total_reward": total_reward,
    }
    if ROUTED_EXPERTS_FIELD in model_flat:
        train_data[ROUTED_EXPERTS_FIELD] = model_flat[ROUTED_EXPERTS_FIELD]
    if TREE_ATTENTION_LAYOUTS in rollout_batch:
        layouts = rollout_batch[TREE_ATTENTION_LAYOUTS]
        edge_width = flat["token_ids"].shape[1] - 1
        edge_sources = torch.full((n, edge_width), -1, dtype=torch.long)
        for row, layout in enumerate(layouts):
            edge_sources[row, : len(layout.edge_source_indices)] = torch.tensor(
                layout.edge_source_indices, dtype=torch.long
            )
        train_data[TREE_ATTENTION_EDGE_SOURCE_INDICES] = edge_sources
        train_data[TREE_ATTENTION_EDGE_TARGET_IDS] = flat["token_ids"][:, 1:]
        train_data[TREE_ATTENTION_EDGE_LENGTHS] = torch.tensor(
            [len(layout.edge_source_indices) for layout in layouts],
            dtype=torch.long,
        )
        train_data[TREE_ATTENTION_LAYOUTS] = layouts
    if PACKED_ATTENTION_SEGMENT_LENGTHS in rollout_batch:
        train_data[PACKED_ATTENTION_SEGMENT_LENGTHS] = rollout_batch[
            PACKED_ATTENTION_SEGMENT_LENGTHS
        ]
    return BatchedDataDict[Any](train_data)


def pack_payload(
    train_batch: Mapping[str, Any],
    *,
    weight_version: int,
    group_id: str,
) -> tuple[list[str], TensorDict, list[dict[str, Any]]]:
    """Pack a producer batch into (sample_ids, fields, tags) for put_samples.

    Args:
        train_batch: Mapping with at least input_lengths plus the tensor/object fields to send.
        weight_version: Trainer weight version stamped on every row's tag.
        group_id: Per-group identifier used as the sample_id prefix; the caller owns uniqueness.

    Returns:
        sample_ids of the form {group_id}_g{i}, a jagged-packed TensorDict, and per-row tags.
    """
    lengths = train_batch["input_lengths"]
    n = int(lengths.shape[0])
    tensor_fields: dict[str, torch.Tensor | np.ndarray] = {
        k: v
        for k, v in train_batch.items()
        if isinstance(v, torch.Tensor)
        or (isinstance(v, np.ndarray) and v.dtype == object)
    }
    token_aligned_fields = TOKEN_ALIGNED_FIELDS
    lengths_by_field = None
    if TREE_ATTENTION_LAYOUTS in train_batch:
        edge_lengths = train_batch[TREE_ATTENTION_EDGE_LENGTHS]
        token_aligned_fields = TOKEN_ALIGNED_FIELDS - TREE_EDGE_ALIGNED_FIELDS
        lengths_by_field = {
            **{field: edge_lengths + 1 for field in TREE_EDGE_SHIFTED_FIELDS},
            **{field: edge_lengths for field in TREE_EDGE_UNSHIFTED_FIELDS},
        }
    fields_td = pack_jagged_fields(
        tensor_fields,
        lengths=lengths,
        token_aligned_fields=token_aligned_fields,
        lengths_by_field=lengths_by_field,
    )
    sample_ids = [f"{group_id}_g{i}" for i in range(n)]
    tags = [
        {
            "weight_version": weight_version,
            "group_id": group_id,
            "rollout_index": rollout_index,
        }
        for rollout_index in range(n)
    ]
    return sample_ids, fields_td, tags
