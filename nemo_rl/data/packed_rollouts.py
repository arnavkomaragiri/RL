# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Metadata and tensor helpers for independently attended calls in one rollout."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


PACKED_ATTENTION_SEGMENT_LENGTHS = "packed_attention_segment_lengths"
PACKED_ATTENTION_SELECTED_SEGMENTS = "packed_attention_selected_segments"
TREE_ATTENTION_LAYOUTS = "tree_attention_layouts"
TREE_ATTENTION_UNIQUE_MESSAGE_LOGS = "tree_attention_unique_message_logs"
TREE_ATTENTION_EDGE_SOURCE_INDICES = "tree_attention_edge_source_indices"
TREE_ATTENTION_EDGE_TARGET_IDS = "tree_attention_edge_target_ids"
TREE_ATTENTION_EDGE_LENGTHS = "tree_attention_edge_lengths"

# Tree model inputs are aligned to unique physical nodes, while policy-loss
# fields retain the original sampled-edge stream. Shifted fields include the
# initial dummy position used by the rest of NeMo-RL's next-token convention.
TREE_EDGE_SHIFTED_FIELDS = frozenset(
    {
        "generation_logprobs",
        "prev_logprobs",
        "reference_policy_logprobs",
        "advantages",
        "token_mask",
    }
)
TREE_EDGE_UNSHIFTED_FIELDS = frozenset(
    {
        TREE_ATTENTION_EDGE_SOURCE_INDICES,
        TREE_ATTENTION_EDGE_TARGET_IDS,
    }
)
TREE_EDGE_ALIGNED_FIELDS = TREE_EDGE_SHIFTED_FIELDS | TREE_EDGE_UNSHIFTED_FIELDS


@dataclass(frozen=True)
class PackedAttentionLayout:
    """How independent call rows map back to logical rollout rows."""

    segment_lengths: Sequence[Sequence[int]]
    output_sequence_length: int


@dataclass(frozen=True)
class TreeAttentionLayout:
    """One rollout's compressed execution tree in DFS segment order.

    Segments are maximal non-branching token runs. A root has parent ``-1``;
    every other segment begins immediately after the final token of its parent.
    ``depths`` are zero-based logical positions of each segment's first token.
    Sampled edges remain separate from unique nodes because one predecessor can
    supervise multiple child tokens, including duplicate samples.
    """

    segment_lengths: tuple[int, ...]
    segment_parents: tuple[int, ...]
    segment_depths: tuple[int, ...]
    edge_source_indices: tuple[int, ...]
    original_token_count: int

    @property
    def unique_token_count(self) -> int:
        return sum(self.segment_lengths)

    @property
    def max_path_length(self) -> int:
        return max(
            depth + length
            for depth, length in zip(self.segment_depths, self.segment_lengths)
        )

    @property
    def valid_attention_pairs(self) -> int:
        return sum(
            depth * length + length * (length + 1) // 2
            for depth, length in zip(self.segment_depths, self.segment_lengths)
        )

    def validate(self) -> None:
        """Reject layouts that could silently change tree attention semantics."""
        segment_count = len(self.segment_lengths)
        if not (segment_count == len(self.segment_parents) == len(self.segment_depths)):
            raise ValueError(
                "tree attention segment lengths, parents, and depths must align"
            )
        if segment_count == 0 or any(length <= 0 for length in self.segment_lengths):
            raise ValueError("tree attention requires non-empty positive segments")
        if self.original_token_count < self.unique_token_count:
            raise ValueError(
                "tree attention original token count cannot be smaller than the "
                "unique token count"
            )

        for segment_index, (length, parent, depth) in enumerate(
            zip(self.segment_lengths, self.segment_parents, self.segment_depths)
        ):
            if parent == -1:
                if depth != 0:
                    raise ValueError("tree attention roots must start at depth zero")
            else:
                if parent < 0 or parent >= segment_index:
                    raise ValueError(
                        "tree attention parents must precede children in DFS order"
                    )
                expected_depth = (
                    self.segment_depths[parent] + self.segment_lengths[parent]
                )
                if depth != expected_depth:
                    raise ValueError(
                        "tree attention child depth must follow its parent segment: "
                        f"segment={segment_index}, depth={depth}, "
                        f"expected={expected_depth}"
                    )
        if any(
            source < 0 or source >= self.unique_token_count
            for source in self.edge_source_indices
        ):
            raise ValueError(
                "tree attention sampled-edge sources must reference unique tokens"
            )


def validate_packed_attention_segment_lengths(
    segment_lengths: Sequence[Sequence[int]],
    input_lengths: Sequence[int] | torch.Tensor,
) -> None:
    """Validate one non-empty, exhaustive segment-length list per rollout row."""
    if torch.is_tensor(input_lengths):
        input_lengths_list = [int(length) for length in input_lengths.tolist()]
    else:
        input_lengths_list = [int(length) for length in input_lengths]
    if len(segment_lengths) != len(input_lengths_list):
        raise ValueError(
            "packed attention metadata must have one entry per rollout row: "
            f"segments={len(segment_lengths)}, rows={len(input_lengths_list)}"
        )
    for row, (row_segments, input_length) in enumerate(
        zip(segment_lengths, input_lengths_list)
    ):
        if not row_segments or any(
            isinstance(length, bool) or not isinstance(length, int) or length <= 0
            for length in row_segments
        ):
            raise ValueError(
                "packed attention segments must be non-empty positive integer lists; "
                f"row {row} has {row_segments!r}"
            )
        if sum(row_segments) != int(input_length):
            raise ValueError(
                "packed attention segments must cover the rollout's unpadded tokens "
                f"exactly; row {row} sums to {sum(row_segments)}, input_length={int(input_length)}"
            )


def split_tensor_at_packed_attention_segments(
    tensor: torch.Tensor,
    segment_lengths: Sequence[Sequence[int]],
    *,
    output_sequence_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split ``[rollout, token, ...]`` rows into padded per-call tensor rows."""
    segments: list[torch.Tensor] = []
    lengths: list[int] = []
    for row, row_segments in enumerate(segment_lengths):
        offset = 0
        for length in row_segments:
            segments.append(tensor[row, offset : offset + length])
            lengths.append(length)
            offset += length

    if not segments:
        raise ValueError("packed attention metadata produced no segments")

    max_length = max(lengths)
    if output_sequence_length is not None:
        if output_sequence_length < max_length:
            raise ValueError(
                "packed attention output_sequence_length cannot truncate calls: "
                f"requested={output_sequence_length}, longest_call={max_length}"
            )
        max_length = output_sequence_length
    output = tensor.new_zeros((len(segments), max_length, *tensor.shape[2:]))
    for index, segment in enumerate(segments):
        output[index, : segment.shape[0]] = segment
    return output, torch.tensor(lengths, dtype=torch.long, device=tensor.device)


def expand_batched_data_for_packed_attention(
    data: BatchedDataDict[Any],
    *,
    output_sequence_length: int | None = None,
) -> tuple[BatchedDataDict[Any], PackedAttentionLayout | None]:
    """Expand logical rollout rows into independently packable model-call rows."""
    if PACKED_ATTENTION_SEGMENT_LENGTHS not in data:
        return data, None
    segment_lengths = data[PACKED_ATTENTION_SEGMENT_LENGTHS]
    if "input_ids" not in data or "input_lengths" not in data:
        raise ValueError(
            f"{PACKED_ATTENTION_SEGMENT_LENGTHS} requires input_ids and input_lengths"
        )

    input_ids = data["input_ids"]
    if not torch.is_tensor(input_ids) or input_ids.ndim < 2:
        raise ValueError("packed attention requires tensor input_ids with shape [B, S]")
    validate_packed_attention_segment_lengths(segment_lengths, data["input_lengths"])

    parent_rows = [
        row_idx
        for row_idx, row_segments in enumerate(segment_lengths)
        for _ in row_segments
    ]
    flattened_lengths = [
        length for row_segments in segment_lengths for length in row_segments
    ]
    expanded = BatchedDataDict[Any]()
    for key, value in data.items():
        if key == PACKED_ATTENTION_SEGMENT_LENGTHS:
            continue
        if isinstance(value, PackedTensor):
            raise NotImplementedError(
                "independent model-call packing does not yet support multimodal data"
            )
        if torch.is_tensor(value):
            if key == "input_lengths":
                expanded[key] = torch.tensor(
                    flattened_lengths,
                    dtype=value.dtype,
                    device=value.device,
                )
            elif value.ndim > 1 and value.shape[1] == input_ids.shape[1]:
                expanded[key], _ = split_tensor_at_packed_attention_segments(
                    value,
                    segment_lengths,
                    output_sequence_length=output_sequence_length,
                )
            else:
                expanded[key] = value.index_select(
                    0, torch.tensor(parent_rows, device=value.device)
                )
        else:
            expanded[key] = [value[row_idx] for row_idx in parent_rows]

    return expanded, PackedAttentionLayout(
        segment_lengths=segment_lengths,
        output_sequence_length=input_ids.shape[1],
    )


def expand_selected_packed_attention_segments(
    data: BatchedDataDict[Any],
    segment_lengths: Sequence[Sequence[int]],
    selected_segments: Sequence[tuple[int, int]],
    *,
    output_sequence_length: int | None = None,
) -> BatchedDataDict[Any]:
    """Materialize selected ``(row, segment)`` calls from logical rows.

    TQ presharding can place calls from one logical rollout on different DP
    ranks. Each rank fetches the owning logical rows once, expands them into
    independent calls, and selects only the calls assigned to that rank.

    Args:
        data: Fetched logical rollout rows.
        segment_lengths: Exact call lengths for each row in ``data``.
        selected_segments: ``(row_index, segment_index)`` pairs in execution
            order for this DP rank.

    Returns:
        A batch containing one independently attended row per selected call.
    """
    if "input_ids" not in data or "input_lengths" not in data:
        raise ValueError(
            f"{PACKED_ATTENTION_SEGMENT_LENGTHS} requires input_ids and input_lengths"
        )
    input_ids = data["input_ids"]
    if not torch.is_tensor(input_ids) or input_ids.ndim < 2:
        raise ValueError("packed attention requires tensor input_ids with shape [B, S]")
    input_ids_is_jagged = bool(input_ids.is_nested)
    input_row_lengths = input_ids.offsets().diff() if input_ids_is_jagged else None
    validate_packed_attention_segment_lengths(segment_lengths, data["input_lengths"])

    selected: list[tuple[int, int, int]] = []
    for row_index, segment_index in selected_segments:
        if not 0 <= row_index < len(segment_lengths):
            raise ValueError(
                f"packed attention selected row {row_index} is out of range"
            )
        if not 0 <= segment_index < len(segment_lengths[row_index]):
            raise ValueError(
                "packed attention selected segment is out of range: "
                f"row={row_index}, segment={segment_index}"
            )
        length = segment_lengths[row_index][segment_index]
        offset = sum(segment_lengths[row_index][:segment_index])
        selected.append((row_index, offset, length))

    if not selected:
        raise ValueError("packed attention preshard selected no model calls")

    selected_lengths = [length for _, _, length in selected]
    max_length = max(selected_lengths)
    if output_sequence_length is not None:
        if output_sequence_length < max_length:
            raise ValueError(
                "packed attention output_sequence_length cannot truncate calls: "
                f"requested={output_sequence_length}, longest_call={max_length}"
            )
        max_length = output_sequence_length
    parent_rows = torch.tensor([row for row, _, _ in selected], device=input_ids.device)

    expanded = BatchedDataDict[Any]()
    for key, value in data.items():
        if isinstance(value, PackedTensor):
            raise NotImplementedError(
                "independent model-call packing does not yet support multimodal data"
            )
        if torch.is_tensor(value):
            if key == "input_lengths":
                expanded[key] = torch.tensor(
                    selected_lengths,
                    dtype=value.dtype,
                    device=value.device,
                )
            elif value.is_nested:
                value_row_lengths = value.offsets().diff()
                if input_row_lengths is None or not torch.equal(
                    value_row_lengths.cpu(), input_row_lengths.cpu()
                ):
                    raise ValueError(
                        "jagged packed-attention fields must have the same row "
                        f"lengths as input_ids; field={key!r}"
                    )
                values = value.values()
                offsets = value.offsets()
                output = values.new_zeros(
                    (len(selected), max_length, *values.shape[1:])
                )
                for index, (row, offset, length) in enumerate(selected):
                    row_start = int(offsets[row].item())
                    output[index, :length] = values[
                        row_start + offset : row_start + offset + length
                    ]
                expanded[key] = output
            elif (
                not input_ids_is_jagged
                and value.ndim > 1
                and value.shape[1] == input_ids.shape[1]
            ):
                output = value.new_zeros((len(selected), max_length, *value.shape[2:]))
                for index, (row, offset, length) in enumerate(selected):
                    output[index, :length] = value[row, offset : offset + length]
                expanded[key] = output
            else:
                expanded[key] = value.index_select(0, parent_rows.to(value.device))
        else:
            expanded[key] = [value[row] for row, _, _ in selected]
    return expanded


def reassemble_packed_attention_segments(
    segments: torch.Tensor,
    segment_lengths: Sequence[Sequence[int]],
    output_sequence_length: int,
) -> torch.Tensor:
    """Concatenate per-call values back into their logical rollout rows."""
    output = segments.new_zeros(
        (len(segment_lengths), output_sequence_length, *segments.shape[2:])
    )
    segment_index = 0
    for row, row_segments in enumerate(segment_lengths):
        offset = 0
        for length in row_segments:
            output[row, offset : offset + length] = segments[segment_index, :length]
            offset += length
            segment_index += 1
    if segment_index != segments.shape[0]:
        raise ValueError(
            "packed segment result count does not match metadata: "
            f"results={segments.shape[0]}, metadata={segment_index}"
        )
    return output
