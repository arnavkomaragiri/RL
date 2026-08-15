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


@dataclass(frozen=True)
class PackedAttentionLayout:
    """How independent call rows map back to logical rollout rows."""

    segment_lengths: Sequence[Sequence[int]]
    output_sequence_length: int


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
    output = tensor.new_zeros((len(segments), max_length, *tensor.shape[2:]))
    for index, segment in enumerate(segments):
        output[index, : segment.shape[0]] = segment
    return output, torch.tensor(lengths, dtype=torch.long, device=tensor.device)


def expand_batched_data_for_packed_attention(
    data: BatchedDataDict[Any],
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
                    value, segment_lengths
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
) -> BatchedDataDict[Any]:
    """Expand logical rows, then retain selected ``(row, segment)`` calls.

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
    with_metadata = BatchedDataDict[Any](dict(data.items()))
    with_metadata[PACKED_ATTENTION_SEGMENT_LENGTHS] = segment_lengths
    expanded, layout = expand_batched_data_for_packed_attention(with_metadata)
    assert layout is not None

    row_offsets: list[int] = []
    offset = 0
    for row_segments in segment_lengths:
        row_offsets.append(offset)
        offset += len(row_segments)

    selected_indices = []
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
        selected_indices.append(row_offsets[row_index] + segment_index)

    if not selected_indices:
        raise ValueError("packed attention preshard selected no model calls")
    return expanded.select_indices(selected_indices)


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
