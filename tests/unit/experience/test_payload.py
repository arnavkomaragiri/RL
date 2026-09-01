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

from __future__ import annotations

import torch

from nemo_rl.data.packed_rollouts import (
    TREE_ATTENTION_EDGE_LENGTHS,
    TREE_ATTENTION_EDGE_SOURCE_INDICES,
    TREE_ATTENTION_EDGE_TARGET_IDS,
    TREE_ATTENTION_LAYOUTS,
)
from nemo_rl.experience.interfaces import Completion, PromptGroupRecord
from nemo_rl.experience.payload import pack_payload, record_to_train_batch
from nemo_rl.algorithms.grpo import _compact_exact_nemo_gym_call_sequences


def _routes(start: int, count: int) -> torch.Tensor:
    token_routes = torch.arange(start, start + count, dtype=torch.int16).view(
        count, 1, 1
    )
    topk_offsets = torch.arange(2, dtype=torch.int16).view(1, 1, 2)
    return (token_routes + topk_offsets).expand(count, 2, 2).contiguous()


def _fallback_routes(count: int) -> torch.Tensor:
    return torch.arange(2, dtype=torch.int16).view(1, 1, 2).expand(count, 2, 2)


def _completion(
    route_start: int,
    reward: float,
    *,
    env_token_ids: tuple[int, ...] = (30,),
    with_routes: bool = True,
) -> Completion:
    message_log = [
        {
            "role": "user",
            "content": "prompt",
            "token_ids": torch.tensor([10, 11]),
            "routed_experts": _routes(route_start, 2),
        },
        {
            "role": "assistant",
            "content": "answer",
            "token_ids": torch.tensor([20, 21]),
            "generation_logprobs": torch.tensor([-0.1, -0.2]),
            "routed_experts": _routes(route_start + 2, 2),
        },
        {
            "role": "user",
            "content": "environment",
            "token_ids": torch.tensor(env_token_ids),
            "routed_experts": _fallback_routes(len(env_token_ids)),
        },
    ]
    if not with_routes:
        for message in message_log:
            message.pop("routed_experts")
    return Completion(
        message_log=message_log,
        env_extras=None,
        truncated=False,
        reward=reward,
    )


def _record(completions: list[Completion]) -> PromptGroupRecord:
    return PromptGroupRecord(
        prompt_idx=0,
        prompt=[
            {
                "role": "user",
                "content": "prompt",
                "token_ids": torch.tensor([10, 11]),
            }
        ],
        extra_env_info=None,
        metadata={"task_name": "test"},
        completions=completions,
        rollout_metrics={},
    )


def test_record_to_train_batch_preserves_routed_experts_in_tq_payload() -> None:
    record = _record(
        [
            _completion(route_start=10, reward=1.0),
            _completion(
                route_start=30,
                reward=2.0,
                env_token_ids=(30, 31),
            ),
        ]
    )

    train_batch = record_to_train_batch(
        record,
        pad_value_dict={"token_ids": 0, "input_ids": 0},
    )

    expected_routes = [
        torch.cat((_routes(10, 4), _fallback_routes(1))),
        torch.cat((_routes(30, 4), _fallback_routes(2))),
    ]
    assert train_batch["input_lengths"].tolist() == [5, 6]
    assert train_batch["routed_experts"].shape == (2, 6, 2, 2)
    assert torch.equal(
        train_batch["routed_experts"][0, :5],
        expected_routes[0],
    )
    assert torch.equal(
        train_batch["routed_experts"][1],
        expected_routes[1],
    )

    sample_ids, fields, tags = pack_payload(
        train_batch,
        weight_version=3,
        group_id="group",
    )
    assert sample_ids == ["group_g0", "group_g1"]
    assert "routed_experts" in fields
    packed_routes = fields["routed_experts"]
    assert packed_routes.is_nested
    packed_rows = list(packed_routes.unbind())
    assert torch.equal(packed_rows[0], expected_routes[0])
    assert torch.equal(packed_rows[1], expected_routes[1])
    assert tags == [
        {"weight_version": 3, "group_id": "group", "rollout_index": 0},
        {"weight_version": 3, "group_id": "group", "rollout_index": 1},
    ]


def test_record_to_train_batch_omits_routed_experts_when_absent() -> None:
    record = _record([_completion(route_start=10, reward=1.0, with_routes=False)])

    train_batch = record_to_train_batch(
        record,
        pad_value_dict={"token_ids": 0, "input_ids": 0},
    )
    assert "routed_experts" not in train_batch

    _, fields, _ = pack_payload(
        train_batch,
        weight_version=3,
        group_id="group",
    )
    assert "routed_experts" not in fields


def _failed_completion() -> Completion:
    """A trajectory whose first generation raised: prompt only, no routes."""
    return Completion(
        message_log=[
            {
                "role": "user",
                "content": "prompt",
                "token_ids": torch.tensor([10, 11]),
            }
        ],
        env_extras=None,
        truncated=False,
        reward=0.0,
    )


def test_record_to_train_batch_backfills_routes_for_failed_completion() -> None:
    """A group is packable when only some completions generated (and so have routes)."""
    record = _record([_completion(route_start=10, reward=1.0), _failed_completion()])

    train_batch = record_to_train_batch(
        record,
        pad_value_dict={"token_ids": 0, "input_ids": 0},
    )

    assert train_batch["input_lengths"].tolist() == [5, 2]
    routes = train_batch["routed_experts"]
    assert routes.shape == (2, 5, 2, 2)
    assert torch.equal(routes[0, :5], torch.cat((_routes(10, 4), _fallback_routes(1))))
    # The completion that never generated gets the all--1 missing-route sentinel,
    # so Megatron routes those tokens with its own router.
    assert torch.equal(routes[1, :2], torch.full((2, 2, 2), -1, dtype=routes.dtype))
    # It is fully loss-masked either way.
    assert train_batch["token_mask"][1, :2].tolist() == [0, 0]

    _, fields, _ = pack_payload(train_batch, weight_version=3, group_id="group")
    assert "routed_experts" in fields
    assert list(fields["routed_experts"].unbind())[1].shape == (2, 2, 2)


def test_record_to_train_batch_builds_tree_nodes_and_sampled_edges() -> None:
    call_1 = [
        {"role": "user", "content": "p", "token_ids": torch.tensor([10, 11])},
        {
            "role": "assistant",
            "content": "a1",
            "token_ids": torch.tensor([20, 21]),
            "generation_logprobs": torch.tensor([-0.1, -0.2]),
        },
    ]
    call_2 = [
        {
            "role": "user",
            "content": "p+a1+tool",
            "token_ids": torch.tensor([30, 31, 32]),
        },
        {
            "role": "assistant",
            "content": "a2",
            "token_ids": torch.tensor([40, 41]),
            "generation_logprobs": torch.tensor([-0.3, -0.4]),
        },
    ]
    completion = Completion(
        message_log=call_2,
        env_extras=None,
        truncated=False,
        reward=1.0,
        training_message_logs=[call_1, call_2],
    )

    train_batch = record_to_train_batch(
        _record([completion]),
        pad_value_dict={"token_ids": 0, "input_ids": 0},
    )

    layout = train_batch[TREE_ATTENTION_LAYOUTS][0]
    assert layout.segment_lengths == (4, 5)
    assert layout.segment_parents == (-1, -1)
    assert train_batch["input_lengths"].tolist() == [9]
    assert train_batch["input_ids"][0, :9].tolist() == [
        10,
        11,
        20,
        21,
        30,
        31,
        32,
        40,
        41,
    ]
    assert train_batch["prompt_ids_for_adv"][0, :3].tolist() == [30, 31, 32]
    assert train_batch[TREE_ATTENTION_EDGE_LENGTHS].tolist() == [4]
    assert train_batch[TREE_ATTENTION_EDGE_SOURCE_INDICES][0, :4].tolist() == [
        1,
        2,
        6,
        7,
    ]
    assert train_batch[TREE_ATTENTION_EDGE_TARGET_IDS][0, :4].tolist() == [
        20,
        21,
        40,
        41,
    ]
    assert int(train_batch["token_mask"].sum().item()) == 4

    _, fields, _ = pack_payload(train_batch, weight_version=0, group_id="group")
    assert TREE_ATTENTION_LAYOUTS not in fields
    assert [row.shape[0] for row in fields["input_ids"].unbind()] == [9]
    assert [row.shape[0] for row in fields["generation_logprobs"].unbind()] == [5]
    assert [
        row.shape[0] for row in fields[TREE_ATTENTION_EDGE_SOURCE_INDICES].unbind()
    ] == [4]


def test_record_to_train_batch_accepts_precompacted_exact_calls() -> None:
    call_1 = [
        {"role": "user", "content": "p", "token_ids": torch.tensor([10, 11])},
        {
            "role": "assistant",
            "content": "a1",
            "token_ids": torch.tensor([20, 21]),
            "generation_logprobs": torch.tensor([-0.1, -0.2]),
            "routed_experts": _routes(10, 2),
        },
    ]
    call_2 = [
        {
            "role": "user",
            "content": "p+a1+tool",
            "token_ids": torch.tensor([30, 31, 32]),
            "routed_experts": _routes(20, 3),
        },
        {
            "role": "assistant",
            "content": "a2",
            "token_ids": torch.tensor([40, 41]),
            "generation_logprobs": torch.tensor([-0.3, -0.4]),
            "routed_experts": _routes(30, 2),
        },
    ]
    tree = _compact_exact_nemo_gym_call_sequences(
        [call_1, call_2],
        materialize_storage=True,
    )
    completion = Completion(
        message_log=call_2,
        env_extras=None,
        truncated=False,
        reward=1.0,
        exact_call_tree=tree,
    )

    train_batch = record_to_train_batch(
        _record([completion]),
        pad_value_dict={"token_ids": 0, "input_ids": 0},
    )

    assert train_batch[TREE_ATTENTION_LAYOUTS] == [tree.layout]
    assert train_batch["input_lengths"].tolist() == [tree.layout.unique_token_count]
    assert train_batch[TREE_ATTENTION_EDGE_LENGTHS].tolist() == [4]
    assert int(train_batch["token_mask"].sum().item()) == 4
