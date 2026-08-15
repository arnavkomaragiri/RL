# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Minimal behavioral invariants for the data-plane wiring.

* ``examples/run_grpo._select_trainer`` dispatches the legacy trainer
  when ``data_plane`` is absent and the sync trainer when enabled.
* The ``DataPlaneClient`` ABC carries every method adapters depend on.
"""

from __future__ import annotations

import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[3]


def test_run_grpo_dispatches_both_trainers():
    """``examples/run_grpo._select_trainer`` returns the TQ-mediated
    ``grpo_train_sync`` iff ``data_plane.enabled`` is true, and the
    legacy ``grpo_train`` otherwise."""
    import sys

    sys.path.insert(0, str(REPO / "examples"))
    try:
        from run_grpo import _select_trainer
    finally:
        sys.path.pop(0)
    from nemo_rl.algorithms.grpo import MasterConfig, grpo_train
    from nemo_rl.algorithms.grpo_sync import grpo_train_sync

    cfg_legacy = MasterConfig.model_construct(data_plane=None)
    assert _select_trainer(cfg_legacy) is grpo_train

    cfg_sync = MasterConfig.model_construct(data_plane={"enabled": True})
    assert _select_trainer(cfg_sync) is grpo_train_sync


def test_nemo_gym_runner_dispatches_both_sync_trainers():
    """The Gym entrypoint must honor data_plane.enabled as run_grpo does."""
    import sys

    sys.path.insert(0, str(REPO / "examples" / "nemo_gym"))
    try:
        from run_grpo_nemo_gym import _select_sync_trainer
    finally:
        sys.path.pop(0)
    from nemo_rl.algorithms.grpo import MasterConfig, grpo_train
    from nemo_rl.algorithms.grpo_sync import grpo_train_sync

    cfg_legacy = MasterConfig.model_construct(data_plane=None)
    assert _select_sync_trainer(cfg_legacy) is grpo_train

    cfg_sync = MasterConfig.model_construct(data_plane={"enabled": True})
    assert _select_sync_trainer(cfg_sync) is grpo_train_sync


def test_harbor_tq_smoke_recipe_selects_sync_tq_configuration():
    from nemo_rl.utils.config import load_config

    recipe = (
        REPO
        / "examples"
        / "configs"
        / "recipes"
        / "llm"
        / (
            "grpo-qwen3-30ba3b-thinking-4n8g-megatron-cp2-r3-async-gym-"
            "harbor-bbh-smoke-tq-simple.yaml"
        )
    )
    cfg = load_config(recipe)

    assert cfg.data_plane.enabled is True
    assert cfg.data_plane.impl == "transfer_queue"
    assert cfg.data_plane.backend == "simple"
    assert cfg.grpo.async_grpo.enabled is False
    assert cfg.grpo.invalid_tool_call_advantage is None
    assert cfg.grpo.malformed_thinking_advantage is None


def test_sync_trainer_rejects_message_level_advantage_penalties():
    from nemo_rl.algorithms.grpo import GRPOConfig, MasterConfig
    from nemo_rl.algorithms.grpo_sync import (
        _raise_if_message_level_advantage_penalties_enabled,
    )

    cfg_disabled = MasterConfig.model_construct(grpo=GRPOConfig())
    _raise_if_message_level_advantage_penalties_enabled(cfg_disabled)

    cfg_enabled = MasterConfig.model_construct(
        grpo=GRPOConfig(
            invalid_tool_call_advantage=-5.0,
            malformed_thinking_advantage=None,
        )
    )
    with pytest.raises(
        NotImplementedError,
        match="grpo.invalid_tool_call_advantage",
    ):
        _raise_if_message_level_advantage_penalties_enabled(cfg_enabled)


@pytest.mark.parametrize(
    "method",
    [
        "register_partition",
        "claim_meta",
        "get_data",
        "put_samples",
        "get_samples",
        "clear_samples",
        "check_consumption_status",
        "close",
    ],
)
def test_data_plane_client_abc_method_present(method: str) -> None:
    """The ``DataPlaneClient`` ABC is the swap surface; a silent rename
    is a breaking change for every adapter."""
    from nemo_rl.data_plane.interfaces import DataPlaneClient

    assert hasattr(DataPlaneClient, method), (
        f"DataPlaneClient ABC is missing required method {method!r}. "
        "This is a breaking change for every adapter."
    )
