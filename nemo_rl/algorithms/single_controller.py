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

"""SingleController: asyncio orchestrator for the RL training loop.

CPU-only Ray actor that runs two concurrent pumps and coordinates the
other actors via lightweight RPCs. SC sends control signals and reads
metadata only — model tensors still move through DataPlane or NCCL.

Data flow:
  _rollout_pump  → gen.generate_and_push(prompt, dp_client) ← RPC to GenWorker
                     GenWorker → dp_client.put_samples(...)
  _train_pump    → sampler.evict/select against TQReplayBuffer
                 → _advantage_stage(meta) → dp_client.get_samples(...)
                                        → adv_estimator.compute_advantage(...)
                                        → dp_client.put_samples(...)
                 → trainer.begin/train_microbatches/finish_train_step (split API,
                     driver-side TQPolicy via asyncio.to_thread)
                     Trainer → dp_client.get_samples(...)   (via its own client)
                 → dp_client.clear_samples(...)             ← SC clears after train
  _sync_weights  → WeightSynchronizer.sync_weights()
"""

from __future__ import annotations

import asyncio
import os
import time
from functools import partial
from typing import Any, Optional, Union

import ray
import torch

from nemo_rl.algorithms.async_utils.staleness_sampler import create_sampler
from nemo_rl.algorithms.grpo import (
    _write_latest_checkpoint_status,
    compute_and_apply_seq_logprob_error_masking,
)
from nemo_rl.algorithms.metric_utils import SetupTimingMetrics
from nemo_rl.algorithms.single_controller_utils.config import (
    AdvantageConfig,
    MasterConfig,
    SingleControllerSaveState,
    validate_sampler_buffer_capacity,
    validate_single_controller_config,
)
from nemo_rl.algorithms.single_controller_utils.setup import SingleControllerActorArgs
from nemo_rl.algorithms.single_controller_utils.utils import (
    advantage_group_ids_from_meta,
    aggregate_step_metrics,
    fields_for_put,
    reduce_advantage_pump_metrics,
    squeeze_trailing_unit_dim,
    tensor_field,
)
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import DP_CALIB_INPUT_FIELDS
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.sglang.sglang_generation import SGLangGeneration
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.models.policy.tq_policy import TQPolicy
from nemo_rl.utils.checkpoint import CheckpointManager
from nemo_rl.utils.logger import Logger
from nemo_rl.utils.timer import TimeoutChecker, Timer

Generation = Union[VllmGeneration, SGLangGeneration]


def _process_memory_metrics(phase: str) -> dict[str, float]:
    """Read process memory counters from procfs, expressed in GiB."""
    metrics: dict[str, float] = {}
    sources = {
        "/proc/self/status": {
            "VmHWM": "rss_high_water",
            "RssAnon": "rss_anon",
            "RssFile": "rss_file",
            "RssShmem": "rss_shmem",
        },
        "/proc/self/smaps_rollup": {
            "Rss": "rss",
            "Pss": "pss",
            "Pss_Anon": "pss_anon",
            "Pss_File": "pss_file",
            "Pss_Shmem": "pss_shmem",
            "Private_Clean": "private_clean",
            "Private_Dirty": "private_dirty",
            "Shared_Clean": "shared_clean",
            "Shared_Dirty": "shared_dirty",
        },
    }
    for path, fields in sources.items():
        try:
            with open(path) as proc_file:
                for line in proc_file:
                    key, separator, value = line.partition(":")
                    metric_name = fields.get(key)
                    if not separator or metric_name is None:
                        continue
                    value_parts = value.split()
                    if not value_parts:
                        continue
                    # Linux exposes these counters in KiB.
                    metrics[f"{phase}_{metric_name}_gib"] = float(value_parts[0]) / (
                        1024**2
                    )
        except OSError:
            continue
    return metrics


@ray.remote(num_cpus=1, num_gpus=0)  # pragma: no cover
class SingleControllerActor:
    """CPU-only Ray actor that orchestrates the RL training loop.

    Owns two concurrent asyncio tasks:
      - _rollout_pump: dispatches prompts to GenerationWorkerActor
      - _train_pump:   claims DataPlane meta, trains, clears consumed rows,
                       then runs _sync_weights (drain gate + weight
                       synchronization) inline after each optimizer step

    All other actors are passive — they expose methods and wait to be called.
    """

    def __init__(
        self,
        master_config: MasterConfig,
        actor_args: SingleControllerActorArgs,
        setup_timing_metrics: SetupTimingMetrics,
    ) -> None:
        """Initialize the SingleController actor.

        Args:
            master_config: SC MasterConfig.
            actor_args: Pre-built actor args from setup_single_controller.
            setup_timing_metrics: Driver-side setup timings; logged here (Logger isn't cloudpickleable).
        """
        validate_single_controller_config(master_config)

        self._advantage_cfg = AdvantageConfig()
        self._partition_id: str = actor_args.partition_id

        self._master_config = master_config
        self._async_cfg = master_config.async_rl
        self._policy_logprobs_required = not (
            master_config.loss_fn.force_on_policy_ratio
            and master_config.grpo.seq_logprob_error_threshold is None
        )
        self._reference_logprobs_required = not bool(
            master_config.grpo.skip_reference_policy_logprobs_calculation
        )
        self._dp_client = actor_args.dp_client
        self._gen: Generation = actor_args.gen_handle
        self._trainer: TQPolicy = actor_args.trainer_handle
        self._dataloader = actor_args.dataloader
        self._weight_synchronizer = actor_args.weight_synchronizer
        self._advantage_estimator = actor_args.advantage_estimator
        self._loss_fn = actor_args.loss_fn
        self._buffer = actor_args.tq_buffer
        self._rollout_manager = actor_args.rollout_manager
        # Rebind so writer and sampler share one buffer instance even
        # when Ray deserializes rollout_manager and tq_buffer separately.
        self._rollout_manager._tq_buffer = self._buffer

        # Built here, not on the driver: Logger backends (wandb/tb/...) hold
        # _thread.lock that Ray can't cloudpickle into the actor.
        self._logger = Logger(master_config.logger)  # type: ignore
        self._logger.log_hyperparams(master_config.model_dump())
        self._logger.log_metrics(
            setup_timing_metrics.to_metrics_dict(),
            step=getattr(
                actor_args, "resume_state", SingleControllerSaveState()
            ).train_steps,
            prefix="timing/setup",
        )
        self._timer = Timer()

        self._checkpointing_cfg = getattr(
            master_config, "checkpointing", {"enabled": False}
        )
        self._checkpointer = (
            CheckpointManager(self._checkpointing_cfg)
            if self._checkpointing_cfg["enabled"]
            else None
        )
        self._checkpoint_timeout = (
            TimeoutChecker(
                self._checkpointing_cfg.get("checkpoint_must_save_by"),
                fit_last_save_time=True,
            )
            if self._checkpointer is not None
            else None
        )
        self._last_checkpoint_path = getattr(actor_args, "last_checkpoint_path", None)

        # Pin clusters so RayVirtualCluster.__del__ doesn't remove the PGs.
        self._train_cluster = actor_args.train_cluster
        self._inference_cluster = actor_args.inference_cluster

        num_prompts_per_step = self._master_config.grpo.num_prompts_per_step
        self._sampler = create_sampler(
            self._buffer,
            self._async_cfg.sampler,
        )
        required_capacity = self._sampler.required_buffer_capacity(num_prompts_per_step)
        validate_sampler_buffer_capacity(
            self._async_cfg,
            required_capacity=required_capacity,
            sampler_name=type(self._sampler).__name__,
        )

        # ── asyncio state ──────────────────────────────────────────────────
        # Gate: cleared during _sync_weights, set when generation may proceed
        self._rollout_permitted: asyncio.Event = asyncio.Event()
        self._rollout_permitted.set()

        # Set only after _rollout_pump exhausts its configured epochs and all
        # dispatched tasks finish successfully. Rollout failures propagate
        # through run() instead of being reported as normal exhaustion.
        self._rollout_exhausted: asyncio.Event = asyncio.Event()

        # Count of in-flight generate_and_push calls
        self._inflight_rollouts: int = 0

        # Cancellation handles for in-flight rollout dispatches.
        self._dispatched_rollouts: set[asyncio.Task[None]] = set()

        # Backpressure valve: max unconsumed rollout groups allowed in DataPlane.
        # Acquired before each rollout dispatch; released when the buffer
        # drops a group (sampler.evict or post-train buffer.remove).
        self._buffer_capacity: asyncio.Semaphore = asyncio.Semaphore(
            self._async_cfg.max_buffered_rollouts
        )

        resume_state = getattr(actor_args, "resume_state", SingleControllerSaveState())
        self._trainer_version = resume_state.trainer_version
        self._train_steps = resume_state.train_steps
        self._total_valid_tokens = resume_state.total_valid_tokens
        self._dataloader_batch_index = 0
        self._restored_source_positions: set[tuple[int, int]] = set()
        self._current_epoch: int = 0
        self._step_log_dict: dict[str, list] = {
            "rewards": [],
            "masked_advantages": [],
            "sequence_lengths": [],
            "seq_logprob_error_records": [],
        }

        print(
            f"SingleControllerActor: "
            f"sampler={self._async_cfg.sampler.name} "
            f"buffer={self._async_cfg.max_buffered_rollouts} "
            f"inflight={self._async_cfg.max_inflight_prompts} "
            f"weight_sync={type(self._weight_synchronizer).__name__}",
            flush=True,
        )

    # ── public API ─────────────────────────────────────────────────────────

    async def run(self) -> dict[str, Any]:
        """Main entry point. Runs until max_train_steps is reached."""
        if self._last_checkpoint_path is not None:
            restore_metrics = await self._buffer.load_completed_rollouts(
                self._last_checkpoint_path,
                current_train_step=self._train_steps,
            )
            self._restored_source_positions = self._buffer.completed_source_positions()
            restored_groups = len(self._buffer)
            available_slots = self._async_cfg.max_buffered_rollouts - restored_groups
            if available_slots < 0:
                raise RuntimeError(
                    "restored TQ groups exceed async_rl.max_buffered_rollouts: "
                    f"{restored_groups} > {self._async_cfg.max_buffered_rollouts}"
                )
            self._buffer_capacity = asyncio.Semaphore(available_slots)
            print(
                "Restored committed TQ experience: "
                f"{restore_metrics['restored_groups']} group(s), "
                f"{restore_metrics['restored_bytes']} byte(s)",
                flush=True,
            )
        self._sampler.restore_dispatch_index(self._train_steps - 1)
        if self._checkpoint_timeout is not None:
            self._checkpoint_timeout.start_iterations()

        # Synchronize weights before starting the pumps
        await self._sync_weights()

        # Start the rollout and train pumps
        rollout_task = asyncio.create_task(self._rollout_pump())
        train_task = asyncio.create_task(self._train_pump())
        try:
            done, _ = await asyncio.wait(
                {rollout_task, train_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if rollout_task in done:
                # Propagate rollout failures immediately. A normally exhausted
                # rollout pump leaves the train pump to drain committed groups.
                await rollout_task
            await train_task
        finally:
            rollout_task.cancel()
            train_task.cancel()
            await asyncio.gather(rollout_task, train_task, return_exceptions=True)
            if self._checkpointer is not None:
                await asyncio.to_thread(self._checkpointer.shutdown)
            self._logger.finish()

        return {
            "train_steps": self._train_steps,
            "trainer_version": self._trainer_version,
        }

    async def ping(self) -> dict[str, Any]:
        """Liveness check — returns immediately if event loop is running."""
        return {
            "alive": True,
            "trainer_version": self._trainer_version,
            "train_steps": self._train_steps,
            "inflight_rollouts": self._inflight_rollouts,
            "rollout_permitted": self._rollout_permitted.is_set(),
            "epoch": self._current_epoch,
        }

    # ── internal helpers ───────────────────────────────────────────────────

    async def _ray_get(self, obj_ref: Any) -> Any:
        """Await a Ray ObjectRef without blocking the asyncio event loop."""
        return await obj_ref

    async def _call_dp(self, method_name: str, **kwargs) -> Any:
        """Call a DataPlaneClient method or a Ray actor exposing that method."""
        method = getattr(self._dp_client, method_name)
        remote = getattr(method, "remote", None)
        if remote is not None:
            return await self._ray_get(remote(**kwargs))
        result = method(**kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result

    async def _wait_for_rollout_permission(self) -> None:
        """Wait until rollout admission is currently enabled.

        ``asyncio.Event.clear`` does not revoke waiter futures already completed
        by ``set``. Rechecking after wakeup prevents such a stale wakeup from
        dispatching a rollout after a terminal pause.
        """
        while True:
            await self._rollout_permitted.wait()
            if self._rollout_permitted.is_set():
                return

    # ── the three pumps + the inline advantage stage ───────────────────────

    async def _rollout_pump(self) -> None:
        """Continuously dispatch rollout tasks until cancellation.

        Per batch:
          0. await sampler.admit(...) to wait until the batch may dispatch and
             obtain its target_step stamp.

        Per prompt:
          1. Acquire _buffer_capacity slot (backpressure)
          2. Acquire sem (cap concurrent in-flight rollouts)
          3. Wait for _rollout_permitted (paused during weight sync)
          4. Call rollout_manager.generate_and_push(prompt) — local async
             RolloutManager reserves a slot, runs the rollout, then commits the
             group via TQReplayBuffer (→ dp_client.put_samples + mark ready)
          5. Decrement _inflight_rollouts
        """
        sem = asyncio.Semaphore(self._async_cfg.max_inflight_prompts)
        self._rollout_exhausted.clear()
        print("rollout_pump: starting", flush=True)

        async def _dispatch_one_prompt(
            prompt: DatumSpec,
            target_step: Optional[int],
            source_batch_index: int,
            source_prompt_index: int,
            task_started_event: asyncio.Event,
        ) -> None:
            task_started_event.set()
            self._inflight_rollouts += 1
            try:
                await self._rollout_manager.generate_and_push(
                    prompt,
                    target_step=target_step,
                    source_batch_index=source_batch_index,
                    source_prompt_index=source_prompt_index,
                )
            except BaseException:
                # On success ownership transfers to the train pump, which
                # releases this permit after consuming the committed group.
                self._buffer_capacity.release()
                raise
            finally:
                self._inflight_rollouts -= 1
                sem.release()

            if self._async_cfg.diagnostics:
                content = ""
                for i in range(len(prompt["message_log"])):
                    if prompt["message_log"][i]["role"] == "user":
                        content = prompt["message_log"][i]["content"]
                        break
                print(f"  rollout done for prompt='{content[:20]}...'", flush=True)

        def _release_permits_if_task_not_started(
            _: asyncio.Task[Any],
            *,
            task_started_event: asyncio.Event,
        ) -> None:
            if not task_started_event.is_set():
                self._buffer_capacity.release()
                sem.release()

        max_epochs = self._master_config.grpo.max_num_epochs
        async with asyncio.TaskGroup() as rollout_tasks:
            while max_epochs is None or self._current_epoch < max_epochs:
                for prompt_batch in self._dataloader:
                    source_batch_index = self._dataloader_batch_index
                    self._dataloader_batch_index += 1
                    if source_batch_index < self._train_steps:
                        continue

                    target_step = await self._sampler.admit(
                        trainer_version_fn=lambda: self._trainer_version
                    )
                    if target_step is not None and target_step != source_batch_index:
                        raise RuntimeError(
                            "in-order sampler/source cursor mismatch after resume: "
                            f"target_step={target_step}, "
                            f"source_batch_index={source_batch_index}"
                        )

                    for prompt_idx in range(prompt_batch.size):
                        source_position = (source_batch_index, prompt_idx)
                        if source_position in self._restored_source_positions:
                            self._restored_source_positions.remove(source_position)
                            continue
                        prompt: DatumSpec = {  # type: ignore
                            k: v[prompt_idx] for k, v in prompt_batch.items()
                        }

                        # check if buffer is full
                        await self._buffer_capacity.acquire()
                        # check if inflight rollouts is full
                        await sem.acquire()
                        # wait for rollout to be permitted
                        await self._wait_for_rollout_permission()

                        task_started_event = asyncio.Event()
                        # dispatch rollout
                        task = rollout_tasks.create_task(
                            _dispatch_one_prompt(
                                prompt,
                                target_step,
                                source_batch_index,
                                prompt_idx,
                                task_started_event,
                            )
                        )
                        self._dispatched_rollouts.add(task)
                        task.add_done_callback(self._dispatched_rollouts.discard)
                        task.add_done_callback(
                            partial(
                                _release_permits_if_task_not_started,
                                task_started_event=task_started_event,
                            )
                        )

                    unresolved_for_batch = [
                        position
                        for position in self._restored_source_positions
                        if position[0] == source_batch_index
                    ]
                    if unresolved_for_batch:
                        raise RuntimeError(
                            "restored TQ source positions exceed the prompt batch: "
                            f"{unresolved_for_batch}"
                        )

                self._current_epoch += 1

        # Drain in-flight so return implies "all rollouts in TQ".
        inflight = list(self._dispatched_rollouts)
        if inflight:
            await asyncio.gather(*inflight, return_exceptions=True)

        self._rollout_exhausted.set()
        print(f"rollout_pump: completed {self._current_epoch} epoch(s)", flush=True)

    async def _cancel_inflight_rollouts(self) -> dict[str, int | float]:
        """Cancel admitted rollouts before a terminal checkpoint.

        ``RolloutManager.generate_and_push`` owns reservation and partial-write
        cleanup, so awaiting the cancelled dispatch tasks establishes a stable
        completed-only replay-buffer boundary before checkpoint serialization.
        The rollout admission gate must be cleared before calling this method.
        """
        if self._rollout_permitted.is_set():
            raise RuntimeError(
                "in-flight rollout cancellation requires rollout admission to be paused"
            )

        started_at = time.monotonic()
        cancelled_tasks: set[asyncio.Task[None]] = set()
        unexpected_errors: list[BaseException] = []
        initial_active_tasks = sum(
            not task.done() for task in self._dispatched_rollouts
        )
        cancellation_rounds = 0
        while True:
            tasks = [task for task in self._dispatched_rollouts if not task.done()]
            if not tasks:
                # Let pending done callbacks and any stale admission waiter run
                # before declaring the tracked task set stable.
                await asyncio.sleep(0)
                tasks = [task for task in self._dispatched_rollouts if not task.done()]
                if not tasks:
                    break

            cancellation_rounds += 1
            cancelled_tasks.update(tasks)
            for task in tasks:
                task.cancel()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            unexpected_errors.extend(
                result
                for result in results
                if isinstance(result, BaseException)
                and not isinstance(result, asyncio.CancelledError)
            )

        self._dispatched_rollouts.difference_update(
            task for task in self._dispatched_rollouts if task.done()
        )
        if unexpected_errors:
            error_types = sorted(type(error).__name__ for error in unexpected_errors)
            print(
                f"terminal rollout drain observed cleanup error(s): {error_types}",
                flush=True,
            )
        if self._inflight_rollouts != 0 or self._dispatched_rollouts:
            raise RuntimeError(
                "terminal rollout drain did not quiesce admitted work: "
                f"inflight={self._inflight_rollouts}, "
                f"tracked_tasks={len(self._dispatched_rollouts)}"
            )

        metrics: dict[str, int | float] = {
            "initial_inflight_rollouts": initial_active_tasks,
            "cancelled_inflight_rollouts": len(cancelled_tasks),
            "late_registered_rollouts": max(
                0, len(cancelled_tasks) - initial_active_tasks
            ),
            "rollout_cancellation_rounds": cancellation_rounds,
            "rollout_cleanup_errors": len(unexpected_errors),
            "remaining_inflight_rollouts": self._inflight_rollouts,
            "remaining_tracked_rollouts": len(self._dispatched_rollouts),
            "rollout_drain_time_s": time.monotonic() - started_at,
        }
        print(f"terminal rollout drain: {metrics}", flush=True)
        return metrics

    async def _train_pump(self) -> None:
        """Per-prompt-group streaming train loop.

        Per step:
          1. sampler.evict drops stale groups from the buffer and clears their TQ rows.
          2. sampler.select returns K prompt groups (or None) and drops them from the
             buffer; DP rows survive so the trainer can read them. Already trainable —
             buffer wrote training-shaped rows at rollout time.
          3. _advantage_stage(train_meta).
          4. trainer.train_microbatches_from_meta + finish_train_step.
          5. dp_client.clear_samples on consumed sample_ids; release _buffer_capacity
             per dropped group, then sync.
        """
        grpo_cfg = self._master_config.grpo

        while self._train_steps < grpo_cfg.max_num_steps:
            version_during_step = self._trainer_version
            groups_dispatched = 0
            min_sample_version = None
            step_open = False
            calibration_batches: list[BatchedDataDict[Any]] = []

            with self._timer.time("total_step_time"):
                while groups_dispatched < grpo_cfg.num_prompts_per_step:
                    # Wait for a selectable batch
                    with self._timer.time("exposed_generation"):
                        await asyncio.sleep(0)

                        # Evict stale groups
                        evicted = await self._sampler.evict(
                            current_train_weight=self._trainer_version,
                        )
                        if evicted:
                            print(
                                f"  evicted {evicted} stale prompt group(s)",
                                flush=True,
                            )
                            for _ in range(evicted):
                                self._buffer_capacity.release()

                        # Select a batch
                        max_prompt_groups = (
                            grpo_cfg.num_prompts_per_step - groups_dispatched
                        )
                        min_prompt_groups = min(
                            self._async_cfg.min_groups_for_streaming_train,
                            max_prompt_groups,
                        )
                        train_meta, num_groups = await self._sampler.select(
                            current_train_weight=self._trainer_version,
                            min_prompt_groups=min_prompt_groups,
                            max_prompt_groups=max_prompt_groups,
                        )

                        # If no batch is selectable, sleep and retry
                        if train_meta is None:
                            if self._rollout_exhausted.is_set():
                                buffered_groups = len(self._buffer)
                                if groups_dispatched == 0 and buffered_groups == 0:
                                    print(
                                        "train_pump: rollout exhausted and "
                                        "buffer drained",
                                        flush=True,
                                    )
                                    return
                                raise RuntimeError(
                                    "rollout exhausted before a complete training "
                                    f"step was assembled: dispatched "
                                    f"{groups_dispatched}/"
                                    f"{grpo_cfg.num_prompts_per_step} prompt "
                                    f"groups with {buffered_groups} group(s) "
                                    f"remaining in the buffer"
                                )
                            await asyncio.sleep(0.005)
                            continue

                        # Release buffer capacity
                        for _ in range(num_groups):
                            self._buffer_capacity.release()

                    # Compute prev_logprobs / ref_logprobs
                    if (
                        self._policy_logprobs_required
                        or self._reference_logprobs_required
                    ):
                        with self._timer.time("logprob_inference_prep"):
                            await asyncio.to_thread(
                                self._trainer.prepare_for_lp_inference
                            )
                        with self._timer.time("policy_and_reference_logprobs"):
                            if self._policy_logprobs_required:
                                await asyncio.to_thread(
                                    self._trainer.get_logprobs_from_meta, train_meta
                                )
                            if self._reference_logprobs_required:
                                await asyncio.to_thread(
                                    self._trainer.get_reference_policy_logprobs_from_meta,
                                    train_meta,
                                )

                    # Compute advantages
                    with self._timer.time("advantage_calculation"):
                        train_meta = await self._advantage_stage(train_meta)

                    # Train
                    with self._timer.time("training_prep"):
                        await asyncio.to_thread(self._trainer.prepare_for_training)
                    with self._timer.time("policy_training"):
                        if not step_open:
                            await asyncio.to_thread(
                                self._trainer.begin_train_step,
                                self._loss_fn,
                            )
                            step_open = True
                        await asyncio.to_thread(
                            self._trainer.train_microbatches_from_meta,
                            train_meta,
                        )

                    if train_meta.sequence_lengths:
                        self._step_log_dict["sequence_lengths"].extend(
                            int(s) for s in train_meta.sequence_lengths
                        )

                    if getattr(self._gen, "requires_kv_scale_sync", False):
                        calibration_fields = [
                            field
                            for field in (train_meta.fields or [])
                            if field in DP_CALIB_INPUT_FIELDS
                        ]
                        calibration_batches.append(
                            await asyncio.to_thread(
                                self._trainer.read_from_dataplane,
                                train_meta,
                                select_fields=calibration_fields,
                            )
                        )

                    # Refresh min_sample_version
                    curr_min_sample_version = min(
                        t["weight_version"]
                        for t in train_meta.tags  # type: ignore
                    )
                    if min_sample_version is not None:
                        min_sample_version = min(
                            min_sample_version, curr_min_sample_version
                        )
                    else:
                        min_sample_version = curr_min_sample_version

                    # Remove consumed sample_ids from the buffer
                    await self._call_dp(
                        "clear_samples",
                        sample_ids=list(train_meta.sample_ids),
                        partition_id=self._partition_id,
                    )

                    groups_dispatched += num_groups

                with self._timer.time("policy_training"):
                    result = await asyncio.to_thread(self._trainer.finish_train_step)

                step_metrics = aggregate_step_metrics(result)
                step_metrics.update(
                    reduce_advantage_pump_metrics(**self._step_log_dict)
                )
                self._step_log_dict = {k: [] for k in self._step_log_dict}

                self._trainer_version += 1
                self._train_steps += 1
                with self._timer.time("weight_sync"):
                    calibration_data = (
                        BatchedDataDict.from_batches(calibration_batches)
                        if calibration_batches
                        else None
                    )
                    await self._sync_weights(calibration_data=calibration_data)

            timing_metrics: dict[str, float] = self._timer.get_timing_metrics(
                reduction_op="sum"
            )  # type: ignore

            total_time = timing_metrics.get("total_step_time", 0.0)
            total_num_gpus = int(ray.cluster_resources().get("GPU", 0))
            if (
                total_time > 0
                and total_num_gpus > 0
                and "global_valid_toks" in step_metrics
            ):
                timing_metrics["valid_tokens_per_sec_per_gpu"] = (
                    step_metrics["global_valid_toks"] / total_time / total_num_gpus
                )

            print("\n⏱️  Timing:")
            print(f"  • Total step time: {total_time:.2f}s")
            for k, v in sorted(
                timing_metrics.items(), key=lambda item: item[1], reverse=True
            ):
                if k == "total_step_time":
                    continue
                percent = (v / total_time * 100) if total_time > 0 else 0.0
                print(f"  • {k}: {v:.2f}s ({percent:.1f}%)")

            # TODO: per-step train_data jsonl dump, vllm metrics logger,
            #   histogram log, rollout_metrics, seq_logprob_error_metrics,
            #   pretty-print "Training Results" block, print_performance_metrics.
            print(f"step_metrics={step_metrics}", flush=True)
            self._logger.log_metrics(
                step_metrics, step=self._train_steps, prefix="train"
            )
            self._logger.log_metrics(
                timing_metrics, step=self._train_steps, prefix="timing/train"
            )
            self._timer.reset()

            # min sample version refers to the version each consumed sample was
            # generated with; lag = training version - oldest sample version.
            lag = version_during_step - min_sample_version  # type: ignore
            print(
                f"train step {self._train_steps}/{grpo_cfg.max_num_steps}  "
                f"trainer_v={self._trainer_version}  "
                f"lag={lag}  ",
                flush=True,
            )

            self._total_valid_tokens += int(step_metrics.get("global_valid_toks", 0))
            should_save_by_timeout = False
            if self._checkpoint_timeout is not None:
                self._checkpoint_timeout.mark_iteration()
                should_save_by_timeout = self._checkpoint_timeout.check_save()

            is_last_step = self._train_steps >= grpo_cfg.max_num_steps
            should_save_by_step = False
            if self._checkpointer is not None:
                ft_save_period = self._checkpointing_cfg.get("ft_save_period")
                should_save_by_step = (
                    is_last_step
                    or self._train_steps % self._checkpointer.save_period == 0
                    or (
                        ft_save_period is not None
                        and self._train_steps % int(ft_save_period) == 0
                    )
                )
                if should_save_by_step or should_save_by_timeout:
                    terminal_drain_metrics: dict[str, int | float] = {}
                    if is_last_step or should_save_by_timeout:
                        self._rollout_permitted.clear()
                        terminal_drain_metrics.update(
                            _process_memory_metrics("pre_terminal_drain")
                        )
                        terminal_drain_metrics.update(
                            await self._cancel_inflight_rollouts()
                        )
                        terminal_drain_metrics.update(
                            _process_memory_metrics("post_terminal_drain")
                        )
                    checkpoint_start = time.monotonic()
                    replay_metrics = await self._save_checkpoint(step_metrics)
                    checkpoint_metrics = {
                        **replay_metrics,
                        **terminal_drain_metrics,
                        "save_time_s": time.monotonic() - checkpoint_start,
                    }
                    self._logger.log_metrics(
                        checkpoint_metrics,
                        step=self._train_steps,
                        prefix="checkpoint",
                    )
                    print(
                        f"checkpoint step {self._train_steps}: "
                        f"{replay_metrics['ready_groups']} committed TQ group(s), "
                        f"{replay_metrics['written_groups']} newly written, "
                        f"{replay_metrics['hardlinked_groups']} hard-linked",
                        flush=True,
                    )

            if should_save_by_timeout:
                print("Checkpoint deadline reached; ending this stage", flush=True)
                return

    async def _save_checkpoint(
        self, step_metrics: dict[str, Any]
    ) -> dict[str, int | float]:
        """Save trainer state and all committed, unconsumed TQ groups."""
        if self._checkpointer is None:
            raise RuntimeError(
                "checkpoint save requested while checkpointing is disabled"
            )

        checkpoint_memory_metrics = _process_memory_metrics("checkpoint_start")
        await asyncio.to_thread(self._checkpointer.finalize_pending)
        checkpoint_memory_metrics.update(
            _process_memory_metrics("checkpoint_after_finalize_pending")
        )
        previous_checkpoint_path = self._checkpointer.get_latest_checkpoint_path()
        dataloader_length = len(self._dataloader)
        save_state = SingleControllerSaveState(
            consumed_samples=(
                self._train_steps * self._master_config.grpo.num_prompts_per_step
            ),
            consumed_prompt_batches=self._train_steps,
            current_step=(
                self._train_steps % dataloader_length if dataloader_length else 0
            ),
            current_epoch=(
                self._train_steps // dataloader_length if dataloader_length else 0
            ),
            train_steps=self._train_steps,
            trainer_version=self._trainer_version,
            total_valid_tokens=self._total_valid_tokens,
        )
        training_info: dict[str, Any] = {
            **vars(save_state),
            # Preserve the legacy GRPO key for checkpoint inspection tooling.
            "total_steps": self._train_steps,
        }
        metric_name = self._checkpointing_cfg["metric_name"]
        if metric_name is not None:
            _, train_metric_name = metric_name.split(":", 1)
            if train_metric_name not in step_metrics:
                raise ValueError(
                    f"Metric {train_metric_name!r} not found in train metrics"
                )
            training_info[metric_name] = step_metrics[train_metric_name]

        checkpoint_path = self._checkpointer.init_tmp_checkpoint(
            self._train_steps,
            training_info,
            self._master_config,
        )
        await asyncio.to_thread(self._trainer.prepare_for_training)
        await asyncio.to_thread(
            self._trainer.save_checkpoint,
            weights_path=os.path.join(checkpoint_path, "policy", "weights"),
            optimizer_path=(
                os.path.join(checkpoint_path, "policy", "optimizer")
                if self._checkpointer.save_optimizer
                else None
            ),
            tokenizer_path=os.path.join(checkpoint_path, "policy", "tokenizer"),
            checkpointing_cfg=self._checkpointing_cfg,
        )
        checkpoint_memory_metrics.update(
            _process_memory_metrics("checkpoint_after_trainer_save")
        )
        replay_metrics = await self._buffer.save_completed_rollouts(
            checkpoint_path,
            previous_checkpoint_path=previous_checkpoint_path,
        )
        checkpoint_memory_metrics.update(
            _process_memory_metrics("checkpoint_after_replay_save")
        )
        self._checkpointer.begin_finalization(
            checkpoint_path,
            wait_fn=self._trainer.finalize_async_save,
        )
        _write_latest_checkpoint_status(
            self._checkpointer,
            last_checkpoint_step=self._train_steps,
        )
        return {**replay_metrics, **checkpoint_memory_metrics}

    async def _sync_weights(
        self,
        *,
        calibration_data: Optional[BatchedDataDict[Any]] = None,
    ) -> None:
        """Pause new rollout dispatches, synchronize weights, resume.

        SC owns the pause gate; in-flight generations continue through the
        refit — vLLM V1 async engine supports weight updates during pending
        requests.

        Flow:
          1. _rollout_permitted.clear()  — no new dispatches
          2. Optionally calibrate FP8 KV-cache scales.
          3. weight_synchronizer.sync_weights(kv_scales=...)
          4. _rollout_permitted.set()   — resume
        """
        self._rollout_permitted.clear()

        # TODO(#2625): Add drain-gate support during refit.

        t0 = time.monotonic()
        kv_scales = None
        if (
            getattr(self._gen, "requires_kv_scale_sync", False)
            and calibration_data is not None
        ):
            print("▶ Computing KV cache scales...", flush=True)
            calibration_result = await asyncio.to_thread(
                self._trainer.calibrate_qkv_fp8_scales,
                calibration_data,
                include_q=True,
            )
            kv_scales = calibration_result["layers"]

        await asyncio.to_thread(
            self._weight_synchronizer.sync_weights,
            kv_scales=kv_scales,
        )
        if self._async_cfg.recompute_kv_cache_after_weight_updates:
            self._gen.invalidate_kv_cache()
        elapsed = time.monotonic() - t0

        print(f"  _sync_weights: sync done in {elapsed:.3f}s", flush=True)
        self._rollout_manager.set_weight_version(self._trainer_version)
        self._rollout_permitted.set()

    async def _advantage_stage(self, meta: KVBatchMeta) -> KVBatchMeta:
        """Fetch advantage inputs, compute advantages, and write them back.

        SC owns the prompt-group-scoped advantage stage because the selected
        ``KVBatchMeta`` still contains complete prompt groups before trainer
        DP sharding. Tensor payloads still move through DataPlane: SC fetches
        only the configured advantage input columns and writes the computed
        ``advantages`` column back under the same ``sample_ids``.
        """
        if self._advantage_estimator is None:
            return meta
        adv_cfg = self._advantage_cfg

        data = await self._call_dp(
            "get_samples",
            sample_ids=meta.sample_ids,
            partition_id=meta.partition_id,
            select_fields=self._advantage_input_fields(),
        )

        advantage_group_ids = advantage_group_ids_from_meta(
            meta,
            expected_group_size=(self._master_config.grpo.num_generations_per_prompt),
        )
        rewards = squeeze_trailing_unit_dim(
            tensor_field(data, adv_cfg.reward_field)
        ).float()
        token_mask = tensor_field(data, adv_cfg.token_mask_field).float()
        sample_mask = squeeze_trailing_unit_dim(
            tensor_field(data, adv_cfg.sample_mask_field)
        ).float()

        if self._policy_logprobs_required:
            generation_logprobs = tensor_field(
                data,
                adv_cfg.generation_logprobs_field,
            )
            policy_logprobs = tensor_field(
                data,
                adv_cfg.policy_logprobs_field,
            )
            original_sample_mask = sample_mask.clone()
            logprob_data = BatchedDataDict(
                {
                    "token_mask": token_mask,
                    "sample_mask": sample_mask,
                    "prev_logprobs": policy_logprobs,
                    "generation_logprobs": generation_logprobs,
                }
            )
            seq_error_record = compute_and_apply_seq_logprob_error_masking(
                train_data=logprob_data,
                rewards=rewards,
                seq_logprob_error_threshold=(
                    self._master_config.grpo.seq_logprob_error_threshold
                ),
            )
            sample_mask = logprob_data["sample_mask"]
            valid_seq_mask = (
                token_mask[:, 1:] * original_sample_mask.unsqueeze(-1)
            ).sum(dim=-1) > 0
            kept_valid_seq_mask = valid_seq_mask & sample_mask.bool()
            seq_error_record["_num_valid_seqs_before_mask"] = float(
                valid_seq_mask.sum().item()
            )
            seq_error_record["_num_valid_seqs_after_mask"] = float(
                kept_valid_seq_mask.sum().item()
            )
            seq_error_record["_num_masked_correct"] = float(
                seq_error_record["masked_correct_pct"]
                * seq_error_record["num_masked_seqs"]
            )
            self._step_log_dict["seq_logprob_error_records"].append(seq_error_record)

        mask = token_mask * sample_mask.unsqueeze(-1)

        repeated_batch: dict[str, torch.Tensor] = {
            "total_reward": rewards,
        }
        for field_name in adv_cfg.repeated_batch_fields:
            repeated_batch[field_name] = squeeze_trailing_unit_dim(
                tensor_field(data, field_name)
            )

        kwargs: dict[str, torch.Tensor] = {}
        if self._policy_logprobs_required:
            kwargs["logprobs_policy"] = policy_logprobs
        if self._reference_logprobs_required:
            kwargs["logprobs_reference"] = tensor_field(
                data,
                adv_cfg.reference_logprobs_field,
            )

        advantages = self._advantage_estimator.compute_advantage(
            prompt_ids=advantage_group_ids,
            rewards=rewards,
            mask=mask,
            repeated_batch=repeated_batch,
            **kwargs,
        )
        response_advantages = torch.masked_select(advantages, mask.bool())
        self._step_log_dict["rewards"].append(rewards.detach().cpu())
        self._step_log_dict["masked_advantages"].append(
            response_advantages.detach().cpu()
        )

        fields_to_put = {adv_cfg.output_field: advantages}
        if self._master_config.grpo.seq_logprob_error_threshold is not None:
            fields_to_put[adv_cfg.sample_mask_field] = sample_mask

        await self._call_dp(
            "put_samples",
            sample_ids=meta.sample_ids,
            partition_id=meta.partition_id,
            fields=fields_for_put(meta, fields_to_put),
        )
        return meta.with_fields(list(fields_to_put))

    # ── utility helpers ────────────────────────────────────────────────────

    def _advantage_input_fields(self) -> list[str]:
        adv_cfg = self._advantage_cfg
        fields = [
            adv_cfg.reward_field,
            adv_cfg.token_mask_field,
            adv_cfg.sample_mask_field,
            *adv_cfg.repeated_batch_fields,
        ]
        if self._policy_logprobs_required:
            fields.extend(
                [
                    adv_cfg.policy_logprobs_field,
                    adv_cfg.generation_logprobs_field,
                ]
            )
        if self._reference_logprobs_required:
            fields.append(adv_cfg.reference_logprobs_field)
        return list(dict.fromkeys(fields))
