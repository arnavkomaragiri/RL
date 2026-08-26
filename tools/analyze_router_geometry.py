#!/usr/bin/env python3
"""Profile MoE router-vector geometry in Hugging Face safetensor checkpoints.

The script is intentionally independent of NeMo-RL, Transformers, PyTorch, and
the safetensors Python package. It reads safetensor headers and selected tensor
ranges directly, so checkpoint processing is sequential and memory-bounded.

Required dependency: NumPy. Matplotlib is optional and used only with --plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import mmap
import re
import struct
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


DEFAULT_ROUTER_REGEX = r"(?:^|\.)(?:gate|router)\.weight$"
DEFAULT_AUXILIARY_REGEX = r"(?:^|\.)mtp(?:\.|$)"
LAYER_REGEX = re.compile(r"(?:^|\.)(?:layers?|h)\.(\d+)(?:\.|$)")
STEP_REGEXES = (
    re.compile(r"(?:global[_-]?step|step)[_=-]?(\d+)", re.IGNORECASE),
    re.compile(r"checkpoint[_-]?(\d+)", re.IGNORECASE),
    re.compile(r"iter(?:ation)?[_=-]?(\d+)", re.IGNORECASE),
)
QUANTILES = (("min", 0.0), ("p05", 0.05), ("p50", 0.5), ("p95", 0.95), ("max", 1.0))


@dataclass(frozen=True)
class CheckpointSpec:
    """One checkpoint and its optional series metadata."""

    path: Path
    label: str
    series: str | None = None
    step: int | None = None


@dataclass(frozen=True)
class ModelMetadata:
    """Configuration fields relevant to router discovery and comparison."""

    architecture: str
    model_type: str
    hidden_size: int | None
    expected_num_experts: int | None
    configured_top_k: int | None


class SafeTensorReader:
    """Read selected uncompressed tensors from one safetensors file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = None
        self._mmap = None
        with path.open("rb") as stream:
            header_size_raw = stream.read(8)
            if len(header_size_raw) != 8:
                raise ValueError(f"{path}: truncated safetensors header length")
            (header_size,) = struct.unpack("<Q", header_size_raw)
            header_raw = stream.read(header_size)
        try:
            header = json.loads(header_raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{path}: invalid safetensors header: {error}") from error
        self.header: dict[str, dict[str, Any]] = {
            key: value for key, value in header.items() if key != "__metadata__"
        }
        self.data_start = 8 + header_size

    def __enter__(self) -> "SafeTensorReader":
        self._file = self.path.open("rb")
        self._mmap = mmap.mmap(self._file.fileno(), length=0, access=mmap.ACCESS_READ)
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def tensor(self, key: str) -> np.ndarray:
        """Return one tensor as a detached float32 NumPy array."""
        if self._mmap is None:
            raise RuntimeError("SafeTensorReader must be used as a context manager")
        metadata = self.header[key]
        shape = tuple(int(value) for value in metadata["shape"])
        start, end = (int(value) for value in metadata["data_offsets"])
        view = memoryview(self._mmap)[self.data_start + start : self.data_start + end]
        dtype = str(metadata["dtype"])
        expected_elements = math.prod(shape)

        if dtype == "BF16":
            bits = np.frombuffer(view, dtype="<u2", count=expected_elements).astype(
                np.uint32
            )
            array = np.left_shift(bits, 16).view(np.float32)
        elif dtype == "F16":
            array = np.frombuffer(view, dtype="<f2", count=expected_elements).astype(
                np.float32
            )
        elif dtype == "F32":
            array = np.frombuffer(view, dtype="<f4", count=expected_elements).copy()
        elif dtype == "F64":
            array = np.frombuffer(view, dtype="<f8", count=expected_elements).astype(
                np.float32
            )
        else:
            raise ValueError(
                f"{self.path}:{key}: unsupported router dtype {dtype!r}; "
                "supported dtypes are BF16, F16, F32, and F64"
            )
        del view
        return array.reshape(shape)


def recursive_config_value(config: dict[str, Any], names: Sequence[str]) -> Any | None:
    """Find the first named scalar in a possibly nested model config."""
    for name in names:
        value = config.get(name)
        if value is not None and not isinstance(value, (dict, list)):
            return value
    for value in config.values():
        if isinstance(value, dict):
            found = recursive_config_value(value, names)
            if found is not None:
                return found
    return None


def load_model_metadata(checkpoint: Path) -> ModelMetadata:
    """Read model metadata without importing Transformers."""
    config_path = checkpoint / "config.json"
    config: dict[str, Any] = {}
    if config_path.is_file():
        with config_path.open(encoding="utf-8") as stream:
            config = json.load(stream)

    architectures = config.get("architectures")
    if architectures is None:
        for value in config.values():
            if isinstance(value, dict) and value.get("architectures") is not None:
                architectures = value["architectures"]
                break
    if isinstance(architectures, list):
        architecture = str(architectures[0]) if architectures else "unknown"
    else:
        architecture = "unknown"
    model_type = str(recursive_config_value(config, ("model_type",)) or "unknown")
    hidden_size = recursive_config_value(config, ("hidden_size", "d_model"))
    num_experts = recursive_config_value(
        config,
        ("num_experts", "n_routed_experts", "moe_num_experts", "num_local_experts"),
    )
    top_k = recursive_config_value(
        config,
        ("num_experts_per_tok", "moe_top_k", "num_experts_per_token", "top_k"),
    )
    return ModelMetadata(
        architecture=architecture,
        model_type=model_type,
        hidden_size=int(hidden_size) if hidden_size is not None else None,
        expected_num_experts=int(num_experts) if num_experts is not None else None,
        configured_top_k=int(top_k) if top_k is not None else None,
    )


def safetensor_weight_map(checkpoint: Path) -> dict[str, Path]:
    """Map tensor names to files for sharded or single-file checkpoints."""
    preferred_index = checkpoint / "model.safetensors.index.json"
    index_paths = (
        [preferred_index]
        if preferred_index.is_file()
        else sorted(checkpoint.glob("*.safetensors.index.json"))
    )
    if index_paths:
        if len(index_paths) != 1:
            raise ValueError(
                f"{checkpoint}: found multiple safetensors index files: {index_paths}"
            )
        with index_paths[0].open(encoding="utf-8") as stream:
            index = json.load(stream)
        return {
            str(key): checkpoint / str(filename)
            for key, filename in index["weight_map"].items()
        }

    tensor_paths = sorted(checkpoint.glob("*.safetensors"))
    if not tensor_paths:
        raise FileNotFoundError(
            f"{checkpoint}: no model.safetensors.index.json or *.safetensors files found"
        )
    result: dict[str, Path] = {}
    for tensor_path in tensor_paths:
        reader = SafeTensorReader(tensor_path)
        for key in reader.header:
            if key in result:
                raise ValueError(
                    f"{checkpoint}: tensor {key!r} occurs in multiple files"
                )
            result[key] = tensor_path
    return result


def infer_step(path: Path) -> int | None:
    """Infer a training step from conventional checkpoint path components."""
    for part in reversed(path.parts):
        for pattern in STEP_REGEXES:
            match = pattern.search(part)
            if match:
                return int(match.group(1))
    return None


def resolve_manifest_path(raw_path: str, manifest: Path) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else manifest.parent / path


def spec_from_mapping(value: dict[str, Any], manifest: Path) -> CheckpointSpec:
    """Convert one manifest object into a checkpoint specification."""
    if "path" not in value:
        raise ValueError(
            f"{manifest}: manifest entry is missing required field 'path': {value}"
        )
    path = resolve_manifest_path(str(value["path"]), manifest)
    label = str(value.get("label") or path.name)
    series = str(value["series"]) if value.get("series") not in (None, "") else None
    step = (
        int(value["step"]) if value.get("step") not in (None, "") else infer_step(path)
    )
    return CheckpointSpec(path=path, label=label, series=series, step=step)


def read_manifest(path: Path) -> list[CheckpointSpec]:
    """Read CSV, JSONL, or plain path/TSV checkpoint manifests."""
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as stream:
            return [
                spec_from_mapping(dict(row), path) for row in csv.DictReader(stream)
            ]

    specs: list[CheckpointSpec] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("{"):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"{path}:{line_number}: invalid JSON: {error}"
                    ) from error
                specs.append(spec_from_mapping(value, path))
                continue
            fields = line.split("\t")
            if len(fields) == 1:
                checkpoint = resolve_manifest_path(fields[0], path)
                specs.append(
                    CheckpointSpec(
                        path=checkpoint,
                        label=checkpoint.name,
                        step=infer_step(checkpoint),
                    )
                )
            elif 2 <= len(fields) <= 4:
                label, raw_checkpoint = fields[:2]
                checkpoint = resolve_manifest_path(raw_checkpoint, path)
                series = fields[2] if len(fields) >= 3 and fields[2] else None
                step = (
                    int(fields[3])
                    if len(fields) == 4 and fields[3]
                    else infer_step(checkpoint)
                )
                specs.append(CheckpointSpec(checkpoint, label, series, step))
            else:
                raise ValueError(
                    f"{path}:{line_number}: expected path or label<TAB>path<TAB>series<TAB>step"
                )
    return specs


def collect_checkpoint_specs(args: argparse.Namespace) -> list[CheckpointSpec]:
    specs = [
        CheckpointSpec(
            path=Path(raw).expanduser(),
            label=Path(raw).name,
            step=infer_step(Path(raw)),
        )
        for raw in args.checkpoints
    ]
    for manifest in args.manifest:
        specs.extend(read_manifest(Path(manifest).expanduser()))
    if not specs:
        raise ValueError("provide at least one checkpoint path or --manifest")

    labels: set[str] = set()
    normalized: list[CheckpointSpec] = []
    for spec in specs:
        resolved = spec.path.resolve()
        if spec.label in labels:
            raise ValueError(f"checkpoint label {spec.label!r} is not unique")
        labels.add(spec.label)
        normalized.append(CheckpointSpec(resolved, spec.label, spec.series, spec.step))
    return normalized


def summary_statistics(values: np.ndarray, prefix: str) -> dict[str, float]:
    """Return stable scalar summaries for one nonempty numeric vector."""
    values64 = np.asarray(values, dtype=np.float64)
    result = {
        f"{prefix}_mean": float(np.mean(values64)),
        f"{prefix}_std": float(np.std(values64)),
    }
    for name, quantile in QUANTILES:
        result[f"{prefix}_{name}"] = float(np.quantile(values64, quantile))
    return result


def orient_router_matrix(
    tensor: np.ndarray,
    *,
    key: str,
    expected_num_experts: int | None,
    hidden_size: int | None,
) -> np.ndarray:
    """Orient one router as [experts, vector_dimension] and validate its shape."""
    if tensor.ndim != 2:
        raise ValueError(
            f"{key}: expected a 2-D router matrix, got shape {tensor.shape}"
        )
    rows, columns = tensor.shape
    if expected_num_experts is not None:
        if rows == expected_num_experts:
            oriented = tensor
        elif columns == expected_num_experts:
            oriented = tensor.T
        else:
            raise ValueError(
                f"{key}: neither dimension of shape {tensor.shape} equals configured "
                f"expert count {expected_num_experts}"
            )
    elif rows < columns:
        oriented = tensor
    elif columns < rows:
        oriented = tensor.T
    else:
        raise ValueError(
            f"{key}: cannot infer expert dimension from square shape {tensor.shape}"
        )

    if hidden_size is not None and oriented.shape[1] != hidden_size:
        print(
            f"warning: {key}: router vector dimension {oriented.shape[1]} differs from "
            f"configured hidden size {hidden_size}",
            file=sys.stderr,
        )
    return np.ascontiguousarray(oriented, dtype=np.float32)


def layer_identity(key: str) -> tuple[int | None, str]:
    match = LAYER_REGEX.search(key)
    if not match:
        return None, key.split(".", maxsplit=1)[0]
    component = key[: match.start()].strip(".") or "model"
    return int(match.group(1)), component


def covariance_factor(gram: np.ndarray) -> np.ndarray:
    """Return a stable square root of a positive-semidefinite Gram matrix."""
    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(gram, dtype=np.float64))
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return (eigenvectors * np.sqrt(eigenvalues)[None, :]).astype(np.float32)


def top_k_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    indices = np.argpartition(scores, kth=scores.shape[1] - top_k, axis=1)[:, -top_k:]
    return np.sort(indices, axis=1)


def simulate_top_k_stability(
    gram: np.ndarray,
    *,
    samples: int,
    chunk_size: int,
    noise_levels: Sequence[float],
    top_k_values: Sequence[int],
    rng: np.random.Generator,
) -> list[dict[str, float | int]]:
    """Estimate routing changes under h' = h + sigma * epsilon, both standard normal."""
    if samples <= 0:
        return []
    factor = covariance_factor(gram)
    num_experts = factor.shape[0]
    valid_top_k = sorted({value for value in top_k_values if 0 < value < num_experts})
    counters = {
        (noise, top_k): [0, 0, 0] for noise in noise_levels for top_k in valid_top_k
    }
    remaining = samples
    while remaining:
        current = min(remaining, chunk_size)
        base_standard = rng.standard_normal((current, num_experts), dtype=np.float32)
        noise_standard = rng.standard_normal((current, num_experts), dtype=np.float32)
        scores = base_standard @ factor.T
        score_noise = noise_standard @ factor.T
        for top_k in valid_top_k:
            baseline = top_k_indices(scores, top_k)
            for noise in noise_levels:
                perturbed = top_k_indices(scores + float(noise) * score_noise, top_k)
                shared_members = np.count_nonzero(
                    np.any(baseline[:, :, None] == perturbed[:, None, :], axis=2),
                    axis=1,
                )
                replaced_experts = top_k - shared_members
                counts = counters[(noise, top_k)]
                counts[0] += int(np.count_nonzero(replaced_experts))
                counts[1] += int(np.sum(replaced_experts))
                counts[2] += current
        remaining -= current
    return [
        {
            "noise_std": float(noise),
            "top_k": int(top_k),
            "flip_rate": differing / total,
            "mean_replaced_experts": replaced_experts / total,
            "samples": total,
        }
        for (noise, top_k), (differing, replaced_experts, total) in sorted(
            counters.items()
        )
    ]


def analyze_router(
    weights: np.ndarray,
    *,
    simulation_samples: int,
    simulation_chunk_size: int,
    noise_levels: Sequence[float],
    top_k_values: Sequence[int],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Compute row, pair, spectral, nearest-neighbor, and simulation metrics."""
    gram = weights @ weights.T
    norm_squared = np.clip(np.diag(gram), 0.0, None)
    norms = np.sqrt(norm_squared)
    if np.any(norms == 0):
        raise ValueError("router contains a zero-norm expert vector")

    cosine = gram / np.outer(norms, norms)
    cosine = np.clip(cosine, -1.0, 1.0)
    np.fill_diagonal(cosine, -np.inf)
    nearest_expert = np.argmax(cosine, axis=1)
    nearest_cosine = cosine[np.arange(len(norms)), nearest_expert]
    nearest_angle = np.degrees(np.arccos(np.clip(nearest_cosine, -1.0, 1.0)))

    upper_i, upper_j = np.triu_indices(len(norms), k=1)
    pair_cosine = cosine[upper_i, upper_j]
    pair_angle = np.degrees(np.arccos(np.clip(pair_cosine, -1.0, 1.0)))
    pair_separator_squared = np.clip(
        norm_squared[upper_i] + norm_squared[upper_j] - 2.0 * gram[upper_i, upper_j],
        0.0,
        None,
    )
    pair_separator = np.sqrt(pair_separator_squared)
    nearest_separator = np.sqrt(
        np.clip(
            norm_squared
            + norm_squared[nearest_expert]
            - 2.0 * gram[np.arange(len(norms)), nearest_expert],
            0.0,
            None,
        )
    )

    eigenvalues = np.clip(np.linalg.eigvalsh(gram.astype(np.float64)), 0.0, None)
    spectral_squared = float(eigenvalues[-1])
    frobenius_squared = float(np.sum(norm_squared, dtype=np.float64))
    probabilities = eigenvalues / max(
        float(np.sum(eigenvalues)), np.finfo(np.float64).tiny
    )
    positive_probabilities = probabilities[probabilities > 0]
    effective_rank = float(
        np.exp(-np.sum(positive_probabilities * np.log(positive_probabilities)))
    )

    row_normalized = weights / norms[:, None]
    normalized_gram = row_normalized @ row_normalized.T
    simulations: list[dict[str, Any]] = []
    simulation_seed = int(rng.integers(0, np.iinfo(np.int64).max))
    for geometry, current_gram in (
        ("original", gram),
        ("row_normalized", normalized_gram),
    ):
        for result in simulate_top_k_stability(
            current_gram,
            samples=simulation_samples,
            chunk_size=simulation_chunk_size,
            noise_levels=noise_levels,
            top_k_values=top_k_values,
            rng=np.random.default_rng(simulation_seed),
        ):
            result["geometry"] = geometry
            simulations.append(result)

    return {
        "norms": norms,
        "weight_rms": norms / math.sqrt(weights.shape[1]),
        "nearest_expert": nearest_expert,
        "nearest_cosine": nearest_cosine,
        "nearest_angle": nearest_angle,
        "nearest_separator": nearest_separator,
        "pair_i": upper_i,
        "pair_j": upper_j,
        "pair_cosine": pair_cosine,
        "pair_angle": pair_angle,
        "pair_separator": pair_separator,
        "spectral_norm": math.sqrt(spectral_squared),
        "frobenius_norm": math.sqrt(frobenius_squared),
        "stable_rank": frobenius_squared
        / max(spectral_squared, np.finfo(np.float64).tiny),
        "effective_rank": effective_rank,
        "simulations": simulations,
    }


def csv_fieldnames(prefixes: Sequence[str], base: Sequence[str]) -> list[str]:
    fields = list(base)
    for prefix in prefixes:
        fields.extend((f"{prefix}_mean", f"{prefix}_std"))
        fields.extend(f"{prefix}_{name}" for name, _ in QUANTILES)
    return fields


EXPERT_FIELDS = (
    "checkpoint",
    "series",
    "step",
    "router_key",
    "component",
    "layer",
    "expert",
    "norm",
    "weight_rms",
    "nearest_expert",
    "nearest_cosine",
    "nearest_angle_deg",
    "nearest_separator_norm",
)
PAIR_FIELDS = (
    "checkpoint",
    "series",
    "step",
    "router_key",
    "component",
    "layer",
    "expert_i",
    "expert_j",
    "cosine",
    "angle_deg",
    "separator_norm",
)
LAYER_BASE_FIELDS = (
    "checkpoint",
    "series",
    "step",
    "model_type",
    "router_key",
    "component",
    "layer",
    "num_experts",
    "vector_dimension",
    "spectral_norm",
    "frobenius_norm",
    "stable_rank",
    "effective_rank",
)
LAYER_FIELDS = csv_fieldnames(
    (
        "norm",
        "weight_rms",
        "pair_cosine",
        "pair_angle_deg",
        "separator_norm",
        "nearest_angle_deg",
    ),
    LAYER_BASE_FIELDS,
)
CHECKPOINT_BASE_FIELDS = (
    "checkpoint",
    "path",
    "series",
    "step",
    "architecture",
    "model_type",
    "configured_hidden_size",
    "configured_num_experts",
    "configured_top_k",
    "num_router_matrices",
    "num_expert_vectors",
    "num_expert_pairs",
    "mean_layer_stable_rank",
    "mean_layer_effective_rank",
)
CHECKPOINT_FIELDS = csv_fieldnames(
    (
        "norm",
        "weight_rms",
        "pair_cosine",
        "pair_angle_deg",
        "separator_norm",
        "nearest_angle_deg",
    ),
    CHECKPOINT_BASE_FIELDS,
)
SIMULATION_FIELDS = (
    "checkpoint",
    "series",
    "step",
    "router_key",
    "component",
    "layer",
    "geometry",
    "noise_std",
    "top_k",
    "flip_rate",
    "mean_replaced_experts",
    "samples",
)
SIMULATION_SUMMARY_FIELDS = (
    "checkpoint",
    "series",
    "step",
    "geometry",
    "noise_std",
    "top_k",
    "num_router_matrices",
    "router_sample_count",
    "flip_rate_mean",
    "flip_rate_min",
    "flip_rate_max",
    "mean_replaced_experts",
)


def write_rows(writer: csv.DictWriter, rows: Iterable[dict[str, Any]]) -> None:
    for row in rows:
        writer.writerow({key: row.get(key) for key in writer.fieldnames})


def summarize_simulations(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-router simulations without adding an interpretation."""
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["checkpoint"],
            row["series"],
            row["step"],
            row["geometry"],
            row["noise_std"],
            row["top_k"],
        )
        grouped.setdefault(key, []).append(row)

    summaries = []
    for key, group in sorted(
        grouped.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        sample_count = sum(int(row["samples"]) for row in group)
        flip_rates = [float(row["flip_rate"]) for row in group]
        summaries.append(
            {
                "checkpoint": key[0],
                "series": key[1],
                "step": key[2],
                "geometry": key[3],
                "noise_std": key[4],
                "top_k": key[5],
                "num_router_matrices": len(group),
                "router_sample_count": sample_count,
                "flip_rate_mean": sum(
                    float(row["flip_rate"]) * int(row["samples"]) for row in group
                )
                / sample_count,
                "flip_rate_min": min(flip_rates),
                "flip_rate_max": max(flip_rates),
                "mean_replaced_experts": sum(
                    float(row["mean_replaced_experts"]) * int(row["samples"])
                    for row in group
                )
                / sample_count,
            }
        )
    return summaries


def analyze_checkpoint(
    spec: CheckpointSpec,
    *,
    router_pattern: re.Pattern[str],
    auxiliary_pattern: re.Pattern[str],
    include_auxiliary: bool,
    writers: dict[str, csv.DictWriter],
    write_pairs: bool,
    simulation_samples: int,
    simulation_chunk_size: int,
    noise_levels: Sequence[float],
    requested_top_k: Sequence[int],
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = load_model_metadata(spec.path)
    weight_map = safetensor_weight_map(spec.path)
    router_keys = sorted(key for key in weight_map if router_pattern.search(key))
    if not include_auxiliary:
        router_keys = [key for key in router_keys if not auxiliary_pattern.search(key)]
    if not router_keys:
        raise ValueError(
            f"{spec.path}: no router tensors matched regex {router_pattern.pattern!r}"
        )

    keys_by_file: dict[Path, list[str]] = {}
    for key in router_keys:
        keys_by_file.setdefault(weight_map[key], []).append(key)

    all_norms: list[np.ndarray] = []
    all_weight_rms: list[np.ndarray] = []
    all_pair_cosine: list[np.ndarray] = []
    all_pair_angle: list[np.ndarray] = []
    all_separator: list[np.ndarray] = []
    all_nearest_angle: list[np.ndarray] = []
    stable_ranks: list[float] = []
    effective_ranks: list[float] = []
    simulation_rows: list[dict[str, Any]] = []
    router_count = 0
    top_k_values = list(requested_top_k)
    if metadata.configured_top_k is not None:
        top_k_values.append(metadata.configured_top_k)

    rng = np.random.default_rng(seed)
    for shard_path in sorted(keys_by_file):
        with SafeTensorReader(shard_path) as reader:
            for key in sorted(keys_by_file[shard_path]):
                weights = orient_router_matrix(
                    reader.tensor(key),
                    key=key,
                    expected_num_experts=metadata.expected_num_experts,
                    hidden_size=metadata.hidden_size,
                )
                layer, component = layer_identity(key)
                metrics = analyze_router(
                    weights,
                    simulation_samples=simulation_samples,
                    simulation_chunk_size=simulation_chunk_size,
                    noise_levels=noise_levels,
                    top_k_values=top_k_values,
                    rng=rng,
                )
                router_count += 1
                all_norms.append(metrics["norms"])
                all_weight_rms.append(metrics["weight_rms"])
                all_pair_cosine.append(metrics["pair_cosine"])
                all_pair_angle.append(metrics["pair_angle"])
                all_separator.append(metrics["pair_separator"])
                all_nearest_angle.append(metrics["nearest_angle"])
                stable_ranks.append(metrics["stable_rank"])
                effective_ranks.append(metrics["effective_rank"])

                common = {
                    "checkpoint": spec.label,
                    "series": spec.series,
                    "step": spec.step,
                    "router_key": key,
                    "component": component,
                    "layer": layer,
                }
                expert_rows = (
                    {
                        **common,
                        "expert": expert,
                        "norm": float(metrics["norms"][expert]),
                        "weight_rms": float(metrics["weight_rms"][expert]),
                        "nearest_expert": int(metrics["nearest_expert"][expert]),
                        "nearest_cosine": float(metrics["nearest_cosine"][expert]),
                        "nearest_angle_deg": float(metrics["nearest_angle"][expert]),
                        "nearest_separator_norm": float(
                            metrics["nearest_separator"][expert]
                        ),
                    }
                    for expert in range(weights.shape[0])
                )
                write_rows(writers["experts"], expert_rows)

                if write_pairs:
                    pair_rows = (
                        {
                            **common,
                            "expert_i": int(metrics["pair_i"][index]),
                            "expert_j": int(metrics["pair_j"][index]),
                            "cosine": float(metrics["pair_cosine"][index]),
                            "angle_deg": float(metrics["pair_angle"][index]),
                            "separator_norm": float(metrics["pair_separator"][index]),
                        }
                        for index in range(len(metrics["pair_i"]))
                    )
                    write_rows(writers["pairs"], pair_rows)

                layer_row: dict[str, Any] = {
                    **common,
                    "model_type": metadata.model_type,
                    "num_experts": weights.shape[0],
                    "vector_dimension": weights.shape[1],
                    "spectral_norm": metrics["spectral_norm"],
                    "frobenius_norm": metrics["frobenius_norm"],
                    "stable_rank": metrics["stable_rank"],
                    "effective_rank": metrics["effective_rank"],
                }
                for values, prefix in (
                    (metrics["norms"], "norm"),
                    (metrics["weight_rms"], "weight_rms"),
                    (metrics["pair_cosine"], "pair_cosine"),
                    (metrics["pair_angle"], "pair_angle_deg"),
                    (metrics["pair_separator"], "separator_norm"),
                    (metrics["nearest_angle"], "nearest_angle_deg"),
                ):
                    layer_row.update(summary_statistics(values, prefix))
                write_rows(writers["layers"], (layer_row,))

                for simulation in metrics["simulations"]:
                    row = {**common, **simulation}
                    simulation_rows.append(row)
                    write_rows(writers["simulations"], (row,))

    aggregate_values = {
        "norm": np.concatenate(all_norms),
        "weight_rms": np.concatenate(all_weight_rms),
        "pair_cosine": np.concatenate(all_pair_cosine),
        "pair_angle_deg": np.concatenate(all_pair_angle),
        "separator_norm": np.concatenate(all_separator),
        "nearest_angle_deg": np.concatenate(all_nearest_angle),
    }
    checkpoint_row: dict[str, Any] = {
        "checkpoint": spec.label,
        "path": str(spec.path),
        "series": spec.series,
        "step": spec.step,
        **asdict(metadata),
        "configured_hidden_size": metadata.hidden_size,
        "configured_num_experts": metadata.expected_num_experts,
        "configured_top_k": metadata.configured_top_k,
        "num_router_matrices": router_count,
        "num_expert_vectors": len(aggregate_values["norm"]),
        "num_expert_pairs": len(aggregate_values["pair_cosine"]),
        "mean_layer_stable_rank": float(np.mean(stable_ranks)),
        "mean_layer_effective_rank": float(np.mean(effective_ranks)),
    }
    for prefix, values in aggregate_values.items():
        checkpoint_row.update(summary_statistics(values, prefix))
    write_rows(writers["checkpoints"], (checkpoint_row,))
    return checkpoint_row, simulation_rows


def write_markdown_summary(
    path: Path, rows: Sequence[dict[str, Any]], errors: Sequence[dict[str, str]]
) -> None:
    columns = (
        "checkpoint",
        "model_type",
        "num_router_matrices",
        "norm_mean",
        "norm_p95",
        "separator_norm_mean",
        "pair_angle_deg_mean",
        "nearest_angle_deg_mean",
    )
    lines = [
        "# Router geometry summary",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            values.append(f"{value:.6g}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        (
            "",
            "## Reported quantities",
            "",
            "- `norm_*`: distribution of expert router-row L2 norms.",
            "- `separator_norm_*`: distribution of pairwise `||w_i-w_j||` values.",
            "- `pair_angle_deg_*`: distribution of angles between expert router rows.",
            "- `nearest_angle_deg_*`: each expert's angle to its most aligned peer.",
            "",
            "Optional simulation results are written to `simulations.csv`; no hypothesis conclusion "
            "is inferred by this report.",
        )
    )
    if errors:
        lines.extend(("", "## Errors", ""))
        lines.extend(f"- `{error['checkpoint']}`: {error['error']}" for error in errors)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plots(
    output_dir: Path, checkpoint_rows: Sequence[dict[str, Any]], layer_csv: Path
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("--plots requires matplotlib") from error

    layers: dict[str, list[dict[str, str]]] = {}
    with layer_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            layers.setdefault(row["checkpoint"], []).append(row)

    figure, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    for checkpoint, rows in layers.items():
        ordered = sorted(
            rows, key=lambda row: int(row["layer"]) if row["layer"] else -1
        )
        layer = [
            int(row["layer"]) if row["layer"] else index
            for index, row in enumerate(ordered)
        ]
        axes[0].plot(
            layer, [float(row["norm_mean"]) for row in ordered], label=checkpoint
        )
        axes[1].plot(
            layer,
            [float(row["nearest_angle_deg_mean"]) for row in ordered],
            label=checkpoint,
        )
    axes[0].set(ylabel="Mean router-row L2 norm", title="Router norms by layer")
    axes[1].set(
        xlabel="Layer",
        ylabel="Mean nearest angle (degrees)",
        title="Nearest-expert angles",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize="small")
    figure.savefig(output_dir / "router_geometry_by_layer.png", dpi=180)
    plt.close(figure)

    series_rows = [row for row in checkpoint_rows if row.get("step") is not None]
    if series_rows:
        figure, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
        by_series: dict[str, list[dict[str, Any]]] = {}
        for row in series_rows:
            by_series.setdefault(
                str(row.get("series") or row["model_type"]), []
            ).append(row)
        for series, rows in by_series.items():
            ordered = sorted(rows, key=lambda row: int(row["step"]))
            axes[0].plot(
                [row["step"] for row in ordered],
                [row["norm_mean"] for row in ordered],
                marker="o",
                label=series,
            )
            axes[1].plot(
                [row["step"] for row in ordered],
                [row["separator_norm_mean"] for row in ordered],
                marker="o",
                label=series,
            )
        axes[0].set(ylabel="Mean router-row norm", title="Router norms over training")
        axes[1].set(
            xlabel="Training step",
            ylabel="Mean separator norm",
            title="Gap-noise gain over training",
        )
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend(fontsize="small")
        figure.savefig(output_dir / "router_geometry_over_training.png", dpi=180)
        plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze MoE router norms, expert-vector angles, and synthetic top-k stability.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "checkpoints", nargs="*", help="Hugging Face safetensor checkpoint directories"
    )
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        help=(
            "CSV with path,label,series,step columns; JSONL objects with those fields; or text "
            "containing paths or label<TAB>path<TAB>series<TAB>step"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("router_geometry_results")
    )
    parser.add_argument("--router-regex", default=DEFAULT_ROUTER_REGEX)
    parser.add_argument("--auxiliary-regex", default=DEFAULT_AUXILIARY_REGEX)
    parser.add_argument("--include-auxiliary-routers", action="store_true")
    parser.add_argument(
        "--write-pairs",
        action="store_true",
        help="Write every expert pair to pairs.csv",
    )
    parser.add_argument(
        "--plots", action="store_true", help="Generate PNG plots; requires matplotlib"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--simulation-samples",
        type=int,
        default=0,
        help="Gaussian h/h+noise samples per router and geometry; zero disables simulation",
    )
    parser.add_argument("--simulation-chunk-size", type=int, default=4096)
    parser.add_argument(
        "--noise-std",
        type=float,
        action="append",
        default=None,
        help="Relative std of isotropic activation perturbation; repeat for multiple values",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        action="append",
        default=[],
        help="Top-k value for simulation; configured model top-k is always included",
    )
    parser.add_argument("--seed", type=int, default=12345)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.simulation_samples < 0:
        raise ValueError("--simulation-samples must be nonnegative")
    if args.simulation_chunk_size <= 0:
        raise ValueError("--simulation-chunk-size must be positive")
    noise_levels = args.noise_std or [1e-4, 1e-3, 1e-2]
    if any(value <= 0 for value in noise_levels):
        raise ValueError("--noise-std values must be positive")
    if any(value <= 0 for value in args.top_k):
        raise ValueError("--top-k values must be positive")

    specs = collect_checkpoint_specs(args)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "experts": output_dir / "experts.csv",
        "layers": output_dir / "layers.csv",
        "checkpoints": output_dir / "checkpoints.csv",
        "simulations": output_dir / "simulations.csv",
        "simulation_summary": output_dir / "simulation_summary.csv",
    }
    if args.write_pairs:
        output_paths["pairs"] = output_dir / "pairs.csv"
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing outputs: {existing}")

    fieldnames = {
        "experts": EXPERT_FIELDS,
        "layers": LAYER_FIELDS,
        "checkpoints": CHECKPOINT_FIELDS,
        "simulations": SIMULATION_FIELDS,
        "simulation_summary": SIMULATION_SUMMARY_FIELDS,
        "pairs": PAIR_FIELDS,
    }
    streams = {
        name: path.open("w", newline="", encoding="utf-8")
        for name, path in output_paths.items()
    }
    writers = {
        name: csv.DictWriter(stream, fieldnames=fieldnames[name], extrasaction="ignore")
        for name, stream in streams.items()
    }
    for writer in writers.values():
        writer.writeheader()

    checkpoint_rows: list[dict[str, Any]] = []
    all_simulation_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    try:
        router_pattern = re.compile(args.router_regex)
        auxiliary_pattern = re.compile(args.auxiliary_regex)
        for index, spec in enumerate(specs, start=1):
            print(f"[{index}/{len(specs)}] {spec.label}: {spec.path}", flush=True)
            try:
                checkpoint_row, simulation_rows = analyze_checkpoint(
                    spec,
                    router_pattern=router_pattern,
                    auxiliary_pattern=auxiliary_pattern,
                    include_auxiliary=args.include_auxiliary_routers,
                    writers=writers,
                    write_pairs=args.write_pairs,
                    simulation_samples=args.simulation_samples,
                    simulation_chunk_size=args.simulation_chunk_size,
                    noise_levels=noise_levels,
                    requested_top_k=args.top_k,
                    seed=args.seed + index,
                )
                checkpoint_rows.append(checkpoint_row)
                all_simulation_rows.extend(simulation_rows)
                print(
                    "  routers={num_router_matrices} norm_mean={norm_mean:.6g} "
                    "separator_mean={separator_norm_mean:.6g} nearest_angle_mean={nearest_angle_deg_mean:.4f}".format(
                        **checkpoint_row
                    ),
                    flush=True,
                )
            except (
                Exception
            ) as error:  # Continue a long checkpoint sweep unless requested otherwise.
                message = f"{type(error).__name__}: {error}"
                errors.append(
                    {"checkpoint": spec.label, "path": str(spec.path), "error": message}
                )
                print(f"  ERROR: {message}", file=sys.stderr, flush=True)
                if args.fail_fast:
                    raise
    finally:
        for stream in streams.values():
            stream.close()

    simulation_summaries = summarize_simulations(all_simulation_rows)
    with output_paths["simulation_summary"].open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=SIMULATION_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(simulation_summaries)

    payload = {
        "schema_version": 1,
        "router_regex": args.router_regex,
        "include_auxiliary_routers": args.include_auxiliary_routers,
        "simulation": {
            "samples_per_router_geometry": args.simulation_samples,
            "noise_std": noise_levels,
            "requested_top_k": args.top_k,
            "seed": args.seed,
        },
        "checkpoints": checkpoint_rows,
        "simulation_summaries": simulation_summaries,
        "errors": errors,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "errors.json").write_text(
        json.dumps(errors, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown_summary(output_dir / "SUMMARY.md", checkpoint_rows, errors)
    if args.plots and checkpoint_rows:
        make_plots(output_dir, checkpoint_rows, output_paths["layers"])

    print(f"Wrote results to {output_dir.resolve()}")
    return 1 if errors else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
