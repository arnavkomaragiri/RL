# BBH Harbor OpenCode conversion

## Hypotest/DSS bundles

Build an uploadable Harbor dataset from a Hypotest task JSONL and a deduplicated
capsule root:

```bash
python3 patches/harbor_opencode/build_hypotest_harbor_bundle.py \
  --tasks-jsonl /path/to/tasks.jsonl \
  --capsules-dir /path/to/clean-capsules \
  --capsule-inventory-jsonl /path/to/capsule_inventory.jsonl \
  --exclude-file patches/capsule_shortcuts/training_mask.txt \
  --docker-image docker.io/example/bixbench@sha256:... \
  --output-root /path/to/harbor-output
```

The output stores capsules once under `capsules/`, Harbor task metadata under
`datasets/train/`, and an ordered NeMo-RL manifest under
`manifests/train.jsonl`. Task names use
`bbh-task__<zero-padded-row-index>__<capsule-uuid>`: this preserves every
paraphrase while making Gym's final `__`-delimited `{task_id}` resolve to the
shared capsule UUID. Mount `capsules/capsule_{task_id}` read-only at `/data` and
keep `environment/data` excluded from Harbor uploads.

Convert a cleaned BBH Harbor split into the OpenCode policy and agentic-judge
task format:

```bash
uv run python patches/harbor_opencode/convert_bbh_to_opencode.py \
  --source-split /path/to/bbh-harbor-rl-v0-clean/datasets/train \
  --output-split /path/to/bbh-harbor-opencode-rl-v0/datasets/train
```

After changing policy or verifier templates, refresh an existing converted
split without copying or rehashing capsule data:

```bash
uv run python patches/harbor_opencode/convert_bbh_to_opencode.py \
  --refresh-split /path/to/bbh-harbor-opencode-rl-v0/datasets/val
```

Generate the NeMo-RL prompt manifest from the materialized tasks after
filtering or refreshing a split:

```bash
uv run python patches/harbor_opencode/write_nemo_rl_manifest.py \
  --split /path/to/bbh-harbor-opencode-rl-v0/datasets/val \
  --exclude-file patches/capsule_shortcuts/validation_mask.txt \
  --output /path/to/bbh-harbor-opencode-rl-v0-val.jsonl
```

Exclusion entries must be an exact task name or the UUID after the task name's
final `__`. Unmatched entries are reported because masks may span dataset
revisions; `--require-all-exclusions` turns those reports into errors for strict
audits. `--input-manifest` applies the same checks when filtering an existing
JSONL manifest and requires `--reference-split` as the source of truth, making an
already-filtered manifest safe to filter again.

For the OpenSandbox scale-up checklist and the volume-template incident notes,
see [the OpenSandbox experiment runbook](./opensandbox_experiment_runbook.md).
