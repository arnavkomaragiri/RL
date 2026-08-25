# BBH Harbor OpenCode conversion

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
