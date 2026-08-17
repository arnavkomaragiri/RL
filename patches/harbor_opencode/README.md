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
  --output /path/to/bbh-harbor-opencode-rl-v0-val.jsonl
```
