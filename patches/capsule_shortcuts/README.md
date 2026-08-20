# Capsule shortcut cleanup

The audit and patch rationale lives in `capsule_shortcut_doc.md`,
`capsule_shortcut_patch_report.md`, `corpus_shortcut_audit_report.md`, and
`validation_shortcut_audit_report.md`. The Harbor source snapshot predates those
patches.

Generate a machine-readable candidate inventory before adding new patches:

```bash
uv run python patches/capsule_shortcuts/discover_shortcuts.py \
  /path/to/datasets/train --output /tmp/train-shortcut-candidates.json
```

The discovery report is intentionally over-inclusive. Executable simulators,
retrieval helpers, and precomputed inputs explicitly required by a protocol must
be reviewed before removal.

Create a clean, hard-linked source split without modifying the original:

```bash
uv run python patches/capsule_shortcuts/prepare_clean_split.py prepare \
  /path/to/bbh-harbor-rl-v0/datasets/train \
  /path/to/bbh-harbor-rl-v0-clean/datasets/train \
  --privileged-context-dir /path/to/bbh-harbor-rl-v0-privileged/train
```

The command applies the validated data patches and removes task UUIDs in
`training_mask.txt`. Before patching, every removed or rewritten file is retained
under the optional privileged-context directory, with its SHA-256 digest and
removal reason recorded in `manifest.json`. Files that require content changes
are replaced atomically, so both the original source split and privileged copy
remain unchanged. The resulting split should be used as the input to
`patches/harbor_opencode/convert_bbh_to_opencode.py`; this ensures the converter hashes the cleaned
capsule contents.

Mask validation tasks whose generated protocols cannot be satisfied by their
capsule data:

```bash
uv run python patches/capsule_shortcuts/prepare_clean_split.py prepare \
  /path/to/bbh-harbor-opencode-rl-v0/datasets/val \
  /path/to/bbh-harbor-opencode-rl-v0-clean/datasets/val \
  --mask-file patches/capsule_shortcuts/validation_mask.txt
```

This leaves the source split unchanged and removes the UUIDs listed in
`validation_mask.txt` from the hard-linked destination.

Audit any split without modifying it:

```bash
uv run python patches/capsule_shortcuts/prepare_clean_split.py audit /path/to/datasets/train
```

Pass `--mask-file patches/capsule_shortcuts/validation_mask.txt` when auditing
the validation split.
