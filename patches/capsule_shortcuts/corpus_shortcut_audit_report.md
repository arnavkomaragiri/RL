# Training Capsule Shortcut Audit

Date: 2026-08-17

## Scope

This pass targets answer-bearing capsule contents only. Rubric ambiguity,
outcome-sensitive preprocessing, and protocols that explicitly permit derived
inputs are deferred to a separate scoring/data-specification pass.

The pre-final-mask snapshot audited for this pass contains 307 tasks and 9,825
capsule files. `discover_shortcuts.py` identified 113 candidate tasks from:

- 231 executable source files;
- 76 potentially answer-bearing documents;
- 198 result-like filenames; and
- 20 archives containing source or result-like members.

These counts are deliberately over-inclusive. Each actionable finding below was
checked against its task protocol and remaining capsule inputs.

## Newly Patchable Capsules

The following capsules contain paper analysis code, completed statistical
outputs, figures/manuscripts that state the answer, or a combination of these.
The remaining observations or intermediate inputs support an independent
analysis after removal.

```text
015ab7f4-b069-4912-ac0b-a3f8bd9c869d
047a8b26-2bbe-415d-803e-903c4d6ed214
08d9001f-42ba-49a2-bfb2-131a3898a97c
14289050-dcc7-43e7-a51e-a0f5b77303b8
1d65c580-960c-4258-b987-1bc4d0469820
33ca12c3-5a9a-45f4-aded-305cc3c27f07
382f157e-0902-4c2d-9c08-ffe9abc4684e
38a25e3c-1dd6-4b1f-b8ae-092134ec5ca9
3e09f25a-92ab-414f-9aee-a5c98f770c45
6043f842-703a-42db-b184-d77bb5d8b5d7
624fca77-a490-48f2-87ff-78e6edea8217
66ff78c6-2792-4d63-90b0-91abdc5bf96d
6fb44b38-5e45-4145-8c6c-9d6b283c8bb2
720852ec-500d-407e-9135-502db964be39
752243b8-06dc-4040-aff7-c38725b9b9ac
77f99441-0959-4031-84b9-de5d7c13c28e
7c15efeb-83d5-4079-87a7-a971b2ab1a27
7eb68875-35ac-4723-beff-b996875dc612
81cc0058-ec89-499a-b973-3402a286c0ce
98ca02a3-9cf7-4997-ae5a-9e8f939c01b0
9ba1ec89-b67b-47a8-bd74-d4441bf1d718
9d9e635a-cb6e-46c7-8da2-00b059d0a55f
a33a7c14-d960-4f5e-8d31-2fc41f9bd4f1
a4c16602-5520-41c4-8aa8-3e69ac62d806
a8350a95-b0d8-4e40-aeed-2bd297043086
b402da23-502b-4ebc-879f-30b3d87a3dc6
b92413f6-a7ea-4701-be98-6a17077c24cd
bd55bc97-8bfb-434b-a732-c5fa58e93423
be59f70b-e862-4173-993c-b4807c07b65f
c8cd05f4-ac6e-40a9-9fd8-27d0922d06c1
dee048b6-83d7-47ec-9c6e-b69a912b159c
e9ddbdbc-c74e-4371-b70a-6f4477cfec18
ec4d9a27-3a39-4d77-88e2-95896f28251c
f0807e09-0c08-4992-b7bc-b30a2cf30c46
```

### Solvability Repairs

- `015ab7f4-b069-4912-ac0b-a3f8bd9c869d`: retain `G`, `nG_keep`, GD,
  MPC, the principal functional gradient, parcel labels, and spin permutations;
  remove the paper scripts, `DE.csv`, repository README, and title figure. A
  verifier independently reconstructed `DE` from the first 14 gradient
  components and matched the supplied matrix within `5e-6` before completing the
  node regressions and spin test.
- `a8350a95-b0d8-4e40-aeed-2bd297043086`: the capsule originally contained only
  an expert Python solution. Copy the matching `GSE222412_rawCountMatrix.csv.gz`
  from capsule `4ee38ef1-ea48-4c6a-856a-ff7c7166413d`, then remove the solution.
- `720852ec-500d-407e-9135-502db964be39`: replace broken download stubs with the
  cleaned alignment/tree archive from related capsule
  `52dad468-cc9e-4b61-9f6c-4e71faeaad64`, then remove the paper and published
  RELAX/PAML results.
- `98ca02a3-9cf7-4997-ae5a-9e8f939c01b0`: retain only the survey, cross, site,
  and image inputs inside `data_and_code.zip`; remove its `code/`, `Figures/`,
  macOS metadata, and the manuscript.

## Newly Masked

`79843673-7806-4dd7-a946-36fd44988b91` contains only the manuscript PDF. The
NSD grating responses, pRF fits, ROI definitions, and subject-level values needed
for the requested nine-parameter model are absent.

## Retained By Design

The following candidate classes are not shortcut-patched in this pass:

- simulator or model source explicitly required by the protocol, such as the
  core MePD-KNDy equations after its ready-made coupled analysis is removed;
- domain tools such as FAPROTAX and data-retrieval-only helpers;
- supplemental or fitted tables the protocol explicitly names as its starting
  input; and
- README files that provide only schema or dependency information.

Some explicitly permitted result tables still make tasks too easy or encode an
answer-conditioned analysis. Those belong in the upcoming ambiguity and
outcome-sensitivity audit because fixing them may require changing both the
capsule and protocol rather than deleting a file alone.

## Privileged Context

Every removed or rewritten file is archived outside the policy capsule under
`capsule_<uuid>/...`. `manifest.json` records the original relative path, size,
SHA-256 digest, and removal reason. This preserves expert analyses for later
teacher, critic, or privileged-judge experiments without exposing them to the
policy environment.

The materialized clean split contains 306 tasks after applying all eight blocked
task masks. Its static audit reports zero remaining validated shortcut findings
and zero blocked tasks.
