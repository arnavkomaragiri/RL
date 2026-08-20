# Validation Capsule Shortcut Audit

Date: 2026-08-17

## Scope

This pass applies the training shortcut standard to all 51 validation capsules.
The source contains 2,088 files and occupies 5.6 GB. Automated discovery marked
17 candidate capsules; task-protocol review and a separate image/workbook/source
inspection found additional answer-bearing artifacts whose filenames were not
covered by the discovery heuristics.

## Patched Capsules

The following five capsules retain sufficient independent inputs after cleanup:

- `1cf79c8c-fb8c-453c-8788-c8958ab6f152`: remove the supplied CHIP burden
  result figure; retain per-sample variant calls, cohort metadata, and CHIP genes.
- `2a8a40d4-05b0-4eec-8bd2-825f61fc9f5d`: remove two fibroblast GO enrichment
  maps; retain raw featureCounts, sample metadata, and gene annotations.
- `30b33e47-92d8-4372-a9f6-da32896493d0`: remove the workbook containing the
  completed multinomial comparisons and exact rubric p-values; retain per-sample
  variant calls and metadata.
- `33b801bb-9b47-4a0a-9314-05325c82fde7`: remove the complete expert notebook,
  which states the conclusion and exact adjusted p-values, and two blood GO
  enrichment maps; retain raw featureCounts, sample metadata, and annotations.
- `7718a922-ce2c-4e59-900b-84fe06050ce6`: remove the supplied cohort-wise CHIP
  effect-type result figure; retain per-sample variant calls and metadata.

The artifact checksum sidecars are removed with their corresponding artifacts.
Every removed file is copied to the privileged-context archive and recorded in
its SHA-256 manifest before the clean split is changed.

## Solvability Repair

`4ef3fcd8-1c35-466f-9d93-49b92f4ea760` requires a blood-versus-fibroblast DEG
comparison but originally contains only blood inputs. The clean split copies the
raw fibroblast featureCounts table and sample metadata from related validation
capsule `2a8a40d4-05b0-4eec-8bd2-825f61fc9f5d`. The expert notebook and
enrichment maps are not copied.

## Masked Capsules

Five capsules cannot satisfy their generated protocols from independent inputs:

- `0923d260-fe1b-4fb4-4398-79edf546e584`: contains only precomputed differential
  expression results, not the required count matrix and genotype/media metadata.
- `1d54e4a7-8b0f-4224-bd31-efcfded0d46c`: lacks the genomic and RNA-seq reads
  required for assembly, alignment, and quantification.
- `a02b761a-02b6-46b5-9d5e-2964d5a74960`: lacks the acute and chronic T-cell
  screens required by the protocol and rubric.
- `d37acffb-d802-4ce0-8caa-66f50667367a`: contains precomputed DE result objects
  but no count matrix or sample metadata required to reproduce them.
- `40cbef03-b5c3-4448-b00f-0ba2965dea9b`: contains aggregate tumor/control
  protein values and completed p-values, but no sample-level abundances needed
  for QC, normalization, testing, effect sizes, or confidence intervals.

## Retained By Design

- BUSCO `full_table.tsv` files are required to identify complete single-copy
  orthologs; IQ-TREE model files are inference metadata rather than conclusions.
- FeatureCounts files with `final` in their names are raw count matrices.
- `MeRIP_RNA_result.xlsx` is the per-gene summary table explicitly specified as
  the starting input for its contingency-table task.
- `Model.csv` is DepMap cell-line metadata needed to align expression and CRISPR
  dependency matrices.
- Precomputed quorum-sensing DE objects are retained only where the protocol
  explicitly permits importing differential-expression results.

The resulting clean validation split contains 46 tasks. Rubric ambiguity and
outcome-sensitive preprocessing remain outside this shortcut-only pass.

## Verification

The materialized clean split reports zero validated shortcut findings and zero
blocked tasks. Its privileged-context archive contains 16 removed files totaling
108,134,027 logical bytes; every archived file matches the size and SHA-256
digest in `manifest.json`. The two repaired fibroblast inputs are byte-identical
to their source-capsule copies.
