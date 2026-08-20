# Capsule Shortcut Patch Report

Date: 2026-05-27

This report covers the shortcut cleanup requested for the capsules listed in
`capsule_shortcut_patches.md`, using the corresponding source BBH task JSONL.

No JSONL rows were edited. Data patches were applied only where the remaining
capsule contents still support a reasonable analysis path.

## Patched Capsules

| UUID | Patch applied | Remaining satisfiable path |
|---|---|---|
| `52dad468-cc9e-4b61-9f6c-4e71faeaad64` | Removed the full `Data4_Selection analysis/PAML/` subtree from `Data_Caecilian.zip` and replaced the nested zip inside `downloaded_only_Data_Caecilian.zip`. | Alignments and trees remain. Requires `codeml`/PAML in the training environment. |
| `dccb5380-8ddc-43e7-9610-cdfefa9bc0c0` | Deleted `TCGA_GBM_IDH_WT_RAP2A_by_subtype.csv`, `downloaded_only_TCGA_GBM_IDH_WT_RAP2A_by_subtype.csv`, and `TCGA_GBM_RAP2A_expression_with_annotations.csv`. | Full TCGA expression, subtype, IDH, and clinical inputs remain. |
| `bf14e7d3-2afb-495b-9e16-b5961623cc04` | Removed precomputed `logFC`, `logCPM`, `PValue`, and `FDR` columns from the RPKM table and regenerated the gzipped copy. | Gene metadata and per-sample RPKM values remain; the task permits RPKM/equivalent expression input. |
| `eb5d2fbe-30ca-4cdd-8705-734248c66e92` | Dropped `Significance` from both workbook sheets. | Effect sizes, standard errors, p-values, ancestry sheets, and phecodes remain. |
| `f52b991d-3d1f-4780-a453-25ddbcc8215d` | Deleted `time-domain-error-30sec-in-cm.pkl` and `time-domain-error-xy.pkl`. | Raw continuous cursor/target and related decoder data remain. |
| `79d5a5bc-0469-4a85-87d1-fe5d255b9823` | Removed the manuscript analysis Rmd and residual, normalized residual, composite-residual, sensorimotor composite, predicted-AAO, and precomputed 6-month Ntg/chow baseline columns from both AD-BXD residual CSVs. | Raw phenotype, strain, sex, genotype, age, diet, and cognitive-measure columns remain, so the baseline aggregates and residual model can be rebuilt. |
| `a33a7c14-d960-4f5e-8d31-2fc41f9bd4f1` | Deleted all eight `downloaded_modified_filtered_SRR319160*_blast_results.txt` files. | FASTQ/FASTA reads and Aedes/reference FASTA files remain. Requires BLAST+ or equivalent species-assignment tooling. |
| `6be3b69f-5284-4f0f-889e-4339cab31746` | Deleted `downloaded_only_PCA_PLINKPRUNED.eigenval` and `downloaded_only_PCA_PLINKPRUNED.eigenvec`. | `.012` genotype matrix and popmaps remain. PCA/FST can be recomputed with PLINK or Python. |

## Blocked Until Data Or Task Is Fixed

These UUIDs should be masked from clean training until the listed issue is
resolved. In each case, removing the shortcut artifact would leave an expectation
in the question that cannot be satisfied from the current capsule contents.

### `96df8728-b367-45b8-b5e2-ed6e9ee0d56f`

- Hypothesis: CUT&Tag sequencing shows PCSK9 was the top SREBP1-bound target
  whose expression was impacted by SREBP1 knockdown.
- Data issue: the capsule contains `GSE292417_SREBP_1_vs_VECTOR_peak_compare_table.tsv.gz`,
  which already includes peak counts, DE statistics, q-values, and gene/TSS
  annotations. The RNA-seq files are also differential-expression result tables,
  not raw RNA-seq count matrices.
- Unsatisfiable expectation if cleaned: the task requires loading raw CUT&Tag
  counts and peak files, running DESeq2 or equivalent, applying BH correction,
  annotating peaks to hg38 genes, and overlapping with RNA-seq results across two
  cell lines. The raw CUT&Tag peak/count inputs needed for that workflow are not
  present.

### `d4ec7768-d0de-4a4f-aae9-a7c16aa7dfaf`

- Hypothesis: long-distance migratory birds show positive Spearman detection
  sensitivity while temperate sedentary birds retain high specificity under the
  climatic niche contrast model.
- Data issue: the capsule includes final step-6 Spearman outputs at top level and
  inside `outputs.zip`, including `6-df_Spearman_nsim100.txt`,
  `6-df_Spearman_prop.txt`, and multiple parameter-variant result tables.
- Unsatisfiable expectation if cleaned: the task expects simulation data or model
  outputs sufficient to compute 100 repeated Spearman tests and then sensitivity
  and specificity by population group. If the final step-6 outputs are removed,
  the raw WorldClim/Movebank/eBird inputs and full runnable intermediate dataset
  needed to regenerate them are not present in the capsule.

### `6bfe94ee-e184-44df-abe2-d00cce381143`

- Hypothesis: the `delta-T` and `delta-M` E. coli auxotroph pair has the highest
  mutual cooperative potential, near 0.44.
- Data issue: the only file in the capsule is `fitted_data(1).pickle`, which
  contains the fitted transformation vectors and biomass stoichiometry values
  used directly by the requested calculation.
- Unsatisfiable expectation if cleaned: the task asks the model to obtain or
  reproduce fitted transformation vectors and stoichiometry. Once the fitted
  pickle is removed, no raw co-culture growth/time-series measurements or fitting
  pipeline inputs remain.

### `42ae1fe4-e8fc-4328-973f-d67f2cd36906`

- Hypothesis: Aedes mosquito virome composition from the DRC-Angola border
  clusters weakly by collection site in PCA space.
- Data issue: the capsule contains only derived `downloaded_only_*.xlsx`
  workbooks: PCA matrices, relative abundance, heatmap/filtering labels, and
  BLAST/KrakenUniq status annotations.
- Unsatisfiable expectation if cleaned: the task requires raw KrakenUniq
  taxonomic reports for the eight mosquito pools, application of RPM and
  index-hopping filters, relative-abundance calculation, and PCA. The raw
  KrakenUniq report files are absent.

### `12eae66c-61bb-4b94-a537-300b1720f8de`

- Hypothesis: most parasitic flatworm `cki` homologs are in conserved
  TIMM21-`cki`-TUBB microsynteny.
- Data issue: the workbook tables already contain upstream/downstream gene IDs
  and descriptions such as `tim21` and `tubb`; those annotations are effectively
  the synteny answer.
- Unsatisfiable expectation if cleaned: the current question explicitly asks the
  model to load Table S1 with upstream/downstream gene annotations. If those
  answer columns are removed, the capsule has no GFF/GTF/genome annotation files
  from which to recompute neighboring genes. This needs either raw annotations
  plus a rewritten coordinate-lookup task, or masking.

### `92b42904-3685-4139-b034-b694ff8e48f6`

- Hypothesis: phoneme and prosody responses are better fit by sigmoid
  psychometric functions than linear functions, with no task difference.
- Data issue: `task_fit_phoneme.mat` and `task_fit_prosody.mat` contain the fit
  outputs. The remaining `task_phoneme.mat` and `task_prosody.mat` files are tiny
  summary MAT files, not the full 29-participant behavioral dataset described by
  the task.
- Unsatisfiable expectation if cleaned: the task requires loading behavioral
  response data for all 29 participants, fitting linear and sigmoid functions per
  participant/task, and running a 2x2 repeated-measures ANOVA. The full
  participant-level data are not present after removing the fit MAT files.

### `bddcbae2-add1-4891-9ea0-9bc87ced9b83`

- Hypothesis: mammalian lifetime somatic mutation burden has a significant
  positive log-log relationship with body mass.
- Data issue: the capsule contains only the manuscript PDF
  `2025.02.05.636604v3.full.pdf`.
- Unsatisfiable expectation if cleaned: the task requires species-level Cagan
  variables for mutation rate, lifespan, and body mass, then an OLS regression.
  Those numeric species-level inputs are not present in the capsule.

## Training Mask List

Mask these UUIDs until raw replacement data and/or a rewritten task is available:

```text
96df8728-b367-45b8-b5e2-ed6e9ee0d56f
d4ec7768-d0de-4a4f-aae9-a7c16aa7dfaf
6bfe94ee-e184-44df-abe2-d00cce381143
42ae1fe4-e8fc-4328-973f-d67f2cd36906
12eae66c-61bb-4b94-a537-300b1720f8de
92b42904-3685-4139-b034-b694ff8e48f6
bddcbae2-add1-4891-9ea0-9bc87ced9b83
```

## Operational Prerequisites For Patched Capsules

The following patched capsules are data-satisfiable but need tooling available
inside the training/eval runtime:

```text
52dad468-cc9e-4b61-9f6c-4e71faeaad64  # needs PAML/codeml
a33a7c14-d960-4f5e-8d31-2fc41f9bd4f1  # needs BLAST+ or equivalent
6be3b69f-5284-4f0f-889e-4339cab31746  # needs PLINK or Python PCA/FST implementation
```

If those tools are not installed and agents cannot install them during training,
mask these three as operationally unsatisfiable as well.
