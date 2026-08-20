# Capsule Shortcut Writeup


## Context

- **Audit run**: `aiapps/grpo-nanov3-hypotest-dfw-sft-rl/nano-sft1519-nowarmup-hypotest-t100-l80k-gbs256-paraphrased-hybridgate-gpt52rubric-async-1off-samplebyage` (80 steps × 16 rollouts = 1280 trajectories, full coverage).
- **Failure mode**: capsules contain a file the original-paper authors used to publish their analysis (PAML codeml output, pre-classified Bonferroni significance column, fitted-model pickle, BLAST result tables, PCA eigenvectors, the manuscript PDF itself). The agent reads that file and reports the contained values as if it had performed the analysis. The rubric scores it as a successful answer; the agent learns shortcut-finding.
- **Dataset**: the source BBH task JSONL. Each row has `id` = capsule UUID,
  `input_data_path` = `capsule_<uuid>`. The actual capsule contents are stored in
  whatever location the environment mounts as `input_data_path`. Patch the
  capsule storage, not the JSONL row (the JSONL only holds
  hypothesis/protocol/rubric/answer text).


---

## Tier A — apply patches (12 capsules, high confidence)

These were flagged as shortcuts by **12–16 of 16 rollouts** independently. The leak is real; patch with confidence.

### A1. `52dad468-cc9e-4b61-9f6c-4e71faeaad64` — PAML caecilian opsins
- **task_idx**: 1704 + 2332 (two paraphrased rows share this capsule — one patch fixes both)
- **Leaked files** (inside `Data_Caecilian.zip` and also extracted under `caecilian_data/`):
  - `Data_Caecilian/Data4_Selection analysis/PAML/RH1/M0/RH1_allamphibians_M0.txt`
  - `Data_Caecilian/Data4_Selection analysis/PAML/RH1/M0_constrained/allamphibians_out_M0_constrained.txt`
  - `Data_Caecilian/Data4_Selection analysis/PAML/LWS/COMPLETE GENE/M0/allamphibians_out_M0.txt`
  - `Data_Caecilian/Data4_Selection analysis/PAML/LWS/COMPLETE GENE/M0_constrained/allamphibians_out_M0_constrained.txt`
- **Leak**: PAML codeml output text containing `lnL`, `omega`, and parameter tables for both M0 (ω free) and M0_constrained (ω=1) — the exact statistics the objective asks the model to compute via codeml.

### A2. `dccb5380-8ddc-43e7-9610-cdfefa9bc0c0` — TCGA RAP2A IDH-WT GBM
- **task_idx**: 2042
- **Leaked files**:
  - `TCGA_GBM_IDH_WT_RAP2A_by_subtype.csv`
  - `downloaded_only_TCGA_GBM_IDH_WT_RAP2A_by_subtype.csv`
- **Leak**: 393 patients already filtered to IDH-WT, RAP2A expression column already extracted from the full expression matrix, GBM molecular subtype already joined.

### A3. `bf14e7d3-2afb-495b-9e16-b5961623cc04` — HSV-1 SUN2/lamin RPKM
- **task_idx**: 1513
- **Leaked file**: `downloaded_only_GSE243613_gene_rpkm_table.txt` (and `.gz` copy)
- **Leak**: RPKM values for all samples + pre-computed `log2FoldChange`, `logCPM`, `PValue`, `FDR` for both mock-vs-HSV1_4h and mock-vs-HSV1_8h contrasts.

### A4. `96df8728-b367-45b8-b5e2-ed6e9ee0d56f` — CUT&Tag PCSK9/SREBP1
- **task_idx**: 349 + 663 (two paraphrases share this capsule)
- **Leaked file**: `GSE292417_SREBP_1_vs_VECTOR_peak_compare_table.tsv.gz`
- **Leak**: pre-computed DESeq2 results (`log2foldchange`, `pvalue`, `qvalue`) **and** peak-to-gene annotations (`geneName`, `distanceToTSS`) for every peak.

### A5. `d4ec7768-d0de-4a4f-aae9-a7c16aa7dfaf` — Climatic Niche Contrast bird migration
- **task_idx**: 2008
- **Leaked files**:
  - `6-df_Spearman_nsim100.txt`
  - `6-df_Spearman_prop.txt`
- **Leak**: pre-computed Spearman ρ, p-values, significance flags, and bird migration classifications for all 100 niche-contrast simulations.

### A6. `eb5d2fbe-30ca-4cdd-8705-734248c66e92` — CHCHD3/OPA1 GREX Bonferroni
- **task_idx**: 2276
- **Leaked file**: `media-2 (1).xlsx` (sheets `MICOS_LIVER_EUR`, `MICOS_LIVER_AFR`)
- **Leak**: the `Significance` column already classifies each association as `within-tissue Bonferroni` / `nominal` / `not significant`.

### A7. `6bfe94ee-e184-44df-abe2-d00cce381143` — E. coli auxotroph cooperative potential
- **task_idx**: 745
- **Leaked file**: `fitted_data(1).pickle`
- **Leak**: a pickle that already contains the fitted 14×14 transformation matrix (element [5]) and biomass stoichiometry vectors (element [6]) — the outputs of a model fit the objective asks the agent to do.

### A8. `42ae1fe4-e8fc-4328-973f-d67f2cd36906` — Aedes mosquito virome PCA
- **task_idx**: 523
- **Leaked files** (all `downloaded_only_*.xlsx`):
  - `PCA_analysis_genus_filtering.xlsx`
  - `PCA_analysis_genus_unfiltering.xlsx`
  - `relative_abundance_genus_unfiltering.xlsx`
  - `genus_filtering_status_labels.xlsx`
  - `family_filtering_status_labels.xlsx`
  - `genus_heatmap_filtering.xlsx`
  - `BLAST_KrakenUniq_results_filtering.xlsx`
- **Leak**: pre-computed filtered relative abundance matrix (8 pools × 28 viral genera in PPM) with RPM filtering + index-hopping QC already applied; PCA-ready table.

### A9. `a33a7c14-d960-4f5e-8d31-2fc41f9bd4f1` — Mosquito species BLAST (Kimpese)
- **task_idx**: 839
- **Leaked files**: 8 files, one per SRR ID — e.g., `downloaded_modified_filtered_SRR31916075_blast_results.txt`, and 7 similar files for `SRR31916076` through `SRR31916082`.
- **Leak**: BLAST results with species IDs already assigned per read.

### A10. `12eae66c-61bb-4b94-a537-300b1720f8de` — Microsynteny TIMM21-cki-TUBB
- **task_idx**: 2305
- **Leaked file**: `downloaded_only_596073_file02.xlsx` (and `file04.xlsx` for free-living flatworms)
- **Leak**: in Table S1 (Parasite CKIs), the columns `5' Gene Description` and `3' Gene Description` already spell out the neighboring genes (`tim21`, `tubb`) for every cki homolog. Table S3 does the same for free-living flatworms.

### A11. `92b42904-3685-4139-b034-b694ff8e48f6` — Phoneme/prosody psychometric fits
- **task_idx**: 1124
- **Leaked files**:
  - `task_fit_phoneme.mat`
  - `task_fit_prosody.mat`
- **Leak**: a `fitinfo` MATLAB struct with adjusted-R², slope, intercept for both linear and sigmoid psychometric fits across all 29 subjects.

### A12. `f52b991d-3d1f-4780-a453-25ddbcc8215d` — Myoelectric interface tracking error
- **task_idx**: 114
- **Leaked file**: `time-domain-error-30sec-in-cm.pkl` (and `time-domain-error-xy.pkl`)
- **Leak**: pre-aggregated early/late 30-second tracking error segments (shape 2×14×8) — exactly the aggregation the comparison hypothesis asks for.

---

## Tier B — verify then patch (3 capsules, medium confidence)

These were flagged by 5–11 of 16 rollouts. The leak is likely real but worth a sanity check before deleting files.

### B1. `17954157-0f01-49da-b056-c15182c47c5a` — TrASPr/Pangolin/SpliceAI predictions
- **task_idx**: 498
- **Leaked file**: `downloaded_only_GTEx_data.tsv`
- **Leak**: TSV columns `TrASPr_pred`, `Pangolin_pred`, `SpliceAI_pred`, `SpliceTF_pred` alongside ground-truth `Labels`, intron coordinates, tissue annotations — i.e., the model predictions whose accuracy the hypothesis asks the agent to compare.

### B2. `6be3b69f-5284-4f0f-889e-4339cab31746` — Reed Warbler FST/PCA
- **task_idx**: 2126
- **Leaked files**:
  - `downloaded_only_PCA_PLINKPRUNED.eigenval`
  - `downloaded_only_PCA_PLINKPRUNED.eigenvec`
- **Leak**: PLINK PCA eigenvalues + eigenvectors already computed on the LD-pruned 13,223-SNP dataset.

### B3. `bddcbae2-add1-4891-9ea0-9bc87ced9b83` — Mammal mutation load (PDF leak)
- **task_idx**: 2025
- **Leaked file**: `2025.02.05.636604v3.full.pdf` — **the manuscript itself is in the capsule**.
- **Leak**: the PDF reports R² = 0.298, p = 0.035 for the exact log Q vs log(M×F) regression the hypothesis asks the agent to perform.

---

## Tier C — verify first, may be false positives (7 capsules)

Each of these was flagged by only 1–2 of 16 rollouts. Manually inspect the named file before deciding whether to patch.

| capsule UUID | task_idx | claimed leak | verification step |
|---|---|---|---|
| `08d9001f-42ba-49a2-bfb2-131a3898a97c` | 442 | `downloaded_modified_cfm_clean.csv` (BXD fear-memory pre-aggregated?) | Check whether this CSV contains raw trial-by-trial freezing or pre-computed per-strain means. |
| `f0807e09-0c08-4992-b7bc-b30a2cf30c46` | 829 | `Spatial_Sorting_Comerford_Soapberrybugs.csv` | Likely the legitimate input (raw insect counts per site). Confirm against objectives before changing. |
| `79d5a5bc-0469-4a85-87d1-fe5d255b9823` | 1383 | `GxE_all_phenotypes_plus_resid_indiv_20240826_filteredforGxEstrains.csv` | Check whether the "resid" column = residuals from a model the objective asks the agent to fit. |
| `bf7e9d6e-f499-4cc6-86d6-5c13e4375922` | 1575 | `Male_CR-R_vs_CR-NR.csv`, `Female_CR-R_vs_CR-NR.csv` | If these are raw metabolomic intensities per sample → legit. If they're already DE-table-style with computed p-values → leak. |
| `a4c16602-5520-41c4-8aa8-3e69ac62d806` | 1756 | `GBM.uncv2.mRNAseq_RSEM_normalized_log2.txt` | Likely legitimate raw expression matrix (RSEM normalized is standard). Confirm objective doesn't ask for normalization from raw counts. |
| `dd2658e0-2c50-4615-9966-02e1faf16dea` | 1821 | `downloaded_only_count_matrix.csv` | Likely legitimate raw count matrix. Confirm before removing. |
| `d9abecb2-c8df-4b12-a316-e7c58e2e770a` | 1938 | `Table_S1_MIRAGE_ASD_genes.xlsx` | Check whether this is just a gene list (legit) or contains pre-computed PPI counts (leak). |

For each Tier C capsule: load the file, inspect column headers, compare to objectives. If the file content matches what the objective asks the agent to *produce* → patch as Tier A. Otherwise leave alone.

---

## Quick reference — capsule UUIDs to patch

**Tier A (12 — patch with confidence):**
```
52dad468-cc9e-4b61-9f6c-4e71faeaad64
dccb5380-8ddc-43e7-9610-cdfefa9bc0c0
bf14e7d3-2afb-495b-9e16-b5961623cc04
96df8728-b367-45b8-b5e2-ed6e9ee0d56f
d4ec7768-d0de-4a4f-aae9-a7c16aa7dfaf
eb5d2fbe-30ca-4cdd-8705-734248c66e92
6bfe94ee-e184-44df-abe2-d00cce381143
42ae1fe4-e8fc-4328-973f-d67f2cd36906
a33a7c14-d960-4f5e-8d31-2fc41f9bd4f1
12eae66c-61bb-4b94-a537-300b1720f8de
92b42904-3685-4139-b034-b694ff8e48f6
f52b991d-3d1f-4780-a453-25ddbcc8215d
```

**Tier B (3 — verify then patch):**
```
17954157-0f01-49da-b056-c15182c47c5a
6be3b69f-5284-4f0f-889e-4339cab31746
bddcbae2-add1-4891-9ea0-9bc87ced9b83
```

**Tier C (7 — verify first, likely false positives):**
```
08d9001f-42ba-49a2-bfb2-131a3898a97c
f0807e09-0c08-4992-b7bc-b30a2cf30c46
79d5a5bc-0469-4a85-87d1-fe5d255b9823
bf7e9d6e-f499-4cc6-86d6-5c13e4375922
a4c16602-5520-41c4-8aa8-3e69ac62d806
dd2658e0-2c50-4615-9966-02e1faf16dea
d9abecb2-c8df-4b12-a316-e7c58e2e770a
```
