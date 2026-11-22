# Single Cell Morphotranscriptomics Project

Welcome to the **Single Cell Morphotranscriptomics Analysis Pipeline**. This repository contains modular, reproducible Python scripts, environment specifications, and a master execution pipeline designed to analyze single-cell transcriptomics integrated with imaging flow cytometry morphology data across megakaryocyte polyploidization states ($2\text{n}, 4\text{n}, 8\text{n}, 16\text{n}, 32\text{n}$).

This guide is specifically tailored for **biologists, scientific software engineers, and lab managers** to easily understand the computational workflows, quality control filters, dataset structures, and biological figure generation.

---

## 🧬 Biological Context & Overview

Megakaryocytes undergo **endomitosis**—repeated rounds of DNA replication without cell division—resulting in polyploid cells ranging from $2\text{n}$ up to $64\text{n}$. Understanding the relationship between genome duplication (ploidy), physical cell morphology (size, granularity, internal complexity), and gene expression requires **multi-omics integration**.

This pipeline combines two complementary data modalities collected at the single-cell level:
1. **Single-Cell Transcriptomics (RNA-seq)**: Quantifies gene expression levels for thousands of genes per cell.
2. **Imaging Flow Cytometry (Morphology)**: Measures physical parameters such as Forward Scatter (`FSC`), Side Scatter (`SSC`), cell area, perimeter, intensity, eccentricity, diffusivity, and Center of Mass (`CoM`) metrics.

---

## 📊 Dataset & Metadata Architecture

The analysis operates on a unified multi-animal cohort (`Metadata_merged_YL_SF_20.10.26_v2.csv`) combining single cells across the original "validation" (10-plate: Mice M1–M5) and "pilot" (4-plate: Mice M6–M7) studies. This unified design replaces arbitrary dataset splits with **Leave-One-Animal-Out Cross-Validation (LOAOCV)**.

### 1. Unified Multi-Animal Cohort (`Metadata_merged_YL_SF_20.10.26_v2.csv`)
- **Animals & Plates**: 7 biological replicates (Mice **M1 through M7**), each with 2 plates (e.g. `M1p1` = mouse 1 plate 1, `M1p2` = mouse 1 plate 2). Total raw capacity: 1,344 wells (192 per animal).
- **Quality Control (QC) Benchmark**: Exactly **393 high-quality single cells** pass full QC across all 7 mice (including the benchmark 291 cells on M1–M5):
  - `Contaminated == False`: Excludes wells exhibiting cross-species or microbial contamination.
  - `Supervised_ploidy`: Retains valid ploidy classes ($2\text{n}, 4\text{n}, 8\text{n}, 16\text{n}, 32\text{n}$) while excluding $64\text{n}$ (due to low cell counts) and unassigned artifacts.
  - `control == 0`: Excludes empty control wells.
  - `number_of_cell == 1`: Verifies single-cell sorting per well.
  - `kallisto.pc_map > 50.0%`: Requires $>50\%$ read alignment rate (refinable via `--min_map_pc`).
  - `kallisto.nb_mapped > 20,000`: Requires $>20,000$ mapped reads per cell (refinable via `--min_mapped_reads`).

### 2. QC-Passed Cell Yield by Animal
| Animal | Plate IDs | Total Wells | QC-Passed Cells | Ploidy Distribution Breakdown |
| :---: | :---: | :---: | :---: | :--- |
| **M1** | `M1p1`, `M1p2` | 192 | 39 | $8\text{n}$ (24), $16\text{n}$ (7), $4\text{n}$ (6), $2\text{n}$ (2) |
| **M2** | `M2p1`, `M2p2` | 192 | 41 | $8\text{n}$ (23), $16\text{n}$ (10), $4\text{n}$ (5), $2\text{n}$ (2), $32\text{n}$ (1) |
| **M3** | `M3p1`, `M3p2` | 192 | 74 | $8\text{n}$ (32), $16\text{n}$ (28), $4\text{n}$ (10), $32\text{n}$ (2), $2\text{n}$ (2) |
| **M4** | `M4p1`, `M4p2` | 192 | 78 | $8\text{n}$ (39), $16\text{n}$ (22), $4\text{n}$ (14), $32\text{n}$ (2), $2\text{n}$ (1) |
| **M5** | `M5p1`, `M5p2` | 192 | 59 | $8\text{n}$ (25), $4\text{n}$ (17), $16\text{n}$ (15), $32\text{n}$ (1), $2\text{n}$ (1) |
| **M6** | `M6p1`, `M6p2` | 192 | 63 | $16\text{n}$ (39), $8\text{n}$ (14), $32\text{n}$ (9), $4\text{n}$ (1) |
| **M7** | `M7p1`, `M7p2` | 192 | 39 | $16\text{n}$ (19), $8\text{n}$ (11), $4\text{n}$ (4), $32\text{n}$ (3), $2\text{n}$ (2) |
| **Total** | **14 Plates** | **1,344** | **393** | **$8\text{n}$ (168), $16\text{n}$ (140), $4\text{n}$ (57), $32\text{n}$ (18), $2\text{n}$ (10)** |

### 3. Leave-One-Animal-Out Cross-Validation (LOAOCV)
Rather than splitting by arbitrary experimental batches ("pilot" vs "validation"), the pipeline tests biological generalization across animals:
- In fold $k$, all cells from mouse $M_k$ are held out as an unseen test cohort.
- Models (ploidy classifiers, continuous regression, PCA projections) are fitted exclusively on the other 6 mice.
- Predictions and projections are evaluated strictly out-of-sample on mouse $M_k$.
- Repeated across all 7 mice to generate unbiased cross-animal generalization benchmarks.

### 4. Flow Cytometry Mappings (`.fcs` & `.cvw`)
Cell barcodes (e.g. `M1p1_A10`) are linked to their physical flow cytometry (`.fcs`) data files and FlowJo workspace (`.cvw`) files:
- Mouse 1: `sortM1p1.fcs` / `sortM1p1.cvw` & `sortM1p2.fcs` / `sortM1p2.cvw`
- Mouse 2: `sortM2p1.fcs` / `sortM2p1.cvw` & `M2p2.fcs` / `M2p2.cvw`
- Mice M3, M4, M5, M6, M7: Plates 1 & 2 respective `.fcs` and `.cvw` files.

---

## 💻 Environment Installation (`requirements.txt`)

All scripts run in Python 3 using the **`sc-analysis`** Conda environment.

### Step 1: Create and Activate Conda Environment
```powershell
# Create conda environment named sc-analysis
conda create -n sc-analysis python=3.10 -y

# Activate the environment
conda activate sc-analysis
```

### Step 2: Install Required Packages
```powershell
# Install all required dependencies
pip install -r requirements.txt
```

---

## 🛠️ Pipeline Scripts Explained (Step-by-Step)

Place input data files (metadata CSV, 10x expression folder `exp/` or `expression/`, and optional pre-saved `mofa_model_rna_morphology.hdf5`) in the gitignored `data/` directory at the repository root.

```
morphotranscriptomics/
├── data/                                    # Gitignored local input data (metadata CSV, exp/, optional HDF5)
├── figures/                                 # Gitignored generated publication figures & analysis outputs
├── MOFA_FACTOR_VARIANCE_RUNBOOK.md          # Runbook for MOFA factor variance (R^2) & heterogeneity sweeps
├── requirements.txt                         # Conda environment package dependencies
├── 01_qc_and_metadata_filtering.py          # Step 1  (qc)           : QC filtering & Multi-Animal metadata harmonization
├── 02_gene_expression_rankings.py           # Step 2  (rankings)     : Vectorized gene rankings & FCS/CVW mapping
├── 03_dimensionality_reduction.py           # Step 3  (umap/dimred)  : Scanpy HVG selection, PCA, & UMAP
├── 04_morphology_gene_correlations.py       # Step 4  (correlations) : Spearman morphology-to-gene correlations & scatter grids
├── 05_mofa_multimodal_analysis.py           # Step 5  (mofa)         : Multi-Omics Factor Analysis (MOFA2)
├── 05b_mvae_multimodal_analysis.py          # Step 5b (mvae)         : PyTorch Multimodal VAE (MVAE)
├── 06_pathway_enrichment.py                 # Step 6  (enrichment)   : Pathway enrichment (gseapy) & WordCloud
├── leave_one_animal_out_cv.py               # Step CV (cv/loaocv)    : Leave-One-Animal-Out Cross-Validation (LOAOCV)
├── cell_cycle.py                            # Shared cited cell-cycle & custom gene exclusion utilities
├── hvg_utils.py                             # Shared gene prevalence (--min_cells_per_gene) & HVG selection utilities
├── preprocessing.py                         # Unified Depth Norm (--depth_norm) -> CDR (--adjust_library_depth) -> CC Regression (--regress_cell_cycle)
└── run_master_analysis.py                   # Master CLI Pipeline Runner
```

---

### Step 1: Quality Control & Metadata Harmonization
**Script**: `01_qc_and_metadata_filtering.py` (`--step 1` or `--step qc`)

- **What it does for the biologist**: Ingests raw sequencing statistics and metadata from `Metadata_merged_YL_SF_20.10.26_v2.csv`. It parses animal IDs (`mouse`), standardizes plate identifiers (`M1p1` = mouse 1 plate 1, etc.), filters out dead/contaminated/empty wells, and verifies the **393 QC-passed cell benchmark** across all 7 animals (including the 291 cell benchmark on M1–M5).
- **Outputs**:
  - `output/qc_filtered_metadata.csv`: Cleaned metadata table with `mouse` and `fold` columns.
  - `output/qc_filtering_summary_per_animal.csv`: Cell counts per animal and ploidy state.
  - `output/01_ploidy_distribution_qc.png`: Bar chart of ploidy counts before vs. after QC.
  - `output/01_ploidy_by_animal_qc.png`: Distribution of ploidy states across all 7 animals.
  - `output/01_kallisto_qc_metrics_violin.png`: Violin plots of read counts and mapping percentages across $2\text{n}$–$32\text{n}$ ploidies.

---

### Step CV: Leave-One-Animal-Out Cross-Validation (LOAOCV)
**Script**: `leave_one_animal_out_cv.py` (`--step cv` or `--step loaocv`)

- **What it does for the biologist**: Replaces the previous 2-dataset split with a rigorous cross-validation loop iterating over all 7 animals ($M1, \dots, M7$). In each fold, one animal is held out while models (Random Forest classifier, Balanced Logistic Regression, Continuous Ridge regression, and PCA projections) are fitted exclusively on the other 6 animals. Evaluates true out-of-sample generalization for predicting genome ploidy ($2\text{n}$ to $32\text{n}$) from imaging morphology.
- **Outputs**:
  - `output/loaocv_summary_metrics.csv`: Per-animal accuracy, balanced accuracy, macro F1, and Spearman correlation.
  - `output/loaocv_cell_predictions.csv`: Out-of-fold predictions for every individual cell.
  - `output/loaocv_metrics_barplot.png`: Bar chart of cross-animal classification performance.
  - `output/loaocv_confusion_matrix.png`: Normalized pooled out-of-fold confusion matrix.
  - `output/loaocv_continuous_ploidy_scatter.png`: Predicted vs True log2 ploidy correlation plot.
  - `output/loaocv_pca_projection.png`: PCA 2D projections of reference vs holdout cells colored by ploidy.

---

### Step 2: Marker Gene Expression Rankings & Cell Mapping
**Script**: `02_gene_expression_rankings.py` (`--step 2` or `--step rankings`)

- **What it does for the biologist**: Filters out non-informative ribosomal genes (`Rps`, `Rpl`) and unannotated genes (`Unnamed`), while retaining mitochondrial (`mt-`/`MT-`) genes by default (pass `--exclude_mt` to filter them out). Finds the top-N highest and lowest expressing single cells for key megakaryocyte marker genes (e.g. `kalrn`, `pf4`, `itga2b`) using fast vectorized lookups. Automatically maps each cell to its mouse ID, plate, well, `.fcs` file, and `.cvw` file.
- **Outputs**:
  - `output/kalrn_top_bottom_cells.csv`: Table containing cell barcodes, expression values, mouse ID, plate, well, and flow file paths.
  - `output/02_gene_rankings_kalrn.png`: Ranking bar plot.

---

### Step 3: Dimensionality Reduction (PCA & UMAP)
**Script**: `03_dimensionality_reduction.py` (`--step 3` or `--step umap`)

- **What it does for the biologist**: Operates on the full unified **393 QC-passed cell cohort (`M1`–`M7`)**, selects Highly Variable Genes (HVGs, default top 2,000; refinable via `--n_hvg`, `--hvg_method`, and `--min_cells_per_gene`), optionally applies library-depth / Cellular Detection Rate (CDR) residualization (`--adjust_library_depth`), scales expression data, performs Principal Component Analysis (PCA), and projects cells onto a 2D UMAP manifold.
- **Optional Gene Filtering**: Supports `--exclude_mt` (opt-in mitochondrial `mt-`/`MT-` gene filtering; **off by default**), `--exclude_cell_cycle` (drops the cited 97-gene Tirosh et al. 2016 / Seurat `cc.genes.updated.2019` set + legacy aliases), `--exclude_genes_file` (drops any user-supplied `.txt`/`.csv`/`.tsv` list, e.g. Mouse `GO:0007049`), and `--exclude_genes` (comma-separated symbols). Every excluded cell-cycle/custom gene and its citation is recorded in `output/03_excluded_genes_audit.csv`.
- **Outputs**:
  - `output/03_umap_supervised_ploidy.png`: UMAP plot where cells are colored by genome ploidy ($2\text{n}, 4\text{n}, 8\text{n}, 16\text{n}, 32\text{n}$).
  - `output/03_umap_dataset_cohort.png`: UMAP plot comparing Train vs. Pilot cohorts.
  - `output/03_umap_gene_kalrn.png`: Expression gradient overlay on the UMAP landscape.
  - `output/03_excluded_genes_audit.csv`: Audit log of excluded genes and provenance (created when filtering is enabled).

---

### Step 4: Morphology-to-Gene Spearman Rank Correlations
**Script**: `04_morphology_gene_correlations.py` (`--step 4` or `--step correlations`)

- **What it does for the biologist**: Computes robust pairwise **Spearman rank correlations ($\rho$)**, Benjamini–Hochberg FDR ($q$-values), and partial Spearman correlations controlling for $\log_2(\text{Ploidy})$ between all expressed genes and imaging flow cytometry morphology parameters across all 393 QC-passed cells (M1–M7), while remaining backwards-compatible with legacy single-cohort datasets (`--legacy_split` or `--cohort 10plate` / `--cohort 4plate`).
- **Optional Gene Filtering**: Supports `--exclude_cell_cycle`, `--exclude_genes_file`, and `--exclude_genes` prior to library-depth adjustment and Spearman correlation testing, writing `output/04_excluded_genes_audit.csv` when filtering is enabled.
- **Outputs**:
  - `output/04_top_gene_morphology_spearman_scatter_grid.png`: 12-panel scatter & regression grid of top positively/negatively correlated gene–morphology pairs colored by `Supervised_ploidy`.
  - `output/04_gene_morphology_spearman_ranking_labeled.png`: Two-panel volcano & ranked bar chart labeling genes with particularly high Spearman $|\rho|$ and key megakaryocyte markers (`Ppbp`, `Pde3a`, `Mylk`, `Kalrn`, `Rock1`, `Exoc6`, `Pf4`, `Itga2b`).
  - `output/04_morphology_gene_spearman_heatmap.png`: Hierarchically clustered Spearman correlation matrix across top genes and morphology features.
  - `output/morphology_gene_spearman_top_pairs.csv` & `output/morphology_gene_spearman_per_gene_summary.csv`: Full tabular correlation statistics.
  - `output/04_excluded_genes_audit.csv`: Audit log of excluded genes and provenance (created when filtering is enabled).

---

### Step 5: Multi-Omics Factor Analysis (MOFA)
**Script**: `05_mofa_multimodal_analysis.py` (`--step 5` or `--step mofa`)

- **What it does for the biologist**: Integrates single-cell RNA expression and cell morphology using linear factor modeling (`mofapy2`). It calculates how much variance ($R^2$) is explained by each latent factor in gene expression vs. cell shape, and identifies top driving gene and morphology features.
- **Recommended Sensitivity Check (Multiple Factor Numbers)**: Because MOFA2 factor decomposition depends on the specified latent dimensionality $K$ (and ARD pruning), we recommend running MOFA across multiple factor counts—for example `K = 5, 10, 15` (`--mofa_factors 5,10,15` in `run_master_analysis.py` or `--factors 5,10,15` in `05_mofa_multimodal_analysis.py`)—to verify that total variance explained ($R^2$) and per-cell reconstruction error heterogeneity are stable across model capacities. See **[MOFA_FACTOR_VARIANCE_RUNBOOK.md](MOFA_FACTOR_VARIANCE_RUNBOOK.md)** for the full step-by-step runbook and diagnostic checklist.
- **Outputs**:
  - `output/mofa_model_rna_morphology.hdf5` (or `*_k5.hdf5`, `*_k10.hdf5`, `*_k15.hdf5` when sweeping): Trained MOFA model(s).
  - `output/05_mofa_r2_variance_explained.png` (or `*_k{K}.png`): Heatmap of $R^2$ variance explained per view.
  - `output/05_mofa_factor1_vs_factor2.png` (or `*_k{K}.png`): Factor score scatter plot separating ploidy states.
  - `output/05_mofa_reconstruction_error_heterogeneity.png` (or `*_k{K}.png`): Per-cell likelihood-aware MOFA reconstruction error (RMSE) stratified by ploidy and animal (`M1`–`M7`).
  - `output/05_mofa_reconstruction_error_per_cell.csv` (or `*_k{K}.csv`): Cell-level MOFA reconstruction error table.
  - `output/05_mofa_factor_sweep_summary.csv` & `output/05_mofa_factor_sweep_error_and_r2.png` (when multiple factor counts are passed): Total $R^2$ and RMSE comparison across tested $K$ values.

---

### Step 5b: PyTorch Multimodal VAE (MVAE)
**Script**: `05b_mvae_multimodal_analysis.py` (`--step 5b` or `--step mvae`)

- **What it does for the biologist**: Uses a deep Multimodal Variational Autoencoder with a **Product-of-Experts (PoE)** encoder to learn a joint non-linear latent space $z$. Evaluates how accurately the neural network reconstructs original RNA expression and imaging morphology parameters.
- **Outputs**:
  - `output/mvae_model_weights.pt`: Saved PyTorch model weights.
  - `output/05b_mvae_training_loss_curve.png`: Convergence loss history (ELBO, Reconstruction, KL divergence).
  - `output/05b_mvae_rna_reconstruction_scatter.png`: Original vs. reconstructed RNA expression scatter plot & $R^2$ score.
  - `output/05b_mvae_morphology_reconstruction_scatter.png`: Original vs. reconstructed morphology parameters scatter plot & $R^2$ score.
  - `output/05b_mvae_latent_space_umap.png`: UMAP of MVAE latent space $z$ colored by ploidy.

---

### Step 6: Biological Pathway Enrichment & WordCloud
**Script**: `06_pathway_enrichment.py` (`--step 6` or `--step enrichment`)

- **What it does for the biologist**: Runs Over-Representation Analysis (ORA / Enrichr) using `gseapy` against MSigDB Hallmark, KEGG, and GO Biological Process databases to highlight enriched biological pathways (e.g., platelet activation, endomitosis, cell cycle, cytoskeletal remodeling).
- **Outputs**:
  - `output/06_pathway_enrichment_dotplot.png`: Pathway enrichment dotplot.
  - `output/06_pathway_wordcloud.png`: Biological pathway WordCloud visualization.

---

## 🔬 Normalization Order, Cell-Cycle Regression & Gene Filtering (Citations)

All RNA-consuming pipeline stages (**Steps 2, 3, 4, 5, and 5b**) share a unified preprocessing pipeline (`preprocessing.py`) executed in the canonical order:
1. **Depth Normalization (`--depth_norm`)**: Defaults to `cp10k_log1p` (library-size normalization to 10,000 counts per cell + $\log(1+x)$). Pass `--depth_norm none` to reproduce legacy raw-count baselines.
2. **Optional Library-Depth / Cellular Detection Rate (CDR) Residualization (`--adjust_library_depth`)**: Regresses out per-cell mean standardized expression / gene-detection rate (Finak et al., 2015) and re-standardizes within each cohort. Always applied by default in Step 4 (unless `--raw_ranks` is passed) and opt-in for Steps 3, 5, and 5b via `--adjust_library_depth`.
3. **Cell-Cycle Scoring & Linear Regression (`--regress_cell_cycle`)**: Computed *after* depth normalization (and *before* optional gene exclusion) using the cited Tirosh et al. (2016) / Seurat `cc.genes.updated.2019` S and G2/M gene lists:
   - `none` *(default)*: No cell-cycle regression applied.
   - `full`: Regresses out both `S_score` and `G2M_score` via OLS (`*_cell_cycle_scores.csv` audit log saved).
   - `difference`: Regresses out `CC_difference = S_score - G2M_score` via OLS (removes S vs. G2/M phase variation while preserving cycling vs. non-cycling / endomitotic state separation).
   - *Step 5 (MOFA2)*: Because MOFA2's Poisson/Bernoulli likelihoods require non-negative integer / binary counts, requesting `--adjust_library_depth` or `--regress_cell_cycle` automatically switches MOFA2's RNA likelihood to `gaussian` with a clear log warning.
4. **Optional Gene Exclusion (`--exclude_mt`, `--exclude_cell_cycle`, `--exclude_genes_file`, `--exclude_genes`, `--exclude_steps`)**: Independent of regression:
   - `--exclude_mt` *(default: off)*: Retains mitochondrial (`mt-`/`MT-`) genes by default across Steps 2–5b; pass `--exclude_mt` to exclude them for sensitivity comparisons.
   - `--exclude_cell_cycle`, `--exclude_genes_file`, `--exclude_genes`: Removes the cited 97 cell-cycle genes (+ 6 legacy aliases) or custom gene lists via `cell_cycle.py`, logging every excluded gene to `*_excluded_genes_audit.csv`.

### Cited Seurat Cell-Cycle Gene Lists
- **Tirosh et al. (2016) / Seurat `cc.genes`**: Canonical human cell-cycle gene signature comprising **43 S-phase** and **54 G2/M-phase** markers (97 genes total) derived from single-cell RNA-seq of proliferating cells (Tirosh et al., 2016, *Science* 352(6282):189–196, DOI: [10.1126/science.aad0501](https://doi.org/10.1126/science.aad0501)).
- **Seurat `cc.genes.updated.2019`**: Updated HGNC gene symbols introduced in Seurat v3 (Butler et al., 2018, *Nature Biotechnology* 36:411–420, DOI: [10.1038/nbt.4096](https://doi.org/10.1038/nbt.4096); Stuart et al., 2019, *Cell* 177(7):1888–1902, DOI: [10.1016/j.cell.2019.05.031](https://doi.org/10.1016/j.cell.2019.05.031)). In `cell_cycle.py`, all 97 canonical mouse ortholog symbols (`TIROSH_2016_S_GENES_MOUSE`, `TIROSH_2016_G2M_GENES_MOUSE`) plus **6 legacy GRCm38/Tirosh aliases** (`Mlf1ip`→`Cenpu`, `Rpa2`→`Polr1b`, `Brip1`→`Mrpl36`, `Mcm2`→`Mcm7`, `Fam64a`→`Pimreg`, `Hn1`→`Jpt1`) are matched case-insensitively to prevent silent misses across older and newer genome annotations.

**Note on endomitotic megakaryocytes**: Megakaryocytes undergo **endomitosis** (aborted mitosis without cytokinesis to reach $4\text{n}$–$32\text{n}+$ ploidy). Because full `S + G2M` regression can attenuate genuine polyploidization biology (Luecken & Theis, 2019, *Molecular Systems Biology* 15(6):e8746, DOI: [10.15252/msb.20188746](https://doi.org/10.15252/msb.20188746)), cell-cycle regression defaults to `none` and both `difference` (`S - G2M`) and `full` (`S + G2M`) modes — as well as explicit gene-list exclusion (`--exclude_cell_cycle`) — are provided for transparent sensitivity comparisons.

### Library-Depth / Cellular Detection Rate (CDR) Adjustment
In megakaryocyte index-sorting, larger polyploid cells (`16n`, `32n`) exhibit lower total mapped read counts / higher dropout ($\text{Spearman}(\text{lib\_size}, \text{Size}) \approx -0.54$). Because global RNA complexity may reflect either genuine maturation-associated transcriptional specialization or technical capture/shearing bias, the pipeline supports comparing **unadjusted** expression alongside **per-cohort library-depth / Cellular Detection Rate (CDR) residualization** (regressing per-cell mean standardized expression out of each gene and re-standardizing, following the CDR covariate adjustment framework of **Finak et al., 2015**, *Genome Biology* 16:278, DOI: [10.1186/s13059-015-0844-5](https://doi.org/10.1186/s13059-015-0844-5)).

---

## 🔄 Backwards Compatibility with Legacy Datasets

Both Step 1 (`01_qc_and_metadata_filtering.py`) and Step 4 (`04_morphology_gene_correlations.py`) retain full backwards compatibility with legacy Train/Pilot 2-dataset splits and older metadata schemas:
- **Legacy Split Mode**: Pass `--legacy_split` to `run_master_analysis.py` or `01_qc_and_metadata_filtering.py` to ingest legacy Train (`Metadata_flow+well_id+supervised_ploidy.csv`, `meta.contaminant.2.tsv` / `meta.contaminant.v1.csv`) and Pilot (`meta.v1+flowSF.csv`) directories instead of the unified `data/Metadata_merged_YL_SF_20.10.26_v2.csv`.
- **Legacy Column & Ploidy Normalization**: Automatically harmonizes older column conventions (`Well_ID`/`Well`/`Barcode`, `Mouse_Plate`/`Sample_Plate`/`orig.ident`, `Supervised_Ploidy_Images`/`Estimated_Ploidy_D.NA.A`, `kallisto_mapped`/`kallisto_mappe_pc`) and ploidy casing (`2N`–`32N` → `2n`–`32n`).
- **Cohort Selection in Step 4**: Use `--cohort 10plate` (M1–M5), `--cohort 4plate` (M6–M7), or `--metadata_file <legacy_csv_or_tsv>` directly.

---

## 🚀 How to Run the Master Pipeline

You can run the entire pipeline end-to-end or individual steps using **`run_master_analysis.py`**:

```powershell
# Activate environment
conda activate sc-analysis

# Run FULL pipeline end-to-end using default local data/ directory (CP10K+log1p depth norm by default):
python run_master_analysis.py --out_dir "output" --gene kalrn

# Run with Depth Normalization -> CDR Residualization -> Cell-Cycle Regression across RNA analyses:
#   --depth_norm: cp10k_log1p (default) | none
#   --adjust_library_depth: apply CDR residualization in Steps 3, 5, 5b (Step 4 applies CDR by default)
#   --regress_cell_cycle: none (default) | full (S + G2M) | difference (S - G2M)
python run_master_analysis.py --depth_norm cp10k_log1p --adjust_library_depth --regress_cell_cycle difference --out_dir "output_cdr_cc_diff"
python run_master_analysis.py --step correlations --depth_norm cp10k_log1p --regress_cell_cycle full --out_dir "output_cc_full"

# Refine read-count QC, gene prevalence, and HVG selection criteria end-to-end:
#   --min_mapped_reads (default: 20000), --min_map_pc (default: 50.0),
#   --min_cells_per_gene (default: 0), --n_hvg (default: 2000),
#   --hvg_method: default | raw_var | log1p_var | seurat | seurat_v3
python run_master_analysis.py --min_mapped_reads 50000 --min_map_pc 60.0 --min_cells_per_gene 5 --n_hvg 3000 --hvg_method seurat --out_dir "output_refined"

# Generate ALL pipeline figures under recommended depth + CDR controls (mitochondrial genes retained by default):
python run_master_analysis.py --out_dir "figures" --gene kalrn --depth_norm cp10k_log1p --adjust_library_depth --regress_cell_cycle none --mofa_factors "5,10,15" --loaocv
python run_master_analysis.py --step mvae --out_dir "figures" --depth_norm cp10k_log1p --adjust_library_depth --regress_cell_cycle none

# Optional: exclude mitochondrial (mt-/MT-) genes across Steps 2-5b (off by default):
python run_master_analysis.py --exclude_mt --out_dir "output_no_mt"

# Run individual steps by number or descriptive alias:
#   1/qc, 2/rankings, 3/umap, 4/correlations, 5/mofa, 5b/mvae, 6/enrichment, cv/loaocv
python run_master_analysis.py --step qc
python run_master_analysis.py --step correlations --exclude_cell_cycle --out_dir "output_no_cc"
python run_master_analysis.py --step mofa
# Recommended: compare MOFA across multiple factor numbers (quote "5,10,15" in PowerShell):
python run_master_analysis.py --step mofa --mofa_factors "5,10,15" --out_dir "figures"
python run_master_analysis.py --step mvae
python run_master_analysis.py --step cv

# Exclude cited cell cycle genes (Tirosh et al. 2016 / Seurat cc.genes.updated.2019) across ALL analyses:
python run_master_analysis.py --exclude_cell_cycle --out_dir "output_no_cc"

# Exclude a custom gene list from a file (e.g., Mouse GO:0007049 Cell Cycle .txt/.csv/.tsv) across ALL analyses:
python run_master_analysis.py --exclude_genes_file "my_genes_to_exclude.txt" --out_dir "output_custom_filter"

# Apply gene filtering to specific steps (numbers or aliases):
python run_master_analysis.py --exclude_cell_cycle --exclude_steps umap,correlations

# Run backwards-compatible legacy Train/Pilot 2-dataset split:
python run_master_analysis.py --legacy_split --out_dir "output_legacy"
```

---

## 📈 Summary of Generated Output Files & Figures (`figures/`)

All publication figures and tabular summaries can be regenerated into the gitignored `figures/` directory using the commands above.

| File / Figure Name | Description | Step |
| :--- | :--- | :---: |
| `qc_filtered_metadata.csv` | Cleaned table of all 393 QC-passed single cells (291 on M1–M5) | Step 1 (`qc`) |
| `01_ploidy_distribution_qc.png` | Bar chart comparing supervised ploidy cell counts ($2\text{n}$–$32\text{n}$) before vs. after QC | Step 1 (`qc`) |
| `01_ploidy_by_animal_qc.png` | Stacked/grouped ploidy breakdown across all 7 biological replicates (`M1`–`M7`, LOAOCV folds) | Step 1 (`qc`) |
| `01_kallisto_qc_metrics_violin.png` | Violin plots of mapped read depth (`kallisto.nb_mapped`) and mapping rate (`kallisto.pc_map`) by ploidy | Step 1 (`qc`) |
| `kalrn_top_bottom_cells.csv` | Top/bottom `Kalrn`-expressing cell mapping to FCS, CVW, Mouse, Plate, and Well | Step 2 (`rankings`) |
| `02_gene_rankings_kalrn.png` | Ranked expression bar chart of highest and lowest `Kalrn`-expressing single cells | Step 2 (`rankings`) |
| `03_umap_supervised_ploidy.png` | Transcriptomic UMAP landscape colored by genome ploidy ($2\text{n}$–$32\text{n}$) | Step 3 (`umap`) |
| `03_umap_dataset_cohort.png` | Transcriptomic UMAP landscape colored by experimental cohort / animal batch | Step 3 (`umap`) |
| `03_umap_gene_kalrn.png` | `Kalrn` expression gradient overlaid on the UMAP manifold | Step 3 (`umap`) |
| `03_umap_gene_pf4.png` | `Pf4` (megakaryocyte marker) expression gradient overlaid on the UMAP manifold | Step 3 (`umap`) |
| `04_top_gene_morphology_spearman_scatter_grid.png` | 12-panel scatter & regression grid of top positively/negatively correlated gene–morphology Spearman pairs colored by ploidy | Step 4 (`correlations`) |
| `04_gene_morphology_spearman_ranking_labeled.png` | Two-panel volcano & ranked bar chart labeling top Spearman $|\rho|$ genes and key megakaryocyte markers (`Ppbp`, `Pde3a`, `Mylk`, `Kalrn`, `Rock1`, `Exoc6`, `Pf4`, `Itga2b`) | Step 4 (`correlations`) |
| `04_morphology_gene_spearman_heatmap.png` | Hierarchically clustered Spearman correlation heatmap across top correlated genes and imaging morphology features | Step 4 (`correlations`) |
| `morphology_gene_spearman_top_pairs.csv` | Top 500 pairwise Spearman $\rho$, BH-FDR $q$, and partial $\rho\mid\log_2(\text{ploidy})$ | Step 4 (`correlations`) |
| `morphology_gene_spearman_per_gene_summary.csv` | Per-gene strongest morphology correlation summary across all 393 cells | Step 4 (`correlations`) |
| `05_mofa_r2_variance_explained_k{5,10,15}.png` | Factor-by-view MOFA2 $R^2$ (%) variance decomposition heatmaps for $K = 5, 10, 15$ | Step 5 (`mofa`) |
| `05_mofa_factor1_vs_factor2_k{5,10,15}.png` | MOFA2 Factor 1 vs. Factor 2 latent score scatter plots colored by ploidy for $K = 5, 10, 15$ | Step 5 (`mofa`) |
| `05_mofa_reconstruction_error_heterogeneity_k{5,10,15}.png` | Per-cell likelihood-aware MOFA2 reconstruction error (RMSE) heterogeneity stratified by ploidy and animal (`M1`–`M7`) for $K = 5, 10, 15$ | Step 5 (`mofa`) |
| `05_mofa_factor_sweep_error_and_r2.png` | Multi-$K$ MOFA2 sweep comparing Total $R^2$ (%) and per-cell RMSE distributions across $K = 5, 10, 15$ | Step 5 (`mofa`) |
| `05_mofa_factor_sweep_summary.csv` | Tabular summary of Total $R^2$ and Mean/Median/IQR RMSE per view across $K = 5, 10, 15$ | Step 5 (`mofa`) |
| `05b_mvae_training_loss_curve.png` | PyTorch Multimodal VAE (PoE) convergence history (Total ELBO, RNA, Morphology, and KL loss) | Step 5b (`mvae`) |
| `05b_mvae_rna_reconstruction_scatter.png` | MVAE observed vs. reconstructed RNA expression scatter plot & $R^2$ score | Step 5b (`mvae`) |
| `05b_mvae_morphology_reconstruction_scatter.png` | MVAE observed vs. reconstructed imaging morphology scatter plot & $R^2$ score | Step 5b (`mvae`) |
| `05b_mvae_latent_space_umap.png` | UMAP projection of the MVAE Product-of-Experts joint latent space $z$ colored by ploidy | Step 5b (`mvae`) |
| `06_pathway_enrichment_dotplot.png` | Enrichr biological pathway over-representation dotplot (`-log10(P-val)`, Combined Score, Adjusted P-value) | Step 6 (`enrichment`) |
| `06_pathway_wordcloud.png` | Biological pathway term frequency WordCloud | Step 6 (`enrichment`) |
| `loaocv_metrics_barplot.png` | Per-animal Leave-One-Animal-Out CV accuracy, balanced accuracy, and macro F1 barplot (`M1`–`M7`) | Step CV (`cv`) |
| `loaocv_confusion_matrix.png` | Normalized pooled out-of-fold ploidy confusion matrix across all 7 holdout animals | Step CV (`cv`) |
| `loaocv_continuous_ploidy_scatter.png` | Out-of-fold continuous $\log_2(\text{ploidy})$ Ridge regression predictions vs. true ploidy colored by animal | Step CV (`cv`) |
| `loaocv_pca_projection.png` | Morphology PCA manifold colored by ploidy (left) and biological replicate animal `M1`–`M7` (right) | Step CV (`cv`) |
| `*_excluded_genes_audit.csv` | Audit log of excluded cell-cycle / custom genes & citations (when gene exclusion is enabled) | Steps 2–5b |

---
