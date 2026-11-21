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

```
Exploration_Data/
├── requirements.txt                         # Conda environment package dependencies
├── 01_qc_and_metadata_filtering.py          # Step 1: QC filtering & Multi-Animal metadata harmonization
├── leave_one_animal_out_cv.py               # Step CV: Leave-One-Animal-Out Cross-Validation (LOAOCV)
├── 02_gene_expression_rankings.py          # Step 2: Vectorized gene rankings & FCS/CVW mapping
├── 03_dimensionality_reduction.py          # Step 3: Scanpy HVG selection, PCA, & UMAP
├── 04_mofa_multimodal_analysis.py          # Step 4: Multi-Omics Factor Analysis (MOFA)
├── 04b_mvae_multimodal_analysis.py         # Step 4b: PyTorch Multimodal VAE (MVAE)
├── 05_pathway_enrichment.py                # Step 5: Pathway enrichment (gseapy) & WordCloud
├── 06_morphology_gene_correlations.py      # Step 6: Spearman morphology-to-gene correlations & scatter grids
├── cell_cycle.py                           # Shared cited cell-cycle & custom gene exclusion utilities
├── hvg_utils.py                            # Shared gene prevalence (--min_cells_per_gene) & HVG selection utilities
└── run_master_analysis.py                  # Master CLI Pipeline Runner
```

---

### Step 1: Quality Control & Metadata Harmonization
**Script**: [`01_qc_and_metadata_filtering.py`](file:///c:/Users/vilsn/Dropbox/Work/Turing/Exploration_Data/01_qc_and_metadata_filtering.py)

- **What it does for the biologist**: Ingests raw sequencing statistics and metadata from `Metadata_merged_YL_SF_20.10.26_v2.csv`. It parses animal IDs (`mouse`), standardizes plate identifiers (`M1p1` = mouse 1 plate 1, etc.), filters out dead/contaminated/empty wells, and verifies the **393 QC-passed cell benchmark** across all 7 animals (including the 291 cell benchmark on M1–M5).
- **Outputs**:
  - `output/qc_filtered_metadata.csv`: Cleaned metadata table with `mouse` and `fold` columns.
  - `output/qc_filtering_summary_per_animal.csv`: Cell counts per animal and ploidy state.
  - `output/01_ploidy_distribution_qc.png`: Bar chart of ploidy counts before vs. after QC.
  - `output/01_ploidy_by_animal_qc.png`: Distribution of ploidy states across all 7 animals.
  - `output/01_kallisto_qc_metrics_violin.png`: Violin plots of read counts and mapping percentages across $2\text{n}$–$32\text{n}$ ploidies.

---

### Step CV: Leave-One-Animal-Out Cross-Validation (LOAOCV)
**Script**: [`leave_one_animal_out_cv.py`](file:///c:/Users/vilsn/Documents/morphotranscriptomics/leave_one_animal_out_cv.py)

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
**Script**: [`02_gene_expression_rankings.py`](file:///c:/Users/vilsn/Dropbox/Work/Turing/Exploration_Data/02_gene_expression_rankings.py)

- **What it does for the biologist**: Filters out non-informative ribosomal genes (`Rps`, `Rpl`) and unannotated genes (`Unnamed`). Finds the top-N highest and lowest expressing single cells for key megakaryocyte marker genes (e.g. `kalrn`, `pf4`, `itga2b`) using fast vectorized lookups. Automatically maps each cell to its mouse ID, plate, well, `.fcs` file, and `.cvw` file.
- **Outputs**:
  - `output/kalrn_top_bottom_cells.csv`: Table containing cell barcodes, expression values, mouse ID, plate, well, and flow file paths.
  - `output/02_gene_rankings_kalrn.png`: Ranking bar plot.

---

### Step 3: Dimensionality Reduction (PCA & UMAP)
**Script**: [`03_dimensionality_reduction.py`](file:///c:/Users/vilsn/Dropbox/Work/Turing/Exploration_Data/03_dimensionality_reduction.py)

- **What it does for the biologist**: Selects Highly Variable Genes (HVGs, default top 2,000; refinable via `--n_hvg`, `--hvg_method`, and `--min_cells_per_gene`), scales expression data, performs Principal Component Analysis (PCA), and projects cells onto a 2D UMAP manifold.
- **Optional Gene Filtering (No Phase Inference)**: Supports `--exclude_cell_cycle` (drops the cited 97-gene Tirosh et al. 2016 / Seurat `cc.genes.updated.2019` set + legacy aliases), `--exclude_genes_file` (drops any user-supplied `.txt`/`.csv`/`.tsv` list, e.g. Mouse `GO:0007049`), and `--exclude_genes` (comma-separated symbols). Every excluded gene and its citation is recorded in `output/03_excluded_genes_audit.csv`. Discrete G1/S/G2M phase classification and regression are intentionally omitted because megakaryocytes undergo endomitosis rather than canonical diploid mitosis (Luecken & Theis, 2019).
- **Outputs**:
  - `output/03_umap_supervised_ploidy.png`: UMAP plot where cells are colored by genome ploidy ($2\text{n}, 4\text{n}, 8\text{n}, 16\text{n}, 32\text{n}$).
  - `output/03_umap_dataset_cohort.png`: UMAP plot comparing Train vs. Pilot cohorts.
  - `output/03_umap_gene_kalrn.png`: Expression gradient overlay on the UMAP landscape.
  - `output/03_excluded_genes_audit.csv`: Audit log of excluded genes and provenance (created when filtering is enabled).

---

### Step 4: Multi-Omics Factor Analysis (MOFA)
**Script**: [`04_mofa_multimodal_analysis.py`](file:///c:/Users/vilsn/Dropbox/Work/Turing/Exploration_Data/04_mofa_multimodal_analysis.py)

- **What it does for the biologist**: Integrates single-cell RNA expression and cell morphology using linear factor modeling (`mofapy2`). It calculates how much variance ($R^2$) is explained by each latent factor in gene expression vs. cell shape, and identifies top driving gene and morphology features.
- **Outputs**:
  - `mofa_model_rna_morphology.hdf5`: Trained MOFA model.
  - `output/04_mofa_r2_variance_explained.png`: Heatmap of $R^2$ variance explained per view.
  - `output/04_mofa_factor1_vs_factor2.png`: Factor score scatter plot separating ploidy states.

---

### Step 4b: PyTorch Multimodal VAE (MVAE)
**Script**: [`04b_mvae_multimodal_analysis.py`](file:///c:/Users/vilsn/Dropbox/Work/Turing/Exploration_Data/04b_mvae_multimodal_analysis.py)

- **What it does for the biologist**: Uses a deep Multimodal Variational Autoencoder with a **Product-of-Experts (PoE)** encoder to learn a joint non-linear latent space $z$. Evaluates how accurately the neural network reconstructs original RNA expression and imaging morphology parameters.
- **Outputs**:
  - `output/mvae_model_weights.pt`: Saved PyTorch model weights.
  - `output/04b_mvae_training_loss_curve.png`: Convergence loss history (ELBO, Reconstruction, KL divergence).
  - `output/04b_mvae_rna_reconstruction_scatter.png`: Original vs. reconstructed RNA expression scatter plot & $R^2$ score.
  - `output/04b_mvae_morphology_reconstruction_scatter.png`: Original vs. reconstructed morphology parameters scatter plot & $R^2$ score.
  - `output/04b_mvae_latent_space_umap.png`: UMAP of MVAE latent space $z$ colored by ploidy.

---

### Step 5: Biological Pathway Enrichment & WordCloud
**Script**: [`05_pathway_enrichment.py`](file:///c:/Users/vilsn/Dropbox/Work/Turing/Exploration_Data/05_pathway_enrichment.py)

- **What it does for the biologist**: Runs Over-Representation Analysis (ORA / Enrichr) using `gseapy` against MSigDB Hallmark, KEGG, and GO Biological Process databases to highlight enriched biological pathways (e.g., platelet activation, endomitosis, cell cycle, cytoskeletal remodeling).
- **Outputs**:
  - `output/05_pathway_enrichment_dotplot.png`: Pathway enrichment dotplot.
  - `output/05_pathway_wordcloud.png`: Biological pathway WordCloud visualization.

---

### Step 6: Morphology-to-Gene Spearman Rank Correlations
**Script**: [`06_morphology_gene_correlations.py`](file:///c:/Users/vilsn/Documents/morphotranscriptomics/06_morphology_gene_correlations.py)

- **What it does for the biologist**: Computes robust pairwise **Spearman rank correlations ($\rho$)**, Benjamini–Hochberg FDR ($q$-values), and partial Spearman correlations controlling for $\log_2(\text{Ploidy})$ between all expressed genes and imaging flow cytometry morphology parameters across all 393 QC-passed cells (M1–M7), while remaining backwards-compatible with legacy single-cohort datasets (`--legacy_split` or `--cohort 10plate` / `--cohort 4plate`).
- **Optional Gene Filtering**: Supports `--exclude_cell_cycle`, `--exclude_genes_file`, and `--exclude_genes` prior to library-depth adjustment and Spearman correlation testing, writing `output/06_excluded_genes_audit.csv` when filtering is enabled.
- **Outputs**:
  - `output/06_top_gene_morphology_spearman_scatter_grid.png`: 12-panel scatter & regression grid of top positively/negatively correlated gene–morphology pairs colored by `Supervised_ploidy`.
  - `output/06_gene_morphology_spearman_ranking_labeled.png`: Two-panel volcano & ranked bar chart labeling genes with particularly high Spearman $|\rho|$ and key megakaryocyte markers (`Ppbp`, `Pde3a`, `Mylk`, `Kalrn`, `Rock1`, `Exoc6`, `Pf4`, `Itga2b`).
  - `output/06_morphology_gene_spearman_heatmap.png`: Hierarchically clustered Spearman correlation matrix across top genes and morphology features.
  - `output/morphology_gene_spearman_top_pairs.csv` & `output/morphology_gene_spearman_per_gene_summary.csv`: Full tabular correlation statistics.
  - `output/06_excluded_genes_audit.csv`: Audit log of excluded genes and provenance (created when filtering is enabled).

---

## 🔬 Cell Cycle Gene Lists & Phase Regression Methodology (Citations)

This pipeline provides optional, auditable cell-cycle gene filtering via [`cell_cycle.py`](file:///c:/Users/vilsn/Documents/morphotranscriptomics/cell_cycle.py) across **Steps 2, 3, 4, 4b, and 6** (`--exclude_cell_cycle`, `--exclude_genes_file`, `--exclude_genes`, `--exclude_steps`).

### Cited Seurat Cell-Cycle Gene Lists
- **Tirosh et al. (2016) / Seurat `cc.genes`**: Canonical human cell-cycle gene signature comprising **43 S-phase** and **54 G2/M-phase** markers (97 genes total) derived from single-cell RNA-seq of proliferating cells (Tirosh et al., 2016, *Science* 352(6282):189–196, DOI: [10.1126/science.aad0501](https://doi.org/10.1126/science.aad0501)).
- **Seurat `cc.genes.updated.2019`**: Updated HGNC gene symbols introduced in Seurat v3 (Butler et al., 2018, *Nature Biotechnology* 36:411–420, DOI: [10.1038/nbt.4096](https://doi.org/10.1038/nbt.4096); Stuart et al., 2019, *Cell* 177(7):1888–1902, DOI: [10.1016/j.cell.2019.05.031](https://doi.org/10.1016/j.cell.2019.05.031)). In `cell_cycle.py`, all 97 canonical mouse ortholog symbols (`TIROSH_2016_S_GENES_MOUSE`, `TIROSH_2016_G2M_GENES_MOUSE`) plus **6 legacy GRCm38/Tirosh aliases** (`Mlf1ip`→`Cenpu`, `Rpa2`→`Polr1b`, `Brip1`→`Mrpl36`, `Mcm2`→`Mcm7`, `Fam64a`→`Pimreg`, `Hn1`→`Jpt1`) are matched case-insensitively to prevent silent misses across older and newer genome annotations.

### Standard Seurat Phase Scoring & Regression Procedures vs. Endomitotic Megakaryocytes
In standard proliferating diploid scRNA-seq workflows, Seurat's **Cell-Cycle Scoring and Regression** procedure (Satija Lab vignette; Butler et al., 2018; Stuart et al., 2019) proceeds in two stages:
1. **Scoring & Discrete Phase Assignment**: `CellCycleScoring(s.features = cc.genes.updated.2019$s.genes, g2m.features = cc.genes.updated.2019$g2m.genes)` computes continuous `S.Score` and `G2M.Score` module scores per cell and assigns discrete `G1`, `S`, or `G2M` phases.
2. **Linear Phase Regression**:
   - *Full cell-cycle removal*: `ScaleData(vars.to.regress = c("S.Score", "G2M.Score"))` regresses out both S and G2/M scores to remove all cell-cycle variance.
   - *Difference-score regression (alternate workflow)*: `CC.Difference <- S.Score - G2M.Score` followed by `ScaleData(vars.to.regress = "CC.Difference")` removes phase differences among actively cycling cells while preserving separation between quiescent (`G1`) and cycling cells.

**Why this pipeline uses explicit gene-list filtering instead of diploid phase regression**: Megakaryocytes undergo **endomitosis** (aborted mitosis without cytokinesis to reach $4\text{n}$–$32\text{n}+$ ploidy). Applying diploid mitotic phase classifiers (`G1`/`S`/`G2M`) or regressing out `S.Score`/`G2M.Score` risks over-specifying endomitotic states and inadvertently removing genuine polyploidization biology (Luecken & Theis, 2019, *Molecular Systems Biology* 15(6):e8746, DOI: [10.15252/msb.20188746](https://doi.org/10.15252/msb.20188746)). Instead, this pipeline keeps the original unfiltered baseline by default and offers transparent, reproducible gene-list exclusion with per-step audit logs (`*_excluded_genes_audit.csv`).

---

## 🔄 Backwards Compatibility with Legacy Datasets

Both Step 1 (`01_qc_and_metadata_filtering.py`) and Step 6 (`06_morphology_gene_correlations.py`) retain full backwards compatibility with legacy Train/Pilot 2-dataset splits and older metadata schemas:
- **Legacy Split Mode**: Pass `--legacy_split` to `run_master_analysis.py` or `01_qc_and_metadata_filtering.py` to ingest legacy Train (`Metadata_flow+well_id+supervised_ploidy.csv`, `meta.contaminant.2.tsv` / `meta.contaminant.v1.csv`) and Pilot (`meta.v1+flowSF.csv`) directories instead of the unified `Metadata_merged_YL_SF_20.10.26_v2.csv`.
- **Legacy Column & Ploidy Normalization**: Automatically harmonizes older column conventions (`Well_ID`/`Well`/`Barcode`, `Mouse_Plate`/`Sample_Plate`/`orig.ident`, `Supervised_Ploidy_Images`/`Estimated_Ploidy_D.NA.A`, `kallisto_mapped`/`kallisto_mappe_pc`) and ploidy casing (`2N`–`32N` → `2n`–`32n`).
- **Cohort Selection in Step 6**: Use `--cohort 10plate` (M1–M5), `--cohort 4plate` (M6–M7), or `--metadata_file <legacy_csv_or_tsv>` directly.

---

## 🚀 How to Run the Master Pipeline

You can run the entire pipeline end-to-end or individual steps using **`run_master_analysis.py`**:

```powershell
# Activate environment
conda activate sc-analysis

# Run FULL pipeline end-to-end (baseline without gene exclusions, matching original notebook)
python run_master_analysis.py --train_dir "D:\Morphotranscriptomic_Data" --test_dir "exp" --out_dir "output" --gene kalrn

# Refine read-count QC, gene prevalence, and HVG selection criteria end-to-end:
#   --min_mapped_reads (default: 20000), --min_map_pc (default: 50.0),
#   --min_cells_per_gene (default: 0), --n_hvg (default: 2000),
#   --hvg_method: default | raw_var | log1p_var | seurat | seurat_v3
python run_master_analysis.py --min_mapped_reads 50000 --min_map_pc 60.0 --min_cells_per_gene 5 --n_hvg 3000 --hvg_method seurat --out_dir "output_refined"

# Exclude cited cell cycle genes (Tirosh et al. 2016 / Seurat cc.genes.updated.2019) across ALL analyses:
python run_master_analysis.py --exclude_cell_cycle --out_dir "output_no_cc"

# Exclude a custom gene list from a file (e.g., Mouse GO:0007049 Cell Cycle .txt/.csv/.tsv) across ALL analyses:
python run_master_analysis.py --exclude_genes_file "my_genes_to_exclude.txt" --out_dir "output_custom_filter"

# Apply gene filtering to JUST ONE analysis (e.g. Step 3 PCA/UMAP only):
python run_master_analysis.py --step 3 --exclude_cell_cycle
# or restrict exclusions to specific steps during a full pipeline run:
python run_master_analysis.py --exclude_cell_cycle --exclude_steps 3,4

# Run individual scripts directly with gene exclusion flags:
python 03_dimensionality_reduction.py --exclude_cell_cycle
python 03_dimensionality_reduction.py --exclude_genes_file "go_0007049.txt"
python 04_mofa_multimodal_analysis.py --exclude_cell_cycle --exclude_genes "Mki67,Top2a"
python 04b_mvae_multimodal_analysis.py --exclude_cell_cycle
python 02_gene_expression_rankings.py --exclude_cell_cycle
python 06_morphology_gene_correlations.py --exclude_cell_cycle

# Run backwards-compatible legacy Train/Pilot 2-dataset split:
python run_master_analysis.py --legacy_split --out_dir "output_legacy"

# Run Step 6 (Morphology-to-Gene Spearman Correlations) with cell-cycle exclusion:
python run_master_analysis.py --step 6 --exclude_cell_cycle --out_dir "output_no_cc"

# Run only Step 1 (Quality Control):
python run_master_analysis.py --step 1

# Run gene expression ranking for a different gene (e.g. pf4):
python run_master_analysis.py --gene pf4

# Run Leave-One-Animal-Out Cross-Validation:
python run_master_analysis.py --step cv
# or standalone:
python leave_one_animal_out_cv.py
```

---

## 📈 Summary of Generated Output Files & Figures

| File / Figure Name | Description | Step |
| :--- | :--- | :---: |
| `qc_filtered_metadata.csv` | Cleaned table of all 291 QC-passed single cells | Step 1 |
| `01_ploidy_distribution_qc.png` | Supervised ploidy cell counts before & after QC | Step 1 |
| `01_kallisto_qc_metrics_violin.png` | Read depth & mapping % violin plots by ploidy | Step 1 |
| `kalrn_top_bottom_cells.csv` | Top/bottom cell mapping to FCS, CVW, Mouse, Plate, Well | Step 2 |
| `02_gene_rankings_kalrn.png` | Bar chart of top & bottom expressing cells | Step 2 |
| `03_umap_supervised_ploidy.png` | UMAP landscape colored by genome ploidy | Step 3 |
| `03_umap_dataset_cohort.png` | UMAP landscape comparing Train vs. Pilot cohorts | Step 3 |
| `03_umap_gene_kalrn.png` | Expression gradient overlay on UMAP | Step 3 |
| `04_mofa_r2_variance_explained.png` | MOFA $R^2$ variance decomposition per view | Step 4 |
| `04_mofa_factor1_vs_factor2.png` | MOFA latent factor scatter plot by ploidy | Step 4 |
| `04b_mvae_training_loss_curve.png` | PyTorch MVAE loss convergence history | Step 4b |
| `04b_mvae_rna_reconstruction_scatter.png` | RNA expression reconstruction scatter & $R^2$ score | Step 4b |
| `04b_mvae_morphology_reconstruction_scatter.png` | Morphology feature reconstruction scatter & $R^2$ score | Step 4b |
| `04b_mvae_latent_space_umap.png` | MVAE PoE latent space UMAP by ploidy | Step 4b |
| `05_pathway_enrichment_dotplot.png` | Enrichr biological pathway dotplot | Step 5 |
| `05_pathway_wordcloud.png` | Biological pathway WordCloud | Step 5 |
| `06_top_gene_morphology_spearman_scatter_grid.png` | 12-panel scatter & regression grid of top gene–morphology Spearman pairs | Step 6 |
| `06_gene_morphology_spearman_ranking_labeled.png` | Volcano & ranked gene bar chart highlighting top Spearman $|\rho|$ & MK markers | Step 6 |
| `06_morphology_gene_spearman_heatmap.png` | Hierarchically clustered gene–morphology Spearman correlation matrix | Step 6 |
| `morphology_gene_spearman_top_pairs.csv` | Top 500 pairwise Spearman $\rho$, BH-FDR $q$, and partial $\rho\mid\text{ploidy}$ | Step 6 |
| `morphology_gene_spearman_per_gene_summary.csv` | Per-gene strongest morphology correlation summary | Step 6 |
| `*_excluded_genes_audit.csv` | Audit log of excluded cell-cycle / custom genes & citations (when enabled) | Steps 2–4b, 6 |

---
