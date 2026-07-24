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

The analysis operates on two distinct data cohorts with harmonized metadata handling:

### 1. Primary Train Dataset (`D:\Morphotranscriptomic_Data`)
- **Main Experimental Cohort**: Single cells isolated from Mice **M1 through M5** across **Plate 1** and **Plate 2**.
- **Quality Control (QC) Benchmark**: Exactly **291 high-quality single cells** pass strict QC filtering in `morphotrans2`:
  - `Contaminated == False`: Excludes wells exhibiting cross-species or microbial contamination.
  - `Supervised_ploidy`: Retains valid ploidy classes ($2\text{n}, 4\text{n}, 8\text{n}, 16\text{n}, 32\text{n}$) while excluding $64\text{n}$ (due to low cell counts) and unassigned artifacts.
  - `control == 0`: Excludes empty control wells.
  - `number_of_cell == 1`: Verifies single-cell sorting per well.
  - `kallisto.pc_map > 50.0%`: Requires $>50\%$ read alignment rate.
  - `kallisto.nb_mapped > 20,000`: Requires $>20,000$ mapped reads per cell.

### 2. Pilot / Test Dataset (`exp/` and local CSVs)
- **Calibration Cohort**: Preliminary sorting runs (e.g. `250530_BM_2m_250602_sortM1plt...`) used for initial gating and cross-dataset validation.

### 3. Flow Cytometry Mappings (`.fcs` & `.cvw`)
Cell barcodes (e.g. `M1p1_A10`) are linked to their physical flow cytometry (`.fcs`) data files and FlowJo workspace (`.cvw`) files:
- Mouse 1: `sortM1p1.fcs` / `sortM1p1.cvw` & `sortM1p2.fcs` / `sortM1p2.cvw`
- Mouse 2: `sortM2p1.fcs` / `sortM2p1.cvw` & `M2p2.fcs` / `M2p2.cvw`
- Mice M3, M4, M5: Plates 1 & 2 respective `.fcs` and `.cvw` files.

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
├── 01_qc_and_metadata_filtering.py          # Step 1: QC filtering & Train/Pilot metadata harmonization
├── 02_gene_expression_rankings.py          # Step 2: Vectorized gene rankings & FCS/CVW mapping
├── 03_dimensionality_reduction.py          # Step 3: Scanpy HVG selection, PCA, & UMAP
├── 04_mofa_multimodal_analysis.py          # Step 4: Multi-Omics Factor Analysis (MOFA)
├── 04b_mvae_multimodal_analysis.py         # Step 4b: PyTorch Multimodal VAE (MVAE)
├── 05_pathway_enrichment.py                # Step 5: Pathway enrichment (gseapy) & WordCloud
└── run_master_analysis.py                  # Master CLI Pipeline Runner
```

---

### Step 1: Quality Control & Metadata Harmonization
**Script**: [`01_qc_and_metadata_filtering.py`](file:///c:/Users/vilsn/Dropbox/Work/Turing/Exploration_Data/01_qc_and_metadata_filtering.py)

- **What it does for the biologist**: Ingests raw sequencing statistics and metadata. It fixes column naming differences between train and pilot experiments (`SampleID` vs `Well` vs `Barcode`), filters out dead/contaminated/empty wells, and verifies the **291 QC-passed cell benchmark**.
- **Outputs**:
  - `output/qc_filtered_metadata.csv`: Cleaned metadata table.
  - `output/01_ploidy_distribution_qc.png`: Bar chart of ploidy counts before vs. after QC.
  - `output/01_kallisto_qc_metrics_violin.png`: Violin plots of read counts and mapping percentages across $2\text{n}$–$32\text{n}$ ploidies.

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

- **What it does for the biologist**: Selects the top 2,000 Highly Variable Genes (HVGs), scales expression data, performs Principal Component Analysis (PCA), and projects cells onto a 2D UMAP manifold.
- **Outputs**:
  - `output/03_umap_supervised_ploidy.png`: UMAP plot where cells are colored by genome ploidy ($2\text{n}, 4\text{n}, 8\text{n}, 16\text{n}, 32\text{n}$).
  - `output/03_umap_dataset_cohort.png`: UMAP plot comparing Train vs. Pilot cohorts.
  - `output/03_umap_gene_kalrn.png`: Expression gradient overlay on the UMAP landscape.

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

## 🚀 How to Run the Master Pipeline

You can run the entire pipeline end-to-end or individual steps using **`run_master_analysis.py`**:

```powershell
# Activate environment
conda activate sc-analysis

# Run FULL pipeline end-to-end for default gene 'kalrn'
python run_master_analysis.py --train_dir "D:\Morphotranscriptomic_Data" --test_dir "exp" --out_dir "output" --gene kalrn

# Run only Step 1 (Quality Control):
python run_master_analysis.py --step 1

# Run gene expression ranking for a different gene (e.g. pf4):
python run_master_analysis.py --gene pf4

# Run PyTorch MVAE Deep Learning training:
python 04b_mvae_multimodal_analysis.py --epochs 100 --latent_dim 10 --out_dir "output"
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

---

## ❓ Support & Questions
For any questions regarding dataset paths or script execution, refer to `GEMINI.md` for project standards or inspect `walkthrough.md`.
