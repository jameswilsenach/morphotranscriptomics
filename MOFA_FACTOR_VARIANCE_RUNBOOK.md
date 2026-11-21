# MOFA Factor Variance & Heterogeneity Runbook

This runbook provides a standard operating procedure for evaluating **MOFA2 latent factor variance ($R^2$)** and **per-cell reconstruction error heterogeneity (RMSE)** across multiple latent dimensionalities ($K \in \{5, 10, 15\}$) in the Single Cell Morphotranscriptomics pipeline.

---

## 1. Purpose & Statistical Rationale

- **Why sweep $K$ (e.g., `5, 10, 15`)?**  
  MOFA2 uses Automatic Relevance Determination (ARD) to prune inactive factors, but the recovered factor structure and total variance explained ($R^2$) still depend on the upper bound $K$ and variational inference local optima. Sweeping `K = 5, 10, 15` lets you check whether biological programmes and total $R^2$ saturate or split as capacity increases.
- **Why inspect per-cell reconstruction error (heterogeneity)?**  
  Global $R^2$ averages across all cells. Per-cell reconstruction RMSE (ported and generalized from Cell 50 of the original `morphotrans2.ipynb` notebook) reveals **cellular heterogeneity**: which ploidy states ($2\text{n}$–$32\text{n}$) or animals (`M1`–`M7`) exhibit higher unexplained variance or outlier cells.
- **Why not average MOFA factors across CV folds?**  
  MOFA latent factors are unidentifiable up to sign flips, permutation, and subspace rotation. Rather than aggregating raw factor indices across folds, evaluate factor-invariant summaries ($R^2$ per view, per-cell reconstruction RMSE, and cross-$K$ sensitivity).

---

## 2. Prerequisites

1. Activate the project Conda environment:
   ```powershell
   conda activate sc-analysis
   ```
2. Ensure Step 1 (QC) has been run so `output/qc_filtered_metadata.csv` exists:
   ```powershell
   python run_master_analysis.py --step qc --out_dir "output"
   ```

---

## 3. Standard Execution Commands

### A. Recommended Multi-Factor Sweep (`K = 5, 10, 15`, Default Likelihoods)
Runs MOFA2 with `rna_likelihood=poisson` and `morph_likelihood=gaussian` across $K \in \{5, 10, 15\}$:

```powershell
python run_master_analysis.py --step mofa --mofa_factors "5,10,15" --out_dir "figures"
```

### B. Direct Step 5 Invocation (with Custom Options)
If you want to compare against a Gaussian RNA baseline (matching the original notebook's z-scored RNA formulation) or reuse existing HDF5 models to regenerate plots only:

```powershell
# Gaussian RNA sensitivity check across K = 5, 10, 15:
python 05_mofa_multimodal_analysis.py --out_dir "output_mofa_gaussian" --factors "5,10,15" --rna_likelihood gaussian

# Cell-cycle gene exclusion sensitivity check across K = 5, 10, 15:
python run_master_analysis.py --step mofa --mofa_factors "5,10,15" --exclude_cell_cycle --out_dir "output_mofa_no_cc"

# Regenerate R2 & heterogeneity plots from already-trained HDF5 files:
python 05_mofa_multimodal_analysis.py --out_dir "figures" --factors "5,10,15" --reuse_model
```

---

## 4. Generated Outputs & How to Interpret Them

| Output File | Description | What to Check |
| :--- | :--- | :--- |
| `05_mofa_factor_sweep_summary.csv` | Summary table across $K \in \{5, 10, 15\}$ with `Total_R2_RNA`, `Total_R2_Morphology`, `Mean_RMSE_*`, `Median_RMSE_*`, and `IQR_RMSE_*`. | Look for the "elbow" where `Total_R2` plateaus and `Mean_RMSE` stops dropping meaningfully. |
| `05_mofa_factor_sweep_error_and_r2.png` | Side-by-side comparison of Total $R^2$ (%) vs. $K$ (left) and per-cell RMSE boxplots vs. $K$ (right). | Confirms whether adding factors ($5 \to 10 \to 15$) genuinely reduces unexplained cell-level error or adds noise. |
| `05_mofa_r2_variance_explained_k{K}.png` | Factor-by-view $R^2$ (%) heatmap for each $K$. | Identify **shared** factors (non-trivial $R^2$ in both RNA and Morphology) vs. **view-specific** factors. |
| `05_mofa_reconstruction_error_heterogeneity_k{K}.png` | Per-cell reconstruction RMSE stratified by **Ploidy** (left) and **Animal `M1`–`M7`** (right). | Spot ploidy states with wide IQR (biological heterogeneity) or single animals with elevated median RMSE (batch/technical outlier). |
| `05_mofa_reconstruction_error_per_cell_k{K}.csv` | Cell-level table (`SampleID`, `Supervised_ploidy`, `mouse`, `Modality`, `Likelihood`, `Error_Metric`, `RMSE`, `Factors_K`). | Use to rank and inspect individual high-error outlier cells against QC read depth or flow morphology. |
| `05_mofa_factor1_vs_factor2_k{K}.png` | 2D scatter of Factor 1 vs. Factor 2 colored by `Supervised_ploidy`. | Verify qualitative consistency of the leading latent axes across $K$. |

---

## 5. Likelihood-Aware Error Metric Reference

Because MOFA2 supports view-specific likelihoods, per-cell reconstruction error is computed on the scale appropriate to each view's generative model ($\eta = Z W_v^\top$):

- **Gaussian view (default for Morphology):**  
  $$\text{RMSE}_i = \sqrt{\frac{1}{D_v}\sum_{j=1}^{D_v} \left(X_{ij} - \eta_{ij}\right)^2}$$
- **Poisson view (default for RNA):** Uses MOFA2's softplus rate link $\hat{\lambda}_{ij} = \log(1 + \exp(\eta_{ij}))$ and evaluates error on the $\log(1 + x)$ scale so high-count genes do not dominate:  
  $$\text{RMSE}_i = \sqrt{\frac{1}{D_v}\sum_{j=1}^{D_v} \left(\log(1 + X_{ij}) - \log(1 + \hat{\lambda}_{ij})\right)^2}$$
- **Bernoulli view:** Uses the sigmoid link $\hat{p}_{ij} = \sigma(\eta_{ij})$ and evaluates Brier-RMSE $\sqrt{\frac{1}{D_v}\sum_{j} (X_{ij} - \hat{p}_{ij})^2}$.

---

## 6. Diagnostic Checklist

1. **Underfitting ($K$ too small, e.g. $K = 5$):** `Total_R2` is substantially lower than at $K = 10$, and per-cell RMSE is elevated across all ploidies.
2. **Diminishing returns / overfitting ($K$ too large, e.g. $K = 15$):** Extra factors have near-zero $R^2$ (pruned by ARD) or explain only tiny single-view variance while `Median_RMSE` barely moves from $K = 10$.
3. **Ploidy-specific heterogeneity:** If higher ploidies ($16\text{n}, 32\text{n}$) show wider RMSE spread even at $K = 15$, this indicates genuine transcriptional/morphological sub-state diversity not captured by low-rank linear factors.
4. **Animal-specific shift:** If one mouse (`M1`–`M7`) has systematically higher RMSE in `05_mofa_reconstruction_error_heterogeneity_k{K}.png`, cross-check against Step 1 QC metrics (`01_kallisto_qc_metrics_violin.png`) and Step CV (`loaocv_summary_metrics.csv`).
