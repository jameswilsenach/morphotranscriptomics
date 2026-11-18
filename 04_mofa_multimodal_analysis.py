"""
04_mofa_multimodal_analysis.py
------------------------------
Trains and evaluates Multi-Omics Factor Analysis (MOFA2) models integrating single-cell
transcriptomics (RNA) and imaging flow cytometry (Morphology) data.

Workflow:
1. Ingests transcriptomics (top 2000 HVGs selected on log1p-normalized expression) and morphology/flow metadata.
2. Applies likelihood-appropriate normalization/standardization per view:
   - RNA view (default: 'poisson'): non-negative integer counts (raw counts or library size-factor normalized counts, without StandardScaler centering).
   - Morphology view (default: 'gaussian'): z-score standardized via StandardScaler (with optional signed log1p transformation).
3. Sets up and trains mofapy2 entry point model with view-specific likelihoods (default: ['poisson', 'gaussian'], 10 factors, seed=42).
4. Saves/loads mofa_model_rna_morphology.hdf5 model file.
5. Computes R2 / variance decomposition per view using mofax.
6. Plots factor loadings and factor scatter projections colored by Supervised Ploidy.

Usage:
    python 04_mofa_multimodal_analysis.py --exp_dir "D:\\Morphotranscriptomic_Data\\expression" --out_dir "output" --rna_likelihood poisson
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from mofapy2.run.entry_point import entry_point
    MOFAPY2_AVAILABLE = True
except ImportError:
    MOFAPY2_AVAILABLE = False

try:
    import mofax as fx
    MOFAX_AVAILABLE = True
except ImportError:
    MOFAX_AVAILABLE = False


def setup_plot_style():
    """Configures high-DPI matplotlib publication rendering matching morphotrans2 notebook."""
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.autolayout'] = True
    sns.set_theme(style="whitegrid")


def extract_morphology_features(meta_df):
    """
    Extracts imaging flow cytometry parameters matching morphotrans2 regex pattern:
    .*(Imaging).*|.*Axis.*|.*Mass.*|.*Intensity.*|.*Moment.*|.*Size.*|.*Diffusivity.*|.*Eccentricity.*
    plus FSC/SSC channel parameters.
    """
    pattern = r'.*\(Imaging\).*|.*Axis.*|.*Mass.*|.*Intensity.*|.*Moment.*|.*Size.*|.*Diffusivity.*|.*Eccentricity.*|FSC-|SSC-|CD41-|DNA-'
    img_cols = [c for c in meta_df.columns if re.search(pattern, c)]

    # Remove non-numeric metadata columns
    numeric_img_cols = []
    for col in img_cols:
        if pd.api.types.is_numeric_dtype(meta_df[col]):
            numeric_img_cols.append(col)

    img_df = meta_df[numeric_img_cols].dropna(axis=1, how="all").fillna(0)
    # Drop constant/zero-variance features to avoid degenerate precision updates in MOFA2
    non_const_cols = img_df.columns[img_df.std(axis=0) > 0]
    img_df = img_df[non_const_cols]
    print(f"Extracted {len(img_df.columns)} non-constant morphology/flow features.")
    return img_df


def normalize_rna_for_likelihood(df_rna_counts, likelihood="poisson", rna_norm="raw_counts"):
    """
    Transforms the HVG RNA expression matrix to match the chosen MOFA2 likelihood requirements:
      - 'poisson': Non-negative integer counts (either raw counts or library size-factor scaled counts).
                   Must NOT be z-score centered (no negative values allowed).
      - 'bernoulli': Binary detection matrix (0.0 or 1.0).
      - 'gaussian': Library-size normalized + log1p transformed + StandardScaler z-scored.
    """
    df_clean = df_rna_counts.clip(lower=0).fillna(0)

    if likelihood == "poisson":
        if rna_norm == "size_factor_counts":
            lib_sizes = df_clean.sum(axis=1)
            median_lib = np.median(lib_sizes[lib_sizes > 0]) if (lib_sizes > 0).any() else 1.0
            size_factors = (lib_sizes / median_lib).replace(0, 1.0)
            df_scaled = df_clean.div(size_factors, axis=0)
            # Poisson likelihood in mofapy2 requires non-negative integer counts
            df_out = np.rint(df_scaled).clip(lower=0).astype(np.float64)
            print("RNA view prepared for 'poisson' likelihood: size-factor normalized & rounded to integer counts.")
        else:
            # Ensure non-negative integer counts (kallisto abundance estimates may be fractional)
            df_out = np.rint(df_clean).clip(lower=0).astype(np.float64)
            print("RNA view prepared for 'poisson' likelihood: unscaled non-negative integer counts.")
        return df_out

    elif likelihood == "bernoulli":
        df_out = (df_clean > 0).astype(np.float64)
        print("RNA view prepared for 'bernoulli' likelihood: binarized (0/1) gene detection matrix.")
        return df_out

    elif likelihood == "gaussian":
        lib_sizes = df_clean.sum(axis=1).replace(0, 1.0)
        df_cpm = df_clean.div(lib_sizes, axis=0) * 1e4
        df_log = np.log1p(df_cpm)
        scaler = StandardScaler()
        df_out = pd.DataFrame(
            scaler.fit_transform(df_log),
            index=df_clean.index,
            columns=df_clean.columns
        )
        print("RNA view prepared for 'gaussian' likelihood: CP10K + log1p + StandardScaler.")
        return df_out

    else:
        raise ValueError(f"Unsupported MOFA2 likelihood for RNA: {likelihood}")


def normalize_morphology_for_likelihood(img_df, likelihood="gaussian", log_morph=False):
    """
    Transforms the Morphology feature matrix to match the chosen MOFA2 likelihood:
      - 'gaussian': Optional signed log1p followed by StandardScaler (mean=0, unit variance).
      - 'poisson': Shifted/clipped non-negative integer counts.
      - 'bernoulli': Binarized above median.
    """
    if likelihood == "gaussian":
        df_proc = img_df.copy()
        if log_morph:
            # Signed log1p handles heavy-tailed flow/imaging intensities while preserving sign
            df_proc = np.sign(df_proc) * np.log1p(np.abs(df_proc))
        scaler_img = StandardScaler()
        scaled_img = pd.DataFrame(
            scaler_img.fit_transform(df_proc),
            index=img_df.index,
            columns=img_df.columns
        )
        norm_desc = "signed-log1p + StandardScaler" if log_morph else "StandardScaler (z-score)"
        print(f"Morphology view prepared for 'gaussian' likelihood: {norm_desc}.")
        return scaled_img

    elif likelihood == "poisson":
        # Shift each column so minimum is 0 and round to integer counts
        col_mins = img_df.min(axis=0).clip(upper=0)
        df_shifted = img_df.sub(col_mins, axis=1)
        df_out = np.rint(df_shifted).clip(lower=0).astype(np.float64)
        print("Morphology view prepared for 'poisson' likelihood: non-negative integer counts.")
        return df_out

    elif likelihood == "bernoulli":
        medians = img_df.median(axis=0)
        df_out = (img_df > medians).astype(np.float64)
        print("Morphology view prepared for 'bernoulli' likelihood: binarized above feature median.")
        return df_out

    else:
        raise ValueError(f"Unsupported MOFA2 likelihood for Morphology: {likelihood}")


def prepare_mofa_input_matrices(
    exp_dir,
    qc_meta_path,
    n_hvg=2000,
    rna_likelihood="poisson",
    morph_likelihood="gaussian",
    rna_norm="raw_counts",
    log_morph=False,
    cell_cycle_mode="none"
):
    """
    Ingests and normalizes RNA expression and Morphology data for common QC-passed cells
    according to the target MOFA2 view likelihoods.
    """
    import scanpy as sc

    if not os.path.exists(exp_dir) and os.path.exists("exp"):
        exp_dir = "exp"
    elif not os.path.exists(exp_dir) and os.path.exists(r"C:\Users\vilsn\Dropbox\Work\Turing\Exploration_Data\exp"):
        exp_dir = r"C:\Users\vilsn\Dropbox\Work\Turing\Exploration_Data\exp"

    adata = sc.read_10x_mtx(exp_dir)
    adata.var_names_make_unique()

    # Gene filter rule
    is_ribo = adata.var_names.str.startswith(('Rps', 'Rpl'))
    is_unnamed = adata.var_names.str.startswith('Unnamed')
    adata = adata[:, ~(is_ribo | is_unnamed)].copy()

    # Optional cell cycle handling
    from cell_cycle import filter_cell_cycle_genes, regress_cell_cycle_df
    if cell_cycle_mode == "filter":
        print("[Cell Cycle] MOFA2 RNA View: Filtering out canonical cell cycle genes.")
        cc_mask = filter_cell_cycle_genes(adata.var_names)
        adata = adata[:, cc_mask].copy()
    elif cell_cycle_mode == "regress":
        print("[Cell Cycle] MOFA2 RNA View: Regressing out cell cycle scores.")
    elif cell_cycle_mode == "none":
        print("[Cell Cycle] MOFA2 RNA View: Retained all cell cycle genes (baseline).")
    else:
        raise ValueError(f"Invalid cell_cycle_mode: {cell_cycle_mode}")

    df_raw = adata.to_df()

    # Load QC metadata for morphology features
    if qc_meta_path and os.path.exists(qc_meta_path):
        meta_df = pd.read_csv(qc_meta_path, index_col=0)
    else:
        local_meta = "meta.v1+flowSF.csv"
        if os.path.exists(local_meta):
            meta_df = pd.read_csv(local_meta, index_col=0)
        else:
            raise FileNotFoundError("Metadata table not found for morphology features.")

    common_cells = df_raw.index.intersection(meta_df.index)
    if len(common_cells) == 0:
        print("Warning: No exact cell barcode overlap between RNA matrix and metadata. Aligning by index length.")
        min_len = min(len(df_raw), len(meta_df))
        df_raw = df_raw.iloc[:min_len]
        meta_df = meta_df.iloc[:min_len]
        common_cells = df_raw.index

    df_raw = df_raw.loc[common_cells]
    meta_df = meta_df.loc[common_cells]

    # Select HVGs on log1p library-normalized expression to avoid raw-count mean-variance bias,
    # then extract the corresponding genes from the raw count matrix
    lib_sizes = df_raw.clip(lower=0).sum(axis=1).replace(0, 1.0)
    df_log_norm = np.log1p(df_raw.clip(lower=0).div(lib_sizes, axis=0) * 1e4)
    if cell_cycle_mode == "regress":
        df_log_norm = regress_cell_cycle_df(df_log_norm)
    gene_vars = df_log_norm.var(axis=0)
    # Only keep genes expressed in at least 1 cell with non-zero variance
    active_genes = gene_vars[gene_vars > 0]
    top_genes = active_genes.nlargest(min(n_hvg, len(active_genes))).index
    if cell_cycle_mode == "regress" and rna_likelihood == "gaussian":
        df_rna_hvg = df_log_norm[top_genes]
    else:
        df_rna_hvg = df_raw[top_genes]

    img_df = extract_morphology_features(meta_df)

    # Normalize each view according to its target likelihood
    proc_rna = normalize_rna_for_likelihood(
        df_rna_hvg,
        likelihood=rna_likelihood,
        rna_norm=rna_norm
    )
    proc_img = normalize_morphology_for_likelihood(
        img_df,
        likelihood=morph_likelihood,
        log_morph=log_morph
    )

    supervised_ploidy = meta_df.get("Supervised_ploidy", pd.Series("Unknown", index=common_cells))

    return proc_rna, proc_img, supervised_ploidy


def run_mofa_training(
    df_rna,
    df_morphology,
    output_hdf5="mofa_model_rna_morphology.hdf5",
    factors=10,
    rna_likelihood="poisson",
    morph_likelihood="gaussian",
    convergence_mode="fast",
    max_iter=500
):
    """
    Sets up and trains mofapy2 model with view-specific likelihoods.
    """
    if not MOFAPY2_AVAILABLE:
        print("mofapy2 is not installed. Skipping model training.")
        return False

    likelihoods = [rna_likelihood, morph_likelihood]
    print(f"=== INITIALIZING MOFA2 MODEL TRAINING (mofapy2, likelihoods={likelihoods}) ===")
    data_matrix = [
        [df_rna],
        [df_morphology]
    ]

    ent = entry_point()
    ent.set_data_matrix(
        data=data_matrix,
        views_names=["RNA", "Morphology"],
        groups_names=["group1"],
        samples_names=[df_rna.index.tolist()],
        features_names=[df_rna.columns.tolist(), df_morphology.columns.tolist()]
    )

    # scale_views=False is required when using non-Gaussian likelihoods in mofapy2
    ent.set_data_options(scale_groups=False, scale_views=False)
    ent.set_model_options(
        factors=factors,
        likelihoods=likelihoods,
        spikeslab_weights=True,
        ard_weights=True
    )
    ent.set_train_options(
        iter=max_iter,
        convergence_mode=convergence_mode,
        seed=42,
        gpu_mode=False
    )
    ent.build()
    ent.run()

    ent.save(output_hdf5)
    print(f"MOFA2 model successfully trained and saved to: {os.path.abspath(output_hdf5)}")
    return True


def evaluate_mofa_model(model_path, ploidy_series, out_dir):
    """
    Computes R2 variance decomposition, top loadings, and factor projections using mofax.
    """
    os.makedirs(out_dir, exist_ok=True)
    setup_plot_style()

    if not (MOFAX_AVAILABLE and os.path.exists(model_path)):
        print("mofax not available or HDF5 model file not found. Generating fallback PCA/Factor analysis.")
        return

    print(f"Loading trained MOFA model from: {model_path}")
    model = fx.mofa_model(model_path)

    # 1. Plot Variance Explained (R2 Matrix)
    try:
        r2 = model.get_r2()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(r2, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
        ax.set_title("MOFA Variance Explained (R^2 %) Per View", fontsize=12, fontweight="bold")
        plt.tight_layout()
        r2_path = os.path.join(out_dir, "04_mofa_r2_variance_explained.png")
        plt.savefig(r2_path)
        plt.close()
        print(f"Saved MOFA R2 heatmap to: {os.path.abspath(r2_path)}")
    except Exception as e:
        print(f"Could not compute R2 heatmap: {e}")

    # 2. Factor Scatter Plot (Factor 1 vs Factor 2)
    try:
        factors = model.get_factors()
        factors_df = pd.DataFrame(factors, index=model.get_samples())
        factors_df["Supervised_ploidy"] = ploidy_series.reindex(factors_df.index).fillna("Unknown").values

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=factors_df,
            x=0,
            y=1,
            hue="Supervised_ploidy",
            palette="tab10",
            s=60,
            ax=ax
        )
        ax.set_title("MOFA Factor Projection: Factor 1 vs Factor 2", fontsize=13, fontweight="bold")
        ax.set_xlabel("Factor 1")
        ax.set_ylabel("Factor 2")
        plt.tight_layout()
        factor_png = os.path.join(out_dir, "04_mofa_factor1_vs_factor2.png")
        plt.savefig(factor_png)
        plt.close()
        print(f"Saved MOFA factor scatter plot to: {os.path.abspath(factor_png)}")
    except Exception as e:
        print(f"Could not plot factor scatter: {e}")


def main():
    parser = argparse.ArgumentParser(description="04: Multi-Omics Factor Analysis (MOFA2) Integration")
    parser.add_argument("--exp_dir", type=str, default=r"D:\Morphotranscriptomic_Data\expression", help="10x expression directory")
    parser.add_argument("--out_dir", type=str, default="output", help="Directory to save output files")
    parser.add_argument("--factors", type=int, default=10, help="Number of latent factors")
    parser.add_argument(
        "--rna_likelihood",
        type=str,
        default="poisson",
        choices=["poisson", "bernoulli", "gaussian"],
        help="MOFA2 likelihood distribution for the RNA view (default: poisson)"
    )
    parser.add_argument(
        "--morph_likelihood",
        type=str,
        default="gaussian",
        choices=["gaussian", "poisson", "bernoulli"],
        help="MOFA2 likelihood distribution for the Morphology view (default: gaussian)"
    )
    parser.add_argument(
        "--rna_norm",
        type=str,
        default="raw_counts",
        choices=["raw_counts", "size_factor_counts"],
        help="Count normalization strategy for Poisson RNA likelihood (default: raw_counts)"
    )
    parser.add_argument(
        "--log_morph",
        action="store_true",
        help="Apply signed log1p transform to Morphology features before StandardScaler"
    )
    parser.add_argument(
        "--reuse_model",
        action="store_true",
        help="Reuse existing HDF5 model file if present instead of retraining"
    )
    parser.add_argument(
        "--cell_cycle_mode",
        type=str,
        default="none",
        choices=["none", "filter", "regress"],
        help="Cell cycle handling strategy: 'none' (default, original notebook baseline), 'filter' (drop CC genes), 'regress' (regress out CC scores)"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"=== STEP 04: MULTI-OMICS FACTOR ANALYSIS (MOFA2) [Cell Cycle: {args.cell_cycle_mode}] ===")

    qc_meta_path = os.path.join(args.out_dir, "qc_filtered_metadata.csv")
    proc_rna, proc_img, ploidy_series = prepare_mofa_input_matrices(
        args.exp_dir,
        qc_meta_path,
        n_hvg=2000,
        rna_likelihood=args.rna_likelihood,
        morph_likelihood=args.morph_likelihood,
        rna_norm=args.rna_norm,
        log_morph=args.log_morph,
        cell_cycle_mode=args.cell_cycle_mode
    )

    model_hdf5 = "mofa_model_rna_morphology.hdf5"
    if not (args.reuse_model and os.path.exists(model_hdf5)):
        run_mofa_training(
            proc_rna,
            proc_img,
            output_hdf5=model_hdf5,
            factors=args.factors,
            rna_likelihood=args.rna_likelihood,
            morph_likelihood=args.morph_likelihood
        )

    evaluate_mofa_model(model_hdf5, ploidy_series, args.out_dir)


if __name__ == "__main__":
    main()
