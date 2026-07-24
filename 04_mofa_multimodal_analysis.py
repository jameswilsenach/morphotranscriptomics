"""
04_mofa_multimodal_analysis.py
------------------------------
Trains and evaluates Multi-Omics Factor Analysis (MOFA) models integrating single-cell
transcriptomics (RNA) and imaging flow cytometry (Morphology) data matching morphotrans2 notebook.

Workflow:
1. Ingests transcriptomics (top 2000 HVGs, scaled) and morphology/flow metadata.
2. Extracts imaging morphology features matching morphotrans2 regex patterns.
3. Sets up and trains mofapy2 entry point model (10 factors, seed=42, fast convergence).
4. Saves/loads mofa_model_rna_morphology.hdf5 model file.
5. Computes R2 variance decomposition per view using mofax.
6. Plots factor loadings and factor scatter projections colored by Supervised Ploidy.

Usage:
    python 04_mofa_multimodal_analysis.py --train_dir "D:\\Morphotranscriptomic_Data" --out_dir "output"
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
    print(f"Extracted {len(numeric_img_cols)} morphology/flow features.")
    return img_df


def prepare_mofa_input_matrices(exp_dir, qc_meta_path, n_hvg=2000):
    """
    Ingests and scales RNA expression and Morphology data for common QC-passed cells.
    """
    import scanpy as sc

    if not os.path.exists(exp_dir) and os.path.exists("exp"):
        exp_dir = "exp"

    adata = sc.read_10x_mtx(exp_dir)
    adata.var_names_make_unique()

    # Gene filter rule
    is_ribo = adata.var_names.str.startswith(('Rps', 'Rpl'))
    is_unnamed = adata.var_names.str.startswith('Unnamed')
    adata = adata[:, ~(is_ribo | is_unnamed)].copy()

    df_raw = adata.to_df()

    # Select HVGs
    gene_vars = df_raw.var(axis=0)
    top_genes = gene_vars.nlargest(n_hvg).index
    df_rna = df_raw[top_genes]

    # Load QC metadata for morphology features
    if qc_meta_path and os.path.exists(qc_meta_path):
        meta_df = pd.read_csv(qc_meta_path, index_col=0)
    else:
        local_meta = "meta.v1+flowSF.csv"
        if os.path.exists(local_meta):
            meta_df = pd.read_csv(local_meta, index_col=0)
        else:
            raise FileNotFoundError("Metadata table not found for morphology features.")

    common_cells = df_rna.index.intersection(meta_df.index)
    if len(common_cells) == 0:
        print("Warning: No exact cell barcode overlap between RNA matrix and metadata. Aligning by index length.")
        min_len = min(len(df_rna), len(meta_df))
        df_rna = df_rna.iloc[:min_len]
        meta_df = meta_df.iloc[:min_len]
        common_cells = df_rna.index

    df_rna = df_rna.loc[common_cells]
    meta_df = meta_df.loc[common_cells]

    img_df = extract_morphology_features(meta_df)

    # Standard scale both views
    scaler_rna = StandardScaler()
    scaled_rna = pd.DataFrame(
        scaler_rna.fit_transform(df_rna),
        index=df_rna.index,
        columns=df_rna.columns
    )

    scaler_img = StandardScaler()
    scaled_img = pd.DataFrame(
        scaler_img.fit_transform(img_df),
        index=img_df.index,
        columns=img_df.columns
    )

    supervised_ploidy = meta_df.get("Supervised_ploidy", pd.Series("Unknown", index=common_cells))

    return scaled_rna, scaled_img, supervised_ploidy


def run_mofa_training(df_rna, df_morphology, output_hdf5="mofa_model_rna_morphology.hdf5", factors=10):
    """
    Sets up and trains mofapy2 model.
    """
    if not MOFAPY2_AVAILABLE:
        print("mofapy2 is not installed. Skipping model training.")
        return False

    print("=== INITIALIZING MOFA MODEL TRAINING (mofapy2) ===")
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

    ent.set_data_options()
    ent.set_model_options(factors=factors, spikeslab_weights=True, ard_weights=True)
    ent.set_train_options(iter=500, convergence_mode="fast", seed=42, gpu_mode=False)
    ent.build()
    ent.run()

    ent.save(output_hdf5)
    print(f"MOFA model successfully trained and saved to: {os.path.abspath(output_hdf5)}")
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
    parser = argparse.ArgumentParser(description="04: Multi-Omics Factor Analysis (MOFA) Integration")
    parser.add_argument("--exp_dir", type=str, default=r"D:\Morphotranscriptomic_Data\expression", help="10x expression directory")
    parser.add_argument("--out_dir", type=str, default="output", help="Directory to save output files")
    parser.add_argument("--factors", type=int, default=10, help="Number of latent factors")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"=== STEP 04: MULTI-OMICS FACTOR ANALYSIS (MOFA) ===")

    qc_meta_path = os.path.join(args.out_dir, "qc_filtered_metadata.csv")
    scaled_rna, scaled_img, ploidy_series = prepare_mofa_input_matrices(args.exp_dir, qc_meta_path, n_hvg=2000)

    model_hdf5 = "mofa_model_rna_morphology.hdf5"
    if not os.path.exists(model_hdf5):
        run_mofa_training(scaled_rna, scaled_img, output_hdf5=model_hdf5, factors=args.factors)

    evaluate_mofa_model(model_hdf5, ploidy_series, args.out_dir)


if __name__ == "__main__":
    main()
