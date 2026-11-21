"""
03_dimensionality_reduction.py
--------------------------------
Performs single-cell transcriptomics preprocessing, Highly Variable Gene (HVG) selection,
PCA, and UMAP manifold projections using Scanpy and AnnData based on morphotrans2 methods.

Features:
1. Ingests 10x expression matrix and filters mitochondrial, ribosomal, and unnamed genes.
2. Filters cells to QC-passed cohort from Step 01.
3. HVG selection (top 2000 most variable genes) & StandardScaler scaling.
4. PCA calculation (retaining 90%+ variance).
5. UMAP manifold projection.
6. Publication-ready UMAP figures colored by Supervised Ploidy, Train vs Pilot cohort, and marker genes (kalrn, pf4).

Usage:
    python 03_dimensionality_reduction.py --exp_dir "D:\\Morphotranscriptomic_Data\\expression" --out_dir "output"
"""

import argparse
import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import seaborn as sns


def setup_plot_style():
    """Configures high-DPI matplotlib publication rendering matching morphotrans2 notebook."""
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.autolayout'] = True
    sns.set_theme(style="whitegrid")


def load_and_preprocess_anndata(
    exp_dir,
    qc_meta_path=None,
    exclude_cell_cycle=False,
    exclude_genes_file=None,
    exclude_genes=None,
    out_dir=None,
):
    """
    Loads 10x expression matrix into Scanpy AnnData, applies gene filters
    (ribosomal/Unnamed plus optional cited cell-cycle or custom gene exclusion),
    and merges QC-passed metadata.
    """
    if not os.path.exists(exp_dir) and os.path.exists("exp"):
        exp_dir = "exp"
    elif not os.path.exists(exp_dir) and os.path.exists(r"C:\Users\vilsn\Dropbox\Work\Turing\Exploration_Data\exp"):
        exp_dir = r"C:\Users\vilsn\Dropbox\Work\Turing\Exploration_Data\exp"

    print(f"Loading AnnData from: {os.path.abspath(exp_dir)}")
    adata = sc.read_10x_mtx(exp_dir)
    adata.var_names_make_unique()

    # Gene filtering matching morphotrans2 rule
    is_ribo = adata.var_names.str.startswith(('Rps', 'Rpl'))
    is_unnamed = adata.var_names.str.startswith('Unnamed')
    genes_to_keep = ~(is_ribo | is_unnamed)
    adata = adata[:, genes_to_keep].copy()

    # Optional cited cell-cycle and/or custom gene exclusion
    if exclude_cell_cycle or exclude_genes_file or exclude_genes:
        from cell_cycle import filter_excluded_genes
        audit_path = os.path.join(out_dir, "03_excluded_genes_audit.csv") if out_dir else None
        keep_mask, _ = filter_excluded_genes(
            adata.var_names,
            exclude_cell_cycle=exclude_cell_cycle,
            exclude_genes_file=exclude_genes_file,
            exclude_genes=exclude_genes,
            audit_csv_path=audit_path,
        )
        adata = adata[:, keep_mask].copy()
    else:
        print("[Gene Filter] No additional gene exclusions applied (original notebook baseline).")

    # Merge QC metadata if available
    if qc_meta_path and os.path.exists(qc_meta_path):
        qc_meta = pd.read_csv(qc_meta_path, index_col=0)
        common_cells = adata.obs_names.intersection(qc_meta.index)
        if len(common_cells) > 0:
            adata = adata[common_cells].copy()
            for col in qc_meta.columns:
                adata.obs[col] = qc_meta.loc[common_cells, col]
            print(f"Filtered AnnData to {len(common_cells)} QC-passed cells.")
        else:
            print("Note: No barcode overlap with QC metadata table. Using all cells in matrix.")

    if "Supervised_ploidy" not in adata.obs.columns:
        adata.obs["Supervised_ploidy"] = "Unknown"

    return adata


def compute_hvg_pca_umap(adata, n_top_genes=2000, hvg_method="default", min_cells_per_gene=0):
    """
    Filters genes by optional minimum cell prevalence, selects top HVGs, scales data,
    and calculates PCA and UMAP embeddings.
    """
    from hvg_utils import resolve_hvg_method, select_hvgs

    df_raw = adata.to_df()
    method = resolve_hvg_method(hvg_method, default_method="raw_var")
    top_hvg = select_hvgs(
        df_raw,
        n_hvg=n_top_genes,
        hvg_method=method,
        min_cells_per_gene=min_cells_per_gene,
    )
    df_hvg = df_raw[top_hvg]

    # Standard scale
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_hvg)

    # PCA
    pca = PCA(n_components=min(50, df_scaled.shape[0], df_scaled.shape[1]))
    pca_embedding = pca.fit_transform(df_scaled)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    idx_90 = np.searchsorted(cum_var, 0.90)
    n_pcs_90 = min(int(idx_90) + 1, len(cum_var))
    print(f"Filtered data shape ({len(top_hvg)} HVGs, method='{method}'): {df_hvg.shape}")
    print(f"Number of PCs retaining 90%+ variance: {n_pcs_90}")
    print(f"Cumulative explained variance (first {n_pcs_90} PCs): {cum_var[n_pcs_90-1]:.4f}")

    # UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    umap_embedding = reducer.fit_transform(pca_embedding[:, :n_pcs_90])

    adata.obsm["X_pca"] = pca_embedding
    adata.obsm["X_umap"] = umap_embedding
    adata.obs["UMAP1"] = umap_embedding[:, 0]
    adata.obs["UMAP2"] = umap_embedding[:, 1]

    return adata


def plot_umap_projections(adata, out_dir, marker_genes=["kalrn", "pf4"]):
    """Generates publication-quality UMAP plots."""
    os.makedirs(out_dir, exist_ok=True)
    setup_plot_style()

    # Plot 1: UMAP by Supervised Ploidy
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=adata.obs,
        x="UMAP1",
        y="UMAP2",
        hue="Supervised_ploidy",
        palette="tab10",
        s=60,
        alpha=0.9,
        ax=ax
    )
    ax.set_title("UMAP Manifold Projection: Supervised Ploidy", fontsize=13, fontweight="bold")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.tight_layout()
    fig1_path = os.path.join(out_dir, "03_umap_supervised_ploidy.png")
    plt.savefig(fig1_path)
    plt.close()

    # Plot 2: UMAP by Dataset Type (Train vs Pilot/Test)
    if "dataset_type" in adata.obs.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=adata.obs,
            x="UMAP1",
            y="UMAP2",
            hue="dataset_type",
            palette="Set1",
            s=60,
            alpha=0.9,
            ax=ax
        )
        ax.set_title("UMAP Manifold Projection: Dataset Cohort", fontsize=13, fontweight="bold")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        plt.tight_layout()
        fig2_path = os.path.join(out_dir, "03_umap_dataset_cohort.png")
        plt.savefig(fig2_path)
        plt.close()

    # Plot 3: UMAP by Marker Gene Expression
    df_raw = adata.to_df()
    for gene in marker_genes:
        match_symbol = [c for c in df_raw.columns if c.lower() == gene.lower()]
        if match_symbol:
            target_gene = match_symbol[0]
            fig, ax = plt.subplots(figsize=(8, 6))
            expr_val = df_raw[target_gene]
            scatter = ax.scatter(
                adata.obs["UMAP1"],
                adata.obs["UMAP2"],
                c=expr_val,
                cmap="viridis",
                s=60,
                alpha=0.9
            )
            fig.colorbar(scatter, ax=ax, label=f"{target_gene} Expression")
            ax.set_title(f"UMAP Expression Overlay: '{target_gene}'", fontsize=13, fontweight="bold")
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            plt.tight_layout()
            gene_png = os.path.join(out_dir, f"03_umap_gene_{gene.lower()}.png")
            plt.savefig(gene_png)
            plt.close()

    print(f"[Dimensionality Reduction] Saved UMAP figures to: {os.path.abspath(out_dir)}")


def main():
    parser = argparse.ArgumentParser(description="03: Single Cell Transcriptomics PCA & UMAP Dimensionality Reduction")
    parser.add_argument("--exp_dir", type=str, default=r"D:\Morphotranscriptomic_Data\expression", help="10x expression directory")
    parser.add_argument("--out_dir", type=str, default="output", help="Directory to save output files")
    parser.add_argument(
        "--n_hvg",
        type=int,
        default=2000,
        help="Number of highly variable genes to select (default: 2000)"
    )
    parser.add_argument(
        "--hvg_method",
        type=str,
        default="default",
        choices=["default", "raw_var", "log1p_var", "seurat", "seurat_v3"],
        help="HVG selection method ('default' uses raw_var for Step 03)"
    )
    parser.add_argument(
        "--min_cells_per_gene",
        type=int,
        default=0,
        help="Minimum number of cells expressing a gene (> 0 counts) required before HVG selection (default: 0, disabled)"
    )
    parser.add_argument(
        "--exclude_cell_cycle",
        action="store_true",
        help="Exclude cited Tirosh et al. (2016) / Seurat cc.genes.updated.2019 cell cycle genes before HVG selection"
    )
    parser.add_argument(
        "--exclude_genes_file",
        type=str,
        default=None,
        help="Path to a .txt/.csv/.tsv file of gene symbols to exclude (e.g., GO:0007049 or custom list)"
    )
    parser.add_argument(
        "--exclude_genes",
        type=str,
        default=None,
        help="Comma-separated inline list of gene symbols to exclude (e.g. 'Mki67,Top2a,Cdk1')"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"=== STEP 03: DIMENSIONALITY REDUCTION (PCA & UMAP) ===")

    qc_meta_path = os.path.join(args.out_dir, "qc_filtered_metadata.csv")
    adata = load_and_preprocess_anndata(
        args.exp_dir,
        qc_meta_path,
        exclude_cell_cycle=args.exclude_cell_cycle,
        exclude_genes_file=args.exclude_genes_file,
        exclude_genes=args.exclude_genes,
        out_dir=args.out_dir,
    )

    adata = compute_hvg_pca_umap(
        adata,
        n_top_genes=args.n_hvg,
        hvg_method=args.hvg_method,
        min_cells_per_gene=args.min_cells_per_gene,
    )

    # Save UMAP coordinates CSV
    umap_df = adata.obs[["Supervised_ploidy", "UMAP1", "UMAP2"]].copy()
    umap_csv = os.path.join(args.out_dir, "umap_coordinates.csv")
    umap_df.to_csv(umap_csv)
    print(f"Exported UMAP coordinates to: {os.path.abspath(umap_csv)}")

    # Plot UMAP projections
    plot_umap_projections(adata, args.out_dir, marker_genes=["kalrn", "pf4"])


if __name__ == "__main__":
    main()
