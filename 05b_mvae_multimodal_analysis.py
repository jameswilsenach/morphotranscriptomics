"""
05b_mvae_multimodal_analysis.py
-------------------------------
Implements a PyTorch Multimodal Variational Autoencoder (MVAE) for joint representation learning,
cross-modality reconstruction, and latent space evaluation across single-cell RNA transcriptomics
and imaging morphology features.

MVAE Architecture:
1. Encoders: Modality-specific encoders (RNA & Morphology) mapping input to latent Gaussian parameters (mean, logvar).
2. Product-of-Experts (PoE) / Latent Fusion: Fuses modality distributions into a joint latent space z.
3. Decoders: Modality-specific decoders reconstructing original RNA expression and Imaging features.
4. Loss Function: Joint ELBO = RNA Recon Loss + Morphology Recon Loss + beta * KL Divergence.

Generated MVAE Figures:
- 05b_mvae_rna_reconstruction_scatter.png (Original vs Reconstructed RNA expression & R^2 score)
- 05b_mvae_morphology_reconstruction_scatter.png (Original vs Reconstructed Morphology parameters & R^2 score)
- 05b_mvae_latent_space_umap.png (MVAE Latent Space z colored by Supervised Ploidy)
- 05b_mvae_training_loss_curve.png (ELBO, Reconstruction, and KL Divergence loss curves)

Usage:
    python 05b_mvae_multimodal_analysis.py --exp_dir "data/expression" --out_dir "output" --epochs 100 --latent_dim 10
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import umap


def setup_plot_style():
    """Configures high-DPI matplotlib publication rendering matching morphotrans2 notebook."""
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.autolayout'] = True
    sns.set_theme(style="whitegrid")


# ==============================================================================
# PYTORCH MVAE MODEL DEFINITION
# ==============================================================================

class Encoder(nn.Module):
    """Modality-specific encoder neural network."""
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        return mean, logvar


class Decoder(nn.Module):
    """Modality-specific decoder neural network."""
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        return self.fc2(h)


class MultimodalVAE(nn.Module):
    """
    Multimodal Variational Autoencoder (MVAE) with Product-of-Experts latent fusion.
    """
    def __init__(self, rna_dim, img_dim, hidden_dim=128, latent_dim=10):
        super(MultimodalVAE, self).__init__()
        self.rna_encoder = Encoder(rna_dim, hidden_dim, latent_dim)
        self.img_encoder = Encoder(img_dim, hidden_dim, latent_dim)

        self.rna_decoder = Decoder(latent_dim, hidden_dim, rna_dim)
        self.img_decoder = Decoder(latent_dim, hidden_dim, img_dim)

        self.latent_dim = latent_dim

    def product_of_experts(self, means, logvars):
        """Fuses independent Gaussian distributions via Product-of-Experts (PoE)."""
        precisions = [torch.exp(-lv) for lv in logvars]
        combined_precision = sum(precisions)
        combined_var = 1.0 / combined_precision
        combined_logvar = torch.log(combined_var)

        combined_mean = sum(m * p for m, p in zip(means, precisions)) * combined_var
        return combined_mean, combined_logvar

    def reparameterize(self, mean, logvar):
        """Reparameterization trick to sample z ~ N(mean, var)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x_rna, x_img):
        mean_rna, logvar_rna = self.rna_encoder(x_rna)
        mean_img, logvar_img = self.img_encoder(x_img)

        # Fuse modal representations
        mean_joint, logvar_joint = self.product_of_experts(
            [mean_rna, mean_img],
            [logvar_rna, logvar_img]
        )

        z = self.reparameterize(mean_joint, logvar_joint)

        recon_rna = self.rna_decoder(z)
        recon_img = self.img_decoder(z)

        return recon_rna, recon_img, mean_joint, logvar_joint


def loss_function_mvae(recon_rna, x_rna, recon_img, x_img, mean, logvar, beta=1.0):
    """
    MVAE Joint Loss = MSE(RNA) + MSE(Morphology) + beta * KLD
    """
    loss_rna = F.mse_loss(recon_rna, x_rna, reduction='sum')
    loss_img = F.mse_loss(recon_img, x_img, reduction='sum')
    
    # KL divergence: -0.5 * sum(1 + logvar - mean^2 - exp(logvar))
    kld = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    total_loss = loss_rna + loss_img + beta * kld
    return total_loss, loss_rna, loss_img, kld


# ==============================================================================
# DATA INGESTION & TRAINING PIPELINE
# ==============================================================================

def extract_morphology_features(meta_df):
    """Extracts imaging morphology parameters via regex matching morphotrans2."""
    pattern = r'.*\(Imaging\).*|.*Axis.*|.*Mass.*|.*Intensity.*|.*Moment.*|.*Size.*|.*Diffusivity.*|.*Eccentricity.*|FSC-|SSC-|CD41-|DNA-'
    img_cols = [c for c in meta_df.columns if re.search(pattern, c)]

    numeric_img_cols = [c for c in img_cols if pd.api.types.is_numeric_dtype(meta_df[c])]
    img_df = meta_df[numeric_img_cols].dropna(axis=1, how="all").fillna(0)
    return img_df


def load_mvae_data(
    exp_dir,
    qc_meta_path,
    mofa_hdf5_path=None,
    n_hvg=2000,
    hvg_method="default",
    min_cells_per_gene=0,
    depth_norm="cp10k_log1p",
    adjust_library_depth=False,
    regress_cell_cycle="none",
    exclude_mt=False,
    exclude_cell_cycle=False,
    exclude_genes_file=None,
    exclude_genes=None,
    out_dir=None,
):
    """
    Ingests, aligns, and standardizes RNA and Morphology matrices across all available
    QC-passed cells (e.g. all 393 cells across M1-M7 by combining 10x matrices and
    pre-saved MOFA HDF5 expression when needed).
    """
    from importlib import import_module
    corr_mod = import_module("04_morphology_gene_correlations")

    # QC Metadata
    if qc_meta_path and os.path.exists(qc_meta_path):
        meta_df = pd.read_csv(qc_meta_path, index_col=0)
        if "SampleID" not in meta_df.columns:
            meta_df["SampleID"] = meta_df.index.astype(str)
        meta_df.index = meta_df.index.astype(str)
    else:
        meta_df = None
        for local_meta in [os.path.join("data", "meta.v1+flowSF.csv"), "meta.v1+flowSF.csv"]:
            if os.path.exists(local_meta):
                meta_df = pd.read_csv(local_meta, index_col=0)
                meta_df.index = meta_df.index.astype(str)
                break
        if meta_df is None:
            raise FileNotFoundError("Metadata table not found for MVAE training.")

    # Load barcode-aligned expression across all available cohorts (10x + MOFA HDF5 fallback)
    # Canonical order: Depth Normalization -> Cell-Cycle Regression -> HVG Selection
    df_rna, cohort_sources = corr_mod.load_expression_backwards_compatible(
        meta_df,
        exp_dir=exp_dir,
        pilot_exp_dir=os.path.join("data", "exp"),
        mofa_hdf5_path=mofa_hdf5_path,
        n_hvg=n_hvg,
        hvg_method=hvg_method,
        min_cells_per_gene=min_cells_per_gene,
        depth_norm=depth_norm,
        regress_cell_cycle=regress_cell_cycle,
        exclude_mt=exclude_mt,
        out_dir=out_dir,
        step_prefix="05b_mvae",
    )

    # Optional cited cell-cycle and/or custom gene exclusion (after CC scoring/regression)
    if exclude_cell_cycle or exclude_genes_file or exclude_genes:
        from cell_cycle import filter_excluded_genes
        audit_path = os.path.join(out_dir, "05b_mvae_excluded_genes_audit.csv") if out_dir else None
        cc_attrs = df_rna.attrs.get("cc_scores_df", None)
        keep_mask, _ = filter_excluded_genes(
            df_rna.columns,
            exclude_cell_cycle=exclude_cell_cycle,
            exclude_genes_file=exclude_genes_file,
            exclude_genes=exclude_genes,
            audit_csv_path=audit_path,
        )
        df_rna = df_rna.loc[:, keep_mask].copy()
        if cc_attrs is not None:
            df_rna.attrs["cc_scores_df"] = cc_attrs
    else:
        print("[Gene Filter] MVAE RNA View: No additional gene exclusions applied (baseline).")

    common_cells = meta_df.index.intersection(df_rna.index)
    if len(common_cells) == 0:
        raise ValueError("No overlapping cell barcodes found between QC metadata and expression matrices.")

    cc_attrs = df_rna.attrs.get("cc_scores_df", None)
    df_rna = df_rna.loc[common_cells].copy()
    if cc_attrs is not None:
        df_rna.attrs["cc_scores_df"] = cc_attrs.reindex(common_cells)
    meta_df = meta_df.loc[common_cells].copy()
    cohort_sources = cohort_sources.loc[common_cells].copy()

    img_df = corr_mod.extract_morphology_features(meta_df)

    # Cohort-level z-score harmonization of morphology and optional library-depth / CDR residualization
    df_rna_h, img_df_h = corr_mod.harmonize_and_adjust_modalities(
        df_rna,
        img_df,
        cohort_sources,
        adjust_library_depth=adjust_library_depth,
        regress_cell_cycle=regress_cell_cycle,
    )
    if adjust_library_depth:
        print("[MVAE] Applied per-cohort library-depth / CDR residualization to RNA view (Finak et al., 2015).")

    scaler_rna = StandardScaler()
    scaled_rna = scaler_rna.fit_transform(df_rna_h)

    scaler_img = StandardScaler()
    scaled_img = scaler_img.fit_transform(img_df_h)

    ploidy_series = meta_df.get("Supervised_ploidy", pd.Series("Unknown", index=common_cells))
    n_mice = meta_df["mouse"].nunique() if "mouse" in meta_df.columns else 0
    print(
        f"[MVAE Data] Loaded {len(common_cells)} aligned QC-passed cells"
        + (f" across {n_mice} animals" if n_mice else "")
        + f" | {df_rna_h.shape[1]} genes | {img_df_h.shape[1]} morphology features."
    )

    return scaled_rna, scaled_img, ploidy_series, df_rna_h.columns, img_df_h.columns, common_cells


def train_mvae(model, dataset, epochs=100, lr=1e-3, batch_size=32):
    """Trains PyTorch MVAE model and tracks loss history."""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"total_loss": [], "loss_rna": [], "loss_img": [], "kld": []}

    model.train()
    print(f"=== TRAINING PYTORCH MVAE MODEL ({epochs} Epochs) ===")
    for epoch in range(1, epochs + 1):
        epoch_total, epoch_rna, epoch_img, epoch_kld = 0, 0, 0, 0
        for batch_rna, batch_img in dataloader:
            optimizer.zero_grad()
            recon_rna, recon_img, mean, logvar = model(batch_rna, batch_img)

            total_loss, loss_rna, loss_img, kld = loss_function_mvae(
                recon_rna, batch_rna, recon_img, batch_img, mean, logvar
            )

            total_loss.backward()
            optimizer.step()

            epoch_total += total_loss.item()
            epoch_rna += loss_rna.item()
            epoch_img += loss_img.item()
            epoch_kld += kld.item()

        n_samples = len(dataset)
        history["total_loss"].append(epoch_total / n_samples)
        history["loss_rna"].append(epoch_rna / n_samples)
        history["loss_img"].append(epoch_img / n_samples)
        history["kld"].append(epoch_kld / n_samples)

        if epoch % 20 == 0 or epoch == epochs:
            print(f"Epoch {epoch:3d}/{epochs} | Total Loss: {history['total_loss'][-1]:.2f} | RNA Loss: {history['loss_rna'][-1]:.2f} | Img Loss: {history['loss_img'][-1]:.2f} | KLD: {history['kld'][-1]:.2f}")

    return history


# ==============================================================================
# RECONSTRUCTION & MVAE FIGURE GENERATION
# ==============================================================================

def generate_mvae_figures(model, x_rna_tensor, x_img_tensor, ploidy_series, cell_ids, history, out_dir):
    """Reconstructs all MVAE evaluation figures."""
    os.makedirs(out_dir, exist_ok=True)
    setup_plot_style()

    model.eval()
    with torch.no_grad():
        recon_rna, recon_img, mean_joint, logvar_joint = model(x_rna_tensor, x_img_tensor)

    orig_rna = x_rna_tensor.numpy()
    pred_rna = recon_rna.numpy()

    orig_img = x_img_tensor.numpy()
    pred_img = recon_img.numpy()
    latent_z = mean_joint.numpy()

    # 1. Figure 1: Training Loss Curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["total_loss"], label="Total ELBO Loss", color="black", linewidth=2)
    ax.plot(history["loss_rna"], label="RNA Recon Loss", color="#1f77b4", linestyle="--")
    ax.plot(history["loss_img"], label="Morphology Recon Loss", color="#ff7f0e", linestyle="--")
    ax.plot(history["kld"], label="KL Divergence", color="#2ca02c", linestyle=":")
    ax.set_title("MVAE Training Loss & Convergence History", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.tight_layout()
    fig1_path = os.path.join(out_dir, "05b_mvae_training_loss_curve.png")
    plt.savefig(fig1_path)
    plt.close()
    print(f"Saved MVAE Loss Curve figure to: {os.path.abspath(fig1_path)}")

    # 2. Figure 2: RNA Reconstruction Scatter & R2 Score
    r2_rna = r2_score(orig_rna.ravel(), pred_rna.ravel())
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(orig_rna.ravel(), pred_rna.ravel(), alpha=0.3, color="#1f77b4", s=10)
    ax.plot([orig_rna.min(), orig_rna.max()], [orig_rna.min(), orig_rna.max()], 'r--', label="Perfect Reconstruction")
    ax.set_title(f"MVAE RNA Expression Reconstruction (R^2 = {r2_rna:.4f})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Original Scaled Expression")
    ax.set_ylabel("Reconstructed Expression")
    ax.legend()
    plt.tight_layout()
    fig2_path = os.path.join(out_dir, "05b_mvae_rna_reconstruction_scatter.png")
    plt.savefig(fig2_path)
    plt.close()
    print(f"Saved RNA Reconstruction Scatter figure (R^2 = {r2_rna:.4f}) to: {os.path.abspath(fig2_path)}")

    # 3. Figure 3: Morphology Reconstruction Scatter & R2 Score
    r2_img = r2_score(orig_img.ravel(), pred_img.ravel())
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(orig_img.ravel(), pred_img.ravel(), alpha=0.4, color="#ff7f0e", s=15)
    ax.plot([orig_img.min(), orig_img.max()], [orig_img.min(), orig_img.max()], 'r--', label="Perfect Reconstruction")
    ax.set_title(f"MVAE Morphology Reconstruction (R^2 = {r2_img:.4f})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Original Scaled Morphology")
    ax.set_ylabel("Reconstructed Morphology")
    ax.legend()
    plt.tight_layout()
    fig3_path = os.path.join(out_dir, "05b_mvae_morphology_reconstruction_scatter.png")
    plt.savefig(fig3_path)
    plt.close()
    print(f"Saved Morphology Reconstruction Scatter figure (R^2 = {r2_img:.4f}) to: {os.path.abspath(fig3_path)}")

    # 4. Figure 4: MVAE Latent Space z UMAP Projection by Ploidy
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    z_umap = reducer.fit_transform(latent_z)

    latent_df = pd.DataFrame({
        "UMAP1": z_umap[:, 0],
        "UMAP2": z_umap[:, 1],
        "Supervised_ploidy": ploidy_series.reindex(cell_ids).fillna("Unknown").values
    }, index=cell_ids)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=latent_df,
        x="UMAP1",
        y="UMAP2",
        hue="Supervised_ploidy",
        palette="tab10",
        s=60,
        ax=ax
    )
    ax.set_title("MVAE Joint Latent Space Projection (Product-of-Experts)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Latent UMAP 1")
    ax.set_ylabel("Latent UMAP 2")
    plt.tight_layout()
    fig4_path = os.path.join(out_dir, "05b_mvae_latent_space_umap.png")
    plt.savefig(fig4_path)
    plt.close()
    print(f"Saved MVAE Latent Space UMAP figure to: {os.path.abspath(fig4_path)}")

    # Export Latent Coordinates CSV
    latent_csv = os.path.join(out_dir, "mvae_latent_coordinates.csv")
    latent_df.to_csv(latent_csv)
    print(f"Exported MVAE latent space coordinates to: {os.path.abspath(latent_csv)}")


def main():
    parser = argparse.ArgumentParser(description="05b: PyTorch Multimodal Variational Autoencoder (MVAE) Training & Reconstruction")
    parser.add_argument("--exp_dir", type=str, default=os.path.join("data", "expression"), help="10x expression directory (falls back to data/exp)")
    parser.add_argument("--mofa_hdf5", type=str, default=None, help="Optional pre-saved MOFA HDF5 path for M1-M5 expression alignment (auto-discovered in data/ if omitted)")
    parser.add_argument("--out_dir", type=str, default="output", help="Directory to save output files")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--latent_dim", type=int, default=10, help="Dimension of latent space z")
    parser.add_argument(
        "--n_hvg",
        type=int,
        default=2000,
        help="Number of highly variable genes to select for the RNA view (default: 2000)"
    )
    parser.add_argument(
        "--hvg_method",
        type=str,
        default="default",
        choices=["default", "raw_var", "log1p_var", "seurat", "seurat_v3"],
        help="HVG selection method ('default' uses raw_var for Step 05b)"
    )
    parser.add_argument(
        "--min_cells_per_gene",
        type=int,
        default=0,
        help="Minimum number of cells expressing a gene (> 0 counts) required before HVG selection (default: 0, disabled)"
    )
    parser.add_argument(
        "--depth_norm",
        type=str,
        default="cp10k_log1p",
        choices=["cp10k_log1p", "none"],
        help="Library-size depth normalization before cell-cycle regression & HVG selection (default: cp10k_log1p)"
    )
    parser.add_argument(
        "--adjust_library_depth",
        action="store_true",
        help="Apply per-cohort library-depth / Cellular Detection Rate (CDR) residualization to RNA view (Finak et al., 2015)"
    )
    parser.add_argument(
        "--regress_cell_cycle",
        type=str,
        default="none",
        choices=["none", "full", "difference"],
        help="Cell-cycle regression after depth normalization: 'none' (default), 'full' (S + G2M), or 'difference' (S - G2M)"
    )
    parser.add_argument(
        "--exclude_mt",
        action="store_true",
        help="Exclude mitochondrial genes (starting with 'mt-' or 'MT-'); off by default"
    )
    parser.add_argument(
        "--exclude_cell_cycle",
        action="store_true",
        help="Exclude cited Tirosh et al. (2016) / Seurat cc.genes.updated.2019 cell cycle genes before MVAE HVG selection"
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
    print(f"=== STEP 05b: PYTORCH MULTIMODAL VAE (MVAE) TRAINING & RECONSTRUCTION ===")

    qc_meta_path = os.path.join(args.out_dir, "qc_filtered_metadata.csv")
    if not os.path.exists(qc_meta_path):
        raise FileNotFoundError(
            f"Missing prerequisite output from Step 1 (QC): '{qc_meta_path}'.\n"
            f"Please run Step 1 first:\n"
            f"  python run_master_analysis.py --step qc --out_dir \"{args.out_dir}\""
        )
    scaled_rna, scaled_img, ploidy_series, rna_cols, img_cols, cell_ids = load_mvae_data(
        args.exp_dir,
        qc_meta_path,
        mofa_hdf5_path=args.mofa_hdf5,
        n_hvg=args.n_hvg,
        hvg_method=args.hvg_method,
        min_cells_per_gene=args.min_cells_per_gene,
        depth_norm=args.depth_norm,
        adjust_library_depth=args.adjust_library_depth,
        regress_cell_cycle=args.regress_cell_cycle,
        exclude_mt=args.exclude_mt,
        exclude_cell_cycle=args.exclude_cell_cycle,
        exclude_genes_file=args.exclude_genes_file,
        exclude_genes=args.exclude_genes,
        out_dir=args.out_dir,
    )

    x_rna_tensor = torch.tensor(scaled_rna, dtype=torch.float32)
    x_img_tensor = torch.tensor(scaled_img, dtype=torch.float32)

    dataset = TensorDataset(x_rna_tensor, x_img_tensor)

    # Initialize MVAE
    model = MultimodalVAE(
        rna_dim=scaled_rna.shape[1],
        img_dim=scaled_img.shape[1],
        hidden_dim=128,
        latent_dim=args.latent_dim
    )

    # Train MVAE
    history = train_mvae(model, dataset, epochs=args.epochs, lr=1e-3, batch_size=32)

    # Save model weights
    model_pth = os.path.join(args.out_dir, "mvae_model_weights.pt")
    torch.save(model.state_dict(), model_pth)
    print(f"Saved PyTorch MVAE model weights to: {os.path.abspath(model_pth)}")

    # Reconstruct all MVAE figures
    generate_mvae_figures(model, x_rna_tensor, x_img_tensor, ploidy_series, cell_ids, history, args.out_dir)


if __name__ == "__main__":
    main()
