"""
04b_mvae_multimodal_analysis.py
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
- 04b_mvae_rna_reconstruction_scatter.png (Original vs Reconstructed RNA expression & R^2 score)
- 04b_mvae_morphology_reconstruction_scatter.png (Original vs Reconstructed Morphology parameters & R^2 score)
- 04b_mvae_latent_space_umap.png (MVAE Latent Space z colored by Supervised Ploidy)
- 04b_mvae_training_loss_curve.png (ELBO, Reconstruction, and KL Divergence loss curves)

Usage:
    python 04b_mvae_multimodal_analysis.py --exp_dir "D:\\Morphotranscriptomic_Data\\expression" --out_dir "output" --epochs 100 --latent_dim 10
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
    n_hvg=2000,
    hvg_method="default",
    min_cells_per_gene=0,
    exclude_cell_cycle=False,
    exclude_genes_file=None,
    exclude_genes=None,
    out_dir=None,
):
    """Ingests and standardizes RNA and Morphology matrices for MVAE training."""
    import scanpy as sc
    from hvg_utils import resolve_hvg_method, select_hvgs

    if not os.path.exists(exp_dir) and os.path.exists("exp"):
        exp_dir = "exp"
    elif not os.path.exists(exp_dir) and os.path.exists(r"C:\Users\vilsn\Dropbox\Work\Turing\Exploration_Data\exp"):
        exp_dir = r"C:\Users\vilsn\Dropbox\Work\Turing\Exploration_Data\exp"

    adata = sc.read_10x_mtx(exp_dir)
    adata.var_names_make_unique()

    # Filter genes
    is_ribo = adata.var_names.str.startswith(('Rps', 'Rpl'))
    is_unnamed = adata.var_names.str.startswith('Unnamed')
    adata = adata[:, ~(is_ribo | is_unnamed)].copy()

    # Optional cited cell-cycle and/or custom gene exclusion
    if exclude_cell_cycle or exclude_genes_file or exclude_genes:
        from cell_cycle import filter_excluded_genes
        audit_path = os.path.join(out_dir, "04b_mvae_excluded_genes_audit.csv") if out_dir else None
        keep_mask, _ = filter_excluded_genes(
            adata.var_names,
            exclude_cell_cycle=exclude_cell_cycle,
            exclude_genes_file=exclude_genes_file,
            exclude_genes=exclude_genes,
            audit_csv_path=audit_path,
        )
        adata = adata[:, keep_mask].copy()
    else:
        print("[Gene Filter] MVAE RNA View: No additional gene exclusions applied (baseline).")

    df_raw = adata.to_df()

    # QC Metadata
    if qc_meta_path and os.path.exists(qc_meta_path):
        meta_df = pd.read_csv(qc_meta_path, index_col=0)
    else:
        local_meta = "meta.v1+flowSF.csv"
        if os.path.exists(local_meta):
            meta_df = pd.read_csv(local_meta, index_col=0)
        else:
            raise FileNotFoundError("Metadata table not found for MVAE training.")

    common_cells = df_raw.index.intersection(meta_df.index)
    if len(common_cells) == 0:
        min_len = min(len(df_raw), len(meta_df))
        df_raw = df_raw.iloc[:min_len]
        meta_df = meta_df.iloc[:min_len]
        common_cells = df_raw.index

    df_raw = df_raw.loc[common_cells]
    meta_df = meta_df.loc[common_cells]

    # HVG selection (default: raw_var) with optional min_cells_per_gene filter
    method = resolve_hvg_method(hvg_method, default_method="raw_var")
    top_genes = select_hvgs(
        df_raw,
        n_hvg=n_hvg,
        hvg_method=method,
        min_cells_per_gene=min_cells_per_gene,
    )
    df_rna = df_raw[top_genes]

    img_df = extract_morphology_features(meta_df)

    scaler_rna = StandardScaler()
    scaled_rna = scaler_rna.fit_transform(df_rna)

    scaler_img = StandardScaler()
    scaled_img = scaler_img.fit_transform(img_df)

    ploidy_series = meta_df.get("Supervised_ploidy", pd.Series("Unknown", index=common_cells))

    return scaled_rna, scaled_img, ploidy_series, df_rna.columns, img_df.columns, common_cells


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
    fig1_path = os.path.join(out_dir, "04b_mvae_training_loss_curve.png")
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
    fig2_path = os.path.join(out_dir, "04b_mvae_rna_reconstruction_scatter.png")
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
    fig3_path = os.path.join(out_dir, "04b_mvae_morphology_reconstruction_scatter.png")
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
    fig4_path = os.path.join(out_dir, "04b_mvae_latent_space_umap.png")
    plt.savefig(fig4_path)
    plt.close()
    print(f"Saved MVAE Latent Space UMAP figure to: {os.path.abspath(fig4_path)}")

    # Export Latent Coordinates CSV
    latent_csv = os.path.join(out_dir, "mvae_latent_coordinates.csv")
    latent_df.to_csv(latent_csv)
    print(f"Exported MVAE latent space coordinates to: {os.path.abspath(latent_csv)}")


def main():
    parser = argparse.ArgumentParser(description="04b: PyTorch Multimodal Variational Autoencoder (MVAE) Training & Reconstruction")
    parser.add_argument("--exp_dir", type=str, default=r"D:\Morphotranscriptomic_Data\expression", help="10x expression directory")
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
        help="HVG selection method ('default' uses raw_var for Step 04b)"
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
    print(f"=== STEP 04b: PYTORCH MULTIMODAL VAE (MVAE) TRAINING & RECONSTRUCTION ===")

    qc_meta_path = os.path.join(args.out_dir, "qc_filtered_metadata.csv")
    scaled_rna, scaled_img, ploidy_series, rna_cols, img_cols, cell_ids = load_mvae_data(
        args.exp_dir,
        qc_meta_path,
        n_hvg=args.n_hvg,
        hvg_method=args.hvg_method,
        min_cells_per_gene=args.min_cells_per_gene,
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
