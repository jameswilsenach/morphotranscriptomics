"""
cell_cycle.py
-------------
Defines canonical mouse and human cell cycle gene sets (S-phase and G2/M-phase)
and utilities to filter or regress out cell cycle effects in single-cell transcriptomics.

References:
- Tirosh et al., 2016 (Science): Dissecting the multicellular ecosystem of metastatic melanoma.
- Macosko et al., 2015 (Cell): Highly Parallel Genome-wide Expression Profiling of Individual Cells.
- Scanpy & Seurat cell cycle scoring benchmarks.
"""

from typing import List, Tuple, Set, Optional, Union
import numpy as np
import pandas as pd


# Canonical mouse S-phase genes (Tirosh et al. / Macosko et al. adapted for Mus musculus)
MOUSE_S_GENES = [
    "Mcm5", "Pcna", "Tyms", "Fen1", "Mcm2", "Mcm4", "Rrm1", "Ung", "Gins2",
    "Cdc6", "Cdca7", "Dtl", "Prim1", "Uhrf1", "Cenpu", "Mlf1ip", "Hells",
    "Rfc2", "Rpa2", "Nasp", "Rad51ap1", "Gmnn", "Wdr76", "Slbp", "Ccne2",
    "Ubr7", "Pold3", "Msh2", "Atad2", "Rad51", "Rrm2", "Cdc45", "Cdc25a",
    "Pola1", "Chaf1b", "Brip1", "E2f8"
]

# Canonical mouse G2/M-phase genes
MOUSE_G2M_GENES = [
    "Hmgb2", "Cdk1", "Nusap1", "Ube2c", "Birc5", "Tpx2", "Top2a", "Ndc80",
    "Cks2", "Nuf2", "Cks1b", "Mki67", "Tmpo", "Cenpf", "Tacc3", "Fam64a",
    "Pimreg", "Smc4", "Ccnb2", "Ckap2l", "Ckap2", "Aurkb", "Bub1", "Kif11",
    "Anp32e", "Tubb4b", "Gtse1", "Kif20b", "Hjurp", "Cdca3", "Hn1", "Jpt1",
    "Cdc20", "Ttk", "Cdc25c", "Kif2c", "Rangap1", "Ncapd2", "Dlgap5", "Cdca2",
    "Cdca8", "Ect2", "Kif23", "Hmmr", "Aurka", "Psrc1", "Anln", "Lbr",
    "Ckap5", "Cenpe", "Ctcf", "Nek2", "G2e3", "Gas2l3", "Cbx5", "Cenpa"
]


def get_cell_cycle_genes(
    gene_names: Optional[Union[List[str], pd.Index]] = None
) -> Tuple[List[str], List[str], Set[str]]:
    """
    Returns (s_genes, g2m_genes, all_cc_genes_set).
    If gene_names is provided, matches gene symbols case-insensitively to match
    the casing used in the dataset (e.g. TitleCase, UPPERCASE, lowercase).
    """
    if gene_names is None:
        s_matched = list(MOUSE_S_GENES)
        g2m_matched = list(MOUSE_G2M_GENES)
        all_cc = set(s_matched + g2m_matched)
        return s_matched, g2m_matched, all_cc

    lookup = {str(g).upper(): str(g) for g in gene_names}
    s_matched = [lookup[g.upper()] for g in MOUSE_S_GENES if g.upper() in lookup]
    g2m_matched = [lookup[g.upper()] for g in MOUSE_G2M_GENES if g.upper() in lookup]
    all_cc = set(s_matched + g2m_matched)
    return s_matched, g2m_matched, all_cc


def filter_cell_cycle_genes(
    gene_names: Union[List[str], pd.Index]
) -> np.ndarray:
    """
    Returns a boolean mask of genes to keep (True for non-cell-cycle genes, False for cell-cycle genes).
    Case-insensitive matching.
    """
    s_genes, g2m_genes, all_cc = get_cell_cycle_genes(gene_names)
    mask = np.array([g not in all_cc for g in gene_names], dtype=bool)
    n_removed = len(gene_names) - int(np.sum(mask))
    print(f"[Cell Cycle] Filtered out {n_removed} cell cycle genes out of {len(gene_names)} total features.")
    return mask


def score_cell_cycle(adata):
    """
    Scores S-phase and G2/M-phase using Scanpy's score_genes_cell_cycle,
    matching gene symbols case-insensitively.
    Adds 'S_score', 'G2M_score', and 'phase' to adata.obs.
    """
    import scanpy as sc

    s_genes, g2m_genes, _ = get_cell_cycle_genes(adata.var_names)
    print(f"[Cell Cycle] Found {len(s_genes)} S-phase genes and {len(g2m_genes)} G2/M-phase genes in dataset.")

    if len(s_genes) == 0 or len(g2m_genes) == 0:
        print("[Cell Cycle] Warning: Insufficient cell cycle genes found to score phases accurately.")
        adata.obs["S_score"] = 0.0
        adata.obs["G2M_score"] = 0.0
        adata.obs["phase"] = "Unknown"
        return adata

    sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)
    print(f"[Cell Cycle] Phase distribution:\n{adata.obs['phase'].value_counts().to_string()}")
    return adata


def regress_cell_cycle_adata(adata):
    """
    Scores cell cycle and regresses out 'S_score' and 'G2M_score'.
    Ensures log1p is applied if data is in raw counts, then performs regression.
    """
    import scanpy as sc

    # Ensure log1p is applied if matrix is raw counts (max > 25)
    max_val = np.max(adata.X.data) if hasattr(adata.X, "data") else np.max(adata.X)
    if max_val > 25:
        print("[Cell Cycle] Applying scanpy.pp.log1p before cell cycle scoring...")
        sc.pp.log1p(adata)

    adata = score_cell_cycle(adata)
    if "S_score" in adata.obs.columns and "G2M_score" in adata.obs.columns:
        print("[Cell Cycle] Regressing out 'S_score' and 'G2M_score'...")
        try:
            sc.pp.regress_out(adata, ["S_score", "G2M_score"], n_jobs=1)
            print("[Cell Cycle] Successfully regressed out cell cycle effects via scanpy.pp.regress_out.")
        except Exception as e:
            print(f"[Cell Cycle] scanpy.pp.regress_out fallback triggered ({e}). Using vectorized OLS...")
            s_score = adata.obs["S_score"].values
            g2m_score = adata.obs["G2M_score"].values
            X_covar = np.column_stack([np.ones(len(adata)), s_score, g2m_score])
            Y = adata.to_df().values
            coef, _, _, _ = np.linalg.lstsq(X_covar, Y, rcond=None)
            residuals = Y - X_covar.dot(coef)
            adata.X = residuals
            print("[Cell Cycle] Successfully regressed out cell cycle effects via vectorized OLS.")
    return adata


def regress_cell_cycle_df(df_log_cpm: pd.DataFrame) -> pd.DataFrame:
    """
    Scores cell cycle using mean S-phase and G2/M-phase gene expression,
    then regresses out both scores using linear regression per gene.
    Returns regressed residual expression DataFrame.
    """
    from sklearn.linear_model import LinearRegression

    s_genes, g2m_genes, _ = get_cell_cycle_genes(df_log_cpm.columns)
    if len(s_genes) == 0 or len(g2m_genes) == 0:
        print("[Cell Cycle] Warning: Not enough cell cycle genes present for linear regression.")
        return df_log_cpm

    s_score = df_log_cpm[s_genes].mean(axis=1).values
    g2m_score = df_log_cpm[g2m_genes].mean(axis=1).values
    X_covar = np.column_stack([s_score, g2m_score])

    print(f"[Cell Cycle] Regressing out mean S/G2M scores across {df_log_cpm.shape[1]} genes...")
    lr = LinearRegression()
    lr.fit(X_covar, df_log_cpm.values)
    residuals = df_log_cpm.values - lr.predict(X_covar) + df_log_cpm.values.mean(axis=0)
    return pd.DataFrame(residuals, index=df_log_cpm.index, columns=df_log_cpm.columns)
