"""
preprocessing.py
----------------
Unified single-cell RNA-seq normalization and cell-cycle regression utilities shared
across all RNA-consuming pipeline stages (Steps 2, 3, 4, 5, and 5b).

Canonical Preprocessing Order:
  1. Depth Normalization (`--depth_norm`):
     - 'cp10k_log1p' (default): Library-size normalization to 10,000 counts per cell (CP10K)
       followed by log1p transformation.
     - 'none': Retains raw expression counts (for legacy raw-count baseline comparisons).
  2. Cell-Cycle Scoring & Linear Regression (`--regress_cell_cycle`):
     - 'none' (default): No cell-cycle regression applied.
     - 'full': Scores S-phase and G2/M-phase modules using the cited Tirosh et al. (2016) /
       Seurat `cc.genes.updated.2019` gene lists (via `cell_cycle.py`) on the depth-normalized
       expression matrix, then linearly regresses out both `S_score` and `G2M_score` via OLS.
     - 'difference': Regresses out `CC_difference = S_score - G2M_score` via OLS, removing
       S vs. G2/M phase variation while preserving cycling vs. non-cycling / endomitotic state
       separation (Satija Lab / Seurat alternate workflow).
  3. Optional Gene Exclusion (`--exclude_cell_cycle`, `--exclude_genes_file`, `--exclude_genes`):
     - Applied AFTER cell-cycle scoring/regression so that S and G2/M markers are available
       for scoring even when both `--regress_cell_cycle` and `--exclude_cell_cycle` are requested.

References:
  - Tirosh, I. et al. (2016). Science, 352(6282), 189-196. DOI: 10.1126/science.aad0501
  - Stuart, T. et al. (2019). Cell, 177(7), 1888-1902. DOI: 10.1016/j.cell.2019.05.031
  - Luecken, M. D., & Theis, F. J. (2019). Mol Syst Biol, 15(6), e8746. DOI: 10.15252/msb.20188746
"""

import os
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


DEPTH_NORM_CHOICES = ["cp10k_log1p", "none"]
REGRESS_CC_CHOICES = ["none", "full", "difference"]


def depth_normalize_expression(
    df_counts: pd.DataFrame,
    method: str = "cp10k_log1p",
    target_sum: float = 1e4,
) -> pd.DataFrame:
    """
    Applies library-size depth normalization to a (cells x genes) expression DataFrame.

    Parameters
    ----------
    df_counts : pd.DataFrame
        Cells x Genes expression matrix.
    method : str
        'cp10k_log1p' for CP10K + log1p normalization, or 'none' to keep raw counts.
    target_sum : float
        Target library size per cell (default: 10,000 for CP10K).
    """
    if method not in DEPTH_NORM_CHOICES:
        raise ValueError(f"Unsupported depth_norm '{method}'. Expected one of: {DEPTH_NORM_CHOICES}")

    if method == "none":
        print("[Depth Norm] method='none': retaining raw expression values.")
        return df_counts.astype(np.float64).copy()

    df_pos = df_counts.astype(np.float64).clip(lower=0.0)
    lib_sizes = df_pos.sum(axis=1)
    n_zero = int((lib_sizes == 0).sum())
    lib_safe = lib_sizes.replace(0.0, 1.0)
    df_norm = np.log1p(df_pos.div(lib_safe, axis=0) * float(target_sum))
    msg = f"[Depth Norm] method='cp10k_log1p' (target_sum={int(target_sum)}): normalized {df_norm.shape[0]} cells x {df_norm.shape[1]} genes."
    if n_zero > 0:
        msg += f" ({n_zero} zero-count cells left at 0.)"
    print(msg)
    return df_norm


def _match_phase_genes(columns: pd.Index) -> Tuple[List[str], List[str]]:
    """
    Matches cited Tirosh 2016 / Seurat 2019 S-phase and G2/M-phase mouse gene symbols
    (plus legacy annotation aliases) case-insensitively against `columns`.
    """
    from cell_cycle import (
        TIROSH_2016_S_GENES_MOUSE,
        TIROSH_2016_G2M_GENES_MOUSE,
        TIROSH_2016_LEGACY_ALIASES_MOUSE,
    )

    s_upper = {g.upper() for g in TIROSH_2016_S_GENES_MOUSE}
    g2m_upper = {g.upper() for g in TIROSH_2016_G2M_GENES_MOUSE}

    # Map legacy aliases to their respective phase sets
    for legacy, modern in TIROSH_2016_LEGACY_ALIASES_MOUSE.items():
        if modern.upper() in s_upper:
            s_upper.add(legacy.upper())
        elif modern.upper() in g2m_upper:
            g2m_upper.add(legacy.upper())

    s_matched = [c for c in columns if str(c).upper() in s_upper]
    g2m_matched = [c for c in columns if str(c).upper() in g2m_upper]
    return s_matched, g2m_matched


def compute_cell_cycle_scores(df_expr: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Computes per-cell `S_score`, `G2M_score`, `CC_difference`, and `phase` on a
    (cells x genes) depth-normalized expression DataFrame using Scanpy's
    `sc.tl.score_genes_cell_cycle` (with safe fallback if Scanpy is unavailable or
    matrix dimensions are very small).
    """
    s_genes, g2m_genes = _match_phase_genes(df_expr.columns)
    if len(s_genes) < 2 or len(g2m_genes) < 2:
        print(
            f"[Cell Cycle] Warning: Only {len(s_genes)} S-phase and {len(g2m_genes)} G2/M-phase "
            f"genes matched in expression matrix; skipping cell-cycle scoring/regression."
        )
        return None

    n_genes = df_expr.shape[1]
    n_bins = min(25, max(2, n_genes // 4))

    try:
        import anndata as ad
        import scanpy as sc

        adata_tmp = ad.AnnData(
            X=df_expr.values.astype(np.float32),
            obs=pd.DataFrame(index=df_expr.index.astype(str)),
            var=pd.DataFrame(index=df_expr.columns.astype(str)),
        )
        sc.tl.score_genes_cell_cycle(
            adata_tmp,
            s_genes=s_genes,
            g2m_genes=g2m_genes,
            n_bins=n_bins,
        )
        scores_df = pd.DataFrame(
            {
                "S_score": adata_tmp.obs["S_score"].astype(np.float64).values,
                "G2M_score": adata_tmp.obs["G2M_score"].astype(np.float64).values,
                "phase": adata_tmp.obs["phase"].astype(str).values,
            },
            index=df_expr.index,
        )
    except Exception as exc:
        # Pure-pandas fallback: mean module expression minus genome-wide per-cell mean
        print(f"[Cell Cycle] Note: Scanpy score_genes_cell_cycle fallback ({exc}); using mean-centered module scores.")
        cell_bg = df_expr.mean(axis=1)
        s_score = df_expr[s_genes].mean(axis=1) - cell_bg
        g2m_score = df_expr[g2m_genes].mean(axis=1) - cell_bg
        phase = np.where((s_score <= 0) & (g2m_score <= 0), "G1", np.where(s_score > g2m_score, "S", "G2M"))
        scores_df = pd.DataFrame(
            {
                "S_score": s_score.astype(np.float64).values,
                "G2M_score": g2m_score.astype(np.float64).values,
                "phase": phase,
            },
            index=df_expr.index,
        )

    scores_df["CC_difference"] = scores_df["S_score"] - scores_df["G2M_score"]
    scores_df["n_s_genes_matched"] = len(s_genes)
    scores_df["n_g2m_genes_matched"] = len(g2m_genes)
    return scores_df


def regress_out_covariates(df_expr: pd.DataFrame, covariates: np.ndarray) -> pd.DataFrame:
    """
    Vectorized OLS residualization of `df_expr` (N x G) against `covariates` (N x P),
    preserving each gene's baseline mean expression.
    """
    Y = df_expr.values.astype(np.float64)
    X = np.asarray(covariates, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, np.newaxis]

    X_c = X - np.mean(X, axis=0, keepdims=True)
    if np.allclose(X_c, 0.0):
        return df_expr.copy()

    Y_mean = np.mean(Y, axis=0, keepdims=True)
    Y_c = Y - Y_mean
    beta, _, _, _ = np.linalg.lstsq(X_c, Y_c, rcond=None)
    Y_adj = Y - (X_c @ beta)
    return pd.DataFrame(Y_adj, index=df_expr.index, columns=df_expr.columns)


def score_and_regress_cell_cycle(
    df_expr: pd.DataFrame,
    mode: str = "none",
    scores_df: Optional[pd.DataFrame] = None,
    audit_csv_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Scores S and G2/M cell-cycle modules (unless `scores_df` is precomputed) and
    linearly regresses out cell-cycle variation from `df_expr`.

    Parameters
    ----------
    df_expr : pd.DataFrame
        Cells x Genes expression matrix (should already be depth-normalized).
    mode : str
        'none' (no-op), 'full' (regress S_score + G2M_score), or 'difference' (regress S_score - G2M_score).
    scores_df : pd.DataFrame, optional
        Precomputed per-cell scores DataFrame aligned to `df_expr.index`.
    audit_csv_path : str, optional
        If provided, saves per-cell S/G2M scores and regression mode to CSV.
    """
    if mode not in REGRESS_CC_CHOICES:
        raise ValueError(f"Unsupported regress_cell_cycle '{mode}'. Expected one of: {REGRESS_CC_CHOICES}")

    if mode == "none":
        return df_expr, None

    if scores_df is None:
        scores_df = compute_cell_cycle_scores(df_expr)
    else:
        scores_df = scores_df.reindex(df_expr.index)

    if scores_df is None or scores_df[["S_score", "G2M_score"]].isna().any().any():
        return df_expr, scores_df

    if mode == "full":
        cov = scores_df[["S_score", "G2M_score"]].values
        cov_desc = "S_score + G2M_score (full)"
    else:
        cov = scores_df[["CC_difference"]].values
        cov_desc = "CC_difference = S_score - G2M_score (difference)"

    df_reg = regress_out_covariates(df_expr, cov)
    n_s = int(scores_df["n_s_genes_matched"].iloc[0]) if "n_s_genes_matched" in scores_df.columns else 0
    n_g2m = int(scores_df["n_g2m_genes_matched"].iloc[0]) if "n_g2m_genes_matched" in scores_df.columns else 0
    print(
        f"[Cell Cycle Regression] mode='{mode}' ({cov_desc}): "
        f"regressed across {df_reg.shape[0]} cells x {df_reg.shape[1]} genes "
        f"(matched {n_s} S + {n_g2m} G2/M markers)."
    )

    if audit_csv_path:
        out_df = scores_df.copy()
        out_df["regress_mode"] = mode
        out_df.index.name = "SampleID"
        os.makedirs(os.path.dirname(os.path.abspath(audit_csv_path)), exist_ok=True)
        out_df.to_csv(audit_csv_path)
        print(f"[Cell Cycle Regression] Saved per-cell cell-cycle scores to: {os.path.abspath(audit_csv_path)}")

    return df_reg, scores_df


def adjust_library_depth_cdr(df_expr: pd.DataFrame) -> pd.DataFrame:
    """
    Applies per-cell library-depth / Cellular Detection Rate (CDR) residualization
    (Finak et al., 2015): z-scores each gene, computes each cell's global mean
    standardized expression (CDR proxy), linearly regresses it out of every gene,
    and re-standardizes each non-constant gene to zero mean and unit variance.
    """
    X = df_expr.values.astype(np.float64)
    col_mean = np.mean(X, axis=0, keepdims=True)
    col_std = np.std(X, axis=0, keepdims=True)
    non_const = col_std[0] > 0
    if not np.any(non_const):
        return df_expr.copy()

    Z = np.zeros_like(X, dtype=np.float64)
    Z[:, non_const] = (X[:, non_const] - col_mean[:, non_const]) / col_std[:, non_const]

    depth_factor = np.mean(Z[:, non_const], axis=1)
    d_c = depth_factor - np.mean(depth_factor)
    d_var = float(np.sum(d_c * d_c))
    if d_var <= 0.0:
        return pd.DataFrame(Z, index=df_expr.index, columns=df_expr.columns)

    beta = np.sum(d_c[:, np.newaxis] * Z[:, non_const], axis=0, keepdims=True) / d_var
    res = Z[:, non_const] - d_c[:, np.newaxis] * beta
    r_std = np.std(res, axis=0, keepdims=True)
    r_std[r_std == 0] = 1.0
    Z[:, non_const] = (res - np.mean(res, axis=0, keepdims=True)) / r_std

    print(
        f"[Depth CDR] Applied library-depth / CDR residualization (Finak et al., 2015) "
        f"across {Z.shape[0]} cells x {int(np.sum(non_const))} non-constant genes."
    )
    return pd.DataFrame(Z, index=df_expr.index, columns=df_expr.columns)


def preprocess_rna_matrix(
    df_counts: pd.DataFrame,
    depth_norm: str = "cp10k_log1p",
    adjust_library_depth: bool = False,
    regress_cell_cycle: str = "none",
    exclude_cell_cycle: bool = False,
    exclude_genes_file: Optional[str] = None,
    exclude_genes: Optional[str] = None,
    out_dir: Optional[str] = None,
    step_prefix: str = "",
    is_prescaled: bool = False,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Executes the canonical RNA preprocessing order:
      1. Depth normalization (`depth_norm`: 'cp10k_log1p' or 'none')
      2. Optional library-depth / CDR residualization (`adjust_library_depth`, Finak et al., 2015)
      3. Cell-cycle scoring & regression (`regress_cell_cycle`: 'none', 'full', 'difference')
      4. Optional gene exclusion (`--exclude_cell_cycle`, `--exclude_genes_file`, `--exclude_genes`)

    Returns (df_processed, cc_scores_df).
    """
    # 1. Depth Normalization
    if is_prescaled:
        if depth_norm != "none":
            print("[Depth Norm] Notice: Input matrix is already z-scored (pre-saved HDF5); skipping CP10K + log1p.")
        df_proc = df_counts.astype(np.float64).copy()
    else:
        if regress_cell_cycle != "none" and depth_norm == "none":
            print(
                "[Cell Cycle Regression] Warning: --regress_cell_cycle requested with --depth_norm none; "
                "CP10K + log1p depth normalization prior to cell-cycle regression is strongly recommended."
            )
        df_proc = depth_normalize_expression(df_counts, method=depth_norm)

    # Precompute cell-cycle scores on the depth-normalized matrix before CDR z-scoring
    # so S/G2M module scores match across CDR-adjusted and unadjusted runs
    precomputed_cc_scores = (
        compute_cell_cycle_scores(df_proc)
        if (adjust_library_depth and regress_cell_cycle != "none")
        else None
    )

    # 2. Optional Library-Depth / CDR Residualization (strictly before cell-cycle regression)
    if adjust_library_depth:
        df_proc = adjust_library_depth_cdr(df_proc)

    # 3. Cell-Cycle Scoring & Regression (runs BEFORE gene exclusion so S/G2M markers are present for scoring)
    cc_audit_path = None
    if out_dir and regress_cell_cycle != "none":
        prefix = f"{step_prefix}_" if step_prefix else ""
        cc_audit_path = os.path.join(out_dir, f"{prefix}cell_cycle_scores.csv")
    df_proc, cc_scores = score_and_regress_cell_cycle(
        df_proc,
        mode=regress_cell_cycle,
        scores_df=precomputed_cc_scores,
        audit_csv_path=cc_audit_path,
    )

    # 3. Optional Gene Exclusion
    if exclude_cell_cycle or exclude_genes_file or exclude_genes:
        from cell_cycle import filter_excluded_genes

        prefix = f"{step_prefix}_" if step_prefix else ""
        audit_path = os.path.join(out_dir, f"{prefix}excluded_genes_audit.csv") if out_dir else None
        keep_mask, _ = filter_excluded_genes(
            df_proc.columns,
            exclude_cell_cycle=exclude_cell_cycle,
            exclude_genes_file=exclude_genes_file,
            exclude_genes=exclude_genes,
            audit_csv_path=audit_path,
        )
        df_proc = df_proc.loc[:, keep_mask].copy()
    else:
        print("[Gene Filter] No additional gene exclusions applied (baseline).")

    return df_proc, cc_scores
