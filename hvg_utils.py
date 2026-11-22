"""
hvg_utils.py
------------
Shared gene-prevalence filtering (`--min_cells_per_gene`) and Highly Variable Gene (HVG)
selection utilities (`--n_hvg`, `--hvg_method`) across pipeline steps 03, 04, 05, and 05b.

Supported HVG selection methods:
  - 'raw_var'   : Top N genes by variance on raw/input counts (baseline for Steps 03, 04, 05b)
  - 'log1p_var' : Top N genes by variance on CP10K + log1p normalized counts (baseline for Step 05)
  - 'seurat'    : Scanpy `sc.pp.highly_variable_genes(..., flavor='seurat')` on CP10K + log1p
  - 'seurat_v3' : Scanpy `sc.pp.highly_variable_genes(..., flavor='seurat_v3')` on raw counts
"""

from typing import Iterable, Optional
import numpy as np
import pandas as pd


HVG_METHOD_CHOICES = ["default", "raw_var", "log1p_var", "seurat", "seurat_v3"]


def resolve_hvg_method(hvg_method: str, default_method: str) -> str:
    """Resolves 'default' to a script's baseline HVG method."""
    if not hvg_method or hvg_method == "default":
        return default_method
    if hvg_method not in HVG_METHOD_CHOICES:
        raise ValueError(
            f"Unsupported hvg_method '{hvg_method}'. Expected one of: {HVG_METHOD_CHOICES}"
        )
    return hvg_method


def filter_genes_by_min_cells(df_counts: pd.DataFrame, min_cells_per_gene: int = 0) -> pd.DataFrame:
    """
    Filters out genes expressed (> 0) in fewer than `min_cells_per_gene` cells.
    When `min_cells_per_gene <= 0`, returns `df_counts` unchanged.
    """
    if min_cells_per_gene is None or min_cells_per_gene <= 0:
        return df_counts

    n_before = df_counts.shape[1]
    n_cells_expressed = (df_counts > 0).sum(axis=0)
    keep_cols = n_cells_expressed[n_cells_expressed >= min_cells_per_gene].index

    if len(keep_cols) == 0:
        raise ValueError(
            f"No genes remained after applying --min_cells_per_gene {min_cells_per_gene} "
            f"(out of {n_before} genes across {df_counts.shape[0]} cells)."
        )

    n_after = len(keep_cols)
    print(
        f"[Gene Filter] min_cells_per_gene >= {min_cells_per_gene}: "
        f"kept {n_after}/{n_before} genes (removed {n_before - n_after})."
    )
    return df_counts[keep_cols]


def _cp10k_log1p(df_counts: pd.DataFrame) -> pd.DataFrame:
    """Library-size normalizes counts to CP10K and applies log1p."""
    df_pos = df_counts.clip(lower=0)
    lib_sizes = df_pos.sum(axis=1).replace(0, 1.0)
    return np.log1p(df_pos.div(lib_sizes, axis=0) * 1e4)


def select_hvgs(
    df_counts: pd.DataFrame,
    n_hvg: int = 2000,
    hvg_method: str = "raw_var",
    min_cells_per_gene: int = 0,
    extra_keep_genes: Optional[Iterable[str]] = None,
    already_log_normalized: bool = False,
    raw_counts_for_filter: Optional[pd.DataFrame] = None,
) -> pd.Index:
    """
    Applies optional `min_cells_per_gene` filtering and selects up to `n_hvg` highly
    variable genes using `hvg_method`.

    Parameters
    ----------
    df_counts : pd.DataFrame
        Cells x Genes expression matrix (raw counts or depth-normalized/regressed expression).
    n_hvg : int
        Maximum number of highly variable genes to select.
    hvg_method : str
        One of 'raw_var', 'log1p_var', 'seurat', 'seurat_v3'.
    min_cells_per_gene : int
        Minimum number of cells with expression > 0 required to retain a gene before HVG selection.
    extra_keep_genes : iterable of str, optional
        Gene symbols (case-insensitive) to force-retain if present after `min_cells_per_gene` filtering.
    already_log_normalized : bool
        If True, `df_counts` is already CP10K + log1p normalized (so `log1p_var` and `seurat`
        will not re-apply CP10K + log1p).
    raw_counts_for_filter : pd.DataFrame, optional
        Optional un-regressed raw count matrix aligned to `df_counts` used for exact
        `min_cells_per_gene` (> 0 counts) filtering and `seurat_v3` raw-count variance estimation.

    Returns
    -------
    pd.Index
        Selected gene column names.
    """
    if n_hvg <= 0:
        raise ValueError(f"--n_hvg must be a positive integer, got {n_hvg}.")

    if raw_counts_for_filter is not None and min_cells_per_gene and min_cells_per_gene > 0:
        raw_aligned = raw_counts_for_filter.reindex(index=df_counts.index, columns=df_counts.columns).fillna(0.0)
        keep_cols = filter_genes_by_min_cells(raw_aligned, min_cells_per_gene=min_cells_per_gene).columns
        df_filt = df_counts[keep_cols]
    else:
        df_filt = filter_genes_by_min_cells(df_counts, min_cells_per_gene=min_cells_per_gene)

    n_avail = df_filt.shape[1]
    n_target = min(int(n_hvg), n_avail)

    if hvg_method == "raw_var":
        gene_vars = df_filt.var(axis=0)
        active = gene_vars[gene_vars > 0]
        if len(active) == 0:
            top_genes = df_filt.columns[:n_target]
        else:
            top_genes = active.nlargest(min(n_target, len(active))).index

    elif hvg_method == "log1p_var":
        df_log = df_filt if already_log_normalized else _cp10k_log1p(df_filt)
        gene_vars = df_log.var(axis=0)
        active = gene_vars[gene_vars > 0]
        if len(active) == 0:
            top_genes = df_filt.columns[:n_target]
        else:
            top_genes = active.nlargest(min(n_target, len(active))).index

    elif hvg_method in ("seurat", "seurat_v3"):
        import anndata as ad
        import scanpy as sc

        # Drop zero-variance genes so scanpy dispersion/variance estimation is well-defined
        gene_vars = df_filt.var(axis=0)
        df_active = df_filt.loc[:, gene_vars > 0]
        if df_active.shape[1] == 0:
            top_genes = df_filt.columns[:n_target]
        else:
            n_req = min(n_target, df_active.shape[1])
            if hvg_method == "seurat":
                df_log = df_active if already_log_normalized else _cp10k_log1p(df_active)
                adata_tmp = ad.AnnData(
                    X=df_log.values.astype(np.float32),
                    obs=pd.DataFrame(index=df_active.index.astype(str)),
                    var=pd.DataFrame(index=df_active.columns.astype(str)),
                )
                sc.pp.highly_variable_genes(
                    adata_tmp,
                    n_top_genes=n_req,
                    flavor="seurat",
                )
            else:
                if raw_counts_for_filter is not None:
                    df_v3 = raw_counts_for_filter.reindex(index=df_active.index, columns=df_active.columns).fillna(0.0)
                else:
                    df_v3 = df_active
                adata_tmp = ad.AnnData(
                    X=df_v3.clip(lower=0).values.astype(np.float32),
                    obs=pd.DataFrame(index=df_active.index.astype(str)),
                    var=pd.DataFrame(index=df_active.columns.astype(str)),
                )
                try:
                    sc.pp.highly_variable_genes(
                        adata_tmp,
                        n_top_genes=n_req,
                        flavor="seurat_v3",
                    )
                except ImportError as exc:
                    raise ImportError(
                        "hvg_method='seurat_v3' requires the optional 'scikit-misc' package "
                        "(install via `pip install scikit-misc`), or use 'seurat', 'log1p_var', or 'raw_var'."
                    ) from exc

            hvg_mask = adata_tmp.var["highly_variable"].values
            top_genes = df_active.columns[hvg_mask]
    else:
        raise ValueError(
            f"Unsupported hvg_method '{hvg_method}'. Expected one of: raw_var, log1p_var, seurat, seurat_v3."
        )

    final_genes = list(top_genes)
    if extra_keep_genes:
        wanted = {g.lower() for g in extra_keep_genes}
        marker_hits = [g for g in df_filt.columns if g.lower() in wanted]
        final_genes = list(dict.fromkeys(final_genes + marker_hits))

    print(
        f"[HVG Selection] method='{hvg_method}', requested={n_hvg}, "
        f"selected={len(final_genes)} (from {n_avail} candidate genes)."
    )
    return pd.Index(final_genes)
