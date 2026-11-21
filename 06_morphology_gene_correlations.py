"""
06_morphology_gene_correlations.py
----------------------------------
Computes robust pairwise Spearman rank correlations between single-cell gene expression
and imaging flow cytometry morphology parameters across megakaryocyte polyploidization states
(2n, 4n, 8n, 16n, 32n).

Features:
1. Backwards-Compatible & Multi-Cohort Data Ingestion:
   - Supports the unified multi-animal metadata file (Metadata_merged_YL_SF_20.10.26_v2.csv,
     393 QC-passed cells across Mice M1-M7: 291 in 10plate M1-M5 + 102 in 4plate M6-M7).
   - Fully backwards compatible with legacy train/pilot datasets (D:\\Morphotranscriptomic_Data,
     exp/, meta.v1+flowSF.csv, Metadata_flow+well_id+supervised_ploidy.csv, meta.contaminant.2.tsv,
     and pre-saved mofa_model_rna_morphology.hdf5).
   - Matches cell identifiers automatically across SampleID, Well_ID, Well, Barcode, and X.
2. Robust Spearman Correlation & Statistical Testing:
   - Computes pairwise Spearman rank correlation (rho), two-sided t-distribution p-values,
     and Benjamini-Hochberg FDR (q-values) across all filtered genes and morphology features.
   - Computes partial Spearman correlation controlling for log2(Ploidy) to distinguish
     ploidy-driven scaling from ploidy-independent morphotranscriptomic coupling.
   - Evaluates within-cohort consistency (10plate M1-M5 vs 4plate M6-M7) when multiple cohorts exist.
3. Publication-Grade Visualizations:
   - 06_top_gene_morphology_spearman_scatter_grid.png:
     Multi-panel scatter & regression grid of top correlated gene-morphology pairs colored by
     Supervised Ploidy, labeling extreme cells and displaying Spearman rho & FDR q-values.
   - 06_gene_morphology_spearman_ranking_labeled.png:
     Two-panel summary (Volcano plot + Top Ranked Gene Lollipop/Bar chart) explicitly labeling
     genes with particularly high Spearman correlation and their associated morphology features.
   - 06_morphology_gene_spearman_heatmap.png:
     Clustered Spearman correlation heatmap of top correlated genes vs imaging morphology features.

Usage:
    # Run on default unified multi-animal dataset (or automatic fallback to available datasets):
    python 06_morphology_gene_correlations.py --out_dir "output"

    # Run on a specific legacy dataset directory / metadata file:
    python 06_morphology_gene_correlations.py --metadata_file "meta.v1+flowSF.csv" --exp_dir "exp" --out_dir "output_legacy"
"""

import os
import sys

# Ensure Conda Library\bin DLLs (Intel MKL / LAPACK) are discoverable on Windows even without conda activate
_conda_lib_bin = os.path.join(sys.prefix, "Library", "bin")
if os.path.isdir(_conda_lib_bin):
    os.environ["PATH"] = _conda_lib_bin + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_conda_lib_bin)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import re
import zipfile
import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import mmread
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


DEFAULT_EXPLORATION_DIR = r"C:\Users\vilsn\Dropbox\Work\Turing\Exploration_Data"
DEFAULT_UNIFIED_META = os.path.join(DEFAULT_EXPLORATION_DIR, "Metadata_merged_YL_SF_20.10.26_v2.csv")
DEFAULT_TRAIN_DIR = r"D:\Morphotranscriptomic_Data"

PLOIDY_ORDER = ["2n", "4n", "8n", "16n", "32n"]
PLOIDY_PALETTE = {
    "2n": "#1f77b4",
    "4n": "#2ca02c",
    "8n": "#ff7f0e",
    "16n": "#d62728",
    "32n": "#9467bd",
    "64n": "#8c564b",
    "Unknown": "#7f7f7f"
}

KEY_MK_MARKERS = [
    "Kalrn", "Ppbp", "Pde3a", "Mylk", "Rock1", "Exoc6", "Add3", "Fhl1", "Itga6",
    "Pf4", "Itga2b", "Mpl", "Runx1", "Fli1", "Gata1", "Nfe2", "Gp1bb", "Tubb1", "Vwf", "Selp"
]


def harmonize_and_adjust_modalities(df_rna, df_morph, cohort_sources, adjust_library_depth=True):
    """
    Harmonizes RNA expression and imaging morphology across cohorts (e.g. 10plate M1-M5 vs 4plate M6-M7)
    and optionally residualizes out per-cell sequencing library depth / global detection bias.

    Why this is critical:
      In megakaryocyte sorting, larger polyploid cells (16n, 32n) exhibit lower total library counts
      (spearman(lib_size, Size) = -0.54), which otherwise makes nearly all genes appear negatively
      correlated with cell size. Adjusting for per-cell library depth and cohort scale reveals the
      true gene-specific morphology correlations (e.g. Ppbp, Mylk, Pde3a, Kalrn, Rock1, Exoc6).
    """
    df_morph_h = df_morph.copy().astype(np.float64)
    df_rna_h = df_rna.copy().astype(np.float64)

    for src in cohort_sources.unique():
        mask = (cohort_sources == src).values
        if mask.sum() < 3:
            continue
        # Cohort-level z-score standardization of morphology features
        sub_m = df_morph_h.iloc[mask]
        m_std = sub_m.std(axis=0).replace(0, 1.0)
        df_morph_h.iloc[mask] = (sub_m - sub_m.mean(axis=0)) / m_std

        if adjust_library_depth:
            sub_r = df_rna_h.iloc[mask]
            # Per-cell mean standardized expression captures global library depth / dropout factor
            depth_factor = sub_r.mean(axis=1).values
            d_c = depth_factor - depth_factor.mean()
            d_var = np.sum(d_c * d_c)
            if d_var > 0:
                beta = np.sum(d_c[:, np.newaxis] * sub_r.values, axis=0, keepdims=True) / d_var
                res = sub_r.values - d_c[:, np.newaxis] * beta
                r_std = np.std(res, axis=0, keepdims=True)
                r_std[r_std == 0] = 1.0
                df_rna_h.iloc[mask] = (res - np.mean(res, axis=0, keepdims=True)) / r_std

    return df_rna_h, df_morph_h


def setup_plot_style():
    """Configures high-DPI matplotlib publication rendering matching project standards."""
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["figure.autolayout"] = False
    sns.set_theme(style="whitegrid")


def categorize_morphology_feature(feat_name):
    """Categorizes an imaging flow cytometry parameter into a biological feature family."""
    f = str(feat_name)
    if re.search(r"Size|FSC-W|SSC.*-W|LightLoss.*-W|SpectralEventWidth", f, re.I):
        return "Cell Size & Width"
    if re.search(r"Total Intensity|DNA-A|CD41-A|FSC-A|SSC.*-A|LightLoss.*-A|ImgB.*-A|Autofluorescence", f, re.I):
        return "Total / Area Intensity"
    if re.search(r"Max Intensity|FSC-H|SSC.*-H|LightLoss.*-H|ImgB.*-H", f, re.I):
        return "Peak / Max Intensity"
    if re.search(r"Moment", f, re.I):
        return "Shape Moments (Long/Short/Radial)"
    if re.search(r"Diffusivity|Eccentricity", f, re.I):
        return "Diffusivity & Eccentricity"
    if re.search(r"Center of Mass|Delta CoM|Correlation", f, re.I):
        return "Spatial CoM & Localization"
    return "Other Flow / Imaging"


def extract_morphology_features(meta_df):
    """
    Extracts numeric imaging flow cytometry parameters matching morphotrans2 regex pattern,
    excluding technical time-of-flight (-T) or constant columns.
    """
    pattern = (
        r".*\(Imaging\).*|.*Axis.*|.*Mass.*|.*Intensity.*|.*Moment.*|"
        r".*Size.*|.*Diffusivity.*|.*Eccentricity.*|FSC-|SSC-|CD41-|DNA-|Correlation"
    )
    img_cols = [c for c in meta_df.columns if re.search(pattern, c)]

    numeric_cols = []
    for col in img_cols:
        # Exclude raw trigger/time-of-flight (-T) technical columns if present alongside area/height/width
        if col.endswith("-T"):
            continue
        series_num = pd.to_numeric(meta_df[col], errors="coerce")
        if series_num.notna().sum() >= max(10, int(0.2 * len(meta_df))) and series_num.std(skipna=True) > 0:
            numeric_cols.append(col)

    img_df = meta_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    # Impute rare missing entries with column median for rank stability
    img_df = img_df.fillna(img_df.median(axis=0)).fillna(0.0)
    non_const = img_df.columns[img_df.std(axis=0) > 0]
    img_df = img_df[non_const]
    print(f"[Morphology] Extracted {len(img_df.columns)} informative morphology/flow features across {len(img_df)} cells.")
    return img_df


def load_qc_metadata_backwards_compatible(
    metadata_file=None,
    qc_meta_path="output/qc_filtered_metadata.csv",
    train_dir=DEFAULT_TRAIN_DIR,
    test_dir="exp",
    cohort_filter="all",
    min_mapped_reads=20000,
    min_map_pc=50.0,
):
    """
    Loads QC-filtered metadata supporting both the unified multi-animal metadata file
    (Metadata_merged_YL_SF_20.10.26_v2.csv) and legacy train/pilot metadata files.
    """
    from importlib import import_module
    qc_mod = import_module("01_qc_and_metadata_filtering")

    if qc_meta_path and os.path.exists(qc_meta_path) and metadata_file is None:
        print(f"[Metadata] Loading pre-filtered QC metadata from: {os.path.abspath(qc_meta_path)}")
        qc_df = pd.read_csv(qc_meta_path, index_col=0)
        if "SampleID" not in qc_df.columns:
            qc_df["SampleID"] = qc_df.index.astype(str)
        qc_df.index = qc_df.index.astype(str)
    else:
        meta_df = qc_mod.load_metadata_sources(
            metadata_file=metadata_file,
            train_dir=train_dir,
            test_dir=test_dir
        )
        full_df, qc_df = qc_mod.apply_qc_filters(
            meta_df,
            min_mapped_reads=min_mapped_reads,
            min_map_pc=min_map_pc,
        )
        if len(qc_df) == 0:
            print("[Metadata] Warning: No cells matched full strict QC in legacy file; falling back to all non-control cells.")
            qc_df = full_df[~full_df.get("is_control", pd.Series(False, index=full_df.index))].copy()

    if cohort_filter and cohort_filter.lower() != "all":
        cf = cohort_filter.lower()
        if "dataset_type" in qc_df.columns:
            mask = qc_df["dataset_type"].astype(str).str.lower().str.contains(cf)
            if mask.any():
                qc_df = qc_df[mask].copy()
                print(f"[Metadata] Filtered to cohort '{cohort_filter}': {len(qc_df)} cells.")

    return qc_df


def _read_10x_dir_as_dataframe(exp_dir):
    """
    Reads a 10x Genomics matrix directory (matrix.mtx[.gz], features.tsv[.gz], barcodes.tsv[.gz])
    using scipy.io.mmread and pandas without requiring scanpy BLAS calls.
    """
    if not exp_dir or not os.path.exists(exp_dir):
        return None

    feat_path = os.path.join(exp_dir, "features.tsv")
    if not os.path.exists(feat_path):
        feat_path = os.path.join(exp_dir, "features.tsv.gz")
    if not os.path.exists(feat_path):
        feat_path = os.path.join(exp_dir, "genes.tsv")

    bc_path = os.path.join(exp_dir, "barcodes.tsv")
    if not os.path.exists(bc_path):
        bc_path = os.path.join(exp_dir, "barcodes.tsv.gz")

    mtx_path = os.path.join(exp_dir, "matrix.mtx")
    if not os.path.exists(mtx_path):
        mtx_path = os.path.join(exp_dir, "matrix.mtx.gz")

    if not (os.path.exists(feat_path) and os.path.exists(bc_path) and os.path.exists(mtx_path)):
        return None

    print(f"[Expression] Loading 10x matrix from: {os.path.abspath(exp_dir)}")
    features_df = pd.read_csv(feat_path, sep="\t", header=None)
    gene_symbols = features_df.iloc[:, 1].astype(str).values if features_df.shape[1] > 1 else features_df.iloc[:, 0].astype(str).values

    barcodes_df = pd.read_csv(bc_path, sep="\t", header=None)
    barcodes = barcodes_df.iloc[:, 0].astype(str).values

    mat = mmread(mtx_path).tocsc()  # shape: (n_genes, n_cells)

    # Filter out ribosomal (Rps, Rpl), mitochondrial (mt-), and Unnamed genes
    s_series = pd.Series(gene_symbols)
    keep_mask = ~(
        s_series.str.startswith(("Rps", "Rpl", "mt-", "MT-", "Unnamed"))
    ).values
    keep_idx = np.where(keep_mask)[0]

    mat_sub = mat[keep_idx, :].T.toarray().astype(np.float64)  # shape: (n_cells, n_kept_genes)
    kept_genes = gene_symbols[keep_idx]

    df_exp = pd.DataFrame(mat_sub, index=barcodes, columns=kept_genes)
    # Deduplicate gene symbols by keeping the highest-variance isoform/entry
    if df_exp.columns.duplicated().any():
        df_exp = df_exp.loc[:, ~df_exp.columns.duplicated(keep="first")]

    return df_exp


def _match_barcodes_to_metadata(df_exp, qc_df):
    """
    Matches expression matrix cell barcodes against metadata identifiers across
    SampleID, index, Well_ID, Well, Barcode, or X for full backwards compatibility.
    Returns a copy of df_exp re-indexed by qc_df.index for the overlapping cells.
    """
    if df_exp is None or df_exp.empty:
        return pd.DataFrame()

    exp_index_set = set(df_exp.index.astype(str))
    candidate_cols = ["SampleID", "Well_ID", "Well", "Barcode", "X"]

    best_map = {}
    # First check direct index match
    direct_overlap = qc_df.index.astype(str).intersection(df_exp.index.astype(str))
    if len(direct_overlap) > 0:
        for cid in direct_overlap:
            best_map[cid] = cid

    # Check candidate metadata columns for any unmatched rows
    for col in candidate_cols:
        if col in qc_df.columns:
            for meta_idx, val in zip(qc_df.index.astype(str), qc_df[col].astype(str)):
                if meta_idx not in best_map and val in exp_index_set:
                    best_map[meta_idx] = val
                # Also check stripped prefixes like '10plate_M1p1_A10' -> 'M1p1_A10'
                if meta_idx not in best_map and "_" in val:
                    stripped = val.split("_", 1)[1]
                    if stripped in exp_index_set:
                        best_map[meta_idx] = stripped

    if not best_map:
        return pd.DataFrame()

    matched_meta_ids = [cid for cid in qc_df.index.astype(str) if cid in best_map]
    matched_exp_ids = [best_map[cid] for cid in matched_meta_ids]

    df_matched = df_exp.loc[matched_exp_ids].copy()
    df_matched.index = matched_meta_ids
    return df_matched


def load_expression_backwards_compatible(
    qc_df,
    exp_dir=None,
    pilot_exp_dir=None,
    mofa_hdf5_path=None,
    n_hvg=2000,
    hvg_method="default",
    min_cells_per_gene=0,
):
    """
    Loads single-cell gene expression aligned to qc_df across all available cohorts.
    Supports:
      1. Primary 10x expression directory (e.g. D:\\Morphotranscriptomic_Data\\expression)
      2. Pilot / 4plate 10x expression directory (exp/ or Exploration_Data/exp)
      3. Pre-saved MOFA HDF5 RNA matrix (mofa_model_rna_morphology.hdf5) for the 291 M1-M5
         cells when the external D: drive is not mounted.
    """
    import h5py

    cohort_frames = []
    cohort_sources = {}

    # Candidate 10x directories
    candidate_dirs = []
    if exp_dir:
        candidate_dirs.append(exp_dir)
    if pilot_exp_dir:
        candidate_dirs.append(pilot_exp_dir)
    candidate_dirs.extend([
        os.path.join(DEFAULT_TRAIN_DIR, "expression"),
        "exp",
        os.path.join(DEFAULT_EXPLORATION_DIR, "exp"),
        r"C:\Users\vilsn\Dropbox\Work\Turing\RNA_Analysis\gene_exp.plate1-4",
    ])

    seen_abs_dirs = set()
    matched_cell_ids = set()

    for d in candidate_dirs:
        if not d or not os.path.exists(d):
            continue
        abs_d = os.path.abspath(d)
        if abs_d in seen_abs_dirs:
            continue
        seen_abs_dirs.add(abs_d)

        df_raw = _read_10x_dir_as_dataframe(abs_d)
        df_matched = _match_barcodes_to_metadata(df_raw, qc_df)
        new_cells = [c for c in df_matched.index if c not in matched_cell_ids]
        if len(new_cells) > 0:
            df_sub = df_matched.loc[new_cells]
            cohort_frames.append(("10x_mtx", abs_d, df_sub))
            for c in new_cells:
                matched_cell_ids.add(c)
                cohort_sources[c] = f"10x:{os.path.basename(abs_d)}"
            print(f"[Expression] Matched {len(new_cells)} QC-passed cells from 10x directory: {abs_d}")

    # Check if there are still unmatched QC-passed cells (e.g. 291 10plate M1-M5 cells when D: is offline)
    unmatched_cells = [c for c in qc_df.index.astype(str) if c not in matched_cell_ids]
    if len(unmatched_cells) > 0:
        h5_candidates = []
        if mofa_hdf5_path:
            h5_candidates.append(mofa_hdf5_path)
        h5_candidates.extend([
            "mofa_model_rna_morphology.hdf5",
            os.path.join(DEFAULT_EXPLORATION_DIR, "mofa_model_rna_morphology.hdf5")
        ])
        for h5_p in h5_candidates:
            if h5_p and os.path.exists(h5_p):
                with h5py.File(h5_p, "r") as f:
                    if "data" in f and "RNA" in f["data"] and "samples" in f and "features" in f:
                        s_raw = f["samples"]["group1"][:]
                        samples = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in s_raw]
                        g_raw = f["features"]["RNA"][:]
                        genes = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in g_raw]
                        rna_mat = f["data"]["RNA"]["group1"][:].astype(np.float64)

                        df_h5 = pd.DataFrame(rna_mat, index=samples, columns=genes)
                        df_h5 = df_h5.loc[:, ~df_h5.columns.duplicated(keep="first")]
                        df_h5_matched = _match_barcodes_to_metadata(df_h5, qc_df.loc[unmatched_cells])
                        if not df_h5_matched.empty:
                            cohort_frames.append(("mofa_hdf5", os.path.abspath(h5_p), df_h5_matched))
                            for c in df_h5_matched.index:
                                matched_cell_ids.add(c)
                                cohort_sources[c] = "10plate_HDF5"
                            print(
                                f"[Expression] Matched {len(df_h5_matched)} QC-passed cells (2000 HVGs) "
                                f"from pre-saved MOFA HDF5: {os.path.abspath(h5_p)}"
                            )
                            break

    if not cohort_frames:
        raise FileNotFoundError(
            "Could not match any cell barcodes between metadata and available expression sources."
        )

    from hvg_utils import resolve_hvg_method, select_hvgs

    resolved_method = resolve_hvg_method(hvg_method, default_method="raw_var")

    # Harmonize across expression cohorts
    if len(cohort_frames) == 1:
        kind, src_path, df_only = cohort_frames[0]
        if kind == "10x_mtx":
            # Select top n_hvg variable genes (with optional min_cells_per_gene) and StandardScaler transform
            final_genes = select_hvgs(
                df_only,
                n_hvg=n_hvg,
                hvg_method=resolved_method,
                min_cells_per_gene=min_cells_per_gene,
                extra_keep_genes=KEY_MK_MARKERS,
            )
            df_hvg = df_only[final_genes]
            df_scaled = pd.DataFrame(
                StandardScaler().fit_transform(df_hvg),
                index=df_hvg.index,
                columns=df_hvg.columns
            )
            return df_scaled, pd.Series(cohort_sources)
        else:
            if n_hvg != 2000 or min_cells_per_gene > 0 or hvg_method != "default":
                print(
                    f"[Expression] Note: Pre-saved MOFA HDF5 contains {df_only.shape[1]} pre-selected HVGs; "
                    f"--n_hvg/--hvg_method/--min_cells_per_gene apply only to raw 10x matrix sources."
                )
            return df_only, pd.Series(cohort_sources)

    # Multiple cohorts (e.g. 291 10plate cells + 102 4plate cells = 393 cells across M1-M7):
    # Find shared genes and standardize within each cohort so z-scores are directly comparable
    shared_genes = None
    for _, _, df_part in cohort_frames:
        cols = pd.Index(df_part.columns)
        shared_genes = cols if shared_genes is None else shared_genes.intersection(cols)

    # If all sources are raw 10x matrices, select top n_hvg HVGs across shared genes
    if all(kind == "10x_mtx" for kind, _, _ in cohort_frames):
        pooled_raw = pd.concat([df_p[shared_genes] for _, _, df_p in cohort_frames], axis=0)
        shared_genes = select_hvgs(
            pooled_raw,
            n_hvg=n_hvg,
            hvg_method=resolved_method,
            min_cells_per_gene=min_cells_per_gene,
            extra_keep_genes=KEY_MK_MARKERS,
        )
    elif n_hvg != 2000 or min_cells_per_gene > 0 or hvg_method != "default":
        print(
            f"[Expression] Note: Combined sources include a pre-saved MOFA HDF5 ({len(shared_genes)} shared genes); "
            f"--n_hvg/--hvg_method/--min_cells_per_gene require raw 10x matrices for all cohorts."
        )

    scaled_parts = []
    for kind, src_path, df_part in cohort_frames:
        sub = df_part[shared_genes].copy()
        # Standardize each cohort to zero mean and unit variance per gene
        scaled_arr = StandardScaler().fit_transform(sub.values)
        scaled_parts.append(pd.DataFrame(scaled_arr, index=sub.index, columns=shared_genes))

    df_combined = pd.concat(scaled_parts, axis=0)
    ordered_ids = [c for c in qc_df.index.astype(str) if c in df_combined.index]
    df_combined = df_combined.loc[ordered_ids]
    print(
        f"[Expression] Combined multi-cohort expression matrix: {df_combined.shape[0]} cells x "
        f"{df_combined.shape[1]} genes across {len(cohort_frames)} sources."
    )
    return df_combined, pd.Series(cohort_sources).reindex(ordered_ids)


def benjamini_hochberg_fdr(p_values):
    """Computes Benjamini-Hochberg FDR q-values for a 1D or 2D array of p-values."""
    orig_shape = p_values.shape
    p_flat = np.asarray(p_values, dtype=np.float64).ravel()
    n = len(p_flat)
    valid_mask = np.isfinite(p_flat)
    q_flat = np.ones(n, dtype=np.float64)

    if valid_mask.any():
        p_valid = p_flat[valid_mask]
        m = len(p_valid)
        order = np.argsort(p_valid)
        ranked_p = p_valid[order]
        ranks = np.arange(1, m + 1, dtype=np.float64)
        q_ranked = ranked_p * m / ranks
        q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
        q_ranked = np.clip(q_ranked, 0.0, 1.0)
        q_valid = np.empty(m, dtype=np.float64)
        q_valid[order] = q_ranked
        q_flat[valid_mask] = q_valid

    return q_flat.reshape(orig_shape)


def compute_spearman_matrix_no_blas(df_rna, df_morph):
    """
    Computes exact pairwise Spearman rank correlations (G x M) and two-sided p-values
    using chunked elementwise NumPy reductions (avoiding BLAS dgemm OpenMP issues).
    """
    n_cells = df_rna.shape[0]
    rna_ranks = df_rna.rank(axis=0, method="average").values.astype(np.float64)     # (N, G)
    morph_ranks = df_morph.rank(axis=0, method="average").values.astype(np.float64) # (N, M)

    rna_c = rna_ranks - rna_ranks.mean(axis=0, keepdims=True)
    morph_c = morph_ranks - morph_ranks.mean(axis=0, keepdims=True)

    rna_ss = np.sqrt(np.sum(rna_c * rna_c, axis=0))     # (G,)
    morph_ss = np.sqrt(np.sum(morph_c * morph_c, axis=0)) # (M,)

    rna_zero = rna_ss == 0
    morph_zero = morph_ss == 0
    rna_ss[rna_zero] = 1.0
    morph_ss[morph_zero] = 1.0

    rna_unit = rna_c / rna_ss[np.newaxis, :]     # (N, G)
    morph_unit = morph_c / morph_ss[np.newaxis, :] # (N, M)

    n_genes = rna_unit.shape[1]
    n_morph = morph_unit.shape[1]
    rho_mat = np.zeros((n_genes, n_morph), dtype=np.float64)

    # Elementwise sum over N in gene chunks to avoid BLAS @ call completely
    chunk_size = 200
    for start in range(0, n_genes, chunk_size):
        end = min(start + chunk_size, n_genes)
        # shape: (N, chunk, 1) * (N, 1, M) -> sum over axis 0 -> (chunk, M)
        rho_mat[start:end, :] = np.sum(
            rna_unit[:, start:end, np.newaxis] * morph_unit[:, np.newaxis, :],
            axis=0
        )

    rho_mat[rna_zero, :] = 0.0
    rho_mat[:, morph_zero] = 0.0
    rho_mat = np.clip(rho_mat, -1.0, 1.0)

    # Two-sided t-distribution p-values for Spearman rank correlation
    dof = max(n_cells - 2, 1)
    denom = np.maximum(1.0 - rho_mat * rho_mat, 1e-15)
    t_stat = np.abs(rho_mat) * np.sqrt(dof / denom)
    p_mat = 2.0 * stats.t.sf(t_stat, df=dof)
    fdr_mat = benjamini_hochberg_fdr(p_mat)

    rho_df = pd.DataFrame(rho_mat, index=df_rna.columns, columns=df_morph.columns)
    pval_df = pd.DataFrame(p_mat, index=df_rna.columns, columns=df_morph.columns)
    fdr_df = pd.DataFrame(fdr_mat, index=df_rna.columns, columns=df_morph.columns)
    return rho_df, pval_df, fdr_df


def compute_partial_spearman_given_ploidy(df_rna, df_morph, ploidy_series):
    """
    Computes partial Spearman correlation between genes and morphology features controlling
    for log2(Supervised_ploidy) via rank residualization.
    """
    ploidy_map = {"2n": 1.0, "4n": 2.0, "8n": 3.0, "16n": 4.0, "32n": 5.0, "64n": 6.0}
    z_ploidy = pd.Series(
        [ploidy_map.get(str(p).strip(), np.nan) for p in ploidy_series],
        index=df_rna.index
    )
    if z_ploidy.isna().all() or z_ploidy.std(skipna=True) == 0:
        return pd.DataFrame(0.0, index=df_rna.columns, columns=df_morph.columns)

    z_rank = z_ploidy.fillna(z_ploidy.median()).rank().values.astype(np.float64)
    z_c = z_rank - z_rank.mean()
    z_var = np.sum(z_c * z_c)
    if z_var == 0:
        return pd.DataFrame(0.0, index=df_rna.columns, columns=df_morph.columns)

    rna_ranks = df_rna.rank(axis=0).values.astype(np.float64)
    morph_ranks = df_morph.rank(axis=0).values.astype(np.float64)

    rna_c = rna_ranks - rna_ranks.mean(axis=0, keepdims=True)
    morph_c = morph_ranks - morph_ranks.mean(axis=0, keepdims=True)

    # Residualize ranks on z_c
    beta_rna = np.sum(z_c[:, np.newaxis] * rna_c, axis=0, keepdims=True) / z_var
    beta_morph = np.sum(z_c[:, np.newaxis] * morph_c, axis=0, keepdims=True) / z_var

    rna_res = rna_c - z_c[:, np.newaxis] * beta_rna
    morph_res = morph_c - z_c[:, np.newaxis] * beta_morph

    rna_ss = np.sqrt(np.sum(rna_res * rna_res, axis=0))
    morph_ss = np.sqrt(np.sum(morph_res * morph_res, axis=0))
    rna_ss[rna_ss == 0] = 1.0
    morph_ss[morph_ss == 0] = 1.0

    rna_u = rna_res / rna_ss[np.newaxis, :]
    morph_u = morph_res / morph_ss[np.newaxis, :]

    n_genes = rna_u.shape[1]
    n_morph = morph_u.shape[1]
    partial_rho = np.zeros((n_genes, n_morph), dtype=np.float64)
    for start in range(0, n_genes, 200):
        end = min(start + 200, n_genes)
        partial_rho[start:end, :] = np.sum(
            rna_u[:, start:end, np.newaxis] * morph_u[:, np.newaxis, :],
            axis=0
        )

    return pd.DataFrame(np.clip(partial_rho, -1.0, 1.0), index=df_rna.columns, columns=df_morph.columns)


def build_correlation_tables(rho_df, pval_df, fdr_df, partial_rho_df, n_cells):
    """Builds long-form pairwise and per-gene summary DataFrames."""
    genes = rho_df.index.values
    morphs = rho_df.columns.values

    g_grid, m_grid = np.meshgrid(genes, morphs, indexing="ij")
    pairs_df = pd.DataFrame({
        "gene": g_grid.ravel(),
        "morphology_feature": m_grid.ravel(),
        "spearman_rho": rho_df.values.ravel(),
        "abs_rho": np.abs(rho_df.values.ravel()),
        "p_value": pval_df.values.ravel(),
        "fdr_q_value": fdr_df.values.ravel(),
        "partial_spearman_rho_given_ploidy": partial_rho_df.values.ravel(),
        "n_cells": n_cells
    })
    pairs_df["morphology_category"] = pairs_df["morphology_feature"].map(categorize_morphology_feature)
    pairs_df.sort_values("abs_rho", ascending=False, inplace=True)
    pairs_df.reset_index(drop=True, inplace=True)

    # Per-gene summary (strongest morphology association per gene)
    best_morph_idx = np.argmax(np.abs(rho_df.values), axis=1)
    row_indices = np.arange(len(genes))
    best_rhos = rho_df.values[row_indices, best_morph_idx]
    best_pvals = pval_df.values[row_indices, best_morph_idx]
    best_fdrs = fdr_df.values[row_indices, best_morph_idx]
    best_partials = partial_rho_df.values[row_indices, best_morph_idx]
    best_morph_names = morphs[best_morph_idx]

    # Also compute mean |rho| across top 5 morphology features for each gene
    sorted_abs = np.sort(np.abs(rho_df.values), axis=1)[:, ::-1]
    mean_top5_abs_rho = sorted_abs[:, :min(5, sorted_abs.shape[1])].mean(axis=1)

    gene_summary_df = pd.DataFrame({
        "gene": genes,
        "best_morphology_feature": best_morph_names,
        "morphology_category": [categorize_morphology_feature(m) for m in best_morph_names],
        "spearman_rho": best_rhos,
        "abs_rho": np.abs(best_rhos),
        "mean_top5_abs_rho": mean_top5_abs_rho,
        "p_value": best_pvals,
        "fdr_q_value": best_fdrs,
        "neg_log10_fdr": -np.log10(np.clip(best_fdrs, 1e-300, 1.0)),
        "partial_spearman_rho_given_ploidy": best_partials,
        "n_sig_morph_features_fdr05": (fdr_df.values < 0.05).sum(axis=1),
        "n_cells": n_cells
    })
    gene_summary_df.sort_values("abs_rho", ascending=False, inplace=True)
    gene_summary_df.reset_index(drop=True, inplace=True)

    return pairs_df, gene_summary_df


def select_representative_scatter_pairs(pairs_df, highlight_genes=None, n_panels=12):
    """
    Selects diverse, high-correlation (gene, morphology_feature) pairs for the scatter grid:
      - Top positive Spearman correlations (unique genes across diverse morphology features)
      - Top negative Spearman correlations (unique genes)
      - Highlighted megakaryocyte marker genes (e.g., Kalrn, Pf4, Itga2b, Mpl) with their
        strongest correlated morphology feature.
    """
    selected_rows = []
    used_genes = set()

    # 1. Top positive correlated genes (up to 5 unique genes)
    pos_df = pairs_df[pairs_df["spearman_rho"] > 0].sort_values("spearman_rho", ascending=False)
    for _, row in pos_df.iterrows():
        if row["gene"] not in used_genes:
            selected_rows.append(row)
            used_genes.add(row["gene"])
        if len(selected_rows) >= 5:
            break

    # 2. Top negative correlated genes (up to 4 unique genes)
    neg_df = pairs_df[pairs_df["spearman_rho"] < 0].sort_values("spearman_rho", ascending=True)
    for _, row in neg_df.iterrows():
        if row["gene"] not in used_genes:
            selected_rows.append(row)
            used_genes.add(row["gene"])
        if len(selected_rows) >= 9:
            break

    # 3. Highlighted megakaryocyte markers (e.g. Kalrn, Pf4, Itga2b, Mpl)
    if highlight_genes:
        for hg in highlight_genes:
            hg_matches = pairs_df[pairs_df["gene"].str.lower() == hg.lower()]
            if not hg_matches.empty:
                best_row = hg_matches.iloc[0]
                if best_row["gene"] not in used_genes and len(selected_rows) < n_panels:
                    selected_rows.append(best_row)
                    used_genes.add(best_row["gene"])

    # 4. Fill remaining panels with highest |rho| unique genes
    for _, row in pairs_df.iterrows():
        if len(selected_rows) >= n_panels:
            break
        if row["gene"] not in used_genes:
            selected_rows.append(row)
            used_genes.add(row["gene"])

    return pd.DataFrame(selected_rows).reset_index(drop=True)


def plot_top_correlated_scatter_grid(
    df_rna,
    df_morph,
    qc_df,
    selected_pairs_df,
    out_dir
):
    """
    Generates Figure 1: Multi-panel scatter & regression grid of top Spearman-correlated
    gene-morphology pairs colored by Supervised_ploidy, labeling extreme cells and
    displaying Spearman rho, FDR q-value, and partial rho|ploidy.
    """
    os.makedirs(out_dir, exist_ok=True)
    setup_plot_style()

    n_panels = len(selected_pairs_df)
    ncols = 4
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.3 * nrows))
    axes = np.atleast_1d(axes).ravel()

    ploidy_series = qc_df.loc[df_rna.index, "Supervised_ploidy"].astype(str) if "Supervised_ploidy" in qc_df.columns else pd.Series("Unknown", index=df_rna.index)
    present_ploidies = [p for p in PLOIDY_ORDER if p in set(ploidy_series)]
    if not present_ploidies:
        present_ploidies = sorted(ploidy_series.unique())

    for idx, row in selected_pairs_df.iterrows():
        ax = axes[idx]
        gene = row["gene"]
        morph = row["morphology_feature"]
        rho = row["spearman_rho"]
        qval = row["fdr_q_value"]
        prho = row["partial_spearman_rho_given_ploidy"]

        x_vals = df_morph[morph].values
        y_vals = df_rna[gene].values

        plot_df = pd.DataFrame({
            "Morphology": x_vals,
            "Expression": y_vals,
            "Ploidy": ploidy_series.values,
            "CellID": df_rna.index.values
        })

        # Scatter points colored by Supervised Ploidy
        sns.scatterplot(
            data=plot_df,
            x="Morphology",
            y="Expression",
            hue="Ploidy",
            hue_order=present_ploidies,
            palette=PLOIDY_PALETTE,
            s=42,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.4,
            legend=(idx == 0),
            ax=ax
        )

        # Robust trend line across all cells
        sns.regplot(
            data=plot_df,
            x="Morphology",
            y="Expression",
            scatter=False,
            ax=ax,
            color="#222222",
            line_kws={"linestyle": "--", "linewidth": 1.6}
        )

        # Label top 2 extreme cells (highest & lowest product of standardized ranks)
        rx = stats.zscore(pd.Series(x_vals).rank())
        ry = stats.zscore(pd.Series(y_vals).rank())
        extremeness = rx * ry if rho >= 0 else -rx * ry
        top_cell_indices = [ int(np.argmax(y_vals)), int(np.argmax(extremeness)) ]
        for c_i in set(top_cell_indices):
            cid = str(plot_df["CellID"].iloc[c_i])
            ax.annotate(
                cid,
                xy=(x_vals[c_i], y_vals[c_i]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7.5,
                color="#333333",
                fontweight="medium",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7)
            )

        q_str = f"{qval:.2e}" if qval < 0.001 else f"{qval:.3f}"
        title_color = "#1b5e20" if rho >= 0 else "#b71c1c"
        ax.set_title(
            f"{gene} vs. {morph}\n"
            f"Spearman ρ = {rho:+.3f} (FDR q = {q_str} | partial ρ|ploidy = {prho:+.2f})",
            fontsize=9.0,
            fontweight="bold",
            color=title_color
        )
        ax.set_xlabel(f"{morph} (Z-score)", fontsize=8.5)
        ax.set_ylabel(f"{gene} Relative Expression (Z-score)", fontsize=8.5)
        ax.tick_params(labelsize=8)

        if idx == 0:
            ax.legend(title="Ploidy", fontsize=7.5, title_fontsize=8, loc="best", frameon=True)

    for extra_idx in range(n_panels, len(axes)):
        axes[extra_idx].axis("off")

    fig.suptitle(
        f"Morphology-to-Gene Spearman Rank Correlations Across Megakaryocyte Polyploidization (n = {len(df_rna)} QC-Passed Cells)",
        fontsize=14,
        fontweight="bold",
        y=0.995
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = os.path.join(out_dir, "06_top_gene_morphology_spearman_scatter_grid.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Plotting] Saved Top Correlated Gene-Morphology Scatter Grid to: {os.path.abspath(out_path)}")
    return out_path


def plot_labeled_high_correlation_genes(gene_summary_df, out_dir, top_n_label=25):
    """
    Generates Figure 2: Two-panel summary figure explicitly labeling genes with particularly
    high Spearman correlation to cellular morphology:
      - Panel A: Volcano-style peak Spearman correlation plot (Spearman rho vs -log10 FDR q-value)
                 colored by Morphology Feature Category, with labeled top positive/negative genes
                 and key MK markers.
      - Panel B: Horizontal lollipop/bar ranking of the Top 25 highest-|rho| genes annotated with
                 their exact best-correlated morphology parameter and Spearman rho.
    """
    os.makedirs(out_dir, exist_ok=True)
    setup_plot_style()

    fig, (ax_volc, ax_bar) = plt.subplots(
        1, 2, figsize=(18, 8.5), gridspec_kw={"width_ratios": [1.15, 1.0]}
    )

    # -------------------------------------------------------------------------
    # Panel A: Peak Spearman rho vs -log10(FDR q-value) with Gene Labels
    # -------------------------------------------------------------------------
    cat_palette = {
        "Cell Size & Width": "#1f77b4",
        "Total / Area Intensity": "#d62728",
        "Peak / Max Intensity": "#ff7f0e",
        "Shape Moments (Long/Short/Radial)": "#2ca02c",
        "Diffusivity & Eccentricity": "#9467bd",
        "Spatial CoM & Localization": "#17becf",
        "Other Flow / Imaging": "#7f7f7f"
    }
    present_cats = [c for c in cat_palette if c in set(gene_summary_df["morphology_category"])]

    sns.scatterplot(
        data=gene_summary_df,
        x="spearman_rho",
        y="neg_log10_fdr",
        hue="morphology_category",
        hue_order=present_cats,
        palette=cat_palette,
        s=45,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.3,
        ax=ax_volc
    )

    # Significance threshold lines
    fdr_05_y = -np.log10(0.05)
    ax_volc.axhline(fdr_05_y, color="#555555", linestyle="--", linewidth=1.1, label="FDR q = 0.05")
    ax_volc.axvline(0.30, color="#2e7d32", linestyle=":", linewidth=1.1)
    ax_volc.axvline(-0.30, color="#c62828", linestyle=":", linewidth=1.1)
    ax_volc.axvline(0.0, color="#999999", linestyle="-", linewidth=0.8, alpha=0.5)

    # Select genes to label: top positive, top negative, and key MK markers with |rho| >= 0.2
    pos_top = gene_summary_df[gene_summary_df["spearman_rho"] > 0].nlargest(11, "abs_rho")
    neg_top = gene_summary_df[gene_summary_df["spearman_rho"] < 0].nlargest(10, "abs_rho")
    mk_hits = gene_summary_df[
        gene_summary_df["gene"].str.lower().isin({m.lower() for m in KEY_MK_MARKERS})
        & (gene_summary_df["abs_rho"] >= 0.20)
    ]
    label_df = pd.concat([pos_top, neg_top, mk_hits], axis=0).drop_duplicates(subset=["gene"])

    # Place labels in two non-overlapping vertical stacks (negative side & positive side)
    y_max = float(gene_summary_df["neg_log10_fdr"].max())
    ax_volc.set_xlim(-0.46, 0.48)
    ax_volc.set_ylim(-0.3, y_max + 1.0)

    neg_labels = label_df[label_df["spearman_rho"] < 0].sort_values("neg_log10_fdr", ascending=False)
    pos_labels = label_df[label_df["spearman_rho"] >= 0].sort_values("neg_log10_fdr", ascending=False)

    neg_y_slots = np.linspace(min(y_max * 0.72, 7.6), 2.2, max(len(neg_labels), 1))
    for slot_y, (_, r) in zip(neg_y_slots, neg_labels.iterrows()):
        gx = float(r["spearman_rho"])
        gy = float(r["neg_log10_fdr"])
        gname = str(r["gene"])
        is_mk = gname.lower() in {m.lower() for m in KEY_MK_MARKERS}
        ax_volc.annotate(
            f"{gname} ({gx:+.2f})",
            xy=(gx, gy),
            xytext=(-0.06, slot_y),
            textcoords="data",
            ha="right",
            va="center",
            fontsize=8.2,
            fontweight="bold" if (is_mk or abs(gx) >= 0.30) else "normal",
            color="#0d47a1" if is_mk else "#b71c1c",
            arrowprops=dict(arrowstyle="->", color="#888888", lw=0.7, alpha=0.75, connectionstyle="arc3,rad=0.05"),
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#dddddd", lw=0.4, alpha=0.92)
        )

    pos_y_slots = np.linspace(y_max + 0.3, 3.0, max(len(pos_labels), 1))
    for slot_y, (_, r) in zip(pos_y_slots, pos_labels.iterrows()):
        gx = float(r["spearman_rho"])
        gy = float(r["neg_log10_fdr"])
        gname = str(r["gene"])
        is_mk = gname.lower() in {m.lower() for m in KEY_MK_MARKERS}
        ax_volc.annotate(
            f"{gname} ({gx:+.2f})",
            xy=(gx, gy),
            xytext=(0.06, slot_y),
            textcoords="data",
            ha="left",
            va="center",
            fontsize=8.2,
            fontweight="bold" if (is_mk or abs(gx) >= 0.28) else "normal",
            color="#0d47a1" if is_mk else "#1b5e20",
            arrowprops=dict(arrowstyle="->", color="#888888", lw=0.7, alpha=0.75, connectionstyle="arc3,rad=-0.05"),
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#dddddd", lw=0.4, alpha=0.92)
        )

    ax_volc.set_title(
        "A. Genome-Wide Peak Morphology Spearman Correlation per Gene\n"
        "(Labeled: High-Correlation Genes & Megakaryocyte Markers)",
        fontsize=12,
        fontweight="bold"
    )
    ax_volc.set_xlabel("Peak Spearman Rank Correlation (ρ) with Cellular Morphology", fontsize=10.5)
    ax_volc.set_ylabel("-log10(Benjamini-Hochberg FDR q-value)", fontsize=10.5)
    ax_volc.legend(title="Best Morphology Category", fontsize=8, title_fontsize=8.5, loc="upper left", frameon=True)

    # -------------------------------------------------------------------------
    # Panel B: Top N Genes Ranked by |Spearman rho| with Morphology Feature Labels
    # -------------------------------------------------------------------------
    top_bar_df = gene_summary_df.head(top_n_label).iloc[::-1].copy()
    bar_colors = ["#2e7d32" if val >= 0 else "#c62828" for val in top_bar_df["spearman_rho"]]

    y_pos = np.arange(len(top_bar_df))
    ax_bar.barh(
        y_pos,
        top_bar_df["spearman_rho"],
        color=bar_colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
        height=0.72
    )
    ax_bar.axvline(0.0, color="black", linewidth=0.9)

    y_labels = [
        f"{row['gene']}  <--  {row['best_morphology_feature']}"
        for _, row in top_bar_df.iterrows()
    ]
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(y_labels, fontsize=8.5, fontweight="bold")

    # Annotate exact rho values at bar tips
    for idx_b, (_, row) in enumerate(top_bar_df.iterrows()):
        val = float(row["spearman_rho"])
        pad = 0.012 if val >= 0 else -0.012
        ha = "left" if val >= 0 else "right"
        ax_bar.text(
            val + pad,
            idx_b,
            f"ρ={val:+.3f} (p|ploidy={row['partial_spearman_rho_given_ploidy']:+.2f})",
            va="center",
            ha=ha,
            fontsize=7.8,
            fontweight="medium",
            color="#111111"
        )

    max_abs = float(top_bar_df["abs_rho"].max())
    ax_bar.set_xlim(-max_abs - 0.22, max_abs + 0.25)
    ax_bar.set_title(
        f"B. Top {top_n_label} Genes with Highest Morphology Spearman |ρ|\n"
        "(Gene  <--  Strongest Correlated Morphology Feature)",
        fontsize=12,
        fontweight="bold"
    )
    ax_bar.set_xlabel("Spearman Rank Correlation (ρ) [Green = Positive, Red = Negative]", fontsize=10.5)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "06_gene_morphology_spearman_ranking_labeled.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Plotting] Saved Labeled High-Correlation Gene Ranking Figure to: {os.path.abspath(out_path)}")
    return out_path


def plot_morphology_gene_heatmap(rho_df, fdr_df, gene_summary_df, out_dir, n_top_genes=30, n_top_morph=25):
    """
    Generates Figure 3: Clustered heatmap of Spearman rank correlations (rho) between the
    top correlated genes (plus key MK markers) and top imaging morphology parameters,
    annotating statistically significant cells (FDR < 0.01).
    """
    os.makedirs(out_dir, exist_ok=True)
    setup_plot_style()

    top_genes = list(gene_summary_df.head(n_top_genes)["gene"].values)
    for mk in KEY_MK_MARKERS:
        matches = [g for g in rho_df.index if g.lower() == mk.lower()]
        if matches and matches[0] not in top_genes:
            top_genes.append(matches[0])

    # Select top morphology features with highest max |rho| across genes
    morph_importance = rho_df.loc[top_genes].abs().max(axis=0).sort_values(ascending=False)
    top_morphs = list(morph_importance.head(min(n_top_morph, len(morph_importance))).index)

    sub_rho = rho_df.loc[top_genes, top_morphs]
    sub_fdr = fdr_df.loc[top_genes, top_morphs]

    # Hierarchical ordering of rows and columns using correlation distance
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist

    if sub_rho.shape[0] > 2:
        row_order = leaves_list(linkage(pdist(sub_rho.values, metric="euclidean"), method="ward"))
    else:
        row_order = np.arange(sub_rho.shape[0])

    if sub_rho.shape[1] > 2:
        col_order = leaves_list(linkage(pdist(sub_rho.values.T, metric="euclidean"), method="ward"))
    else:
        col_order = np.arange(sub_rho.shape[1])

    sub_rho = sub_rho.iloc[row_order, col_order]
    sub_fdr = sub_fdr.iloc[row_order, col_order]

    # Significance star annotations (* FDR < 0.05, ** FDR < 0.01, *** FDR < 0.001)
    annot_mat = np.full(sub_rho.shape, "", dtype=object)
    for i in range(sub_rho.shape[0]):
        for j in range(sub_rho.shape[1]):
            q = sub_fdr.iat[i, j]
            r = sub_rho.iat[i, j]
            if q < 0.001 and abs(r) >= 0.25:
                annot_mat[i, j] = f"{r:+.2f}*"
            elif abs(r) >= 0.30:
                annot_mat[i, j] = f"{r:+.2f}"

    fig, ax = plt.subplots(figsize=(15, 11))
    sns.heatmap(
        sub_rho,
        cmap="RdBu_r",
        center=0.0,
        vmin=-0.55,
        vmax=0.55,
        annot=annot_mat,
        fmt="",
        annot_kws={"size": 6.8, "weight": "bold"},
        linewidths=0.3,
        linecolor="#eeeeee",
        cbar_kws={"label": "Spearman Rank Correlation (ρ)"},
        ax=ax
    )

    ax.set_title(
        "Morphology-to-Gene Spearman Rank Correlation Matrix (Top Correlated Genes & MK Markers)\n"
        "(Annotated values indicate |ρ| ≥ 0.25 with * for Benjamini-Hochberg FDR q < 0.001)",
        fontsize=12.5,
        fontweight="bold"
    )
    ax.set_xlabel("Imaging Flow Cytometry & Scatter Morphology Parameters", fontsize=10.5)
    ax.set_ylabel("High-Correlation Genes & Megakaryocyte Markers", fontsize=10.5)
    plt.xticks(rotation=45, ha="right", fontsize=8.5)
    plt.yticks(rotation=0, fontsize=8.5)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "06_morphology_gene_spearman_heatmap.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Plotting] Saved Morphology-Gene Spearman Correlation Heatmap to: {os.path.abspath(out_path)}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="06: Morphology-to-Gene Spearman Rank Correlation Analysis (Backwards Compatible)"
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default=None,
        help="Path to unified or legacy metadata CSV (defaults to Metadata_merged_YL_SF_20.10.26_v2.csv or legacy fallback)"
    )
    parser.add_argument(
        "--qc_meta",
        type=str,
        default="output/qc_filtered_metadata.csv",
        help="Path to pre-filtered QC metadata CSV from Step 01"
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default=os.path.join(DEFAULT_TRAIN_DIR, "expression"),
        help="Path to primary 10x expression directory"
    )
    parser.add_argument(
        "--pilot_exp_dir",
        type=str,
        default=None,
        help="Optional path to secondary/pilot 10x expression directory"
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default=DEFAULT_TRAIN_DIR,
        help="Legacy train directory fallback"
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="exp",
        help="Legacy test/pilot directory fallback"
    )
    parser.add_argument(
        "--cohort",
        type=str,
        default="all",
        help="Cohort filter: 'all' (default), '10plate' (M1-M5), or '4plate' (M6-M7)"
    )
    parser.add_argument(
        "--highlight_genes",
        type=str,
        default="Kalrn,Ppbp,Pde3a,Mylk,Rock1,Exoc6,Pf4,Itga2b,Mpl,Vwf",
        help="Comma-separated marker genes to highlight in correlation plots"
    )
    parser.add_argument(
        "--raw_ranks",
        action="store_true",
        help="Use unadjusted expression ranks without residualizing per-cell library depth"
    )
    parser.add_argument(
        "--n_panels",
        type=int,
        default=12,
        help="Number of scatter panels in the top gene-morphology scatter grid (default: 12)"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="output",
        help="Output directory for CSV tables and figures"
    )
    parser.add_argument(
        "--min_mapped_reads",
        type=int,
        default=20000,
        help="Minimum kallisto.nb_mapped reads per cell if fallback QC is run (default: 20000)"
    )
    parser.add_argument(
        "--min_map_pc",
        type=float,
        default=50.0,
        help="Minimum kallisto.pc_map mapping percentage per cell if fallback QC is run (default: 50.0)"
    )
    parser.add_argument(
        "--min_cells_per_gene",
        type=int,
        default=0,
        help="Minimum number of cells expressing a gene (> 0 counts) before HVG selection (default: 0, disabled)"
    )
    parser.add_argument(
        "--n_hvg",
        type=int,
        default=2000,
        help="Number of highly variable genes to select from 10x matrices (default: 2000)"
    )
    parser.add_argument(
        "--hvg_method",
        type=str,
        default="default",
        choices=["default", "raw_var", "log1p_var", "seurat", "seurat_v3"],
        help="HVG selection method ('default' uses raw_var for Step 06)"
    )
    parser.add_argument(
        "--exclude_cell_cycle",
        action="store_true",
        help="Exclude cited Tirosh et al. (2016) / Seurat cc.genes.updated.2019 cell cycle genes before correlation analysis"
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
    print("==========================================================================")
    print("   STEP 06: MORPHOLOGY-TO-GENE SPEARMAN CORRELATION ANALYSIS              ")
    print("==========================================================================")

    # 1. Load QC-passed metadata (unified M1-M7 or legacy fallback)
    qc_df = load_qc_metadata_backwards_compatible(
        metadata_file=args.metadata_file,
        qc_meta_path=args.qc_meta,
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        cohort_filter=args.cohort,
        min_mapped_reads=args.min_mapped_reads,
        min_map_pc=args.min_map_pc,
    )

    # 2. Load gene expression aligned to QC-passed cells across available sources
    df_rna_raw, cohort_sources = load_expression_backwards_compatible(
        qc_df=qc_df,
        exp_dir=args.exp_dir,
        pilot_exp_dir=args.pilot_exp_dir,
        n_hvg=args.n_hvg,
        hvg_method=args.hvg_method,
        min_cells_per_gene=args.min_cells_per_gene,
    )

    # Optional cited cell-cycle and/or custom gene exclusion (applied prior to depth adjustment & Spearman correlation)
    if args.exclude_cell_cycle or args.exclude_genes_file or args.exclude_genes:
        from cell_cycle import filter_excluded_genes
        audit_path = os.path.join(args.out_dir, "06_excluded_genes_audit.csv")
        keep_mask, _ = filter_excluded_genes(
            df_rna_raw.columns,
            exclude_cell_cycle=args.exclude_cell_cycle,
            exclude_genes_file=args.exclude_genes_file,
            exclude_genes=args.exclude_genes,
            audit_csv_path=audit_path,
        )
        df_rna_raw = df_rna_raw.loc[:, keep_mask].copy()
    else:
        print("[Gene Filter] No additional gene exclusions applied (original notebook baseline).")

    # Align metadata & morphology features to the matched expression cells
    qc_aligned = qc_df.loc[df_rna_raw.index].copy()
    df_morph_raw = extract_morphology_features(qc_aligned)

    # Harmonize across cohorts and adjust for per-cell library depth bias (unless --raw_ranks is set)
    df_rna, df_morph = harmonize_and_adjust_modalities(
        df_rna_raw,
        df_morph_raw,
        cohort_sources=cohort_sources,
        adjust_library_depth=not args.raw_ranks
    )

    # 3. Compute Spearman rank correlations, BH-FDR q-values, and partial Spearman | ploidy
    print(f"[Spearman] Computing pairwise Spearman correlations for {df_rna.shape[1]} genes x {df_morph.shape[1]} morphology features (n = {len(df_rna)} cells)...")
    rho_df, pval_df, fdr_df = compute_spearman_matrix_no_blas(df_rna, df_morph)
    partial_rho_df = compute_partial_spearman_given_ploidy(
        df_rna,
        df_morph,
        qc_aligned.get("Supervised_ploidy", pd.Series("Unknown", index=qc_aligned.index))
    )

    pairs_df, gene_summary_df = build_correlation_tables(
        rho_df, pval_df, fdr_df, partial_rho_df, n_cells=len(df_rna)
    )

    # Export CSV tables
    top_pairs_csv = os.path.join(args.out_dir, "morphology_gene_spearman_top_pairs.csv")
    pairs_df.head(500).to_csv(top_pairs_csv, index=False)

    gene_summary_csv = os.path.join(args.out_dir, "morphology_gene_spearman_per_gene_summary.csv")
    gene_summary_df.to_csv(gene_summary_csv, index=False)

    print(f"[Export] Saved top 500 gene-morphology Spearman pairs to: {os.path.abspath(top_pairs_csv)}")
    print(f"[Export] Saved per-gene strongest morphology correlation summary to: {os.path.abspath(gene_summary_csv)}")

    print("\n--- TOP 20 GENES BY STRONGEST SPEARMAN CORRELATION WITH MORPHOLOGY ---")
    display_cols = [
        "gene", "best_morphology_feature", "spearman_rho",
        "fdr_q_value", "partial_spearman_rho_given_ploidy", "morphology_category"
    ]
    print(gene_summary_df[display_cols].head(20).to_string(index=False))

    highlight_list = [g.strip() for g in args.highlight_genes.split(",") if g.strip()]
    mk_sub = gene_summary_df[gene_summary_df["gene"].str.lower().isin({h.lower() for h in highlight_list})]
    if not mk_sub.empty:
        print("\n--- KEY MEGAKARYOCYTE MARKER GENE CORRELATIONS ---")
        print(mk_sub[display_cols].to_string(index=False))

    # 4. Generate publication figures
    selected_pairs_df = select_representative_scatter_pairs(
        pairs_df,
        highlight_genes=highlight_list,
        n_panels=args.n_panels
    )
    plot_top_correlated_scatter_grid(df_rna, df_morph, qc_aligned, selected_pairs_df, args.out_dir)
    plot_labeled_high_correlation_genes(gene_summary_df, args.out_dir, top_n_label=25)
    plot_morphology_gene_heatmap(rho_df, fdr_df, gene_summary_df, args.out_dir)

    print("==========================================================================")
    print("   STEP 06 COMPLETED SUCCESSFULLY                                         ")
    print("==========================================================================")


if __name__ == "__main__":
    main()
