"""
05_mofa_multimodal_analysis.py
------------------------------
Trains and evaluates Multi-Omics Factor Analysis (MOFA2) models integrating single-cell
transcriptomics (RNA) and imaging flow cytometry (Morphology) data.

Workflow:
1. Ingests transcriptomics (top 2000 HVGs selected on log1p-normalized expression) and morphology/flow metadata.
2. Applies likelihood-appropriate normalization/standardization per view:
   - RNA view (default: 'poisson'): non-negative integer counts (raw counts or library size-factor normalized counts, without StandardScaler centering).
   - Morphology view (default: 'gaussian'): z-score standardized via StandardScaler (with optional signed log1p transformation).
3. Sets up and trains mofapy2 entry point model with view-specific likelihoods (default: ['poisson', 'gaussian'], 10 factors, seed=42).
4. Saves/loads output/mofa_model_rna_morphology.hdf5 model file.
5. Computes R2 / variance decomposition per view using mofax.
6. Plots factor loadings and factor scatter projections colored by Supervised Ploidy.

Usage:
    python 05_mofa_multimodal_analysis.py --exp_dir "data/expression" --out_dir "output" --rna_likelihood poisson
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

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False


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
    mofa_hdf5_path=None,
    n_hvg=2000,
    hvg_method="default",
    min_cells_per_gene=0,
    rna_likelihood="poisson",
    morph_likelihood="gaussian",
    rna_norm="raw_counts",
    log_morph=False,
    depth_norm="cp10k_log1p",
    regress_cell_cycle="none",
    adjust_library_depth=False,
    exclude_mt=False,
    exclude_cell_cycle=False,
    exclude_genes_file=None,
    exclude_genes=None,
    out_dir=None,
):
    """
    Ingests, aligns, and normalizes RNA expression and Morphology data across all available
    QC-passed cells (e.g. all 393 cells across M1-M7 by combining 10x matrices and
    pre-saved MOFA HDF5 expression when needed) according to the target MOFA2 view likelihoods.
    Returns (proc_rna, proc_img, supervised_ploidy, mouse_series, effective_rna_likelihood).
    """
    from importlib import import_module
    corr_mod = import_module("04_morphology_gene_correlations")

    # Load QC metadata for morphology features
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
            raise FileNotFoundError("Metadata table not found for morphology features.")

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
        step_prefix="05_mofa",
    )

    # Optional cited cell-cycle and/or custom gene exclusion (after CC scoring/regression)
    if exclude_cell_cycle or exclude_genes_file or exclude_genes:
        from cell_cycle import filter_excluded_genes
        audit_path = os.path.join(out_dir, "05_mofa_excluded_genes_audit.csv") if out_dir else None
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
        print("[Gene Filter] MOFA2 RNA View: No additional gene exclusions applied (baseline).")

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

    # Multi-cohort z-scored RNA, CDR residualization, or cell-cycle regression requires Gaussian likelihood
    effective_rna_likelihood = rna_likelihood
    is_prescaled = bool((df_rna.values < 0).any())
    if regress_cell_cycle != "none" or is_prescaled or adjust_library_depth:
        if rna_likelihood != "gaussian":
            reasons = []
            if regress_cell_cycle != "none":
                reasons.append(f"cell-cycle regression ('{regress_cell_cycle}')")
            if adjust_library_depth:
                reasons.append("library-depth/CDR residualization")
            if is_prescaled and not reasons:
                reasons.append("multi-cohort z-scored HDF5 + 10x alignment")
            print(
                f"[MOFA2] Warning: Switching RNA likelihood '{rna_likelihood}' -> 'gaussian' "
                f"({'; '.join(reasons)} produces continuous values incompatible with {rna_likelihood})."
            )
            effective_rna_likelihood = "gaussian"

    # Harmonize morphology across cohorts, optionally residualize per-cell library depth (Finak et al., 2015),
    # and enforce cell-cycle regression strictly after depth adjustment
    df_rna_h, img_df_h = corr_mod.harmonize_and_adjust_modalities(
        df_rna,
        img_df,
        cohort_sources,
        adjust_library_depth=adjust_library_depth,
        regress_cell_cycle=regress_cell_cycle,
    )
    if adjust_library_depth:
        print("[MOFA2] Applied per-cohort library-depth / CDR residualization to RNA view (Finak et al., 2015).")

    if effective_rna_likelihood == "gaussian":
        scaler_rna = StandardScaler()
        proc_rna = pd.DataFrame(
            scaler_rna.fit_transform(df_rna_h),
            index=df_rna_h.index,
            columns=df_rna_h.columns,
        )
        print("RNA view prepared for 'gaussian' likelihood: cohort-harmonized z-scores.")
    else:
        proc_rna = normalize_rna_for_likelihood(
            df_rna_h,
            likelihood=effective_rna_likelihood,
            rna_norm=rna_norm
        )

    proc_img = normalize_morphology_for_likelihood(
        img_df_h,
        likelihood=morph_likelihood,
        log_morph=log_morph
    )

    supervised_ploidy = meta_df.get("Supervised_ploidy", pd.Series("Unknown", index=common_cells))
    if "mouse" in meta_df.columns:
        mouse_series = meta_df["mouse"].fillna("Unknown").astype(str)
    elif "Mouse_Plate" in meta_df.columns:
        extracted = meta_df["Mouse_Plate"].astype(str).str.extract(r'^(?:Mouse|M)?(\d+)', expand=False, flags=re.IGNORECASE)
        mouse_series = extracted.apply(lambda x: f"M{x}" if pd.notna(x) else "Unknown")
    else:
        mouse_series = pd.Series("Unknown", index=common_cells)

    n_mice = mouse_series[mouse_series != "Unknown"].nunique()
    print(
        f"[MOFA2 Data] Loaded {len(common_cells)} aligned QC-passed cells"
        + (f" across {n_mice} animals" if n_mice else "")
        + f" | {proc_rna.shape[1]} genes | {proc_img.shape[1]} morphology features."
    )

    return proc_rna, proc_img, supervised_ploidy, mouse_series, effective_rna_likelihood


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
        likelihoods=likelihoods,
        views_names=["RNA", "Morphology"],
        groups_names=["group1"],
        samples_names=[df_rna.index.tolist()],
        features_names=[df_rna.columns.tolist(), df_morphology.columns.tolist()]
    )

    # scale_views=False is required when using non-Gaussian likelihoods in mofapy2
    ent.set_data_options(scale_groups=False, scale_views=False)
    ent.set_model_options(
        factors=factors,
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


def parse_factor_counts(factors_arg):
    """
    Parses a single factor count (e.g. '10') or comma-separated list (e.g. '5,10,15')
    into a deduplicated list of positive integers.
    """
    counts = []
    for tok in str(factors_arg).split(","):
        tok = tok.strip()
        if not tok:
            continue
        k = int(tok)
        if k <= 0:
            raise ValueError(f"Factor count must be a positive integer, got: {k}")
        if k not in counts:
            counts.append(k)
    if not counts:
        raise ValueError("At least one positive integer factor count must be provided to --factors.")
    return counts


def _decode_h5_strings(arr):
    """Decodes HDF5 byte-string arrays to plain Python strings."""
    return [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in arr]


def compute_mofa_reconstruction_errors(
    model_path,
    proc_rna,
    proc_img,
    ploidy_series,
    mouse_series,
    rna_likelihood="poisson",
    morph_likelihood="gaussian",
    factors_k=None,
):
    """
    Computes likelihood-aware per-cell MOFA reconstruction error (RMSE) across views
    (ported and generalized from Cell 50 of the original morphotrans2 notebook):
      - gaussian  : RMSE(X, Z W^T) in z-score units
      - poisson   : RMSE(log1p(X), log1p(softplus(Z W^T))) using MOFA2's softplus link
      - bernoulli : RMSE(X, sigmoid(Z W^T))
    """
    if not (H5PY_AVAILABLE and os.path.exists(model_path)):
        print("h5py not available or HDF5 model file not found. Skipping reconstruction error computation.")
        return pd.DataFrame()

    view_inputs = {
        "RNA": (proc_rna, rna_likelihood),
        "Morphology": (proc_img, morph_likelihood),
    }

    records = []
    with h5py.File(model_path, "r") as f:
        if "expectations" not in f or "Z" not in f["expectations"] or "W" not in f["expectations"]:
            print("Warning: MOFA HDF5 missing expectations/Z or expectations/W; skipping reconstruction error.")
            return pd.DataFrame()

        # Load Z (N_cells x K)
        z_group = f["expectations"]["Z"]
        group_key = "group1" if "group1" in z_group else list(z_group.keys())[0]
        Z = z_group[group_key][:]

        # Sample names
        if "samples" in f and group_key in f["samples"]:
            sample_names = _decode_h5_strings(f["samples"][group_key][:])
        else:
            sample_names = proc_rna.index.tolist()

        if Z.shape[0] != len(sample_names) and Z.shape[1] == len(sample_names):
            Z = Z.T  # (N_cells, K)
        n_cells, k_active = Z.shape
        k_val = int(factors_k) if factors_k is not None else int(k_active)

        ploidy_aligned = ploidy_series.reindex(sample_names).fillna("Unknown").astype(str).str.strip().str.lower().values
        mouse_aligned = mouse_series.reindex(sample_names).fillna("Unknown").astype(str).values

        for view_name in f["expectations"]["W"].keys():
            W_v = f["expectations"]["W"][view_name][:]
            if W_v.shape[0] == k_active and W_v.shape[1] != k_active:
                W_v = W_v.T  # (D_v, K)

            # Align observed feature matrix X_v (N_cells x D_v)
            feat_names = None
            if "features" in f and view_name in f["features"]:
                feat_names = _decode_h5_strings(f["features"][view_name][:])

            df_in, likelihood = view_inputs.get(view_name, (None, "gaussian"))
            X_v = None
            if df_in is not None:
                if feat_names is not None and set(feat_names).issubset(df_in.columns):
                    X_v = df_in.reindex(index=sample_names, columns=feat_names).fillna(0.0).values
                elif df_in.shape == (n_cells, W_v.shape[0]):
                    X_v = df_in.reindex(index=sample_names).fillna(0.0).values
            if X_v is None and "data" in f and view_name in f["data"]:
                d_node = f["data"][view_name]
                X_raw = d_node[group_key][:] if isinstance(d_node, h5py.Group) and group_key in d_node else d_node[:]
                if X_raw.shape[0] == W_v.shape[0] and X_raw.shape[1] == n_cells:
                    X_raw = X_raw.T
                X_v = np.nan_to_num(X_raw, nan=0.0)

            if X_v is None or X_v.shape != (n_cells, W_v.shape[0]):
                print(f"Warning: Could not align data matrix for view '{view_name}'; skipping RMSE.")
                continue

            eta = Z @ W_v.T  # Linear predictor (N_cells, D_v)
            if likelihood == "poisson":
                rate_hat = np.logaddexp(0.0, np.clip(eta, -30.0, 30.0))
                rmse = np.sqrt(np.mean((np.log1p(np.clip(X_v, 0.0, None)) - np.log1p(rate_hat)) ** 2, axis=1))
                metric_label = "RMSE (log1p Poisson rate)"
            elif likelihood == "bernoulli":
                prob_hat = 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))
                rmse = np.sqrt(np.mean((X_v - prob_hat) ** 2, axis=1))
                metric_label = "RMSE (Bernoulli prob)"
            else:
                rmse = np.sqrt(np.mean((X_v - eta) ** 2, axis=1))
                metric_label = "RMSE (Gaussian z-score)"

            for i in range(n_cells):
                records.append({
                    "SampleID": sample_names[i],
                    "Supervised_ploidy": ploidy_aligned[i],
                    "mouse": mouse_aligned[i],
                    "Modality": view_name,
                    "Likelihood": likelihood,
                    "Error_Metric": metric_label,
                    "RMSE": float(rmse[i]),
                    "Factors_K": k_val,
                })

    return pd.DataFrame(records)


def plot_mofa_heterogeneity_errors(rmse_df, out_dir, suffix=""):
    """
    Plots per-cell MOFA reconstruction error (heterogeneity) across views,
    stratified by Supervised Ploidy and Animal (mouse), and saves the per-cell CSV.
    """
    if rmse_df.empty:
        return

    os.makedirs(out_dir, exist_ok=True)
    setup_plot_style()

    csv_path = os.path.join(out_dir, f"05_mofa_reconstruction_error_per_cell{suffix}.csv")
    rmse_df.to_csv(csv_path, index=False)
    print(f"Saved per-cell MOFA reconstruction errors to: {os.path.abspath(csv_path)}")

    title_suffix = f" ({suffix.lstrip('_').upper()})" if suffix else ""
    ploidy_order = [p for p in ["2n", "4n", "8n", "16n", "32n"] if p in rmse_df["Supervised_ploidy"].unique()]
    mice = sorted([m for m in rmse_df["mouse"].unique() if m != "Unknown"])
    show_mouse_panel = len(mice) > 1

    n_cols = 2 if show_mouse_panel else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(7.5 * n_cols, 5.5))
    if n_cols == 1:
        axes = [axes]

    # Panel A: Heterogeneity by Ploidy across Views
    ax0 = axes[0]
    plot_ploidy = rmse_df[rmse_df["Supervised_ploidy"].isin(ploidy_order)] if ploidy_order else rmse_df
    sns.boxplot(
        data=plot_ploidy,
        x="Modality",
        y="RMSE",
        hue="Supervised_ploidy" if ploidy_order else None,
        hue_order=ploidy_order if ploidy_order else None,
        palette="viridis",
        width=0.65,
        fliersize=2.5,
        ax=ax0,
    )
    ax0.set_title(f"MOFA Reconstruction Error Heterogeneity by Ploidy{title_suffix}", fontsize=11, fontweight="bold")
    ax0.set_xlabel("Omics Modality / View")
    ax0.set_ylabel("Per-Cell Reconstruction Error (RMSE)")
    if ploidy_order:
        ax0.legend(title="Ploidy", frameon=True, loc="upper right")

    # Panel B: Heterogeneity by Animal (Mouse) across Views
    if show_mouse_panel:
        ax1 = axes[1]
        plot_mouse = rmse_df[rmse_df["mouse"].isin(mice)]
        sns.boxplot(
            data=plot_mouse,
            x="Modality",
            y="RMSE",
            hue="mouse",
            hue_order=mice,
            palette="Set2",
            width=0.65,
            fliersize=2.5,
            ax=ax1,
        )
        ax1.set_title(f"MOFA Reconstruction Error Heterogeneity by Animal{title_suffix}", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Omics Modality / View")
        ax1.set_ylabel("Per-Cell Reconstruction Error (RMSE)")
        ax1.legend(title="Animal", frameon=True, loc="upper right", ncol=2)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"05_mofa_reconstruction_error_heterogeneity{suffix}.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved MOFA heterogeneity error plot to: {os.path.abspath(fig_path)}")


def plot_mofa_factor_sweep_comparison(sweep_df, all_rmse_df, out_dir):
    """
    Plots total variance explained (R^2 %) and per-cell reconstruction error (RMSE)
    across the swept factor counts K (e.g. K = 5, 10, 15).
    """
    if sweep_df.empty and all_rmse_df.empty:
        return

    os.makedirs(out_dir, exist_ok=True)
    setup_plot_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left panel: Total R^2 (%) per view vs K
    ax0 = axes[0]
    r2_cols = [c for c in sweep_df.columns if c.startswith("Total_R2_")]
    if r2_cols:
        for col in r2_cols:
            view_label = col.replace("Total_R2_", "")
            ax0.plot(
                sweep_df["Factors_K"],
                sweep_df[col],
                marker="o",
                linewidth=2.2,
                markersize=7,
                label=view_label,
            )
        ax0.set_xticks(sweep_df["Factors_K"].tolist())
        ax0.set_xlabel("Number of MOFA Factors (K)")
        ax0.set_ylabel("Total Variance Explained (R^2 %)")
        ax0.set_title("MOFA Factor Sweep: Variance Explained (R^2) vs. K", fontsize=12, fontweight="bold")
        ax0.legend(title="View", frameon=True)
    else:
        ax0.text(0.5, 0.5, "R^2 summary unavailable", ha="center", va="center")
        ax0.axis("off")

    # Right panel: Per-cell Reconstruction Error (RMSE) across K
    ax1 = axes[1]
    if not all_rmse_df.empty:
        df_plot = all_rmse_df.copy()
        df_plot["Factors_K_str"] = df_plot["Factors_K"].astype(str)
        k_order = [str(k) for k in sorted(all_rmse_df["Factors_K"].unique())]
        sns.boxplot(
            data=df_plot,
            x="Factors_K_str",
            y="RMSE",
            hue="Modality",
            order=k_order,
            palette="Set1",
            width=0.55,
            fliersize=2.5,
            ax=ax1,
        )
        ax1.set_xlabel("Number of MOFA Factors (K)")
        ax1.set_ylabel("Per-Cell Reconstruction Error (RMSE)")
        ax1.set_title("MOFA Factor Sweep: Reconstruction Error Heterogeneity vs. K", fontsize=12, fontweight="bold")
        ax1.legend(title="View", frameon=True)
    else:
        ax1.text(0.5, 0.5, "Reconstruction error unavailable", ha="center", va="center")
        ax1.axis("off")

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "05_mofa_factor_sweep_error_and_r2.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved MOFA factor sweep comparison plot to: {os.path.abspath(fig_path)}")


def evaluate_mofa_model(
    model_path,
    ploidy_series,
    out_dir,
    suffix="",
    proc_rna=None,
    proc_img=None,
    mouse_series=None,
    rna_likelihood="poisson",
    morph_likelihood="gaussian",
    factors_k=None,
):
    """
    Computes R2 variance decomposition, factor projections, and per-cell reconstruction
    error heterogeneity plots for a trained MOFA2 model.
    Returns (summary_dict, rmse_df).
    """
    os.makedirs(out_dir, exist_ok=True)
    setup_plot_style()
    summary = {}
    rmse_df = pd.DataFrame()

    if not os.path.exists(model_path):
        print("HDF5 model file not found. Skipping MOFA evaluation.")
        return summary, rmse_df

    title_suffix = f" ({suffix.lstrip('_').upper()})" if suffix else ""

    if MOFAX_AVAILABLE:
        print(f"Loading trained MOFA model from: {model_path}")
        model = fx.mofa_model(model_path)

        # 1. Plot Variance Explained (R2 Matrix)
        try:
            r2_raw = model.get_r2()
            if {"Factor", "View", "R2"}.issubset(r2_raw.columns):
                r2_pivot = r2_raw.pivot_table(index="Factor", columns="View", values="R2", aggfunc="mean")
                factor_order = sorted(
                    r2_pivot.index,
                    key=lambda x: int(re.sub(r"\D+", "", str(x)) or 0)
                )
                r2 = r2_pivot.reindex(factor_order)
            else:
                r2 = r2_raw.apply(pd.to_numeric, errors="coerce")

            for col in r2.columns:
                summary[f"Total_R2_{col}"] = float(r2[col].sum())
            fig, ax = plt.subplots(figsize=(8, max(5, 0.4 * len(r2) + 2)))
            sns.heatmap(r2, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
            ax.set_title(f"MOFA Variance Explained (R^2 %) Per View{title_suffix}", fontsize=12, fontweight="bold")
            plt.tight_layout()
            r2_path = os.path.join(out_dir, f"05_mofa_r2_variance_explained{suffix}.png")
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
            ax.set_title(f"MOFA Factor Projection: Factor 1 vs Factor 2{title_suffix}", fontsize=13, fontweight="bold")
            ax.set_xlabel("Factor 1")
            ax.set_ylabel("Factor 2")
            plt.tight_layout()
            factor_png = os.path.join(out_dir, f"05_mofa_factor1_vs_factor2{suffix}.png")
            plt.savefig(factor_png)
            plt.close()
            print(f"Saved MOFA factor scatter plot to: {os.path.abspath(factor_png)}")
        except Exception as e:
            print(f"Could not plot factor scatter: {e}")
    else:
        print("mofax not available; skipping R2 heatmap and factor scatter.")

    # 3. Per-Cell Reconstruction Error & Heterogeneity Plots
    if proc_rna is not None and proc_img is not None:
        if mouse_series is None:
            mouse_series = pd.Series("Unknown", index=ploidy_series.index)
        try:
            rmse_df = compute_mofa_reconstruction_errors(
                model_path=model_path,
                proc_rna=proc_rna,
                proc_img=proc_img,
                ploidy_series=ploidy_series,
                mouse_series=mouse_series,
                rna_likelihood=rna_likelihood,
                morph_likelihood=morph_likelihood,
                factors_k=factors_k,
            )
            if not rmse_df.empty:
                for view_name, grp in rmse_df.groupby("Modality"):
                    summary[f"Mean_RMSE_{view_name}"] = float(grp["RMSE"].mean())
                    summary[f"Median_RMSE_{view_name}"] = float(grp["RMSE"].median())
                    summary[f"IQR_RMSE_{view_name}"] = float(grp["RMSE"].quantile(0.75) - grp["RMSE"].quantile(0.25))
                plot_mofa_heterogeneity_errors(rmse_df, out_dir, suffix=suffix)
        except Exception as e:
            print(f"Could not compute/plot MOFA reconstruction error heterogeneity: {e}")

    return summary, rmse_df


def main():
    parser = argparse.ArgumentParser(description="05: Multi-Omics Factor Analysis (MOFA2) Integration")
    parser.add_argument("--exp_dir", type=str, default=os.path.join("data", "expression"), help="10x expression directory (falls back to data/exp)")
    parser.add_argument("--out_dir", type=str, default="output", help="Directory to save output files")
    parser.add_argument(
        "--factors",
        type=str,
        default="10",
        help="Number of latent factors, or comma-separated list to sweep (e.g. '5,10,15'; default: '10')"
    )
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
        help="HVG selection method ('default' uses log1p_var for Step 05)"
    )
    parser.add_argument(
        "--min_cells_per_gene",
        type=int,
        default=0,
        help="Minimum number of cells expressing a gene (> 0 counts) required before HVG selection (default: 0, disabled)"
    )
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
        "--mofa_hdf5",
        type=str,
        default=None,
        help="Optional pre-saved MOFA HDF5 path for M1-M5 expression alignment (auto-discovered in data/ if omitted)"
    )
    parser.add_argument(
        "--depth_norm",
        type=str,
        default="cp10k_log1p",
        choices=["cp10k_log1p", "none"],
        help="Library-size depth normalization before cell-cycle regression & HVG selection (default: cp10k_log1p)"
    )
    parser.add_argument(
        "--regress_cell_cycle",
        type=str,
        default="none",
        choices=["none", "full", "difference"],
        help="Cell-cycle regression after depth normalization: 'none' (default), 'full' (S + G2M), or 'difference' (S - G2M); automatically switches RNA likelihood to gaussian"
    )
    parser.add_argument(
        "--adjust_library_depth",
        action="store_true",
        help="Apply per-cohort library-depth / Cellular Detection Rate (CDR) residualization to RNA view (Finak et al., 2015)"
    )
    parser.add_argument(
        "--reuse_model",
        action="store_true",
        help="Reuse existing HDF5 model file if present instead of retraining"
    )
    parser.add_argument(
        "--exclude_mt",
        action="store_true",
        help="Exclude mitochondrial genes (starting with 'mt-' or 'MT-'); off by default"
    )
    parser.add_argument(
        "--exclude_cell_cycle",
        action="store_true",
        help="Exclude cited Tirosh et al. (2016) / Seurat cc.genes.updated.2019 cell cycle genes before MOFA2 HVG selection"
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
    print(f"=== STEP 05: MULTI-OMICS FACTOR ANALYSIS (MOFA2) ===")

    qc_meta_path = os.path.join(args.out_dir, "qc_filtered_metadata.csv")
    if not os.path.exists(qc_meta_path):
        raise FileNotFoundError(
            f"Missing prerequisite output from Step 1 (QC): '{qc_meta_path}'.\n"
            f"Please run Step 1 first:\n"
            f"  python run_master_analysis.py --step qc --out_dir \"{args.out_dir}\""
        )
    proc_rna, proc_img, ploidy_series, mouse_series, effective_rna_likelihood = prepare_mofa_input_matrices(
        args.exp_dir,
        qc_meta_path,
        mofa_hdf5_path=args.mofa_hdf5,
        n_hvg=args.n_hvg,
        hvg_method=args.hvg_method,
        min_cells_per_gene=args.min_cells_per_gene,
        rna_likelihood=args.rna_likelihood,
        morph_likelihood=args.morph_likelihood,
        rna_norm=args.rna_norm,
        log_morph=args.log_morph,
        depth_norm=args.depth_norm,
        regress_cell_cycle=args.regress_cell_cycle,
        adjust_library_depth=args.adjust_library_depth,
        exclude_mt=args.exclude_mt,
        exclude_cell_cycle=args.exclude_cell_cycle,
        exclude_genes_file=args.exclude_genes_file,
        exclude_genes=args.exclude_genes,
        out_dir=args.out_dir,
    )

    factor_counts = parse_factor_counts(args.factors)
    multi_k = len(factor_counts) > 1
    sweep_records = []
    rmse_frames = []

    for k in factor_counts:
        suffix = f"_k{k}" if multi_k else ""
        model_hdf5 = os.path.join(args.out_dir, f"mofa_model_rna_morphology{suffix}.hdf5")
        if multi_k:
            print(f"\n--- MOFA2 Factor Sweep: K = {k} ---")
        if not (args.reuse_model and os.path.exists(model_hdf5)):
            run_mofa_training(
                proc_rna,
                proc_img,
                output_hdf5=model_hdf5,
                factors=k,
                rna_likelihood=effective_rna_likelihood,
                morph_likelihood=args.morph_likelihood
            )

        summary, rmse_df = evaluate_mofa_model(
            model_hdf5,
            ploidy_series,
            args.out_dir,
            suffix=suffix,
            proc_rna=proc_rna,
            proc_img=proc_img,
            mouse_series=mouse_series,
            rna_likelihood=effective_rna_likelihood,
            morph_likelihood=args.morph_likelihood,
            factors_k=k,
        )
        sweep_records.append({"Factors_K": k, "Model_Path": os.path.abspath(model_hdf5), **summary})
        if not rmse_df.empty:
            rmse_frames.append(rmse_df)

    if multi_k and sweep_records:
        sweep_df = pd.DataFrame(sweep_records)
        sweep_csv = os.path.join(args.out_dir, "05_mofa_factor_sweep_summary.csv")
        sweep_df.to_csv(sweep_csv, index=False)
        print(f"\nSaved MOFA2 multi-factor sweep summary to: {os.path.abspath(sweep_csv)}")
        all_rmse_df = pd.concat(rmse_frames, ignore_index=True) if rmse_frames else pd.DataFrame()
        plot_mofa_factor_sweep_comparison(sweep_df, all_rmse_df, args.out_dir)


if __name__ == "__main__":
    main()
