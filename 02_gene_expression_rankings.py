"""
02_gene_expression_rankings.py
------------------------------
Ingests 10x single-cell expression matrix data and performs vectorized gene expression ranking,
cell selection, and FCS/CVW file mapping based on morphotrans2 notebook methods.

Features:
1. Gene filtering: Excludes mitochondrial (mt-), ribosomal (Rps, Rpl), and Unnamed genes as in morphotrans2.
2. Vectorized top-N and bottom-N expression ranking for target gene symbols (default: kalrn, pf4).
3. QC integration: Merges step 01 QC filters (Contaminated==False, Ploidy>=2n, single cell, kallisto stats).
4. Automated mapping: Maps cell barcodes to FCS (.fcs) and FlowJo (.cvw) files, Mouse ID, Plate, and Well annotations.
5. Handles metadata formatting inconsistencies across Train and Pilot/Test datasets.

Usage:
    python 02_gene_expression_rankings.py --gene kalrn --n 10 --exp_dir "data/expression" --out_dir "output"
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
from scipy.io import mmread
import matplotlib.pyplot as plt
import seaborn as sns


def setup_plot_style():
    """Configures high-DPI matplotlib publication rendering matching morphotrans2 notebook."""
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.autolayout'] = True
    sns.set_theme(style="whitegrid")


def load_10x_expression_data(exp_dir):
    """
    Loads 10x expression matrix, cell barcodes, and gene feature annotations.
    Checks the provided exp_dir first, falling back to data/exp or exp/.
    """
    if not os.path.exists(exp_dir):
        for fallback in [os.path.join("data", "exp"), "exp"]:
            if os.path.exists(fallback):
                exp_dir = fallback
                break
    
    print(f"Loading 10x matrix from: {os.path.abspath(exp_dir)}")

    # 1. Features / Genes
    features_path = os.path.join(exp_dir, "features.tsv")
    if not os.path.exists(features_path):
        features_path = os.path.join(exp_dir, "features.tsv.gz")

    features_df = pd.read_csv(
        features_path,
        sep="\t",
        header=None,
        names=["gene_id", "gene_symbol", "feature_type"]
    )

    # 2. Barcodes / Cells
    barcodes_path = os.path.join(exp_dir, "barcodes.tsv")
    if not os.path.exists(barcodes_path):
        barcodes_path = os.path.join(exp_dir, "barcodes.tsv.gz")

    barcodes_df = pd.read_csv(barcodes_path, header=None, names=["cell_barcode"])
    barcodes = barcodes_df["cell_barcode"].astype(str).values

    # 3. Sparse Expression Matrix
    matrix_path = os.path.join(exp_dir, "matrix.mtx")
    if not os.path.exists(matrix_path):
        matrix_path = os.path.join(exp_dir, "matrix.mtx.gz")

    matrix = mmread(matrix_path).tocsr()

    return matrix, features_df, barcodes


def filter_genes(
    features_df,
    matrix,
    exclude_cell_cycle=False,
    exclude_genes_file=None,
    exclude_genes=None,
    out_dir=None,
):
    """
    Applies morphotrans2 gene filtering logic:
    - Excludes ribosomal genes starting with ('Rps', 'Rpl')
    - Excludes genes starting with 'Unnamed'
    - Optionally excludes cited cell-cycle genes and/or user-supplied genes
    """
    symbols = features_df["gene_symbol"].astype(str)
    
    is_mt = symbols.str.startswith("mt-")
    is_ribo = symbols.str.startswith(("Rps", "Rpl"))
    is_unnamed = symbols.str.startswith("Unnamed")

    # morphotrans2 notebook rule: genes_to_keep = ~(is_ribo | is_unnamed)
    genes_to_keep = ~(is_ribo | is_unnamed)

    if exclude_cell_cycle or exclude_genes_file or exclude_genes:
        from cell_cycle import filter_excluded_genes
        audit_path = os.path.join(out_dir, "02_excluded_genes_audit.csv") if out_dir else None
        cc_mask, _ = filter_excluded_genes(
            symbols,
            exclude_cell_cycle=exclude_cell_cycle,
            exclude_genes_file=exclude_genes_file,
            exclude_genes=exclude_genes,
            audit_csv_path=audit_path,
        )
        genes_to_keep = genes_to_keep & cc_mask
    else:
        print("[Gene Filter] No additional gene exclusions applied (baseline).")

    keep_indices = np.where(genes_to_keep)[0]

    filtered_features = features_df.iloc[keep_indices].reset_index(drop=True)
    filtered_matrix = matrix[keep_indices, :]

    print(f"Original genes: {len(features_df)} | Filtered genes: {len(filtered_features)}")
    return filtered_matrix, filtered_features, keep_indices


def map_barcodes_to_fcs_cvw(cell_df):
    """
    Maps cell barcodes to FCS (.fcs) and FlowJo workspace (.cvw) files for Mice M1-M5
    across Plate 1 and Plate 2 based on morphotrans2 directory mapping.
    """
    fcs_map = {
        "m1p1": ("260119_BM_lowhi_250128_sortM1p1.fcs", "260119_BM_lowhi_250128_sortM1p1.cvw"),
        "m1p2": ("260119_BM_lowhi_250128_sortM1p2.fcs", "260119_BM_lowhi_250128_sortM1p2.cvw"),
        "m2p1": ("260119_BM_lowhi_250209_sortM2p1.fcs", "260119_BM_lowhi_250209_sortM2p1.cvw"),
        "m2p2": ("260119_BM_lowhi_250209_M2p2.fcs", "260119_BM_lowhi_250209_M2p2.cvw"),
        "m3p1": ("260119_BM_lowhi_250209_M3p1.fcs", "260119_BM_lowhi_250209_M3p1.cvw"),
        "m3p2": ("260119_BM_lowhi_250209_M3p2.fcs", "260119_BM_lowhi_250209_M3p2.cvw"),
        "m4p1": ("260119_BM_lowhi_250209_M4p1.fcs", "260119_BM_lowhi_250209_M4p1.cvw"),
        "m4p2": ("260119_BM_lowhi_250209_M4p2.fcs", "260119_BM_lowhi_250209_M4p2.cvw"),
        "m5p1": ("260119_BM_lowhi_250209_M5p1.fcs", "260119_BM_lowhi_250209_M5p1.cvw"),
        "m5p2": ("260119_BM_lowhi_250209_M5p2.fcs", "260119_BM_lowhi_250209_M5p2.cvw"),
    }

    fcs_list, cvw_list, mouse_list, plate_list, well_list = [], [], [], [], []
    for idx, row in cell_df.iterrows():
        barcode = str(row["cell_barcode"])
        m = re.match(r"^M(\d+)p(\d+)_(.+)$", barcode, re.IGNORECASE)
        if m:
            m_num, p_num, well = m.group(1), m.group(2), m.group(3)
            mp_code = f"m{m_num}p{p_num}"
            mouse = f"M{m_num}"
            plate = f"Plate {p_num}"
        else:
            mp_code = "unknown"
            mouse, plate, well = "N/A", "N/A", "N/A"

        fcs, cvw = fcs_map.get(mp_code, ("UNKNOWN", "UNKNOWN"))
        fcs_list.append(fcs)
        cvw_list.append(cvw)
        mouse_list.append(mouse)
        plate_list.append(plate)
        well_list.append(well)

    cell_df["Mouse"] = mouse_list
    cell_df["Plate"] = plate_list
    cell_df["Well"] = well_list
    cell_df["FCS_File"] = fcs_list
    cell_df["CVW_File"] = cvw_list

    return cell_df


def rank_gene_expression(
    gene_symbol,
    matrix,
    features_df,
    barcodes,
    qc_meta_path=None,
    n=10,
    depth_norm="cp10k_log1p",
    regress_cell_cycle="none",
    exclude_cell_cycle=False,
    exclude_genes_file=None,
    exclude_genes=None,
    out_dir=None,
):
    """
    Vectorized extraction of top-N and bottom-N cell expression ranks for target gene.
    Applies unified preprocessing (Depth Normalization -> Cell-Cycle Regression ->
    Optional Gene Exclusions) on QC-passed cells before ranking.
    """
    from preprocessing import preprocess_rna_matrix

    # Build (cells x genes) DataFrame on QC-passed cells first
    qc_meta = None
    if qc_meta_path and os.path.exists(qc_meta_path):
        qc_meta = pd.read_csv(qc_meta_path)

    bc_series = pd.Series(barcodes, dtype=str)
    if qc_meta is not None and "SampleID" in qc_meta.columns:
        qc_ids = set(qc_meta["SampleID"].astype(str))
        if "qc_pass" in qc_meta.columns:
            qc_ids = set(qc_meta.loc[qc_meta["qc_pass"] == True, "SampleID"].astype(str))
        cell_mask = bc_series.isin(qc_ids).values
        if cell_mask.any():
            cell_indices = np.where(cell_mask)[0]
        else:
            cell_indices = np.arange(len(barcodes))
    else:
        cell_indices = np.arange(len(barcodes))

    gene_symbols = features_df["gene_symbol"].astype(str).values
    mat_qc = matrix[:, cell_indices].T.toarray().astype(np.float64)
    df_counts = pd.DataFrame(mat_qc, index=barcodes[cell_indices], columns=gene_symbols)
    if df_counts.columns.duplicated().any():
        df_counts = df_counts.loc[:, ~df_counts.columns.duplicated(keep="first")]

    # Canonical order: Depth Normalization -> Cell-Cycle Regression -> Gene Exclusion
    df_proc, _ = preprocess_rna_matrix(
        df_counts,
        depth_norm=depth_norm,
        regress_cell_cycle=regress_cell_cycle,
        exclude_cell_cycle=exclude_cell_cycle,
        exclude_genes_file=exclude_genes_file,
        exclude_genes=exclude_genes,
        out_dir=out_dir,
        step_prefix="02",
    )

    cols_lower = pd.Series(df_proc.columns, dtype=str).str.lower()
    exact_hits = df_proc.columns[cols_lower == gene_symbol.lower()].tolist()
    if exact_hits:
        matched_symbol = exact_hits[0]
    else:
        partial_hits = df_proc.columns[cols_lower.str.contains(gene_symbol.lower())].tolist()
        if not partial_hits:
            raise ValueError(f"Gene symbol '{gene_symbol}' not found in preprocessed features.")
        matched_symbol = partial_hits[0]

    print(f"Target Gene Identified: '{matched_symbol}'")

    cell_df = pd.DataFrame({
        "cell_barcode": df_proc.index.astype(str),
        "expression": df_proc[matched_symbol].values,
    })

    # Merge QC metadata from Step 01 if available
    if qc_meta is not None:
        cell_df = cell_df.merge(
            qc_meta,
            left_on="cell_barcode",
            right_on="SampleID",
            how="left"
        )

    # Map FCS & CVW file annotations
    cell_df = map_barcodes_to_fcs_cvw(cell_df)

    # Apply QC filtering if qc_pass column exists
    if "qc_pass" in cell_df.columns:
        filtered_df = cell_df[cell_df["qc_pass"] == True].copy()
    else:
        filtered_df = cell_df.copy()

    # Sort using vectorized pandas/numpy sort
    filtered_df.sort_values(by="expression", ascending=False, inplace=True)

    top_df = filtered_df.head(n).copy()
    top_df["rank_group"] = "top"
    top_df["rank"] = np.arange(1, len(top_df) + 1)

    bottom_df = filtered_df.tail(n).iloc[::-1].copy()
    bottom_df["rank_group"] = "bottom"
    bottom_df["rank"] = np.arange(1, len(bottom_df) + 1)

    ranked_df = pd.concat([top_df, bottom_df], ignore_index=True)
    ranked_df["gene_symbol"] = matched_symbol

    return ranked_df, matched_symbol


def plot_expression_rankings(ranked_df, gene_symbol, out_dir, depth_norm="cp10k_log1p", regress_cell_cycle="none"):
    """Generates publication quality ranking bar plot for top/bottom cells."""
    os.makedirs(out_dir, exist_ok=True)
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.barplot(
        data=ranked_df,
        x="cell_barcode",
        y="expression",
        hue="rank_group",
        palette={"top": "#2ca02c", "bottom": "#d62728"},
        ax=ax
    )

    ylab = "Expression (CP10K log1p)" if depth_norm == "cp10k_log1p" else "Raw Expression Count"
    if regress_cell_cycle != "none":
        ylab += f" [CC-regressed: {regress_cell_cycle}]"

    ax.set_title(f"Top & Bottom Expressing Cells for Gene '{gene_symbol}'", fontsize=13, fontweight="bold")
    ax.set_xlabel("Cell Barcode")
    ax.set_ylabel(ylab)
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    out_png = os.path.join(out_dir, f"02_gene_rankings_{gene_symbol.lower()}.png")
    plt.savefig(out_png)
    plt.close()

    print(f"[Gene Ranking] Saved expression bar plot to: {os.path.abspath(out_png)}")


def main():
    parser = argparse.ArgumentParser(description="02: Vectorized Gene Expression Ranking & Cell FCS/CVW Mapping")
    parser.add_argument("--gene", type=str, default="kalrn", help="Target gene symbol (e.g. kalrn, pf4)")
    parser.add_argument("--n", type=int, default=10, help="Number of top and bottom cells to select")
    parser.add_argument("--exp_dir", type=str, default=os.path.join("data", "expression"), help="10x expression directory (falls back to data/exp)")
    parser.add_argument("--out_dir", type=str, default="output", help="Directory to save output files")
    parser.add_argument(
        "--depth_norm",
        type=str,
        default="cp10k_log1p",
        choices=["cp10k_log1p", "none"],
        help="Library-size depth normalization before ranking/regression (default: cp10k_log1p)"
    )
    parser.add_argument(
        "--regress_cell_cycle",
        type=str,
        default="none",
        choices=["none", "full", "difference"],
        help="Cell-cycle regression after depth normalization: 'none' (default), 'full' (S + G2M), or 'difference' (S - G2M)"
    )
    parser.add_argument(
        "--exclude_cell_cycle",
        action="store_true",
        help="Exclude cited Tirosh et al. (2016) / Seurat cc.genes.updated.2019 cell cycle genes from ranking pool"
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
    print(f"=== STEP 02: GENE EXPRESSION RANKINGS & CELL MAPPING ===")

    # Check for prerequisite QC metadata from Step 01
    qc_meta_path = os.path.join(args.out_dir, "qc_filtered_metadata.csv")
    if not os.path.exists(qc_meta_path):
        raise FileNotFoundError(
            f"Missing prerequisite output from Step 1 (QC): '{qc_meta_path}'.\n"
            f"Please run Step 1 first:\n"
            f"  python run_master_analysis.py --step qc --out_dir \"{args.out_dir}\""
        )

    # Load 10x matrix data
    matrix, features_df, barcodes = load_10x_expression_data(args.exp_dir)

    # Filter ribosomal and Unnamed genes first (optional exclusions run after depth norm & CC regression)
    matrix, features_df, _ = filter_genes(
        features_df,
        matrix,
        exclude_cell_cycle=False,
        exclude_genes_file=None,
        exclude_genes=None,
        out_dir=args.out_dir,
    )

    # Extract expression ranks after unified preprocessing
    ranked_df, matched_symbol = rank_gene_expression(
        gene_symbol=args.gene,
        matrix=matrix,
        features_df=features_df,
        barcodes=barcodes,
        qc_meta_path=qc_meta_path,
        n=args.n,
        depth_norm=args.depth_norm,
        regress_cell_cycle=args.regress_cell_cycle,
        exclude_cell_cycle=args.exclude_cell_cycle,
        exclude_genes_file=args.exclude_genes_file,
        exclude_genes=args.exclude_genes,
        out_dir=args.out_dir,
    )

    # Export structured CSV
    cols_to_export = [
        "gene_symbol", "rank_group", "rank", "cell_barcode", "expression",
        "Supervised_ploidy", "Mouse", "Plate", "Well", "FCS_File", "CVW_File"
    ]
    export_cols = [c for c in cols_to_export if c in ranked_df.columns]
    output_csv = os.path.join(args.out_dir, f"{matched_symbol.lower()}_top_bottom_cells.csv")
    ranked_df[export_cols].to_csv(output_csv, index=False)
    print(f"Successfully exported cell mapping to: {os.path.abspath(output_csv)}")

    # Generate ranking plot
    plot_expression_rankings(
        ranked_df,
        matched_symbol,
        args.out_dir,
        depth_norm=args.depth_norm,
        regress_cell_cycle=args.regress_cell_cycle,
    )


if __name__ == "__main__":
    main()
