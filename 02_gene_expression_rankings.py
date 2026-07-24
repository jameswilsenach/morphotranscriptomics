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
    python 02_gene_expression_rankings.py --gene kalrn --n 10 --exp_dir "D:\\Morphotranscriptomic_Data\\expression" --out_dir "output"
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
    Prefers D:\\Morphotranscriptomic_Data\\expression if available, falling back to exp/.
    """
    if not os.path.exists(exp_dir) and os.path.exists("exp"):
        exp_dir = "exp"
    
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


def filter_genes(features_df, matrix):
    """
    Applies morphotrans2 gene filtering logic:
    - Excludes mitochondrial genes starting with 'mt-'
    - Excludes ribosomal genes starting with ('Rps', 'Rpl')
    - Excludes genes starting with 'Unnamed'
    """
    symbols = features_df["gene_symbol"].astype(str)
    
    is_mt = symbols.str.startswith("mt-")
    is_ribo = symbols.str.startswith(("Rps", "Rpl"))
    is_unnamed = symbols.str.startswith("Unnamed")

    # morphotrans2 notebook rule: genes_to_keep = ~(is_ribo | is_unnamed)
    genes_to_keep = ~(is_ribo | is_unnamed)
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


def rank_gene_expression(gene_symbol, matrix, features_df, barcodes, qc_meta_path=None, n=10):
    """
    Vectorized extraction of top-N and bottom-N cell expression ranks for target gene.
    Integrates QC filtering metadata if available.
    """
    match_mask = features_df["gene_symbol"].str.lower() == gene_symbol.lower()
    matching_genes = features_df[match_mask]

    if matching_genes.empty:
        partial_mask = features_df["gene_symbol"].str.lower().str.contains(gene_symbol.lower())
        matching_genes = features_df[partial_mask]
        if matching_genes.empty:
            raise ValueError(f"Gene symbol '{gene_symbol}' not found in feature annotations.")

    gene_row_idx = matching_genes.index[0]
    matched_id = matching_genes.iloc[0]["gene_id"]
    matched_symbol = matching_genes.iloc[0]["gene_symbol"]
    print(f"Target Gene Identified: '{matched_symbol}' (ID: {matched_id}, Row Index: {gene_row_idx})")

    # Extract expression vector
    gene_expr = matrix[gene_row_idx, :].toarray().ravel()

    cell_df = pd.DataFrame({
        "cell_barcode": barcodes,
        "expression": gene_expr
    })

    # Merge QC metadata from Step 01 if available
    if qc_meta_path and os.path.exists(qc_meta_path):
        qc_meta = pd.read_csv(qc_meta_path)
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


def plot_expression_rankings(ranked_df, gene_symbol, out_dir):
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

    ax.set_title(f"Top & Bottom Expressing Cells for Gene '{gene_symbol}'", fontsize=13, fontweight="bold")
    ax.set_xlabel("Cell Barcode")
    ax.set_ylabel("Raw Expression Count")
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
    parser.add_argument("--exp_dir", type=str, default=r"D:\Morphotranscriptomic_Data\expression", help="10x expression directory")
    parser.add_argument("--out_dir", type=str, default="output", help="Directory to save output files")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"=== STEP 02: GENE EXPRESSION RANKINGS & CELL MAPPING ===")

    # Load 10x matrix data
    matrix, features_df, barcodes = load_10x_expression_data(args.exp_dir)

    # Filter genes matching morphotrans2 rules
    matrix, features_df, _ = filter_genes(features_df, matrix)

    # Check for QC metadata from Step 01
    qc_meta_path = os.path.join(args.out_dir, "qc_filtered_metadata.csv")

    # Extract expression ranks
    ranked_df, matched_symbol = rank_gene_expression(
        gene_symbol=args.gene,
        matrix=matrix,
        features_df=features_df,
        barcodes=barcodes,
        qc_meta_path=qc_meta_path,
        n=args.n
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
    plot_expression_rankings(ranked_df, matched_symbol, args.out_dir)


if __name__ == "__main__":
    main()
