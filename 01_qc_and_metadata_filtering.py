"""
01_qc_and_metadata_filtering.py
--------------------------------
Ingests single cell morphotranscriptomics expression metadata and applies quality control (QC)
filtering based on morphotrans2 notebook methods.

Handles metadata formatting inconsistencies between Train (D:\\Morphotranscriptomic_Data) and Test/Pilot (exp/).

Exact morphotrans2 QC Filter Criteria (Yields 291 QC-passed cells on Train dataset):
1. Contaminated == False
2. Supervised_ploidy in ['2n', '4n', '8n', '16n', '32n'] (excludes '64n' and NaN)
3. control == 0 / False
4. number_of_cell == 1
5. kallisto.pc_map > 50.0
6. kallisto.nb_mapped > 20,000

Usage:
    python 01_qc_and_metadata_filtering.py --train_dir "D:\\Morphotranscriptomic_Data" --test_dir "exp" --out_dir "output"
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def setup_plot_style():
    """Configures high-DPI matplotlib publication rendering matching morphotrans2 notebook."""
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.autolayout'] = True
    sns.set_theme(style="whitegrid")


def normalize_metadata(df, dataset_name="Train"):
    """
    Harmonizes column naming and data types across formatting inconsistencies
    between Train (D:\\Morphotranscriptomic_Data) and Test/Pilot (exp/) datasets.
    """
    df = df.copy()

    # Standardize sample identifier column to 'SampleID'
    if "SampleID" not in df.columns:
        if "Well" in df.columns:
            df["SampleID"] = df["Well"]
        elif "Barcode" in df.columns:
            df["SampleID"] = df["Barcode"]
        elif df.index.name in ["SampleID", "Well", "Barcode"]:
            df["SampleID"] = df.index
        else:
            df["SampleID"] = df.iloc[:, 0].astype(str)

    df["SampleID"] = df["SampleID"].astype(str)
    df.set_index("SampleID", drop=False, inplace=True)

    # Standardize Ploidy column
    if "Supervised_ploidy" not in df.columns:
        if "Estimated_Ploidy_D.NA.A" in df.columns:
            df["Supervised_ploidy"] = df["Estimated_Ploidy_D.NA.A"]
        elif "ploidy" in df.columns:
            df["Supervised_ploidy"] = df["ploidy"]
        else:
            df["Supervised_ploidy"] = "Unknown"

    df["Supervised_ploidy"] = df["Supervised_ploidy"].astype(str).str.strip()

    # Standardize Contaminated column (Boolean)
    if "Contaminated" in df.columns:
        df["Contaminated_bool"] = df["Contaminated"].apply(
            lambda x: True if str(x).upper() in ["TRUE", "T", "1"] else False
        )
    else:
        df["Contaminated_bool"] = False

    # Standardize control column
    if "control" in df.columns:
        df["is_control"] = df["control"].apply(
            lambda x: True if str(x).upper() in ["TRUE", "T", "1"] else False
        )
    else:
        df["is_control"] = False

    # Standardize single cell count
    if "number_of_cell" in df.columns:
        df["single_cell"] = df["number_of_cell"].apply(
            lambda x: True if str(x) in ["1", "1.0"] else False
        )
    else:
        df["single_cell"] = True

    # Standardize Kallisto read counts & percentage
    if "kallisto.nb_mapped" not in df.columns and "kallisto_mapped_reads" in df.columns:
        df["kallisto.nb_mapped"] = df["kallisto_mapped_reads"]
    
    if "kallisto.pc_map" not in df.columns and "kallisto_mapping_pc" in df.columns:
        df["kallisto.pc_map"] = df["kallisto_mapping_pc"]

    df["kallisto.nb_mapped"] = pd.to_numeric(df.get("kallisto.nb_mapped", np.nan), errors="coerce")
    df["kallisto.pc_map"] = pd.to_numeric(df.get("kallisto.pc_map", np.nan), errors="coerce")
    df["dataset_type"] = dataset_name

    return df


def load_metadata_sources(train_dir, test_dir):
    """
    Loads metadata from both Train (D:\\Morphotranscriptomic_Data) and local Test/Pilot directories.
    """
    metadata_frames = []

    # 1. Train Metadata Files
    train_meta_path = os.path.join(train_dir, "Metadata_flow+well_id+supervised_ploidy.csv")
    train_contam_path = os.path.join(train_dir, "expression", "meta.contaminant.2.tsv")

    if not os.path.exists(train_contam_path) and os.path.exists("meta.contaminant.2.tsv"):
        train_contam_path = "meta.contaminant.2.tsv"

    if os.path.exists(train_contam_path):
        train_contam_df = pd.read_csv(train_contam_path, sep="\t")
        norm_train_contam = normalize_metadata(train_contam_df, dataset_name="Train")

        if os.path.exists(train_meta_path):
            train_flow_df = pd.read_csv(train_meta_path)
            norm_train_flow = normalize_metadata(train_flow_df, dataset_name="Train")
            # Merge flow ploidy into contaminant metadata
            merged_train = norm_train_contam.merge(
                norm_train_flow[["Supervised_ploidy"]],
                left_index=True,
                right_index=True,
                how="left",
                suffixes=("", "_flow")
            )
            if "Supervised_ploidy_flow" in merged_train.columns:
                merged_train["Supervised_ploidy"] = merged_train["Supervised_ploidy_flow"].fillna(merged_train["Supervised_ploidy"])
            metadata_frames.append(merged_train)
        else:
            metadata_frames.append(norm_train_contam)

    # 2. Test / Pilot Metadata Files (Local fallback)
    local_meta_path = "meta.v1+flowSF.csv"
    if os.path.exists(local_meta_path):
        local_df = pd.read_csv(local_meta_path)
        norm_local = normalize_metadata(local_df, dataset_name="Pilot/Test")
        metadata_frames.append(norm_local)

    if not metadata_frames:
        raise FileNotFoundError(f"No valid metadata files found in {train_dir} or local directory.")

    combined_meta = pd.concat(metadata_frames, axis=0, ignore_index=False)
    combined_meta = combined_meta.loc[~combined_meta.index.duplicated(keep="first")]
    return combined_meta


def apply_qc_filters(df):
    """
    Applies strict morphotrans2 QC filtering pipeline:
    - Not contaminated (Contaminated == False)
    - Supervised ploidy in ['2n', '4n', '8n', '16n', '32n'] (excludes '64n' and NA)
    - control == False / 0
    - number_of_cell == 1
    - kallisto.pc_map > 50.0
    - kallisto.nb_mapped > 20000
    """
    valid_ploidies = ["2n", "4n", "8n", "16n", "32n"]

    cond_not_contam = ~df["Contaminated_bool"]
    cond_valid_ploidy = df["Supervised_ploidy"].isin(valid_ploidies)
    cond_not_control = ~df["is_control"]
    cond_single_cell = df["single_cell"]

    # Kallisto mapping metrics rule matching morphotrans2 cell 14
    cond_map_pc = df["kallisto.pc_map"].isna() | (df["kallisto.pc_map"] > 50.0)
    cond_nb_mapped = df["kallisto.nb_mapped"].isna() | (df["kallisto.nb_mapped"] > 20000)

    full_qc_mask = (
        cond_not_contam &
        cond_valid_ploidy &
        cond_not_control &
        cond_single_cell &
        cond_map_pc &
        cond_nb_mapped
    )

    df["qc_pass"] = full_qc_mask
    qc_passed_df = df[full_qc_mask].copy()

    return df, qc_passed_df


def plot_qc_metrics(df, out_dir):
    """Generates publication quality QC distribution plots."""
    os.makedirs(out_dir, exist_ok=True)
    setup_plot_style()

    # Figure 1: Ploidy breakdown before and after QC
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    valid_df = df[df["Supervised_ploidy"].isin(["2n", "4n", "8n", "16n", "32n", "64n"])]
    sns.countplot(data=valid_df, x="Supervised_ploidy", hue="dataset_type", ax=ax[0], palette="viridis")
    ax[0].set_title("All Cells: Supervised Ploidy Distribution", fontsize=12, fontweight="bold")
    ax[0].set_xlabel("Supervised Ploidy State")
    ax[0].set_ylabel("Cell Count")

    passed_df = df[df["qc_pass"] & df["Supervised_ploidy"].isin(["2n", "4n", "8n", "16n", "32n"])]
    sns.countplot(data=passed_df, x="Supervised_ploidy", hue="dataset_type", ax=ax[1], palette="crest")
    ax[1].set_title("QC-Passed Cells: Supervised Ploidy Distribution (Benchmark: 291 cells)", fontsize=12, fontweight="bold")
    ax[1].set_xlabel("Supervised Ploidy State")
    ax[1].set_ylabel("Cell Count")

    plt.tight_layout()
    fig1_path = os.path.join(out_dir, "01_ploidy_distribution_qc.png")
    plt.savefig(fig1_path)
    plt.close()

    # Figure 2: Violin plots of mapped reads & mapping % across ploidies for Train set
    train_passed = passed_df[passed_df["kallisto.nb_mapped"].notna()]
    if not train_passed.empty:
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        sns.violinplot(data=train_passed, x="Supervised_ploidy", y="kallisto.nb_mapped", ax=ax[0], inner="quartile", palette="Set2")
        ax[0].set_title("Kallisto Mapped Reads by Ploidy", fontsize=12, fontweight="bold")
        ax[0].set_ylabel("Mapped Read Count")
        ax[0].set_yscale("log")

        sns.violinplot(data=train_passed, x="Supervised_ploidy", y="kallisto.pc_map", ax=ax[1], inner="quartile", palette="Set2")
        ax[1].set_title("Kallisto Mapping Percentage (%) by Ploidy", fontsize=12, fontweight="bold")
        ax[1].set_ylabel("% Mapped Reads")

        plt.tight_layout()
        fig2_path = os.path.join(out_dir, "01_kallisto_qc_metrics_violin.png")
        plt.savefig(fig2_path)
        plt.close()

    print(f"[QC Plotting] Saved QC visualization figures to: {os.path.abspath(out_dir)}")


def main():
    parser = argparse.ArgumentParser(description="01: Single Cell Morphotranscriptomics Quality Control & Metadata Filtering")
    parser.add_argument("--train_dir", type=str, default=r"D:\Morphotranscriptomic_Data", help="Path to main Train data directory")
    parser.add_argument("--test_dir", type=str, default="exp", help="Path to local Test/Pilot expression directory")
    parser.add_argument("--out_dir", type=str, default="output", help="Directory to save output CSVs and plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"=== STEP 01: QUALITY CONTROL & METADATA FILTERING ===")
    print(f"Train directory: {args.train_dir}")
    print(f"Test directory : {args.test_dir}")

    # Load and harmonize metadata
    meta_df = load_metadata_sources(args.train_dir, args.test_dir)
    print(f"Total raw metadata records loaded: {len(meta_df)}")

    # Apply QC filters
    full_df, qc_df = apply_qc_filters(meta_df)
    print(f"Total cells passing ALL QC FILTERS: {len(qc_df)} (Target morphotrans2 benchmark: 291 cells)")

    if "Supervised_ploidy" in qc_df.columns:
        print("\nPloidy breakdown of QC-passed cells:")
        print(qc_df["Supervised_ploidy"].value_counts().to_string())

    # Save summary table
    summary_dict = {
        "Total_Raw_Cells": len(full_df),
        "Contaminated_Cells": int(full_df["Contaminated_bool"].sum()),
        "Valid_Ploidy_Cells": int(full_df["Supervised_ploidy"].isin(["2n", "4n", "8n", "16n", "32n"]).sum()),
        "Single_Cells": int(full_df["single_cell"].sum()),
        "QC_Passed_Cells": len(qc_df)
    }
    summary_df = pd.DataFrame([summary_dict])
    summary_csv = os.path.join(args.out_dir, "qc_filtering_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved QC summary to: {os.path.abspath(summary_csv)}")

    # Save filtered metadata CSV
    filtered_csv = os.path.join(args.out_dir, "qc_filtered_metadata.csv")
    qc_df.to_csv(filtered_csv)
    print(f"Saved QC-passed metadata table to: {os.path.abspath(filtered_csv)}")

    # Generate plots
    plot_qc_metrics(full_df, args.out_dir)


if __name__ == "__main__":
    main()
