"""
run_master_analysis.py
----------------------
Master orchestration script for the Single Cell Morphotranscriptomics Project handover.

Runs the complete pipeline end-to-end or individual analysis stages cleanly via CLI:
  Step 1: 01_qc_and_metadata_filtering.py     (Quality Control & Multi-Animal / Legacy metadata harmonization)
  Step 2: 02_gene_expression_rankings.py     (Vectorized gene expression lookup & FCS/CVW mapping)
  Step 3: 03_dimensionality_reduction.py     (Scanpy/AnnData HVG selection, PCA, & UMAP embedding)
  Step 4: 04_mofa_multimodal_analysis.py     (Multi-Omics Factor Analysis integration)
  Step 5: 05_pathway_enrichment.py           (Pathway enrichment dotplots & WordCloud visualization)
  Step 6: 06_morphology_gene_correlations.py (Morphology-to-gene Spearman rank correlation scatter grid & gene rankings)
  Step CV: leave_one_animal_out_cv.py        (Leave-One-Animal-Out Cross-Validation across all mice)

Usage Examples:
    # Run Morphology-to-Gene Spearman Correlation Analysis:
    python run_master_analysis.py --step 6

    # Run Leave-One-Animal-Out Cross-Validation:
    python run_master_analysis.py --step cv

    # Run full pipeline end-to-end including LOAOCV:
    python run_master_analysis.py --loaocv --out_dir "output"

    # Run backwards-compatible legacy Train/Pilot split:
    python run_master_analysis.py --legacy_split --out_dir "output_legacy"
"""

import argparse
import os
import sys
import subprocess
import time


def run_command(cmd):
    """Executes a python sub-process and monitors stdout/stderr logs."""
    print(f"\n[EXEC] Running command:\n  {' '.join(cmd)}")
    start_time = time.time()

    res = subprocess.run(cmd, text=True)

    elapsed = time.time() - start_time
    if res.returncode == 0:
        print(f"[SUCCESS] Completed stage in {elapsed:.2f} seconds.")
    else:
        print(f"[ERROR] Stage failed with exit code {res.returncode}.")
        sys.exit(res.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Master Orchestration Script - Single Cell Morphotranscriptomics Project",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--metadata_file", type=str, default=r"C:\Users\vilsn\Dropbox\Work\Turing\Exploration_Data\Metadata_merged_YL_SF_20.10.26_v2.csv", help="Path to unified multi-animal metadata CSV")
    parser.add_argument("--train_dir", type=str, default=r"D:\Morphotranscriptomic_Data", help="Path to main Train dataset directory")
    parser.add_argument("--test_dir", type=str, default="exp", help="Path to local Test/Pilot expression directory")
    parser.add_argument("--out_dir", type=str, default="output", help="Directory to save generated CSVs and figures")
    parser.add_argument("--gene", type=str, default="kalrn", help="Target gene symbol for expression rankings (default: kalrn)")
    parser.add_argument("--step", type=str, default="all", choices=["all", "1", "2", "3", "4", "5", "6", "cv"], help="Specific step to run, '6' for correlations, 'cv' for LOAOCV, or 'all'")
    parser.add_argument("--loaocv", action="store_true", help="Enable Leave-One-Animal-Out Cross-Validation stage")
    parser.add_argument("--legacy_split", action="store_true", help="Force legacy Train/Pilot 2-dataset split ingestion for backwards compatibility")
    parser.add_argument(
        "--rna_likelihood",
        type=str,
        default="poisson",
        choices=["poisson", "bernoulli", "gaussian"],
        help="MOFA2 likelihood for RNA view in Step 4 (default: poisson)"
    )
    parser.add_argument(
        "--rna_norm",
        type=str,
        default="raw_counts",
        choices=["raw_counts", "size_factor_counts"],
        help="Count normalization strategy for Poisson RNA likelihood in Step 4 (default: raw_counts)"
    )

    args = parser.parse_args()

    python_exe = sys.executable
    os.makedirs(args.out_dir, exist_ok=True)

    print("==========================================================================")
    print("      SINGLE CELL MORPHOTRANSCRIPTOMICS MASTER ANALYSIS RUNNER           ")
    print("==========================================================================")
    print(f" Python Executable : {python_exe}")
    print(f" Metadata File     : {os.path.abspath(args.metadata_file) if args.metadata_file else 'Default'}")
    print(f" Train Directory   : {os.path.abspath(args.train_dir)}")
    print(f" Test Directory    : {os.path.abspath(args.test_dir)}")
    print(f" Output Directory  : {os.path.abspath(args.out_dir)}")
    print(f" Target Gene       : {args.gene}")
    print(f" Target Step       : {args.step}")
    print(f" LOAOCV Enabled    : {args.loaocv or args.step == 'cv'}")
    print(f" Legacy Split Mode : {args.legacy_split}")
    print("==========================================================================")

    exp_train_dir = os.path.join(args.train_dir, "expression")
    if not os.path.exists(exp_train_dir):
        if os.path.exists(args.test_dir):
            exp_train_dir = args.test_dir
        elif os.path.exists(r"C:\Users\vilsn\Dropbox\Work\Turing\Exploration_Data\exp"):
            exp_train_dir = r"C:\Users\vilsn\Dropbox\Work\Turing\Exploration_Data\exp"

    # Step 1: Quality Control & Metadata Harmonization
    if args.step in ["all", "1"]:
        print("\n>>> STEP 1: Quality Control & Metadata Harmonization")
        cmd = [
            python_exe, "01_qc_and_metadata_filtering.py",
            "--metadata_file", args.metadata_file,
            "--train_dir", args.train_dir,
            "--test_dir", args.test_dir,
            "--out_dir", args.out_dir
        ]
        if args.legacy_split:
            cmd.append("--legacy_split")
        run_command(cmd)

    # Step 2: Gene Expression Rankings & FCS/CVW Cell Mapping
    if args.step in ["all", "2"]:
        print(f"\n>>> STEP 2: Gene Expression Rankings & FCS/CVW Mapping for '{args.gene}'")
        cmd = [
            python_exe, "02_gene_expression_rankings.py",
            "--gene", args.gene,
            "--exp_dir", exp_train_dir,
            "--out_dir", args.out_dir
        ]
        run_command(cmd)

    # Step 3: Dimensionality Reduction (PCA & UMAP)
    if args.step in ["all", "3"]:
        print("\n>>> STEP 3: Dimensionality Reduction (Scanpy PCA & UMAP)")
        cmd = [
            python_exe, "03_dimensionality_reduction.py",
            "--exp_dir", exp_train_dir,
            "--out_dir", args.out_dir
        ]
        run_command(cmd)

    # Step 4: Multi-Omics Factor Analysis (MOFA) Integration
    if args.step in ["all", "4"]:
        print(f"\n>>> STEP 4: Multi-Omics Factor Analysis (MOFA2 Integration, RNA likelihood='{args.rna_likelihood}')")
        cmd = [
            python_exe, "04_mofa_multimodal_analysis.py",
            "--exp_dir", exp_train_dir,
            "--out_dir", args.out_dir,
            "--rna_likelihood", args.rna_likelihood,
            "--rna_norm", args.rna_norm
        ]
        run_command(cmd)

    # Step 5: Biological Pathway Enrichment & WordCloud
    if args.step in ["all", "5"]:
        print("\n>>> STEP 5: Biological Pathway Enrichment & WordCloud Generation")
        cmd = [
            python_exe, "05_pathway_enrichment.py",
            "--gene_list", f"{args.gene.upper()},PF4,RUNX1,ITGA2B,MPL,FLI1,GATA1,NFE2",
            "--out_dir", args.out_dir
        ]
        run_command(cmd)

    # Step 6: Morphology-to-Gene Spearman Rank Correlations
    if args.step in ["all", "6"]:
        print("\n>>> STEP 6: Morphology-to-Gene Spearman Rank Correlation Analysis")
        cmd = [
            python_exe, "06_morphology_gene_correlations.py",
            "--qc_meta", os.path.join(args.out_dir, "qc_filtered_metadata.csv"),
            "--exp_dir", exp_train_dir,
            "--train_dir", args.train_dir,
            "--test_dir", args.test_dir,
            "--out_dir", args.out_dir
        ]
        run_command(cmd)

    # Step CV: Leave-One-Animal-Out Cross-Validation (LOAOCV)
    if args.step in ["all", "cv"] or args.loaocv:
        print("\n>>> STEP CV: Leave-One-Animal-Out Cross-Validation (LOAOCV)")
        cmd = [
            python_exe, "leave_one_animal_out_cv.py",
            "--metadata_file", args.metadata_file,
            "--qc_meta", os.path.join(args.out_dir, "qc_filtered_metadata.csv"),
            "--out_dir", args.out_dir
        ]
        run_command(cmd)

    print("\n==========================================================================")
    print("      ALL MORPHOTRANSCRIPTOMICS ANALYSIS STAGES COMPLETED SUCCESSFULLY     ")
    print(f" Outputs saved to: {os.path.abspath(args.out_dir)}")
    print("==========================================================================")


if __name__ == "__main__":
    main()
