"""
run_master_analysis.py
----------------------
Master orchestration script for the Single Cell Morphotranscriptomics Project handover.

Runs the complete pipeline end-to-end or individual analysis stages cleanly via CLI:
  Step 1: 01_qc_and_metadata_filtering.py     (Quality Control & Train/Pilot metadata harmonization)
  Step 2: 02_gene_expression_rankings.py     (Vectorized gene expression lookup & FCS/CVW mapping)
  Step 3: 03_dimensionality_reduction.py     (Scanpy/AnnData HVG selection, PCA, & UMAP embedding)
  Step 4: 04_mofa_multimodal_analysis.py     (Multi-Omics Factor Analysis integration)
  Step 5: 05_pathway_enrichment.py           (Pathway enrichment dotplots & WordCloud visualization)

Usage Examples:
    # Run full pipeline end-to-end:
    python run_master_analysis.py --train_dir "D:\\Morphotranscriptomic_Data" --test_dir "exp" --out_dir "output" --gene kalrn

    # Run specific step (e.g. Step 1 QC filtering only):
    python run_master_analysis.py --step 1

    # Run for a different marker gene (e.g. pf4):
    python run_master_analysis.py --gene pf4
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

    parser.add_argument("--train_dir", type=str, default=r"D:\Morphotranscriptomic_Data", help="Path to main Train dataset directory")
    parser.add_argument("--test_dir", type=str, default="exp", help="Path to local Test/Pilot expression directory")
    parser.add_argument("--out_dir", type=str, default="output", help="Directory to save generated CSVs and figures")
    parser.add_argument("--gene", type=str, default="kalrn", help="Target gene symbol for expression rankings (default: kalrn)")
    parser.add_argument("--step", type=str, default="all", choices=["all", "1", "2", "3", "4", "5"], help="Specific step to run or 'all'")
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
    print(f" Train Directory   : {os.path.abspath(args.train_dir)}")
    print(f" Test Directory    : {os.path.abspath(args.test_dir)}")
    print(f" Output Directory  : {os.path.abspath(args.out_dir)}")
    print(f" Target Gene       : {args.gene}")
    print(f" Target Step       : {args.step}")
    print("==========================================================================")

    exp_train_dir = os.path.join(args.train_dir, "expression")
    if not os.path.exists(exp_train_dir):
        exp_train_dir = args.test_dir

    # Step 1: Quality Control & Metadata Harmonization
    if args.step in ["all", "1"]:
        print("\n>>> STEP 1: Quality Control & Metadata Harmonization")
        cmd = [
            python_exe, "01_qc_and_metadata_filtering.py",
            "--train_dir", args.train_dir,
            "--test_dir", args.test_dir,
            "--out_dir", args.out_dir
        ]
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

    print("\n==========================================================================")
    print("      ALL MORPHOTRANSCRIPTOMICS ANALYSIS STAGES COMPLETED SUCCESSFULLY     ")
    print(f" Outputs saved to: {os.path.abspath(args.out_dir)}")
    print("==========================================================================")


if __name__ == "__main__":
    main()
