"""
run_master_analysis.py
----------------------
Master orchestration script for the Single Cell Morphotranscriptomics Project handover.

Runs the complete pipeline end-to-end or individual analysis stages cleanly via CLI:
  Step 1  (qc)           : 01_qc_and_metadata_filtering.py     (Quality Control & Multi-Animal / Legacy metadata harmonization)
  Step 2  (rankings)     : 02_gene_expression_rankings.py      (Vectorized gene expression lookup & FCS/CVW mapping)
  Step 3  (umap/dimred)  : 03_dimensionality_reduction.py      (Scanpy/AnnData HVG selection, PCA, & UMAP embedding)
  Step 4  (correlations) : 04_morphology_gene_correlations.py  (Morphology-to-gene Spearman rank correlation scatter grid & gene rankings)
  Step 5  (mofa)         : 05_mofa_multimodal_analysis.py      (Multi-Omics Factor Analysis integration)
  Step 5b (mvae)         : 05b_mvae_multimodal_analysis.py     (PyTorch Multimodal VAE integration, optional/standalone)
  Step 6  (enrichment)   : 06_pathway_enrichment.py            (Pathway enrichment dotplots & WordCloud visualization)
  Step CV (cv/loaocv)    : leave_one_animal_out_cv.py          (Leave-One-Animal-Out Cross-Validation across all mice)

Usage Examples:
    # Run Step 1 (Quality Control):
    python run_master_analysis.py --step qc

    # Run Morphology-to-Gene Spearman Correlation Analysis (Step 4):
    python run_master_analysis.py --step correlations

    # Run MOFA2 Multimodal Integration (Step 5):
    python run_master_analysis.py --step mofa

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


STEP_ALIASES = {
    "all": "all",
    "1": "1",
    "qc": "1",
    "2": "2",
    "rankings": "2",
    "3": "3",
    "umap": "3",
    "dimred": "3",
    "4": "4",
    "correlations": "4",
    "corr": "4",
    "5": "5",
    "mofa": "5",
    "5b": "5b",
    "mvae": "5b",
    "6": "6",
    "enrichment": "6",
    "pathways": "6",
    "cv": "cv",
    "loaocv": "cv",
}

STEP_LABELS = {
    "1": "Step 1 (qc)",
    "2": "Step 2 (rankings)",
    "3": "Step 3 (umap)",
    "4": "Step 4 (correlations)",
    "5": "Step 5 (mofa)",
    "5b": "Step 5b (mvae)",
    "6": "Step 6 (enrichment)",
    "cv": "Step CV (loaocv)",
}

# Steps that require output/qc_filtered_metadata.csv produced by Step 1 (qc)
STEPS_REQUIRING_QC = {"2", "3", "4", "5", "5b", "cv"}


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


def check_step_dependencies(canonical_step, out_dir):
    """
    Validates that prerequisite outputs from earlier stages exist when running an individual step.
    Provides an actionable message suggesting the prior dependency step if missing.
    """
    if canonical_step in STEPS_REQUIRING_QC:
        qc_meta_path = os.path.join(out_dir, "qc_filtered_metadata.csv")
        if not os.path.exists(qc_meta_path):
            step_label = STEP_LABELS.get(canonical_step, canonical_step)
            print(
                f"\n[DEPENDENCY ERROR] {step_label} requires '{qc_meta_path}', which does not exist yet.\n"
                f"Please run Step 1 (QC) first:\n"
                f"  python run_master_analysis.py --step qc --out_dir \"{out_dir}\"\n"
                f"Or run the full pipeline from Step 1:\n"
                f"  python run_master_analysis.py --out_dir \"{out_dir}\""
            )
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Master Orchestration Script - Single Cell Morphotranscriptomics Project",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--metadata_file",
        type=str,
        default=os.path.join("data", "Metadata_merged_YL_SF_20.10.26_v2.csv"),
        help="Path to unified multi-animal metadata CSV (default: data/Metadata_merged_YL_SF_20.10.26_v2.csv)"
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default="data",
        help="Path to main Train dataset directory (default: data)"
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default=os.path.join("data", "exp"),
        help="Path to local Test/Pilot expression directory (default: data/exp)"
    )
    parser.add_argument("--out_dir", type=str, default="output", help="Directory to save generated CSVs and figures")
    parser.add_argument("--gene", type=str, default="kalrn", help="Target gene symbol for expression rankings (default: kalrn)")
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=list(STEP_ALIASES.keys()),
        help=(
            "Specific step to run by number or alias: "
            "'all', '1'/'qc', '2'/'rankings', '3'/'umap'/'dimred', "
            "'4'/'correlations'/'corr', '5'/'mofa', '5b'/'mvae', "
            "'6'/'enrichment'/'pathways', 'cv'/'loaocv'"
        )
    )
    parser.add_argument("--loaocv", action="store_true", help="Enable Leave-One-Animal-Out Cross-Validation stage")
    parser.add_argument("--legacy_split", action="store_true", help="Force legacy Train/Pilot 2-dataset split ingestion for backwards compatibility")
    parser.add_argument(
        "--mofa_factors",
        type=str,
        default="10",
        help="Number of MOFA2 latent factors in Step 5, or comma-separated list to sweep (e.g. '5,10,15'; default: '10')"
    )
    parser.add_argument(
        "--rna_likelihood",
        type=str,
        default="poisson",
        choices=["poisson", "bernoulli", "gaussian"],
        help="MOFA2 likelihood for RNA view in Step 5 (default: poisson)"
    )
    parser.add_argument(
        "--rna_norm",
        type=str,
        default="raw_counts",
        choices=["raw_counts", "size_factor_counts"],
        help="Count normalization strategy for Poisson RNA likelihood in Step 5 (default: raw_counts)"
    )
    parser.add_argument(
        "--min_mapped_reads",
        type=int,
        default=20000,
        help="Minimum kallisto.nb_mapped reads per cell in Step 1 QC (default: 20000)"
    )
    parser.add_argument(
        "--min_map_pc",
        type=float,
        default=50.0,
        help="Minimum kallisto.pc_map mapping percentage per cell in Step 1 QC (default: 50.0)"
    )
    parser.add_argument(
        "--min_cells_per_gene",
        type=int,
        default=0,
        help="Minimum number of cells expressing a gene (> 0 counts) before HVG selection in Steps 3, 4, 5, 5b (default: 0, disabled)"
    )
    parser.add_argument(
        "--n_hvg",
        type=int,
        default=2000,
        help="Number of highly variable genes to select in Steps 3, 4, 5, 5b (default: 2000)"
    )
    parser.add_argument(
        "--hvg_method",
        type=str,
        default="default",
        choices=["default", "raw_var", "log1p_var", "seurat", "seurat_v3"],
        help="HVG selection method in Steps 3, 4, 5, 5b ('default' keeps each step's baseline method)"
    )
    parser.add_argument(
        "--depth_norm",
        type=str,
        default="cp10k_log1p",
        choices=["cp10k_log1p", "none"],
        help="Library-size depth normalization applied across Steps 2, 3, 4, 5, 5b before cell-cycle regression (default: cp10k_log1p)"
    )
    parser.add_argument(
        "--regress_cell_cycle",
        type=str,
        default="none",
        choices=["none", "full", "difference"],
        help="Cell-cycle regression applied AFTER depth normalization across Steps 2, 3, 4, 5, 5b: 'none' (default), 'full' (S + G2M), or 'difference' (S - G2M)"
    )
    parser.add_argument(
        "--exclude_cell_cycle",
        action="store_true",
        help="Exclude cited Tirosh et al. (2016) / Seurat cc.genes.updated.2019 cell cycle genes (97 genes + legacy aliases)"
    )
    parser.add_argument(
        "--exclude_genes_file",
        type=str,
        default=None,
        help="Path to a .txt/.csv/.tsv file of gene symbols to exclude (e.g., Mouse GO:0007049 or custom list)"
    )
    parser.add_argument(
        "--exclude_genes",
        type=str,
        default=None,
        help="Comma-separated inline list of gene symbols to exclude (e.g. 'Mki67,Top2a,Cdk1')"
    )
    parser.add_argument(
        "--exclude_steps",
        type=str,
        default="all",
        help="Analyses to apply gene exclusions to: 'all' (default: 2,3,4,5,5b) or comma-separated step numbers/aliases (e.g. '3,4' or 'umap,correlations')"
    )

    args = parser.parse_args()
    canonical_step = STEP_ALIASES[args.step]

    python_exe = sys.executable
    os.makedirs(args.out_dir, exist_ok=True)

    # Validate prerequisite outputs if running an individual downstream step
    check_step_dependencies(canonical_step, args.out_dir)

    # Determine which steps receive gene exclusion flags (supports both numbers and aliases)
    if args.exclude_steps.lower() in ["all", "*"]:
        exclude_active_steps = {"2", "3", "4", "5", "5b"}
    else:
        exclude_active_steps = set()
        for tok in args.exclude_steps.split(","):
            t = tok.strip().lower()
            if t:
                exclude_active_steps.add(STEP_ALIASES.get(t, t))

    any_exclusion = bool(args.exclude_cell_cycle or args.exclude_genes_file or args.exclude_genes)

    def append_exclusion_flags(cmd, step_id):
        if step_id not in exclude_active_steps:
            return cmd
        if args.exclude_cell_cycle:
            cmd.append("--exclude_cell_cycle")
        if args.exclude_genes_file:
            cmd.extend(["--exclude_genes_file", args.exclude_genes_file])
        if args.exclude_genes:
            cmd.extend(["--exclude_genes", args.exclude_genes])
        return cmd

    print("==========================================================================")
    print("      SINGLE CELL MORPHOTRANSCRIPTOMICS MASTER ANALYSIS RUNNER           ")
    print("==========================================================================")
    print(f" Python Executable : {python_exe}")
    print(f" Metadata File     : {os.path.abspath(args.metadata_file) if args.metadata_file else 'Default'}")
    print(f" Train Directory   : {os.path.abspath(args.train_dir)}")
    print(f" Test Directory    : {os.path.abspath(args.test_dir)}")
    print(f" Output Directory  : {os.path.abspath(args.out_dir)}")
    print(f" Target Gene       : {args.gene}")
    print(f" Target Step       : {args.step} -> {STEP_LABELS.get(canonical_step, canonical_step)}")
    print(f" LOAOCV Enabled    : {args.loaocv or canonical_step == 'cv'}")
    print(f" Legacy Split Mode : {args.legacy_split}")
    print(f" QC Min Reads      : > {args.min_mapped_reads}")
    print(f" QC Min Map %      : > {args.min_map_pc}")
    print(f" Min Cells / Gene  : {args.min_cells_per_gene}")
    print(f" HVG Selection     : n_hvg={args.n_hvg}, method='{args.hvg_method}'")
    print(f" Depth Norm        : {args.depth_norm}")
    print(f" Regress Cell Cycle: {args.regress_cell_cycle}")
    if any_exclusion:
        print(f" Exclude Cell Cycle: {args.exclude_cell_cycle}")
        if args.exclude_genes_file:
            print(f" Exclude Genes File: {os.path.abspath(args.exclude_genes_file)}")
        if args.exclude_genes:
            print(f" Exclude Inline    : {args.exclude_genes}")
        print(f" Exclusion Steps   : {', '.join(sorted(exclude_active_steps))}")
    else:
        print(" Gene Exclusions   : None (original notebook baseline)")
    print("==========================================================================")

    exp_train_dir = os.path.join(args.train_dir, "expression")
    if not os.path.exists(exp_train_dir):
        for candidate in [args.test_dir, os.path.join("data", "exp"), "exp"]:
            if candidate and os.path.exists(candidate):
                exp_train_dir = candidate
                break

    # Step 1 (qc): Quality Control & Metadata Harmonization
    if canonical_step in ["all", "1"]:
        print("\n>>> STEP 1 (qc): Quality Control & Metadata Harmonization")
        cmd = [
            python_exe, "01_qc_and_metadata_filtering.py",
            "--metadata_file", args.metadata_file,
            "--train_dir", args.train_dir,
            "--test_dir", args.test_dir,
            "--out_dir", args.out_dir,
            "--min_mapped_reads", str(args.min_mapped_reads),
            "--min_map_pc", str(args.min_map_pc),
        ]
        if args.legacy_split:
            cmd.append("--legacy_split")
        run_command(cmd)

    # Step 2 (rankings): Gene Expression Rankings & FCS/CVW Cell Mapping
    if canonical_step in ["all", "2"]:
        print(f"\n>>> STEP 2 (rankings): Gene Expression Rankings & FCS/CVW Mapping for '{args.gene}'")
        cmd = append_exclusion_flags([
            python_exe, "02_gene_expression_rankings.py",
            "--gene", args.gene,
            "--exp_dir", exp_train_dir,
            "--out_dir", args.out_dir,
            "--depth_norm", args.depth_norm,
            "--regress_cell_cycle", args.regress_cell_cycle,
        ], "2")
        run_command(cmd)

    # Step 3 (umap): Dimensionality Reduction (PCA & UMAP)
    if canonical_step in ["all", "3"]:
        print("\n>>> STEP 3 (umap): Dimensionality Reduction (Scanpy PCA & UMAP)")
        cmd = append_exclusion_flags([
            python_exe, "03_dimensionality_reduction.py",
            "--exp_dir", exp_train_dir,
            "--out_dir", args.out_dir,
            "--depth_norm", args.depth_norm,
            "--regress_cell_cycle", args.regress_cell_cycle,
            "--n_hvg", str(args.n_hvg),
            "--hvg_method", args.hvg_method,
            "--min_cells_per_gene", str(args.min_cells_per_gene),
        ], "3")
        run_command(cmd)

    # Step 4 (correlations): Morphology-to-Gene Spearman Rank Correlations
    if canonical_step in ["all", "4"]:
        print("\n>>> STEP 4 (correlations): Morphology-to-Gene Spearman Rank Correlation Analysis")
        cmd = append_exclusion_flags([
            python_exe, "04_morphology_gene_correlations.py",
            "--qc_meta", os.path.join(args.out_dir, "qc_filtered_metadata.csv"),
            "--exp_dir", exp_train_dir,
            "--train_dir", args.train_dir,
            "--test_dir", args.test_dir,
            "--out_dir", args.out_dir,
            "--min_mapped_reads", str(args.min_mapped_reads),
            "--min_map_pc", str(args.min_map_pc),
            "--depth_norm", args.depth_norm,
            "--regress_cell_cycle", args.regress_cell_cycle,
            "--n_hvg", str(args.n_hvg),
            "--hvg_method", args.hvg_method,
            "--min_cells_per_gene", str(args.min_cells_per_gene),
        ], "4")
        run_command(cmd)

    # Step 5 (mofa): Multi-Omics Factor Analysis (MOFA) Integration
    if canonical_step in ["all", "5"]:
        print(f"\n>>> STEP 5 (mofa): Multi-Omics Factor Analysis (MOFA2, factors={args.mofa_factors}, RNA likelihood='{args.rna_likelihood}')")
        cmd = append_exclusion_flags([
            python_exe, "05_mofa_multimodal_analysis.py",
            "--exp_dir", exp_train_dir,
            "--out_dir", args.out_dir,
            "--factors", str(args.mofa_factors),
            "--rna_likelihood", args.rna_likelihood,
            "--rna_norm", args.rna_norm,
            "--depth_norm", args.depth_norm,
            "--regress_cell_cycle", args.regress_cell_cycle,
            "--n_hvg", str(args.n_hvg),
            "--hvg_method", args.hvg_method,
            "--min_cells_per_gene", str(args.min_cells_per_gene),
        ], "5")
        run_command(cmd)

    # Step 5b (mvae): PyTorch Multimodal VAE (run when explicitly selected via --step 5b or --step mvae)
    if canonical_step == "5b":
        print("\n>>> STEP 5b (mvae): PyTorch Multimodal VAE (MVAE) Integration")
        cmd = append_exclusion_flags([
            python_exe, "05b_mvae_multimodal_analysis.py",
            "--exp_dir", exp_train_dir,
            "--out_dir", args.out_dir,
            "--depth_norm", args.depth_norm,
            "--regress_cell_cycle", args.regress_cell_cycle,
            "--n_hvg", str(args.n_hvg),
            "--hvg_method", args.hvg_method,
            "--min_cells_per_gene", str(args.min_cells_per_gene),
        ], "5b")
        run_command(cmd)

    # Step 6 (enrichment): Biological Pathway Enrichment & WordCloud
    if canonical_step in ["all", "6"]:
        print("\n>>> STEP 6 (enrichment): Biological Pathway Enrichment & WordCloud Generation")
        cmd = [
            python_exe, "06_pathway_enrichment.py",
            "--gene_list", f"{args.gene.upper()},PF4,RUNX1,ITGA2B,MPL,FLI1,GATA1,NFE2",
            "--out_dir", args.out_dir
        ]
        run_command(cmd)

    # Step CV (loaocv): Leave-One-Animal-Out Cross-Validation (LOAOCV)
    if canonical_step in ["all", "cv"] or args.loaocv:
        print("\n>>> STEP CV (loaocv): Leave-One-Animal-Out Cross-Validation (LOAOCV)")
        cmd = [
            python_exe, "leave_one_animal_out_cv.py",
            "--metadata_file", args.metadata_file,
            "--qc_meta", os.path.join(args.out_dir, "qc_filtered_metadata.csv"),
            "--out_dir", args.out_dir,
            "--min_mapped_reads", str(args.min_mapped_reads),
            "--min_map_pc", str(args.min_map_pc),
        ]
        run_command(cmd)

    print("\n==========================================================================")
    print("      ALL MORPHOTRANSCRIPTOMICS ANALYSIS STAGES COMPLETED SUCCESSFULLY     ")
    print(f" Outputs saved to: {os.path.abspath(args.out_dir)}")
    print("==========================================================================")


if __name__ == "__main__":
    main()
