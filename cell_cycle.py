"""
cell_cycle.py
-------------
Cited gene-set filtering utilities for single-cell transcriptomics.

Provides transparent, non-parametric gene exclusion (filtering) prior to HVG selection,
dimensionality reduction (PCA/UMAP), and multimodal integration (MOFA2/MVAE), along with
an auditable log of every excluded gene and its provenance.

NOTE ON METHODOLOGY:
This module intentionally does NOT perform discrete cell-cycle phase inference
(G1/S/G2M classification) or score-based linear regression (`sc.pp.regress_out`).
Megakaryocytes undergo endomitosis (aborted mitosis without cytokinesis to reach
4n-32n+ ploidy), so diploid mitotic phase classifiers derived from proliferating
cancer cells (Tirosh et al., 2016) over-specify endomitotic states, and regressing
out cell-cycle scores risks removing genuine polyploidization biology
(Luecken & Theis, 2019).

Citations:
1. Tirosh, I. et al. (2016). Dissecting the multicellular ecosystem of metastatic
   melanoma by single-cell RNA-seq. Science, 352(6282), 189-196.
   DOI: 10.1126/science.aad0501
2. Stuart, T. et al. (2019). Comprehensive Integration of Single-Cell Data.
   Cell, 177(7), 1888-1902. (Seurat v3 `cc.genes.updated.2019` HGNC symbol update).
   DOI: 10.1016/j.cell.2019.05.031
3. Luecken, M. D., & Theis, F. J. (2019). Current best practices in single-cell
   RNA-seq analysis: a tutorial. Molecular Systems Biology, 15(6), e8746.
   DOI: 10.15252/msb.20188746
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


TIROSH_CITATION = (
    "Tirosh et al. (2016) Science 352:189-196 (DOI:10.1126/science.aad0501); "
    "Seurat cc.genes.updated.2019, Stuart et al. (2019) Cell 177:1888-1902 (DOI:10.1016/j.cell.2019.05.031)"
)

# Exact 43 S-phase-associated genes from Seurat `cc.genes.updated.2019` (Tirosh et al., 2016),
# formatted in standard Mus musculus TitleCase symbol convention.
TIROSH_2016_S_GENES_MOUSE: List[str] = [
    "Mcm5", "Pcna", "Tyms", "Fen1", "Mcm7", "Mcm4", "Rrm1", "Ung", "Gins2",
    "Mcm6", "Cdca7", "Dtl", "Prim1", "Uhrf1", "Cenpu", "Hells", "Rfc2",
    "Polr1b", "Nasp", "Rad51ap1", "Gmnn", "Wdr76", "Slbp", "Ccne2", "Ubr7",
    "Pold3", "Msh2", "Atad2", "Rad51", "Rrm2", "Cdc45", "Cdc6", "Exo1",
    "Tipin", "Dscc1", "Blm", "Casp8ap2", "Usp1", "Clspn", "Pola1", "Chaf1b",
    "Mrpl36", "E2f8",
]

# Exact 54 G2/M-phase-associated genes from Seurat `cc.genes.updated.2019` (Tirosh et al., 2016),
# formatted in standard Mus musculus TitleCase symbol convention.
TIROSH_2016_G2M_GENES_MOUSE: List[str] = [
    "Hmgb2", "Cdk1", "Nusap1", "Ube2c", "Birc5", "Tpx2", "Top2a", "Ndc80",
    "Cks2", "Nuf2", "Cks1b", "Mki67", "Tmpo", "Cenpf", "Tacc3", "Pimreg",
    "Smc4", "Ccnb2", "Ckap2l", "Ckap2", "Aurkb", "Bub1", "Kif11", "Anp32e",
    "Tubb4b", "Gtse1", "Kif20b", "Hjurp", "Cdca3", "Jpt1", "Cdc20", "Ttk",
    "Cdc25c", "Kif2c", "Rangap1", "Ncapd2", "Dlgap5", "Cdca2", "Cdca8",
    "Ect2", "Kif23", "Hmmr", "Aurka", "Psrc1", "Anln", "Lbr", "Ckap5",
    "Cenpe", "Ctcf", "Nek2", "G2e3", "Gas2l3", "Cbx5", "Cenpa",
]

# Historical symbols from original Tirosh et al. (2016) / older GRCm38 Ensembl annotations
# that were renamed in `cc.genes.updated.2019` (included to avoid silent misses on older refs).
TIROSH_2016_LEGACY_ALIASES_MOUSE: Dict[str, str] = {
    "Mlf1ip": "Cenpu",
    "Rpa2": "Polr1b",
    "Brip1": "Mrpl36",
    "Mcm2": "Mcm7",
    "Fam64a": "Pimreg",
    "Hn1": "Jpt1",
}


def get_cited_cell_cycle_genes(
    include_legacy_aliases: bool = True,
) -> Dict[str, Tuple[str, str]]:
    """
    Returns a dictionary mapping UPPERCASE gene symbol -> (source_label, citation).
    Contains the exact 97 canonical genes from Tirosh et al. (2016) / Seurat
    cc.genes.updated.2019 (43 S + 54 G2/M), plus optional legacy annotation aliases.
    """
    gene_map: Dict[str, Tuple[str, str]] = {}
    for g in TIROSH_2016_S_GENES_MOUSE:
        gene_map[g.upper()] = ("Tirosh2016_Seurat2019_S", TIROSH_CITATION)
    for g in TIROSH_2016_G2M_GENES_MOUSE:
        gene_map[g.upper()] = ("Tirosh2016_Seurat2019_G2M", TIROSH_CITATION)
    if include_legacy_aliases:
        for legacy, modern in TIROSH_2016_LEGACY_ALIASES_MOUSE.items():
            if legacy.upper() not in gene_map:
                gene_map[legacy.upper()] = (
                    f"Tirosh2016_LegacyAlias_for_{modern}",
                    TIROSH_CITATION,
                )
    return gene_map


def load_exclude_genes_file(filepath: str) -> List[str]:
    """
    Loads a user-supplied list of gene symbols to exclude from a .txt, .csv, or .tsv file.

    Supported formats:
      - Plain text (.txt): one gene symbol per line (blank lines and '#' comments ignored).
      - Delimited table (.csv / .tsv): uses column matching ('gene_symbol', 'gene', 'symbol',
        'mgi_symbol', 'mgi symbol', 'feature', 'name') if present, otherwise first column.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Exclude genes file not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    genes: List[str] = []

    if ext in (".csv", ".tsv"):
        sep = "\t" if ext == ".tsv" else ","
        df = pd.read_csv(filepath, sep=sep)
        preferred = {"gene_symbol", "gene", "symbol", "mgi_symbol", "mgi symbol", "feature", "name"}
        target_col = df.columns[0]
        for col in df.columns:
            if str(col).strip().lower() in preferred:
                target_col = col
                break
        genes = [str(v).strip() for v in df[target_col].dropna().tolist() if str(v).strip()]
    else:
        with open(filepath, "r", encoding="utf-8") as fh:
            for line in fh:
                cleaned = line.strip()
                if not cleaned or cleaned.startswith("#"):
                    continue
                # Take first whitespace/comma/tab-delimited token if extra annotation exists
                token = cleaned.replace(",", " ").split()[0].strip()
                if token:
                    genes.append(token)

    print(f"[Gene Filter] Loaded {len(genes)} gene symbols from: {os.path.abspath(filepath)}")
    return genes


def filter_excluded_genes(
    gene_names: Union[List[str], pd.Index],
    exclude_cell_cycle: bool = False,
    exclude_genes_file: Optional[str] = None,
    exclude_genes: Optional[Union[str, List[str]]] = None,
    audit_csv_path: Optional[str] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Builds a boolean keep-mask (True = keep, False = exclude) for `gene_names`
    using case-insensitive matching against:
      1. Cited Tirosh et al. (2016) / Seurat cc.genes.updated.2019 cell cycle genes
         (if `exclude_cell_cycle=True`).
      2. External gene list file (if `exclude_genes_file` is provided).
      3. Inline comma-separated or list gene symbols (if `exclude_genes` is provided).

    Writes an auditable CSV of every excluded gene and its provenance if
    `audit_csv_path` is provided and at least one gene is excluded.
    """
    exclude_lookup: Dict[str, Tuple[str, str]] = {}

    if exclude_cell_cycle:
        exclude_lookup.update(get_cited_cell_cycle_genes(include_legacy_aliases=True))

    if exclude_genes_file:
        file_genes = load_exclude_genes_file(exclude_genes_file)
        file_label = f"CustomFile:{os.path.basename(exclude_genes_file)}"
        for g in file_genes:
            key = g.upper()
            if key not in exclude_lookup:
                exclude_lookup[key] = (file_label, os.path.abspath(exclude_genes_file))

    if exclude_genes:
        if isinstance(exclude_genes, str):
            inline_list = [g.strip() for g in exclude_genes.split(",") if g.strip()]
        else:
            inline_list = [str(g).strip() for g in exclude_genes if str(g).strip()]
        for g in inline_list:
            key = g.upper()
            if key not in exclude_lookup:
                exclude_lookup[key] = ("CLI_Inline_Exclude", "User CLI --exclude_genes")

    if not exclude_lookup:
        return np.ones(len(gene_names), dtype=bool), pd.DataFrame(
            columns=["gene_symbol", "source", "citation"]
        )

    keep_mask = np.ones(len(gene_names), dtype=bool)
    audit_rows: List[Dict[str, str]] = []

    for idx, raw_gene in enumerate(gene_names):
        g_str = str(raw_gene)
        key = g_str.upper()
        if key in exclude_lookup:
            keep_mask[idx] = False
            source, citation = exclude_lookup[key]
            audit_rows.append(
                {"gene_symbol": g_str, "source": source, "citation": citation}
            )

    audit_df = pd.DataFrame(audit_rows, columns=["gene_symbol", "source", "citation"])
    n_removed = len(audit_rows)
    print(f"[Gene Filter] Excluded {n_removed} genes out of {len(gene_names)} total features.")

    if audit_csv_path and n_removed > 0:
        os.makedirs(os.path.dirname(os.path.abspath(audit_csv_path)), exist_ok=True)
        audit_df.to_csv(audit_csv_path, index=False)
        print(f"[Gene Filter] Saved excluded-gene audit log to: {os.path.abspath(audit_csv_path)}")

    return keep_mask, audit_df
