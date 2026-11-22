"""
supp_contamination_morphology_ridge.py
--------------------------------------
Supplementary Analysis: Predicting Contamination Status from Imaging Flow Cytometry Morphology
via Nested Leave-One-Animal-Out Cross-Validation (LOAOCV) Logistic Ridge Regression.

Evaluates whether wells flagged as `Contaminated == True` in the unified multi-animal metadata
(`data/Metadata_merged_YL_SF_20.10.26_v2.csv`) can be inferred purely from single-cell imaging
flow cytometry morphology parameters (97 numeric optical/morphological features across all
1,204 non-control single cells with valid morphology measurements across Mice M1-M7).

Methodology (Nested LOAOCV):
  - Outer loop (7 folds, Mice M1-M7):
      Hold out all single cells from mouse M_k as an unseen biological replicate test set.
  - Inner loop (6 folds over the remaining 6 training mice):
      Perform Leave-One-Animal-Out CV across a log-spaced grid of L2 regularization strengths
      C in [1e-4, 1e3], fitting `StandardScaler` strictly within each inner training split,
      and select C maximizing mean inner-fold ROC-AUC.
  - Refit L2-regularized Logistic Regression (class_weight="balanced") on all 6 training mice
    with the selected C and evaluate strictly out-of-sample on holdout mouse M_k.
  - Statistical significance of pooled out-of-fold ROC-AUC is assessed via within-mouse label
    permutation testing (1,000 permutations preserving per-animal contamination prevalence).

Outputs (saved to --out_dir):
  - supp_contamination_morphology_ridge.png          (4-panel publication supplementary figure)
  - supp_contamination_ridge_fold_metrics.csv        (Per-animal LOAOCV metrics & selected C)
  - supp_contamination_ridge_cell_predictions.csv    (Cell-level out-of-fold probabilities)
  - supp_contamination_ridge_coefficients.csv        (Mean +/- SD standardized L2 coefficients)

Usage:
    python supp_contamination_morphology_ridge.py --metadata_file "data/Metadata_merged_YL_SF_20.10.26_v2.csv" --out_dir "output"
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
    balanced_accuracy_score,
    accuracy_score,
    f1_score,
    brier_score_loss,
)


METADATA_NON_MORPH_COLS = {
    "Well_ID", "X", "orig.ident", "nCount_RNA", "nFeature_RNA", "SampleID",
    "Sample_Well", "Sample_Plate", "Contaminated", "Contaminated_bool",
    "meta_1", "control", "is_control", "number_of_cell", "single_cell",
    "trim.galore.nb_prefilter", "trim.galore.nb_postfilter", "trim.galore.pc_keep",
    "kallisto.nb_mapped", "kallisto.pc_map", "mouse", "fold",
    "Supervised_ploidy", "percent.mt", "dataset", "dataset_type",
    "DEG_artifact_label", "Cell_cycle", "Artefact_type",
}


def setup_plot_style():
    """Configures high-DPI matplotlib publication rendering matching project standards."""
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    sns.set_theme(style="whitegrid", context="paper")


def extract_all_morphology_features(df):
    """
    Extracts all numeric imaging flow cytometry morphology parameters from the metadata table,
    excluding sequencing QC, sample ID, and biological annotation columns.
    """
    morph_cols = [
        c for c in df.columns
        if c not in METADATA_NON_MORPH_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not morph_cols:
        pattern = r".*\(Imaging\).*|.*Axis.*|.*Mass.*|.*Intensity.*|.*Moment.*|.*Size.*|.*Diffusivity.*|.*Eccentricity.*|FSC-|SSC-|CD41-|DNA-"
        morph_cols = [
            c for c in df.columns
            if re.search(pattern, c) and pd.api.types.is_numeric_dtype(df[c])
        ]
    return morph_cols


def prepare_contamination_dataset(metadata_file):
    """
    Loads the unified metadata file, standardizes animal/contamination columns,
    and retains all non-control single cells with valid imaging morphology features.
    """
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    sep = "\t" if metadata_file.lower().endswith(".tsv") else ","
    df = pd.read_csv(metadata_file, sep=sep)

    # Standardize SampleID
    if "SampleID" not in df.columns or df["SampleID"].isna().all():
        for c in ["Well_ID", "Well", "Barcode"]:
            if c in df.columns:
                df["SampleID"] = df[c]
                break
        else:
            df["SampleID"] = df.iloc[:, 0].astype(str)
    df["SampleID"] = df["SampleID"].astype(str)

    # Standardize mouse ID
    if "mouse" not in df.columns or df["mouse"].isna().all():
        for c in ["Mouse_Plate", "Sample_Plate", "orig.ident", "SampleID"]:
            if c in df.columns:
                extracted = df[c].astype(str).str.extract(r"^(?:10plate_)?(?:Mouse|M)?(\d+)", expand=False, flags=re.IGNORECASE)
                df["mouse"] = extracted.apply(lambda x: f"M{x}" if pd.notna(x) else "Unknown")
                break
    else:
        df["mouse"] = df["mouse"].astype(str).str.strip()

    # Standardize Contaminated boolean target
    contam_col = None
    for c in ["Contaminated", "contaminated", "Contaminated_bool"]:
        if c in df.columns:
            contam_col = c
            break
    if contam_col is None:
        raise ValueError("No 'Contaminated' column found in metadata file.")

    df["Contaminated_bool"] = df[contam_col].apply(
        lambda x: True if str(x).strip().upper() in ["TRUE", "T", "1"] else False
    )

    # Filter to non-control single cells with imaging morphology measurements
    if "control" in df.columns:
        is_ctrl = df["control"].apply(lambda x: str(x).strip().upper() in ["TRUE", "T", "1"])
        df = df[~is_ctrl].copy()
    if "number_of_cell" in df.columns:
        df = df[pd.to_numeric(df["number_of_cell"], errors="coerce") == 1].copy()

    morph_cols = extract_all_morphology_features(df)
    has_morph = ~df[morph_cols].isna().all(axis=1)
    df = df[has_morph].copy()

    # Drop any zero-variance morphology features
    morph_cols = [c for c in morph_cols if df[c].std(skipna=True) > 0]

    # Exclude Unknown animals
    df = df[df["mouse"] != "Unknown"].copy()
    df.reset_index(drop=True, inplace=True)
    return df, morph_cols


def run_nested_loaocv_logistic_ridge(
    df,
    morph_cols,
    c_grid=None,
    class_weight="balanced",
    n_permutations=1000,
    seed=42,
):
    """
    Runs Nested Leave-One-Animal-Out Cross-Validation for L2-regularized Logistic Regression
    (Logistic Ridge Regression) predicting `Contaminated_bool` from imaging morphology.
    """
    if c_grid is None:
        c_grid = np.logspace(-4, 3, 29)
    else:
        c_grid = np.asarray(c_grid, dtype=float)

    mice = sorted(df["mouse"].unique())
    X_raw = df[morph_cols].copy()
    X = X_raw.values.astype(np.float64)
    y = df["Contaminated_bool"].astype(int).values
    groups = df["mouse"].values
    datasets = df["dataset"].values if "dataset" in df.columns else np.array(["Combined"] * len(df))

    oof_probs = np.zeros(len(df), dtype=np.float64)
    oof_preds = np.zeros(len(df), dtype=int)
    fold_rows = []
    coef_list = []
    inner_auc_matrix = np.zeros((len(mice), len(c_grid)), dtype=np.float64)
    per_mouse_roc = {}

    for fold_idx, holdout in enumerate(mice):
        tr_idx = np.where(groups != holdout)[0]
        te_idx = np.where(groups == holdout)[0]
        tr_mice = sorted(np.unique(groups[tr_idx]))

        # Inner LOAOCV over the remaining training mice to select C
        for c_idx, c_val in enumerate(c_grid):
            inner_aucs = []
            for val_m in tr_mice:
                in_tr = tr_idx[groups[tr_idx] != val_m]
                in_va = tr_idx[groups[tr_idx] == val_m]

                # Impute any sporadic NaNs using inner-train medians only
                tr_medians = np.nanmedian(X[in_tr], axis=0)
                X_in_tr_raw = np.where(np.isnan(X[in_tr]), tr_medians, X[in_tr])
                X_in_va_raw = np.where(np.isnan(X[in_va]), tr_medians, X[in_va])

                scaler = StandardScaler()
                X_in_tr = scaler.fit_transform(X_in_tr_raw)
                X_in_va = scaler.transform(X_in_va_raw)

                clf = LogisticRegression(
                    C=float(c_val),
                    class_weight=class_weight,
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=seed,
                )
                clf.fit(X_in_tr, y[in_tr])
                p_va = clf.predict_proba(X_in_va)[:, 1]
                inner_aucs.append(roc_auc_score(y[in_va], p_va))

            inner_auc_matrix[fold_idx, c_idx] = float(np.mean(inner_aucs))

        best_c_idx = int(np.argmax(inner_auc_matrix[fold_idx]))
        best_c = float(c_grid[best_c_idx])
        best_inner_auc = float(inner_auc_matrix[fold_idx, best_c_idx])

        # Refit on full outer training set (all mice except holdout)
        tr_medians = np.nanmedian(X[tr_idx], axis=0)
        X_tr_raw = np.where(np.isnan(X[tr_idx]), tr_medians, X[tr_idx])
        X_te_raw = np.where(np.isnan(X[te_idx]), tr_medians, X[te_idx])

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_raw)
        X_te_s = scaler.transform(X_te_raw)

        clf = LogisticRegression(
            C=best_c,
            class_weight=class_weight,
            solver="lbfgs",
            max_iter=2000,
            random_state=seed,
        )
        clf.fit(X_tr_s, y[tr_idx])
        p_te = clf.predict_proba(X_te_s)[:, 1]
        pred_te = clf.predict(X_te_s)

        oof_probs[te_idx] = p_te
        oof_preds[te_idx] = pred_te
        coef_list.append(clf.coef_[0])

        fpr, tpr, _ = roc_curve(y[te_idx], p_te)
        auc_te = float(roc_auc_score(y[te_idx], p_te))
        per_mouse_roc[holdout] = {
            "fpr": fpr,
            "tpr": tpr,
            "auc": auc_te,
            "dataset": str(datasets[te_idx[0]]),
        }

        fold_rows.append(
            {
                "Fold": fold_idx + 1,
                "Holdout_Animal": holdout,
                "Dataset_Cohort": str(datasets[te_idx[0]]),
                "Train_Cells": int(len(tr_idx)),
                "Test_Cells": int(len(te_idx)),
                "Test_Contaminated_N": int(y[te_idx].sum()),
                "Test_Contaminated_Rate": float(y[te_idx].mean()),
                "Selected_C": best_c,
                "Inner_LOAOCV_ROC_AUC": best_inner_auc,
                "ROC_AUC": auc_te,
                "PR_AUC": float(average_precision_score(y[te_idx], p_te)),
                "Balanced_Accuracy": float(balanced_accuracy_score(y[te_idx], pred_te)),
                "Accuracy": float(accuracy_score(y[te_idx], pred_te)),
                "F1_Score": float(f1_score(y[te_idx], pred_te, zero_division=0)),
                "Brier_Score": float(brier_score_loss(y[te_idx], p_te)),
            }
        )

    fold_df = pd.DataFrame(fold_rows)

    # Pooled out-of-fold metrics
    pooled_auc = float(roc_auc_score(y, oof_probs))
    pooled_pr = float(average_precision_score(y, oof_probs))
    pooled_bal = float(balanced_accuracy_score(y, oof_preds))
    pooled_acc = float(accuracy_score(y, oof_preds))
    pooled_f1 = float(f1_score(y, oof_preds, zero_division=0))
    pooled_brier = float(brier_score_loss(y, oof_probs))
    pooled_fpr, pooled_tpr, _ = roc_curve(y, oof_probs)

    # Within-mouse permutation test preserving per-animal class prevalence
    rng = np.random.default_rng(seed)
    perm_aucs = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        y_perm = y.copy()
        for m in mice:
            idx = np.where(groups == m)[0]
            y_perm[idx] = rng.permutation(y_perm[idx])
        perm_aucs[i] = roc_auc_score(y_perm, oof_probs)
    perm_p = float((np.sum(perm_aucs >= pooled_auc) + 1) / (n_permutations + 1))

    # Feature coefficients summary across outer folds
    coef_arr = np.vstack(coef_list)
    coef_df = pd.DataFrame(
        {
            "Morphology_Feature": morph_cols,
            "Mean_Std_Coef": np.mean(coef_arr, axis=0),
            "SD_Std_Coef": np.std(coef_arr, axis=0),
            "Abs_Mean_Coef": np.abs(np.mean(coef_arr, axis=0)),
            "Sign_Consistency": np.maximum(
                np.mean(coef_arr > 0, axis=0), np.mean(coef_arr < 0, axis=0)
            ),
        }
    ).sort_values("Abs_Mean_Coef", ascending=False).reset_index(drop=True)

    # Cell-level predictions table
    pred_df = pd.DataFrame(
        {
            "SampleID": df["SampleID"].values,
            "mouse": df["mouse"].values,
            "Sample_Plate": df["Sample_Plate"].values if "Sample_Plate" in df.columns else "NA",
            "dataset": datasets,
            "Supervised_ploidy": df["Supervised_ploidy"].values if "Supervised_ploidy" in df.columns else "NA",
            "Contaminated_True": y.astype(bool),
            "Pred_Contaminated_Prob": oof_probs,
            "Pred_Contaminated_Label": oof_preds.astype(bool),
        }
    )

    summary = {
        "n_cells": int(len(df)),
        "n_contaminated": int(y.sum()),
        "n_clean": int((1 - y).sum()),
        "prevalence": float(y.mean()),
        "n_features": int(len(morph_cols)),
        "mean_auc": float(fold_df["ROC_AUC"].mean()),
        "sd_auc": float(fold_df["ROC_AUC"].std()),
        "mean_pr_auc": float(fold_df["PR_AUC"].mean()),
        "sd_pr_auc": float(fold_df["PR_AUC"].std()),
        "mean_bal_acc": float(fold_df["Balanced_Accuracy"].mean()),
        "sd_bal_acc": float(fold_df["Balanced_Accuracy"].std()),
        "pooled_auc": pooled_auc,
        "pooled_pr_auc": pooled_pr,
        "pooled_bal_acc": pooled_bal,
        "pooled_acc": pooled_acc,
        "pooled_f1": pooled_f1,
        "pooled_brier": pooled_brier,
        "perm_p_value": perm_p,
        "n_permutations": int(n_permutations),
        "pooled_fpr": pooled_fpr,
        "pooled_tpr": pooled_tpr,
    }

    return {
        "fold_df": fold_df,
        "pred_df": pred_df,
        "coef_df": coef_df,
        "inner_auc_matrix": inner_auc_matrix,
        "c_grid": c_grid,
        "per_mouse_roc": per_mouse_roc,
        "summary": summary,
    }


def plot_supplementary_figure(results, out_path):
    """
    Generates a 4-panel publication-grade supplementary figure summarizing the
    Nested LOAOCV Logistic Ridge Regression analysis of contamination from morphology.
    """
    setup_plot_style()
    fold_df = results["fold_df"]
    coef_df = results["coef_df"]
    inner_auc_matrix = results["inner_auc_matrix"]
    c_grid = results["c_grid"]
    per_mouse_roc = results["per_mouse_roc"]
    summary = results["summary"]

    fig, axes = plt.subplots(2, 2, figsize=(16.8, 12.2))
    fig.subplots_adjust(hspace=0.34, wspace=0.44, top=0.90, bottom=0.07, left=0.06, right=0.98)

    mice = fold_df["Holdout_Animal"].tolist()
    mouse_palette = {
        "M1": "#1f77b4",
        "M2": "#2ca02c",
        "M3": "#ff7f0e",
        "M4": "#d62728",
        "M5": "#9467bd",
        "M6": "#8c564b",
        "M7": "#e377c2",
    }

    # -------------------------------------------------------------------------
    # Panel A: Out-of-Fold ROC Curves (Per-Animal + Pooled OOF)
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    for m in mice:
        info = per_mouse_roc[m]
        ls = "--" if info["dataset"] == "4plate" else "-"
        ax.plot(
            info["fpr"],
            info["tpr"],
            color=mouse_palette.get(m, "#555555"),
            linestyle=ls,
            linewidth=1.5,
            alpha=0.80,
            label=f"{m} ({info['dataset']}, AUC={info['auc']:.3f})",
        )

    p_str = f"p < {1.0 / (summary['n_permutations'] + 1):.3f}" if summary["perm_p_value"] <= 1.0 / (summary["n_permutations"] + 1) + 1e-12 else f"p = {summary['perm_p_value']:.4f}"
    ax.plot(
        summary["pooled_fpr"],
        summary["pooled_tpr"],
        color="black",
        linewidth=2.8,
        label=f"Pooled OOF (AUC={summary['pooled_auc']:.3f}, {p_str})",
    )
    ax.plot([0, 1], [0, 1], color="#7f7f7f", linestyle=":", linewidth=1.3, label="Chance (AUC=0.500)")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10.5, fontweight="bold")
    ax.set_title("A. Nested LOAOCV ROC Curves (Morphology → Contamination)", fontsize=11.5, fontweight="bold", loc="left", pad=8)
    ax.legend(loc="lower right", fontsize=8.2, frameon=True, framealpha=0.92)
    ax.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Panel B: Per-Animal LOAOCV Performance Barplot
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    labels = [f"{r['Holdout_Animal']}\n({r['Dataset_Cohort']})" for _, r in fold_df.iterrows()] + ["Mean\n(±SD)"]
    x = np.arange(len(labels))
    width = 0.26

    auc_vals = fold_df["ROC_AUC"].tolist() + [summary["mean_auc"]]
    pr_vals = fold_df["PR_AUC"].tolist() + [summary["mean_pr_auc"]]
    bal_vals = fold_df["Balanced_Accuracy"].tolist() + [summary["mean_bal_acc"]]

    auc_err = [0] * len(mice) + [summary["sd_auc"]]
    pr_err = [0] * len(mice) + [summary["sd_pr_auc"]]
    bal_err = [0] * len(mice) + [summary["sd_bal_acc"]]

    b1 = ax.bar(x - width, auc_vals, width, label="ROC-AUC", color="#1f77b4", edgecolor="white", linewidth=0.6)
    b2 = ax.bar(x, pr_vals, width, label="PR-AUC", color="#ff7f0e", edgecolor="white", linewidth=0.6)
    b3 = ax.bar(x + width, bal_vals, width, label="Balanced Acc.", color="#2ca02c", edgecolor="white", linewidth=0.6)

    # Error bars only on the Mean (±SD) summary bars
    mean_x = x[-1]
    ax.errorbar(mean_x - width, summary["mean_auc"], yerr=summary["sd_auc"], fmt="none", ecolor="#222222", capsize=3.5, linewidth=1.2)
    ax.errorbar(mean_x, summary["mean_pr_auc"], yerr=summary["sd_pr_auc"], fmt="none", ecolor="#222222", capsize=3.5, linewidth=1.2)
    ax.errorbar(mean_x + width, summary["mean_bal_acc"], yerr=summary["sd_bal_acc"], fmt="none", ecolor="#222222", capsize=3.5, linewidth=1.2)

    ax.axhline(0.50, color="#444444", linestyle="--", linewidth=1.1, label="Chance ROC / Bal. Acc. (0.50)")
    ax.axhline(summary["prevalence"], color="#d62728", linestyle=":", linewidth=1.2, label=f"Contam. Prevalence ({summary['prevalence']:.2f})")

    # Annotate ROC-AUC values cleanly above bars (and above error bar cap for Mean)
    for bar, val, err in zip(b1, auc_vals, auc_err):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            min(val + err + 0.02, 0.99),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.8,
            fontweight="bold",
            color="#1f77b4",
        )

    ax.axvline(len(mice) - 0.5, color="#999999", linestyle="-", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.8)
    ax.set_ylim(0.0, 1.06)
    ax.set_ylabel("Out-of-Sample Score", fontsize=10.5, fontweight="bold")
    ax.set_title("B. Per-Animal Out-of-Sample Generalization (M1–M7)", fontsize=11.5, fontweight="bold", loc="left", pad=8)
    ax.legend(loc="upper right", fontsize=8.0, ncol=2, frameon=True, framealpha=0.92)
    ax.grid(True, axis="y", alpha=0.3)

    # -------------------------------------------------------------------------
    # Panel C: Inner-LOAOCV Regularization (C) Tuning Profile
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    mean_inner = np.mean(inner_auc_matrix, axis=0)
    sd_inner = np.std(inner_auc_matrix, axis=0)

    for i, m in enumerate(mice):
        ax.plot(
            c_grid,
            inner_auc_matrix[i],
            color=mouse_palette.get(m, "#888888"),
            alpha=0.45,
            linewidth=1.1,
        )

    ax.plot(c_grid, mean_inner, color="#1f2937", linewidth=2.5, label="Mean Inner-LOAOCV ROC-AUC")
    ax.fill_between(
        c_grid,
        mean_inner - sd_inner,
        mean_inner + sd_inner,
        color="#1f2937",
        alpha=0.14,
        label="± 1 SD across outer folds",
    )

    # Highlight selected C per outer fold
    selected_cs = fold_df["Selected_C"].values
    selected_aucs = fold_df["Inner_LOAOCV_ROC_AUC"].values
    ax.scatter(
        selected_cs,
        selected_aucs,
        color="#d62728",
        edgecolor="white",
        s=55,
        zorder=5,
        label=f"Selected C per fold (median={np.median(selected_cs):.4g})",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Inverse L2 Regularization Strength C (log scale; smaller C = stronger Ridge penalty)", fontsize=9.8, fontweight="bold")
    ax.set_ylabel("Inner-LOAOCV Mean ROC-AUC", fontsize=10.5, fontweight="bold")
    ax.set_title("C. Nested Inner-LOAOCV Hyperparameter Tuning (L2 Ridge C)", fontsize=11.5, fontweight="bold", loc="left", pad=8)
    ax.legend(loc="lower left", fontsize=8.3, frameon=True, framealpha=0.92)
    ax.grid(True, which="both", alpha=0.25)

    # -------------------------------------------------------------------------
    # Panel D: Top 15 Standardized Logistic Ridge Coefficients
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    top_k = min(15, len(coef_df))
    top_df = coef_df.head(top_k).iloc[::-1].copy()

    colors = ["#d62728" if v > 0 else "#1f77b4" for v in top_df["Mean_Std_Coef"]]
    y_pos = np.arange(len(top_df))

    ax.barh(
        y_pos,
        top_df["Mean_Std_Coef"],
        xerr=top_df["SD_Std_Coef"],
        color=colors,
        alpha=0.88,
        capsize=3,
        edgecolor="white",
        linewidth=0.6,
    )
    ax.axvline(0.0, color="#333333", linewidth=1.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_df["Morphology_Feature"], fontsize=8.5)
    ax.set_xlabel("Mean Standardized Logistic Ridge Coefficient (± SD across 7 folds)", fontsize=9.8, fontweight="bold")
    ax.set_title("D. Top Morphology Predictors of Contamination Status", fontsize=11.5, fontweight="bold", loc="left", pad=8)

    # Custom legend for sign
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#d62728", alpha=0.88, label="Higher odds of Contamination (+β)"),
        Patch(facecolor="#1f77b4", alpha=0.88, label="Lower odds of Contamination (−β)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8.3, frameon=True, framealpha=0.92)
    ax.grid(True, axis="x", alpha=0.3)

    # Overall super-title
    fig.suptitle(
        "Supplementary Figure: Inferring Contamination Status from Imaging Flow Cytometry Morphology\n"
        f"Nested LOAOCV Logistic Ridge Regression (n = {summary['n_cells']:,} single cells across M1–M7 | "
        f"{summary['n_contaminated']} Contaminated [{summary['prevalence']*100:.1f}%], "
        f"{summary['n_clean']} Clean | {summary['n_features']} morphology features)",
        fontsize=13.0,
        fontweight="bold",
        y=0.975,
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] Supplementary figure: {os.path.abspath(out_path)}")


def main():
    parser = argparse.ArgumentParser(
        description="Supplementary Analysis: Logistic Ridge Regression predicting Contamination from Morphology via Nested LOAOCV"
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default=os.path.join("data", "Metadata_merged_YL_SF_20.10.26_v2.csv"),
        help="Path to unified multi-animal metadata CSV (default: data/Metadata_merged_YL_SF_20.10.26_v2.csv)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="output",
        help="Directory to save supplementary figure and CSV outputs (default: output)",
    )
    parser.add_argument(
        "--n_permutations",
        type=int,
        default=1000,
        help="Number of within-mouse permutations for pooled OOF ROC-AUC p-value (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print("==========================================================================")
    print("  SUPPLEMENTARY ANALYSIS: CONTAMINATION PREDICTION FROM MORPHOLOGY       ")
    print("  Nested Leave-One-Animal-Out CV (LOAOCV) Logistic Ridge Regression      ")
    print("==========================================================================")

    df, morph_cols = prepare_contamination_dataset(args.metadata_file)
    print(f" Metadata File     : {os.path.abspath(args.metadata_file)}")
    print(f" Single Cells (n)  : {len(df):,} ({int(df['Contaminated_bool'].sum())} Contaminated [{df['Contaminated_bool'].mean()*100:.1f}%], {int((~df['Contaminated_bool']).sum())} Clean)")
    print(f" Morphology Feats  : {len(morph_cols)} imaging flow cytometry features")
    print(f" Animals (Folds)   : {sorted(df['mouse'].unique())}")

    results = run_nested_loaocv_logistic_ridge(
        df,
        morph_cols,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )

    fold_df = results["fold_df"]
    pred_df = results["pred_df"]
    coef_df = results["coef_df"]
    s = results["summary"]

    print("\n--- Per-Animal Nested LOAOCV Results ---")
    print(fold_df[["Holdout_Animal", "Dataset_Cohort", "Test_Cells", "Test_Contaminated_Rate", "Selected_C", "ROC_AUC", "PR_AUC", "Balanced_Accuracy", "F1_Score"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(
        f"\nMean per-animal : ROC-AUC = {s['mean_auc']:.3f} +/- {s['sd_auc']:.3f} | "
        f"PR-AUC = {s['mean_pr_auc']:.3f} +/- {s['sd_pr_auc']:.3f} | "
        f"Balanced Acc = {s['mean_bal_acc']:.3f} +/- {s['sd_bal_acc']:.3f}"
    )
    print(
        f"Pooled OOF      : ROC-AUC = {s['pooled_auc']:.3f} (within-mouse perm p = {s['perm_p_value']:.4f}) | "
        f"PR-AUC = {s['pooled_pr_auc']:.3f} (baseline = {s['prevalence']:.3f}) | "
        f"Balanced Acc = {s['pooled_bal_acc']:.3f} | Acc = {s['pooled_acc']:.3f} | F1 = {s['pooled_f1']:.3f}"
    )

    # Save tabular outputs
    fold_csv = os.path.join(args.out_dir, "supp_contamination_ridge_fold_metrics.csv")
    pred_csv = os.path.join(args.out_dir, "supp_contamination_ridge_cell_predictions.csv")
    coef_csv = os.path.join(args.out_dir, "supp_contamination_ridge_coefficients.csv")
    fig_path = os.path.join(args.out_dir, "supp_contamination_morphology_ridge.png")

    # Append pooled summary row to fold_df for convenient downstream inspection
    pooled_row = pd.DataFrame(
        [
            {
                "Fold": "Pooled_OOF",
                "Holdout_Animal": "ALL_M1_M7",
                "Dataset_Cohort": "Combined",
                "Train_Cells": s["n_cells"],
                "Test_Cells": s["n_cells"],
                "Test_Contaminated_N": s["n_contaminated"],
                "Test_Contaminated_Rate": s["prevalence"],
                "Selected_C": float(np.median(fold_df["Selected_C"])),
                "Inner_LOAOCV_ROC_AUC": float(fold_df["Inner_LOAOCV_ROC_AUC"].mean()),
                "ROC_AUC": s["pooled_auc"],
                "PR_AUC": s["pooled_pr_auc"],
                "Balanced_Accuracy": s["pooled_bal_acc"],
                "Accuracy": s["pooled_acc"],
                "F1_Score": s["pooled_f1"],
                "Brier_Score": s["pooled_brier"],
            }
        ]
    )
    pd.concat([fold_df, pooled_row], ignore_index=True).to_csv(fold_csv, index=False)
    pred_df.to_csv(pred_csv, index=False)
    coef_df.to_csv(coef_csv, index=False)

    print(f"[Saved] Fold metrics : {os.path.abspath(fold_csv)}")
    print(f"[Saved] Predictions  : {os.path.abspath(pred_csv)}")
    print(f"[Saved] Coefficients : {os.path.abspath(coef_csv)}")

    plot_supplementary_figure(results, fig_path)


if __name__ == "__main__":
    main()
