"""
leave_one_animal_out_cv.py
--------------------------
Leave-One-Animal-Out Cross-Validation (LOAOCV / LOGO-CV) for Single-Cell Morphotranscriptomics.

Replaces the legacy pilot/validation 2-dataset split with a rigorous cross-validation framework
iterating across all biological replicates (Mice M1 through M7).

For each mouse M_k:
  - Training / Reference set: all cells from the other 6 mice
  - Test / Holdout set: all cells from mouse M_k (both plates, e.g. M1p1 and M1p2)
  - Evaluates out-of-sample generalizability of cellular morphology for ploidy state
    prediction (2n, 4n, 8n, 16n, 32n) and latent space projections.

Outputs:
  - output/loaocv_summary_metrics.csv
  - output/loaocv_cell_predictions.csv
  - output/loaocv_metrics_barplot.png
  - output/loaocv_confusion_matrix.png
  - output/loaocv_continuous_ploidy_scatter.png
  - output/loaocv_pca_projection.png

Usage:
    python leave_one_animal_out_cv.py --metadata_file "C:\\Users\\vilsn\\Dropbox\\Work\\Turing\\Exploration_Data\\Metadata_merged_YL_SF_20.10.26_v2.csv" --out_dir "output"
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    r2_score,
    confusion_matrix,
    mean_squared_error
)
from scipy.stats import spearmanr, pearsonr


def setup_plot_style():
    """Configures high-DPI matplotlib publication rendering matching project standards."""
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.autolayout'] = True
    sns.set_theme(style="whitegrid")


def extract_morphology_features(df):
    """
    Extracts imaging flow cytometry parameters matching regex pattern:
    .*(Imaging).*|.*Axis.*|.*Mass.*|.*Intensity.*|.*Moment.*|.*Size.*|.*Diffusivity.*|.*Eccentricity.*
    plus FSC/SSC channel parameters.
    Handles legacy naming conventions and imputes any sporadic missing values.
    """
    pattern = r'.*\(Imaging\).*|.*Axis.*|.*Mass.*|.*Intensity.*|.*Moment.*|.*Size.*|.*Diffusivity.*|.*Eccentricity.*|FSC-|SSC-|CD41-|DNA-'
    morph_cols = [c for c in df.columns if re.search(pattern, c)]
    numeric_morph_cols = [c for c in morph_cols if pd.api.types.is_numeric_dtype(df[c])]
    return numeric_morph_cols


def run_leave_one_animal_out_cv(qc_df, out_dir="output"):
    """
    Executes Leave-One-Animal-Out Cross-Validation across all mice present in the cohort.
    Backwards-compatible with legacy multi-dataset, multi-plate, and multi-file formats.
    """
    os.makedirs(out_dir, exist_ok=True)
    setup_plot_style()
    qc_df = qc_df.copy()

    # Robust animal / group identification with full backward compatibility
    if "mouse" not in qc_df.columns or qc_df["mouse"].nunique() <= 1:
        if "Mouse_Plate" in qc_df.columns:
            extracted = qc_df["Mouse_Plate"].astype(str).str.extract(r'^(?:Mouse|M)?(\d+)', expand=False, flags=re.IGNORECASE)
            qc_df["mouse"] = extracted.apply(lambda x: f"M{x}" if pd.notna(x) else "Unknown")
        elif "Sample_Plate" in qc_df.columns and qc_df["Sample_Plate"].astype(str).str.contains(r'^(?:Mouse|M)?\d+', regex=True).any():
            extracted = qc_df["Sample_Plate"].astype(str).str.extract(r'^(?:Mouse|M)?(\d+)', expand=False, flags=re.IGNORECASE)
            qc_df["mouse"] = extracted.apply(lambda x: f"M{x}" if pd.notna(x) else "Unknown")
        elif "dataset_type" in qc_df.columns and qc_df["dataset_type"].nunique() > 1:
            print("[LOAOCV] Notice: Animal IDs not present. Falling back to legacy dataset grouping (Leave-One-Dataset-Out)...")
            qc_df["mouse"] = qc_df["dataset_type"]
        elif "dataset" in qc_df.columns and qc_df["dataset"].nunique() > 1:
            print("[LOAOCV] Notice: Animal IDs not present. Falling back to legacy dataset grouping (Leave-One-Dataset-Out)...")
            qc_df["mouse"] = qc_df["dataset"]
        elif "Sample_Plate" in qc_df.columns and qc_df["Sample_Plate"].nunique() > 1:
            print("[LOAOCV] Notice: Animal IDs not present. Falling back to plate grouping (Leave-One-Plate-Out)...")
            qc_df["mouse"] = qc_df["Sample_Plate"]
        else:
            raise ValueError("Could not resolve biological replicates (mouse, plate, or dataset) for cross-validation.")

    # Standardize ploidy column if using legacy naming
    if "Supervised_ploidy" not in qc_df.columns:
        for c in ["Estimated_Ploidy_D.NA.A", "Supervised_Ploidy_Images", "Supervised_Ploidy", "ploidy"]:
            if c in qc_df.columns:
                qc_df["Supervised_ploidy"] = qc_df[c]
                break

    qc_df["Supervised_ploidy"] = qc_df["Supervised_ploidy"].astype(str).str.strip().str.lower()
    valid_ploidies = ["2n", "4n", "8n", "16n", "32n"]
    qc_df = qc_df[qc_df["Supervised_ploidy"].isin(valid_ploidies)].copy()

    mice = sorted([m for m in qc_df["mouse"].unique() if m != "Unknown"])
    print("==========================================================================")
    print("      LEAVE-ONE-ANIMAL-OUT CROSS-VALIDATION (LOAOCV) PIPELINE             ")
    print("==========================================================================")
    print(f" Total QC-Passed Cells : {len(qc_df)}")
    print(f" Unique Groups ({len(mice)})     : {mice}")
    for m in mice:
        n_m = (qc_df["mouse"] == m).sum()
        ploidies = qc_df.loc[qc_df["mouse"] == m, "Supervised_ploidy"].value_counts().to_dict()
        print(f"   - Group {m:2s}: {n_m:2d} cells | Ploidy distribution: {ploidies}")
    print("==========================================================================")

    # Features and Targets
    numeric_morph_cols = extract_morphology_features(qc_df)
    print(f"\nExtracted {len(numeric_morph_cols)} imaging morphology features for modeling.")

    # Fill any sparse NaNs with median of available data
    X_df = qc_df[numeric_morph_cols].fillna(qc_df[numeric_morph_cols].median())
    X_all = X_df.values
    y_class = qc_df["Supervised_ploidy"].values

    ploidy_map = {"2n": 2, "4n": 4, "8n": 8, "16n": 16, "32n": 32}
    y_num = np.array([ploidy_map.get(p, np.nan) for p in y_class])
    y_log_num = np.log2(y_num)

    fold_metrics = []
    cell_predictions = []
    pca_embeddings_dict = {}

    for fold_idx, holdout_mouse in enumerate(mice, 1):
        print(f"\n>>> Fold {fold_idx}/{len(mice)}: Holdout Mouse '{holdout_mouse}'")
        train_mask = (qc_df["mouse"] != holdout_mouse).values
        test_mask = (qc_df["mouse"] == holdout_mouse).values

        n_train = int(np.sum(train_mask))
        n_test = int(np.sum(test_mask))
        print(f"    Train cohort: {n_train} cells (all mice except {holdout_mouse})")
        print(f"    Test cohort : {n_test} cells (mouse {holdout_mouse} holdout)")

        # Scale features using reference cohort only
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_all[train_mask])
        X_test_scaled = scaler.transform(X_all[test_mask])

        y_train = y_class[train_mask]
        y_test = y_class[test_mask]

        y_train_num = y_log_num[train_mask]
        y_test_num = y_log_num[test_mask]

        # -------------------------------------------------------------
        # Model 1: Random Forest Classifier
        # -------------------------------------------------------------
        rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        rf.fit(X_train_scaled, y_train)
        y_pred_rf = rf.predict(X_test_scaled)

        acc_rf = accuracy_score(y_test, y_pred_rf)
        bal_acc_rf = balanced_accuracy_score(y_test, y_pred_rf)
        f1_macro_rf = f1_score(y_test, y_pred_rf, average="macro", zero_division=0)
        f1_weighted_rf = f1_score(y_test, y_pred_rf, average="weighted", zero_division=0)

        # -------------------------------------------------------------
        # Model 2: Logistic Regression (Balanced)
        # -------------------------------------------------------------
        lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        lr.fit(X_train_scaled, y_train)
        y_pred_lr = lr.predict(X_test_scaled)
        acc_lr = accuracy_score(y_test, y_pred_lr)
        bal_acc_lr = balanced_accuracy_score(y_test, y_pred_lr)

        # -------------------------------------------------------------
        # Model 3: Continuous Log2 Ploidy Regressor (Ridge)
        # -------------------------------------------------------------
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_scaled, y_train_num)
        y_pred_ridge = ridge.predict(X_test_scaled)

        # Map continuous regression predictions back to discrete ploidies
        rounded_log = np.clip(np.round(y_pred_ridge), 1, 5).astype(int)
        log_to_ploidy = {1: "2n", 2: "4n", 3: "8n", 4: "16n", 5: "32n"}
        y_pred_ridge_class = [log_to_ploidy[r] for r in rounded_log]
        acc_ridge_class = accuracy_score(y_test, y_pred_ridge_class)

        r2 = r2_score(y_test_num, y_pred_ridge)
        spear, _ = spearmanr(y_test_num, y_pred_ridge)
        pears, _ = pearsonr(y_test_num, y_pred_ridge)
        rmse = np.sqrt(mean_squared_error(y_test_num, y_pred_ridge))

        # -------------------------------------------------------------
        # Manifold: PCA fit on train, project test
        # -------------------------------------------------------------
        pca = PCA(n_components=10)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        pca_embeddings_dict[holdout_mouse] = {
            "test_pca": X_test_pca,
            "y_test": y_test,
            "explained_var": pca.explained_variance_ratio_[:2].sum()
        }

        print(f"    -> RF Accuracy: {acc_rf:.3f} | Balanced Acc: {bal_acc_rf:.3f} | Macro F1: {f1_macro_rf:.3f}")
        print(f"    -> Ridge Spearman Rho: {spear:.3f} | Pearson r: {pears:.3f} | RMSE: {rmse:.3f}")

        fold_metrics.append({
            "Fold": fold_idx,
            "Holdout_Animal": holdout_mouse,
            "Train_Cells": n_train,
            "Test_Cells": n_test,
            "RF_Accuracy": acc_rf,
            "RF_Balanced_Acc": bal_acc_rf,
            "RF_Macro_F1": f1_macro_rf,
            "RF_Weighted_F1": f1_weighted_rf,
            "LR_Accuracy": acc_lr,
            "LR_Balanced_Acc": bal_acc_lr,
            "Ridge_Class_Accuracy": acc_ridge_class,
            "Ridge_R2": r2,
            "Ridge_Pearson": pears,
            "Ridge_Spearman": spear,
            "Ridge_RMSE": rmse
        })

        # Record cell-level predictions
        test_indices = np.where(test_mask)[0]
        test_ids = qc_df.index[test_indices].tolist()
        plate_vals = qc_df.iloc[test_indices]["Sample_Plate"].values if "Sample_Plate" in qc_df.columns else ["NA"] * n_test

        for i in range(n_test):
            cell_predictions.append({
                "SampleID": test_ids[i],
                "mouse": holdout_mouse,
                "Sample_Plate": plate_vals[i],
                "True_Ploidy": y_test[i],
                "Pred_Ploidy_RF": y_pred_rf[i],
                "Pred_Ploidy_LR": y_pred_lr[i],
                "Pred_Ploidy_Ridge_Discretized": y_pred_ridge_class[i],
                "True_Log2_Ploidy": y_test_num[i],
                "Pred_Log2_Ploidy_Ridge": y_pred_ridge[i],
                "Correct_RF": bool(y_pred_rf[i] == y_test[i])
            })

    # Summary table
    metrics_df = pd.DataFrame(fold_metrics)
    summary_csv = os.path.join(out_dir, "loaocv_summary_metrics.csv")
    metrics_df.to_csv(summary_csv, index=False)
    print(f"\nSaved LOAOCV summary metrics to: {os.path.abspath(summary_csv)}")

    # Cell-level predictions table
    preds_df = pd.DataFrame(cell_predictions)
    preds_csv = os.path.join(out_dir, "loaocv_cell_predictions.csv")
    preds_df.to_csv(preds_csv, index=False)
    print(f"Saved cell-level out-of-fold predictions to: {os.path.abspath(preds_csv)}")

    # Overall pooled metrics
    overall_acc = accuracy_score(preds_df["True_Ploidy"], preds_df["Pred_Ploidy_RF"])
    overall_bal = balanced_accuracy_score(preds_df["True_Ploidy"], preds_df["Pred_Ploidy_RF"])
    overall_f1 = f1_score(preds_df["True_Ploidy"], preds_df["Pred_Ploidy_RF"], average="macro")
    overall_spear, _ = spearmanr(preds_df["True_Log2_Ploidy"], preds_df["Pred_Log2_Ploidy_Ridge"])
    overall_pears, _ = pearsonr(preds_df["True_Log2_Ploidy"], preds_df["Pred_Log2_Ploidy_Ridge"])

    print("\n==========================================================================")
    print("      LEAVE-ONE-ANIMAL-OUT CROSS-VALIDATION AGGREGATE RESULTS              ")
    print("==========================================================================")
    print(f" Mean RF Accuracy        : {metrics_df['RF_Accuracy'].mean():.4f} +/- {metrics_df['RF_Accuracy'].std():.4f}")
    print(f" Mean RF Balanced Acc    : {metrics_df['RF_Balanced_Acc'].mean():.4f} +/- {metrics_df['RF_Balanced_Acc'].std():.4f}")
    print(f" Mean RF Macro F1        : {metrics_df['RF_Macro_F1'].mean():.4f} +/- {metrics_df['RF_Macro_F1'].std():.4f}")
    print(f" Mean Ridge Spearman Rho : {metrics_df['Ridge_Spearman'].mean():.4f} +/- {metrics_df['Ridge_Spearman'].std():.4f}")
    print(f" Pooled Out-of-Fold Acc  : {overall_acc:.4f}")
    print(f" Pooled Out-of-Fold BalAcc: {overall_bal:.4f}")
    print(f" Pooled Out-of-Fold Spearman: {overall_spear:.4f}")
    print("==========================================================================")

    # -------------------------------------------------------------
    # Visualization 1: LOAOCV Performance per Animal Barplot
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_df = pd.melt(
        metrics_df,
        id_vars=["Holdout_Animal"],
        value_vars=["RF_Accuracy", "RF_Balanced_Acc", "RF_Macro_F1"],
        var_name="Metric",
        value_name="Score"
    )
    metric_labels = {
        "RF_Accuracy": "Overall Accuracy",
        "RF_Balanced_Acc": "Balanced Accuracy",
        "RF_Macro_F1": "Macro F1-Score"
    }
    bar_df["Metric"] = bar_df["Metric"].map(metric_labels)

    sns.barplot(data=bar_df, x="Holdout_Animal", y="Score", hue="Metric", ax=ax, palette="mako")
    ax.set_title("Leave-One-Animal-Out Cross-Validation: Ploidy Prediction per Holdout Mouse", fontsize=12, fontweight="bold")
    ax.set_xlabel("Holdout Animal (Evaluated on Unseen Mouse)")
    ax.set_ylabel("Out-of-Fold Score")
    ax.set_ylim(0, 1.05)
    ax.axhline(overall_acc, color="red", linestyle="--", alpha=0.7, label=f"Pooled Accuracy ({overall_acc:.2f})")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig1_path = os.path.join(out_dir, "loaocv_metrics_barplot.png")
    plt.savefig(fig1_path)
    plt.close()

    # -------------------------------------------------------------
    # Visualization 2: Pooled Out-of-Fold Confusion Matrix
    # -------------------------------------------------------------
    labels_order = ["2n", "4n", "8n", "16n", "32n"]
    present_labels = [lbl for lbl in labels_order if lbl in preds_df["True_Ploidy"].values]
    cm = confusion_matrix(preds_df["True_Ploidy"], preds_df["Pred_Ploidy_RF"], labels=present_labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=present_labels,
        yticklabels=present_labels,
        ax=ax,
        cbar_kws={'label': 'Normalized Recall'}
    )
    ax.set_title(f"LOAOCV: Pooled Out-of-Fold Ploidy Confusion Matrix\n(Overall Accuracy: {overall_acc:.1%})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Ploidy State")
    ax.set_ylabel("True Ploidy State")
    plt.tight_layout()
    fig2_path = os.path.join(out_dir, "loaocv_confusion_matrix.png")
    plt.savefig(fig2_path)
    plt.close()

    # -------------------------------------------------------------
    # Visualization 3: Continuous Log2 Ploidy Regression Scatter
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=preds_df,
        x="True_Log2_Ploidy",
        y="Pred_Log2_Ploidy_Ridge",
        hue="mouse",
        style="True_Ploidy",
        palette="tab10",
        s=65,
        alpha=0.85,
        ax=ax
    )
    # Regression trend line
    sns.regplot(
        data=preds_df,
        x="True_Log2_Ploidy",
        y="Pred_Log2_Ploidy_Ridge",
        scatter=False,
        ax=ax,
        color="black",
        line_kws={"linestyle": "--", "linewidth": 1.5, "label": f"Fit (Spearman rho = {overall_spear:.3f})"}
    )
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["2n (1)", "4n (2)", "8n (3)", "16n (4)", "32n (5)"])
    ax.set_title(f"LOAOCV: Out-of-Fold Continuous Ploidy Generalization\n(Spearman rho = {overall_spear:.3f}, Pearson r = {overall_pears:.3f})", fontsize=12, fontweight="bold")
    ax.set_xlabel("True Ploidy State (Log2 scale)")
    ax.set_ylabel("Predicted Log2 Ploidy (Ridge from Morphology)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    fig3_path = os.path.join(out_dir, "loaocv_continuous_ploidy_scatter.png")
    plt.savefig(fig3_path)
    plt.close()

    # -------------------------------------------------------------
    # Visualization 4: PCA Manifold Projections across Animal Folds
    # -------------------------------------------------------------
    scaler_full = StandardScaler()
    X_full_scaled = scaler_full.fit_transform(X_all)
    pca_full = PCA(n_components=2)
    pca_coords = pca_full.fit_transform(X_full_scaled)
    qc_df["PCA1"] = pca_coords[:, 0]
    qc_df["PCA2"] = pca_coords[:, 1]

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    sns.scatterplot(
        data=qc_df,
        x="PCA1",
        y="PCA2",
        hue="Supervised_ploidy",
        palette="crest",
        s=50,
        alpha=0.85,
        ax=ax[0]
    )
    ax[0].set_title(f"Morphology PCA Space: Colored by Ploidy State\n(PC1: {pca_full.explained_variance_ratio_[0]:.1%}, PC2: {pca_full.explained_variance_ratio_[1]:.1%})", fontsize=11, fontweight="bold")

    sns.scatterplot(
        data=qc_df,
        x="PCA1",
        y="PCA2",
        hue="mouse",
        palette="Set1",
        s=50,
        alpha=0.85,
        ax=ax[1]
    )
    ax[1].set_title("Morphology PCA Space: Colored by Animal (LOAOCV Replicates)", fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig4_path = os.path.join(out_dir, "loaocv_pca_projection.png")
    plt.savefig(fig4_path)
    plt.close()

    print(f"[LOAOCV Plotting] Saved all figures to: {os.path.abspath(out_dir)}")
    return metrics_df, preds_df


def main():
    parser = argparse.ArgumentParser(description="Leave-One-Animal-Out Cross-Validation for Single Cell Morphotranscriptomics")
    parser.add_argument("--metadata_file", type=str, default=r"C:\Users\vilsn\Dropbox\Work\Turing\Exploration_Data\Metadata_merged_YL_SF_20.10.26_v2.csv", help="Path to unified multi-animal metadata CSV")
    parser.add_argument("--qc_meta", type=str, default="output/qc_filtered_metadata.csv", help="Path to pre-filtered QC metadata CSV (optional)")
    parser.add_argument("--out_dir", type=str, default="output", help="Directory to save LOAOCV results and plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Ingest QC metadata
    if os.path.exists(args.qc_meta):
        print(f"Loading QC-filtered metadata from: {os.path.abspath(args.qc_meta)}")
        qc_df = pd.read_csv(args.qc_meta, index_col=0)
    else:
        print("QC metadata table not found in output directory. Running QC filtering from raw metadata...")
        from 01_qc_and_metadata_filtering import load_metadata_sources, apply_qc_filters
        raw_df = load_metadata_sources(metadata_file=args.metadata_file)
        _, qc_df = apply_qc_filters(raw_df)
        qc_df.to_csv(os.path.join(args.out_dir, "qc_filtered_metadata.csv"))

    run_leave_one_animal_out_cv(qc_df, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
