# Authored by Hilary Utaegbulam 

"""Testing pipeline, head-tail disambiguation, and advanced analysis plots."""
from __future__ import annotations
import os
import json
import csv
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.stats import pearsonr

from .constants import (
    Config, CLASS_BG, CLASS_MICHEL, CLASS_BREMS, CLASS_ID_TO_NAME,
)
from .physics_utils import (
    spatial_softmax2d_logits, dsnt_expectation, unit_from_points,
)
from .losses import (
    prior_bias_from_seg_dense, brems_sumcos,
)
from .models import DenseModel, DenseModelSMP, DenseModelConvNeXt
from .training_utils import replace_cfg, build_models
from .dataset import make_dense_loader, _pixel_energy_weight
from .training import run_epoch_dense
from .metrics import _fmt
from .visualization import (
    viz_sample_dense, save_consistency_analysis_plots,
    _save_cosine_histograms, _write_angle_csv_with_summary,
    compute_heatmap_confidence, _apply_priors_and_order_for_eval,
)

from scipy.stats import pearsonr

def eval_s1s2_core(model, batch_iterable, cfg, device, save_dir, view_tag="Z"):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    all_S1, all_S2 = [], []

    for batch in batch_iterable:
        if batch is None:
            continue
        img = batch["image"].to(device) # [B,1,H,W]
        out = model(img)

        logits_ab_adj, mx_sw, my_sw = _apply_priors_and_order_for_eval(out, cfg)

        if not (cfg.enable_seg and ("seg_logits" in out)):
            continue

        S1, S2, dS, c1, c2, Bmap = brems_sumcos(mx_sw, my_sw, out["seg_logits"], brems_idx=CLASS_BREMS)
        all_S1.extend(S1.detach().cpu().tolist())
        all_S2.extend(S2.detach().cpu().tolist())

    if len(all_S1) == 0:
        print("[S1/S2] No scores computed (seg head disabled or empty eval).")
        return None

    S1a = np.asarray(all_S1, dtype=np.float32)
    S2a = np.asarray(all_S2, dtype=np.float32)
    dSa = np.abs(S1a - S2a)

    # Plots
    bins = np.linspace(-1, 1, 50)

    plt.figure(figsize=(8,5), dpi=180)
    plt.hist(S1a, bins=bins, alpha=0.7, label='S1 (Start Score)')
    plt.hist(S2a, bins=bins, alpha=0.7, label='S2 (End Score)')
    plt.xlabel("Physics Score (weighted cosine)"); plt.ylabel("Count")
    plt.title(f"{view_tag}: Distribution of S1 and S2")
    plt.legend(); plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    f1 = os.path.join(save_dir, f"{view_tag}_hist_S1_S2.png"); plt.savefig(f1, dpi=600); plt.close()
    print(f"[S1/S2] Saved: {f1}")

    plt.figure(figsize=(8,5), dpi=180)
    for arr, lbl in [(S1a, "S1"), (S2a, "S2")]:
        vals = np.sort(arr); cdf = np.linspace(0,1,len(vals), endpoint=True)
        plt.plot(vals, cdf, label=lbl)
    plt.xlabel("Physics Score"); plt.ylabel("CDF")
    plt.title(f"{view_tag}: CDF of S1 and S2")
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    f2 = os.path.join(save_dir, f"{view_tag}_cdf_S1_S2.png"); plt.savefig(f2, dpi=600); plt.close()
    print(f"[S1/S2] Saved: {f2}")

    plt.figure(figsize=(8,5), dpi=180)
    plt.hist(dSa, bins=np.linspace(0,2,50), alpha=0.85)        
    plt.xlabel("|ΔS|"); plt.ylabel("Count")
    plt.title(f"{view_tag}: Distribution of |ΔS| = |S1 - S2|")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    f3 = os.path.join(save_dir, f"{view_tag}_hist_abs_dS.png"); plt.savefig(f3, dpi=600); plt.close()
    print(f"[S1/S2] Saved: {f3}")

    plt.figure(figsize=(5,5), dpi=180)
    plt.scatter(S1a, S2a, s=6, alpha=0.5)
    lims = [-1.0, 1.0]
    plt.plot(lims, lims, 'k--', linewidth=1)
    plt.xlim(lims); plt.ylim(lims)
    plt.xlabel("S1"); plt.ylabel("S2")
    plt.title(f"{view_tag}: S1 vs S2")
    plt.tight_layout()
    f4 = os.path.join(save_dir, f"{view_tag}_scatter_S1_vs_S2.png"); plt.savefig(f4, dpi=600); plt.close()
    print(f"[S1/S2] Saved: {f4}")

    summary = {
        "N": int(S1a.size),
        "S1_mean": float(S1a.mean()),  "S1_median": float(np.median(S1a)),
        "S2_mean": float(S2a.mean()),  "S2_median": float(np.median(S2a)),
        "abs_dS_mean": float(dSa.mean()),
        "frac_abs_dS_lt_0p05": float((dSa < 0.05).mean()),
        "frac_abs_dS_lt_0p10": float((dSa < 0.10).mean()),
    }
    return summary

def test_worker_single_view(gpu, view, cfg, results_dict):
    """
    Worker function for parallel testing on a specific GPU for one view.
    results_dict: shared dictionary to store results
    """
    torch.cuda.set_device(gpu)
    device = torch.device(f'cuda:{gpu}')
    
    result = run_testing(view, device, cfg)
    results_dict[view] = result


def run_testing(view: str, device, cfg: Config):
    """
    Full end-to-end test routine:
      * loads best checkpoint
      * computes overall test metrics
      * generates per-sample plots (single-panel arrows + Bragg 2x2)
      * collects and saves final histograms / confusion-matrix heatmap
    """
    print(f"\n==== TESTING View {view} ====")
    view_dir = os.path.join(cfg.out_dir, f"runs_{view}")
    os.makedirs(view_dir, exist_ok=True)

    cfg_test = replace_cfg(cfg, batch_size=8, multi_view=False, workers=0)
    test_ds, test_ld = make_dense_loader('test', view, cfg_test)
    print(f"    -> Found {len(test_ds)} samples in the test set.")

    if cfg.multi_view:
        view_dir = os.path.join(cfg.out_dir, "runs_multi")
        best_model_path = os.path.join(view_dir, 'best_checkpoint.pt')  
    else:
        view_dir = os.path.join(cfg.out_dir, f"runs_{view}")
        best_model_path = os.path.join(view_dir, 'best_checkpoint.pt') 

    if not os.path.exists(best_model_path):
        print(f" -> ERROR: Could not find best model at {best_model_path}. Probably wrong path?")
        return

    arch = cfg.dense_arch.lower()
    if arch == "convnextv2_unet":
        model = DenseModelConvNeXt(
            in_ch=1, feat_ch=cfg.dense_feat_ch, num_classes=cfg.num_classes,
            enable_seg=cfg.enable_seg, enable_dirreg=cfg.enable_dirreg,
            variant=cfg.cnv2_variant, decoder_scale=cfg.cnv2_decoder_scale,
            use_transpose=cfg.cnv2_use_transpose, skip_proj=cfg.cnv2_skip_proj,
        )
    elif arch == "smp_unet":
        model = DenseModelSMP(
            in_ch=cfg.smp_in_channels, feat_ch=cfg.dense_feat_ch, num_classes=cfg.num_classes,
            enable_seg=cfg.enable_seg, enable_dirreg=cfg.enable_dirreg,
            encoder_name=cfg.smp_encoder, encoder_weights=cfg.smp_encoder_weights,
            decoder_channels=cfg.smp_decoder_channels,
        )
    else:
        model = DenseModel(
            in_ch=1, base=cfg.dense_base, feat_ch=cfg.dense_feat_ch,
            num_classes=cfg.num_classes, enable_seg=cfg.enable_seg, enable_dirreg=cfg.enable_dirreg
        )

    ckpt = torch.load(best_model_path, map_location=device)

    if cfg.multi_view:
        if view not in ckpt:
            print(f"    -> ERROR: View '{view}' not found in checkpoint. Available: {list(ckpt.keys())}")
            return
        model_state = ckpt[view]['model'] 
        epoch_info = ckpt.get('epoch', 'N/A')
    else:
        if "model" not in ckpt:
            print("    -> ERROR: Checkpoint missing 'model' state_dict.")
            return
        model_state = ckpt["model"]
        epoch_info = ckpt.get('epoch', 'N/A')

    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    print(f"    -> Loaded best model from epoch {epoch_info}.")

    
    logs = run_epoch_dense(model, test_ld, opt=None, device=device, cfg=cfg,
                       epoch=cfg.epochs - 1, total_epochs=cfg.epochs, train=False)
    print(f"[{view}] TEST METRICS ({_fmt(logs)})")

   
    test_plot_dir = os.path.join(view_dir, "test_plots")
    os.makedirs(test_plot_dir, exist_ok=True)

    n_total_events = len(test_ds)
    n_plot_events = max(1, int(n_total_events * cfg.test_plot_fraction))
    
    plot_indices = set(random.sample(range(n_total_events), n_plot_events))  # Random


    # subfolders for hard cases
    gt90_dir  = os.path.join(test_plot_dir, "ang_gt90")
    gt150_dir = os.path.join(test_plot_dir, "ang_gt150")
    os.makedirs(gt90_dir, exist_ok=True)
    os.makedirs(gt150_dir, exist_ok=True)

    all_cos, all_weights, all_names = [], [], []


    all_angular_errors_deg, all_u_preds, all_u_gts = [], [], []
    all_brems_fractions, all_seg_preds, all_seg_gts = [], [], []
    all_confidences = []  # <=== NEW

    with torch.no_grad():
        for i, sample in enumerate(test_ds):
            if sample is None:
                continue
            should_plot = (i in plot_indices) and (not cfg.skip_per_event_plots)
            # Per-sample viz (returns metrics/arrays for summaries)
    
            cfg_temp = replace_cfg(cfg, skip_per_event_plots=(not should_plot))

            res = viz_sample_dense(cfg_temp, test_plot_dir, view, model, sample, device)
            if not isinstance(res, dict):
                continue

            if "angular_error_deg" in res and "cos_theta" in res:
                all_angular_errors_deg.append(float(res["angular_error_deg"]))
                all_cos.append(float(res["cos_theta"]))
                all_weights.append(_pixel_energy_weight(sample))
                # Use the base name of the same filename 
                base_name = sample["meta"]["fname"]
                all_names.append(base_name)

            # Grab "confidence"
            if "confidence" in res and res["confidence"] is not None:
                all_confidences.append(float(res["confidence"]))
            else:
                all_confidences.append(abs(float(res.get("cos_theta", 0.0))))

            if not cfg.skip_per_event_plots:
                try:
                    ang = float(res.get("angular_error_deg", 0.0))
                    paths = res.get("paths", {})
                    vec = paths.get("vec", None)
                    vec_orig = paths.get("vec_orig", None)

                    def _copy_if_exists(src_path: str, dst_dir: str):
                        if (src_path is not None) and os.path.exists(src_path):
                            shutil.copy(src_path, os.path.join(dst_dir, os.path.basename(src_path)))

                    if ang > 90.0:
                        _copy_if_exists(vec, gt90_dir)
                        _copy_if_exists(vec_orig, gt90_dir)
                    if ang > 150.0:
                        _copy_if_exists(vec, gt150_dir)
                        _copy_if_exists(vec_orig, gt150_dir)
                except Exception:
                    pass


            # Collectors for summary plots
            if "u_pred" in res:
                all_u_preds.append(np.asarray(res["u_pred"], dtype=float))
            if "u_gt" in res:
                all_u_gts.append(np.asarray(res["u_gt"], dtype=float))

            if res.get("brems_fraction") is not None:
                all_brems_fractions.append(float(res["brems_fraction"]))

            seg_pred = res.get("seg_pred", None)
            seg_gt   = res.get("seg_gt", None)
            if (seg_pred is not None) and (seg_gt is not None):
                all_seg_preds.append(np.asarray(seg_pred).ravel())
                all_seg_gts.append(np.asarray(seg_gt).ravel())

    print("    -> Building summary plots...")

    all_u_gts = np.array(all_u_gts)
    all_u_preds = np.array(all_u_preds)

    # Angular error histogram 
    if len(all_angular_errors_deg) > 0:
        ang_arr = np.asarray(all_angular_errors_deg, dtype=float)
        plt.figure(figsize=(5,4), dpi=150)
        plt.hist(ang_arr, bins=180)
        plt.xlabel("Angular error (degrees)")
        plt.ylabel("Count")
        plt.title(f"{view}: Angular Error Histogram (N={len(ang_arr)})")
        plt.tight_layout()
        out_ang = os.path.join(test_plot_dir, f"{view}_hist_angle_deg.png")
        plt.savefig(out_ang, dpi=600)
        plt.close()
        print(f"       - Saved: {out_ang}")

    # Angular error histogram with mean/median lines
    if len(all_angular_errors_deg) > 0:
        plt.figure(figsize=(10, 6), dpi=150)
        plt.hist(all_angular_errors_deg, bins=180, range=(0, 180), color='royalblue', alpha=0.75)
        plt.xlabel("Angular Error (Degrees)")
        plt.ylabel("Number of Events")
        plt.title(f"Angular Error Distribution for {view} View (N={len(all_angular_errors_deg)})")
        plt.grid(axis='y', alpha=0.5)
        mean_error = np.mean(all_angular_errors_deg) if all_angular_errors_deg else 0.0
        median_error = np.median(all_angular_errors_deg) if all_angular_errors_deg else 0.0
        plt.axvline(mean_error, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_error:.2f}°')
        plt.axvline(median_error, color='orange', linestyle='dashed', linewidth=2, label=f'Median: {median_error:.2f}°')
        plt.legend()
        plt.savefig(os.path.join(view_dir, f"summary_test_angular_error_hist_{view}.png"), dpi=600)
        plt.close()

    # Predicted vs. True Vector Components
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=600)
    ax1.scatter(all_u_gts[:, 0], all_u_preds[:, 0], alpha=0.3, s=10)
    ax1.plot([-1, 1], [-1, 1], 'r--', label='y=x (Ideal)')
    ax1.set_xlabel("True ux Component")
    ax1.set_ylabel("Predicted ux Component")
    ax1.set_title("ux Component Correlation")
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', 'box')
    ax2.scatter(all_u_gts[:, 1], all_u_preds[:, 1], alpha=0.3, s=10)
    ax2.plot([-1, 1], [-1, 1], 'r--', label='y=x (Ideal)')
    ax2.set_xlabel("True uy Component")
    ax2.set_ylabel("Predicted uy Component")
    ax2.set_title("uy Component Correlation")
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', 'box')
    plt.tight_layout()
    plt.savefig(os.path.join(view_dir, f"summary_test_components_{view}.png"), dpi=600)
    plt.close()

    # Angular error vs. Brems fraction (if brems exists)
    if all_brems_fractions:
        plt.figure(figsize=(10, 6), dpi=600)
        plt.scatter(all_brems_fractions, all_angular_errors_deg, alpha=0.4, s=15)
        plt.xlabel("Bremsstrahlung Fraction (Brems Pixels / Total Michel Pixels)")
        plt.ylabel("Angular Difference (Degrees)")
        plt.title(f"Model Performance vs. Bremsstrahlung Fraction - View {view}")
        plt.xlim(0, 1)
        plt.ylim(0, 180)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(view_dir, f"summary_test_error_vs_brems_{view}.png"), dpi=600)
        plt.close()

    # Brems fraction histogram 
    if len(all_brems_fractions) > 0:
        bf = np.asarray(all_brems_fractions, dtype=float)
        plt.figure(figsize=(5,4), dpi=600)
        plt.hist(bf, bins=80, range=(0.0, 1.0))
        plt.xlabel("Brems fraction in (Michel ∪ Brems) [GT]")
        plt.ylabel("Count")
        plt.title(f"{view}: Brems Fraction Histogram (N={len(bf)})")
        plt.tight_layout()
        out_bf = os.path.join(test_plot_dir, f"{view}_hist_brems_fraction.png")
        plt.savefig(out_bf, dpi=600)
        plt.close()
        print(f"       - Saved: {out_bf}")

    # Segmentation confusion matrix heat map
    if (len(all_seg_preds) > 0) and (len(all_seg_gts) > 0):
        y_pred = np.concatenate(all_seg_preds, axis=0)
        y_true = np.concatenate(all_seg_gts, axis=0)

        keep = (y_true != IGNORE_INDEX)
        y_true = y_true[keep]
        y_pred = y_pred[keep]

        labels_present = np.unique(np.concatenate([y_true, y_pred]))
        labels_present = labels_present[(labels_present >= 0) & (labels_present < int(cfg.num_classes))]

        if labels_present.size > 0:
           
            cm = confusion_matrix(y_true, y_pred, labels=labels_present)
            plt.figure(figsize=(6.4, 5.6), dpi=600)
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=True,
                xticklabels=[CLASS_ID_TO_NAME.get(int(i), str(int(i))) for i in labels_present],
                yticklabels=[CLASS_ID_TO_NAME.get(int(i), str(int(i))) for i in labels_present],
            )
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"{view}: Segmentation Confusion Matrix")
            plt.tight_layout()
            out_cm = os.path.join(test_plot_dir, f"{view}_seg_confusion_matrix.png")
            plt.savefig(out_cm, dpi=600)
            plt.close()
            print(f"        - Saved: {out_cm}")

            # Normalize the confusion matrix by row (true label) sums to get per-class accuracy
            with np.errstate(divide='ignore', invalid='ignore'):
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized) 

            plt.figure(figsize=(6.4, 5.6), dpi=600)
            sns.heatmap(
                cm_normalized,
                annot=True,
                fmt=".2f", 
                cmap="Blues",
                cbar=True,
                xticklabels=[CLASS_ID_TO_NAME.get(int(i), str(int(i))) for i in labels_present],
                yticklabels=[CLASS_ID_TO_NAME.get(int(i), str(int(i))) for i in labels_present],
            )
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"{view}: Normalized Segmentation Confusion Matrix")
            plt.tight_layout()
            
            out_cm_normalized = os.path.join(test_plot_dir, f"{view}_seg_confusion_matrix_normalized.png")
            plt.savefig(out_cm_normalized, dpi=600)
            plt.close()
            print(f"        - Saved: {out_cm_normalized}")
        else:
            print("       - No valid labels present to build confusion matrix.")

    if len(all_cos) > 0:
        cos_arr = np.asarray(all_cos, dtype=float)
        ang_arr = np.asarray(all_angular_errors_deg, dtype=float)
        w_arr   = np.asarray(all_weights, dtype=float)

        
        _save_cosine_histograms(view, cos_arr, w_arr, test_plot_dir)

        csv_path = os.path.join(test_plot_dir, f"{view}_angles_and_cos_summary.csv")
        _write_angle_csv_with_summary(csv_path, all_names, ang_arr, cos_arr, w_arr)
        print(f"       - Saved: {csv_path}")

    test_ds, test_ld = make_dense_loader(
        'test', view, cfg_test
    )

    save_dir = os.path.join(test_plot_dir, "physics_metrics")
    summary = eval_s1s2_core(model, test_ld, cfg, device, save_dir, view_tag=view)

    if summary is not None:
        with open(os.path.join(save_dir, f"{view}_s1s2_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    print("    -> Building additional analysis plots...")
    
    all_u_preds_arr = np.array(all_u_preds)
    all_u_gts_arr = np.array(all_u_gts)
    all_angular_errors_arr = np.array(all_angular_errors_deg)
    all_confidences_arr = np.array(all_confidences)
    
    if len(all_confidences_arr) > 0 and len(all_angular_errors_arr) > 0:
        correctness = (all_angular_errors_arr < 20.0).astype(float)
        plot_calibration_curve(
            all_confidences_arr, correctness,
            os.path.join(test_plot_dir, f"{view}_calibration.png"),
            view
        )
    
    # Confidence vs Error
    if len(all_confidences_arr) > 0 and len(all_angular_errors_arr) > 0:
        plot_confidence_vs_error(
            all_confidences_arr, all_angular_errors_arr,
            os.path.join(test_plot_dir, f"{view}_confidence_vs_error.png"),
            view
        )
    
    # Direction-specific plots
    if len(all_u_preds_arr) > 0 and len(all_u_gts_arr) > 0:
        plot_direction_polar(
            all_u_preds_arr, all_u_gts_arr, all_angular_errors_arr,
            os.path.join(test_plot_dir, f"{view}_direction_polar.png"),
            view
        )
        
        plot_predicted_direction_distribution(
            all_u_preds_arr, all_u_gts_arr,
            os.path.join(test_plot_dir, f"{view}_direction_distribution.png"),
            view
        )

    print(f"[{view}] Testing complete.")
    
    # Return information for multi-view analysis
    return {
        'view': view,
        'u_pred': all_u_preds_arr,
        'u_gt': all_u_gts_arr,
        'angular_errors': all_angular_errors_arr,
        'confidences': all_confidences_arr if len(all_confidences_arr) > 0 else None,
        'filenames': all_names
    }

def plot_calibration_curve(
    confidences: np.ndarray,
    correctness: np.ndarray,
    save_path: str,
    view: str,
    n_bins: int = 10
):
    """
    Reliability diagram (calibration plot).
    Should show if model confidence matches actual accuracy.
    """
    # Bin by confidence
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    bin_conf_means = []
    bin_acc_means = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = (bin_indices == i)
        if mask.sum() > 0:
            bin_conf_means.append(confidences[mask].mean())
            bin_acc_means.append(correctness[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_conf_means.append((bins[i] + bins[i+1]) / 2)
            bin_acc_means.append(0)
            bin_counts.append(0)
    
    bin_conf_means = np.array(bin_conf_means)
    bin_acc_means = np.array(bin_acc_means)
    bin_counts = np.array(bin_counts)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=600)
    
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')
    ax1.plot(bin_conf_means, bin_acc_means, 'o-', linewidth=2, markersize=8, label='Model')
    
    for i, (conf, acc, count) in enumerate(zip(bin_conf_means, bin_acc_means, bin_counts)):
        if count > 0:
            ax1.text(conf, acc + 0.03, str(count), ha='center', fontsize=8, color='gray')
    
    ax1.set_xlabel('Predicted Confidence', fontsize=12)
    ax1.set_ylabel('Actual Accuracy', fontsize=12)
    ax1.set_title(f'{view}: Calibration Curve\n(numbers = sample count per bin)', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Right: Histogram of confidences
    ax2.hist(confidences, bins=50, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Predicted Confidence', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title(f'{view}: Confidence Distribution', fontsize=13)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"       - Saved: {save_path}")


def plot_confidence_vs_error(
    confidences: np.ndarray,
    angular_errors: np.ndarray,
    save_path: str,
    view: str
):
    """
    Scatter plot: confidence vs angular error.
    Should show negative correlation, hopefully.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=600)
    
    # Left: Scatter plot
    ax1.scatter(confidences, angular_errors, alpha=0.3, s=10, c=angular_errors, 
                cmap='coolwarm', vmin=0, vmax=180)
    ax1.set_xlabel('Confidence', fontsize=12)
    ax1.set_ylabel('Angular Error (degrees)', fontsize=12)
    ax1.set_title(f'{view}: Confidence vs Error', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 180)
    

    z = np.polyfit(confidences, angular_errors, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(0, 1, 100)
    ax1.plot(x_trend, p(x_trend), "r--", linewidth=2, alpha=0.8, 
             label=f'Linear fit: y={z[0]:.1f}x+{z[1]:.1f}')
    ax1.legend(fontsize=10)
    
    # Compute correlation
    corr = np.corrcoef(confidences, angular_errors)[0, 1]
    ax1.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
             transform=ax1.transAxes, fontsize=11, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Right: Binned statistics
    bins = np.linspace(0, 1, 11)
    bin_indices = np.digitize(confidences, bins) - 1
    bin_indices = np.clip(bin_indices, 0, 9)
    
    bin_means = []
    bin_stds = []
    bin_centers = []
    
    for i in range(10):
        mask = (bin_indices == i)
        if mask.sum() > 5:  
            bin_means.append(angular_errors[mask].mean())
            bin_stds.append(angular_errors[mask].std())
            bin_centers.append((bins[i] + bins[i+1]) / 2)
    
    if bin_means:
        ax2.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='o-', 
                     linewidth=2, markersize=8, capsize=5)
        ax2.set_xlabel('Confidence Bin', fontsize=12)
        ax2.set_ylabel('Mean Angular Error (degrees)', fontsize=12)
        ax2.set_title(f'{view}: Error by Confidence Bin', fontsize=13)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"       - Saved: {save_path}")


def plot_direction_polar(
    u_pred: np.ndarray,
    u_gt: np.ndarray,
    angular_errors: np.ndarray,
    save_path: str,
    view: str,
    n_sectors: int = 12
):
    """
    Polar plot showing error vs true direction.
    Are there directional biases?
    """
    angles_gt = np.arctan2(u_gt[:, 1], u_gt[:, 0])  # Range: [-π, π]
    
    sector_edges = np.linspace(-np.pi, np.pi, n_sectors + 1)
    sector_centers = (sector_edges[:-1] + sector_edges[1:]) / 2
    sector_indices = np.digitize(angles_gt, sector_edges) - 1
    sector_indices = np.clip(sector_indices, 0, n_sectors - 1)
    
    sector_errors = []
    sector_counts = []
    
    for i in range(n_sectors):
        mask = (sector_indices == i)
        if mask.sum() > 0:
            sector_errors.append(angular_errors[mask].mean())
            sector_counts.append(mask.sum())
        else:
            sector_errors.append(0)
            sector_counts.append(0)
    
    sector_errors = np.array(sector_errors)
    sector_counts = np.array(sector_counts)
    
    fig = plt.figure(figsize=(10, 10), dpi=600)
    ax = fig.add_subplot(111, projection='polar')
    
    width = 2 * np.pi / n_sectors
    bars = ax.bar(sector_centers, sector_errors, width=width, alpha=0.7, 
                   edgecolor='black', linewidth=1.5)
    
    # Color by error magnitude
    colors = plt.cm.coolwarm(sector_errors / max(sector_errors.max(), 1))
    for bar, color in zip(bars, colors):
        bar.set_facecolor(color)
    
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_title(f'{view}: Mean Angular Error by True Direction\n' + 
                 f'(Rose diagram with {n_sectors} sectors)', 
                 fontsize=14, pad=20)
    ax.set_ylabel('Mean Angular Error (degrees)', fontsize=11, labelpad=30)
    
    
    for angle, error, count in zip(sector_centers, sector_errors, sector_counts):
        if count > 0:
            ax.text(angle, error + 2, str(count), ha='center', fontsize=8, color='gray')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"       - Saved: {save_path}")


def plot_predicted_direction_distribution(
    u_pred: np.ndarray,
    u_gt: np.ndarray,
    save_path: str,
    view: str
):
    """
    2D histogram showing coverage of direction space.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=600)
    
    
    ax = axes[0]
    h = ax.hist2d(u_pred[:, 0], u_pred[:, 1], bins=50, range=[[-1, 1], [-1, 1]], 
                  cmap='viridis', cmin=1)
    plt.colorbar(h[3], ax=ax, label='Count')
    ax.set_xlabel('ux (predicted)', fontsize=12)
    ax.set_ylabel('uy (predicted)', fontsize=12)
    ax.set_title(f'{view}: Predicted Direction Distribution', fontsize=13)
    ax.set_aspect('equal')
    
    circle = plt.Circle((0, 0), 1, fill=False, color='red', linewidth=2, linestyle='--')
    ax.add_patch(circle)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.grid(True, alpha=0.3)
    
    # True directions (for comparison)
    ax = axes[1]
    h = ax.hist2d(u_gt[:, 0], u_gt[:, 1], bins=50, range=[[-1, 1], [-1, 1]], 
                  cmap='viridis', cmin=1)
    plt.colorbar(h[3], ax=ax, label='Count')
    ax.set_xlabel('ux (ground truth)', fontsize=12)
    ax.set_ylabel('uy (ground truth)', fontsize=12)
    ax.set_title(f'{view}: True Direction Distribution', fontsize=13)
    ax.set_aspect('equal')
    
    
    circle = plt.Circle((0, 0), 1, fill=False, color='red', linewidth=2, linestyle='--')
    ax.add_patch(circle)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"       - Saved: {save_path}")

def plot_multiview_error_correlation(
    view_errors: dict,
    save_path: str
):
    """
    Correlation matrix: when U gets it wrong, do V/Z also get it wrong?
    """
    
    views = ['U', 'V', 'Z']
    n_views = len(views)
    
    # Compute correlation matrix
    corr_matrix = np.zeros((n_views, n_views))
    p_values = np.zeros((n_views, n_views))
    
    for i, v1 in enumerate(views):
        for j, v2 in enumerate(views):
            if i == j:
                corr_matrix[i, j] = 1.0
                p_values[i, j] = 0.0
            else:
                corr, p = pearsonr(view_errors[v1], view_errors[v2])
                corr_matrix[i, j] = corr
                p_values[i, j] = p
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 7), dpi=600)
    
    im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Pearson Correlation', fontsize=12)
    
    # Set ticks
    ax.set_xticks(np.arange(n_views))
    ax.set_yticks(np.arange(n_views))
    ax.set_xticklabels(views, fontsize=12)
    ax.set_yticklabels(views, fontsize=12)
    
    for i in range(n_views):
        for j in range(n_views):
            if i != j:
                sig = '***' if p_values[i, j] < 0.001 else ('**' if p_values[i, j] < 0.01 else ('*' if p_values[i, j] < 0.05 else ''))
                text = ax.text(j, i, f'{corr_matrix[i, j]:.3f}\n{sig}',
                              ha="center", va="center", color="black", fontsize=11, fontweight='bold')
    
    ax.set_title('Multi-View Error Correlation Matrix\n(* p<0.05, ** p<0.01, *** p<0.001)', 
                 fontsize=14, pad=15)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"       - Saved: {save_path}")


def plot_multiview_consistency(
    view_predictions: dict,
    save_path: str
):
    """
    3-view angular consistency plot.
    Shows spread between U/V/Z predictions for same event.
    """
    u_pred_U = view_predictions['U']
    u_pred_V = view_predictions['V']
    u_pred_Z = view_predictions['Z']
    
    N = u_pred_U.shape[0]
    
    # Compute pairwise angular differences for each event
    angle_UV = np.zeros(N)
    angle_UZ = np.zeros(N)
    angle_VZ = np.zeros(N)
    
    for i in range(N):
        # U vs V
        cos_uv = np.clip(np.dot(u_pred_U[i], u_pred_V[i]), -1, 1)
        angle_UV[i] = np.degrees(np.arccos(np.abs(cos_uv)))
        
        # U vs Z
        cos_uz = np.clip(np.dot(u_pred_U[i], u_pred_Z[i]), -1, 1)
        angle_UZ[i] = np.degrees(np.arccos(np.abs(cos_uz)))
        
        # V vs Z
        cos_vz = np.clip(np.dot(u_pred_V[i], u_pred_Z[i]), -1, 1)
        angle_VZ[i] = np.degrees(np.arccos(np.abs(cos_vz)))
    
    # Max spread per event
    max_spread = np.maximum(np.maximum(angle_UV, angle_UZ), angle_VZ)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=600)
    
    # Pairwise angle distributions
    ax = axes[0, 0]
    ax.hist(angle_UV, bins=50, alpha=0.5, label='U-V', color='blue')
    ax.hist(angle_UZ, bins=50, alpha=0.5, label='U-Z', color='green')
    ax.hist(angle_VZ, bins=50, alpha=0.5, label='V-Z', color='red')
    ax.set_xlabel('Angular Difference (degrees)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Pairwise View Disagreement', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Max spread distribution
    ax = axes[0, 1]
    ax.hist(max_spread, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax.set_xlabel('Max Angular Spread (degrees)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Maximum View Disagreement per Event\nMean: {max_spread.mean():.2f}°, Median: {np.median(max_spread):.2f}°', 
                 fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    ax.axvline(max_spread.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    ax.axvline(np.median(max_spread), color='orange', linestyle='--', linewidth=2, label='Median')
    ax.legend(fontsize=10)
    
    # Scatter UV vs UZ
    ax = axes[1, 0]
    ax.scatter(angle_UV, angle_UZ, alpha=0.3, s=10, c=max_spread, cmap='coolwarm', vmin=0, vmax=90)
    ax.set_xlabel('U-V Angular Difference (degrees)', fontsize=12)
    ax.set_ylabel('U-Z Angular Difference (degrees)', fontsize=12)
    ax.set_title('View Disagreement Correlation', fontsize=13)
    ax.plot([0, 90], [0, 90], 'k--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 90)
    
    # CDF
    ax = axes[1, 1]
    for angles, label, color in [(angle_UV, 'U-V', 'blue'), 
                                   (angle_UZ, 'U-Z', 'green'), 
                                   (angle_VZ, 'V-Z', 'red')]:
        sorted_angles = np.sort(angles)
        cdf = np.arange(1, len(sorted_angles) + 1) / len(sorted_angles)
        ax.plot(sorted_angles, cdf, label=label, linewidth=2, color=color)
    
    ax.set_xlabel('Angular Difference (degrees)', fontsize=12)
    ax.set_ylabel('CDF', fontsize=12)
    ax.set_title('Cumulative Distribution of View Disagreement', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 90)
    
    for p in [50, 90, 95]:
        val = np.percentile(max_spread, p)
        ax.axvline(val, color='gray', linestyle=':', alpha=0.5)
        ax.text(val, 0.05, f'{p}%\n{val:.1f}°', fontsize=8, ha='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"       - Saved: {save_path}")


def plot_voting_pattern_distribution(
    per_event_results: list,
    save_path: str,
    filter_consensus_wrong: bool = False
):
    """
    Voting Pattern Distribution (8 combinations).

    """
    # All 8 patterns
    patterns = [
        ('CCC', [True, True, True]),    # Unanimous correct
        ('CCF', [True, True, False]),   # U+V correct, Z wrong
        ('CFC', [True, False, True]),   # U+Z correct, V wrong
        ('FCC', [False, True, True]),   # V+Z correct, U wrong
        ('CFF', [True, False, False]),  # Only U correct
        ('FCF', [False, True, False]),  # Only V correct
        ('FFC', [False, False, True]),  # Only Z correct
        ('FFF', [False, False, False]), # Unanimous incorrect
    ]
    
    pattern_counts = {label: 0 for label, _ in patterns}
    
    for fname, votes, confidences, consensus_correct in per_event_results:
        if filter_consensus_wrong and consensus_correct:
            continue
        
        for label, pattern in patterns:
            if votes == pattern:
                pattern_counts[label] += 1
                break
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=600)
    
    labels = [label for label, _ in patterns]
    counts = [pattern_counts[label] for label in labels]
    colors = ['limegreen', 'gold', 'gold', 'gold', 'orange', 'orange', 'orange', 'crimson']
    
    bars = ax.bar(labels, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar, count in zip(bars, counts):
        if count > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Voting Pattern (U, V, Z)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    
    title_suffix = " (Consensus Wrong Only)" if filter_consensus_wrong else ""
    ax.set_title(f'Voting Pattern Distribution{title_suffix}\n' +
                 'C=Correct, F=Flipped | Green=Unanimous, Yellow=Majority, Orange=Minority, Red=All Wrong',
                 fontsize=13)
    
    ax.grid(axis='y', alpha=0.3)
    
    descriptions = [
        'All Correct',
        'U+V, Z wrong',
        'U+Z, V wrong',
        'V+Z, U wrong',
        'Only U',
        'Only V',
        'Only Z',
        'All Wrong'
    ]
    
    for i, (bar, desc) in enumerate(zip(bars, descriptions)):
        ax.text(i, -max(counts)*0.08, desc, ha='center', va='top', fontsize=8, rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"       - Saved: {save_path}")

def plot_voting_pattern_confusion_matrix_combined(
    per_event_results: list,
    save_path: str,
    show_percentages: bool = True
):
    """
    Confusion Matrix of All 8 Voting Patterns; DUAL VIEW.
    """
    pattern_groups = [
        ('CCC', [True, True, True], 'Unanimous Correct'),
        ('FCC', [False, True, True], 'Single Dissenter'),
        ('CFC', [True, False, True], 'Single Dissenter'),
        ('CCF', [True, True, False], 'Single Dissenter'),
        ('CFF', [True, False, False], 'Double Dissenter'),
        ('FCF', [False, True, False], 'Double Dissenter'),
        ('FFC', [False, False, True], 'Double Dissenter'),
        ('FFF', [False, False, False], 'Unanimous Incorrect'),
    ]
    
    pattern_stats = {}
    for label, pattern, group in pattern_groups:
        pattern_stats[label] = {
            'total': 0,
            'consensus_correct': 0,
            'consensus_wrong': 0,
            'group': group,
            'pattern': pattern
        }
    
    for fname, votes, confidences, consensus_correct in per_event_results:
        for label, pattern, group in pattern_groups:
            if votes == pattern:
                pattern_stats[label]['total'] += 1
                if consensus_correct:
                    pattern_stats[label]['consensus_correct'] += 1
                else:
                    pattern_stats[label]['consensus_wrong'] += 1
                break
            
    labels = [label for label, _, _ in pattern_groups]
    groups = [pattern_stats[label]['group'] for label in labels]
    
    matrix_data = []
    for label in labels:
        stats = pattern_stats[label]
        total = stats['total']
        correct = stats['consensus_correct']
        wrong = stats['consensus_wrong']
        accuracy = (correct / max(1, total)) * 100
        matrix_data.append([total, correct, wrong, accuracy])
    
    matrix_data = np.array(matrix_data)
    
    fig = plt.figure(figsize=(18, 14), dpi=600)
    gs = GridSpec(3, 2, height_ratios=[2.5, 2.5, 2], hspace=0.35, wspace=0.25)
    

    group_colors = {
        'Unanimous Correct': 'limegreen',
        'Single Dissenter': 'gold', 
        'Double Dissenter': 'orange',
        'Unanimous Incorrect': 'crimson'
    }
    
    ax_row = fig.add_subplot(gs[0, 0])
    
    matrix_row_norm = matrix_data.copy()
    for row in range(8):
        row_total = matrix_data[row, 0]
        if row_total > 0:
            # Columns 1-2: normalize by row total (shows proportions)
            matrix_row_norm[row, 1] = (matrix_data[row, 1] / row_total) * 100
            matrix_row_norm[row, 2] = (matrix_data[row, 2] / row_total) * 100
            # Column 0: normalize by global max (shows relative frequency)
            matrix_row_norm[row, 0] = (matrix_data[row, 0] / matrix_data[:, 0].max()) * 100
        else:
            matrix_row_norm[row, :3] = 0
    matrix_row_norm[:, 3] = matrix_data[:, 3]  # Accuracy already in %
    
    im_row = ax_row.imshow(matrix_row_norm, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    ax_row.set_xticks(np.arange(4))
    ax_row.set_yticks(np.arange(8))
    ax_row.set_xticklabels(['Total\n(rel. freq.)', 'Correct\n(% of row)', 
                            'Wrong\n(% of row)', 'Accuracy\n(%)'], 
                           fontsize=10, fontweight='bold')
    ax_row.set_yticklabels([f"{label}\n(U,V,Z)" for label in labels], 
                           fontsize=9, fontweight='bold')
    
    for i, (label, group) in enumerate(zip(labels, groups)):
        ax_row.text(-0.6, i, group.split()[0], va='center', ha='right', 
                   fontsize=8, fontweight='bold', color=group_colors[group],
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor=group_colors[group], linewidth=2))
    
    for i in range(8):
        for j in range(4):
            actual_value = matrix_data[i, j]
            norm_value = matrix_row_norm[i, j]
            
            if j == 0:  
                text = f'{int(actual_value)}'
            elif j == 3:  
                text = f'{actual_value:.1f}%'
            else:  
                text = f'{int(actual_value)}\n({norm_value:.0f}%)'
            
            text_color = 'white' if norm_value > 50 else 'black'
            ax_row.text(j, i, text, ha='center', va='center', 
                       fontsize=9, fontweight='bold', color=text_color)
    
    ax_row.set_title('ROW-NORMALIZED: Pattern Reliability\n' +
                    '"Given this voting pattern, what happens?"\n' +
                    'Bright GREEN in Correct = reliable pattern',
                    fontsize=12, fontweight='bold', pad=10)
    
    cbar_row = plt.colorbar(im_row, ax=ax_row, fraction=0.046, pad=0.04)
    cbar_row.set_label('Normalized %', fontsize=10)
    
    ax_col = fig.add_subplot(gs[0, 1])
    
    matrix_col_norm = matrix_data.copy()
    for col in range(3): 
        col_max = matrix_data[:, col].max()
        if col_max > 0:
            matrix_col_norm[:, col] = (matrix_data[:, col] / col_max) * 100
    matrix_col_norm[:, 3] = matrix_data[:, 3]  
    
    im_col = ax_col.imshow(matrix_col_norm, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    ax_col.set_xticks(np.arange(4))
    ax_col.set_yticks(np.arange(8))
    ax_col.set_xticklabels(['Total\n(% of max)', 'Correct\n(% of max)', 
                            'Wrong\n(% of max)', 'Accuracy\n(%)'], 
                           fontsize=10, fontweight='bold')
    ax_col.set_yticklabels([f"{label}\n(U,V,Z)" for label in labels], 
                           fontsize=9, fontweight='bold')
    
    for i, (label, group) in enumerate(zip(labels, groups)):
        ax_col.text(-0.6, i, group.split()[0], va='center', ha='right', 
                   fontsize=8, fontweight='bold', color=group_colors[group],
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor=group_colors[group], linewidth=2))
    
    for i in range(8):
        for j in range(4):
            actual_value = matrix_data[i, j]
            norm_value = matrix_col_norm[i, j]
            
            if j == 3:  # Accuracy
                text = f'{actual_value:.1f}%'
            else:  # Counts
                text = f'{int(actual_value)}\n({norm_value:.0f}%)'
            
            text_color = 'white' if norm_value > 50 else 'black'
            ax_col.text(j, i, text, ha='center', va='center', 
                       fontsize=9, fontweight='bold', color=text_color)
    
    ax_col.set_title('COLUMN-NORMALIZED: Failure Attribution\n' +
                    '"Within each outcome, which patterns dominate?"\n' +
                    'Bright in Wrong column = major failure contributor',
                    fontsize=12, fontweight='bold', pad=10)
    
   
    cbar_col = plt.colorbar(im_col, ax=ax_col, fraction=0.046, pad=0.04)
    cbar_col.set_label('% of Column Max', fontsize=10)
    
    
    ax_text = fig.add_subplot(gs[1, :])
    ax_text.axis('off')
    
    ax_bar = fig.add_subplot(gs[2, :])
    
    x = np.arange(8)
    width = 0.35
    
    bars1 = ax_bar.bar(x - width/2, matrix_data[:, 1], width, label='Consensus Correct', 
                       color='limegreen', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax_bar.bar(x + width/2, matrix_data[:, 2], width, label='Consensus Wrong', 
                       color='crimson', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax_bar.set_xlabel('Voting Pattern', fontsize=12, fontweight='bold')
    ax_bar.set_ylabel('Raw Count', fontsize=12, fontweight='bold')
    ax_bar.set_title('RAW COUNTS: Consensus Outcome by Voting Pattern', 
                     fontsize=13, fontweight='bold')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=10, fontweight='bold')
    ax_bar.legend(fontsize=11, loc='upper right')
    ax_bar.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax_bar.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
   
    group_ranges = {
        'Unanimous Correct': (0, 1),
        'Single Dissenter': (1, 4),
        'Double Dissenter': (4, 7),
        'Unanimous Incorrect': (7, 8)
    }
    
    for group, (start, end) in group_ranges.items():
        ax_bar.axvspan(start - 0.5, end - 0.5, alpha=0.1, 
                      color=group_colors[group], zorder=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"       - Saved: {save_path}")

def plot_voting_pattern_confusion_matrix_row_normalized(
    per_event_results: list,
    save_path: str
):

    pattern_groups = [
        ('CCC', [True, True, True], 'Unanimous Correct'),
        ('FCC', [False, True, True], 'Single Dissenter'),
        ('CFC', [True, False, True], 'Single Dissenter'),
        ('CCF', [True, True, False], 'Single Dissenter'),
        ('CFF', [True, False, False], 'Double Dissenter'),
        ('FCF', [False, True, False], 'Double Dissenter'),
        ('FFC', [False, False, True], 'Double Dissenter'),
        ('FFF', [False, False, False], 'Unanimous Incorrect'),
    ]
    
    pattern_stats = {}
    for label, pattern, group in pattern_groups:
        pattern_stats[label] = {
            'total': 0,
            'consensus_correct': 0,
            'consensus_wrong': 0,
            'group': group,
            'pattern': pattern
        }
    
    for fname, votes, confidences, consensus_correct in per_event_results:
        for label, pattern, group in pattern_groups:
            if votes == pattern:
                pattern_stats[label]['total'] += 1
                if consensus_correct:
                    pattern_stats[label]['consensus_correct'] += 1
                else:
                    pattern_stats[label]['consensus_wrong'] += 1
                break
    
    labels = [label for label, _, _ in pattern_groups]
    groups = [pattern_stats[label]['group'] for label in labels]
    
    matrix_data = []
    for label in labels:
        stats = pattern_stats[label]
        total = stats['total']
        correct = stats['consensus_correct']
        wrong = stats['consensus_wrong']
        accuracy = (correct / max(1, total)) * 100
        matrix_data.append([total, correct, wrong, accuracy])
    
    matrix_data = np.array(matrix_data)
    

    fig = plt.figure(figsize=(12, 10), dpi=600)
    gs = GridSpec(2, 1, height_ratios=[3, 2], hspace=0.3)
    
    group_colors = {
        'Unanimous Correct': 'limegreen',
        'Single Dissenter': 'gold', 
        'Double Dissenter': 'orange',
        'Unanimous Incorrect': 'crimson'
    }

    ax_main = fig.add_subplot(gs[0])
    
    matrix_row_norm = matrix_data.copy()
    for row in range(8):
        row_total = matrix_data[row, 0]
        if row_total > 0:
            matrix_row_norm[row, 1] = (matrix_data[row, 1] / row_total) * 100
            matrix_row_norm[row, 2] = (matrix_data[row, 2] / row_total) * 100
            matrix_row_norm[row, 0] = (matrix_data[row, 0] / matrix_data[:, 0].max()) * 100
        else:
            matrix_row_norm[row, :3] = 0
    matrix_row_norm[:, 3] = matrix_data[:, 3]
    

    im = ax_main.imshow(matrix_row_norm, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    
    ax_main.set_xticks(np.arange(4))
    ax_main.set_yticks(np.arange(8))
    ax_main.set_xticklabels(['Total\n(rel. freq.)', 'Consensus Correct\n(% of row)', 
                             'Consensus Wrong\n(% of row)', 'Accuracy\n(%)'], 
                            fontsize=11, fontweight='bold')
    ax_main.set_yticklabels([f"{label}\n(U,V,Z)" for label in labels], 
                            fontsize=10, fontweight='bold')
    

    for i, (label, group) in enumerate(zip(labels, groups)):
        ax_main.text(-0.6, i, group.split()[0], va='center', ha='right', 
                    fontsize=9, fontweight='bold', color=group_colors[group],
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             edgecolor=group_colors[group], linewidth=2))

    for i in range(8):
        for j in range(4):
            actual_value = matrix_data[i, j]
            norm_value = matrix_row_norm[i, j]
            
            if j == 0:
                text = f'{int(actual_value)}'
            elif j == 3:
                text = f'{actual_value:.1f}%'
            else:
                text = f'{int(actual_value)}\n({norm_value:.0f}%)'
            
            text_color = 'white' if norm_value > 50 else 'black'
            ax_main.text(j, i, text, ha='center', va='center', 
                        fontsize=10, fontweight='bold', color=text_color)
    
    ax_main.set_title('ROW-NORMALIZED: Pattern Reliability\n' +
                     '"Given this voting pattern, what happens?"\n' +
                     'Bright GREEN in Correct = reliable | Bright RED in Wrong = unreliable',
                     fontsize=13, fontweight='bold', pad=15)
    
 
    cbar = plt.colorbar(im, ax=ax_main, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized % (within row)', fontsize=11, fontweight='bold')
    
    ax_bar = fig.add_subplot(gs[1])
    
    x = np.arange(8)
    width = 0.35
    
    bars1 = ax_bar.bar(x - width/2, matrix_data[:, 1], width, label='Consensus Correct', 
                       color='limegreen', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax_bar.bar(x + width/2, matrix_data[:, 2], width, label='Consensus Wrong', 
                       color='crimson', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax_bar.set_xlabel('Voting Pattern', fontsize=12, fontweight='bold')
    ax_bar.set_ylabel('Raw Count', fontsize=12, fontweight='bold')
    ax_bar.set_title('Raw Counts by Voting Pattern', fontsize=13, fontweight='bold')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=10, fontweight='bold')
    ax_bar.legend(fontsize=11, loc='upper right')
    ax_bar.grid(axis='y', alpha=0.3)
    

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax_bar.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    group_ranges = {
        'Unanimous Correct': (0, 1),
        'Single Dissenter': (1, 4),
        'Double Dissenter': (4, 7),
        'Unanimous Incorrect': (7, 8)
    }
    
    for group, (start, end) in group_ranges.items():
        ax_bar.axvspan(start - 0.5, end - 0.5, alpha=0.1, 
                      color=group_colors[group], zorder=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"       - Saved: {save_path}")


def plot_voting_pattern_confusion_matrix_col_normalized(
    per_event_results: list,
    save_path: str
):
    """
    "Within each outcome, which patterns dominate?"
    """
    pattern_groups = [
        ('CCC', [True, True, True], 'Unanimous Correct'),
        ('FCC', [False, True, True], 'Single Dissenter'),
        ('CFC', [True, False, True], 'Single Dissenter'),
        ('CCF', [True, True, False], 'Single Dissenter'),
        ('CFF', [True, False, False], 'Double Dissenter'),
        ('FCF', [False, True, False], 'Double Dissenter'),
        ('FFC', [False, False, True], 'Double Dissenter'),
        ('FFF', [False, False, False], 'Unanimous Incorrect'),
    ]
    
    pattern_stats = {}
    for label, pattern, group in pattern_groups:
        pattern_stats[label] = {
            'total': 0,
            'consensus_correct': 0,
            'consensus_wrong': 0,
            'group': group,
            'pattern': pattern
        }
    
    for fname, votes, confidences, consensus_correct in per_event_results:
        for label, pattern, group in pattern_groups:
            if votes == pattern:
                pattern_stats[label]['total'] += 1
                if consensus_correct:
                    pattern_stats[label]['consensus_correct'] += 1
                else:
                    pattern_stats[label]['consensus_wrong'] += 1
                break
    
    labels = [label for label, _, _ in pattern_groups]
    groups = [pattern_stats[label]['group'] for label in labels]
    
    matrix_data = []
    for label in labels:
        stats = pattern_stats[label]
        total = stats['total']
        correct = stats['consensus_correct']
        wrong = stats['consensus_wrong']
        accuracy = (correct / max(1, total)) * 100
        matrix_data.append([total, correct, wrong, accuracy])
    
    matrix_data = np.array(matrix_data)
    
    fig = plt.figure(figsize=(12, 10), dpi=600)
    gs = GridSpec(2, 1, height_ratios=[3, 2], hspace=0.3)
    
    group_colors = {
        'Unanimous Correct': 'limegreen',
        'Single Dissenter': 'gold', 
        'Double Dissenter': 'orange',
        'Unanimous Incorrect': 'crimson'
    }
    
    ax_main = fig.add_subplot(gs[0])
    
    matrix_col_norm = matrix_data.copy()
    for col in range(3):
        col_max = matrix_data[:, col].max()
        if col_max > 0:
            matrix_col_norm[:, col] = (matrix_data[:, col] / col_max) * 100
    matrix_col_norm[:, 3] = matrix_data[:, 3]
    
    im = ax_main.imshow(matrix_col_norm, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    ax_main.set_xticks(np.arange(4))
    ax_main.set_yticks(np.arange(8))
    ax_main.set_xticklabels(['Total\n(% of max)', 'Consensus Correct\n(% of max)', 
                             'Consensus Wrong\n(% of max)', 'Accuracy\n(%)'], 
                            fontsize=11, fontweight='bold')
    ax_main.set_yticklabels([f"{label}\n(U,V,Z)" for label in labels], 
                            fontsize=10, fontweight='bold')
    
    for i, (label, group) in enumerate(zip(labels, groups)):
        ax_main.text(-0.6, i, group.split()[0], va='center', ha='right', 
                    fontsize=9, fontweight='bold', color=group_colors[group],
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             edgecolor=group_colors[group], linewidth=2))
    
    for i in range(8):
        for j in range(4):
            actual_value = matrix_data[i, j]
            norm_value = matrix_col_norm[i, j]
            
            if j == 3:
                text = f'{actual_value:.1f}%'
            else:
                text = f'{int(actual_value)}\n({norm_value:.0f}%)'
            
            text_color = 'white' if norm_value > 50 else 'black'
            ax_main.text(j, i, text, ha='center', va='center', 
                        fontsize=10, fontweight='bold', color=text_color)
    
    ax_main.set_title('COLUMN-NORMALIZED: Failure Attribution\n' +
                     '"Within each outcome, which patterns dominate?"\n' +
                     'Bright in Wrong column = major failure contributor',
                     fontsize=13, fontweight='bold', pad=15)
    
    cbar = plt.colorbar(im, ax=ax_main, fraction=0.046, pad=0.04)
    cbar.set_label('% of Column Max', fontsize=11, fontweight='bold')
    
    ax_bar = fig.add_subplot(gs[1])
    
    totals = matrix_data[:, 0]
    correct_pct = np.zeros(8)
    wrong_pct = np.zeros(8)
    
    for i in range(8):
        if totals[i] > 0:
            correct_pct[i] = (matrix_data[i, 1] / totals[i]) * 100
            wrong_pct[i] = (matrix_data[i, 2] / totals[i]) * 100
    
    x = np.arange(8)
    
    bars1 = ax_bar.bar(x, correct_pct, label='% Consensus Correct', 
                       color='limegreen', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax_bar.bar(x, wrong_pct, bottom=correct_pct, label='% Consensus Wrong', 
                       color='crimson', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax_bar.set_xlabel('Voting Pattern', fontsize=12, fontweight='bold')
    ax_bar.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax_bar.set_title('Percentage Breakdown by Pattern', fontsize=13, fontweight='bold')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=10, fontweight='bold')
    ax_bar.set_ylim(0, 100)
    ax_bar.legend(fontsize=11, loc='upper right')
    ax_bar.grid(axis='y', alpha=0.3)
    
    for i in range(8):
        if correct_pct[i] > 5:  
            ax_bar.text(i, correct_pct[i]/2, f'{correct_pct[i]:.0f}%', 
                       ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        if wrong_pct[i] > 5:
            ax_bar.text(i, correct_pct[i] + wrong_pct[i]/2, f'{wrong_pct[i]:.0f}%', 
                       ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    

    group_ranges = {
        'Unanimous Correct': (0, 1),
        'Single Dissenter': (1, 4),
        'Double Dissenter': (4, 7),
        'Unanimous Incorrect': (7, 8)
    }
    
    for group, (start, end) in group_ranges.items():
        ax_bar.axvspan(start - 0.5, end - 0.5, alpha=0.1, 
                      color=group_colors[group], zorder=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"       - Saved: {save_path}")

def plot_voting_pattern_conditioned_on_consensus_wrong(
    per_event_results: list,
    save_path: str
):
    """
    Which voting patterns produce your remaining errors after consensus?
    """
    wrong_events = [(fname, votes, confs, cc) for fname, votes, confs, cc 
                    in per_event_results if not cc]
    
    if len(wrong_events) == 0:
        print("       - No consensus-wrong events to plot!")
        return
    
    
    patterns = [
        ('CCC', [True, True, True]),
        ('CCF', [True, True, False]),
        ('CFC', [True, False, True]),
        ('FCC', [False, True, True]),
        ('CFF', [True, False, False]),
        ('FCF', [False, True, False]),
        ('FFC', [False, False, True]),
        ('FFF', [False, False, False]),
    ]
    
    
    pattern_counts = {label: 0 for label, _ in patterns}
    for fname, votes, confs, cc in wrong_events:
        for label, pattern in patterns:
            if votes == pattern:
                pattern_counts[label] += 1
                break
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=600)
    
   
    labels = [label for label, _ in patterns]
    counts = [pattern_counts[label] for label in labels]
    total_wrong = sum(counts)
    percentages = [(c / max(1, total_wrong)) * 100 for c in counts]
    
    colors = ['limegreen', 'gold', 'gold', 'gold', 'orange', 'orange', 'orange', 'crimson']
    bars = ax1.bar(labels, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    
    for bar, count, pct in zip(bars, counts, percentages):
        if count > 0:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{count}\n({pct:.1f}%)', ha='center', va='bottom', 
                    fontsize=9, fontweight='bold')
    
    ax1.set_xlabel('Voting Pattern (U, V, Z)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count (Consensus Wrong Only)', fontsize=12, fontweight='bold')
    ax1.set_title(f'THE MONEY PLOT: Failure Modes\n' +
                 f'(N={total_wrong} consensus-wrong events)', 
                 fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    

    threshold = 0.05  # 5%
    pie_data = []
    pie_labels = []
    pie_colors = []
    other_count = 0
    
    for label, count, color in zip(labels, counts, colors):
        if count / max(1, total_wrong) >= threshold:
            pie_data.append(count)
            pie_labels.append(f'{label}\n{count} events')
            pie_colors.append(color)
        else:
            other_count += count
    
    if other_count > 0:
        pie_data.append(other_count)
        pie_labels.append(f'Other\n{other_count} events')
        pie_colors.append('gray')
    
    ax2.pie(pie_data, labels=pie_labels, colors=pie_colors, autopct='%1.1f%%',
           startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax2.set_title('Dominant Failure Modes\n(Patterns >5% shown separately)', 
                 fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"       - Saved: {save_path}")

def plot_angular_error_by_agreement(
    results_dict: dict,
    per_event_results: list,
    save_path: str
):
    """
    Angular Error Distribution Conditioned on Agreement.
    
    Four categories:
    - Consensus Correct & Unanimous (3/3)
    - Consensus Correct & Majority (2/3)
    - Consensus Wrong & Unanimous (3/3)
    - Consensus Wrong & Majority (2/3)
    """
    # Build filename -> (U_error, V_error, Z_error) lookup
    view_results = results_dict['view_results']
    
    error_lookup = {}
    for v in ['U', 'V', 'Z']:
        for fname, is_correct, cos_theta, conf, ang_err in view_results[v]['per_event']:
            if fname not in error_lookup:
                error_lookup[fname] = {}
            error_lookup[fname][v] = ang_err
    
    # Categorize events
    categories = {
        'Consensus Correct & Unanimous (3/3)': [],
        'Consensus Correct & Majority (2/3)': [],
        'Consensus Wrong & Unanimous (3/3)': [],
        'Consensus Wrong & Majority (2/3)': [],
    }
    
    for fname, votes, confidences, consensus_correct in per_event_results:
        if fname not in error_lookup:
            continue
        
        
        errors = [error_lookup[fname].get(v, 0) for v in ['U', 'V', 'Z']]
        mean_error = np.mean(errors)
        
        
        n_votes = sum(votes)
        is_unanimous = (n_votes == 3 or n_votes == 0)
        
        
        if consensus_correct:
            if is_unanimous:
                categories['Consensus Correct & Unanimous (3/3)'].append(mean_error)
            else:
                categories['Consensus Correct & Majority (2/3)'].append(mean_error)
        else:
            if is_unanimous:
                categories['Consensus Wrong & Unanimous (3/3)'].append(mean_error)
            else:
                categories['Consensus Wrong & Majority (2/3)'].append(mean_error)
    
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=600)
    axes = axes.flatten()
    
    colors = ['limegreen', 'gold', 'crimson', 'orange']
    
    for i, (cat, errors) in enumerate(categories.items()):
        ax = axes[i]
        
        if len(errors) > 0:
            ax.hist(errors, bins=50, range=(0, 180), alpha=0.8, color=colors[i], edgecolor='black')
            ax.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.2f}°')
            ax.axvline(np.median(errors), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.2f}°')
            ax.legend(fontsize=9)
        
        ax.set_xlabel('Angular Error (degrees)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'{cat}\nN={len(errors)}', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_xlim(0, 180)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"       - Saved: {save_path}")


def evaluate_head_tail_disambiguation_single_view(
    view: str,
    device,
    cfg: Config,
    use_confidence: bool = False,
    batch_size: int = 32
) -> dict:
    """
    Evaluate binary head-tail disambiguation for a single view (batched version).
    """
    print(f"HEAD-TAIL DISAMBIGUATION: View {view}")
    
    if cfg.multi_view:
        model_dir = os.path.join(cfg.out_dir, "runs_multi")
        best_model_path = os.path.join(model_dir, 'best_checkpoint.pt')  
        checkpoint_key = view  # 'U', 'V', or 'Z'
    else:
        model_dir = os.path.join(cfg.out_dir, f"runs_{view}")
        best_model_path = os.path.join(model_dir, 'best_checkpoint.pt') 
        checkpoint_key = 'dense'
    
    if not os.path.exists(best_model_path):
        print(f"ERROR: Model not found at {best_model_path}")
        return None
    
    # Build model
    arch = cfg.dense_arch.lower()
    if arch == "convnextv2_unet":
        model = DenseModelConvNeXt(
            in_ch=1, feat_ch=cfg.dense_feat_ch, num_classes=cfg.num_classes,
            enable_seg=cfg.enable_seg, enable_dirreg=cfg.enable_dirreg,
            variant=cfg.cnv2_variant, decoder_scale=cfg.cnv2_decoder_scale,
            use_transpose=cfg.cnv2_use_transpose, skip_proj=cfg.cnv2_skip_proj,
        )
    elif arch == "smp_unet":
        model = DenseModelSMP(
            in_ch=cfg.smp_in_channels, feat_ch=cfg.dense_feat_ch, num_classes=cfg.num_classes,
            enable_seg=cfg.enable_seg, enable_dirreg=cfg.enable_dirreg,
            encoder_name=cfg.smp_encoder, encoder_weights=cfg.smp_encoder_weights,
            decoder_channels=cfg.smp_decoder_channels,
        )
    else:
        model = DenseModel(
            in_ch=1, base=cfg.dense_base, feat_ch=cfg.dense_feat_ch,
            num_classes=cfg.num_classes, enable_seg=cfg.enable_seg, enable_dirreg=cfg.enable_dirreg
        )
    
    
    ckpt = torch.load(best_model_path, map_location=device)
    
    if checkpoint_key not in ckpt:
        print(f"ERROR: Key '{checkpoint_key}' not found in checkpoint. Available keys: {list(ckpt.keys())}")
        return None
    
    model_state = ckpt[checkpoint_key]['model']
    
    # Handle potential 'module.' prefix from DDP 
    if list(model_state.keys())[0].startswith('module.'):
        print("  Detected 'module.' prefix, stripping...")
        model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
    
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    print(f"  Loaded model from epoch {ckpt.get('epoch', 'N/A')}")
    
    # Load test dataset with batching
    cfg_test = replace_cfg(cfg, batch_size=batch_size, multi_view=False, workers=0)
    test_ds, test_ld = make_dense_loader('test', view, cfg_test)
    print(f"  Test samples: {len(test_ds)}")
    print(f"  Batch size: {batch_size}")
    
    # Evaluate
    results = []
    n_correct = 0
    n_flipped = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_ld):
            if batch is None:
                continue
            
            # Forward pass
            img = batch["image"].to(device)              # [B,1,H,W]
            u_gt = batch["u_gt"].cpu().numpy()           # [B,2]
            fnames = batch["meta"]["fname"]              # List[str] of length B
            
            B = img.shape[0]
            
            out = model(img)
            logits_ab = out["logits_ab"]                 # [B,2,H,W]
            
            # Apply priors if enabled (match training behavior)
            logits_ab_adj, mx_sw, my_sw = _apply_priors_and_order_for_eval(out, cfg)
            
            # DSNT
            prob = spatial_softmax2d_logits(logits_ab_adj, temperature=cfg.dsnt_temp)  # [B,2,H,W]
            mx_t, my_t = dsnt_expectation(prob)          # Each [B,2]
            
            # Predicted direction (batch)
            ux, uy, _ = unit_from_points(mx_t[:,0], my_t[:,0], mx_t[:,1], my_t[:,1])
            u_pred = torch.stack([ux, uy], 1).detach().cpu().numpy()  # [B,2]
            
            # Process each sample in batch
            for i in range(B):
                fname = fnames[i]
                u_p = u_pred[i]  # [2]
                u_g = u_gt[i]    # [2]
                
                # Binary classification
                cos_theta = float(np.clip(u_p[0]*u_g[0] + u_p[1]*u_g[1], -1.0, 1.0))
                is_correct = (cos_theta > 0.0)  # Simple hemisphere
                angular_error_deg = float(np.degrees(np.arccos(np.abs(cos_theta))))
                
                # Confidence (per-sample computation)
                if use_confidence:
                    prob_np = prob[i].detach().cpu().numpy()  # [2,H,W]
                    mx_np = mx_t[i].detach().cpu().numpy()    # [2]
                    my_np = my_t[i].detach().cpu().numpy()    # [2]
                    conf_A, conf_B = compute_heatmap_confidence(prob_np, mx_np, my_np)
                    confidence = min(conf_A, conf_B)  # Overall confidence
                else:
                    confidence = abs(cos_theta)  # Fallback: use cos_theta as proxy
                
                # Record
                results.append((fname, is_correct, cos_theta, confidence, angular_error_deg))
                
                if is_correct:
                    n_correct += 1
                else:
                    n_flipped += 1
            
            # Progress
            if (batch_idx + 1) % 10 == 0:
                print(f"    Processed {(batch_idx + 1) * batch_size} / {len(test_ds)} samples...", end='\r')
    
    print()  # progress line
    
    total = n_correct + n_flipped
    accuracy = n_correct / max(1, total)
    
    print(f"\n  Results:")
    print(f"    Correct:  {n_correct}/{total} ({100*accuracy:.2f}%)")
    print(f"    Flipped:  {n_flipped}/{total} ({100*(1-accuracy):.2f}%)")
    
    return {
        'view': view,
        'total': total,
        'correct': n_correct,
        'flipped': n_flipped,
        'accuracy': accuracy,
        'per_event': results
    }


def evaluate_multiview_head_tail_consensus(
    cfg: Config,
    device,
    consensus_method: str = 'majority'
) -> dict:
    """
    Evaluate head-tail disambiguation using multi-view consensus.
    
    Returns:
        {
            'total': int,
            'consensus_correct': int,
            'consensus_flipped': int,
            'consensus_accuracy': float,
            'per_view_accuracy': {'U': float, 'V': float, 'Z': float},
            'agreement_stats': {...},
            'per_event': [(fname, votes, confidences, consensus), ...]
        }
    """
    print(f"MULTI-VIEW HEAD-TAIL CONSENSUS ({consensus_method})")
    
    # Evaluate each view
    use_conf = (consensus_method == 'confidence_weighted')
    view_results = {}
    for v in ['U', 'V', 'Z']:
        res = evaluate_head_tail_disambiguation_single_view(v, device, cfg, use_confidence=use_conf)
        if res is None:
            print(f"ERROR: Failed to evaluate view {v}")
            return None
        view_results[v] = res
    
    # Align events by filename
    # Build dictionaries: fname -> (is_correct, confidence)
    def _build_lookup(results):
        return {fname: (is_correct, conf) for fname, is_correct, _, conf, _ in results['per_event']}
    
    U_lookup = _build_lookup(view_results['U'])
    V_lookup = _build_lookup(view_results['V'])
    Z_lookup = _build_lookup(view_results['Z'])
    
    # Find common events
    common_fnames = set(U_lookup.keys()) & set(V_lookup.keys()) & set(Z_lookup.keys())
    print(f"\n  Common events across U/V/Z: {len(common_fnames)}")
    
    # Consensus voting
    consensus_results = []
    n_consensus_correct = 0
    n_consensus_flipped = 0
    
    # Agreement tracking
    n_unanimous = 0  # All 3 agree
    n_majority = 0   # 2 of 3 agree
    n_split = 0      # Complete disagreement or tie
    
    for fname in sorted(common_fnames):
        U_correct, U_conf = U_lookup[fname]
        V_correct, V_conf = V_lookup[fname]
        Z_correct, Z_conf = Z_lookup[fname]
        
        votes = [U_correct, V_correct, Z_correct]
        confidences = [U_conf, V_conf, Z_conf]
        
        if consensus_method == 'majority':
            # Simple majority vote
            n_votes_correct = sum(votes)
            consensus_correct = (n_votes_correct >= 2)
        
        elif consensus_method == 'confidence_weighted':
            # Weighted by confidence
            correct_weight = sum(conf for vote, conf in zip(votes, confidences) if vote)
            flipped_weight = sum(conf for vote, conf in zip(votes, confidences) if not vote)
            consensus_correct = (correct_weight > flipped_weight)
        
        else:
            raise ValueError(f"Unknown consensus method: {consensus_method}")
        
        # Track agreement
        n_votes_correct = sum(votes)
        if n_votes_correct == 3 or n_votes_correct == 0:
            n_unanimous += 1
        elif n_votes_correct == 2 or n_votes_correct == 1:
            n_majority += 1
        
        consensus_results.append((fname, votes, confidences, consensus_correct))
        
        if consensus_correct:
            n_consensus_correct += 1
        else:
            n_consensus_flipped += 1
    
    total = len(common_fnames)
    consensus_accuracy = n_consensus_correct / max(1, total)
    
    print(f"\n  Consensus Results:")
    print(f"    Correct:  {n_consensus_correct}/{total} ({100*consensus_accuracy:.2f}%)")
    print(f"    Flipped:  {n_consensus_flipped}/{total} ({100*(1-consensus_accuracy):.2f}%)")
    print(f"\n  Agreement:")
    print(f"    Unanimous (3/3): {n_unanimous} ({100*n_unanimous/max(1,total):.1f}%)")
    print(f"    Majority (2/3):  {n_majority} ({100*n_majority/max(1,total):.1f}%)")
    
    return {
        'total': total,
        'consensus_correct': n_consensus_correct,
        'consensus_flipped': n_consensus_flipped,
        'consensus_accuracy': consensus_accuracy,
        'per_view_accuracy': {
            'U': view_results['U']['accuracy'],
            'V': view_results['V']['accuracy'],
            'Z': view_results['Z']['accuracy'],
        },
        'agreement_stats': {
            'unanimous': n_unanimous,
            'majority': n_majority,
            'unanimous_frac': n_unanimous / max(1, total),
            'majority_frac': n_majority / max(1, total),
        },
        'per_event': consensus_results,
        'view_results': view_results,
    }


def save_head_tail_results(results: dict, save_dir: str, prefix: str = ""):
    """
    Save head-tail disambiguation results to CSV, JSON, and plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if 'per_event' in results and 'view' in results:
        view = results['view']
        
        # CSV
        csv_path = os.path.join(save_dir, f"{prefix}head_tail_{view}.csv")
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['filename', 'is_correct', 'cos_theta', 'confidence', 'angular_error_deg'])
            for fname, is_correct, cos_theta, conf, ang in results['per_event']:
                w.writerow([fname, int(is_correct), f"{cos_theta:.6f}", f"{conf:.6f}", f"{ang:.6f}"])
        print(f"  Saved: {csv_path}")
        
        # Bar chart
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=150)
        ax.bar(['Correct', 'Flipped'], [results['correct'], results['flipped']], 
               color=['limegreen', 'crimson'], alpha=0.8)
        ax.set_ylabel('Count')
        ax.set_title(f"{view}: Head-Tail Disambiguation\nAccuracy: {100*results['accuracy']:.2f}%")
        ax.grid(axis='y', alpha=0.3)
        for i, (label, count) in enumerate([('Correct', results['correct']), 
                                            ('Flipped', results['flipped'])]):
            ax.text(i, count + 10, str(count), ha='center', fontweight='bold')
        plt.tight_layout()
        bar_path = os.path.join(save_dir, f"{prefix}head_tail_{view}_bar.png")
        plt.savefig(bar_path, dpi=600)
        plt.close()
        print(f"  Saved: {bar_path}")
        
        # Cosine histogram (Figure 7 style)
        cos_values = [cos_theta for _, _, cos_theta, _, _ in results['per_event']]
        fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=150)
        ax.hist(cos_values, bins=50, range=(-1, 1), alpha=0.8, color='steelblue')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Decision boundary')
        ax.set_xlabel(r'$\cos(\theta)$ between predicted and true direction')
        ax.set_ylabel('Count')
        ax.set_title(f"{view}: Distribution of cos(θ) — Figure 7 Style\n"
                     f"Correct (>0): {results['correct']}  |  Flipped (<0): {results['flipped']}")
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        cos_hist_path = os.path.join(save_dir, f"{prefix}head_tail_{view}_cos_hist.png")
        plt.savefig(cos_hist_path, dpi=600)
        plt.close()
        print(f"  Saved: {cos_hist_path}")
    
    elif 'consensus_accuracy' in results:
        # Multi-view consensus results
        
        # CSV
        csv_path = os.path.join(save_dir, f"{prefix}head_tail_consensus.csv")
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['filename', 'U_correct', 'V_correct', 'Z_correct', 
                       'U_conf', 'V_conf', 'Z_conf', 'consensus_correct'])
            for fname, votes, confs, consensus in results['per_event']:
                w.writerow([fname, int(votes[0]), int(votes[1]), int(votes[2]),
                           f"{confs[0]:.6f}", f"{confs[1]:.6f}", f"{confs[2]:.6f}",
                           int(consensus)])
        print(f"  Saved: {csv_path}")
        
        # Summary JSON
        summary = {
            'total_events': results['total'],
            'consensus_accuracy': results['consensus_accuracy'],
            'per_view_accuracy': results['per_view_accuracy'],
            'agreement_stats': results['agreement_stats'],
        }
        json_path = os.path.join(save_dir, f"{prefix}head_tail_summary.json")
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved: {json_path}")
        
        # Comparison bar chart
        fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=150)
        views = ['U', 'V', 'Z', 'Consensus']
        accuracies = [
            results['per_view_accuracy']['U'],
            results['per_view_accuracy']['V'],
            results['per_view_accuracy']['Z'],
            results['consensus_accuracy']
        ]
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
        bars = ax.bar(views, [100*a for a in accuracies], color=colors, alpha=0.8)
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Head-Tail Disambiguation Accuracy: Per-View vs Consensus')
        ax.set_ylim([0, 105])
        ax.grid(axis='y', alpha=0.3)
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{100*acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()
        comp_path = os.path.join(save_dir, f"{prefix}head_tail_comparison.png")
        plt.savefig(comp_path, dpi=600)
        plt.close()
        print(f"  Saved: {comp_path}")
        
        # Agreement pie chart
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=150)
        sizes = [results['agreement_stats']['unanimous'], 
                results['agreement_stats']['majority']]
        labels = [f"Unanimous (3/3)\n{sizes[0]} events", 
                 f"Majority (2/3)\n{sizes[1]} events"]
        colors_pie = ['limegreen', 'gold']
        ax.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', 
              startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax.set_title('Multi-View Agreement Distribution')
        plt.tight_layout()
        pie_path = os.path.join(save_dir, f"{prefix}head_tail_agreement_pie.png")
        plt.savefig(pie_path, dpi=600)
        plt.close()
        print(f"  Saved: {pie_path}")

        plot_voting_pattern_distribution(
            results['per_event'],
            os.path.join(save_dir, f"{prefix}head_tail_voting_patterns_all.png"),
            filter_consensus_wrong=False
        )
        
        plot_voting_pattern_distribution(
            results['per_event'],
            os.path.join(save_dir, f"{prefix}head_tail_voting_patterns_wrong_only.png"),
            filter_consensus_wrong=True
        )
        
        plot_voting_pattern_conditioned_on_consensus_wrong(
            results['per_event'],
            os.path.join(save_dir, f"{prefix}head_tail_money_plot.png")
        )

        plot_voting_pattern_confusion_matrix_combined(
            results['per_event'],
            os.path.join(save_dir, f"{prefix}head_tail_confusion_matrix.png")
        )

        plot_voting_pattern_confusion_matrix_row_normalized(
            results['per_event'],
            os.path.join(save_dir, f"{prefix}head_tail_CM_row_normalized.png")
        )

        plot_voting_pattern_confusion_matrix_col_normalized(
            results['per_event'],
            os.path.join(save_dir, f"{prefix}head_tail_CM_col_normalized.png")
        )
        
        if 'view_results' in results:
            plot_angular_error_by_agreement(
                results,
                results['per_event'],
                os.path.join(save_dir, f"{prefix}head_tail_angular_by_agreement.png")
            )
        


def run_head_tail_analysis(cfg: Config, device):
    """
    Main entry point for head-tail disambiguation analysis.
    """
    print("HEAD-TAIL DISAMBIGUATION ANALYSIS")
    
    if cfg.multi_view:
        # Multi-view consensus
        print("\nRunning multi-view consensus analysis...")
        
        # Test both methods
        for method in ['majority', 'confidence_weighted']:
            print(f"Method: {method}")
            
            results = evaluate_multiview_head_tail_consensus(cfg, device, consensus_method=method)
            
            if results is not None:
                save_dir = os.path.join(cfg.out_dir, "runs_multi", "test_plots")
                prefix = f"{method}_"
                save_head_tail_results(results, save_dir, prefix=prefix)
    
    else:
        # Single-view analysis
        print("\nRunning single-view analysis...")
        
        for view in cfg.views:
            results = evaluate_head_tail_disambiguation_single_view(view, device, cfg, use_confidence=False)
            
            if results is not None:
                save_dir = os.path.join(cfg.out_dir, f"runs_{view}", "test_plots")
                save_head_tail_results(results, save_dir)
    
    print("HEAD-TAIL ANALYSIS COMPLETE")


