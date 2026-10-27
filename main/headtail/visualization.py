# Authored by Hilary Utaegbulam

from __future__ import annotations
import csv
import os
import random
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import torch
import torch.nn as nn

try:
    from scipy import ndimage as _ndi
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

from .constants import Config, CLASS_BG, CLASS_MICHEL, CLASS_BREMS
from .physics_utils import spatial_softmax2d_logits, dsnt_expectation, unit_from_points
from .losses import prior_bias_from_seg_dense

def _topk_mask(arr: np.ndarray, top_percent: float, roi: Optional[Tuple[int,int,int,int]] = None) -> np.ndarray:
    a = arr
    if roi is not None:
        y0,y1,x0,x1 = roi
        sub = a[y0:y1, x0:x1]
        if sub.size == 0:
            return np.zeros_like(a, dtype=bool)
        thr = np.percentile(sub, 100.0 - float(top_percent))
        m = np.zeros_like(a, dtype=bool)
        m[y0:y1, x0:x1] = sub >= thr
        return m
    else:
        thr = np.percentile(a, 100.0 - float(top_percent))
        return a >= thr

def _zscore_at(adc: np.ndarray, y: float, x: float, patch: int = 7) -> float:
    H, W = adc.shape
    yy = int(round(float(y))); xx = int(round(float(x)))
    hh = max(1, int(patch) // 2)
    y0, y1 = max(0, yy - hh), min(H, yy + hh + 1)
    x0, x1 = max(0, xx - hh), min(W, xx + hh + 1)
    win = adc[y0:y1, x0:x1]
    if win.size == 0:
        return float('nan')
    mu, sd = float(win.mean()), float(win.std() + 1e-6)
    return float((adc[min(max(0, yy), H-1), min(max(0, xx), W-1)] - mu) / sd)

def _compute_bragg_numbers(prob_2ch: np.ndarray,
                           adc: np.ndarray,
                           mx: np.ndarray, my: np.ndarray,
                           top_percent: float,
                           zpatch: int,
                           roi: Optional[Tuple[int,int,int,int]] = None) -> Tuple[Tuple[float,float], Tuple[float,float]]:
    H, W = adc.shape
    assert prob_2ch.shape == (2, H, W), "prob_2ch must be (2,H,W)"
    A = prob_2ch[0]; B = prob_2ch[1]

    R10_A = _topk_mask(A, top_percent, roi=roi)
    R10_B = _topk_mask(B, top_percent, roi=roi)
    A10   = _topk_mask(adc, top_percent, roi=roi)

    num_A = R10_A.sum()
    num_B = R10_B.sum()
    overlap_A = float((R10_A & A10).sum() / max(1, num_A))
    overlap_B = float((R10_B & A10).sum() / max(1, num_B))

    zA = _zscore_at(adc, float(my[0]), float(mx[0]), patch=zpatch)
    zB = _zscore_at(adc, float(my[1]), float(mx[1]), patch=zpatch)
    return (overlap_A, overlap_B), (zA, zB)

def save_consistency_analysis_plots(
    metrics_over_time: list,
    save_dir: str,
    prefix: str = ""
):
    
    os.makedirs(save_dir, exist_ok=True)
    
    if len(metrics_over_time) == 0:
        print("  WARNING: No consistency metrics to plot")
        return
    
    epochs = list(range(len(metrics_over_time)))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    
    ax = axes[0, 0]
    acc_before = [m['aggregate']['acc_before'] for m in metrics_over_time]
    acc_after = [m['aggregate']['acc_after'] for m in metrics_over_time]
    
    ax.plot(epochs, [100*a for a in acc_before], 'r--', label='Before A→B', linewidth=2)
    ax.plot(epochs, [100*a for a in acc_after], 'g-', label='After A→B', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Overall Accuracy: A→B Impact', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    ax = axes[0, 1]
    for v in ['U', 'V', 'Z']:
        gains = [100*m['per_view'][v]['acc_gain'] for m in metrics_over_time]
        ax.plot(epochs, gains, label=f'{v} gain', linewidth=2, marker='o', markersize=4)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy Gain (%)', fontsize=12)
    ax.set_title('Per-View Accuracy Gain from A→B', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    flip_prec = [100*m['aggregate']['flip_precision'] for m in metrics_over_time]
    ax.plot(epochs, flip_prec, 'b-', linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Flip Precision (%)', fontsize=12)
    ax.set_title('A→B Flip Decision Quality\n(When we flip, are we right?)', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    ax = axes[1, 1]
    for v in ['U', 'V', 'Z']:
        flips = [m['per_view'][v]['n_flipped_total'] for m in metrics_over_time]
        ax.plot(epochs, flips, label=f'{v} flips', linewidth=2, marker='^', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Number of Flips', fontsize=12)
    ax.set_title('Flip Frequency by View\n(Should decrease as model improves)', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{prefix}consistency_analysis.png')
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {save_path}")

def save_bragg_figure_dense(
    cfg,
    view: str,
    sample: dict,
    u_pred_1x2: np.ndarray,
    prob_2ch: np.ndarray,
    mx_2: np.ndarray,
    my_2: np.ndarray,
    out_path: str,
    angular_error_deg: float = None,
):
    if not cfg.bragg_make:
        return

    adc  = sample["image"].squeeze(0).numpy()
    u_gt = sample["u_gt"].numpy()
    H, W = adc.shape
    cx, cy = W/2.0, H/2.0
    L = min(H, W) * 0.2

    A = prob_2ch[0]
    B = prob_2ch[1]

    def _levels(arr, n= int(cfg.bragg_contour_levels)):
        lo = float(np.percentile(arr, 50.0))
        hi = float(np.percentile(arr, 99.5))
        hi = max(hi, 1e-12)
        lo = min(lo, hi * 0.7)
        return np.linspace(lo, hi, max(2, n))

    levA = _levels(A)
    levB = _levels(B)

    (ovA, ovB), (zA, zB) = _compute_bragg_numbers(
        prob_2ch=prob_2ch, adc=adc, mx=mx_2, my=my_2,
        top_percent=cfg.bragg_topk_percent, zpatch=cfg.bragg_zscore_patch,
        roi=cfg.bragg_roi
    )

    fig = plt.figure(figsize=(6.5, 7.0), dpi=int(cfg.bragg_dpi))
    gs = GridSpec(2, 2, height_ratios=[2.2, 1.8], figure=fig)

    ax_top = fig.add_subplot(gs[0, :])
    ax_top.imshow(adc, origin='lower', cmap='gray', aspect='auto')

    ax_top.contour(A, levels=levA, alpha=float(cfg.bragg_contour_alpha), cmap='Blues', linewidths=0.8)
    ax_top.contour(B, levels=levB, alpha=float(cfg.bragg_contour_alpha), cmap='Oranges', linewidths=0.8)

    ax_top.arrow(cx, cy, L*u_gt[0],       L*u_gt[1],       color='lime',    width=0.0, head_width=2.5, length_includes_head=True)
    ax_top.arrow(cx, cy, L*u_pred_1x2[0], L*u_pred_1x2[1], color='magenta', width=0.0, head_width=2.5, length_includes_head=True)

    ax_top.text(0.01, 0.98, f"A {ovA:.2f}/{zA:.1f}   B {ovB:.2f}/{zB:.1f}",
                transform=ax_top.transAxes, va='top', ha='left',
                fontsize=7, color='w', bbox=dict(facecolor='k', alpha=0.3, pad=2, edgecolor='none'))
    if cfg.plot_show_error_text and (angular_error_deg is not None):
        ax_top.text(0.99, 0.02, f"Err: {angular_error_deg:.1f}°",
                    transform=ax_top.transAxes, va='bottom', ha='right',
                    fontsize=8, color='w',
                    bbox=dict(facecolor='k', alpha=0.35, pad=2, edgecolor='none'))

    ax_top.set_xticks([]); ax_top.set_yticks([])
    ax_top.set_title(f"{view} — {sample['meta']['fname']}")

    ax_A = fig.add_subplot(gs[1, 0])
    ax_A.imshow(adc, origin='lower', cmap='gray', aspect='auto')
    ax_A.contour(A, levels=levA, alpha=0.9, cmap='Blues', linewidths=0.9)
    ax_A.plot([mx_2[0]], [my_2[0]], marker='+', markersize=6, color='tab:blue', mew=1.2)
    ax_A.set_xticks([]); ax_A.set_yticks([])

    ax_B = fig.add_subplot(gs[1, 1])
    ax_B.imshow(adc, origin='lower', cmap='gray', aspect='auto')
    ax_B.contour(B, levels=levB, alpha=0.9, cmap='Oranges', linewidths=0.9)
    ax_B.plot([mx_2[1]], [my_2[1]], marker='+', markersize=6, color='tab:orange', mew=1.2)
    ax_B.set_xticks([]); ax_B.set_yticks([])

    plt.tight_layout()
    plt.savefig(out_path, dpi=int(cfg.bragg_dpi))
    plt.close(fig)

def save_block_heatmap_figure_dense(
    cfg,
    view: str,
    sample: dict,
    prob_2ch: np.ndarray,
    mx_2: np.ndarray,
    my_2: np.ndarray,
    out_path: str,
    *,
    angular_error_deg: float = None,
    bg_dim=0.35,
    boost=1.6,
    max_alpha=0.85,
):

    adc = sample["image"].squeeze(0).numpy()
    H, W = adc.shape
    A = prob_2ch[0].copy()
    B = prob_2ch[1].copy()

    A = np.power(np.clip(A / (A.max() + 1e-12), 0, 1), 1.0 / boost)
    B = np.power(np.clip(B / (B.max() + 1e-12), 0, 1), 1.0 / boost)

    alphaA = np.clip(A, 0.0, max_alpha)
    alphaB = np.clip(B, 0.0, max_alpha)

    im_kwargs = dict(origin="lower", interpolation="nearest", aspect="auto")

    fig = plt.figure(figsize=(6.5, 7.0), dpi=int(cfg.bragg_dpi))
    gs  = GridSpec(2, 2, height_ratios=[2.2, 1.8], figure=fig)

    ax_top = fig.add_subplot(gs[0, :])
    ax_top.set_facecolor("black")
    ax_top.imshow(adc, cmap="gray", origin="lower", aspect="auto", alpha=bg_dim)
    ax_top.imshow(np.zeros_like(A), cmap="Reds", alpha=alphaA, **im_kwargs)
    ax_top.imshow(np.zeros_like(B), cmap="Blues", alpha=alphaB, **im_kwargs)
    ax_top.plot([mx_2[0]], [my_2[0]], marker="+", ms=7, mew=1.5, color="red")
    ax_top.plot([mx_2[1]], [my_2[1]], marker="+", ms=7, mew=1.5, color="blue")
    ax_top.set_xticks([]); ax_top.set_yticks([])
    ax_top.set_title(f"{view} — {sample['meta']['fname']}")

    if cfg.plot_show_error_text and (angular_error_deg is not None):
        ax_top.text(0.99, 0.02, f"Err: {angular_error_deg:.1f}°",
                    transform=ax_top.transAxes, va='bottom', ha='right',
                    fontsize=8, color='w',
                    bbox=dict(facecolor='k', alpha=0.35, pad=2, edgecolor='none'))

    ax_A = fig.add_subplot(gs[1, 0])
    ax_A.set_facecolor("black")
    ax_A.imshow(adc, cmap="gray", origin="lower", aspect="auto", alpha=bg_dim)
    ax_A.imshow(np.zeros_like(A), cmap="Reds", alpha=alphaA, **im_kwargs)
    ax_A.plot([mx_2[0]], [my_2[0]], marker="+", ms=7, mew=1.5, color="red")
    ax_A.set_xticks([]); ax_A.set_yticks([])

    ax_B = fig.add_subplot(gs[1, 1])
    ax_B.set_facecolor("black")
    ax_B.imshow(adc, cmap="gray", origin="lower", aspect="auto", alpha=bg_dim)
    ax_B.imshow(np.zeros_like(B), cmap="Blues", alpha=alphaB, **im_kwargs)
    ax_B.plot([mx_2[1]], [my_2[1]], marker="+", ms=7, mew=1.5, color="blue")
    ax_B.set_xticks([]); ax_B.set_yticks([])

    plt.tight_layout()
    plt.savefig(out_path, dpi=int(cfg.bragg_dpi))
    plt.close(fig)

def _apply_priors_and_order_for_eval(out, cfg):
    logits_ab = out["logits_ab"]
    seg_logits = out.get("seg_logits", None)

    logits_ab_adj = logits_ab
    if cfg.enable_seg and (seg_logits is not None) and (cfg.prior_mode in ("soft","hard")):
        prior = prior_bias_from_seg_dense(seg_logits)
        if cfg.prior_mode == "soft" and cfg.prior_alpha > 0:
            logits_ab_adj = logits_ab_adj + cfg.prior_alpha * (prior - 0.5)
        elif cfg.prior_mode == "hard":
            with torch.no_grad():
                B_, _, H, W = prior.shape
                keep_mask = torch.zeros_like(prior, dtype=torch.bool)
                q = float(cfg.prior_topk_percent)
                for b in range(B_):
                    vals = prior[b,0].flatten()
                    k = max(1, int(len(vals) * (q / 100.0)))
                    thr = torch.topk(vals, k).values.min()
                    keep_mask[b,0] = (prior[b,0] >= thr)
            m = keep_mask.expand(-1,2,-1,-1)
            logits_ab_adj = torch.where(m, logits_ab_adj, torch.finfo(logits_ab_adj.dtype).min)

    prob = spatial_softmax2d_logits(logits_ab_adj, temperature=cfg.dsnt_temp)
    mx, my = dsnt_expectation(prob)
    mx_sw, my_sw = mx, my

    return logits_ab_adj, mx_sw, my_sw

 
@torch.no_grad()

def _save_cosine_histograms(view: str, cos_list: np.ndarray, weights: np.ndarray, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    bins = np.linspace(-1.0, 1.0, 50)

    plt.figure(figsize=(6,4), dpi=600)
    plt.hist(cos_list, bins=bins, alpha=0.9)
    plt.xlabel(r"$\cos\theta$")
    plt.ylabel("Counts")
    plt.title(f"{view}: cos(angle)")
    plt.tight_layout()
    p1 = os.path.join(out_dir, f"{view}_cos_hist_unweighted.png")
    plt.savefig(p1, dpi=600); plt.close()

    plt.figure(figsize=(6,4), dpi=600)
    plt.hist(cos_list, bins=bins, weights=weights, alpha=0.9)
    plt.xlabel(r"$\cos\theta$")
    plt.ylabel("Pixel-weighted counts")
    plt.title(f"{view}: cos(angle) — pixel-weighted")
    plt.tight_layout()
    p2 = os.path.join(out_dir, f"{view}_cos_hist_pixelweighted.png")
    plt.savefig(p2, dpi=600); plt.close()

    print(f"       - Saved: {p1}")
    print(f"       - Saved: {p2}")

def _write_angle_csv_with_summary(csv_path: str,
                                  names: list,
                                  ang_deg: np.ndarray,
                                  cos_list: np.ndarray,
                                  weights: np.ndarray):
    N = len(ang_deg)
    frac_lt = lambda t: float((ang_deg < t).mean()) if N > 0 else 0.0
    frac_gt = lambda t: float((ang_deg > t).mean()) if N > 0 else 0.0

    anti_parallel_frac = frac_gt(90.0)
    near_flip_frac = frac_gt(150.0)

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fname", "ang_deg", "cos_theta", "pixel_weight",
                    "flag_gt90", "flag_gt150"])
        for nm, a, c, pw in zip(names, ang_deg, cos_list, weights):
            w.writerow([nm, f"{a:.6f}", f"{c:.6f}", f"{pw:.6f}",
                        int(a > 90.0), int(a > 150.0)])

        w.writerow([])
        w.writerow(["SUMMARY"])
        w.writerow(["N_events", N])
        if N > 0:
            w.writerow(["mean_ang_deg", float(np.mean(ang_deg))])
            w.writerow(["median_ang_deg", float(np.median(ang_deg))])
            w.writerow(["%<25deg", frac_lt(25.0)])
            w.writerow(["%<20deg", frac_lt(20.0)])
            w.writerow(["%<10deg", frac_lt(10.0)])
            w.writerow(["%<5deg",  frac_lt(5.0)])
            w.writerow(["%<90deg", 1.0 - anti_parallel_frac])
            w.writerow(["anti_parallel_fraction_(>90deg)", anti_parallel_frac])
            w.writerow(["near_flip_fraction_(>150deg)", near_flip_frac])

def save_segmentation_overlay_figure(
    cfg,
    view: str,
    sample: dict,
    seg_pred: np.ndarray,
    seg_gt: np.ndarray,
    out_path: str,
    dpi: int = 600,
    seg_logits: np.ndarray = None,
    show_probs: bool = True,
):

    def _to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    adc  = _to_np(sample["image"]).squeeze(0)
    pred = _to_np(seg_pred).astype(np.int32)
    gt   = _to_np(seg_gt).astype(np.int32)
    H, W = adc.shape

    COLS = {
        "gt_mi_fill":  (0.0, 0.85, 0.85, 0.22),
        "gt_br_fill":  (1.0, 0.75, 0.25, 0.22),
        "gt_mi_edge":  (0.0, 0.55, 0.55, 1.00),
        "gt_br_edge":  (0.80,0.40, 0.00, 1.00),
        "pr_mi_edge":  (0.00,0.90, 0.90, 1.00),
        "pr_br_edge":  (1.00,0.60, 0.00, 1.00),
    }

    def _bool_edges(mask_bool: np.ndarray) -> np.ndarray:
        if not mask_bool.any():
            return mask_bool
        if _HAVE_SCIPY:
            er = _ndi.binary_erosion(mask_bool, structure=np.ones((3,3), bool))
            return mask_bool & (~er)
        m = mask_bool.astype(np.uint8)
        up    = np.pad(m[1:,:],   ((0,1),(0,0)))
        down  = np.pad(m[:-1,:],  ((1,0),(0,0)))
        left  = np.pad(m[:,1:],   ((0,0),(0,1)))
        right = np.pad(m[:,:-1],  ((0,0),(1,0)))
        edge = (m != up) | (m != down) | (m != left) | (m != right)
        return edge.astype(bool)

    def _rgba_from_mask(mask_bool: np.ndarray, rgba) -> np.ndarray:
        out = np.zeros((H, W, 4), dtype=np.float32)
        if mask_bool.any():
            out[mask_bool] = rgba
        return out

    def _percentile_window(a, lo=70.0, hi=99.7):
        lo_v, hi_v = np.percentile(a, [lo, hi])
        if hi_v <= lo_v: hi_v = lo_v + 1e-6
        return float(lo_v), float(hi_v)

    gt_mi   = (gt   == CLASS_MICHEL)
    gt_br   = (gt   == CLASS_BREMS)
    pr_mi   = (pred == CLASS_MICHEL)
    pr_br   = (pred == CLASS_BREMS)

    gt_mi_edge = _bool_edges(gt_mi)
    gt_br_edge = _bool_edges(gt_br)
    pr_mi_edge = _bool_edges(pr_mi)
    pr_br_edge = _bool_edges(pr_br)

    probs = None
    if (seg_logits is not None) and show_probs:
        L = _to_np(seg_logits)
        L = L - L.max(axis=0, keepdims=True)
        ex = np.exp(L, dtype=np.float64)
        probs = ex / np.clip(ex.sum(axis=0, keepdims=True), 1e-12, None)
        has_mi = CLASS_MICHEL < probs.shape[0]
        has_br = CLASS_BREMS  < probs.shape[0]
    else:
        has_mi = has_br = False

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=dpi)

    ax = axes[0]
    ax.imshow(adc, origin="lower", cmap="gray", interpolation="nearest", aspect="auto")

    ax.imshow(_rgba_from_mask(gt_mi, COLS["gt_mi_fill"]), origin="lower", interpolation="nearest", aspect="auto")
    ax.imshow(_rgba_from_mask(gt_br, COLS["gt_br_fill"]), origin="lower", interpolation="nearest", aspect="auto")

    ax.imshow(_rgba_from_mask(gt_mi_edge, COLS["gt_mi_edge"]), origin="lower", interpolation="nearest", aspect="auto")
    ax.imshow(_rgba_from_mask(gt_br_edge, COLS["gt_br_edge"]), origin="lower", interpolation="nearest", aspect="auto")

    ax.set_title(f"{view} — Ground Truth", fontsize=16, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1]
    ax.imshow(adc, origin="lower", cmap="gray", interpolation="nearest", aspect="auto")

    if has_mi:
        lo, hi = _percentile_window(probs[CLASS_MICHEL], 70, 99.7)
        ax.imshow(probs[CLASS_MICHEL], origin="lower", cmap="Blues", alpha=0.35,
                  vmin=lo, vmax=hi, interpolation="nearest", aspect="auto")
    if has_br:
        lo, hi = _percentile_window(probs[CLASS_BREMS], 70, 99.7)
        ax.imshow(probs[CLASS_BREMS], origin="lower", cmap="Oranges", alpha=0.35,
                  vmin=lo, vmax=hi, interpolation="nearest", aspect="auto")

    ax.imshow(_rgba_from_mask(pr_mi_edge, COLS["pr_mi_edge"]), origin="lower", interpolation="nearest", aspect="auto")
    ax.imshow(_rgba_from_mask(pr_br_edge, COLS["pr_br_edge"]), origin="lower", interpolation="nearest", aspect="auto")

    ax.set_title(f"{view} — Prediction",
                 fontsize=16, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    legend_elements = [
        Patch(facecolor=COLS["gt_mi_edge"], label='Michel (Cyan)'),
        Patch(facecolor=COLS["gt_br_edge"], label='Brems (Orange)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, frameon=False, fontsize=14)
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

def save_thesis_vector_figure(
    cfg,
    view: str,
    sample: dict,
    u_pred: np.ndarray,
    u_gt: np.ndarray,
    prob_2ch: np.ndarray,
    mx_2: np.ndarray,
    my_2: np.ndarray,
    out_path: str,
    angular_error_deg: float,
    show_title: bool = False
):
    adc = sample["image"].squeeze(0).numpy()
    H, W = adc.shape
    cx, cy = W / 2.0, H / 2.0
    L = min(H, W) * 0.2
    
    x = np.arange(W)
    y = np.arange(H)
    X, Y = np.meshgrid(x, y)

    img_white_bg = np.ma.masked_where(adc < 1e-3, adc)
    cmap_white_bg = plt.get_cmap('viridis').copy()
    cmap_white_bg.set_bad('white')

    img_dark_bg = adc
    
    fig = plt.figure(figsize=(10, 6), dpi=600)
    gs = GridSpec(2, 2, width_ratios=[1.5, 1], height_ratios=[1, 1], wspace=0.1, hspace=0.1)
    
    ax_main = fig.add_subplot(gs[:, 0])
    ax_main.imshow(img_white_bg, origin='lower', cmap=cmap_white_bg, aspect='auto')
    
    ax_main.arrow(cx, cy, L*u_gt[0], L*u_gt[1], color='lime', width=0.0, 
                  head_width=2.5, length_includes_head=True, label="Ground Truth", zorder=5)
    ax_main.arrow(cx, cy, L*u_pred[0], L*u_pred[1], color='magenta', width=0.0, 
                  head_width=2.5, length_includes_head=True, label="Prediction", zorder=6)
    
    ax_main.legend(loc='upper left', fontsize=9, framealpha=0.9, facecolor='white', edgecolor='black')
    
    stats_text = f"Err: {angular_error_deg:.1f}°"
    ax_main.text(0.02, 0.02, stats_text, transform=ax_main.transAxes, 
                 color='black', fontsize=12, fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))
    
    ax_main.set_xlabel('Time Tick', fontsize=14)
    ax_main.set_ylabel('Wire Number', fontsize=14)
    ax_main.tick_params(axis='both', labelsize=10)
    if show_title:
        ax_main.set_title(f"{view} View", fontsize=14, fontweight='bold')

    contour_levels = np.r_[0.0, np.geomspace(1e-6, 1.0, 12)]

    ax_a = fig.add_subplot(gs[0, 1])
    ax_a.set_facecolor('black')
    ax_a.imshow(img_dark_bg, origin='lower', cmap='gray', aspect='auto', alpha=0.5) 
    
    prob_a = prob_2ch[0]
    prob_a_norm = prob_a / (prob_a.max() + 1e-9)
    

    ax_a.contourf(
    X, Y, prob_a_norm,
    levels=contour_levels,
    cmap='winter',
    alpha=0.85,
    extend='min'
    )
    ax_a.contour(
        X, Y, prob_a_norm,
        levels=contour_levels[2:],
        colors='cyan',
        linewidths=0.4,
        alpha=0.8
    )

    
    ax_a.scatter([mx_2[0]], [my_2[0]], c='cyan', marker='x', s=60, linewidth=2)
    ax_a.tick_params(axis='both', labelsize=8)
    ax_a.tick_params(axis='x', labelbottom=False)
    
    ax_a.text(0.02, 0.9, "Tail Probability", transform=ax_a.transAxes, 
              color='black', fontsize=10, fontweight='bold', 
              bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1.5))

    ax_b = fig.add_subplot(gs[1, 1])
    ax_b.set_facecolor('black')
    ax_b.imshow(img_dark_bg, origin='lower', cmap='gray', aspect='auto', alpha=0.5)
    
    prob_b = prob_2ch[1]
    prob_b_norm = prob_b / (prob_b.max() + 1e-9)
    

    ax_b.contourf(
        X, Y, prob_b_norm,
        levels=contour_levels,
        cmap='autumn_r',
        alpha=0.85,
        extend='min'
    )
    ax_b.contour(
        X, Y, prob_b_norm,
        levels=contour_levels[2:],
        colors='red',
        linewidths=0.4,
        alpha=0.8
    )

    
    ax_b.scatter([mx_2[1]], [my_2[1]], c='red', marker='x', s=60, linewidth=2)
    ax_b.set_xlabel('Time Tick', fontsize=14)
    ax_b.tick_params(axis='both', labelsize=8)
    
    ax_b.text(0.02, 0.9, "Head Probability", transform=ax_b.transAxes, 
              color='black', fontsize=10, fontweight='bold', 
              bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1.5))

    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close(fig)

def save_thesis_segmentation_figure(
    cfg,
    view: str,
    sample: dict,
    seg_pred: np.ndarray,
    seg_gt: np.ndarray,
    out_path: str,
    show_title: bool = False
):
    adc = sample["image"].squeeze(0).numpy()
    H, W = seg_gt.shape
    
    img_white_bg = np.ma.masked_where(adc < 1e-3, adc)
    cmap_white_bg = plt.get_cmap('viridis').copy()
    cmap_white_bg.set_bad('white')

    gt_overlay = np.zeros((H, W, 4))
    mask_michel = (seg_gt == 3) 
    mask_brems  = (seg_gt == 4)
    
    c_michel = [0, 1, 1, 0.6]
    c_brems  = [1, 0.5, 0, 0.6]
    
    gt_overlay[mask_michel] = c_michel
    gt_overlay[mask_brems]  = c_brems

    error_map = np.zeros((H, W, 4))
    gt_signal = (seg_gt == 3) | (seg_gt == 4)
    pred_signal = (seg_pred == 3) | (seg_pred == 4)
    
    tp_mask = (gt_signal) & (seg_gt == seg_pred)
    fp_mask = (pred_signal) & (~gt_signal)
    fn_mask = (gt_signal) & (~pred_signal)
    conf_mask = (gt_signal) & (pred_signal) & (seg_gt != seg_pred)

    c_tp   = [0, 0.8, 0, 0.7]
    c_fp   = [1, 0, 0, 0.7]
    c_fn   = [0, 0, 1, 0.7]
    c_conf = [1, 0.8, 0, 0.7]

    error_map[tp_mask] = c_tp
    error_map[fp_mask] = c_fp
    error_map[fn_mask] = c_fn
    error_map[conf_mask] = c_conf

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=600)
    
    axes[0].imshow(img_white_bg, origin='lower', cmap=cmap_white_bg, aspect='auto')
    axes[0].imshow(gt_overlay, origin='lower', aspect='auto')
    axes[0].set_xticks([]); axes[0].set_yticks([])
    if show_title:
        axes[0].set_title(f"{view}: Ground Truth", fontsize=14, fontweight='bold')
    
    legend_elements_gt = [
        Patch(facecolor=c_michel, edgecolor='none', label='Michel'),
        Patch(facecolor=c_brems, edgecolor='none', label='Brems'),
    ]
    axes[0].legend(handles=legend_elements_gt, loc='upper left', framealpha=0.9, facecolor='white', edgecolor='black')

    axes[1].imshow(img_white_bg, origin='lower', cmap=cmap_white_bg, aspect='auto')
    axes[1].imshow(error_map, origin='lower', aspect='auto')
    axes[1].set_xticks([]); axes[1].set_yticks([])
    if show_title:
        axes[1].set_title(f"{view}: Error Analysis", fontsize=14, fontweight='bold')

    legend_elements_err = [
        Patch(facecolor=c_tp, edgecolor='none', label='Correct (TP)'),
        Patch(facecolor=c_fp, edgecolor='none', label='Ghost (FP)'),
        Patch(facecolor=c_fn, edgecolor='none', label='Missed (FN)'),
        Patch(facecolor=c_conf, edgecolor='none', label='Class Swap'),
    ]
    axes[1].legend(handles=legend_elements_err, loc='upper left', framealpha=0.9, facecolor='white', edgecolor='black')

    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close(fig)

def save_validation_plots(model, val_ds, device, cfg, epoch, save_dir, k=5):
    
    epoch_dir = os.path.join(save_dir, f"epoch_{epoch:04d}")
    os.makedirs(epoch_dir, exist_ok=True)
    
    indices = random.sample(range(len(val_ds)), min(k, len(val_ds)))
    
    model.eval()
    with torch.no_grad():
        for idx in indices:
            sample = val_ds[idx]
            if sample is None:
                continue
            
            try:
                viz_sample_dense(cfg, epoch_dir, sample['meta']['view'], model, sample, device)
            except Exception as e:
                print(f"    WARNING: Failed to visualize sample {idx}: {e}")
    
    print(f"    -> Saved {len(indices)} validation plots to {epoch_dir}")

def viz_sample_dense(cfg: Config, out_dir: str, view: str, model: nn.Module, sample, device):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        img = sample["image"].unsqueeze(0).to(device)
        out = model(img)
        logits_ab = out["logits_ab"]
        prob = spatial_softmax2d_logits(logits_ab, temperature=cfg.dsnt_temp)
        mx_t, my_t = dsnt_expectation(prob)
        ux, uy, _ = unit_from_points(mx_t[:,0], my_t[:,0], mx_t[:,1], my_t[:,1])
        u_pred = torch.stack([ux, uy], 1)[0].detach().cpu().numpy()
        u_gt   = sample["u_gt"].numpy()

        cos_theta = float(np.clip(u_pred[0]*u_gt[0] + u_pred[1]*u_gt[1], -1.0, 1.0))
        angular_error_deg = float(np.degrees(np.arccos(cos_theta)))

        SHOW_COLORBAR = True

        img_np = sample["image"].squeeze(0).numpy().copy()

        if "original_size" in sample:
            H_orig, W_orig = sample["original_size"]
            img_np[H_orig:, :] = 0
            img_np[:, W_orig:] = 0
        H, W = img_np.shape
        cx, cy = W/2.0, H/2.0
        L = min(H, W) * 0.2

        f1 = None
        f1_orig = None
        f2 = None
        f3 = None
        f_seg = None

        if not cfg.skip_per_event_plots:
            fig, ax = plt.subplots(1,1,figsize=(6,4),dpi=600)
            
            img_masked = np.ma.masked_where(img_np == 0, img_np)
            cmap = plt.get_cmap('viridis').copy()
            cmap.set_bad('white')
            
            im = ax.imshow(img_masked, origin='lower', cmap=cmap, aspect='auto')
            
            if SHOW_COLORBAR:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="4.5%", pad=0.08)
                cbar = plt.colorbar(im, cax=cax)
                cbar.set_label('ADC', rotation=270, labelpad=15)
            
            ax.arrow(cx, cy, L*u_gt[0],   L*u_gt[1],   color='lime',    width=0.0, head_width=2.5, length_includes_head=True, label="GT")
            ax.arrow(cx, cy, L*u_pred[0], L*u_pred[1], color='magenta', width=0.0, head_width=2.5, length_includes_head=True, label="Pred")
            ax.legend(loc='upper right')
            ax.set_title(f"{view} Plane Readout (Input to Model)")

            ax.set_xlabel('Time Tick', fontsize=14)
            ax.set_ylabel('Wire Number', fontsize=14)

            if cfg.plot_show_error_text:
                ax.text(0.02, 0.98, f"Err: {angular_error_deg:.1f}°",
                        transform=ax.transAxes, va='top', ha='left',
                        fontsize=14, color='w',
                        bbox=dict(facecolor='k', alpha=0.35, pad=2, edgecolor='none'))

            f1 = os.path.join(out_dir, f"{sample['meta']['fname'].split('.')[0]}__{view}_vec.png")
            plt.tight_layout(); plt.savefig(f1, dpi=600); plt.close(fig)

            if "image_pre_masks" in sample:
                img_np_orig = sample["image_pre_masks"].squeeze(0).numpy().copy()
                if "original_size" in sample:
                    H_orig, W_orig = sample["original_size"]
                    img_np_orig[H_orig:, :] = 0
                    img_np_orig[:, W_orig:] = 0

                H2, W2 = img_np_orig.shape
                cx2, cy2 = W2/2.0, H2/2.0
                L2 = min(H2, W2) * 0.2

                fig2, ax2 = plt.subplots(1,1,figsize=(6,4),dpi=600)
                
                if "seg_mask" in sample:
                    seg_gt = sample["seg_mask"].detach().cpu().numpy()
                    is_background = (seg_gt == CLASS_BG)
                    is_padded = (img_np_orig == 0)
                    is_masked = is_background | is_padded
                    img_orig_masked = np.ma.masked_where(is_masked, img_np_orig)
                else:
                    img_orig_masked = img_np_orig
                
                cmap_orig = plt.get_cmap('viridis').copy()
                cmap_orig.set_bad('white')
                
                im2 = ax2.imshow(img_orig_masked, origin='lower', cmap=cmap_orig, aspect='auto')
                
                if SHOW_COLORBAR:
                    divider2 = make_axes_locatable(ax2)
                    cax2 = divider2.append_axes("right", size="4.5%", pad=0.08)
                    cbar2 = plt.colorbar(im2, cax=cax2)
                    cbar2.set_label('ADC', rotation=270, labelpad=15)
                
                ax2.arrow(cx2, cy2, L2*u_gt[0],   L2*u_gt[1],   color='lime',    width=0.0, head_width=2.5, length_includes_head=True, label="GT")
                ax2.arrow(cx2, cy2, L2*u_pred[0], L2*u_pred[1], color='magenta', width=0.0, head_width=2.5, length_includes_head=True, label="Pred")
                ax2.legend(loc='upper right') 
                ax2.set_title(f"{view} Plane Readout (Original Event)")

                ax2.set_xlabel('Time Tick', fontsize=14)
                ax2.set_ylabel('Wire Number', fontsize=14)

                if cfg.plot_show_error_text:
                    ax2.text(0.02, 0.98, f"Err: {angular_error_deg:.1f}°",
                             transform=ax2.transAxes, va='top', ha='left',
                             fontsize=14, color='w',
                             bbox=dict(facecolor='k', alpha=0.35, pad=2, edgecolor='none'))

                f1_orig = os.path.join(out_dir, f"{sample['meta']['fname'].split('.')[0]}__{view}_vec__origbg.png")
                plt.tight_layout(); plt.savefig(f1_orig, dpi=600); plt.close(fig2)

            prob_np = prob[0].detach().cpu().numpy()
            mx = mx_t[0].detach().cpu().numpy()
            my = my_t[0].detach().cpu().numpy()
            f2 = os.path.join(out_dir, f"{sample['meta']['fname'].split('.')[0]}__{view}__2x2_bragg.png")
            save_bragg_figure_dense(
                cfg=cfg,
                view=view,
                sample=sample,
                u_pred_1x2=u_pred,
                prob_2ch=prob_np,
                mx_2=mx,
                my_2=my,
                out_path=f2,
                angular_error_deg=angular_error_deg,
            )

            f3 = os.path.join(out_dir, f"{sample['meta']['fname'].split('.')[0]}__{view}__blocks.png")
            save_block_heatmap_figure_dense(
                cfg, view, sample, prob_np, mx, my, f3,
                angular_error_deg=angular_error_deg,
            )

            f_thesis_vec = os.path.join(out_dir, f"{sample['meta']['fname'].split('.')[0]}__{view}__thesis_vectors.png")
            save_thesis_vector_figure(
                cfg=cfg,
                view=view,
                sample=sample,
                u_pred=u_pred,
                u_gt=u_gt,
                prob_2ch=prob_np,
                mx_2=mx,
                my_2=my,
                out_path=f_thesis_vec,
                angular_error_deg=angular_error_deg,
                show_title=False
            )

        seg_pred_np = None
        seg_gt_np   = None
        if cfg.enable_seg and ("seg_logits" in out) and ("seg_mask" in sample):
            seg_pred_np = out["seg_logits"].argmax(1)[0].detach().cpu().numpy()
            seg_gt_np   = sample["seg_mask"].detach().cpu().numpy()

            if not cfg.skip_per_event_plots:
                f_seg = os.path.join(out_dir, f"{sample['meta']['fname'].split('.')[0]}__{view}__segmentation.png")
                save_segmentation_overlay_figure(
                    cfg=cfg,
                    view=view,
                    sample=sample,
                    seg_pred=seg_pred_np,
                    seg_gt=seg_gt_np,
                    out_path=f_seg,
                    dpi=600,
                    seg_logits=out["seg_logits"][0].detach().cpu().numpy(),
                    show_probs=True
                )

                f_thesis_seg = os.path.join(out_dir, f"{sample['meta']['fname'].split('.')[0]}__{view}__thesis_seg_error.png")
                save_thesis_segmentation_figure(
                    cfg=cfg,
                    view=view,
                    sample=sample,
                    seg_pred=seg_pred_np,
                    seg_gt=seg_gt_np,
                    out_path=f_thesis_seg,
                    show_title=False
                )

        bragg_overlap = None
        bragg_z = None
        try:
            adc = sample["image"].squeeze(0).numpy()
            (ovA, ovB), (zA, zB) = _compute_bragg_numbers(
                prob_2ch=prob_np, adc=adc, mx=mx, my=my,
                top_percent=cfg.bragg_topk_percent,
                zpatch=cfg.bragg_zscore_patch,
                roi=cfg.bragg_roi,
            )
            bragg_overlap = (ovA, ovB)
            bragg_z = (zA, zB)
        except Exception:
            pass

        brems_fraction = None
        if seg_gt_np is not None:
            michel_mask = (seg_gt_np == CLASS_MICHEL)
            brems_mask  = (seg_gt_np == CLASS_BREMS)
            denom = int((michel_mask | brems_mask).sum())
            if denom > 0:
                brems_fraction = float(brems_mask.sum() / denom)

        confidence = None
        try:
            conf_A, conf_B = compute_heatmap_confidence(prob_np, mx, my)
            confidence = min(conf_A, conf_B)
        except Exception:
            confidence = abs(float(cos_theta))

        return {
            "angular_error_deg": float(angular_error_deg),
            "cos_theta": float(cos_theta),
            "u_pred": u_pred,
            "u_gt": u_gt,
            "seg_pred": seg_pred_np,
            "seg_gt": seg_gt_np,
            "bragg_overlap": bragg_overlap,
            "bragg_z": bragg_z,
            "brems_fraction": brems_fraction,
            "confidence": confidence,
            "paths": {
                "vec": f1,
                "vec_orig": f1_orig if "image_pre_masks" in sample else None
            }
        }

def compute_heatmap_confidence(
    prob_2ch: np.ndarray,
    mx: np.ndarray,
    my: np.ndarray,
    *,
    radius_px: Optional[float] = None,
    radius_frac: float = 0.05,
    min_radius_px: float = 2.0,
    weights: Tuple[float, float, float, float] = (0.35, 0.25, 0.30, 0.10),
    overall_mode: str = "min",
    renormalize: bool = True,
    eps: float = 1e-12,
    return_overall: bool = False,
) -> Tuple[float, float] | Tuple[float, float, float]:

    assert prob_2ch.ndim == 3 and prob_2ch.shape[0] == 2, \
        "prob_2ch should have shape [2, H, W]"
    assert mx.shape[0] == 2 and my.shape[0] == 2, \
        "mx and my should have shape [2]"

    def _normalize(p: np.ndarray) -> np.ndarray:
        if not renormalize:
            return p
        s = float(np.sum(p))
        if not np.isfinite(s) or s <= eps:
            H, W = p.shape
            return np.full((H, W), 1.0 / (H * W), dtype=np.float64)
        return (p / s).astype(np.float64)

    def _channel_confidence(
        prob: np.ndarray,
        peak_x: float,
        peak_y: float
    ) -> float:
        H, W = prob.shape
        p = _normalize(prob)

        peak_x = float(np.clip(peak_x, 0.0, W - 1.0))
        peak_y = float(np.clip(peak_y, 0.0, H - 1.0))

        r = radius_px
        if r is None:
            r = max(min_radius_px, radius_frac * float(min(H, W)))
        r = float(r)

        yy, xx = np.ogrid[0:H, 0:W]
        dx = xx - peak_x
        dy = yy - peak_y
        dist_sq = dx * dx + dy * dy

        near_mask = dist_sq <= (r * r)
        concentration = float(p[near_mask].sum())

        p_safe = np.clip(p, eps, 1.0)
        entropy = -float(np.sum(p * np.log(p_safe)))
        max_entropy = float(np.log(H * W))
        entropy_score = 1.0 - (entropy / max(max_entropy, eps))
        entropy_score = float(np.clip(entropy_score, 0.0, 1.0))

        Er2 = float(np.sum(p * dist_sq))

        corners = np.array([
            [0.0, 0.0],
            [W - 1.0, 0.0],
            [0.0, H - 1.0],
            [W - 1.0, H - 1.0],
        ], dtype=np.float64)
        cx = corners[:, 0] - peak_x
        cy = corners[:, 1] - peak_y
        r2_far = float(np.max(cx * cx + cy * cy))
        spread_score = 1.0 - (Er2 / max(r2_far, 1.0))
        spread_score = float(np.clip(spread_score, 0.0, 1.0))

        peak_height = float(np.max(p))

        w_conc, w_ent, w_spr, w_peak = weights
        w_sum = float(w_conc + w_ent + w_spr + w_peak)
        if w_sum <= 0:
            w_conc, w_ent, w_spr, w_peak = 0.35, 0.25, 0.30, 0.10
            w_sum = 1.0
        w_conc /= w_sum
        w_ent  /= w_sum
        w_spr  /= w_sum
        w_peak /= w_sum

        conf = (
            w_conc * concentration +
            w_ent  * entropy_score +
            w_spr  * spread_score +
            w_peak * peak_height
        )

        return float(np.clip(conf, 0.0, 1.0))

    conf_A = _channel_confidence(prob_2ch[0], mx[0], my[0])
    conf_B = _channel_confidence(prob_2ch[1], mx[1], my[1])

    if overall_mode not in ("min", "mean", "geom"):
        raise ValueError("overall_mode must be one of: 'min', 'mean', 'geom'")

    if overall_mode == "min":
        overall = min(conf_A, conf_B)
    elif overall_mode == "mean":
        overall = 0.5 * (conf_A + conf_B)
    else:
        overall = float(np.sqrt(max(conf_A, 0.0) * max(conf_B, 0.0)))

    if return_overall:
        return conf_A, conf_B, float(np.clip(overall, 0.0, 1.0))

    return conf_A, conf_B
