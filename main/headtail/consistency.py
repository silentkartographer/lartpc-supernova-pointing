# Authored by Hilary Utaegbulam

"""Cross-view consistency: Methods A & B, 3D reconstruction, geometric validation."""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple

from .constants import *


def method_a_tick_sign_majority_batch(
    ux_img: dict,
    uy_img: dict,
) -> Tuple[dict, dict, dict]:
    """
    Simple head-tail consensus on tick sign (x-component in image space).
    drift_sign cancels when comparing tick signs across views for the same event.
    """
    views = ['U', 'V', 'Z']
    B = ux_img['U'].shape[0]
    device = ux_img['U'].device

    ux_out = {k: v.clone() for k, v in ux_img.items()}
    uy_out = {k: v.clone() for k, v in uy_img.items()}

    flipped_A = {v: torch.zeros(B, dtype=torch.bool, device=device) for v in views}

    for b in range(B):
        signs = {v: torch.sign(ux_img[v][b]) for v in views}

        sign_counts = {+1.0: 0, -1.0: 0, 0.0: 0}
        for s in signs.values():
            sign_counts[float(s)] += 1

        s_maj = +1.0 if sign_counts[+1.0] >= sign_counts[-1.0] else -1.0

        for v in views:
            if signs[v] != s_maj and signs[v] != 0.0:
                ux_out[v][b] *= -1
                uy_out[v][b] *= -1
                flipped_A[v][b] = True

    return ux_out, uy_out, {'flipped_A': flipped_A}


def _get_params(view):
    theta = THETA_DEG[view]
    pitch = PITCH_U if view == 'U' else (PITCH_V if view == 'V' else PITCH_Z)
    K = VBULK * T_TICK / pitch
    theta_rad = math.radians(theta)
    ny, nz = -math.sin(theta_rad), math.cos(theta_rad)
    return K, ny, nz


def reconstruct_3d_from_pair_batch(
    ux_img_A: torch.Tensor,  # [B]
    uy_img_A: torch.Tensor,  # [B]
    ux_img_B: torch.Tensor,  # [B]
    uy_img_B: torch.Tensor,  # [B]
    view_A: str,
    view_B: str,
    drift_sign: Optional[torch.Tensor] = None,  # [B] or scalar; +1 TPC1, -1 TPC0
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Two-plane reconstruction: 2D image directions → 3D direction.

    Physics convention
    ------------------
    The tick direction in image space maps to the physical x-axis (drift axis)
    with a sign that depends on which TPC the event is in:

        dticks/ds = drift_sign * ux_3d / (VBULK * T_TICK)

    Returns: [B, 3] unit 3D directions (ux, uy, uz)
    """
    device = ux_img_A.device
    B = ux_img_A.shape[0]

    if drift_sign is None:
        drift_sign = torch.ones(B, device=device, dtype=ux_img_A.dtype)
    elif not isinstance(drift_sign, torch.Tensor):
        drift_sign = torch.full((B,), float(drift_sign),
                                device=device, dtype=ux_img_A.dtype)
    else:
        drift_sign = drift_sign.to(device=device, dtype=ux_img_A.dtype)
        if drift_sign.dim() == 0:
            drift_sign = drift_sign.expand(B)

    K_A, ny_A, nz_A = _get_params(view_A)
    K_B, ny_B, nz_B = _get_params(view_B)

    ux_safe_A = ux_img_A.sign() * ux_img_A.abs().clamp(min=eps)
    ux_safe_B = ux_img_B.sign() * ux_img_B.abs().clamp(min=eps)

    slope_A = uy_img_A / ux_safe_A   # wire / tick  (can be negative)
    slope_B = uy_img_B / ux_safe_B

    # a_V = slope_V / K_V = (uy*ny_V + uz*nz_V) / (drift_sign * ux_3d)
    a_A = slope_A / K_A
    a_B = slope_B / K_B

    
    det = ny_A * nz_B - ny_B * nz_A
    det = det + eps * (1.0 if det >= 0 else -1.0)

    uy = drift_sign * (nz_B * a_A - nz_A * a_B) / det
    uz = drift_sign * (ny_A * a_B - ny_B * a_A) / det
    ux = torch.ones_like(uy)

    u_3d = torch.stack([ux, uy, uz], dim=1)
    u_3d = F.normalize(u_3d, dim=1, eps=eps)

    # Compare u vs -u using forward-consistency in the two views
    uxA_exp, uyA_exp = predict_view_from_3d_batch(u_3d, view_A, drift_sign=drift_sign, eps=eps)
    uxB_exp, uyB_exp = predict_view_from_3d_batch(u_3d, view_B, drift_sign=drift_sign, eps=eps)

    score = (ux_img_A * uxA_exp + uy_img_A * uyA_exp) + \
            (ux_img_B * uxB_exp + uy_img_B * uyB_exp)
    flip = score < 0
    u_3d[flip] = -u_3d[flip]

    return u_3d


def predict_view_from_3d_batch(
    u_3d: torch.Tensor,
    view: str,
    drift_sign: Optional[torch.Tensor] = None,
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward mapping: 3D → 2D image direction.
        dticks/ds = drift_sign * ux_3d / (VBULK * T_TICK)

    Returns (ux_img, uy_img), each [B], unit-normalised.
    """
    B = u_3d.shape[0]
    device = u_3d.device

    if drift_sign is None:
        drift_sign = torch.ones(B, device=device, dtype=u_3d.dtype)
    elif not isinstance(drift_sign, torch.Tensor):
        drift_sign = torch.full((B,), float(drift_sign), device=device, dtype=u_3d.dtype)
    else:
        drift_sign = drift_sign.to(device=device, dtype=u_3d.dtype)
        if drift_sign.dim() == 0:
            drift_sign = drift_sign.expand(B)

    theta = THETA_DEG[view]
    pitch = PITCH_U if view == 'U' else (PITCH_V if view == 'V' else PITCH_Z)
    theta_rad = math.radians(theta)
    ny, nz = -math.sin(theta_rad), math.cos(theta_rad)

    ux_3d, uy_3d, uz_3d = u_3d[:, 0], u_3d[:, 1], u_3d[:, 2]

    dticks_ds = drift_sign * ux_3d / (VBULK * T_TICK)
    dwires_ds = (uy_3d * ny + uz_3d * nz) / pitch

    u_2d = F.normalize(torch.stack([dticks_ds, dwires_ds], dim=1), dim=1, eps=eps)
    return u_2d[:, 0], u_2d[:, 1]


def method_b_geometric_consistency_batch(
    ux_img: dict,
    uy_img: dict,
    drift_sign: Optional[torch.Tensor] = None,
    flip_threshold: float = 0.85,
    drop_threshold: float = 0.3,
    eps: float = 1e-9,
) -> Tuple[dict, dict, dict]:
    """
    Geometric consistency round-robin: reconstruct each view from the other two
    and compare via cosine similarity. Flips the most decisive outlier if it
    contradicts the prediction strongly enough, or drops it if ambiguous.
    """
    views = ['U', 'V', 'Z']
    pairs = [
        ('U', 'V', 'Z'),
        ('U', 'Z', 'V'),
        ('V', 'Z', 'U'),
    ]

    B = ux_img['U'].shape[0]
    device = ux_img['U'].device

    if drift_sign is None:
        drift_sign = torch.ones(B, device=device, dtype=ux_img['U'].dtype)

    ux_out = {k: v.clone() for k, v in ux_img.items()}
    uy_out = {k: v.clone() for k, v in uy_img.items()}

    flipped_B = {v: torch.zeros(B, dtype=torch.bool, device=device) for v in views}
    dropped   = {v: torch.zeros(B, dtype=torch.bool, device=device) for v in views}

    C_scores = {}
    for view_A, view_B, view_C in pairs:
        u_3d = reconstruct_3d_from_pair_batch(
            ux_img[view_A], uy_img[view_A],
            ux_img[view_B], uy_img[view_B],
            view_A, view_B,
            drift_sign=drift_sign,
            eps=eps,
        )
        ux_exp, uy_exp = predict_view_from_3d_batch(u_3d, view_C, drift_sign=drift_sign, eps=eps)
        C = (ux_img[view_C] * ux_exp + uy_img[view_C] * uy_exp).clamp(-1.0, 1.0)
        C_scores[f"{view_A}{view_B}_to_{view_C}"] = C

    for b in range(B):
        scores_b = {
            'Z': C_scores['UV_to_Z'][b],
            'V': C_scores['UZ_to_V'][b],
            'U': C_scores['VZ_to_U'][b],
        }
        abs_scores    = {v: abs(float(c)) for v, c in scores_b.items()}
        most_decisive = max(abs_scores, key=abs_scores.get)
        C_val         = float(scores_b[most_decisive])

        if C_val < -flip_threshold:
            ux_out[most_decisive][b] *= -1
            uy_out[most_decisive][b] *= -1
            flipped_B[most_decisive][b] = True
        elif abs(C_val) < drop_threshold:
            dropped[most_decisive][b] = True

    info = {
        'flipped_B': flipped_B,
        'dropped':   dropped,
        'C_scores':  C_scores,
    }
    return ux_out, uy_out, info


def apply_consistency_pipeline(
    ux_img: dict,
    uy_img: dict,
    drift_sign: Optional[torch.Tensor] = None,
    cfg=None,
) -> dict:
    """
    Full A → B pipeline.

    Args:
        ux_img, uy_img : per-view image-space unit directions {'U':[B],'V':[B],'Z':[B]}
        drift_sign     : [B] tensor, +1 for TPC1, -1 for TPC0. Defaults to all +1.
        cfg            : reads consistency_flip_threshold and consistency_drop_threshold.
    """
    flip_thr = getattr(cfg, 'consistency_flip_threshold', 0.85) if cfg else 0.85
    drop_thr = getattr(cfg, 'consistency_drop_threshold', 0.30) if cfg else 0.30

    ux_a, uy_a, info_a = method_a_tick_sign_majority_batch(ux_img, uy_img)
    ux_b, uy_b, info_b = method_b_geometric_consistency_batch(
        ux_a, uy_a,
        drift_sign=drift_sign,
        flip_threshold=flip_thr,
        drop_threshold=drop_thr,
    )

    return {
        'ux_corrected': ux_b,
        'uy_corrected': uy_b,
        'info_A': info_a,
        'info_B': info_b,
    }


def evaluate_consistency_corrections(
    ux_img_original: dict,
    uy_img_original: dict,
    ux_img_corrected: dict,
    uy_img_corrected: dict,
    u_gt: dict,
    consistency_info: dict,
    eps: float = 1e-9,
) -> dict:
    """Validate A→B corrections against ground truth."""
    views = ['U', 'V', 'Z']
    B = ux_img_original['U'].shape[0]
    device = ux_img_original['U'].device

    metrics = {'per_view': {}, 'aggregate': {}}

    for v in views:
        u_orig = F.normalize(
            torch.stack([ux_img_original[v], uy_img_original[v]], dim=1), dim=1, eps=eps)
        u_corr = F.normalize(
            torch.stack([ux_img_corrected[v], uy_img_corrected[v]], dim=1), dim=1, eps=eps)
        u_true = F.normalize(u_gt[v], dim=1, eps=eps)

        cos_orig = (u_orig * u_true).sum(dim=1).clamp(-1, 1)
        cos_corr = (u_corr * u_true).sum(dim=1).clamp(-1, 1)

        correct_orig = (cos_orig > 0).float()
        correct_corr = (cos_corr > 0).float()

        flipped_A     = consistency_info['info_A']['flipped_A'][v]
        flipped_B     = consistency_info['info_B']['flipped_B'][v]
        flipped_total = flipped_A | flipped_B
        dropped       = consistency_info['info_B']['dropped'][v]

        not_dropped = ~dropped
        acc_before  = float(correct_orig.mean())
        acc_after   = float(correct_corr[not_dropped].mean()) if not_dropped.any() else 0.0

        was_wrong  = (correct_orig == 0)
        is_correct = (correct_corr == 1)

        tp_flip = int((was_wrong & flipped_total & is_correct).sum())
        fp_flip = int((correct_orig.bool() & flipped_total & ~is_correct).sum())
        tn_flip = int((correct_orig.bool() & ~flipped_total & is_correct).sum())
        fn_flip = int((was_wrong & ~flipped_total & ~is_correct).sum())

        flip_precision = tp_flip / max(1, tp_flip + fp_flip)
        flip_recall    = tp_flip / max(1, tp_flip + fn_flip)

        if v == 'Z':
            C_score = consistency_info['info_B']['C_scores']['UV_to_Z']
        elif v == 'V':
            C_score = consistency_info['info_B']['C_scores']['UZ_to_V']
        else:
            C_score = consistency_info['info_B']['C_scores']['VZ_to_U']

        C_abs     = C_score.abs()
        high_conf = C_abs > 0.7
        low_conf  = C_abs < 0.3

        acc_high_conf = float(correct_corr[high_conf].mean()) if high_conf.any() else 0.0
        acc_low_conf  = float(correct_corr[low_conf].mean())  if low_conf.any()  else 0.0

        metrics['per_view'][v] = {
            'n_total':         B,
            'n_flipped_A':     int(flipped_A.sum()),
            'n_flipped_B':     int(flipped_B.sum()),
            'n_flipped_total': int(flipped_total.sum()),
            'n_dropped':       int(dropped.sum()),
            'acc_before':      acc_before,
            'acc_after':       acc_after,
            'acc_gain':        acc_after - acc_before,
            'flip_tp':         tp_flip,
            'flip_fp':         fp_flip,
            'flip_tn':         tn_flip,
            'flip_fn':         fn_flip,
            'flip_precision':  flip_precision,
            'flip_recall':     flip_recall,
            'acc_high_conf':   acc_high_conf,
            'acc_low_conf':    acc_low_conf,
            'mean_C_abs':      float(C_abs.mean()),
        }

    total_before = sum(m['acc_before'] for m in metrics['per_view'].values()) / 3
    total_after  = sum(m['acc_after']  for m in metrics['per_view'].values()) / 3
    total_tp     = sum(m['flip_tp'] for m in metrics['per_view'].values())
    total_fp     = sum(m['flip_fp'] for m in metrics['per_view'].values())

    metrics['aggregate'] = {
        'acc_before':    total_before,
        'acc_after':     total_after,
        'acc_gain':      total_after - total_before,
        'total_flipped': sum(m['n_flipped_total'] for m in metrics['per_view'].values()),
        'flip_precision': total_tp / max(1, total_tp + total_fp),
    }
    return metrics


def log_consistency_metrics(metrics: dict, epoch: int, phase: str, rank: int = 0):
    if rank != 0:
        return
    print(f"\n{'='*80}")
    print(f"A→B Consistency Validation | Epoch {epoch} | {phase}")
    print(f"{'='*80}")
    agg = metrics['aggregate']
    print(f"OVERALL:")
    print(f"  Accuracy Before: {100*agg['acc_before']:.2f}%")
    print(f"  Accuracy After:  {100*agg['acc_after']:.2f}%")
    print(f"  Gain:            {100*agg['acc_gain']:+.2f}%")
    print(f"  Total Flipped:   {agg['total_flipped']}")
    print(f"  Flip Precision:  {100*agg['flip_precision']:.2f}%")
    print(f"\nPER-VIEW:")
    for v in ['U', 'V', 'Z']:
        m = metrics['per_view'][v]
        print(f"  {v}:")
        print(f"    Acc: {100*m['acc_before']:.1f}% → {100*m['acc_after']:.1f}% ({100*m['acc_gain']:+.1f}%)")
        print(f"    Flipped: A={m['n_flipped_A']}, B={m['n_flipped_B']}, Dropped={m['n_dropped']}")
        print(f"    Flip Quality: Precision={100*m['flip_precision']:.1f}%, Recall={100*m['flip_recall']:.1f}%")
        print(f"    C-score: mean|C|={m['mean_C_abs']:.3f}, acc@high={100*m['acc_high_conf']:.1f}%, acc@low={100*m['acc_low_conf']:.1f}%")
    print(f"{'='*80}\n")


def compute_tick_consistency_losses(
    mx_sw_U: torch.Tensor,
    mx_sw_V: torch.Tensor,
    mx_sw_Z: torch.Tensor,
    cfg,
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Multi-view tick consistency losses."""
    device = mx_sw_U.device
    B = mx_sw_U.shape[0]

    tA_U, tB_U = mx_sw_U[:, 0], mx_sw_U[:, 1]
    tA_V, tB_V = mx_sw_V[:, 0], mx_sw_V[:, 1]
    tA_Z, tB_Z = mx_sw_Z[:, 0], mx_sw_Z[:, 1]

    dU = tB_U - tA_U
    dV = tB_V - tA_V
    dZ = tB_Z - tA_Z

    DT_SCALE = float(cfg.tick_scale)
    dU_n = dU / DT_SCALE
    dV_n = dV / DT_SCALE
    dZ_n = dZ / DT_SCALE

    L_sign = (
        F.relu(-dU_n * dV_n) +
        F.relu(-dU_n * dZ_n) +
        F.relu(-dV_n * dZ_n)
    ).mean()

    d_stack  = torch.stack([dU_n, dV_n, dZ_n], dim=-1)
    mu       = d_stack.mean(dim=-1, keepdim=True)
    resid    = d_stack - mu

    tau       = float(cfg.tick_tau) / DT_SCALE
    resid_eff = torch.sign(resid) * F.relu(resid.abs() - tau)
    L_val     = F.smooth_l1_loss(resid_eff, torch.zeros_like(resid_eff))

    logs = {
        'tick_sign_loss': L_sign.detach(),
        'tick_val_loss':  L_val.detach(),
        'tick_dU_mean':   dU.mean().detach(),
        'tick_dV_mean':   dV.mean().detach(),
        'tick_dZ_mean':   dZ.mean().detach(),
        'tick_delta_std': d_stack.std().detach(),
    }
    return L_sign, L_val, logs
