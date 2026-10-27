# Authored by Hilary Utaegbulam

"""Loss functions: segmentation, DSNT stabilizers, physics-informed brems losses."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np

from .constants import CLASS_MICHEL, CLASS_BREMS

def overlap_loss_dense(prob_ab: torch.Tensor) -> torch.Tensor:
    # prob_ab: [B,2,H,W]
    v = prob_ab[:,0]; h = prob_ab[:,1]
    return (v * h).sum(dim=(1,2)).mean()

def entropy_loss_dense(prob_ab: torch.Tensor, eps=1e-12) -> torch.Tensor:
    p = prob_ab.clamp_min(eps)
    H = -(p * p.log()).sum(dim=(2,3)).mean()  # mean over B,C
    return H


def dice_loss_dense(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_bg: bool = True,
    ignore_index: Optional[int] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    # probs: [B,C,H,W]
    probs = F.softmax(logits, dim=1)

    if ignore_index is not None:
        valid = target.ne(int(ignore_index))                             # [B,H,W] bool
        target_valid = torch.where(valid, target, torch.zeros_like(target))
    else:
        valid = torch.ones_like(target, dtype=torch.bool)
        target_valid = target

    target_valid = target_valid.clamp(min=0, max=num_classes - 1)

    t_one = F.one_hot(target_valid, num_classes=num_classes).permute(0, 3, 1, 2).float()

    if ignore_bg and num_classes > 0:
        probs = probs[:, 1:]
        t_one = t_one[:, 1:]

    valid_f = valid.unsqueeze(1).float()                                 # [B,1,H,W]
    inter = (probs * t_one * valid_f).sum(dim=(0, 2, 3))
    denom = ((probs + t_one) * valid_f).sum(dim=(0, 2, 3)).clamp_min(eps)

    dice = 1.0 - (2.0 * inter / denom).mean()
    return dice

def ce_loss_dense(logits: torch.Tensor, target: torch.Tensor, class_weights: Optional[torch.Tensor]=None) -> torch.Tensor:
    return F.cross_entropy(logits, target, weight=class_weights)


def focal_loss_dense(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[torch.Tensor] = None,
    ignore_index: Optional[int] = None,
) -> torch.Tensor:
    B, C, H, W = logits.shape

    if ignore_index is None:
        ce = F.cross_entropy(logits, target, reduction="none")
        valid = torch.ones_like(target, dtype=torch.bool)
    else:
        ce = F.cross_entropy(logits, target, reduction="none", ignore_index=int(ignore_index))
        valid = target.ne(int(ignore_index))

    pt = torch.exp(-ce)                       
    loss = ((1 - pt).pow(gamma)) * ce         

    if alpha is not None:
        if alpha.ndim == 0:
            alpha_bg = 1.0 - float(alpha.item())
            alpha_vec = torch.full((C,), float(alpha.item()), dtype=logits.dtype, device=logits.device)
            alpha_vec[0] = alpha_bg
        else:
            alpha_vec = alpha.to(dtype=logits.dtype, device=logits.device)
            assert alpha_vec.shape[0] == C, "alpha must have length C"

        idx = target.clamp(min=0, max=C-1).long()
        alpha_map = alpha_vec[idx]            # [B,H,W]
        loss = loss * alpha_map

    # masked mean over valid pixels only
    loss = (loss * valid).sum() / valid.sum().clamp_min(1)
    return loss


def compute_seg_loss_dense(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_type: str,
    dice_weight: float = 1.0,
    focal_gamma: float = 2.0,
    focal_alpha: Optional[torch.Tensor] = None,
    num_classes: int = 1,
    ignore_index: Optional[int] = None,   # <--- add this
) -> Tuple[torch.Tensor, torch.Tensor]:

    """
    Returns (seg_loss, dice_term_for_logging)
    - If 'dice' not present, dice_term_for_logging = 0 (for meters).
    """
    lt = loss_type.lower()
    dice_term = logits.new_tensor(0.0)

    if lt == "ce":
        seg_loss = F.cross_entropy(logits, target, ignore_index=ignore_index)
    elif lt == "dice":
        seg_loss = dice_loss_dense(logits, target, num_classes=num_classes,
                                   ignore_bg=True, ignore_index=ignore_index)
        dice_term = seg_loss.detach()
    elif lt == "ce+dice":
        ce = F.cross_entropy(logits, target, ignore_index=ignore_index)
        dice = dice_loss_dense(logits, target, num_classes=num_classes,
                               ignore_bg=True, ignore_index=ignore_index)
        seg_loss = ce + dice_weight * dice
        dice_term = dice.detach()
    elif lt == "focal+dice":
        fl = focal_loss_dense(logits, target, gamma=focal_gamma, alpha=focal_alpha,
                              ignore_index=ignore_index)
        dice = dice_loss_dense(logits, target, num_classes=num_classes,
                               ignore_bg=True, ignore_index=ignore_index)
        seg_loss = fl + dice_weight * dice
        dice_term = dice.detach()
    else:
        raise ValueError(f"Unknown seg_loss_type='{loss_type}'")

    return seg_loss, dice_term


def prior_bias_from_seg_dense(seg_logits: torch.Tensor, michel_idx=CLASS_MICHEL, brems_idx=CLASS_BREMS) -> torch.Tensor:
    probs = F.softmax(seg_logits, dim=1)
    keep = probs[:, michel_idx:michel_idx+1, ...]
    if brems_idx < probs.shape[1]:
        keep = keep + probs[:, brems_idx:brems_idx+1, ...]
    return keep.clamp(0,1)  # [B,1,H,W]


def brems_sumcos(mx, my, seg_logits, brems_idx, eps=1e-6):
    Bsz, C, H, W = seg_logits.shape
    device = seg_logits.device
    seg_prob = torch.softmax(seg_logits, dim=1)
    Bmap = seg_prob[:, brems_idx:brems_idx+1, ...]  # [B,1,H,W]

    xs = torch.linspace(0, W-1, W, device=device).view(1,1,1,W).expand(Bsz,1,H,W)
    ys = torch.linspace(0, H-1, H, device=device).view(1,1,H,1).expand(Bsz,1,H,W)

    xA, xB = mx[:,0].view(Bsz,1,1,1), mx[:,1].view(Bsz,1,1,1)
    yA, yB = my[:,0].view(Bsz,1,1,1), my[:,1].view(Bsz,1,1,1)

    vx, vy = (xB - xA), (yB - yA)
    vn = torch.sqrt(vx*vx + vy*vy + eps)
    ux, uy = vx / vn, vy / vn

    rx1, ry1 = xs - xA, ys - yA
    r1n = torch.sqrt(rx1*rx1 + ry1*ry1 + eps)
    rx1, ry1 = rx1 / r1n, ry1 / r1n
    c1 = (ux * rx1 + uy * ry1).clamp(-1, 1)

    rx2, ry2 = xs - xB, ys - yB
    r2n = torch.sqrt(rx2*rx2 + ry2*ry2 + eps)
    rx2, ry2 = rx2 / r2n, ry2 / r2n
    c2 = (-(ux) * rx2 + (-(uy)) * ry2).clamp(-1, 1)

    wsum = Bmap.sum(dim=(2,3), keepdim=True).clamp_min(eps)
    S1 = (Bmap * c1).sum(dim=(2,3), keepdim=True) / wsum
    S2 = (Bmap * c2).sum(dim=(2,3), keepdim=True) / wsum

    S1 = S1.squeeze(-1).squeeze(-1).squeeze(-1)
    S2 = S2.squeeze(-1).squeeze(-1).squeeze(-1)
    dS = (S1 - S2)
    return S1, S2, dS, c1, c2, Bmap

def loss_brems_margin(S1, S2, margin=0.10):
    # dS = (S1 - S2).abs()
    # return F.relu(margin - dS).mean()
    return F.relu(margin - (S1 - S2)).mean() # enforces S1 >= S2 + margin

def loss_brems_maxmin(S1, S2, lambda_min=0.25):
    Smax = torch.maximum(S1, S2)
    Smin = torch.minimum(S1, S2)
    return (-Smax + lambda_min * Smin).mean()

def loss_brems_contrastive_soft(c_field, Bmap, T=0.5, gamma=1.0):
    w_pos = Bmap * F.relu(c_field).pow(gamma)
    w_neg = Bmap * F.relu(-c_field).pow(gamma)
    eps = 1e-6
    Spos = (w_pos * c_field).sum((2,3)) / (w_pos.sum((2,3)) + eps)
    Sneg = (w_neg * c_field).sum((2,3)) / (w_neg.sum((2,3)) + eps)
    sep  = (Spos - Sneg).squeeze(1)
    return F.softplus(-(sep / T)).mean()

def loss_brems_contrastive_sampled(c_field, Bmap, K=64, T=0.5):
    Bsz, _, H, W = Bmap.shape
    device = c_field.device
    c = c_field.view(Bsz, -1)
    w = Bmap.view(Bsz, -1)
    pos_w = w * F.relu(c); neg_w = w * F.relu(-c)
    eps = 1e-12; losses = []
    for b in range(Bsz):
        pw = pos_w[b]; nw = neg_w[b]
        if pw.sum() < eps or nw.sum() < eps: continue
        pos_idx = torch.multinomial((pw / pw.sum()).clamp_min(eps), num_samples=K, replacement=True)
        neg_idx = torch.multinomial((nw / nw.sum()).clamp_min(eps), num_samples=K, replacement=True)
        c_pos = c[b, pos_idx]; c_neg = c[b, neg_idx]
        logits = (c_pos - c_neg) / T
        losses.append(-F.logsigmoid(logits).mean())
    if len(losses) == 0:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()

def brems_physics_loss(
    mx, my, seg_logits, brems_idx,
    use_A=False, use_B=False, use_C=False,
    A_margin=0.10,
    B_lambda_min=0.25,
    C_mode="soft",
    C_T=0.5, C_gamma=1.0, C_K=64,
    weights=(1.0, 1.0, 1.0)
):
    S1, S2, dS, c1, c2, Bmap = brems_sumcos(mx, my, seg_logits, brems_idx)
    L_total = mx.new_tensor(0.0)
    logs = {"S1": S1.detach(), "S2": S2.detach(), "dS": dS.detach()}

    if use_A:
        LA = loss_brems_margin(S1, S2, margin=A_margin)
        L_total = L_total + weights[0] * LA
        logs["L_A"] = LA.detach()
    if use_B:
        LB = loss_brems_maxmin(S1, S2, lambda_min=B_lambda_min)
        L_total = L_total + weights[1] * LB
        logs["L_B"] = LB.detach()
    if use_C:
        # choose c1 or c2 based on which is larger (more forward)
        # use_h1 = (S1 >= S2).view(-1,1,1,1)
        # c_star = torch.where(use_h1, c1, c2)

        # always measure forwardness from START (A)
        # c_star = c1  

        # try two headed contrastive loss 
        if C_mode == "twoheaded":
            LCA = loss_brems_contrastive_soft(c1, Bmap, T=C_T, gamma=C_gamma)
            LCB = loss_brems_contrastive_soft(-c2, Bmap, T=C_T, gamma=C_gamma)
            LC = 0.5 * (LCA + LCB)
        elif C_mode == "soft":
            # choose c1 or c2 based on which is larger (more forward)
            # use_h1 = (S1 >= S2).view(-1,1,1,1)
            # c_star = torch.where(use_h1, c1, c2)

            # always measure forwardness from START (A)
            c_star = c1 
            LC = loss_brems_contrastive_soft(c_star, Bmap, T=C_T, gamma=C_gamma)
        else:
            # choose c1 or c2 based on which is larger (more forward)
            # use_h1 = (S1 >= S2).view(-1,1,1,1)
            # c_star = torch.where(use_h1, c1, c2)
            
            # always measure forwardness from START (A)
            c_star = c1 
            LC = loss_brems_contrastive_sampled(c_star, Bmap, K=C_K, T=C_T)
        L_total = L_total + weights[2] * LC
        logs["L_C"] = LC.detach()
    return L_total, logs

