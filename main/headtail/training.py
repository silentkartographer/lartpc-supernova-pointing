# Authored by Hilary Utaegbulam

"""Training loops: run_epoch_dense and run_epoch_multi with DDP support."""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Tuple, Optional

from .constants import (
    Config, IGNORE_INDEX, CLASS_BREMS, CLASS_MICHEL,
)
from .physics_utils import (
    spatial_softmax2d_logits, dsnt_expectation, unit_from_points,
    angular_loss, ray_distance_loss, ray_length_loss,
    _gaussian_2d,
)
from .losses import (
    overlap_loss_dense, entropy_loss_dense, compute_seg_loss_dense,
    prior_bias_from_seg_dense, brems_physics_loss,
)
from .consistency import (
    apply_consistency_pipeline, evaluate_consistency_corrections,
    log_consistency_metrics, compute_tick_consistency_losses,
)
import math

def run_epoch_dense(model, loader, opt, device, cfg, epoch: int, total_epochs: int, train: bool, is_distributed=False):
    if train: 
        model.train()
    else:
        model.eval()

    if cfg.overlap_weight0 is None: cfg.overlap_weight0 = cfg.overlap_weight
    if cfg.entropy_weight0 is None: cfg.entropy_weight0 = cfg.entropy_weight
    if cfg.sep_margin_px0 is None:  cfg.sep_margin_px0  = cfg.sep_margin_px

    t = float(epoch) / max(1, total_epochs - 1)

    cfg.dsnt_temp = (1.0 - t) * cfg.dsnt_temp_hi + t * cfg.dsnt_temp_lo
    cfg.overlap_weight = cfg.overlap_weight0 * (1.0 - t)
    cfg.entropy_weight = cfg.entropy_weight0 * (1.0 - t)
    cfg.sep_margin_px  = cfg.sep_margin_px0  * (1.0 - t)

    T = max(1, getattr(cfg, "brems_warmup_epochs", 0))
    ramp = min(1.0, float(epoch) / float(T)) if T > 0 else 1.0
    cfg.lambda_brems_now = ramp * cfg.lambda_brems

    meters = {"loss":0.0, "ang":0.0, "sep":0.0, "ov":0.0, "ent":0.0, "seg":0.0, "dice":0.0, "N":0}

    for batch in loader:
        if batch is None: continue
        img = batch["image"].to(device)
        u_gt = batch["u_gt"].to(device)
        if train and opt is not None: opt.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            out = model(img)
            logits_ab = out["logits_ab"]
            keep_mask = None
            if cfg.enable_seg and ("seg_logits" in out):
                seg_logits = out["seg_logits"]
                seg_t = batch["seg_mask"].to(device)

                focal_alpha = None
                if cfg.seg_loss_type.lower().startswith("focal"):
                    if cfg.seg_focal_alpha is not None:
                        if isinstance(cfg.seg_focal_alpha, (list, tuple)):
                            focal_alpha = torch.tensor(cfg.seg_focal_alpha, dtype=seg_logits.dtype, device=device)
                        else:
                            focal_alpha = torch.tensor(float(cfg.seg_focal_alpha), dtype=seg_logits.dtype, device=device)

                seg_loss, dice_term = compute_seg_loss_dense(
                    seg_logits, seg_t,
                    loss_type=cfg.seg_loss_type,
                    dice_weight=cfg.seg_dice_weight if "dice" in cfg.seg_loss_type.lower() else 0.0,
                    focal_gamma=cfg.seg_focal_gamma,
                    focal_alpha=focal_alpha,
                    num_classes=cfg.num_classes,
                    ignore_index=IGNORE_INDEX,
                )

                keep_mask = None
                if cfg.prior_mode in ("soft","hard"):
                    prior = prior_bias_from_seg_dense(seg_logits)
                    if cfg.prior_mode == "soft" and cfg.prior_alpha > 0:
                        logits_ab = logits_ab + cfg.prior_alpha * (prior - 0.5)
                    elif cfg.prior_mode == "hard":
                        with torch.no_grad():
                            B_,_,H,W = prior.shape
                            keep_mask = torch.zeros_like(prior, dtype=torch.bool)
                            q = cfg.prior_topk_percent
                            for b in range(B_):
                                vals = prior[b,0].flatten()
                                k = max(1, int(len(vals) * (q/100.0)))
                                thr = torch.topk(vals, k).values.min()
                                keep_mask[b,0] = (prior[b,0] >= thr)
                        m = keep_mask.expand(-1,2,-1,-1)
                        logits_ab = torch.where(m, logits_ab, torch.finfo(logits_ab.dtype).min)
            else:
                seg_loss = torch.tensor(0.0, device=device)
                dice_term = torch.tensor(0.0, device=device)

            prob = spatial_softmax2d_logits(logits_ab, temperature=cfg.dsnt_temp)
            mx, my = dsnt_expectation(prob)

            ux_ab, uy_ab, sep = unit_from_points(mx[:,0], my[:,0], mx[:,1], my[:,1])
            u_ab = torch.stack([ux_ab, uy_ab], dim=1)

            order_mode_eff="fixed"

            if order_mode_eff == "fixed":
                u_pred = u_ab
                mx_sw, my_sw = mx, my
            elif order_mode_eff == "endpoints":
                u_end = F.normalize(batch["u_endpoints"].to(device), dim=1, eps=1e-9)
                cos = (F.normalize(u_ab, dim=1) * u_end).sum(dim=1)
                need_swap = (cos < 0.0).view(-1, 1)
                mx_sw = torch.where(need_swap, torch.stack([mx[:,1], mx[:,0]], dim=1), mx)
                my_sw = torch.where(need_swap, torch.stack([my[:,1], my[:,0]], dim=1), my)
                ux_sw, uy_sw, sep = unit_from_points(mx_sw[:,0], my_sw[:,0], mx_sw[:,1], my_sw[:,1])
                u_pred = torch.stack([ux_sw, uy_sw], dim=1)
            elif order_mode_eff == "perm_inv":
                ux_ba, uy_ba, _ = unit_from_points(mx[:,1], my[:,1], mx[:,0], my[:,0])
                u_ba = torch.stack([ux_ba, uy_ba], dim=1)
                u_gt_n = F.normalize(batch["u_gt"].to(device), dim=1, eps=1e-9)
                cos_ab = (F.normalize(u_ab, dim=1) * u_gt_n).sum(dim=1)
                cos_ba = (F.normalize(u_ba, dim=1) * u_gt_n).sum(dim=1)
                take_ba = (cos_ba > cos_ab).view(-1,1)
                mx_sw = torch.where(take_ba, torch.stack([mx[:,1], mx[:,0]], dim=1), mx)
                my_sw = torch.where(take_ba, torch.stack([my[:,1], my[:,0]], dim=1), my)
                u_pred = torch.where(take_ba, u_ba, u_ab)
            else:
                raise ValueError(f"Unknown dsnt_order_mode={order_mode_eff!r}")

            coord_loss = torch.tensor(0.0, device=device)
            kl_loss    = torch.tensor(0.0, device=device)
            ray_perp_loss = torch.tensor(0.0, device=device)
            ray_fwd_loss  = torch.tensor(0.0, device=device)
            ray_len_loss  = torch.tensor(0.0, device=device)
            ray_cos_loss  = torch.tensor(0.0, device=device)
            
            if ("head_xy" in batch) and ("tail_xy" in batch):
                Bsz = img.shape[0]
                H, W = prob.shape[-2], prob.shape[-1]
            
                gxA = batch["head_xy"].to(device)[:,0]; gyA = batch["head_xy"].to(device)[:,1]
                gxB = batch["tail_xy"].to(device)[:,0]; gyB = batch["tail_xy"].to(device)[:,1]
            
                def _in_bounds(x, y):
                    return (x >= 0) & (x <= (W - 1)) & (y >= 0) & (y <= (H - 1))
            
                validA = _in_bounds(gxA, gyA)
                
                if cfg.gt_vector_mode == "truth_momentum_pdf":
                    u_norm = torch.linalg.vector_norm(u_gt, dim=1)
                    valid = validA & (u_norm > 1e-6)
            
                    if valid.any():
                        xA, yA = mx_sw[:,0][valid], my_sw[:,0][valid]
                        xB, yB = mx_sw[:,1][valid], my_sw[:,1][valid]
                        gxA_v, gyA_v = gxA[valid], gyA[valid]
            
                        coord_loss = F.smooth_l1_loss(xA, gxA_v) + F.smooth_l1_loss(yA, gyA_v)
                        
                        A_xy = torch.stack([xA, yA], dim=1)
                        B_xy = torch.stack([xB, yB], dim=1)
                        
                        u_gt_valid = u_gt[valid]
                        
                        L_perp, L_fwd, t = ray_distance_loss(A_xy, B_xy, u_gt_valid)
                        ray_perp_loss = L_perp
                        ray_fwd_loss = L_fwd
                        
                        if cfg.lambda_ray_len > 0:
                            ray_len_loss = ray_length_loss(t, t_min=cfg.ray_t_min)
                        
                        if train and epoch <= cfg.ray_warmup_epochs and cfg.lambda_ray_cos > 0:
                            u_pred_valid = u_pred[valid]
                            cos_sim = (F.normalize(u_pred_valid, dim=1) * F.normalize(u_gt_valid, dim=1)).sum(dim=1)
                            ray_cos_loss = (1.0 - cos_sim).mean()
                        
                        targ = torch.zeros(valid.sum().item(), 1, H, W, device=prob.device, dtype=prob.dtype)
                        sig = float(getattr(cfg, "hm_sigma", 1.5))
                        valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
                        for j, b in enumerate(valid_idx.tolist()):
                            cxA_j, cyA_j = batch["head_xy"][b]
                            targ[j,0] = _gaussian_2d(H, W, cxA_j, cyA_j, sigma=sig, device=prob.device, dtype=prob.dtype)
                        
                        logits_A = logits_ab[valid, 0:1, :, :]
                        pA = torch.softmax(logits_A.view(logits_A.size(0), 1, -1), dim=-1)
                        pA = pA.view_as(logits_A).clamp_min(1e-12)
                        
                        tA = targ.clamp_min(1e-12)
                        kl_loss = (tA * (tA.log() - pA.log())).sum(dim=(1,2,3)).mean()
                        
                else:
                    gxB = batch["tail_xy"].to(device)[:,0]
                    gyB = batch["tail_xy"].to(device)[:,1]
                    validB = _in_bounds(gxB, gyB)
                    valid  = validA & validB

                    if valid.any():
                        xA, yA = mx_sw[:,0][valid], my_sw[:,0][valid]
                        xB, yB = mx_sw[:,1][valid], my_sw[:,1][valid]
                        gxA_v, gyA_v = gxA[valid], gyA[valid]
                        gxB_v, gyB_v = gxB[valid], gyB[valid]

                        coord_loss = (
                            F.smooth_l1_loss(xA, gxA_v) + F.smooth_l1_loss(yA, gyA_v) +
                            F.smooth_l1_loss(xB, gxB_v) + F.smooth_l1_loss(yB, gyB_v)
                        )

                        targ = torch.zeros(valid.sum().item(), 2, H, W, device=prob.device, dtype=prob.dtype)
                        sig = float(getattr(cfg, "hm_sigma", 1.5))
                        valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
                        for j, b in enumerate(valid_idx.tolist()):
                            cxA_j, cyA_j = batch["head_xy"][b]
                            cxB_j, cyB_j = batch["tail_xy"][b]
                            targ[j,0] = _gaussian_2d(H, W, cxA_j, cyA_j, sigma=sig, device=prob.device, dtype=prob.dtype)
                            targ[j,1] = _gaussian_2d(H, W, cxB_j, cyB_j, sigma=sig, device=prob.device, dtype=prob.dtype)

                        logits_V = logits_ab[valid]
                        p = torch.softmax(logits_V.view(logits_V.size(0), 2, -1), dim=-1)
                        p = p.view_as(logits_V).clamp_min(1e-12)
                        
                        t = targ.clamp_min(1e-12)
                        kl_loss = (t * (t.log() - p.log())).sum(dim=(1,2,3)).mean()

            L_phys = torch.tensor(0.0, device=device)
            phys_logs = {}
            if cfg.enable_seg and ("seg_logits" in out):
                L_phys, phys_logs = brems_physics_loss(
                    mx_sw, my_sw, out["seg_logits"], brems_idx=CLASS_BREMS,
                    use_A=cfg.use_brems_A,
                    use_B=cfg.use_brems_B,
                    use_C=cfg.use_brems_C,
                    A_margin=cfg.brems_margin,
                    B_lambda_min=cfg.brems_lambda_min,
                    C_mode=cfg.brems_C_mode,
                    C_T=cfg.brems_C_T, C_gamma=cfg.brems_C_gamma, C_K=cfg.brems_C_K,
                    weights=cfg.brems_inner_weights,
                )

            L_ang = angular_loss(u_pred, u_gt)
            L_sep = F.relu(cfg.sep_margin_px - sep).mean() if cfg.sep_margin_px > 0 else torch.tensor(0.0, device=device)
            L_ov  = cfg.overlap_weight * overlap_loss_dense(prob) if cfg.overlap_weight>0 else torch.tensor(0.0, device=device)
            L_ent = cfg.entropy_weight * entropy_loss_dense(prob) if cfg.entropy_weight>0 else torch.tensor(0.0, device=device)

            L_dir = torch.tensor(0.0, device=device)
            if cfg.enable_dirreg and ("dir_vec" in out):
                u_dir = F.normalize(out["dir_vec"], dim=1)
                L_dir = cfg.dirreg_weight * angular_loss(u_dir, u_gt)

            lb = getattr(cfg, "lambda_brems_now", cfg.lambda_brems)
            loss = (
                cfg.lambda_ang * L_ang
                + L_sep + L_ov + L_ent
                + seg_loss * cfg.lambda_seg
                + L_dir
                + lb * L_phys
                + getattr(cfg, "lambda_coord", 0.1) * coord_loss
                + getattr(cfg, "lambda_kl", 0.05) * kl_loss
                + cfg.lambda_ray_perp * ray_perp_loss
                + cfg.lambda_ray_fwd  * ray_fwd_loss
                + cfg.lambda_ray_len  * ray_len_loss
                + cfg.lambda_ray_cos  * ray_cos_loss
            )

        if train and opt is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        B = img.shape[0]
        meters["loss"] += float(loss.item()) * B
        meters["ang"]  += float(L_ang.item()) * B
        meters["sep"]  += float(sep.mean().item()) * B
        meters["ov"]   += float(L_ov.item()) * B
        meters["ent"]  += float(L_ent.item()) * B
        meters["seg"]  += float(seg_loss.item()) * B
        meters["dice"] += float(dice_term.item()) * B
        meters["N"] += B

        def _to_float(x):
            if isinstance(x, torch.Tensor):
                return float(x.detach().item())
            return float(x)
        
        if any(k in phys_logs for k in ("L_A", "L_B", "L_C")):
            LA = _to_float(phys_logs.get("L_A", 0.0))
            LB = _to_float(phys_logs.get("L_B", 0.0))
            LC = _to_float(phys_logs.get("L_C", 0.0))
            total_phys = LA + LB + LC
        
            meters.setdefault("phys", 0.0); meters["phys"] += total_phys * B
            meters.setdefault("L_A", 0.0);  meters["L_A"]  += LA * B
            meters.setdefault("L_B", 0.0);  meters["L_B"]  += LB * B
            meters.setdefault("L_C", 0.0);  meters["L_C"]  += LC * B
        
        if all(k in phys_logs for k in ("S1", "S2", "dS")):
            S1 = phys_logs["S1"].detach().float()
            S2 = phys_logs["S2"].detach().float()
            dS_abs = phys_logs["dS"].detach().float().abs()
        
            meters.setdefault("S1_mean", 0.0);       meters["S1_mean"]       += float(S1.mean().item()) * B
            meters.setdefault("S2_mean", 0.0);       meters["S2_mean"]       += float(S2.mean().item()) * B
            meters.setdefault("|dS|_mean", 0.0);     meters["|dS|_mean"]     += float(dS_abs.mean().item()) * B
        
            thr = float(getattr(cfg, "brems_delta_near0", 0.05))
            frac_near = float((dS_abs < thr).float().mean().item())
            meters.setdefault("frac_|dS|<thr", 0.0); meters["frac_|dS|<thr"] += frac_near * B

    if is_distributed:
        for k in list(meters.keys()):
            if k != "N":
                tensor = torch.tensor(meters[k], dtype=torch.float32, device=device)
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                meters[k] = float(tensor.item())
        
        n_tensor = torch.tensor(meters["N"], dtype=torch.float32, device=device)
        dist.all_reduce(n_tensor, op=dist.ReduceOp.SUM)
        meters["N"] = int(n_tensor.item())

    N = max(1, meters["N"])
    for k in list(meters.keys()):
        if k!="N":
            meters[k] /= N
    return meters

def _forward_one_view(sample, model, device, cfg, epoch, train):
    img = sample["image"].to(device)
    u_gt = F.normalize(sample["u_gt"].to(device), dim=1)
    out = model(img)
    logits_ab = out["logits_ab"]
    seg_logits = out.get("seg_logits", None)

    logits_ab_adj = logits_ab
    if cfg.enable_seg and (seg_logits is not None) and (cfg.prior_mode in ("soft","hard")):
        prior = prior_bias_from_seg_dense(seg_logits)
        if cfg.prior_mode == "soft" and cfg.prior_alpha > 0:
            logits_ab_adj = logits_ab_adj + cfg.prior_alpha * (prior - 0.5)
        elif cfg.prior_mode == "hard":
            with torch.no_grad():
                B_, _, H_, W_ = prior.shape
                keep_mask = torch.zeros_like(prior, dtype=torch.bool)
                q = float(cfg.prior_topk_percent)
                for b in range(B_):
                    vals = prior[b,0].flatten()
                    k = max(1, int(len(vals) * (q / 100.0)))
                    thr = torch.topk(vals, k).values.min()
                    keep_mask[b,0] = (prior[b,0] >= thr)
            msk2 = keep_mask.expand(-1, 2, -1, -1)
            logits_ab_adj = torch.where(msk2, logits_ab_adj, torch.finfo(logits_ab_adj.dtype).min)

    if cfg.enable_seg and (seg_logits is not None):
        seg_t = sample["seg_mask"].to(device)
        focal_alpha = None
        if cfg.seg_loss_type.lower().startswith("focal") and (cfg.seg_focal_alpha is not None):
            if isinstance(cfg.seg_focal_alpha, (list, tuple)):
                focal_alpha = torch.tensor(cfg.seg_focal_alpha, dtype=seg_logits.dtype, device=device)
            else:
                focal_alpha = torch.tensor(float(cfg.seg_focal_alpha), dtype=seg_logits.dtype, device=device)

        seg_loss, dice_term = compute_seg_loss_dense(
            seg_logits, seg_t,
            loss_type=cfg.seg_loss_type,
            dice_weight=cfg.seg_dice_weight if "dice" in cfg.seg_loss_type.lower() else 0.0,
            focal_gamma=cfg.seg_focal_gamma,
            focal_alpha=focal_alpha,
            num_classes=cfg.num_classes,
            ignore_index=IGNORE_INDEX,
        )
    else:
        seg_loss = img.new_tensor(0.0); dice_term = img.new_tensor(0.0)

    prob = spatial_softmax2d_logits(logits_ab_adj, temperature=cfg.dsnt_temp)
    mx, my = dsnt_expectation(prob)
    ux_ab, uy_ab, sep = unit_from_points(mx[:,0], my[:,0], mx[:,1], my[:,1])
    u_pred = torch.stack([ux_ab, uy_ab], dim=1)
    mx_sw, my_sw = mx, my

    coord_loss = img.new_tensor(0.0)
    kl_loss    = img.new_tensor(0.0)
    ray_perp_loss = img.new_tensor(0.0)
    ray_fwd_loss  = img.new_tensor(0.0)
    ray_len_loss  = img.new_tensor(0.0)
    ray_cos_loss  = img.new_tensor(0.0)

    H, W = prob.shape[-2:]
    need_tail = (cfg.gt_vector_mode != "truth_momentum_pdf")

    if ("head_xy" in sample) and ((not need_tail) or ("tail_xy" in sample)):
        gxA = sample["head_xy"].to(device)[:,0]
        gyA = sample["head_xy"].to(device)[:,1]

        def _in_bounds(x, y):
            return (x >= 0) & (x <= (W - 1)) & (y >= 0) & (y <= (H - 1))

        validA = _in_bounds(gxA, gyA)

        if cfg.gt_vector_mode == "truth_momentum_pdf":
            u_norm = torch.linalg.vector_norm(u_gt, dim=1)
            valid  = validA & (u_norm > 1e-6)

            if valid.any():
                xA, yA = mx_sw[:,0][valid], my_sw[:,0][valid]
                xB, yB = mx_sw[:,1][valid], my_sw[:,1][valid]
                gxA_v, gyA_v = gxA[valid], gyA[valid]

                coord_loss = F.smooth_l1_loss(xA, gxA_v) + F.smooth_l1_loss(yA, gyA_v)

                A_xy = torch.stack([xA, yA], dim=1)
                B_xy = torch.stack([xB, yB], dim=1)
                u_gt_valid = u_gt[valid]
                L_perp, L_fwd, t = ray_distance_loss(A_xy, B_xy, u_gt_valid)
                ray_perp_loss = L_perp
                ray_fwd_loss  = L_fwd
                if cfg.lambda_ray_len > 0:
                    ray_len_loss = ray_length_loss(t, t_min=cfg.ray_t_min)

                if train and (epoch <= getattr(cfg, "ray_warmup_epochs", 0)) and (cfg.lambda_ray_cos > 0):
                    u_pred_valid = u_pred[valid]
                    cos_sim = (F.normalize(u_pred_valid, dim=1) * F.normalize(u_gt_valid, dim=1)).sum(dim=1)
                    ray_cos_loss = (1.0 - cos_sim).mean()

                targ = torch.zeros(valid.sum().item(), 1, H, W, device=prob.device, dtype=prob.dtype)
                sig = float(getattr(cfg, "hm_sigma", 1.5))
                valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
                for j, b in enumerate(valid_idx.tolist()):
                    cxA_j, cyA_j = sample["head_xy"][b]
                    targ[j,0] = _gaussian_2d(H, W, cxA_j, cyA_j, sigma=sig, device=prob.device, dtype=prob.dtype)
                logits_A = logits_ab_adj[valid, 0:1, :, :]
                pA = torch.softmax(logits_A.view(logits_A.size(0), 1, -1), dim=-1).view_as(logits_A).clamp_min(1e-12)
                tA = targ.clamp_min(1e-12)
                kl_loss = (tA * (tA.log() - pA.log())).sum(dim=(1,2,3)).mean()

        else:
            gxB = sample["tail_xy"].to(device)[:,0]
            gyB = sample["tail_xy"].to(device)[:,1]
            validB = _in_bounds(gxB, gyB)
            valid  = validA & validB

            if valid.any():
                xA, yA = mx_sw[:,0][valid], my_sw[:,0][valid]
                xB, yB = mx_sw[:,1][valid], my_sw[:,1][valid]
                gxA_v, gyA_v = gxA[valid], gyA[valid]
                gxB_v, gyB_v = gxB[valid], gyB[valid]

                coord_loss = (
                    F.smooth_l1_loss(xA, gxA_v) + F.smooth_l1_loss(yA, gyA_v) +
                    F.smooth_l1_loss(xB, gxB_v) + F.smooth_l1_loss(yB, gyB_v)
                )

                targ = torch.zeros(valid.sum().item(), 2, H, W, device=prob.device, dtype=prob.dtype)
                sig = float(getattr(cfg, "hm_sigma", 1.5))
                valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
                for j, b in enumerate(valid_idx.tolist()):
                    cxA_j, cyA_j = sample["head_xy"][b]
                    cxB_j, cyB_j = sample["tail_xy"][b]
                    targ[j,0] = _gaussian_2d(H, W, cxA_j, cyA_j, sigma=sig, device=prob.device, dtype=prob.dtype)
                    targ[j,1] = _gaussian_2d(H, W, cxB_j, cyB_j, sigma=sig, device=prob.device, dtype=prob.dtype)

                logits_V = logits_ab_adj[valid]
                p = torch.softmax(logits_V.view(logits_V.size(0), 2, -1), dim=-1).view_as(logits_V).clamp_min(1e-12)
                t = targ.clamp_min(1e-12)
                kl_loss = (t * (t.log() - p.log())).sum(dim=(1,2,3)).mean()

    L_ang = angular_loss(u_pred, u_gt)
    sep_mean = sep.mean()
    L_sep = F.relu(cfg.sep_margin_px - sep).mean() if cfg.sep_margin_px > 0 else img.new_tensor(0.0)
    L_ov  = cfg.overlap_weight * overlap_loss_dense(prob) if cfg.overlap_weight > 0 else img.new_tensor(0.0)
    L_ent = cfg.entropy_weight * entropy_loss_dense(prob) if cfg.entropy_weight > 0 else img.new_tensor(0.0)

    L_dir = img.new_tensor(0.0)
    if cfg.enable_dirreg and ("dir_vec" in out):
        u_dir = F.normalize(out["dir_vec"], dim=1)
        L_dir = cfg.dirreg_weight * angular_loss(u_dir, u_gt)

    L_phys = img.new_tensor(0.0)
    LA = img.new_tensor(0.0); LB = img.new_tensor(0.0); LC = img.new_tensor(0.0)
    S1m = img.new_tensor(0.0); S2m = img.new_tensor(0.0); dSm = img.new_tensor(0.0)
    frac_near = img.new_tensor(0.0)
    if cfg.enable_seg and ("seg_logits" in out):
        L_phys_out, phys_logs = brems_physics_loss(
            mx_sw, my_sw, out["seg_logits"], brems_idx=CLASS_BREMS,
            use_A=cfg.use_brems_A, use_B=cfg.use_brems_B, use_C=cfg.use_brems_C,
            A_margin=cfg.brems_margin,
            B_lambda_min=cfg.brems_lambda_min,
            C_mode=cfg.brems_C_mode,
            C_T=cfg.brems_C_T, C_gamma=cfg.brems_C_gamma, C_K=cfg.brems_C_K,
            weights=cfg.brems_inner_weights,
        )
        L_phys = L_phys_out if torch.is_tensor(L_phys_out) else img.new_tensor(float(L_phys_out))
        if isinstance(phys_logs, dict):
            LA = phys_logs.get("L_A", LA)
            LB = phys_logs.get("L_B", LB)
            LC = phys_logs.get("L_C", LC)
            if "S1" in phys_logs:  S1m = phys_logs["S1"].mean()
            if "S2" in phys_logs:  S2m = phys_logs["S2"].mean()
            if "dS" in phys_logs:  dSm = phys_logs["dS"].detach().float().abs().mean()
            if "frac_near" in phys_logs: frac_near = phys_logs["frac_near"]

    lb = getattr(cfg, "lambda_brems_now", cfg.lambda_brems)
    loss_v = (
        cfg.lambda_ang * L_ang + L_sep + L_ov + L_ent
        + cfg.lambda_seg * seg_loss + L_dir
        + lb * L_phys
        + getattr(cfg, "lambda_coord", 0.1) * coord_loss
        + getattr(cfg, "lambda_kl", 0.05) * kl_loss
        + cfg.lambda_ray_perp * ray_perp_loss
        + cfg.lambda_ray_fwd  * ray_fwd_loss
        + cfg.lambda_ray_len  * ray_len_loss
        + cfg.lambda_ray_cos  * ray_cos_loss
    )

    logs = {
        "loss": loss_v, "ang": L_ang, "sep": sep_mean, "ov": L_ov, "ent": L_ent,
        "seg": seg_loss, "dice": dice_term, "coord": coord_loss, "kl": kl_loss,
        "ray_perp": ray_perp_loss, "ray_fwd": ray_fwd_loss, "ray_len": ray_len_loss, "ray_cos": ray_cos_loss,
        "phys": L_phys, "L_A": LA, "L_B": LB, "L_C": LC,
        "S1_mean": S1m, "S2_mean": S2m, "|dS|_mean": dSm, "frac_|dS|<thr": frac_near,
        "mx_sw": mx_sw,
        "my_sw": my_sw 
    }
    return logs, prob

def run_epoch_multi(models, loader, opts, device, cfg, epoch, total_epochs, train, is_distributed=False):
    views = ["U", "V", "Z"]

    for m in models.values():
        m.train(mode=train)

    if getattr(cfg, "overlap_weight0", None) is None:
        cfg.overlap_weight0 = cfg.overlap_weight
    if getattr(cfg, "entropy_weight0", None) is None:
        cfg.entropy_weight0 = cfg.entropy_weight
    if getattr(cfg, "sep_margin_px0", None) is None:
        cfg.sep_margin_px0 = cfg.sep_margin_px

    t = float(epoch) / max(1, total_epochs - 1)
    cfg.dsnt_temp      = (1.0 - t) * cfg.dsnt_temp_hi + t * cfg.dsnt_temp_lo
    cfg.overlap_weight = cfg.overlap_weight0 * (1.0 - t)
    cfg.entropy_weight = cfg.entropy_weight0 * (1.0 - t)
    cfg.sep_margin_px  = cfg.sep_margin_px0  * (1.0 - t)

    T = max(1, getattr(cfg, "brems_warmup_epochs", 0))
    ramp = min(1.0, float(epoch) / float(T)) if T > 0 else 1.0
    cfg.lambda_brems_now = ramp * cfg.lambda_brems

    def _base():
        return {
            "loss":0.0, "ang":0.0, "sep":0.0, "ov":0.0, "ent":0.0, "seg":0.0, "dice":0.0,
            "coord":0.0, "kl":0.0, "ray_perp":0.0, "ray_fwd":0.0, "ray_len":0.0, "ray_cos":0.0,
            "phys":0.0, "L_A":0.0, "L_B":0.0, "L_C":0.0,
            "S1_mean":0.0, "S2_mean":0.0, "|dS|_mean":0.0, "frac_|dS|<thr":0.0,
            "x_var":0.0, "x_gt":0.0, "tick_sign":0.0, "tick_val":0.0, "N":0
        }
    meters = _base()
    per_view_meters = {v: _base() for v in views}

    for batch in loader:
        if batch is None:
            continue

        B = batch[views[0]]["image"].shape[0]
        if train and isinstance(opts, dict):
            for v in views:
                o = opts.get(v)
                if o is not None:
                    o.zero_grad(set_to_none=True)

        all_mx_A, all_mx_B = [], []
        gt_xA_all, gt_xB_all = None, None

        per_view_losses = []

        per_view_logs = {}
        for v in views:
            logs, prob = _forward_one_view(batch[v], models[v], device, cfg, epoch, train)
            per_view_logs[v] = logs
            per_view_losses.append(logs["loss"])

            pv = per_view_meters[v]
            for k in (
                "loss","ang","sep","ov","ent","seg","dice","coord","kl",
                "ray_perp","ray_fwd","ray_len","ray_cos","phys","L_A","L_B","L_C",
                "S1_mean","S2_mean","|dS|_mean","frac_|dS|<thr"
            ):
                val = logs[k]
                pv[k] += (float(val.detach().cpu()) if torch.is_tensor(val) else float(val)) * B
            pv["N"] += B

            mx_sw = logs["mx_sw"]
            all_mx_A.append(mx_sw[:,0])
            all_mx_B.append(mx_sw[:,1])

            smp = batch[v]
            if gt_xA_all is None and ("head_xy" in smp):
                gt_xA_all = smp["head_xy"].to(device)[:,0]
            if gt_xB_all is None and ("tail_xy" in smp):
                gt_xB_all = smp["tail_xy"].to(device)[:,0]
        use_B = (cfg.gt_vector_mode != "truth_momentum_pdf")

        mx_A_stack = torch.stack(all_mx_A, dim=1)
        L_x_var = mx_A_stack.var(dim=1, unbiased=False).mean()

        if use_B:
            mx_B_stack = torch.stack(all_mx_B, dim=1)
            L_x_var = L_x_var + mx_B_stack.var(dim=1, unbiased=False).mean()

        L_x_gt = torch.zeros((), device=device)
        if gt_xA_all is not None:
            for v_idx in range(len(views)):
                L_x_gt = L_x_gt + F.smooth_l1_loss(all_mx_A[v_idx], gt_xA_all)

        if use_B and gt_xB_all is not None:
            for v_idx in range(len(views)):
                L_x_gt = L_x_gt + F.smooth_l1_loss(all_mx_B[v_idx], gt_xB_all)

        if gt_xA_all is not None or gt_xB_all is not None:
            L_x_gt = L_x_gt / float(len(views) * ((gt_xA_all is not None) + (gt_xB_all is not None)))

        scale = float(prob.shape[-1])
        L_x_var = L_x_var / (scale * scale)
        L_x_gt  = L_x_gt  / scale

        L_tick_sign = torch.zeros((), device=device)
        L_tick_val = torch.zeros((), device=device)
        
        if cfg.lambda_tick_sign > 0 or cfg.lambda_tick_val > 0:
            mx_sw_U = per_view_logs['U']['mx_sw']
            mx_sw_V = per_view_logs['V']['mx_sw']
            mx_sw_Z = per_view_logs['Z']['mx_sw']
            
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
            
            if cfg.lambda_tick_sign > 0:
                L_sign_UV = F.relu(-dU_n * dV_n)
                L_sign_UZ = F.relu(-dU_n * dZ_n)
                L_sign_VZ = F.relu(-dV_n * dZ_n)
                L_tick_sign = (L_sign_UV + L_sign_UZ + L_sign_VZ).mean()
            
            if cfg.lambda_tick_val > 0:
                d_stack = torch.stack([dU_n, dV_n, dZ_n], dim=-1)
                
                mu = d_stack.mean(dim=-1, keepdim=True)
                
                resid = d_stack - mu
                
                tau = float(cfg.tick_tau) / DT_SCALE
                resid_abs = resid.abs()
                resid_eff = torch.sign(resid) * F.relu(resid_abs - tau)
                
                L_tick_val = F.smooth_l1_loss(resid_eff, torch.zeros_like(resid_eff))

        avg_per_view = sum(per_view_losses) / 3.0
        total_loss = (
            avg_per_view + 
            cfg.lambda_x_constraint * L_x_var + 
            cfg.lambda_x_gt * L_x_gt +
            cfg.lambda_tick_sign * L_tick_sign +
            cfg.lambda_tick_val * L_tick_val
        )

        if train and isinstance(opts, dict):
            total_loss.backward()
            for v in views:
                o = opts.get(v)
                if o is not None:
                    try:
                        torch.nn.utils.clip_grad_norm_(models[v].parameters(), 5.0)
                    except Exception:
                        pass
                    o.step()

        meters["loss"] += float(total_loss.detach().cpu()) * B
        meters["x_var"] += float(L_x_var.detach().cpu()) * B
        meters["x_gt"]  += float(L_x_gt.detach().cpu()) * B
        meters["tick_sign"] += float(L_tick_sign.detach().cpu()) * B
        meters["tick_val"] += float(L_tick_val.detach().cpu()) * B
        meters["N"] += B

        if getattr(cfg, 'use_consistency_checks', False):
            ux_img_orig = {}
            uy_img_orig = {}
            u_gt_dict = {}
            
            for v in views:
                mx = per_view_logs[v]["mx_sw"]

                img_shape = batch[v]["image"].shape
                prob_shape = prob.shape  
                

                my = per_view_logs[v].get("my_sw")
                if my is None:
                    
                    raise RuntimeError("my_sw not in logs - add it in _forward_one_view")
                
                dx = mx[:, 1] - mx[:, 0]
                dy = my[:, 1] - my[:, 0]
                norm = torch.sqrt(dx*dx + dy*dy + 1e-9)
                
                ux_img_orig[v] = dx / norm
                uy_img_orig[v] = dy / norm
                
                u_gt_dict[v] = batch[v]["u_gt"].to(device)
            
            consistency_result = apply_consistency_pipeline(
                ux_img_orig, uy_img_orig, cfg
            )
            
            val_metrics = evaluate_consistency_corrections(
                ux_img_original=ux_img_orig,
                uy_img_original=uy_img_orig,
                ux_img_corrected=consistency_result['ux_corrected'],
                uy_img_corrected=consistency_result['uy_corrected'],
                u_gt=u_gt_dict,
                consistency_info=consistency_result
            )
            
            meters['consistency_metrics'] = val_metrics
            
            meters['consistency_acc_before'] = val_metrics['aggregate']['acc_before']
            meters['consistency_acc_after'] = val_metrics['aggregate']['acc_after']
            meters['consistency_acc_gain'] = val_metrics['aggregate']['acc_gain']
            meters['consistency_flip_precision'] = val_metrics['aggregate']['flip_precision']

            
            for v in ['U', 'V', 'Z']:
                vm = val_metrics['per_view'][v]
                meters[f'consistency_{v}_acc_before'] = vm['acc_before']
                meters[f'consistency_{v}_acc_after'] = vm['acc_after']
                meters[f'consistency_{v}_flip_prec'] = vm['flip_precision']
                meters[f'consistency_{v}_flip_recall'] = vm['flip_recall']
            
            
                

    if is_distributed:
        for k in ("loss","x_var","x_gt","tick_sign","tick_val"):
            tensor = torch.tensor(meters[k], dtype=torch.float32, device=device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            meters[k] = float(tensor.item())
        
        n_tensor = torch.tensor(meters["N"], dtype=torch.float32, device=device)
        dist.all_reduce(n_tensor, op=dist.ReduceOp.SUM)
        meters["N"] = int(n_tensor.item())

        for v in views:
            pv = per_view_meters[v]
            for k in list(pv.keys()):
                if k != "N":
                    t = torch.tensor(pv[k], dtype=torch.float32, device=device)
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    pv[k] = float(t.item())
            
            n_t = torch.tensor(pv["N"], dtype=torch.float32, device=device)
            dist.all_reduce(n_t, op=dist.ReduceOp.SUM)
            pv["N"] = int(n_t.item())

    N = max(1, meters["N"])
    for k in ("loss","x_var","x_gt","tick_sign","tick_val"):
        meters[k] /= N

    for v in views:
        Nv = max(1, per_view_meters[v]["N"])
        for k in list(per_view_meters[v].keys()):
            if k != "N":
                per_view_meters[v][k] /= Nv

    for k in ("ang","sep","ov","ent","seg","dice","coord","kl",
              "ray_perp","ray_fwd","ray_len","ray_cos",
              "phys","L_A","L_B","L_C","S1_mean","S2_mean","|dS|_mean","frac_|dS|<thr"):
        meters[k] = (per_view_meters["U"][k] + per_view_meters["V"][k] + per_view_meters["Z"][k]) / 3.0

    meters["per_view"] = per_view_meters
    return meters
