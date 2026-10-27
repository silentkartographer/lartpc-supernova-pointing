# Authored by Hilary Utaegbulam

"""Logging formatters, CSV writers, and metric computation utilities."""
from __future__ import annotations
import os
import csv
import math
import os, csv

def _fmt(logs: dict) -> str:
    """
    Formats metrics. "ang" is shown in degrees (since logs['ang'] is 1 - cosθ).
    """
    parts = []

    def _pick(k, fmt="{:.4f}"):
        return (k in logs) and (logs[k] is not None)

    if _pick("loss"): parts.append(f"loss {float(logs['loss']):.4f}")

    if _pick("ang"):
        ang_loss_val = float(logs["ang"])
        cos_theta = max(-1.0, min(1.0, 1.0 - ang_loss_val))
        ang_deg = math.degrees(math.acos(cos_theta))
        parts.append(f"ang {ang_deg:.2f}deg")

    if _pick("sep"):  parts.append(f"sep {float(logs['sep']):.2f}")
    if _pick("ov"):   parts.append(f"ov {float(logs['ov']):.4f}")
    if _pick("ent"):  parts.append(f"ent {float(logs['ent']):.4f}")
    if _pick("seg"):  parts.append(f"seg {float(logs['seg']):.4f}")
    if _pick("dice"): parts.append(f"dice {float(logs['dice']):.4f}")

    have_phys = any(k in logs for k in ("phys", "L_A", "L_B", "L_C", "S1_mean", "S2_mean", "|dS|_mean", "frac_|dS|<thr"))
    if have_phys:
        phys_bits = []

        # total physics loss (sum of enabled A/B/C, averaged over the epoch)
        if _pick("phys"): phys_bits.append(f"phys {float(logs['phys']):.4f}")

        
        abcs = []
        if _pick("L_A"): abcs.append(f"A {float(logs['L_A']):.4f}")
        if _pick("L_B"): abcs.append(f"B {float(logs['L_B']):.4f}")
        if _pick("L_C"): abcs.append(f"C {float(logs['L_C']):.4f}")
        if abcs: phys_bits.append("(" + " ".join(abcs) + ")")

        # summary stats from the brems metrics
        # means of S1, S2, |ΔS| and the fraction of near-zero |ΔS|
        if _pick("S1_mean"):        phys_bits.append(f"S1 {float(logs['S1_mean']):.3f}")
        if _pick("S2_mean"):        phys_bits.append(f"S2 {float(logs['S2_mean']):.3f}")
        if _pick("|dS|_mean"):      phys_bits.append(f"|dS| {float(logs['|dS|_mean']):.3f}")
        if _pick("frac_|dS|<thr"):  phys_bits.append(f"near {float(logs['frac_|dS|<thr']):.3f}")

        if phys_bits:
            parts.append("phys{" + " ".join(phys_bits) + "}")

    return " | ".join(parts)


def _fmt_multi(logs: dict) -> str:
    """
    Format multi-view metrics for logging.
    """
    parts = []
    
    # Total loss
    if "loss" in logs:
        parts.append(f"loss {float(logs['loss']):.4f}")
    
    # Angular error (convert from 1-cos to degrees)
    if "ang" in logs:
        ang_loss_val = float(logs["ang"])
        cos_theta = max(-1.0, min(1.0, 1.0 - ang_loss_val))
        ang_deg = math.degrees(math.acos(cos_theta))
        parts.append(f"ang {ang_deg:.2f}°")
    
    # X-coordinate constraints
    if "x_var" in logs:
        parts.append(f"x_var {float(logs['x_var']):.4f}")
    if "x_gt" in logs:
        parts.append(f"x_gt {float(logs['x_gt']):.4f}")
    
    # Tick consistency
    if "tick_sign" in logs:
        parts.append(f"tick_sign {float(logs['tick_sign']):.4f}")
    if "tick_val" in logs:
        parts.append(f"tick_val {float(logs['tick_val']):.4f}")
    
    # Other losses
    if "sep" in logs:
        parts.append(f"sep {float(logs['sep']):.2f}")
    if "seg" in logs:
        parts.append(f"seg {float(logs['seg']):.4f}")
    if "dice" in logs:
        parts.append(f"dice {float(logs['dice']):.4f}")
    
    return " | ".join(parts)


def _scalar(d, k, default=0.0):
    v = d.get(k, default)
    try:
        return float(v)
    except Exception:
        return default

def _fmt_perview(tag, logs):
    parts = [f"[{tag}]"]

    # overall loss
    if "loss" in logs:
        parts.append(f"loss {_scalar(logs,'loss'):.4f}")

    # angle (display as degrees)
    if "ang" in logs:
        ang_loss_val = _scalar(logs, "ang")
        cos_theta = max(-1.0, min(1.0, 1.0 - ang_loss_val))
        try:
            ang_deg = math.degrees(math.acos(cos_theta))
            parts.append(f"ang {ang_deg:.2f}deg")
        except Exception:
            parts.append(f"ang_raw {ang_loss_val:.4f}")

    for key, fmt in (("sep",".2f"), ("ov",".4f"), ("ent",".4f"),
                     ("seg",".4f"), ("dice",".4f")):
        if key in logs:
            parts.append(f"{key} {_scalar(logs,key):{fmt}}")

    # physics block
    phys_bits = []
    if "phys" in logs:
        phys_bits.append(f"sum {_scalar(logs,'phys'):.4f}")
    abcs = []
    if "L_A" in logs: abcs.append(f"A {_scalar(logs,'L_A'):.4f}")
    if "L_B" in logs: abcs.append(f"B {_scalar(logs,'L_B'):.4f}")
    if "L_C" in logs: abcs.append(f"C {_scalar(logs,'L_C'):.4f}")
    if abcs: phys_bits.append("(" + " ".join(abcs) + ")")
    for k,lab,fmt in (("S1_mean","S1",".3f"),
                      ("S2_mean","S2",".3f"),
                      ("|dS|_mean","|dS|",".3f"),
                      ("frac_|dS|<thr","near",".3f")):
        if k in logs: phys_bits.append(f"{lab} {_scalar(logs,k):{fmt}}")
    if phys_bits:
        parts.append("phys{" + " ".join(phys_bits) + "}")

    # ray block 
    ray_bits = []
    if "ray_perp" in logs: ray_bits.append(f"perp {_scalar(logs,'ray_perp'):.4f}")
    if "ray_fwd"  in logs: ray_bits.append(f"fwd {_scalar(logs,'ray_fwd'):.4f}")
    if "ray_len"  in logs: ray_bits.append(f"len {_scalar(logs,'ray_len'):.4f}")
    if "ray_cos"  in logs: ray_bits.append(f"cos {_scalar(logs,'ray_cos'):.4f}")
    if ray_bits:
        parts.append("ray{" + " ".join(ray_bits) + "}")

    # x-constraints 
    xb = []
    if "x_var" in logs: xb.append(f"var {_scalar(logs,'x_var'):.4f}")
    if "x_gt"  in logs: xb.append(f"gt {_scalar(logs,'x_gt'):.4f}")
    if xb:
        parts.append("x{" + " ".join(xb) + "}")

    return " | ".join(parts)

def log_grad_norms(model, label=""):
    total = 0.0
    cnt = 0
    for _, p in model.named_parameters():
        if p.grad is not None:
            g2 = float(p.grad.detach().data.norm(2).cpu())
            total += g2
            cnt += 1
    if cnt > 0:
        print(f"[grad] {label} mean_grad_l2={total/cnt:.4f} (from {cnt} tensors)")

def _csv_log_epoch(csv_path, epoch, phase, logs):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = [
        "epoch","phase","view","loss","ang","coord","kl",
        "ray_perp","ray_fwd","ray_len","ray_cos",
        "sep","ov","ent","seg","dice",
        "phys","L_A","L_B","L_C","S1_mean","S2_mean","|dS|_mean","frac_|dS|<thr",
        "x_var","x_gt"
    ]
    write_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        def _row(view_tag, d):
            r = {k: None for k in fieldnames}
            r.update({"epoch": epoch, "phase": phase, "view": view_tag})
            for k in fieldnames:
                if k in d:
                    try:
                        r[k] = float(d[k])
                    except Exception:
                        pass
            return r
        # global
        w.writerow(_row("ALL", logs))
        # per-view
        if isinstance(logs.get("per_view", None), dict):
            for v in ("U","V","Z"):
                if v in logs["per_view"]:
                    w.writerow(_row(v, logs["per_view"][v]))


def _brems_lambda_at_epoch(cfg, epoch: int) -> float:
    """
    Linear warmup for brems physics loss.
    Goes from 0 to cfg.lambda_brems over cfg.brems_warmup_epochs.
    """
    if cfg.brems_warmup_epochs <= 0:
        return float(cfg.lambda_brems)
    t = min(1.0, max(0.0, epoch / float(cfg.brems_warmup_epochs)))
    return float(cfg.lambda_brems) * t
