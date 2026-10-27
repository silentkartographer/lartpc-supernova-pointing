# Authored by Hilary Utaegbulam

"""Physics coordinate transforms, DSNT math, angular/ray losses."""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .constants import *

def _n_perp_from_theta_deg(theta_deg: float) -> tuple[float, float]:
    th = math.radians(theta_deg)
    # wire direction e_wire = (cosθ, sinθ) in (y,z)
    # perpendicular that increases wire index: n_perp = (-sinθ, +cosθ)
    return (-math.sin(th), math.cos(th))


def truth_momentum_arrow_from_pdf_plane(
    vpx: float, vpy: float, vpz: float,
    *, plane: str, drift_sign: int,
    mom_arrow_pix_len: float = 40.0
) -> tuple[float, float] | None:
    """
    Generalized 3D→2D mapping for any plane {U,V,Z}.
    - dticks/ds = drift_sign * u_x / (VBULK * T_TICK)
    - dwires/ds = drift_sign * ((u_y,u_z)·n_perp(θ_plane)) / PITCH_plane
    Returns (dy_pixels, dx_pixels) normalized to mom_arrow_pix_len, or None if invalid.
    """
    p = np.array([float(vpx), float(vpy), float(vpz)], dtype=float)
    n = float(np.linalg.norm(p))
    if not np.isfinite(n) or n == 0.0:
        return None
    ux, uy, uz = (p / n)

    dticks_ds = (drift_sign * ux) / (VBULK * T_TICK)

    plane = str(plane).upper()
    if plane == 'Z':
        pitch = PITCH_Z
        theta = THETA_DEG['Z']
    elif plane == 'U':
        pitch = PITCH_U
        theta = THETA_DEG['U']
    elif plane == 'V':
        pitch = PITCH_V
        theta = THETA_DEG['V']
    else:
        return None

    ny, nz = _n_perp_from_theta_deg(theta)
    # The wire-index axis lies in the readout plane, so it carries no drift sign.
    dwires_ds = (uy * ny + uz * nz) / pitch 

    dy, dx = float(dwires_ds), float(dticks_ds)
    mag = float(np.hypot(dy, dx))
    if mag == 0.0 or not np.isfinite(mag):
        return None
    uy_img, ux_img = dy / mag, dx / mag
    return (uy_img * mom_arrow_pix_len, ux_img * mom_arrow_pix_len)


def spatial_softmax2d_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    B, C, H, W = logits.shape
    T = max(float(temperature), 1e-8)
    x = logits.view(B*C, -1) / T
    x = x - x.max(dim=-1, keepdim=True).values
    p = F.softmax(x, dim=-1)
    return p.view(B, C, H, W)

def dsnt_expectation(prob: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # return (mu_x, mu_y) per channel
    B, C, H, W = prob.shape
    xs = torch.linspace(0, W-1, W, device=prob.device).view(1,1,1,W)
    ys = torch.linspace(0, H-1, H, device=prob.device).view(1,1,H,1)
    mx = (prob * xs).sum(dim=(2,3))
    my = (prob * ys).sum(dim=(2,3))
    return mx, my  # [B,C], [B,C]

def unit_from_points(mxA, myA, mxB, myB, eps=1e-9):
    # returns unit vector (ux,uy) and length (for separation loss)
    vx = mxB - mxA
    vy = myB - myA
    ln = torch.sqrt(vx*vx + vy*vy + eps)
    ux = vx / ln
    uy = vy / ln
    return ux, uy, ln

def angular_loss(u_pred: torch.Tensor, u_true: torch.Tensor, eps=1e-9) -> torch.Tensor:
    # u_* shape [B,2], each supposed to be unit vectors
    u_pred = F.normalize(u_pred, dim=1, eps=eps)
    u_true = F.normalize(u_true, dim=1, eps=eps)
    cos = (u_pred * u_true).sum(dim=1).clamp(-1.0, 1.0)
    return (1.0 - cos).mean()


def ray_distance_loss(A_xy, B_xy, u_dir, eps=1e-8):
    """
    Ray-based supervision for point B when using momentum ground truth.
    
    A_xy: [B,2] predicted A points (x,y order)
    B_xy: [B,2] predicted B points (x,y order)
    u_dir: [B,2] ground truth direction vectors (will be normalized)
    
    Returns:
        L_perp: mean perpendicular distance from B to ray
        L_fwd: mean forwardness hinge loss
        t: axial distances along ray (for optional length loss)
    """
    u = F.normalize(u_dir, dim=1, eps=eps)  # [B,2] unit vectors
    r = B_xy - A_xy                         # [B,2] vector from A to B
    
    # Perpendicular distance: |r × u| in 2D
    cross_mag = torch.abs(r[:, 0] * u[:, 1] - r[:, 1] * u[:, 0])
    
    # Axial coordinate along ray: t = r · u
    t = (r * u).sum(dim=1)
    
    # Forwardness hinge: penalize if B is behind A (t < 0)
    fwd_hinge = F.relu(-t)
    
    return cross_mag.mean(), fwd_hinge.mean(), t

def ray_length_loss(t, t_min=5.0, eps=1e-8):
    """
    Prevent B from collapsing onto A.

    t: [B] axial distances along ray
    t_min: minimum ddistance in pixel units
    
    Returns:
        Mean softplus penalty for distances less than t_min
    """
    return F.softplus(t_min - t).mean()

def _gaussian_2d(H, W, cx, cy, sigma=1.5, device=None, dtype=None):
    ys = torch.arange(H, device=device, dtype=dtype).view(H, 1)
    xs = torch.arange(W, device=device, dtype=dtype).view(1, W)
    g = torch.exp(-0.5 * ((xs - cx)**2 + (ys - cy)**2) / (sigma * sigma))
    g = g / g.sum().clamp_min(1e-12)
    return g
