# Authored by Hilary Utaegbulam

"""Geometric transforms and augmentations.
"""
from __future__ import annotations
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple

def _make_affine_grid(H: int, W: int,
                      degrees: float, scale_range: Tuple[float,float],
                      translate: Tuple[int,int],
                      device):
    """
    Returns:
      grid: (1,H,W,2) sampling grid for F.grid_sample (dst -> src)
      affine: dict with forward-mapping params for point warping:
              {
                'R_np': 2x2 numpy array (forward rotation*scale),
                'tx': float, 'ty': float,          # pixel translations (same units as x,y)
                'cx': float, 'cy': float           # image center in pixels
              }
    """
    angle = random.uniform(-degrees, degrees)
    sc = random.uniform(scale_range[0], scale_range[1])
    tx = random.uniform(-translate[0], translate[0])
    ty = random.uniform(-translate[1], translate[1])

    theta = math.radians(angle) 
    c, s = math.cos(theta), math.sin(theta)

    R = torch.tensor([[sc*c, -sc*s],
                      [sc*s,  sc*c]], dtype=torch.float32, device=device)
    
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0

    Rinv = torch.inverse(R)
    yy, xx = torch.meshgrid(torch.arange(H, device=device),
                            torch.arange(W, device=device), indexing='ij')
    
    grid = torch.stack([xx, yy], dim=-1).float() # (H,W,2)
    grid_centered = grid - torch.tensor([cx, cy], device=device)
    src = (grid_centered.reshape(-1,2) @ Rinv.T).reshape(H,W,2)
    src = src + torch.tensor([cx - tx, cy - ty], device=device)

    gx = (2.0 * src[...,0] / max(W-1,1)) - 1.0
    gy = (2.0 * src[...,1] / max(H-1,1)) - 1.0
    grid_out = torch.stack([gx, gy], dim=-1).unsqueeze(0)  # (1,H,W,2)

    affine = {
        'R_np': np.array([[sc*c, -sc*s],
                          [sc*s,  sc*c]], dtype=np.float64),  # forward
        'tx': float(tx), 'ty': float(ty),
        'cx': float(cx), 'cy': float(cy),
    }
    return grid_out, affine

def _warp_point_xy_np(x: float, y: float, R_np: np.ndarray, tx: float, ty: float, cx: float, cy: float) -> Tuple[float,float]:
    """
    Forward-map a source point (x_s, y_s) -> destination (x_d, y_d) used as:
      [x_d, y_d]^T = [cx, cy]^T + R * ( [x_s, y_s]^T - [cx, cy]^T + [tx, ty]^T )
    This should match the grid generation used for F.grid_sample.
    """
    v = np.array([x, y], dtype=np.float64) - np.array([cx, cy], dtype=np.float64)
    v = R_np @ (v + np.array([tx, ty], dtype=np.float64))
    v = v + np.array([cx, cy], dtype=np.float64)
    return float(v[0]), float(v[1])

def _warp_dir_xy_np(ux: float, uy: float, R_np: np.ndarray) -> Tuple[float,float]:
    """Rotate/scale a direction by R, then renormalize to unit length."""
    u = R_np @ np.array([ux, uy], dtype=np.float64)
    n = float(np.hypot(u[0], u[1])) or 1.0
    return float(u[0]/n), float(u[1]/n)

# AUGMENTATIONS FOR POINTING

def aug_dead_stripes(
    img: np.ndarray,
    p_col: float = 0.2,
    p_row: float = 0.2,
    max_cols: int = 6,
    max_rows: int = 6,
    threshold: float = 1e-6
) -> np.ndarray:
    """
    Randomly zero out entire columns (dead wires) and rows (bad ticks).
    Only affects columns/rows that contain actual data since most of the data is sparse(mean > threshold).
    """
    H, W = img.shape
    img_out = img.copy()
    
    # Dead columns (wires)
    if random.random() < p_col:
        col_means = np.abs(img).mean(axis=0)
        data_cols = np.where(col_means > threshold)[0]
        if len(data_cols) > 0:
            n_cols = random.randint(1, min(max_cols, len(data_cols)))
            dead_cols = np.random.choice(data_cols, size=n_cols, replace=False)
            img_out[:, dead_cols] = 0.0
    
    # Dead rows (time ticks)
    if random.random() < p_row:
        row_means = np.abs(img).mean(axis=1)
        data_rows = np.where(row_means > threshold)[0]
        if len(data_rows) > 0:
            n_rows = random.randint(1, min(max_rows, len(data_rows)))
            dead_rows = np.random.choice(data_rows, size=n_rows, replace=False)
            img_out[dead_rows, :] = 0.0
    
    return img_out


def aug_random_erasing(
    img: np.ndarray,
    p: float = 0.2,
    area: Tuple[float, float] = (0.005, 0.03),
    aspect_ratio: Tuple[float, float] = (0.3, 3.3),
    threshold: float = 1e-6,
    max_attempts: int = 50
) -> np.ndarray:
    """
    Random erasing that targets actual data regions.
    
    max_attempts is the maximum attempts to find a patch overlapping data so the process doesnt continue forever.
    """
    if random.random() > p:
        return img
    
    H, W = img.shape
    img_out = img.copy()
    
    # Find data pixels
    data_mask = np.abs(img) > threshold
    if not data_mask.any():
        return img_out
    
    # Try to find a patch that overlaps with data
    for _ in range(max_attempts):
        target_area = random.uniform(area[0], area[1]) * H * W
        aspect = random.uniform(aspect_ratio[0], aspect_ratio[1])
        
        h = int(round(np.sqrt(target_area * aspect)))
        w = int(round(np.sqrt(target_area / aspect)))
        
        if h >= H or w >= W:
            continue
        
        y = random.randint(0, H - h)
        x = random.randint(0, W - w)
        
        patch_mask = data_mask[y:y+h, x:x+w]
        if patch_mask.any():
            img_out[y:y+h, x:x+w] = 0.0
            break
    
    return img_out


def aug_bragg_jitter(
    img: np.ndarray,
    p: float = 0.3,
    percentile: float = 95.0,
    scale_range: Tuple[float, float] = (0.8, 1.2),
    threshold: float = 1e-6
) -> np.ndarray:
    """
    Randomly scale high-ADC pixels (Bragg peaks).
    """
    if random.random() > p:
        return img
    
    img_out = img.astype(np.float32)
    
    # Find high-value pixels
    nonzero = np.abs(img_out) > threshold
    if not nonzero.any():
        return img_out
    
    thr = np.percentile(np.abs(img_out[nonzero]), percentile)
    high_mask = np.abs(img_out) >= thr
    
    if high_mask.any():
        scale = random.uniform(scale_range[0], scale_range[1])
        img_out[high_mask] *= scale
    
    return img_out


def aug_gain_offset(
    img: np.ndarray,
    gain_range: Tuple[float, float] = (0.9, 1.1),
    offset_range: Tuple[float, float] = (-0.02, 0.02)
) -> np.ndarray:
    """
    Apply random gain and offset: img_out = gain * img + offset
    """
    gain = random.uniform(gain_range[0], gain_range[1])
    offset = random.uniform(offset_range[0], offset_range[1])
    return gain * img + offset


def aug_gaussian_noise(
    img: np.ndarray,
    sigma: float = 0.02,
    clip_min: float = 0.0,
    clip_max: Optional[float] = None
) -> np.ndarray:
    """
    Add Gaussian noise 
    """
    noise = np.random.normal(0, sigma, img.shape)
    img_out = img + noise
    
    if clip_max is None:
        clip_max = np.abs(img).max()
    
    return np.clip(img_out, clip_min, clip_max)
