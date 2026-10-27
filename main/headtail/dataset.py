# Authored by Hilary Utaegbulam

"""Dataset classes, collation functions, and data loading utilities."""
from __future__ import annotations
import os
import glob
import json
import hashlib
import random
import copy
from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Tuple, Optional, Dict, List

from .constants import (
    Config, PARTITION_SEED, PARTITION_FRAC, TRAIN_FRACTION, IGNORE_INDEX,
    PDG_ELECTRON_POS, PDG_ELECTRON_NEG, PDG_MUON_POS, PDG_MUON_NEG,
    PDG_MICHEL, PDG_BREMS,
    CLASS_BG, CLASS_ELECTRON, CLASS_MUON, CLASS_MICHEL, CLASS_BREMS,
)
from .physics_utils import truth_momentum_arrow_from_pdf_plane
from .augmentations import (
    _make_affine_grid, _warp_point_xy_np, _warp_dir_xy_np,
    aug_dead_stripes, aug_random_erasing, aug_bragg_jitter,
    aug_gain_offset, aug_gaussian_noise,
)
import traceback

def _passes_partition(filename: str, *, role: str, frac: float, seed: int) -> bool:
    h = hashlib.md5(f"{role}\n{filename}\n{seed}".encode('utf-8')).hexdigest()
    v = int(h[:8], 16) / 0xFFFFFFFF
    return v < float(frac)

def _map_pixid_to_class(pixid: np.ndarray, enable_brems: bool, **kwargs) -> np.ndarray:
    """Maps pixid labels to segmentation classes (0-4). Uses new label definitions."""
    cls = np.zeros_like(pixid, dtype=np.int64)  # Default: CLASS_BG

    cls[(pixid == PDG_ELECTRON_POS) | (pixid == PDG_ELECTRON_NEG)] = CLASS_ELECTRON
    cls[(pixid == PDG_MUON_POS) | (pixid == PDG_MUON_NEG)] = CLASS_MUON

    michel_mask = (np.abs(pixid) == PDG_MICHEL)
    cls[michel_mask] = CLASS_MICHEL

    if enable_brems:
        brems_mask = (np.abs(pixid) >= PDG_BREMS)
        cls[brems_mask & ~michel_mask] = CLASS_BREMS

    return cls



def _pad_to_multiple_mask(mask: torch.Tensor, multiple: int = 32, fill: int = IGNORE_INDEX) -> torch.Tensor:
    # mask: [H,W] Long
    h, w = mask.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return mask
    return F.pad(mask.unsqueeze(0), (0, pad_w, 0, pad_h), 'constant', fill).squeeze(0)

def _pixel_energy_weight(sample) -> float:
    adc = sample["image"].squeeze(0).numpy()
    if "seg_mask" in sample:
        seg = sample["seg_mask"].numpy()
        mask = (seg == CLASS_MICHEL) | (seg == CLASS_BREMS)
        if mask.any():
            return float(adc[mask].sum())
    return float(adc.sum())

        

class NpzEventDataset(Dataset):
    def __init__(self,
                preprocessed_root: str,
                split: str,
                view: str,
                cfg: Config,
                use_raw_bg: bool = False,
                use_partition: bool = True,
                partition_role: str = 'vector',
                partition_frac: float = PARTITION_FRAC,
                partition_seed: int = PARTITION_SEED,
                train_fraction: float = TRAIN_FRACTION,
                augment: bool = False,
                aug_degrees: float = 7.5,
                aug_scale: Tuple[float,float] = (0.95, 1.05),
                aug_translate: Tuple[int,int] = (4, 4),
                zscore_norm: bool = False,
                enable_seg_labels: bool = True,
                enable_brems_label: bool = True,
                gt_vector_mode: str = 'truth_momentum_pdf',
                resize_to: Optional[Tuple[int, int]] = None,
                use_truth_vertex: bool = False,
                vertex_tick_mode: str = 'relative',
                t0_offset_ticks: int = 0,
                wire_index_base: int = 0,
                ):
        super().__init__()
        self.preprocessed_root = preprocessed_root
        self.split = split
        self.view = view.upper()
        assert self.view in ('U','V','Z'), f"view must be U, V, or Z, got {view}"
        self.use_raw_bg = use_raw_bg
        self.use_partition = use_partition
        self.partition_role = partition_role
        self.partition_frac = float(partition_frac)
        self.partition_seed = int(partition_seed)
        self.train_fraction = float(train_fraction)
        self.augment = bool(augment)
        self.aug_degrees = float(aug_degrees)
        self.aug_scale = aug_scale
        self.aug_translate = aug_translate
        self.zscore_norm = bool(zscore_norm)
        self.enable_seg_labels = bool(enable_seg_labels)
        self.enable_brems_label = bool(enable_brems_label)
        self.gt_vector_mode = gt_vector_mode.lower()
        self.resize_to = resize_to

        self.aug_dead_stripes = getattr(cfg, 'aug_dead_stripes', True)
        self.aug_dead_p_col = getattr(cfg, 'aug_dead_p_col', 0.2)
        self.aug_dead_p_row = getattr(cfg, 'aug_dead_p_row', 0.2)
        self.aug_dead_max_cols = getattr(cfg, 'aug_dead_max_cols', 6)
        self.aug_dead_max_rows = getattr(cfg, 'aug_dead_max_rows', 6)
        
        self.aug_random_erasing = getattr(cfg, 'aug_random_erasing', True)
        self.aug_erasing_p = getattr(cfg, 'aug_erasing_p', 0.2)
        self.aug_erasing_area = getattr(cfg, 'aug_erasing_area', (0.005, 0.03))
        
        self.aug_bragg_jitter = getattr(cfg, 'aug_bragg_jitter', True)
        self.aug_bragg_p = getattr(cfg, 'aug_bragg_p', 0.3)
        self.aug_bragg_percentile = getattr(cfg, 'aug_bragg_percentile', 95.0)
        self.aug_bragg_scale = getattr(cfg, 'aug_bragg_scale', (0.8, 1.2))
        
        self.aug_gain_offset = getattr(cfg, 'aug_gain_offset', True)
        self.aug_gain_range = getattr(cfg, 'aug_gain_range', (0.9, 1.1))
        self.aug_offset_range = getattr(cfg, 'aug_offset_range', (-0.02, 0.02))
        
        self.aug_gaussian_noise = getattr(cfg, 'aug_gaussian_noise', True)
        self.aug_noise_sigma = getattr(cfg, 'aug_noise_sigma', 0.02)
        
        # Truth vertex params
        self.use_truth_vertex = bool(use_truth_vertex)
        self.vertex_tick_mode = str(vertex_tick_mode).lower()
        self.t0_offset_ticks = int(t0_offset_ticks)
        self.wire_index_base = int(wire_index_base)
        
        # Validate gt_vector_mode
        assert self.gt_vector_mode in ('truth_momentum_pdf', 'com'), \
            f"gt_vector_mode must be 'truth_momentum_pdf' or 'com', got {gt_vector_mode}"
        
        # Discover files from preprocessed directory
        preproc_dir = os.path.join(preprocessed_root, split, 'all')
        if not os.path.isdir(preproc_dir):
            raise FileNotFoundError(f"Preprocessed directory missing: {preproc_dir}")
        
        file_map = {os.path.basename(p): p for p in glob.glob(os.path.join(preproc_dir, '*.npz'))}
        fnames = sorted(file_map.keys())
        
        if self.use_partition:
            fnames = [f for f in fnames if _passes_partition(
                f, role=self.partition_role, frac=self.partition_frac, seed=self.partition_seed
            )]
        
        if split == 'train' and self.train_fraction < 1.0:
            rng = np.random.default_rng(self.partition_seed)
            rng.shuffle(fnames)
            keep = int(max(1, round(self.train_fraction * len(fnames))))
            fnames = fnames[:keep]
        
        self.file_pairs = [(file_map[f],) for f in fnames]  # Single-element tuples

        print(f"    -> NpzEventDataset initialized for split='{split}', view='{view}' with {len(self.file_pairs)} files")
        
        if len(self.file_pairs) == 0:
            raise RuntimeError(f"No NPZ files found in {preproc_dir} for view={self.view}")
        
        # Probe image dimensions from first file
        first_file = self.file_pairs[0][0]
        with np.load(first_file) as arr:
            img = arr[f'{self.view}_data']
            self.H, self.W = int(img.shape[0]), int(img.shape[1])

    @staticmethod
    def _pad_to_multiple(tensor: torch.Tensor, multiple: int = 32) -> torch.Tensor:
        """Unets require inputs to be divisible by 32. Pads with zeros if needed."""
        h, w = tensor.shape[-2:]
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        return F.pad(tensor, (0, pad_w, 0, pad_h), 'constant', 0)
        


    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        (path_preproc,) = self.file_pairs[idx]  # Unpack single-element tuple
        
        try:
            with np.load(path_preproc, allow_pickle=True) as data:
                # Load pre-masked image (muon/ non-brems electrons already removed)
                img_np = data[f'{self.view}_data']
                
                # Can load raw image for visualization/augmentation, but not used for GT vector calculation
                if self.use_raw_bg and f'{self.view}_data_raw' in data:
                    img_pre_masks_np = data[f'{self.view}_data_raw']
                else:
                    img_pre_masks_np = img_np.copy()
                
                # Load ground truth segmentation
                pix_np = data[f'{self.view}_mask']
                
                # Load metadata
                metadata = data['metadata'].item() if 'metadata' in data else {}

            
            if self.resize_to is not None:
                target_h, target_w = self.resize_to
                img_tensor = torch.from_numpy(img_np).float().unsqueeze(0).unsqueeze(0)
                pix_tensor = torch.from_numpy(pix_np).long().unsqueeze(0).unsqueeze(0)
                img_resized = F.interpolate(img_tensor, size=(target_h, target_w), mode='bilinear', align_corners=False)
                pix_resized = F.interpolate(pix_tensor.float(), size=(target_h, target_w), mode='nearest').long()
                img_np = img_resized.squeeze(0).squeeze(0).numpy()
                pix_np = pix_resized.squeeze(0).squeeze(0).numpy()

            H_orig, W_orig = img_np.shape

            
            truth_vertex_rc = None
            michel_info = metadata.get('per_view', {}).get(self.view, {}).get('michel', {})
            
            if self.use_truth_vertex and michel_info.get('vwno') is not None and michel_info.get('vtck') is not None:
                row = float(michel_info['vwno']) - self.wire_index_base
                start = float(michel_info.get('start', 0.0) or 0.0)
                vtck = float(michel_info['vtck'])
                
                if self.vertex_tick_mode == 'relative':
                    col = vtck - start + self.t0_offset_ticks
                else:
                    col = vtck + self.t0_offset_ticks
                
                row = float(np.clip(np.round(row), 0, H_orig - 1))
                col = float(np.clip(np.round(col), 0, W_orig - 1))
                truth_vertex_rc = (row, col)

            
            michel_mask_gt = (np.abs(pix_np) == PDG_MICHEL)
            brems_mask_gt = (np.abs(pix_np) >= PDG_BREMS)
            full_michel_mask_gt = michel_mask_gt | brems_mask_gt
            
            if not np.any(michel_mask_gt):
                return None

            # Determine vertex (use truth if available, else nearest Michel pixel)
            if truth_vertex_rc is not None:
                vertex = truth_vertex_rc
            else:
                # Fallback: use centroid of Michel mask
                ys, xs = np.nonzero(michel_mask_gt)
                if ys.size == 0:
                    return None
                vertex = (float(np.mean(ys)), float(np.mean(xs)))

            # Build GT direction vector 
            d = None  # (dy, dx)
            
            if self.gt_vector_mode == 'truth_momentum_pdf':
                # Use truth momentum from metadata
                vpx = michel_info.get('vpx')
                vpy = michel_info.get('vpy')
                vpz = michel_info.get('vpz')
                tpc = michel_info.get('tpc')
                
                # Determine drift sign from TPC
                if tpc is not None and np.isfinite(tpc):
                    tpc_int = int(round(tpc))
                    drift_sign = +1 if tpc_int == 1 else -1
                else:
                    drift_sign = +1
                
                if (vpx is not None and vpy is not None and vpz is not None and
                    np.isfinite(vpx) and np.isfinite(vpy) and np.isfinite(vpz)):
                    
                    d_arrow = truth_momentum_arrow_from_pdf_plane(
                        vpx, vpy, vpz,
                        plane=self.view,
                        drift_sign=drift_sign,
                        mom_arrow_pix_len=1.0
                    )
                    
                    if d_arrow is not None:
                        d = d_arrow
                    else:
                        return None
                else:
                    return None
            
            elif self.gt_vector_mode == 'com':
                # Center of mass vector
                ys, xs = np.nonzero(full_michel_mask_gt)
                if ys.size == 0:
                    return None
                com_y, com_x = float(np.mean(ys)), float(np.mean(xs))
                d = (com_y - vertex[0], com_x - vertex[1])
            
            else:
                # Other modes have been removed but could be reimplemented here
                return None

            if d is None:
                return None
            dy, dx = d
            n = float(np.hypot(dx, dy))
            if n == 0.0:
                return None
            ux, uy = dx / n, dy / n

            # Endpoints for visualization
            head_xy = (float(vertex[1]), float(vertex[0]))  # (x, y)
            tail_xy = (head_xy[0] + ux, head_xy[1] + uy)  # Default fallback
            u_endpoints = (ux, uy)

            img = torch.from_numpy(img_np).unsqueeze(0).float()
            img_pre = torch.from_numpy(img_pre_masks_np).unsqueeze(0).float()
            
            if self.zscore_norm:
                mu = img.mean()
                sd = img.std().clamp_min(1e-6)
                img = (img - mu) / sd
                img_pre = (img_pre - mu) / sd

            m = None
            if self.enable_seg_labels:
                seg_cls = _map_pixid_to_class(pix_np, self.enable_brems_label)
                m = torch.from_numpy(seg_cls).long()

            # Augmentation
            if self.augment and self.split == 'train':
                # Apply label-invariant augmentations to numpy array first
                img_np_aug = img_np.copy()
                
                # Dead stripes
                if self.aug_dead_stripes:
                    img_np_aug = aug_dead_stripes(
                        img_np_aug,
                        p_col=self.aug_dead_p_col,
                        p_row=self.aug_dead_p_row,
                        max_cols=self.aug_dead_max_cols,
                        max_rows=self.aug_dead_max_rows
                    )
                
                # Random erasing
                if self.aug_random_erasing:
                    img_np_aug = aug_random_erasing(
                        img_np_aug,
                        p=self.aug_erasing_p,
                        area=self.aug_erasing_area
                    )
                
                # Bragg jitter
                if self.aug_bragg_jitter:
                    img_np_aug = aug_bragg_jitter(
                        img_np_aug,
                        p=self.aug_bragg_p,
                        percentile=self.aug_bragg_percentile,
                        scale_range=self.aug_bragg_scale
                    )
                
                # Gain/offset
                if self.aug_gain_offset:
                    img_np_aug = aug_gain_offset(
                        img_np_aug,
                        gain_range=self.aug_gain_range,
                        offset_range=self.aug_offset_range
                    )
                
                # Gaussian noise
                if self.aug_gaussian_noise:
                    img_np_aug = aug_gaussian_noise(
                        img_np_aug,
                        sigma=self.aug_noise_sigma
                    )
                
                # Convert to tensor
                img = torch.from_numpy(img_np_aug).unsqueeze(0).float()
                img_pre = torch.from_numpy(img_pre_masks_np).unsqueeze(0).float()
                
                # Apply z-score normalization after augmentation
                if self.zscore_norm:
                    mu = img.mean()
                    sd = img.std().clamp_min(1e-6)
                    img = (img - mu) / sd
                    img_pre = (img_pre - mu) / sd
                
                # Geometric augmentation 
                H, W = img.shape[-2:]
                grid, affine = _make_affine_grid(H, W, self.aug_degrees, self.aug_scale, self.aug_translate, img.device)

                img = F.grid_sample(img.unsqueeze(0), grid, mode='bilinear', padding_mode='zeros', align_corners=False).squeeze(0)
                img_pre = F.grid_sample(img_pre.unsqueeze(0), grid, mode='bilinear', padding_mode='zeros', align_corners=False).squeeze(0)

                if m is not None:
                    m_f = m.unsqueeze(0).unsqueeze(0).float()
                    m = F.grid_sample(m_f, grid, mode='nearest', padding_mode='zeros', align_corners=False).squeeze(0).squeeze(0).long()
                    gx, gy = grid[0, :, :, 0], grid[0, :, :, 1]
                    valid = (gx >= -1.0) & (gx <= 1.0) & (gy >= -1.0) & (gy <= 1.0)
                    m = torch.where(valid, m, torch.full_like(m, IGNORE_INDEX))

                # Warp points and vectors
                hx, hy = head_xy
                hx_w, hy_w = _warp_point_xy_np(hx, hy, affine['R_np'], affine['tx'], affine['ty'], affine['cx'], affine['cy'])
                head_xy = (hx_w, hy_w)
                
                tx_, ty_ = tail_xy
                tx_w, ty_w = _warp_point_xy_np(tx_, ty_, affine['R_np'], affine['tx'], affine['ty'], affine['cx'], affine['cy'])
                tail_xy = (tx_w, ty_w)

                ux_w, uy_w = _warp_dir_xy_np(ux, uy, affine['R_np'])
                ux, uy = ux_w, uy_w
            
            else:
                # No augmentation - convert to tensor
                img = torch.from_numpy(img_np).unsqueeze(0).float()
                img_pre = torch.from_numpy(img_pre_masks_np).unsqueeze(0).float()
                
                if self.zscore_norm:
                    mu = img.mean()
                    sd = img.std().clamp_min(1e-6)
                    img = (img - mu) / sd
                    img_pre = (img_pre - mu) / sd

            # Pad to multiple of 32 
            img = self._pad_to_multiple(img, 32)
            img_pre = self._pad_to_multiple(img_pre, 32)
            if m is not None:
                m = _pad_to_multiple_mask(m, 32, fill=IGNORE_INDEX)

            sample = {
                "image": img,
                "image_pre_masks": img_pre,
                "u_gt": torch.tensor([ux, uy], dtype=torch.float32),
                "u_endpoints": torch.tensor([u_endpoints[0], u_endpoints[1]], dtype=torch.float32),
                "head_xy": torch.tensor([head_xy[0], head_xy[1]], dtype=torch.float32),
                "tail_xy": torch.tensor([tail_xy[0], tail_xy[1]], dtype=torch.float32),
                "meta": {"fname": os.path.basename(path_preproc), "view": self.view},
                "michel_truth_info": michel_info,
                "original_size": (H_orig, W_orig),  # <-- ADD THIS
            }
            if m is not None:
                sample["seg_mask"] = m

            return sample

        except Exception as e:
            print(f"ERROR loading item {idx} ({os.path.basename(path_preproc)}): {e}")
            traceback.print_exc()
            return None



# Multi-View Dataset
class MultiViewDataset(Dataset):
    """
    Loads all three views (U, V, Z) for each event.
    Each __getitem__ returns a dict with keys 'U', 'V', 'Z'.
    
    The NPZ files for U/V/Z correspond to the same events.
    """
    def __init__(self,
                preprocessed_root: str,
                split: str,
                **kwargs):  
        super().__init__()
        self.views = ['U', 'V', 'Z']
        self.ds = {}
        
        print(f"    -> Creating MultiViewDataset for split='{split}'")
        
        # Create one NpzEventDataset per view
        for v in self.views:
            print(f"       Loading view {v}...")
            kwargs_no_cfg = {k: v for k, v in kwargs.items() if k != 'cfg'}
            # self.ds[v] = NpzEventDataset(
            #     preprocessed_root=preprocessed_root,
            #     split=split,
            #     view=v,
            #     cfg=kwargs.get('cfg'),
            #     **kwargs
            # )
            self.ds[v] = NpzEventDataset(
                preprocessed_root=preprocessed_root,
                split=split,
                view=v,
                cfg=kwargs.get('cfg'), 
                **kwargs_no_cfg  
            )
        
        # Assume all views have the same number of events (they should!)
        lengths = [len(self.ds[v]) for v in self.views]
        self.length = min(lengths)
        
        if len(set(lengths)) > 1:
            print(f"       WARNING: Views have different lengths: {dict(zip(self.views, lengths))}")
            print(f"       Using minimum length: {self.length}")
        else:
            print(f"       All views have {self.length} events")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        """
        Returns dict with keys 'U', 'V', 'Z', each containing a sample dict
        from the corresponding NpzEventDataset.
        
        Returns None if ANY view fails to load to ensure all views are present.
        """
        out = {}
        for v in self.views:
            smp = self.ds[v][idx]
            if smp is None:
                # If any view fails, skip this event entirely
                return None
            out[v] = smp
        return out



# Multi-View Collate Function
def _collate_single(blist, enable_seg=True):
    """
    Collate function for single-view batches.
    """
    blist = [b for b in blist if b is not None]
    if not blist:
        return None
    imgs = torch.stack([b["image"] for b in blist], 0)
    ugt = torch.stack([b["u_gt"] for b in blist], 0)
    uend = torch.stack([b["u_endpoints"] for b in blist], 0)
    hxy = torch.stack([b["head_xy"] for b in blist], 0)
    txy = torch.stack([b["tail_xy"] for b in blist], 0)
    michel_info_list = [b.get("michel_truth_info", {}) for b in blist]
    meta = {
        "fname": [b["meta"]["fname"] for b in blist],
        "view":  [b["meta"]["view"]  for b in blist],
        "michel_truth_info": michel_info_list
    }
    out = {
        "image": imgs,
        "u_gt": ugt,
        "u_endpoints": uend,
        "head_xy": hxy,
        "tail_xy": txy,
        "meta": meta,
    }
    if enable_seg and ("seg_mask" in blist[0]):
        masks = torch.stack([b["seg_mask"] for b in blist], 0)
        if masks.numel() > 0:
            out["seg_mask"] = masks
    return out


def _collate_multi(blist):
    """
    Collate function for multi-view batches.
    
    Input: list of dicts, each with keys 'U', 'V', 'Z'
    where each value should be a sample dict from NpzEventDataset
    
    Output: dict with keys 'U', 'V', 'Z', each containing batched tensors:
    {
        'U': {'image': [B,1,H,W], 'u_gt': [B,2], ...},
        'V': {'image': [B,1,H,W], 'u_gt': [B,2], ...},
        'Z': {'image': [B,1,H,W], 'u_gt': [B,2], ...}
    }
    """
    # Filter out None samples
    blist = [b for b in blist if b is not None]
    if not blist:
        return None
    
    views = ['U', 'V', 'Z']
    out = {}
    
    for v in views:
        # Extract all samples for this view
        view_samples = [b[v] for b in blist]
        
        # Stack all tensors
        imgs = torch.stack([s["image"] for s in view_samples], 0)
        ugts = torch.stack([s["u_gt"] for s in view_samples], 0)
        uends = torch.stack([s["u_endpoints"] for s in view_samples], 0)
        hxy = torch.stack([s["head_xy"] for s in view_samples], 0)
        txy = torch.stack([s["tail_xy"] for s in view_samples], 0)
        
        # Collect metadata
        michel_info_list = [s.get("michel_truth_info", {}) for s in view_samples]
        metas = {
            "fname": [s["meta"]["fname"] for s in view_samples],
            "view": [s["meta"]["view"] for s in view_samples],
            "michel_truth_info": michel_info_list
        }
        
        # Build output dict for this view
        out[v] = {
            "image": imgs,
            "u_gt": ugts,
            "u_endpoints": uends,
            "head_xy": hxy,
            "tail_xy": txy,
            "meta": metas,
        }
        
        # Segmentation masks 
        if "seg_mask" in view_samples[0]:
            masks = torch.stack([s["seg_mask"] for s in view_samples], 0)
            if masks.numel() > 0:
                out[v]["seg_mask"] = masks
    
    return out


# Dataloader helpers

def make_dense_loader(split, view, cfg, is_distributed=False, rank=0, world_size=1):
    """
    Creates dataloader for either single-view or multi-view training.
    """
    if cfg.multi_view:
        # MULTI-VIEW MODE 
        if rank == 0:
            print(f"  [make_dense_loader] Multi-view mode for split={split}")
        
        ds = MultiViewDataset(
            preprocessed_root=cfg.preprocessed_root,
            split=split,
            cfg=cfg,
            use_raw_bg=getattr(cfg, 'use_raw_bg', False),
            use_partition=True,
            partition_role='vector',
            partition_frac=cfg.partition_frac,
            partition_seed=cfg.partition_seed,
            train_fraction=cfg.train_fraction,
            augment=cfg.augment,
            zscore_norm=cfg.zscore_norm,
            enable_seg_labels=cfg.enable_seg,
            enable_brems_label=cfg.enable_brems,
            gt_vector_mode=cfg.gt_vector_mode,
            resize_to=cfg.resize_to,
            use_truth_vertex=getattr(cfg, 'use_truth_vertex', False),
            vertex_tick_mode=getattr(cfg, 'vertex_tick_mode', 'relative'),
            t0_offset_ticks=getattr(cfg, 't0_offset_ticks', 0),
            wire_index_base=getattr(cfg, 'wire_index_base', 0),
        )
        
        # DDP sampler for training, but regular for val/test
        sampler = None
        shuffle = False
        if is_distributed and split == 'train':
            sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
        elif split == 'train':
            shuffle = True
        
        ld = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=cfg.workers,
            pin_memory=True,
            collate_fn=_collate_multi,
            drop_last=(split == 'train'),
        )
        
    else:
        # SINGLE-VIEW MODE 
        ds = NpzEventDataset(
            preprocessed_root=cfg.preprocessed_root,
            split=split,
            view=view,
            cfg=cfg,
            use_raw_bg=getattr(cfg, 'use_raw_bg', False),
            use_partition=True,
            partition_role='vector',
            partition_frac=cfg.partition_frac,
            partition_seed=cfg.partition_seed,
            train_fraction=cfg.train_fraction,
            augment=cfg.augment,
            zscore_norm=cfg.zscore_norm,
            enable_seg_labels=cfg.enable_seg,
            enable_brems_label=cfg.enable_brems,
            gt_vector_mode=cfg.gt_vector_mode,
            resize_to=cfg.resize_to,
            use_truth_vertex=getattr(cfg, 'use_truth_vertex', False),
            vertex_tick_mode=getattr(cfg, 'vertex_tick_mode', 'relative'),
            t0_offset_ticks=getattr(cfg, 't0_offset_ticks', 0),
            wire_index_base=getattr(cfg, 'wire_index_base', 0),
        )
        
        # DDP sampler for training
        sampler = None
        shuffle = False
        if is_distributed and split == 'train':
            sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
        elif split == 'train':
            shuffle = True
        
        ld = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=cfg.workers,
            pin_memory=True,
            collate_fn=partial(_collate_single, enable_seg=cfg.enable_seg),
            drop_last=(split == 'train'),
        )
    
    return ds, ld
