"""Sparse 3-class segmentation (muon / Michel+brems / other) of raw wire-plane images.

The sparse input mask is built from the raw ADC image only: MAD whitening -> multiscale
Difference-of-Gaussians score -> hysteresis growth. A WarpConvNet sparse U-Net then labels
every retained pixel.
"""

import os
import glob
import random
import argparse
import shutil
import json
import functools
from dataclasses import dataclass, field
from typing import Tuple, Union, Callable, Sequence, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warp as wp

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.geometry.base.geometry import Geometry
from warpconvnet.nn.modules.sequential import Sequential
from warpconvnet.nn.modules.sparse_conv import SparseConv2d
from warpconvnet.nn.modules.activations import ReLU


def apply_feature_transform(
    input: Union[Geometry, Tensor],
    transform: Callable,
):
    if isinstance(input, Geometry):
        return input.replace(batched_features=transform(input.feature_tensor))
    else:
        assert isinstance(input, Tensor), f"Expected Tensor, got {type(input)}"
        return transform(input)


def cat(*inputs: Geometry, dim: int = -1):
    if len(inputs) == 1 and isinstance(inputs[0], Sequence):
        inputs = inputs[0]

    assert all(isinstance(input, Geometry) for input in inputs), \
        f"Expected all inputs to be Geometry objects, got {[type(i) for i in inputs]}"

    base_offsets = inputs[0].offsets.long()
    for i, input in enumerate(inputs[1:]):
        if not torch.allclose(input.offsets.long(), base_offsets):
            raise ValueError(f"Input {i+1} offsets do not match input 0 offsets.")

    feats_list = [getattr(i, "feature_tensor", i.features) for i in inputs]
    concatenated_features = torch.cat(feats_list, dim=dim)
    return inputs[0].replace(batched_features=concatenated_features)


def _pick(x, view, default=None):
    if isinstance(x, dict):
        return x.get(view, default if default is not None else next(iter(x.values())))
    return x


PDG_ELECTRON_POS = 11
PDG_ELECTRON_NEG = -11
PDG_MUON_POS = 13
PDG_MUON_NEG = -13
PDG_MICHEL = 9900011
PDG_BREMS = 9910000


def get_class_names(num_classes: int) -> Dict[int, str]:
    if num_classes == 2:
        return {0: "not_michel_brems", 1: "michel_brems"}
    elif num_classes == 3:
        return {0: "other", 1: "muon", 2: "michel_brems"}
    elif num_classes == 4:
        return {0: "other", 1: "muon", 2: "michel", 3: "brems"}
    else:
        raise ValueError(f"Unsupported num_classes: {num_classes}")


def _build_features(
    img_tensor:   torch.Tensor,
    z_tensor:     torch.Tensor,
    score_tensor: torch.Tensor,
    mask:         torch.Tensor,
    high_v:       float,
    in_channels:  int,
) -> torch.Tensor:
    if in_channels == 1:
        return img_tensor[mask].unsqueeze(1)
    elif in_channels == 4:
        return torch.stack([
            img_tensor[mask],
            z_tensor[mask],
            score_tensor[mask],
            (score_tensor[mask] > float(high_v)).float(),
        ], dim=1)
    else:
        raise ValueError(f"in_channels must be 1 or 4, got {in_channels}")


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', ignore_index=-1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        valid_mask = (targets != self.ignore_index)

        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        logits  = logits[valid_mask]
        targets = targets[valid_mask]
        probs   = probs[valid_mask]

        ce_loss      = F.cross_entropy(logits, targets, reduction='none')
        p_t          = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma
        loss         = focal_weight * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss    = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=-1):
        super().__init__()
        self.smooth       = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        valid = (targets != self.ignore_index)

        if not valid.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        targets_valid   = targets[valid]
        probs_valid     = probs[valid]
        num_classes     = probs.shape[1]
        targets_one_hot = F.one_hot(targets_valid, num_classes).float()

        intersection = (probs_valid * targets_one_hot).sum(0)
        cardinality  = probs_valid.sum(0) + targets_one_hot.sum(0)
        dice         = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, focal_loss, dice_loss, focal_weight: float = 20.0 / 21.0):
        super().__init__()
        self.focal        = focal_loss
        self.dice         = dice_loss
        self.focal_weight = focal_weight
        self.dice_weight  = 1.0 - focal_weight

    def forward(self, logits, targets):
        return (self.focal_weight * self.focal(logits, targets)
                + self.dice_weight  * self.dice(logits, targets))


@dataclass
class SparseUNetConfig:
    raw_root:      str = ""
    cleansig_root: str = ""
    out_dir:       str = ""

    input_source: str             = "raw"
    views:        Tuple[str, ...] = ("U", "V", "Z")
    num_classes:  int             = 3

    low_thr:  Union[float, Dict[str, float]] = field(
        default_factory=lambda: {"U": 0.8, "V": 0.8, "Z": 0.4})
    high_thr: Union[float, Dict[str, float]] = field(
        default_factory=lambda: {"U": 1.8, "V": 1.8, "Z": 2.8})
    sigmas: Union[Tuple[float, ...], Dict[str, Tuple[float, ...]]] = field(
        default_factory=lambda: {
            "U": (0.8, 1.2, 1.8, 2.6, 3.6),
            "V": (0.8, 1.2, 1.8, 2.6, 3.6),
            "Z": (0.8, 1.2, 1.8, 2.6),
        })
    dog_k:            float = 1.6
    truncate:         float = 3.0
    scale_normalize:  bool  = True
    use_abs_z:        bool  = True
    hyst_max_iters:   int   = 512
    use_maxima_seeds: Union[bool, Dict[str, bool]] = field(
        default_factory=lambda: {"U": True, "V": True, "Z": False})
    seed_min: Union[float, Dict[str, float], None] = None

    init_dim: int = 32
    planes: Tuple[int, ...] = field(
        default_factory=lambda: (32, 64, 128, 256, 384, 384,
                                 256, 192, 128,  96,  96,  96))
    layers: Tuple[int, ...] = field(
        default_factory=lambda: (2, 2, 3, 3, 3, 3,
                                 3, 3, 2, 2, 2, 2))

    in_channels: int = 4

    use_attention:      bool  = True
    attn_heads:         int   = 8
    attn_dropout:       float = 0.0
    attn_patch_size:    int   = 64
    attn_patch_size_d5: int   = 128
    attn_patch_size_d4: int   = 256

    epochs:         int   = 100
    batch_size:     int   = 64
    lr:             float = 1e-3
    weight_decay:   float = 0.01
    grad_clip_norm: float = 1.0

    loss_type:    str   = "combined"
    focal_gamma:  float = 2.0
    focal_weight: float = 20.0 / 21.0
    dice_smooth:  float = 1.0
    use_class_weights:    bool                        = True
    manual_class_weights: Union[Tuple[float, ...], None] = None

    scheduler_type: str   = "cosine"
    cosine_T0:      int   = 20
    cosine_T_mult:  int   = 2
    cosine_eta_min: float = 1e-6

    num_workers:   int  = 4
    amp:           bool = True
    save_dir:      str  = "./checkpoints"
    patience:      int  = 15
    warmup_epochs: int  = 5
    viz_k:         int  = 10


@torch.no_grad()
def robust_zscore_gpu(img_hw: torch.Tensor, eps: float = 1e-6):
    if img_hw.dim() != 2:
        raise ValueError("robust_zscore_gpu expects img shaped (H,W)")
    x   = img_hw.flatten()
    med = x.median()
    mad = (x - med).abs().median()
    z   = (img_hw - med) / (1.4826 * mad + eps)
    return z, med, 1.4826 * mad


@torch.no_grad()
def _gaussian_kernel_1d(sigma: float, truncate: float, device, dtype):
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    radius = int(truncate * sigma + 0.5)
    xs     = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kern   = torch.exp(-0.5 * (xs / sigma) ** 2)
    return kern / kern.sum()


@torch.no_grad()
def gaussian_blur_sep_gpu(img_hw: torch.Tensor, sigma: float,
                           truncate: float = 3.0) -> torch.Tensor:
    if img_hw.dim() != 2:
        raise ValueError("gaussian_blur_sep_gpu expects img shaped (H,W)")
    device = img_hw.device
    dtype  = img_hw.dtype
    k1d    = _gaussian_kernel_1d(float(sigma), float(truncate), device=device, dtype=dtype)
    k      = int(k1d.numel())
    pad    = k // 2
    x      = img_hw.unsqueeze(0).unsqueeze(0)
    kx     = k1d.view(1, 1, 1, k)
    x      = F.pad(x, (pad, pad, 0, 0), mode="reflect")
    x      = F.conv2d(x, kx, padding=0)
    ky     = k1d.view(1, 1, k, 1)
    x      = F.pad(x, (0, 0, pad, pad), mode="reflect")
    x      = F.conv2d(x, ky, padding=0)
    return x.squeeze(0).squeeze(0)


@torch.no_grad()
def multiscale_dog_score_gpu(
    img_hw: torch.Tensor,
    sigmas: Tuple[float, ...],
    k: float = 1.6,
    truncate: float = 3.0,
    scale_normalize: bool = True,
    use_abs_response: bool = True,
) -> torch.Tensor:
    if img_hw.dim() != 2:
        raise ValueError("multiscale_dog_score_gpu expects img shaped (H,W)")
    score = torch.zeros_like(img_hw)
    for s in sigmas:
        s     = float(s)
        blur1 = gaussian_blur_sep_gpu(img_hw, sigma=s,            truncate=truncate)
        blur2 = gaussian_blur_sep_gpu(img_hw, sigma=s * float(k), truncate=truncate)
        dog   = blur1 - blur2
        if scale_normalize:
            dog = dog * (s ** 2)
        resp  = dog.abs() if use_abs_response else dog
        score = torch.maximum(score, resp)
    return score


@torch.no_grad()
def hysteresis_grow_gpu(
    seeds:   torch.Tensor,
    allowed: torch.Tensor,
    max_iters: int = 512
) -> torch.Tensor:
    if seeds.dtype   != torch.bool: seeds   = seeds.bool()
    if allowed.dtype != torch.bool: allowed = allowed.bool()
    cur = seeds & allowed
    for _ in range(int(max_iters)):
        cur_f = cur.float().unsqueeze(0).unsqueeze(0)
        dil   = F.max_pool2d(cur_f, kernel_size=3, stride=1, padding=1) > 0
        nxt   = dil.squeeze(0).squeeze(0) & allowed
        if torch.equal(nxt, cur):
            break
        cur = nxt
    return cur


@torch.no_grad()
def create_sparse_mask_iceberg_gpu(
    score_hw: torch.Tensor,
    low:      float,
    high:     float,
    max_iters: int = 512,
    use_maxima_seeds: bool  = True,
    seed_min:         float = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if score_hw.dim() != 2:
        raise ValueError("create_sparse_mask_iceberg_gpu expects score shaped (H,W)")
    low     = float(low)
    high    = float(high)
    allowed = score_hw > low
    seeds   = score_hw > high
    if use_maxima_seeds:
        if seed_min is None:
            seed_min = 0.5 * (low + high)
        s         = score_hw.unsqueeze(0).unsqueeze(0)
        pooled    = F.max_pool2d(s, kernel_size=3, stride=1, padding=1).squeeze(0).squeeze(0)
        local_max = (score_hw >= (pooled - 1e-12)) & (score_hw > float(seed_min))
        seeds     = seeds | local_max
    seeds = seeds & allowed
    mask  = hysteresis_grow_gpu(seeds=seeds, allowed=allowed, max_iters=max_iters)
    return mask, seeds, allowed


def _map_pixid_to_classes(pixid: np.ndarray, num_classes: int) -> np.ndarray:
    if num_classes == 2:
        cls = np.zeros_like(pixid, dtype=np.int64)
        cls[(np.abs(pixid) == PDG_MICHEL) | (np.abs(pixid) >= PDG_BREMS)] = 1
        return cls
    elif num_classes == 3:
        cls = np.zeros_like(pixid, dtype=np.int64)
        cls[np.abs(pixid) == PDG_MUON_POS] = 1
        cls[(np.abs(pixid) == PDG_MICHEL) | (np.abs(pixid) >= PDG_BREMS)] = 2
        return cls
    elif num_classes == 4:
        cls = np.zeros_like(pixid, dtype=np.int64)
        cls[np.abs(pixid) == PDG_MUON_POS] = 1
        cls[np.abs(pixid) == PDG_MICHEL]   = 2
        cls[np.abs(pixid) >= PDG_BREMS]    = 3
        return cls
    else:
        raise ValueError(f"Unsupported num_classes: {num_classes}")


class SparseNeutrinoDataset(Dataset):
    def __init__(self, raw_root, cleansig_root, split, view, cfg):
        super().__init__()
        self.split = split
        self.view  = view.upper()
        self.cfg   = cfg

        raw_dir   = os.path.join(raw_root,      split, "all")
        clean_dir = os.path.join(cleansig_root, split, "all")

        if not (os.path.isdir(raw_dir) and os.path.isdir(clean_dir)):
            raw_dir   = os.path.join(raw_root,      split)
            clean_dir = os.path.join(cleansig_root, split)
            if not (os.path.isdir(raw_dir) and os.path.isdir(clean_dir)):
                raise FileNotFoundError(
                    f"Missing directories: {raw_dir} or {clean_dir}")

        raw_files   = {os.path.basename(p): p
                       for p in glob.glob(os.path.join(raw_dir,   "*.npz"))}
        clean_files = {os.path.basename(p): p
                       for p in glob.glob(os.path.join(clean_dir, "*.npz"))}

        common          = sorted(set(raw_files.keys()) & set(clean_files.keys()))
        self.file_pairs = [(raw_files[f], clean_files[f]) for f in common]

        if len(self.file_pairs) == 0:
            raise RuntimeError(f"No matched NPZ pairs found for {split}/{view}")

        print(f"  [{split}/{view}] Found {len(self.file_pairs)} samples")

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        path_raw, path_clean = self.file_pairs[idx]
        try:
            with np.load(path_raw,   allow_pickle=True) as data_r, \
                 np.load(path_clean, allow_pickle=True) as data_c:

                img_key  = f"{self.view}_data"
                mask_key = f"{self.view}_mask"

                img_np    = data_r[img_key] if self.cfg.input_source == "raw" \
                            else data_c[img_key]
                pixid_np  = data_c[mask_key]
                labels_np = _map_pixid_to_classes(pixid_np, self.cfg.num_classes)

                return {
                    "image":          torch.from_numpy(img_np).float(),
                    "labels_dense":   torch.from_numpy(labels_np).long(),
                    "original_shape": img_np.shape,
                    "fname":          os.path.basename(path_clean),
                    "view":           self.view,
                }
        except Exception as e:
            print(f"Error loading {os.path.basename(path_clean)}: {e}")
            return None


def collate_sparse_fn(batch, cfg, device):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    all_coords = []
    all_feats  = []
    all_labels = []
    offsets    = torch.zeros(len(batch) + 1, dtype=torch.int32)
    shapes     = []
    images     = []
    fnames     = []
    views      = []

    for i, sample in enumerate(batch):
        img_tensor   = sample["image"].to(device)
        labels_dense = sample["labels_dense"]
        view         = sample["view"]

        low_v        = float(_pick(cfg.low_thr,          view))
        high_v       = float(_pick(cfg.high_thr,         view))
        sigmas_v     = tuple(_pick(cfg.sigmas,           view))
        use_maxima_v = bool( _pick(cfg.use_maxima_seeds, view))
        seed_min_v   = _pick(cfg.seed_min, view, default=None)

        z_tensor, z_med, z_sigma = robust_zscore_gpu(img_tensor, eps=1e-6)
        z_for_score              = z_tensor.abs() if cfg.use_abs_z else z_tensor

        score_tensor = multiscale_dog_score_gpu(
            img_hw=z_for_score, sigmas=sigmas_v, k=cfg.dog_k,
            truncate=cfg.truncate, scale_normalize=cfg.scale_normalize,
            use_abs_response=True,
        )

        mask, seeds, allowed = create_sparse_mask_iceberg_gpu(
            score_hw=score_tensor, low=low_v, high=high_v,
            max_iters=cfg.hyst_max_iters, use_maxima_seeds=use_maxima_v,
            seed_min=seed_min_v,
        )

        coords = torch.nonzero(mask, as_tuple=False)
        if len(coords) == 0:
            continue

        feats = _build_features(
            img_tensor, z_tensor, score_tensor, mask, high_v, cfg.in_channels)

        labels_dense_gpu = labels_dense.to(device)
        labels           = labels_dense_gpu[mask].long()

        all_coords.append(coords)
        all_feats.append(feats)
        all_labels.append(labels)
        offsets[len(all_coords)] = offsets[len(all_coords) - 1] + len(coords)
        shapes.append(sample["original_shape"])
        images.append(sample["image"])
        fnames.append(sample["fname"])
        views.append(view)

    if len(all_coords) == 0:
        return None

    coords = torch.cat(all_coords, dim=0).int()
    feats  = torch.cat(all_feats,  dim=0).float()
    labels = torch.cat(all_labels, dim=0).long()

    return {
        "coords":          coords,
        "feats":           feats,
        "labels":          labels,
        "offsets":         offsets[:len(all_coords) + 1].to(device),
        "original_shapes": shapes,
        "original_images": images,
        "fnames":          fnames,
        "views":           views,
        "batch_size":      len(all_coords),
    }


def compute_class_weights_from_dataset(dataset, cfg, device, max_samples=1000):
    num_classes = cfg.num_classes
    counts      = torch.zeros(num_classes, dtype=torch.float64)
    n           = len(dataset)
    idxs        = range(n) if max_samples is None \
                  else random.sample(range(n), min(max_samples, n))

    for i in tqdm(idxs, desc="Computing class weights", leave=False):
        sample = dataset[i]
        if sample is None:
            continue

        img_tensor   = sample["image"].to(device)
        labels_dense = sample["labels_dense"]
        view         = sample["view"]

        low_v        = float(_pick(cfg.low_thr,          view))
        high_v       = float(_pick(cfg.high_thr,         view))
        sigmas_v     = tuple(_pick(cfg.sigmas,           view))
        use_maxima_v = bool( _pick(cfg.use_maxima_seeds, view))
        seed_min_v   = _pick(cfg.seed_min, view, default=None)

        z_tensor, _, _ = robust_zscore_gpu(img_tensor, eps=1e-6)
        z_for_score    = z_tensor.abs() if cfg.use_abs_z else z_tensor

        score_tensor = multiscale_dog_score_gpu(
            img_hw=z_for_score, sigmas=sigmas_v, k=cfg.dog_k,
            truncate=cfg.truncate, scale_normalize=cfg.scale_normalize,
            use_abs_response=True,
        )

        mask, _, _ = create_sparse_mask_iceberg_gpu(
            score_hw=score_tensor, low=low_v, high=high_v,
            max_iters=cfg.hyst_max_iters, use_maxima_seeds=use_maxima_v,
            seed_min=seed_min_v,
        )

        labels = labels_dense.to(device)[mask].long()
        for c in range(num_classes):
            counts[c] += (labels == c).sum().item()

    total   = counts.sum().clamp(min=1.0)
    freqs   = counts / total
    weights = 1.0 / (freqs + 1e-12)
    weights = weights / weights.mean()
    return weights.float()


class ConvBlock2d(Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 activation=nn.ReLU(inplace=True), bias=False):
        super().__init__()
        self.conv = SparseConv2d(in_channels, out_channels, kernel_size, stride, bias=bias)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act  = nn.Identity() if activation is None else activation
        self.add_module("conv", self.conv)
        self.add_module("norm", self.norm)
        self.add_module("act",  self.act)


class ConvTrBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 activation=nn.ReLU(inplace=True), bias=False):
        super().__init__()
        self.conv_tr = SparseConv2d(in_channels, out_channels, kernel_size, stride,
                                    transposed=True, bias=bias)
        self.norm_act = Sequential(
            nn.BatchNorm1d(out_channels),
            nn.Identity() if activation is None else activation,
        )

    def forward(self, x: Voxels, out_spatial_sparsity: Voxels) -> Voxels:
        out = self.conv_tr(x, out_spatial_sparsity)
        out = self.norm_act(out)
        return out


class BasicBlock2d(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, bias=False):
        super().__init__()
        self.conv1      = ConvBlock2d(in_channels, out_channels, kernel_size=3,
                                      stride=stride, bias=bias)
        self.conv2      = ConvBlock2d(out_channels, out_channels, kernel_size=3,
                                      activation=None, bias=bias)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = ConvBlock2d(in_channels, out_channels, kernel_size=1,
                                          stride=stride, activation=None, bias=bias)
        self.relu = ReLU(inplace=True)

    def forward(self, x: Voxels) -> Voxels:
        identity = x
        out      = self.conv1(x)
        out      = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out


class SparseVoxelAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, patch_size: int, dropout: float):
        super().__init__()
        self.patch_size = patch_size
        self.attn_norm  = nn.LayerNorm(dim)
        self.attn       = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn      = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: Voxels) -> Voxels:
        feats   = x.feature_tensor
        offsets = x.offsets.long()
        out     = feats.clone()

        # each event attends only to its own sparse points
        for i in range(len(offsets) - 1):
            s, e = int(offsets[i]), int(offsets[i + 1])
            if s == e:
                continue
            chunk  = feats[s:e].unsqueeze(0)
            normed = self.attn_norm(chunk)
            attn_out, _ = self.attn(normed, normed, normed)
            out[s:e] = feats[s:e] + attn_out.squeeze(0)

        out = out + self.ffn(self.ffn_norm(out))

        # replace(batched_features=...) preserves stride and all other metadata
        return x.replace(batched_features=out)


def _make_bottleneck_attn(dim: int, num_heads: int, patch_size: int,
                           dropout: float) -> SparseVoxelAttention:
    return SparseVoxelAttention(
        dim=dim, num_heads=num_heads,
        patch_size=patch_size, dropout=dropout,
    )


class SparseUNet2D(nn.Module):
    def __init__(self, cfg: SparseUNetConfig):
        super().__init__()
        self.cfg    = cfg
        init_dim    = cfg.init_dim
        planes      = cfg.planes
        layers      = cfg.layers
        in_ch       = cfg.in_channels
        num_classes = cfg.num_classes

        assert len(planes) == 12, \
            f"planes must have 12 entries for 6-scale U-Net, got {len(planes)}"
        assert len(layers) == 12, \
            f"layers must have 12 entries for 6-scale U-Net, got {len(layers)}"

        self.conv0 = ConvBlock2d(in_ch, init_dim, kernel_size=1, bias=False)

        self.conv1_enc  = ConvBlock2d(init_dim,  init_dim,  kernel_size=2, stride=2)
        self.block1_enc = self._make_layer(init_dim,  planes[0], layers[0])

        self.conv2_enc  = ConvBlock2d(planes[0], planes[0], kernel_size=2, stride=2)
        self.block2_enc = self._make_layer(planes[0], planes[1], layers[1])

        self.conv3_enc  = ConvBlock2d(planes[1], planes[1], kernel_size=2, stride=2)
        self.block3_enc = self._make_layer(planes[1], planes[2], layers[2])

        self.conv4_enc  = ConvBlock2d(planes[2], planes[2], kernel_size=2, stride=2)
        self.block4_enc = self._make_layer(planes[2], planes[3], layers[3])

        self.conv5_enc  = ConvBlock2d(planes[3], planes[3], kernel_size=2, stride=2)
        self.block5_enc = self._make_layer(planes[3], planes[4], layers[4])

        self.conv6_enc  = ConvBlock2d(planes[4], planes[4], kernel_size=2, stride=2)
        self.block6_enc = self._make_layer(planes[4], planes[5], layers[5])

        if cfg.use_attention:
            self.bottleneck_attn: nn.Module = _make_bottleneck_attn(
                dim=planes[5],
                num_heads=cfg.attn_heads,
                patch_size=cfg.attn_patch_size,
                dropout=cfg.attn_dropout,
            )
        else:
            self.bottleneck_attn = None

        self.convtr6_dec = ConvTrBlock2d(planes[5], planes[6], kernel_size=2, stride=2)
        self.block_d5    = self._make_layer(planes[6] + planes[4], planes[6], layers[6])
        self.attn_d5: nn.Module = (_make_bottleneck_attn(
            dim=planes[6], num_heads=cfg.attn_heads,
            patch_size=cfg.attn_patch_size_d5, dropout=cfg.attn_dropout,
        ) if cfg.use_attention else None)

        self.convtr5_dec = ConvTrBlock2d(planes[6], planes[7], kernel_size=2, stride=2)
        self.block_d4    = self._make_layer(planes[7] + planes[3], planes[7], layers[7])
        self.attn_d4: nn.Module = (_make_bottleneck_attn(
            dim=planes[7], num_heads=cfg.attn_heads,
            patch_size=cfg.attn_patch_size_d4, dropout=cfg.attn_dropout,
        ) if cfg.use_attention else None)

        self.convtr4_dec = ConvTrBlock2d(planes[7], planes[8], kernel_size=2, stride=2)
        self.block_d3    = self._make_layer(planes[8] + planes[2], planes[8], layers[8])

        self.convtr3_dec = ConvTrBlock2d(planes[8], planes[9], kernel_size=2, stride=2)
        self.block_d2    = self._make_layer(planes[9] + planes[1], planes[9], layers[9])

        self.convtr2_dec = ConvTrBlock2d(planes[9],  planes[10], kernel_size=2, stride=2)
        self.block_d1    = self._make_layer(planes[10] + planes[0], planes[10], layers[10])

        self.convtr1_dec = ConvTrBlock2d(planes[10], planes[11], kernel_size=2, stride=2)
        self.block_d0    = self._make_layer(planes[11] + init_dim, planes[11], layers[11])

        self.final = SparseConv2d(planes[11], num_classes, kernel_size=1, bias=True)

    def _make_layer(self, in_channels, out_channels, blocks):
        layer_list = [BasicBlock2d(in_channels, out_channels)]
        for _ in range(1, blocks):
            layer_list.append(BasicBlock2d(out_channels, out_channels))
        return nn.Sequential(*layer_list)

    def forward(self, x: Voxels) -> Voxels:
        skip0 = self.conv0(x)

        out   = self.conv1_enc(skip0)
        skip1 = self.block1_enc(out)

        out   = self.conv2_enc(skip1)
        skip2 = self.block2_enc(out)

        out   = self.conv3_enc(skip2)
        skip3 = self.block3_enc(out)

        out   = self.conv4_enc(skip3)
        skip4 = self.block4_enc(out)

        out   = self.conv5_enc(skip4)
        skip5 = self.block5_enc(out)

        out   = self.conv6_enc(skip5)
        out   = self.block6_enc(out)

        if self.bottleneck_attn is not None:
            out = self.bottleneck_attn(out)

        out = self.convtr6_dec(out, skip5)
        out = cat(out, skip5)
        out = self.block_d5(out)
        if self.attn_d5 is not None:
            out = self.attn_d5(out)

        out = self.convtr5_dec(out, skip4)
        out = cat(out, skip4)
        out = self.block_d4(out)
        if self.attn_d4 is not None:
            out = self.attn_d4(out)

        out = self.convtr4_dec(out, skip3)
        out = cat(out, skip3)
        out = self.block_d3(out)

        out = self.convtr3_dec(out, skip2)
        out = cat(out, skip2)
        out = self.block_d2(out)

        out = self.convtr2_dec(out, skip1)
        out = cat(out, skip1)
        out = self.block_d1(out)

        out = self.convtr1_dec(out, skip0)
        out = cat(out, skip0)
        out = self.block_d0(out)

        return self.final(out)


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, num_classes: int):
    pred   = pred.view(-1)
    target = target.view(-1)
    k      = (target >= 0) & (target < num_classes)
    inds   = num_classes * target[k] + pred[k]
    cm     = torch.bincount(inds, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    diag   = cm.diagonal().float()
    denom  = (cm.sum(0) + cm.sum(1) - diag).clamp(min=1.0)
    iou    = diag / denom
    miou   = iou.mean()
    acc    = diag.sum() / cm.sum().clamp(min=1.0)
    return {
        "acc":           acc.item(),
        "miou":          miou.item(),
        "iou_per_class": iou.cpu().tolist(),
        "cm":            cm.cpu(),
    }


def create_loss_fn(cfg, class_weights=None):
    if cfg.loss_type == "focal":
        return FocalLoss(alpha=class_weights, gamma=cfg.focal_gamma, ignore_index=-1)
    elif cfg.loss_type == "dice":
        return DiceLoss(smooth=cfg.dice_smooth, ignore_index=-1)
    elif cfg.loss_type == "combined":
        focal = FocalLoss(alpha=class_weights, gamma=cfg.focal_gamma, ignore_index=-1)
        dice  = DiceLoss(smooth=cfg.dice_smooth, ignore_index=-1)
        return CombinedLoss(focal, dice, focal_weight=cfg.focal_weight)
    else:
        return None


def _apply_loss(logits, labels, cfg, loss_fn, class_weights):
    if cfg.loss_type in ["focal", "dice", "combined"]:
        return loss_fn(logits, labels), logits.argmax(1)

    elif cfg.loss_type == "ce":
        loss = F.cross_entropy(logits, labels,
                               weight=class_weights, ignore_index=-1)
        return loss, logits.argmax(1)

    elif cfg.loss_type == "bce":
        if cfg.num_classes != 2:
            raise ValueError("BCE loss only supported for 2-class mode")
        probs   = torch.sigmoid(logits[:, 0])
        valid   = (labels >= 0)
        targets = (labels == 1).float()
        loss    = F.binary_cross_entropy(probs[valid], targets[valid])
        return loss, (probs > 0.5).long()

    else:
        raise ValueError(f"Unknown loss type: {cfg.loss_type}")


def train_epoch(model, loader, optimizer, scaler, cfg, epoch, device,
                rank=0, loss_fn=None, class_weights=None):
    model.train()
    total_loss = total_acc = total_miou = 0.0
    n_batches  = 0

    iter_loader = loader
    if rank == 0:
        iter_loader = tqdm(loader, desc=f"Train {epoch:03d}",
                           dynamic_ncols=True, leave=False)

    for batch in iter_loader:
        if batch is None:
            continue

        coords  = batch["coords"].to(device, non_blocking=True)
        feats   = batch["feats"].to(device,  non_blocking=True)
        labels  = batch["labels"].to(device, non_blocking=True)
        offsets = batch["offsets"].to(device, non_blocking=True)

        voxels = Voxels(coords, feats, offsets)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=cfg.amp):
            out_voxels = model(voxels)
            logits     = out_voxels.features
            loss, pred = _apply_loss(logits, labels, cfg, loss_fn, class_weights)

        if scaler is not None:
            scaler.scale(loss).backward()
            if cfg.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

        with torch.no_grad():
            metrics = compute_metrics(pred, labels, num_classes=cfg.num_classes)

        total_loss += loss.item()
        total_acc  += metrics["acc"]
        total_miou += metrics["miou"]
        n_batches  += 1

        if rank == 0 and hasattr(iter_loader, "set_postfix"):
            iter_loader.set_postfix({
                "loss": f"{total_loss / n_batches:.4f}",
                "acc":  f"{total_acc  / n_batches:.4f}",
                "miou": f"{total_miou / n_batches:.4f}",
            })

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        mt = torch.tensor(
            [total_loss, total_acc, total_miou, n_batches], device=device)
        dist.all_reduce(mt, op=dist.ReduceOp.SUM)
        total_loss, total_acc, total_miou = mt[:3].tolist()
        n_batches = int(mt[3].item())

    return {
        "loss": total_loss / max(1, n_batches),
        "acc":  total_acc  / max(1, n_batches),
        "miou": total_miou / max(1, n_batches),
    }


@torch.no_grad()
def eval_epoch(model, loader, cfg, device, rank=0, loss_fn=None, class_weights=None):
    model.eval()
    total_loss = 0.0
    cm_total   = torch.zeros(
        (cfg.num_classes, cfg.num_classes), dtype=torch.long, device=device)
    n_batches  = 0

    iter_loader = loader
    if rank == 0:
        iter_loader = tqdm(loader, desc="Val",
                           dynamic_ncols=True, leave=False)

    for batch in iter_loader:
        if batch is None:
            continue

        coords  = batch["coords"].to(device, non_blocking=True)
        feats   = batch["feats"].to(device,  non_blocking=True)
        labels  = batch["labels"].to(device, non_blocking=True)
        offsets = batch["offsets"].to(device, non_blocking=True)

        voxels     = Voxels(coords, feats, offsets)
        out_voxels = model(voxels)
        logits     = out_voxels.features
        loss, pred = _apply_loss(logits, labels, cfg, loss_fn, class_weights)

        metrics    = compute_metrics(pred, labels, num_classes=cfg.num_classes)
        total_loss += loss.item()
        cm_total   += metrics["cm"].to(device)
        n_batches  += 1

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(cm_total, op=dist.ReduceOp.SUM)
        lt = torch.tensor([total_loss, n_batches], device=device)
        dist.all_reduce(lt, op=dist.ReduceOp.SUM)
        total_loss = lt[0].item()
        n_batches  = int(lt[1].item())

    diag  = cm_total.diagonal().float()
    denom = (cm_total.sum(0) + cm_total.sum(1) - diag).clamp(min=1.0)
    iou   = diag / denom
    miou  = iou.mean()
    acc   = diag.sum() / cm_total.sum().clamp(min=1.0)

    return {
        "loss":          total_loss / max(1, n_batches),
        "acc":           acc.item(),
        "miou":          miou.item(),
        "iou_per_class": iou.cpu().tolist(),
        "cm":            cm_total.cpu(),
    }


@torch.no_grad()
def visualize_predictions(model, dataset, cfg, device, save_dir, k=None, view=None):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    CLASS_NAMES = get_class_names(cfg.num_classes)

    if cfg.num_classes == 2:
        class_colors = {0: [1.0, 0.0, 0.0], 1: [0.0, 1.0, 1.0]}
    elif cfg.num_classes == 3:
        class_colors = {0: [0.5, 0.5, 0.5], 1: [1.0, 0.0, 0.0], 2: [0.0, 1.0, 1.0]}
    else:
        class_colors = {0: [0.5, 0.5, 0.5], 1: [1.0, 0.0, 0.0],
                        2: [0.0, 1.0, 1.0],  3: [1.0, 0.0, 1.0]}

    n       = len(dataset)
    indices = list(range(n)) if (k is None or k <= 0 or k >= n) \
              else random.sample(range(n), k)

    for idx in indices:
        sample = dataset[idx]
        if sample is None:
            continue

        img_tensor       = sample["image"].to(device)
        labels_dense_cpu = sample["labels_dense"].cpu().numpy()
        H, W             = sample["original_shape"]
        fname            = sample["fname"]
        sample_view      = sample["view"]

        low_v        = float(_pick(cfg.low_thr,          sample_view))
        high_v       = float(_pick(cfg.high_thr,         sample_view))
        sigmas_v     = tuple(_pick(cfg.sigmas,           sample_view))
        use_maxima_v = bool( _pick(cfg.use_maxima_seeds, sample_view))
        seed_min_v   = _pick(cfg.seed_min, sample_view, default=None)

        z_tensor, z_med, z_sigma = robust_zscore_gpu(img_tensor, eps=1e-6)
        z_for_score              = z_tensor.abs() if cfg.use_abs_z else z_tensor

        score_tensor = multiscale_dog_score_gpu(
            img_hw=z_for_score, sigmas=sigmas_v, k=cfg.dog_k,
            truncate=cfg.truncate, scale_normalize=cfg.scale_normalize,
            use_abs_response=True,
        )

        mask_gpu, seeds_gpu, allowed_gpu = create_sparse_mask_iceberg_gpu(
            score_hw=score_tensor, low=low_v, high=high_v,
            max_iters=cfg.hyst_max_iters, use_maxima_seeds=use_maxima_v,
            seed_min=seed_min_v,
        )

        coords = torch.nonzero(mask_gpu, as_tuple=False)
        if len(coords) == 0:
            continue

        feats = _build_features(
            img_tensor, z_tensor, score_tensor, mask_gpu, high_v, cfg.in_channels)

        labels_dense_gpu = sample["labels_dense"].to(device)
        labels_sparse    = labels_dense_gpu[mask_gpu].long()

        offsets    = torch.tensor([0, len(coords)], dtype=torch.int32, device=device)
        voxels     = Voxels(coords.int(), feats, offsets)
        out_voxels = model(voxels)
        logits     = out_voxels.features

        if cfg.loss_type == "bce":
            pred_sparse = (torch.sigmoid(logits[:, 0]) > 0.5).long()
        else:
            pred_sparse = logits.argmax(1)

        img_np     = img_tensor.cpu().numpy()
        z_np       = z_tensor.cpu().numpy()
        score_np   = score_tensor.cpu().numpy()
        mask_np    = mask_gpu.cpu().numpy()
        seeds_np   = seeds_gpu.cpu().numpy()
        allowed_np = allowed_gpu.cpu().numpy()
        coords_np  = coords.cpu().numpy()
        pred_np    = pred_sparse.cpu().numpy()

        total_pixels     = mask_np.size
        kept_pixels      = int(np.sum(mask_np))
        global_retention = 100.0 * kept_pixels / max(1, total_pixels)

        if cfg.num_classes == 4:
            michel_pixels    = (labels_dense_cpu == 2)
            total_michel     = int(np.sum(michel_pixels))
            kept_michel      = int(np.sum(mask_np & michel_pixels))
            michel_retention = 100.0 * kept_michel / float(total_michel) \
                               if total_michel > 0 else 0.0
            brems_pixels     = (labels_dense_cpu == 3)
            total_brems      = int(np.sum(brems_pixels))
            kept_brems       = int(np.sum(mask_np & brems_pixels))
            brems_retention  = 100.0 * kept_brems / float(total_brems) \
                               if total_brems > 0 else 0.0
        else:
            mb_class     = cfg.num_classes - 1
            mb_pixels    = (labels_dense_cpu == mb_class)
            total_mb     = int(np.sum(mb_pixels))
            kept_mb      = int(np.sum(mask_np & mb_pixels))
            mb_retention = 100.0 * kept_mb / float(total_mb) if total_mb > 0 else 0.0

        pred_img = np.full((H, W), -1, dtype=np.int64)
        for i, (y, x) in enumerate(coords_np):
            pred_img[y, x] = int(pred_np[i])

        gt_overlay = np.zeros((H, W, 3), dtype=np.float32)
        for cls_id, color in class_colors.items():
            gt_overlay[labels_dense_cpu == cls_id] = color

        pred_overlay = np.zeros((H, W, 3), dtype=np.float32)
        for cls_id, color in class_colors.items():
            pred_overlay[pred_img == cls_id] = color

        mask_3ch            = np.stack([mask_np] * 3, axis=-1)
        pred_overlay_dimmed = pred_overlay.copy()
        pred_overlay_dimmed[~mask_3ch] *= 0.1

        fig, axes = plt.subplots(2, 4, figsize=(24, 10))
        sigmas_str = ",".join([f"{s:.1f}" for s in sigmas_v])

        if cfg.num_classes == 4:
            retention_str = (f"Global: {global_retention:.2f}% | "
                             f"Michel: {michel_retention:.1f}% | "
                             f"Brems: {brems_retention:.1f}%")
        else:
            retention_str = (f"Global: {global_retention:.2f}% | "
                             f"Michel/Brems: {mb_retention:.1f}%")

        plt.suptitle(
            f"File: {fname} | View: {sample_view} | Mode: {cfg.num_classes}-class\n"
            f"Params: low={low_v:.2f}, high={high_v:.2f}, sigmas=({sigmas_str}), "
            f"maxima_seeds={use_maxima_v}\n"
            f"Retention — {retention_str}",
            fontsize=14, y=0.98,
        )

        im0 = axes[0, 0].imshow(img_np, cmap="viridis",
                                 vmin=-50, vmax=50, origin="lower")
        axes[0, 0].set_title("1. Raw ADC Input", fontsize=12)
        plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

        im1 = axes[0, 1].imshow(z_np, cmap="RdBu_r",
                                 vmin=-5, vmax=5, origin="lower")
        axes[0, 1].set_title(
            f"2. MAD Z-score\n(med={z_med:.1f}, σ_rob={z_sigma:.1f})", fontsize=12)
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

        im2 = axes[0, 2].imshow(score_np, cmap="inferno", origin="lower")
        axes[0, 2].set_title(
            f"3. DoG Blob Score\n(low={low_v:.2f}, high={high_v:.2f})", fontsize=12)
        axes[0, 2].contour(score_np > low_v,  colors="white",
                           linewidths=0.5, alpha=0.6, origin="lower")
        axes[0, 2].contour(score_np > high_v, colors="cyan",
                           linewidths=0.8, alpha=0.8, origin="lower")
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

        axes[0, 3].imshow(allowed_np.astype(float),
                          cmap="gray", origin="lower", alpha=0.7)
        axes[0, 3].contour(seeds_np.astype(float), colors="cyan",
                           linewidths=0.8, alpha=0.9, origin="lower")
        axes[0, 3].contour(mask_np.astype(float), colors="red",
                           linewidths=0.7, alpha=0.8, origin="lower")
        axes[0, 3].set_title(
            "4. Hysteresis\nAllowed(gray) | Seeds(cyan) | Mask(red)", fontsize=12)

        if cfg.num_classes == 2:
            gt_legend = "Red=NOT_MB, Cyan=MB"
        elif cfg.num_classes == 3:
            gt_legend = "Gray=Other, Red=Muon, Cyan=MB"
        else:
            gt_legend = "Gray=Other, Red=Muon, Cyan=Michel, Mag=Brems"

        axes[1, 0].imshow(gt_overlay, origin="lower")
        axes[1, 0].set_title(f"5. Ground Truth\n({gt_legend})", fontsize=12)

        pred_subtitle = (f"Michel: {michel_retention:.1f}% | Brems: {brems_retention:.1f}%"
                         if cfg.num_classes == 4
                         else f"MB Kept: {mb_retention:.1f}%")
        axes[1, 1].imshow(pred_overlay_dimmed, origin="lower")
        axes[1, 1].set_title(
            f"6. Prediction (Dimmed=Filtered)\n{pred_subtitle}", fontsize=12)

        combined = np.zeros((H, W, 3), dtype=np.float32)
        correct  = (pred_img == labels_dense_cpu) & (pred_img >= 0)
        wrong    = (pred_img != labels_dense_cpu) & (pred_img >= 0)
        combined[correct]      = [0.0, 1.0, 0.0]
        combined[wrong]        = [1.0, 0.0, 0.0]
        combined[pred_img < 0] *= 0.1
        axes[1, 2].imshow(combined, origin="lower")
        axes[1, 2].set_title(
            "7. Accuracy\n(Green=Correct, Red=Wrong)", fontsize=12)

        axes[1, 3].axis('off')
        stats_text  = f"Statistics ({cfg.num_classes}-class):\n\n"
        stats_text += f"Global:\n  Total: {total_pixels:,}\n"
        stats_text += f"  Kept: {kept_pixels:,} ({global_retention:.2f}%)\n\n"
        if cfg.num_classes == 4:
            stats_text += (f"Michel (class 2):\n  Total: {total_michel:,}\n"
                           f"  Kept: {kept_michel:,} ({michel_retention:.1f}%)\n\n"
                           f"Brems (class 3):\n  Total: {total_brems:,}\n"
                           f"  Kept: {kept_brems:,} ({brems_retention:.1f}%)\n\n")
        else:
            stats_text += (f"Michel/Brems:\n  Total: {total_mb:,}\n"
                           f"  Kept: {kept_mb:,} ({mb_retention:.1f}%)\n\n")
        stats_text += (f"Parameters:\n"
                       f"  low_thr: {low_v:.2f}\n  high_thr: {high_v:.2f}\n"
                       f"  sigmas: {sigmas_str}\n  maxima_seeds: {use_maxima_v}\n"
                       f"  in_channels: {cfg.in_channels}\n"
                       f"  focal_weight: {cfg.focal_weight:.4f}")
        axes[1, 3].text(
            0.1, 0.9, stats_text, transform=axes[1, 3].transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        base      = fname.split(".")[0]
        prefix    = f"{view}_" if view is not None else ""
        save_path = os.path.join(save_dir, f"{prefix}{base}_viz.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

    print(f"  Saved {len(indices)} visualizations to {save_dir}")


def plot_confusion_matrix(cm, class_names, save_path,
                           normalize=False, norm_axis=1):
    cm_np = cm.numpy()
    if normalize:
        cm_np = cm_np.astype(np.float64)
        sums  = cm_np.sum(axis=norm_axis, keepdims=True)
        sums[sums == 0.0] = 1.0
        cm_np = cm_np / sums
        fmt, title = ".2f", "Normalized Confusion Matrix"
    else:
        cm_np = cm_np.astype(np.int64)
        fmt, title = "d", "Confusion Matrix (Counts)"

    fig_size = (8, 7) if len(class_names) <= 3 else (10, 9)
    plt.figure(figsize=fig_size)
    sns.heatmap(cm_np, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()


def save_checkpoint(model, optimizer, cfg, epoch, metrics, save_dir, is_best=False):
    os.makedirs(save_dir, exist_ok=True)
    model_state = (model.module.state_dict()
                   if isinstance(model, DDP) else model.state_dict())
    checkpoint = {
        "epoch":           epoch,
        "model_state":     model_state,
        "optimizer_state": optimizer.state_dict(),
        "metrics":         metrics,
        "cfg":             cfg.__dict__,
    }
    last_path = os.path.join(save_dir, "last_checkpoint.pt")
    torch.save(checkpoint, last_path)
    if is_best:
        shutil.copy(last_path, os.path.join(save_dir, "best_checkpoint.pt"))
        if cfg.num_classes == 4:
            m_iou = metrics['iou_per_class'][2]
            b_iou = metrics['iou_per_class'][3]
            print(f"    ✓ Best saved (Michel={m_iou:.4f}, Brems={b_iou:.4f}, "
                  f"mIoU={metrics.get('miou', -1):.4f})")
        else:
            mb_iou = metrics['iou_per_class'][cfg.num_classes - 1]
            print(f"    ✓ Best saved (MB IoU={mb_iou:.4f}, "
                  f"mIoU={metrics.get('miou', -1):.4f})")


def load_checkpoint(checkpoint_path, model, optimizer, device):
    if not os.path.exists(checkpoint_path):
        return 0, None
    print(f"  Loading checkpoint from {checkpoint_path}")
    ckpt  = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model_state"]
    if len(state) > 0 and list(state.keys())[0].startswith("module."):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    (model.module if isinstance(model, DDP) else model).load_state_dict(state)
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt["epoch"], ckpt.get("metrics", {})


def seed_everything(seed: int, rank: int = 0):
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_scheduler(optimizer, cfg):
    if cfg.scheduler_type == "cosine":
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.cosine_T0,
            T_mult=cfg.cosine_T_mult,
            eta_min=cfg.cosine_eta_min,
        )
    elif cfg.scheduler_type == "plateau":
        return ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    else:
        raise ValueError(
            f"Unknown scheduler_type '{cfg.scheduler_type}'. "
            f"Choose 'cosine' or 'plateau'.")


def train_worker(gpu, view, cfg: SparseUNetConfig):
    rank       = gpu
    world_size = torch.cuda.device_count()

    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:29501",
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")
    seed_everything(42, rank=rank)

    CLASS_NAMES = get_class_names(cfg.num_classes)

    if rank == 0:
        print(f"\nTraining Sparse UNet (6-scale): View {view} | GPUs={world_size}")
        print(f"  Mode: {cfg.num_classes}-class | {list(CLASS_NAMES.values())}")
        print(f"  Loss: {cfg.loss_type}", end="")
        if cfg.loss_type in ("focal", "combined"):
            print(f" (gamma={cfg.focal_gamma})", end="")
        if cfg.loss_type == "combined":
            print(f" | focal:dice = {cfg.focal_weight:.4f}:{1-cfg.focal_weight:.4f}", end="")
        print()
        print(f"  Scheduler: {cfg.scheduler_type}", end="")
        if cfg.scheduler_type == "cosine":
            print(f" (T0={cfg.cosine_T0}, Tmult={cfg.cosine_T_mult}, "
                  f"eta_min={cfg.cosine_eta_min})")
        else:
            print()
        print(f"  In-channels: {cfg.in_channels}")
        print(f"  Attention: {cfg.use_attention}" +
              (f" (heads={cfg.attn_heads})" if cfg.use_attention else ""))
        print(f"  Planes: {cfg.planes}")
        print(f"  Layers: {cfg.layers}")
        print(f"  Batch size: {cfg.batch_size}")

    train_ds = SparseNeutrinoDataset(
        cfg.raw_root, cfg.cleansig_root, "train", view, cfg)
    val_ds   = SparseNeutrinoDataset(
        cfg.raw_root, cfg.cleansig_root, "val",   view, cfg)

    class_weights = None
    if cfg.manual_class_weights is not None:
        weights = torch.tensor(cfg.manual_class_weights,
                               dtype=torch.float32, device=device)
        if rank == 0:
            print("  Manual class weights:", weights.tolist())
        class_weights = weights
    elif cfg.use_class_weights:
        if rank == 0:
            weights = compute_class_weights_from_dataset(
                train_ds, cfg, device, max_samples=500)
            print("  Auto class weights:",
                  [float(w) for w in weights.tolist()])
        else:
            weights = torch.ones(cfg.num_classes, dtype=torch.float32)
        weights = weights.to(device)
        if world_size > 1:
            dist.broadcast(weights, src=0)
        class_weights = weights
    else:
        if rank == 0:
            print("  No class weights")

    loss_fn = create_loss_fn(cfg, class_weights)

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(
        val_ds,   num_replicas=world_size, rank=rank, shuffle=False)

    from functools import partial
    collate_fn = partial(collate_sparse_fn, cfg=cfg, device=device)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=train_sampler,
        num_workers=0, collate_fn=collate_fn,
        pin_memory=False, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, sampler=val_sampler,
        num_workers=0, collate_fn=collate_fn,
        pin_memory=False, drop_last=False,
    )

    model     = SparseUNet2D(cfg).to(device)
    model     = DDP(model, device_ids=[gpu])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = _build_scheduler(optimizer, cfg)
    scaler    = torch.amp.GradScaler("cuda", enabled=cfg.amp)

    best_metric       = -1.0
    epochs_no_improve = 0

    for epoch in range(cfg.epochs):
        train_sampler.set_epoch(epoch)

        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, cfg, epoch, device,
            rank=rank, loss_fn=loss_fn, class_weights=class_weights,
        )
        val_metrics = eval_epoch(
            model, val_loader, cfg, device,
            rank=rank, loss_fn=loss_fn, class_weights=class_weights,
        )

        # tracking metric: Michel+Brems IoU (weighted for 4-class)
        if cfg.num_classes == 4:
            tracking_metric = (val_metrics["iou_per_class"][2]
                               + 3.0 * val_metrics["iou_per_class"][3]) / 4.0
        else:
            tracking_metric = val_metrics["iou_per_class"][cfg.num_classes - 1]

        if cfg.scheduler_type == "cosine":
            scheduler.step(epoch + 1)   # +1 so restart triggers at epoch boundary
        else:
            scheduler.step(tracking_metric)

        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\n[Epoch {epoch:03d}]  lr={current_lr:.2e}")
            print(f"  Train: loss={train_metrics['loss']:.4f} "
                  f"acc={train_metrics['acc']:.4f} "
                  f"miou={train_metrics['miou']:.4f}")
            print(f"  Val:   loss={val_metrics['loss']:.4f} "
                  f"acc={val_metrics['acc']:.4f} "
                  f"miou={val_metrics['miou']:.4f}")
            for i, name in CLASS_NAMES.items():
                print(f"    IoU[{name}]: {val_metrics['iou_per_class'][i]:.4f}")
            if cfg.num_classes == 4:
                print(f"    Tracking metric: {tracking_metric:.4f}")

            if epoch % 10 == 0:
                viz_dir = os.path.join(cfg.save_dir, f"val_viz_epoch_{epoch:03d}")
                print("  Generating validation visualizations...")
                visualize_predictions(
                    model.module, val_ds, cfg, device, viz_dir,
                    k=cfg.viz_k, view=view)

        is_best = tracking_metric > best_metric
        if is_best:
            best_metric       = tracking_metric
            epochs_no_improve = 0
        else:
            if epoch > cfg.warmup_epochs:
                epochs_no_improve += 1

        if rank == 0:
            save_checkpoint(
                model, optimizer, cfg, epoch, val_metrics,
                cfg.save_dir, is_best=is_best)

        stop_flag = torch.tensor(0, device=device)
        if (rank == 0 and epoch > cfg.warmup_epochs
                and epochs_no_improve >= cfg.patience):
            print(f"\n  Early stopping triggered (patience={cfg.patience})")
            stop_flag.fill_(1)

        dist.broadcast(stop_flag, src=0)
        dist.barrier()
        if stop_flag.item() == 1:
            break

    if rank == 0:
        print(f"\nDone. Best tracking metric: {best_metric:.4f}\n")
    dist.destroy_process_group()


def test_model(view: str, cfg: SparseUNetConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTesting: View {view} | {cfg.num_classes}-class")

    CLASS_NAMES = get_class_names(cfg.num_classes)
    print(f"  {list(CLASS_NAMES.values())}")

    test_ds = SparseNeutrinoDataset(
        cfg.raw_root, cfg.cleansig_root, "test", view, cfg)

    from functools import partial
    collate_fn  = partial(collate_sparse_fn, cfg=cfg, device=device)
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )

    model     = SparseUNet2D(cfg).to(device)
    best_path = os.path.join(cfg.save_dir, "best_checkpoint.pt")

    if not os.path.exists(best_path):
        print(f"  ERROR: No checkpoint at {best_path}")
        return

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Loaded best model from epoch {ckpt.get('epoch', -1)}")

    loss_fn = create_loss_fn(cfg, None)
    metrics = eval_epoch(model, test_loader, cfg, device, rank=0,
                         loss_fn=loss_fn, class_weights=None)

    print(f"\nTest Results:\n  Accuracy: {metrics['acc']:.4f}"
          f"\n  mIoU:     {metrics['miou']:.4f}")
    for i, name in CLASS_NAMES.items():
        print(f"    IoU[{name}]: {metrics['iou_per_class'][i]:.4f}")

    plot_dir = os.path.join(os.path.dirname(cfg.save_dir), f"test_plots_{view}")
    os.makedirs(plot_dir, exist_ok=True)
    plot_confusion_matrix(metrics["cm"], list(CLASS_NAMES.values()),
                          os.path.join(plot_dir, f"{view}_cm_raw.png"),  normalize=False)
    plot_confusion_matrix(metrics["cm"], list(CLASS_NAMES.values()),
                          os.path.join(plot_dir, f"{view}_cm_norm.png"), normalize=True)
    print(f"  Saved confusion matrices to {plot_dir}")

    k_viz = max(1, int(0.1 * len(test_ds)))
    print(f"  Generating visualizations for {k_viz} test events ...")
    visualize_predictions(model, test_ds, cfg, device, plot_dir,
                          k=k_viz, view=view)


def parse_json_or_value(s, type_fn=float):
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return type_fn(s)


def parse_sigmas(s):
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict):
            return {k: tuple(v) for k, v in parsed.items()}
        return tuple(parsed)
    except (json.JSONDecodeError, TypeError):
        return tuple(float(x) for x in s.split(','))


def main():
    parser = argparse.ArgumentParser(
        description="Sparse 2D UNet (6-scale) for Neutrino Segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--raw_root",      type=str, required=True)
    parser.add_argument("--cleansig_root", type=str, required=True)
    parser.add_argument("--out_dir",       type=str, required=True)

    parser.add_argument("--num_classes", type=int, default=3, choices=[2, 3, 4])

    parser.add_argument("--low_thr",            type=str,   default=None)
    parser.add_argument("--high_thr",           type=str,   default=None)
    parser.add_argument("--sigmas",             type=str,   default=None)
    parser.add_argument("--dog_k",              type=float, default=1.6)
    parser.add_argument("--truncate",           type=float, default=3.0)
    parser.add_argument("--no_scale_normalize", action="store_true")
    parser.add_argument("--no_abs_z",           action="store_true")
    parser.add_argument("--hyst_max_iters",     type=int,   default=512)
    parser.add_argument("--use_maxima_seeds",   type=str,   default=None)
    parser.add_argument("--seed_min",           type=str,   default=None)

    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--no_amp",       action="store_true")

    parser.add_argument("--loss_type",    type=str,   default="combined",
                        choices=["ce", "bce", "focal", "dice", "combined"])
    parser.add_argument("--focal_gamma",  type=float, default=2.0)
    parser.add_argument("--focal_weight", type=float, default=20.0 / 21.0,
                        help="Focal weight in combined loss. Default 20/21 ≈ 0.952 "
                             "gives 20:1 focal:dice ratio.")
    parser.add_argument("--dice_smooth",  type=float, default=1.0)
    parser.add_argument("--no_class_weights",     action="store_true")
    parser.add_argument("--manual_class_weights", type=str, default=None)

    parser.add_argument("--scheduler_type", type=str, default="cosine",
                        choices=["cosine", "plateau"])
    parser.add_argument("--cosine_T0",      type=int,   default=20)
    parser.add_argument("--cosine_T_mult",  type=int,   default=2)
    parser.add_argument("--cosine_eta_min", type=float, default=1e-6)

    parser.add_argument("--in_channels",        type=int,   default=4, choices=[1, 4])
    parser.add_argument("--no_attention",        action="store_true")
    parser.add_argument("--attn_heads",          type=int,   default=8)
    parser.add_argument("--attn_dropout",        type=float, default=0.0)
    parser.add_argument("--attn_patch_size",     type=int,   default=64)
    parser.add_argument("--attn_patch_size_d5",  type=int,   default=128)
    parser.add_argument("--attn_patch_size_d4",  type=int,   default=256)

    parser.add_argument("--test_only",     action="store_true")
    parser.add_argument("--view",          type=str, default=None,
                        choices=["U", "V", "Z"])
    parser.add_argument("--patience",      type=int,   default=25)
    parser.add_argument("--warmup_epochs", type=int,   default=5)
    parser.add_argument("--viz_k",         type=int,   default=10)

    args = parser.parse_args()

    mode_str = {2: "2-CLASS", 3: "3-CLASS", 4: "4-CLASS"}[args.num_classes]
    print(f"Sparse UNet (6-scale) — {mode_str} | loss={args.loss_type} | "
          f"scheduler={args.scheduler_type} | in_channels={args.in_channels} | "
          f"attention={not args.no_attention}")

    print("\nInitializing Warp...")
    wp.init()
    print("✓ Done\n")

    low_thr  = parse_json_or_value(args.low_thr,  float) if args.low_thr  \
               else {"U": 0.8, "V": 0.8, "Z": 0.4}
    high_thr = parse_json_or_value(args.high_thr, float) if args.high_thr \
               else {"U": 1.8, "V": 1.8, "Z": 2.8}
    sigmas   = parse_sigmas(args.sigmas) if args.sigmas else {
        "U": (0.8, 1.2, 1.8, 2.6, 3.6),
        "V": (0.8, 1.2, 1.8, 2.6, 3.6),
        "Z": (0.8, 1.2, 1.8, 2.6),
    }
    use_maxima_seeds = parse_json_or_value(args.use_maxima_seeds, bool) \
                       if args.use_maxima_seeds \
                       else {"U": True, "V": True, "Z": False}
    seed_min = parse_json_or_value(args.seed_min, float) \
               if args.seed_min else None

    manual_weights = None
    if args.manual_class_weights:
        try:
            manual_weights = json.loads(args.manual_class_weights)
            if (not isinstance(manual_weights, list)
                    or len(manual_weights) != args.num_classes):
                raise ValueError(f"Expected list of {args.num_classes} values")
            manual_weights = tuple(float(w) for w in manual_weights)
            print(f"Manual class weights: {manual_weights}")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing manual_class_weights: {e}")
            exit(1)

    cfg = SparseUNetConfig(
        raw_root=args.raw_root,
        cleansig_root=args.cleansig_root,
        out_dir=args.out_dir,
        num_classes=args.num_classes,
        low_thr=low_thr,
        high_thr=high_thr,
        sigmas=sigmas,
        dog_k=args.dog_k,
        truncate=args.truncate,
        scale_normalize=(not args.no_scale_normalize),
        use_abs_z=(not args.no_abs_z),
        hyst_max_iters=args.hyst_max_iters,
        use_maxima_seeds=use_maxima_seeds,
        seed_min=seed_min,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        amp=(not args.no_amp),
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        focal_weight=args.focal_weight,
        dice_smooth=args.dice_smooth,
        use_class_weights=(not args.no_class_weights),
        manual_class_weights=manual_weights,
        in_channels=args.in_channels,
        use_attention=(not args.no_attention),
        attn_heads=args.attn_heads,
        attn_dropout=args.attn_dropout,
        attn_patch_size=args.attn_patch_size,
        attn_patch_size_d5=args.attn_patch_size_d5,
        attn_patch_size_d4=args.attn_patch_size_d4,
        scheduler_type=args.scheduler_type,
        cosine_T0=args.cosine_T0,
        cosine_T_mult=args.cosine_T_mult,
        cosine_eta_min=args.cosine_eta_min,
        patience=args.patience,
        warmup_epochs=args.warmup_epochs,
        viz_k=args.viz_k,
        save_dir=os.path.join(args.out_dir, "checkpoints"),
    )

    views  = [args.view] if args.view else list(cfg.views)
    n_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {n_gpus}\n")

    for view in views:
        view_cfg          = cfg
        view_cfg.save_dir = os.path.join(args.out_dir, f"checkpoints_{view}")

        if args.test_only:
            test_model(view, view_cfg)
        else:
            if n_gpus > 1:
                print(f"Starting multi-GPU training for view {view}...")
                mp.spawn(train_worker, args=(view, view_cfg),
                         nprocs=n_gpus, join=True)
            else:
                print(f"Starting single-GPU training for view {view}...")
                train_worker(0, view, view_cfg)

            test_model(view, view_cfg)


if __name__ == "__main__":
    main()
