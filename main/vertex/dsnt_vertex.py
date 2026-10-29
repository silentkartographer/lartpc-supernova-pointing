"""Muon-Michel vertex finding with a DSNT U-Net (DenseNet169 encoder), one model per
wire plane, coupled by a cross-view time-tick constraint.
"""
import os
import glob
import random
import argparse
import shutil
import json
from dataclasses import dataclass, field
from typing import Tuple, Dict, List

from main.paths import OUTPUT_ROOT

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import zoom
from tqdm import tqdm
import segmentation_models_pytorch as smp


@dataclass
class VertexConfig:
    raw_root: str
    out_dir: str
    views: Tuple[str, ...] = ("U", "V", "Z")
    encoder_name: str = "densenet169"
    encoder_weights: str = None
    epochs: int = 100
    batch_size: int = 8
    lr: float = 2.5e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    lambda_js: float = 1.0
    sigma_t: float = 1.0
    lambda_x_var: float = 200.0
    lambda_x_gt: float = 1.0
    num_workers: int = 0
    amp: bool = True
    save_dir: str = "./vertex_checkpoints"
    patience: int = 15
    warmup_epochs: int = 5
    viz_k: int = 10


class VertexDataset(Dataset):
    def __init__(self, raw_root: str, split: str, view: str):
        super().__init__()
        self.split = split
        self.view = view.upper()

        raw_dir = os.path.join(raw_root, split, "all")
        if not os.path.isdir(raw_dir):
            raw_dir = os.path.join(raw_root, split)
            if not os.path.isdir(raw_dir):
                raise FileNotFoundError(f"Missing directory: {raw_dir}")

        self.file_paths = sorted(glob.glob(os.path.join(raw_dir, "*.npz")))
        if len(self.file_paths) == 0:
            raise RuntimeError(f"No NPZ files found for {split}/{view}")

        print(f"  [{split}/{view}] Found {len(self.file_paths)} samples")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        try:
            with np.load(path, allow_pickle=True) as data:
                metadata = data['metadata']
                if metadata.size == 0:
                    return None

                meta_dict = metadata.item()
                michel_info = meta_dict.get('per_view', {}).get(self.view, {}).get('michel', {})

                if not michel_info or not isinstance(michel_info, dict):
                    return None

                vwno = michel_info.get('vwno', None)
                vtck = michel_info.get('vtck', None)
                if vwno is None or vtck is None:
                    return None

                img_np = data[f"{self.view}_data"]
                H, W = img_np.shape

                vertex_wire = float(vwno)

                start = float(michel_info.get('start', 0.0) or 0.0)
                vertex_tick = float(vtck) - start

                vertex_wire = float(np.clip(vertex_wire, 0, H - 1))
                vertex_tick = float(np.clip(vertex_tick, 0, W - 1))

                return {
                    'image': torch.from_numpy(img_np).float().unsqueeze(0),
                    'vertex_wire': vertex_wire,
                    'vertex_tick': vertex_tick,
                    'original_shape': img_np.shape,
                    'fname': os.path.basename(path),
                    'view': self.view,
                }

        except Exception:
            return None


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    return {
        'images': torch.stack([b['image'] for b in batch]),
        'vertex_wires': torch.tensor([b['vertex_wire'] for b in batch], dtype=torch.float32),
        'vertex_ticks': torch.tensor([b['vertex_tick'] for b in batch], dtype=torch.float32),
        'shapes': [b['original_shape'] for b in batch],
        'fnames': [b['fname'] for b in batch],
        'views': [b['view'] for b in batch],
    }


def spatial_softmax2d(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    B, C, H, W = logits.shape
    assert C == 1
    logits_flat = logits.view(B, -1) / temperature
    return F.softmax(logits_flat, dim=1).view(B, 1, H, W)


def dsnt_expectation(probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    B, C, H, W = probs.shape
    assert C == 1
    device = probs.device

    y_coords = torch.linspace(-1, 1, H, device=device).view(H, 1).expand(H, W)
    x_coords = torch.linspace(-1, 1, W, device=device).view(1, W).expand(H, W)

    p = probs.squeeze(1)
    coords_y = (p * y_coords).sum(dim=(1, 2))
    coords_x = (p * x_coords).sum(dim=(1, 2))

    return coords_x, coords_y


def coords_to_pixels(coords_norm: torch.Tensor, size: int) -> torch.Tensor:
    return (coords_norm + 1.0) * (size - 1) / 2.0


def pixels_to_coords(coords_pixels: torch.Tensor, size: int) -> torch.Tensor:
    return 2.0 * coords_pixels / (size - 1) - 1.0


def jensen_shannon_divergence(probs: torch.Tensor,
                               target_coords: Tuple[torch.Tensor, torch.Tensor],
                               sigma: float) -> torch.Tensor:
    B, C, H, W = probs.shape
    device = probs.device
    coords_x_gt, coords_y_gt = target_coords

    y_coords = torch.linspace(-1, 1, H, device=device).view(H, 1).expand(H, W)
    x_coords = torch.linspace(-1, 1, W, device=device).view(1, W).expand(H, W)

    target_probs = []
    for b in range(B):
        dx = x_coords - coords_x_gt[b]
        dy = y_coords - coords_y_gt[b]
        g = torch.exp(-(dx**2 + dy**2) / (2 * sigma**2))
        target_probs.append(g / (g.sum() + 1e-12))

    target_probs = torch.stack(target_probs).unsqueeze(1)

    p = probs.view(B, -1)
    q = target_probs.view(B, -1)
    m = 0.5 * (p + q)

    kl_pm = (p * (torch.log(p + 1e-12) - torch.log(m + 1e-12))).sum(dim=1)
    kl_qm = (q * (torch.log(q + 1e-12) - torch.log(m + 1e-12))).sum(dim=1)

    return (0.5 * (kl_pm + kl_qm)).mean()


class UNetDSNT(nn.Module):
    def __init__(self, encoder_name: str = "resnet18", encoder_weights: str = "imagenet"):
        super().__init__()

        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=1,
            classes=int(64*4),
        )
        self.heatmap_head = nn.Conv2d(int(64*4), 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.unet(x)
        logits = self.heatmap_head(feats)

        probs = spatial_softmax2d(logits)
        coords_x, coords_y = dsnt_expectation(probs)
        coords = torch.stack([coords_x, coords_y], dim=1)

        return logits, coords


def compute_loss_single_view(logits, coords_pred, vertex_wire, vertex_tick,
                              image_shape, cfg: VertexConfig) -> Dict[str, torch.Tensor]:
    H_img, W_img = image_shape

    coords_y_gt = pixels_to_coords(vertex_wire, H_img)
    coords_x_gt = pixels_to_coords(vertex_tick, W_img)

    coords_x_pred, coords_y_pred = coords_pred[:, 0], coords_pred[:, 1]
    loss_euc = torch.sqrt((coords_x_pred - coords_x_gt)**2 + (coords_y_pred - coords_y_gt)**2).mean()

    probs = spatial_softmax2d(logits)
    sigma_norm = ((2.0 * cfg.sigma_t / H_img) + (2.0 * cfg.sigma_t / W_img)) / 2.0
    loss_js = jensen_shannon_divergence(probs, (coords_x_gt, coords_y_gt), sigma_norm)

    return {
        'loss_total': loss_euc + cfg.lambda_js * loss_js,
        'loss_euc': loss_euc,
        'loss_js': loss_js,
        'coords_x_pred': coords_x_pred,
        'coords_y_pred': coords_y_pred,
        'coords_x_gt': coords_x_gt,
        'coords_y_gt': coords_y_gt,
    }


def compute_multiview_constraints(losses_dict: Dict, cfg: VertexConfig) -> torch.Tensor:
    # tick coordinate should be consistent across all three views
    coords_x_all = torch.stack([losses_dict[v]['coords_x_pred'] for v in cfg.views], dim=1)
    coords_x_gt = losses_dict[cfg.views[0]]['coords_x_gt']

    mean_x = coords_x_all.mean(dim=1)
    var_x = ((coords_x_all - mean_x.unsqueeze(1))**2).mean()

    loss_x_var = cfg.lambda_x_var * var_x
    loss_x_gt = cfg.lambda_x_gt * ((mean_x - coords_x_gt)**2).mean()

    return loss_x_var + loss_x_gt


def train_epoch_multiview(models, loaders, optimizers, scaler, cfg: VertexConfig,
                           epoch: int, device, rank: int = 0) -> Dict[str, float]:
    for m in models.values():
        m.train()

    total_losses = {v: 0.0 for v in cfg.views}
    total_combined = 0.0
    n_batches = 0

    iters = {v: iter(loaders[v]) for v in cfg.views}
    min_batches = min(len(loaders[v]) for v in cfg.views)

    pbar = tqdm(range(min_batches), desc=f"Train {epoch:03d}", dynamic_ncols=True, leave=False) \
        if rank == 0 else range(min_batches)

    for _ in pbar:
        batches = {}
        for v in cfg.views:
            try:
                batches[v] = next(iters[v])
            except StopIteration:
                continue

        if len(batches) != len(cfg.views) or any(b is None for b in batches.values()):
            continue

        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)

        losses_dict = {}
        with torch.cuda.amp.autocast(enabled=cfg.amp):
            for v in cfg.views:
                b = batches[v]
                images = b['images'].to(device, non_blocking=True)
                v_wires = b['vertex_wires'].to(device, non_blocking=True)
                v_ticks = b['vertex_ticks'].to(device, non_blocking=True)

                logits, coords_pred = models[v](images)
                losses_dict[v] = compute_loss_single_view(
                    logits, coords_pred, v_wires, v_ticks, b['shapes'][0], cfg
                )

            loss_combined = sum(ld['loss_total'] for ld in losses_dict.values()) + \
                            compute_multiview_constraints(losses_dict, cfg)

        if scaler is not None:
            scaler.scale(loss_combined).backward()
            for v in cfg.views:
                if cfg.grad_clip_norm is not None:
                    scaler.unscale_(optimizers[v])
                    torch.nn.utils.clip_grad_norm_(models[v].parameters(), cfg.grad_clip_norm)
                scaler.step(optimizers[v])
            scaler.update()
        else:
            loss_combined.backward()
            for v in cfg.views:
                if cfg.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(models[v].parameters(), cfg.grad_clip_norm)
                optimizers[v].step()

        for v in cfg.views:
            total_losses[v] += losses_dict[v]['loss_total'].item()
        total_combined += loss_combined.item()
        n_batches += 1

        if rank == 0 and hasattr(pbar, 'set_postfix'):
            pbar.set_postfix({v: f"{total_losses[v]/n_batches:.4f}" for v in cfg.views} |
                             {'combined': f"{total_combined/n_batches:.4f}"})

    if dist.is_initialized() and dist.get_world_size() > 1:
        vals = [total_losses[v] for v in cfg.views] + [total_combined, n_batches]
        t = torch.tensor(vals, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        for i, v in enumerate(cfg.views):
            total_losses[v] = t[i].item()
        total_combined = t[-2].item()
        n_batches = int(t[-1].item())

    n = max(1, n_batches)
    avg = {v: total_losses[v] / n for v in cfg.views}
    avg['combined'] = total_combined / n
    return avg


@torch.no_grad()
def eval_epoch_multiview(models, loaders, cfg: VertexConfig, device, rank: int = 0) -> Dict[str, float]:
    for m in models.values():
        m.eval()

    total_errors = {v: 0.0 for v in cfg.views}
    total_losses = {v: 0.0 for v in cfg.views}
    total_combined = 0.0
    n_batches = 0

    iters = {v: iter(loaders[v]) for v in cfg.views}
    min_batches = min(len(loaders[v]) for v in cfg.views)

    pbar = tqdm(range(min_batches), desc="Val", dynamic_ncols=True, leave=False) \
        if rank == 0 else range(min_batches)

    for _ in pbar:
        batches = {}
        for v in cfg.views:
            try:
                batches[v] = next(iters[v])
            except StopIteration:
                continue

        if len(batches) != len(cfg.views) or any(b is None for b in batches.values()):
            continue

        losses_dict = {}
        for v in cfg.views:
            b = batches[v]
            images = b['images'].to(device, non_blocking=True)
            v_wires = b['vertex_wires'].to(device, non_blocking=True)
            v_ticks = b['vertex_ticks'].to(device, non_blocking=True)
            H_img, W_img = b['shapes'][0]

            logits, coords_pred = models[v](images)
            ld = compute_loss_single_view(logits, coords_pred, v_wires, v_ticks, b['shapes'][0], cfg)
            losses_dict[v] = ld
            total_losses[v] += ld['loss_total'].item()

            x_px = coords_to_pixels(coords_pred[:, 0], W_img)
            y_px = coords_to_pixels(coords_pred[:, 1], H_img)
            total_errors[v] += torch.sqrt((x_px - v_ticks)**2 + (y_px - v_wires)**2).mean().item()

        loss_combined = sum(ld['loss_total'] for ld in losses_dict.values()) + \
                        compute_multiview_constraints(losses_dict, cfg)
        total_combined += loss_combined.item()
        n_batches += 1

    if dist.is_initialized() and dist.get_world_size() > 1:
        vals = ([total_errors[v] for v in cfg.views] +
                [total_losses[v] for v in cfg.views] +
                [total_combined, n_batches])
        t = torch.tensor(vals, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        nv = len(cfg.views)
        for i, v in enumerate(cfg.views):
            total_errors[v] = t[i].item()
            total_losses[v] = t[nv + i].item()
        total_combined = t[-2].item()
        n_batches = int(t[-1].item())

    n = max(1, n_batches)
    result = {v: total_errors[v] / n for v in cfg.views}
    result['mean'] = sum(result.values()) / len(cfg.views)
    result['combined_loss'] = total_combined / n
    return result


def save_checkpoint(models, optimizers, cfg: VertexConfig, epoch: int,
                     metrics: Dict, save_dir: str, is_best: bool = False):
    os.makedirs(save_dir, exist_ok=True)
    for v in cfg.views:
        state = models[v].module.state_dict() if isinstance(models[v], DDP) else models[v].state_dict()
        ckpt = {
            'epoch': epoch,
            'model_state': state,
            'optimizer_state': optimizers[v].state_dict(),
            'metrics': metrics,
            'cfg': cfg.__dict__,
        }
        view_dir = os.path.join(save_dir, f"checkpoints_{v}")
        os.makedirs(view_dir, exist_ok=True)
        last = os.path.join(view_dir, "last_checkpoint.pt")
        torch.save(ckpt, last)
        if is_best:
            shutil.copy(last, os.path.join(view_dir, "best_checkpoint.pt"))


def load_checkpoint(checkpoint_path: str, model: nn.Module,
                     optimizer: torch.optim.Optimizer, device):
    if not os.path.exists(checkpoint_path):
        return 0, None

    print(f"  Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    state = ckpt['model_state']
    if list(state.keys())[0].startswith('module.'):
        state = {k.replace('module.', ''): v for k, v in state.items()}

    target = model.module if isinstance(model, DDP) else model
    target.load_state_dict(state)
    optimizer.load_state_dict(ckpt['optimizer_state'])

    return ckpt['epoch'], ckpt.get('metrics', {})


def seed_everything(seed: int, rank: int = 0):
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def debug_dataset_coordinates(dataset: Dataset, n_samples: int = 100):
    print(f"\nDEBUG: {dataset.view} view")

    wires, ticks, shapes = [], [], []
    for i in range(min(n_samples, len(dataset))):
        s = dataset[i]
        if s is None:
            continue
        wires.append(s['vertex_wire'])
        ticks.append(s['vertex_tick'])
        shapes.append(s['original_shape'])

    wires = np.array(wires)
    ticks = np.array(ticks)
    unique_shapes = set(shapes)

    print(f"  Valid: {len(wires)}  shapes: {unique_shapes}")
    print(f"  Wire (y): [{wires.min():.1f}, {wires.max():.1f}], mean={wires.mean():.1f}")
    print(f"  Tick (x): [{ticks.min():.1f}, {ticks.max():.1f}], mean={ticks.mean():.1f}")

    if len(unique_shapes) == 1:
        H, W = list(unique_shapes)[0]
        in_y = ((wires >= 0) & (wires < H)).mean() * 100
        in_x = ((ticks >= 0) & (ticks < W)).mean() * 100
        print(f"  In-bounds: wires {in_y:.1f}%  ticks {in_x:.1f}%")
        if in_y < 100:
            print(f"  ⚠ wire max={wires.max():.0f} but H={H}")
        if in_x < 100:
            print(f"  ⚠ tick max={ticks.max():.0f} but W={W}")
    print()


@torch.no_grad()
def visualize_predictions(models, datasets, cfg: VertexConfig, device, epoch: int, k: int = 10):
    for m in models.values():
        m.eval()

    viz_dir = os.path.join(cfg.out_dir, 'visualizations', f'epoch_{epoch:03d}')
    os.makedirs(viz_dir, exist_ok=True)

    for view in cfg.views:
        view_dir = os.path.join(viz_dir, view)
        os.makedirs(view_dir, exist_ok=True)
        dataset = datasets[view]

        for idx in random.sample(range(len(dataset)), min(k, len(dataset))):
            sample = dataset[idx]
            if sample is None:
                continue

            image = sample['image'].unsqueeze(0).to(device)
            gt_wire = sample['vertex_wire']
            gt_tick = sample['vertex_tick']
            H, W = sample['original_shape']
            fname = sample['fname']

            logits, coords_pred = models[view](image)

            px_x = coords_to_pixels(coords_pred[0, 0], W).item()
            px_y = coords_to_pixels(coords_pred[0, 1], H).item()
            err = np.sqrt((px_x - gt_tick)**2 + (px_y - gt_wire)**2)

            probs = spatial_softmax2d(logits)
            heatmap = probs[0, 0].cpu().numpy()
            img_np = image[0, 0].cpu().numpy()

            if heatmap.shape != (H, W):
                heatmap = zoom(heatmap, (H / heatmap.shape[0], W / heatmap.shape[1]), order=1)

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            im0 = axes[0].imshow(img_np, cmap='viridis', origin='lower', aspect='auto')
            axes[0].plot(gt_tick, gt_wire, 'r+', markersize=20, markeredgewidth=3,
                         label=f'GT ({gt_tick:.1f}, {gt_wire:.1f})')
            axes[0].set(title=f'Input | View {view}', xlabel='Tick (x)', ylabel='Wire (y)')
            axes[0].legend(loc='upper right')
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            im1 = axes[1].imshow(heatmap, cmap='hot', origin='lower', aspect='auto')
            axes[1].plot(gt_tick, gt_wire, 'g+', markersize=20, markeredgewidth=3, label='GT')
            axes[1].plot(px_x, px_y, 'c+', markersize=20, markeredgewidth=3,
                         label=f'Pred ({px_x:.1f}, {px_y:.1f})')
            axes[1].set(title='Heatmap (DSNT)', xlabel='Tick (x)', ylabel='Wire (y)')
            axes[1].legend(loc='upper right')
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            im2 = axes[2].imshow(img_np, cmap='viridis', origin='lower', aspect='auto', alpha=0.7)
            for (tx, ty, color) in [(gt_tick, gt_wire, 'g'), (px_x, px_y, 'm')]:
                axes[2].plot(tx, ty, f'{color}o', markersize=15, markerfacecolor='none', markeredgewidth=3)
                axes[2].plot([tx-5, tx+5], [ty, ty], f'{color}-', linewidth=2)
                axes[2].plot([tx, tx], [ty-5, ty+5], f'{color}-', linewidth=2)
            axes[2].plot([gt_tick, px_x], [gt_wire, px_y], 'r--', linewidth=2, alpha=0.7,
                         label=f'Error: {err:.2f} px')
            axes[2].set(title=f'Overlay | err={err:.2f} px', xlabel='Tick (x)', ylabel='Wire (y)')
            axes[2].legend(loc='upper right')
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

            plt.suptitle(f'Epoch {epoch} | {view} | {fname}\n'
                         f'GT: ({gt_tick:.1f}, {gt_wire:.1f})  Pred: ({px_x:.1f}, {px_y:.1f})  '
                         f'Error: {err:.2f} px', fontsize=14, y=1.02)
            plt.tight_layout()
            plt.savefig(os.path.join(view_dir, fname.replace('.npz', '_viz.png')),
                        dpi=150, bbox_inches='tight')
            plt.close(fig)

    print(f"  Saved visualizations to {viz_dir}")


@torch.no_grad()
def test_multiview(models, test_datasets, cfg: VertexConfig, device,
                   out_dir: str, visualize_pct: float = 0.3):
    print("\nTEST EVALUATION")
    for m in models.values():
        m.eval()

    results = {v: {'errors': [], 'gt_wires': [], 'gt_ticks': [],
                   'pred_wires': [], 'pred_ticks': [], 'fnames': []}
               for v in cfg.views}

    test_dir = os.path.join(out_dir, 'test_results')
    os.makedirs(test_dir, exist_ok=True)

    for v in cfg.views:
        print(f"\nEvaluating {v}...")
        dataset = test_datasets[v]

        for idx in tqdm(range(len(dataset)), desc=f"Test {v}", dynamic_ncols=True):
            s = dataset[idx]
            if s is None:
                continue

            image = s['image'].unsqueeze(0).to(device)
            gt_wire, gt_tick = s['vertex_wire'], s['vertex_tick']
            H, W = s['original_shape']

            _, coords_pred = models[v](image)
            px_x = coords_to_pixels(coords_pred[0, 0], W).item()
            px_y = coords_to_pixels(coords_pred[0, 1], H).item()
            err = np.sqrt((px_x - gt_tick)**2 + (px_y - gt_wire)**2)

            results[v]['errors'].append(err)
            results[v]['gt_wires'].append(gt_wire)
            results[v]['gt_ticks'].append(gt_tick)
            results[v]['pred_wires'].append(px_y)
            results[v]['pred_ticks'].append(px_x)
            results[v]['fnames'].append(s['fname'])

    print("\nRESULTS")
    stats = {}
    for v in cfg.views:
        e = np.array(results[v]['errors'])
        stats[v] = {
            'mean_error': float(np.mean(e)), 'median_error': float(np.median(e)),
            'std_error': float(np.std(e)), 'min_error': float(np.min(e)),
            'max_error': float(np.max(e)), 'p95_error': float(np.percentile(e, 95)),
            'p99_error': float(np.percentile(e, 99)), 'n_samples': len(e),
        }
        print(f"  {v}: n={stats[v]['n_samples']}  mean={stats[v]['mean_error']:.2f}  "
              f"median={stats[v]['median_error']:.2f}  std={stats[v]['std_error']:.2f}  "
              f"p95={stats[v]['p95_error']:.2f}  p99={stats[v]['p99_error']:.2f} px")

    all_errors = np.concatenate([np.array(results[v]['errors']) for v in cfg.views])
    stats['overall'] = {
        'mean_error': float(np.mean(all_errors)), 'median_error': float(np.median(all_errors)),
        'std_error': float(np.std(all_errors)), 'p95_error': float(np.percentile(all_errors, 95)),
        'p99_error': float(np.percentile(all_errors, 99)),
    }
    print(f"\n  All views: mean={stats['overall']['mean_error']:.2f}  "
          f"median={stats['overall']['median_error']:.2f}  "
          f"p95={stats['overall']['p95_error']:.2f} px")

    with open(os.path.join(test_dir, 'test_statistics.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    for v in cfg.views:
        with open(os.path.join(test_dir, f'{v}_predictions.csv'), 'w') as f:
            f.write('filename,gt_wire,gt_tick,pred_wire,pred_tick,error_px\n')
            for i in range(len(results[v]['fnames'])):
                f.write(f"{results[v]['fnames'][i]},"
                        f"{results[v]['gt_wires'][i]:.2f},{results[v]['gt_ticks'][i]:.2f},"
                        f"{results[v]['pred_wires'][i]:.2f},{results[v]['pred_ticks'][i]:.2f},"
                        f"{results[v]['errors'][i]:.2f}\n")

    viz_dir = os.path.join(test_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    for v in cfg.views:
        view_dir = os.path.join(viz_dir, v)
        os.makedirs(view_dir, exist_ok=True)
        dataset = test_datasets[v]
        viz_indices = np.linspace(0, len(dataset) - 1, int(len(dataset) * visualize_pct), dtype=int)
        print(f"  Visualizing {len(viz_indices)} samples for {v}...")

        for idx in tqdm(viz_indices, desc=f"Viz {v}", dynamic_ncols=True):
            s = dataset[idx]
            if s is None:
                continue

            image = s['image'].unsqueeze(0).to(device)
            gt_wire, gt_tick = s['vertex_wire'], s['vertex_tick']
            H, W = s['original_shape']
            fname = s['fname']

            logits, coords_pred = models[v](image)
            px_x = coords_to_pixels(coords_pred[0, 0], W).item()
            px_y = coords_to_pixels(coords_pred[0, 1], H).item()
            err = np.sqrt((px_x - gt_tick)**2 + (px_y - gt_wire)**2)

            probs = spatial_softmax2d(logits)
            heatmap = probs[0, 0].cpu().numpy()
            if heatmap.shape != (H, W):
                heatmap = zoom(heatmap, (H / heatmap.shape[0], W / heatmap.shape[1]), order=1)
            img_np = image[0, 0].cpu().numpy()

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            im0 = axes[0].imshow(img_np, cmap='viridis', origin='lower', aspect='auto')
            axes[0].plot(gt_tick, gt_wire, 'r+', markersize=20, markeredgewidth=3,
                         label=f'GT ({gt_tick:.1f}, {gt_wire:.1f})')
            axes[0].set(title=f'Input | View {v}', xlabel='Tick (x)', ylabel='Wire (y)')
            axes[0].legend(loc='upper right')
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            im1 = axes[1].imshow(heatmap, cmap='hot', origin='lower', aspect='auto')
            axes[1].plot(gt_tick, gt_wire, 'g+', markersize=20, markeredgewidth=3, label='GT')
            axes[1].plot(px_x, px_y, 'c+', markersize=20, markeredgewidth=3,
                         label=f'Pred ({px_x:.1f}, {px_y:.1f})')
            axes[1].set(title='Heatmap', xlabel='Tick (x)', ylabel='Wire (y)')
            axes[1].legend(loc='upper right')
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            im2 = axes[2].imshow(img_np, cmap='viridis', origin='lower', aspect='auto', alpha=0.7)
            for (tx, ty, color) in [(gt_tick, gt_wire, 'g'), (px_x, px_y, 'm')]:
                axes[2].plot(tx, ty, f'{color}o', markersize=15, markerfacecolor='none', markeredgewidth=3)
                axes[2].plot([tx-5, tx+5], [ty, ty], f'{color}-', linewidth=2)
                axes[2].plot([tx, tx], [ty-5, ty+5], f'{color}-', linewidth=2)
            axes[2].plot([gt_tick, px_x], [gt_wire, px_y], 'r--', linewidth=2, alpha=0.7,
                         label=f'Error: {err:.2f} px')
            axes[2].set(title=f'Overlay | err={err:.2f} px', xlabel='Tick (x)', ylabel='Wire (y)')
            axes[2].legend(loc='upper right')
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

            plt.suptitle(f'TEST | {v} | {fname}\n'
                         f'GT: ({gt_tick:.1f}, {gt_wire:.1f})  Pred: ({px_x:.1f}, {px_y:.1f})  '
                         f'Error: {err:.2f} px', fontsize=14, y=1.02)
            plt.tight_layout()
            plt.savefig(os.path.join(view_dir, fname.replace('.npz', '_test.png')),
                        dpi=150, bbox_inches='tight')
            plt.close(fig)

    print(f"\nSaved test results to {test_dir}\n")
    return stats


def train_worker(gpu: int, cfg: VertexConfig):
    rank = gpu
    world_size = torch.cuda.device_count()

    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:29500',
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(gpu)
    device = torch.device(f'cuda:{gpu}')
    seed_everything(42, rank=rank)

    if rank == 0:
        print(f"Multi-View Vertex Prediction | GPUs={world_size}  views={cfg.views}  "
              f"encoder={cfg.encoder_name}  bs={cfg.batch_size}")
        print(f"  lambda_js={cfg.lambda_js}  sigma={cfg.sigma_t}  "
              f"lambda_x_var={cfg.lambda_x_var}  lambda_x_gt={cfg.lambda_x_gt}\n")

    train_datasets = {v: VertexDataset(cfg.raw_root, 'train', v) for v in cfg.views}
    val_datasets   = {v: VertexDataset(cfg.raw_root, 'val',   v) for v in cfg.views}

    if rank == 0:
        for v in cfg.views:
            debug_dataset_coordinates(val_datasets[v])

    train_samplers = {v: DistributedSampler(train_datasets[v], world_size, rank, shuffle=True)
                      for v in cfg.views}
    val_samplers   = {v: DistributedSampler(val_datasets[v],   world_size, rank, shuffle=False)
                      for v in cfg.views}

    loader_kw = dict(num_workers=cfg.num_workers, collate_fn=collate_fn, pin_memory=True)
    train_loaders = {v: DataLoader(train_datasets[v], batch_size=cfg.batch_size,
                                    sampler=train_samplers[v], drop_last=True, **loader_kw)
                     for v in cfg.views}
    val_loaders   = {v: DataLoader(val_datasets[v], batch_size=cfg.batch_size,
                                    sampler=val_samplers[v],   drop_last=False, **loader_kw)
                     for v in cfg.views}

    models = {v: DDP(UNetDSNT(cfg.encoder_name, cfg.encoder_weights).to(device),
                     device_ids=[gpu])
              for v in cfg.views}

    optimizers = {v: torch.optim.Adam(models[v].parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
                  for v in cfg.views}
    schedulers = {v: ReduceLROnPlateau(optimizers[v], mode='min', factor=0.5, patience=5)
                  for v in cfg.views}

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(cfg.epochs):
        for s in train_samplers.values():
            s.set_epoch(epoch)

        train_losses = train_epoch_multiview(models, train_loaders, optimizers, scaler,
                                              cfg, epoch, device, rank)
        val_metrics  = eval_epoch_multiview(models, val_loaders, cfg, device, rank)

        for v in cfg.views:
            schedulers[v].step(val_metrics['combined_loss'])

        if rank == 0:
            print(f"[Epoch {epoch:03d}]  "
                  f"train: {', '.join(f'{v}={train_losses[v]:.4f}' for v in cfg.views)}"
                  f"  combined={train_losses['combined']:.4f}")
            print(f"  val err (px): {', '.join(f'{v}={val_metrics[v]:.4f}' for v in cfg.views)}"
                  f"  mean={val_metrics['mean']:.4f}  loss={val_metrics['combined_loss']:.4f}")

            if epoch % 10 == 0 or epoch == 0:
                unwrapped = {v: m.module if isinstance(m, DDP) else m for v, m in models.items()}
                visualize_predictions(unwrapped, val_datasets, cfg, device, epoch, k=cfg.viz_k)

        is_best = val_metrics['combined_loss'] < best_loss
        if is_best:
            best_loss = val_metrics['combined_loss']
            epochs_no_improve = 0
        elif epoch > cfg.warmup_epochs:
            epochs_no_improve += 1

        if rank == 0:
            save_checkpoint(models, optimizers, cfg, epoch, val_metrics, cfg.save_dir, is_best)
            if is_best:
                print(f"  ✓ best model (val combined loss={best_loss:.4f})")

        stop = torch.tensor(0, device=device)
        if rank == 0 and epoch > cfg.warmup_epochs and epochs_no_improve >= cfg.patience:
            print(f"\n  Early stopping (patience={cfg.patience})")
            stop.fill_(1)
        dist.broadcast(stop, src=0)
        dist.barrier()
        if stop.item() == 1:
            break

    if rank == 0:
        print(f"\nDone. Best val combined loss: {best_loss:.4f}\n")

        test_datasets = {v: VertexDataset(cfg.raw_root, 'test', v) for v in cfg.views}

        for v in cfg.views:
            ckpt = os.path.join(cfg.save_dir, f"checkpoints_{v}", "best_checkpoint.pt")
            if os.path.exists(ckpt):
                load_checkpoint(ckpt, models[v], optimizers[v], device)
            else:
                print(f"  Warning: no best checkpoint for {v}, using current weights")

        unwrapped = {v: m.module if isinstance(m, DDP) else m for v, m in models.items()}
        test_multiview(unwrapped, test_datasets, cfg, device, cfg.out_dir, visualize_pct=0.3)

    if dist.is_initialized():
        dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--raw_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str,
        default=str(OUTPUT_ROOT / "vertex"))

    parser.add_argument("--encoder_name",    type=str,   default="densenet169")
    parser.add_argument("--encoder_weights", type=str,   default="None")
    parser.add_argument("--epochs",          type=int,   default=600)
    parser.add_argument("--batch_size",      type=int,   default=8)
    parser.add_argument("--lr",              type=float, default=2.5e-4)
    parser.add_argument("--weight_decay",    type=float, default=1e-4)
    parser.add_argument("--lambda_js",       type=float, default=1.0)
    parser.add_argument("--sigma_t",         type=float, default=1.0)
    parser.add_argument("--lambda_x_var",    type=float, default=200.0)
    parser.add_argument("--lambda_x_gt",     type=float, default=1.0)
    parser.add_argument("--patience",        type=int,   default=25)
    parser.add_argument("--warmup_epochs",   type=int,   default=5)
    parser.add_argument("--no_amp",          action="store_true")

    args = parser.parse_args()

    cfg = VertexConfig(
        raw_root=args.raw_root,
        out_dir=args.out_dir,
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights if args.encoder_weights.lower() != "none" else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_js=args.lambda_js,
        sigma_t=args.sigma_t,
        lambda_x_var=args.lambda_x_var,
        lambda_x_gt=args.lambda_x_gt,
        patience=args.patience,
        warmup_epochs=args.warmup_epochs,
        amp=not args.no_amp,
        save_dir=os.path.join(args.out_dir, "checkpoints_densenet169"),
    )

    n_gpus = torch.cuda.device_count()
    print(f"GPUs available: {n_gpus}")

    if n_gpus > 1:
        mp.spawn(train_worker, args=(cfg,), nprocs=n_gpus, join=True)
    else:
        train_worker(0, cfg)


if __name__ == "__main__":
    main()
