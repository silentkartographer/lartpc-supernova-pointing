"""Evaluate the vertex models (predicted vs. true tick/wire hexbins, error histograms,
cross-view tick spread).
"""
import os
import argparse
import json
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import segmentation_models_pytorch as smp

from main.paths import DEFAULT_VERTEX_DIR, OUTPUT_ROOT


def spatial_softmax2d(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    B, C, H, W = logits.shape
    return F.softmax(logits.view(B, -1) / temperature, dim=1).view(B, 1, H, W)


def dsnt_expectation(probs: torch.Tensor):
    B, C, H, W = probs.shape
    device = probs.device
    y_coords = torch.linspace(-1, 1, H, device=device).view(H, 1).expand(H, W)
    x_coords = torch.linspace(-1, 1, W, device=device).view(1, W).expand(H, W)
    p = probs.squeeze(1)
    return (p * x_coords).sum(dim=(1, 2)), (p * y_coords).sum(dim=(1, 2))


def coords_to_pixels(coords_norm: torch.Tensor, size: int) -> torch.Tensor:
    return (coords_norm + 1.0) * (size - 1) / 2.0


class UNetDSNT(nn.Module):
    def __init__(self, encoder_name="densenet169", encoder_weights=None):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=1,
            classes=int(64*4),
        )
        self.heatmap_head = nn.Conv2d(int(64*4), 1, kernel_size=1)

    def forward(self, x):
        logits = self.heatmap_head(self.unet(x))
        probs = spatial_softmax2d(logits)
        cx, cy = dsnt_expectation(probs)
        return logits, torch.stack([cx, cy], dim=1)


class VertexDataset(Dataset):
    def __init__(self, raw_root, split, view):
        self.view = view.upper()
        raw_dir = os.path.join(raw_root, split, "all")
        if not os.path.isdir(raw_dir):
            raw_dir = os.path.join(raw_root, split)
        self.file_paths = sorted(glob.glob(os.path.join(raw_dir, "*.npz")))
        if not self.file_paths:
            raise RuntimeError(f"No NPZ files in {raw_dir}")
        print(f"  [{split}/{view}] {len(self.file_paths)} samples")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        try:
            with np.load(self.file_paths[idx], allow_pickle=True) as data:
                meta = data['metadata']
                if meta.size == 0:
                    return None
                meta = meta.item()
                michel = meta.get('per_view', {}).get(self.view, {}).get('michel', {})
                if not michel or not isinstance(michel, dict):
                    return None
                vwno = michel.get('vwno')
                vtck = michel.get('vtck')
                if vwno is None or vtck is None:
                    return None
                img = data[f"{self.view}_data"]
                H, W = img.shape
                start = float(michel.get('start', 0.0) or 0.0)
                wire = float(np.clip(float(vwno), 0, H - 1))
                tick = float(np.clip(float(vtck) - start, 0, W - 1))
                return {
                    'image': torch.from_numpy(img).float().unsqueeze(0),
                    'vertex_wire': wire,
                    'vertex_tick': tick,
                    'shape': img.shape,
                    'fname': os.path.basename(self.file_paths[idx]),
                }
        except Exception:
            return None


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return {
        'images':       torch.stack([b['image'] for b in batch]),
        'vertex_wires': torch.tensor([b['vertex_wire'] for b in batch]),
        'vertex_ticks': torch.tensor([b['vertex_tick'] for b in batch]),
        'shapes':       [b['shape'] for b in batch],
        'fnames':       [b['fname'] for b in batch],
    }


@torch.no_grad()
def run_inference(model, dataset, device, batch_size=16):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=0, pin_memory=True)
    model.eval()

    gt_wires, gt_ticks = [], []
    pred_wires, pred_ticks = [], []

    for batch in tqdm(loader, leave=False, dynamic_ncols=True):
        if batch is None:
            continue
        images = batch['images'].to(device)
        H, W = batch['shapes'][0]

        _, coords = model(images)
        px_x = coords_to_pixels(coords[:, 0], W).cpu().numpy()
        px_y = coords_to_pixels(coords[:, 1], H).cpu().numpy()

        gt_wires.extend(batch['vertex_wires'].numpy())
        gt_ticks.extend(batch['vertex_ticks'].numpy())
        pred_wires.extend(px_y)
        pred_ticks.extend(px_x)

    gt_wire   = np.array(gt_wires)
    gt_tick   = np.array(gt_ticks)
    pred_wire = np.array(pred_wires)
    pred_tick = np.array(pred_ticks)

    err_wire   = pred_wire - gt_wire
    err_tick   = pred_tick - gt_tick
    err_euclid = np.sqrt(err_wire**2 + err_tick**2)

    return gt_wire, gt_tick, pred_wire, pred_tick, err_wire, err_tick, err_euclid


def load_model(ckpt_path, encoder_name, device):
    model = UNetDSNT(encoder_name=encoder_name, encoder_weights=None).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt['model_state']
    if list(state.keys())[0].startswith('module.'):
        state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    print(f"  Loaded {ckpt_path}  (epoch {ckpt.get('epoch', '?')})")
    return model


VIEWS = ("U", "V", "Z")
VIEW_COLORS = {"U": "#4e9af1", "V": "#f1824e", "Z": "#6db86d"}


def percentile_label(arr, pcts=(50, 95, 99)):
    vals = [np.percentile(arr, p) for p in pcts]
    return "  ".join(f"p{p}={v:.1f}" for p, v in zip(pcts, vals))


def plot_error_distributions(results, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()

    clip = 150

    for i, view in enumerate(VIEWS):
        ax = axes[i]
        e = np.clip(results[view]['err_euclid'], 0, clip)
        ax.hist(e, bins=80, color=VIEW_COLORS[view], edgecolor='none', alpha=0.85)
        ax.axvline(np.median(results[view]['err_euclid']), color='k', lw=1.5,
                   linestyle='--', label=f"median={np.median(results[view]['err_euclid']):.1f}")
        ax.axvline(np.mean(results[view]['err_euclid']), color='k', lw=1.5,
                   linestyle=':', label=f"mean={np.mean(results[view]['err_euclid']):.1f}")
        ax.set_title(f"View {view}  (n={len(results[view]['err_euclid'])})", fontsize=12)
        ax.set_xlabel("Euclidean error (px)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.text(0.97, 0.95, percentile_label(results[view]['err_euclid']),
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        if np.max(results[view]['err_euclid']) > clip:
            ax.set_xlabel(f"Euclidean error (px, clipped at {clip})")

    ax = axes[3]
    all_err = np.concatenate([results[v]['err_euclid'] for v in VIEWS])
    ax.hist(np.clip(all_err, 0, clip), bins=80, color='#9b59b6', edgecolor='none', alpha=0.85)
    ax.axvline(np.median(all_err), color='k', lw=1.5, linestyle='--',
               label=f"median={np.median(all_err):.1f}")
    ax.axvline(np.mean(all_err), color='k', lw=1.5, linestyle=':',
               label=f"mean={np.mean(all_err):.1f}")
    ax.set_title(f"All views combined  (n={len(all_err)})", fontsize=12)
    ax.set_xlabel(f"Euclidean error (px, clipped at {clip})")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)
    ax.text(0.97, 0.95, percentile_label(all_err),
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    fig.suptitle("Vertex prediction — Euclidean error distributions (test set)", fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig1_error_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


def plot_cdfs(results, out_dir):
    fig, ax = plt.subplots(figsize=(9, 6))

    all_err = np.concatenate([results[v]['err_euclid'] for v in VIEWS])

    for view in VIEWS:
        e = np.sort(results[view]['err_euclid'])
        cdf = np.arange(1, len(e)+1) / len(e)
        ax.plot(e, cdf, color=VIEW_COLORS[view], lw=2, label=f"View {view}")

    e_all = np.sort(all_err)
    cdf_all = np.arange(1, len(e_all)+1) / len(e_all)
    ax.plot(e_all, cdf_all, color='#9b59b6', lw=2.5, linestyle='--', label="Combined")

    for pct, ls in [(0.5, ':'), (0.75, '-.'), (0.95, '--')]:
        ax.axhline(pct, color='gray', lw=0.8, linestyle=ls, alpha=0.6)
        ax.text(ax.get_xlim()[1] * 0.98 if ax.get_xlim()[1] > 1 else 200,
                pct + 0.01, f"{int(pct*100)}th", color='gray', fontsize=8, ha='right')

    ax.set_xlim(0, np.percentile(all_err, 99) * 1.1)
    ax.set_xlabel("Euclidean error (px)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("CDF of vertex prediction error (test set)")
    ax.legend()
    ax.grid(alpha=0.3)

    path = os.path.join(out_dir, "fig2_cdf.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


def plot_signed_residuals(results, out_dir):
    fig, axes = plt.subplots(3, 2, figsize=(13, 14))

    for row, view in enumerate(VIEWS):
        ew = results[view]['err_wire']
        et = results[view]['err_tick']

        ax = axes[row, 0]
        clip_w = np.percentile(np.abs(ew), 99) * 1.5
        ax.hist(np.clip(ew, -clip_w, clip_w), bins=80,
                color=VIEW_COLORS[view], edgecolor='none', alpha=0.85)
        ax.axvline(0, color='k', lw=1)
        ax.axvline(np.median(ew), color='r', lw=1.5, linestyle='--',
                   label=f"median={np.median(ew):.2f}")
        ax.set_title(f"View {view} — Wire residual (pred − GT)")
        ax.set_xlabel("ΔWire (px)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.text(0.03, 0.95, f"bias={np.mean(ew):.2f}  std={np.std(ew):.2f}",
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax = axes[row, 1]
        clip_t = np.percentile(np.abs(et), 99) * 1.5
        ax.hist(np.clip(et, -clip_t, clip_t), bins=80,
                color=VIEW_COLORS[view], edgecolor='none', alpha=0.85)
        ax.axvline(0, color='k', lw=1)
        ax.axvline(np.median(et), color='r', lw=1.5, linestyle='--',
                   label=f"median={np.median(et):.2f}")
        ax.set_title(f"View {view} — Tick residual (pred − GT)")
        ax.set_xlabel("ΔTick (px)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.text(0.03, 0.95, f"bias={np.mean(et):.2f}  std={np.std(et):.2f}",
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    fig.suptitle("Signed residuals: wire (y) and tick (x) axes", fontsize=14)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig3_signed_residuals.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


def plot_error_vs_position(results, out_dir):
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))

    for row, view in enumerate(VIEWS):
        gt_w = results[view]['gt_wire']
        gt_t = results[view]['gt_tick']
        err  = results[view]['err_euclid']

        for col, (pos, label) in enumerate([(gt_w, "GT wire (y, px)"), (gt_t, "GT tick (x, px)")]):
            ax = axes[row, col]
            sc = ax.scatter(pos, err, c=err, cmap='plasma', norm=LogNorm(vmin=0.5, vmax=max(err.max(), 1)),
                            s=8, alpha=0.5)
            plt.colorbar(sc, ax=ax, label="Error (px)")

            order = np.argsort(pos)
            win = max(len(pos) // 20, 5)
            rm_x = [np.median(pos[order[max(0, i-win):i+win]]) for i in range(0, len(pos), win)]
            rm_y = [np.median(err[order[max(0, i-win):i+win]]) for i in range(0, len(err), win)]
            ax.plot(rm_x, rm_y, 'w-', lw=2.5, label="running median")
            ax.plot(rm_x, rm_y, 'k--', lw=1.5)

            ax.set_xlabel(label)
            ax.set_ylabel("Euclidean error (px)")
            ax.set_title(f"View {view} — error vs {label.split(' ')[1]}")
            ax.legend(fontsize=8)

    fig.suptitle("Error vs GT position — checking for spatial bias / edge effects", fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig4_error_vs_position.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


def plot_pred_vs_gt_2d(results, out_dir):
    fig, axes = plt.subplots(3, 2, figsize=(13, 18))

    for row, view in enumerate(VIEWS):
        for col, (gt, pred, label) in enumerate([
            (results[view]['gt_wire'], results[view]['pred_wire'], "Wire (y)"),
            (results[view]['gt_tick'], results[view]['pred_tick'], "Tick (x)"),
        ]):
            ax = axes[row, col]
            lo = min(gt.min(), pred.min())
            hi = max(gt.max(), pred.max())
            hb = ax.hexbin(gt, pred, gridsize=50, cmap='inferno', mincnt=1,
                           extent=[lo, hi, lo, hi])
            ax.plot([lo, hi], [lo, hi], 'w--', lw=1.5, alpha=0.8, label="ideal")
            plt.colorbar(hb, ax=ax, label="Count")
            ax.set_xlabel(f"GT {label} (px)")
            ax.set_ylabel(f"Pred {label} (px)")
            ax.set_title(f"View {view} — {label}")
            ax.legend(fontsize=8)

    fig.suptitle("Predicted vs GT coordinates (hexbin density)", fontsize=14)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig5_pred_vs_gt.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


def plot_cross_view_tick(results, out_dir):
    n = min(len(results[v]['pred_tick']) for v in VIEWS)
    ticks = np.stack([results[v]['pred_tick'][:n] for v in VIEWS], axis=1)
    gt_tick = results[VIEWS[0]]['gt_tick'][:n]

    tick_std = ticks.std(axis=1)
    tick_mean = ticks.mean(axis=1)
    mean_tick_err = np.abs(tick_mean - gt_tick)

    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    ax0 = fig.add_subplot(gs[0, :2])
    ax0.hist(tick_std, bins=60, color='steelblue', edgecolor='none', alpha=0.85)
    ax0.axvline(np.median(tick_std), color='k', lw=1.5, linestyle='--',
                label=f"median std={np.median(tick_std):.2f} px")
    ax0.set_xlabel("Std of tick prediction across U/V/Z views (px)")
    ax0.set_ylabel("Count")
    ax0.set_title("Cross-view tick consistency\n(lower = views agree)")
    ax0.legend()

    mean_euclid = np.stack([results[v]['err_euclid'][:n] for v in VIEWS], axis=1).mean(axis=1)
    ax1 = fig.add_subplot(gs[0, 2])
    ax1.scatter(tick_std, mean_euclid, s=6, alpha=0.4, color='steelblue')
    ax1.set_xlabel("Tick std across views (px)")
    ax1.set_ylabel("Mean Euclidean error (px)")
    ax1.set_title("Cross-view disagreement\nvs error")
    ax1.set_xlim(0, np.percentile(tick_std, 99) * 1.2)
    ax1.set_ylim(0, np.percentile(mean_euclid, 99) * 1.2)

    for i, (va, vb) in enumerate([("U", "V"), ("U", "Z"), ("V", "Z")]):
        ax = fig.add_subplot(gs[1, i])
        lo = min(results[va]['pred_tick'][:n].min(), results[vb]['pred_tick'][:n].min())
        hi = max(results[va]['pred_tick'][:n].max(), results[vb]['pred_tick'][:n].max())
        ax.scatter(results[va]['pred_tick'][:n], results[vb]['pred_tick'][:n],
                   s=5, alpha=0.35, color='steelblue')
        ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label="ideal")
        r = np.corrcoef(results[va]['pred_tick'][:n], results[vb]['pred_tick'][:n])[0, 1]
        ax.set_xlabel(f"Tick pred {va} (px)")
        ax.set_ylabel(f"Tick pred {vb} (px)")
        ax.set_title(f"{va} vs {vb}  r={r:.3f}")
        ax.legend(fontsize=8)

    fig.suptitle("Cross-view tick (x) consistency — should be identical across views", fontsize=13)
    path = os.path.join(out_dir, "fig6_cross_view_tick.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


def plot_failure_breakdown(results, out_dir):
    thresholds = [5, 15, 50]
    labels = [f"< {thresholds[0]} px (good)",
              f"{thresholds[0]}–{thresholds[1]} px (ok)",
              f"{thresholds[1]}–{thresholds[2]} px (bad)",
              f"> {thresholds[2]} px (failure)"]
    colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']

    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    all_err = np.concatenate([results[v]['err_euclid'] for v in VIEWS])

    for ax, (view, err) in zip(axes[:3], [(v, results[v]['err_euclid']) for v in VIEWS]):
        counts = [
            (err < thresholds[0]).sum(),
            ((err >= thresholds[0]) & (err < thresholds[1])).sum(),
            ((err >= thresholds[1]) & (err < thresholds[2])).sum(),
            (err >= thresholds[2]).sum(),
        ]
        bars = ax.bar(range(4), counts, color=colors, edgecolor='none')
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
        ax.set_title(f"View {view}")
        ax.set_ylabel("Count")
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{100*cnt/len(err):.1f}%", ha='center', va='bottom', fontsize=8)

    ax = axes[3]
    counts = [
        (all_err < thresholds[0]).sum(),
        ((all_err >= thresholds[0]) & (all_err < thresholds[1])).sum(),
        ((all_err >= thresholds[1]) & (all_err < thresholds[2])).sum(),
        (all_err >= thresholds[2]).sum(),
    ]
    bars = ax.bar(range(4), counts, color=colors, edgecolor='none')
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.set_title("All views combined")
    ax.set_ylabel("Count")
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{100*cnt/len(all_err):.1f}%", ha='center', va='bottom', fontsize=8)

    fig.suptitle("Error category breakdown (failure mode analysis)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig7_failure_breakdown.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


def print_summary(results):
    print(f"\n{'View':<6} {'n':>5} {'mean':>7} {'median':>8} {'std':>7} {'p75':>7} {'p95':>7} {'p99':>7}")
    print(f"{'-'*57}")
    all_err = []
    for v in VIEWS:
        e = results[v]['err_euclid']
        all_err.append(e)
        print(f"{v:<6} {len(e):>5} {np.mean(e):>7.2f} {np.median(e):>8.2f} "
              f"{np.std(e):>7.2f} {np.percentile(e,75):>7.2f} "
              f"{np.percentile(e,95):>7.2f} {np.percentile(e,99):>7.2f}")
    all_err = np.concatenate(all_err)
    print(f"{'-'*57}")
    print(f"{'ALL':<6} {len(all_err):>5} {np.mean(all_err):>7.2f} {np.median(all_err):>8.2f} "
          f"{np.std(all_err):>7.2f} {np.percentile(all_err,75):>7.2f} "
          f"{np.percentile(all_err,95):>7.2f} {np.percentile(all_err,99):>7.2f}")
    print("  All units: pixels (Euclidean error)")

    print("\nFailure rates:")
    for label, lo, hi in [("< 5 px (good)", 0, 5), ("5–15 px (ok)", 5, 15),
                           ("15–50 px (bad)", 15, 50), ("> 50 px (failure)", 50, np.inf)]:
        n = ((all_err >= lo) & (all_err < hi)).sum()
        print(f"  {label:<22}: {n:>4} / {len(all_err)}  ({100*n/len(all_err):.1f}%)")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root",       type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_VERTEX_DIR)
    parser.add_argument("--encoder_name",   type=str, default="densenet169")
    parser.add_argument("--out_dir",        type=str, default=str(OUTPUT_ROOT / "vertex_eval"))
    parser.add_argument("--batch_size",     type=int, default=16)
    parser.add_argument("--split",          type=str, default="test")
    parser.add_argument("--device",         type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"\nDevice: {device}")
    print(f"Output: {args.out_dir}\n")

    results = {}
    for view in VIEWS:
        print(f"--- View {view} ---")
        ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoints_{view}", "best_checkpoint.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoints_{view}", "last_checkpoint.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"No checkpoint found for view {view} in {args.checkpoint_dir}")

        model = load_model(ckpt_path, args.encoder_name, device)
        dataset = VertexDataset(args.raw_root, args.split, view)
        gt_wire, gt_tick, pred_wire, pred_tick, err_wire, err_tick, err_euclid = \
            run_inference(model, dataset, device, args.batch_size)

        results[view] = {
            'gt_wire': gt_wire, 'gt_tick': gt_tick,
            'pred_wire': pred_wire, 'pred_tick': pred_tick,
            'err_wire': err_wire, 'err_tick': err_tick,
            'err_euclid': err_euclid,
        }
        del model

    print_summary(results)

    print("Generating figures...")
    plot_error_distributions(results, args.out_dir)
    plot_cdfs(results, args.out_dir)
    plot_signed_residuals(results, args.out_dir)
    plot_error_vs_position(results, args.out_dir)
    plot_pred_vs_gt_2d(results, args.out_dir)
    plot_cross_view_tick(results, args.out_dir)
    plot_failure_breakdown(results, args.out_dir)

    np_out = os.path.join(args.out_dir, "results.npz")
    np.savez(np_out, **{f"{v}_{k}": arr for v, d in results.items() for k, arr in d.items()})
    print(f"\nSaved raw arrays to {np_out}")
    print("Done.\n")


if __name__ == "__main__":
    main()
