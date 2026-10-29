#!/usr/bin/env python3
"""Poisson-stream deployment study. Michel (~1/min) and no-Michel (~1/10 s) events arrive
as a Poisson process; a Michel trigger classifier gates the full direction pipeline while
latency, memory, and GPU utilization are logged.
"""
# Simulates a Poisson arrival process:
#   Michel events:    ~1/min   (stopping muon decay; proxy for SN nu eES)
#   No-Michel events: ~1/10 s  (generic background)
#
# Outputs (under output_dir):
#   raw_events/event_{n:06d}.npz      — raw U/V/Z for fired events
#   pipeline_input/event_{n:06d}.npz  — seg-preprocessed U/V/Z for fired events
#   event_log.csv                     — per-event metadata, confusion counts, utilization
#   memory_log.csv                    — stage-separated memory snapshots for fired events
#   direction_log.csv                 — directions + cos_theta for fired events

from __future__ import annotations
import argparse
import csv
import glob
import math
import os
import random
import tempfile
import threading
import time

import numpy as np
import psutil
import pynvml
import torch
import torch.nn.functional as F
import warp as wp
from torchvision import models as tvm

from main.vertex.dsnt_vertex import UNetDSNT, VertexConfig
from main.preprocessing.remove_muon import load_sparse_model, process_file
from main.headtail.models import DenseModelSMP
from main.headtail.visualization import _apply_priors_and_order_for_eval
from main.headtail.physics_utils import unit_from_points
from main.headtail.consistency import apply_consistency_pipeline, predict_view_from_3d_batch
from main.paths import (
    DEFAULT_SEG_CKPTS,
    DEFAULT_VERTEX_CKPTS,
    DEFAULT_BREMS_CKPT,
    DEFAULT_CLASSIFIER_CKPT,
    OUTPUT_ROOT,
)

wp.init()
pynvml.nvmlInit()

try:
    import wandb
    _WANDB_OK = True
except Exception as e:
    _WANDB_OK = False
    print(f"WARNING: wandb unavailable ({type(e).__name__}: {e})")

_VIEWS = ["U", "V", "Z"]
_NAN = float("nan")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim_duration",            type=float, default=1000.0)  # 7200 s = 2 hrs
    ap.add_argument("--michel_rate",             type=float, default=1.0 / 60.0)
    ap.add_argument("--nomichel_rate",           type=float, default=1.0 / 10.0)
    ap.add_argument("--threshold",               type=float, default=None)
    ap.add_argument("--seed",                    type=int,   default=42)
    ap.add_argument("--device",                  type=str,   default="cuda:0")
    ap.add_argument("--gpu_index",               type=int,   default=0,
                    help="GPU index for NVML utilization query.")
    ap.add_argument("--util_sample_interval_ms", type=float, default=5.0)

    # paths
    ap.add_argument("--michel_pool_dir",   type=str, required=True,
                    help="Directory of Michel NPZs (rawdata/all/*.npz).")
    ap.add_argument("--nomichel_root",     type=str, required=True,
                    help="Root dir containing no_michel/rawdata/{train,val,test}/all.")
    ap.add_argument("--ckpt_brems",        type=str, default=DEFAULT_BREMS_CKPT)
    ap.add_argument("--classifier_ckpt",   type=str, default=DEFAULT_CLASSIFIER_CKPT)
    ap.add_argument("--output_dir",        type=str, default=str(OUTPUT_ROOT / "simulation_output"))

    # wandb
    ap.add_argument("--use_wandb",      action="store_true")
    ap.add_argument("--wandb_project",  type=str, default="sn-michel-simulation")
    ap.add_argument("--wandb_run_name", type=str, default=None)
    ap.add_argument("--wandb_entity",   type=str, default=None)

    ap.add_argument("--penalize_fp", action="store_true",
                    help="Assign cos_theta=-1 for FP events instead of nan.")
    return ap.parse_args()


def collect_michel_pool(root):
    return sorted(glob.glob(os.path.join(root, "*.npz")))


def collect_nomichel_pool(root):
    paths = []
    for split in ("train", "val", "test"):
        pattern = os.path.join(root, "no_michel", "rawdata", split, "all", "*.npz")
        paths.extend(glob.glob(pattern))
    return sorted(paths)


def _summarize_array(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    return float(arr.mean()), float(np.percentile(arr, 95)), float(arr.max())


def zero_util_stats():
    return {
        "gpu_mean_pct": 0.0, "gpu_p95_pct": 0.0, "gpu_max_pct": 0.0,
        "cpu_mean_pct": 0.0, "cpu_p95_pct": 0.0, "cpu_max_pct": 0.0,
        "rss_mean_gb":  0.0, "rss_p95_gb":  0.0, "rss_max_gb":  0.0,
    }


class StageUtilSampler:
    # Samples GPU util (NVML), CPU process util (psutil), and process RSS in GB.
    # CPU util can exceed 100 on multi-core machines — that's expected.

    def __init__(self, nvml_handle, interval_s=0.005):
        self.nvml_handle = nvml_handle
        self.interval_s = max(float(interval_s), 1e-4)
        self.proc = psutil.Process()
        self._stop = threading.Event()
        self._thread = None
        self._gpu_samples = []
        self._cpu_samples = []
        self._rss_samples = []

    def _sample_once(self):
        try:
            gpu_util = float(pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle).gpu)
        except Exception:
            gpu_util = float("nan")
        try:
            cpu_util = float(self.proc.cpu_percent(interval=None))
        except Exception:
            cpu_util = float("nan")
        try:
            rss_gb = float(self.proc.memory_info().rss / 1e9)
        except Exception:
            rss_gb = float("nan")
        self._gpu_samples.append(gpu_util)
        self._cpu_samples.append(cpu_util)
        self._rss_samples.append(rss_gb)

    def _run(self):
        self.proc.cpu_percent(interval=None)
        while not self._stop.wait(self.interval_s):
            self._sample_once()

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._sample_once()
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        gpu_mean, gpu_p95, gpu_max = _summarize_array(np.array(self._gpu_samples, dtype=float))
        cpu_mean, cpu_p95, cpu_max = _summarize_array(np.array(self._cpu_samples, dtype=float))
        rss_mean, rss_p95, rss_max = _summarize_array(np.array(self._rss_samples, dtype=float))
        return {
            "gpu_mean_pct": gpu_mean, "gpu_p95_pct": gpu_p95, "gpu_max_pct": gpu_max,
            "cpu_mean_pct": cpu_mean, "cpu_p95_pct": cpu_p95, "cpu_max_pct": cpu_max,
            "rss_mean_gb":  rss_mean, "rss_p95_gb":  rss_p95, "rss_max_gb":  rss_max,
        }


def load_pipeline_models(args, device):
    ckpt_seg = DEFAULT_SEG_CKPTS
    ckpt_vtx = DEFAULT_VERTEX_CKPTS

    brems_cfg = argparse.Namespace(
        dense_arch="smp_unet",
        smp_encoder="densenet169",
        smp_in_channels=1,
        dense_feat_ch=256,
        num_classes=5,
        enable_seg=True,
        enable_dirreg=False,
        smp_encoder_weights=None,
        smp_decoder_channels=(256, 128, 64, 32, 16),
        dsnt_temp=1.0,
        prior_mode="off",
        prior_alpha=0.3,
        prior_topk_percent=20.0,
        consistency_flip_threshold=0.90,
        consistency_drop_threshold=0.25,
        bragg_make=False,
    )

    print("Loading vertex models...")
    vertex_models = {}
    for v in _VIEWS:
        ckpt = torch.load(ckpt_vtx[v], map_location=device, weights_only=False)
        vcfg = VertexConfig(**ckpt["cfg"])
        m = UNetDSNT(encoder_name=vcfg.encoder_name, encoder_weights=None).to(device)
        state = ckpt["model_state"]
        if list(state.keys())[0].startswith("module."):
            state = {k.replace("module.", ""): vv for k, vv in state.items()}
        m.load_state_dict(state)
        m.eval()
        vertex_models[v] = m
        print(f"  vertex {v} OK")

    print("Loading seg models...")
    seg_models = {}
    for v in _VIEWS:
        m, cfg_seg = load_sparse_model(ckpt_seg[v], device)
        seg_models[v] = (m, cfg_seg)
        print(f"  seg {v} OK")

    print("Loading brems-fl models...")
    ckpt_b = torch.load(args.ckpt_brems, map_location=device, weights_only=False)
    brems_models = {}
    for v in _VIEWS:
        m = DenseModelSMP(
            in_ch=brems_cfg.smp_in_channels,
            feat_ch=brems_cfg.dense_feat_ch,
            num_classes=brems_cfg.num_classes,
            enable_seg=brems_cfg.enable_seg,
            enable_dirreg=brems_cfg.enable_dirreg,
            encoder_name=brems_cfg.smp_encoder,
            encoder_weights=brems_cfg.smp_encoder_weights,
            decoder_channels=brems_cfg.smp_decoder_channels,
        ).to(device)
        state = ckpt_b[v]["model"]
        if list(state.keys())[0].startswith("module."):
            state = {k.replace("module.", ""): vv for k, vv in state.items()}
        m.load_state_dict(state)
        m.eval()
        brems_models[v] = m
        print(f"  brems {v} OK")

    return vertex_models, seg_models, brems_models, brems_cfg


def load_classifier(ckpt_path, device, threshold_override=None):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})

    m = tvm.efficientnet_b5(weights=None)
    m.classifier[-1] = torch.nn.Linear(m.classifier[-1].in_features, 1)

    state = ckpt["model"]
    if list(state.keys())[0].startswith("module."):
        state = {k.replace("module.", ""): vv for k, vv in state.items()}

    m.load_state_dict(state)
    m.to(device)
    m.eval()

    if threshold_override is not None:
        threshold = float(threshold_override)
        print(f"Classifier threshold (CLI): {threshold:.4f}")
    else:
        threshold = float(saved_args.get("test_threshold", 0.5))
        print(f"Classifier threshold (checkpoint): {threshold:.4f}")
        print("  Note: pass --threshold <val> to override if needed.")

    return m, threshold


@torch.no_grad()
def classify_npz(npz_path, model, device, threshold):
    # returns (fired, score, classifier_time_s, classifier_peak_gpu_gb)
    with np.load(npz_path, allow_pickle=True) as d:
        U = d["U_data"].squeeze().astype(np.float32)
        V = d["V_data"].squeeze().astype(np.float32)
        Z = d["Z_data"].squeeze().astype(np.float32)

    H, W = U.shape

    def _resize(x):
        if x.shape == (H, W):
            return x
        return F.interpolate(
            torch.from_numpy(x)[None, None],
            size=(H, W), mode="bilinear", align_corners=False,
        )[0, 0].numpy()

    V = _resize(V)
    Z = _resize(Z)

    x = torch.stack([
        torch.from_numpy(U),
        torch.from_numpy(V),
        torch.from_numpy(Z),
    ], dim=0).float()

    mean = x.mean(dim=(1, 2), keepdim=True)
    std  = x.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    x    = (x - mean) / std

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    logit = model(x.unsqueeze(0).to(device))
    score = float(torch.sigmoid(logit).cpu().item())
    classifier_time = time.perf_counter() - t0

    classifier_peak_gpu = (
        torch.cuda.max_memory_allocated(device) / 1e9
        if torch.cuda.is_available() else 0.0
    )
    return score >= threshold, score, classifier_time, classifier_peak_gpu


def _extract_drift_sign(npz_data, default=1.0):
    try:
        meta = npz_data["metadata"].item()
        for v in _VIEWS:
            try:
                tpc = meta["per_view"][v]["michel"]["tpc"]
                return 1.0 if int(tpc) == 1 else -1.0
            except (KeyError, TypeError):
                pass
        tpc = meta.get("tpc", None)
        if tpc is not None:
            return 1.0 if int(tpc) == 1 else -1.0
    except Exception:
        pass
    return float(default)


def _extract_gt_direction(meta):
    for v in ("U", "V", "Z"):
        try:
            pv = meta["per_view"][v]["michel"]
            return float(pv["vpx"]), float(pv["vpy"]), float(pv["vpz"])
        except (KeyError, TypeError):
            pass
    try:
        return float(meta["vpx"]), float(meta["vpy"]), float(meta["vpz"])
    except (KeyError, TypeError):
        pass
    return None, None, None


@torch.no_grad()
def _infer_brems_view(model, img_np, device, brems_cfg):
    H, W = img_np.shape
    img_t = torch.from_numpy(img_np).float().to(device).unsqueeze(0).unsqueeze(0)

    pad_h = (32 - H % 32) % 32
    pad_w = (32 - W % 32) % 32
    if pad_h or pad_w:
        img_t = F.pad(img_t, (0, pad_w, 0, pad_h), "constant", 0.0)

    out = model(img_t)
    logits_ab, mx, my = _apply_priors_and_order_for_eval(out, brems_cfg)
    ux, uy, _ = unit_from_points(mx[:, 0], my[:, 0], mx[:, 1], my[:, 1])
    return float(ux[0].cpu()), float(uy[0].cpu())


def run_direction_pipeline(raw_path, vertex_models, seg_models,
                           brems_models, brems_cfg, device):
    raw_npz = np.load(raw_path, allow_pickle=True)
    drift_sign = _extract_drift_sign(raw_npz)
    raw_arrays = {v: raw_npz[f"{v}_data"].squeeze().astype(np.float32) for v in _VIEWS}

    clean_path = raw_path.replace("/rawdata/", "/cleansig/")
    fd, tmp_path = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    for v in _VIEWS:
        seg_model, cfg_seg = seg_models[v]
        process_file(
            raw_path=raw_path,
            clean_path=clean_path,
            output_path=tmp_path,
            seg_model=seg_model,
            vertex_model=vertex_models[v],
            device=device,
            cfg_seg=cfg_seg,
            view=v,
            apply_bragg_suppression=True,
            vertex_csv_writer=None,
        )

    preproc_data = dict(np.load(tmp_path, allow_pickle=True))
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    ux_raw, uy_raw = {}, {}
    for v in _VIEWS:
        img = preproc_data[f"{v}_data"].astype(np.float32)
        ux_raw[v], uy_raw[v] = _infer_brems_view(brems_models[v], img, device, brems_cfg)

    ds_t = torch.tensor([drift_sign], device=device, dtype=torch.float32)
    ux_t = {v: torch.tensor([ux_raw[v]], device=device) for v in _VIEWS}
    uy_t = {v: torch.tensor([uy_raw[v]], device=device) for v in _VIEWS}

    cons = apply_consistency_pipeline(ux_t, uy_t, ds_t, brems_cfg)
    ux_corr = {v: float(cons["ux_corrected"][v][0]) for v in _VIEWS}
    uy_corr = {v: float(cons["uy_corrected"][v][0]) for v in _VIEWS}

    gt_direction_3d = None
    gt_2d_per_view = {}
    cos_thetas = {}

    try:
        meta = raw_npz["metadata"].item()
        vpx, vpy, vpz = _extract_gt_direction(meta)
        if vpx is not None:
            norm = math.sqrt(vpx**2 + vpy**2 + vpz**2)
            if norm > 1e-9:
                vpx, vpy, vpz = vpx / norm, vpy / norm, vpz / norm
                gt_direction_3d = (vpx, vpy, vpz)
                gt3d = torch.tensor([[vpx, vpy, vpz]], device=device, dtype=torch.float32)
                for v in _VIEWS:
                    gux_t, guy_t = predict_view_from_3d_batch(gt3d, v, drift_sign=ds_t)
                    gt_ux, gt_uy = float(gux_t[0].cpu()), float(guy_t[0].cpu())
                    gt_2d_per_view[v] = (gt_ux, gt_uy)
                    cos_thetas[v] = float(np.clip(
                        ux_corr[v] * gt_ux + uy_corr[v] * gt_uy, -1.0, 1.0
                    ))
    except Exception:
        pass

    return (
        preproc_data, raw_arrays,
        ux_corr, uy_corr,
        drift_sign, gt_direction_3d, gt_2d_per_view, cos_thetas,
    )


def snapshot_memory(device):
    cpu_gb = psutil.Process().memory_info().rss / 1e9
    if torch.cuda.is_available():
        gpu_alloc = torch.cuda.memory_allocated(device) / 1e9
        gpu_res   = torch.cuda.memory_reserved(device) / 1e9
    else:
        gpu_alloc = 0.0
        gpu_res   = 0.0
    return gpu_alloc, gpu_res, cpu_gb


def _fmt(x):
    if isinstance(x, float) and math.isnan(x):
        return "nan"
    return f"{x:.5f}"


def generate_event_sequence(sim_duration, michel_rate, nomichel_rate, rng):
    total_rate = michel_rate + nomichel_rate
    p_michel   = michel_rate / total_rate
    n_expected = int(total_rate * sim_duration * 3.0) + 500
    inter = rng.exponential(1.0 / total_rate, size=n_expected)
    times = np.cumsum(inter)
    times = times[times <= sim_duration]
    labels = (rng.random(len(times)) < p_michel).astype(int)
    return list(zip(times.tolist(), labels.tolist()))


def open_event_log(path):
    f = open(path, "w", newline="")
    w = csv.writer(f)
    w.writerow([
        "event_count", "sim_time_s", "true_label", "pred_label",
        "classifier_score", "classifier_time_s", "pipeline_fired", "outcome",
        "cumulative_TP", "cumulative_TN", "cumulative_FP", "cumulative_FN",
        "wall_time_pipeline_s",
        "classifier_gpu_util_mean_pct", "classifier_gpu_util_p95_pct", "classifier_gpu_util_max_pct",
        "classifier_cpu_proc_util_mean_pct", "classifier_cpu_proc_util_p95_pct", "classifier_cpu_proc_util_max_pct",
        "pipeline_gpu_util_mean_pct", "pipeline_gpu_util_p95_pct", "pipeline_gpu_util_max_pct",
        "pipeline_cpu_proc_util_mean_pct", "pipeline_cpu_proc_util_p95_pct", "pipeline_cpu_proc_util_max_pct",
    ])
    return f, w


def open_memory_log(path):
    f = open(path, "w", newline="")
    w = csv.writer(f)
    w.writerow([
        "event_count", "sim_time_s", "pipeline_fired",
        "baseline_gpu_gb", "baseline_cpu_rss_gb",
        "classifier_peak_gpu_gb", "classifier_peak_cpu_rss_gb",
        "pipeline_peak_gpu_gb", "pipeline_peak_cpu_rss_gb",
        "gpu_allocated_gb", "gpu_reserved_gb", "cpu_rss_gb",
    ])
    return f, w


def open_direction_log(path):
    f = open(path, "w", newline="")
    w = csv.writer(f)
    w.writerow([
        "event_count", "sim_time_s", "true_label", "outcome", "classifier_score",
        "drift_sign",
        "gt_vpx", "gt_vpy", "gt_vpz",
        "ux_corr_U", "uy_corr_U", "ux_corr_V", "uy_corr_V", "ux_corr_Z", "uy_corr_Z",
        "gt_ux_U", "gt_uy_U", "gt_ux_V", "gt_uy_V", "gt_ux_Z", "gt_uy_Z",
        "cos_theta_U", "cos_theta_V", "cos_theta_Z",
    ])
    return f, w


def main():
    args = parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu_index)
    util_interval_s = args.util_sample_interval_ms / 1000.0

    print(f"Device: {device}")
    print(f"Utilization sampling interval: {args.util_sample_interval_ms:.2f} ms")

    use_wandb = args.use_wandb and _WANDB_OK
    if args.use_wandb and not _WANDB_OK:
        print("WARNING: wandb unavailable — skipping W&B logging.")

    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config=vars(args),
        )
        print(f"W&B run: {wandb.run.url}")

    out_dir     = args.output_dir
    raw_dir     = os.path.join(out_dir, "raw_events")
    preproc_dir = os.path.join(out_dir, "pipeline_input")
    for d in (out_dir, raw_dir, preproc_dir):
        os.makedirs(d, exist_ok=True)

    michel_pool   = collect_michel_pool(args.michel_pool_dir)
    nomichel_pool = collect_nomichel_pool(args.nomichel_root)
    if not michel_pool:
        raise RuntimeError(f"No Michel NPZs found in {args.michel_pool_dir}")
    if not nomichel_pool:
        raise RuntimeError(f"No no-Michel NPZs found under {args.nomichel_root}")

    print(f"Michel pool:    {len(michel_pool):5d} files")
    print(f"No-Michel pool: {len(nomichel_pool):5d} files")

    vertex_models, seg_models, brems_models, brems_cfg = load_pipeline_models(args, device)
    classifier, threshold = load_classifier(args.classifier_ckpt, device, args.threshold)
    print(f"Classifier threshold: {threshold:.4f}")

    base_gpu, _, base_cpu = snapshot_memory(device)
    print(f"Baseline memory — GPU: {base_gpu:.3f} GB  CPU: {base_cpu:.3f} GB")

    event_seq = generate_event_sequence(
        args.sim_duration, args.michel_rate, args.nomichel_rate, rng
    )
    n_total  = len(event_seq)
    n_michel = sum(1 for _, lbl in event_seq if lbl == 1)

    print(f"\nSimulating {args.sim_duration:.0f} s  |  total events: {n_total}"
          f"  (Michel: {n_michel}  no-Michel: {n_total - n_michel})")
    print(f"Output → {out_dir}\n")

    ev_f,  ev_w  = open_event_log(os.path.join(out_dir, "event_log.csv"))
    mem_f, mem_w = open_memory_log(os.path.join(out_dir, "memory_log.csv"))
    dir_f, dir_w = open_direction_log(os.path.join(out_dir, "direction_log.csv"))

    counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

    for event_count, (sim_time, true_label) in enumerate(event_seq, start=1):
        npz_path = (
            michel_pool[rng.integers(len(michel_pool))]
            if true_label == 1 else
            nomichel_pool[rng.integers(len(nomichel_pool))]
        )

        baseline_gpu_gb, _, baseline_cpu_rss_gb = snapshot_memory(device)

        # classifier stage
        classifier_sampler = StageUtilSampler(nvml_handle=nvml_handle, interval_s=util_interval_s).start()

        fired, score, classifier_time, classifier_peak_gpu = classify_npz(
            npz_path, classifier, device, threshold
        )

        classifier_util = classifier_sampler.stop()
        pred_label = int(fired)
        classifier_peak_cpu_rss = classifier_util["rss_max_gb"]

        if   true_label == 1 and pred_label == 1: outcome = "TP"
        elif true_label == 0 and pred_label == 0: outcome = "TN"
        elif true_label == 0 and pred_label == 1: outcome = "FP"
        else:                                      outcome = "FN"
        counts[outcome] += 1

        wall_time_pipeline    = 0.0
        pipeline_peak_gpu     = 0.0
        pipeline_peak_cpu_rss = 0.0
        pipeline_util         = zero_util_stats()

        if fired:
            # direction pipeline stage
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)

            pipeline_sampler = StageUtilSampler(nvml_handle=nvml_handle, interval_s=util_interval_s).start()

            t0 = time.perf_counter()
            (
                preproc_data, raw_arrays,
                ux_corr, uy_corr,
                drift_sign, gt_dir3d, gt_2d, cos_thetas,
            ) = run_direction_pipeline(
                npz_path, vertex_models, seg_models, brems_models, brems_cfg, device,
            )
            wall_time_pipeline = time.perf_counter() - t0

            pipeline_util = pipeline_sampler.stop()
            pipeline_peak_gpu = (
                torch.cuda.max_memory_allocated(device) / 1e9
                if torch.cuda.is_available() else 0.0
            )
            pipeline_peak_cpu_rss = pipeline_util["rss_max_gb"]

            if outcome == "FP" and args.penalize_fp:
                cos_thetas = {v: -1.0 for v in _VIEWS}

            np.savez_compressed(
                os.path.join(raw_dir, f"event_{event_count:06d}.npz"),
                U_data=raw_arrays["U"], V_data=raw_arrays["V"], Z_data=raw_arrays["Z"],
            )
            np.savez_compressed(
                os.path.join(preproc_dir, f"event_{event_count:06d}.npz"),
                U_data=preproc_data["U_data"].astype(np.float32),
                V_data=preproc_data["V_data"].astype(np.float32),
                Z_data=preproc_data["Z_data"].astype(np.float32),
            )

            gpu_alloc, gpu_res, cpu_rss = snapshot_memory(device)

            mem_w.writerow([
                event_count, f"{sim_time:.4f}", 1,
                f"{baseline_gpu_gb:.4f}", f"{baseline_cpu_rss_gb:.4f}",
                f"{classifier_peak_gpu:.4f}", f"{classifier_peak_cpu_rss:.4f}",
                f"{pipeline_peak_gpu:.4f}", f"{pipeline_peak_cpu_rss:.4f}",
                f"{gpu_alloc:.4f}", f"{gpu_res:.4f}", f"{cpu_rss:.4f}",
            ])
            mem_f.flush()

            def _gt2(v, i):
                return gt_2d[v][i] if v in gt_2d else _NAN
            def _ct(v):
                return cos_thetas[v] if v in cos_thetas else _NAN

            vpx = gt_dir3d[0] if gt_dir3d else _NAN
            vpy = gt_dir3d[1] if gt_dir3d else _NAN
            vpz = gt_dir3d[2] if gt_dir3d else _NAN

            dir_w.writerow([
                event_count, f"{sim_time:.4f}", true_label, outcome, f"{score:.5f}",
                f"{drift_sign:+.0f}",
                _fmt(vpx), _fmt(vpy), _fmt(vpz),
                f"{ux_corr['U']:.5f}", f"{uy_corr['U']:.5f}",
                f"{ux_corr['V']:.5f}", f"{uy_corr['V']:.5f}",
                f"{ux_corr['Z']:.5f}", f"{uy_corr['Z']:.5f}",
                _fmt(_gt2("U", 0)), _fmt(_gt2("U", 1)),
                _fmt(_gt2("V", 0)), _fmt(_gt2("V", 1)),
                _fmt(_gt2("Z", 0)), _fmt(_gt2("Z", 1)),
                _fmt(_ct("U")), _fmt(_ct("V")), _fmt(_ct("Z")),
            ])
            dir_f.flush()

            if use_wandb:
                wdict = {
                    "pipeline/wall_time_s":             wall_time_pipeline,
                    "pipeline/classifier_time_s":       classifier_time,
                    "pipeline/classifier_peak_gpu_gb":  classifier_peak_gpu,
                    "pipeline/classifier_peak_cpu_rss": classifier_peak_cpu_rss,
                    "pipeline/pipeline_peak_gpu_gb":    pipeline_peak_gpu,
                    "pipeline/pipeline_peak_cpu_rss":   pipeline_peak_cpu_rss,
                    "pipeline/gpu_allocated_gb":        gpu_alloc,
                    "pipeline/cpu_rss_gb":              cpu_rss,
                    "util/classifier_gpu_mean_pct":     classifier_util["gpu_mean_pct"],
                    "util/classifier_gpu_p95_pct":      classifier_util["gpu_p95_pct"],
                    "util/classifier_gpu_max_pct":      classifier_util["gpu_max_pct"],
                    "util/classifier_cpu_mean_pct":     classifier_util["cpu_mean_pct"],
                    "util/classifier_cpu_p95_pct":      classifier_util["cpu_p95_pct"],
                    "util/classifier_cpu_max_pct":      classifier_util["cpu_max_pct"],
                    "util/pipeline_gpu_mean_pct":       pipeline_util["gpu_mean_pct"],
                    "util/pipeline_gpu_p95_pct":        pipeline_util["gpu_p95_pct"],
                    "util/pipeline_gpu_max_pct":        pipeline_util["gpu_max_pct"],
                    "util/pipeline_cpu_mean_pct":       pipeline_util["cpu_mean_pct"],
                    "util/pipeline_cpu_p95_pct":        pipeline_util["cpu_p95_pct"],
                    "util/pipeline_cpu_max_pct":        pipeline_util["cpu_max_pct"],
                }
                for v in _VIEWS:
                    if v in cos_thetas:
                        wdict[f"pipeline/cos_theta_{v}"] = cos_thetas[v]
                wandb.log(wdict, step=event_count)

        ev_w.writerow([
            event_count, f"{sim_time:.4f}", true_label, pred_label, f"{score:.5f}",
            f"{classifier_time:.5f}", int(fired), outcome,
            counts["TP"], counts["TN"], counts["FP"], counts["FN"],
            f"{wall_time_pipeline:.5f}",
            f"{classifier_util['gpu_mean_pct']:.3f}", f"{classifier_util['gpu_p95_pct']:.3f}",
            f"{classifier_util['gpu_max_pct']:.3f}",  f"{classifier_util['cpu_mean_pct']:.3f}",
            f"{classifier_util['cpu_p95_pct']:.3f}",  f"{classifier_util['cpu_max_pct']:.3f}",
            f"{pipeline_util['gpu_mean_pct']:.3f}",   f"{pipeline_util['gpu_p95_pct']:.3f}",
            f"{pipeline_util['gpu_max_pct']:.3f}",    f"{pipeline_util['cpu_mean_pct']:.3f}",
            f"{pipeline_util['cpu_p95_pct']:.3f}",    f"{pipeline_util['cpu_max_pct']:.3f}",
        ])
        ev_f.flush()

        tp, tn, fp, fn = counts["TP"], counts["TN"], counts["FP"], counts["FN"]
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        bacc = 0.5 * (sensitivity + specificity)

        if use_wandb:
            wandb.log({
                "event/sim_time_s":        sim_time,
                "event/true_label":        true_label,
                "event/pred_label":        pred_label,
                "event/score":             score,
                "event/classifier_time_s": classifier_time,
                "event/pipeline_fired":    int(fired),
                "metrics/sensitivity":     sensitivity,
                "metrics/specificity":     specificity,
                "metrics/bacc":            bacc,
                "metrics/TP": tp, "metrics/TN": tn,
                "metrics/FP": fp, "metrics/FN": fn,
            }, step=event_count)

        if event_count % 100 == 0 or event_count == n_total:
            print(f"[{event_count:5d}/{n_total}] t={sim_time:7.1f}s  "
                  f"TP={tp} TN={tn} FP={fp} FN={fn}  "
                  f"sens={sensitivity:.3f} spec={specificity:.3f} bacc={bacc:.3f}")

    ev_f.close()
    mem_f.close()
    dir_f.close()

    tp, tn, fp, fn = counts["TP"], counts["TN"], counts["FP"], counts["FN"]
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    bacc = 0.5 * (sensitivity + specificity)

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print(f"  Simulated duration: {args.sim_duration:.0f} s")
    print(f"  Total events:       {tp + tn + fp + fn}")
    print(f"             pred 0     pred 1")
    print(f"  true 0   {tn:8d}  {fp:8d}   (TN / FP)")
    print(f"  true 1   {fn:8d}  {tp:8d}   (FN / TP)")
    print(f"  Sensitivity:  {sensitivity:.4f}")
    print(f"  Specificity:  {specificity:.4f}")
    print(f"  Balanced acc: {bacc:.4f}")
    print(f"  Pipeline fired: {tp + fp}  (TP={tp} FP={fp})")
    print(f"  Missed Michels: {fn}")
    print("=" * 60)

    if use_wandb:
        wandb.log({
            "final/sensitivity": sensitivity, "final/specificity": specificity,
            "final/bacc": bacc,
            "final/TP": tp, "final/TN": tn, "final/FP": fp, "final/FN": fn,
        })
        wandb.finish()


if __name__ == "__main__":
    t_start = time.perf_counter()
    main()
    elapsed = time.perf_counter() - t_start
    print(f"\nTotal wall-clock time: {elapsed:.1f} s  ({elapsed/60:.2f} min)")

# Runtime notes (single A100, ~19.6 ms per fired event):
#   Michel exhaustion   (36,120 s simulated)  -> ~12 min wall clock
#   No-Michel exhaustion (129,870 s simulated) -> ~42 min wall clock
# Four passes through the event pool:
#   bash scripts/07_stream_simulation.sh --use_wandb --sim_duration 519480.0
