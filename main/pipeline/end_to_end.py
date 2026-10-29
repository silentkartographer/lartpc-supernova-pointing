#!/usr/bin/env python3
"""End-to-end inference. Raw U/V/Z images -> segmentation -> vertex -> Bragg suppression
-> two-point DSNT head-tail model -> voting -> 3D direction. With --evaluate, truth-based
plots (cos theta before/after voting, 3D angular error) are also produced.
"""
import os
import csv
import glob
import math
import tempfile
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random as _random
import matplotlib.pyplot as _plt

import warp as wp
from warpconvnet.geometry.types.voxels import Voxels

from larpoint.vertex.dsnt_vertex import (
    UNetDSNT,
    VertexConfig,
    coords_to_pixels,
    spatial_softmax2d,
)
from larpoint.segmentation.sparse_unet import (
    SparseUNet2D,
    SparseUNetConfig,
)
from larpoint.preprocessing.remove_muon import (
    load_sparse_model,
    load_vertex_model,
    process_file,
)

from larpoint.headtail.models import DenseModelSMP, DenseModel, DenseModelConvNeXt
from larpoint.headtail.physics_utils import (
    spatial_softmax2d_logits,
    dsnt_expectation,
    unit_from_points,
    spatial_softmax2d_logits as _ssm,
)
from larpoint.headtail.consistency import (
    apply_consistency_pipeline,
    reconstruct_3d_from_pair_batch,
    predict_view_from_3d_batch,
)
from larpoint.headtail.visualization import (
    _apply_priors_and_order_for_eval,
    save_thesis_vector_figure,
    save_bragg_figure_dense,
)
from larpoint.paths import (
    DEFAULT_SEG_CKPTS,
    DEFAULT_VERTEX_CKPTS,
    DEFAULT_BREMS_CKPT,
)


_plot_registry = {}


def _try_import_plot(fn_name):
    try:
        from larpoint.pipeline import plots as _m
        fn = getattr(_m, fn_name, None)
        if fn is not None:
            _plot_registry[fn_name] = fn
        else:
            print(f"  WARNING: {fn_name} not found in plots.py")
    except ImportError as e:
        print(f"  WARNING: could not import plots ({e}) — plots disabled")


# Group A — blind (no GT required); comment out any line to skip
_try_import_plot("plot_flip_drop_rates")
_try_import_plot("plot_cscore_distributions")
_try_import_plot("plot_angular_change")
_try_import_plot("plot_dsnt_sep_distributions")
_try_import_plot("plot_sep_vs_cscore_heatmap")
_try_import_plot("plot_pairwise_3d_agreement")
_try_import_plot("plot_event_gallery")

# Group B — evaluate (requires GT); comment out any line to skip
_try_import_plot("plot_costheta_before_after")
_try_import_plot("plot_angular_error_before_after")
_try_import_plot("plot_summary_accuracy")
_try_import_plot("plot_flip_confusion")
_try_import_plot("plot_3d_angular_error")
_try_import_plot("plot_3d_geometry_sanity")
_try_import_plot("plot_3d_high_confidence")


def _run_plot(name, *args, **kwargs):
    fn = _plot_registry.get(name)
    if fn is None:
        return
    try:
        fn(*args, **kwargs)
    except Exception as exc:
        import traceback
        print(f"  [plot] ERROR in {name}: {exc}")
        traceback.print_exc()


def load_vertex_models(ckpt_paths: dict, device: torch.device) -> dict:
    models = {}
    for view in ["U", "V", "Z"]:
        path = ckpt_paths[view]
        print(f"  [Vertex/{view}] {path}")
        ckpt = torch.load(path, map_location=device, weights_only=False)

        cfg_dict = ckpt.get("cfg")
        if cfg_dict is None:
            raise RuntimeError(f"Vertex checkpoint {view} missing 'cfg' key.")
        cfg = VertexConfig(**cfg_dict)

        model = UNetDSNT(encoder_name=cfg.encoder_name, encoder_weights=None).to(device)

        state = ckpt["model_state"]
        if list(state.keys())[0].startswith("module."):
            state = {k.replace("module.", ""): v for k, v in state.items()}

        model.load_state_dict(state)
        model.eval()
        models[view] = model
        print(f"    OK encoder={cfg.encoder_name}, epoch={ckpt.get('epoch','?')}")

    return models


@torch.no_grad()
def predict_vertex(vertex_models, img_np, view, device):
    model = vertex_models[view]
    img_t = torch.from_numpy(img_np).float().to(device).unsqueeze(0).unsqueeze(0)
    _, coords = model(img_t)
    H, W = img_np.shape
    tick_px = coords_to_pixels(coords[0, 0], W).item()
    wire_px = coords_to_pixels(coords[0, 1], H).item()
    return tick_px, wire_px


def load_seg_models(ckpt_paths: dict, device: torch.device) -> dict:
    models = {}
    for view in ["U", "V", "Z"]:
        path = ckpt_paths[view]
        print(f"  [Seg/{view}] {path}")
        model, cfg_seg = load_sparse_model(path, device)
        models[view] = (model, cfg_seg)
        print(f"    OK num_classes={cfg_seg.num_classes}, attn={cfg_seg.use_attention}")
    return models


def preprocess_single_file(raw_path, clean_path, output_path,
                            seg_models, vertex_models, device,
                            views=None, apply_bragg=True,
                            no_save=False):
    if views is None:
        views = ["U", "V", "Z"]

    # in no_save mode, write to a temp file, read back, then delete
    if no_save:
        fd, write_path = tempfile.mkstemp(suffix=".npz")
        os.close(fd)
        os.remove(write_path)
    else:
        write_path = output_path

    for view in views:
        seg_model, cfg_seg = seg_models[view]
        vm = vertex_models[view] if apply_bragg else None
        try:
            process_file(
                raw_path=raw_path,
                clean_path=clean_path,
                output_path=write_path,
                seg_model=seg_model,
                vertex_model=vm,
                device=device,
                cfg_seg=cfg_seg,
                view=view,
                apply_bragg_suppression=apply_bragg,
                vertex_csv_writer=None,
            )
        except Exception as e:
            print(f"  ERROR {os.path.basename(raw_path)} {view}: {e}")
            if no_save and os.path.exists(write_path):
                os.remove(write_path)
            return None

    if no_save:
        try:
            raw_npz   = np.load(raw_path, allow_pickle=True)
            ds        = _extract_drift_sign(raw_npz)
            tmp_npz   = np.load(write_path, allow_pickle=True)
            data_dict = dict(tmp_npz)
            data_dict["drift_sign"] = ds
        finally:
            if os.path.exists(write_path):
                os.remove(write_path)
        return data_dict

    return None


def _build_brems_model(cfg, device):
    arch = cfg.dense_arch.lower()
    if arch == "smp_unet":
        model = DenseModelSMP(
            in_ch=cfg.smp_in_channels,
            feat_ch=cfg.dense_feat_ch,
            num_classes=cfg.num_classes,
            enable_seg=cfg.enable_seg,
            enable_dirreg=cfg.enable_dirreg,
            encoder_name=cfg.smp_encoder,
            encoder_weights=cfg.smp_encoder_weights,
            decoder_channels=cfg.smp_decoder_channels,
        )
    elif arch == "convnextv2_unet":
        model = DenseModelConvNeXt(
            in_ch=1, feat_ch=cfg.dense_feat_ch,
            num_classes=cfg.num_classes,
            enable_seg=cfg.enable_seg, enable_dirreg=cfg.enable_dirreg,
            variant=cfg.cnv2_variant,
            decoder_scale=cfg.cnv2_decoder_scale,
            use_transpose=cfg.cnv2_use_transpose,
            skip_proj=cfg.cnv2_skip_proj,
        )
    else:
        model = DenseModel(
            in_ch=1, base=cfg.dense_base, feat_ch=cfg.dense_feat_ch,
            num_classes=cfg.num_classes,
            enable_seg=cfg.enable_seg, enable_dirreg=cfg.enable_dirreg,
        )
    return model.to(device)


def load_brems_models(ckpt_path: str, cfg, device: torch.device) -> dict:
    print(f"  [BremsFL] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    print(f"    Checkpoint keys: {[k for k in ckpt.keys() if k != 'rng_states']}")
    print(f"    Epoch: {ckpt.get('epoch', '?')}")

    models = {}
    for view in ["U", "V", "Z"]:
        if view not in ckpt:
            raise RuntimeError(
                f"View '{view}' not in checkpoint. "
                f"Available keys: {list(ckpt.keys())}"
            )
        model = _build_brems_model(cfg, device)
        state = ckpt[view]["model"]
        if list(state.keys())[0].startswith("module."):
            state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
        model.eval()
        models[view] = model
        print(f"    OK {view}")

    return models


def _extract_drift_sign(npz_data, default: float = 1.0) -> float:
    try:
        meta = npz_data["metadata"].item()
        # common path: per_view → {U|V|Z} → michel → tpc
        for v in ("U", "V", "Z"):
            try:
                tpc = meta["per_view"][v]["michel"]["tpc"]
                return 1.0 if int(tpc) == 1 else -1.0
            except (KeyError, TypeError):
                pass
        # flat path
        tpc = meta.get("tpc", None)
        if tpc is not None:
            return 1.0 if int(tpc) == 1 else -1.0
    except Exception:
        pass
    return float(default)


@torch.no_grad()
def _infer_brems_single_view(model, img_np, cfg, device):
    H, W = img_np.shape
    img_t = (torch.from_numpy(img_np)
             .float().to(device).unsqueeze(0).unsqueeze(0))

    # pad to multiple of 32
    pad_h = (32 - H % 32) % 32
    pad_w = (32 - W % 32) % 32
    if pad_h > 0 or pad_w > 0:
        img_t = F.pad(img_t, (0, pad_w, 0, pad_h), "constant", 0.0)

    out = model(img_t)

    logits_ab_adj, mx_sw, my_sw = _apply_priors_and_order_for_eval(out, cfg)

    ux, uy, _ = unit_from_points(
        mx_sw[:, 0], my_sw[:, 0],
        mx_sw[:, 1], my_sw[:, 1],
    )

    sep = float(torch.sqrt(
        (mx_sw[0, 0] - mx_sw[0, 1]) ** 2 +
        (my_sw[0, 0] - my_sw[0, 1]) ** 2
    ).cpu())

    return float(ux[0].cpu()), float(uy[0].cpu()), sep, out


def run_brems_inference(
    preproc_dir,
    out_dir: str,
    brems_models: dict,
    cfg,
    device: torch.device,
    views: list = None,
    preproc_data: dict = None,
) -> list:
    if views is None:
        views = ["U", "V", "Z"]

    pred_dir = os.path.join(out_dir, "brems_predictions")
    os.makedirs(pred_dir, exist_ok=True)

    if preproc_data is not None:
        event_items = list(preproc_data.items())
        print(f"  Using {len(event_items)} in-memory preprocessed events")
    else:
        npz_files = sorted(glob.glob(os.path.join(preproc_dir, "*.npz")))
        if not npz_files:
            print(f"  WARNING: No NPZ files found in {preproc_dir}")
            return []
        event_items = None
        print(f"  Found {len(npz_files)} preprocessed NPZ files")

    fnames      = []
    ux_raw      = {v: [] for v in views}
    uy_raw      = {v: [] for v in views}
    sep_raw     = {v: [] for v in views}
    drift_signs = []

    iterator = (
        tqdm(event_items, desc="Directional Inference", unit="file")
        if preproc_data is not None else
        tqdm(npz_files,   desc="Directional Inference", unit="file")
    )

    for item in iterator:
        if preproc_data is not None:
            fname, data = item
            images = {v: data[f"{v}_data"] for v in views}
            ds     = float(data.get("drift_sign", 1.0))
        else:
            npz_path = item
            fname    = os.path.basename(npz_path)
            try:
                data   = np.load(npz_path, allow_pickle=True)
                images = {v: data[f"{v}_data"] for v in views}
                ds     = _extract_drift_sign(data)
            except Exception as e:
                print(f"  ERROR loading {fname}: {e}")
                continue

        ok = True
        ux_ev, uy_ev, sep_ev = {}, {}, {}
        for v in views:
            img = images[v].astype(np.float32)
            if img.max() == 0:
                ok = False
                break
            ux_v, uy_v, sep_v, _ = _infer_brems_single_view(
                brems_models[v], img, cfg, device
            )
            ux_ev[v]  = ux_v
            uy_ev[v]  = uy_v
            sep_ev[v] = sep_v

        if not ok:
            print(f"  SKIP {fname}: blank view")
            continue

        fnames.append(fname)
        drift_signs.append(ds)
        for v in views:
            ux_raw[v].append(ux_ev[v])
            uy_raw[v].append(uy_ev[v])
            sep_raw[v].append(sep_ev[v])

    if not fnames:
        print("  No valid events processed.")
        return []

    N = len(fnames)
    print(f"  Processed {N} events, applying A→B consistency...")

    ux_t         = {v: torch.tensor(ux_raw[v], device=device) for v in views}
    uy_t         = {v: torch.tensor(uy_raw[v], device=device) for v in views}
    drift_sign_t = torch.tensor(drift_signs, device=device, dtype=torch.float32)

    cons      = apply_consistency_pipeline(ux_t, uy_t, drift_sign_t, cfg)
    ux_corr   = cons["ux_corrected"]
    uy_corr   = cons["uy_corrected"]
    flipped_A = cons["info_A"]["flipped_A"]
    flipped_B = cons["info_B"]["flipped_B"]
    dropped   = cons["info_B"]["dropped"]
    C_scores  = cons["info_B"]["C_scores"]

    pairs     = [("U", "V"), ("U", "Z"), ("V", "Z")]
    pair_keys = ["UV", "UZ", "VZ"]

    u3d_pair = {}
    for (vA, vB), pk in zip(pairs, pair_keys):
        u3d_pair[pk] = reconstruct_3d_from_pair_batch(
            ux_corr[vA], uy_corr[vA],
            ux_corr[vB], uy_corr[vB],
            vA, vB,
            drift_sign=drift_sign_t,
        )

    u3d_avg = F.normalize(
        torch.stack(list(u3d_pair.values()), dim=0).mean(dim=0), dim=1
    )

    for v in views:
        predict_view_from_3d_batch(u3d_avg, v, drift_sign=drift_sign_t)

    ux_raw_t = {v: torch.tensor(ux_raw[v], device=device) for v in views}
    uy_raw_t = {v: torch.tensor(uy_raw[v], device=device) for v in views}

    u3d_raw_list = []
    for vA, vB in pairs:
        u3d_raw_list.append(
            reconstruct_3d_from_pair_batch(
                ux_raw_t[vA], uy_raw_t[vA],
                ux_raw_t[vB], uy_raw_t[vB],
                vA, vB,
                drift_sign=drift_sign_t,
            )
        )
    u3d_raw_avg = F.normalize(
        torch.stack(u3d_raw_list, dim=0).mean(dim=0), dim=1
    )

    results = []
    for i, fname in enumerate(fnames):
        dropped_any = any(dropped[v][i].item() for v in views)

        ev = {
            "fname": fname,
            # Corrected 2D per-view
            "ux_U": float(ux_corr["U"][i]), "uy_U": float(uy_corr["U"][i]),
            "ux_V": float(ux_corr["V"][i]), "uy_V": float(uy_corr["V"][i]),
            "ux_Z": float(ux_corr["Z"][i]), "uy_Z": float(uy_corr["Z"][i]),
            # Raw 2D (before correction)
            "ux_U_raw": ux_raw["U"][i], "uy_U_raw": uy_raw["U"][i],
            "ux_V_raw": ux_raw["V"][i], "uy_V_raw": uy_raw["V"][i],
            "ux_Z_raw": ux_raw["Z"][i], "uy_Z_raw": uy_raw["Z"][i],
            # DSNT separation per view
            "sep_U": sep_raw["U"][i],
            "sep_V": sep_raw["V"][i],
            "sep_Z": sep_raw["Z"][i],
            # Drift sign
            "drift_sign": drift_signs[i],
            # Corrected 3D (average of 3 pairs)
            "u3d_x": float(u3d_avg[i, 0]),
            "u3d_y": float(u3d_avg[i, 1]),
            "u3d_z": float(u3d_avg[i, 2]),
            # Raw 3D (before correction)
            "u3d_raw_x": float(u3d_raw_avg[i, 0]),
            "u3d_raw_y": float(u3d_raw_avg[i, 1]),
            "u3d_raw_z": float(u3d_raw_avg[i, 2]),
            # Per-pair 3D (for pairwise agreement plot)
            "u3d_UV_x": float(u3d_pair["UV"][i, 0]),
            "u3d_UV_y": float(u3d_pair["UV"][i, 1]),
            "u3d_UV_z": float(u3d_pair["UV"][i, 2]),
            "u3d_UZ_x": float(u3d_pair["UZ"][i, 0]),
            "u3d_UZ_y": float(u3d_pair["UZ"][i, 1]),
            "u3d_UZ_z": float(u3d_pair["UZ"][i, 2]),
            "u3d_VZ_x": float(u3d_pair["VZ"][i, 0]),
            "u3d_VZ_y": float(u3d_pair["VZ"][i, 1]),
            "u3d_VZ_z": float(u3d_pair["VZ"][i, 2]),
            # Flip diagnostics
            "flipped_A_U": bool(flipped_A["U"][i]),
            "flipped_A_V": bool(flipped_A["V"][i]),
            "flipped_A_Z": bool(flipped_A["Z"][i]),
            "flipped_B_U": bool(flipped_B["U"][i]),
            "flipped_B_V": bool(flipped_B["V"][i]),
            "flipped_B_Z": bool(flipped_B["Z"][i]),
            # Drop flags
            "dropped_U": bool(dropped["U"][i]),
            "dropped_V": bool(dropped["V"][i]),
            "dropped_Z": bool(dropped["Z"][i]),
            "dropped_any": dropped_any,
            # C scores
            "C_UV_to_Z": float(C_scores["UV_to_Z"][i]),
            "C_UZ_to_V": float(C_scores["UZ_to_V"][i]),
            "C_VZ_to_U": float(C_scores["VZ_to_U"][i]),
        }
        results.append(ev)

        out_npz = os.path.join(pred_dir, fname)
        np.savez_compressed(
            out_npz,
            ux_corr   = np.array([ev["ux_U"], ev["ux_V"], ev["ux_Z"]], dtype=np.float32),
            uy_corr   = np.array([ev["uy_U"], ev["uy_V"], ev["uy_Z"]], dtype=np.float32),
            ux_raw    = np.array([ev["ux_U_raw"], ev["ux_V_raw"], ev["ux_Z_raw"]], dtype=np.float32),
            uy_raw    = np.array([ev["uy_U_raw"], ev["uy_V_raw"], ev["uy_Z_raw"]], dtype=np.float32),
            sep       = np.array([ev["sep_U"], ev["sep_V"], ev["sep_Z"]], dtype=np.float32),
            u3d       = np.array([ev["u3d_x"], ev["u3d_y"], ev["u3d_z"]], dtype=np.float32),
            u3d_raw   = np.array([ev["u3d_raw_x"], ev["u3d_raw_y"], ev["u3d_raw_z"]], dtype=np.float32),
            u3d_UV    = np.array([ev["u3d_UV_x"], ev["u3d_UV_y"], ev["u3d_UV_z"]], dtype=np.float32),
            u3d_UZ    = np.array([ev["u3d_UZ_x"], ev["u3d_UZ_y"], ev["u3d_UZ_z"]], dtype=np.float32),
            u3d_VZ    = np.array([ev["u3d_VZ_x"], ev["u3d_VZ_y"], ev["u3d_VZ_z"]], dtype=np.float32),
            drift_sign = np.float32(ev["drift_sign"]),
            views     = np.array(views),
        )

    csv_path = os.path.join(pred_dir, "predictions_summary.csv")
    if results:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        print(f"  Saved summary CSV: {csv_path}")

    n_dropped = sum(1 for r in results if r["dropped_any"])
    n_flipped = sum(
        1 for r in results
        if any(r[f"flipped_A_{v}"] or r[f"flipped_B_{v}"] for v in views)
    )
    print(f"  Events: {N}  |  Any flip: {n_flipped}  |  Any drop: {n_dropped}")
    return results


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


def _is_well_behaved(raw_npz_path, views=("Z", "U", "V"), boundary_px=2):
    PDG_MUON_POS = 13
    PDG_MUON_NEG = -13
    try:
        data = np.load(raw_npz_path, allow_pickle=True)
        for v in views:
            key = f"{v}_mask"
            if key not in data.files:
                continue
            mask    = data[key]
            muon_px = (mask == PDG_MUON_POS) | (mask == PDG_MUON_NEG)
            if not muon_px.any():
                continue
            H, W = muon_px.shape
            rows, cols = np.where(muon_px)
            near_edge = (
                (rows < boundary_px) |
                (rows >= H - boundary_px) |
                (cols < boundary_px) |
                (cols >= W - boundary_px)
            )
            if near_edge.any():
                return True
    except Exception:
        pass
    return False


def augment_results_with_gt(results, raw_dir, views, device,
                             preproc_dir=None, preproc_data=None):
    print(f"  [GT] Loading truth from {raw_dir} ...")
    n_ok = 0

    for r in tqdm(results, desc="Loading GT", unit="file"):
        fname    = r["fname"]
        raw_path = os.path.join(raw_dir, fname)

        if not os.path.exists(raw_path):
            alt = os.path.join(os.path.dirname(raw_dir), fname)
            if os.path.exists(alt):
                raw_path = alt
            else:
                print(f"  [GT] WARNING: raw file not found: {raw_path}")
                continue

        try:
            with np.load(raw_path, allow_pickle=True) as data:
                meta = data["metadata"].item()
                vpx, vpy, vpz = _extract_gt_direction(meta)
        except Exception as e:
            print(f"  [GT] WARNING: could not read {fname}: {e}")
            continue

        muon_near_edge = False
        try:
            if preproc_data is not None:
                pd        = preproc_data.get(fname, {})
                mask_keys = list(pd.keys())
                mask_src  = pd
            elif preproc_dir is not None:
                _pf       = np.load(os.path.join(preproc_dir, fname), allow_pickle=True)
                mask_keys = list(_pf.files)
                mask_src  = _pf
            else:
                mask_keys = []
                mask_src  = {}

            for v in ("Z", "U", "V"):
                key = f"{v}_mask"
                if key not in mask_keys:
                    continue
                px_mask = mask_src[key]
                muon_px = (px_mask == 13) | (px_mask == -13)
                if not muon_px.any():
                    continue
                H_m, W_m = muon_px.shape
                rows, cols = np.where(muon_px)
                if (
                    (rows < 2).any() or (rows >= H_m - 2).any() or
                    (cols < 2).any() or (cols >= W_m - 2).any()
                ):
                    muon_near_edge = True
                    break
        except Exception as e:
            print(f"  [WB] WARNING: could not check well-behaved for {fname}: {e}")
        r["well_behaved"] = muon_near_edge

        if vpx is None:
            print(f"  [GT] WARNING: no GT direction in metadata for {fname}")
            continue

        # normalise GT 3D
        norm = math.sqrt(vpx**2 + vpy**2 + vpz**2)
        if norm < 1e-9:
            continue
        vpx, vpy, vpz = vpx / norm, vpy / norm, vpz / norm

        r["gt_vpx"] = vpx
        r["gt_vpy"] = vpy
        r["gt_vpz"] = vpz

        ds   = r.get("drift_sign", 1.0)
        ds_t = torch.tensor([ds],           device=device, dtype=torch.float32)
        gt3d = torch.tensor([[vpx, vpy, vpz]], device=device, dtype=torch.float32)

        # GT 2D projections (geometry-aware)
        for v in views:
            gt_ux_v, gt_uy_v = predict_view_from_3d_batch(gt3d, v, drift_sign=ds_t)
            r[f"gt_ux_{v}"] = float(gt_ux_v[0].cpu())
            r[f"gt_uy_{v}"] = float(gt_uy_v[0].cpu())

        # 2D cos(θ): raw and corrected vs GT projection
        for v in views:
            r[f"cos_raw_{v}"]  = float(
                r[f"ux_{v}_raw"] * r[f"gt_ux_{v}"] +
                r[f"uy_{v}_raw"] * r[f"gt_uy_{v}"]
            )
            r[f"cos_corr_{v}"] = float(
                r[f"ux_{v}"] * r[f"gt_ux_{v}"] +
                r[f"uy_{v}"] * r[f"gt_uy_{v}"]
            )

        # 3D cos(θ): raw and corrected vs GT
        gt3d_np = np.array([vpx, vpy, vpz])
        r["cos_3d_corr"] = float(np.dot(
            [r["u3d_x"], r["u3d_y"], r["u3d_z"]], gt3d_np
        ))
        r["cos_3d_raw"] = float(np.dot(
            [r["u3d_raw_x"], r["u3d_raw_y"], r["u3d_raw_z"]], gt3d_np
        ))

        # geometry sanity: reconstruct 3D from GT 2D projections
        gt_ux_t = {v: torch.tensor([r[f"gt_ux_{v}"]], device=device, dtype=torch.float32)
                   for v in views}
        gt_uy_t = {v: torch.tensor([r[f"gt_uy_{v}"]], device=device, dtype=torch.float32)
                   for v in views}

        sanity_list = []
        for vA, vB in [("U", "V"), ("U", "Z"), ("V", "Z")]:
            sanity_list.append(
                reconstruct_3d_from_pair_batch(
                    gt_ux_t[vA], gt_uy_t[vA],
                    gt_ux_t[vB], gt_uy_t[vB],
                    vA, vB,
                    drift_sign=ds_t,
                )
            )
        u3d_sanity = F.normalize(
            torch.stack(sanity_list, dim=0).mean(dim=0), dim=1
        )
        r["cos_3d_gt_reco"] = float(
            np.dot(u3d_sanity[0].cpu().numpy(), gt3d_np)
        )

        n_ok += 1

    print(f"  [GT] Loaded truth for {n_ok}/{len(results)} events")


def parse_args():
    p = argparse.ArgumentParser(
        description="Michel direction pipeline — stages 1–4",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    p.add_argument("--raw_root",      required=True)
    p.add_argument("--cleansig_root", required=False)
    p.add_argument("--out_dir",       required=True)

    # Seg checkpoints
    p.add_argument("--ckpt_seg_U", default=DEFAULT_SEG_CKPTS["U"])
    p.add_argument("--ckpt_seg_V", default=DEFAULT_SEG_CKPTS["V"])
    p.add_argument("--ckpt_seg_Z", default=DEFAULT_SEG_CKPTS["Z"])

    # Vertex checkpoints
    p.add_argument("--ckpt_vertex_U", default=DEFAULT_VERTEX_CKPTS["U"])
    p.add_argument("--ckpt_vertex_V", default=DEFAULT_VERTEX_CKPTS["V"])
    p.add_argument("--ckpt_vertex_Z", default=DEFAULT_VERTEX_CKPTS["Z"])

    # Brems-fl checkpoint (single shared file)
    p.add_argument("--ckpt_brems", default=DEFAULT_BREMS_CKPT,
        help="Shared multi-view checkpoint (contains U/V/Z keys)")

    # Run control
    p.add_argument("--views", default="all",
        help="'all' or comma-separated, e.g. 'U,V'")
    p.add_argument("--no_bragg", action="store_true")
    p.add_argument("--skip_preprocess", action="store_true",
        help="Skip stages 1&2; use --preproc_dir directly")
    p.add_argument("--preproc_dir", default=None,
        help="Path to existing preprocessed NPZs (used with --skip_preprocess)")

    # Evaluate / plot options
    p.add_argument("--evaluate", action="store_true",
        help="Load GT from raw NPZs and produce Group B evaluation plots")
    p.add_argument("--no_save_preproc", action="store_true",
        help="Skip writing preprocessed NPZs to disk; pass images directly "
             "to the inference stage.")
    p.add_argument("--n_gallery", type=int, default=24,
        help="Number of events to include in the event gallery PDF")

    # Brems-fl model hyperparams
    p.add_argument("--dense_arch",   default="smp_unet",
        choices=["smp_unet", "dense", "convnextv2_unet"])
    p.add_argument("--smp_encoder",  default="densenet169")
    p.add_argument("--smp_in_channels", type=int, default=1)
    p.add_argument("--dense_feat_ch",   type=int, default=256)
    p.add_argument("--num_classes",     type=int, default=5)
    p.add_argument("--enable_seg",
        type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--enable_dirreg",
        type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--smp_encoder_weights", default=None)
    p.add_argument("--smp_decoder_channels", default="256,128,64,32,16",
        help="Comma-separated decoder channel sizes")

    # ConvNeXt options (only used when dense_arch=convnextv2_unet)
    p.add_argument("--cnv2_variant",       default="tiny")
    p.add_argument("--cnv2_decoder_scale", type=float, default=0.5)
    p.add_argument("--cnv2_use_transpose",
        type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--cnv2_skip_proj",
        type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--dense_base", type=int, default=32)

    # DSNT / consistency
    p.add_argument("--dsnt_temp",     type=float, default=1.0)
    p.add_argument("--prior_mode",    default="off",
        choices=["off", "soft", "hard"])
    p.add_argument("--prior_alpha",   type=float, default=0.3)
    p.add_argument("--prior_topk_percent", type=float, default=20.0)
    p.add_argument("--consistency_flip_threshold", type=float, default=0.90)
    p.add_argument("--consistency_drop_threshold", type=float, default=0.25)

    return p.parse_args()


def _make_brems_cfg(args):
    import argparse as _ap
    cfg = _ap.Namespace(**vars(args))
    cfg.smp_decoder_channels = tuple(
        int(x) for x in str(args.smp_decoder_channels).split(",")
    )
    cfg.bragg_make = False
    return cfg


def main():
    args   = parse_args()
    views  = (["U", "V", "Z"] if args.views.lower() == "all"
              else [v.strip().upper() for v in args.views.split(",")])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg    = _make_brems_cfg(args)

    print(f"Michel Direction Pipeline | device={device} | views={views}")
    print(f"  Bragg={not args.no_bragg}  skip_preprocess={args.skip_preprocess}  "
          f"no_save_preproc={args.no_save_preproc}  evaluate={args.evaluate}")
    print(f"  dense_arch={args.dense_arch}  smp_encoder={args.smp_encoder}")
    print(f"  out_dir={args.out_dir}")
    print(f"  Active plots: {sorted(_plot_registry.keys())}")

    os.makedirs(args.out_dir, exist_ok=True)

    if args.skip_preprocess:
        preproc_data = None
        preproc_dir  = args.preproc_dir or os.path.join(args.out_dir, "preprocessed")
        if not os.path.isdir(preproc_dir):
            raise ValueError(
                f"--skip_preprocess: preprocessed dir not found: {preproc_dir}\n"
                f"  Either run without --skip_preprocess first, "
                f"or pass --preproc_dir explicitly."
            )
        print(f"\n[Preprocess] SKIPPED — using {preproc_dir}")
    else:
        print("\n[0] Initializing Warp...")
        wp.init()
        print("    OK")

        print("\n[1] Loading vertex models...")
        vertex_models = load_vertex_models(
            {"U": args.ckpt_vertex_U,
             "V": args.ckpt_vertex_V,
             "Z": args.ckpt_vertex_Z},
            device,
        )

        print("\n[2] Loading segmentation models...")
        seg_models = load_seg_models(
            {"U": args.ckpt_seg_U,
             "V": args.ckpt_seg_V,
             "Z": args.ckpt_seg_Z},
            device,
        )

        raw_dir = (os.path.join(args.raw_root, "all")
                   if os.path.isdir(os.path.join(args.raw_root, "all"))
                   else args.raw_root)
        clean_dir = (os.path.join(args.cleansig_root, "all")
                     if os.path.isdir(os.path.join(args.cleansig_root, "all"))
                     else args.cleansig_root)

        raw_files = sorted(glob.glob(os.path.join(raw_dir, "*.npz")))

        if args.no_save_preproc:
            preproc_dir  = None
            preproc_data = {}
            print(f"\n[3] Preprocessing in-memory (--no_save_preproc)   {len(raw_files)} files")

            for raw_f in tqdm(raw_files, desc="Preprocessing", unit="file"):
                fname   = os.path.basename(raw_f)
                clean_f = os.path.join(clean_dir, fname)

                if not os.path.exists(clean_f):
                    print(f"    WARNING no cleansig for {fname}, skipping")
                    continue

                data_dict = preprocess_single_file(
                    raw_path=raw_f,
                    clean_path=clean_f,
                    output_path=None,
                    seg_models=seg_models,
                    vertex_models=vertex_models,
                    device=device,
                    views=views,
                    apply_bragg=not args.no_bragg,
                    no_save=True,
                )
                if data_dict is not None:
                    preproc_data[fname] = data_dict

            print(f"    OK in-memory preprocessing complete ({len(preproc_data)} events)")

        else:
            preproc_dir  = os.path.join(args.out_dir, "preprocessed")
            preproc_data = None
            os.makedirs(preproc_dir, exist_ok=True)
            print(f"\n[3] Preprocessing → {preproc_dir}   ({len(raw_files)} files)")

            for raw_f in tqdm(raw_files, desc="Preprocessing", unit="file"):
                fname   = os.path.basename(raw_f)
                clean_f = os.path.join(clean_dir, fname)
                out_f   = os.path.join(preproc_dir, fname)

                if not os.path.exists(clean_f):
                    print(f"    WARNING no cleansig for {fname}, skipping")
                    continue

                preprocess_single_file(
                    raw_path=raw_f,
                    clean_path=clean_f,
                    output_path=out_f,
                    seg_models=seg_models,
                    vertex_models=vertex_models,
                    device=device,
                    views=views,
                    apply_bragg=not args.no_bragg,
                )

            print("    OK preprocessing complete")

    print("\n[4] Loading brems-fl direction models...")
    brems_models = load_brems_models(args.ckpt_brems, cfg, device)

    print("\n[5] Running brems-fl inference + A→B consistency...")
    results = run_brems_inference(
        preproc_dir=preproc_dir,
        out_dir=args.out_dir,
        brems_models=brems_models,
        cfg=cfg,
        device=device,
        views=views,
        preproc_data=preproc_data if not args.skip_preprocess else None,
    )

    pred_dir = os.path.join(args.out_dir, "brems_predictions")
    plot_dir = os.path.join(args.out_dir, "plots")

    if results and _plot_registry:
        print(f"\n[6] Group A (blind) plots → {plot_dir}")
        _run_plot("plot_flip_drop_rates",         results, plot_dir, views=views)
        _run_plot("plot_cscore_distributions",    results, plot_dir, views=views)
        _run_plot("plot_angular_change",          results, plot_dir, views=views)
        _run_plot("plot_dsnt_sep_distributions",  results, plot_dir, views=views)
        _run_plot("plot_sep_vs_cscore_heatmap",   results, plot_dir, views=views)
        _run_plot("plot_pairwise_3d_agreement",   results, plot_dir)
        if preproc_dir is not None:
            _run_plot("plot_event_gallery",
                      results, preproc_dir, plot_dir,
                      views=views, n_events=args.n_gallery)
        else:
            print("  [plot] plot_event_gallery skipped (--no_save_preproc: no images on disk)")

    if args.evaluate and results:
        gt_raw_dir = (os.path.join(args.raw_root, "all")
                      if os.path.isdir(os.path.join(args.raw_root, "all"))
                      else args.raw_root)

        print(f"\n[7] Loading GT directions from {gt_raw_dir} ...")
        augment_results_with_gt(results, gt_raw_dir, views, device,
                                preproc_dir=preproc_dir,
                                preproc_data=preproc_data)

        event_plot_dir = os.path.join(args.out_dir, "event_plots")
        os.makedirs(event_plot_dir, exist_ok=True)

        gt_results = [r for r in results if "gt_vpx" in r]
        sample_evs = _random.sample(gt_results, min(35, len(gt_results)))
        print(f"\n[Event Plots] {len(sample_evs)} events → {event_plot_dir}")

        for ev in tqdm(sample_evs, desc="Event plots"):
            fname = ev["fname"]

            if preproc_data is not None:
                data_preproc = preproc_data[fname]
            else:
                data_preproc = np.load(os.path.join(preproc_dir, fname), allow_pickle=True)

            raw_npz_path = os.path.join(gt_raw_dir, fname)
            try:
                data_raw = np.load(raw_npz_path, allow_pickle=True)
            except Exception as e:
                print(f"  [event plot] could not load raw NPZ for {fname}: {e}")
                data_raw = None

            for v in views:
                img_np = data_preproc[f"{v}_data"].astype(np.float32)
                H_orig, W_orig = img_np.shape
                _, _, _, raw_out = _infer_brems_single_view(
                    brems_models[v], img_np, cfg, device)

                logits_ab_adj, mx_t, my_t = _apply_priors_and_order_for_eval(raw_out, cfg)
                prob_2ch = _ssm(logits_ab_adj, temperature=cfg.dsnt_temp)[0].detach().cpu().numpy()
                prob_2ch = prob_2ch[:, :H_orig, :W_orig]
                mx_2 = mx_t[0].detach().cpu().numpy()
                my_2 = my_t[0].detach().cpu().numpy()

                u_pred  = np.array([ev[f"ux_{v}"],    ev[f"uy_{v}"]],    dtype=np.float32)
                u_gt    = np.array([ev[f"gt_ux_{v}"], ev[f"gt_uy_{v}"]], dtype=np.float32)
                ang_err = float(np.degrees(np.arccos(np.clip(
                    u_pred[0]*u_gt[0] + u_pred[1]*u_gt[1], -1.0, 1.0))))

                sample_dict = {
                    "image": torch.from_numpy(img_np).unsqueeze(0),
                    "meta":  {"fname": fname},
                    "u_gt":  torch.from_numpy(u_gt),
                }

                stem = fname.replace(".npz", "")

                try:
                    save_thesis_vector_figure(
                        cfg=cfg, view=v, sample=sample_dict,
                        u_pred=u_pred, u_gt=u_gt,
                        prob_2ch=prob_2ch, mx_2=mx_2, my_2=my_2,
                        out_path=os.path.join(event_plot_dir, f"{stem}_{v}_vec.png"),
                        angular_error_deg=ang_err,
                    )
                except Exception as e:
                    print(f"  [vec plot] {fname} {v}: {e}")

                if data_raw is not None:
                    try:
                        raw_img = data_raw[f"{v}_data"].astype(np.float32)
                        H, W = raw_img.shape
                        cx, cy = W / 2.0, H / 2.0
                        L = min(H, W) * 0.2

                        fig, ax = _plt.subplots(figsize=(6, 4), dpi=150)
                        im = ax.imshow(raw_img, origin="lower", cmap="viridis",
                                       aspect="auto")
                        _plt.colorbar(im, ax=ax, label="ADC")
                        ax.arrow(cx, cy, L*u_gt[0],   L*u_gt[1],
                                 color="lime",    width=0, head_width=2.5,
                                 length_includes_head=True, label="GT")
                        ax.arrow(cx, cy, L*u_pred[0], L*u_pred[1],
                                 color="magenta", width=0, head_width=2.5,
                                 length_includes_head=True, label="Pred")
                        ax.legend(loc="upper right")
                        ax.set_title(f"{v} Plane Readout (Original Event)")
                        ax.set_xlabel("Time Tick"); ax.set_ylabel("Wire Number")
                        ax.text(0.02, 0.98, f"Err: {ang_err:.1f}°",
                                transform=ax.transAxes, va="top", color="w",
                                fontsize=11, fontweight="bold",
                                bbox=dict(facecolor="k", alpha=0.35, pad=2, edgecolor="none"))
                        _plt.tight_layout()
                        _plt.savefig(os.path.join(event_plot_dir, f"{stem}_{v}_orig.png"),
                                     dpi=150, bbox_inches="tight")
                        _plt.close(fig)
                    except Exception as e:
                        print(f"  [orig plot] {fname} {v}: {e}")

                try:
                    save_bragg_figure_dense(
                        cfg=cfg, view=v, sample=sample_dict,
                        u_pred_1x2=u_pred,
                        prob_2ch=prob_2ch, mx_2=mx_2, my_2=my_2,
                        out_path=os.path.join(event_plot_dir, f"{stem}_{v}_bragg.png"),
                        angular_error_deg=ang_err,
                    )
                except Exception as e:
                    print(f"  [bragg plot] {fname} {v}: {e}")

        n_gt = sum(1 for r in results if "gt_vpx" in r)
        print(f"    GT available for {n_gt}/{len(results)} events")

        if n_gt > 0 and _plot_registry:
            print(f"\n[8] Group B (evaluate) plots → {plot_dir}")
            _run_plot("plot_costheta_before_after",    results, plot_dir, views=views)
            _run_plot("plot_angular_error_before_after", results, plot_dir, views=views)
            _run_plot("plot_summary_accuracy",         results, plot_dir, views=views)
            _run_plot("plot_flip_confusion",           results, plot_dir, views=views)
            _run_plot("plot_3d_angular_error",         results, plot_dir)
            _run_plot("plot_3d_geometry_sanity",       results, plot_dir)
            _run_plot("plot_3d_high_confidence",       results, plot_dir, views=views)

    if args.evaluate and results:
        wb_results = [r for r in results if r.get("well_behaved", False)]
        n_wb       = len(wb_results)
        n_gt_total = sum(1 for r in results if "gt_vpx" in r)
        print(f"\n[9] Well-behaved events: {n_wb}/{n_gt_total} "
              f"({100*n_wb/max(n_gt_total,1):.1f}%)")

        if wb_results and _plot_registry:
            wb_plot_dir = os.path.join(args.out_dir, "plots_well_behaved")
            print(f"    Plots → {wb_plot_dir}")

            _run_plot("plot_flip_drop_rates",        wb_results, wb_plot_dir, views=views)
            _run_plot("plot_cscore_distributions",   wb_results, wb_plot_dir, views=views)
            _run_plot("plot_angular_change",         wb_results, wb_plot_dir, views=views)
            _run_plot("plot_dsnt_sep_distributions", wb_results, wb_plot_dir, views=views)
            _run_plot("plot_sep_vs_cscore_heatmap",  wb_results, wb_plot_dir, views=views)
            _run_plot("plot_pairwise_3d_agreement",  wb_results, wb_plot_dir)

            if preproc_dir is not None:
                _run_plot("plot_event_gallery",
                          wb_results, preproc_dir, wb_plot_dir,
                          views=views, n_events=args.n_gallery)

            wb_gt = [r for r in wb_results if "gt_vpx" in r]
            if wb_gt:
                _run_plot("plot_costheta_before_after",    wb_results, wb_plot_dir, views=views)
                _run_plot("plot_angular_error_before_after", wb_results, wb_plot_dir, views=views)
                _run_plot("plot_summary_accuracy",         wb_results, wb_plot_dir, views=views)
                _run_plot("plot_flip_confusion",           wb_results, wb_plot_dir, views=views)
                _run_plot("plot_3d_angular_error",         wb_results, wb_plot_dir)
                _run_plot("plot_3d_geometry_sanity",       wb_results, wb_plot_dir)
                _run_plot("plot_3d_high_confidence",       wb_results, wb_plot_dir, views=views)

    print(f"\nDone.")
    if preproc_dir is not None:
        print(f"  Preprocessed NPZs : {preproc_dir}")
    else:
        print(f"  Preprocessed NPZs : (in-memory only, --no_save_preproc)")
    print(f"  Predictions       : {pred_dir}/")
    print(f"  Summary CSV       : {pred_dir}/predictions_summary.csv")
    if _plot_registry:
        print(f"  Plots             : {plot_dir}/")
        if args.evaluate:
            print(f"  Well-behaved plots: {os.path.join(args.out_dir, 'plots_well_behaved')}/")


if __name__ == "__main__":
    main()
