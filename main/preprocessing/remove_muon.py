#!/usr/bin/env python3
"""Remove the parent muon and its Bragg peak from raw images.

For every event and view: segment with the sparse U-Net, keep only Michel+brems pixels, predict
the muon-Michel vertex with the DSNT vertex model, and zero the high-ADC pixels inside a
10-pixel disk around that vertex. The output NPZs are the input to the head-tail model.
"""
import os
import glob
import argparse
import csv
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import warp as wp
from warpconvnet.geometry.types.voxels import Voxels

from larpoint.segmentation.sparse_unet import (
    SparseUNet2D,
    SparseUNetConfig,
    robust_zscore_gpu,
    multiscale_dog_score_gpu,
    create_sparse_mask_iceberg_gpu,
    _build_features,
    _pick,
)
from larpoint.vertex.dsnt_vertex import (
    UNetDSNT,
    VertexConfig,
    coords_to_pixels,
    spatial_softmax2d,
)
from larpoint.paths import (
    DEFAULT_SEG_CKPTS,
    DEFAULT_VERTEX_CKPTS,
    OUTPUT_ROOT,
)
HAVE_WARP = True


def load_sparse_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = ckpt.get("cfg", None)
    if cfg_dict is None:
        raise RuntimeError("Checkpoint missing cfg; cannot reconstruct model config safely.")

    cfg = SparseUNetConfig(**cfg_dict)
    print(f"  Seg model: num_classes={cfg.num_classes}, low_thr={cfg.low_thr}, high_thr={cfg.high_thr}")

    model = SparseUNet2D(cfg).to(device)
    state = ckpt.get("model_state", ckpt)
    if list(state.keys())[0].startswith("module."):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def load_vertex_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = ckpt.get("cfg", None)
    if cfg_dict is None:
        raise RuntimeError("Vertex checkpoint missing cfg")

    cfg = VertexConfig(**cfg_dict)
    print(f"  Vertex model: encoder={cfg.encoder_name}")

    model = UNetDSNT(encoder_name=cfg.encoder_name, encoder_weights=None).to(device)
    state = ckpt.get("model_state", ckpt)
    if list(state.keys())[0].startswith("module."):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def suppress_bragg_using_predictions(
    adc_image: np.ndarray,
    pred_michel_mask: np.ndarray,
    pred_muon_mask: np.ndarray,
    vertex_wire: float,
    vertex_tick: float,
    radius_px: int = 10,
    percentile: float = 90.0,
    use_muon_alignment: bool = True,
    muon_align_deg: float = 35.0,
):
    H, W = adc_image.shape
    adc_out = adc_image.copy()

    yy, xx = np.ogrid[:H, :W]
    dy = yy - vertex_wire
    dx = xx - vertex_tick
    dist2 = dx**2 + dy**2

    root_mask = pred_michel_mask.astype(bool) & (dist2 <= radius_px**2)
    if not np.any(root_mask):
        return adc_out

    if use_muon_alignment and pred_muon_mask is not None and np.any(pred_muon_mask):
        muon_pixels = np.column_stack(np.where(pred_muon_mask))
        if len(muon_pixels) > 10:
            mu_dy = muon_pixels[:, 0].mean() - vertex_wire
            mu_dx = muon_pixels[:, 1].mean() - vertex_tick
            mu_norm = np.sqrt(mu_dx**2 + mu_dy**2)

            if mu_norm > 1e-6:
                mu_dy /= mu_norm
                mu_dx /= mu_norm
                cos_theta = (dx * mu_dx + dy * mu_dy) / (np.sqrt(dist2) + 1e-9)
                cos_threshold = np.cos(np.deg2rad(muon_align_deg))
                aligned_mask = root_mask & (cos_theta >= cos_threshold) & (dist2 > 0)
            else:
                aligned_mask = root_mask
        else:
            aligned_mask = root_mask
    else:
        aligned_mask = root_mask

    if not np.any(aligned_mask):
        return adc_out

    vals = adc_out[aligned_mask]
    thr = float(np.percentile(vals, percentile))
    adc_out[aligned_mask & (adc_out >= thr)] = 0.0
    return adc_out


@torch.no_grad()
def suppress_bragg_vertex_only(
    adc_image: np.ndarray,
    vertex_wire: float,
    vertex_tick: float,
    radius_px: int = 10,
    percentile: float = 95.0,
    always_zero_vertex: bool = True,
):
    H, W = adc_image.shape
    adc_out = adc_image.copy()

    if vertex_wire is None or vertex_tick is None:
        return adc_out
    if not (0 <= vertex_wire < H and 0 <= vertex_tick < W):
        return adc_out

    yy, xx = np.ogrid[:H, :W]
    dy = yy - vertex_wire
    dx = xx - vertex_tick
    dist2 = dx**2 + dy**2

    disk = dist2 <= radius_px**2
    if not np.any(disk):
        return adc_out

    thr = float(np.percentile(adc_out[disk], percentile))
    zero_mask = disk & (adc_out >= thr)

    if always_zero_vertex:
        vy = int(np.clip(round(vertex_wire), 0, H - 1))
        vx = int(np.clip(round(vertex_tick), 0, W - 1))
        zero_mask[vy, vx] = True

    adc_out[zero_mask] = 0.0
    return adc_out


def validate_vertex_prediction(
    vertex_wire: float,
    vertex_tick: float,
    pred_michel_mask: np.ndarray,
    pred_muon_mask: np.ndarray,
    margin_px: int = 20,
):
    H, W = pred_michel_mask.shape

    if not (0 <= vertex_wire < H and 0 <= vertex_tick < W):
        return False, f"out_of_bounds(wire={vertex_wire:.1f},tick={vertex_tick:.1f})"

    y_int = int(np.clip(vertex_wire, 0, H - 1))
    x_int = int(np.clip(vertex_tick, 0, W - 1))

    y_min, y_max = max(0, y_int - margin_px), min(H, y_int + margin_px + 1)
    x_min, x_max = max(0, x_int - margin_px), min(W, x_int + margin_px + 1)

    region_michel = pred_michel_mask[y_min:y_max, x_min:x_max]
    if not np.any(region_michel):
        if pred_muon_mask is not None:
            region_muon = pred_muon_mask[y_min:y_max, x_min:x_max]
            if not np.any(region_muon):
                return False, f"no_signal_near_vertex(margin={margin_px}px)"
        else:
            return False, f"no_michel_near_vertex(margin={margin_px}px)"

    return True, "valid"


@torch.no_grad()
def process_file(
    raw_path, clean_path, output_path,
    seg_model, vertex_model, device, cfg_seg, view,
    apply_bragg_suppression=True,
    vertex_csv_writer=None,
):
    fname = os.path.basename(clean_path)

    with np.load(raw_path, allow_pickle=True) as data_r, \
         np.load(clean_path, allow_pickle=True) as data_c:
        img_np     = data_r[f"{view}_data"]
        mask_gt_np = data_c[f"{view}_mask"]
        metadata   = data_c["metadata"].item()

    H, W = img_np.shape
    img_tensor = torch.from_numpy(img_np).float().to(device)

    # Step 1: vertex prediction
    vertex_wire_pred = vertex_tick_pred = None
    vertex_valid  = False
    vertex_reason = "not_attempted"

    if apply_bragg_suppression:
        try:
            img_for_vertex = img_tensor.unsqueeze(0).unsqueeze(0)
            _, coords_pred = vertex_model(img_for_vertex)
            vertex_tick_pred = coords_to_pixels(coords_pred[0, 0], W).item()
            vertex_wire_pred = coords_to_pixels(coords_pred[0, 1], H).item()
        except Exception as e:
            vertex_reason = f"prediction_failed({str(e)[:50]})"

    # Step 2: ICEBERG masking
    low_v        = float(_pick(cfg_seg.low_thr, view))
    high_v       = float(_pick(cfg_seg.high_thr, view))
    sigmas_v     = tuple(_pick(cfg_seg.sigmas, view))
    use_maxima_v = bool(_pick(cfg_seg.use_maxima_seeds, view))
    seed_min_v   = _pick(cfg_seg.seed_min, view, default=None)

    z_tensor, z_med, z_sigma = robust_zscore_gpu(img_tensor, eps=1e-6)
    z_for_score = z_tensor.abs() if cfg_seg.use_abs_z else z_tensor

    score_tensor = multiscale_dog_score_gpu(
        img_hw=z_for_score, sigmas=sigmas_v, k=cfg_seg.dog_k,
        truncate=cfg_seg.truncate, scale_normalize=cfg_seg.scale_normalize,
        use_abs_response=True,
    )
    input_mask, seeds, allowed = create_sparse_mask_iceberg_gpu(
        score_hw=score_tensor, low=low_v, high=high_v,
        max_iters=cfg_seg.hyst_max_iters, use_maxima_seeds=use_maxima_v,
        seed_min=seed_min_v,
    )

    # Step 3: segmentation
    coords = torch.nonzero(input_mask, as_tuple=False).int()

    if len(coords) == 0:
        pred_mask_np      = np.zeros((H, W), dtype=np.uint8)
        pred_muon_mask_np = None
    else:
        feats   = _build_features(
            img_tensor, z_tensor, score_tensor, input_mask, high_v, cfg_seg.in_channels)
        offsets = torch.tensor([0, len(coords)], dtype=torch.int32, device=device)
        voxels  = Voxels(coords, feats, offsets)

        logits   = seg_model(voxels).features
        pred_cls = logits.argmax(dim=1)

        pred_mask_gpu      = torch.zeros((H, W), dtype=torch.bool, device=device)
        pred_muon_mask_gpu = torch.zeros((H, W), dtype=torch.bool, device=device)

        if cfg_seg.num_classes == 2:
            keep_idx = pred_cls == 1
            muon_idx = None
        elif cfg_seg.num_classes == 3:
            keep_idx = pred_cls == 2
            muon_idx = pred_cls == 1
        elif cfg_seg.num_classes == 4:
            keep_idx = (pred_cls == 2) | (pred_cls == 3)
            muon_idx = pred_cls == 1

        if keep_idx.any():
            kc = coords[keep_idx].long()
            pred_mask_gpu[kc[:, 0], kc[:, 1]] = True

        if muon_idx is not None and muon_idx.any():
            mc = coords[muon_idx].long()
            pred_muon_mask_gpu[mc[:, 0], mc[:, 1]] = True
            pred_muon_mask_np = pred_muon_mask_gpu.cpu().numpy()
        else:
            pred_muon_mask_np = None

        pred_mask_np = pred_mask_gpu.cpu().numpy().astype(np.uint8)

    muon_removal_reverted = False
    if not np.any(pred_mask_np):
        input_mask_np_temp = input_mask.cpu().numpy()
        if np.any(input_mask_np_temp):
            print(f"⚠️  {fname}: Michel mask empty after muon removal, keeping all ICEBERG pixels")
            pred_mask_np = input_mask_np_temp.astype(np.uint8)
            muon_removal_reverted = True

    # Step 4: validate vertex
    if apply_bragg_suppression and vertex_wire_pred is not None and vertex_tick_pred is not None:
        if (0 <= vertex_wire_pred < H) and (0 <= vertex_tick_pred < W):
            vertex_valid  = True
            vertex_reason = "in_bounds"
        else:
            vertex_valid  = False
            vertex_reason = f"out_of_bounds(wire={vertex_wire_pred:.1f},tick={vertex_tick_pred:.1f})"

    # Step 5: Bragg suppression
    if apply_bragg_suppression and vertex_valid:
        try:
            img_suppressed_np = suppress_bragg_vertex_only(
                adc_image=img_np,
                vertex_wire=vertex_wire_pred,
                vertex_tick=vertex_tick_pred,
                radius_px=10,
                percentile=85.0,
                always_zero_vertex=True,
            )
            bragg_applied = True
        except Exception as e:
            img_suppressed_np = img_np.copy()
            vertex_reason     = f"suppression_failed({str(e)[:50]})"
            bragg_applied     = False
    else:
        img_suppressed_np = img_np.copy()
        bragg_applied = False

    # Step 6: final masking
    pred_bool = pred_mask_np.astype(bool)

    img_masked_np = np.zeros_like(img_suppressed_np)
    img_masked_np[pred_bool] = img_suppressed_np[pred_bool]

    img_masked_no_bragg_np = np.zeros_like(img_np)
    img_masked_no_bragg_np[pred_bool] = img_np[pred_bool].copy()

    bragg_suppression_reverted = False
    if bragg_applied and not np.any(img_masked_np) and np.any(img_masked_no_bragg_np):
        print(f"⚠️  {fname}: Bragg suppression blanked output, reverting")
        img_masked_np     = img_masked_no_bragg_np.copy()
        img_suppressed_np = img_np.copy()
        bragg_applied     = False
        bragg_suppression_reverted = True
        vertex_reason    += "_reverted_blank"

    input_mask_np     = input_mask.detach().cpu().numpy().astype(np.uint8)
    pred_mask_u8      = pred_mask_np.astype(np.uint8)
    pred_muon_mask_u8 = (pred_muon_mask_np.astype(np.uint8)
                         if pred_muon_mask_np is not None
                         else np.zeros((H, W), dtype=np.uint8))

    # Step 7: save
    new_data = {
        # Primary output: Michel/Brems pixels only, Bragg peak removed
        f"{view}_data": img_masked_np.copy(),
        # Same mask but without Bragg suppression — for comparison
        f"{view}_data_no_bragg": img_masked_no_bragg_np.copy(),
        # Unmodified raw ADC
        f"{view}_data_raw": img_np.copy(),
        # Full image with Bragg suppression applied (before Michel masking)
        f"{view}_data_bragg_suppressed": img_suppressed_np.copy(),
        f"{view}_pred_mask": pred_mask_u8.copy(),
        f"{view}_pred_muon_mask": pred_muon_mask_u8.copy(),
        f"{view}_input_mask": input_mask_np.copy(),
        f"{view}_mask": mask_gt_np.copy(),
        f"{view}_pred_vertex_wire": np.array([vertex_wire_pred if vertex_wire_pred is not None else -999.0]),
        f"{view}_pred_vertex_tick": np.array([vertex_tick_pred if vertex_tick_pred is not None else -999.0]),
    }

    final_dict = {}
    if os.path.exists(output_path):
        try:
            with np.load(output_path, allow_pickle=True) as existing_f:
                for k in existing_f.files:
                    final_dict[k] = (existing_f[k].copy()
                                     if isinstance(existing_f[k], np.ndarray)
                                     else existing_f[k])
        except Exception as e:
            print(f"Warning: could not read {output_path}: {e}")

    final_dict["metadata"] = metadata
    final_dict.update(new_data)
    np.savez(output_path, **final_dict)

    if vertex_csv_writer is not None:
        vertex_csv_writer.writerow({
            "filename": fname,
            "view": view,
            "vertex_wire_pred": f"{vertex_wire_pred:.2f}" if vertex_wire_pred is not None else "N/A",
            "vertex_tick_pred": f"{vertex_tick_pred:.2f}" if vertex_tick_pred is not None else "N/A",
            "vertex_valid": str(vertex_valid),
            "vertex_reason": vertex_reason,
            "bragg_applied": str(bragg_applied),
            "muon_removal_reverted": str(muon_removal_reverted),
            "bragg_suppression_reverted": str(bragg_suppression_reverted),
        })

    try:
        with np.load(output_path, allow_pickle=True) as verify:
            if (f"{view}_data" in verify.files and f"{view}_data_raw" in verify.files
                    and np.array_equal(verify[f"{view}_data"], verify[f"{view}_data_raw"])):
                print(f"❌ WARNING: {fname} — masked and raw are identical!")
    except Exception as e:
        print(f"❌ Verification failed for {output_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Offline segmentation + vertex-based Bragg suppression preprocessing"
    )
    parser.add_argument("--raw_root",      type=str, required=True)
    parser.add_argument("--cleansig_root", type=str, required=True)
    parser.add_argument("--out_dir",       type=str, default=str(OUTPUT_ROOT / "preprocess"))
    parser.add_argument("--views",         type=str, required=True)
    parser.add_argument("--single_dir",    action="store_true",
        help="Process a single directory without train/val/test splits")

    parser.add_argument("--ckpt_seg_U",    type=str, default=DEFAULT_SEG_CKPTS["U"])
    parser.add_argument("--ckpt_seg_V",    type=str, default=DEFAULT_SEG_CKPTS["V"])
    parser.add_argument("--ckpt_seg_Z",    type=str, default=DEFAULT_SEG_CKPTS["Z"])
    parser.add_argument("--ckpt_vertex_U", type=str, default=DEFAULT_VERTEX_CKPTS["U"])
    parser.add_argument("--ckpt_vertex_V", type=str, default=DEFAULT_VERTEX_CKPTS["V"])
    parser.add_argument("--ckpt_vertex_Z", type=str, default=DEFAULT_VERTEX_CKPTS["Z"])
    parser.add_argument("--no_bragg_suppression", action="store_true")

    args = parser.parse_args()

    wp.init()
    print("✓ Warp initialized")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    seg_ckpts    = {"U": args.ckpt_seg_U,    "V": args.ckpt_seg_V,    "Z": args.ckpt_seg_Z}
    vertex_ckpts = {"U": args.ckpt_vertex_U, "V": args.ckpt_vertex_V, "Z": args.ckpt_vertex_Z}

    if args.views.lower() == "all":
        views = ["U", "V", "Z"]
    else:
        views = [v.strip().upper() for v in args.views.replace(",", " ").split()]

    apply_bragg = not args.no_bragg_suppression
    print(f"Bragg suppression: {'ON' if apply_bragg else 'OFF'}\n")

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "vertex_predictions.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        "filename", "view", "vertex_wire_pred", "vertex_tick_pred",
        "vertex_valid", "vertex_reason", "bragg_applied",
        "muon_removal_reverted", "bragg_suppression_reverted",
    ])
    csv_writer.writeheader()
    print(f"Logging vertex predictions to: {csv_path}\n")

    for view in views:
        seg_ckpt    = seg_ckpts[view]
        vertex_ckpt = vertex_ckpts[view]

        if seg_ckpt is None or not os.path.exists(seg_ckpt):
            print(f"Skipping {view}: seg checkpoint not found ({seg_ckpt})")
            continue

        apply_bragg_view = apply_bragg
        if apply_bragg and (vertex_ckpt is None or not os.path.exists(vertex_ckpt)):
            print(f"WARNING: vertex checkpoint not found for {view}, disabling Bragg suppression for this view")
            apply_bragg_view = False

        print(f"\nView: {view}")
        seg_model, cfg_seg = load_sparse_model(seg_ckpt, device)
        vertex_model = load_vertex_model(vertex_ckpt, device) if apply_bragg_view else None

        splits = [None] if args.single_dir else ["train", "val", "test"]

        for split in splits:
            if split is None:
                raw_dir   = os.path.join(args.raw_root, "all")
                clean_dir = os.path.join(args.cleansig_root, "all")
                out_dir   = os.path.join(args.out_dir, "all")
                print("  Processing single directory")
            else:
                raw_dir   = os.path.join(args.raw_root, split, "all")
                clean_dir = os.path.join(args.cleansig_root, split, "all")
                if not os.path.exists(raw_dir):
                    raw_dir   = os.path.join(args.raw_root, split)
                    clean_dir = os.path.join(args.cleansig_root, split)
                out_dir = os.path.join(args.out_dir, split, "all")
                print(f"  Split: {split}")

            os.makedirs(out_dir, exist_ok=True)
            raw_files = sorted(glob.glob(os.path.join(raw_dir, "*.npz")))

            if not raw_files:
                print(f"  No files in {raw_dir}")
                continue

            for raw_f in tqdm(raw_files, desc=f"{view}/{split or 'all'}"):
                fname   = os.path.basename(raw_f)
                clean_f = os.path.join(clean_dir, fname)
                out_f   = os.path.join(out_dir, fname)

                if not os.path.exists(clean_f):
                    continue

                try:
                    process_file(
                        raw_f, clean_f, out_f,
                        seg_model, vertex_model, device, cfg_seg, view,
                        apply_bragg_suppression=apply_bragg_view,
                        vertex_csv_writer=csv_writer if apply_bragg_view else None,
                    )
                except Exception as e:
                    print(f"\nError on {fname}: {e}")
                    if apply_bragg_view and csv_writer is not None:
                        csv_writer.writerow({
                            "filename": fname, "view": view,
                            "vertex_wire_pred": "ERROR", "vertex_tick_pred": "ERROR",
                            "vertex_valid": "False",
                            "vertex_reason": f"processing_error({str(e)[:50]})",
                            "bragg_applied": "False",
                            "muon_removal_reverted": "False",
                            "bragg_suppression_reverted": "False",
                        })

    csv_file.close()
    print(f"\nDone. Vertex log: {csv_path}\n")


if __name__ == "__main__":
    main()
