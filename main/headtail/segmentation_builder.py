# Authored by Hilary Utaegbulam

"""Michel/brems segmentation builder: morphology, vertex finding, Dijkstra pathfinding."""
from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, List
from collections import deque
import heapq

from .constants import PDG_MICHEL, PDG_BREMS, PDG_MUON_POS, PDG_MUON_NEG
from types import SimpleNamespace
from collections import deque

try:
    from scipy import ndimage as ndi
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

try:
    from skimage.morphology import skeletonize as _skel
    HAVE_SKIMAGE = True
    def skeleton(mask: np.ndarray) -> np.ndarray:
        return _skel(mask.astype(bool))
except Exception:
    HAVE_SKIMAGE = False
    def skeleton(mask: np.ndarray) -> np.ndarray:
        return mask.astype(bool)

def _neighbors8(y, x, H, W):
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            yy, xx = y + dy, x + dx
            if 0 <= yy < H and 0 <= xx < W:
                yield yy, xx

def _nearest_true_pixel(mask: np.ndarray, ref_rc: Tuple[float, float]) -> Optional[Tuple[int,int]]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    vy, vx = float(ref_rc[0]), float(ref_rc[1])
    k = int(np.argmin((ys - vy)**2 + (xs - vx)**2))
    return (int(ys[k]), int(xs[k]))

def michel_end_dijkstra(
    michel_mask: np.ndarray,
    vertex_rc: Tuple[float, float],
    *,
    max_jump_dist: float = 5.0,
    jump_penalty_factor: float = 1.5
) -> Optional[Tuple[int, int]]:
    """
    Finds the Michel end by Dijkstra with 8-nb steps (+1 cost) and
    gap-bridging jumps (penalized by dist**jump_penalty_factor).
    """
    if vertex_rc is None or not np.any(michel_mask):
        return None

    coords = np.array(np.nonzero(michel_mask)).T
    if coords.shape[0] == 0:
        return None

    try:
        kdtree = cKDTree(coords)
    except Exception as e:
        print(f"Warning: cKDTree failed: {e}")
        if coords.shape[0] > 0:
            dists_sq = np.sum((coords - np.array(vertex_rc))**2, axis=1)
            farthest_idx = np.argmax(dists_sq)
            return tuple(coords[farthest_idx])
        return None

    # start at nearest Michel pixel to vertex
    _, start_idx = kdtree.query(vertex_rc)
    start_node = tuple(coords[start_idx])

    pq = [(0.0, start_node)]
    costs = {start_node: 0.0}

    while pq:
        cost, (y, x) = heapq.heappop(pq)
        if cost > costs.get((y, x), float('inf')):
            continue

        # cheap 8-connected steps
        for ny, nx in _neighbors8(y, x, michel_mask.shape[0], michel_mask.shape[1]):
            if michel_mask[ny, nx]:
                new_cost = cost + 1.0
                if new_cost < costs.get((ny, nx), float('inf')):
                    costs[(ny, nx)] = new_cost
                    heapq.heappush(pq, (new_cost, (ny, nx)))

        # jump steps to bridge gaps
        jump_indices = kdtree.query_ball_point([y, x], r=max_jump_dist)
        for idx in jump_indices:
            ny, nx = coords[idx]
            if abs(ny - y) <= 1 and abs(nx - x) <= 1:
                continue
            dist = float(np.hypot(ny - y, nx - x))
            jump_cost = dist ** jump_penalty_factor
            new_cost = cost + jump_cost
            if new_cost < costs.get((ny, nx), float('inf')):
                costs[(ny, nx)] = new_cost
                heapq.heappush(pq, (new_cost, (ny, nx)))

    if not costs:
        return None

    farthest_node, _ = max(costs.items(), key=lambda kv: kv[1])
    return (int(farthest_node[0]), int(farthest_node[1]))




# MICHEL / BREMS BUILDER 


def _disk(r: int) -> np.ndarray:
    r = int(max(0, r))
    if r == 0: return np.ones((1,1), bool)
    Y, X = np.ogrid[-r:r+1, -r:r+1]
    return (Y*Y + X*X) <= r*r

def bridge_gaps(mask: np.ndarray, radius: int) -> np.ndarray:
    if SCIPY_OK and radius > 0:
        return ndi.binary_closing(mask.astype(bool), structure=_disk(radius))
    return mask.astype(bool)

def largest_component(mask: np.ndarray) -> np.ndarray:
    if not SCIPY_OK: return mask.astype(bool)
    lab, n = ndi.label(mask, structure=np.ones((3,3), int))
    if n == 0: return mask & False
    sizes = np.bincount(lab.ravel())
    return lab == (np.argmax(sizes[1:]) + 1)

def _component_length(mask: np.ndarray, mode: str = "skeleton") -> int:
    if mode == "skeleton" and HAVE_SKIMAGE:
        return int(np.count_nonzero(skeleton(mask)))
    return int(np.count_nonzero(mask))

def choose_michel_close_to_mu(mu_skel: np.ndarray,
                              mi_mask: np.ndarray,
                              min_len: int = 0,
                              len_mode: str = "skeleton") -> np.ndarray:
    if not SCIPY_OK:
        return largest_component(mi_mask)
    lab, n = ndi.label(mi_mask, structure=np.ones((3,3), int))
    if n == 0:
        return mi_mask & False
    if n == 1:
        return mi_mask.astype(bool)
    dt_mu = ndi.distance_transform_edt(~mu_skel.astype(bool))
    cand = []
    for L in range(1, n+1):
        comp = (lab == L)
        ys, xs = np.nonzero(comp)
        if ys.size == 0: continue
        d = float(np.min(dt_mu[ys, xs])); ln = _component_length(comp, mode=len_mode)
        cand.append((d, ln, comp))
    if not cand: return mi_mask & False
    cand.sort(key=lambda t: t[0])
    for d, ln, comp in cand:
        if ln >= int(min_len): return comp
    return cand[0][2]

def vertex_bwmv(mu_skel: np.ndarray, primary_skel: np.ndarray, knobs: dict) -> Optional[Tuple[float,float]]:
    if not SCIPY_OK: return None
    dist_img, indices = ndi.distance_transform_edt(~primary_skel, return_indices=True)
    ymu, xmu = np.nonzero(mu_skel)
    if ymu.size == 0: return None
    d = dist_img[ymu, xmu]
    dmin = float(np.min(d)) if d.size else np.inf
    delta = float(knobs.get("REGION_DELTA_PX", 1.5))
    step  = float(knobs.get("REGION_DELTA_STEP", 0.5))
    dmax  = float(knobs.get("REGION_DELTA_MAX", 3.0))
    min_pairs = int(knobs.get("REGION_MIN_PAIRS", 12))
    for _ in range(8):
        keep = np.where(d <= dmin + delta)[0]
        if keep.size >= min_pairs or delta >= dmax:
            break
        delta += step
    if keep.size == 0:
        if d.size == 0: return None
        keep = np.array([int(np.argmin(d))])
    iy = indices[0, ymu[keep], xmu[keep]]
    ix = indices[1, ymu[keep], xmu[keep]]
    mids = np.column_stack(((ymu[keep] + iy)/2.0, (xmu[keep] + ix)/2.0))
    w    = 1.0 / (1.0 + d[keep])
    vtx_y = float(np.sum(w * mids[:,0]) / np.sum(w))
    vtx_x = float(np.sum(w * mids[:,1]) / np.sum(w))
    return (vtx_y, vtx_x)

def vertex_ncv(mu_skel: np.ndarray, primary_skel: np.ndarray, knobs: dict) -> Optional[Tuple[float,float]]:
    if not SCIPY_OK: return None
    dist_img, indices = ndi.distance_transform_edt(~primary_skel, return_indices=True)
    ymu, xmu = np.nonzero(mu_skel)
    if ymu.size == 0: return None
    d = dist_img[ymu, xmu]
    k = int(np.argmin(d))
    iy = int(indices[0, ymu[k], xmu[k]]); ix = int(indices[1, ymu[k], xmu[k]])
    return (float(iy), float(ix))

def _find_path_bfs(component_mask: np.ndarray, start_rc: Tuple[int, int]) -> Tuple[List[Tuple[int,int]], Tuple[int,int]]:
    if not component_mask[start_rc]:
        raise ValueError("start_rc not within component_mask.")
    q = deque([start_rc]); parent = {start_rc: None}; far = start_rc
    rows, cols = np.nonzero(component_mask); S = set(zip(rows, cols))
    while q:
        r, c = q.popleft(); far = (r, c)
        for dr in [-1,0,1]:
            for dc in [-1,0,1]:
                if dr==0 and dc==0: continue
                nb = (r+dr, c+dc)
                if nb in S and nb not in parent:
                    parent[nb] = (r, c); q.append(nb)
    path = []; cur = far
    while cur is not None:
        path.append(cur); cur = parent[cur]
    return path[::-1], far

def compute_michel_initial_dir(main_track_mask: np.ndarray,
                               mi_start_rc: Tuple[int, int],
                               steps: int = 5) -> Optional[Tuple[float, float]]:
    if not np.any(main_track_mask):
        return None
    try:
        path, _ = _find_path_bfs(main_track_mask, mi_start_rc)
    except Exception:
        return None
    if len(path) < 2: return None
    k = int(max(1, min(steps, len(path)-1)))
    seg = path[1:k+1]
    sy, sx = path[0]
    ys = np.array([p[0] for p in seg], dtype=float)
    xs = np.array([p[1] for p in seg], dtype=float)
    dy, dx = (float(np.mean(ys) - sy), float(np.mean(xs) - sx))
    fdy, fdx = (float(seg[0][0] - sy), float(seg[0][1] - sx))
    if (fdx * dx + fdy * dy) < 0.0:
        dy, dx = -dy, -dx
    return (dy, dx)

def compute_michel_com_vector(seg_result: dict) -> Optional[Tuple[float, float]]:
    """
    Computes a vector from the vertex to the COM of the full Michel cluster.
    Returns (dy, dx) in image coordinates or None if invalid.
    """
    vertex = seg_result.get("vertex")
    main_mask = seg_result.get("main_track_mask")
    brems_mask = seg_result.get("brems_mask")

    if vertex is None or main_mask is None or brems_mask is None:
        return None

    
    full_michel_mask = main_mask | brems_mask
    
    
    ys, xs = np.nonzero(full_michel_mask)
    if ys.size == 0:
        return None
    
    com_y = float(np.mean(ys))
    com_x = float(np.mean(xs))
    
    vy, vx = vertex # vertex is (y, x)
    
    
    dy = com_y - vy
    dx = com_x - vx
    
    return (dy, dx)

def grow_muon_by_connectivity(pixid: np.ndarray, knobs: dict) -> np.ndarray:
    mu_seed = (pixid == PDG_MUON_NEG) | (pixid == PDG_MUON_POS)
    if not np.any(mu_seed): return mu_seed

    
    bridge_r = int(knobs.get("MU_BRIDGE_RADIUS", 1))
    blob = bridge_gaps(mu_seed, bridge_r)

    # Everything that is NOT muon and NOT background is forbidden to enter
    forbid = (pixid != 0) & ~mu_seed

    if not SCIPY_OK:  
        return largest_component(mu_seed) 

    
    st = np.ones((3,3), bool)  # 8-connectivity
    last = None
    blob = blob.astype(bool)
    # hard cap just in case; usually converges in < 50 iters
    for _ in range(int(knobs.get("MU_MAX_ITERS", 2000))):
        last = blob
        ring = ndi.binary_dilation(blob, structure=st)
        blob = (ring | blob) & ~forbid
        if np.array_equal(blob, last):
            break
    return blob

# Bragg suppression in Michel near the vertex (old)

def _disk_local(r: int) -> np.ndarray:
    r = int(max(0, r))
    if r == 0: return np.ones((1,1), bool)
    Y, X = np.ogrid[-r:r+1, -r:r+1]
    return (Y*Y + X*X) <= r*r

def _unit_local(dy, dx):
    n = float(np.hypot(dy, dx))
    if n == 0.0: return 0.0, 0.0, 0.0
    return dy/n, dx/n, n

def _mu_axis_from_mask_near_vertex(mask: np.ndarray, vertex_rc, r_init=6, r_max=20):
    """Return (umu_y, umu_x) PCA axis of muon pixels near vertex, or None."""
    if mask is None or vertex_rc is None or not np.any(mask):
        return None
    vy, vx = float(vertex_rc[0]), float(vertex_rc[1])
    H, W = mask.shape
    Y, X = np.mgrid[0:H, 0:W]
    for r in range(int(r_init), int(r_max)+1, 2):
        ring = mask & ((Y - vy)**2 + (X - vx)**2 <= r*r)
        ys, xs = np.nonzero(ring)
        if ys.size >= 3:
            pts = np.column_stack((ys.astype(float) - vy, xs.astype(float) - vx))
            u, s, vh = np.linalg.svd(pts, full_matrices=False)
            dy, dx = vh[0, 0], vh[0, 1]
            n = float(np.hypot(dy, dx))
            if n == 0.0: 
                return None
            dy, dx = dy/n, dx/n
            if np.mean(pts @ np.array([dy, dx])) < 0:
                dy, dx = -dy, -dx
            return (dy, dx)
    return None

def suppress_bragg_in_michel_near_vertex(adc_img: np.ndarray, res, knobs: dict):
    """
    Zero high-ADC, (optionally) muon-aligned pixels that bled into the Michel mask near the vertex.
    """
    if (res is None) or (res.vertex is None) or (res.main_track_mask is None):
        #print("[bragg] early-exit: missing result, vertex, or Michel mask")
        return adc_img

    adc_out = adc_img.copy()
    vy, vx = float(res.vertex[0]), float(res.vertex[1])
    H, W = adc_out.shape

    rr = float(knobs.get("BRAGG_ROOT_RADIUS_PX", 10.0))
    use_align = bool(knobs.get("BRAGG_USE_ALIGN", True))
    cos_thresh = np.cos(np.deg2rad(float(knobs.get("BRAGG_MU_ALIGN_DEG", 35.0))))
    p = float(knobs.get("BRAGG_IN_MICHEL_PERCENTILE", 90.0))
    r_dil = int(knobs.get("BRAGG_DILATE_RADIUS", 0))

    # candidates = Michel pixels within rr of the vertex
    Y, X = np.mgrid[0:H, 0:W]
    rely, relx = (Y - vy), (X - vx)
    dist2 = rely*rely + relx*relx
    root_mask = (res.main_track_mask.astype(bool)) & (dist2 <= rr*rr)
    if not np.any(root_mask):
        #print("[bragg] early-exit: root_mask empty")
        return adc_out

    aligned_mask = root_mask
    if use_align:
        # Try to infer muon axis from any available muon mask near vertex
        mu_mask_for_axis = getattr(res, "mu_main_track_mask", None)
        if (mu_mask_for_axis is None) or (not np.any(mu_mask_for_axis)):
            mu_mask_for_axis = getattr(res, "mu_comp", None)
        umu = _mu_axis_from_mask_near_vertex(mu_mask_for_axis, res.vertex,
                                             r_init=int(knobs.get("AXIS_R_INIT", 6)),
                                             r_max=int(knobs.get("AXIS_R_MAX", 20)))
        if umu is None:
            pass
            #print("[bragg] no mu axis -> using radius-only")
        else:
            umu_y, umu_x = umu
            proj = rely*umu_y + relx*umu_x
            dist = np.hypot(rely, relx)
            with np.errstate(invalid='ignore', divide='ignore'):
                cosang_mu = proj / np.maximum(dist, 1e-6)
            aligned_mask = root_mask & (proj > 0) & (cosang_mu >= cos_thresh)
            if not np.any(aligned_mask):
                #print("[bragg] aligned_mask empty -> radius-only")
                aligned_mask = root_mask

    vals = adc_out[aligned_mask]
    if vals.size == 0:
        #print("[bragg] early-exit: no candidate ADCs")
        return adc_out

    thr = float(np.percentile(vals, p))
    zero_mask = aligned_mask & (adc_out >= thr)
    if r_dil > 0 and SCIPY_OK:
        zero_mask = ndi.binary_dilation(zero_mask, structure=_disk_local(r_dil))

    nz = int(np.count_nonzero(zero_mask))
    #print(f"[bragg] candidates={int(np.count_nonzero(aligned_mask))}, to_zero={nz}, thr={thr:.2f}, p={p}")
    if nz > 0:
        adc_out[zero_mask] = 0.0
    return adc_out


def segment_michel_track_and_brems(pixid: np.ndarray,
                                   knobs: dict,
                                   truth_vertex_rc: Optional[Tuple[float, float]] = None
                                   ) -> Optional[dict]:
    """
    Returns:
      {
        'vertex': (y,x) float,
        'mi_start': (y,x) int or None,
        'main_track_mask': bool array (Michel truth mask),
        'brems_mask': bool array (Brems truth mask),
      }
      - If truth_vertex_rc is provided, it is used directly.
      - Michel mask is abs(pixid) == PDG_MICHEL (9900011)
      - Brems mask is abs(pixid) >= PDG_BREMS   (9910000 base)
    """
    # truth masks (your new label scheme)
    michel_mask = (np.abs(pixid) == PDG_MICHEL)
    brems_mask  = (np.abs(pixid) >= PDG_BREMS)

    mu_mask = (pixid == PDG_MUON_NEG) | (pixid == PDG_MUON_POS)
    mu_comp = largest_component(mu_mask)
    mu_skel = skeleton(mu_comp)

    if not np.any(michel_mask):
        return None

    # vertex
    if truth_vertex_rc is not None:
        vertex = (float(truth_vertex_rc[0]), float(truth_vertex_rc[1]))
    else:
        if not np.any(mu_skel):
            return None
        bridged_mi = bridge_gaps(michel_mask, int(knobs.get("MI_BRIDGE_RADIUS", 2)))
        if not np.any(bridged_mi):
            return None
        primary_mi_comp = choose_michel_close_to_mu(
            mu_skel, bridged_mi,
            min_len=int(knobs.get("MIN_MICHEL_LEN_PIX", 6)),
            len_mode=knobs.get("MICHEL_LEN_MODE", "skeleton"),
        )
        if not np.any(primary_mi_comp):
            return None
        primary_skel = skeleton(primary_mi_comp)
        mode = str(knobs.get("VERTEX_MODE", "NCV")).upper()
        vertex = vertex_ncv(mu_skel, primary_skel, knobs) if mode == "NCV" else vertex_bwmv(mu_skel, primary_skel, knobs)
        if vertex is None:
            return None

    vy, vx = float(vertex[0]), float(vertex[1])

    # mi_start: nearest true Michel pixel to vertex 
    mi_start = _nearest_true_pixel(michel_mask, (vy, vx))

    
    if mi_start is not None and bool(knobs.get("SNAP_VERTEX_TO_ENDPOINT", True)):
        snap_max = float(knobs.get("VERTEX_SNAP_MAX_DIST_PX", 6.0))
        d2 = (mi_start[0] - vy)**2 + (mi_start[1] - vx)**2
        if d2 <= snap_max * snap_max:
            vertex = (float(mi_start[0]), float(mi_start[1]))

    # Return truth masks 
    return {
        "vertex": vertex,
        "mi_start": mi_start,
        "main_track_mask": michel_mask.astype(bool),
        "brems_mask": brems_mask.astype(bool),
    }



# DSNT Utilities 

# def spatial_softmax2d_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
#     B, C, H, W = logits.shape
#     T = max(float(temperature), 1e-8)
#     x = logits.reshape(B*C, -1) / T
#     x = x - x.max(dim=-1, keepdim=True).values
#     p = F.softmax(x, dim=-1)
#     return p.reshape(B, C, H, W)
