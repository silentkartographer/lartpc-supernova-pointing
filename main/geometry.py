"""Wire-plane geometry for ICEBERG / DUNE horizontal-drift LArTPCs.

A pure-NumPy reference implementation of every coordinate transformation used by in this project:

* 3D direction  ->  2D (time tick, wire) direction in each of the U, V, Z readout planes
* 2D directions from two planes  ->  3D direction
* the U/V/Z geometric consistency condition  a_U + a_V = 2 a_Z cos(phi)
* the two-stage voting system (tick-sign majority, then round-robin held-out consistency)

The derivations are written out in ``docs/coordinate_transforms.md``; the equation numbers in
the docstrings below refer to that page. The batched PyTorch versions used during training and
inference live in :mod:`main.headtail.consistency` and are tested against this module.

Conventions
-----------
* A 2D image direction is ``q = (d_tick, d_wire)`` -- time tick first, wire index second.
* ``s_d`` is the drift sign: +1 for TPC 1, -1 for TPC 0. It fixes the sign relating increasing
  time tick to the global x axis. The wire axis lies in the readout plane and carries no drift sign.
* ``phi_P`` parameterizes the wire normal as (0, -sin(phi_P), cos(phi_P)):
  Z = 0 deg, U = +35.71 deg, V = -35.71 deg.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------------------------
# ICEBERG readout constants
# --------------------------------------------------------------------------------------------
PLANES: Tuple[str, ...] = ("U", "V", "Z")

PHI_DEG: Dict[str, float] = {"U": +35.71, "V": -35.71, "Z": 0.0}
"""Plane-angle parameter defining (0, -sin(phi), cos(phi)), in degrees."""

PITCH_CM: Dict[str, float] = {"U": 0.4667, "V": 0.4667, "Z": 0.479}
"""Wire pitch of each plane (cm per wire)."""

V_DRIFT_CM_PER_NS: float = 1.60e-4
"""Bulk electron drift speed in liquid argon (cm/ns)."""

T_TICK_NS: float = 500.0
"""ADC sampling period (ns per tick)."""

TICK_CM: float = V_DRIFT_CM_PER_NS * T_TICK_NS
"""Drift distance per time tick (0.08 cm)."""

PAIRS: Tuple[Tuple[str, str, str], ...] = (("U", "V", "Z"), ("U", "Z", "V"), ("V", "Z", "U"))
"""(P, Q, held-out R) combinations used by the round-robin consistency test."""


def drift_sign(tpc: int) -> int:
    """Drift sign ``s_d`` for a TPC index: +1 for TPC 1, -1 for TPC 0."""
    return +1 if int(tpc) == 1 else -1


# --------------------------------------------------------------------------------------------
# Plane geometry
# --------------------------------------------------------------------------------------------
def wire_normal(plane: str) -> np.ndarray:
    """Unit vector across the wires of ``plane``, pointing along increasing wire index.

    ``w_P = (0, -sin(phi_P), cos(phi_P))``  (Eq. 1)
    """
    phi = np.deg2rad(PHI_DEG[plane])
    return np.array([0.0, -np.sin(phi), np.cos(phi)])


def wire_direction(plane: str) -> np.ndarray:
    """Unit vector along the wires of ``plane`` (in the readout plane, orthogonal to the normal)."""
    phi = np.deg2rad(PHI_DEG[plane])
    return np.array([0.0, np.cos(phi), np.sin(phi)])


def kappa(plane: str) -> float:
    """Plane conversion factor ``kappa_P = v_drift * t_tick / pitch_P`` (wires per tick per unit slope)."""
    return TICK_CM / PITCH_CM[plane]


# --------------------------------------------------------------------------------------------
# Forward map: 3D -> 2D
# --------------------------------------------------------------------------------------------
def project(u: np.ndarray, plane: str, s_d: float = +1.0, normalize: bool = True) -> np.ndarray:
    """Project 3D direction(s) into the (tick, wire) image of ``plane``.

    ``d_2D,P = [ s_d u_x / (v_drift t_tick),  (u . w_P) / pitch_P ]``  (Eq. 5)

    Parameters
    ----------
    u : array (..., 3)
        3D direction(s). Need not be unit length.
    plane : {"U", "V", "Z"}
    s_d : float or array (...)
        Drift sign(s), +1 for TPC 1 and -1 for TPC 0.
    normalize : bool
        If True return unit image vectors; otherwise return d(ticks)/ds and d(wires)/ds per cm.

    Returns
    -------
    array (..., 2) ordered (tick, wire).
    """
    u = np.asarray(u, dtype=float)
    s_d = np.asarray(s_d, dtype=float)
    u_hat = u / np.linalg.norm(u, axis=-1, keepdims=True)
    d_tick = s_d * u_hat[..., 0] / TICK_CM
    d_wire = (u_hat @ wire_normal(plane)) / PITCH_CM[plane]
    q = np.stack([d_tick, d_wire], axis=-1)
    if normalize:
        q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return q


def slope(u: np.ndarray, plane: str, s_d: float = +1.0) -> np.ndarray:
    """Projected slope d(wires)/d(ticks) of a 3D direction in ``plane``  (Eq. 6).

    The slope is invariant under u -> -u, so it carries the track *axis* but not its head-tail
    orientation; the signed 2D vector from :func:`project` is needed for that.
    """
    q = project(u, plane, s_d, normalize=False)
    return q[..., 1] / q[..., 0]


def normalized_slope(q: np.ndarray, plane: str) -> np.ndarray:
    """Plane-independent slope ``a_P = (1/kappa_P) * d_wire / d_tick`` from an image vector  (Eq. 8).

    For a physical direction ``a_P = (u . w_P) / (s_d u_x)`` -- it depends only on the 3D direction
    and the wire angle. Returns NaN when the tick component is zero.
    """
    q = np.asarray(q, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = q[..., 1] / q[..., 0] / kappa(plane)
    return np.where(q[..., 0] == 0.0, np.nan, a)


def consistency_residual(a_U: np.ndarray, a_V: np.ndarray, a_Z: np.ndarray) -> np.ndarray:
    """Residual of the U/V/Z consistency condition  ``a_U + a_V - 2 a_Z cos(phi)``  (Eq. 10).

    Exactly zero (up to rounding) when the three projections come from one 3D line.
    Uses the symmetric ICEBERG/DUNE geometry phi_U = -phi_V = phi, phi_Z = 0.
    """
    cos_phi = np.cos(np.deg2rad(PHI_DEG["U"]))
    return np.asarray(a_U) + np.asarray(a_V) - 2.0 * np.asarray(a_Z) * cos_phi


# --------------------------------------------------------------------------------------------
# Inverse map: two 2D views -> 3D
# --------------------------------------------------------------------------------------------
def reconstruct_from_pair(
    q_P: np.ndarray, q_Q: np.ndarray, P: str, Q: str, s_d: float = +1.0
) -> np.ndarray:
    """3D unit direction from the image directions of two non-parallel planes  (Eqs. 11-15).

    Each plane gives one linear constraint ``a_P = n_Py b_y + n_Pz b_z`` on
    ``b_y = u_y / (s_d u_x)`` and ``b_z = u_z / (s_d u_x)``. Solving the 2x2 system fixes the 3D
    line ``(1, s_d b_y, s_d b_z)``; the orientation ``eta = +/-1`` is the one whose forward
    projections best agree with the *signed* inputs ``q_P`` and ``q_Q``.

    Returns an array (3,) -- or (..., 3) for batched inputs.
    """
    n_P, n_Q = wire_normal(P), wire_normal(Q)
    a_P, a_Q = normalized_slope(q_P, P), normalized_slope(q_Q, Q)
    det = n_P[1] * n_Q[2] - n_Q[1] * n_P[2]
    b_y = (n_Q[2] * a_P - n_P[2] * a_Q) / det
    b_z = (n_P[1] * a_Q - n_Q[1] * a_P) / det
    s_d = np.asarray(s_d, dtype=float)
    u = np.stack([np.ones_like(b_y), s_d * b_y, s_d * b_z], axis=-1)
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)

    # choose eta: maximise agreement with the two signed input directions
    qP_hat = np.asarray(q_P, float) / np.linalg.norm(q_P, axis=-1, keepdims=True)
    qQ_hat = np.asarray(q_Q, float) / np.linalg.norm(q_Q, axis=-1, keepdims=True)
    score = np.sum(qP_hat * project(u, P, s_d), axis=-1) + np.sum(qQ_hat * project(u, Q, s_d), axis=-1)
    eta = np.where(score < 0.0, -1.0, 1.0)
    return u * eta[..., None]


def heldout_score(q: Mapping[str, np.ndarray], P: str, Q: str, R: str, s_d: float = +1.0) -> float:
    """Round-robin consistency score ``C_{PQ->R}``  (Eq. 16).

    Reconstruct 3D from planes P and Q, project into the held-out plane R, and take the cosine
    with R's own prediction: +1 agree, -1 reversed, ~0 uninformative.
    """
    u = reconstruct_from_pair(q[P], q[Q], P, Q, s_d)
    q_R = np.asarray(q[R], float)
    return float(np.clip(np.dot(q_R / np.linalg.norm(q_R), project(u, R, s_d)), -1.0, 1.0))


# --------------------------------------------------------------------------------------------
# Voting system
# --------------------------------------------------------------------------------------------
@dataclass
class VoteResult:
    """Output of :func:`vote` for one event."""

    q: Dict[str, np.ndarray]
    """Corrected (tick, wire) direction per plane."""
    flipped_tick: Dict[str, bool] = field(default_factory=dict)
    """Planes reversed by stage 1 (tick-sign majority)."""
    flipped_geom: Dict[str, bool] = field(default_factory=dict)
    """Planes reversed by stage 2 (round-robin consistency)."""
    dropped: Dict[str, bool] = field(default_factory=dict)
    """Planes flagged as uninformative by stage 2."""
    scores: Dict[str, float] = field(default_factory=dict)
    """Held-out consistency score C for each plane (after stage 1)."""
    u3d: Optional[np.ndarray] = None
    """Final 3D direction (drop-aware), or None if two or more planes were dropped."""


def tick_sign_vote(q: Mapping[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, bool]]:
    """Voting stage 1: the tick axis is shared by all planes, so its sign must agree.

    A non-zero tick component that disagrees with the majority is reversed (both components
    multiplied by -1). Ties go to the positive sign; a zero tick component does not vote.
    """
    signs = {p: float(np.sign(q[p][0])) for p in PLANES}
    n_pos = sum(s > 0 for s in signs.values())
    n_neg = sum(s < 0 for s in signs.values())
    majority = +1.0 if n_pos >= n_neg else -1.0
    out, flipped = {}, {}
    for p in PLANES:
        flip = signs[p] != 0.0 and signs[p] != majority
        out[p] = -np.asarray(q[p], float) if flip else np.asarray(q[p], float).copy()
        flipped[p] = bool(flip)
    return out, flipped


def geometric_vote(
    q: Mapping[str, np.ndarray],
    s_d: float = +1.0,
    flip_threshold: float = 0.90,
    drop_threshold: float = 0.25,
) -> Tuple[Dict[str, np.ndarray], Dict[str, bool], Dict[str, bool], Dict[str, float]]:
    """Voting stage 2: round-robin held-out consistency.

    Compute ``C_{UV->Z}``, ``C_{UZ->V}``, ``C_{VZ->U}``; act on the most decisive one (largest |C|):
    reverse that plane if ``C < -flip_threshold``, drop it if ``|C| < drop_threshold``.
    """
    scores = {R: heldout_score(q, P, Q, R, s_d) for P, Q, R in PAIRS}
    order = ("Z", "V", "U")  # same tie-breaking order as main.headtail.consistency
    decisive = max(order, key=lambda p: abs(scores[p]))
    C = scores[decisive]
    out = {p: np.asarray(q[p], float).copy() for p in PLANES}
    flipped = {p: False for p in PLANES}
    dropped = {p: False for p in PLANES}
    if C < -flip_threshold:
        out[decisive] = -out[decisive]
        flipped[decisive] = True
    elif abs(C) < drop_threshold:
        dropped[decisive] = True
    return out, flipped, dropped, scores


def reconstruct_3d(
    q: Mapping[str, np.ndarray], s_d: float = +1.0, dropped: Optional[Mapping[str, bool]] = None
) -> Optional[np.ndarray]:
    """Combine the pairwise 3D candidates into one direction  (Eq. 17).

    With no dropped plane, the three pairwise unit directions are summed and normalised. With one
    dropped plane only the remaining pair is used. With two or more dropped planes, returns None.
    """
    dropped = dropped or {p: False for p in PLANES}
    kept = [p for p in PLANES if not dropped.get(p, False)]
    if len(kept) < 2:
        return None
    pairs = [(P, Q) for P, Q, _ in PAIRS if P in kept and Q in kept]
    total = sum(reconstruct_from_pair(q[P], q[Q], P, Q, s_d) for P, Q in pairs)
    n = np.linalg.norm(total)
    return None if n == 0.0 else total / n


def vote(
    q: Mapping[str, np.ndarray],
    s_d: float = +1.0,
    flip_threshold: float = 0.90,
    drop_threshold: float = 0.25,
) -> VoteResult:
    """Full voting system for one event: tick-sign vote, round-robin vote, drop-aware 3D direction."""
    q1, flipped_tick = tick_sign_vote(q)
    q2, flipped_geom, dropped, scores = geometric_vote(q1, s_d, flip_threshold, drop_threshold)
    return VoteResult(
        q=q2,
        flipped_tick=flipped_tick,
        flipped_geom=flipped_geom,
        dropped=dropped,
        scores=scores,
        u3d=reconstruct_3d(q2, s_d, dropped),
    )


def angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle between two (batches of) vectors, in degrees."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    c = np.sum(a * b, axis=-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))
