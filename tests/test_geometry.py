"""Tests for the wire-plane geometry (larpoint.geometry).

Run with:  pytest -q
"""
import numpy as np
import pytest

from larpoint import geometry as g


def random_directions(n, seed=0, min_abs_ux=0.05):
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(4 * n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    u = u[np.abs(u[:, 0]) > min_abs_ux]  # slopes diverge as the drift component -> 0
    return u[:n]


def views_of(u, s_d):
    return {p: g.project(u, p, s_d) for p in g.PLANES}


# ---------------------------------------------------------------------------- plane geometry
def test_wire_frame_is_orthonormal():
    x = np.array([1.0, 0.0, 0.0])
    for p in g.PLANES:
        w, l = g.wire_normal(p), g.wire_direction(p)
        assert np.isclose(np.linalg.norm(w), 1.0)
        assert np.isclose(np.dot(w, l), 0.0)
        assert np.isclose(np.dot(w, x), 0.0)  # readout plane is normal to the drift axis


def test_collection_plane_measures_z():
    assert np.allclose(g.wire_normal("Z"), [0.0, 0.0, 1.0])


def test_tick_length():
    assert np.isclose(g.TICK_CM, 0.08)


# ---------------------------------------------------------------------------- forward map
@pytest.mark.parametrize("s_d", [+1, -1])
def test_projection_reverses_with_direction(s_d):
    u = random_directions(200)
    for p in g.PLANES:
        assert np.allclose(g.project(-u, p, s_d), -g.project(u, p, s_d))


@pytest.mark.parametrize("s_d", [+1, -1])
def test_slope_cannot_see_head_tail(s_d):
    u = random_directions(200)
    for p in g.PLANES:
        assert np.allclose(g.slope(u, p, s_d), g.slope(-u, p, s_d))


def test_drift_sign_flips_only_tick_component():
    u = random_directions(50)
    for p in g.PLANES:
        a = g.project(u, p, +1, normalize=False)
        b = g.project(u, p, -1, normalize=False)
        assert np.allclose(a[:, 0], -b[:, 0])
        assert np.allclose(a[:, 1], b[:, 1])


def test_normalized_slope_matches_closed_form():
    u = random_directions(300, seed=1)
    for s_d in (+1, -1):
        for p in g.PLANES:
            a = g.normalized_slope(g.project(u, p, s_d), p)
            expected = (u @ g.wire_normal(p)) / (s_d * u[:, 0])
            assert np.allclose(a, expected)


# ---------------------------------------------------------------------------- consistency condition
@pytest.mark.parametrize("s_d", [+1, -1])
def test_consistency_condition_holds_for_real_tracks(s_d):
    u = random_directions(1000, seed=2)
    q = views_of(u, s_d)
    a = {p: g.normalized_slope(q[p], p) for p in g.PLANES}
    assert np.allclose(g.consistency_residual(a["U"], a["V"], a["Z"]), 0.0, atol=1e-9)


def test_consistency_condition_detects_perturbed_view():
    u = random_directions(1000, seed=3)
    q = views_of(u, +1)
    rot = np.deg2rad(3.0)
    R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    q["Z"] = q["Z"] @ R.T
    a = {p: g.normalized_slope(q[p], p) for p in g.PLANES}
    assert np.median(np.abs(g.consistency_residual(a["U"], a["V"], a["Z"]))) > 1e-3


# ---------------------------------------------------------------------------- inverse map
@pytest.mark.parametrize("s_d", [+1, -1])
@pytest.mark.parametrize("pair", [("U", "V"), ("U", "Z"), ("V", "Z")])
def test_pair_reconstruction_round_trip(s_d, pair):
    u = random_directions(1000, seed=4)
    P, Q = pair
    q = views_of(u, s_d)
    u_rec = g.reconstruct_from_pair(q[P], q[Q], P, Q, s_d)
    assert np.max(g.angle_deg(u_rec, u)) < 1e-4


@pytest.mark.parametrize("s_d", [+1, -1])
def test_three_view_reconstruction_round_trip(s_d):
    for u in random_directions(200, seed=5):
        q = views_of(u, s_d)
        assert g.angle_deg(g.reconstruct_3d(q, s_d), u) < 1e-4


def test_heldout_score_is_plus_one_for_consistent_views():
    for u in random_directions(100, seed=6):
        q = views_of(u, +1)
        for P, Q, R in g.PAIRS:
            assert np.isclose(g.heldout_score(q, P, Q, R, +1), 1.0)


def test_heldout_score_is_minus_one_for_reversed_view():
    for u in random_directions(100, seed=7):
        q = views_of(u, +1)
        q["Z"] = -q["Z"]
        assert np.isclose(g.heldout_score(q, "U", "V", "Z", +1), -1.0)


# ---------------------------------------------------------------------------- voting
@pytest.mark.parametrize("bad", ["U", "V", "Z"])
@pytest.mark.parametrize("s_d", [+1, -1])
def test_vote_repairs_one_reversed_view(bad, s_d):
    for u in random_directions(100, seed=8):
        q = views_of(u, s_d)
        q[bad] = -q[bad]
        res = g.vote(q, s_d)
        assert res.flipped_tick[bad]
        for p in g.PLANES:
            assert np.allclose(res.q[p], g.project(u, p, s_d))
        assert g.angle_deg(res.u3d, u) < 1e-4


def test_vote_leaves_consistent_event_untouched():
    for u in random_directions(100, seed=9):
        q = views_of(u, +1)
        res = g.vote(q, +1)
        assert not any(res.flipped_tick.values())
        assert not any(res.flipped_geom.values())
        assert not any(res.dropped.values())


def test_reconstruct_3d_is_drop_aware():
    u = random_directions(1, seed=10)[0]
    q = views_of(u, +1)
    q["U"] = np.array([0.3, -0.95])  # garbage in U
    only_vz = g.reconstruct_3d(q, +1, dropped={"U": True, "V": False, "Z": False})
    assert g.angle_deg(only_vz, u) < 1e-4
    assert g.reconstruct_3d(q, +1, dropped={"U": True, "V": True, "Z": False}) is None


# ---------------------------------------------------------------------------- agreement with the torch code
def test_matches_batched_torch_implementation():
    torch = pytest.importorskip("torch")
    tc = pytest.importorskip("larpoint.headtail.consistency", reason="Research PyTorch module is uploaded in a later stage")

    rng = np.random.default_rng(11)
    u = random_directions(600, seed=11)
    s_d = rng.choice([-1.0, 1.0], size=len(u))
    q = {p: g.project(u, p, s_d) for p in g.PLANES}
    # corrupt: random reversals, large rotations, and some pure-garbage views,
    # so that every branch of both voting stages is exercised
    for p in g.PLANES:
        flip = rng.random(len(u)) < 0.25
        q[p][flip] *= -1
        ang = np.deg2rad(rng.normal(0.0, 45.0, size=len(u)))
        c, s = np.cos(ang), np.sin(ang)
        q[p] = np.stack([c * q[p][:, 0] - s * q[p][:, 1], s * q[p][:, 0] + c * q[p][:, 1]], axis=1)
        bad = rng.random(len(u)) < 0.1
        r = rng.normal(size=(bad.sum(), 2))
        q[p][bad] = r / np.linalg.norm(r, axis=1, keepdims=True)

    ux = {p: torch.tensor(q[p][:, 0], dtype=torch.float64) for p in g.PLANES}
    uy = {p: torch.tensor(q[p][:, 1], dtype=torch.float64) for p in g.PLANES}
    cfg = type("Cfg", (), {"consistency_flip_threshold": 0.90, "consistency_drop_threshold": 0.25})()
    out = tc.apply_consistency_pipeline(ux, uy, torch.tensor(s_d, dtype=torch.float64), cfg)

    n_tick = n_geom = n_drop = 0
    for i in range(len(u)):
        res = g.vote({p: q[p][i] for p in g.PLANES}, s_d[i])
        for p in g.PLANES:
            torch_q = np.array([out["ux_corrected"][p][i].item(), out["uy_corrected"][p][i].item()])
            assert np.allclose(res.q[p], torch_q, atol=1e-6)
            assert res.flipped_tick[p] == bool(out["info_A"]["flipped_A"][p][i])
            assert res.flipped_geom[p] == bool(out["info_B"]["flipped_B"][p][i])
            assert res.dropped[p] == bool(out["info_B"]["dropped"][p][i])
            n_tick += res.flipped_tick[p]
            n_geom += res.flipped_geom[p]
            n_drop += res.dropped[p]
    assert n_tick > 0 and n_geom > 0 and n_drop > 0  # every branch was compared
