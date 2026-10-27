"""Draw the geometry diagrams used in docs/coordinate_transforms.md.

    python tools/make_geometry_figures.py

Everything is computed with larpoint.geometry.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from larpoint import geometry as g  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "assets" / "geometry"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#1b1f24"
MUTED = "#57606a"
COLORS = {"U": "#0969da", "V": "#bc4c00", "Z": "#1a7f37"}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
    }
)


def arrow(ax, start, end, color, lw=2.6, ms=18, ls="-", z=5):
    ax.add_patch(
        FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color, linestyle=ls, zorder=z)
    )


# ------------------------------------------------------------------------------------------
# 1. Wire orientations in the readout (yz) plane
# ------------------------------------------------------------------------------------------
def wire_planes():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.9))
    for ax, p in zip(axes, ("U", "V", "Z")):
        c = COLORS[p]
        phi = g.PHI_DEG[p]
        w = g.wire_normal(p)      # (x, y, z)
        l = g.wire_direction(p)
        # plot axes: horizontal = z, vertical = y
        wz, wy = w[2], w[1]
        lz, ly = l[2], l[1]
        for s in np.linspace(-1.6, 1.6, 15):
            p0 = np.array([s * wz, s * wy])
            a, b = p0 - 3 * np.array([lz, ly]), p0 + 3 * np.array([lz, ly])
            ax.plot([a[0], b[0]], [a[1], b[1]], color=c, alpha=0.28, lw=1.3)
        arrow(ax, (0, 0), (1.05 * wz, 1.05 * wy), c, lw=3)
        ax.text(1.18 * wz + (0.02 if p != "Z" else 0), 1.18 * wy - 0.05, r"$\hat{\mathbf{w}}_%s$" % p,
                color=c, fontsize=17, ha="center", va="center")
        arrow(ax, (0, 0), (0.75 * lz, 0.75 * ly), MUTED, lw=1.8, ms=13)
        ax.text(0.86 * lz + 0.13, 0.86 * ly, r"$\hat{\boldsymbol{\ell}}_%s$" % p, color=MUTED, fontsize=15,
                ha="center", va="center")
        if phi != 0:
            th1, th2 = (np.degrees(np.arctan2(wy, wz)), 0.0) if phi > 0 else (0.0, np.degrees(np.arctan2(wy, wz)))
            ax.add_patch(Arc((0, 0), 0.9, 0.9, theta1=th1, theta2=th2, color=INK, lw=1.4))
            mid = np.radians((th1 + th2) / 2)
            ax.text(0.62 * np.cos(mid), 0.62 * np.sin(mid), r"$\phi_%s$" % p, fontsize=14, ha="center",
                    va="center", color=INK)
            ax.plot([0, 1.25], [0, 0], color=MUTED, lw=0.8, ls=":")
        ax.text(0.5, 1.02, f"{p} plane", transform=ax.transAxes, ha="center", va="bottom", fontsize=16,
                fontweight="bold", color=c)
        ax.text(0.5, -0.04, r"$\phi_%s = %+.2f^\circ$" % (p, phi) if phi else r"$\phi_Z = 0^\circ$ (collection)",
                transform=ax.transAxes, ha="center", va="top", fontsize=13, color=INK, clip_on=False)
        ax.set_xlim(-1.35, 1.6)
        ax.set_ylim(-1.35, 1.35)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#d0d7de")
        # tiny axis key
        ax.annotate("", xy=(-1.05, -1.18), xytext=(-1.25, -1.18), arrowprops=dict(arrowstyle="-|>", color=MUTED))
        ax.annotate("", xy=(-1.25, -0.98), xytext=(-1.25, -1.18), arrowprops=dict(arrowstyle="-|>", color=MUTED))
        ax.text(-1.0, -1.18, "z", color=MUTED, va="center", fontsize=11)
        ax.text(-1.25, -0.92, "y", color=MUTED, ha="center", fontsize=11)
    fig.suptitle("Readout planes seen from the drift (x) axis: wires (faint), wire normal $\\hat{\\mathbf{w}}_P$, wire direction $\\hat{\\boldsymbol{\\ell}}_P$",
                 fontsize=13.5, color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(OUT / "wire_planes.png", dpi=170, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------------------------------
# 2. One 3D direction, three 2D views, and the head-tail ambiguity
# ------------------------------------------------------------------------------------------
def projection():
    u = np.array([0.42, 0.55, 0.72])
    u /= np.linalg.norm(u)
    s_d = +1

    fig = plt.figure(figsize=(15, 4.7))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.35, 1, 1, 1], wspace=0.28, left=0.02, right=0.985, top=0.80,
                          bottom=0.08)

    # --- 3D sketch
    ax = fig.add_subplot(gs[0], projection="3d")
    ax.set_proj_type("ortho")
    # APA at x = 0 (yz plane), drift volume x in [0, 1]
    yy, zz = np.meshgrid([-1, 1], [-1, 1])
    ax.plot_surface(np.zeros_like(yy), yy, zz, color="#8c959f", alpha=0.18, shade=False)
    for x in (0, 1):
        ax.plot([x, x, x, x, x], [-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], color="#8c959f", lw=1)
    for y, z in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        ax.plot([0, 1], [y, y], [z, z], color="#d0d7de", lw=1)
    r0 = np.array([0.55, -0.3, -0.35])
    r1 = r0 + 0.9 * u
    ax.quiver(*r0, *(r1 - r0), color="#cf222e", lw=3, arrow_length_ratio=0.18)
    ax.scatter(*r0, color=INK, s=28)
    ax.text(*(r0 + np.array([0, -0.05, -0.22])), "vertex", color=INK, fontsize=11)
    ax.text(*(r1 + np.array([0.02, 0.02, 0.06])), r"$\hat{\mathbf{u}}$", color="#cf222e", fontsize=18)
    ax.quiver(1.1, 1.15, 1.0, -0.6, 0, 0, color=MUTED, lw=1.8, arrow_length_ratio=0.25)
    ax.text(0.85, 1.2, 1.15, "drift", color=MUTED, fontsize=11)
    ax.text(0.0, -1.0, 1.15, "wire planes\n(anode)", color=MUTED, fontsize=10.5)
    ax.set_xlabel("x (drift)", labelpad=-8)
    ax.set_ylabel("y", labelpad=-8)
    ax.set_zlabel("z", labelpad=-8)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=18, azim=-58)
    ax.set_box_aspect((1, 1.1, 1.1))
    ax.set_title("3D initial electron direction", fontsize=14, fontweight="bold", color=INK, pad=2)

    # --- 2D views (image convention: tick horizontal, wire vertical)
    L = 34.0
    for k, p in enumerate(("U", "V", "Z")):
        a2 = fig.add_subplot(gs[k + 1])
        q = g.project(u, p, s_d) * L
        c = COLORS[p]
        t0, w0 = 0.0, 0.0
        a2.plot([t0 - 1.5 * q[0], t0 + 1.5 * q[0]], [w0 - 1.5 * q[1], w0 + 1.5 * q[1]], color="#d0d7de", lw=6,
                solid_capstyle="round", zorder=1)
        arrow(a2, (t0, w0), (t0 + q[0], w0 + q[1]), c, lw=3, ms=20)
        arrow(a2, (t0, w0), (t0 - q[0], w0 - q[1]), c, lw=2, ms=16, ls=(0, (3, 2)))
        a2.scatter([t0], [w0], color=INK, s=24, zorder=6)
        a2.text(t0 + q[0] * 1.05, w0 + q[1] * 1.05 + 4, "head", color=c, ha="center", fontsize=11)
        a2.text(t0 - q[0] * 1.05, w0 - q[1] * 1.05 - 6, "reversed", color=c, ha="center", fontsize=11, alpha=0.85)
        m = g.slope(u, p, s_d)
        a2.text(0.03, 0.04, f"slope = {m:+.3f} wire/tick\n(same for both arrows)", transform=a2.transAxes,
                fontsize=10, color=MUTED, va="bottom")
        a2.set_xlim(-52, 52); a2.set_ylim(-52, 52)
        a2.set_aspect("equal")
        a2.set_xlabel("time tick")
        if k == 0:
            a2.set_ylabel("wire number")
        a2.set_xticks([]); a2.set_yticks([])
        a2.set_title(f"{p} view", fontsize=14, fontweight="bold", color=c)
    fig.text(0.665, 0.91, r"$\mathbf{d}_{2D,P} \propto \left(\, s_d\,u_x / v_{\rm drift} t_{\rm tick},\;\; "
             r"\hat{\mathbf{u}}\cdot\hat{\mathbf{w}}_P / \mathrm{pitch}_P \right)$", ha="center", fontsize=15,
             color=INK)
    fig.savefig(OUT / "projection_3d_to_2d.png", dpi=170, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------------------------------
# 3. The consistency condition in action
# ------------------------------------------------------------------------------------------
def consistency():
    rng = np.random.default_rng(0)
    u = rng.normal(size=(60000, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    u = u[np.abs(u[:, 0]) > 0.05][:20000]
    q = {p: g.project(u, p, +1) for p in g.PLANES}

    def resid(qd):
        a = {p: g.normalized_slope(qd[p], p) for p in g.PLANES}
        return g.consistency_residual(a["U"], a["V"], a["Z"])

    r_ok = resid(q)
    qb = {p: v.copy() for p, v in q.items()}
    ang = np.radians(rng.normal(0, 3.0, size=len(u)))
    c, s = np.cos(ang), np.sin(ang)
    qb["Z"] = np.stack([c * q["Z"][:, 0] - s * q["Z"][:, 1], s * q["Z"][:, 0] + c * q["Z"][:, 1]], axis=1)
    r_bad = resid(qb)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    floor = 1e-16
    bins = np.logspace(-16.2, 2, 90)
    ax.hist(np.maximum(np.abs(r_ok), floor), bins=bins, color=COLORS["Z"], alpha=0.85,
            label="three projections of one 3D direction")
    ax.hist(np.maximum(np.abs(r_bad), floor), bins=bins, color=COLORS["V"], alpha=0.75,
            label=r"Z view rotated by $\mathcal{N}(0, 3^\circ)$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$|\,a_U + a_V - 2\,a_Z\cos\phi\,|$   (values below $10^{-16}$, incl. exact zeros, in the first bin)")
    ax.set_ylabel("directions")
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Consistency residual: zero to machine precision for a real track", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "consistency_residual.png", dpi=170, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    wire_planes()
    projection()
    consistency()
    print("wrote", *sorted(p.name for p in OUT.glob("*.png")))
