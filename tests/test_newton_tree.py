"""L0d.1b — vascular-tree contact in the Newton AVBD solve.

The safety anchor is PARITY: on a single-edge tree the new accumulate_tree_barrier
must match the battle-tested accumulate_tube_barrier (same math, one edge). Then a
branch-navigation test shows a wire is held in the lumen along a branch."""

import numpy as np
import pytest

pytest.importorskip("warp")
pytest.importorskip("newton")

from lumen.assets import procedural
from lumen.core import VascularTree
from lumen.newton.sim import NewtonGuidewireSim


def _straight_vessel(M=40, L=80.0):
    return np.stack([np.zeros(M), np.zeros(M), np.linspace(0, L, M)], axis=1)


def _device(n=11, x=1.0, z0=4.0):
    return np.stack([np.full(n, x), np.zeros(n), np.linspace(z0, z0 + 2.0 * (n - 1), n)], axis=1)


def test_single_edge_tree_matches_the_tube_kernel():
    # PARITY: a one-edge tree (straight_tube asset) vs the single-tube path, same scene,
    # rigid wall. The tree kernel adds an edge loop + junction flags but identical
    # contact math, so node radii must track closely.
    vessel, R = _straight_vessel(), 2.0
    dev = _device()
    common = dict(radius=0.2, stretch_stiffness=1e4, bend_stiffness=4e1,
                  kappa=2e3, d_hat=0.3, vbd_iterations=12, device="cpu")
    tube = NewtonGuidewireSim(vessel, R, dev, **common)
    tree = NewtonGuidewireSim(vessel, R, dev,
                              tree=VascularTree(procedural.straight_tube(80.0, R, n=40)), **common)
    for _ in range(120):
        tube.step(dt=2.5e-2, substeps=5, preload=(120.0, 0.0, 0.0))
        tree.step(dt=2.5e-2, substeps=5, preload=(120.0, 0.0, 0.0))
    r_tube, r_tree = tube.node_radii(), tree.node_radii()
    assert np.isfinite(r_tree).all()
    assert np.max(np.abs(r_tube - r_tree)) < 0.05          # parity with the proven kernel


def _branch_setup(angle_deg=30.0):
    asset = procedural.bifurcation(trunk=50.0, branch=50.0, radius=2.0, angle_deg=angle_deg)
    tree = VascularTree(asset)
    apex = np.asarray([n.position_mm for n in asset.nodes if n.id == "apex"][0], float)
    ld = np.array([-np.sin(np.radians(angle_deg)), 0.0, np.cos(np.radians(angle_deg))])
    # a short wire seated just past the apex, oriented up the LEFT branch
    n = 11
    dev = (apex + 2.0 * ld)[None, :] + np.arange(n)[:, None] * 2.0 * ld[None, :]
    return asset, tree, apex, ld, dev


def test_tree_contact_holds_a_wire_in_a_branch():
    asset, tree, apex, ld, dev = _branch_setup()
    trunk_pts = np.asarray(asset.edges[0].centerline_mm)
    R, d_hat = 2.0, 0.3

    def run(enable):
        sim = NewtonGuidewireSim(trunk_pts, R, dev, radius=0.2, kappa=2e3, d_hat=d_hat,
                                 vbd_iterations=12, device="cpu", tree=tree)
        if not enable:
            sim.solver._tree_enabled = False
        # push toward the left branch wall (off-axis preload) to test the barrier holds
        for _ in range(150):
            sim.step(dt=2.5e-2, substeps=5, preload=(80.0, 0.0, 0.0))
        return sim

    on = run(True)
    off = run(False)
    assert np.isfinite(on.node_radii()).all()
    assert on.node_radii().max() <= R + d_hat + 0.1        # held within the branch lumen band
    assert off.node_radii().max() > 2.0 * R                # without contact it escapes
    # the wire's nodes live in the left branch (not flung onto the trunk/right)
    tip_edge = tree.project(on.body_positions()[-1]).edge_id
    assert tip_edge in ("left", "right")


def test_tree_path_builds_and_steps():
    asset, tree, apex, ld, dev = _branch_setup()
    trunk_pts = np.asarray(asset.edges[0].centerline_mm)
    sim = NewtonGuidewireSim(trunk_pts, 2.0, dev, vbd_iterations=8, device="cpu", tree=tree)
    sim.step(dt=1.5e-2, substeps=3)
    assert np.isfinite(sim.body_positions()).all()


def test_batched_tree_rejected():
    asset = procedural.bifurcation()
    tree = VascularTree(asset)
    trunk_pts = np.asarray(asset.edges[0].centerline_mm)
    with pytest.raises(NotImplementedError, match="single-env"):
        NewtonGuidewireSim(trunk_pts, 2.0, _device(), n_envs=2, device="cpu", tree=tree)
