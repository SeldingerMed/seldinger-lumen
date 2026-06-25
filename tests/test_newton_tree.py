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
        # push off-axis so the barrier must hold the wire against a branch wall
        for _ in range(150):
            sim.step(dt=2.5e-2, substeps=5, preload=(60.0, 0.0, 0.0))
        return sim

    on = run(True)
    off = run(False)
    assert np.isfinite(on.node_radii()).all()
    assert on.node_radii().max() <= R + d_hat + 0.1        # held within the branch lumen band
    assert off.node_radii().max() > 2.0 * R                # without contact it escapes
    # (branch IDENTITY isn't asserted: which edge the tip projects to is preload-sensitive
    # and, near a junction, subject to the documented min-r ownership ceiling — that's a
    # navigation/selection concern. The contact claim is the held-vs-escape pair above.)


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


def test_tree_rejects_unsupported_physics():
    # flow/clot + a sim-level lumen_field aren't wired for the edge graph -> fail loud
    # (deformable_wall IS supported now, L0d.1d).
    asset = procedural.bifurcation()
    tree = VascularTree(asset)
    trunk_pts = np.asarray(asset.edges[0].centerline_mm)
    with pytest.raises(NotImplementedError, match="flow/clot"):
        NewtonGuidewireSim(trunk_pts, 2.0, _device(), device="cpu", tree=tree,
                           clot_segment=(10.0, 20.0))


def test_tree_deformable_wall_deflects_under_load():
    # L0d.1d: the tree wall is a per-edge HGO wall (R0+w shared with the barrier). Drive
    # the relaxation directly with a known load (deterministic, no dependence on how hard
    # the dynamics press): it deflects, and a stiffer wall (larger C10) deflects less.
    from lumen.newton.hgo_wall import HGOParams
    asset = procedural.bifurcation(trunk=50.0, branch=50.0, radius=2.0, angle_deg=25.0)
    tree = VascularTree(asset)
    trunk_pts = np.asarray(asset.edges[0].centerline_mm)

    def deflect(C10):
        sim = NewtonGuidewireSim(trunk_pts, 2.0, _device(), device="cpu", tree=tree,
                                 deformable_wall=True,
                                 hgo_params=HGOParams(C10=C10, k1=C10 * 0.5, k2=1.0, thickness=0.3))
        wall = sim.solver._tree_wall
        wall.wall_load.assign(np.full(wall.wall_load.shape, 50.0, np.float32))  # uniform load
        wall.update_from_load()
        return wall.max_deflection()

    soft = deflect(1.5e3)
    stiff = deflect(2.0e4)
    assert soft > 1e-3                       # the per-edge HGO wall actually deforms
    assert soft > stiff                      # softer wall yields more (HGO monotone)


def test_deformable_tree_wall_uses_per_edge_arclength():
    # H1 fix: unequal-length edges get their OWN s_max for the HGO cell area (not one mean),
    # so the relaxation is correct per edge.
    from lumen.newton.hgo_wall import HGOParams
    asset = procedural.bifurcation(trunk=50.0, branch=30.0, radius=2.0)   # edges 50, 30, 30
    tree = VascularTree(asset)
    sim = NewtonGuidewireSim(np.asarray(asset.edges[0].centerline_mm), 2.0, _device(),
                             device="cpu", tree=tree, deformable_wall=True,
                             hgo_params=HGOParams(C10=4e3, k1=2e3, k2=1.0, thickness=0.3))
    w = sim.solver._tree_wall
    assert np.allclose(w.s_max, [50.0, 30.0, 30.0])         # per-edge arc-length, not the mean
    n = w.n_cells
    assert abs(w.cell_area[0] / w.cell_area[n] - 50.0 / 30.0) < 0.01   # ds scales with edge length


def test_deformable_wall_with_route_actuation():
    # GLM L3: the physically interesting combo — deformable wall + base actuation that
    # follows a route past a junction — runs stably and the wall deforms.
    from lumen.envs.tree_nav import _route_polyline
    from lumen.newton.hgo_wall import HGOParams
    asset = procedural.bifurcation(trunk=50.0, branch=50.0, radius=2.0, angle_deg=25.0)
    tree = VascularTree(asset)
    route = tree.route("left_out", "trunk_in")
    route_pts = _route_polyline(tree, route, "trunk_in")
    dev = np.stack([np.full(10, 0.3), np.zeros(10), np.linspace(2, 20, 10)], axis=1)
    sim = NewtonGuidewireSim(np.asarray(asset.edges[0].centerline_mm), 2.0, dev, device="cpu",
                             tree=tree, route_centerline=route_pts, deformable_wall=True,
                             vbd_iterations=10,
                             hgo_params=HGOParams(C10=3e3, k1=1.5e3, k2=1.0, thickness=0.3))
    for _ in range(30):
        sim.step(dt=2.5e-2, substeps=5, insertion=2.0, preload=(40.0, 0.0, 0.0))
    # this is a STABILITY check for the deformable-wall + route-actuation combo: assert it
    # ran to finite state (the deflection magnitude depends on how hard the dynamics press,
    # which the deterministic deflection test covers separately).
    assert np.isfinite(sim.body_positions()).all()
    assert np.isfinite(sim.wall_max_deflection())          # wall state is a valid number


def test_reset_clears_tree_wall_deformation():
    from lumen.newton.hgo_wall import HGOParams
    asset = procedural.bifurcation(trunk=50.0, branch=50.0, radius=2.0)
    tree = VascularTree(asset)
    sim = NewtonGuidewireSim(np.asarray(asset.edges[0].centerline_mm), 2.0, _device(),
                             device="cpu", tree=tree, deformable_wall=True,
                             hgo_params=HGOParams(C10=2e3, k1=1e3, k2=1.0, thickness=0.3))
    w = sim.solver._tree_wall
    w.wall_load.assign(np.full(w.wall_load.shape, 50.0, np.float32)); w.update_from_load()
    assert w.max_deflection() > 1e-3                        # deformed
    sim.reset()
    assert w.max_deflection() == 0.0                        # reset clears the wall (CodeRabbit #25)


def test_tree_deformable_wall_steps_stably():
    from lumen.newton.hgo_wall import HGOParams
    asset = procedural.bifurcation(trunk=50.0, branch=50.0, radius=2.0)
    tree = VascularTree(asset)
    trunk_pts = np.asarray(asset.edges[0].centerline_mm)
    sim = NewtonGuidewireSim(trunk_pts, 2.0, _device(), device="cpu", tree=tree,
                             deformable_wall=True, vbd_iterations=10,
                             hgo_params=HGOParams(C10=4e3, k1=2e3, k2=1.0, thickness=0.3))
    for _ in range(40):
        sim.step(dt=2.5e-2, substeps=5, preload=(80.0, 0.0, 0.0))
    assert np.isfinite(sim.node_radii()).all()
    assert sim.node_radii().max() <= 2.0 + 0.3 + 0.4       # held within the (deformed) band
