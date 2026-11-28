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


def test_coaxial_tree_contact_builds_and_steps():
    from lumen.envs.tree_nav import _route_polyline
    asset = procedural.bifurcation(trunk=50.0, branch=50.0, radius=2.0, angle_deg=25.0)
    tree = VascularTree(asset)
    route_pts = _route_polyline(tree, tree.route("left_out", "trunk_in"), "trunk_in")
    trunk_pts = np.asarray(asset.edges[0].centerline_mm)
    dev = np.stack([np.full(10, 0.2), np.zeros(10), np.linspace(4.0, 22.0, 10)], axis=1)
    cath = np.stack([np.zeros(10), np.zeros(10), np.linspace(2.0, 20.0, 10)], axis=1)
    sim = NewtonGuidewireSim(trunk_pts, 2.0, dev, catheter_points=cath,
                             catheter_radius=0.65, catheter_inner_radius=0.5,
                             tree=tree, route_centerline=route_pts,
                             vbd_iterations=8, device="cpu")
    for _ in range(8):
        sim.step(dt=2.5e-2, substeps=5, insertion=1.0, insertion_cath=0.5)
    assert np.isfinite(sim.body_positions()).all()
    assert np.isfinite(sim.catheter_positions()).all()
    assert sim.node_radii().max() <= 2.0 + 0.3 + 0.2
    assert sim.catheter_node_radii().max() <= 2.0 + 0.3 + 0.2


def test_batched_tree_contact_steps_independently_on_one_procedural_tree():
    asset = procedural.bifurcation()
    tree = VascularTree(asset)
    trunk_pts = np.asarray(asset.edges[0].centerline_mm)
    sim = NewtonGuidewireSim(trunk_pts, 2.0, _device(), n_envs=2, device="cpu", tree=tree,
                             vbd_iterations=8)

    sim.step(dt=2.5e-2, substeps=5, insertion=[0.0, 1.0],
             preload=([20.0, 0.0, 0.0]))
    pos = sim.env_positions()

    assert pos.shape == (2, sim.n_per_env, 3)
    assert np.isfinite(pos).all()
    assert sim.node_radii().reshape(2, sim.n_per_env).max(axis=1).max() <= 2.0 + 0.3 + 0.2
    assert not np.allclose(pos[0], pos[1])       # per-env actuation/contact state is independent


def test_batched_tree_flow_uses_edge_graph_fields():
    # #55: Flow on a vascular tree must be keyed by env×edge blocks, not by a
    # single route-centered centerline. Two envs share one procedural tree/contact
    # graph while solving independent per-edge flow fields and local node drag.
    from lumen.newton.flow import FlowField

    asset = procedural.bifurcation()
    tree = VascularTree(asset)
    trunk_pts = np.asarray(asset.edges[0].centerline_mm)
    sim = NewtonGuidewireSim(trunk_pts, 2.0, _device(), n_envs=2, device="cpu",
                             tree=tree, flow=FlowField(), vbd_iterations=8)

    sim.step(dt=2.5e-2, substeps=3, insertion=[0.0, 1.0])

    assert np.isfinite(sim.env_positions()).all()
    assert sim.flow.tree_pressure_fields().shape[:2] == (2, len(tree.edges))
    assert sim.flow.tree_velocity_fields().shape[:2] == (2, len(tree.edges))
    assert np.isfinite(sim.flow.tree_downstream_Q()).all()


def test_tree_clot_remains_guarded_with_edge_graph_message():
    # #55: tree flow is edge-aware; clot remains intentionally blocked until callers
    # can specify per-edge clot segments instead of one ambiguous route-centered range.
    asset = procedural.bifurcation()
    tree = VascularTree(asset)
    trunk_pts = np.asarray(asset.edges[0].centerline_mm)
    with pytest.raises(NotImplementedError, match="edge-aware tree clot"):
        NewtonGuidewireSim(trunk_pts, 2.0, _device(), device="cpu", tree=tree,
                           clot_segment=(10.0, 20.0))


def test_tree_rejects_sim_level_lumen_field():
    # A tree takes per-edge R0 from its asset; a sim-level linear lumen_field would
    # be ambiguous across branches. FlowField is covered by the edge-graph test.
    from lumen.core import LumenField

    asset = procedural.bifurcation()
    tree = VascularTree(asset)
    trunk_pts = np.asarray(asset.edges[0].centerline_mm)
    with pytest.raises(NotImplementedError, match="sim-level lumen_field"):
        NewtonGuidewireSim(trunk_pts, 2.0, _device(), device="cpu", tree=tree,
                           lumen_field=LumenField.cylinder(50.0, 2.0))


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



def _tapered_tree(n_s, n_th, R0=2.0, R1=4.0, L=80.0):
    """One straight 80 mm vessel whose radius widens linearly from R0 to R1, its wall baked
    into the tree-contact grid at n_s x n_th samples."""
    from lumen.assets.schema import Asset, DeviceSpawn, Frame, Node
    from lumen.core.lumen_field import LumenField
    from lumen.newton.tube_vbd import TubeVBDSolver  # noqa: F401  (needs warp + newton)

    pts = _straight_vessel(M=81, L=L)
    lf = LumenField(np.array([0.0, L]), np.array([0.0]), np.array([[R0], [R1]]))
    edge = procedural._edge_from_polyline("e0", "in", "out", pts, lf)
    asset = Asset(frame=Frame(), nodes=[Node("in", tuple(pts[0])), Node("out", tuple(pts[-1]))],
                  edges=[edge], device_spawn=DeviceSpawn(node_id="in"))
    return VascularTree(asset)


def _wall_force(tree, p, n_s=None, n_th=None, qd=None, mu=0.0):
    """The tree barrier's force on one node of radius 0.2 at p (kappa 2e3, d_hat 0.3), with
    Newton spatial velocity ``qd`` (linear, angular) and friction ``mu`` both ways."""
    import warp as wp

    from lumen.newton.tube_barrier_kernel import accumulate_tree_barrier

    sim = NewtonGuidewireSim(_straight_vessel(), 2.0, _device(), radius=0.2, kappa=2e3,
                             d_hat=0.3, device="cpu", tree=tree)
    s = sim.solver
    if n_s is not None:
        s.set_tree_contact(tree, sim._contact_bodies, kappa=2e3, d_hat=0.3, n_s=n_s, n_th=n_th,
                           body_radii=[0.2] * len(sim._contact_bodies))
    bid = sim._contact_bodies[3]
    q = sim.state_0.body_q.numpy()
    q[bid, :3] = p
    bq = wp.array(q, dtype=wp.transform, device="cpu")
    bf = wp.zeros(len(q), dtype=wp.vec3, device="cpu")
    bh = wp.zeros(len(q), dtype=wp.mat33, device="cpu")
    load = wp.zeros_like(s._tree_wall_load)
    cg = wp.array(np.array([bid], np.int32), dtype=wp.int32, device="cpu")
    v = sim.state_0.body_qd.numpy() * 0.0
    if qd is not None:
        v[bid] = qd
    bqd = wp.array(v, dtype=wp.spatial_vector, device="cpu")
    wp.launch(accumulate_tree_barrier, dim=1,
              inputs=[cg, s._tree_wire_mask, s._tree_body_radius, bq, bqd,
                      s._tree_P, s._tree_Tg, s._tree_M1, s._tree_cum_s, s._tree_vstart,
                      s._tree_vcount, s._tree_smax, s._tree_start_junc, s._tree_end_junc,
                      s._tree_n_edges, s._tree_R0, s._tree_w, s._tree_ns, s._tree_nth,
                      s._tree_n_per_env, s._tree_kappa, s._tree_d_hat, s._tree_mode,
                      float(mu), float(mu), 0.0, 5e-3],
              outputs=[bf, bh, load], device="cpu")
    return bf.numpy()[bid]


def test_tree_contact_friction_resists_sliding_not_spinning():
    """Newton's body_qd is (linear, angular). On a scanned case's tree contact a node pressed
    into the wall and sliding along the vessel feels friction against the slide; spinning in
    place it feels none along the vessel (the kernel used to read the spin as the slide)."""
    tree = _tapered_tree(41, 16, R0=2.0, R1=2.0)
    p = np.array([1.7, 0.0, 40.0])                       # 0.1 mm into the barrier band
    normal = abs(_wall_force(tree, p, qd=(0, 0, 0, 0, 0, 0), mu=0.5)[0])
    assert normal > 50.0
    slide = _wall_force(tree, p, qd=(0, 0, 1.0, 0, 0, 0), mu=0.5)
    spin = _wall_force(tree, p, qd=(0, 0, 0, 0, 0, 1.0), mu=0.5)
    assert slide[2] < -0.45 * normal                     # about mu * normal, against +z
    assert abs(spin[2]) < 1e-3 * normal


def test_a_node_past_a_junction_end_in_the_wall_is_pushed_back():
    """A node behind the start of a wider vessel at a junction, 0.24 mm outside both vessels'
    walls, is owned by that vessel by its distance to the end point. The force is measured by
    the same distance and pushes the node back toward the junction. Measured across the
    owner's axis, it had 2.1 mm of room and felt no force at all."""
    from test_tree import _elbow_tree

    tree = _elbow_tree()
    q = np.array([-2.8, 0.0, -0.5])
    f = _wall_force(tree, q)
    back = -q / np.linalg.norm(q)
    assert float(np.dot(f, back)) > 100.0
    assert np.linalg.norm(f - np.dot(f, back) * back) < 1e-3 * np.linalg.norm(f)
    # inside the junction, more than d_hat from the wall: no force
    assert np.linalg.norm(_wall_force(tree, np.array([-1.5, 0.0, -0.5]))) < 1e-9


@pytest.mark.parametrize("n_s", [None, 5])
def test_the_contact_kernel_feels_the_wall_between_its_samples(n_s):
    """On a vessel that widens from 2 to 4 mm the wall at s = 30 mm is 2.75 mm. A 0.2 mm node
    2.35 mm off the axis is 0.2 mm from that wall, inside the 0.3 mm barrier, and is pushed
    back with kappa * (0.3 - 0.2). The kernel read the wall at the nearest of 40 samples
    (one per 2 mm here, 10 mm on a 400 mm aorta); with samples 20 mm apart (n_s = 5) the
    nearest one says 3.0 mm and the node would feel nothing."""
    tree = _tapered_tree(n_s, 1)
    assert tree.blended_R(0, 30.0, 0.0) == pytest.approx(2.75)
    f = _wall_force(tree, np.array([2.35, 0.0, 30.0]), n_s=n_s, n_th=16 if n_s else None)
    assert f[0] == pytest.approx(-2e3 * (0.3 - 0.2), rel=0.02)
    assert abs(f[1]) < 1e-3 and abs(f[2]) < 1e-3


def test_the_wall_is_sampled_finely_along_a_long_vessel():
    """By default the baked wall has a sample at least every 2 mm along the longest vessel,
    and the lumen fields' own angles."""
    from lumen import cases

    tree = VascularTree(cases.load("aorta-cta")[0])
    longest = max(e.frame.length for e in tree.edges)
    sim = NewtonGuidewireSim(np.asarray(tree.edges[0].frame.points), 2.0,
                             np.asarray(tree.edges[0].frame.points[:8]), radius=0.2,
                             device="cpu", tree=tree)
    assert longest / (sim.solver._tree_ns - 1) <= 2.0
    assert sim.solver._tree_nth == max(len(e.lf.theta) for e in tree.edges) == 32
