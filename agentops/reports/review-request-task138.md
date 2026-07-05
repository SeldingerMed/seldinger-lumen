You are the mandatory independent Seldinger engineering reviewer using openai-codex/gpt-5.5 / ChatGPT 5.5.

Return exactly one of:
REVIEW_RESULT: PASS
Required fixes: none
Notes: <optional>

or:
REVIEW_RESULT: CHANGES_REQUESTED
Required fixes:
1. <specific actionable fix>
Evidence: <file/line/test/command rationale>

Task: task id 138; SeldingerMed/seldinger-lumen#53 Implement batched coaxial guidewire + catheter assemblies.
Repo: /Users/colin/Desktop/projects/seldinger/.worktrees/seldinger-lumen-task132
Branch/PR: feat/task132-batched-coaxial, existing PR #77 https://github.com/SeldingerMed/seldinger-lumen/pull/77 (overlaps issue, do not duplicate).
Repo boundary: Apache-2.0 open-core. No private clinical data/PHI/assets allowed.

Issue acceptance:
- Represent guidewire + catheter rods for each env in one Newton model.
- Allocate per-env catheter base actuation arrays, not the current single catheter base.
- Generalize _contact_bodies/n_per_env_contact so wall contact and coaxial coupling map bodies to the right env.
- Preserve two-way guidewire-catheter coupling and existing single-env tests.
- Add a test that creates at least two envs with guidewire + catheter rods and steps them independently.
- Existing coaxial tests continue to pass.
- Throughput path remains device-side with no per-substep host synchronization.

Implementation summary:
- Removes the n_envs != 1 NotImplementedError for coaxial assemblies.
- Builds one guidewire+catheter rod assembly per env, with contiguous guidewire/catheter blocks and per-env catheter bases.
- Introduces n_cath_per_env and n_per_env_contact; passes n_per_env_contact to tube/tree contact.
- Extends coaxial coupling kernel/solver setup with per-env catheter and assembly sizes so guidewire nodes couple only to their env's catheter centerline, preserving two-way reactions.
- Adds two-env independent guidewire/catheter actuation regression and updates solver support docs.

Local checks run this turn:
- python -m ruff check . => PASS (All checks passed!)
- python -m pytest tests/test_newton_coaxial.py tests/test_newton_coaxial_coupling.py tests/test_newton_batched.py tests/test_solver_support_docs.py -q => PASS (23 passed in 8.67s)
- python -m pytest -q => PASS (354 passed in 338.71s)

Intended PR body:
## Summary
- Builds batched coaxial guidewire+catheter assemblies as contiguous per-env Newton body blocks.
- Adds per-env catheter actuation arrays and scopes coaxial coupling to each env's catheter centerline while preserving two-way reactions.
- Updates the solver support matrix and adds a two-env coaxial regression with independent guidewire/catheter actions.

## Tests
- python -m pytest tests/test_newton_coaxial.py tests/test_newton_coaxial_coupling.py tests/test_newton_batched.py tests/test_solver_support_docs.py -q
- python -m ruff check .
- python -m pytest -q

## Risk / data / license / PHI
- No secrets, credentials, PHI, private clinical data, or new external assets.
- Apache-2.0 repo; implementation uses existing Newton/Warp code paths.

Fixes #53

Risk/secrets/PHI/license notes:
- No secrets or credentials in diff.
- No PHI/private clinical data.
- No new data/assets/dependencies; uses existing Newton/Warp code paths in Apache-2.0 repo.

Diff stat:
 docs/SOLVER_SUPPORT.md              | 21 ++++-------
 lumen/newton/sim.py                 | 74 ++++++++++++++++++++++---------------
 lumen/newton/tube_barrier_kernel.py | 19 +++++++---
 lumen/newton/tube_vbd.py            | 24 ++++++++++--
 tests/test_newton_coaxial.py        | 31 ++++++++++++++--
 tests/test_solver_support_docs.py   |  5 +--
 6 files changed, 116 insertions(+), 58 deletions(-)

Full diff:
diff --git a/docs/SOLVER_SUPPORT.md b/docs/SOLVER_SUPPORT.md
index 1b3ed24..438108d 100644
--- a/docs/SOLVER_SUPPORT.md
+++ b/docs/SOLVER_SUPPORT.md
@@ -1,54 +1,49 @@
 # Newton solver support matrix
 
 This matrix is the contract for `lumen.newton.sim.NewtonGuidewireSim`: what works in a single simulation environment, what is vectorized across `n_envs > 1`, and which combinations intentionally fail fast. It tracks the explicit `NotImplementedError` paths in `lumen/newton/sim.py` so users do not discover solver limits only at runtime.
 
 Legend: ✅ supported, ⚠️ supported with stated limits, 🚧 intentionally blocked / follow-up filed. Follow-up links point at the implementation issues that own each remaining batched feature gap.
 
 | Solver path | Single env (`n_envs=1`) | Batched envs (`n_envs>1`) | Runtime guard | Follow-up |
 |---|---:|---:|---|---|
 | Guidewire + tube wall contact | ✅ | ✅ | none | — |
 | Deformable HGO wall | ✅ | ✅ | none | — |
 | Anisotropic friction | ✅ | ✅ | none | — |
 | 1-D `FlowField` coupling | ✅ | ✅ | none | — |
 | Lumped `NewtonFlow` analytic fallback | ✅ | 🚧 | `batched flow requires the 1-D FlowField` | — |
 | Finite clot deformation/damage | ✅ | ✅ with `FlowField`/device coupling | none for batched clot alone | — |
-| Coaxial guidewire + catheter assembly | ✅ | 🚧 | `coaxial assemblies are single-env` | [#53](https://github.com/SeldingerMed/seldinger-lumen/issues/53) |
-<<<<<<< HEAD
+| Coaxial guidewire + catheter assembly | ✅ | ✅ | none | — |
 | Stent-retriever capture/slip/fragmentation | ✅ | ✅ with `FlowField`/clot coupling | `batched stent-retriever retrieval requires the 1-D FlowField coupling path` for non-`FlowField` batched sims | — |
 | Vascular-tree contact | ✅ | ✅ | none | — |
 | Tree + sim-level `lumen_field` | 🚧 | 🚧 | `tree contact takes R0 from each edge's lumen field` | [#55](https://github.com/SeldingerMed/seldinger-lumen/issues/55) |
 | Tree + flow/clot coupling | 🚧 | 🚧 | `edge-aware tree flow/clot coupling is not wired yet` | [#55](https://github.com/SeldingerMed/seldinger-lumen/issues/55) |
 | Aneurysm + flow diverter | ✅ with `FlowField` | ✅ with `FlowField` | none | — |
 | Aneurysm without `FlowField` | 🚧 | 🚧 | `an aneurysm needs the 1-D FlowField` | [#56](https://github.com/SeldingerMed/seldinger-lumen/issues/56) |
 
 ## Follow-up implementation tracker
 
 | Gap | Implementation issue | Required closure evidence |
 |---|---|---|
-| Batched coaxial guidewire + catheter assemblies | [#53](https://github.com/SeldingerMed/seldinger-lumen/issues/53) | A two-env coaxial construction/step test with independent guidewire and catheter bases, plus unchanged single-env coaxial coverage. |
-<<<<<<< HEAD
-| Batched stent-retriever clot retrieval | [#54](https://github.com/SeldingerMed/seldinger-lumen/issues/54) | A two-env retrieval test where capture/slip/fragmentation state diverges per env without host-state bleed-through. |
 | Tree flow/clot coupling | [#55](https://github.com/SeldingerMed/seldinger-lumen/issues/55) | Edge-aware flow/clot coverage on graph edges. Batched tree contact is covered by a two-env tree contact test on a procedural tree; flow/clot stays guarded until it has graph fields instead of a single route centerline. |
-| Batched aneurysm flow-diverter simulations | [#56](https://github.com/SeldingerMed/seldinger-lumen/issues/56) | A two-env aneurysm test with per-env sac state and neck-pressure reads from the matching batched `FlowField`. |
 
-## Why the remaining gaps exist
+## Closed batched gaps
 
-### Coaxial batching (#53)
+### Coaxial batching ([#53](https://github.com/SeldingerMed/seldinger-lumen/issues/53))
 
-The single-env coaxial path adds one catheter rod, one catheter base, and one set of catheter insertion/twist arrays. Batched support must allocate one catheter assembly per env and preserve the body-to-env mapping for both tube contact and coaxial guidewire-catheter coupling. Until then, `n_envs > 1` would mix bodies across envs, so the constructor fails fast.
+Batched coaxial guidewire + catheter assemblies now allocate one guidewire rod, one catheter rod, one guidewire base, and one catheter base per env. Bodies are created as contiguous per-env assemblies so tube/tree wall contact can map body ids to the correct env wall block, while the coaxial coupling kernel restricts each guidewire to its own env's catheter centerline. Closure evidence: a two-env coaxial construction/step test drives independent guidewire and catheter base arrays and preserves the existing single-env coaxial coverage.
 
-### Stent-retriever batching (#54)
+### Stent-retriever batching ([#54](https://github.com/SeldingerMed/seldinger-lumen/issues/54))
 
-Retrieval currently performs capture, slip, fragmentation, and force-balance updates as per-env host logic. Batched retrieval needs independent device or batched host state for each clot/retriever pair so one env's capture event cannot affect another env.
+Batched stent-retriever capture/slip/fragmentation is supported when the sim uses the 1-D `FlowField` clot/device coupling path. The remaining guard requires `FlowField` for batched retrieval because the analytic lumped flow path is still single-env.
 
-### Tree flow/clot (#55)
+### Vascular-tree contact ([#55](https://github.com/SeldingerMed/seldinger-lumen/issues/55))
 
 Tree contact uses per-edge lumen fields and route-centered actuation, and is now safe in batched simulations by allocating independent env×edge wall deformation/load blocks over the shared procedural tree graph. Flow drag and clot grids remain intentionally blocked because they are still parameterized by one linear centerline; tree + flow/clot must first become edge-aware rather than reusing the straight/route centerline arrays.
 
-### Aneurysm batching (#56)
+### Aneurysm batching ([#56](https://github.com/SeldingerMed/seldinger-lumen/issues/56))
 
 Aneurysm flow diversion is now batched when the sim uses the 1-D `FlowField`: each env owns an independent `AneurysmSac`, can use distinct aneurysm/diverter parameters, and reads the corresponding env block from the batched pressure field. The physics limit remains the same as the single-env path: sac→parent back-reaction is not fed into the 1-D parent-flow solve, so the model captures diverter-induced sac stasis but not a neck draw that perturbs parent-vessel through-flow.
 
 ## Development rule
 
 When adding a new solver combination, update this file in the same PR as the implementation and add a regression test that covers both the newly supported path and any combinations that remain intentionally blocked. Do not remove a `NotImplementedError` unless the corresponding row moves from 🚧 to ✅/⚠️.
diff --git a/lumen/newton/sim.py b/lumen/newton/sim.py
index a5a8fd0..f0cbadf 100644
--- a/lumen/newton/sim.py
+++ b/lumen/newton/sim.py
@@ -1,538 +1,554 @@
 """Newton-based guidewire simulation with tube-intrinsic contact (doc §3.2).
 
 The guidewire is a Newton ``add_rod`` cable (stretch + bend/twist), integrated by
 the forked ``TubeVBDSolver`` (``lumen.newton.tube_vbd``) which injects the
 tube-intrinsic barrier (force + Hessian) natively into VBD's per-color AVBD solve
 — so contact is implicit and stable, not an external predictor force. This is the
 faithful replatform of Layer 0 onto Newton (doc §3.2: a domain-specialized module
 inside the engine, replacing generic device-vs-mesh collision).
 
 Proximal-end actuation (insertion = advance the kinematic base along the centerline
 arc-length, staying in the lumen through curves; rotation = spin its quaternion)
 matches the continuum action space (§1.2).
 """
 
 from __future__ import annotations
 
 import warnings
 
 import numpy as np
 import warp as wp
 import newton
 
 from lumen.core.frame import CenterlineFrame
 from lumen.newton.tube_vbd import TubeVBDSolver
 from lumen.newton.forces import add_world_force, add_body_forces, actuate_bases
 from lumen.newton.coupling import compose_radius_k, flow_drag_k
 
 
 class NewtonGuidewireSim:
     def __init__(self, vessel_centerline: np.ndarray, R: float,
                  device_points: np.ndarray, radius: float = 0.2,
                  stretch_stiffness: float = 1.0e4, bend_stiffness: float = 5.0e1,
                  bend_damping: float = 1.0, density: float = 1.0,
                  kappa: float = 2.0e3, d_hat: float = 0.3,
                  barrier_mode: str = "compliant",
                  deformable_wall: bool = False, hgo_params=None,
                  mu_along: float = 0.0, mu_across: float = 0.0,
                  gamma_fric_deg: float = 40.0, lumen_field=None, flow=None,
                  clot_segment=None, clot_height: float = 1.6, clot_params=None,
                  stentriever=None, aneurysm=None, flow_diverter=None, n_envs: int = 1,
                  vbd_iterations: int = 10, device: str | None = None,
                  catheter_points=None, catheter_radius: float = 0.65,
                  catheter_stretch_stiffness: float = 2.0e4,
                  catheter_bend_stiffness: float = 1.5e2,
                  couple_coaxial: bool = True, catheter_inner_radius: float = 0.5,
                  coax_kappa: float = 2.0e3, coax_d_hat: float = 0.1,
                  coax_two_way: bool = True, tree=None, route_centerline=None):
         from lumen.hardware import configure_backend_logging, detect_device
         configure_backend_logging()
         self.device = device or detect_device()      # cuda if available, else cpu
         self.R, self.kappa, self.d_hat = R, kappa, d_hat
         self.n_envs = int(n_envs)
-        # L0d.2a — optional COAXIAL microcatheter: a second rod (larger radius, stiffer)
-        # sharing the lumen with the guidewire. Single-env for now; flow/clot/stentriever
-        # run through the host path, with aspiration keyed to the catheter tip and
-        # flow-drag applied to the guidewire.
+        # L0d.2a — optional COAXIAL microcatheter: a second rod (larger radius,
+        # stiffer) sharing the lumen with the guidewire. Batched coaxial builds one
+        # guidewire+catheter assembly per env in a single Newton model; per-env bodies
+        # stay contiguous so contact/coupling kernels map body ids to the right wall
+        # block without host synchronization in substeps.
         self.coaxial = catheter_points is not None
         if self.coaxial:
             if len(catheter_points) < 2:          # fail fast, before _add_rod's opaque error
                 raise ValueError("catheter_points needs >= 2 nodes (a rod centerline)")
-            if self.n_envs != 1:
-                raise NotImplementedError("coaxial assemblies are single-env (batched coaxial is future)")
         self.contact_frame = CenterlineFrame(vessel_centerline)
         # Batched envs share one vessel (the contact is wire-vs-wall, never wire-vs-wire,
         # so E rods in one model are independent). For n_envs>1 the wall/clot/flow co-sim
         # runs through on-device kernels (no per-substep host round-trip); n_envs==1 keeps
         # the original host path. The lumped Windkessel (NewtonFlow) is single-env only —
         # batched flow needs the 1-D FlowField.
         self._flow_is_field = flow is not None and hasattr(flow, "set_lumen")
         if self.n_envs > 1 and flow is not None and not self._flow_is_field:
             raise NotImplementedError(
                 "batched flow requires the 1-D FlowField; the lumped NewtonFlow is "
                 "single-env (analytic fallback)")
 
         builder = newton.ModelBuilder(gravity=0.0)
         builder.default_shape_cfg.density = density
 
         def _add_rod(seed_pts, rod_radius, stretch, bend):
             pts = [wp.vec3(*map(float, p)) for p in seed_pts]
             quats = newton.utils.create_parallel_transport_cable_quaternions(pts)
             # Newton 1.4 interprets damping as an absolute physical coefficient; Lumen's
             # public knob remains a stiffness-relative multiplier for continuity with
             # the validated pre-1.4 behavior.
             bend_damping_abs = bend_damping * bend
             bodies, _ = builder.add_rod(pts, quats, radius=rod_radius, stretch_stiffness=stretch,
                                         bend_stiffness=bend, bend_damping=bend_damping_abs,
                                         body_frame_origin="com")
             builder.body_mass[bodies[0]] = 0.0       # kinematic base (proximal actuation)
             builder.body_inv_mass[bodies[0]] = 0.0
             builder.body_inertia[bodies[0]] = wp.mat33(0.0)
             builder.body_inv_inertia[bodies[0]] = wp.mat33(0.0)
             return bodies
 
-        # guidewire (one rod per env). self.bodies stays the GUIDEWIRE so positions /
+        # One assembly per env. self.bodies stays the GUIDEWIRE so positions /
         # projection / flow are backward-compatible; the catheter rides _contact_bodies.
         self.bodies, self.bases = [], []
+        self.cath_bodies, self.cath_bases = [], []
         for _ in range(self.n_envs):
             gw = _add_rod(device_points, radius, stretch_stiffness, bend_stiffness)
             self.bodies.extend(gw)
             self.bases.append(gw[0])
+            if self.coaxial:
+                cath = _add_rod(catheter_points, catheter_radius,
+                                catheter_stretch_stiffness, catheter_bend_stiffness)
+                self.cath_bodies.extend(cath)
+                self.cath_bases.append(cath[0])
         if self.n_envs <= 0:
             raise ValueError("n_envs must be > 0")
         n_bodies = len(self.bodies)
         if n_bodies % self.n_envs != 0:
             raise ValueError(
                 f"bodies length ({n_bodies}) must be evenly divisible by n_envs ({self.n_envs}) "
                 "when computing n_per_env (integer division would cause incorrect env indexing)"
             )
         self.n_per_env = n_bodies // self.n_envs   # add_rod: N+1 points -> N bodies
-        self.base = self.bases[0]
-        # optional coaxial microcatheter (single-env): a second, larger, stiffer rod
-        self.cath_bodies, self.cath_bases = [], []
+        self.n_cath_per_env = 0
+        self.n_per_env_contact = self.n_per_env
         if self.coaxial:
-            cath = _add_rod(catheter_points, catheter_radius,
-                            catheter_stretch_stiffness, catheter_bend_stiffness)
-            self.cath_bodies.extend(cath)
-            self.cath_bases.append(cath[0])
+            n_cath_bodies = len(self.cath_bodies)
+            if n_cath_bodies % self.n_envs != 0:
+                raise ValueError("catheter bodies length must be evenly divisible by n_envs")
+            self.n_cath_per_env = n_cath_bodies // self.n_envs
+            self.n_per_env_contact = self.n_per_env + self.n_cath_per_env
+        self.base = self.bases[0]
         self._contact_bodies = self.bodies + self.cath_bodies   # both rods hit the wall
         if self.coaxial:
             # the guidewire and catheter are coupled by the radial constraint (L0d.2b),
             # NOT by capsule contact — disable Newton body-body collision between the two
             # rods so the catheter slides freely over the guidewire (else it drags it).
             shape_of: dict[int, list[int]] = {}
             for s, bod in enumerate(builder.shape_body):
                 shape_of.setdefault(int(bod), []).append(s)
             gw_shapes = [s for bod in self.bodies for s in shape_of.get(bod, [])]
             cath_shapes = [s for bod in self.cath_bodies for s in shape_of.get(bod, [])]
             for sa in gw_shapes:
                 for sb in cath_shapes:
                     builder.add_shape_collision_filter_pair(sa, sb)
         builder.color()
         self.model = builder.finalize(device=self.device)
 
         with warnings.catch_warnings():
             warnings.filterwarnings("ignore", message="SolverVBD damping behavior changed.*",
                                     category=UserWarning)
             self.solver = TubeVBDSolver(self.model, iterations=vbd_iterations)
         self.tree = tree
         if tree is not None:                          # multi-edge vascular tree (batched contact)
             # the tree uses each edge's own lumen field for the base R0, so a sim-level
             # lumen_field doesn't apply; flow/clot project a single centerline. deformable_
             # wall + hgo_params ARE supported (per-edge HGO wall, L0d.1d).
             if lumen_field is not None:
                 raise NotImplementedError("tree contact takes R0 from each edge's lumen "
                                           "field; a sim-level lumen_field doesn't apply")
             if flow is not None or clot_segment is not None:
                 raise NotImplementedError(
                     "edge-aware tree flow/clot coupling is not wired yet: flow drag and "
                     "clot grids need per-edge graph fields, not a single route centerline")
-            n_per_env_contact = len(self._contact_bodies) if self.coaxial else self.n_per_env
+            n_per_env_contact = self.n_per_env_contact
             self.solver.set_tree_contact(tree, self._contact_bodies, kappa=kappa, d_hat=d_hat,
                                          barrier_mode=barrier_mode, mu_along=mu_along,
                                          mu_across=mu_across, gamma_fric_deg=gamma_fric_deg,
                                          actuation_centerline=route_centerline,
                                          deformable_wall=deformable_wall, hgo_params=hgo_params,
                                          n_envs=self.n_envs, n_per_env=n_per_env_contact)
         else:
-            # single-env coaxial passes n_per_env = ALL bodies so every body maps to env 0
-            # (env = body_id // n_per_env); non-coaxial keeps the per-env guidewire blocking.
-            n_per_env_contact = len(self._contact_bodies) if self.coaxial else self.n_per_env
+            n_per_env_contact = self.n_per_env_contact
             self.solver.set_tube_contact(vessel_centerline, R, self._contact_bodies,
                                          kappa=kappa, d_hat=d_hat,
                                          barrier_mode=barrier_mode,
                                          deformable_wall=deformable_wall,
                                          hgo_params=hgo_params, mu_along=mu_along,
                                          mu_across=mu_across, gamma_fric_deg=gamma_fric_deg,
                                          lumen_field=lumen_field,
                                          n_envs=self.n_envs, n_per_env=n_per_env_contact)
         self.state_0 = self.model.state()
         self.state_1 = self.model.state()
         self.control = self.model.control()
         self.contacts = self.model.contacts()
         self.body_ids = wp.array(np.array(self._contact_bodies, dtype=np.int32),  # forces on both rods
                                  dtype=wp.int32, device=self.device)
         self._guidewire_body_ids = wp.array(np.array(self.bodies, dtype=np.int32),
                                             dtype=wp.int32, device=self.device)
         # cache wall s-grid for aneurysm interp (fixed per sim; avoids recompute in hot path)
         w = getattr(self.solver, "_wall", None)
         self._s_grid = np.linspace(0.0, w.s_max, w.n_s) if w is not None else None
         # on-device base actuation (no per-substep body_q host round-trip), one per env
         self._base_ids = wp.array(np.array(self.bases, dtype=np.int32), dtype=wp.int32,
                                   device=self.device)
         self._ins_arr = wp.zeros(self.n_envs, dtype=wp.float32, device=self.device)
         self._tw_arr = wp.zeros(self.n_envs, dtype=wp.float32, device=self.device)
         if self.coaxial:                              # independent catheter proximal actuation
             self._cath_base_ids = wp.array(np.array(self.cath_bases, dtype=np.int32),
                                            dtype=wp.int32, device=self.device)
-            self._cath_ins_arr = wp.zeros(1, dtype=wp.float32, device=self.device)
-            self._cath_tw_arr = wp.zeros(1, dtype=wp.float32, device=self.device)
+            self._cath_ins_arr = wp.zeros(self.n_envs, dtype=wp.float32, device=self.device)
+            self._cath_tw_arr = wp.zeros(self.n_envs, dtype=wp.float32, device=self.device)
             if couple_coaxial:                        # gw rides inside the catheter lumen (L0d.2b)
                 # forward the gw radius so the coupling keeps the gw SURFACE (not just its
                 # centre) inside the catheter inner lumen (the barrier limit is
                 # catheter_inner_radius − gw radius; the solver rejects an impossible fit).
                 self.solver.set_coaxial_coupling(self.bodies, self.cath_bodies,
                                                  catheter_inner_radius, kappa=coax_kappa,
                                                  d_hat=coax_d_hat, two_way=coax_two_way,
-                                                 gw_radius=radius)
+                                                 gw_radius=radius, n_envs=self.n_envs,
+                                                 n_gw_per_env=self.n_per_env,
+                                                 n_cath_per_env=self.n_cath_per_env,
+                                                 n_assembly_per_env=self.n_per_env_contact)
         self.flow = flow                 # optional NewtonFlow (lumped) or FlowField (1-D)
         if self._flow_is_field:
             # Bind the FlowField to this sim's batch/device. Its device arrays are
             # sized to n_envs, so a FlowField must not be shared across sims with
             # different shapes — refuse to rebind an already-used or conflicting one.
             if getattr(flow, "_P_d", None) is not None:
                 raise ValueError("this FlowField is already bound to a sim (device "
                                  "arrays allocated); use one FlowField per NewtonGuidewireSim")
             if getattr(flow, "n_envs", 1) not in (1, self.n_envs):
                 raise ValueError(f"FlowField.n_envs ({flow.n_envs}) conflicts with sim "
                                  f"n_envs ({self.n_envs})")
             if getattr(flow, "device", "cpu") not in ("cpu", self.device):
                 raise ValueError(f"FlowField.device ({flow.device}) conflicts with sim "
                                  f"device ({self.device})")
             flow.n_envs = self.n_envs
             flow.device = self.device
         # optional finite-extent deformable clot (shares the wall's s,θ grid)
         self.clot = None
         if clot_segment is not None:
             from lumen.newton.clot import ClotField
             w = self.solver._wall
             self.clot = ClotField(s_max=w.s_max, n_s=w.n_s, n_th=w.n_th, R_base=R,
                                   s0=clot_segment[0], s1=clot_segment[1],
                                   height=clot_height, params=clot_params,
                                   n_envs=self.n_envs, device=self.device)
         self.stentriever = stentriever   # optional device for clot retrieval
         if self.n_envs > 1 and self.stentriever is not None:
             if self.clot is None:
                 raise ValueError("stentriever requires clot_segment so it has a clot to retrieve")
             if not self._flow_is_field:
                 raise NotImplementedError(
                     "batched stent-retriever retrieval requires the 1-D FlowField coupling path")
             if isinstance(self.stentriever, (list, tuple)) and len(self.stentriever) != self.n_envs:
                 raise ValueError("batched stentriever list length must match n_envs")
         # optional saccular aneurysm + flow diverter. Needs the 1-D FlowField, which
         # supplies the pressure P(s_neck) that drives each sac. Batched sims keep one
         # AneurysmSac per env and read the matching env's pressure block from the
         # batched FlowField snapshot after every device-side flow solve.
         self.aneurysm = aneurysm
         self.flow_diverter = flow_diverter
         self.aneurysm_sac = None
         if aneurysm is not None:
             if not self._flow_is_field:
                 raise NotImplementedError("an aneurysm needs the 1-D FlowField (it reads "
                                           "the neck pressure P(s)); pass flow=FlowField(...)")
             s_max = self.solver._wall.s_max
             from lumen.newton.aneurysm import AneurysmSac
             self.aneurysms = self._per_env_objects(aneurysm, "aneurysm")
             self.flow_diverters = self._per_env_objects(flow_diverter, "flow_diverter",
                                                         allow_none=True)
             for an in self.aneurysms:
                 if not (0.0 <= an.s_neck <= s_max):    # else np.interp silently clamps
                     raise ValueError(f"aneurysm s_neck ({an.s_neck}) is outside the "
                                      f"vessel arc-length [0, {s_max:.1f}]")
             if flow is None:
                 raise ValueError("flow is required for aneurysm flow diversion")
             self.aneurysm_sacs = [AneurysmSac(an, visc=flow.p.visc) for an in self.aneurysms]
             self.aneurysm_sac = self.aneurysm_sacs[0]  # backward-compatible single-env handle
         elif flow_diverter is not None:
             raise ValueError("flow_diverter set without an aneurysm to divert from")
         else:
             self.aneurysms = []
             self.flow_diverters = []
             self.aneurysm_sacs = []
         # batched on-device coupling scratch (n_envs>1): node drag arrays + zero occ
         self._use_device_coupling = self.n_envs > 1 and (self.clot is not None
                                                          or self._flow_is_field)
         if self._use_device_coupling:
             w = self.solver._wall
             n_node = self.n_per_env * self.n_envs
             self._snodes_d = wp.zeros(n_node, dtype=wp.float32, device=self.device)
             self._tang_d = wp.zeros(n_node, dtype=wp.vec3, device=self.device)
             self._zero_occ_d = wp.zeros(self.n_envs * w.n_s, dtype=wp.float32,
                                         device=self.device)
         # snapshot for fast reset (avoid rebuilding the model/solver each episode)
         self._init_body_q = self.state_0.body_q.numpy().copy()
 
     def _per_env_objects(self, value, name: str, allow_none: bool = False):
         """Return one object per env; scalars broadcast, explicit sequences must match.
 
         The objects are config/state holders, not numeric arrays. Strings are treated as
         scalars for the generic helper even though current callers pass dataclasses.
         """
         if value is None:
             if allow_none:
                 return [None] * self.n_envs
             raise ValueError(f"{name} is required")
         if isinstance(value, (list, tuple, np.ndarray)):
             if len(value) != self.n_envs:
                 raise ValueError(f"{name} length ({len(value)}) must match n_envs ({self.n_envs})")
             if not allow_none and any(v is None for v in value):
                 raise ValueError(f"{name} entries cannot be None")
             return list(value)
         return [value for _ in range(self.n_envs)]
 
     def reset(self):
         """Restore the initial state cheaply (no model/solver rebuild) — for RL."""
         dev = self.device
         self.state_0.body_q = wp.array(self._init_body_q.copy(), dtype=wp.transform, device=dev)
         self.state_1.body_q = wp.array(self._init_body_q.copy(), dtype=wp.transform, device=dev)
         self.state_0.body_qd.zero_()
         self.state_1.body_qd.zero_()
         self.solver.body_q_prev = wp.array(self._init_body_q.copy(), dtype=wp.transform, device=dev)
         w = getattr(self.solver, "_wall", None)
         if w is not None:
             w.w[:] = 0.0
             w.w_field.zero_()
             w.wall_load.zero_()
         tw = getattr(self.solver, "_tree_wall", None)        # tree path: clear wall deformation + load
         if tw is not None:
             tw.w[:] = 0.0
             tw.w_field.zero_()
             tw.wall_load.zero_()
         if self.clot is not None:
             self.clot.o0_env[:] = self.clot._initial_o0_env
             self.clot.o_env[:] = self.clot.o0_env
             self.clot.D_env[:] = 0.0
             self.clot.mask_env[:] = self.clot.o0_env > 1e-6
             self.clot.retrieved_env[:] = 0.0
             self.clot._sync_public_from_env(0)
             self.clot.sync_to_device()
         if self.aneurysm_sac is not None:
             self.aneurysm_sac.reset()
         for sac in self.aneurysm_sacs:
             sac.reset()
 
     def _actuate(self, base_ids, ins_arr, tw_arr, insertion, twist, n):
         """Centerline-following insertion/twist of `n` kinematic bases (one per env, or
         the single catheter base). Scalars broadcast; arrays are per-env RL actions."""
         ins = np.broadcast_to(np.asarray(insertion, np.float32), (n,))
         tw = np.broadcast_to(np.asarray(twist, np.float32), (n,))
         if not ins.any() and not tw.any():
             return
         ins_arr.assign(np.ascontiguousarray(ins))
         tw_arr.assign(np.ascontiguousarray(tw))
         sv = self.solver
         wp.launch(actuate_bases, dim=n,
                   inputs=[base_ids, ins_arr, tw_arr,
                           sv._tube_P, sv._tube_Tg, sv._tube_cum_s, sv._tube_M, sv._tube_s_max],
                   outputs=[self.state_0.body_q], device=self.device)
 
     def _actuate_base(self, insertion, twist):
         # translate/rotate each guidewire kinematic base along the centerline (#23)
         self._actuate(self._base_ids, self._ins_arr, self._tw_arr, insertion, twist, self.n_envs)
 
     def _actuate_catheter(self, insertion, twist):
         self._actuate(self._cath_base_ids, self._cath_ins_arr, self._cath_tw_arr,
-                      insertion, twist, 1)
+                      insertion, twist, self.n_envs)
 
     def step(self, dt: float = 2.5e-2, substeps: int = 5,
-             insertion: float = 0.0, twist: float = 0.0, preload=(0.0, 0.0, 0.0),
-             aspiration: float = 0.0, insertion_cath: float = 0.0, twist_cath: float = 0.0):
+             insertion: float | np.ndarray = 0.0, twist: float | np.ndarray = 0.0,
+             preload=(0.0, 0.0, 0.0),
+             aspiration: float = 0.0, insertion_cath: float | np.ndarray = 0.0,
+             twist_cath: float | np.ndarray = 0.0):
         """Advance the simulation by `dt` total, as `substeps` sub-steps of
         `dt/substeps` each (the standard substep convention).
 
         `insertion_cath`/`twist_cath` independently actuate the coaxial microcatheter
         (ignored when there is no catheter)."""
         sub_dt = dt / substeps
         if self.flow is not None and aspiration:
             self.flow.aspiration = aspiration        # aspiration recovers downstream flow
         if self._use_device_coupling:                # n_envs>1 with clot/flow: on-device
-            self._step_device(sub_dt, substeps, insertion, twist, preload)
+            self._step_device(sub_dt, substeps, insertion, twist, preload,
+                              insertion_cath, twist_cath)
             return
         self._step_host(sub_dt, substeps, insertion, twist, preload, aspiration, dt,
                         insertion_cath, twist_cath)
 
-    def _step_device(self, sub_dt, substeps, insertion, twist, preload):
+    def _step_device(self, sub_dt, substeps, insertion, twist, preload,
+                     insertion_cath: float | np.ndarray = 0.0,
+                     twist_cath: float | np.ndarray = 0.0):
         """Batched per-substep co-sim entirely on device (no host round-trip):
         compose R0 (pulse − clot occlusion) -> flow solve -> local drag -> contact
         solve -> clot update. Each env reads/writes its own wall/clot/flow block."""
         wall = self.solver._wall
         n_s, n_th, s_max, ncell = wall.n_s, wall.n_th, wall.s_max, wall.n_cells
         field_flow = self._flow_is_field
         if field_flow:                               # per-step host prep for local drag
             pos = self.state_0.body_q.numpy()[self.bodies, :3].reshape(
                 self.n_envs, self.n_per_env, 3)
             tang = np.zeros_like(pos)
             tang[:, 1:-1] = pos[:, 2:] - pos[:, :-2]
             tang[:, 0] = pos[:, 1] - pos[:, 0]
             tang[:, -1] = pos[:, -1] - pos[:, -2]
             tang /= (np.linalg.norm(tang, axis=2, keepdims=True) + 1e-12)
             s_nodes = self.contact_frame.project_s(pos.reshape(-1, 3))   # vectorized
             self._tang_d.assign(np.ascontiguousarray(tang.reshape(-1, 3).astype(np.float32)))
             self._snodes_d.assign(s_nodes.astype(np.float32))
             self.flow.set_tips(s_nodes, s_max, n_s)
         occ_arr = self.clot.o_d if self.clot is not None else self._zero_occ_d
         for _ in range(substeps):
             self.state_0.clear_forces()
             self._actuate_base(np.asarray(insertion, np.float32) / substeps,
-                               np.asarray(twist, np.float32) / substeps)   # batched: no coaxial
+                               np.asarray(twist, np.float32) / substeps)
+            if self.coaxial:
+                self._actuate_catheter(np.asarray(insertion_cath, np.float32) / substeps,
+                                       np.asarray(twist_cath, np.float32) / substeps)
             if any(preload):
                 wp.launch(add_world_force, dim=self.body_ids.shape[0],
                           inputs=[self.body_ids, float(preload[0]), float(preload[1]),
                                   float(preload[2]), 1],
                           outputs=[self.state_0.body_f], device=self.device)
             pulse = self.flow.pulse_factor() if self.flow is not None else 1.0
             wp.launch(compose_radius_k, dim=self.n_envs * ncell,
                       inputs=[wall._R0_base_d, occ_arr, float(pulse), n_s, n_th],
                       outputs=[wall.r0_field, wall.clot_mask_field], device=self.device)
             if field_flow:
                 self.flow.advance(sub_dt)
                 self.flow.solve_device(wall.r0_field, n_s, n_th, s_max)
                 if self.aneurysm_sacs:
                     # Synchronize the just-solved batched pressure field to host and
                     # update each env's independent 0-D sac from its own P(s_neck) and
                     # flow-diverter coverage. This preserves the device-side FlowField
                     # solve while avoiding cross-env host state bleed-through.
                     P_env = self.flow._P_d.numpy().reshape(self.n_envs, n_s)
                     s_grid = self._s_grid if self._s_grid is not None else np.linspace(0.0, s_max, n_s)
                     for env, (an, sac, fd) in enumerate(zip(self.aneurysms,
                                                            self.aneurysm_sacs,
                                                            self.flow_diverters)):
                         P_neck = float(np.interp(an.s_neck, s_grid, P_env[env]))
                         div = fd.diversion(an) if fd is not None else 0.0
                         sac.update(P_neck, sub_dt, diversion=div)
                 wp.launch(flow_drag_k, dim=self.n_envs * self.n_per_env,
                           inputs=[self._snodes_d, self._tang_d, self.body_ids,
                                   self.flow._v_d, self.n_per_env, n_s, float(s_max),
                                   float(self.flow.p.drag_coeff)],
                           outputs=[self.state_0.body_f], device=self.device)
             self.solver.step(self.state_0, self.state_1, self.control, self.contacts, sub_dt)
             if self.clot is not None:
                 self.clot.update_device(wall.wall_load, sub_dt)
             self.state_0, self.state_1 = self.state_1, self.state_0
         self._retrieve_batched_if_needed(insertion, dt=sub_dt * substeps)
 
     def _retrieve_batched_if_needed(self, insertion, dt: float) -> None:
         if self.stentriever is None or self.clot is None:
             return
         ins = np.broadcast_to(np.asarray(insertion, dtype=float), (self.n_envs,))
         if not np.any(ins < 0.0):
             return
         self.clot.sync_from_device()
         stents = (self.stentriever if isinstance(self.stentriever, (list, tuple))
                   else [self.stentriever] * self.n_envs)
         engagement = np.array([
             st.engagement_strength_for_mask(self.clot.s_grid, self.clot.mask_env[e])
             for e, st in enumerate(stents)
         ], dtype=float)
         aspiration = np.full(self.n_envs,
                              self.flow.aspiration if self.flow is not None else 0.0,
                              dtype=float)
         wall = self.solver._wall
         if self._flow_is_field and self.flow is not None and getattr(self.flow, "_P_d", None) is not None:
             P = self.flow._P_d.numpy().reshape(self.n_envs, wall.n_s)
             s_flow = np.linspace(0.0, wall.s_max, wall.n_s)
             for e in range(self.n_envs):
                 mask = self.clot.mask_env[e]
                 if not mask.any():
                     continue
                 clot_s = self.clot.s_grid[mask]
                 P_prox = float(np.interp(clot_s[0], s_flow, P[e]))
                 P_dist = float(np.interp(clot_s[-1], s_flow, P[e]))
                 r_open = np.maximum(self.R - self.clot.o_env[e, mask], self.flow.p.R_floor)
                 A_clot = np.pi * float(np.mean(r_open)) ** 2
                 aspiration[e] += (P_dist - P_prox) * A_clot
         self.last_retrieval = self.clot.retrieve_batched(-ins, engagement, aspiration, dt)
 
     def _step_host(self, sub_dt, substeps, insertion, twist, preload, aspiration, dt,
-                   insertion_cath=0.0, twist_cath=0.0):
-        """Single-env (n_envs==1) path: the original host-side co-sim, unchanged."""
+                   insertion_cath: float | np.ndarray = 0.0,
+                   twist_cath: float | np.ndarray = 0.0):
+        """Host-side co-sim path for single-env or batched sims without device flow/clot."""
         # flow drag acts along the (slowly-varying) device tangents — compute once/step
         tang = None
         s_nodes = None
         if self.flow is not None:
             pos = self.state_0.body_q.numpy()[self.bodies, :3]
             tang = np.zeros_like(pos)
             tang[1:-1] = pos[2:] - pos[:-2]
             tang[0] = pos[1] - pos[0]
             tang[-1] = pos[-1] - pos[-2]
             tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12)
             if self._flow_is_field:                  # device-node arc-lengths for local v(s)
                 s_nodes = np.array([self.contact_frame.project(p).s for p in pos])
                 if self.coaxial and len(self.cath_bodies):
                     cath_pos = self.state_0.body_q.numpy()[self.cath_bodies, :3]
                     cath_s = np.array([self.contact_frame.project(p).s for p in cath_pos])
                     self.flow.set_tip(float(cath_s.max()))   # aspiration point = catheter tip
                 else:
                     self.flow.set_tip(float(s_nodes.max()))   # no catheter: deepest wire node
         for _ in range(substeps):
             self.state_0.clear_forces()
             self._actuate_base(np.asarray(insertion, np.float32) / substeps,
                                np.asarray(twist, np.float32) / substeps)
             if self.coaxial:                          # independent microcatheter actuation
                 self._actuate_catheter(insertion_cath / substeps, twist_cath / substeps)
             if any(preload):
                 wp.launch(add_world_force, dim=self.body_ids.shape[0],
                           inputs=[self.body_ids, float(preload[0]), float(preload[1]),
                                   float(preload[2]), 1],
                           outputs=[self.state_0.body_f], device=self.device)
             wall = getattr(self.solver, "_wall", None)
             # compose the shared effective base radius R0(s,θ,t) = R0_base × pulse −
             # clot occlusion, which the contact kernel reads (+ wall w). M1: pulse
             # modulates the *vessel* wall only, NOT the clot (a clot is incompressible
             # tissue — it doesn't breathe with the cardiac cycle).
             if wall is not None and (self.flow is not None or self.clot is not None):
                 pulse = self.flow.pulse_factor() if self.flow is not None else 1.0
                 base = wall._R0_base * np.float32(pulse)
                 if self.clot is not None:
                     occ = self.clot.occlusion_grid().astype(np.float32)
                     base = base - occ
                     wall.set_clot_mask(occ)              # H1: clot bears its own load
                 wall.r0_field.assign(base.astype(np.float32))
             if self.flow is not None:
                 self.flow.advance(sub_dt)
                 if self._flow_is_field:
                     # feed the SHARED lumen radius (θ-averaged open radius) to the 1-D
                     # network, solve P(s)/v(s), then drag each node by its LOCAL v(s).
                     r_open_s = base.reshape(wall.n_s, wall.n_th).mean(axis=1)
                     self.flow.set_lumen(r_open_s, wall.s_max)
                     self.flow.solve()
                     if self.aneurysm_sac is not None:    # drive the sac from P(s_neck)
                         # Host path is only for n_envs==1 (see _use_device_coupling which requires n>1
                         # for field flow); n>1 aneurysms go through the batched _step_device path.
                         # The [0] indexing and singular handle are for single-env + backward compat.
                         an = self.aneurysms[0]
                         fd = self.flow_diverters[0]
                         P_neck = float(np.interp(an.s_neck, self.flow.s_grid,
                                                  self.flow.pressure_field()))
                         div = fd.diversion(an) if fd is not None else 0.0
                         self.aneurysm_sac.update(P_neck, sub_dt, diversion=div)
                     drag = self.flow.drag_at(s_nodes)[:, None]
                 else:
                     drag = self.flow.drag_per_unit_tangent()
                 dvecs = wp.array((drag * tang).astype(np.float32),
                                  dtype=wp.vec3, device=self.device)
                 wp.launch(add_body_forces, dim=self._guidewire_body_ids.shape[0],
                           inputs=[self._guidewire_body_ids, dvecs, 1],
                           outputs=[self.state_0.body_f], device=self.device)
             self.solver.step(self.state_0, self.state_1, self.control,
                              self.contacts, sub_dt)
             if self.clot is not None:                # clot deforms/damages from contact load
                 occ = self.clot.update(wall.wall_load.numpy(), sub_dt)
                 if self.flow is not None and not self._flow_is_field:
                     self.flow.occlusion = occ        # lumped model: feed the scalar blockage
             self.state_0, self.state_1 = self.state_1, self.state_0
         # stent-retriever: on retraction, drag the clot proximally (retrieve/slip/fragment)
         if self.stentriever is not None and self.clot is not None and insertion < 0.0:
             if self._flow_is_field and self.clot.mask.any():
                 # mobilising force = retrograde pressure gradient the aspiration sink
                 # builds across the clot (real hemodynamics), plus any direct command.
diff --git a/lumen/newton/tube_barrier_kernel.py b/lumen/newton/tube_barrier_kernel.py
index c50890f..575a57e 100644
--- a/lumen/newton/tube_barrier_kernel.py
+++ b/lumen/newton/tube_barrier_kernel.py
@@ -1,151 +1,158 @@
 """Tube-intrinsic barrier as a native AVBD constraint, over a deformable wall.
 
 Injected into the forked SolverVBD's per-color rigid solve. Adds BOTH the barrier
 reaction force and its Hessian (κ·eᵣ⊗eᵣ) to the per-body 6×6 system so contact is
 treated *implicitly* with inertia + cable joints — what makes stiff contact stable
 in VBD (doc §3.5). The lumen radius is the SHARED field R(s,θ) = R0 + w(s,θ): the
 barrier reads the deformed radius and deposits the contact normal load onto the
 wall cell, so the HGO wall (lumen.newton.hgo_wall) and the contact share R
 (doc §3.5.6). With w≡0 the wall is rigid.
 
 Barrier (doc §3.5.3): compliant fast tier E=½κδ² or bounded IPC-log option; the
 rigorous penetration-free IPC is the accurate tier (§3.3).
 
 Precision (#21): geometry arrays are float32 here (Warp GPU throughput), while
 lumen.core.frame is float64. The resulting s/r differences are ~1e-6 at the
 scales we run — negligible and an intentional speed/precision trade. Promote the
 arrays to float64 only if very long centerlines or large coordinates demand it.
 """
 
 from __future__ import annotations
 
 import warp as wp
 
 
 @wp.kernel
 def accumulate_coaxial_coupling(
     color_group: wp.array(dtype=wp.int32),
     gw_mask: wp.array(dtype=wp.int32),         # 1 for guidewire bodies (the ones constrained)
     body_q: wp.array(dtype=wp.transform),
-    cath_ids: wp.array(dtype=wp.int32),        # catheter body indices, ordered along the rod
+    cath_ids: wp.array(dtype=wp.int32),        # catheter body indices, grouped per env, ordered along each rod
     n_cath: int,
+    n_cath_per_env: int,
+    n_assembly_per_env: int,
     r_inner: float,                            # catheter inner-lumen radius the gw rides within
     kappa: float, d_hat: float, two_way: int,
     body_forces: wp.array(dtype=wp.vec3),
     body_hessian_ll: wp.array(dtype=wp.mat33),
 ):
     """Sliding coaxial coupling (L0d.2b): keep each guidewire node within the catheter's
     inner lumen. The catheter centerline is read LIVE from body_q (no host rebuild) — so
     as the catheter bends, the gw is barriered to follow, while sliding freely along the
     axis (no tangential force). Structurally the tube barrier, but the 'wall' is the
     dynamic catheter axis and the barrier pulls INWARD (gw stays inside, r < r_inner).
 
     two_way=1 (L0d.2d): the catheter feels the equal-opposite reaction — the contact
     force is split (Newton's third law) onto the two catheter nodes of the nearest
     segment by the barycentric weight, so a stiff guidewire pushes/straightens the
     catheter (a responsive catheter, not a rigid tube). two_way=0 is the one-way model."""
     t = wp.tid()
     bid = color_group[t]
     if gw_mask[bid] == 0:
         return
     p = wp.transform_get_translation(body_q[bid])
     best = float(1.0e30)
     bk = int(0)
     bu = float(0.0)
-    for k in range(n_cath - 1):
+    env = bid // n_assembly_per_env
+    cath0 = env * n_cath_per_env
+    cathN = cath0 + n_cath_per_env
+    if cath0 < 0 or cathN > n_cath or n_cath_per_env < 2:
+        return
+    for k in range(cath0, cathN - 1):
         a = wp.transform_get_translation(body_q[cath_ids[k]])
         ab = wp.transform_get_translation(body_q[cath_ids[k + 1]]) - a
         L2 = wp.dot(ab, ab)
         u = wp.clamp(wp.dot(p - a, ab) / (L2 + 1.0e-12), 0.0, 1.0)
         dd = p - (a + u * ab)
         d2 = wp.dot(dd, dd)
         if d2 < best:
             best = d2
             bk = k
             bu = u
     # a guidewire node axially BEYOND either catheter opening has telescoped out — it's
     # free there, not riding inside, so no coupling (CodeRabbit #22).
-    c0 = wp.transform_get_translation(body_q[cath_ids[0]])
-    c1 = wp.transform_get_translation(body_q[cath_ids[1]])
-    cn = wp.transform_get_translation(body_q[cath_ids[n_cath - 1]])
-    cm = wp.transform_get_translation(body_q[cath_ids[n_cath - 2]])
+    c0 = wp.transform_get_translation(body_q[cath_ids[cath0]])
+    c1 = wp.transform_get_translation(body_q[cath_ids[cath0 + 1]])
+    cn = wp.transform_get_translation(body_q[cath_ids[cathN - 1]])
+    cm = wp.transform_get_translation(body_q[cath_ids[cathN - 2]])
     if wp.dot(p - c0, c1 - c0) < 0.0 or wp.dot(p - cn, cn - cm) > 0.0:
         return
     a = wp.transform_get_translation(body_q[cath_ids[bk]])
     b = wp.transform_get_translation(body_q[cath_ids[bk + 1]])
     foot = a + bu * (b - a)
     tang = wp.normalize(b - a)
     radial = (p - foot) - wp.dot(p - foot, tang) * tang
     r = wp.length(radial)
     er = radial / (r + 1.0e-9)
     dwall = r_inner - r                          # clearance to the catheter inner wall
     if dwall < d_hat:
         bp = -kappa * (d_hat - dwall)            # compliant barrier, pulls inward (bp<0, er outward)
         f = bp * er
         body_forces[bid] = body_forces[bid] + f
         body_hessian_ll[bid] = body_hessian_ll[bid] + kappa * wp.outer(er, er)
         if two_way == 1:                         # catheter feels -f, split by the barycentric u,
             ca = cath_ids[bk]                    # WITH the Hessian (implicit, else it overshoots)
             cb = cath_ids[bk + 1]
             h = kappa * wp.outer(er, er)
             wp.atomic_add(body_forces, ca, -(1.0 - bu) * f)
             wp.atomic_add(body_forces, cb, -bu * f)
             wp.atomic_add(body_hessian_ll, ca, (1.0 - bu) * h)
             wp.atomic_add(body_hessian_ll, cb, bu * h)
 
 
 @wp.func
 def _barrier_dd(d: float, d_hat: float, kappa: float, mode: int):
     """Return (b'(d), b''(d)) for the wall-distance barrier (see module docstring)."""
     if mode == 1:
         dd = wp.max(d, 0.05 * d_hat)
         ln = wp.log(dd / d_hat)
         diff = dd - d_hat
         bp = -kappa * (2.0 * diff * ln + diff * diff / dd)
         bpp = -kappa * (2.0 * ln + 4.0 * diff / dd - diff * diff / (dd * dd))
         bp = wp.max(bp, -50.0 * kappa * d_hat)
         return bp, wp.clamp(bpp, 0.0, 200.0 * kappa)
     return -kappa * (d_hat - d), kappa
 
 
 @wp.kernel
 def accumulate_tube_barrier(
     color_group: wp.array(dtype=wp.int32),
     wire_mask: wp.array(dtype=wp.int32),
     body_q: wp.array(dtype=wp.transform),
     body_qd: wp.array(dtype=wp.spatial_vector),
     P: wp.array(dtype=wp.vec3),               # centerline vertices
     Tg: wp.array(dtype=wp.vec3),              # centerline tangents
     M1: wp.array(dtype=wp.vec3),              # rotation-minimizing reference normals (per vertex)
     cum_s: wp.array(dtype=wp.float32),        # cumulative arc-length (per vertex)
     M: int,
     R0_grid: wp.array(dtype=wp.float32),      # [n_envs*n_s*n_th] BASE lumen radius R0(s,θ)
     s_max: float, n_s: int, n_th: int, n_per_env: int,
     w_field: wp.array(dtype=wp.float32),      # [n_envs*n_s*n_th] radial displacement (shared R)
     kappa: float, d_hat: float, mode: int,
     mu_along: float, mu_across: float, gamma_fric: float, dt: float,  # anisotropic friction
     body_forces: wp.array(dtype=wp.vec3),     # in/out
     body_hessian_ll: wp.array(dtype=wp.mat33),  # in/out
     wall_load: wp.array(dtype=wp.float32),    # [n_s*n_th] accumulated normal load (out)
 ):
     t = wp.tid()
     bid = color_group[t]
     if wire_mask[bid] == 0:
         return
     p = wp.transform_get_translation(body_q[bid])
     best = float(1.0e30)
     bj = int(0)
     bu = float(0.0)
     for j in range(M - 1):
         a = P[j]
         ab = P[j + 1] - a
         L2 = wp.dot(ab, ab)
         u = wp.clamp(wp.dot(p - a, ab) / L2, 0.0, 1.0)
         dd = p - (a + u * ab)
         d2 = wp.dot(dd, dd)
         if d2 < best:
             best = d2
             bj = j
             bu = u
     # #22 — open vessel ends: a node axially past either opening has left the
     # vessel; no wall contact there (don't deposit load at a boundary cell).
diff --git a/lumen/newton/tube_vbd.py b/lumen/newton/tube_vbd.py
index f05642c..66e03c7 100644
--- a/lumen/newton/tube_vbd.py
+++ b/lumen/newton/tube_vbd.py
@@ -138,162 +138,164 @@ class TubeVBDSolver(SolverVBD):
                     inputs=[
                         dt,
                         color_group,
                         self.body_q_prev,
                         state_in.body_q,
                         model.body_com,
                         self.body_inv_mass_effective,
                         self.friction_epsilon,
                         self.body_body_contact_penalty_k,
                         self.body_body_contact_material_ke,
                         self.body_body_contact_material_kd,
                         self.body_body_contact_material_mu,
                         self.body_body_contact_lambda,
                         self.body_body_contact_C0,
                         self.rigid_contact_alpha,
                         self.rigid_contact_hard,
                         contacts.rigid_contact_count,
                         contacts.rigid_contact_shape0,
                         contacts.rigid_contact_shape1,
                         contacts.rigid_contact_point0,
                         contacts.rigid_contact_point1,
                         contacts.rigid_contact_offset0,
                         contacts.rigid_contact_offset1,
                         contacts.rigid_contact_normal,
                         contacts.rigid_contact_margin0,
                         contacts.rigid_contact_margin1,
                         model.shape_body,
                         self.body_body_contact_buffer_pre_alloc,
                         self.body_body_contact_counts,
                         self.body_body_contact_indices,
                     ],
                     outputs=[
                         self.body_forces,
                         self.body_torques,
                         self.body_hessian_ll,
                         self.body_hessian_al,
                         self.body_hessian_aa,
                     ],
                     device=self.device,
                 )
 
             if getattr(self, "_tube_enabled", False):
                 wp.launch(
                     kernel=accumulate_tube_barrier,
                     dim=color_group.size,
                     inputs=[color_group, self._tube_wire_mask, state_in.body_q,
                             state_in.body_qd,
                             self._tube_P, self._tube_Tg, self._tube_M1,
                             self._tube_cum_s, self._tube_M,
                             self._wall.r0_field, self._tube_s_max, self._tube_ns,
                             self._tube_nth, self._tube_n_per_env, self._wall.w_field,
                             self._tube_kappa, self._tube_d_hat, self._tube_mode,
                             self._tube_mu_along, self._tube_mu_across,
                             self._tube_gamma_fric, dt],
                     outputs=[self.body_forces, self.body_hessian_ll,
                              self._tube_wall_load],
                     device=self.device,
                 )
             if getattr(self, "_tree_enabled", False):
                 wp.launch(
                     kernel=accumulate_tree_barrier,
                     dim=color_group.size,
                     inputs=[color_group, self._tree_wire_mask, state_in.body_q,
                             state_in.body_qd,
                             self._tree_P, self._tree_Tg, self._tree_M1, self._tree_cum_s,
                             self._tree_vstart, self._tree_vcount, self._tree_smax,
                             self._tree_start_junc, self._tree_end_junc, self._tree_n_edges,
                             self._tree_R0, self._tree_w, self._tree_ns, self._tree_nth,
                             self._tree_n_per_env, self._tree_kappa, self._tree_d_hat,
                             self._tree_mode,
                             self._tree_mu_along, self._tree_mu_across,
                             self._tree_gamma_fric, dt],
                     outputs=[self.body_forces, self.body_hessian_ll, self._tree_wall_load],
                     device=self.device,
                 )
             if getattr(self, "_coax_enabled", False):
                 wp.launch(
                     kernel=accumulate_coaxial_coupling,
                     dim=color_group.size,
                     inputs=[color_group, self._coax_gw_mask, state_in.body_q,
-                            self._coax_cath_ids, self._coax_n_cath, self._coax_r_inner,
-                            self._coax_kappa, self._coax_d_hat, self._coax_two_way],
+                            self._coax_cath_ids, self._coax_n_cath,
+                            self._coax_n_cath_per_env, self._coax_n_assembly_per_env,
+                            self._coax_r_inner, self._coax_kappa, self._coax_d_hat,
+                            self._coax_two_way],
                     outputs=[self.body_forces, self.body_hessian_ll],
                     device=self.device,
                 )
             wp.launch(
                 kernel=solve_rigid_body,
                 inputs=[
                     dt,
                     color_group,
                     state_in.body_q,
                     self.body_q_prev,
                     model.body_q,
                     model.body_mass,
                     self.body_inv_mass_effective,
                     model.body_inertia,
                     self.body_inertia_q,
                     model.body_com,
                     self.rigid_adjacency,
                     model.joint_type,
                     model.joint_enabled,
                     model.joint_parent,
                     model.joint_child,
                     model.joint_X_p,
                     model.joint_X_c,
                     model.joint_axis,
                     model.joint_qd_start,
                     model.joint_target_q_start,
                     self.joint_constraint_start,
                     self.joint_penalty_k,
                     self.joint_penalty_kd,
                     self.joint_sigma_start,
                     self.joint_C_fric,
                     model.joint_target_ke,
                     model.joint_target_kd,
                     control.joint_target_q,
                     control.joint_target_qd,
                     model.joint_limit_lower,
                     model.joint_limit_upper,
                     model.joint_limit_ke,
                     model.joint_limit_kd,
                     self.joint_lambda_lin,
                     self.joint_lambda_ang,
                     self.joint_C0_lin,
                     self.joint_C0_ang,
                     self.joint_is_hard,
                     self.rigid_joint_alpha,
                     model.joint_dof_dim,
                     self.joint_rest_angle,
                     self.body_forces,
                     self.body_torques,
                     self.body_hessian_ll,
                     self.body_hessian_al,
                     self.body_hessian_aa,
                 ],
                 outputs=[
                     state_in.body_q,
                 ],
                 dim=color_group.size,
                 device=self.device,
             )
 
         if contacts is not None:
             contact_launch_dim = contacts.rigid_contact_max
             wp.launch(
                 kernel=update_duals_body_body_contacts,
                 dim=contact_launch_dim,
                 inputs=[
                     contacts.rigid_contact_count,
                     contacts.rigid_contact_shape0,
                     contacts.rigid_contact_shape1,
                     contacts.rigid_contact_point0,
                     contacts.rigid_contact_point1,
                     contacts.rigid_contact_offset0,
                     contacts.rigid_contact_offset1,
                     contacts.rigid_contact_normal,
                     contacts.rigid_contact_margin0,
                     contacts.rigid_contact_margin1,
                     model.shape_body,
                     state_in.body_q,
                     self.body_q_prev,
                     self.body_body_contact_material_mu,
@@ -359,179 +361,195 @@ class TubeVBDSolver(SolverVBD):
                     self.rigid_joint_alpha,
                     self.joint_penalty_k_max,
                     self.rigid_linear_beta,
                     self.rigid_angular_beta,
                     model.joint_target_ke,
                     control.joint_target_q,
                     model.joint_limit_lower,
                     model.joint_limit_upper,
                     model.joint_limit_ke,
                     self.joint_rest_angle,
                     self.joint_penalty_k,  # input/output
                     self.joint_lambda_lin,  # input/output
                     self.joint_lambda_ang,  # input/output
                 ],
                 device=self.device,
             )
 
     def set_tube_contact(self, centerline, R, wire_body_ids, kappa=2.0e3, d_hat=0.3,
                          barrier_mode="compliant", deformable_wall=False,
                          hgo_params=None, n_s=40, n_th=16,
                          mu_along=0.0, mu_across=0.0, gamma_fric_deg=40.0,
                          lumen_field=None, n_envs=1, n_per_env=None):
         """barrier_mode: 'compliant' (fast tier) | 'log' (bounded IPC option).
 
         R is the base lumen radius: a scalar (cylinder) OR, via ``lumen_field`` (a
         ``lumen.core.LumenField``), the true R(s,θ) anatomy (stenosis/aneurysm/
         patient). The contact reads R0(s,θ)+w(s,θ) — the shared field of §3.5.6.
         deformable_wall=True activates the HGO wall; False = rigid base R0(s,θ).
         """
         from lumen.core.frame import CenterlineFrame
         from lumen.newton.hgo_wall import WallField
         f = CenterlineFrame(_np.asarray(centerline))
         dev = self.device
         self._tube_P = _wp.array(f.points.astype(_np.float32), dtype=_wp.vec3, device=dev)
         self._tube_Tg = _wp.array(f.tangents.astype(_np.float32), dtype=_wp.vec3, device=dev)
         self._tube_M1 = _wp.array(f.m1.astype(_np.float32), dtype=_wp.vec3, device=dev)
         self._tube_cum_s = _wp.array(f.cum_s.astype(_np.float32), dtype=_wp.float32, device=dev)
         self._tube_M = len(f.points)
         s_max = float(f.length)
         self._tube_s_max = s_max
         self._tube_ns = int(n_s)
         self._tube_nth = int(n_th)
         self._tube_kappa = float(kappa)
         self._tube_d_hat = float(d_hat)
         self._tube_mode = 1 if barrier_mode == "log" else 0
         self._tube_mu_along = float(mu_along)
         self._tube_mu_across = float(mu_across)
         self._tube_gamma_fric = float(_np.radians(gamma_fric_deg))
         self._tube_deformable = bool(deformable_wall)
         # base radius grid R0(s,θ) on the wall cells (cell = i_s*n_th + i_th)
         if lumen_field is None:
             R0_grid = float(R)
         else:
             ss = _np.linspace(0.0, s_max, n_s)
             th = -_np.pi + (_np.arange(n_th) + 0.5) / n_th * 2.0 * _np.pi
             R0_grid = _np.array([[lumen_field.eval(float(s), float(t)) for t in th]
                                  for s in ss]).ravel()
         self._tube_n_envs = int(n_envs)
         if n_per_env is not None:
             self._tube_n_per_env = int(n_per_env)
         else:
             if n_envs <= 0:
                 raise ValueError("n_envs must be > 0")
             n_wires = len(wire_body_ids)
             if n_wires % n_envs != 0:
                 raise ValueError(
                     f"wire_body_ids length ({n_wires}) must be evenly divisible by n_envs "
                     f"({n_envs}) when computing n_per_env; got n_per_env={n_wires // n_envs} "
                     "(integer division would cause incorrect env indexing)"
                 )
             self._tube_n_per_env = n_wires // n_envs
         self._wall = WallField(R0=R0_grid, s_max=s_max, n_s=n_s, n_th=n_th,
                                params=hgo_params, device=dev, n_envs=n_envs)
         self._tube_wall_load = self._wall.wall_load
         mask = _np.zeros(self.model.body_count, dtype=_np.int32)
         mask[_np.asarray(wire_body_ids, dtype=_np.int32)] = 1
         self._tube_wire_mask = _wp.array(mask, dtype=_wp.int32, device=dev)
         self._tube_enabled = True
 
     def set_coaxial_coupling(self, gw_body_ids, cath_body_ids, r_inner,
-                             kappa=2.0e3, d_hat=0.3, two_way=True, gw_radius=0.0):
+                             kappa=2.0e3, d_hat=0.3, two_way=True, gw_radius=0.0,
+                             n_envs=1, n_gw_per_env=None, n_cath_per_env=None,
+                             n_assembly_per_env=None):
         """Constrain the guidewire to ride inside the catheter's inner lumen (radius
         `r_inner`), reading the catheter's LIVE centerline each AVBD iteration so the gw
         follows the catheter as it bends, sliding freely axially (L0d.2b). `two_way`
         (L0d.2d) deposits the equal-opposite reaction on the catheter (responsive).
 
         `gw_radius` keeps the guidewire SURFACE (centre + radius), not just its centre,
         inside the lumen: the barrier limit is r_inner − gw_radius, and the band d_hat
         is clamped to half that clearance (so it stays a near-wall barrier, not a
         constant central pull)."""
         if len(cath_body_ids) < 2:
             raise ValueError("coaxial coupling needs a catheter centerline of >= 2 nodes")
         dev = self.device
         mask = _np.zeros(self.model.body_count, dtype=_np.int32)
         mask[_np.asarray(gw_body_ids, dtype=_np.int32)] = 1
         self._coax_gw_mask = _wp.array(mask, dtype=_wp.int32, device=dev)
         self._coax_cath_ids = _wp.array(_np.asarray(cath_body_ids, _np.int32),
                                         dtype=_wp.int32, device=dev)
         self._coax_n_cath = len(cath_body_ids)
+        if n_envs <= 0:
+            raise ValueError("n_envs must be > 0 for coaxial coupling")
+        if n_cath_per_env is None:
+            if len(cath_body_ids) % n_envs != 0:
+                raise ValueError("catheter body count must be evenly divisible by n_envs")
+            n_cath_per_env = len(cath_body_ids) // n_envs
+        if n_gw_per_env is None:
+            if len(gw_body_ids) % n_envs != 0:
+                raise ValueError("guidewire body count must be evenly divisible by n_envs")
+            n_gw_per_env = len(gw_body_ids) // n_envs
+        if n_assembly_per_env is None:
+            n_assembly_per_env = int(n_gw_per_env) + int(n_cath_per_env)
+        self._coax_n_cath_per_env = int(n_cath_per_env)
+        self._coax_n_assembly_per_env = int(n_assembly_per_env)
         if float(gw_radius) >= float(r_inner):       # impossible fit -> fail fast, not near-singular
             raise ValueError(f"guidewire radius ({gw_radius}) must be < catheter inner "
                              f"radius ({r_inner}); the guidewire can't fit inside")
         r_eff = float(r_inner) - float(gw_radius)     # gw SURFACE stays inside the lumen
         self._coax_r_inner = r_eff
         self._coax_kappa = float(kappa)
         self._coax_d_hat = min(float(d_hat), 0.5 * r_eff)        # proper near-wall band
         self._coax_two_way = 1 if two_way else 0
         self._coax_enabled = True
     def set_tree_contact(self, tree, wire_body_ids, kappa=2.0e3, d_hat=0.3,
                          barrier_mode="compliant", n_s=40, n_th=16,
                          mu_along=0.0, mu_across=0.0, gamma_fric_deg=40.0,
                          actuation_centerline=None, deformable_wall=False, hgo_params=None,
                          n_envs=1, n_per_env=None):
         """Multi-edge (vascular-tree) contact: each wire node contacts its nearest edge,
         with R branch-blended across junctions (the §3.5.2 work, pre-baked into the grid
         here so the kernel stays simple). `deformable_wall=True` gives each env×edge
         block an HGO wall (w field shared with the barrier, like the single tube,
         §3.5.6), with each edge's OWN arc-length feeding its cell area (correct for
         unequal-length edges). `tree` is a ``lumen.core.VascularTree``.
 
         `actuation_centerline` is the path the kinematic base follows for insertion
         (centerline-following). Pass the full route polyline (trunk→target branch) so the
         base can be pushed PAST the junction into a branch; default = the entry edge (the
         base stops at the apex — only trunk targets reachable)."""
         if getattr(self, "_tube_enabled", False):     # one contact model per solver, never both
             raise RuntimeError("tube contact already set on this solver; cannot also enable tree "
                                "contact (the barriers would double up)")
         dev = self.device
         P, Tg, M1, cum_s = [], [], [], []
         vstart, vcount, smax, sj, ej = [], [], [], [], []
         ss = _np.linspace(0.0, 1.0, n_s)              # fractional s; scaled per edge below
         th = -_np.pi + (_np.arange(n_th) + 0.5) / n_th * 2.0 * _np.pi
         R0_blocks = []
         off = 0
         for i, e in enumerate(tree.edges):
             f = e.frame
             vstart.append(off)
             vcount.append(len(f.points))
             off += len(f.points)
             smax.append(float(f.length))
             sj.append(1 if tree.is_junction(e.node_a) else 0)
             ej.append(1 if tree.is_junction(e.node_b) else 0)
             P.append(f.points); Tg.append(f.tangents); M1.append(f.m1); cum_s.append(f.cum_s)
             # bake the branch-blended R(s,θ) for this edge into its grid block
             block = _np.array([[tree.blended_R(i, float(sf) * f.length, float(t)) for t in th]
                                for sf in ss])
             R0_blocks.append(block.ravel())
         self._tree_P = _wp.array(_np.concatenate(P).astype(_np.float32), dtype=_wp.vec3, device=dev)
         self._tree_Tg = _wp.array(_np.concatenate(Tg).astype(_np.float32), dtype=_wp.vec3, device=dev)
         self._tree_M1 = _wp.array(_np.concatenate(M1).astype(_np.float32), dtype=_wp.vec3, device=dev)
         self._tree_cum_s = _wp.array(_np.concatenate(cum_s).astype(_np.float32), dtype=_wp.float32, device=dev)
         self._tree_vstart = _wp.array(_np.array(vstart, _np.int32), dtype=_wp.int32, device=dev)
         self._tree_vcount = _wp.array(_np.array(vcount, _np.int32), dtype=_wp.int32, device=dev)
         self._tree_smax = _wp.array(_np.array(smax, _np.float32), dtype=_wp.float32, device=dev)
         self._tree_start_junc = _wp.array(_np.array(sj, _np.int32), dtype=_wp.int32, device=dev)
         self._tree_end_junc = _wp.array(_np.array(ej, _np.int32), dtype=_wp.int32, device=dev)
         self._tree_n_edges = len(tree.edges)
         self._tree_n_envs = int(n_envs)
         if n_per_env is not None:
             self._tree_n_per_env = int(n_per_env)
         else:
             if self._tree_n_envs <= 0:
                 raise ValueError("n_envs must be > 0")
             n_wires = len(wire_body_ids)
             if n_wires % self._tree_n_envs != 0:
                 raise ValueError(
                     f"wire_body_ids length ({n_wires}) must be evenly divisible by n_envs "
                     f"({self._tree_n_envs}) when computing n_per_env; got n_per_env={n_wires // self._tree_n_envs} "
                     "(integer division would cause incorrect env indexing in tree barrier kernel)"
                 )
             self._tree_n_per_env = n_wires // self._tree_n_envs
         # one HGO wall over all env×edge blocks. r0_field is the blended base R0; w_field
         # is the shared deformation the barrier reads and the contact load relaxes. rigid
         # (deformable_wall=False) just leaves w≡0. The per-edge blocks are tiled per env.
         from lumen.newton.hgo_wall import WallField
         edge_R0 = _np.concatenate(R0_blocks).astype(_np.float32)
         self._tree_wall = WallField(R0=_np.tile(edge_R0, self._tree_n_envs),
                                     s_max=_np.tile(_np.asarray(smax, float), self._tree_n_envs),
                                     n_s=n_s, n_th=n_th, params=hgo_params,
diff --git a/tests/test_newton_coaxial.py b/tests/test_newton_coaxial.py
index bfe1dcc..1a88d28 100644
--- a/tests/test_newton_coaxial.py
+++ b/tests/test_newton_coaxial.py
@@ -1,128 +1,151 @@
 """L0d.2a — coaxial assemblies: a microcatheter rod alongside the guidewire.
 
 Two rods share the lumen with INDEPENDENT proximal actuation; here they interact
 only through the shared wall contact (the sliding gw-in-catheter coupling is L0d.2b)."""
 
 import numpy as np
 import pytest
 
 pytest.importorskip("warp")
 pytest.importorskip("newton")
 
 from lumen.newton.sim import NewtonGuidewireSim
 
 
 def _vessel(M=40, L=80.0):
     return np.stack([np.zeros(M), np.zeros(M), np.linspace(0, L, M)], axis=1)
 
 
 def _rod(n, z0, x=0.3, sp=2.0):
     return np.stack([np.full(n, x), np.zeros(n), z0 + np.arange(n) * sp], axis=1)
 
 
 def _coaxial(**kw):
     # catheter proximal (z 0..16), guidewire distal (z 18..36) — telescoping tandem
     return NewtonGuidewireSim(_vessel(), 2.0, _rod(10, 18.0), radius=0.2,
                               catheter_points=_rod(9, 0.0), catheter_radius=0.65,
                               vbd_iterations=10, device="cpu", **kw)
 
 
 def test_coaxial_builds_with_two_rods():
     sim = _coaxial()
     assert sim.coaxial
     assert len(sim.bodies) == 9 and len(sim.cath_bodies) == 8     # 10/9 points -> 9/8 bodies
     assert sim.body_positions().shape == (9, 3)
     assert sim.catheter_positions().shape == (8, 3)
     sim.step(dt=1.5e-2, substeps=3)
     assert np.isfinite(sim.body_positions()).all()
     assert np.isfinite(sim.catheter_positions()).all()
 
 
 def test_independent_actuation_of_each_device():
     sim = _coaxial()
     gw0 = sim.body_positions()[-1, 2]      # guidewire tip z
     ct0 = sim.catheter_positions()[-1, 2]  # catheter tip z
     for _ in range(10):                    # advance the CATHETER only
         sim.step(dt=2.5e-2, substeps=5, insertion_cath=2.0)
     assert sim.catheter_positions()[-1, 2] > ct0 + 3.0           # catheter advanced
     assert abs(sim.body_positions()[-1, 2] - gw0) < 2.0          # guidewire ~unchanged
 
     sim2 = _coaxial()
     gw0 = sim2.body_positions()[-1, 2]
     ct0 = sim2.catheter_positions()[-1, 2]
     for _ in range(10):                    # advance the GUIDEWIRE only
         sim2.step(dt=2.5e-2, substeps=5, insertion=2.0)
     assert sim2.body_positions()[-1, 2] > gw0 + 3.0             # guidewire advanced
     assert abs(sim2.catheter_positions()[-1, 2] - ct0) < 2.0    # catheter ~unchanged
 
 
 def test_both_rods_held_in_lumen():
     sim = _coaxial()
     for _ in range(80):                    # press both against the wall
         sim.step(dt=2.5e-2, substeps=5, preload=(100.0, 0.0, 0.0))
     R = 2.0
     assert sim.node_radii().max() <= R + 0.3 + 0.1             # guidewire held
     assert sim.catheter_node_radii().max() <= R + 0.3 + 0.1    # catheter held
 
 
-def test_coaxial_rejects_batched():
-    with pytest.raises(NotImplementedError, match="single-env"):
-        NewtonGuidewireSim(_vessel(), 2.0, _rod(10, 18.0), catheter_points=_rod(9, 0.0),
-                           n_envs=2, device="cpu")
+def test_batched_coaxial_envs_are_independent_under_per_env_actions():
+    sim = NewtonGuidewireSim(_vessel(), 2.0, _rod(10, 18.0), radius=0.2,
+                             catheter_points=_rod(9, 0.0), catheter_radius=0.65,
+                             catheter_inner_radius=0.5, n_envs=2,
+                             vbd_iterations=10, device="cpu")
+    assert sim.coaxial
+    assert sim.n_per_env == 9
+    assert sim.n_cath_per_env == 8
+    assert sim.n_per_env_contact == 17
+    assert sim.bases == [0, 17]
+    assert sim.cath_bases == [9, 26]
+
+    gw0 = sim.env_positions()[:, -1, 2].copy()
+    cath0 = sim.catheter_positions().reshape(2, sim.n_cath_per_env, 3)[:, -1, 2].copy()
+    for _ in range(8):
+        sim.step(dt=2.5e-2, substeps=4,
+                 insertion=np.array([0.0, 1.5], dtype=np.float32),
+                 insertion_cath=np.array([1.25, 0.0], dtype=np.float32))
+
+    gw_tip = sim.env_positions()[:, -1, 2]
+    cath_tip = sim.catheter_positions().reshape(2, sim.n_cath_per_env, 3)[:, -1, 2]
+    assert np.isfinite(sim.env_positions()).all()
+    assert np.isfinite(sim.catheter_positions()).all()
+    assert gw_tip[1] > gw0[1] + 2.0
+    assert abs(gw_tip[0] - gw0[0]) < 1.5
+    assert cath_tip[0] > cath0[0] + 2.0
+    assert abs(cath_tip[1] - cath0[1]) < 1.5
 
 
 def test_coaxial_wires_thrombectomy_flow_clot_and_stentriever():
     from lumen.newton.devices import Stentriever
     from lumen.newton.flow import FlowField, FlowFieldParams
     sim = NewtonGuidewireSim(_vessel(M=60, L=120.0), 2.0, _rod(11, 40.0), radius=0.2,
                              catheter_points=_rod(13, 34.0), catheter_radius=0.65,
                              catheter_inner_radius=0.5,
                              flow=FlowField(FlowFieldParams(P_pulse=0.0)),
                              clot_segment=(55.0, 70.0), clot_height=1.2,
                              stentriever=Stentriever(deployed_center=62.0),
                              device="cpu")
     sim.step(dt=2.5e-2, substeps=2, aspiration=0.4)
     assert sim.clot is not None
     assert sim.clot.o.max() > 0.0
     sim.step(dt=2.5e-2, substeps=2, insertion=-0.5, aspiration=0.4)
     assert getattr(sim, "last_retrieval", {}).get("status") in {"retrieve", "fragment", "miss"}
     assert np.isfinite(sim.body_positions()).all()
     assert np.isfinite(sim.catheter_positions()).all()
     assert sim.flow.pressure_field() is not None
 
 
 def test_no_catheter_is_backward_compatible():
     sim = NewtonGuidewireSim(_vessel(), 2.0, _rod(11, 4.0), device="cpu")
     assert not sim.coaxial and sim.cath_bodies == []
     assert sim.catheter_positions().shape == (0, 3)
     sim.step(dt=1.5e-2, substeps=3)
     assert np.isfinite(sim.body_positions()).all()
 
 
 def test_coaxial_with_deformable_wall():
     # GLM L1 / CodeRabbit #21: coaxial + deformable_wall is allowed (both rods press the
     # same single-env vessel wall) — verify it deflects and stays finite, not rejected.
     from lumen.newton.hgo_wall import HGOParams
     sim = NewtonGuidewireSim(_vessel(), 2.0, _rod(10, 18.0), radius=0.2,
                              catheter_points=_rod(9, 0.0), catheter_radius=0.65,
                              deformable_wall=True,
                              hgo_params=HGOParams(C10=3e3, k1=1.5e3, k2=1.0, thickness=0.3),
                              device="cpu")
     for _ in range(60):
         sim.step(dt=2.5e-2, substeps=5, preload=(120.0, 0.0, 0.0))
     assert np.isfinite(sim.body_positions()).all() and np.isfinite(sim.catheter_positions()).all()
     assert sim.wall_max_deflection() > 1e-4        # both rods deform the shared wall
 
 
 def test_degenerate_catheter_rejected():
     # CodeRabbit #22: a < 2-node catheter centerline can't define a rod -> fail fast up front
     with pytest.raises(ValueError, match=">= 2 nodes"):
         NewtonGuidewireSim(_vessel(), 2.0, _rod(10, 18.0), catheter_points=_rod(1, 0.0), device="cpu")
 
 
 def test_guidewire_too_thick_for_catheter_rejected():
     # CodeRabbit #6: a guidewire that can't fit inside the catheter inner lumen is a hard
     # error, not a near-singular clamp (gw radius >= inner radius).
     with pytest.raises(ValueError, match="must be <"):
         NewtonGuidewireSim(_vessel(), 2.0, _rod(10, 18.0), radius=0.5,
                            catheter_points=_rod(9, 0.0), catheter_inner_radius=0.5, device="cpu")
diff --git a/tests/test_solver_support_docs.py b/tests/test_solver_support_docs.py
index 2022e0a..5a8bc24 100644
--- a/tests/test_solver_support_docs.py
+++ b/tests/test_solver_support_docs.py
@@ -1,75 +1,74 @@
 import ast
 from pathlib import Path
 
 
 ROOT = Path(__file__).resolve().parents[1]
 
 
 def _not_implemented_messages(source: str) -> set[str]:
     tree = ast.parse(source)
     messages = set()
     for node in ast.walk(tree):
         if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
             continue
         func = node.exc.func
         if not isinstance(func, ast.Name) or func.id != "NotImplementedError":
             continue
         if node.exc.args and isinstance(node.exc.args[0], ast.Constant):
             messages.add(str(node.exc.args[0].value))
     return messages
 
 
 def test_solver_support_matrix_tracks_batched_guardrails():
     support = (ROOT / "docs" / "SOLVER_SUPPORT.md").read_text()
     sim = (ROOT / "lumen" / "newton" / "sim.py").read_text()
     not_implemented_messages = _not_implemented_messages(sim)
 
     required_guards = [
-        ("coaxial assemblies are single-env (batched coaxial is future)", "coaxial assemblies are single-env"),
         (
             "batched flow requires the 1-D FlowField; the lumped NewtonFlow is single-env (analytic fallback)",
             "batched flow requires the 1-D FlowField",
         ),
         (
-            "batched stent-retriever retrieval is not ported (per-env host force balance); run retrieval single-env",
-            "batched stent-retriever retrieval is not ported",
+            "batched stent-retriever retrieval requires the 1-D FlowField coupling path",
+            "batched stent-retriever retrieval requires the 1-D FlowField coupling path",
         ),
         (
             "tree contact takes R0 from each edge's lumen field; a sim-level lumen_field doesn't apply",
             "tree contact takes R0 from each edge's lumen field",
         ),
         (
             "edge-aware tree flow/clot coupling is not wired yet: flow drag and clot grids need per-edge graph fields, not a single route centerline",
             "edge-aware tree flow/clot coupling is not wired yet",
         ),
         (
             "an aneurysm needs the 1-D FlowField (it reads the neck pressure P(s)); pass flow=FlowField(...)",
             "an aneurysm needs the 1-D FlowField",
         ),
     ]
     for source_guard, doc_guard in required_guards:
         assert source_guard in not_implemented_messages
         assert doc_guard in support
 
     assert "| 1-D `FlowField` coupling | ✅ | ✅ | none | — |" in support
     assert "| Vascular-tree contact | ✅ | ✅ | none | — |" in support
     assert "| Stent-retriever capture/slip/fragmentation | ✅ | ✅ with `FlowField`/clot coupling |" in support
     for issue_ref in ("53", "55", "56"):
         assert f"| #{issue_ref} |" not in support
         assert f"[#{issue_ref}](https://github.com/SeldingerMed/seldinger-lumen/issues/{issue_ref})" in support
 
     assert "## Follow-up implementation tracker" in support
     for closure_evidence in (
         "two-env coaxial construction/step test",
         "two-env tree contact test",
     ):
         assert closure_evidence in support
 
 
 def test_readme_and_architecture_link_solver_support_matrix():
     readme = (ROOT / "README.md").read_text()
     architecture = (ROOT / "ARCHITECTURE.md").read_text()
 
     link = "docs/SOLVER_SUPPORT.md"
     assert link in readme
     assert link in architecture
