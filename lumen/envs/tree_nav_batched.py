"""Batched tree navigation: E independent `TreeNavEnv` worlds stepped together in one Newton model.

Same scene, observation, action, reward, success and safety rules as `TreeNavEnv`, applied per world.
The worlds share the model but not their state: each has its own guidewire rod, actuation and
episode counter, and `reset_env(e)` restores one world without touching the others (what an RL
vector env needs for auto-reset). The wall is rigid (the `TreeNavEnv` scene), so a world's state is
its rod: body poses, velocities and the solver's previous poses.
"""

from __future__ import annotations

import numpy as np

from lumen.envs.tree_nav import TreeNavEnv


class BatchedTreeNavEnv(TreeNavEnv):
    """`TreeNavEnv` with `n_envs` worlds; observations (E, 5), actions (E, 2)."""

    def __init__(self, asset, n_envs: int = 8, **kw):
        if isinstance(n_envs, bool) or int(n_envs) != n_envs or n_envs < 1:
            raise ValueError(f"n_envs must be a positive integer, got {n_envs!r}")
        self.n_envs = int(n_envs)
        super().__init__(asset, **kw)

    # -- construction / reset -------------------------------------------------------------------
    def _build_sim(self):
        from lumen.newton.sim import NewtonGuidewireSim
        pts = np.asarray(self.route_frame.points)
        self.sim = NewtonGuidewireSim(pts, self.R, self._device_points(), radius=self.device_radius_mm,
                                      kappa=3e3, d_hat=0.3, vbd_iterations=8, device=self.device,
                                      tree=self.tree, route_centerline=pts, n_envs=self.n_envs)
        n = self.sim.n_per_env
        self._ids = [np.asarray(self.sim.bodies[e * n:(e + 1) * n], dtype=np.int64) for e in range(self.n_envs)]

    def reset(self, *, seed=None, options=None):
        if getattr(self, "sim", None) is None:
            self._build_sim()
        else:
            self.sim.reset()
        E = self.n_envs
        self.steps = np.zeros(E, dtype=np.int64)
        self._impulse = np.zeros(E, dtype=float)
        feats = [self._features_from(p) for p in self.sim.env_positions()]
        self._prev = np.array([abs(f["s"] - self.target_s) for f in feats])
        return np.stack([self._obs(f) for f in feats]), {}

    def reset_env(self, e: int):
        """Restore world `e` to its initial state; the other worlds are untouched."""
        import warp as wp
        ids, dev, sim = self._ids[e], self.sim.device, self.sim
        init = sim._init_body_q
        for holder, name in ((sim.state_0, "body_q"), (sim.state_1, "body_q"), (sim.solver, "body_q_prev")):
            q = getattr(holder, name).numpy()
            q[ids] = init[ids]
            setattr(holder, name, wp.array(q, dtype=wp.transform, device=dev))
        for st in (sim.state_0, sim.state_1):
            qd = st.body_qd.numpy()
            qd[ids] = 0.0
            st.body_qd = wp.array(qd, dtype=st.body_qd.dtype, device=dev)
        self.steps[e] = 0
        self._impulse[e] = 0.0
        f = self._features_from(sim.env_positions()[e])
        self._prev[e] = abs(f["s"] - self.target_s)
        return self._obs(f)

    # -- stepping -------------------------------------------------------------------------------
    def step(self, action):
        actions = np.asarray(action, dtype=np.float32).reshape(self.n_envs, -1)
        parsed = [self._parse_action(a) for a in actions]
        ins = np.array([a for a, _ in parsed], dtype=np.float32) * self.max_insertion
        tw = np.array([t for _, t in parsed], dtype=np.float32) * self.max_twist
        self.sim.step(dt=5e-3 * self.substeps, substeps=self.substeps, insertion=ins, twist=tw)
        self.steps += 1
        imp = np.broadcast_to(np.asarray(self.sim.wall_load_impulse(), dtype=float), (self.n_envs,))
        self._impulse += imp
        E = self.n_envs
        obs = np.zeros((E, 5), np.float32)
        rew = np.zeros(E, np.float32)
        term = np.zeros(E, bool)
        trunc = self.steps >= self.max_steps
        infos = []
        for e, pos in enumerate(self.sim.env_positions()):
            f = self._features_from(pos)
            o = self._obs(f)
            if not (np.isfinite(o).all() and np.isfinite([f["s"], f["r"], f["max_r"]]).all()):
                rew[e], term[e] = -100.0, True
                infos.append({"route_s": 0.0, "dist": 1e6, "max_pen": 0.0, "success": False,
                              "safe_success": False, "unsafe": False, "diverged": True,
                              "wall_load_impulse": float(self._impulse[e]), "wall_load_units": "sim_units"})
                continue
            dist = abs(f["s"] - self.target_s)
            r = (self._prev[e] - dist if f["on_route"] else -0.1) - 0.5 * f["max_pen"] - 0.01
            self._prev[e] = dist
            unsafe = bool(f["max_pen"] > self.safety_max_pen)
            success = bool(dist <= self.success_tol and f["on_route"])
            if success:
                r += 10.0
            if unsafe:
                r -= 10.0
            obs[e], rew[e] = o, r
            term[e] = success or (unsafe and self.terminate_on_unsafe)
            infos.append({"route_s": f["s"], "dist": dist, "max_r": f["max_r"], "max_pen": f["max_pen"],
                          "success": success, "safe_success": bool(success and not unsafe), "unsafe": unsafe,
                          "edge": f["edge"], "off_route": not f["on_route"],
                          "wall_load_impulse": float(self._impulse[e]), "wall_load_units": "sim_units"})
        return obs, rew, term, trunc & ~term, infos
