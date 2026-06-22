"""Close the learning loop (doc M5): train a navigation policy on the Layer-0 sim.

Uses the **batched** sim from the GPU-throughput track: a CEM (cross-entropy method)
population of K candidate policies is evaluated in ONE batched rollout — env e runs
candidate e — so policy search rides the same parallelism that makes the fast tier
fast. Gradient-free, pure numpy (no torch): right-sized for a small policy and a
natural fit for the non-differentiable fast tier.

Policy: linear, action = clip(W·obs + b, -1, 1) over the 5-D NavEnv obs. It can
represent the proportional baseline (W=[0,0,0,0,4]) and improve on it by using the
tip-radius / position features to back off before contact.
"""

from __future__ import annotations

import numpy as np

from lumen.core.frame import CenterlineFrame

THETA_DIM = 6                                  # 5 weights + 1 bias (linear policy)


def linear_action(obs, theta):
    """obs (K,5), theta (K,6) -> action (K,) in [-1,1]."""
    return np.clip(np.sum(theta[:, :5] * obs, axis=1) + theta[:, 5], -1.0, 1.0)


class BatchedNav:
    """Vectorized NavEnv-equivalent rollout over K parallel guidewires sharing one
    vessel (one per CEM candidate). Mirrors NavEnv's obs and reward; the tip radius
    is used as the contact proxy (cheap) — the trained policy is then scored on the
    real NavEnv leaderboard, which uses the true max-node contact."""

    def __init__(self, vessel, R, K, target_frac=0.7, max_insertion=2.0,
                 substeps=4, lumen_field=None, device="cpu"):
        from lumen.newton.sim import NewtonGuidewireSim
        self.frame = CenterlineFrame(vessel)
        self.L = float(self.frame.length)
        self.R, self.K = float(R), int(K)
        self.target_s = target_frac * self.L
        self.max_insertion, self.substeps = max_insertion, substeps
        p0, t0, m1 = self.frame.points[0], self.frame.tangents[0], self.frame.m1[0]
        dev = (p0 + 0.5 * R * m1)[None, :] + np.arange(10)[:, None] * 2.0 * t0[None, :]
        self.sim = NewtonGuidewireSim(vessel, R, dev, radius=0.2, kappa=3e3, d_hat=0.3,
                                      lumen_field=lumen_field, n_envs=K, vbd_iterations=8,
                                      device=device)

    def _tip_obs(self):
        tips = self.sim.env_positions()[:, -1, :]               # (K,3)
        s = np.empty(self.K); r = np.empty(self.K); th = np.empty(self.K)
        for e in range(self.K):
            pr = self.frame.project(tips[e]); s[e], r[e], th[e] = pr.s, pr.r, pr.theta
        obs = np.stack([s / self.L, r / self.R, np.sin(th), np.cos(th),
                        (self.target_s - s) / self.L], axis=1).astype(np.float32)
        return obs, s, r

    def rollout(self, theta, max_steps=40, success_tol=2.5):
        """Return (returns[K], success[K], steps[K]) for the K candidate policies."""
        self.sim.reset()
        obs, s, r = self._tip_obs()
        prev = np.abs(s - self.target_s)
        ret = np.zeros(self.K); done = np.zeros(self.K, bool); steps = np.zeros(self.K)
        succ = np.zeros(self.K, bool)
        for _ in range(max_steps):
            a = linear_action(obs, theta) * self.max_insertion
            a = np.where(done, 0.0, a)                          # freeze finished envs
            self.sim.step(dt=5e-3 * self.substeps, substeps=self.substeps,
                          insertion=a.astype(np.float32))
            obs, s, r = self._tip_obs()
            dist = np.abs(s - self.target_s)
            contact_pen = np.maximum(0.0, r - self.R)
            rew = (prev - dist) - 0.5 * contact_pen - 0.01
            hit = (dist < success_tol) & (~done)
            rew = np.where(hit, rew + 10.0, rew)
            ret = np.where(done, ret, ret + rew)
            steps = np.where(done, steps, steps + 1)
            succ = succ | hit
            done = done | (dist < success_tol)
            prev = dist
            if not np.isfinite(obs).all():                      # guard a diverged candidate
                obs = np.nan_to_num(obs)
            if done.all():
                break
        return ret, succ, steps


def train_cem(vessel=None, R=None, lumen_field=None, anatomies=None, pop=64,
              elite_frac=0.25, iters=25, target_frac=0.7, max_insertion=2.0,
              seed=0, device="cpu", log=None):
    """CEM over the linear policy; population evaluated in one batched rollout/iter.

    `anatomies` (list of (vessel, R, lumen_field)) trains with domain randomisation —
    each candidate is scored by its MEAN return across all anatomies, so the policy
    must generalise (a single straight-tube fit overfits and fails a stenosis). If
    omitted, trains on the single (vessel, R, lumen_field). Returns (best_theta, history)."""
    rng = np.random.default_rng(seed)
    if anatomies is None:
        anatomies = [(vessel, R, lumen_field)]
    envs = [BatchedNav(v, r, pop, target_frac=target_frac, max_insertion=max_insertion,
                       lumen_field=lf, device=device) for (v, r, lf) in anatomies]
    mu = np.zeros(THETA_DIM, np.float32)
    mu[4] = 2.0                                                 # warm-start near "push toward target"
    sigma = np.full(THETA_DIM, 1.5, np.float32)
    n_elite = max(2, int(elite_frac * pop))
    best_theta, best_ret, hist = mu.copy(), -1e9, []
    for it in range(iters):
        theta = (mu[None, :] + sigma[None, :] * rng.standard_normal((pop, THETA_DIM))
                 ).astype(np.float32)
        theta[0] = mu                                          # keep the current mean in the pop
        rets, succs = [], []
        for env in envs:
            r_e, s_e, _ = env.rollout(theta)
            rets.append(r_e); succs.append(s_e)
        ret = np.mean(rets, axis=0)                            # generalise: mean across anatomies
        succ = np.mean(succs, axis=0)
        order = np.argsort(ret)[::-1]
        elite = theta[order[:n_elite]]
        mu = elite.mean(axis=0)
        sigma = elite.std(axis=0) + 1e-2                       # floor to avoid collapse
        if ret[order[0]] > best_ret:
            best_ret, best_theta = float(ret[order[0]]), theta[order[0]].copy()
        rec = {"iter": it, "mean_return": float(ret.mean()),
               "best_return": float(ret.max()), "success_rate": float(succ.mean())}
        hist.append(rec)
        if log:
            log(rec)
    return best_theta, hist


def make_policy(theta):
    """Wrap trained params as a NavEnv-compatible policy(obs)->action(1,)."""
    th = np.asarray(theta, np.float32)[None, :]
    def policy(obs):
        return np.array([linear_action(np.asarray(obs, np.float32)[None, :], th)[0]],
                        dtype=np.float32)
    return policy
