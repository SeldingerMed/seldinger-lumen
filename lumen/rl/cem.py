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

THETA_DIM = 6                                  # default: 5 obs weights + 1 bias (state obs)


def linear_action(obs, theta):
    """Linear policy over a d-dim obs: action = clip(W·obs + b). theta is (K, d+1)
    (d weights + bias); d is inferred from obs so this works for state OR image obs."""
    d = obs.shape[1]
    return np.clip(np.sum(theta[:, :d] * obs, axis=1) + theta[:, d], -1.0, 1.0)


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
        self.obs_dim = 5                                         # [s/L, r/R, sinθ, cosθ, remaining]

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
            # finite guard immediately after _tip_obs, before dist/rew/prev;
            # mark diverged done to stop NaN propagation into returns and elite selection
            finite = np.isfinite(obs).all(axis=1) & np.isfinite(s) & np.isfinite(r)
            if not finite.all():
                obs = np.nan_to_num(obs)
                s = np.nan_to_num(s)
                r = np.nan_to_num(r)
            done = done | (~finite)
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
            if done.all():
                break
        return ret, succ, steps


def train_cem(vessel=None, R=None, lumen_field=None, anatomies=None, pop=64,
              elite_frac=0.25, iters=25, target_frac=0.7, max_insertion=2.0,
              seed=0, device="cpu", log=None, env_factory=None, warm_start=(4, 2.0)):
    """CEM over the linear policy; population evaluated in one batched rollout/iter.

    `anatomies` (list of (vessel, R, lumen_field)) trains with domain randomisation —
    each candidate is scored by its MEAN return across all anatomies, so the policy
    must generalise (a single straight-tube fit overfits and fails a stenosis). If
    omitted, trains on the single (vessel, R, lumen_field).

    `env_factory(vessel, R, lumen_field, pop) -> env` builds the rollout env (default
    state-obs BatchedNav; pass a fluoro-obs builder for image-based control). The
    policy dimension is taken from env.obs_dim. `warm_start=(idx, val)` seeds the
    progress feature. Returns (best_theta, history)."""
    if pop <= 0:
        raise ValueError(f"pop must be greater than 0, got {pop}")
    if iters <= 0:
        raise ValueError(f"iters must be greater than 0, got {iters}")
    if not (0 < elite_frac < 1):
        raise ValueError(f"elite_frac must be between 0 and 1 (exclusive), got {elite_frac}")
    if anatomies is not None:
        if not isinstance(anatomies, (list, tuple)) or len(anatomies) == 0:
            raise ValueError("anatomies must be either None or a non-empty list of (vessel, R, lumen_field) tuples")
    rng = np.random.default_rng(seed)
    if anatomies is None:
        anatomies = [(vessel, R, lumen_field)]
    if env_factory is None:
        def env_factory(v, r, lf, pop):
            return BatchedNav(v, r, pop, target_frac=target_frac,
                              max_insertion=max_insertion, lumen_field=lf, device=device)
    envs = [env_factory(v, r, lf, pop) for (v, r, lf) in anatomies]
    theta_dim = envs[0].obs_dim + 1                             # obs weights + bias
    mu = np.zeros(theta_dim, np.float32)
    wi, wv = warm_start
    if wi is not None and wi < theta_dim:
        mu[wi] = wv                                             # warm-start the progress feature
    sigma = np.full(theta_dim, 1.5, np.float32)
    n_elite = max(2, int(elite_frac * pop))
    best_theta, best_ret, hist = mu.copy(), -1e9, []
    for it in range(iters):
        theta = (mu[None, :] + sigma[None, :] * rng.standard_normal((pop, theta_dim))
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
