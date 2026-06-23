"""L1.3 — image-based navigation: the policy observes the FLUORO, not privileged state.

Closes the perception-in-the-real-modality loop (doc §3.6, §6.1): each step renders
the env's device to a synthetic fluoro (Layer 1), detects the device tip in the image
(§4.2 front-end), and feeds image-space features to the policy. Trained with the same
batched CEM as the state-obs policy — env e renders candidate e — so it rides the
fast tier's parallelism. The reward still uses the true arc-length (a privileged
training signal); only the OBSERVATION is image-derived, as at deployment.

Obs (4-D, all from the image / known goal):
  [tip_u/nu, tip_v/nv, (target_v − tip_v)/nv, tip_present]
The third feature is the image-space "remaining" — the progress signal to warm-start.

Honest scope: this proves the image-observation PIPELINE is correct and carries the
navigation signal (test_image_obs_signal pins that obs[2] runs from large at the inlet
to ~0 at the target). On this *easy, monotonic* insertion task the privileged reward
also solves it (forward always reaches the target; the env freezes on a hit), so it
does NOT prove vision is strictly necessary — that needs a task requiring a choice or a
precise stop (branched anatomy / no auto-freeze), which a small CEM can't crack
quickly. Left for a harder L1.x task.

ponytail: per-env render is a Python loop over K (one DRR per candidate per step) —
fine at reference scale; batch/vectorise the raycast when perception is the bottleneck.
"""

from __future__ import annotations

import numpy as np

from lumen.rl.cem import BatchedNav
from lumen.sensors.perception import detect_device_tip


class FluoroBatchedNav(BatchedNav):
    """BatchedNav whose observation is read from a rendered fluoro (one C-arm, shared
    across envs). True arc-length is still used for the reward; the policy only sees
    the image features."""

    def __init__(self, vessel, R, K, sensor, carm=None, view_axis=(1.0, 0.0, 0.0), **kw):
        super().__init__(vessel, R, K, **kw)
        self.sensor = sensor
        # H1: size the C-arm to the whole VESSEL (the reachable extent), not the seed
        # device — else the target and the inserted tip project off the detector and
        # the image observation carries no signal.
        self.carm = carm or sensor.default_carm(np.asarray(self.frame.points), axis=view_axis)
        # target pixel = projection of the centerline point at the target arc-length
        j = int(np.argmin(np.abs(self.frame.cum_s - self.target_s)))
        self.target_uv = self.carm.project(self.frame.points[j])
        # M1: which image extreme is the inserted tip depends on the vessel's projected
        # direction, not a hardcoded max-v. Compare the start vs target detector-v.
        start_v = self.carm.project(self.frame.points[0])[1]
        self.leading = "max_v" if self.target_uv[1] >= start_v else "min_v"
        self.obs_dim = 4

    def _tip_obs(self):
        pos = self.sim.env_positions()                          # (K, n, 3)
        s = np.empty(self.K); r = np.empty(self.K)
        obs = np.zeros((self.K, 4), np.float32)
        tv = self.target_uv[1]
        nu, nv = self.carm.nu, self.carm.nv                     # M3: normalize by the IMAGE size
        for e in range(self.K):
            pr = self.frame.project(pos[e, -1])                 # true tip (reward only)
            s[e], r[e] = pr.s, pr.r
            img, _ = self.sensor.render(pos[e], carm=self.carm)  # this env's fluoro
            u, v, present = detect_device_tip(img, leading=self.leading)
            progress = (tv - v) / nv if present else 0.0        # neutral when undetected
            obs[e] = [u / nu, v / nv, progress, present]
        return obs, s, r


def fluoro_env_factory(sensor, view_axis=(1.0, 0.0, 0.0), target_frac=0.7,
                       max_insertion=2.0, device="cpu"):
    """Build a `train_cem` env_factory that yields image-observation envs."""
    def factory(vessel, R, lumen_field, pop):
        return FluoroBatchedNav(vessel, R, pop, sensor, view_axis=view_axis,
                                target_frac=target_frac, max_insertion=max_insertion,
                                lumen_field=lumen_field, device=device)
    return factory
