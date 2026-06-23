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
        seed_dev = self.sim.body_positions()[:self.sim.n_per_env]
        self.carm = carm or sensor.default_carm(seed_dev, axis=view_axis)
        # target pixel = projection of the centerline point at the target arc-length
        j = int(np.argmin(np.abs(self.frame.cum_s - self.target_s)))
        self.target_uv = self.carm.project(self.frame.points[j])
        self.obs_dim = 4

    def _tip_obs(self):
        pos = self.sim.env_positions()                          # (K, n, 3)
        s = np.empty(self.K); r = np.empty(self.K)
        obs = np.zeros((self.K, 4), np.float32)
        tu, tv = self.target_uv
        for e in range(self.K):
            pr = self.frame.project(pos[e, -1])                 # true tip (reward only)
            s[e], r[e] = pr.s, pr.r
            img, _ = self.sensor.render(pos[e], carm=self.carm)  # this env's fluoro
            u, v, present = detect_device_tip(img)
            obs[e] = [u / self.sensor.nu, v / self.sensor.nv,
                      (tv - v) / self.sensor.nv, present]
        return obs, s, r


def fluoro_env_factory(sensor, view_axis=(1.0, 0.0, 0.0), target_frac=0.7,
                       max_insertion=2.0, device="cpu"):
    """Build a `train_cem` env_factory that yields image-observation envs."""
    def factory(vessel, R, lumen_field, pop):
        return FluoroBatchedNav(vessel, R, pop, sensor, view_axis=view_axis,
                                target_frac=target_frac, max_insertion=max_insertion,
                                lumen_field=lumen_field, device=device)
    return factory
