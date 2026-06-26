"""Gymnasium registration for the lumen navigation envs (doc M5: Gym integration).

The envs (`NavEnv`, `TreeNavEnv`) follow the Gym reset/step convention WITHOUT a hard
gymnasium dependency, so policies can train on Layer 0 with or without gymnasium. This
module is the optional bridge: when gymnasium IS installed, `register_gym_envs()` puts
the canonical scenes in its registry so external groups can `gymnasium.make("Lumen/...")`
— the standard way a benchmark is consumed and submitted to.

The same scene factories back the benchmark suite (`lumen.bench`), so a `gymnasium.make`
env and a benchmark task are the identical scene.
"""

from __future__ import annotations


def make_nav_tube(**kw):
    """Straight-tube navigation (the easy tier)."""
    from lumen.assets import procedural
    from lumen.envs.nav_gym import NavEnv
    return NavEnv(asset=procedural.straight_tube(length=80.0, radius=2.0), **kw)


def make_nav_stenotic(severity=0.5, **kw):
    """Navigation past a mid-vessel narrowing (the medium tier)."""
    from lumen.assets import procedural
    from lumen.envs.nav_gym import NavEnv
    return NavEnv(asset=procedural.stenotic_tube(length=80.0, radius=2.0, severity=severity), **kw)


def make_tree_nav(target_node="left_out", angle_deg=25.0, **kw):
    """Branch navigation on a bifurcation tree (the hard tier)."""
    from lumen.assets import procedural
    from lumen.envs.tree_nav import TreeNavEnv
    asset = procedural.bifurcation(trunk=50.0, branch=50.0, radius=2.0, angle_deg=angle_deg)
    return TreeNavEnv(asset, target_node=target_node, **kw)


# id -> factory. The benchmark suite (lumen.bench) reuses these so a make() env and a
# bench task are the same scene.
LUMEN_ENVS = {
    "Lumen/NavTube-v0": make_nav_tube,
    "Lumen/NavStenotic-v0": make_nav_stenotic,
    "Lumen/NavTreeBranch-v0": make_tree_nav,
}


def _gym_entry_point(lumen_factory):
    """Wrap a raw lumen-env factory as a gymnasium.Env subclass instance — gymnasium.make
    requires a gymnasium.Env, but the lumen envs deliberately don't subclass it (no hard
    dependency). The wrapper just delegates reset/step and surfaces the spaces + R."""
    import gymnasium
    import numpy as np
    from gymnasium import spaces

    class LumenGymEnv(gymnasium.Env):
        metadata = {"render_modes": []}

        def __init__(self, **kw):
            self._env = lumen_factory(**kw)
            self.action_space = getattr(self._env, "action_space",
                                        spaces.Box(-1.0, 1.0, (1,), np.float32))
            self.observation_space = getattr(self._env, "observation_space",
                                             spaces.Box(-np.inf, np.inf, (5,), np.float32))

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return self._env.reset(seed=seed, options=options)

        def step(self, action):
            return self._env.step(action)

        @property
        def R(self):                            # the bench/eval reads env.R for the safety metric
            return self._env.R

    return lambda **kw: LumenGymEnv(**kw)


def register_gym_envs() -> list[str]:
    """Register the lumen envs with gymnasium (idempotent). Returns the registered ids.

    Raises ImportError only if gymnasium is absent (callers guard); an already-registered
    id is skipped, so calling this twice is safe."""
    import gymnasium

    registered = []
    for env_id, factory in LUMEN_ENVS.items():
        if env_id not in gymnasium.registry:
            gymnasium.register(id=env_id, entry_point=_gym_entry_point(factory))
        registered.append(env_id)
    return registered
