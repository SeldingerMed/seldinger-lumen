"""L2.2 — replay & dataset iteration over a corpus of episodes (doc §5).

The consumer end of the capture seam: iterate a directory of `lumen-episode/0`
episodes with lazy observation loading (so a large corpus doesn't have to fit in
memory), and replay one episode step-by-step. This is the handoff surface Layer 3
(world model / policy) trains on — and it works the same whether the episodes were
produced by the open synthetic recorder (L2.1) or a private patient pipeline,
because both write the same schema.

Each yielded `Episode` gets a runtime `.root` attribute (its directory) so
`step.load_obs(ep.root)` works without threading the path through by hand.
"""

from __future__ import annotations

import pathlib
from collections import Counter

import numpy as np

from lumen.data.schema import Episode, validate

_SKIP = {".git", "node_modules", ".venv", "venv", "__pycache__", "obs"}


class EpisodeDataset:
    """Iterate every episode directory (one holding a `manifest.json`) under `root`.

    Indexable and iterable; each access loads one episode and, unless
    `validate_on_load=False`, validates it against its on-disk sidecars (fail fast in
    a corpus rather than crash deep in training). Observations stay on disk until
    `step.load_obs(ep.root)` is called."""

    def __init__(self, root, validate_on_load: bool = True):
        self.root = str(root)
        self.validate_on_load = validate_on_load
        self.dirs = [str(p.parent) for p in sorted(pathlib.Path(self.root).rglob("manifest.json"))
                     if not any(part in _SKIP for part in p.parts)]

    def __len__(self) -> int:
        return len(self.dirs)

    def __getitem__(self, i: int) -> Episode:
        d = self.dirs[i]
        ep = Episode.load(d)
        if self.validate_on_load:
            validate(ep, root=d)
        ep.root = d                      # runtime handle for lazy obs (not serialized)
        return ep

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def replay(episode: Episode, root: str | None = None):
    """Yield `(t, action, kinematics, obs)` per step. `obs` is lazily loaded from the
    sidecar (None for a "none"-modality step, or if no root is known). `root` defaults
    to the episode's runtime `.root` (set by EpisodeDataset)."""
    root = root or getattr(episode, "root", None)
    for s in episode.steps:
        obs = s.load_obs(root) if (root and s.obs_ref) else None
        yield s.t, s.action, s.kinematics, obs


def summarize(dataset) -> dict:
    """Corpus-level summary (success rate, mean steps/final_dist, label counts) — the
    leaderboard-shaped dict (no dedicated leaderboard writer exists yet). Accepts an
    EpisodeDataset or any iterable of Episodes."""
    eps = list(dataset)
    n = len(eps)
    if n == 0:
        return {"episodes": 0, "success_rate": 0.0, "mean_steps": 0.0,
                "mean_final_dist": 0.0, "labels": {}}
    return {"episodes": n,
            "success_rate": float(np.mean([e.outcome.success for e in eps])),
            "mean_steps": float(np.mean([e.outcome.steps for e in eps])),
            "mean_final_dist": float(np.mean([e.outcome.final_dist for e in eps])),
            "labels": dict(Counter(e.outcome.label for e in eps))}


if __name__ == "__main__":  # self-check (pure numpy — no Newton needed)
    import tempfile

    from lumen.data.schema import EpisodeMeta, Outcome, Step

    def _ep(label, n, success):
        return Episode(meta=EpisodeMeta(asset_ref=f"{label}.json"),
                       steps=[Step(t=i * 0.1, action={"insertion": 1.0},
                                   kinematics={"tip_mm": [0.0, 0.0, float(i)], "tip_s": float(i)},
                                   obs_modality="fluoro", obs_ref=f"{i:03d}.npy",
                                   obs=np.full((3, 3), float(i)))
                             for i in range(n)],
                       outcome=Outcome(success=success, final_dist=0.4 if success else 9.0,
                                       steps=n, label=label))

    with tempfile.TemporaryDirectory() as root:
        _ep("straight", 3, True).save(f"{root}/a")
        _ep("stenosis", 5, False).save(f"{root}/b")
        ds = EpisodeDataset(root)
        assert len(ds) == 2
        steps = list(replay(ds[0]))                       # ds[0] has .root attached
        assert len(steps) == ds[0].outcome.steps
        t, action, kin, obs = steps[2]
        assert obs.shape == (3, 3) and np.array_equal(obs, np.full((3, 3), 2.0))   # lazy obs ok
        s = summarize(ds)
        assert s["episodes"] == 2 and 0.0 < s["success_rate"] < 1.0
        assert set(s["labels"]) == {"straight", "stenosis"}
    print("replay self-check ok")
