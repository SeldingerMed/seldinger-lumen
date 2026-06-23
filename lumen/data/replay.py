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

import os
import pathlib
import warnings
from collections import Counter

import numpy as np

from lumen.data.schema import Episode, validate

# dirs to skip when someone points root at a big tree (e.g. the repo). NOT "obs":
# rglob only matches manifest.json, which an episode's obs/ sidecar dir never holds,
# so skipping "obs" only hid a legitimate episode dir literally named "obs".
_SKIP = {".git", "node_modules", ".venv", "venv", "__pycache__"}


class EpisodeDataset:
    """Iterate every episode directory (one holding a `manifest.json`) under `root`.

    Indexable and iterable; each access loads one episode and, unless
    `validate_on_load=False`, validates it against its on-disk sidecars (fail fast in
    a corpus rather than crash deep in training). Observations stay on disk until
    `step.load_obs(ep.root)` is called."""

    def __init__(self, root, validate_on_load: bool = True):
        self.root = str(root)
        self.validate_on_load = validate_on_load
        if not os.path.isdir(self.root):     # a typo'd path silently yielding 0 episodes is a footgun
            warnings.warn(f"EpisodeDataset root {self.root!r} is not a directory; 0 episodes",
                          stacklevel=2)
        # dirs snapshotted at construction; rebuild the dataset to pick up new captures.
        self.dirs = [str(p.parent) for p in sorted(pathlib.Path(self.root).rglob("manifest.json"))
                     if not any(part in _SKIP for part in p.parts)]

    def __len__(self) -> int:
        return len(self.dirs)

    def __getitem__(self, i):
        if isinstance(i, slice):                              # PyTorch-dataset idiom ds[:10]
            return [self[j] for j in range(*i.indices(len(self)))]
        d = self.dirs[i]
        try:                                                 # turn KeyError/JSONDecodeError/validate
            ep = Episode.load(d)                             # failures into a clear, path-tagged error
            if self.validate_on_load:
                validate(ep, root=d)
        except Exception as e:
            raise ValueError(f"episode at {d} failed to load/validate: {e}") from e
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
        # copy the dicts: a consumer mutating them in place must not corrupt the
        # Episode for the next replay (the Step holds them by reference).
        yield s.t, dict(s.action), dict(s.kinematics), obs


def summarize(dataset) -> dict:
    """Corpus-level summary (success rate, mean steps/final_dist, label counts) — the
    leaderboard-shaped dict (no dedicated leaderboard writer exists yet). Accepts an
    EpisodeDataset or any iterable of Episodes.

    Single-pass and manifest-only: the outcome lives in the manifest, so for an
    EpisodeDataset this reads manifests WITHOUT loading sidecars or full validation —
    a 10k-episode corpus costs 10k small JSON reads, not 10k sidecar checks.

    Kind-aware: success_rate / mean_steps / mean_final_dist are computed over
    NAVIGATION episodes only (the metrics are meaningless for wall-probe calibration
    episodes, whose "steps" are views). `kinds` counts each notes["episode_kind"]
    (default "navigation" for episodes that predate the discriminator)."""
    items = ((Episode.load(d) for d in dataset.dirs)
             if isinstance(dataset, EpisodeDataset) else dataset)
    n = nav = succ = steps = dist = 0
    labels: Counter = Counter()
    kinds: Counter = Counter()
    for ep in items:                                          # one streaming pass, no list()
        n += 1
        labels[ep.outcome.label] += 1
        kind = ep.meta.notes.get("episode_kind", "navigation") if isinstance(ep.meta.notes, dict) else "navigation"
        kinds[kind] += 1
        if kind == "navigation":                             # nav metrics over nav episodes only
            nav += 1
            succ += bool(ep.outcome.success)
            steps += ep.outcome.steps
            dist += ep.outcome.final_dist
    base = {"episodes": n, "navigation": nav, "labels": dict(labels), "kinds": dict(kinds)}
    if nav == 0:
        return {**base, "success_rate": 0.0, "mean_steps": 0.0, "mean_final_dist": 0.0}
    return {**base, "success_rate": succ / nav, "mean_steps": steps / nav,
            "mean_final_dist": dist / nav}


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
