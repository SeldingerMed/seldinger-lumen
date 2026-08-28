"""Optional adapters for Gymnasium-based SB3 and CleanRL training loops."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path


def _make_env(env_id: str, **kwargs):
    import gymnasium

    from lumen.envs.registration import register_gym_envs

    register_gym_envs()
    return gymnasium.make(env_id, **kwargs)


def _seed_spaces(env, seed: int) -> None:
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)


def make_sb3_env(
    env_id: str = "Lumen/NavTube-v0",
    *,
    seed: int | None = 0,
    **kwargs,
):
    """Build a registered Lumen env wrapped for Stable-Baselines3.

    ``stable-baselines3`` remains optional. The returned ``Monitor`` receives the
    initial seed, while later resets remain under the caller/model's control.
    """
    try:
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "make_sb3_env requires stable-baselines3; install it separately"
        ) from exc

    env = Monitor(_make_env(env_id, **kwargs))
    if seed is not None:
        _seed_spaces(env, seed)
        env.reset(seed=seed)
    return env


def make_cleanrl_env(
    env_id: str = "Lumen/NavTube-v0",
    *,
    seed: int = 0,
    idx: int = 0,
    capture_video: bool = False,
    video_dir: str | Path = "videos",
    **kwargs,
) -> Callable[[], object]:
    """Return the environment thunk expected by CleanRL's vector setup.

    The thunk wraps episode statistics, seeds both spaces with ``seed + idx``, and
    leaves reset ownership to ``SyncVectorEnv``. Only worker zero records video;
    requesting video from a headless environment raises ``ValueError``.
    """
    def thunk():
        import gymnasium

        env = _make_env(env_id, **kwargs)
        if capture_video and idx == 0:
            if not getattr(env, "render_mode", None):
                env.close()
                raise ValueError(
                    "capture_video requires an environment render_mode; "
                    "Lumen environments are headless"
                )
            env = gymnasium.wrappers.RecordVideo(env, str(video_dir))
        env = gymnasium.wrappers.RecordEpisodeStatistics(env)
        _seed_spaces(env, seed + idx)
        return env

    return thunk


def make_cleanrl_vector_env(
    env_id: str = "Lumen/NavTube-v0",
    *,
    num_envs: int = 1,
    seed: int = 0,
    capture_video: bool = False,
    video_dir: str | Path = "videos",
    **kwargs,
):
    """Build CleanRL's standard synchronous vector environment."""
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
        raise ValueError(f"num_envs must be a positive integer, got {num_envs!r}")

    import gymnasium

    return gymnasium.vector.SyncVectorEnv([
        make_cleanrl_env(
            env_id,
            seed=seed,
            idx=idx,
            capture_video=capture_video,
            video_dir=video_dir,
            **kwargs,
        )
        for idx in range(num_envs)
    ])


__all__ = ["make_sb3_env", "make_cleanrl_env", "make_cleanrl_vector_env"]
