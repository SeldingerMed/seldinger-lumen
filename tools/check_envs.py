"""Run Gymnasium's API checker over every registered Lumen environment."""

from __future__ import annotations

import gymnasium
from gymnasium.utils.env_checker import check_env

from lumen.envs.registration import LUMEN_ENVS, register_gym_envs


def main() -> None:
    register_gym_envs()
    for env_id in LUMEN_ENVS:
        env = gymnasium.make(env_id)
        try:
            check_env(env.unwrapped, skip_render_check=True)
        except Exception as exc:
            raise RuntimeError(f"Gymnasium env checker failed for {env_id}") from exc
        finally:
            env.close()
        print(f"env checker ok: {env_id}")


if __name__ == "__main__":
    main()
