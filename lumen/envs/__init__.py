"""Task environments over the lumen solver (doc M5: Isaac Lab / Gym integration).

A navigation env wraps the differentiable solver as a standard reset/step RL
environment on procedural anatomy. It follows the Gym/Gymnasium convention
(returns obs, reward, terminated, truncated, info) but does not hard-depend on
gymnasium -- so policies can train on Layer 0 alone before any real data exists
(doc §2, "policies can train on Layer 0 alone"). Spaces are exposed if gymnasium
is installed.
"""

from lumen.envs.nav_gym import NavEnv
from lumen.envs.tree_nav import TreeNavEnv

__all__ = ["NavEnv", "TreeNavEnv"]
