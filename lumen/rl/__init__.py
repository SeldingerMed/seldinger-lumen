"""RL / policy search over the Layer-0 sim (doc M5): close the learning loop.

A gradient-free CEM trainer that evaluates a population of policies in one batched
rollout (env e = candidate e), riding the fast tier's parallelism. Pure numpy.
"""

from lumen.rl.cem import BatchedNav, make_policy, train_cem
from lumen.rl.fluoro_nav import FluoroBatchedNav, fluoro_env_factory
from lumen.rl.adapters import make_cleanrl_env, make_cleanrl_vector_env, make_sb3_env

__all__ = [
    "train_cem",
    "make_policy",
    "BatchedNav",
    "FluoroBatchedNav",
    "fluoro_env_factory",
    "make_sb3_env",
    "make_cleanrl_env",
    "make_cleanrl_vector_env",
]
