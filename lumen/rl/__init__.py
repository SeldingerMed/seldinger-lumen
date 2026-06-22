"""RL / policy search over the Layer-0 sim (doc M5): close the learning loop.

A gradient-free CEM trainer that evaluates a population of policies in one batched
rollout (env e = candidate e), riding the fast tier's parallelism. Pure numpy.
"""

from lumen.rl.cem import BatchedNav, make_policy, train_cem

__all__ = ["train_cem", "make_policy", "BatchedNav"]
