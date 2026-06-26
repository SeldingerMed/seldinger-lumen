"""Accurate tier (doc §3.3): a self-contained, penetration-free IPC reference and a
differentiable calibration path, used to cross-validate and calibrate the fast tier.
Not on the RL hot path — reference quality, numpy/autodiff."""

from lumen.accurate.ipc import IPCParams, IPCTubeReference, ipc_barrier
from lumen.accurate.stochastic import (contact_reaction, deterministic_grad,
                                       recover_by_smoothed_descent, smoothed_value_and_grad)

__all__ = ["IPCTubeReference", "IPCParams", "ipc_barrier",
           "calibrate_hgo", "hgo_pressure_curve",
           "contact_reaction", "smoothed_value_and_grad", "deterministic_grad",
           "recover_by_smoothed_descent"]


def __getattr__(name):                # lazy: the differentiable path needs warp
    if name in ("calibrate_hgo", "hgo_pressure_curve"):
        from lumen.accurate import diff
        return getattr(diff, name)
    raise AttributeError(name)
