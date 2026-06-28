"""Hardware detection for the Newton/Warp backend.

Layer 0 runs on the Newton engine (Warp), which executes on BOTH the CPU
(Warp's LLVM backend) and CUDA GPUs. There is no separate CPU engine: the same
solver code runs everywhere, and this module just picks the device.

    detect_device()  -> "cuda" if a CUDA GPU is visible to Warp, else "cpu"
    describe()       -> full hardware/software report

``NewtonGuidewireSim`` calls ``detect_device()`` for its default device.
"""

from __future__ import annotations

import json
import os
from importlib import metadata

VALIDATED_WARP_VERSION = "1.14.0"
VALIDATED_NEWTON_VERSION = "1.4.0.dev0"
VALIDATED_NEWTON_REF = "6dfe7303d9ca50f7505cac31bee9885c813d89d7"
BACKEND_LOG_ENV = "LUMEN_BACKEND_LOG_LEVEL"


def _newton_install_ref(newton_module) -> str | None:
    """Best-effort installed Newton VCS ref; None means the exact ref is unknown."""
    for attr in ("__commit__", "__git_commit__", "__git_revision__"):
        ref = getattr(newton_module, attr, None)
        if ref:
            return str(ref)
    try:
        dist = metadata.distribution("newton")
        direct_url = dist.read_text("direct_url.json")
        if not direct_url:
            return None
        data = json.loads(direct_url)
        vcs_info = data.get("vcs_info") or {}
        return vcs_info.get("commit_id") or vcs_info.get("requested_revision")
    except Exception:
        return None


def configure_backend_logging(level: str | None = None) -> None:
    """Set Warp's default log level before backend initialization.

    Lumen examples and hardware probes should print Lumen results, not Warp's module
    load chatter. Set ``LUMEN_BACKEND_LOG_LEVEL=info`` or ``debug`` to opt back into
    verbose backend diagnostics.
    """
    level = (level if level is not None else os.environ.get(BACKEND_LOG_ENV, "warning")).lower()
    try:
        import warp as wp
    except Exception:
        return
    levels = {
        "debug": wp.LOG_DEBUG,
        "info": wp.LOG_INFO,
        "warning": wp.LOG_WARNING,
        "warn": wp.LOG_WARNING,
        "error": wp.LOG_ERROR,
    }
    if level not in levels:
        raise ValueError(f"{BACKEND_LOG_ENV} must be one of {sorted(levels)}, got {level!r}")
    wp.config.log_level = levels[level]


def detect_device(prefer: str = "auto") -> str:
    """Return the Warp device to run on: 'cuda' (if available) or 'cpu'.

    prefer='cpu' forces CPU; prefer='cuda' still falls back to CPU if no GPU.
    """
    if prefer == "cpu":
        return "cpu"
    try:
        import warp as wp
    except Exception:
        return "cpu"
    configure_backend_logging()
    try:
        wp.init()
        return "cuda" if wp.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        return "cpu"


def describe() -> dict:
    """Report the hardware/software the Layer-0 stack will use."""
    info = {"device": detect_device(), "warp": None, "cuda_devices": 0,
            "newton": None, "newton_available": False,
            "validated": {"warp": VALIDATED_WARP_VERSION,
                          "newton": VALIDATED_NEWTON_VERSION,
                          "newton_ref": VALIDATED_NEWTON_REF},
            "backend_validated": False}
    configure_backend_logging()
    try:
        import warp as wp
        wp.init()
        info["warp"] = wp.config.version
        info["cuda_devices"] = wp.get_cuda_device_count()
    except Exception:
        pass
    newton_ref = None
    try:
        import newton
        info["newton"] = newton.__version__
        newton_ref = _newton_install_ref(newton)
        info["newton_ref"] = newton_ref
        info["newton_available"] = True
    except Exception:
        pass
    info["backend_validated"] = (
        info["warp"] == VALIDATED_WARP_VERSION
        and info["newton"] == VALIDATED_NEWTON_VERSION
        and newton_ref == VALIDATED_NEWTON_REF
    )
    return info


if __name__ == "__main__":
    import json
    print(json.dumps(describe(), indent=2))
