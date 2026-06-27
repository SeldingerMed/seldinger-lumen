"""Hardware detection for the Newton/Warp backend.

Layer 0 runs on the Newton engine (Warp), which executes on BOTH the CPU
(Warp's LLVM backend) and CUDA GPUs. There is no separate CPU engine: the same
solver code runs everywhere, and this module just picks the device.

    detect_device()  -> "cuda" if a CUDA GPU is visible to Warp, else "cpu"
    describe()       -> full hardware/software report

``NewtonGuidewireSim`` calls ``detect_device()`` for its default device.
"""

from __future__ import annotations

VALIDATED_WARP_VERSION = "1.14.0"
VALIDATED_NEWTON_VERSION = "1.4.0.dev0"
VALIDATED_NEWTON_REF = "6dfe7303d9ca50f7505cac31bee9885c813d89d7"


def detect_device(prefer: str = "auto") -> str:
    """Return the Warp device to run on: 'cuda' (if available) or 'cpu'.

    prefer='cpu' forces CPU; prefer='cuda' still falls back to CPU if no GPU.
    """
    if prefer == "cpu":
        return "cpu"
    try:
        import warp as wp
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
    try:
        import warp as wp
        wp.init()
        info["warp"] = wp.config.version
        info["cuda_devices"] = wp.get_cuda_device_count()
    except Exception:
        pass
    try:
        import newton
        info["newton"] = newton.__version__
        info["newton_available"] = True
    except Exception:
        pass
    info["backend_validated"] = (
        info["warp"] == VALIDATED_WARP_VERSION
        and info["newton"] == VALIDATED_NEWTON_VERSION
    )
    return info


if __name__ == "__main__":
    import json
    print(json.dumps(describe(), indent=2))
