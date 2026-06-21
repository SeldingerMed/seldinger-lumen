"""Backend selection: favour Warp/Newton, fall back to PyTorch (hardware-gated).

Priority (doc intent -- build on Warp/Newton; the PyTorch path is a portable
fallback, not the target):

    1. warp-cuda   Warp kernels on a CUDA GPU      (production throughput)
    2. warp-cpu    Warp kernels on the CPU device  (no-GPU dev / CI of kernels)
    3. torch       PyTorch reference solver         (always available fallback)

`select_backend()` reports what is available; callers pick the contact
implementation accordingly. The Warp and torch narrowphases are kept in parity
(tests/test_warp_parity.py) so results are backend-independent.
"""

from __future__ import annotations


def warp_status():
    try:
        import warp as wp
        wp.init()
        return True, wp.get_cuda_device_count() > 0
    except Exception:
        return False, False


def torch_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def select_backend(prefer="auto") -> str:
    """Return 'warp-cuda' | 'warp-cpu' | 'torch'. prefer='torch' forces fallback."""
    if prefer == "torch":
        return "torch"
    has_warp, warp_cuda = warp_status()
    if has_warp and warp_cuda:
        return "warp-cuda"
    if prefer == "warp" or has_warp:
        return "warp-cpu" if has_warp else "torch"
    return "torch"


def describe() -> dict:
    has_warp, warp_cuda = warp_status()
    return {"selected": select_backend(), "warp_available": has_warp,
            "warp_cuda": warp_cuda, "torch_cuda": torch_cuda()}


if __name__ == "__main__":
    import json
    print(json.dumps(describe(), indent=2))
