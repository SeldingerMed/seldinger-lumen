"""GPU validation: run the Warp narrowphase on CUDA, batched, and report.

Self-reports the selected backend, verifies CUDA-vs-CPU parity of the kernel, and
measures batched throughput (the doc's >=1e4 env-steps/s target, §3.1). Intended
to be run on a CUDA box (e.g. a RunPod pod):  python -m benchmarks.warp_gpu_check
"""

from __future__ import annotations

import time

import numpy as np

import warp as wp

from lumen.physics.warp_contact import WarpTubeContact
from lumen.physics import backend


def _scene(B, N, M=60, L=120.0, R=2.0, seed=0):
    cl = np.stack([np.zeros(M), np.zeros(M), np.linspace(0, L, M)], axis=1)
    rng = np.random.default_rng(seed)
    x = np.stack([np.full(N, 1.0), np.zeros(N), np.linspace(4, L - 4, N)], axis=1)
    x = np.broadcast_to(x, (B, N, 3)).copy() + 0.4 * rng.standard_normal((B, N, 3))
    return cl, x.astype(np.float32), R


def main():
    print("backend:", backend.describe())
    has_cuda = wp.get_cuda_device_count() > 0
    devices = ["cpu"] + (["cuda"] if has_cuda else [])

    cl, x, R = _scene(B=8, N=16)
    # CPU-vs-CUDA parity
    if has_cuda:
        g_cpu, f_cpu = WarpTubeContact(cl, R, device="cpu").forces(x)
        g_gpu, f_gpu = WarpTubeContact(cl, R, device="cuda").forces(x)
        print(f"cpu-vs-cuda parity: gap {np.abs(g_cpu-g_gpu).max():.2e}  "
              f"force {np.abs(f_cpu-f_gpu).max():.2e}")

    for dev in devices:
        for B in (1024, 8192, 65536):
            cl, x, R = _scene(B=B, N=16)
            wc = WarpTubeContact(cl, R, device=dev)
            wc.forces(x)                                    # warmup / compile
            t0 = time.perf_counter()
            iters = 20
            for _ in range(iters):
                wc.forces(x)
            dt = (time.perf_counter() - t0) / iters
            node_evals = B * 16
            print(f"[{dev:4s}] B={B:6d}  {dt*1e3:8.3f} ms/step  "
                  f"{node_evals/dt/1e6:8.1f} M node-evals/s  "
                  f"({B/dt/1e3:.1f}k env-steps/s)")


if __name__ == "__main__":
    main()
