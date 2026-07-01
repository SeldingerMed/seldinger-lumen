# Newton solver support matrix

This matrix is the contract for `lumen.newton.sim.NewtonGuidewireSim`: what works in a single simulation environment, what is vectorized across `n_envs > 1`, and which combinations intentionally fail fast. It tracks the explicit `NotImplementedError` paths in `lumen/newton/sim.py` so users do not discover solver limits only at runtime.

Legend: ✅ supported, ⚠️ supported with stated limits, 🚧 intentionally blocked / follow-up filed.

| Solver path | Single env (`n_envs=1`) | Batched envs (`n_envs>1`) | Runtime guard | Follow-up |
|---|---:|---:|---|---|
| Guidewire + tube wall contact | ✅ | ✅ | none | — |
| Deformable HGO wall | ✅ | ✅ | none | — |
| Anisotropic friction | ✅ | ✅ | none | — |
| 1-D `FlowField` coupling | ✅ | ✅ | none | — |
| Lumped `NewtonFlow` analytic fallback | ✅ | 🚧 | `batched flow requires the 1-D FlowField` | — |
| Finite clot deformation/damage | ✅ | ✅ with `FlowField`/device coupling | none for batched clot alone | — |
| Coaxial guidewire + catheter assembly | ✅ | 🚧 | `coaxial assemblies are single-env` | #53 |
| Stent-retriever capture/slip/fragmentation | ✅ | 🚧 | `batched stent-retriever retrieval is not ported` | #54 |
| Vascular-tree contact | ✅ | 🚧 | `tree contact is single-env` | #55 |
| Tree + sim-level `lumen_field` | 🚧 | 🚧 | `tree contact takes R0 from each edge's lumen field` | #55 |
| Tree + flow/clot coupling | 🚧 | 🚧 | `tree + flow/clot is not wired` | #55 |
| Aneurysm + flow diverter | ✅ with `FlowField` | 🚧 | `aneurysm flow diversion is single-env` | #56 |
| Aneurysm without `FlowField` | 🚧 | 🚧 | `an aneurysm needs the 1-D FlowField` | #56 |

## Why the remaining gaps exist

### Coaxial batching (#53)

The single-env coaxial path adds one catheter rod, one catheter base, and one set of catheter insertion/twist arrays. Batched support must allocate one catheter assembly per env and preserve the body-to-env mapping for both tube contact and coaxial guidewire-catheter coupling. Until then, `n_envs > 1` would mix bodies across envs, so the constructor fails fast.

### Stent-retriever batching (#54)

Retrieval currently performs capture, slip, fragmentation, and force-balance updates as per-env host logic. Batched retrieval needs independent device or batched host state for each clot/retriever pair so one env's capture event cannot affect another env.

### Tree batching and tree flow/clot (#55)

Tree contact uses per-edge lumen fields and route-centered actuation. Batched support needs a safe env dimension over that edge graph. Flow drag and clot grids are also currently parameterized by one linear centerline, so tree + flow/clot must first become edge-aware rather than reusing the straight/route centerline arrays.

### Aneurysm batching (#56)

Aneurysm flow diversion uses the 1-D `FlowField` neck pressure to update one sac state on the host. Batched support must store sac state per env and read the corresponding batched pressure/flow-diverter state before it can be used for RL-throughput rollouts.

## Development rule

When adding a new solver combination, update this file in the same PR as the implementation and add a regression test that covers both the newly supported path and any combinations that remain intentionally blocked. Do not remove a `NotImplementedError` unless the corresponding row moves from 🚧 to ✅/⚠️.
