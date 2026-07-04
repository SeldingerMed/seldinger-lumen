# Newton solver support matrix

This matrix is the contract for `lumen.newton.sim.NewtonGuidewireSim`: what works in a single simulation environment, what is vectorized across `n_envs > 1`, and which combinations intentionally fail fast. It tracks the explicit `NotImplementedError` paths in `lumen/newton/sim.py` so users do not discover solver limits only at runtime.

Legend: ✅ supported, ⚠️ supported with stated limits, 🚧 intentionally blocked / follow-up filed. Follow-up links point at the implementation issues that own each remaining batched feature gap.

| Solver path | Single env (`n_envs=1`) | Batched envs (`n_envs>1`) | Runtime guard | Follow-up |
|---|---:|---:|---|---|
| Guidewire + tube wall contact | ✅ | ✅ | none | — |
| Deformable HGO wall | ✅ | ✅ | none | — |
| Anisotropic friction | ✅ | ✅ | none | — |
| 1-D `FlowField` coupling | ✅ | ✅ | none | — |
| Lumped `NewtonFlow` analytic fallback | ✅ | 🚧 | `batched flow requires the 1-D FlowField` | — |
| Finite clot deformation/damage | ✅ | ✅ with `FlowField`/device coupling | none for batched clot alone | — |
| Coaxial guidewire + catheter assembly | ✅ | 🚧 | `coaxial assemblies are single-env` | [#53](https://github.com/SeldingerMed/seldinger-lumen/issues/53) |
| Stent-retriever capture/slip/fragmentation | ✅ | 🚧 | `batched stent-retriever retrieval is not ported` | [#54](https://github.com/SeldingerMed/seldinger-lumen/issues/54) |
| Vascular-tree contact | ✅ | 🚧 | `tree contact is single-env` | [#55](https://github.com/SeldingerMed/seldinger-lumen/issues/55) |
| Tree + sim-level `lumen_field` | 🚧 | 🚧 | `tree contact takes R0 from each edge's lumen field` | [#55](https://github.com/SeldingerMed/seldinger-lumen/issues/55) |
| Tree + flow/clot coupling | 🚧 | 🚧 | `tree + flow/clot is not wired` | [#55](https://github.com/SeldingerMed/seldinger-lumen/issues/55) |
| Aneurysm + flow diverter | ✅ with `FlowField` | ✅ with `FlowField` | none | — |
| Aneurysm without `FlowField` | 🚧 | 🚧 | `an aneurysm needs the 1-D FlowField` | [#56](https://github.com/SeldingerMed/seldinger-lumen/issues/56) |

## Follow-up implementation tracker

| Gap | Implementation issue | Required closure evidence |
|---|---|---|
| Batched coaxial guidewire + catheter assemblies | [#53](https://github.com/SeldingerMed/seldinger-lumen/issues/53) | A two-env coaxial construction/step test with independent guidewire and catheter bases, plus unchanged single-env coaxial coverage. |
| Batched stent-retriever clot retrieval | [#54](https://github.com/SeldingerMed/seldinger-lumen/issues/54) | A two-env retrieval test where capture/slip/fragmentation state diverges per env without host-state bleed-through. |
| Batched vascular-tree contact and tree flow/clot coupling | [#55](https://github.com/SeldingerMed/seldinger-lumen/issues/55) | A two-env tree contact test and either edge-aware flow/clot coverage or an updated guard/doc row for any intentionally remaining sub-gap. |

## Why the remaining gaps exist

### Coaxial batching (#53)

The single-env coaxial path adds one catheter rod, one catheter base, and one set of catheter insertion/twist arrays. Batched support must allocate one catheter assembly per env and preserve the body-to-env mapping for both tube contact and coaxial guidewire-catheter coupling. Until then, `n_envs > 1` would mix bodies across envs, so the constructor fails fast.

### Stent-retriever batching (#54)

Retrieval currently performs capture, slip, fragmentation, and force-balance updates as per-env host logic. Batched retrieval needs independent device or batched host state for each clot/retriever pair so one env's capture event cannot affect another env.

### Tree batching and tree flow/clot (#55)

Tree contact uses per-edge lumen fields and route-centered actuation. Batched support needs a safe env dimension over that edge graph. Flow drag and clot grids are also currently parameterized by one linear centerline, so tree + flow/clot must first become edge-aware rather than reusing the straight/route centerline arrays.

### Aneurysm batching (#56)

Aneurysm flow diversion is now batched when the sim uses the 1-D `FlowField`: each env owns an independent `AneurysmSac`, can use distinct aneurysm/diverter parameters, and reads the corresponding env block from the batched pressure field. The physics limit remains the same as the single-env path: sac→parent back-reaction is not fed into the 1-D parent-flow solve, so the model captures diverter-induced sac stasis but not a neck draw that perturbs parent-vessel through-flow.

## Development rule

When adding a new solver combination, update this file in the same PR as the implementation and add a regression test that covers both the newly supported path and any combinations that remain intentionally blocked. Do not remove a `NotImplementedError` unless the corresponding row moves from 🚧 to ✅/⚠️.
