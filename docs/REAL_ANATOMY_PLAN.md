# Real anatomy and viewer: build checklist

Working plan for three additions to Lumen. Lumen stays a simulation and RL training environment
and generated anatomy stays the default.

1. Users load their own CTA or MRA (`lumen import`).
2. A 3D viewer with several views, a C-arm view and replay.
3. A few open real cases shipped with Lumen.

Out of scope: angiography import, 2D/3D fusion, bulk dataset tooling.

## Acceptance tests

| # | Test | Starts from | Compared against | Status |
|---|---|---|---|---|
| A1 | Mask to navigable asset, exact shapes | masks rasterised from formulas (tube, flipped axes, Y fork) | the formulas; gate: half a voxel of the coarsest spacing | passing (`tests/test_scan_import.py`) |
| A2 | Mask to navigable asset, real anatomy | an open aortic CTA with its supplied vessel mask | the mask's own surface (wall error report); a scripted wire reaches a branch target | first case run by hand; not yet a command |
| A3 | Real DICOM, chest or abdomen | a DICOM folder, no mask, no flag naming a model | a wire reaches a target chosen by vessel name | not started |
| A4 | Real DICOM, head | a DICOM folder, no mask, no flag naming a model | a wire reaches a target chosen by vessel name | not started |
| A5 | Ask before install | a machine with no segmenter installed | the prompt names model, size and licence; "no" installs nothing; `--yes` is silent | not started |
| A6 | Viewer | an imported case and a generated case | both replay smoothly in a browser; look signed off | not started |
| A7 | Bundled open cases | a fresh install | `lumen anatomy` lists them and each one runs | not started |

## Steps

### 1. Load your own scan

- [x] Lift the surface-to-tree converter into the library (`lumen/assets/surface_import.py`)
- [x] Mask to closed surface in patient millimetres (`lumen/assets/scan_import.py`)
- [x] Default entry at the caudal end of the vessel
- [ ] Scan readers: DICOM series (CT and MR, classic and enhanced), NIfTI, NRRD
- [ ] Body region from the scan itself (tags, then physical extent)
- [ ] Segmenter registry with ask-then-install into a private environment
  - [ ] TotalSegmentator (neck to pelvis), from `SeldingerMed/TotalSegmentator`
  - [ ] RSNA 2025 intracranial model (head), from `SeldingerMed/RSNA2025_Intracranial-Aneurysm-Detection`
- [ ] Vessel names on branches; targets addressable by name
- [ ] `lumen import` command, report and preview image
- [ ] User guide page

### 2. Viewer

- [ ] Three style stills for sign-off
- [ ] `lumen view`: 3D scene, free rotation, preset views, named branches
- [ ] C-arm view (LAO/RAO, cranial/caudal)
- [ ] Replay of saved runs with a scrub bar

### 3. Bundled open cases

- [ ] Pick four or five open cases (aortic arch with iliacs, carotid, intracranial)
- [ ] Relax the firewall check to allow credited open cases
- [ ] List them in `lumen anatomy` with dataset credit

## Decisions log

- 2026-11-25. Coordinates are DICOM patient LPS millimetres, written as frame `patient_lps_mm`.
  NIfTI input is converted on read.
- 2026-11-25. The mask is smoothed per axis by 0.6 of that axis's spacing. One isotropic value
  sized for 3 mm slices shrank iliac radii by about 0.5 mm on the first real case.
- 2026-11-25. Only the largest face-connected piece of a mask is kept, decided on the resampled
  grid the converter uses. Deciding it on the raw mask with corner connectivity kept a stub the
  converter could not reach.
- 2026-11-25. Segmenters run in Lumen's own private environment as a subprocess, so their
  PyTorch and nnU-Net pins never touch the user's environment and the core stays NumPy only.
