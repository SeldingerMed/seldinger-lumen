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
| A1 | Mask to navigable asset, exact shapes | masks rasterised from formulas (tube, flipped axes, Y fork, a broken tube) | the formulas; gate: half a voxel of the coarsest spacing | passing (`tests/test_scan_import.py`) |
| A2 | Mask to navigable asset, real anatomy | an open aortic CTA (AVT D10) with its supplied vessel mask | the mask's own surface (wall error report); a push-only wire reaches a branch target | run by hand on one case with `lumen import --mask` |
| A3 | Real scan, neck to pelvis | a scan, no mask, no flag naming a model | a wire reaches a target chosen by vessel name | run by hand on two cases. AVT D10 (NRRD, arch to iliacs): a wire that is only pushed reached 4 of the 6 named targets within 500 steps (aorta, brachiocephalic trunk, both iliacs) and stopped 13 mm short of the left common carotid and left subclavian. A head-and-neck CTA DICOM folder (both models): 13 of 16 (not the anterior communicating artery, right anterior cerebral artery or right infraclinoid carotid) |
| A4 | Real scan, head | a DICOM folder, no mask, no flag naming a model | a wire reaches a target chosen by vessel name | import run by hand on one head-only CTA (head model chosen without a flag; both carotids, middle and anterior cerebral arteries named). A wire that is only pushed reached 4 of the 9 named targets within 500 steps. A scripted wire that pulls back and turns its shaped tip when it stalls reached 6 of 9 within 600 steps (not the two middle cerebral arteries or the right anterior cerebral artery). No trained policy has been tried |
| A5 | Ask before install | a machine with no segmenter installed | the prompt names model, size and licence; "no" installs nothing; `--yes` is silent | passing (`tests/test_scan_pipeline.py`), and run by hand in a terminal with a fresh install folder |
| A6 | Viewer | an imported case and a generated case | both replay smoothly in a browser; look signed off | not started |
| A7 | Bundled open cases | a fresh install | `lumen anatomy` lists them and each one runs | not started |

Not yet tried on any scan: MRA, an abdomen-only CTA, a scan from a scanner vendor other than the
open sets above.

## Steps

### 1. Load your own scan

- [x] Lift the surface-to-tree converter into the library (`lumen/assets/surface_import.py`)
- [x] Mask to closed surface in patient millimetres (`lumen/assets/scan_import.py`)
- [x] Default entry at the caudal end of the vessel
- [x] Scan readers: DICOM series (CT and MR, classic and enhanced), NIfTI, NRRD (`lumen/scan/io.py`)
- [x] Body region from the picture itself, header text only as a fallback (`lumen/scan/region.py`)
- [x] Segmenter registry with ask-then-install into a private environment (`lumen/scan/segmenters.py`)
  - [x] TotalSegmentator (neck to pelvis), from `SeldingerMed/TotalSegmentator`
  - [x] RSNA 2025 intracranial model (head), from `SeldingerMed/RSNA2025_Intracranial-Aneurysm-Detection`
- [x] Walls sharpened against the scan; missed stretches filled where contrast crosses the gap (`lumen/scan/refine.py`)
- [x] Vessel names on branches; targets addressable by name (`lumen/scan/pipeline.py`)
- [x] `lumen import` command and report
- [ ] Preview image written next to the report
- [x] User guide page (`docs/import.md`)
- [ ] A steering policy run to every named target on the imported cases
- [ ] The wire can leave the vessel on imported anatomy. Pushed only, the deepest wall overlap
      was 0.9 to 6.6 mm across the six targets of the aortic case, up to 7.4 mm on the
      head-and-neck case and 13.7 mm on the head case; on the generated bifurcation and
      tortuous tree the same wire stays under 1 mm. Checked against the scan's surface itself
      on the head case: the tip was up to 15 mm outside it, at 2, 1 and 0.5 mm of push per
      step. Two runs were traced step by step. Pushed only towards the right middle cerebral
      artery, the wire was on the route inside one branch when its middle bowed out through
      that branch's wall at a bend, a little further each step as the base kept advancing.
      Under the scripted wire it began after the tip took a side branch off the route. In
      both the base keeps moving along the route whatever the wall pushes back with, so the
      wall gives way; that reading of the cause is not confirmed. Fix before training on
      imported anatomy
- [ ] The environment lets the 18 mm device be withdrawn to the entry point, where it bunches
      up and buckles through the wall; withdrawal should stop at the entry

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
- 2026-11-25. Body region is read from the picture first (outline size and bone, slice by slice).
  Header text is only a fallback: it is often missing or wrong.
- 2026-11-25. Both models install from Seldinger forks at a pinned commit. The TotalSegmentator
  fork turns usage statistics off by default; the RSNA fork makes the published stage 1 model
  loadable and loads weights as data only. RSNA weight files are checked against pinned hashes.
- 2026-11-25. Models draw vessels a little fat and on a coarse grid, so each named vessel is
  redrawn at the half-way brightness between its lumen and the surrounding tissue, within 2 mm
  of the model's outline.
- 2026-11-25. A stretch a model missed is filled only where the scan shows contrast across the
  gap: up to 30 mm between two pieces of the same named vessel, up to 15 mm otherwise. Pieces
  within 2 mm of the tree are joined; anything else that does not connect is left out and its
  volume reported.
- 2026-11-25. A scan needing both models asks about both before any work starts, then installs
  in the background while the first model runs.

