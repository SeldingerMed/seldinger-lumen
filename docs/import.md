# Importing a real scan

`lumen import` turns a CTA or MRA into anatomy the simulator can navigate. Generated anatomy
keeps working exactly as before; this adds real anatomy next to it.

```bash
pip install "seldinger-lumen[scan]"
lumen import /path/to/dicom-folder --out my-case
```

You can give a folder of DICOM files, a NIfTI file (`.nii`, `.nii.gz`) or an NRRD file. Everything
runs on your machine. The scan is never uploaded.

## What happens

1. **Read.** Lumen reads the scan in patient millimetres. In a DICOM folder with several series
   it takes the one with the most slices (`--list-series` shows them, `--series` picks another).
2. **Work out the body region.** Lumen looks at the picture itself, slice by slice, to decide
   whether it shows the head, the neck or trunk, or both. You do not have to say.
3. **Find and name the arteries.** Two open models do this, one per region:

   | The scan shows | Model | Names it gives |
   |---|---|---|
   | neck, chest, abdomen, pelvis | TotalSegmentator | aorta, brachiocephalic trunk, subclavian, common and internal carotid, iliac arteries |
   | head | RSNA 2025 intracranial model | internal carotid (infra- and supraclinoid), middle and anterior cerebral, anterior and posterior communicating, basilar tip, other posterior circulation |

   A scan that shows both regions gets both models and one joined tree.
4. **Sharpen.** Each named vessel is redrawn at the half-way brightness between its lumen and
   the tissue around it, so the wall follows the contrast in your scan and not the model's
   coarser grid.
5. **Build the tree.** The vessel surface becomes a branching tree with a fitted wall, which is
   what the solver uses.

## The first time: Lumen asks before installing a model

Lumen does not ship either model. The first time a scan needs one, Lumen tells you what it is,
how large it is and its licence, and asks:

```
This scan shows the head. To find the arteries of the circle of Willis and their feeders,
Lumen needs the RSNA 2025 intracranial artery model
(Apache-2.0, about 2.5 GB with PyTorch), installed once into /Users/you/.lumen.
It comes from https://github.com/SeldingerMed/RSNA2025_Intracranial-Aneurysm-Detection.
It runs on this machine; the scan is not sent anywhere.
Install it now? [y/N]
```

- **Yes** installs it once, quietly, into `~/.lumen` in its own Python environment, so its
  PyTorch never touches yours. Later imports do not ask again.
- **No** installs nothing and stops. You can still import with your own mask (below).
- `--yes` answers yes in advance, for scripts.
- `LUMEN_HOME` moves the install folder. `LUMEN_WEIGHTS` moves just the model weights.

Both models are installed from Seldinger's forks at a fixed commit, and the head model's weight
files are checked against known hashes before use.

## Using your own mask

If you already have the vessels segmented, no model is needed:

```bash
lumen import --mask vessels.nii.gz --out my-case
```

A mask with several label values and names (for example a 3D Slicer `.seg.nrrd`) keeps its names.

## What you get

```
my-case/
  asset.json    the anatomy the solver navigates
  surface.stl   the vessel surface in patient millimetres
  names.json    vessel names and the node to aim for in each
  report.json   region, models used, wall error, every segment's length and radius
```

`lumen import` prints the named vessels you can aim for. To train or run on the case:

```python
from lumen.scan.pipeline import make_env

env = make_env("my-case", target="left middle cerebral artery")
obs, _ = env.reset()
```

`target` takes a vessel name (any unique part of it, any case) or a node id from `asset.json`.

## Where the device enters

By default the device enters at the lowest end of the vessel tree: the groin end of an
aorto-iliac scan, the bottom of a neck or head scan. `--entry x,y,z` sets another point, in
patient millimetres (LPS).

## What is not supported

- Angiography and other projection images. Only cross-sectional scans.
- Scans with gantry tilt or unevenly spaced slices. Lumen stops and says so.
- Vessels the models do not name (renal and visceral branches, vertebral arteries, external
  carotids) are not found from a scan. Supply a mask to include them.
- Imported anatomy is for simulation and training. It is not a clinical measurement.
