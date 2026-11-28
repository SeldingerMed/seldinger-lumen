# Importing a real scan

`lumen import` turns a CTA or MRA into anatomy the simulator can navigate. Generated anatomy
keeps working exactly as before; this adds real anatomy next to it.

```bash
pip install -e ".[scan]"    # from your clone of this repository
lumen import /path/to/dicom-folder --out my-case
```

You can give a folder of DICOM files, a NIfTI file (`.nii`, `.nii.gz`) or an NRRD file. Everything
runs on your machine. The scan is never uploaded.

## What happens

1. **Read.** Lumen reads the scan in patient millimetres. In a DICOM folder with several series
   it takes the one with the most slices (`--list-series` shows them, `--series` picks another).
   A NIfTI or NRRD CT stored without negative numbers (1024 added) is recognised from its
   values and read in Hounsfield units; `--modality CT` or `--modality MR` settles it when a
   file is read the wrong way.
2. **Work out the body region.** Lumen looks at the picture itself, slice by slice, to decide
   whether it shows the head, the neck or trunk, or both. You do not have to say.
3. **Find and name the arteries.** Two open models do this, one per region:

   | The scan shows | Model | Names it gives |
   |---|---|---|
   | neck, chest, abdomen, pelvis | TotalSegmentator | aorta, brachiocephalic trunk, subclavian, common and internal carotid, iliac arteries |
   | head | RSNA 2025 intracranial model | internal carotid (infra- and supraclinoid), middle and anterior cerebral, anterior and posterior communicating, basilar tip, other posterior circulation |

   A scan that shows both regions gets both models and one joined tree.
4. **Sharpen and mend.** Each named vessel is redrawn at the half-way brightness between its
   lumen and the tissue around it, so the wall follows the contrast in your scan and not the
   model's coarser grid. Where a model missed a short stretch of a vessel and the scan shows
   contrast running across the gap, the stretch is filled in.
5. **Build the tree.** The vessel surface becomes a branching tree with a fitted wall, which is
   what the solver uses. The wall is measured every half millimetre along each vessel by rays
   cast round its axis. A ray that grazes a gap in the surface reads a radius the sections either
   side of it do not share; such a reading, more than 0.3 mm from the median of its ray over
   2 mm, is replaced by that median, unless the sections next to it lean the same way (the
   tapering minimum of a stenosis is kept). `report.json` lists every stretch that was filled in
   or joined, and in `conversion.rays_despiked_frac` the share of all wall readings replaced
   this way (readings already filled in at a branch opening are not counted again).

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
- **No** installs nothing. You can still import with your own mask (below).
- A scan that shows both the head and the neck needs both models. Lumen asks about both
  before it starts, then installs in the background: the second model installs while the
  first is already finding arteries. Saying no to one still imports the part the other covers.
- `--yes` answers yes in advance, for scripts.
- `LUMEN_HOME` moves the install folder. `LUMEN_WEIGHTS` moves just the model weights.
  The install folder must be on a normal Mac or Linux disk. An exFAT, FAT or NTFS drive
  cannot hold a Python environment (Lumen says so before asking); the weights can go there.

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
The wire is the one an operator would use on the route: a stiff 0.035 inch wire when every
vessel on it is at least 6 mm across (aorta, iliacs, the arch vessels), a microwire anywhere
narrower. `wire="micro"` or `wire="0.035"` chooses it yourself.

## Where the device enters

By default the device enters at the lowest end of the vessel tree: the groin end of an
aorto-iliac scan, the bottom of a neck or head scan. A head scan usually cuts both carotids
(and often the vertebrals) at the bottom of the picture, and each cut end is a way in, as the
artery an operator would bring the wire up: a run aimed at a named vessel starts from the way
in whose route to it is shortest and runs through the vessel itself, so the right middle
cerebral artery is reached up the right carotid. `names.json` records the ways in (`access`)
and each vessel's start (`start_node`).

`--entry x,y,z` sets the entry point, in patient millimetres (LPS). It is then the only way in.

## What is not supported

- Angiography and other projection images. Only cross-sectional scans.
- Scans with gantry tilt or unevenly spaced slices. Lumen stops and says so.
- Where the head model marks an aneurysm, the sac is kept as part of the vessel and listed as
  a target named `aneurysm`. The model can mark one that is not there, and the tree treats
  the sac as a short side branch, not as a rounded pouch.
- Vessels the models do not name (renal and visceral branches, vertebral arteries, external
  carotids) are not found from a scan. Supply a mask to include them.
- The anatomy is one connected tree. Arteries that do not join it inside the scan are left
  out; in a head-only scan that is usually the vertebrobasilar system when no posterior
  communicating artery is seen. `report.json` gives the volume left out.
- Imported anatomy is for simulation and training. It is not a clinical measurement.
