# Register a vessel tree from one projection

Choose the [full-tree control](#case-1-full-tree-control) or [two selective exposures](#case-2-selective-exposures). Both accept the same pose-file format; use the case ID from the blind manifest.

## Case 1: full-tree control

Open [the blind image](full-tree-v1/problem/image.png). The task is to place the supplied 3D centerlines so their projection follows the dark branches, including the small forks near the top. You can inspect the PNG without installing Lumen. To give someone the problem blind, copy **only** `full-tree-v1/problem/`; keep `answer/` until they submit their pose. The answer is public, not a hidden benchmark.

## What a reader needs

`problem.json` (`lumen-registration/1`) identifies the case, image, geometry, procedural provenance and camera. `geometry.json` lists ordered branches. Each `id` is unique; `parent` is null at the inlet or the ID of the branch whose last point starts this one. `points_mm` are ordered XYZ triples in a right-handed **model** frame. `radius_mm` is the proximal drawing radius, not an estimated measurement; the renderer tapers each branch to 60% at its tip. `territory` is a synthetic group label, not a clinical territory.

Submit UTF-8 JSON with exactly `schema` (`lumen-rigid-pose/1`), `case_id` (from the manifest) and `model_to_camera_mm` (four arrays of four numbers). Multiply the row-major matrix by the **column** `[x,y,z,1]` about the model origin. The upper left block must be a proper rotation (absolute tolerance 3e-6 on its Gram matrix and determinant, allowing six-decimal export), the translation column is in mm, and the last row is `[0,0,0,1]`. Do not invert it or rotate about the centroid.

The camera source is the origin: +X right, +Y down, +Z toward the detector. Pixel `(0,0)` is the top-left pixel **center**. For camera point `(X,Y,Z)` with `0 < Z <= source_to_detector_mm`, project to column `cx_px + fx_px*X/Z`, row `cy_px + fy_px*Y/Z`. No distortion. The generator uses Lumen's `CArm.project` and flips its native row before downsampling. Overlapping projected segments add darkness; this drawing is not a calibrated attenuation volume.

## Try the score, then reveal

From the repository root, this deliberately rough estimate uses no answer data. It places the model halfway between source and detector with no rotation. Python's standard library is enough:

```sh
PACK=lumen/registration_pack
CASE="$PACK/full-tree-v1"
python3 - "$CASE/problem/problem.json" <<'PY' > estimate.json
import json, sys
with open(sys.argv[1]) as f:
    p = json.load(f)
d = p['camera']['source_to_detector_mm'] / 2
print(json.dumps({'schema':'lumen-rigid-pose/1', 'case_id':p['case_id'],
                  'model_to_camera_mm':[[1,0,0,0],[0,1,0,0],[0,0,1,d],[0,0,0,1]]}))
PY
python3 -I -S "$PACK/score.py" "$CASE" estimate.json
python3 -I -S "$PACK/score.py" "$CASE" "$CASE/answer/pose.json"
```

The second call is a zero-error control. `rms_3d_mm` is the root mean squared Euclidean distance between corresponding model samples transformed by your pose and by the reference. `rms_reprojection_px` compares their detector coordinates, not pixel intensities. Each listed sample counts once, including repeated junction endpoints. Off-image estimates still count; points behind the source or beyond the detector are rejected. These scores check a supplied pose, not whether one image uniquely determines it.

Open the [reference overlay](full-tree-v1/answer/overlay.png) to see the inlet (purple), west (rust), apical (green) and east (blue) territories. `answer/territories.json` maps the geometry's group labels to binary PNG masks (255 = assigned, 0 = other). At overlaps, the largest accumulated contribution wins; a mask is not the isolated projection of its territory. `answer/pose.json` is the reference transform.

## Rebuild offline

With Lumen and NumPy installed, choose a **new** output directory:

```sh
python3 -m lumen.registration_pack.generate rebuilt-full-tree
```

The generator samples the fixed cubic branch recipe and reads the frozen pose from `answer/`. It writes both halves, without fetching data or invoking a solver. All geometry, images and masks are procedural and covered by the repository's Apache-2.0 license. This is one static registration fixture, not a dynamics episode, clinical image or validation study.


## Case 2: selective exposures

Open [blind A](selective-a-v1/problem/image.png) or [blind B](selective-b-v1/problem/image.png).
Both come from the same 16-branch model in the same pose, with the same camera. Only the contrast-filled territory changes. Give a reader just one `problem/` directory and ask for the pose of the **whole tree**. A white pixel is not evidence that a modeled branch is absent. The `territory` tags in the geometry identify all the possible groups; neither manifest names the filled group. For independent blind attempts, collect both estimates before revealing either answer.

Use the estimate example above with `CASE="$PACK/selective-a-v1"` or `CASE="$PACK/selective-b-v1"`. For a blind submission, stop after writing `estimate.json`. Scoring even your own estimate requires `answer/pose.json`. After submission, the evaluator can score your file and compare the two supplied poses with Python alone:

```sh
PACK=lumen/registration_pack
for CASE in "$PACK/selective-a-v1" "$PACK/selective-b-v1"; do
  python3 -I -S "$PACK/score.py" "$CASE" "$CASE/answer/pose.json"
  python3 -I -S "$PACK/score.py" "$CASE" "$CASE/answer/wrong_pose.json"
done
```

### Reveal: a good outline at the wrong depth

Compare [A reference](selective-a-v1/answer/overlay.png) with [A wrong](selective-a-v1/answer/wrong_overlay.png), then [B reference](selective-b-v1/answer/overlay.png) with [B wrong](selective-b-v1/answer/wrong_overlay.png). Blue traces the inner edge of the projected filled territory on the **original blind image**. Both wrong outlines follow the forks; look at the tips to see where the blue line stops short.

The wrong hypothesis keeps the rotation but moves the entire model 12 mm away from the source along the ray through the mean of the filled centerline samples. That point stays on its ray. The surrounding branches shrink in projection. These are separate perspective renders, not a translated or resized PNG.

| Exposure | Filled group | Foreground Dice, reference / wrong | Whole-tree 3D RMS, reference / wrong | Whole-tree reprojection RMS, reference / wrong |
| --- | --- | --- | --- | --- |
| A | west | 1.000 / 0.845 | 0 / 12 mm | 0 / 6.19 px |
| B | east | 1.000 / 0.865 | 0 / 12 mm | 0 / 6.35 px |

The image-space test asks for foreground Dice of at least 0.75. The independent pose test treats a 3D RMS error of 10 mm or more as wrong. Only the wrong hypothesis passes the first while failing the second. These cutoffs illustrate depth ambiguity in this synthetic fixture; they are not clinical tolerances.

**Answer files.** Use the camera intrinsics and axes from the corresponding `problem/problem.json` with `answer/pose.json`. `visibility.json` partitions the geometry IDs into `visible` and `hidden`; only `visible` branches enter the attenuation calculation. `territories.json` points to the binary filled-territory mask. `wrong_mask.png` is the foreground at the wrong pose. `projections.json` has `reference` and `wrong` maps: for each branch ID, `[column,row]` pairs in model sample order, in native image pixels. Hidden projections are provided for checking the full model, not as evidence of contrast.

**Reproduce the image-space diagnostic.** With NumPy and the source checkout, compare the PNG pixels rather than the proportion of the white background that matches:

```sh
python3 - <<'PY'
from pathlib import Path
from lumen.registration_pack.generate import read_png
for case in sorted(Path('lumen/registration_pack').glob('selective-?-v1')):
    observed = read_png(case / 'problem/image.png') < 255
    for name in ('mask.png', 'wrong_mask.png'):
        predicted = read_png(case / 'answer' / name) == 255
        print(case.name, name, 2 * (observed & predicted).sum() / (observed.sum() + predicted.sum()))
PY
python3 -m lumen.registration_pack.selective rebuilt-selective
```

Use a new destination. Both exposures use the original procedural tree and renderer, with no external images, source volumes or weights. The Apache-2.0 license covers the generated fixtures as well as the code.
