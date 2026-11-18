# Register a full tree from one projection

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
