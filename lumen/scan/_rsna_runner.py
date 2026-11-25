"""Run the RSNA 2025 intracranial artery model on one head scan.

This file is executed by Lumen inside the model's own private Python environment (it imports
``torch`` and ``nnxnet``, which Lumen itself does not depend on). It must not import ``lumen``.

    python _rsna_runner.py --scan head.nii.gz --weights <dir> --out labels.nii.gz --device mps

The steps follow the authors' published inference notebook:

1. the scan is put in LPS order;
2. stage 1 (a 2-D network on the three centre planes) boxes the part of the head that holds the
   arteries;
3. the box is z-scored, resized to 224 x 224 x 224 and passed through the stage 2 network;
4. the first segmentation head gives one label per voxel: 13 named arteries plus "aneurysm".

Only the first segmentation head is evaluated (the classification heads and the second decoder
are not needed to trace arteries and roughly double the memory). No test-time augmentation.

The output is a label volume on the input grid with the input affine: 0 background, 1..13 the
arteries in ``ARTERIES`` order, 14 aneurysm sac.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

STAGE1 = "Dataset180_2D_vessel_box_seg_stable/nnUNetTrainer__nnUNetPlans__2d"
STAGE2 = "Dataset660_26classes_resize224_4661/onlyMirror01_lr4e3_100epochs_ps224"

# first segmentation head, channels 1..14 (dataset.json of the published model)
ARTERIES = (
    "other posterior circulation", "basilar tip",
    "right posterior communicating artery", "left posterior communicating artery",
    "right infraclinoid internal carotid artery", "left infraclinoid internal carotid artery",
    "right supraclinoid internal carotid artery", "left supraclinoid internal carotid artery",
    "right middle cerebral artery", "left middle cerebral artery",
    "right anterior cerebral artery", "left anterior cerebral artery",
    "anterior communicating artery", "aneurysm",
)


def _to_lps(img):
    import nibabel as nib

    src = nib.io_orientation(img.affine)
    dst = nib.orientations.axcodes2ornt(("L", "P", "S"))
    fwd = nib.orientations.ornt_transform(src, dst)
    inv = nib.orientations.ornt_transform(dst, src)
    return img.as_reoriented(fwd), inv


def _roi(zyx, spacing_xyz, weights: Path, device):
    """Stage 1: the box (z0, z1, y0, y1, x0, x1) that holds the intracranial arteries."""
    import torch
    from nnxnet.inference.predict_from_raw_data_2D_orthogonal_planes_fast import nnXNetPredictor

    predictor = nnXNetPredictor(tile_step_size=0.5, use_mirroring=False, use_gaussian=True,
                                perform_everything_on_device=True, device=device, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=str(weights / STAGE1), use_folds=(0,),
        checkpoint_name="checkpoint_final.pth")
    predictor.initialize_network_and_gaussian()
    with torch.no_grad():
        box = predictor.predict_from_multi_axial_slices(
            zyx[None], spacing_xyz, np.array([1, 0.55, 0.5]), max_batch_size=16)
    return tuple(int(v) for v in box)


def _network(weights: Path):
    """Stage 2 network with its published weights, first segmentation head only."""
    import torch
    from nnxnet.training.nnXNetTrainer.variants.network_architecture.\
        ResEncoderUNet_two_seg_with_cls_modality import ResEncoderUNet_two_seg_with_cls_modality

    allowed = [np.dtype, np._core.multiarray.scalar, np.dtypes.Float64DType,
               np.dtypes.Float32DType, np.dtypes.Int64DType]
    with torch.serialization.safe_globals(allowed):
        ckpt = torch.load(str(weights / STAGE2 / "fold_0" / "checkpoint_final.pth"),
                          map_location="cpu", weights_only=True)
    init = ckpt["init_args"]
    conf = init["plans"]["configurations"][init["configuration"]]
    kwargs = dict(conf["architecture"]["arch_kwargs"])
    kwargs.update(conv_op=torch.nn.Conv3d, norm_op=torch.nn.InstanceNorm3d, dropout_op=None,
                  nonlin=torch.nn.LeakyReLU)
    net = ResEncoderUNet_two_seg_with_cls_modality(
        in_channels=1, out_channels_1=15, out_channels_2=14, deep_supervision=False, **kwargs)
    net.load_state_dict(ckpt["network_weights"], strict=True)
    net.eval()

    class Tracer(torch.fx.Tracer):
        def is_leaf_module(self, module, name):
            return module.__class__.__name__ == "ClassificationHead" or super().is_leaf_module(module, name)

    graph = Tracer().trace(net, concrete_args={"only_forward_cls": False})
    out = next(n for n in graph.nodes if n.op == "output")
    out.args = (out.args[0][0],)
    head1 = torch.fx.GraphModule(net, graph)
    head1.graph.eliminate_dead_code()
    head1.delete_all_unused_submodules()
    head1.recompile()
    return head1.eval()


def _labels(crop_zyx, weights: Path, device):
    """Stage 2: per-voxel artery label on the cropped box."""
    import torch

    x = torch.from_numpy(np.ascontiguousarray(crop_zyx, dtype=np.float32))[None]
    x = (x - x.mean()) / x.std().clamp_min(1e-8)
    x = torch.nn.functional.interpolate(x[None], size=(224, 224, 224), mode="trilinear",
                                        align_corners=True)
    net = _network(weights).to(device)
    with torch.inference_mode():
        logits = net(x.to(device)).float().cpu()
    del net
    # back to the box's own grid one channel at a time (keeps memory flat), then pick the best
    best = torch.full(crop_zyx.shape, -np.inf)
    lab = torch.zeros(crop_zyx.shape, dtype=torch.uint8)
    for c in range(logits.shape[1]):
        up = torch.nn.functional.interpolate(logits[:, c:c + 1], size=crop_zyx.shape,
                                             mode="trilinear", align_corners=True)[0, 0]
        win = up > best
        best[win] = up[win]
        lab[win] = c
    return lab.numpy()


def main(argv=None) -> int:
    import nibabel as nib
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    device = torch.device({"gpu": "cuda"}.get(args.device, args.device))
    weights = Path(args.weights)

    img = nib.load(args.scan)
    lps, back = _to_lps(img)
    zyx = np.asarray(lps.dataobj, dtype=np.float32).transpose(2, 1, 0)
    spacing = np.asarray(lps.header.get_zooms()[:3], float)
    z0, z1, y0, y1, x0, x1 = _roi(zyx, spacing, weights, device)
    box_mm = [(z1 - z0) * spacing[2], (y1 - y0) * spacing[1], (x1 - x0) * spacing[0]]
    if min(z1 - z0, y1 - y0, x1 - x0) < 8:
        print(json.dumps({"error": "no arteries were located in this scan"}))
        return 3
    lab = _labels(zyx[z0:z1, y0:y1, x0:x1], weights, device)
    full = np.zeros(zyx.shape, np.uint8)
    full[z0:z1, y0:y1, x0:x1] = lab
    native = nib.orientations.apply_orientation(full.transpose(2, 1, 0), back)
    if native.shape != img.shape[:3]:
        raise RuntimeError(f"label grid {native.shape} does not match the scan {img.shape}")
    out = nib.Nifti1Image(np.ascontiguousarray(native, dtype=np.uint8), img.affine)
    nib.save(out, args.out)
    found = {int(v): ARTERIES[int(v) - 1] for v in np.unique(native) if v > 0}
    print(json.dumps({"box_zyx": [z0, z1, y0, y1, x0, x1], "box_mm": [round(float(v), 1) for v in box_mm],
                      "voxel_mm_in_box": [round(float(b) / 224, 3) for b in box_mm],
                      "found": found, "device": str(device)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
