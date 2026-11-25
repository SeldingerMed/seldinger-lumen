"""Artery segmenters for ``lumen import``: which one a scan needs, asking before installing it,
and running it.

Lumen ships no segmentation model. Two open models are known:

* ``body`` -- TotalSegmentator (Apache-2.0): aorta, arch branches, common and internal carotid
  arteries, iliac arteries, on CT; aorta and iliac arteries on MR.
* ``head`` -- the RSNA 2025 intracranial model (Apache-2.0): the named arteries of the circle
  of Willis and their feeders, on CTA and MRA.

The first time a scan needs one that is not installed, Lumen says what it is, how big it is and
under which licence it comes, and asks. On yes it installs the model into Lumen's own folder
(``~/.lumen`` or ``$LUMEN_HOME``), in a private Python environment, so the model's PyTorch and
nnU-Net versions never touch the environment Lumen runs in. Each model runs as a subprocess on
the user's machine; the scan is never sent anywhere.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lumen.scan.region import BODY, HEAD

_RAS_TO_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])


class SegmenterError(RuntimeError):
    """A segmenter is needed but is not available, or it failed."""


class Declined(SegmenterError):
    """The user said no to installing a segmenter."""


@dataclass(frozen=True)
class Segmenter:
    key: str                  # folder name under <home>/segmenters
    region: str               # "body" or "head"
    title: str
    finds: str                # what it names, for the prompt
    licence: str
    repo: str                 # our fork
    commit: str               # pinned commit of our fork
    size: str                 # rough download size, for the prompt
    python: str               # Python version of the private environment


TOTALSEGMENTATOR = Segmenter(
    key="totalsegmentator", region=BODY, title="TotalSegmentator",
    finds="the aorta, its arch branches, the carotid and iliac arteries",
    licence="Apache-2.0", repo="https://github.com/SeldingerMed/TotalSegmentator",
    commit="c73ff4fc39bdc503ed3e5d0e135c7bc57569c51e", size="about 2 GB with PyTorch",
    python="3.11")

RSNA_INTRACRANIAL = Segmenter(
    key="rsna-intracranial", region=HEAD, title="the RSNA 2025 intracranial artery model",
    finds="the arteries of the circle of Willis and their feeders",
    licence="Apache-2.0", repo="https://github.com/SeldingerMed/RSNA2025_Intracranial-Aneurysm-Detection",
    commit="c76f73b2019d1e8a88dcaf83c7df8df80277657b", size="about 2.5 GB with PyTorch",
    python="3.11")

SEGMENTERS = {BODY: TOTALSEGMENTATOR, HEAD: RSNA_INTRACRANIAL}

# TotalSegmentator classes used, in the order names are reported
BODY_ARTERIES = ("aorta", "brachiocephalic_trunk", "subclavian_artery_right", "subclavian_artery_left",
                 "common_carotid_artery_right", "common_carotid_artery_left",
                 "iliac_artery_left", "iliac_artery_right")
NECK_ARTERIES = ("internal_carotid_artery_right", "internal_carotid_artery_left")
# RSNA 2025 first segmentation head, label values 1..14 (must match lumen/scan/_rsna_runner.py)
HEAD_ARTERIES = (
    "other posterior circulation", "basilar tip",
    "right posterior communicating artery", "left posterior communicating artery",
    "right infraclinoid internal carotid artery", "left infraclinoid internal carotid artery",
    "right supraclinoid internal carotid artery", "left supraclinoid internal carotid artery",
    "right middle cerebral artery", "left middle cerebral artery",
    "right anterior cerebral artery", "left anterior cerebral artery",
    "anterior communicating artery", "aneurysm",
)


def pretty(name: str) -> str:
    """``common_carotid_artery_left`` -> ``left common carotid artery``."""
    words = name.replace("-", "_").split("_")
    side = [w for w in words if w in ("left", "right")]
    rest = [w for w in words if w not in ("left", "right")]
    return " ".join(side + rest)


def for_regions(regions) -> list[Segmenter]:
    return [SEGMENTERS[r] for r in (BODY, HEAD) if r in regions]


# ---------------------------------------------------------------------------------------------
# where things live
# ---------------------------------------------------------------------------------------------
def home() -> Path:
    return Path(os.environ.get("LUMEN_HOME") or Path.home() / ".lumen").expanduser()


def _dir(seg: Segmenter) -> Path:
    return home() / "segmenters" / seg.key


def weights_dir(seg: Segmenter) -> Path:
    """Model weights folder; ``$LUMEN_WEIGHTS`` moves weights (the bulk of the download) to
    another disk."""
    base = os.environ.get("LUMEN_WEIGHTS")
    return (Path(base).expanduser() / seg.key) if base else _dir(seg) / "weights"


def _python(seg: Segmenter) -> Path:
    sub = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    return _dir(seg) / "env" / sub


def installed(seg: Segmenter) -> bool:
    try:
        rec = json.loads((_dir(seg) / "installed.json").read_text())
    except (OSError, ValueError):
        return False
    return rec.get("commit") == seg.commit and _python(seg).exists()


# ---------------------------------------------------------------------------------------------
# asking
# ---------------------------------------------------------------------------------------------
def _consent_file() -> Path:
    return home() / "consent.json"


def _consents() -> dict:
    try:
        return json.loads(_consent_file().read_text())
    except (OSError, ValueError):
        return {}


def prompt_text(seg: Segmenter, why: str) -> str:
    return (f"{why} To find {seg.finds}, Lumen needs {seg.title}\n"
            f"({seg.licence}, {seg.size}), installed once into {home()}.\n"
            f"It comes from {seg.repo}. It runs on this machine; the scan is not sent anywhere.\n"
            f"Install it now? [y/N] ")


def ensure(seg: Segmenter, *, why: str, assume_yes: bool = False, ask=None, say=print,
           install=None) -> None:
    """Make sure ``seg`` is installed, asking first unless the user already agreed.

    ``ask`` is ``input`` by default (only on a terminal). Raises ``Declined`` on no, and
    ``SegmenterError`` when it cannot ask (no terminal and no ``assume_yes``).
    """
    if installed(seg):
        return
    agreed = assume_yes or bool(_consents().get(seg.key, {}).get("agreed"))
    if not agreed:
        if ask is None:
            if not (sys.stdin and sys.stdin.isatty()):
                raise SegmenterError(
                    f"this scan needs {seg.title}, which is not installed. Run the command in a "
                    f"terminal to be asked, or pass --yes to install it without being asked.")
            ask = input
        answer = str(ask(prompt_text(seg, why))).strip().lower()
        if answer not in ("y", "yes"):
            raise Declined(f"{seg.title} was not installed. To import without it, pass your own "
                           f"vessel mask with --mask.")
    rec = _consents()
    rec[seg.key] = {"agreed": True, "when": time.strftime("%Y-%m-%dT%H:%M:%S")}
    _consent_file().parent.mkdir(parents=True, exist_ok=True)
    _consent_file().write_text(json.dumps(rec, indent=1))
    say(f"Installing {seg.title} (this happens once) ...")
    (install or _install)(seg, say)
    say(f"{seg.title} is installed.")


# ---------------------------------------------------------------------------------------------
# installing
# ---------------------------------------------------------------------------------------------
def _run(cmd, log: Path, env=None, what="command") -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as f:
        f.write(("\n$ " + " ".join(map(str, cmd)) + "\n").encode())
        f.flush()
        rc = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    if rc != 0:
        tail = log.read_text(errors="replace")[-1500:]
        raise SegmenterError(f"{what} failed (exit {rc}); full log: {log}\n{tail}")


def _archive(seg: Segmenter) -> str:
    return f"{seg.repo}/archive/{seg.commit}.zip"


def _make_env(seg: Segmenter, log: Path) -> None:
    env_dir = _dir(seg) / "env"
    if env_dir.exists():
        shutil.rmtree(env_dir)
    uv = shutil.which("uv")
    if uv:
        _run([uv, "venv", "--python", seg.python, env_dir], log, what="creating the environment")
    else:
        _run([sys.executable, "-m", "venv", env_dir], log, what="creating the environment")


def _pip(seg: Segmenter, args: list, log: Path) -> None:
    uv = shutil.which("uv")
    if uv:
        _run([uv, "pip", "install", "--python", _python(seg), *args], log, what="installing packages")
    else:
        _run([_python(seg), "-m", "pip", "install", *args], log, what="installing packages")


_KAGGLE = "https://www.kaggle.com/api/v1/models/{handle}/download/{path}"
_S1 = "Dataset180_2D_vessel_box_seg_stable/nnUNetTrainer__nnUNetPlans__2d"
_S2 = "Dataset660_26classes_resize224_4661/onlyMirror01_lr4e3_100epochs_ps224"
_H1 = "pengchengshi/dataset180_2d_vessel_box_seg_stable/pytorch/default/1"
_H2 = "pengchengshi/dataset660_26classes_resize224_4661/pytorch/default/2"
# (model handle, file, bytes, sha256). The authors publish no checksums; these are the hashes of
# the files this version of Lumen was tested with, so every install runs the same weights.
RSNA_WEIGHTS = (
    (_H1, f"{_S1}/dataset.json", 234, "c119a43d2caa9553c4d954785884102a708623574be51f979024245b6d703e9e"),
    (_H1, f"{_S1}/plans.json", 5783, "3f6078018304a619a981ef7e4e4ea68ff34ce757ee3360e8ffc15d106c682f3c"),
    (_H1, f"{_S1}/fold_0/checkpoint_final.pth", 268241288, "1782b71c378c6602976671cf909b63068bfefbcb12d5e33bb41884b7ea0159ef"),
    (_H2, f"{_S2}/dataset.json", 1563, "1867f11cc437d9a28c35cf9763f44cdfa34c5719c5abcf3278383faf6e692a4b"),
    (_H2, f"{_S2}/plans.json", 15817, "4be258af7b9fd0aa6c7883d820868130f454853ecedbae9984c24b59e1e5cf20"),
    (_H2, f"{_S2}/fold_0/checkpoint_final.pth", 875399410,
     "e60b539d025a8ecccf77d3ff1a45ef6888b5e059671cc8e24fff572d520fa521"),
)


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _fetch(url: str, dest: Path, size: int, sha256: str) -> None:
    """Download one weight file, refusing anything that is not exactly the tested file."""
    import urllib.request

    if dest.exists() and dest.stat().st_size == size and _sha256(dest) == sha256:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "seldinger-lumen"})
    got = 0
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(part, "wb") as f:
            while True:
                block = resp.read(1 << 20)
                if not block:
                    break
                got += len(block)
                if got > size:
                    raise SegmenterError(f"{dest.name}: the download is larger than expected")
                f.write(block)
    except OSError as exc:
        part.unlink(missing_ok=True)
        raise SegmenterError(f"could not download {dest.name}: {exc}") from exc
    if got != size or _sha256(part) != sha256:
        part.unlink(missing_ok=True)
        raise SegmenterError(f"{dest.name}: the downloaded file is not the one Lumen was tested "
                             f"with; nothing was installed")
    part.replace(dest)


def _install(seg: Segmenter, say=print) -> None:
    log = _dir(seg) / "install.log"
    _make_env(seg, log)
    if seg.key == TOTALSEGMENTATOR.key:
        _pip(seg, [f"TotalSegmentator @ {_archive(seg)}"], log)
        w = weights_dir(seg)
        w.mkdir(parents=True, exist_ok=True)
        cfg = w / "config.json"
        if not cfg.exists():       # belt and braces: the fork already defaults to no statistics
            cfg.write_text(json.dumps({"totalseg_id": "lumen", "send_usage_stats": False,
                                       "prediction_counter": 0, "statistics_disclaimer_shown": True},
                                      indent=1))
    elif seg.key == RSNA_INTRACRANIAL.key:
        _pip(seg, [f"nnxnet @ {_archive(seg)}#subdirectory=nnXNet"], log)
        for handle, rel, size, sha in RSNA_WEIGHTS:
            _fetch(_KAGGLE.format(handle=handle, path=rel), weights_dir(seg) / rel, size, sha)
    else:
        raise SegmenterError(f"no installer for {seg.key}")
    (_dir(seg) / "installed.json").write_text(json.dumps(
        {"commit": seg.commit, "repo": seg.repo, "when": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))


# ---------------------------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------------------------
def write_nifti(scan, path: Path) -> None:
    """Write the scan as NIfTI (RAS) for a segmenter; the file carries no patient tags."""
    import nibabel as nib

    img = nib.Nifti1Image(np.asarray(scan.data, dtype=np.float32), _RAS_TO_LPS @ np.asarray(scan.affine, float))
    img.header.set_xyzt_units("mm")
    nib.save(img, str(path))


def _read_labels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import nibabel as nib

    img = nib.load(str(path))
    return np.asarray(img.dataobj).astype(np.int32), _RAS_TO_LPS @ np.asarray(img.affine, float)


def _torch_env(extra: dict | None = None) -> dict:
    """Environment for a segmenter subprocess.

    A machine that caps Apple-GPU memory with ``PYTORCH_MPS_HIGH_WATERMARK_RATIO`` below
    PyTorch's default low watermark (1.4) makes every model load fail with "invalid low
    watermark ratio". Keep the user's cap and set the low watermark under it.
    """
    env = dict(os.environ, **(extra or {}))
    high = env.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO")
    if high and "PYTORCH_MPS_LOW_WATERMARK_RATIO" not in env:
        try:
            if 0.0 < float(high) < 1.4:
                env["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = f"{0.8 * float(high):.3f}"
        except ValueError:
            pass
    return env


def _device(seg: Segmenter) -> str:
    code = ("import torch;print('gpu' if torch.cuda.is_available() else "
            "'mps' if torch.backends.mps.is_available() else 'cpu')")
    out = subprocess.run([str(_python(seg)), "-c", code], capture_output=True, text=True)
    return out.stdout.strip().splitlines()[-1] if out.returncode == 0 and out.stdout.strip() else "cpu"


def _class_map(seg: Segmenter, task: str) -> dict:
    code = ("import json;from totalsegmentator.map_to_binary import class_map;"
            f"print(json.dumps(class_map[{task!r}]))")
    out = subprocess.run([str(_python(seg)), "-c", code], capture_output=True, text=True)
    if out.returncode != 0:
        raise SegmenterError(f"cannot read the class list of {seg.title}: {out.stderr[-400:]}")
    return {int(k): v for k, v in json.loads(out.stdout.strip().splitlines()[-1]).items()}


def run_body(scan, work: Path, say=print):
    """Run TotalSegmentator on ``scan``; return a ``LabelVolume`` of named arteries."""
    from lumen.assets.scan_import import LabelVolume

    seg = TOTALSEGMENTATOR
    work.mkdir(parents=True, exist_ok=True)
    src, log = work / "scan.nii.gz", work / "totalsegmentator.log"
    write_nifti(scan, src)
    env = _torch_env({"TOTALSEG_HOME_DIR": str(weights_dir(seg))})
    exe = _python(seg).parent / ("TotalSegmentator.exe" if os.name == "nt" else "TotalSegmentator")
    device = _device(seg)
    mr = scan.modality == "MR"
    task = "total_mr" if mr else "total"
    cmap = _class_map(seg, task)
    wanted = [n for n in BODY_ARTERIES if n in cmap.values()]
    say(f"Finding arteries with {seg.title} on {device} ...")
    out = work / "body.nii.gz"
    _run([exe, "-i", src, "-o", out, "--ml", "--task", task, "--device", device,
          "--roi_subset", *wanted], log, env=env, what=seg.title)
    lab, aff = _read_labels(out)
    data = np.zeros(lab.shape, np.int16)
    names: dict = {}
    for idx, name in cmap.items():
        if name in wanted and np.any(lab == idx):
            names[len(names) + 1] = pretty(name)
            data[lab == idx] = len(names)
    carotid = any("carotid" in v for v in names.values())
    if carotid and not mr:                      # the neck task adds the internal carotids
        out2 = work / "neck.nii.gz"
        _run([exe, "-i", src, "-o", out2, "--ml", "--task", "headneck_bones_vessels",
              "--device", device], log, env=env, what=seg.title)
        lab2, _ = _read_labels(out2)
        for idx, name in _class_map(seg, "headneck_bones_vessels").items():
            if name in NECK_ARTERIES and np.any(lab2 == idx):
                names[len(names) + 1] = pretty(name)
                data[(lab2 == idx) & (data == 0)] = len(names)
    if not names:
        raise SegmenterError(f"{seg.title} found no arteries in this scan")
    return LabelVolume(data, aff, names)


def run_head(scan, work: Path, say=print, head_from_mm: float | None = None):
    """Run the RSNA 2025 intracranial model on ``scan``; return a ``LabelVolume`` of named
    arteries on the scan's own grid.

    ``head_from_mm`` (patient z, mm) is where the head starts in a scan that also shows the neck
    or trunk; the model is given only the part above it, as it was trained on head scans.
    """
    from lumen.assets.scan_import import LabelVolume
    from lumen.scan.io import ScanVolume

    seg = RSNA_INTRACRANIAL
    work.mkdir(parents=True, exist_ok=True)
    src, out, log = work / "scan.nii.gz", work / "head.nii.gz", work / "rsna.log"
    A = np.asarray(scan.affine, float)
    data = np.asarray(scan.data)
    sl = [slice(None)] * 3
    if head_from_mm is not None:
        sp = np.linalg.norm(A[:3, :3], axis=0)
        ax = int(np.argmax(np.abs(A[2, :3] / sp)))
        z = A[2, 3] + A[2, ax] * np.arange(data.shape[ax])
        keep = np.nonzero(z >= head_from_mm)[0]
        if len(keep) >= 16:
            sl[ax] = slice(int(keep.min()), int(keep.max()) + 1)
    sub_aff = A.copy()
    sub_aff[:3, 3] = (A @ np.array([sl[0].start or 0, sl[1].start or 0, sl[2].start or 0, 1.0]))[:3]
    write_nifti(ScanVolume(data[tuple(sl)], sub_aff), src)
    device = _device(seg)
    say(f"Finding arteries with {seg.title} on {device} ...")
    runner = Path(__file__).with_name("_rsna_runner.py")
    _run([_python(seg), runner, "--scan", src, "--weights", weights_dir(seg), "--out", out,
          "--device", device], log, env=_torch_env(), what=seg.title)
    lab, aff = _read_labels(out)
    if not np.allclose(aff, sub_aff, atol=1e-3) or lab.shape != data[tuple(sl)].shape:
        raise SegmenterError(f"{seg.title} returned a different grid from the scan")
    full = np.zeros(data.shape, np.int16)
    full[tuple(sl)] = lab
    names = {i + 1: n for i, n in enumerate(HEAD_ARTERIES) if np.any(full == i + 1)}
    if not names:
        raise SegmenterError(f"{seg.title} found no arteries in this scan")
    return LabelVolume(full, A, names)
