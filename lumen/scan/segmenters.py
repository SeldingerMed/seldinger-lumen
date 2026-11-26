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

import contextlib
import json
import os
import shutil
import subprocess
import sys
import threading
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
    return isinstance(rec, dict) and rec.get("commit") == seg.commit and _python(seg).exists()


# ---------------------------------------------------------------------------------------------
# asking
# ---------------------------------------------------------------------------------------------
def _consent_file() -> Path:
    return home() / "consent.json"


_CONSENT_LOCK = threading.Lock()   # a background install may forget while the main thread agrees


def _consents() -> dict:
    """What the user has agreed to; a missing or damaged file means nothing was agreed."""
    try:
        rec = json.loads(_consent_file().read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(rec, dict):
        return {}
    return {k: v for k, v in rec.items() if isinstance(v, dict)}


def prompt_text(seg: Segmenter, why: str) -> str:
    return (f"{why} To find {seg.finds}, Lumen needs {seg.title}\n"
            f"({seg.licence}, {seg.size}), installed once into {home()}.\n"
            f"It comes from {seg.repo}. It runs on this machine; the scan is not sent anywhere.\n"
            f"Install it now? [y/N] ")


_NO_ENV_FILESYSTEMS = ("exfat", "msdos", "vfat", "ntfs", "ntfs3")   # a Python environment does not install on these


_PROC_MOUNTS = Path("/proc/mounts")


def _filesystem(path: Path) -> str:
    """The filesystem type holding ``path`` (macOS and Linux; '' when it cannot be told)."""
    while not path.exists() and path != path.parent:
        path = path.parent
    if sys.platform.startswith("linux"):
        try:
            real, best, kind = os.path.realpath(path), "", ""
            for line in _PROC_MOUNTS.read_text().splitlines():
                parts = line.split()
                if len(parts) < 3:
                    continue
                point = parts[1].replace("\\040", " ")
                if (real == point or real.startswith(point.rstrip("/") + "/")) and len(point) > len(best):
                    best, kind = point, parts[2].lower()
            return kind
        except OSError:
            return ""
    if sys.platform != "darwin":
        return ""

    def _out(cmd):
        return subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL,
                              timeout=10, check=True).stdout

    try:
        # `df -P` ends each line with the mount point; `mount` lists "dev on <point> (type, ...)"
        dev = _out(["df", "-P", str(path)]).splitlines()[-1].split()[0]
        for line in _out(["mount"]).splitlines():
            if line.startswith(dev + " on ") and line.rstrip().endswith(")"):
                return line[line.rindex("(") + 1:].split(",")[0].strip().lower()
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        pass
    return ""


def check_home(seg: Segmenter) -> None:
    """Refuse, before asking or downloading, a folder that cannot hold the model's environment."""
    fs = _filesystem(_dir(seg))
    if fs in _NO_ENV_FILESYSTEMS:
        raise SegmenterError(
            f"Lumen's folder for model environments ({home()}) is on a drive formatted as {fs}, which cannot "
            f"hold a Python environment. Set LUMEN_HOME to a folder on your main disk; the large "
            f"weight files can still go on that drive with LUMEN_WEIGHTS.")


def agree(seg: Segmenter, *, why: str, assume_yes: bool = False, ask=None) -> None:
    """Get the user's yes to installing ``seg`` (remembered, so they are asked once).

    ``ask`` is ``input`` by default (only on a terminal). Raises ``Declined`` on no, and
    ``SegmenterError`` when it cannot ask (no terminal and no ``assume_yes``).
    """
    check_home(seg)
    agreed = assume_yes or bool(_consents().get(seg.key, {}).get("agreed"))
    if not agreed:
        if ask is None:
            if not (sys.stdin and sys.stdin.isatty()):
                raise SegmenterError(
                    f"this scan needs {seg.title}, which is not installed. Run the command in a "
                    f"terminal to be asked, or pass --yes to install it without being asked.")
            ask = _ask_on_stderr
        try:
            answer = str(ask(prompt_text(seg, why))).strip().lower()
        except EOFError:                                    # input closed: the same as no
            answer = ""
        if answer not in ("y", "yes"):
            raise Declined(f"{seg.title} was not installed. To import without it, pass your own "
                           f"vessel mask with --mask.")
    with _CONSENT_LOCK:
        rec = _consents()
        rec[seg.key] = {"agreed": True, "when": time.strftime("%Y-%m-%dT%H:%M:%S")}
        _consent_file().parent.mkdir(parents=True, exist_ok=True)
        _consent_file().write_text(json.dumps(rec, indent=1))


def _ask_on_stderr(text: str) -> str:
    """Ask on the terminal through stderr, so stdout (``--json``) carries only the result."""
    sys.stderr.write(text)
    sys.stderr.flush()
    line = sys.stdin.readline()
    if not line:
        raise EOFError
    return line


def forget(seg: Segmenter) -> None:
    """Drop a recorded yes, so the next import asks again (used when the install failed)."""
    with _CONSENT_LOCK:
        rec = _consents()
        if rec.pop(seg.key, None) is not None:
            _consent_file().write_text(json.dumps(rec, indent=1))


def ensure(seg: Segmenter, *, why: str, assume_yes: bool = False, ask=None, say=print,
           install=None) -> None:
    """Make sure ``seg`` is installed, asking first unless the user already agreed."""
    if installed(seg):
        return
    agree(seg, why=why, assume_yes=assume_yes, ask=ask)
    say(f"Installing {seg.title} (this happens once) ...")
    try:
        (install or _install)(seg, say)
    except BaseException:
        forget(seg)                                         # a failed install is asked about again
        raise
    say(f"{seg.title} is installed.")


class BackgroundInstall:
    """An install running quietly on its own thread; ``wait()`` returns when it is usable."""

    def __init__(self, seg: Segmenter, install=None):
        import threading

        self.seg, self.error = seg, None
        self._thread = threading.Thread(target=self._work, args=(install or _install,), daemon=True)
        self._thread.start()

    def _work(self, install) -> None:
        try:
            install(self.seg, lambda *_: None)
        except Exception as exc:                            # reported by wait(), on the caller's thread
            self.error = exc
            forget(self.seg)                                # a failed install is asked about again

    def wait(self) -> None:
        self._thread.join()
        if self.error is not None:
            if isinstance(self.error, SegmenterError):
                raise self.error
            raise SegmenterError(f"installing {self.seg.title} failed: {self.error}") from self.error


# ---------------------------------------------------------------------------------------------
# installing
# ---------------------------------------------------------------------------------------------
RUN_TIMEOUT_S = 3600.0   # no install step or model run on one scan should take an hour


def _run(cmd, log: Path, env=None, what="command", timeout: float = RUN_TIMEOUT_S) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as f:
        f.write(("\n$ " + " ".join(map(str, cmd)) + "\n").encode())
        f.flush()
        try:
            # no terminal for the child: it runs beside the prompt and must never read from it
            rc = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, env=env,
                                stdin=subprocess.DEVNULL, timeout=timeout).returncode
        except subprocess.TimeoutExpired as exc:
            raise SegmenterError(f"{what} took longer than {timeout / 60:.0f} minutes and was "
                                 f"stopped; full log: {log}") from exc
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
        _run([_pinned_python(seg), "-m", "venv", env_dir], log, what="creating the environment")


def _pinned_python(seg: Segmenter) -> str:
    """The interpreter the model was tried on, when uv is not there to fetch it."""
    if "{}.{}".format(*sys.version_info[:2]) == seg.python:
        return sys.executable
    found = shutil.which(f"python{seg.python}")
    if found:
        return found
    raise SegmenterError(
        f"{seg.title} needs Python {seg.python} for its own environment, and this is Python "
        "{}.{}. Install uv (https://docs.astral.sh/uv/), which fetches the right Python by "
        "itself, or install Python {} and run the import again.".format(
            *sys.version_info[:2], seg.python))


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
    except SegmenterError:
        part.unlink(missing_ok=True)
        raise
    except BaseException as exc:
        # whatever stopped it (network error, a short read, Ctrl-C), no partial file stays
        part.unlink(missing_ok=True)
        if isinstance(exc, Exception):
            raise SegmenterError(f"could not download {dest.name}: {exc}") from exc
        raise
    if got != size or _sha256(part) != sha256:
        part.unlink(missing_ok=True)
        raise SegmenterError(f"{dest.name}: the downloaded file is not the one Lumen was tested "
                             f"with; nothing was installed")
    part.replace(dest)


_USE_FLOCK = True


def _holder_gone(path: Path) -> bool:
    """Whether the process holding the lock file is gone.

    The file holds a PID. ``os.kill(pid, 0)`` probes liveness on Unix but terminates the process
    on Windows, so there the file's age decides: an install never takes three hours.
    """
    try:
        owner = int(path.read_text().strip() or "0")
    except (OSError, ValueError):
        return True
    if os.name == "nt":
        return time.time() - path.stat().st_mtime > 3 * 3600
    try:
        os.kill(owner, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False                                     # it exists; it belongs to someone else
    return False


@contextlib.contextmanager
def _install_lock(seg: Segmenter):
    """One install of a model at a time, across processes.

    A lock file is held for the whole install: ``flock`` where it exists, an exclusive create
    elsewhere, retried until the other process finishes. A stale lock (its process is gone) is
    taken over, so a crashed install does not block forever.
    """
    path = _dir(seg) / "install.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError:
        fcntl = None
    if not _USE_FLOCK:                                   # tests exercise the portable path
        fcntl = None
    if fcntl is not None:
        f = path.open("a")
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
            f.close()
        return
    import time as _time

    while True:
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            break
        except FileExistsError:
            if _holder_gone(path):
                try:
                    path.unlink()                            # the holder is gone; take the lock over
                    continue
                except OSError:
                    pass
            _time.sleep(2)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def _install(seg: Segmenter, say=print) -> None:
    with _install_lock(seg):
        if installed(seg):                                  # another Lumen finished it meanwhile
            return
        _install_locked(seg, say)


def _install_locked(seg: Segmenter, say=print) -> None:
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
        # NumPy 2 is required, not just preferred: the weights are loaded as data only, through
        # an allow-list of NumPy 2 type names (see _rsna_runner._network)
        _pip(seg, [f"nnxnet @ {_archive(seg)}#subdirectory=nnXNet", "numpy>=2.0"], log)
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


PROBE_TIMEOUT_S = 300.0   # a one-line question to the model's Python; minutes means it is stuck


def _ask_env(seg: Segmenter, code: str, what: str) -> subprocess.CompletedProcess:
    """Run a one-liner in the model's environment, never waiting on it for ever."""
    try:
        return subprocess.run([str(_python(seg)), "-c", code], capture_output=True, text=True,
                              stdin=subprocess.DEVNULL, timeout=PROBE_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise SegmenterError(f"{what} of {seg.title} took longer than "
                             f"{PROBE_TIMEOUT_S / 60:.0f} minutes and was stopped") from exc
    except OSError as exc:
        raise SegmenterError(f"{what} of {seg.title} could not start: {exc}") from exc


def _device(seg: Segmenter) -> str:
    code = ("import torch;print('gpu' if torch.cuda.is_available() else "
            "'mps' if torch.backends.mps.is_available() else 'cpu')")
    out = _ask_env(seg, code, "checking the processor")
    return out.stdout.strip().splitlines()[-1] if out.returncode == 0 and out.stdout.strip() else "cpu"


def _class_map(seg: Segmenter, task: str) -> dict:
    code = ("import json;from totalsegmentator.map_to_binary import class_map;"
            f"print(json.dumps(class_map[{task!r}]))")
    out = _ask_env(seg, code, "reading the class list")
    if out.returncode != 0:
        raise SegmenterError(f"cannot read the class list of {seg.title}: {out.stderr[-400:]}")
    return {int(k): v for k, v in json.loads(out.stdout.strip().splitlines()[-1]).items()}


def run_body(scan, work: Path, say=print):
    """Run TotalSegmentator on ``scan``; return a ``LabelVolume`` of named arteries."""
    seg = TOTALSEGMENTATOR
    work.mkdir(parents=True, exist_ok=True)
    src, log = work / "scan.nii.gz", work / "totalsegmentator.log"
    env = _torch_env({"TOTALSEG_HOME_DIR": str(weights_dir(seg))})
    exe = _python(seg).parent / ("TotalSegmentator.exe" if os.name == "nt" else "TotalSegmentator")
    device = _device(seg)
    from lumen.scan.io import is_ct

    mr = not is_ct(scan)                         # NIfTI and NRRD carry no modality: judged by values
    task = "total_mr" if mr else "total"
    cmap = _class_map(seg, task)
    wanted = [n for n in BODY_ARTERIES if n in cmap.values()]
    if not wanted:
        # an empty --roi_subset would run the full model and then keep nothing: stop clearly
        raise SegmenterError(f"{seg.title} task '{task}' lists none of the arteries Lumen uses; "
                             f"the installed version may not match the pinned fork")
    say(f"Finding arteries with {seg.title} on {device} ...")
    out = work / "body.nii.gz"
    write_nifti(scan, src)
    _run([exe, "-i", src, "-o", out, "--ml", "--task", task, "--device", device,
          "--roi_subset", *wanted], log, env=env, what=seg.title)
    return _body_labels(seg, out, task, neck=(src, log, exe, device, env, work), run=_run)


def _body_labels(seg: Segmenter, out: Path, task: str, neck=None, run=None):
    """Named arteries from the body output, plus the internal carotids from the neck output.

    ``neck`` is (src, log, exe, device, env, work) for the CT follow-up run; ``run`` performs it.
    """
    from lumen.assets.scan_import import LabelVolume

    lab, aff = _read_labels(out)
    cmap = _class_map(seg, task)
    wanted = [n for n in BODY_ARTERIES if n in cmap.values()]
    data = np.zeros(lab.shape, np.int16)
    names: dict = {}
    for idx, name in cmap.items():
        if name in wanted and np.any(lab == idx):
            names[len(names) + 1] = pretty(name)
            data[lab == idx] = len(names)
    if any("carotid" in v for v in names.values()) and task == "total" and neck is not None:
        src, log, exe, device, env, work = neck
        out2 = work / "neck.nii.gz"
        run([exe, "-i", src, "-o", out2, "--ml", "--task", "headneck_bones_vessels",
             "--device", device], log, env=env, what=seg.title)
        lab2, _ = _read_labels(out2)
        for idx, name in _class_map(seg, "headneck_bones_vessels").items():
            fresh = (lab2 == idx) & (data == 0)
            if name in NECK_ARTERIES and np.any(fresh):            # only if it adds voxels
                names[len(names) + 1] = pretty(name)
                data[fresh] = len(names)
    if not names:
        raise SegmenterError(f"{seg.title} found no arteries in this scan")
    return LabelVolume(data, aff, names)


def run_head(scan, work: Path, say=print, head_from_mm: float | None = None):
    """Run the RSNA 2025 intracranial model on ``scan``; return a ``LabelVolume`` of named
    arteries on the scan's own grid.

    ``head_from_mm`` (patient z, mm) is where the head starts in a scan that also shows the neck
    or trunk; the model is given only the part above it, as it was trained on head scans.
    """
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
    device = _device(seg)
    say(f"Finding arteries with {seg.title} on {device} ...")
    write_nifti(ScanVolume(data[tuple(sl)], sub_aff), src)
    runner = Path(__file__).with_name("_rsna_runner.py")
    _run([_python(seg), runner, "--scan", src, "--weights", weights_dir(seg), "--out", out,
          "--device", device], log, env=_torch_env(), what=seg.title)
    return _head_labels(out, seg, data.shape, A, sl)


def _head_labels(out: Path, seg: Segmenter, shape, aff: np.ndarray, sl):
    from lumen.assets.scan_import import LabelVolume

    lab, lab_aff = _read_labels(out)
    sub_aff = aff.copy()
    sub_aff[:3, 3] = (aff @ np.array([sl[0].start or 0, sl[1].start or 0, sl[2].start or 0, 1.0]))[:3]
    if not np.allclose(lab_aff, sub_aff, atol=1e-3) or lab.shape != tuple(shape[ax] if sl[ax] == slice(None) else sl[ax].stop - (sl[ax].start or 0) for ax in range(3)):
        raise SegmenterError(f"{seg.title} returned a different grid from the scan")
    full = np.zeros(shape, np.int16)
    full[tuple(sl)] = lab
    names = {i + 1: n for i, n in enumerate(HEAD_ARTERIES) if np.any(full == i + 1)}
    if not names:
        raise SegmenterError(f"{seg.title} found no arteries in this scan")
    return LabelVolume(full, aff, names)
