"""Import segmented cross-sectional anatomy into Lumen assets.

This module is deliberately dependency-light. The stable seam is:

    volume -> binary mask -> centerline/radius graph -> Asset

Heavy DICOM readers and ML segmenters can sit in optional wrappers, but the core
conversion path remains NumPy-only so imported masks are easy to test and replay.
"""

from __future__ import annotations

import csv
import json
import struct
import warnings
import zlib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

import numpy as np

from lumen.assets.schema import Asset, DeviceSpawn, Edge, Frame, Node


_MAX_PLANAR_FALLBACK_PIXELS = 262_144
_WARN_PLANAR_FALLBACK_PIXELS = 65_536


@dataclass
class Volume:
    data: np.ndarray
    spacing_mm: tuple = (1.0, 1.0, 1.0)
    origin_mm: tuple = (0.0, 0.0, 0.0)

    def __post_init__(self):
        self.data = np.asarray(self.data)
        if self.data.ndim != 3:
            raise ValueError(f"volume must be 3-D, got shape {self.data.shape}")
        self.spacing_mm = _triple(self.spacing_mm, "spacing_mm")
        self.origin_mm = _triple(self.origin_mm, "origin_mm")
        if any(s <= 0 for s in self.spacing_mm):
            raise ValueError(f"spacing_mm values must be positive, got {self.spacing_mm}")

    @property
    def mask(self):
        return np.asarray(self.data, dtype=bool)


@dataclass
class PlanarImage:
    data: np.ndarray
    pixel_spacing_mm: tuple = (1.0, 1.0)
    origin_mm: tuple = (0.0, 0.0, 0.0)
    direction: tuple = (1.0, 0.0, 0.0, 1.0)

    def __post_init__(self):
        self.data = np.asarray(self.data)
        if self.data.ndim != 2:
            raise ValueError(f"planar image must be 2-D, got shape {self.data.shape}")
        spacing = np.asarray(self.pixel_spacing_mm, dtype=float).reshape(-1)
        if len(spacing) != 2:
            raise ValueError("pixel_spacing_mm must have two values")
        if np.any(spacing <= 0):
            raise ValueError(f"pixel_spacing_mm values must be positive, got {tuple(spacing)}")
        self.pixel_spacing_mm = tuple(float(x) for x in spacing)
        self.origin_mm = _triple(self.origin_mm, "origin_mm")
        direction = np.asarray(self.direction, dtype=float).reshape(-1)
        root = int(round(np.sqrt(len(direction))))
        if root * root != len(direction) or root < 2:
            raise ValueError("direction must be a flattened square matrix with at least 4 values")
        if not np.isfinite(direction).all():
            raise ValueError("direction values must be finite")
        self.direction = tuple(float(x) for x in direction)

    @property
    def mask(self):
        return np.asarray(self.data, dtype=bool)


@dataclass(frozen=True)
class BoxAnnotation:
    """A 2-D vessel box annotation in image-pixel coordinates.

    Coordinates follow the usual image convention: x grows right, y grows down.
    ``group`` identifies one vessel segment/polyline; boxes with the same group
    are ordered into one edge. ``radius_mm`` can override the box-derived radius.
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    group: str = "vessel"
    order: float | None = None
    radius_mm: float | None = None


def _triple(value, name: str) -> tuple[float, float, float]:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if len(arr) != 3:
        raise ValueError(f"{name} must have three values")
    return tuple(float(x) for x in arr)


def load_npz_volume(path) -> Volume:
    """Load a 3-D ``.npz`` volume or mask.

    Expected arrays:
    - ``volume`` for raw intensity, or ``mask`` for an existing segmentation.
    - optional ``spacing_mm`` and ``origin_mm`` triples.
    """
    with np.load(Path(path), allow_pickle=False) as data:
        if "volume" in data:
            arr = data["volume"]
        elif "mask" in data:
            arr = data["mask"].astype(bool)
        else:
            raise ValueError("npz must contain a 'volume' or 'mask' array")
        spacing = data["spacing_mm"] if "spacing_mm" in data else (1.0, 1.0, 1.0)
        origin = data["origin_mm"] if "origin_mm" in data else (0.0, 0.0, 0.0)
    return Volume(arr, spacing_mm=spacing, origin_mm=origin)


def _spacing2(value) -> tuple[float, float]:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if len(arr) < 2:
        raise ValueError("spacing metadata must have at least two values")
    return float(arr[0]), float(arr[1])


def _dicom_meta(image, key: str) -> str | None:
    if hasattr(image, "HasMetaDataKey") and hasattr(image, "GetMetaData"):
        try:
            if image.HasMetaDataKey(key):
                return str(image.GetMetaData(key)).strip()
        except RuntimeError:
            return None
    return None


def _dicom_first_float(image, key: str) -> float | None:
    raw = _dicom_meta(image, key)
    if raw is None:
        return None
    first = raw.split("\\")[0].strip()
    try:
        return float(first)
    except ValueError:
        return None


def _apply_dicom_presentation(arr: np.ndarray, image) -> np.ndarray:
    slope = _dicom_first_float(image, "0028|1053")  # Rescale Slope
    intercept = _dicom_first_float(image, "0028|1052")  # Rescale Intercept
    if slope is not None or intercept is not None:
        arr = arr.astype(np.float32, copy=False) * (1.0 if slope is None else slope)
        if intercept is not None:
            arr = arr + intercept
    center = _dicom_first_float(image, "0028|1050")  # Window Center
    width = _dicom_first_float(image, "0028|1051")   # Window Width
    if center is None or width is None or width <= 0.0:
        return arr

    lo = center - 0.5 * width
    hi = center + 0.5 * width
    out = np.clip((arr.astype(np.float32, copy=False) - lo) / (hi - lo), 0.0, 1.0)
    photometric = (_dicom_meta(image, "0028|0004") or "").upper()
    if photometric == "MONOCHROME1":
        out = 1.0 - out
    return out.astype(np.float32, copy=False)


def _read_dicom_frame_image(sitk, path: Path):
    if not hasattr(sitk, "ImageFileReader"):
        return sitk.ReadImage(str(path)), None
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    if hasattr(reader, "LoadPrivateTagsOn"):
        reader.LoadPrivateTagsOn()
    if hasattr(reader, "ReadImageInformation"):
        reader.ReadImageInformation()
    return reader.Execute(), reader


def load_planar_array(path, frame_index: int | None = None) -> PlanarImage:
    """Load a 2-D `.png`, `.npy`, `.npz`, or DICOM image/mask frame with metadata.

    `.npz` inputs may contain `pixel_spacing_mm` or `spacing_mm` plus optional
    `origin_mm`. PNG previews are dependency-free 1/2/4/8/16-bit grayscale and
    8/16-bit RGB/RGBA reads. Indexed-color PNGs are interpreted as label maps
    where raw palette indices are meaningful; use Pillow upstream for large PNGs
    or regular palette-image color conversion.
    Multi-frame DICOM cine inputs require an explicit ``frame_index``. DICOM loading
    uses SimpleITK when the optional imaging extra is installed.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".png":
        if frame_index is not None:
            raise ValueError(
                "PNG planar import supports only one frame; frame_index is not applicable"
            )
        return PlanarImage(_read_png_planar(path))
    if suffix == ".npy":
        arr = _select_planar_frame(np.load(path, allow_pickle=False), frame_index, "NPY")
        return PlanarImage(arr)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if "mask" in data:
                arr = data["mask"]
            elif "volume" in data:
                arr = data["volume"]
            else:
                raise ValueError("input must contain a 'mask' or 'volume' array")
            if "pixel_spacing_mm" in data:
                spacing = _spacing2(data["pixel_spacing_mm"])
            elif "spacing_mm" in data:
                spacing = _spacing2(data["spacing_mm"])
            else:
                spacing = (1.0, 1.0)
            origin = data["origin_mm"] if "origin_mm" in data else (0.0, 0.0, 0.0)
            arr = _select_planar_frame(arr, frame_index, "NPZ")
        return PlanarImage(arr, pixel_spacing_mm=spacing, origin_mm=origin)
    return load_dicom_frame(path, frame_index=frame_index)


def _read_png_planar(path: Path) -> np.ndarray:
    # Keep PNG previews dependency-free for the core package; users that already
    # depend on Pillow can decode upstream and pass arrays into the planar APIs.
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"{path}: not a PNG file")
    pos = 8
    width = height = bit_depth = color_type = interlace = None
    payload = b""
    # Minimal PNG chunk walk: IHDR defines layout; IDAT chunks form one zlib stream.
    while pos < len(data):
        if pos + 8 > len(data):
            raise ValueError(f"{path}: truncated PNG chunk header")
        n = int.from_bytes(data[pos:pos + 4], "big")
        typ = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + n]
        pos += 12 + n
        if typ == b"IHDR":
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack(
                ">IIBBBBB", chunk
            )
        elif typ == b"IDAT":
            payload += chunk
        elif typ == b"IEND":
            break
    if None in (width, height, bit_depth, color_type, interlace):
        raise ValueError(f"{path}: PNG is missing IHDR")
    allowed_depths = {0: {1, 2, 4, 8, 16}, 2: {8, 16}, 3: {1, 2, 4, 8}, 4: {8, 16}, 6: {8, 16}}
    if bit_depth not in allowed_depths.get(int(color_type), set()):
        raise ValueError(f"{path}: unsupported PNG bit depth {bit_depth} for color type "
                         f"{color_type}")
    if interlace != 0:
        raise ValueError(f"{path}: interlaced PNG previews are not supported")
    channels_by_type = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
    channels = channels_by_type.get(int(color_type))
    if channels is None:
        raise ValueError(f"{path}: unsupported PNG color type {color_type}")
    try:
        raw = zlib.decompress(payload)
    except zlib.error as e:
        raise ValueError(f"{path}: PNG decompression failed: {e}") from e
    # PNG scanlines carry a leading filter byte; undo filters before sample unpacking.
    rows = _png_unfilter(raw, int(width), int(height), int(channels), int(bit_depth))
    if bit_depth < 8:
        samples = _png_unpack_subbyte_samples(rows, int(width), int(height),
                                              int(channels), int(bit_depth))
        max_value = float((1 << int(bit_depth)) - 1)
        out_dtype = np.uint8
    elif bit_depth == 16:
        samples = np.frombuffer(rows.tobytes(), dtype=">u2").reshape(
            int(height), int(width), int(channels)
        ).astype(np.uint16)
        max_value = 65535.0
        out_dtype = np.uint16
    else:
        samples = rows.reshape(int(height), int(width), int(channels))
        max_value = 255.0
        out_dtype = np.uint8
    if color_type == 0:
        return samples[:, :, 0]
    if color_type == 3:
        return samples[:, :, 0]
    if color_type == 4:
        return samples[:, :, 0]
    rgb = samples[:, :, :3].astype(np.float32)
    return np.clip(0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2],
                   0, max_value).astype(out_dtype)


def _png_unfilter(raw: bytes, width: int, height: int, channels: int,
                  bit_depth: int) -> np.ndarray:
    bits_per_pixel = channels * bit_depth
    filter_stride = max(1, (bits_per_pixel + 7) // 8)
    row_len = (width * bits_per_pixel + 7) // 8
    expected = height * (row_len + 1)
    if len(raw) != expected:
        raise ValueError(f"PNG decompressed data has {len(raw)} bytes; expected {expected}")
    out = np.zeros((height, row_len), dtype=np.uint8)
    offset = 0
    for y in range(height):
        filter_type = raw[offset]
        offset += 1
        row = np.frombuffer(raw[offset:offset + row_len], dtype=np.uint8).astype(np.int16)
        offset += row_len
        left = np.zeros(row_len, dtype=np.int16)
        left[filter_stride:] = row[:-filter_stride]
        up = out[y - 1].astype(np.int16) if y else np.zeros(row_len, dtype=np.int16)
        up_left = np.zeros(row_len, dtype=np.int16)
        if y:
            up_left[filter_stride:] = out[y - 1, :-filter_stride].astype(np.int16)
        if filter_type == 0:
            recon = row
        elif filter_type == 1:
            recon = row + left
        elif filter_type == 2:
            recon = row + up
        elif filter_type == 3:
            recon = row + ((left + up) // 2)
        elif filter_type == 4:
            recon = row + _png_paeth(left, up, up_left)
        else:
            raise ValueError(f"PNG uses unsupported row filter {filter_type}")
        out[y] = np.asarray(recon % 256, dtype=np.uint8)
    return out


def _png_unpack_subbyte_samples(rows: np.ndarray, width: int, height: int, channels: int,
                                bit_depth: int) -> np.ndarray:
    bits = np.unpackbits(rows, axis=1, bitorder="big")
    sample_count = width * channels
    sample_bits = bits[:, :sample_count * bit_depth].reshape(height, sample_count, bit_depth)
    weights = (1 << np.arange(bit_depth - 1, -1, -1, dtype=np.uint8)).reshape(1, 1, bit_depth)
    samples = np.sum(sample_bits * weights, axis=2, dtype=np.uint8)
    return samples.reshape(height, width, channels)


def _png_paeth(left: np.ndarray, up: np.ndarray, up_left: np.ndarray) -> np.ndarray:
    p = left + up - up_left
    pa = np.abs(p - left)
    pb = np.abs(p - up)
    pc = np.abs(p - up_left)
    return np.where((pa <= pb) & (pa <= pc), left, np.where(pb <= pc, up, up_left))


def _select_planar_frame(arr, frame_index: int | None, label: str) -> np.ndarray:
    data = np.asarray(arr)
    if data.ndim == 3:
        if frame_index is None:
            raise ValueError(
                f"{label} planar import got {data.shape[0]} frames; "
                "pass frame_index to select one"
            )
        idx = int(frame_index)
        if idx < 0 or idx >= data.shape[0]:
            raise ValueError(
                f"{label} frame_index {idx} is out of range for {data.shape[0]} frames"
            )
        return data[idx]
    return data


def load_dicom_frame(path, frame_index: int | None = None) -> PlanarImage:
    """Load one DICOM frame as a 2-D `PlanarImage`.

    Multi-slice series should use `load_dicom_series`; this helper is for single
    angiography/fluoro frames that are then thresholded or segmented to planar assets.
    Multi-frame cine DICOMs require ``frame_index`` so imports do not silently pick
    an arbitrary frame.
    If VOI window tags are present, pixel data is converted to display-ready
    ``float32`` in ``[0, 1]`` and MONOCHROME1 polarity is inverted for previews.
    """
    try:
        import SimpleITK as sitk
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError("load_dicom_frame requires the optional SimpleITK dependency") from e

    image, metadata_source = _read_dicom_frame_image(sitk, Path(path))
    metadata_source = metadata_source or image
    dim = int(image.GetDimension())
    direction = np.asarray(image.GetDirection(), float).reshape(dim, dim)
    if not np.isfinite(direction).all():
        raise ValueError("DICOM direction cosines must be finite")
    arr = _apply_dicom_presentation(np.asarray(sitk.GetArrayFromImage(image)), metadata_source)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 3:
        if frame_index is None:
            raise ValueError(
                f"DICOM frame import got {arr.shape[0]} frames; pass frame_index to select one"
            )
        idx = int(frame_index)
        if idx < 0 or idx >= arr.shape[0]:
            raise ValueError(
                f"DICOM frame_index {idx} is out of range for {arr.shape[0]} frames"
            )
        arr = arr[idx]
    if arr.ndim != 2:
        raise ValueError(f"DICOM frame import expects one 2-D frame, got array shape {arr.shape}")
    spacing = _spacing2(image.GetSpacing())
    origin_raw = np.asarray(image.GetOrigin(), dtype=float).reshape(-1)
    origin_values = []
    for axis_index in range(3):
        if axis_index < len(origin_raw):
            origin_values.append(float(origin_raw[axis_index]))
        else:
            origin_values.append(0.0)
    origin = tuple(origin_values)
    return PlanarImage(arr, pixel_spacing_mm=spacing, origin_mm=origin,
                       direction=tuple(float(x) for x in direction.reshape(-1)))


def load_box_annotations(path, group_key: str = "group",
                         image_size_px: tuple[float, float] | None = None,
                         image_id=None,
                         image_file: str | None = None,
                         ) -> list[BoxAnnotation]:
    """Load 2-D vessel box annotations from JSON or CSV.

    JSON may be a list of boxes, an object with a ``boxes`` array, COCO-style
    ``annotations`` with ``bbox=[x,y,width,height]``, polygon ``segmentation``,
    or uncompressed RLE ``segmentation`` masks, Label Studio rectangle exports,
    LabelMe rectangle/polygon shapes, or VGG Image Annotator rectangle regions.
    XML may be a CVAT or Pascal VOC image annotation export. Use
    ``image_id`` or ``image_file`` for multi-image COCO exports.
    CSV files need headers.
    `.txt` files are read as YOLO normalized labels
    ``class x_center y_center width height`` and require ``image_size_px=(width,height)``.
    Accepted coordinate names are ``x_min/y_min/x_max/y_max``, ``x0/y0/x1/y1``,
    or ``left/top/width/height``.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"invalid JSON in {path}: {e}") from e
        rows = _box_rows_from_json(payload, group_key=group_key,
                                   image_size_px=image_size_px,
                                   image_id=image_id,
                                   image_file=image_file)
        if not isinstance(rows, list):
            raise ValueError("box JSON must be a list or an object with a 'boxes' list")
        return [_box_from_mapping(row, group_key=group_key) for row in rows]
    if suffix == ".xml":
        rows = _xml_box_rows(path, group_key=group_key, image_file=image_file)
        return [_box_from_mapping(row, group_key=group_key) for row in rows]
    if suffix in {".txt", ".labels"}:
        return _load_yolo_box_annotations(path, image_size_px=image_size_px)
    with path.open(newline="") as f:
        return [_box_from_mapping(row, group_key=group_key) for row in csv.DictReader(f)]


def load_dicom_series(path) -> Volume:
    """Load a DICOM series with SimpleITK when the optional imaging extra is present."""
    try:
        import SimpleITK as sitk
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError("load_dicom_series requires the optional SimpleITK dependency") from e

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(path))
    if not series_ids:
        raise ValueError(f"no DICOM series found under {path!r}")
    files = reader.GetGDCMSeriesFileNames(str(path), series_ids[0])
    reader.SetFileNames(files)
    image = reader.Execute()
    direction = np.asarray(image.GetDirection(), float).reshape(3, 3)
    if not np.allclose(direction, np.eye(3), atol=1e-6):
        raise ValueError("DICOM direction cosines are not identity; resample to an "
                         "axis-aligned volume before importing into the current Asset schema")
    # SimpleITK arrays are z,y,x; Lumen volumes use x,y,z.
    data = np.asarray(sitk.GetArrayFromImage(image)).transpose(2, 1, 0)
    return Volume(data, spacing_mm=image.GetSpacing(), origin_mm=image.GetOrigin())


def box_annotation_preview(image, boxes) -> np.ndarray:
    """Return an RGB QA overlay for box annotations on a 2-D source image.

    The source image is contrast-normalized to grayscale. Box borders are red and
    the ordered centerline implied by each group is green, giving import users a
    quick visual check before they trust the generated asset graph.
    """
    frame = np.asarray(image.data if isinstance(image, PlanarImage) else image)
    if frame.ndim != 2:
        raise ValueError(f"box preview expects a 2-D image, got shape {frame.shape}")
    gray = _display01(frame)
    rgb = np.repeat((0.55 * gray)[..., None], 3, axis=2)
    annotations = [_coerce_box(b) for b in boxes]
    h, w = rgb.shape[:2]
    by_group: dict[str, list[BoxAnnotation]] = {}
    for box in annotations:
        _draw_box(rgb, box, color=(1.0, 0.05, 0.05), width=2)
        by_group.setdefault(box.group, []).append(box)
    for group_boxes in by_group.values():
        centers = np.asarray([_box_center_radius_px(b)[0] for b in _order_boxes(group_boxes)],
                             dtype=float)
        if len(centers) == 1:
            _draw_disk(rgb, centers[0, 0], centers[0, 1], radius=2, color=(0.1, 1.0, 0.25))
        else:
            for a, b in zip(centers[:-1], centers[1:]):
                _draw_line(rgb, a[0], a[1], b[0], b[1], color=(0.1, 1.0, 0.25), width=2)
    return np.clip(rgb[:h, :w], 0.0, 1.0)


def planar_mask_asset_preview(image, mask, asset: Asset) -> np.ndarray:
    """Return an RGB QA overlay for a 2-D mask import and extracted asset graph."""
    frame = np.asarray(image.data if isinstance(image, PlanarImage) else image)
    if frame.ndim != 2:
        raise ValueError(f"mask preview expects a 2-D image, got shape {frame.shape}")
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != frame.shape:
        raise ValueError(f"mask preview shape {mask.shape} does not match image shape {frame.shape}")
    gray = _display01(frame)
    rgb = np.repeat((0.55 * gray)[..., None], 3, axis=2)
    rgb[mask, 1] = np.maximum(rgb[mask, 1], 0.85)

    spacing = np.asarray(asset.frame.spacing_mm[:2], dtype=float)
    origin = np.asarray(asset.frame.origin_mm[:2], dtype=float)
    if np.any(spacing <= 0):
        spacing = np.array([1.0, 1.0], dtype=float)
    for edge in asset.edges:
        pts = np.asarray(edge.centerline_mm, dtype=float)
        if len(pts) == 0:
            continue
        px = (pts[:, 0] - origin[0]) / spacing[0]
        py = (pts[:, 1] - origin[1]) / spacing[1]
        for a, b in zip(np.stack([px, py], axis=1)[:-1], np.stack([px, py], axis=1)[1:]):
            _draw_line(rgb, a[0], a[1], b[0], b[1], color=(1.0, 0.05, 0.05), width=2)
        _draw_disk(rgb, px[0], py[0], radius=2, color=(1.0, 1.0, 1.0))
    return np.clip(rgb, 0.0, 1.0)


def asset_planar_import_report(asset: Asset, source: str | None = None,
                               image_shape_px: tuple[int, int] | None = None,
                               annotation_count: int | None = None,
                               preview_image: str | None = None,
                               annotation_selector: dict | None = None,
                               image_direction=None) -> dict:
    """Return JSON-ready QA metrics for a 2-D imported asset graph."""
    degree = {node.id: 0 for node in asset.nodes}
    all_points = []
    all_radii = []
    total_length = 0.0
    report_warnings = []
    for edge in asset.edges:
        degree[edge.node_a] = degree.get(edge.node_a, 0) + 1
        degree[edge.node_b] = degree.get(edge.node_b, 0) + 1
        pts = np.asarray(edge.centerline_mm, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
            report_warnings.append(f"{edge.id}: empty or invalid centerline")
            continue
        if not np.isfinite(pts).all():
            report_warnings.append(f"{edge.id}: non-finite centerline coordinate")
        all_points.append(pts)
        if len(pts) > 1:
            total_length += float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        radii = np.asarray(edge.R, dtype=float).reshape(-1)
        if radii.size == 0:
            report_warnings.append(f"{edge.id}: missing radius samples")
        elif np.any(radii <= 0.0) or not np.isfinite(radii).all():
            report_warnings.append(f"{edge.id}: non-positive or non-finite radius sample")
        if radii.size:
            all_radii.append(radii)

    points = np.concatenate(all_points, axis=0) if all_points else np.zeros((0, 3), dtype=float)
    radii = np.concatenate(all_radii, axis=0) if all_radii else np.zeros(0, dtype=float)
    spacing = np.asarray(asset.frame.spacing_mm[:2], dtype=float)
    origin = np.asarray(asset.frame.origin_mm[:2], dtype=float)
    shape = None
    if image_shape_px is not None:
        shape_arr = np.asarray(image_shape_px, dtype=int).reshape(-1)
        if len(shape_arr) != 2:
            raise ValueError("image_shape_px must be (height, width)")
        shape = [int(shape_arr[0]), int(shape_arr[1])]
        if len(points):
            px = (points[:, 0] - origin[0]) / spacing[0]
            py = (points[:, 1] - origin[1]) / spacing[1]
            outside = (
                (px < -0.5)
                | (py < -0.5)
                | (px > shape[1] - 0.5)
                | (py > shape[0] - 0.5)
            )
            if np.any(outside):
                report_warnings.append("centerline: points fall outside supplied image_shape_px")
    direction = None
    if image_direction is not None:
        direction_arr = np.asarray(image_direction, dtype=float).reshape(-1)
        root = int(round(np.sqrt(len(direction_arr))))
        if root * root != len(direction_arr) or root < 2:
            raise ValueError("image_direction must be a flattened square matrix")
        if not np.isfinite(direction_arr).all():
            raise ValueError("image_direction values must be finite")
        direction = [float(x) for x in direction_arr]

    if len(points):
        bounds = {
            "x": [round(float(points[:, 0].min()), 4), round(float(points[:, 0].max()), 4)],
            "y": [round(float(points[:, 1].min()), 4), round(float(points[:, 1].max()), 4)],
            "z": [round(float(points[:, 2].min()), 4), round(float(points[:, 2].max()), 4)],
        }
    else:
        bounds = {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]}
    if radii.size:
        radius_summary = {
            "min": round(float(radii.min()), 4),
            "max": round(float(radii.max()), 4),
            "mean": round(float(radii.mean()), 4),
        }
    else:
        radius_summary = {"min": 0.0, "max": 0.0, "mean": 0.0}

    report = {
        "ok": not report_warnings,
        "source": str(source) if source is not None else None,
        "preview_image": str(preview_image) if preview_image is not None else None,
        "image_shape_px": shape,
        "image_direction": direction,
        "annotation_count": int(annotation_count) if annotation_count is not None else None,
        "annotation_selector": annotation_selector,
        "frame": {
            "name": asset.frame.name,
            "spacing_mm": [float(x) for x in asset.frame.spacing_mm],
            "origin_mm": [float(x) for x in asset.frame.origin_mm],
        },
        "nodes": int(len(asset.nodes)),
        "edges": int(len(asset.edges)),
        "branch_nodes": int(sum(1 for count in degree.values() if count >= 3)),
        "terminal_nodes": int(sum(1 for count in degree.values() if count == 1)),
        "total_centerline_mm": round(float(total_length), 4),
        "bounds_mm": bounds,
        "radius_mm": radius_summary,
        "warnings": report_warnings,
    }
    return report


def _display01(frame) -> np.ndarray:
    arr = np.asarray(frame, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=float)
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        return np.zeros_like(arr, dtype=float)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _draw_disk(img: np.ndarray, x: float, y: float, radius: int, color) -> None:
    h, w = img.shape[:2]
    cx, cy = int(round(x)), int(round(y))
    rr = int(max(1, radius))
    y0, y1 = max(0, cy - rr), min(h, cy + rr + 1)
    x0, x1 = max(0, cx - rr), min(w, cx + rr + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= rr ** 2
    img[y0:y1, x0:x1][mask] = color


def _draw_line(img: np.ndarray, x0: float, y0: float, x1: float, y1: float,
               color, width: int = 1) -> None:
    span_px = max(abs(float(x1) - float(x0)), abs(float(y1) - float(y0)))
    steps = max(2, int(np.ceil(span_px * 1.5)))
    for t in np.linspace(0.0, 1.0, steps):
        _draw_disk(img, (1.0 - t) * x0 + t * x1, (1.0 - t) * y0 + t * y1,
                   radius=width, color=color)


def _draw_box(img: np.ndarray, box: BoxAnnotation, color, width: int = 1) -> None:
    x0 = float(np.clip(box.x_min, 0, img.shape[1] - 1))
    x1 = float(np.clip(box.x_max, 0, img.shape[1] - 1))
    y0 = float(np.clip(box.y_min, 0, img.shape[0] - 1))
    y1 = float(np.clip(box.y_max, 0, img.shape[0] - 1))
    _draw_line(img, x0, y0, x1, y0, color=color, width=width)
    _draw_line(img, x1, y0, x1, y1, color=color, width=width)
    _draw_line(img, x1, y1, x0, y1, color=color, width=width)
    _draw_line(img, x0, y1, x0, y0, color=color, width=width)


def segment_threshold(volume: Volume, threshold: float, foreground: str = "above") -> Volume:
    """Segment a volume by thresholding.

    This is a baseline backend, not a clinical segmenter. It is useful for synthetic
    CTA/MRA/airway fixtures and as the common output shape for model-backed segmenters.
    """
    if foreground == "above":
        mask = np.asarray(volume.data) >= float(threshold)
    elif foreground == "below":
        mask = np.asarray(volume.data) <= float(threshold)
    else:
        raise ValueError("foreground must be 'above' or 'below'")
    return Volume(mask, spacing_mm=volume.spacing_mm, origin_mm=volume.origin_mm)


@dataclass
class _SliceComponent:
    id: int
    z: int
    pixels: frozenset
    center_vox: np.ndarray
    center_mm: np.ndarray
    radius_mm: float


def asset_from_mask(mask, spacing_mm=(1.0, 1.0, 1.0), origin_mm=(0.0, 0.0, 0.0),
                    min_component_voxels: int = 4,
                    provenance: str = "segmented(imported)") -> Asset:
    """Convert an axial binary vessel/airway mask into a Lumen ``Asset``.

    The extractor is intentionally conservative: it follows connected components in
    adjacent axial slices, estimates each cross-section's equivalent circular radius,
    and compresses degree-2 runs into graph edges. It is a good import seam for
    already-segmented demo volumes; arbitrary tortuous clinical centerlines can later
    replace this extractor behind the same ``Asset`` boundary.
    """
    vol = Volume(np.asarray(mask, dtype=bool), spacing_mm=spacing_mm, origin_mm=origin_mm)
    if min_component_voxels <= 0:
        raise ValueError("min_component_voxels must be positive")
    comps = _slice_components(vol, min_component_voxels)
    if not comps:
        raise ValueError("mask contains no components large enough to import")
    links = _link_components(comps, vol.spacing_mm)
    return _components_to_asset(comps, links, vol.spacing_mm, vol.origin_mm, provenance)


def asset_from_box_annotations(boxes, pixel_spacing_mm=(1.0, 1.0),
                               origin_mm=(0.0, 0.0, 0.0), z_mm: float = 0.0,
                               merge_tolerance_mm: float = 2.0,
                               min_boxes_per_edge: int = 2,
                               provenance: str = "2d-boxes(imported)") -> Asset:
    """Convert 2-D vessel boxes into a planar Lumen ``Asset``.

    This is the lightweight import path for angiography/video annotations: a row of
    boxes traces each visible vessel segment, the box centers become the edge
    centerline, and the short box side estimates the local lumen radius. Grouped
    segments whose endpoints fall within ``merge_tolerance_mm`` share a branch node.
    Groups smaller than ``min_boxes_per_edge`` raise instead of being skipped so
    incomplete annotation exports fail visibly.
    """
    annotations = [_coerce_box(b) for b in boxes]
    if not annotations:
        raise ValueError("at least one box annotation is required")
    spacing_xy = np.asarray(pixel_spacing_mm, dtype=float).reshape(-1)
    if len(spacing_xy) != 2:
        raise ValueError("pixel_spacing_mm must have two values: (x, y)")
    if np.any(spacing_xy <= 0):
        raise ValueError(f"pixel_spacing_mm values must be positive, got {tuple(spacing_xy)}")
    origin = np.asarray(_triple(origin_mm, "origin_mm"), dtype=float)
    merge_tolerance_mm = float(merge_tolerance_mm)
    if merge_tolerance_mm < 0:
        raise ValueError("merge_tolerance_mm must be >= 0")
    if min_boxes_per_edge < 1:
        raise ValueError("min_boxes_per_edge must be >= 1")

    by_group: dict[str, list[BoxAnnotation]] = {}
    for box in annotations:
        by_group.setdefault(str(box.group), []).append(box)

    nodes: list[Node] = []
    node_positions: list[np.ndarray] = []

    def node_for(point: np.ndarray) -> str:
        for i, pos in enumerate(node_positions):
            if float(np.linalg.norm(point - pos)) <= merge_tolerance_mm:
                return nodes[i].id
        nid = f"n{len(nodes)}"
        nodes.append(Node(nid, tuple(float(x) for x in point)))
        node_positions.append(point.copy())
        return nid

    edges: list[Edge] = []
    spawn_node = None
    for group, group_boxes in sorted(by_group.items(), key=lambda kv: kv[0]):
        if len(group_boxes) < min_boxes_per_edge:
            raise ValueError(
                f"group {group!r} has {len(group_boxes)} boxes; expected at least "
                f"{min_boxes_per_edge}")
        ordered = _order_boxes(group_boxes)
        pts = []
        radii = []
        for box in ordered:
            center_px, radius_px = _box_center_radius_px(box)
            p = origin + np.array([
                center_px[0] * spacing_xy[0],
                center_px[1] * spacing_xy[1],
                float(z_mm),
            ])
            pts.append(p)
            if box.radius_mm is None:
                radii.append(max(radius_px * float(min(spacing_xy)), 0.5 * float(min(spacing_xy))))
            else:
                radii.append(float(box.radius_mm))
        pts = np.asarray(pts, float)
        radii = np.asarray(radii, float)
        if np.any(radii <= 0) or not np.isfinite(radii).all():
            raise ValueError(f"group {group!r} has non-positive or non-finite radius")
        node_a = node_for(pts[0])
        node_b = node_for(pts[-1])
        if node_a == node_b and len(pts) > 1:
            raise ValueError(f"group {group!r} collapses to one merged endpoint")
        if spawn_node is None or tuple(pts[0][:2]) < tuple(node_positions[int(spawn_node[1:])][:2]):
            spawn_node = node_a
        edges.append(_edge_for_points(f"e{len(edges)}", node_a, node_b, pts, radii))

    if spawn_node is None:
        raise ValueError("no edges were generated from boxes")
    return Asset(
        frame=Frame(name="image_xy_mm",
                    spacing_mm=(float(spacing_xy[0]), float(spacing_xy[1]), 1.0),
                    origin_mm=tuple(float(x) for x in origin)),
        nodes=nodes,
        edges=edges,
        device_spawn=DeviceSpawn(node_id=spawn_node),
        provenance=provenance,
    )


def asset_from_planar_mask(mask2d, pixel_spacing_mm=(1.0, 1.0),
                           origin_mm=(0.0, 0.0, 0.0), z_mm: float = 0.0,
                           samples: int = 48, min_component_pixels: int = 16,
                           provenance: str = "2d-mask(imported)") -> Asset:
    """Convert a segmented 2-D angiography frame mask into a planar Lumen asset.

    Each connected component is thinned to a centerline skeleton, converted to a
    branch graph, and assigned a local radius from the mask distance field. If a
    component is too small or degenerate for graph tracing, it falls back to a
    principal-axis fit. This gives model- or human-segmented DICOM frames the same
    asset seam as boxes and 3-D masks while preserving visible bifurcations.
    """
    mask = np.asarray(mask2d, bool)
    if mask.ndim != 2:
        raise ValueError(f"planar mask must be 2-D, got shape {mask.shape}")
    spacing_xy = np.asarray(pixel_spacing_mm, dtype=float).reshape(-1)
    if len(spacing_xy) != 2:
        raise ValueError("pixel_spacing_mm must have two values: (x, y)")
    if np.any(spacing_xy <= 0):
        raise ValueError(f"pixel_spacing_mm values must be positive, got {tuple(spacing_xy)}")
    origin = np.asarray(_triple(origin_mm, "origin_mm"), dtype=float)
    if samples < 2:
        raise ValueError("samples must be >= 2")
    if min_component_pixels <= 0:
        raise ValueError("min_component_pixels must be positive")

    components = [
        pixels for pixels in _connected_components_2d(mask)
        if len(pixels) >= int(min_component_pixels)
    ]
    if not components:
        raise ValueError("planar mask contains no components large enough to import")

    nodes: list[Node] = []
    edges: list[Edge] = []
    spawn_node = None
    min_radius = 0.5 * float(min(spacing_xy))

    def add_node(point: np.ndarray) -> str:
        nid = f"n{len(nodes)}"
        nodes.append(Node(nid, tuple(float(x) for x in point)))
        return nid

    def add_edge(node_a: str, node_b: str, pts, radii) -> None:
        if node_a == node_b:
            return
        edges.append(_edge_for_points(f"e{len(edges)}", node_a, node_b, pts, radii))

    for pixels in sorted(components, key=lambda p: min(p)):
        graph = _component_planar_skeleton_graph(
            pixels, spacing_xy=spacing_xy, origin=origin, z_mm=float(z_mm),
            min_radius=min_radius,
        )
        if graph is None:
            pts, radii = _component_planar_centerline(
                pixels, spacing_xy=spacing_xy, origin=origin, z_mm=float(z_mm),
                samples=int(samples),
            )
            if len(pts) < 2:
                continue
            node_a = add_node(pts[0])
            node_b = add_node(pts[-1])
            add_edge(node_a, node_b, pts, radii)
            first_node = node_a
        else:
            first_cluster = _planar_graph_spawn_cluster(graph)
            node_map = {cluster_id: add_node(point)
                        for cluster_id, point in graph["nodes"].items()}
            for segment in graph["segments"]:
                node_a = node_map[segment["node_a"]]
                node_b = node_map[segment["node_b"]]
                pts = segment["points"]
                radii = segment["radii"]
                if segment["node_b"] == first_cluster:
                    node_a, node_b = node_b, node_a
                    pts = np.asarray(pts, dtype=float)[::-1]
                    radii = np.asarray(radii, dtype=float)[::-1]
                add_edge(node_a, node_b, pts, radii)
            first_node = node_map[first_cluster]
        node_positions = {node.id: node.position_mm for node in nodes}
        first_pos = np.asarray(node_positions[first_node], float)
        if spawn_node is None:
            spawn_node = first_node
            continue
        spawn_pos = np.asarray(node_positions[spawn_node], float)
        if tuple(first_pos[:2]) < tuple(spawn_pos[:2]):
            spawn_node = first_node

    if not edges:
        raise ValueError("planar mask components did not produce any centerline edges")
    return Asset(
        frame=Frame(name="image_xy_mm",
                    spacing_mm=(float(spacing_xy[0]), float(spacing_xy[1]), 1.0),
                    origin_mm=tuple(float(x) for x in origin)),
        nodes=nodes,
        edges=edges,
        device_spawn=DeviceSpawn(node_id=spawn_node),
        provenance=provenance,
    )


def _planar_graph_spawn_cluster(graph: dict) -> int:
    degree = {cluster_id: 0 for cluster_id in graph["nodes"]}
    for segment in graph["segments"]:
        degree[segment["node_a"]] += 1
        degree[segment["node_b"]] += 1
    terminals = [cluster_id for cluster_id, deg in degree.items() if deg == 1]
    candidates = terminals or list(graph["nodes"])
    if any(deg > 2 for deg in degree.values()):
        return min(candidates, key=lambda cid: (graph["nodes"][cid][1], graph["nodes"][cid][0]))
    return min(candidates, key=lambda cid: (graph["nodes"][cid][0], graph["nodes"][cid][1]))


def _component_planar_skeleton_graph(pixels, spacing_xy, origin, z_mm: float,
                                     min_radius: float):
    rows = [p[0] for p in pixels]
    cols = [p[1] for p in pixels]
    pad = 2
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    local = np.zeros((r1 - r0 + 1 + 2 * pad, c1 - c0 + 1 + 2 * pad), dtype=bool)
    for r, c in pixels:
        local[r - r0 + pad, c - c0 + pad] = True

    skeleton = _thin_skeleton(local)
    skel_pixels = {tuple(int(x) for x in p) for p in np.argwhere(skeleton)}
    if len(skel_pixels) < 2:
        return None
    radius_map = _distance_to_background_mm(local, spacing_xy)

    neighbors = {p: [q for q in _neighbors8(p, skeleton.shape) if q in skel_pixels]
                 for p in skel_pixels}
    special_pixels = {p for p, ns in neighbors.items() if len(ns) != 2}
    if len(special_pixels) < 2:
        return None

    clusters = _connected_special_clusters(special_pixels, skeleton.shape)
    special_to_cluster = {
        p: cluster_id
        for cluster_id, cluster in enumerate(clusters)
        for p in cluster
    }

    def pixel_point(p):
        r = p[0] + r0 - pad
        c = p[1] + c0 - pad
        return np.array([
            origin[0] + c * spacing_xy[0],
            origin[1] + r * spacing_xy[1],
            origin[2] + z_mm,
        ], dtype=float)

    def pixel_radius(p):
        return max(float(radius_map[p]), min_radius)

    graph_nodes = {}
    for cluster_id, cluster in enumerate(clusters):
        pts = np.asarray([pixel_point(p) for p in cluster], dtype=float)
        graph_nodes[cluster_id] = pts.mean(axis=0)

    segments = []
    visited_edges: set[frozenset] = set()
    for cluster_id, cluster in enumerate(clusters):
        for start in sorted(cluster):
            for cur in sorted(neighbors[start]):
                edge_key = frozenset((start, cur))
                if edge_key in visited_edges:
                    continue
                if cur in special_to_cluster:
                    other = special_to_cluster[cur]
                    if other == cluster_id:
                        visited_edges.add(edge_key)
                        continue
                    visited_edges.add(edge_key)
                    pts = np.vstack([graph_nodes[cluster_id], graph_nodes[other]])
                    radii = [
                        max(pixel_radius(start), min_radius),
                        max(pixel_radius(cur), min_radius),
                    ]
                    segments.append({"node_a": cluster_id, "node_b": other,
                                     "points": pts, "radii": radii})
                    continue

                prev = start
                path = [cur]
                visited_edges.add(edge_key)
                while cur not in special_to_cluster:
                    choices = [n for n in sorted(neighbors[cur]) if n != prev]
                    if not choices:
                        break
                    nxt = choices[0]
                    visited_edges.add(frozenset((cur, nxt)))
                    prev, cur = cur, nxt
                    path.append(cur)
                if cur not in special_to_cluster:
                    continue
                other = special_to_cluster[cur]
                if other == cluster_id:
                    continue
                internal = path[:-1] if path[-1] in special_to_cluster else path
                pts = [graph_nodes[cluster_id]]
                pts.extend(pixel_point(p) for p in internal)
                pts.append(graph_nodes[other])
                radii = [max(pixel_radius(start), min_radius)]
                radii.extend(pixel_radius(p) for p in internal)
                radii.append(max(pixel_radius(cur), min_radius))
                if len(pts) >= 2:
                    segments.append({"node_a": cluster_id, "node_b": other,
                                     "points": np.asarray(pts, dtype=float),
                                     "radii": np.asarray(radii, dtype=float)})

    if not segments:
        return None
    graph_nodes, segments = _compress_skeleton_graph(graph_nodes, segments)
    return {"nodes": graph_nodes, "segments": segments}


def _compress_skeleton_graph(graph_nodes: dict, segments: list[dict]) -> tuple[dict, list]:
    degree = {node_id: 0 for node_id in graph_nodes}
    for segment in segments:
        a, b = segment["node_a"], segment["node_b"]
        degree[a] += 1
        degree[b] += 1
    specials = {node_id for node_id, deg in degree.items() if deg != 2}
    if len(specials) < 2:
        return graph_nodes, segments

    adjacency = {node_id: [] for node_id in graph_nodes}
    for segment in segments:
        adjacency[segment["node_a"]].append(segment["node_b"])
        adjacency[segment["node_b"]].append(segment["node_a"])

    by_pair = {frozenset((segment["node_a"], segment["node_b"])): segment
               for segment in segments}

    def oriented_segment(a, b):
        segment = by_pair[frozenset((a, b))]
        seg_pts = np.asarray(segment["points"], dtype=float)
        seg_radii = np.asarray(segment["radii"], dtype=float)
        if segment["node_a"] != a:
            seg_pts = seg_pts[::-1]
            seg_radii = seg_radii[::-1]
        return seg_pts, seg_radii

    compressed = []
    visited_pairs: set[frozenset] = set()
    for start in sorted(specials, key=lambda cid: tuple(graph_nodes[cid][:2])):
        for nxt in sorted(adjacency[start]):
            first_key = frozenset((start, nxt))
            if first_key in visited_pairs:
                continue
            visited_pairs.add(first_key)
            pts, radii = oriented_segment(start, nxt)
            prev, cur = start, nxt
            while cur not in specials:
                choices = [n for n in sorted(adjacency[cur]) if n != prev]
                if not choices:
                    break
                nxt = choices[0]
                edge_key = frozenset((cur, nxt))
                if edge_key in visited_pairs:
                    break
                visited_pairs.add(edge_key)
                seg_pts, seg_radii = oriented_segment(cur, nxt)
                pts = np.vstack([pts, seg_pts[1:]])
                radii = np.concatenate([radii, seg_radii[1:]])
                prev, cur = cur, nxt
            if cur in specials and cur != start:
                compressed.append({
                    "node_a": start,
                    "node_b": cur,
                    "points": np.asarray(pts, dtype=float),
                    "radii": np.asarray(radii, dtype=float),
                })

    if not compressed:
        return graph_nodes, segments
    used = {segment["node_a"] for segment in compressed} | {
        segment["node_b"] for segment in compressed
    }
    return {node_id: graph_nodes[node_id] for node_id in sorted(used)}, compressed


def _thin_skeleton(mask2d) -> np.ndarray:
    """Zhang-Suen thinning for small/medium binary masks, implemented in NumPy loops.

    Large segmentation masks should be downsampled before import or thinned with
    a compiled imaging stack upstream.
    """
    mask = np.asarray(mask2d, dtype=bool)
    try:  # pragma: no cover - optional dependency path
        from skimage.morphology import skeletonize
    except ImportError:
        skeletonize = None
    if skeletonize is not None:
        return np.asarray(skeletonize(mask), dtype=bool)
    if mask.size > _MAX_PLANAR_FALLBACK_PIXELS:
        raise ImportError(
            "large planar mask skeletonization requires scikit-image; install the optional "
            "imaging stack or downsample before import"
        )
    if mask.size > _WARN_PLANAR_FALLBACK_PIXELS:
        warnings.warn(
            "using NumPy Zhang-Suen thinning fallback for a large planar mask; "
            "install scikit-image for compiled skeletonization",
            RuntimeWarning,
            stacklevel=2,
        )
    skel = mask.copy()
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            remove = []
            rows, cols = np.nonzero(skel)
            for r, c in zip(rows, cols):
                if r == 0 or c == 0 or r == skel.shape[0] - 1 or c == skel.shape[1] - 1:
                    continue
                p2 = skel[r - 1, c]
                p3 = skel[r - 1, c + 1]
                p4 = skel[r, c + 1]
                p5 = skel[r + 1, c + 1]
                p6 = skel[r + 1, c]
                p7 = skel[r + 1, c - 1]
                p8 = skel[r, c - 1]
                p9 = skel[r - 1, c - 1]
                ring = [p2, p3, p4, p5, p6, p7, p8, p9]
                count = int(sum(ring))
                if count < 2 or count > 6:
                    continue
                transitions = sum((not ring[i]) and ring[(i + 1) % 8] for i in range(8))
                if transitions != 1:
                    continue
                if step == 0:
                    keep = (p2 and p4 and p6) or (p4 and p6 and p8)
                else:
                    keep = (p2 and p4 and p8) or (p2 and p6 and p8)
                if not keep:
                    remove.append((r, c))
            if remove:
                changed = True
                for r, c in remove:
                    skel[r, c] = False
    return skel


def _distance_to_background_mm(mask2d, spacing_xy) -> np.ndarray:
    """Distance from foreground pixels to background, in millimeters.

    SciPy is intentionally optional for this package; use its exact Euclidean
    distance transform when installed, otherwise fall back to a bounded exact
    separable transform suitable for small/medium QA masks.
    """
    mask = np.asarray(mask2d, dtype=bool)
    try:  # pragma: no cover - optional dependency path
        from scipy import ndimage
    except ImportError:
        ndimage = None
    if ndimage is not None:
        # SciPy arrays use (row, col) spacing, while Lumen planar spacing is (x, y).
        return ndimage.distance_transform_edt(
            mask,
            sampling=(float(spacing_xy[1]), float(spacing_xy[0])),
        )
    if mask.size > _MAX_PLANAR_FALLBACK_PIXELS:
        raise ImportError(
            "large planar mask distance transforms require scipy; install the optional "
            "imaging stack or downsample before import"
        )
    max_spacing = max(float(spacing_xy[0]), float(spacing_xy[1]))
    high = (float(sum(mask.shape) + 1) * max_spacing) ** 2
    foreground_cost = np.where(mask, high, 0.0).astype(float)
    row_pass = np.column_stack([
        _edt_1d_squared(foreground_cost[:, col], float(spacing_xy[1]))
        for col in range(foreground_cost.shape[1])
    ])
    dist2 = np.vstack([
        _edt_1d_squared(row_pass[row], float(spacing_xy[0]))
        for row in range(row_pass.shape[0])
    ])
    return np.sqrt(np.maximum(dist2, 0.0))


def _edt_1d_squared(values, spacing: float) -> np.ndarray:
    """Squared 1-D lower-envelope distance transform for finite costs."""
    costs = np.asarray(values, dtype=float)
    n = len(costs)
    if n == 0:
        return costs.copy()
    spacing2 = float(spacing) ** 2
    sites = np.zeros(n, dtype=int)
    breaks = np.empty(n + 1, dtype=float)
    out = np.empty(n, dtype=float)
    k = 0
    sites[0] = 0
    breaks[0] = -np.inf
    breaks[1] = np.inf
    for q in range(1, n):
        while True:
            p = sites[k]
            q_cost = costs[q] + spacing2 * q * q
            p_cost = costs[p] + spacing2 * p * p
            numerator = q_cost - p_cost
            site_gap = q - p
            denominator = 2.0 * spacing2 * site_gap
            split = numerator / denominator
            if split > breaks[k]:
                break
            k -= 1
        k += 1
        sites[k] = q
        breaks[k] = split
        breaks[k + 1] = np.inf
    k = 0
    for q in range(n):
        while breaks[k + 1] < q:
            k += 1
        p = sites[k]
        out[q] = spacing2 * (q - p) ** 2 + costs[p]
    return out


def _neighbors8(pixel, shape):
    r, c = pixel
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
                yield rr, cc


def _connected_special_clusters(special_pixels: set[tuple[int, int]], shape) -> list[set]:
    remaining = set(special_pixels)
    clusters = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        cluster = {seed}
        frontier = [seed]
        while frontier:
            pixel = frontier.pop()
            for neighbor in _neighbors8(pixel, shape):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    cluster.add(neighbor)
                    frontier.append(neighbor)
        clusters.append(cluster)
    return clusters


def _component_planar_centerline(pixels, spacing_xy, origin, z_mm: float, samples: int):
    rc = np.asarray(list(pixels), dtype=float)
    xy_px = np.stack([rc[:, 1], rc[:, 0]], axis=1)  # image mask row/col -> x/y pixels
    xy_mm = origin[:2] + xy_px * spacing_xy[None, :]
    center = xy_mm.mean(axis=0)
    cov = np.cov((xy_mm - center).T)
    if np.ndim(cov) == 0 or not np.isfinite(cov).all():
        raise ValueError("planar mask component is degenerate")
    vals, vecs = np.linalg.eigh(cov)
    principal_idx = int(np.argmax(vals))
    axis = np.asarray(vecs[:, principal_idx], dtype=float)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    # deterministic start/end: left-to-right unless the principal direction is mostly vertical.
    if (abs(axis[0]) >= abs(axis[1]) and axis[0] < 0) or (
        abs(axis[1]) > abs(axis[0]) and axis[1] < 0
    ):
        axis = -axis
    normal = np.array([-axis[1], axis[0]])
    t = (xy_mm - center) @ axis
    u = (xy_mm - center) @ normal
    t_min, t_max = float(t.min()), float(t.max())
    if t_max <= t_min:
        raise ValueError("planar mask component has no measurable length")

    bins = np.linspace(t_min, t_max, int(samples) + 1)
    sample_centers, radii = [], []
    min_radius = 0.5 * float(min(spacing_xy))
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (t >= lo) & (t <= hi if hi == bins[-1] else t < hi)
        if np.count_nonzero(sel) < 2:
            continue
        local_xy = xy_mm[sel].mean(axis=0)
        width = float(u[sel].max() - u[sel].min()) + float(min(spacing_xy))
        sample_centers.append([float(local_xy[0]), float(local_xy[1]), float(origin[2] + z_mm)])
        radii.append(max(0.5 * width, min_radius))

    if len(sample_centers) < 2:
        # Fallback for tiny but valid components: line between principal extrema.
        order = np.argsort(t)
        endpoints = xy_mm[[order[0], order[-1]]]
        width = float(u.max() - u.min()) + float(min(spacing_xy))
        sample_centers = [[float(x), float(y), float(origin[2] + z_mm)] for x, y in endpoints]
        radii = [max(0.5 * width, min_radius), max(0.5 * width, min_radius)]
    return np.asarray(sample_centers, dtype=float), np.asarray(radii, dtype=float)


def _box_rows_from_json(payload, group_key: str = "group", image_size_px=None,
                        image_id=None, image_file: str | None = None):
    label_studio = _label_studio_rows_from_json(payload, group_key=group_key,
                                                image_size_px=image_size_px,
                                                image_file=image_file)
    if label_studio:
        return label_studio
    via = _via_rows_from_json(payload, group_key=group_key, image_file=image_file)
    if via:
        return via
    labelme = _labelme_rows_from_json(payload, group_key=group_key, image_file=image_file)
    if labelme:
        return labelme
    if not isinstance(payload, dict):
        return payload
    if "boxes" in payload:
        return payload["boxes"]
    if "annotations" not in payload:
        return payload
    annotations = payload["annotations"]
    if not isinstance(annotations, list):
        raise ValueError("COCO JSON 'annotations' must be a list")
    images = [item for item in payload.get("images", []) if isinstance(item, dict)]
    selected_image_id = image_id
    if image_file is not None:
        matches = [img for img in images if _image_file_matches(img.get("file_name"), image_file)]
        if not matches:
            raise ValueError(f"COCO JSON contains no image with file_name {image_file!r}")
        if len(matches) > 1:
            raise ValueError(f"COCO JSON has multiple images with file_name {image_file!r}; "
                             "use image_id instead")
        matched_image_id = matches[0].get("id")
        if selected_image_id is not None and not _coco_ids_equal(selected_image_id, matched_image_id):
            raise ValueError(
                f"COCO image_id {selected_image_id!r} does not match "
                f"image_file {image_file!r} (matched image id {matched_image_id!r})"
            )
        selected_image_id = matched_image_id
    if selected_image_id is None and len({ann.get("image_id") for ann in annotations
                                          if isinstance(ann, dict) and "image_id" in ann}) > 1:
        raise ValueError("multi-image COCO import requires image_id or image_file")
    categories = {
        item.get("id"): item.get("name")
        for item in payload.get("categories", [])
        if isinstance(item, dict)
    }
    rows = []
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        if selected_image_id is not None and not _coco_ids_equal(
            ann.get("image_id"), selected_image_id
        ):
            continue
        bbox = ann.get("bbox")
        if bbox is not None:
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                raise ValueError("COCO annotation 'bbox' must be [x, y, width, height]")
            x0, y0, w, h = bbox
            x1 = float(x0) + float(w)
            y1 = float(y0) + float(h)
        else:
            bounds = _coco_segmentation_bounds(ann.get("segmentation"))
            if bounds is None:
                continue
            x0, y0, x1, y1 = bounds
        attrs = ann.get("attributes", {})
        attrs = attrs if isinstance(attrs, dict) else {}
        group = ann.get(group_key, attrs.get(group_key))
        if group is None:
            group = attrs.get("group", categories.get(ann.get("category_id"), ann.get("category_id")))
        row = {
            "x_min": x0,
            "y_min": y0,
            "x_max": x1,
            "y_max": y1,
            "group": "vessel" if group is None else group,
        }
        if "order" in ann or "order" in attrs:
            row["order"] = ann.get("order", attrs.get("order"))
        if "radius_mm" in ann or "radius_mm" in attrs:
            row["radius_mm"] = ann.get("radius_mm", attrs.get("radius_mm"))
        rows.append(row)
    if not rows:
        raise ValueError("COCO JSON contains no annotations with bbox or polygon segmentation entries")
    return rows


def _via_rows_from_json(payload, group_key: str = "group",
                        image_file: str | None = None) -> list[dict]:
    """Extract VGG Image Annotator rectangle regions as pixel-space box rows."""
    if not isinstance(payload, dict):
        return []
    records_obj = payload.get("_via_img_metadata")
    if isinstance(records_obj, dict):
        records = [record for record in records_obj.values() if isinstance(record, dict)]
    else:
        records = [
            record for record in payload.values()
            if isinstance(record, dict) and "regions" in record and "filename" in record
        ]
    if not records:
        return []

    refs = [record.get("filename") for record in records]
    if image_file is not None:
        matches = [record for record, ref in zip(records, refs) if _image_file_matches(ref, image_file)]
        if not matches:
            raise ValueError(f"VIA export contains no image_file {image_file!r}")
        if len(matches) > 1:
            raise ValueError(f"VIA export has multiple images for image_file {image_file!r}")
        records = matches
    elif len({_image_ref_basename(ref) for ref in refs if ref}) > 1:
        raise ValueError("multi-image VIA import requires image_file")

    rows = []
    order = 0
    for record in records:
        regions = record.get("regions", [])
        if isinstance(regions, dict):
            regions_iter = [region for region in regions.values()]
        elif isinstance(regions, list):
            regions_iter = regions
        else:
            continue
        for region in regions_iter:
            if not isinstance(region, dict):
                continue
            shape = region.get("shape_attributes", {})
            if not isinstance(shape, dict) or shape.get("name") != "rect":
                continue
            if not {"x", "y", "width", "height"} <= set(shape):
                raise ValueError("VIA rectangle regions require x/y/width/height")
            attrs = region.get("region_attributes", {})
            attrs = attrs if isinstance(attrs, dict) else {}
            group = (
                attrs.get(group_key)
                or attrs.get("group")
                or attrs.get("label")
                or attrs.get("category")
                or "vessel"
            )
            row = {
                "x_min": shape["x"],
                "y_min": shape["y"],
                "x_max": float(shape["x"]) + float(shape["width"]),
                "y_max": float(shape["y"]) + float(shape["height"]),
                group_key: group,
                "order": attrs.get("order", order),
            }
            if "radius_mm" in attrs:
                row["radius_mm"] = attrs["radius_mm"]
            rows.append(row)
            order += 1
    if not rows:
        raise ValueError("VIA export contains no rectangle regions")
    return rows


def _labelme_rows_from_json(payload, group_key: str = "group",
                            image_file: str | None = None) -> list[dict]:
    """Extract LabelMe rectangle/polygon shapes as axis-aligned box rows."""
    if not isinstance(payload, dict) or "shapes" not in payload:
        return []
    if image_file is not None:
        ref = payload.get("imagePath") or payload.get("image_path")
        if ref is None:
            raise ValueError("LabelMe import with image_file requires imagePath metadata")
        if not _image_file_matches(ref, image_file):
            raise ValueError(f"LabelMe export imagePath {ref!r} does not match image_file {image_file!r}")

    shapes = payload.get("shapes")
    if not isinstance(shapes, list):
        raise ValueError("LabelMe JSON 'shapes' must be a list")
    rows = []
    order = 0
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        shape_type = str(shape.get("shape_type") or "polygon").lower()
        if shape_type not in {"rectangle", "polygon"}:
            continue
        points = shape.get("points")
        if not isinstance(points, list) or len(points) < 2:
            raise ValueError("LabelMe rectangle/polygon shapes require at least two points")
        arr = np.asarray(points, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2 or not np.isfinite(arr).all():
            raise ValueError("LabelMe shape points must be finite [x, y] coordinates")
        if shape_type == "rectangle" and len(arr) != 2:
            raise ValueError("LabelMe rectangle shapes require exactly two corner points")
        x0, y0 = arr.min(axis=0)
        x1, y1 = arr.max(axis=0)
        flags = shape.get("flags", {})
        flags = flags if isinstance(flags, dict) else {}
        group = (
            shape.get(group_key)
            or flags.get(group_key)
            or flags.get("group")
            or shape.get("label")
            or shape.get("group_id")
            or "vessel"
        )
        row = {
            "x_min": float(x0),
            "y_min": float(y0),
            "x_max": float(x1),
            "y_max": float(y1),
            group_key: group,
            "order": shape.get("order", flags.get("order", order)),
        }
        if "radius_mm" in shape or "radius_mm" in flags:
            row["radius_mm"] = shape.get("radius_mm", flags.get("radius_mm"))
        rows.append(row)
        order += 1
    if not rows:
        raise ValueError("LabelMe export contains no rectangle or polygon shapes")
    return rows


def _coco_ids_equal(a, b) -> bool:
    return a == b or (a is not None and b is not None and str(a) == str(b))


def _coco_segmentation_bounds(segmentation) -> tuple[float, float, float, float] | None:
    if isinstance(segmentation, dict):
        return _coco_rle_segmentation_bounds(segmentation)
    return _coco_polygon_segmentation_bounds(segmentation)


def _coco_polygon_segmentation_bounds(segmentation) -> tuple[float, float, float, float] | None:
    if segmentation is None:
        return None
    if not isinstance(segmentation, (list, tuple)):
        raise ValueError("COCO polygon segmentation must be a list of x/y coordinates")
    if not segmentation:
        return None

    if all(np.isscalar(value) for value in segmentation):
        polygons = [segmentation]
    else:
        polygons = segmentation

    coords = []
    for polygon in polygons:
        if not isinstance(polygon, (list, tuple)):
            raise ValueError("COCO polygon segmentation must contain polygon coordinate lists")
        try:
            values = np.asarray(polygon, dtype=float).reshape(-1)
        except (TypeError, ValueError) as e:
            raise ValueError("COCO polygon segmentation must contain finite x/y pairs") from e
        if len(values) == 0:
            continue
        if len(values) < 6 or len(values) % 2:
            raise ValueError("COCO polygon segmentation must contain at least three x/y pairs")
        if not np.isfinite(values).all():
            raise ValueError("COCO polygon segmentation must contain finite x/y pairs")
        coords.append(values.reshape(-1, 2))
    if not coords:
        return None
    xy = np.vstack(coords)
    return (
        float(xy[:, 0].min()),
        float(xy[:, 1].min()),
        float(xy[:, 0].max()),
        float(xy[:, 1].max()),
    )


def _coco_rle_segmentation_bounds(segmentation: dict) -> tuple[float, float, float, float] | None:
    size = segmentation.get("size")
    counts = segmentation.get("counts")
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError("COCO RLE segmentation 'size' must be [height, width]")
    height, width = (int(size[0]), int(size[1]))
    if height <= 0 or width <= 0:
        raise ValueError("COCO RLE segmentation size values must be positive")
    if isinstance(counts, str):
        raise ValueError("COCO compressed RLE counts are not supported; provide bbox or uncompressed counts")
    if not isinstance(counts, (list, tuple)):
        raise ValueError("COCO RLE segmentation 'counts' must be an uncompressed run-length list")
    runs = np.asarray(counts, dtype=float).reshape(-1)
    if not np.isfinite(runs).all() or np.any(runs < 0):
        raise ValueError("COCO RLE segmentation counts must be finite non-negative values")
    if not np.allclose(runs, np.round(runs), atol=0.0):
        raise ValueError("COCO RLE segmentation counts must be integers")
    runs = runs.astype(int)
    total = int(height * width)
    if int(runs.sum()) != total:
        raise ValueError("COCO RLE segmentation counts must sum to height * width")

    mask = np.zeros(total, dtype=bool)
    cursor = 0
    foreground = False
    for run in runs:
        next_cursor = cursor + int(run)
        if foreground and run:
            mask[cursor:next_cursor] = True
        cursor = next_cursor
        foreground = not foreground
    foreground_idx = np.flatnonzero(mask)
    if len(foreground_idx) == 0:
        return None
    ys = foreground_idx % height
    xs = foreground_idx // height
    return (
        float(xs.min()),
        float(ys.min()),
        float(xs.max() + 1),
        float(ys.max() + 1),
    )


def _label_studio_rows_from_json(payload, group_key: str = "group", image_size_px=None,
                                 image_file: str | None = None):
    """Extract Label Studio rectanglelabels exports as pixel-space box rows.

    Label Studio stores rectangle coordinates as percentages of the source image.
    Most exports include ``original_width`` and ``original_height`` on each result;
    ``image_size_px`` is accepted as a fallback for stripped-down exports.
    """
    tasks = payload if isinstance(payload, list) else [payload]
    if not all(isinstance(task, dict) for task in tasks):
        return []
    fallback_size = None
    if image_size_px is not None:
        size = np.asarray(image_size_px, dtype=float).reshape(-1)
        if len(size) != 2 or np.any(size <= 0):
            raise ValueError(f"image_size_px must be positive (width, height), got {image_size_px}")
        fallback_size = (float(size[0]), float(size[1]))
    if not any(_label_studio_task_has_rectangles(task) for task in tasks):
        return []

    task_refs = [_label_studio_task_image_ref(task) for task in tasks]
    if image_file is not None:
        matches = [i for i, ref in enumerate(task_refs) if _image_file_matches(ref, image_file)]
        if not matches:
            raise ValueError(f"Label Studio export contains no task for image_file {image_file!r}")
        if len(matches) > 1:
            raise ValueError(f"Label Studio export has multiple tasks for image_file {image_file!r}")
        tasks = [tasks[matches[0]]]
    elif len({ref for ref in task_refs if ref}) > 1:
        raise ValueError("multi-task Label Studio import requires image_file")

    rows = []
    order = 0
    for task in tasks:
        annotations = task.get("annotations") or task.get("completions")
        if annotations is None and isinstance(task.get("result"), list):
            annotations = [task]
        if not isinstance(annotations, list):
            continue
        for ann in annotations:
            if not isinstance(ann, dict):
                continue
            results = ann.get("result", [])
            if not isinstance(results, list):
                continue
            for result in results:
                if not isinstance(result, dict):
                    continue
                value = result.get("value", {})
                if not isinstance(value, dict):
                    continue
                if not {"x", "y", "width", "height"} <= set(value):
                    continue
                labels = value.get(group_key)
                if labels is None:
                    labels = value.get("rectanglelabels", value.get("labels"))
                if isinstance(labels, (list, tuple)):
                    group = labels[0] if labels else "vessel"
                elif labels in (None, ""):
                    group = result.get("from_name", "vessel")
                else:
                    group = labels
                width = result.get("original_width", task.get("original_width"))
                height = result.get("original_height", task.get("original_height"))
                if width is None or height is None:
                    if fallback_size is None:
                        raise ValueError("Label Studio rectangle import requires "
                                         "original_width/original_height metadata or "
                                         "image_size_px=(width, height)")
                    width, height = fallback_size
                width, height = float(width), float(height)
                if width <= 0 or height <= 0:
                    raise ValueError("Label Studio original_width/original_height must be positive")
                x0 = float(value["x"]) * width / 100.0
                y0 = float(value["y"]) * height / 100.0
                x1 = x0 + float(value["width"]) * width / 100.0
                y1 = y0 + float(value["height"]) * height / 100.0
                row = {
                    "x_min": x0,
                    "y_min": y0,
                    "x_max": x1,
                    "y_max": y1,
                    "group": group,
                    "order": float(order),
                }
                if "radius_mm" in value:
                    row["radius_mm"] = value["radius_mm"]
                rows.append(row)
                order += 1
    return rows


def _xml_box_rows(path: Path, group_key: str = "group",
                  image_file: str | None = None) -> list[dict]:
    root = ET.parse(path).getroot()
    if _xml_tag(root) == "annotation" and any(_xml_tag(elem) == "object" for elem in root):
        return _pascal_voc_box_rows(root, group_key=group_key, image_file=image_file)
    return _cvat_box_rows(root, group_key=group_key, image_file=image_file)


def _cvat_box_rows(root, group_key: str = "group",
                   image_file: str | None = None) -> list[dict]:
    images = [elem for elem in root.iter() if _xml_tag(elem) == "image"]
    if not images:
        raise ValueError("CVAT XML contains no <image> elements")
    if image_file is not None:
        matches = [image for image in images if _image_file_matches(image.get("name"), image_file)]
        if not matches:
            raise ValueError(f"CVAT XML contains no image named {image_file!r}")
        if len(matches) > 1:
            raise ValueError(f"CVAT XML has multiple images named {image_file!r}")
        images = matches
    elif len({_image_ref_basename(image.get("name")) for image in images if image.get("name")}) > 1:
        raise ValueError("multi-image CVAT import requires image_file")

    rows = []
    order = 0
    for image in images:
        for box in image:
            if _xml_tag(box) != "box":
                continue
            attrs = {attr.get("name"): (attr.text or "").strip()
                     for attr in box if _xml_tag(attr) == "attribute" and attr.get("name")}
            group = attrs.get(group_key) or attrs.get("group") or box.get(group_key) or box.get("label")
            row = {
                "x_min": box.get("xtl"),
                "y_min": box.get("ytl"),
                "x_max": box.get("xbr"),
                "y_max": box.get("ybr"),
                group_key: "vessel" if group in (None, "") else group,
                "order": attrs.get("order", box.get("order", order)),
            }
            if "radius_mm" in attrs or box.get("radius_mm") is not None:
                row["radius_mm"] = attrs.get("radius_mm", box.get("radius_mm"))
            rows.append(row)
            order += 1
    if not rows:
        raise ValueError("CVAT XML contains no <box> annotations")
    return rows


def _pascal_voc_box_rows(root, group_key: str = "group",
                         image_file: str | None = None) -> list[dict]:
    refs = [
        _xml_text(root.find("filename")),
        _xml_text(root.find("path")),
    ]
    if image_file is not None and not any(_image_file_matches(ref, image_file) for ref in refs):
        raise ValueError(f"Pascal VOC XML does not describe image_file {image_file!r}")

    rows = []
    for order, obj in enumerate(elem for elem in root if _xml_tag(elem) == "object"):
        box = next((child for child in obj if _xml_tag(child) == "bndbox"), None)
        if box is None:
            continue
        row = {
            "x_min": _xml_text(box.find("xmin")),
            "y_min": _xml_text(box.find("ymin")),
            "x_max": _xml_text(box.find("xmax")),
            "y_max": _xml_text(box.find("ymax")),
            group_key: _xml_text(obj.find("name")) or "vessel",
            "order": _xml_text(obj.find("order")) or order,
        }
        radius = _xml_text(obj.find("radius_mm"))
        if radius is not None:
            row["radius_mm"] = radius
        rows.append(row)
    if not rows:
        raise ValueError("Pascal VOC XML contains no <object><bndbox> annotations")
    return rows


def _xml_tag(elem) -> str:
    return str(elem.tag).rsplit("}", 1)[-1]


def _xml_text(elem) -> str | None:
    if elem is None or elem.text is None:
        return None
    value = elem.text.strip()
    return value or None


def _label_studio_task_image_ref(task: dict) -> str | None:
    data = task.get("data")
    if isinstance(data, dict):
        for key in ("image", "img", "file", "filename", "image_url"):
            value = data.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _label_studio_task_has_rectangles(task: dict) -> bool:
    annotations = task.get("annotations") or task.get("completions")
    if annotations is None and isinstance(task.get("result"), list):
        annotations = [task]
    if not isinstance(annotations, list):
        return False
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        results = ann.get("result", [])
        if not isinstance(results, list):
            continue
        for result in results:
            if not isinstance(result, dict):
                continue
            value = result.get("value", {})
            if isinstance(value, dict) and {"x", "y", "width", "height"} <= set(value):
                return True
    return False


def _image_file_matches(candidate, image_file: str) -> bool:
    if not candidate:
        return False
    return (
        candidate == image_file
        or _image_ref_basename(candidate) == _image_ref_basename(image_file)
    )


def _image_ref_basename(value) -> str:
    raw = str(value)
    parsed = urlparse(raw)
    path = parsed.path if parsed.scheme or parsed.netloc else raw.split("?", 1)[0].split("#", 1)[0]
    return Path(path).name


def _load_yolo_box_annotations(path: Path, image_size_px=None) -> list[BoxAnnotation]:
    if image_size_px is None:
        raise ValueError("YOLO label import requires image_size_px=(width, height)")
    size = np.asarray(image_size_px, dtype=float).reshape(-1)
    if len(size) != 2 or np.any(size <= 0):
        raise ValueError(f"image_size_px must be positive (width, height), got {image_size_px}")
    width, height = float(size[0]), float(size[1])
    boxes: list[BoxAnnotation] = []
    for order, raw in enumerate(path.read_text().splitlines()):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 5:
            raise ValueError("YOLO label rows must contain class x_center y_center width height")
        cls, xc, yc, bw, bh = parts[:5]
        xc, yc, bw, bh = (float(xc), float(yc), float(bw), float(bh))
        if not np.isfinite([xc, yc, bw, bh]).all() or bw <= 0 or bh <= 0:
            raise ValueError(f"invalid YOLO row {raw!r}")
        x0 = (xc - 0.5 * bw) * width
        y0 = (yc - 0.5 * bh) * height
        x1 = (xc + 0.5 * bw) * width
        y1 = (yc + 0.5 * bh) * height
        boxes.append(BoxAnnotation(x0, y0, x1, y1, group=f"class_{cls}", order=float(order)))
    if not boxes:
        raise ValueError("YOLO label file contains no boxes")
    for box in boxes:
        _validate_box(box)
    return boxes


def _box_from_mapping(row, group_key: str = "group") -> BoxAnnotation:
    if not isinstance(row, dict):
        raise ValueError("box entries must be objects/mappings")
    if {"x_min", "y_min", "x_max", "y_max"} <= set(row):
        x0, y0, x1, y1 = row["x_min"], row["y_min"], row["x_max"], row["y_max"]
    elif {"x0", "y0", "x1", "y1"} <= set(row):
        x0, y0, x1, y1 = row["x0"], row["y0"], row["x1"], row["y1"]
    elif {"left", "top", "width", "height"} <= set(row):
        x0, y0 = row["left"], row["top"]
        x1 = float(x0) + float(row["width"])
        y1 = float(y0) + float(row["height"])
    else:
        raise ValueError("box row must contain x_min/y_min/x_max/y_max, "
                         "x0/y0/x1/y1, or left/top/width/height")
    group = row.get(group_key, row.get("group", "vessel"))
    order = row.get("order", None)
    radius = row.get("radius_mm", None)
    return BoxAnnotation(float(x0), float(y0), float(x1), float(y1),
                         group=str(group),
                         order=None if order in (None, "") else float(order),
                         radius_mm=None if radius in (None, "") else float(radius))


def _coerce_box(box) -> BoxAnnotation:
    if isinstance(box, BoxAnnotation):
        out = box
    elif isinstance(box, dict):
        out = _box_from_mapping(box)
    else:
        vals = list(box)
        if len(vals) not in (4, 5, 6, 7):
            raise ValueError("box sequences must have 4 values, optionally followed by "
                             "group, order, and radius_mm")
        group = vals[4] if len(vals) >= 5 else "vessel"
        order = vals[5] if len(vals) >= 6 else None
        radius = vals[6] if len(vals) >= 7 else None
        out = BoxAnnotation(float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]),
                            group=str(group),
                            order=None if order is None else float(order),
                            radius_mm=None if radius is None else float(radius))
    _validate_box(out)
    return out


def _validate_box(box: BoxAnnotation) -> None:
    vals = np.asarray([box.x_min, box.y_min, box.x_max, box.y_max], dtype=float)
    if not np.isfinite(vals).all():
        raise ValueError(f"box coordinates must be finite, got {box}")
    if box.x_max <= box.x_min or box.y_max <= box.y_min:
        raise ValueError(f"box must have positive width and height, got {box}")
    if box.radius_mm is not None and box.radius_mm <= 0:
        raise ValueError(f"radius_mm must be positive when provided, got {box.radius_mm}")


def _box_center_radius_px(box: BoxAnnotation) -> tuple[np.ndarray, float]:
    center = np.array([(box.x_min + box.x_max) * 0.5, (box.y_min + box.y_max) * 0.5],
                      dtype=float)
    radius = 0.5 * min(box.x_max - box.x_min, box.y_max - box.y_min)
    return center, float(radius)


def _order_boxes(boxes: list[BoxAnnotation]) -> list[BoxAnnotation]:
    if all(b.order is not None for b in boxes):
        return sorted(boxes, key=lambda b: (float(b.order), b.y_min, b.x_min))
    centers = np.asarray([_box_center_radius_px(b)[0] for b in boxes], dtype=float)
    if len(centers) == 1:
        return list(boxes)
    span = np.ptp(centers, axis=0)
    axis = int(np.argmax(span))
    return [box for _, box in sorted(zip(centers[:, axis], boxes),
                                     key=lambda item: (float(item[0]), item[1].y_min,
                                                       item[1].x_min))]


def _slice_components(vol: Volume, min_voxels: int) -> list[_SliceComponent]:
    comps: list[_SliceComponent] = []
    spacing = np.asarray(vol.spacing_mm, float)
    origin = np.asarray(vol.origin_mm, float)
    next_id = 0
    for z in range(vol.data.shape[2]):
        labels = _connected_components_2d(vol.data[:, :, z])
        for pixels in labels:
            if len(pixels) < min_voxels:
                continue
            arr = np.asarray(list(pixels), dtype=float)
            xy = arr.mean(axis=0)
            center_vox = np.array([xy[0], xy[1], float(z)])
            center_mm = origin + center_vox * spacing
            area_mm2 = len(pixels) * spacing[0] * spacing[1]
            radius_mm = max(float(np.sqrt(area_mm2 / np.pi)), 0.5 * min(spacing[:2]))
            comps.append(_SliceComponent(next_id, z, pixels, center_vox, center_mm, radius_mm))
            next_id += 1
    return comps


def _connected_components_2d(mask2d) -> list[frozenset]:
    mask2d = np.asarray(mask2d, bool)
    seen = np.zeros(mask2d.shape, dtype=bool)
    out = []
    nx, ny = mask2d.shape
    for seed in np.argwhere(mask2d):
        sx, sy = int(seed[0]), int(seed[1])
        if seen[sx, sy]:
            continue
        stack = [(sx, sy)]
        seen[sx, sy] = True
        pixels = []
        while stack:
            x, y = stack.pop()
            pixels.append((x, y))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    xx, yy = x + dx, y + dy
                    if 0 <= xx < nx and 0 <= yy < ny and mask2d[xx, yy] and not seen[xx, yy]:
                        seen[xx, yy] = True
                        stack.append((xx, yy))
        out.append(frozenset(pixels))
    return out


def _link_components(comps: list[_SliceComponent], spacing_mm) -> set[tuple[int, int]]:
    by_z: dict[int, list[_SliceComponent]] = {}
    for c in comps:
        by_z.setdefault(c.z, []).append(c)
    links = set()
    z_values = sorted(by_z)
    xy_spacing = np.asarray(spacing_mm[:2], float)
    for za, zb in zip(z_values[:-1], z_values[1:]):
        if zb != za + 1:
            continue
        for a in by_z[za]:
            candidates = []
            for b in by_z[zb]:
                overlap = bool(a.pixels & b.pixels)
                delta_xy = (a.center_vox[:2] - b.center_vox[:2]) * xy_spacing
                dist = float(np.linalg.norm(delta_xy))
                if overlap or dist <= a.radius_mm + b.radius_mm + max(xy_spacing):
                    candidates.append((dist, b.id))
            if candidates:
                for _, bid in sorted(candidates)[:2]:
                    links.add((a.id, bid))
    return links


def _components_to_asset(comps: list[_SliceComponent], links: set[tuple[int, int]],
                         spacing_mm, origin_mm, provenance: str) -> Asset:
    comp = {c.id: c for c in comps}
    adj: dict[int, set[int]] = {c.id: set() for c in comps}
    for a, b in links:
        adj[a].add(b)
        adj[b].add(a)
    endpoints = {c.id for c in comps if not adj[c.id]}
    min_z = min(c.z for c in comps)
    max_z = max(c.z for c in comps)
    endpoints |= {c.id for c in comps if c.z in (min_z, max_z)}
    specials = {cid for cid, ns in adj.items() if len(ns) != 2} | endpoints
    node_ids = {cid: f"n{cid}" for cid in sorted(specials)}
    nodes = [Node(node_ids[cid], tuple(comp[cid].center_mm)) for cid in sorted(specials)]

    edges: list[Edge] = []
    visited: set[frozenset] = set()
    for start in sorted(specials):
        for nxt in sorted(adj[start]):
            key = frozenset((start, nxt))
            if key in visited:
                continue
            path = [start]
            prev, cur = start, nxt
            visited.add(key)
            while cur not in specials:
                path.append(cur)
                choices = [n for n in sorted(adj[cur]) if n != prev]
                if not choices:
                    break
                prev, cur = cur, choices[0]
                visited.add(frozenset((prev, cur)))
            path.append(cur)
            if path[0] == path[-1] or path[-1] not in specials:
                continue
            edges.append(_edge_for_path(f"e{len(edges)}", node_ids[path[0]],
                                        node_ids[path[-1]], [comp[i] for i in path]))

    if not edges and len(comps) == 1:
        c = comps[0]
        p0 = c.center_mm - np.array([0.0, 0.0, 0.5 * spacing_mm[2]])
        p1 = c.center_mm + np.array([0.0, 0.0, 0.5 * spacing_mm[2]])
        nodes = [Node("n0", tuple(p0)), Node("n1", tuple(p1))]
        edges = [_edge_for_points("e0", "n0", "n1", [p0, p1], [c.radius_mm, c.radius_mm])]
        spawn = "n0"
    else:
        spawn_comp = min(specials, key=lambda cid: (comp[cid].z, comp[cid].center_mm[2]))
        spawn = node_ids[spawn_comp]

    return Asset(
        frame=Frame(name="world_mm", spacing_mm=spacing_mm, origin_mm=origin_mm),
        nodes=nodes,
        edges=edges,
        device_spawn=DeviceSpawn(node_id=spawn),
        provenance=provenance,
    )


def _edge_for_path(edge_id: str, node_a: str, node_b: str,
                   comps: list[_SliceComponent]) -> Edge:
    pts = [c.center_mm for c in comps]
    radii = [c.radius_mm for c in comps]
    return _edge_for_points(edge_id, node_a, node_b, pts, radii)


def _edge_for_points(edge_id: str, node_a: str, node_b: str, pts, radii) -> Edge:
    pts = np.asarray(pts, float)
    radii = np.asarray(radii, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1) if len(pts) > 1 else np.array([])
    s = np.concatenate([[0.0], np.cumsum(seg)])
    theta = np.array([0.0])
    return Edge(
        id=edge_id,
        node_a=node_a,
        node_b=node_b,
        centerline_mm=pts.tolist(),
        s_grid=s.tolist(),
        theta_grid=theta.tolist(),
        R=radii[:, None].tolist(),
    )
