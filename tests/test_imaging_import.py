import json
import numpy as np
import struct
import sys
import zlib

from lumen.assets.imaging import (asset_from_box_annotations, asset_from_mask,
                                  asset_from_planar_mask, asset_planar_import_report,
                                  load_dicom_frame,
                                  load_box_annotations, load_npz_volume,
                                  load_planar_array, segment_threshold)
from lumen.sensors import FluoroSensor


def _disk_mask(shape=(36, 36, 16)):
    mask = np.zeros(shape, dtype=bool)
    yy, xx = np.mgrid[0:shape[1], 0:shape[0]]
    for z in range(2, 14):
        centers = [(18.0, 18.0)] if z < 8 else [(14.0, 18.0), (22.0, 18.0)]
        radius = 4.0 if z < 8 else 3.0
        for cx, cy in centers:
            disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
            mask[:, :, z] |= disk.T
    return mask


def _curved_planar_mask(shape=(72, 128)):
    mask = np.zeros(shape, dtype=bool)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    for x in range(10, shape[1] - 10):
        cy = 0.5 * shape[0] + 13.0 * np.sin((x - 10) / 18.0)
        radius = 4.0 + 1.0 * np.sin(x / 13.0)
        mask |= (xx - x) ** 2 + (yy - cy) ** 2 <= radius ** 2
    return mask


def _branched_planar_mask(shape=(96, 96)):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    mask = np.zeros(shape, dtype=bool)
    segments = [
        ((48.0, 10.0), (48.0, 52.0), 5.0),
        ((48.0, 52.0), (26.0, 84.0), 4.0),
        ((48.0, 52.0), (72.0, 82.0), 4.0),
    ]
    for (x0, y0), (x1, y1), radius in segments:
        dx, dy = x1 - x0, y1 - y0
        t = ((xx - x0) * dx + (yy - y0) * dy) / (dx * dx + dy * dy)
        t = np.clip(t, 0.0, 1.0)
        px = x0 + t * dx
        py = y0 + t * dy
        mask |= (xx - px) ** 2 + (yy - py) ** 2 <= radius ** 2
    return mask


def _write_gray16_png(path, arr) -> None:
    data = np.asarray(arr, dtype=np.uint16)
    h, w = data.shape

    def chunk(typ, payload):
        body = typ + payload
        return (
            struct.pack(">I", len(payload))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + data[y].astype(">u2", copy=False).tobytes() for y in range(h))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def _write_gray1_png(path, arr) -> None:
    data = np.asarray(arr, dtype=np.uint8)
    h, w = data.shape

    def chunk(typ, payload):
        body = typ + payload
        return (
            struct.pack(">I", len(payload))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + np.packbits(data[y], bitorder="big").tobytes() for y in range(h))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 1, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def _write_palette_png(path, indices) -> None:
    data = np.asarray(indices, dtype=np.uint8)
    h, w = data.shape
    palette = bytes([
        0, 0, 0,
        255, 0, 0,
        0, 255, 0,
        0, 0, 255,
    ])

    def chunk(typ, payload):
        body = typ + payload
        return (
            struct.pack(">I", len(payload))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + data[y].tobytes() for y in range(h))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 3, 0, 0, 0))
        + chunk(b"PLTE", palette)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def _coco_uncompressed_rle(mask) -> dict:
    data = np.asarray(mask, dtype=bool)
    counts = []
    current = False
    run = 0
    for value in data.reshape(-1, order="F"):
        if bool(value) == current:
            run += 1
        else:
            counts.append(run)
            current = bool(value)
            run = 1
    counts.append(run)
    return {"size": [int(data.shape[0]), int(data.shape[1])], "counts": counts}


def test_npz_volume_can_be_thresholded_and_imported_as_branching_asset(tmp_path):
    intensity = np.where(_disk_mask(), 240.0, 20.0).astype(np.float32)
    path = tmp_path / "cta.npz"
    np.savez(path, volume=intensity, spacing_mm=np.array([0.5, 0.5, 1.25]),
             origin_mm=np.array([10.0, 20.0, 30.0]))

    vol = load_npz_volume(path)
    mask = segment_threshold(vol, threshold=100.0, foreground="above")
    asset = asset_from_mask(mask.mask, spacing_mm=mask.spacing_mm, origin_mm=mask.origin_mm,
                            min_component_voxels=8)

    assert asset.frame.spacing_mm == (0.5, 0.5, 1.25)
    assert asset.frame.origin_mm == (10.0, 20.0, 30.0)
    assert asset.provenance == "segmented(imported)"
    assert len(asset.edges) >= 3
    assert any(n.id == asset.device_spawn.node_id for n in asset.nodes)
    assert all(np.asarray(edge.R).min() > 0.0 for edge in asset.edges)


def test_fluoro_can_render_tree_wide_asset_roadmap_from_imported_mask():
    asset = asset_from_mask(_disk_mask(), spacing_mm=(0.5, 0.5, 1.0),
                            min_component_voxels=8)
    trunk = next(edge for edge in asset.edges if edge.node_a == asset.device_spawn.node_id)
    wire = np.asarray(trunk.centerline_mm, float)

    scene = FluoroSensor(res=32, nu=48, nv=48, n_samples=80).render_scene(
        wire, radius=0.25, contrast_asset=asset, mu_contrast=0.16)

    assert scene["image"].shape == (48, 48)
    assert scene["masks"]["device"].sum() > 0
    assert scene["masks"]["vessel"].sum() > scene["masks"]["device"].sum()


def test_box_annotations_import_planar_asset_with_radius_profile():
    boxes = [
        {"x0": 10, "y0": 40, "x1": 18, "y1": 52},
        {"x0": 10, "y0": 10, "x1": 18, "y1": 22},
        {"x0": 10, "y0": 25, "x1": 22, "y1": 37},
    ]

    asset = asset_from_box_annotations(boxes, pixel_spacing_mm=(0.5, 0.25),
                                       origin_mm=(100.0, 200.0, 5.0),
                                       z_mm=2.0)
    edge = asset.edges[0]
    pts = np.asarray(edge.centerline_mm, float)
    radii = np.asarray(edge.R, float)[:, 0]

    assert asset.frame.name == "image_xy_mm"
    assert asset.frame.spacing_mm == (0.5, 0.25, 1.0)
    assert asset.provenance == "2d-boxes(imported)"
    assert np.all(np.diff(pts[:, 1]) > 0.0)
    assert np.allclose(pts[:, 2], 7.0)
    assert np.all(radii > 0.0)
    # The first box's short side is 8 px; with 0.25 mm min spacing, radius is 1 mm.
    assert np.isclose(radii[0], 1.0)


def test_load_box_annotations_accepts_labelme_rectangles_and_polygons(tmp_path):
    src = tmp_path / "labelme.json"
    src.write_text(json.dumps({
        "version": "5.4.1",
        "imagePath": "frame_001.png",
        "imageWidth": 128,
        "imageHeight": 96,
        "shapes": [
            {
                "label": "left_branch",
                "shape_type": "rectangle",
                "points": [[10.0, 20.0], [24.0, 38.0]],
                "flags": {"order": 2, "radius_mm": 1.4},
            },
            {
                "label": "left_branch",
                "shape_type": "polygon",
                "points": [[14.0, 48.0], [28.0, 51.0], [26.0, 66.0], [12.0, 63.0]],
                "flags": {"order": 3},
            },
        ],
    }))

    boxes = load_box_annotations(src, image_file="signed/frame_001.png?token=abc")

    assert len(boxes) == 2
    assert boxes[0].group == "left_branch"
    assert boxes[0].order == 2
    assert boxes[0].radius_mm == 1.4
    assert (boxes[0].x_min, boxes[0].y_min, boxes[0].x_max, boxes[0].y_max) == (10.0, 20.0, 24.0, 38.0)
    assert (boxes[1].x_min, boxes[1].y_min, boxes[1].x_max, boxes[1].y_max) == (12.0, 48.0, 28.0, 66.0)


def test_box_annotation_groups_merge_into_renderable_branch_asset():
    boxes = [
        {"x0": 28, "y0": 0, "x1": 36, "y1": 8, "group": "trunk"},
        {"x0": 28, "y0": 14, "x1": 36, "y1": 22, "group": "trunk"},
        {"x0": 28, "y0": 28, "x1": 36, "y1": 36, "group": "trunk"},
        {"x0": 28, "y0": 28, "x1": 36, "y1": 36, "group": "left"},
        {"x0": 16, "y0": 42, "x1": 24, "y1": 50, "group": "left"},
        {"x0": 28, "y0": 28, "x1": 36, "y1": 36, "group": "right"},
        {"x0": 42, "y0": 42, "x1": 50, "y1": 50, "group": "right"},
    ]

    asset = asset_from_box_annotations(boxes, pixel_spacing_mm=(0.4, 0.4),
                                       merge_tolerance_mm=0.5)

    assert len(asset.edges) == 3
    degree = {}
    for edge in asset.edges:
        degree[edge.node_a] = degree.get(edge.node_a, 0) + 1
        degree[edge.node_b] = degree.get(edge.node_b, 0) + 1
    assert max(degree.values()) == 3

    trunk = next(edge for edge in asset.edges if edge.node_a == asset.device_spawn.node_id)
    wire = np.asarray(trunk.centerline_mm, float)
    scene = FluoroSensor(res=24, nu=32, nv=32, n_samples=60).render_scene(
        wire, radius=0.2, contrast_asset=asset, mu_contrast=0.18)
    assert scene["image"].max() > 0.0
    assert scene["masks"]["vessel"].sum() > scene["masks"]["device"].sum()


def test_planar_import_report_summarizes_graph_geometry_and_warnings():
    boxes = [
        {"x0": 28, "y0": 0, "x1": 36, "y1": 8, "group": "trunk"},
        {"x0": 28, "y0": 14, "x1": 36, "y1": 22, "group": "trunk"},
        {"x0": 28, "y0": 28, "x1": 36, "y1": 36, "group": "trunk"},
        {"x0": 28, "y0": 28, "x1": 36, "y1": 36, "group": "left"},
        {"x0": 16, "y0": 42, "x1": 24, "y1": 50, "group": "left"},
        {"x0": 28, "y0": 28, "x1": 36, "y1": 36, "group": "right"},
        {"x0": 42, "y0": 42, "x1": 50, "y1": 50, "group": "right"},
    ]
    asset = asset_from_box_annotations(boxes, pixel_spacing_mm=(0.4, 0.4),
                                       merge_tolerance_mm=0.5)

    report = asset_planar_import_report(asset, source="boxes.json", image_shape_px=(96, 96),
                                        annotation_count=len(boxes))

    assert report["ok"] is True
    assert report["source"] == "boxes.json"
    assert report["image_shape_px"] == [96, 96]
    assert report["annotation_count"] == 7
    assert report["nodes"] == len(asset.nodes)
    assert report["edges"] == 3
    assert report["branch_nodes"] == 1
    assert report["total_centerline_mm"] > 20.0
    assert report["radius_mm"]["min"] > 0.0
    assert report["bounds_mm"]["x"][0] < report["bounds_mm"]["x"][1]
    assert report["bounds_mm"]["y"][0] < report["bounds_mm"]["y"][1]
    assert report["warnings"] == []


def test_planar_import_report_records_source_image_direction():
    boxes = [
        {"x0": 28, "y0": 0, "x1": 36, "y1": 8, "group": "trunk"},
        {"x0": 28, "y0": 14, "x1": 36, "y1": 22, "group": "trunk"},
    ]
    asset = asset_from_box_annotations(boxes, pixel_spacing_mm=(0.4, 0.4))

    report = asset_planar_import_report(
        asset,
        source="rotated-frame.dcm",
        image_shape_px=(96, 96),
        image_direction=(0.0, -1.0, 1.0, 0.0),
    )

    assert report["image_direction"] == [0.0, -1.0, 1.0, 0.0]
    assert report["warnings"] == []


def test_coco_box_annotations_load_with_attributes(tmp_path):
    src = tmp_path / "coco.json"
    src.write_text(
        """
        {
          "images": [{"id": 7, "file_name": "frame.png", "width": 128, "height": 96}],
          "annotations": [
            {"id": 1, "image_id": 7, "bbox": [10, 20, 8, 12],
             "attributes": {"group": "trunk", "order": 0, "radius_mm": 1.2}},
            {"id": 2, "image_id": 7, "bbox": [16, 38, 10, 10],
             "attributes": {"group": "trunk", "order": 1}},
            {"id": 3, "image_id": 7, "bbox": [34, 52, 12, 8],
             "attributes": {"group": "right", "order": 0}}
          ],
          "categories": [{"id": 1, "name": "vessel"}]
        }
        """
    )

    boxes = load_box_annotations(src)

    assert len(boxes) == 3
    assert boxes[0].group == "trunk"
    assert boxes[0].order == 0
    assert boxes[0].radius_mm == 1.2
    assert (boxes[0].x_min, boxes[0].y_min, boxes[0].x_max, boxes[0].y_max) == (
        10.0, 20.0, 18.0, 32.0)
    assert boxes[2].group == "right"


def test_coco_polygon_segmentations_import_as_box_annotations(tmp_path):
    src = tmp_path / "coco_polygons.json"
    src.write_text(json.dumps({
        "images": [{"id": 7, "file_name": "frame.png", "width": 128, "height": 96}],
        "categories": [{"id": 1, "name": "trunk"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 7,
                "category_id": 1,
                "segmentation": [[10, 20, 18, 22, 16, 34, 9, 31]],
                "attributes": {"order": 0, "radius_mm": 1.2},
            },
            {
                "id": 2,
                "image_id": 7,
                "category_id": 1,
                "segmentation": [[12, 42, 24, 44, 22, 55, 11, 53]],
                "attributes": {"order": 1},
            },
        ],
    }))

    boxes = load_box_annotations(src, image_file="frame.png")

    assert len(boxes) == 2
    assert [box.group for box in boxes] == ["trunk", "trunk"]
    assert boxes[0].order == 0
    assert boxes[0].radius_mm == 1.2
    assert (boxes[0].x_min, boxes[0].y_min, boxes[0].x_max, boxes[0].y_max) == (
        9.0, 20.0, 18.0, 34.0)
    assert (boxes[1].x_min, boxes[1].y_min, boxes[1].x_max, boxes[1].y_max) == (
        11.0, 42.0, 24.0, 55.0)
    asset = asset_from_box_annotations(boxes, pixel_spacing_mm=(0.5, 0.5))
    assert len(asset.edges) == 1
    assert np.asarray(asset.edges[0].R, float)[0, 0] == 1.2


def test_coco_uncompressed_rle_segmentations_import_as_box_annotations(tmp_path):
    first_mask = np.zeros((10, 12), dtype=bool)
    first_mask[2:7, 3:6] = True
    second_mask = np.zeros((10, 12), dtype=bool)
    second_mask[7:9, 8:11] = True
    src = tmp_path / "coco_rle.json"
    src.write_text(json.dumps({
        "images": [{"id": 7, "file_name": "frame.png", "width": 12, "height": 10}],
        "categories": [{"id": 1, "name": "trunk"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 7,
                "category_id": 1,
                "segmentation": _coco_uncompressed_rle(first_mask),
                "attributes": {"order": 0, "radius_mm": 1.2},
            },
            {
                "id": 2,
                "image_id": 7,
                "category_id": 1,
                "segmentation": _coco_uncompressed_rle(second_mask),
                "attributes": {"order": 1},
            },
        ],
    }))

    boxes = load_box_annotations(src, image_file="frame.png")

    assert len(boxes) == 2
    assert [box.group for box in boxes] == ["trunk", "trunk"]
    assert boxes[0].radius_mm == 1.2
    assert (boxes[0].x_min, boxes[0].y_min, boxes[0].x_max, boxes[0].y_max) == (
        3.0, 2.0, 6.0, 7.0)
    assert (boxes[1].x_min, boxes[1].y_min, boxes[1].x_max, boxes[1].y_max) == (
        8.0, 7.0, 11.0, 9.0)
    asset = asset_from_box_annotations(boxes, pixel_spacing_mm=(0.5, 0.5))
    assert len(asset.edges) == 1


def test_coco_box_annotations_filter_multi_image_exports(tmp_path):
    src = tmp_path / "multi_image_coco.json"
    src.write_text(
        """
        {
          "images": [
            {"id": 7, "file_name": "frame_a.png", "width": 128, "height": 96},
            {"id": 8, "file_name": "frame_b.png", "width": 128, "height": 96}
          ],
          "annotations": [
            {"id": 1, "image_id": 7, "bbox": [10, 20, 8, 12],
             "attributes": {"group": "trunk", "order": 0}},
            {"id": 2, "image_id": 7, "bbox": [18, 42, 10, 10],
             "attributes": {"group": "trunk", "order": 1}},
            {"id": 3, "image_id": 8, "bbox": [70, 10, 12, 8],
             "attributes": {"group": "other", "order": 0}}
          ],
          "categories": [{"id": 1, "name": "vessel"}]
        }
        """
    )

    try:
        load_box_annotations(src)
    except ValueError as e:
        assert "image_id or image_file" in str(e)
    else:
        raise AssertionError("multi-image COCO import should require an image selector")

    boxes_by_id = load_box_annotations(src, image_id=7)
    boxes_by_string_id = load_box_annotations(src, image_id="7")
    boxes_by_file = load_box_annotations(src, image_file="frame_a.png")

    assert [b.group for b in boxes_by_id] == ["trunk", "trunk"]
    assert [b.group for b in boxes_by_string_id] == ["trunk", "trunk"]
    assert [b.group for b in boxes_by_file] == ["trunk", "trunk"]
    assert all(b.x_min < 30.0 for b in boxes_by_id)


def test_coco_box_annotations_match_image_file_basename(tmp_path):
    src = tmp_path / "nested_coco.json"
    src.write_text(
        """
        {
          "images": [
            {"id": 7, "file_name": "study_a/frame_a.png", "width": 128, "height": 96},
            {"id": 8, "file_name": "study_b/frame_b.png", "width": 128, "height": 96}
          ],
          "annotations": [
            {"id": 1, "image_id": 7, "bbox": [10, 20, 8, 12],
             "attributes": {"group": "trunk", "order": 0}},
            {"id": 2, "image_id": 7, "bbox": [18, 42, 10, 10],
             "attributes": {"group": "trunk", "order": 1}},
            {"id": 3, "image_id": 8, "bbox": [70, 10, 12, 8],
             "attributes": {"group": "other", "order": 0}}
          ]
        }
        """
    )

    boxes = load_box_annotations(src, image_file="frame_a.png")

    assert [b.group for b in boxes] == ["trunk", "trunk"]


def test_coco_box_annotations_match_signed_url_image_file_basename(tmp_path):
    src = tmp_path / "signed_url_coco.json"
    src.write_text(
        """
        {
          "images": [
            {"id": 7, "file_name": "https://example.test/study/frame_a.png?token=abc#view",
             "width": 128, "height": 96},
            {"id": 8, "file_name": "https://example.test/study/frame_b.png?token=def",
             "width": 128, "height": 96}
          ],
          "annotations": [
            {"id": 1, "image_id": 7, "bbox": [10, 20, 8, 12],
             "attributes": {"group": "trunk", "order": 0}},
            {"id": 2, "image_id": 7, "bbox": [18, 42, 10, 10],
             "attributes": {"group": "trunk", "order": 1}},
            {"id": 3, "image_id": 8, "bbox": [70, 10, 12, 8],
             "attributes": {"group": "other", "order": 0}}
          ]
        }
        """
    )

    boxes = load_box_annotations(src, image_file="frame_a.png")

    assert [b.group for b in boxes] == ["trunk", "trunk"]


def test_coco_box_annotations_reject_mismatched_image_selectors(tmp_path):
    src = tmp_path / "multi_image_coco.json"
    src.write_text(
        """
        {
          "images": [
            {"id": 7, "file_name": "frame_a.png", "width": 128, "height": 96},
            {"id": 8, "file_name": "frame_b.png", "width": 128, "height": 96}
          ],
          "annotations": [
            {"id": 1, "image_id": 7, "bbox": [10, 20, 8, 12]},
            {"id": 2, "image_id": 8, "bbox": [70, 10, 12, 8]}
          ]
        }
        """
    )

    try:
        load_box_annotations(src, image_id=8, image_file="frame_a.png")
    except ValueError as e:
        assert "image_id" in str(e)
        assert "image_file" in str(e)
    else:
        raise AssertionError("mismatched COCO selectors should be rejected")


def test_coco_box_annotations_reject_ambiguous_image_file_basename(tmp_path):
    src = tmp_path / "ambiguous_coco.json"
    src.write_text(
        """
        {
          "images": [
            {"id": 7, "file_name": "study_a/frame.png", "width": 128, "height": 96},
            {"id": 8, "file_name": "study_b/frame.png", "width": 128, "height": 96}
          ],
          "annotations": [
            {"id": 1, "image_id": 7, "bbox": [10, 20, 8, 12]},
            {"id": 2, "image_id": 8, "bbox": [70, 10, 12, 8]}
          ]
        }
        """
    )

    try:
        load_box_annotations(src, image_file="frame.png")
    except ValueError as e:
        assert "multiple images" in str(e)
    else:
        raise AssertionError("ambiguous COCO image_file basename should be rejected")


def test_via_rectangle_regions_import_by_image_file(tmp_path):
    src = tmp_path / "via.json"
    src.write_text(
        """
        {
          "_via_img_metadata": {
            "frame_a.png123": {
              "filename": "/data/study/frame_a.png",
              "regions": [
                {
                  "shape_attributes": {
                    "name": "rect",
                    "x": 10,
                    "y": 20,
                    "width": 8,
                    "height": 12
                  },
                  "region_attributes": {
                    "group": "trunk",
                    "order": "0",
                    "radius_mm": "1.2"
                  }
                },
                {
                  "shape_attributes": {
                    "name": "rect",
                    "x": 18,
                    "y": 42,
                    "width": 10,
                    "height": 10
                  },
                  "region_attributes": {
                    "group": "trunk",
                    "order": "1"
                  }
                }
              ]
            },
            "frame_b.png456": {
              "filename": "/data/study/frame_b.png",
              "regions": [
                {
                  "shape_attributes": {
                    "name": "rect",
                    "x": 70,
                    "y": 10,
                    "width": 12,
                    "height": 8
                  },
                  "region_attributes": {
                    "group": "other",
                    "order": "0"
                  }
                }
              ]
            }
          }
        }
        """
    )

    try:
        load_box_annotations(src)
    except ValueError as e:
        assert "image_file" in str(e)
    else:
        raise AssertionError("multi-image VIA import should require image_file")

    boxes = load_box_annotations(src, image_file="frame_a.png")

    assert [b.group for b in boxes] == ["trunk", "trunk"]
    assert [b.order for b in boxes] == [0.0, 1.0]
    assert boxes[0].radius_mm == 1.2
    assert (boxes[0].x_min, boxes[0].y_min, boxes[0].x_max, boxes[0].y_max) == (
        10.0, 20.0, 18.0, 32.0)
    assert all(b.x_max < 30.0 for b in boxes)


def test_via_rectangle_regions_import_flat_file_mapping(tmp_path):
    src = tmp_path / "via_flat.json"
    src.write_text(
        """
        {
          "frame.png123": {
            "filename": "frame.png",
            "regions": {
              "0": {
                "shape_attributes": {"name": "rect", "x": 10, "y": 20, "width": 8, "height": 12},
                "region_attributes": {"label": "vessel", "order": 0}
              }
            }
          }
        }
        """
    )

    boxes = load_box_annotations(src, group_key="label")

    assert len(boxes) == 1
    assert boxes[0].group == "vessel"
    assert boxes[0].order == 0


def test_cvat_xml_box_annotations_load_image_boxes(tmp_path):
    src = tmp_path / "cvat.xml"
    src.write_text(
        """
        <annotations>
          <image id="0" name="frame_a.png" width="128" height="96">
            <box label="trunk" xtl="10" ytl="20" xbr="18" ybr="32">
              <attribute name="order">0</attribute>
            </box>
            <box label="trunk" xtl="18" ytl="42" xbr="28" ybr="52">
              <attribute name="order">1</attribute>
              <attribute name="radius_mm">2.5</attribute>
            </box>
          </image>
        </annotations>
        """
    )

    boxes = load_box_annotations(src)

    assert [b.group for b in boxes] == ["trunk", "trunk"]
    assert [(b.x_min, b.y_min, b.x_max, b.y_max) for b in boxes] == [
        (10.0, 20.0, 18.0, 32.0),
        (18.0, 42.0, 28.0, 52.0),
    ]
    assert [b.order for b in boxes] == [0.0, 1.0]
    assert boxes[1].radius_mm == 2.5


def test_cvat_xml_multi_image_export_requires_image_file(tmp_path):
    src = tmp_path / "multi_image_cvat.xml"
    src.write_text(
        """
        <annotations>
          <image id="0" name="/data/frame_a.png" width="128" height="96">
            <box label="trunk" xtl="10" ytl="20" xbr="18" ybr="32"/>
            <box label="trunk" xtl="18" ytl="42" xbr="28" ybr="52"/>
          </image>
          <image id="1" name="/data/frame_b.png" width="128" height="96">
            <box label="other" xtl="70" ytl="10" xbr="82" ybr="18"/>
          </image>
        </annotations>
        """
    )

    try:
        load_box_annotations(src)
    except ValueError as e:
        assert "image_file" in str(e)
    else:
        raise AssertionError("multi-image CVAT import should require image_file")

    boxes = load_box_annotations(src, image_file="frame_a.png")

    assert [b.group for b in boxes] == ["trunk", "trunk"]
    assert all(b.x_max < 30.0 for b in boxes)


def test_pascal_voc_xml_box_annotations_load_object_bndboxes(tmp_path):
    src = tmp_path / "frame_a.xml"
    src.write_text(
        """
        <annotation>
          <folder>study</folder>
          <filename>frame_a.png</filename>
          <path>/data/study/frame_a.png</path>
          <object>
            <name>trunk</name>
            <bndbox>
              <xmin>10</xmin>
              <ymin>20</ymin>
              <xmax>18</xmax>
              <ymax>32</ymax>
            </bndbox>
          </object>
          <object>
            <name>trunk</name>
            <order>1</order>
            <radius_mm>2.5</radius_mm>
            <bndbox>
              <xmin>18</xmin>
              <ymin>42</ymin>
              <xmax>28</xmax>
              <ymax>52</ymax>
            </bndbox>
          </object>
        </annotation>
        """
    )

    boxes = load_box_annotations(src, image_file="frame_a.png")

    assert [b.group for b in boxes] == ["trunk", "trunk"]
    assert [(b.x_min, b.y_min, b.x_max, b.y_max) for b in boxes] == [
        (10.0, 20.0, 18.0, 32.0),
        (18.0, 42.0, 28.0, 52.0),
    ]
    assert [b.order for b in boxes] == [0.0, 1.0]
    assert boxes[1].radius_mm == 2.5


def test_label_studio_rectangle_exports_load_percent_boxes(tmp_path):
    src = tmp_path / "label_studio.json"
    src.write_text(
        """
        [
          {
            "annotations": [
              {
                "result": [
                  {
                    "type": "rectanglelabels",
                    "original_width": 200,
                    "original_height": 100,
                    "value": {
                      "x": 10,
                      "y": 20,
                      "width": 15,
                      "height": 10,
                      "rectanglelabels": ["trunk"]
                    }
                  },
                  {
                    "type": "rectanglelabels",
                    "original_width": 200,
                    "original_height": 100,
                    "value": {
                      "x": 12,
                      "y": 42,
                      "width": 16,
                      "height": 8,
                      "rectanglelabels": ["trunk"]
                    }
                  },
                  {
                    "type": "rectanglelabels",
                    "original_width": 200,
                    "original_height": 100,
                    "value": {
                      "x": 58,
                      "y": 70,
                      "width": 18,
                      "height": 12,
                      "rectanglelabels": ["right"]
                    }
                  }
                ]
              }
            ]
          }
        ]
        """
    )

    boxes = load_box_annotations(src)
    asset = asset_from_box_annotations(boxes, pixel_spacing_mm=(0.5, 0.5),
                                       min_boxes_per_edge=1)

    assert len(boxes) == 3
    assert [b.group for b in boxes] == ["trunk", "trunk", "right"]
    assert boxes[0].order == 0
    assert np.allclose(
        [boxes[0].x_min, boxes[0].y_min, boxes[0].x_max, boxes[0].y_max],
        [20.0, 20.0, 50.0, 30.0],
    )
    assert len(asset.edges) == 2


def test_label_studio_multi_task_export_requires_image_file(tmp_path):
    src = tmp_path / "label_studio_multi.json"
    src.write_text(
        """
        [
          {
            "data": {"image": "/data/upload/frame_a.png"},
            "annotations": [{"result": [
              {"type": "rectanglelabels", "original_width": 200, "original_height": 100,
               "value": {"x": 10, "y": 20, "width": 15, "height": 10,
                         "rectanglelabels": ["trunk"]}},
              {"type": "rectanglelabels", "original_width": 200, "original_height": 100,
               "value": {"x": 12, "y": 42, "width": 16, "height": 8,
                         "rectanglelabels": ["trunk"]}}
            ]}]
          },
          {
            "data": {"image": "/data/upload/frame_b.png"},
            "annotations": [{"result": [
              {"type": "rectanglelabels", "original_width": 200, "original_height": 100,
               "value": {"x": 70, "y": 20, "width": 12, "height": 8,
                         "rectanglelabels": ["other"]}}
            ]}]
          }
        ]
        """
    )

    try:
        load_box_annotations(src)
    except ValueError as e:
        assert "image_file" in str(e)
    else:
        raise AssertionError("multi-task Label Studio import should require image_file")

    boxes = load_box_annotations(src, image_file="frame_a.png")

    assert [b.group for b in boxes] == ["trunk", "trunk"]


def test_label_studio_image_file_selector_matches_signed_url_basename(tmp_path):
    src = tmp_path / "label_studio_urls.json"
    src.write_text(
        """
        [
          {
            "data": {"image": "https://example.test/uploads/frame_a.png?token=abc#view"},
            "annotations": [{"result": [
              {"type": "rectanglelabels", "original_width": 200, "original_height": 100,
               "value": {"x": 10, "y": 20, "width": 15, "height": 10,
                         "rectanglelabels": ["trunk"]}},
              {"type": "rectanglelabels", "original_width": 200, "original_height": 100,
               "value": {"x": 12, "y": 42, "width": 16, "height": 8,
                         "rectanglelabels": ["trunk"]}}
            ]}]
          },
          {
            "data": {"image": "https://example.test/uploads/frame_b.png?token=def"},
            "annotations": [{"result": [
              {"type": "rectanglelabels", "original_width": 200, "original_height": 100,
               "value": {"x": 70, "y": 20, "width": 12, "height": 8,
                         "rectanglelabels": ["other"]}}
            ]}]
          }
        ]
        """
    )

    boxes = load_box_annotations(src, image_file="frame_a.png")

    assert [b.group for b in boxes] == ["trunk", "trunk"]


def test_yolo_box_annotations_require_image_size_and_import_asset(tmp_path):
    src = tmp_path / "labels.txt"
    src.write_text(
        "\n".join([
            "0 0.250000 0.200000 0.100000 0.120000",
            "0 0.260000 0.420000 0.110000 0.100000",
            "1 0.640000 0.700000 0.120000 0.080000",
            "",
        ])
    )

    try:
        load_box_annotations(src)
    except ValueError as e:
        assert "image_size_px" in str(e)
    else:
        raise AssertionError("YOLO normalized labels should require image_size_px")

    boxes = load_box_annotations(src, image_size_px=(200, 100))
    asset = asset_from_box_annotations(boxes, pixel_spacing_mm=(0.5, 0.5),
                                       min_boxes_per_edge=1)

    assert [b.group for b in boxes] == ["class_0", "class_0", "class_1"]
    assert boxes[0].order == 0
    assert np.allclose(
        [boxes[0].x_min, boxes[0].y_min, boxes[0].x_max, boxes[0].y_max],
        [40.0, 14.0, 60.0, 26.0],
    )
    assert len(asset.edges) == 2


def test_planar_mask_imports_curved_vessel_and_renders():
    asset = asset_from_planar_mask(
        _curved_planar_mask(),
        pixel_spacing_mm=(0.35, 0.35),
        origin_mm=(100.0, 200.0, 5.0),
        z_mm=2.0,
        samples=40,
        min_component_pixels=20,
    )
    edge = asset.edges[0]
    pts = np.asarray(edge.centerline_mm, float)
    radii = np.asarray(edge.R, float)[:, 0]

    assert asset.frame.name == "image_xy_mm"
    assert asset.frame.spacing_mm == (0.35, 0.35, 1.0)
    assert asset.provenance == "2d-mask(imported)"
    assert len(pts) >= 20
    assert np.ptp(pts[:, 0]) > 30.0
    assert np.ptp(pts[:, 1]) > 5.0
    assert np.allclose(pts[:, 2], 7.0)
    assert radii.min() > 0.0

    scene = FluoroSensor(res=24, nu=32, nv=32, n_samples=60).render_scene(
        pts, radius=0.2, contrast_asset=asset, mu_contrast=0.18)
    assert scene["image"].max() > 0.0
    assert scene["masks"]["vessel"].sum() > scene["masks"]["device"].sum()


def test_planar_mask_import_preserves_visible_bifurcation_graph():
    asset = asset_from_planar_mask(
        _branched_planar_mask(),
        pixel_spacing_mm=(0.4, 0.4),
        min_component_pixels=20,
    )

    degree = {}
    for edge in asset.edges:
        degree[edge.node_a] = degree.get(edge.node_a, 0) + 1
        degree[edge.node_b] = degree.get(edge.node_b, 0) + 1

    assert len(asset.edges) == 3
    assert max(degree.values()) == 3
    spawn = next(node for node in asset.nodes if node.id == asset.device_spawn.node_id)
    assert spawn.position_mm[1] < 8.0
    assert all(np.asarray(edge.R, float).min() > 0.0 for edge in asset.edges)

    trunk = next(edge for edge in asset.edges if edge.node_a == asset.device_spawn.node_id)
    wire = np.asarray(trunk.centerline_mm, float)
    scene = FluoroSensor(res=24, nu=32, nv=32, n_samples=60).render_scene(
        wire, radius=0.2, contrast_asset=asset, mu_contrast=0.18)
    assert scene["masks"]["vessel"].sum() > scene["masks"]["device"].sum()


def test_planar_array_loader_preserves_npz_spacing_and_origin(tmp_path):
    mask = _curved_planar_mask()
    path = tmp_path / "frame_mask.npz"
    np.savez(path, mask=mask, pixel_spacing_mm=np.array([0.31, 0.42]),
             origin_mm=np.array([12.0, 34.0, 5.0]))

    image = load_planar_array(path)

    assert image.data.shape == mask.shape
    assert image.pixel_spacing_mm == (0.31, 0.42)
    assert image.origin_mm == (12.0, 34.0, 5.0)


def test_planar_array_loader_reads_png_preview_image_without_imaging_extra(tmp_path):
    from lumen.sensors import write_png

    rgb = np.zeros((12, 16, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(16, dtype=np.uint8)[None, :] * 8
    rgb[..., 1] = 40
    path = tmp_path / "frame.png"
    write_png(path, rgb)

    image = load_planar_array(path)

    assert image.data.shape == (12, 16)
    assert image.pixel_spacing_mm == (1.0, 1.0)
    assert image.origin_mm == (0.0, 0.0, 0.0)
    assert image.data[:, -1].mean() > image.data[:, 0].mean()


def test_planar_array_loader_reads_16bit_grayscale_png_mask(tmp_path):
    data = np.zeros((10, 14), dtype=np.uint16)
    data[:, 7:] = 4096
    data[2:8, 3:6] = 65535
    path = tmp_path / "mask16.png"
    _write_gray16_png(path, data)

    image = load_planar_array(path)

    assert image.data.dtype == np.uint16
    assert image.data.shape == data.shape
    assert np.array_equal(image.data, data)
    assert image.pixel_spacing_mm == (1.0, 1.0)


def test_planar_array_loader_reads_palette_png_label_indices(tmp_path):
    labels = np.zeros((10, 14), dtype=np.uint8)
    labels[:, 7:] = 2
    labels[2:8, 3:6] = 1
    path = tmp_path / "labels.png"
    _write_palette_png(path, labels)

    image = load_planar_array(path)

    assert image.data.dtype == np.uint8
    assert image.data.shape == labels.shape
    assert np.array_equal(image.data, labels)
    assert sorted(np.unique(image.data).tolist()) == [0, 1, 2]


def test_planar_array_loader_reads_1bit_grayscale_png_mask(tmp_path):
    labels = np.zeros((9, 15), dtype=np.uint8)
    labels[:, 7:] = 1
    labels[2:7, 3:6] = 1
    path = tmp_path / "binary_mask.png"
    _write_gray1_png(path, labels)

    image = load_planar_array(path)

    assert image.data.dtype == np.uint8
    assert image.data.shape == labels.shape
    assert np.array_equal(image.data, labels)
    assert sorted(np.unique(image.data).tolist()) == [0, 1]


def test_planar_array_loader_selects_npy_stack_frame_index(tmp_path):
    frames = np.zeros((3, 16, 18), dtype=np.float32)
    frames[2, 3:10, 4:12] = 1.0
    path = tmp_path / "cine_stack.npy"
    np.save(path, frames)

    image = load_planar_array(path, frame_index=2)

    assert image.data.shape == (16, 18)
    assert np.array_equal(image.data, frames[2])
    assert image.pixel_spacing_mm == (1.0, 1.0)
    assert image.origin_mm == (0.0, 0.0, 0.0)


def test_planar_array_loader_requires_frame_index_for_npy_stack(tmp_path):
    frames = np.zeros((3, 16, 18), dtype=np.float32)
    path = tmp_path / "cine_stack.npy"
    np.save(path, frames)

    try:
        load_planar_array(path)
    except ValueError as e:
        assert "frame_index" in str(e)
    else:
        raise AssertionError("NPY frame stacks should require frame_index")


def test_planar_array_loader_selects_npz_stack_frame_index(tmp_path):
    frames = np.zeros((3, 16, 18), dtype=np.float32)
    frames[1, 4:12, 5:14] = 1.0
    path = tmp_path / "cine_stack.npz"
    np.savez(path, volume=frames, pixel_spacing_mm=np.array([0.2, 0.3]),
             origin_mm=np.array([4.0, 5.0, 6.0]))

    image = load_planar_array(path, frame_index=1)

    assert image.data.shape == (16, 18)
    assert np.array_equal(image.data, frames[1])
    assert image.pixel_spacing_mm == (0.2, 0.3)
    assert image.origin_mm == (4.0, 5.0, 6.0)


def test_planar_array_loader_requires_frame_index_for_npz_stack(tmp_path):
    frames = np.zeros((3, 16, 18), dtype=np.float32)
    path = tmp_path / "cine_stack.npz"
    np.savez(path, volume=frames)

    try:
        load_planar_array(path)
    except ValueError as e:
        assert "frame_index" in str(e)
    else:
        raise AssertionError("NPZ frame stacks should require frame_index")


def test_dicom_frame_loader_preserves_spacing_and_origin(monkeypatch):
    class FakeImage:
        def GetDimension(self):
            return 2

        def GetDirection(self):
            return (1.0, 0.0, 0.0, 1.0)

        def GetSpacing(self):
            return (0.22, 0.33)

        def GetOrigin(self):
            return (11.0, 22.0)

    class FakeSITK:
        @staticmethod
        def ReadImage(path):
            assert path.endswith("frame.dcm")
            return FakeImage()

        @staticmethod
        def GetArrayFromImage(image):
            return np.ones((8, 12), dtype=np.float32)

    monkeypatch.setitem(sys.modules, "SimpleITK", FakeSITK)

    frame = load_dicom_frame("frame.dcm")

    assert frame.data.shape == (8, 12)
    assert frame.pixel_spacing_mm == (0.22, 0.33)
    assert frame.origin_mm == (11.0, 22.0, 0.0)


def test_dicom_frame_loader_accepts_nonidentity_direction_metadata(monkeypatch):
    class FakeImage:
        def GetDimension(self):
            return 2

        def GetDirection(self):
            return (0.0, -1.0, 1.0, 0.0)

        def GetSpacing(self):
            return (0.22, 0.33)

        def GetOrigin(self):
            return (11.0, 22.0)

    class FakeSITK:
        @staticmethod
        def ReadImage(path):
            assert path.endswith("rotated-frame.dcm")
            return FakeImage()

        @staticmethod
        def GetArrayFromImage(image):
            return np.arange(12, dtype=np.float32).reshape(3, 4)

    monkeypatch.setitem(sys.modules, "SimpleITK", FakeSITK)

    frame = load_dicom_frame("rotated-frame.dcm")

    assert frame.data.shape == (3, 4)
    assert frame.pixel_spacing_mm == (0.22, 0.33)
    assert frame.origin_mm == (11.0, 22.0, 0.0)
    assert frame.direction == (0.0, -1.0, 1.0, 0.0)


def test_dicom_frame_loader_applies_window_and_monochrome1_presentation(monkeypatch):
    class FakeImage:
        _meta = {
            "0028|0004": "MONOCHROME1",
            "0028|1050": "0",
            "0028|1051": "100",
        }

        def GetDimension(self):
            return 2

        def GetDirection(self):
            return (1.0, 0.0, 0.0, 1.0)

        def GetSpacing(self):
            return (0.4, 0.4)

        def GetOrigin(self):
            return (0.0, 0.0)

        def HasMetaDataKey(self, key):
            return key in self._meta

        def GetMetaData(self, key):
            return self._meta[key]

    class FakeSITK:
        @staticmethod
        def ReadImage(path):
            assert path.endswith("angio.dcm")
            return FakeImage()

        @staticmethod
        def GetArrayFromImage(image):
            return np.array([[-100.0, -50.0, 0.0, 50.0, 100.0]], dtype=np.float32)

    monkeypatch.setitem(sys.modules, "SimpleITK", FakeSITK)

    frame = load_dicom_frame("angio.dcm")

    assert frame.data.dtype == np.float32
    assert np.allclose(frame.data, [[1.0, 1.0, 0.5, 0.0, 0.0]])


def test_dicom_frame_loader_applies_rescale_before_window(monkeypatch):
    class FakeImage:
        _meta = {
            "0028|0004": "MONOCHROME2",
            "0028|1050": "0",
            "0028|1051": "100",
            "0028|1052": "-100",  # Rescale Intercept
            "0028|1053": "2",     # Rescale Slope
        }

        def GetDimension(self):
            return 2

        def GetDirection(self):
            return (1.0, 0.0, 0.0, 1.0)

        def GetSpacing(self):
            return (0.4, 0.4)

        def GetOrigin(self):
            return (0.0, 0.0)

        def HasMetaDataKey(self, key):
            return key in self._meta

        def GetMetaData(self, key):
            return self._meta[key]

    class FakeSITK:
        @staticmethod
        def ReadImage(path):
            assert path.endswith("rescale.dcm")
            return FakeImage()

        @staticmethod
        def GetArrayFromImage(image):
            return np.array([[0.0, 25.0, 50.0, 75.0, 100.0]], dtype=np.float32)

    monkeypatch.setitem(sys.modules, "SimpleITK", FakeSITK)

    frame = load_dicom_frame("rescale.dcm")

    assert frame.data.dtype == np.float32
    assert np.allclose(frame.data, [[0.0, 0.0, 0.5, 1.0, 1.0]])


def test_dicom_frame_loader_uses_image_file_reader_metadata(monkeypatch):
    calls = []

    class FakeImage:
        def GetDimension(self):
            return 2

        def GetDirection(self):
            return (1.0, 0.0, 0.0, 1.0)

        def GetSpacing(self):
            return (0.4, 0.4)

        def GetOrigin(self):
            return (0.0, 0.0)

    class FakeReader:
        _meta = {
            "0028|0004": "MONOCHROME2",
            "0028|1050": "10",
            "0028|1051": "20",
        }

        def SetFileName(self, path):
            calls.append(("SetFileName", path))

        def LoadPrivateTagsOn(self):
            calls.append(("LoadPrivateTagsOn",))

        def ReadImageInformation(self):
            calls.append(("ReadImageInformation",))

        def Execute(self):
            calls.append(("Execute",))
            return FakeImage()

        def HasMetaDataKey(self, key):
            return key in self._meta

        def GetMetaData(self, key):
            return self._meta[key]

    class FakeSITK:
        @staticmethod
        def ImageFileReader():
            return FakeReader()

        @staticmethod
        def ReadImage(path):
            raise AssertionError("load_dicom_frame should prefer ImageFileReader metadata")

        @staticmethod
        def GetArrayFromImage(image):
            return np.array([[0.0, 10.0, 20.0]], dtype=np.float32)

    monkeypatch.setitem(sys.modules, "SimpleITK", FakeSITK)

    frame = load_dicom_frame("reader-meta.dcm")

    assert calls == [
        ("SetFileName", "reader-meta.dcm"),
        ("LoadPrivateTagsOn",),
        ("ReadImageInformation",),
        ("Execute",),
    ]
    assert np.allclose(frame.data, [[0.0, 0.5, 1.0]])


def test_dicom_frame_loader_requires_index_for_multiframe(monkeypatch):
    class FakeImage:
        def GetDimension(self):
            return 3

        def GetDirection(self):
            return (1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0)

        def GetSpacing(self):
            return (0.2, 0.3, 1.0)

        def GetOrigin(self):
            return (4.0, 5.0, 0.0)

    class FakeSITK:
        @staticmethod
        def ReadImage(path):
            assert path.endswith("cine.dcm")
            return FakeImage()

        @staticmethod
        def GetArrayFromImage(image):
            return np.stack([
                np.full((4, 5), 1.0, dtype=np.float32),
                np.full((4, 5), 2.0, dtype=np.float32),
            ])

    monkeypatch.setitem(sys.modules, "SimpleITK", FakeSITK)

    try:
        load_dicom_frame("cine.dcm")
    except ValueError as e:
        assert "frame_index" in str(e)
    else:
        raise AssertionError("multi-frame DICOM import should require frame_index")


def test_dicom_frame_loader_selects_multiframe_index(monkeypatch):
    class FakeImage:
        def GetDimension(self):
            return 3

        def GetDirection(self):
            return (1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0)

        def GetSpacing(self):
            return (0.2, 0.3, 1.0)

        def GetOrigin(self):
            return (4.0, 5.0, 0.0)

    class FakeSITK:
        @staticmethod
        def ReadImage(path):
            assert path.endswith("cine.dcm")
            return FakeImage()

        @staticmethod
        def GetArrayFromImage(image):
            return np.stack([
                np.full((4, 5), 1.0, dtype=np.float32),
                np.full((4, 5), 2.0, dtype=np.float32),
                np.full((4, 5), 3.0, dtype=np.float32),
            ])

    monkeypatch.setitem(sys.modules, "SimpleITK", FakeSITK)

    frame = load_dicom_frame("cine.dcm", frame_index=1)

    assert frame.data.shape == (4, 5)
    assert np.all(frame.data == 2.0)
    assert frame.pixel_spacing_mm == (0.2, 0.3)
    assert frame.origin_mm == (4.0, 5.0, 0.0)
