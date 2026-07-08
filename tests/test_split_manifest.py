import json

import pytest

from lumen.data import read_split_manifest, split_index_records
from lumen.data.split import SplitManifest, SplitName, SplitSummary


def _write_index(path):
    records = [
        {"episode": "case_a", "label": "success", "obs_modality": "fluoro", "frame": 0},
        {"episode": "case_a", "label": "success", "obs_modality": "fluoro", "frame": 1},
        {"episode": "case_b", "label": "failure", "obs_modality": "fluoro", "frame": 0},
        {"episode": "case_c", "label": "success", "obs_modality": "luminal", "frame": 0},
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_split_manifest_public_types_are_importable():
    assert SplitName is not None
    assert SplitSummary is not None
    assert SplitManifest is not None


def test_read_split_manifest_roundtrips_from_directory_and_file(tmp_path):
    index_path = tmp_path / "index.jsonl"
    _write_index(index_path)
    out_dir = tmp_path / "splits"

    manifest = split_index_records(index_path, out_dir, ratios=(2, 1, 0), seed=3)

    from_dir = read_split_manifest(out_dir)
    from_file = read_split_manifest(out_dir / "manifest.json")
    assert from_dir == manifest == from_file
    assert from_dir["ratios"] == {"train": pytest.approx(2 / 3), "val": pytest.approx(1 / 3), "test": 0.0}
    assert set(from_dir["assignments"].values()).issubset({"train", "val", "test"})


def test_read_split_manifest_rejects_malformed_manifest(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"ratios": {"train": 1.0}, "splits": {}}))

    with pytest.raises(ValueError, match="missing required fields"):
        read_split_manifest(manifest_path)

    manifest_path.write_text(json.dumps({
        "source_index": "index.jsonl",
        "out_dir": "splits",
        "group_by": "episode",
        "seed": 0,
        "ratios": {"train": 1.0, "val": 0.0},
        "stratify": ["label"],
        "records": 1,
        "episodes": 1,
        "assignments": {"case_a": "train"},
        "splits": {"train": {}, "val": {}, "test": {}},
    }))
    with pytest.raises(ValueError, match="ratios must contain train, val, and test"):
        read_split_manifest(manifest_path)
