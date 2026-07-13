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


def _valid_manifest():
    return {
        "source_index": "index.jsonl",
        "out_dir": "splits",
        "group_by": "episode",
        "seed": 0,
        "ratios": {"train": 1.0, "val": 0.0, "test": 0.0},
        "stratify": ["label"],
        "records": 1,
        "episodes": 1,
        "assignments": {"case_a": "train"},
        "splits": {
            "train": {
                "records": 1,
                "episodes": 1,
                "labels": {"success": 1},
                "modalities": {"fluoro": 1},
            },
            "val": {"records": 0, "episodes": 0, "labels": {}, "modalities": {}},
            "test": {"records": 0, "episodes": 0, "labels": {}, "modalities": {}},
        },
    }


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
    assert from_dir["splits"]["train"]["labels"]["success"] == 2
    assert from_dir["splits"]["train"]["modalities"]["fluoro"] == 3


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

    invalid = _valid_manifest()
    invalid["ratios"]["train"] = "1.0"
    manifest_path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="ratios must be non-negative numbers"):
        read_split_manifest(manifest_path)

    invalid = _valid_manifest()
    invalid["ratios"]["train"] = True
    manifest_path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="ratios must be non-negative numbers"):
        read_split_manifest(manifest_path)

    invalid = _valid_manifest()
    invalid["ratios"]["train"] = -0.5
    manifest_path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="ratios must be non-negative numbers"):
        read_split_manifest(manifest_path)

    invalid = _valid_manifest()
    invalid["splits"]["train"]["records"] = True
    manifest_path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="records must be a non-negative integer"):
        read_split_manifest(manifest_path)

    invalid = _valid_manifest()
    invalid["splits"]["train"]["records"] = -1
    manifest_path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="records must be a non-negative integer"):
        read_split_manifest(manifest_path)

    invalid = _valid_manifest()
    invalid["splits"]["train"]["labels"] = {"success": "one"}
    manifest_path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="labels must map strings to integer counts"):
        read_split_manifest(manifest_path)

    invalid = _valid_manifest()
    invalid["splits"]["train"]["labels"] = {"success": True}
    manifest_path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="labels must map strings to integer counts"):
        read_split_manifest(manifest_path)

    invalid = _valid_manifest()
    invalid["splits"]["train"]["labels"] = {"success": -1}
    manifest_path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="labels must map strings to integer counts"):
        read_split_manifest(manifest_path)


def test_read_split_manifest_rejects_malformed_field_types(tmp_path):
    manifest_path = tmp_path / "manifest.json"

    invalid = _valid_manifest()
    invalid["source_index"] = 123
    manifest_path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="source_index must be a string"):
        read_split_manifest(manifest_path)

    invalid = _valid_manifest()
    invalid["stratify"] = ["label", 7]
    manifest_path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="stratify must be a list of strings"):
        read_split_manifest(manifest_path)

    invalid = _valid_manifest()
    invalid["assignments"] = {"case_a": "holdout"}
    manifest_path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="assignments must map strings to train, val, or test"):
        read_split_manifest(manifest_path)

    invalid = _valid_manifest()
    invalid["seed"] = True
    manifest_path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="seed must be an integer"):
        read_split_manifest(manifest_path)

    # Negative seeds are valid (random.Random accepts them) and must round-trip.
    ok = _valid_manifest()
    ok["seed"] = -7
    manifest_path.write_text(json.dumps(ok))
    assert read_split_manifest(manifest_path)["seed"] == -7


def test_read_split_manifest_rejects_inconsistent_summary_totals(tmp_path):
    manifest_path = tmp_path / "manifest.json"

    invalid = _valid_manifest()
    invalid["splits"]["train"]["records"] = 2
    manifest_path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="split record counts do not match records"):
        read_split_manifest(manifest_path)

    invalid = _valid_manifest()
    invalid["splits"]["train"]["episodes"] = 0
    manifest_path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="episode count does not match assignments"):
        read_split_manifest(manifest_path)

    invalid = _valid_manifest()
    invalid["episodes"] = 2
    manifest_path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="split episode counts do not match episodes"):
        read_split_manifest(manifest_path)

    invalid = _valid_manifest()
    invalid["ratios"] = {"train": 0.0, "val": 0.0, "test": 0.0}
    manifest_path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="at least one positive"):
        read_split_manifest(manifest_path)


def test_read_split_manifest_rejects_non_json_file_path(tmp_path):
    with pytest.raises(ValueError, match="directory or .json file"):
        read_split_manifest(tmp_path / "manifest.txt")
    with pytest.raises(ValueError, match="directory or .json file"):
        read_split_manifest(tmp_path / "manifest")
