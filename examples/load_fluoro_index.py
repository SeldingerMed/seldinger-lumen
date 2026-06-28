"""Load a fluoro JSONL index as a small CV training batch.

    python examples/load_fluoro_index.py episodes/fluoro.jsonl --limit 8

Run `lumen index <episodes_dir> --modality fluoro --require-cv-labels --out
episodes/fluoro.jsonl` first. For fixed-shape batching, check the index with
`lumen inspect-index episodes/fluoro.jsonl --require-uniform-arrays`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    # Allow `python examples/load_fluoro_index.py ...` from a source checkout
    # without requiring an editable install first.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lumen.data import iter_index_records


def _keypoint_uv(keypoints: dict, name: str) -> list[float]:
    kp = keypoints.get(name) if isinstance(keypoints, dict) else None
    if not isinstance(kp, dict) or not kp.get("present", True) or kp.get("uv") is None:
        return [float("nan"), float("nan")]
    return [float(kp["uv"][0]), float(kp["uv"][1])]


def load_batch(index_path, limit: int | None = 8) -> dict:
    """Load up to ``limit`` fluoro rows from a Lumen JSONL index into NumPy arrays."""
    rows = []
    for sample in iter_index_records(index_path, load_arrays=True):
        if sample.get("obs_modality") != "fluoro":
            continue
        rows.append(sample)
        if limit is not None and len(rows) >= limit:
            break
    if not rows:
        raise ValueError("index contains no fluoro rows")
    try:
        obs = np.stack([row["obs"] for row in rows])
        device_mask = np.stack([row["device_mask"] for row in rows])
        vessel_mask = np.stack([row["vessel_mask"] for row in rows])
    except ValueError as e:
        raise ValueError(
            "index rows are not fixed-shape; run `lumen inspect-index "
            f"{index_path} --require-uniform-arrays` before batching"
        ) from e
    return {
        "obs": obs,
        "device_mask": device_mask,
        "vessel_mask": vessel_mask,
        "tip_uv": np.asarray([_keypoint_uv(row.get("keypoints", {}), "tip") for row in rows],
                             dtype=float),
        "base_uv": np.asarray([_keypoint_uv(row.get("keypoints", {}), "base") for row in rows],
                              dtype=float),
        "labels": [row.get("label") for row in rows],
        "episodes": [row.get("episode") for row in rows],
        "step_index": np.asarray([row.get("step_index") for row in rows], dtype=int),
    }


def main(index_path=None, limit: int | None = 8) -> dict:
    if index_path is None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("index_path")
        parser.add_argument("--limit", type=int, default=8)
        args = parser.parse_args()
        index_path = args.index_path
        limit = args.limit
    batch = load_batch(index_path, limit=limit)
    print(f"obs: {batch['obs'].shape} {batch['obs'].dtype}")
    print(f"device_mask: {batch['device_mask'].shape} {batch['device_mask'].dtype}")
    print(f"vessel_mask: {batch['vessel_mask'].shape} {batch['vessel_mask'].dtype}")
    print(f"tip_uv: {batch['tip_uv'].shape} {batch['tip_uv'].dtype}")
    print(f"base_uv: {batch['base_uv'].shape} {batch['base_uv'].dtype}")
    print(f"labels: {sorted(set(batch['labels']))}")
    return batch


def cli() -> None:
    try:
        main()
    except ValueError as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    cli()
