"""Deterministic train/validation/test splits for Lumen dataloader indexes."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
import json
from pathlib import Path
import random
from typing import Iterable

SPLIT_NAMES = ("train", "val", "test")
DEFAULT_RATIOS = (0.8, 0.1, 0.1)
DEFAULT_STRATIFY_FIELDS = ("label", "obs_modality")


def _read_jsonl(index_path: str | Path) -> list[dict]:
    rows = []
    with open(index_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"line {line_no}: invalid JSON: {e.msg}") from e
            if not isinstance(row, dict):
                raise ValueError(f"line {line_no}: expected JSON object, got {type(row).__name__}")
            rows.append(row)
    if not rows:
        raise ValueError("index has no records")
    return rows


def _normalized_ratios(ratios: Iterable[float]) -> tuple[float, float, float]:
    values = tuple(float(value) for value in ratios)
    if len(values) != 3:
        raise ValueError("ratios must contain exactly three values: train val test")
    if any(value < 0.0 for value in values):
        raise ValueError("ratios must be non-negative")
    total = sum(values)
    if total <= 0.0:
        raise ValueError("at least one ratio must be positive")
    return tuple(value / total for value in values)  # type: ignore[return-value]


def _target_counts(n_groups: int, ratios: tuple[float, float, float]) -> dict[str, int]:
    raw = [n_groups * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    remaining = n_groups - sum(counts)
    order = sorted(range(3), key=lambda i: (raw[i] - counts[i], ratios[i]), reverse=True)
    for i in order[:remaining]:
        counts[i] += 1
    return dict(zip(SPLIT_NAMES, counts, strict=True))


def _stratum_key(record: dict, fields: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(record.get(field, "<missing>")) for field in fields)


def _group_records(rows: list[dict], group_by: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        group = row.get(group_by)
        if not group:
            raise ValueError(f"record missing non-empty group field {group_by!r}")
        groups[str(group)].append(row)
    return dict(groups)


def _interleaved_groups(groups: dict[str, list[dict]], stratify_fields: tuple[str, ...], seed: int) -> list[str]:
    rng = random.Random(seed)
    strata: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for group, rows in groups.items():
        keys = {_stratum_key(row, stratify_fields) for row in rows}
        if len(keys) > 1:
            raise ValueError(
                f"group {group!r} has inconsistent stratify fields {list(stratify_fields)}: "
                f"found {len(keys)} distinct values across {len(rows)} rows"
            )
        strata[keys.pop()].append(group)

    queues = []
    for key in sorted(strata):
        members = sorted(strata[key])
        rng.shuffle(members)
        queues.append(deque(members))

    order = []
    while any(queues):
        for queue in queues:
            if queue:
                order.append(queue.popleft())
    return order


def _assign_groups(groups: dict[str, list[dict]], ratios: tuple[float, float, float],
                   stratify_fields: tuple[str, ...], seed: int) -> dict[str, str]:
    targets = _target_counts(len(groups), ratios)
    remaining = dict(targets)
    assignments = {}
    for group in _interleaved_groups(groups, stratify_fields, seed):
        split = max(SPLIT_NAMES, key=lambda name: (remaining[name], -SPLIT_NAMES.index(name)))
        if remaining[split] <= 0:
            split = next(name for name in SPLIT_NAMES if remaining[name] > 0)
        assignments[group] = split
        remaining[split] -= 1
    return assignments


def _summarize_split(rows: list[dict], group_by: str) -> dict:
    return {
        "records": len(rows),
        "episodes": len({str(row.get(group_by, "<missing>")) for row in rows}),
        "labels": dict(sorted(Counter(str(row.get("label", "<missing>")) for row in rows).items())),
        "modalities": dict(sorted(Counter(str(row.get("obs_modality", "<missing>")) for row in rows).items())),
    }


def split_index_records(index_path: str | Path, out_dir: str | Path,
                        ratios: Iterable[float] = DEFAULT_RATIOS, seed: int = 0,
                        stratify_fields: Iterable[str] = DEFAULT_STRATIFY_FIELDS,
                        group_by: str = "episode") -> dict:
    """Write deterministic train/val/test JSONL splits for an existing index.

    Splits are assigned at episode granularity by default, so frames from one
    procedure never leak across train/validation/test folds. ``stratify_fields``
    controls the round-robin ordering used before filling the requested global
    split quotas; it is intentionally lightweight and dependency-free.
    """
    ratios = _normalized_ratios(ratios)
    stratify_fields = tuple(str(field) for field in stratify_fields)

    rows = _read_jsonl(index_path)
    groups = _group_records(rows, group_by)
    assignments = _assign_groups(groups, ratios, stratify_fields, int(seed))

    split_rows = {name: [] for name in SPLIT_NAMES}
    for row in rows:
        split_rows[assignments[str(row[group_by])]].append(row)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLIT_NAMES:
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for row in split_rows[split]:
                f.write(json.dumps(row, sort_keys=True) + "\n")

    manifest = {
        "source_index": str(index_path),
        "out_dir": str(out_dir),
        "group_by": group_by,
        "seed": int(seed),
        "ratios": dict(zip(SPLIT_NAMES, ratios, strict=True)),
        "stratify": list(stratify_fields),
        "records": len(rows),
        "episodes": len(groups),
        "assignments": dict(sorted(assignments.items())),
        "splits": {name: _summarize_split(split_rows[name], group_by) for name in SPLIT_NAMES},
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest
