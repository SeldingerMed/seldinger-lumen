"""Deterministic train/validation/test splits for Lumen dataloader indexes."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
import json
from pathlib import Path
import random
from typing import Iterable, Literal, TypedDict, cast

SplitName = Literal["train", "val", "test"]
IndexRecord = dict[str, object]
Ratios = tuple[float, float, float]


class SplitSummary(TypedDict):
    records: int
    episodes: int
    labels: dict[str, int]
    modalities: dict[str, int]


class SplitManifest(TypedDict):
    source_index: str
    out_dir: str
    group_by: str
    seed: int
    ratios: dict[SplitName, float]
    stratify: list[str]
    records: int
    episodes: int
    assignments: dict[str, SplitName]
    splits: dict[SplitName, SplitSummary]


SPLIT_NAMES: tuple[SplitName, SplitName, SplitName] = ("train", "val", "test")
DEFAULT_RATIOS: Ratios = (0.8, 0.1, 0.1)
DEFAULT_STRATIFY_FIELDS = ("label", "obs_modality")


def _read_jsonl(index_path: str | Path) -> list[IndexRecord]:
    rows: list[IndexRecord] = []
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
            rows.append(cast(IndexRecord, row))
    if not rows:
        raise ValueError("index has no records")
    return rows


def _normalized_ratios(ratios: Iterable[float]) -> Ratios:
    values = tuple(float(value) for value in ratios)
    if len(values) != 3:
        raise ValueError("ratios must contain exactly three values: train val test")
    if any(value < 0.0 for value in values):
        raise ValueError("ratios must be non-negative")
    total = sum(values)
    if total <= 0.0:
        raise ValueError("at least one ratio must be positive")
    return (values[0] / total, values[1] / total, values[2] / total)


def _target_counts(n_groups: int, ratios: Ratios) -> dict[SplitName, int]:
    raw = [n_groups * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    remaining = n_groups - sum(counts)
    order = sorted(range(3), key=lambda i: (raw[i] - counts[i], ratios[i]), reverse=True)
    for i in order[:remaining]:
        counts[i] += 1
    return dict(zip(SPLIT_NAMES, counts, strict=True))


def _stratum_key(record: IndexRecord, fields: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(record.get(field, "<missing>")) for field in fields)


def _group_records(rows: list[IndexRecord], group_by: str) -> dict[str, list[IndexRecord]]:
    groups: dict[str, list[IndexRecord]] = defaultdict(list)
    for row in rows:
        group = row.get(group_by)
        if not group:
            raise ValueError(f"record missing non-empty group field {group_by!r}")
        groups[str(group)].append(row)
    return dict(groups)


def _interleaved_groups(
    groups: dict[str, list[IndexRecord]], stratify_fields: tuple[str, ...], seed: int
) -> list[str]:
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


def _assign_groups(
    groups: dict[str, list[IndexRecord]], ratios: Ratios, stratify_fields: tuple[str, ...], seed: int
) -> dict[str, SplitName]:
    targets = _target_counts(len(groups), ratios)
    remaining = dict(targets)
    assignments: dict[str, SplitName] = {}
    for group in _interleaved_groups(groups, stratify_fields, seed):
        split = max(SPLIT_NAMES, key=lambda name: (remaining[name], -SPLIT_NAMES.index(name)))
        if remaining[split] <= 0:
            split = cast(SplitName, next(name for name in SPLIT_NAMES if remaining[name] > 0))
        assignments[group] = split
        remaining[split] -= 1
    return assignments


def _summarize_split(rows: list[IndexRecord], group_by: str) -> SplitSummary:
    # Labels/modalities are fixed dataloader-index conventions, independent of
    # the optional fields used to stratify group assignment.
    return {
        "records": len(rows),
        "episodes": len({str(row.get(group_by, "<missing>")) for row in rows}),
        "labels": dict(sorted(Counter(str(row.get("label", "<missing>")) for row in rows).items())),
        "modalities": dict(sorted(Counter(str(row.get("obs_modality", "<missing>")) for row in rows).items())),
    }


def _manifest_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_dir():
        return path / "manifest.json"
    if path.suffix == ".json":
        return path
    raise ValueError(f"split manifest path must be a directory or .json file, got {path}")


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_count_map(value: object, field: str, manifest_path: Path) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"split manifest {manifest_path} {field} must be an object")
    bad_items = [key for key, count in value.items() if not isinstance(key, str) or not _is_int_count(count)]
    if bad_items:
        raise ValueError(f"split manifest {manifest_path} {field} must map strings to integer counts")


def read_split_manifest(path: str | Path) -> SplitManifest:
    """Load and validate a split manifest produced by :func:`split_index_records`.

    ``path`` may point directly at ``manifest.json`` or at the split output directory.
    The validation catches missing/renamed fields, malformed split names, and
    non-numeric summary counts before downstream training code consumes bad folds.
    """
    manifest_path = _manifest_path(path)
    with open(manifest_path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"split manifest {manifest_path} must contain a JSON object")

    required = {
        "source_index", "out_dir", "group_by", "seed", "ratios", "stratify",
        "records", "episodes", "assignments", "splits",
    }
    missing = sorted(required.difference(raw))
    if missing:
        raise ValueError(f"split manifest {manifest_path} missing required fields: {missing}")
    ratios_obj = raw.get("ratios")
    splits_obj = raw.get("splits")
    if not isinstance(ratios_obj, dict) or set(ratios_obj) != set(SPLIT_NAMES):
        raise ValueError(f"split manifest {manifest_path} ratios must contain train, val, and test")
    if any(not _is_number(ratios_obj[name]) for name in SPLIT_NAMES):
        raise ValueError(f"split manifest {manifest_path} ratios must be numeric")
    if not isinstance(splits_obj, dict) or set(splits_obj) != set(SPLIT_NAMES):
        raise ValueError(f"split manifest {manifest_path} splits must contain train, val, and test")
    for split in SPLIT_NAMES:
        summary = splits_obj[split]
        if not isinstance(summary, dict):
            raise ValueError(f"split manifest {manifest_path} split {split!r} summary must be an object")
        for field in ("records", "episodes"):
            if not _is_int_count(summary.get(field)):
                raise ValueError(f"split manifest {manifest_path} split {split!r} {field} must be an integer")
        _validate_count_map(summary.get("labels"), f"split {split!r} labels", manifest_path)
        _validate_count_map(summary.get("modalities"), f"split {split!r} modalities", manifest_path)
    return cast(SplitManifest, raw)


def split_index_records(index_path: str | Path, out_dir: str | Path,
                        ratios: Iterable[float] = DEFAULT_RATIOS, seed: int = 0,
                        stratify_fields: Iterable[str] = DEFAULT_STRATIFY_FIELDS,
                        group_by: str = "episode") -> SplitManifest:
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

    split_rows: dict[SplitName, list[IndexRecord]] = {name: [] for name in SPLIT_NAMES}
    for row in rows:
        split_rows[assignments[str(row[group_by])]].append(row)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLIT_NAMES:
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for row in split_rows[split]:
                f.write(json.dumps(row, sort_keys=True) + "\n")

    manifest: SplitManifest = {
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
