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
            fallback = next((name for name in SPLIT_NAMES if remaining[name] > 0), None)
            if fallback is None:
                raise ValueError("no split capacity remaining for group assignment")
            split = fallback
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
    # bool is a subclass of int in Python; exclude it so True/False aren't ratios.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_string(value: object, field: str, manifest_path: Path) -> None:
    if not isinstance(value, str):
        raise ValueError(f"split manifest {manifest_path} {field} must be a string")


def _validate_int(value: object, field: str, manifest_path: Path) -> None:
    # Seeds may be negative (random.Random accepts them); only exclude bool/non-int.
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"split manifest {manifest_path} {field} must be an integer")


def _validated_split_count(value: object, field: str, manifest_path: Path) -> int:
    if not _is_int_count(value):
        raise ValueError(f"split manifest {manifest_path} {field} must be a non-negative integer")
    return cast(int, value)


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
    for field in ("source_index", "out_dir", "group_by"):
        _validate_string(raw.get(field), field, manifest_path)
    _validate_int(raw.get("seed"), "seed", manifest_path)
    total_records = _validated_split_count(raw.get("records"), "records", manifest_path)
    total_episodes = _validated_split_count(raw.get("episodes"), "episodes", manifest_path)

    stratify_obj = raw.get("stratify")
    if not isinstance(stratify_obj, list) or any(not isinstance(field, str) for field in stratify_obj):
        raise ValueError(f"split manifest {manifest_path} stratify must be a list of strings")
    assignments_obj = raw.get("assignments")
    if not isinstance(assignments_obj, dict):
        raise ValueError(f"split manifest {manifest_path} assignments must be an object")
    assignment_counts: Counter[SplitName] = Counter()
    # JSON object parsing leaves one value per group key; validate that parsed map.
    for group, split_value in assignments_obj.items():
        if not isinstance(group, str):
            raise ValueError(f"split manifest {manifest_path} assignments must map strings to train, val, or test")
        if split_value not in SPLIT_NAMES:
            raise ValueError(f"split manifest {manifest_path} invalid split assignment: {split_value!r}")
        assignment_counts[cast(SplitName, split_value)] += 1

    ratios_obj = raw.get("ratios")
    splits_obj = raw.get("splits")
    if not isinstance(ratios_obj, dict) or set(ratios_obj) != set(SPLIT_NAMES):
        raise ValueError(f"split manifest {manifest_path} ratios must contain train, val, and test")
    ratios: dict[SplitName, float] = {}
    for name in SPLIT_NAMES:
        ratio_value = ratios_obj[name]
        if not _is_number(ratio_value) or ratio_value < 0.0:
            raise ValueError(f"split manifest {manifest_path} ratios must be non-negative numbers")
        ratios[name] = float(ratio_value)
    if not isinstance(splits_obj, dict) or set(splits_obj) != set(SPLIT_NAMES):
        raise ValueError(f"split manifest {manifest_path} splits must contain train, val, and test")
    split_record_total = 0
    split_episode_total = 0

    for split in SPLIT_NAMES:
        summary = splits_obj[split]
        if not isinstance(summary, dict):
            raise ValueError(f"split manifest {manifest_path} split {split!r} summary must be an object")
        summary_records = _validated_split_count(
            summary.get("records"), f"split {split!r} records", manifest_path
        )
        summary_episodes = _validated_split_count(
            summary.get("episodes"), f"split {split!r} episodes", manifest_path
        )
        _validate_count_map(summary.get("labels"), f"split {split!r} labels", manifest_path)
        _validate_count_map(summary.get("modalities"), f"split {split!r} modalities", manifest_path)
        split_record_total += summary_records
        split_episode_total += summary_episodes
        if assignment_counts[split] != summary_episodes:
            raise ValueError(
                f"split manifest {manifest_path} split {split!r} episode count does not match assignments"
            )

    ratio_total = sum(ratios[name] for name in SPLIT_NAMES)
    # ``set(ratios_obj)`` was validated above, so every split name is present here.
    if ratio_total == 0.0:
        raise ValueError(f"split manifest {manifest_path} ratios must include at least one positive value")
    if split_record_total != total_records:
        raise ValueError(f"split manifest {manifest_path} split record counts do not match records")
    if split_episode_total != total_episodes:
        raise ValueError(f"split manifest {manifest_path} split episode counts do not match episodes")
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
