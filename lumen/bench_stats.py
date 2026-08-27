"""Deterministic interquartile-mean and bootstrap statistics for benchmark metrics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Integral

import numpy as np


STATS_PROTOCOL_VERSION = "lumen-stats/1"
DEFAULT_CONFIDENCE = 0.95
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
_BOOTSTRAP_CHUNK = 256


def _samples(values) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("samples must be finite numeric values") from exc
    if array.size == 0:
        raise ValueError("samples must not be empty")
    if not np.isfinite(array).all():
        raise ValueError("samples must be finite numeric values")
    return array


def _positive_int(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _seed(value) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError("seed must be a non-negative integer")
    value = int(value)
    if value < 0:
        raise ValueError("seed must be a non-negative integer")
    return value


def _confidence(value: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be strictly between 0 and 1") from exc
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    return value


def _iqm_rows(sorted_samples: np.ndarray) -> np.ndarray:
    """Compute IQM for rows that have already been sorted."""
    n = sorted_samples.shape[-1]
    trim = 0.25 * n
    left_full = int(np.floor(trim))
    fractional = trim - left_full
    weights = np.ones(n, dtype=np.float64)
    if left_full:
        weights[:left_full] = 0.0
        weights[-left_full:] = 0.0
    if fractional:
        weights[left_full] -= fractional
        weights[-left_full - 1] -= fractional
    return np.sum(sorted_samples * weights, axis=-1) / np.sum(weights)


def iqm(values) -> float:
    """Return the finite-sample interquartile mean with fractional trimming.

    The lowest and highest 25% of observations are removed, including fractional
    endpoint weights when the sample size is not divisible by four. This is the
    IQM used by robust RL evaluation protocols, rather than a mean of observations
    selected by an inclusive quantile mask.
    """
    array = np.sort(_samples(values))
    return float(_iqm_rows(array.reshape(1, -1))[0])


interquartile_mean = iqm


@dataclass(frozen=True)
class BootstrapCI:
    """A reproducible percentile bootstrap interval for an IQM estimate."""

    estimate: float
    lower: float
    upper: float
    confidence: float
    resamples: int
    seed: int
    statistic: str = "iqm"

    def to_dict(self) -> dict:
        return asdict(self)


def bootstrap_iqm_ci(
    values,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> BootstrapCI:
    """Return a deterministic percentile bootstrap CI for the sample IQM."""
    array = _samples(values)
    confidence = _confidence(confidence)
    n_resamples = _positive_int(n_resamples, "n_resamples")
    seed = _seed(seed)
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_resamples, dtype=np.float64)
    for start in range(0, n_resamples, _BOOTSTRAP_CHUNK):
        stop = min(start + _BOOTSTRAP_CHUNK, n_resamples)
        indices = rng.integers(0, array.size, size=(stop - start, array.size))
        resampled = np.sort(array[indices], axis=1)
        estimates[start:stop] = _iqm_rows(resampled)
    alpha = (1.0 - confidence) / 2.0
    return BootstrapCI(
        estimate=iqm(array),
        lower=float(np.quantile(estimates, alpha)),
        upper=float(np.quantile(estimates, 1.0 - alpha)),
        confidence=confidence,
        resamples=n_resamples,
        seed=seed,
    )


def bootstrap_iqm_difference_ci(
    first,
    second,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> BootstrapCI:
    """Return a deterministic independent bootstrap CI for IQM(first)-IQM(second)."""
    first = _samples(first)
    second = _samples(second)
    confidence = _confidence(confidence)
    n_resamples = _positive_int(n_resamples, "n_resamples")
    seed = _seed(seed)
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_resamples, dtype=np.float64)
    for start in range(0, n_resamples, _BOOTSTRAP_CHUNK):
        stop = min(start + _BOOTSTRAP_CHUNK, n_resamples)
        first_indices = rng.integers(0, first.size, size=(stop - start, first.size))
        second_indices = rng.integers(0, second.size, size=(stop - start, second.size))
        first_bootstrap = np.sort(first[first_indices], axis=1)
        second_bootstrap = np.sort(second[second_indices], axis=1)
        estimates[start:stop] = _iqm_rows(first_bootstrap) - _iqm_rows(second_bootstrap)
    alpha = (1.0 - confidence) / 2.0
    return BootstrapCI(
        estimate=iqm(first) - iqm(second),
        lower=float(np.quantile(estimates, alpha)),
        upper=float(np.quantile(estimates, 1.0 - alpha)),
        confidence=confidence,
        resamples=n_resamples,
        seed=seed,
        statistic="iqm_difference",
    )


def summarize_metric(
    values,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> dict:
    """Return mean, IQM, and a reproducible bootstrap IQM interval."""
    array = _samples(values)
    interval = bootstrap_iqm_ci(
        array,
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
    )
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "iqm": interval.estimate,
        "bootstrap_ci": {
            "lower": interval.lower,
            "upper": interval.upper,
        },
        "confidence": interval.confidence,
        "n_resamples": interval.resamples,
        "seed": interval.seed,
        "statistic": interval.statistic,
    }


def summarize_metrics(
    values: Mapping[str, object],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> dict:
    """Summarize each metric vector under one frozen statistics protocol."""
    if not isinstance(values, Mapping) or not values:
        raise ValueError("values must be a non-empty metric mapping")
    confidence = _confidence(confidence)
    n_resamples = _positive_int(n_resamples, "n_resamples")
    seed = _seed(seed)
    metrics = {}
    for offset, name in enumerate(sorted(values)):
        if not isinstance(name, str) or not name:
            raise ValueError("metric names must be non-empty strings")
        metrics[name] = summarize_metric(
            values[name],
            confidence=confidence,
            n_resamples=n_resamples,
            seed=seed + offset,
        )
    return {
        "protocol": STATS_PROTOCOL_VERSION,
        "confidence": confidence,
        "n_resamples": n_resamples,
        "seed": seed,
        "metrics": metrics,
    }


def validate_statistics(statistics: dict, expected_metrics=None) -> dict:
    """Validate a serialized statistics payload before it is published."""
    errors = []
    if not isinstance(statistics, dict):
        raise ValueError("statistics must be a mapping")
    if statistics.get("protocol") != STATS_PROTOCOL_VERSION:
        errors.append(f"statistics.protocol must be {STATS_PROTOCOL_VERSION!r}")
    metrics = statistics.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        errors.append("statistics.metrics must be a non-empty mapping")
        metrics = {}
    if expected_metrics is not None and set(metrics) != set(expected_metrics):
        errors.append("statistics.metrics must match the expected metric names")
    try:
        confidence = _confidence(statistics.get("confidence"))
    except ValueError:
        errors.append("statistics.confidence is invalid")
        confidence = None
    try:
        n_resamples = _positive_int(statistics.get("n_resamples"), "n_resamples")
    except ValueError:
        errors.append("statistics.n_resamples is invalid")
        n_resamples = None
    try:
        _seed(statistics.get("seed"))
    except ValueError:
        errors.append("statistics.seed is invalid")
    for name, summary in metrics.items():
        if not isinstance(summary, dict):
            errors.append(f"statistics.metrics.{name} must be a mapping")
            continue
        if not isinstance(summary.get("n"), Integral) or int(summary["n"]) <= 0:
            errors.append(f"statistics.metrics.{name}.n must be positive")
        for field in ("mean", "iqm"):
            try:
                if not np.isfinite(float(summary.get(field))):
                    raise ValueError
            except (TypeError, ValueError):
                errors.append(f"statistics.metrics.{name}.{field} must be finite")
        interval = summary.get("bootstrap_ci")
        if not isinstance(interval, dict):
            errors.append(f"statistics.metrics.{name}.bootstrap_ci must be a mapping")
        else:
            try:
                lower = float(interval["lower"])
                upper = float(interval["upper"])
                if not np.isfinite([lower, upper]).all() or lower > upper:
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                errors.append(f"statistics.metrics.{name}.bootstrap_ci must be finite and ordered")
        if summary.get("confidence") != confidence:
            errors.append(f"statistics.metrics.{name}.confidence must match the protocol")
        if summary.get("n_resamples") != n_resamples:
            errors.append(f"statistics.metrics.{name}.n_resamples must match the protocol")
        try:
            _seed(summary.get("seed"))
        except ValueError:
            errors.append(f"statistics.metrics.{name}.seed must be a non-negative integer")
        if summary.get("statistic") != "iqm":
            errors.append(f"statistics.metrics.{name}.statistic must be 'iqm'")
    if errors:
        raise ValueError("invalid statistics payload: " + "; ".join(errors))
    return statistics
