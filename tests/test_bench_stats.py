import pytest

from lumen.bench_stats import (
    STATS_PROTOCOL_VERSION,
    bootstrap_iqm_ci,
    bootstrap_iqm_difference_ci,
    iqm,
    summarize_metrics,
    validate_statistics,
)


def test_iqm_uses_fractional_quartile_trimming():
    assert iqm([0.0, 1.0, 2.0, 3.0]) == pytest.approx(1.5)
    assert iqm([0.0, 1.0, 2.0, 3.0, 4.0]) == pytest.approx(2.0)
    assert iqm([1.0, 2.0]) == pytest.approx(1.5)


def test_bootstrap_iqm_is_reproducible_and_bounded():
    values = [0.0, 1.0, 2.0, 10.0]
    first = bootstrap_iqm_ci(values, n_resamples=300, seed=17)
    second = bootstrap_iqm_ci(values, n_resamples=300, seed=17)

    assert first == second
    assert first.lower <= first.estimate <= first.upper
    assert first.confidence == 0.95
    assert first.resamples == 300
    assert first.seed == 17


def test_bootstrap_difference_uses_iqm_difference_estimate():
    interval = bootstrap_iqm_difference_ci(
        [1.0, 2.0, 3.0, 4.0],
        [0.0, 1.0, 2.0, 3.0],
        n_resamples=200,
        seed=3,
    )

    assert interval.statistic == "iqm_difference"
    assert interval.estimate == pytest.approx(1.0)
    assert interval.lower <= interval.estimate <= interval.upper


def test_summarize_metrics_records_frozen_protocol_metadata():
    payload = summarize_metrics(
        {"success_rate": [1.0, 0.0, 1.0, 1.0], "mean_return": [2.0, 4.0, 8.0, 10.0]},
        n_resamples=200,
        seed=9,
    )

    assert payload["protocol"] == STATS_PROTOCOL_VERSION
    assert payload["n_resamples"] == 200
    assert payload["metrics"]["success_rate"]["n"] == 4
    assert payload["metrics"]["mean_return"]["mean"] == pytest.approx(6.0)
    assert payload["metrics"]["success_rate"]["seed"] != payload["metrics"]["mean_return"]["seed"]
    assert validate_statistics(payload) is payload


def test_statistics_reject_empty_nonfinite_and_invalid_parameters():
    with pytest.raises(ValueError, match="must not be empty"):
        iqm([])
    with pytest.raises(ValueError, match="finite"):
        iqm([1.0, float("nan")])
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_iqm_ci([1.0], confidence=1.0)
    with pytest.raises(ValueError, match="positive integer"):
        bootstrap_iqm_ci([1.0], n_resamples=0)


def test_statistics_validator_rejects_tampered_interval():
    payload = summarize_metrics({"score": [1.0, 2.0, 3.0]}, n_resamples=100)
    payload["metrics"]["score"]["bootstrap_ci"]["lower"] = 4.0
    with pytest.raises(ValueError, match="finite and ordered"):
        validate_statistics(payload)
