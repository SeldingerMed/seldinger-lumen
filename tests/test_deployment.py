import json

import numpy as np
import pytest

from lumen.deployment import (
    BenchTrace,
    BenchTraceMetadata,
    BenchValidationCriteria,
    DeploymentConfig,
    SafetyEnvelope,
    check_safety_envelope,
    run_deployment,
    validate_bench_trace,
)


def _trace_metadata(source, **overrides):
    values = {
        "source": source,
        "force_units": "N",
        "torque_units": "N*m",
        "device_id": "guidewire-014-v1",
        "phantom_id": "phantom-a-v1",
        "actuation_protocol": "advance-rotate-v1",
        "calibration_reference": "calibration-2026-08-v1",
        "provenance": "procedural-phantom-a",
    }
    values.update(overrides)
    return BenchTraceMetadata(**values)
class FakeInterface:
    def __init__(self, telemetry_rows=None):
        self.telemetry_rows = list(telemetry_rows or [{"force_n": 0.2, "torque_nm": -0.1}])
        self.observations = 0
        self.actions = []
        self.stopped = False
        self.closed = False

    def observe(self):
        self.observations += 1
        return np.array([0.25, 0.5], dtype=np.float32)

    def apply_action(self, action):
        self.actions.append(np.asarray(action).copy())

    def telemetry(self):
        index = min(len(self.actions), len(self.telemetry_rows) - 1)
        return self.telemetry_rows[index]

    def safe_stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


def test_deployment_loop_validates_actions_and_delivers_step_hook():
    interface = FakeInterface()
    seen = []
    result = run_deployment(
        interface,
        lambda obs: np.array([0.5, -0.25]),
        config=DeploymentConfig(max_steps=2),
        on_step=seen.append,
    )

    assert result.status == "max_steps"
    assert result.protocol_valid and result.safe
    assert result.steps == 2
    assert result.preflight_telemetry == {"force_n": 0.2, "torque_nm": -0.1}
    assert len(interface.actions) == 2
    assert len(seen) == 2
    assert seen[0].action.dtype == np.float32
    assert interface.observations == 3
    assert interface.stopped and interface.closed
    assert json.loads(json.dumps(result.to_dict()))["safe"] is True


def test_predict_policy_tuple_is_supported():
    interface = FakeInterface()

    class PredictPolicy:
        def predict(self, observation, deterministic):
            assert deterministic is True
            return np.array([0.1, 0.2]), {"state": 1}

    result = run_deployment(interface, PredictPolicy(), config=DeploymentConfig(max_steps=1))
    assert result.protocol_valid and result.steps == 1
    np.testing.assert_allclose(interface.actions[0], [0.1, 0.2])


@pytest.mark.parametrize(
    "policy, message",
    [
        (lambda _obs: np.array([np.nan, 0.0]), "finite"),
        (lambda _obs: np.array([1.1, 0.0]), "lie in"),
        (lambda _obs: np.array([0.0]), "shape"),
    ],
)
def test_invalid_policy_action_stops_before_actuation(policy, message):
    interface = FakeInterface()
    result = run_deployment(interface, policy, config=DeploymentConfig(max_steps=1))

    assert result.status == "error"
    assert message in result.error
    assert result.steps == 0
    assert interface.actions == []
    assert interface.stopped and interface.closed


def test_unsafe_telemetry_is_reported_and_stopped():
    interface = FakeInterface([
        {"force_n": 0.1, "torque_nm": 0.1},
        {"force_n": -2.5, "torque_nm": 0.1},
    ])
    seen = []
    result = run_deployment(
        interface,
        lambda _obs: [0.0, 0.0],
        config=DeploymentConfig(
            max_steps=4,
            envelope=SafetyEnvelope(max_force_n=2.0, max_torque_nm=1.0),
        ),
        on_step=seen.append,
    )

    assert result.status == "unsafe"
    assert result.steps == 1
    assert result.violations == ["force_n=2.5 exceeds limit 2"]
    assert not result.safe and interface.stopped and interface.closed
    assert len(seen) == 1 and not seen[0].safety.safe

def test_initially_unsafe_telemetry_blocks_first_action():
    interface = FakeInterface([{"force_n": 3.0, "torque_nm": 0.0}])
    called = []
    result = run_deployment(
        interface,
        lambda _obs: called.append(True) or [0.0, 0.0],
        config=DeploymentConfig(
            max_steps=2,
            envelope=SafetyEnvelope(max_force_n=2.0),
        ),
    )

    assert result.status == "unsafe"
    assert result.steps == 0
    assert result.preflight_telemetry == {"force_n": 3.0, "torque_nm": 0.0}
    assert called == [] and interface.actions == []
    assert interface.stopped and interface.closed


def test_missing_configured_telemetry_fails_closed():
    check = check_safety_envelope({"force_n": 0.1}, SafetyEnvelope(max_torque_nm=1.0))
    assert not check.safe
    assert check.violations == ("missing telemetry: torque_nm",)


def test_safety_envelope_uses_force_and_torque_magnitude():
    check = check_safety_envelope(
        {"force_n": -0.5, "torque_nm": -0.25},
        SafetyEnvelope(max_force_n=0.5, max_torque_nm=0.25),
    )
    assert check.safe
    assert check.values == {"force_n": 0.5, "torque_nm": 0.25}

@pytest.mark.parametrize("key", ["penetration_mm", "ood_score"])
def test_nonnegative_safety_signals_reject_negative_values(key):
    envelope = SafetyEnvelope(max_penetration_mm=1.0, max_ood_score=1.0)
    telemetry = {"penetration_mm": 0.0, "ood_score": 0.0}
    telemetry[key] = -0.001
    check = check_safety_envelope(telemetry, envelope)
    assert not check.safe
    assert check.violations == (f"negative telemetry: {key}",)


def test_nonnegative_safety_signals_accept_zero_and_exact_limit():
    envelope = SafetyEnvelope(max_penetration_mm=1.0, max_ood_score=1.0)
    check = check_safety_envelope(
        {"penetration_mm": 0.0, "ood_score": 0.0},
        envelope,
    )
    assert check.safe
    check = check_safety_envelope(
        {"penetration_mm": 1.0, "ood_score": 1.0},
        envelope,
    )
    assert check.safe


def test_bench_trace_interpolates_and_checks_whip_behavior():
    simulated = BenchTrace(
        time_s=[0.0, 1.0, 2.0],
        force_n=[0.0, 2.0, 4.0],
        torque_nm=[0.0, 1.0, 0.0],
        metadata=_trace_metadata("simulation"),
    )
    measured = BenchTrace(
        time_s=[0.0, 0.5, 1.5, 2.0],
        force_n=[0.0, 1.0, 3.0, 4.0],
        torque_nm=[0.0, 0.5, 0.5, 0.0],
        metadata=_trace_metadata("bench"),
    )
    report = validate_bench_trace(
        simulated,
        measured,
        BenchValidationCriteria(
            force_rmse_max_n=0.0,
            torque_rmse_max_nm=0.0,
            force_peak_error_max_n=0.0,
            torque_peak_error_max_nm=0.0,
            max_whip_event_delta=0,
            whip_deadband_nm=0.05,
        ),
    )

    assert report.passed
    assert report.failures == ()
    assert report.metrics["force_rmse_n"] == pytest.approx(0.0)
    assert report.metrics["torque_rmse_nm"] == pytest.approx(0.0)
    assert report.metrics["simulated_whip_events"] == 1
    assert report.metrics["measured_whip_events"] == 1
    payload = report.to_dict()
    assert payload["protocol"] == "lumen-bench-validation/1"
    assert payload["criteria"]["force_rmse_max_n"] == 0.0
    assert payload["criteria"]["whip_deadband_nm"] == 0.05

def test_whip_counts_use_common_measured_grid_for_unequal_rates():
    simulated = BenchTrace(
        [0.0, 0.25, 0.5, 0.75, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, -1.0, 0.0],
        _trace_metadata("simulation"),
    )
    measured = BenchTrace(
        [0.0, 0.5, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        _trace_metadata("bench"),
    )
    report = validate_bench_trace(
        simulated,
        measured,
        BenchValidationCriteria(
            force_rmse_max_n=0.0,
            torque_rmse_max_nm=0.0,
            max_whip_event_delta=0,
            whip_deadband_nm=0.01,
        ),
    )
    assert report.passed
    assert report.metrics["simulated_whip_events"] == 0
    assert report.metrics["measured_whip_events"] == 0


def test_bench_trace_reports_criterion_and_coverage_failures():
    simulated = BenchTrace([0.0, 1.0], [0.0, 1.0], [0.0, 1.0],
                            _trace_metadata("simulation"))
    measured = BenchTrace([-0.1, 0.5], [0.0, 4.0], [0.0, 3.0],
                          _trace_metadata("bench"))
    report = validate_bench_trace(
        simulated,
        measured,
        BenchValidationCriteria(force_rmse_max_n=0.1, torque_rmse_max_nm=0.1),
    )
    assert not report.passed
    assert report.failures == ("measured time range is outside simulated trace coverage",)

    measured = BenchTrace([0.0, 1.0], [0.0, 4.0], [0.0, 3.0],
                          _trace_metadata("bench"))
    report = validate_bench_trace(
        simulated,
        measured,
        BenchValidationCriteria(force_rmse_max_n=0.1, torque_rmse_max_nm=0.1),
    )
    assert not report.passed
    assert "force RMSE exceeds criterion" in report.failures
    assert "torque RMSE exceeds criterion" in report.failures

@pytest.mark.parametrize("field", ["force_rmse_max_n", "torque_rmse_max_nm"])
def test_bench_criteria_require_rmse_thresholds(field):
    values = {"force_rmse_max_n": 0.1, "torque_rmse_max_nm": 0.1}
    values[field] = None
    with pytest.raises(ValueError, match=field):
        BenchValidationCriteria(**values)

def test_bench_trace_rejects_unmatched_identity_and_persists_metadata():
    with pytest.raises(ValueError, match="calibration_reference"):
        BenchTraceMetadata(
            source="simulation",
            force_units="N",
            torque_units="N*m",
            device_id="guidewire",
            phantom_id="phantom",
            actuation_protocol="advance-rotate",
            calibration_reference="uncalibrated",
            provenance="procedural",
        )
    simulated = BenchTrace([0.0, 1.0], [0.0, 1.0], [0.0, 1.0],
                            _trace_metadata("simulation"))
    measured = BenchTrace(
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        _trace_metadata("bench", force_units="native"),
    )
    report = validate_bench_trace(
        simulated,
        measured,
        BenchValidationCriteria(force_rmse_max_n=0.1, torque_rmse_max_nm=0.1),
    )
    assert not report.passed
    assert report.metrics["metadata_match"] == 0
    assert any("force_units mismatch" in failure for failure in report.failures)
    payload = report.to_dict()
    assert payload["simulated_metadata"]["source"] == "simulation"
    assert payload["measured_metadata"]["force_units"] == "native"


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"max_steps": 0}, "max_steps"),
        ({"action_shape": (0, 2)}, "action_shape"),
        ({"action_low": 1.0, "action_high": 1.0}, "action_low"),
    ],
)
def test_deployment_config_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        DeploymentConfig(**kwargs)


def test_bench_trace_rejects_non_monotonic_time_and_wrong_lengths():
    with pytest.raises(ValueError, match="strictly increasing"):
        BenchTrace([0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                   _trace_metadata("simulation"))
    with pytest.raises(ValueError, match="equal lengths"):
        BenchTrace([0.0, 1.0], [0.0], [0.0, 0.0],
                   _trace_metadata("simulation"))


def test_run_deployment_records_interface_errors_and_cleanup_errors():
    class BrokenInterface(FakeInterface):
        def safe_stop(self):
            raise RuntimeError("stop link down")

        def close(self):
            raise RuntimeError("close link down")

    result = run_deployment(
        BrokenInterface(),
        lambda _obs: np.array([np.inf, 0.0]),
        config=DeploymentConfig(max_steps=1),
    )
    assert isinstance(result.error, str) and "finite" in result.error
    assert "safe stop failed" in result.cleanup_error
    assert "close failed" in result.cleanup_error
