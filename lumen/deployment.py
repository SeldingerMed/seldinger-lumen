"""Deployment boundary and benchtop force/torque validation.

The open package cannot ship a robot or patient data. It does ship the contract that
private simulator, phantom, and hardware bridges implement: expose an observation,
accept one normalized action, report telemetry, and stop safely on a violation.

The benchtop protocol compares simulator and phantom traces in the same time domain.
Raw traces carry explicit units and calibration identity, but comparison is accepted
only after both traces are converted to canonical calibrated N and N*m units; it is
not a clinical or regulatory calibration claim.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


DEPLOYMENT_PROTOCOL_VERSION = "lumen-deployment/1"
BENCH_VALIDATION_PROTOCOL_VERSION = "lumen-bench-validation/1"


class DeploymentProtocolError(ValueError):
    """A policy, interface, or telemetry source violated the deployment contract."""


class DeploymentInterface(Protocol):
    """Minimal interface for a simulator, phantom rig, or private device bridge.

    ``action`` is normalized by :func:`run_deployment` before ``apply_action`` is
    called. Implementations must make ``safe_stop`` idempotent.
    """

    def observe(self) -> Any: ...

    def apply_action(self, action: np.ndarray) -> None: ...

    def telemetry(self) -> Mapping[str, Any]: ...

    def safe_stop(self) -> None: ...


@dataclass(frozen=True)
class SafetyEnvelope:
    """Runtime limits for absolute force/torque and scalar safety signals.

    ``None`` disables a limit. Force and torque are compared by magnitude; penetration
    and OOD scores are non-negative scalar values. Limits must come from the private
    device/phantom calibration rather than being inferred from this package.
    """

    max_force_n: float | None = None
    max_torque_nm: float | None = None
    max_penetration_mm: float | None = None
    max_ood_score: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("max_force_n", self.max_force_n),
            ("max_torque_nm", self.max_torque_nm),
            ("max_penetration_mm", self.max_penetration_mm),
            ("max_ood_score", self.max_ood_score),
        ):
            if value is None:
                continue
            try:
                normalized = float(value)
            except (TypeError, ValueError, OverflowError):
                normalized = float("nan")
            if isinstance(value, (bool, np.bool_)) or not np.isfinite(normalized) or normalized < 0.0:
                raise ValueError(f"{name} must be None or a finite non-negative number")
            object.__setattr__(self, name, normalized)


@dataclass(frozen=True)
class SafetyCheck:
    """Result of applying a :class:`SafetyEnvelope` to one telemetry sample."""

    safe: bool
    values: dict[str, float] = field(default_factory=dict)
    violations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "safe": self.safe,
            "values": dict(self.values),
            "violations": list(self.violations),
        }


def check_safety_envelope(
    telemetry: Mapping[str, Any], envelope: SafetyEnvelope
) -> SafetyCheck:
    """Fail closed when a configured safety signal is missing or non-finite."""
    if not isinstance(telemetry, Mapping):
        raise DeploymentProtocolError("telemetry must be a mapping")
    configured = {
        "force_n": envelope.max_force_n,
        "torque_nm": envelope.max_torque_nm,
        "penetration_mm": envelope.max_penetration_mm,
        "ood_score": envelope.max_ood_score,
    }
    values: dict[str, float] = {}
    violations: list[str] = []
    for name, limit in configured.items():
        if limit is None:
            continue
        if name not in telemetry:
            violations.append(f"missing telemetry: {name}")
            continue
        try:
            value = float(telemetry[name])
        except (TypeError, ValueError, OverflowError):
            violations.append(f"non-numeric telemetry: {name}")
            continue
        if not np.isfinite(value):
            violations.append(f"non-finite telemetry: {name}")
            continue
        if name in {"force_n", "torque_nm"}:
            value = abs(value)
        elif value < 0.0:
            violations.append(f"negative telemetry: {name}")
            continue
        values[name] = value
        if value > limit:
            violations.append(f"{name}={value:g} exceeds limit {limit:g}")
    return SafetyCheck(not violations, values, tuple(violations))


@dataclass(frozen=True)
class DeploymentConfig:
    """Action and loop validation settings for one deployment session."""

    max_steps: int = 100
    action_shape: tuple[int, ...] = (2,)
    action_low: float = -1.0
    action_high: float = 1.0
    envelope: SafetyEnvelope = field(default_factory=SafetyEnvelope)
    deterministic: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_steps, (bool, np.bool_))
            or not isinstance(self.max_steps, (int, np.integer))
            or self.max_steps <= 0
        ):
            raise ValueError("max_steps must be a positive integer")
        try:
            shape = tuple(self.action_shape)
        except TypeError as exc:
            raise ValueError("action_shape must contain positive dimensions") from exc
        if not shape or any(
            isinstance(dim, (bool, np.bool_))
            or not isinstance(dim, (int, np.integer))
            or dim <= 0
            for dim in shape
        ):
            raise ValueError("action_shape must contain positive dimensions")
        object.__setattr__(self, "action_shape", tuple(int(dim) for dim in shape))
        for name, value in (("action_low", self.action_low), ("action_high", self.action_high)):
            try:
                normalized = float(value)
            except (TypeError, ValueError, OverflowError):
                normalized = float("nan")
            if isinstance(value, (bool, np.bool_)) or not np.isfinite(normalized):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, normalized)
        if self.action_low >= self.action_high:
            raise ValueError("action_low must be less than action_high")
        if not isinstance(self.deterministic, (bool, np.bool_)):
            raise ValueError("deterministic must be a boolean")
        if not isinstance(self.envelope, SafetyEnvelope):
            raise ValueError("envelope must be a SafetyEnvelope")


@dataclass(frozen=True)
class DeploymentStep:
    """One action and the telemetry/safety result observed after it."""

    index: int
    observation: Any
    action: np.ndarray
    telemetry: Mapping[str, Any]
    safety: SafetyCheck


@dataclass
class DeploymentResult:
    """Audit record from one deployment loop."""

    status: str
    steps: int
    preflight_telemetry: dict[str, Any] | None = None
    actions: list[list[float]] = field(default_factory=list)
    telemetry: list[dict[str, Any]] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    error: str | None = None
    cleanup_error: str | None = None

    @property
    def protocol_valid(self) -> bool:
        return self.error is None

    @property
    def safe(self) -> bool:
        return (
            self.protocol_valid
            and self.cleanup_error is None
            and not self.violations
            and self.status != "unsafe"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": DEPLOYMENT_PROTOCOL_VERSION,
            "status": self.status,
            "steps": self.steps,
            "preflight_telemetry": _json_safe(self.preflight_telemetry),
            "actions": self.actions,
            "telemetry": [_json_safe(item) for item in self.telemetry],
            "violations": list(self.violations),
            "error": self.error,
            "cleanup_error": self.cleanup_error,
            "protocol_valid": self.protocol_valid,
            "safe": self.safe,
        }


@dataclass(frozen=True)
class BenchTraceMetadata:
    """Identity and calibration metadata required for trace comparison."""

    source: str
    force_units: str
    torque_units: str
    device_id: str
    phantom_id: str
    actuation_protocol: str
    calibration_reference: str
    provenance: str

    def __post_init__(self) -> None:
        fields = (
            "source",
            "force_units",
            "torque_units",
            "device_id",
            "phantom_id",
            "actuation_protocol",
            "calibration_reference",
            "provenance",
        )
        for name in fields:
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
            object.__setattr__(self, name, value.strip())
        if self.source not in {"simulation", "bench"}:
            raise ValueError("source must be 'simulation' or 'bench'")
        if self.calibration_reference.lower() in {"none", "unknown", "uncalibrated"}:
            raise ValueError("calibration_reference must identify a calibration")

    def to_dict(self) -> dict[str, str]:
        return {
            "source": self.source,
            "force_units": self.force_units,
            "torque_units": self.torque_units,
            "device_id": self.device_id,
            "phantom_id": self.phantom_id,
            "actuation_protocol": self.actuation_protocol,
            "calibration_reference": self.calibration_reference,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class BenchTrace:
    """Time-sampled force/torque trace with explicit calibration identity."""

    time_s: np.ndarray
    force_n: np.ndarray
    torque_nm: np.ndarray
    metadata: BenchTraceMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, BenchTraceMetadata):
            raise ValueError("metadata must be a BenchTraceMetadata")
        arrays = {}
        for name, value in (
            ("time_s", self.time_s),
            ("force_n", self.force_n),
            ("torque_nm", self.torque_nm),
        ):
            try:
                arr = np.asarray(value, dtype=np.float64)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} must be a numeric one-dimensional array") from exc
            if arr.ndim != 1 or arr.size == 0:
                raise ValueError(f"{name} must be a non-empty one-dimensional array")
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} must contain only finite values")
            arrays[name] = arr.copy()
        n = arrays["time_s"].size
        if arrays["force_n"].size != n or arrays["torque_nm"].size != n:
            raise ValueError("time_s, force_n, and torque_nm must have equal lengths")
        if n > 1 and not np.all(np.diff(arrays["time_s"]) > 0.0):
            raise ValueError("time_s must be strictly increasing")
        for name, arr in arrays.items():
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)


@dataclass(frozen=True)
class BenchValidationCriteria:
    """Predeclared acceptance limits for one force/torque comparison."""

    force_rmse_max_n: float
    torque_rmse_max_nm: float
    force_peak_error_max_n: float | None = None
    torque_peak_error_max_nm: float | None = None
    max_whip_event_delta: int | None = None
    whip_deadband_nm: float = 0.0

    def __post_init__(self) -> None:
        required = {"force_rmse_max_n", "torque_rmse_max_nm", "whip_deadband_nm"}
        for name, value in (
            ("force_rmse_max_n", self.force_rmse_max_n),
            ("torque_rmse_max_nm", self.torque_rmse_max_nm),
            ("force_peak_error_max_n", self.force_peak_error_max_n),
            ("torque_peak_error_max_nm", self.torque_peak_error_max_nm),
            ("whip_deadband_nm", self.whip_deadband_nm),
        ):
            if value is None:
                if name in required:
                    raise ValueError(f"{name} must be a finite non-negative number")
                continue
            try:
                normalized = float(value)
            except (TypeError, ValueError, OverflowError):
                normalized = float("nan")
            if (
                isinstance(value, (bool, np.bool_))
                or not np.isfinite(normalized)
                or normalized < 0.0
            ):
                raise ValueError(f"{name} must be None or a finite non-negative number")
            object.__setattr__(self, name, normalized)
        if self.max_whip_event_delta is not None and (
            isinstance(self.max_whip_event_delta, (bool, np.bool_))
            or not isinstance(self.max_whip_event_delta, (int, np.integer))
            or self.max_whip_event_delta < 0
        ):
            raise ValueError("max_whip_event_delta must be None or a non-negative integer")
        if self.max_whip_event_delta is not None:
            object.__setattr__(self, "max_whip_event_delta", int(self.max_whip_event_delta))


@dataclass(frozen=True)
class BenchValidationReport:
    """Portable result of :func:`validate_bench_trace`."""

    passed: bool
    metrics: dict[str, float | int]
    failures: tuple[str, ...] = ()
    criteria: dict[str, Any] = field(default_factory=dict)
    simulated_metadata: dict[str, str] = field(default_factory=dict)
    measured_metadata: dict[str, str] = field(default_factory=dict)
    protocol: str = BENCH_VALIDATION_PROTOCOL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "passed": self.passed,
            "metrics": dict(self.metrics),
            "criteria": dict(self.criteria),
            "simulated_metadata": dict(self.simulated_metadata),
            "measured_metadata": dict(self.measured_metadata),
            "failures": list(self.failures),
        }


def _whip_events(torque: np.ndarray, deadband_nm: float) -> int:
    """Count reversals in torque slope after a predeclared deadband."""
    if torque.size < 3:
        return 0
    slope = np.diff(torque)
    slope[np.abs(slope) <= deadband_nm] = 0.0
    slope = slope[slope != 0.0]
    if slope.size < 2:
        return 0
    return int(np.count_nonzero(np.signbit(slope[1:]) != np.signbit(slope[:-1])))


def _criteria_dict(criteria: BenchValidationCriteria) -> dict[str, Any]:
    return {
        "force_rmse_max_n": criteria.force_rmse_max_n,
        "torque_rmse_max_nm": criteria.torque_rmse_max_nm,
        "force_peak_error_max_n": criteria.force_peak_error_max_n,
        "torque_peak_error_max_nm": criteria.torque_peak_error_max_nm,
        "max_whip_event_delta": criteria.max_whip_event_delta,
        "whip_deadband_nm": criteria.whip_deadband_nm,
    }


def _metadata_mismatches(simulated: BenchTraceMetadata,
                         measured: BenchTraceMetadata) -> list[str]:
    failures = []
    if simulated.source != "simulation":
        failures.append("simulated trace source must be 'simulation'")
    if measured.source != "bench":
        failures.append("measured trace source must be 'bench'")
    if simulated.force_units != "N" or measured.force_units != "N":
        failures.append("force_units must be 'N' after calibration")
    if simulated.torque_units != "N*m" or measured.torque_units != "N*m":
        failures.append("torque_units must be 'N*m' after calibration")
    for name in (
        "force_units",
        "torque_units",
        "device_id",
        "phantom_id",
        "actuation_protocol",
        "calibration_reference",
        "provenance",
    ):
        left, right = getattr(simulated, name), getattr(measured, name)
        if left != right:
            failures.append(f"{name} mismatch: simulated={left!r}, measured={right!r}")
    return failures

def validate_bench_trace(
    simulated: BenchTrace,
    measured: BenchTrace,
    criteria: BenchValidationCriteria,
) -> BenchValidationReport:
    """Compare calibrated simulation and benchtop traces on measured timestamps.

    The measured time range must be covered by the simulation trace; extrapolation is
    rejected instead of silently manufacturing agreement. Force and torque errors use
    linear interpolation, while the whip proxy compares reversals in torque slope on
    the same measured grid. Trace identity, units, calibration, and actuation metadata
    must match before any numerical comparison is accepted.
    """
    if not isinstance(simulated, BenchTrace) or not isinstance(measured, BenchTrace):
        raise TypeError("simulated and measured must be BenchTrace instances")
    if not isinstance(criteria, BenchValidationCriteria):
        raise TypeError("criteria must be a BenchValidationCriteria")

    simulated_metadata = simulated.metadata.to_dict()
    measured_metadata = measured.metadata.to_dict()
    metrics: dict[str, float | int] = {
        "simulated_samples": int(simulated.time_s.size),
        "measured_samples": int(measured.time_s.size),
        "time_start_s": float(measured.time_s[0]),
        "time_end_s": float(measured.time_s[-1]),
    }
    failures = _metadata_mismatches(simulated.metadata, measured.metadata)
    metrics["metadata_match"] = int(not failures)
    if failures:
        return BenchValidationReport(
            False,
            metrics,
            tuple(failures),
            criteria=_criteria_dict(criteria),
            simulated_metadata=simulated_metadata,
            measured_metadata=measured_metadata,
        )

    metrics["coverage_valid"] = 1
    if measured.time_s[0] < simulated.time_s[0] or measured.time_s[-1] > simulated.time_s[-1]:
        failures.append("measured time range is outside simulated trace coverage")
        metrics["coverage_valid"] = 0
        return BenchValidationReport(
            False,
            metrics,
            tuple(failures),
            criteria=_criteria_dict(criteria),
            simulated_metadata=simulated_metadata,
            measured_metadata=measured_metadata,
        )

    sim_force = np.interp(measured.time_s, simulated.time_s, simulated.force_n)
    sim_torque = np.interp(measured.time_s, simulated.time_s, simulated.torque_nm)
    force_error = sim_force - measured.force_n
    torque_error = sim_torque - measured.torque_nm
    simulated_whip_events = _whip_events(sim_torque, criteria.whip_deadband_nm)
    measured_whip_events = _whip_events(measured.torque_nm, criteria.whip_deadband_nm)
    metrics.update({
        "force_mae_n": float(np.mean(np.abs(force_error))),
        "force_rmse_n": float(np.sqrt(np.mean(force_error ** 2))),
        "force_peak_error_n": float(abs(np.max(np.abs(sim_force))
                                         - np.max(np.abs(measured.force_n)))),
        "torque_mae_nm": float(np.mean(np.abs(torque_error))),
        "torque_rmse_nm": float(np.sqrt(np.mean(torque_error ** 2))),
        "torque_peak_error_nm": float(abs(np.max(np.abs(sim_torque))
                                          - np.max(np.abs(measured.torque_nm)))),
        "simulated_whip_events": simulated_whip_events,
        "measured_whip_events": measured_whip_events,
        "whip_event_delta": abs(simulated_whip_events - measured_whip_events),
    })
    if metrics["force_rmse_n"] > criteria.force_rmse_max_n:
        failures.append("force RMSE exceeds criterion")
    if metrics["torque_rmse_nm"] > criteria.torque_rmse_max_nm:
        failures.append("torque RMSE exceeds criterion")
    if (criteria.force_peak_error_max_n is not None
            and metrics["force_peak_error_n"] > criteria.force_peak_error_max_n):
        failures.append("force peak error exceeds criterion")
    if (criteria.torque_peak_error_max_nm is not None
            and metrics["torque_peak_error_nm"] > criteria.torque_peak_error_max_nm):
        failures.append("torque peak error exceeds criterion")
    if (criteria.max_whip_event_delta is not None
            and metrics["whip_event_delta"] > criteria.max_whip_event_delta):
        failures.append("torque whip-event delta exceeds criterion")
    return BenchValidationReport(
        not failures,
        metrics,
        tuple(failures),
        criteria=_criteria_dict(criteria),
        simulated_metadata=simulated_metadata,
        measured_metadata=measured_metadata,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _validate_observation(observation: Any) -> Any:
    try:
        values = np.asarray(observation)
        finite = np.isfinite(values.astype(np.float64, copy=False)).all()
    except (TypeError, ValueError, OverflowError) as exc:
        raise DeploymentProtocolError("observation must be a finite numeric array") from exc
    if values.size == 0 or not finite:
        raise DeploymentProtocolError("observation must be a non-empty finite numeric array")
    return observation


def _policy_action(policy: Any, observation: Any, deterministic: bool) -> Any:
    predict = getattr(policy, "predict", None)
    if callable(predict):
        result = predict(observation, deterministic=deterministic)
        return result[0] if isinstance(result, tuple) else result
    if not callable(policy):
        raise DeploymentProtocolError("policy must be callable or expose predict()")
    return policy(observation)


def _validate_action(action: Any, config: DeploymentConfig) -> np.ndarray:
    try:
        values = np.asarray(action, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DeploymentProtocolError("policy action must be a numeric array") from exc
    if values.shape != config.action_shape:
        raise DeploymentProtocolError(
            f"policy action shape must be {config.action_shape}, got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise DeploymentProtocolError("policy action must contain only finite values")
    if np.any(values < config.action_low) or np.any(values > config.action_high):
        raise DeploymentProtocolError(
            f"policy action must lie in [{config.action_low}, {config.action_high}]"
        )
    return values.astype(np.float32)


def _call_safe_stop(interface: Any) -> Exception | None:
    stop = getattr(interface, "safe_stop", None)
    if not callable(stop):
        return DeploymentProtocolError("deployment interface has no safe_stop()")
    try:
        stop()
    except Exception as exc:  # pragma: no cover - backend-specific failure
        return exc
    return None

def _read_telemetry(interface: Any) -> dict[str, Any]:
    raw = interface.telemetry()
    if not isinstance(raw, Mapping):
        raise DeploymentProtocolError("telemetry() must return a mapping")
    return dict(raw)


def run_deployment(
    interface: DeploymentInterface,
    policy: Any,
    *,
    config: DeploymentConfig | None = None,
    on_step: Callable[[DeploymentStep], None] | None = None,
) -> DeploymentResult:
    """Run a bounded, fail-closed policy loop through a deployment interface.

    The initial telemetry sample is checked before the first action. The loop never
    clips an action. A malformed observation/action/telemetry sample, policy error,
    callback error, or safety violation invokes ``safe_stop()`` before optional
    ``close()``. Cleanup errors are recorded separately.
    """
    config = config or DeploymentConfig()
    result = DeploymentResult(status="error", steps=0)
    try:
        observation = _validate_observation(interface.observe())
        result.preflight_telemetry = _read_telemetry(interface)
        preflight = check_safety_envelope(result.preflight_telemetry, config.envelope)
        if not preflight.safe:
            result.status = "unsafe"
            result.violations.extend(preflight.violations)
        else:
            for index in range(config.max_steps):
                observation = _validate_observation(observation)
                action = _validate_action(
                    _policy_action(policy, observation, config.deterministic), config
                )
                interface.apply_action(action.copy())
                telemetry = _read_telemetry(interface)
                safety = check_safety_envelope(telemetry, config.envelope)
                result.actions.append(action.tolist())
                result.telemetry.append(telemetry)
                result.steps += 1
                if not safety.safe:
                    result.violations.extend(safety.violations)
                if on_step is not None:
                    on_step(DeploymentStep(index, observation, action, telemetry, safety))
                if not safety.safe:
                    result.status = "unsafe"
                    break
                observation = interface.observe()
            else:
                result.status = "max_steps"
    except Exception as exc:
        result.status = "error"
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        stop_error = _call_safe_stop(interface)
        if stop_error is not None:
            result.cleanup_error = (
                f"safe stop failed: {type(stop_error).__name__}: {stop_error}"
            )
        close = getattr(interface, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:  # pragma: no cover - backend-specific failure
                message = f"close failed: {type(exc).__name__}: {exc}"
                result.cleanup_error = (
                    f"{result.cleanup_error}; {message}"
                    if result.cleanup_error else message
                )
    return result


__all__ = [
    "BENCH_VALIDATION_PROTOCOL_VERSION",
    "DEPLOYMENT_PROTOCOL_VERSION",
    "BenchTrace",
    "BenchTraceMetadata",
    "BenchValidationCriteria",
    "BenchValidationReport",
    "DeploymentInterface",
    "DeploymentProtocolError",
    "DeploymentResult",
    "DeploymentStep",
    "SafetyCheck",
    "SafetyEnvelope",
    "check_safety_envelope",
    "run_deployment",
    "validate_bench_trace",
]
