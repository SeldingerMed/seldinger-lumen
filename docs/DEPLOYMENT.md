# Deployment and benchtop validation

Lumen exposes an open interface for a simulator, vascular phantom, or private device
bridge without shipping a robot driver or patient data. The interface is intentionally
small:

```python
class DeploymentInterface:
    def observe(self): ...
    def apply_action(self, action): ...
    def telemetry(self): ...
    def safe_stop(self): ...
```

`run_deployment()` validates every observation and action before the action reaches the
interface. Actions are finite arrays with the configured shape and normalized bounds
(default: two values in `[-1, 1]`). Invalid actions are rejected, not clipped. A
configured telemetry limit is fail-closed: a missing, non-numeric, or non-finite value
is a violation. Force and torque limits use absolute magnitude; penetration and OOD
limits use their non-negative scalar values.

`safe_stop()` is invoked on every exit, including a normal bounded shutdown, before an
optional `close()`. `DeploymentResult` retains preflight and per-step telemetry,
actions, violations, and separate cleanup errors. The result can be serialized with
`to_dict()` without serializing observations, which may contain large or private images.

## Benchtop force/torque protocol

`BenchTrace` is the common trace format for a calibrated simulation or phantom
measurement:

- `time_s`: strictly increasing sample times;
- `force_n`: scalar device-force samples in newtons (`N`);
- `torque_nm`: scalar device-torque samples in newton-metres (`N*m`);
- `metadata`: source (`simulation` or `bench`), units, matched device and phantom
  IDs, actuation protocol, calibration reference, and provenance.

`validate_bench_trace(simulated, measured, criteria)` requires both traces to be
calibrated to the canonical `N`/`N*m` units and rejects unit, calibration, device,
phantom, actuation, provenance, and source-role mismatches before comparing numbers.
It interpolates the simulation onto the measured timestamps. The measured window
must be inside the simulation window; there is no extrapolation. The report persists
both metadata and includes force/torque MAE/RMSE, peak-magnitude errors, and a
a deterministic whip proxy: reversals in the torque slope on the measured timestamp
grid, after a predeclared torque deadband. Criteria are predeclared by the calibration
owner and can gate force RMSE, torque RMSE, force/torque peak error, and whip-event-count
difference.

```python
from lumen.deployment import (
    BenchTrace,
    BenchTraceMetadata,
    BenchValidationCriteria,
    validate_bench_trace,
)

identity = dict(
    force_units="N",
    torque_units="N*m",
    device_id="guidewire-014-v1",
    phantom_id="phantom-a-v1",
    actuation_protocol="advance-rotate-v1",
    calibration_reference="calibration-2026-08-v1",
    provenance="procedural-phantom-a",
)
simulated_trace = BenchTrace(
    time_s=[0.0, 1.0],
    force_n=[0.0, 1.0],
    torque_nm=[0.0, 0.1],
    metadata=BenchTraceMetadata(source="simulation", **identity),
)
phantom_trace = BenchTrace(
    time_s=[0.0, 1.0],
    force_n=[0.0, 1.0],
    torque_nm=[0.0, 0.1],
    metadata=BenchTraceMetadata(source="bench", **identity),
)
report = validate_bench_trace(
    simulated_trace,
    phantom_trace,
    BenchValidationCriteria(
        force_rmse_max_n=0.15,
        torque_rmse_max_nm=0.02,
        force_peak_error_max_n=0.30,
        torque_peak_error_max_nm=0.04,
        max_whip_event_delta=1,
        whip_deadband_nm=0.002,
    ),
)
if not report.passed:
    raise RuntimeError(report.to_dict())
```

The report is a simulator/phantom agreement record, not a calibrated clinical force
threshold or regulatory clearance. Device geometry, phantom materials, sampling,
units, and acceptance criteria must be recorded by the private bench owner. Real
measurements and patient-derived assets stay outside this repository.
