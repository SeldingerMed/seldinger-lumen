"""Native safety endpoints and explicit physical calibration.

The fast solver reports wall load in ``sim_units``.  Those values are deliberately
not presented as newtons: HGO parameters and geometry are a consistent simulator
system until a matched device/anatomy/phantom calibration is supplied.  This
module provides the explicit seam for that calibration without shipping a global
conversion constant.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


_NATIVE_WALL_LOAD_UNITS = "sim_units"
_PHYSICAL_WALL_LOAD_UNITS = "N"


def describe_wall_load_units() -> str:
    """Describe the native wall-load quantity and its calibration boundary."""
    return (
        "wall_load is a non-dimensional simulator load in sim units; converting it "
        "to newtons requires a matched-device, matched-anatomy physical calibration."
    )


@dataclass(frozen=True)
class WallLoadCalibration:
    """Linear calibration fitted from paired simulator and physical measurements.

    This is a calibration *record*, not a claim that any default mapping exists.
    The paired samples must come from the same device-tip geometry, anatomy/material
    configuration, and measurement protocol that the converted episodes use.
    """

    slope_n_per_sim: float
    intercept_n: float
    rmse_n: float
    r_squared: float
    sample_count: int
    sim_min: float
    sim_max: float
    provenance: str
    sim_units: str = _NATIVE_WALL_LOAD_UNITS
    physical_units: str = _PHYSICAL_WALL_LOAD_UNITS
    fit_method: str = "ordinary_least_squares"

    @classmethod
    def from_paired_measurements(
        cls,
        sim_loads,
        physical_forces,
        *,
        provenance: str,
    ) -> "WallLoadCalibration":
        """Fit a calibration from paired native loads and measured forces.

        At least two finite samples with distinct simulator loads are required.
        ``provenance`` must identify the physical acquisition/calibration record;
        callers should persist it alongside the resulting scorecard.
        """
        x = np.asarray(sim_loads, dtype=float).reshape(-1)
        y = np.asarray(physical_forces, dtype=float).reshape(-1)
        if x.size != y.size or x.size < 2:
            raise ValueError("paired simulator loads and physical forces need >= 2 samples")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("paired simulator loads and physical forces must be finite")
        if np.ptp(x) <= 0.0:
            raise ValueError("paired simulator loads need at least two distinct values")
        if not isinstance(provenance, str) or not provenance.strip():
            raise ValueError("provenance must identify the paired physical calibration data")

        slope, intercept = np.polyfit(x, y, 1)
        predicted = slope * x + intercept
        residual = predicted - y
        ss_res = float(np.dot(residual, residual))
        centred = y - float(y.mean())
        ss_tot = float(np.dot(centred, centred))
        r_squared = 1.0 if ss_tot == 0.0 and ss_res == 0.0 else (
            1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
        )
        return cls(
            slope_n_per_sim=float(slope),
            intercept_n=float(intercept),
            rmse_n=float(np.sqrt(ss_res / x.size)),
            r_squared=float(r_squared),
            sample_count=int(x.size),
            sim_min=float(x.min()),
            sim_max=float(x.max()),
            provenance=provenance.strip(),
        )

    def convert(self, sim_load, *, allow_extrapolation: bool = False):
        """Convert native load(s) to force(s) using this recorded fit.

        Extrapolation is rejected by default because it silently turns an empirical
        calibration into an unsupported injury-risk claim.  Set
        ``allow_extrapolation=True`` only for an explicitly documented analysis.
        """
        values = np.asarray(sim_load, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("sim_load must be finite")
        if not allow_extrapolation and (
            (values < self.sim_min).any() or (values > self.sim_max).any()
        ):
            raise ValueError("sim_load falls outside the paired calibration range")
        result = self.slope_n_per_sim * values + self.intercept_n
        return float(result) if result.ndim == 0 else result

    def to_dict(self) -> dict:
        """Return a JSON-serializable calibration record."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "WallLoadCalibration":
        """Restore and validate a serialized calibration record."""
        if not isinstance(payload, dict):
            raise ValueError("calibration payload must be a mapping")
        required = {
            "slope_n_per_sim", "intercept_n", "rmse_n", "r_squared", "sample_count",
            "sim_min", "sim_max", "provenance",
        }
        missing = sorted(required.difference(payload))
        if missing:
            raise ValueError(f"calibration payload missing fields: {', '.join(missing)}")
        cal = cls(**payload)
        if (cal.sim_units != _NATIVE_WALL_LOAD_UNITS
                or cal.physical_units != _PHYSICAL_WALL_LOAD_UNITS
                or cal.fit_method != "ordinary_least_squares"):
            raise ValueError("calibration record has unsupported units or fit method")
        if cal.sample_count < 2 or cal.sim_min >= cal.sim_max:
            raise ValueError("calibration record has an invalid sample range")
        if not np.isfinite([
            cal.slope_n_per_sim, cal.intercept_n, cal.rmse_n, cal.r_squared,
            cal.sim_min, cal.sim_max,
        ]).all():
            raise ValueError("calibration record must contain finite diagnostics")
        if not cal.provenance.strip():
            raise ValueError("calibration record needs provenance")
        return cal


def calibrate_wall_load(sim_load, calibration: WallLoadCalibration | None = None, *,
                        allow_extrapolation: bool = False):
    """Convert native load only when an explicit paired-data calibration is supplied."""
    if calibration is None:
        raise ValueError(
            "wall-load conversion requires WallLoadCalibration fitted from paired data"
        )
    if not isinstance(calibration, WallLoadCalibration):
        raise TypeError("calibration must be a WallLoadCalibration")
    return calibration.convert(sim_load, allow_extrapolation=allow_extrapolation)
