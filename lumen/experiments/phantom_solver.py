"""Measured incremental base motion on the fixed left-route CPU Newton fork.

Positions are encoder readings relative to the prescribed reset pose. Map them
into model coordinates *before* testing cumulative travel. The model rail is
100 mm long; bound travel to the first 49 mm so neither base enters a daughter.
The shaped wire and route are fixed before seeing either measured trajectory.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from decimal import Decimal

from lumen.assets.procedural import bifurcation
from lumen.deployment import DeploymentConfig, SafetyEnvelope, run_deployment
from lumen.envs.tree_nav import TreeNavEnv
from lumen.experiments.phantom_capture import MAX_RUN_LENGTH, MAX_WIRE_MM, MAX_ROLL_ABS_DEG

INITIAL_WIRE = INITIAL_CATHETER = [0.0, 0.0]
SUPPORTED_INTERVAL_S = 0.020
MAX_STEP_INSERTION_MM = 2.0
MAX_STEP_TWIST_DEG = 30.0
SCALES = (MAX_STEP_INSERTION_MM, math.radians(MAX_STEP_TWIST_DEG))


def convert_samples(samples, signs):
    """Return incremental mm, rad per tool in wire-then-catheter order."""
    if not isinstance(samples, list) or not 1 <= len(samples) <= MAX_RUN_LENGTH:
        raise ValueError(f"samples must contain 1..{MAX_RUN_LENGTH} rows")
    if not isinstance(signs, dict) or any(
        not isinstance(signs.get(k), list)
        or len(signs[k]) != 2
        or any(type(v) is not int or v not in (-1, 1) for v in signs[k])
        for k in ("wire", "catheter")
    ):
        raise ValueError("axis signs must be integer +/-1 pairs")
    tools = ["wire"] if samples[0]["catheter"] is None else ["wire", "catheter"]
    step_bounds = (MAX_STEP_INSERTION_MM, MAX_STEP_TWIST_DEG)
    commands, prev = [], None
    for i, row in enumerate(samples):
        if not math.isfinite(row["t_s"]) or (
            prev is not None
            and not math.isclose(
                row["t_s"] - samples[i - 1]["t_s"], SUPPORTED_INTERVAL_S, rel_tol=0, abs_tol=1e-8
            )
        ):
            raise ValueError("unsupported cadence: expected 20 ms intervals")
        if ("catheter" in tools) != (row["catheter"] is not None):
            raise ValueError("incomplete catheter measurements")
        state = []
        for tool in tools:
            v = row[tool]
            if (
                not isinstance(v, (list, tuple))
                or len(v) != 2
                or not all(type(n) in (int, float) and math.isfinite(n) for n in v)
            ):
                raise ValueError(f"missing or invalid measured {tool} motion")
            pos, roll = (a * b for a, b in zip(v, signs[tool]))
            if not i and (pos or roll):
                raise ValueError("unknown initial pose: zero both axes at reset")
            if not 0 <= pos <= MAX_WIRE_MM or abs(roll) > MAX_ROLL_ABS_DEG:
                raise ValueError(f"{tool} outside cumulative replay bounds")
            state.extend([pos, roll])
        if prev is not None:
            delta = [b - a for a, b in zip(prev, state)]
            # Compare against the exact bound via decimal subtraction of the
            # stringified inputs: finite decimal magnitudes subtract exactly,
            # so float rounding at the bound (e.g. 4.03 - 2.03 ->
            # 2.0000000000000004) is normalized to the exact bound instead of
            # being rejected. Genuinely oversized steps are still refused, not
            # clipped.
            for k, (a, b) in enumerate(zip(prev, state)):
                exact = Decimal(str(b)) - Decimal(str(a))
                bound = Decimal(str(step_bounds[k % 2]))
                if abs(exact) > bound:
                    raise ValueError("per-step motion exceeds bound; refusing to clip")
                normalized = float(bound if exact > 0 else -bound)
                if abs(exact) == bound and delta[k] != normalized:
                    delta[k] = normalized
            commands.append(tuple(math.radians(v) if k % 2 else v for k, v in enumerate(delta)))
        prev = state
    return commands


class PhantomReplayInterface:
    """In-process DeploymentInterface; there is no hardware backend selection."""

    def __init__(self, coaxial=False):
        env = TreeNavEnv(bifurcation(), target_node="left_out", device="cpu")
        self.sim = env.sim
        wire = env._device_points()
        self.wire_half = np.linalg.norm(wire[-1] - wire[-2]) / 2
        if coaxial:
            from lumen.newton.sim import NewtonGuidewireSim

            route = np.asarray(env.route_frame.points)
            pts, normals = env._route_seed(np.arange(8, dtype=float) * 2.0)
            catheter = env._clamp_seed_inside_lumen(pts - 0.35 * env.R * normals, margin_mm=0.9)
            self.cath_half = np.linalg.norm(catheter[-1] - catheter[-2]) / 2
            self.sim = NewtonGuidewireSim(
                route,
                env.R,
                env._device_points(),
                radius=0.2,
                kappa=3e3,
                d_hat=0.3,
                vbd_iterations=8,
                device="cpu",
                tree=env.tree,
                route_centerline=route,
                catheter_points=catheter,
                catheter_radius=0.65,
            )
        self.coaxial = coaxial
        self.scales = np.array(SCALES * (2 if coaxial else 1))
        self.commands = deque()

    def observe(self):
        return self.commands[0] / self.scales if self.commands else np.zeros_like(self.scales)

    def apply_action(self, action):
        ins, twist, *catheter = self.scales * action
        kwargs = dict(insertion_cath=catheter[0], twist_cath=catheter[1]) if catheter else {}
        self.sim.step(dt=SUPPORTED_INTERVAL_S, substeps=4, insertion=ins, twist=twist, **kwargs)
        self.commands.popleft()

    def telemetry(self):
        poses = self.sim.state_0.body_q.numpy()

        def projected(body, half):
            pose = poses[body]
            x, y, z, w = pose[3:]
            # add_rod uses body_frame_origin='com', with +local Z distal.
            axis = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
            tip = pose[:3] + half * axis
            if not np.isfinite(tip).all():
                raise ValueError("nonfinite simulated endpoint")
            return tip[[0, 2]].tolist()

        return {
            "wire": projected(self.sim.bodies[-1], self.wire_half),
            "catheter": projected(self.sim.cath_bodies[-1], self.cath_half)
            if self.coaxial
            else None,
            "penetration_mm": float(np.max(self.sim.surface_penetration())),
        }

    def safe_stop(self):
        self.commands.clear()


def replay(samples, signs):
    commands = convert_samples(samples, signs)
    interface = PhantomReplayInterface(samples[0]["catheter"] is not None)
    telemetry = [interface.telemetry()]
    if commands:
        interface.commands.extend(commands)
        result = run_deployment(
            interface,
            lambda x: x,
            config=DeploymentConfig(
                max_steps=len(commands),
                action_shape=interface.scales.shape,
                envelope=SafetyEnvelope(max_penetration_mm=2.0),
            ),
        )
        if not result.safe or result.steps != len(commands):
            raise ValueError(f"incomplete simulation: {result.error or result.violations}")
        telemetry.extend(result.telemetry)
    return [
        dict(t_s=s["t_s"], event="sample" if i else "reset", wire=t["wire"], catheter=t["catheter"])
        for i, (s, t) in enumerate(zip(samples, telemetry))
    ]
