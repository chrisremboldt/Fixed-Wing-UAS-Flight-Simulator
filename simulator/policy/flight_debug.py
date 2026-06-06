"""Optional flight-control debug logging for policy / trim-assist runs."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..dynamics import FlightDynamics
from ..intruders import IntruderManager
from ..state import ControlInputs

logger = logging.getLogger('uas.flight')


def flight_debug_enabled() -> bool:
    return os.environ.get('UAS_FLIGHT_DEBUG', '').lower() in {'1', 'true', 'yes', 'on'}


def configure_flight_debug() -> None:
    if not flight_debug_enabled():
        return
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s [flight] %(message)s',
            datefmt='%H:%M:%S',
        ))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class ThreatSnapshot:
    closest_forward_m: float
    closest_any_m: float
    forward_count: int
    total_count: int


def measure_threats(
    dynamics: FlightDynamics,
    intruder_manager: Optional[IntruderManager],
    forward_cone_deg: float,
) -> ThreatSnapshot:
    if intruder_manager is None or not intruder_manager.intruders:
        return ThreatSnapshot(float('inf'), float('inf'), 0, 0)

    own = dynamics.state
    R_body = own.quaternion.to_dcm().T
    half_cone = math.radians(forward_cone_deg) / 2.0
    closest_forward = float('inf')
    closest_any = float('inf')
    forward_count = 0

    for intruder in intruder_manager.intruders:
        rel_body = R_body @ (intruder.dynamics.state.position - own.position)
        dist = float(np.linalg.norm(rel_body))
        closest_any = min(closest_any, dist)
        if rel_body[0] <= 50.0:
            continue
        az = math.atan2(rel_body[1], rel_body[0])
        if abs(az) <= half_cone:
            forward_count += 1
            closest_forward = min(closest_forward, dist)

    return ThreatSnapshot(
        closest_forward,
        closest_any,
        forward_count,
        len(intruder_manager.intruders),
    )


class FlightDebugLogger:
    """Periodic structured logs for trim-assist / policy control loops."""

    def __init__(self, interval_s: float = 0.25):
        self.interval_s = max(interval_s, 0.05)
        self._last_log_time = -float('inf')
        self._step = 0

    def maybe_log(
        self,
        *,
        dynamics: FlightDynamics,
        intruder_manager: Optional[IntruderManager],
        raw_authority: float,
        smoothed_authority: float,
        baseline: ControlInputs,
        policy_controls: Optional[ControlInputs],
        output: ControlInputs,
        action: Optional[np.ndarray],
        forward_cone_deg: float,
        tag: str = 'step',
    ) -> None:
        if not flight_debug_enabled():
            return

        self._step += 1
        t = dynamics.state.time
        if t - self._last_log_time < self.interval_s:
            return
        self._last_log_time = t

        state = dynamics.state
        fm = dynamics.forces_moments
        threat = measure_threats(dynamics, intruder_manager, forward_cone_deg)

        action_str = 'n/a'
        if action is not None:
            action_str = (
                f'[{action[0]:+.2f},{action[1]:+.2f},'
                f'{action[2]:+.2f},{action[3]:+.2f}]'
            )

        policy_str = 'n/a'
        if policy_controls is not None:
            policy_str = (
                f'e={policy_controls.elevator:+.3f} '
                f'a={policy_controls.aileron:+.3f} '
                f'r={policy_controls.rudder:+.3f}'
            )

        logger.info(
            '%s t=%.2fs #%d alt=%.0fm pitch=%.1f° roll=%.1f° spd=%.1fm/s '
            'q=%+.2f p=%+.2f | intruders=%d fwd=%d closest_fwd=%s closest_any=%s | '
            'auth raw=%.3f smooth=%.3f | baseline e=%+.3f a=%+.3f | policy %s | '
            'out e=%+.3f a=%+.3f thr=%.3f | action %s%s',
            tag,
            t,
            self._step,
            state.altitude,
            math.degrees(state.theta),
            math.degrees(state.phi),
            fm.airspeed,
            state.q,
            state.p,
            threat.total_count,
            threat.forward_count,
            f'{threat.closest_forward_m:.0f}m'
            if math.isfinite(threat.closest_forward_m) else '—',
            f'{threat.closest_any_m:.0f}m'
            if math.isfinite(threat.closest_any_m) else '—',
            raw_authority,
            smoothed_authority,
            baseline.elevator,
            baseline.aileron,
            policy_str,
            output.elevator,
            output.aileron,
            output.throttle,
            action_str,
            f' | CRASH {dynamics.crash_state.crash_message}'
            if dynamics.crash_state.crashed else '',
        )

    def log_startup(
        self,
        *,
        dynamics: FlightDynamics,
        trim_altitude: float,
        trim_airspeed: float,
        trim_controls: ControlInputs,
        recovery_altitude: float,
            recovery_heading_rad: float,
        intruder_manager: Optional[IntruderManager],
        forward_cone_deg: float,
    ) -> None:
        if not flight_debug_enabled():
            return
        configure_flight_debug()
        state = dynamics.state
        logger.info(
            'startup ownship alt=%.0fm pitch=%.1f° hdg=%.0f° spd=%.1f | '
            'trim target alt=%.0fm spd=%.0f trim_e=%+.3f thr=%.3f | '
            'recovery target alt=%.0fm hdg=%.0f°',
            state.altitude,
            math.degrees(state.theta),
            math.degrees(state.psi),
            float(np.linalg.norm(state.velocity_body)),
            trim_altitude,
            trim_airspeed,
            trim_controls.elevator,
            trim_controls.throttle,
            recovery_altitude,
            math.degrees(recovery_heading_rad),
        )
        if intruder_manager is not None:
            threat = measure_threats(dynamics, intruder_manager, forward_cone_deg)
            logger.info(
                'startup intruders=%d forward_in_cone=%d closest_fwd=%s',
                threat.total_count,
                threat.forward_count,
                f'{threat.closest_forward_m:.0f}m'
                if math.isfinite(threat.closest_forward_m) else '—',
            )
            for intruder in intruder_manager.intruders:
                s = intruder.dynamics.state
                rel = s.position - state.position
                rel_body = state.quaternion.to_dcm().T @ rel
                logger.info(
                    '  intruder %d alt=%.0fm dist=%.0fm body=[%.0f,%.0f,%.0f] az=%.0f°',
                    intruder.id,
                    s.altitude,
                    float(np.linalg.norm(rel)),
                    rel_body[0],
                    rel_body[1],
                    rel_body[2],
                    math.degrees(math.atan2(rel_body[1], rel_body[0])),
                )
