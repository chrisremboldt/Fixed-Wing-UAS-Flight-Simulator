"""
Trim-assisted policy control for sim-to-sim transfer.

The trained policy expects GPU-rendered pixels and Warp physics. Until render
parity is complete, blending trim-hold cruise with threat-gated policy
commands keeps the aircraft flyable while still exercising avoidance logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..aircraft import AircraftConfig
from ..dynamics import FlightDynamics
from ..intruders import IntruderManager
from ..px4_bridge import ControlInterventionPolicy
from ..state import AircraftState, ControlInputs
from .actions import training_action_to_controls
from .flight_debug import FlightDebugLogger, configure_flight_debug
from .model_policy import ModelPolicy


@dataclass
class TrimAssistConfig:
    """Threat-gated blending between trim cruise and policy commands."""

    engage_distance_m: float = 1000.0
    full_authority_distance_m: float = 350.0
    forward_cone_deg: float = 45.0
    max_authority: float = 0.6
    hold_trim_throttle: bool = True
    min_throttle: float = 0.15
    # When engaged authority is low, recover with PID holds instead of static trim.
    use_recovery_autopilot: bool = True
    authority_smoothing_s: float = 0.35
    authority_warmup_s: float = 3.0  # hold policy off briefly after spawn/reset


def compute_threat_authority(
    dynamics: FlightDynamics,
    intruder_manager: Optional[IntruderManager],
    config: TrimAssistConfig,
) -> float:
    """Return blend factor in [0, max_authority] based on forward-sector intruders."""
    if intruder_manager is None or not intruder_manager.intruders:
        return 0.0

    own = dynamics.state
    R_body = own.quaternion.to_dcm().T
    cone = math.radians(config.forward_cone_deg)
    closest = float('inf')

    for intruder in intruder_manager.intruders:
        rel_body = R_body @ (intruder.dynamics.state.position - own.position)
        if rel_body[0] <= 50.0:
            continue
        az = math.atan2(rel_body[1], rel_body[0])
        if abs(az) > cone / 2:
            continue
        closest = min(closest, float(np.linalg.norm(rel_body)))

    if closest == float('inf'):
        return 0.0
    if closest <= config.full_authority_distance_m:
        return config.max_authority
    if closest >= config.engage_distance_m:
        return 0.0

    span = config.engage_distance_m - config.full_authority_distance_m
    t = (config.engage_distance_m - closest) / span
    return config.max_authority * t


def _normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


@dataclass
class LevelRecoveryController:
    """
    Return toward trim + captured cruise targets after policy intervention.

    Uses trim deflections as feedforward with capped attitude feedback and
    per-step slew limits so disengagement does not slam the surfaces.
    """

    trim_controls: ControlInputs
    target_heading: float = 0.0
    target_altitude: float = 1000.0
    pitch_kp: float = 0.55
    pitch_kd: float = 0.22
    roll_kp: float = 0.75
    roll_kd: float = 0.22
    yaw_kp: float = 0.35
    altitude_kp: float = 0.004
    max_surface_correction: float = 0.10
    max_slew_per_step: float = 0.025
    _last_controls: Optional[ControlInputs] = None

    def reset(self) -> None:
        self._last_controls = None

    def update(self, state: AircraftState) -> ControlInputs:
        heading_error = _normalize_angle(self.target_heading - state.psi)
        altitude_error = self.target_altitude - state.altitude

        pitch_corr = np.clip(
            self.pitch_kp * state.theta + self.pitch_kd * state.q,
            -self.max_surface_correction,
            self.max_surface_correction,
        )
        roll_corr = np.clip(
            self.roll_kp * state.phi + self.roll_kd * state.p,
            -self.max_surface_correction,
            self.max_surface_correction,
        )
        yaw_corr = np.clip(
            self.yaw_kp * heading_error,
            -self.max_surface_correction,
            self.max_surface_correction,
        )
        alt_corr = np.clip(
            self.altitude_kp * altitude_error,
            -0.05,
            0.05,
        )

        target = ControlInputs(
            elevator=float(np.clip(
                self.trim_controls.elevator + pitch_corr + alt_corr, -0.35, 0.35,
            )),
            aileron=float(np.clip(
                self.trim_controls.aileron + roll_corr + yaw_corr, -0.35, 0.35,
            )),
            rudder=float(np.clip(
                self.trim_controls.rudder + 0.10 * roll_corr - 0.15 * state.r,
                -0.35,
                0.35,
            )),
            throttle=self.trim_controls.throttle,
        )

        if self._last_controls is None:
            self._last_controls = target
            return target

        slew = self.max_slew_per_step
        blended = ControlInputs(
            elevator=self._last_controls.elevator + np.clip(
                target.elevator - self._last_controls.elevator, -slew, slew,
            ),
            aileron=self._last_controls.aileron + np.clip(
                target.aileron - self._last_controls.aileron, -slew, slew,
            ),
            rudder=self._last_controls.rudder + np.clip(
                target.rudder - self._last_controls.rudder, -slew, slew,
            ),
            throttle=target.throttle,
        )
        self._last_controls = blended
        return blended


def blend_trim_and_policy(
    trim_controls: ControlInputs,
    policy_controls: ControlInputs,
    authority: float,
    *,
    hold_trim_throttle: bool = True,
    min_throttle: float = 0.15,
) -> ControlInputs:
    """Linear blend; throttle can stay at trim for cruise stability."""
    auth = float(np.clip(authority, 0.0, 1.0))
    throttle = trim_controls.throttle
    if not hold_trim_throttle:
        throttle = float(np.clip(
            (1.0 - auth) * trim_controls.throttle + auth * policy_controls.throttle,
            min_throttle,
            1.0,
        ))

    return ControlInputs(
        elevator=(1.0 - auth) * trim_controls.elevator + auth * policy_controls.elevator,
        aileron=(1.0 - auth) * trim_controls.aileron + auth * policy_controls.aileron,
        rudder=(1.0 - auth) * trim_controls.rudder + auth * policy_controls.rudder,
        throttle=throttle,
    )


class TrimAssistedModelPolicy:
    """
    Wraps ModelPolicy with trim cruise baseline and threat-gated authority.

    Implements ControlInterventionPolicy for PX4 bridge use.
    """

    def __init__(
        self,
        model_policy: ModelPolicy,
        trim_controls: ControlInputs,
        config: Optional[TrimAssistConfig] = None,
        *,
        recovery_state: Optional[AircraftState] = None,
    ):
        self.model_policy = model_policy
        self.trim_controls = trim_controls
        self.config = config or TrimAssistConfig()
        self.recovery = LevelRecoveryController(
            trim_controls=trim_controls,
            target_heading=recovery_state.psi if recovery_state else 0.0,
            target_altitude=recovery_state.altitude if recovery_state else 1000.0,
        )
        self._smoothed_authority = 0.0
        self._last_authority_time: Optional[float] = None
        self._recovery_active = False
        self._flight_log = FlightDebugLogger()
        configure_flight_debug()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str,
        trim_controls: ControlInputs,
        device: str = 'cpu',
        config: Optional[TrimAssistConfig] = None,
        renderer_backend: str = 'training',
        throttle_mode: str = 'symmetric',
        render_device: str = 'cpu',
        surface_scale: float = 1.0,
        recovery_state: Optional[AircraftState] = None,
    ) -> 'TrimAssistedModelPolicy':
        return cls(
            ModelPolicy(
                checkpoint,
                device=device,
                deterministic=True,
                renderer_backend=renderer_backend,
                render_device=render_device,
                throttle_mode=throttle_mode,
                surface_scale=surface_scale,
            ),
            trim_controls,
            config=config,
            recovery_state=recovery_state,
        )

    @property
    def smoothed_authority(self) -> float:
        return self._smoothed_authority

    def _smooth_authority(self, raw_authority: float, sim_time: float) -> float:
        if self._last_authority_time is None:
            self._smoothed_authority = raw_authority
        else:
            dt = max(sim_time - self._last_authority_time, 0.0)
            tau = max(self.config.authority_smoothing_s, 1e-3)
            alpha = 1.0 - math.exp(-dt / tau)
            self._smoothed_authority += alpha * (raw_authority - self._smoothed_authority)
        self._last_authority_time = sim_time
        return self._smoothed_authority

    def _baseline_controls(
        self,
        dynamics: FlightDynamics,
        *,
        raw_authority: float,
        smoothed_authority: float,
    ) -> ControlInputs:
        if not self.config.use_recovery_autopilot:
            return self.trim_controls

        engage_threshold = 0.05
        if smoothed_authority > engage_threshold or raw_authority > engage_threshold:
            self._recovery_active = True

        if not self._recovery_active:
            return self.trim_controls

        return self.recovery.update(dynamics.state)

    def compute_controls(
        self,
        dynamics: FlightDynamics,
        intruder_manager: Optional[IntruderManager],
    ) -> ControlInputs:
        raw_authority = compute_threat_authority(dynamics, intruder_manager, self.config)
        if dynamics.state.time < self.config.authority_warmup_s:
            raw_authority = 0.0
        authority = self._smooth_authority(raw_authority, dynamics.state.time)
        baseline = self._baseline_controls(
            dynamics,
            raw_authority=raw_authority,
            smoothed_authority=authority,
        )
        if authority <= 1e-6:
            self._flight_log.maybe_log(
                dynamics=dynamics,
                intruder_manager=intruder_manager,
                raw_authority=raw_authority,
                smoothed_authority=authority,
                baseline=baseline,
                policy_controls=None,
                output=baseline,
                action=None,
                forward_cone_deg=self.config.forward_cone_deg,
            )
            return baseline

        action = self.model_policy.predict_action(dynamics, intruder_manager)
        policy_controls = training_action_to_controls(
            action,
            dynamics.aircraft,
            throttle_mode=self.model_policy.throttle_mode,
            surface_scale=self.model_policy.surface_scale,
        )
        blended = blend_trim_and_policy(
            baseline,
            policy_controls,
            authority,
            hold_trim_throttle=self.config.hold_trim_throttle,
            min_throttle=self.config.min_throttle,
        )
        self._flight_log.maybe_log(
            dynamics=dynamics,
            intruder_manager=intruder_manager,
            raw_authority=raw_authority,
            smoothed_authority=authority,
            baseline=baseline,
            policy_controls=policy_controls,
            output=blended,
            action=action,
            forward_cone_deg=self.config.forward_cone_deg,
        )
        return blended

    def intervene(
        self,
        controls: ControlInputs,
        dynamics: FlightDynamics,
        intruder_manager: Optional[IntruderManager],
    ) -> ControlInputs:
        return self.compute_controls(dynamics, intruder_manager)
