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
from ..state import ControlInputs
from .actions import training_action_to_controls
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
    ):
        self.model_policy = model_policy
        self.trim_controls = trim_controls
        self.config = config or TrimAssistConfig()

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
        )

    def compute_controls(
        self,
        dynamics: FlightDynamics,
        intruder_manager: Optional[IntruderManager],
    ) -> ControlInputs:
        authority = compute_threat_authority(dynamics, intruder_manager, self.config)
        if authority <= 0.0:
            return self.trim_controls

        action = self.model_policy.predict_action(dynamics, intruder_manager)
        policy_controls = training_action_to_controls(
            action,
            dynamics.aircraft,
            throttle_mode=self.model_policy.throttle_mode,
            surface_scale=self.model_policy.surface_scale,
        )
        return blend_trim_and_policy(
            self.trim_controls,
            policy_controls,
            authority,
            hold_trim_throttle=self.config.hold_trim_throttle,
            min_throttle=self.config.min_throttle,
        )

    def intervene(
        self,
        controls: ControlInputs,
        dynamics: FlightDynamics,
        intruder_manager: Optional[IntruderManager],
    ) -> ControlInputs:
        return self.compute_controls(dynamics, intruder_manager)
