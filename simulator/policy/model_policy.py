"""PyTorch policy loader and control hook (RFC 001)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np

from ..aircraft import AircraftConfig
from ..dynamics import FlightDynamics
from ..intruders import IntruderManager
from ..px4_bridge import ControlInterventionPolicy
from ..state import ControlInputs
from .actions import training_action_to_controls
from .architecture import PufferPolicy, load_policy_checkpoint
from .observation import ObservationSpec, PixelObservationBuilder

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore
    HAS_TORCH = False


class ModelPolicy:
    """
    Loads scratch_built_daa checkpoint and drives / intervenes on controls.

    Uses pixel observations and absolute control mapping from training.
    """

    def __init__(
        self,
        checkpoint: Path | str,
        spec: Optional[ObservationSpec] = None,
        device: str = 'cpu',
        *,
        deterministic: bool = True,
        control_mode: Literal['absolute', 'blend'] = 'absolute',
        blend_alpha: float = 1.0,
        renderer_backend: str = 'training',
        render_device: str = 'cpu',
        throttle_mode: str = 'symmetric',
    ):
        if not HAS_TORCH:
            raise ImportError('torch is required for ModelPolicy')

        self.spec = spec or ObservationSpec(
            renderer_backend=renderer_backend,
            render_device=render_device,
        )
        self.throttle_mode = throttle_mode
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.control_mode = control_mode
        self.blend_alpha = float(np.clip(blend_alpha, 0.0, 1.0))
        self.observation_builder = PixelObservationBuilder(self.spec)
        self.policy = load_policy_checkpoint(str(checkpoint), device=self.device)

    def predict_action(
        self,
        dynamics: FlightDynamics,
        intruder_manager: Optional[IntruderManager],
    ) -> np.ndarray:
        """Return [throttle, aileron, elevator, rudder] in [-1, 1]."""
        obs = self.observation_builder.build(dynamics, intruder_manager)
        tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.deterministic:
                action = self.policy.deterministic_action(tensor)
            else:
                action, _, _, _ = self.policy.get_action_and_value(tensor, deterministic=False)

        return action.squeeze(0).cpu().numpy()

    def action_to_controls(
        self,
        action: np.ndarray,
        aircraft: AircraftConfig,
    ) -> ControlInputs:
        return training_action_to_controls(
            action, aircraft, throttle_mode=self.throttle_mode,
        )

    def intervene(
        self,
        controls: ControlInputs,
        dynamics: FlightDynamics,
        intruder_manager: Optional[IntruderManager],
    ) -> ControlInputs:
        action = self.predict_action(dynamics, intruder_manager)
        policy_controls = self.action_to_controls(action, dynamics.aircraft)

        if self.control_mode == 'absolute' or self.blend_alpha >= 1.0:
            return policy_controls

        alpha = self.blend_alpha
        return ControlInputs(
            elevator=(1 - alpha) * controls.elevator + alpha * policy_controls.elevator,
            aileron=(1 - alpha) * controls.aileron + alpha * policy_controls.aileron,
            rudder=(1 - alpha) * controls.rudder + alpha * policy_controls.rudder,
            throttle=(1 - alpha) * controls.throttle + alpha * policy_controls.throttle,
        )

    def control_callback(
        self,
        dynamics: FlightDynamics,
        intruder_manager: Optional[IntruderManager],
    ):
        """Factory for dynamics.run(control_callback=...)."""

        def _callback(state, time: float) -> ControlInputs:
            return self.action_to_controls(
                self.predict_action(dynamics, intruder_manager),
                dynamics.aircraft,
            )

        return _callback
