"""Interactive visualization policy controller (RFC 001 / issue #11)."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ..aircraft import AircraftConfig
from ..dynamics import FlightDynamics
from ..environment import Environment
from ..intruders import IntruderManager
from ..state import ControlInputs
from ..trim import TrimCondition, compute_trim
from .evaluation import min_intruder_distance
from .model_policy import ModelPolicy
from .training_config import TrainingFidelityConfig
from .training_init import apply_training_initial_state
from .trim_assist import TrimAssistConfig, TrimAssistedModelPolicy, compute_threat_authority


@dataclass
class InteractivePolicyConfig:
    policy_path: str
    device: str = 'cpu'
    training_fidelity: bool = True
    full_policy: bool = False
    renderer: str = 'training'
    render_device: str = 'cpu'
    engage_distance: float = 1000.0
    max_authority: float = 0.6
    surface_scale: float = 1.5
    cruise_throttle_floor: bool = True
    obs_thumbnail_size: int = 64
    obs_thumbnail_interval: int = 2


class InteractivePolicyController:
    """
    Drives the WebSocket simulation loop with trim-assisted or full policy control.

    Returns control inputs plus telemetry for the Three.js overlay.
    """

    def __init__(
        self,
        aircraft: AircraftConfig,
        environment: Environment,
        config: InteractivePolicyConfig,
        *,
        seed: int = 42,
    ):
        self.config = config
        self._step_count = 0
        self._last_thumbnail_b64: str | None = None

        tf = TrainingFidelityConfig.defaults()
        sim_config = tf.simulation_config() if config.training_fidelity else None

        from ..dynamics import SimulationConfig

        self.dynamics = FlightDynamics(
            aircraft,
            environment,
            sim_config or SimulationConfig(dt=0.01),
        )

        cruise_speed = tf.cruise_speed_mps if config.training_fidelity else 25.0
        cruise_alt = 1000.0 if config.training_fidelity else 100.0

        if config.training_fidelity:
            apply_training_initial_state(self.dynamics, seed)

        trim = compute_trim(
            TrimCondition(airspeed=cruise_speed, altitude=cruise_alt),
            aircraft,
            environment,
        )
        if not config.training_fidelity and trim.success:
            self.dynamics.reset(trim.state)
            self.dynamics.controls = trim.controls
        elif not config.training_fidelity:
            self.dynamics.reset()

        throttle_floor = trim.controls.throttle if (config.cruise_throttle_floor and trim.success) else None

        if config.full_policy:
            self._mode = 'full_policy'
            self._policy = ModelPolicy(
                config.policy_path,
                device=config.device,
                renderer_backend=config.renderer,
                render_device=config.render_device,
                throttle_mode=tf.throttle_mode if config.training_fidelity else 'symmetric',
                surface_scale=config.surface_scale,
                cruise_throttle_floor=throttle_floor,
            )
            self._trim_assisted: TrimAssistedModelPolicy | None = None
            self._get_controls = self._controls_full_policy
        else:
            if not trim.success:
                raise RuntimeError('Trim failed; cannot run trim-assisted policy mode')
            self._mode = 'trim_assisted'
            self._trim_assisted = TrimAssistedModelPolicy.from_checkpoint(
                config.policy_path,
                trim.controls,
                device=config.device,
                config=TrimAssistConfig(
                    engage_distance_m=config.engage_distance,
                    max_authority=config.max_authority,
                ),
                renderer_backend=config.renderer,
                throttle_mode=tf.throttle_mode if config.training_fidelity else 'symmetric',
                render_device=config.render_device,
                surface_scale=config.surface_scale,
                recovery_state=self.dynamics.state,
            )
            self._policy = self._trim_assisted.model_policy
            self._get_controls = self._controls_trim_assisted

    @property
    def mode(self) -> str:
        return self._mode

    def _controls_full_policy(
        self,
        dynamics: FlightDynamics,
        intruder_manager: Optional[IntruderManager],
    ) -> ControlInputs:
        action = self._policy.predict_action(dynamics, intruder_manager)
        return self._policy.action_to_controls(action, dynamics.aircraft)

    def _controls_trim_assisted(
        self,
        dynamics: FlightDynamics,
        intruder_manager: Optional[IntruderManager],
    ) -> ControlInputs:
        assert self._trim_assisted is not None
        return self._trim_assisted.compute_controls(dynamics, intruder_manager)

    def _encode_thumbnail(self, obs: np.ndarray) -> str:
        size = self.config.obs_thumbnail_size
        h, w = obs.shape[:2]
        step_y = max(1, h // size)
        step_x = max(1, w // size)
        thumb = obs[::step_y, ::step_x, :][:size, :size, :]
        return base64.b64encode(thumb.tobytes()).decode('ascii')

    def step(
        self,
        dynamics: FlightDynamics,
        intruder_manager: Optional[IntruderManager],
    ) -> tuple[ControlInputs, dict]:
        """Compute controls and policy telemetry for one physics step."""
        action = self._policy.predict_action(dynamics, intruder_manager)
        controls = self._get_controls(dynamics, intruder_manager)

        authority = 1.0
        if self._trim_assisted is not None:
            authority = self._trim_assisted.smoothed_authority

        self._step_count += 1
        thumbnail_b64 = self._last_thumbnail_b64
        if self._step_count % max(1, self.config.obs_thumbnail_interval) == 0:
            obs = self._policy.observation_builder.build(dynamics, intruder_manager)
            thumbnail_b64 = self._encode_thumbnail(obs)
            self._last_thumbnail_b64 = thumbnail_b64

        min_sep = min_intruder_distance(dynamics, intruder_manager) if intruder_manager else float('inf')

        telemetry = {
            'active': True,
            'mode': self._mode,
            'min_separation_m': min_sep if np.isfinite(min_sep) else None,
            'authority': float(authority),
            'action': [float(x) for x in action[:4]],
            'controls': {
                'elevator': controls.elevator,
                'aileron': controls.aileron,
                'rudder': controls.rudder,
                'throttle': controls.throttle,
            },
            'obs_thumbnail_b64': thumbnail_b64,
            'obs_thumbnail_size': self.config.obs_thumbnail_size,
        }
        return controls, telemetry


def build_interactive_policy_setup(
    aircraft: AircraftConfig,
    config: InteractivePolicyConfig,
    *,
    seed: int = 42,
    enable_intruders: bool = True,
) -> tuple[FlightDynamics, Optional[IntruderManager], InteractivePolicyController, Callable]:
    """
    Build dynamics, intruders, and policy controller for interactive visualization.

    Returns:
        dynamics, intruder_manager, controller, controls_provider callback for SimulationServer
    """
    from ..intruders import IntruderConfig, IntruderManager

    environment = Environment()
    controller = InteractivePolicyController(aircraft, environment, config, seed=seed)
    dynamics = controller.dynamics

    intruder_manager = None
    if enable_intruders:
        tf = TrainingFidelityConfig.defaults()
        if config.training_fidelity:
            intruder_config = tf.intruder_config()
        else:
            intruder_config = IntruderConfig(
                spawn_rate=0.2,
                max_intruders=5,
                spawn_distance_min=300.0,
                spawn_distance_max=1200.0,
            )
        intruder_manager = IntruderManager(intruder_config, environment)
        intruder_manager.random.seed(seed)
        if config.training_fidelity:
            intruder_manager.spawn_initial_intruders(dynamics.state)

    def controls_provider(dyn: FlightDynamics, mgr: Optional[IntruderManager]):
        return controller.step(dyn, mgr)

    return dynamics, intruder_manager, controller, controls_provider
