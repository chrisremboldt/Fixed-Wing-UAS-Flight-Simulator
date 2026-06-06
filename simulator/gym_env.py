"""
Gymnasium-compatible wrapper for RL fine-tuning on UAS sim physics (RFC 002 spike).

Single-env step loop with pixel observations from the training renderer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .aircraft import AircraftConfig
from .dynamics import FlightDynamics
from .environment import Environment
from .intruders import IntruderConfig, IntruderManager
from .policy import (
    EvalConfig,
    TrainingFidelityConfig,
    apply_training_initial_state,
    run_episode,
)
from .policy.actions import training_action_to_controls
from .policy.model_policy import ModelPolicy
from .trim import TrimCondition, compute_trim


@dataclass
class UASDAAEnvConfig:
    policy_path: str | None = None
    duration: float = 20.0
    dt: float = 0.02
    seed: int = 42
    device: str = 'cpu'
    training_fidelity: bool = True


class UASDAAEnv:
    """
    Minimal gym-style environment for policy eval and future RL fine-tuning.

    Observation: uint8 RGB (128, 128, 3)
    Action: [throttle, aileron, elevator, rudder] in [-1, 1]
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        aircraft: Optional[AircraftConfig] = None,
        config: Optional[UASDAAEnvConfig] = None,
    ):
        self.aircraft = aircraft or AircraftConfig()
        self.config = config or UASDAAEnvConfig(policy_path='final_model.pt')
        self._episode_seed = self.config.seed
        self._step_count = 0
        self._max_steps = int(self.config.duration / self.config.dt)

        self._environment = Environment()
        self._tf = TrainingFidelityConfig.defaults() if self.config.training_fidelity else None
        self._dynamics: FlightDynamics | None = None
        self._intruders: IntruderManager | None = None
        self._policy: ModelPolicy | None = None

        if self.config.policy_path:
            self._policy = ModelPolicy(
                self.config.policy_path,
                device=self.config.device,
                renderer_backend='training',
            )

    @property
    def observation_space_shape(self) -> tuple[int, int, int]:
        return (128, 128, 3)

    @property
    def action_space_shape(self) -> tuple[int, ...]:
        return (4,)

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._episode_seed = seed

        from .dynamics import SimulationConfig

        sim_config = (
            self._tf.simulation_config(self.config.dt)
            if self._tf
            else SimulationConfig(dt=self.config.dt)
        )
        self._dynamics = FlightDynamics(self.aircraft, self._environment, sim_config)

        cruise_speed = self._tf.cruise_speed_mps if self._tf else 25.0
        cruise_alt = 1000.0 if self._tf else 100.0

        if self._tf:
            apply_training_initial_state(self._dynamics, self._episode_seed)

        trim = compute_trim(
            TrimCondition(airspeed=cruise_speed, altitude=cruise_alt),
            self.aircraft,
            self._environment,
        )
        if not self._tf and trim.success:
            self._dynamics.reset(trim.state)

        intruder_config = IntruderConfig(
            spawn_rate=self._tf.spawn_rate if self._tf else 0.2,
            max_intruders=self._tf.max_intruders if self._tf else 5,
        )
        self._intruders = IntruderManager(intruder_config, self._environment)
        self._intruders.random.seed(self._episode_seed)
        self._step_count = 0

        obs = self._build_obs()
        return obs, {'seed': self._episode_seed}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._dynamics is not None

        controls = training_action_to_controls(
            action,
            self.aircraft,
            throttle_mode=self._tf.throttle_mode if self._tf else 'symmetric',
            surface_scale=self._tf.surface_scale if self._tf else 1.0,
        )
        self._dynamics.step(controls)

        if self._intruders is not None:
            if self._intruders.should_spawn_intruder(self.config.dt, self._dynamics.state):
                self._intruders.spawn_intruder(self._dynamics.state)
            self._intruders.update_intruders(self.config.dt, self._dynamics.state)

        self._step_count += 1
        terminated = self._dynamics.crash_state.crashed
        truncated = self._step_count >= self._max_steps
        reward = 0.0 if not (terminated or truncated) else -1.0 if terminated else 1.0

        obs = self._build_obs()
        info = {'sim_time': self._dynamics.state.time}
        return obs, reward, terminated, truncated, info

    def _build_obs(self) -> np.ndarray:
        assert self._dynamics is not None
        if self._policy is not None:
            return self._policy.observation_builder.build(self._dynamics, self._intruders)
        from .policy import PixelObservationBuilder
        return PixelObservationBuilder().build(self._dynamics, self._intruders)

    def run_policy_episode(self) -> Any:
        """Convenience: run full eval episode via existing harness."""
        eval_config = EvalConfig.from_training_defaults(
            self.config.policy_path or 'final_model.pt',
            duration=self.config.duration,
            seed=self._episode_seed,
        )
        return run_episode(self.aircraft, eval_config, self._episode_seed)
