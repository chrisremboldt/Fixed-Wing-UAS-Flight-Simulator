"""Batch policy evaluation with training-style metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..aircraft import AircraftConfig
from ..dynamics import FlightDynamics, SimulationConfig
from ..environment import Environment
from ..intruders import IntruderConfig, IntruderManager
from ..state import ControlInputs
from ..trim import TrimCondition, TrimResult, compute_trim
from .model_policy import ModelPolicy
from .training_config import TrainingFidelityConfig
from .training_init import apply_training_initial_state
from .trim_assist import TrimAssistConfig, TrimAssistedModelPolicy

NMAC_DISTANCE_M = 152.4
GROUND_ALTITUDE_M = 50.0


@dataclass
class EvalConfig:
    policy_path: str
    duration: float = 20.0
    dt: float = 0.02
    seed: int = 42
    device: str = 'cpu'
    spawn_rate: float = 0.2
    max_intruders: int = 5
    training_fidelity: bool = True
    full_policy: bool = False
    renderer: str = 'training'
    render_device: str = 'cpu'
    throttle_mode: str = 'clamp'
    stochastic: bool = False
    engage_distance: float = 1000.0
    max_authority: float = 0.6
    policy_throttle: bool = False
    surface_scale: float = 1.0
    cruise_throttle_floor: bool = True

    @classmethod
    def from_training_defaults(
        cls,
        policy_path: str,
        *,
        duration: float = 20.0,
        seed: int = 42,
        full_policy: bool = False,
        device: str = 'cpu',
    ) -> 'EvalConfig':
        tf = TrainingFidelityConfig.defaults()
        return cls(
            policy_path=policy_path,
            duration=duration,
            dt=tf.dt,
            seed=seed,
            device=device,
            spawn_rate=tf.spawn_rate,
            max_intruders=tf.max_intruders,
            training_fidelity=True,
            full_policy=full_policy,
            renderer=tf.renderer_backend,
            throttle_mode=tf.throttle_mode,
            surface_scale=tf.surface_scale,
            cruise_throttle_floor=tf.cruise_throttle_floor,
        )


@dataclass
class EpisodeResult:
    seed: int
    sim_time: float
    min_separation_m: float
    final_altitude_m: float
    final_airspeed_mps: float
    intruders_spawned: int
    nmac_proximity_steps: int
    crashed: bool
    ground_collision: bool
    nmac_violation: bool
    timeout: bool
    crash_message: str = ''

    @property
    def success(self) -> bool:
        return not self.crashed and not self.nmac_violation and not self.ground_collision


@dataclass
class BatchEvalResult:
    episodes: list[EpisodeResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return 100.0 * sum(ep.success for ep in self.episodes) / len(self.episodes)

    @property
    def nmac_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return 100.0 * sum(ep.nmac_violation for ep in self.episodes) / len(self.episodes)

    @property
    def ground_collision_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return 100.0 * sum(ep.ground_collision for ep in self.episodes) / len(self.episodes)

    @property
    def timeout_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return 100.0 * sum(ep.timeout for ep in self.episodes) / len(self.episodes)

    def mean_min_separation(self) -> float:
        vals = [ep.min_separation_m for ep in self.episodes if np.isfinite(ep.min_separation_m)]
        return float(np.mean(vals)) if vals else float('inf')

    def mean_sim_time(self) -> float:
        return float(np.mean([ep.sim_time for ep in self.episodes])) if self.episodes else 0.0


def min_intruder_distance(dynamics: FlightDynamics, intruder_manager: IntruderManager) -> float:
    if not intruder_manager.intruders:
        return float('inf')
    own = dynamics.state.position
    dists = [
        float(np.linalg.norm(intruder.dynamics.state.position - own))
        for intruder in intruder_manager.intruders
    ]
    return min(dists)


def _setup_episode(
    aircraft: AircraftConfig,
    config: EvalConfig,
    seed: int,
) -> tuple[FlightDynamics, IntruderManager, TrimResult, Callable[[], ControlInputs]]:
    environment = Environment()
    sim_config = SimulationConfig(dt=config.dt)
    dynamics = FlightDynamics(aircraft, environment, sim_config)

    cruise_speed = TrainingFidelityConfig.defaults().cruise_speed_mps if config.training_fidelity else 25.0
    cruise_altitude = 1000.0 if config.training_fidelity else 100.0

    if config.training_fidelity:
        apply_training_initial_state(dynamics, seed)
    trim = compute_trim(
        TrimCondition(airspeed=cruise_speed, altitude=cruise_altitude),
        aircraft,
        environment,
    )
    if not config.training_fidelity:
        if trim.success:
            dynamics.reset(trim.state)
        else:
            dynamics.reset()

    intruder_config = IntruderConfig(
        spawn_rate=config.spawn_rate,
        max_intruders=config.max_intruders,
        spawn_distance_min=300.0,
        spawn_distance_max=1200.0,
    )
    intruder_manager = IntruderManager(intruder_config, environment)
    intruder_manager.random.seed(seed)

    throttle_floor = trim.controls.throttle if (config.cruise_throttle_floor and trim.success) else None

    if config.full_policy:
        policy = ModelPolicy(
            config.policy_path,
            device=config.device,
            deterministic=not config.stochastic,
            renderer_backend=config.renderer,
            render_device=config.render_device,
            throttle_mode=config.throttle_mode,
            surface_scale=config.surface_scale,
            cruise_throttle_floor=throttle_floor,
        )
        get_controls = lambda: policy.action_to_controls(
            policy.predict_action(dynamics, intruder_manager),
            aircraft,
        )
    else:
        if not trim.success:
            raise RuntimeError('Trim failed; cannot run trim-assisted policy mode')
        assisted = TrimAssistedModelPolicy.from_checkpoint(
            config.policy_path,
            trim.controls,
            device=config.device,
            config=TrimAssistConfig(
                engage_distance_m=config.engage_distance,
                max_authority=config.max_authority,
                hold_trim_throttle=not config.policy_throttle,
            ),
            renderer_backend=config.renderer,
            throttle_mode=config.throttle_mode,
            render_device=config.render_device,
            surface_scale=config.surface_scale,
        )
        get_controls = lambda: assisted.compute_controls(dynamics, intruder_manager)

    return dynamics, intruder_manager, trim, get_controls


def run_episode(
    aircraft: AircraftConfig,
    config: EvalConfig,
    seed: int,
) -> EpisodeResult:
    dynamics, intruder_manager, _trim, get_controls = _setup_episode(aircraft, config, seed)

    steps = int(config.duration / config.dt)
    min_dist_overall = float('inf')
    nmac_events = 0
    spawn_count = 0
    crash_message = ''

    for _ in range(steps):
        if intruder_manager.should_spawn_intruder(config.dt, dynamics.state):
            intruder_manager.spawn_intruder(dynamics.state)
            spawn_count += 1

        dynamics.step(get_controls())
        intruder_manager.update_intruders(config.dt, dynamics.state)

        d = min_intruder_distance(dynamics, intruder_manager)
        min_dist_overall = min(min_dist_overall, d)
        if d < NMAC_DISTANCE_M:
            nmac_events += 1

        if dynamics.crash_state.crashed:
            crash_message = dynamics.crash_state.crash_message
            break

        if np.any(np.isnan(dynamics.state.to_array())):
            crash_message = 'NaN in state'
            break

    ground_collision = dynamics.state.altitude < GROUND_ALTITUDE_M and dynamics.crash_state.crashed
    nmac_violation = min_dist_overall < NMAC_DISTANCE_M
    timeout = (
        not dynamics.crash_state.crashed
        and dynamics.state.time >= config.duration - config.dt * 0.5
    )

    return EpisodeResult(
        seed=seed,
        sim_time=dynamics.state.time,
        min_separation_m=min_dist_overall,
        final_altitude_m=dynamics.state.altitude,
        final_airspeed_mps=float(np.linalg.norm(dynamics.state.velocity_body)),
        intruders_spawned=spawn_count,
        nmac_proximity_steps=nmac_events,
        crashed=dynamics.crash_state.crashed,
        ground_collision=ground_collision,
        nmac_violation=nmac_violation,
        timeout=timeout,
        crash_message=crash_message,
    )


def run_batch_evaluation(
    aircraft: AircraftConfig,
    config: EvalConfig,
    num_episodes: int,
    base_seed: int = 42,
) -> BatchEvalResult:
    episodes = [run_episode(aircraft, config, base_seed + i) for i in range(num_episodes)]
    return BatchEvalResult(episodes=episodes)


def format_batch_summary(batch: BatchEvalResult) -> str:
    return '\n'.join([
        '',
        '=' * 70,
        'BATCH RESULTS',
        '=' * 70,
        f'  Episodes: {len(batch.episodes)}',
        f'  Success rate: {batch.success_rate:.1f}%',
        f'  NMAC violations: {batch.nmac_rate:.1f}%',
        f'  Ground collisions: {batch.ground_collision_rate:.1f}%',
        f'  Timeouts: {batch.timeout_rate:.1f}%',
        f'  Mean min separation: {batch.mean_min_separation():.1f} m',
        f'  Mean episode length: {batch.mean_sim_time():.1f} s',
    ])
