"""Batch policy evaluation with training-style metrics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ..aircraft import AircraftConfig
from ..dynamics import CrashType, FlightDynamics, SimulationConfig
from ..environment import Environment
from ..intruders import IntruderConfig, IntruderManager
from ..state import ControlInputs
from ..trim import TrimCondition, TrimResult, compute_trim
from .model_policy import ModelPolicy
from .scenarios import ScenarioConfig, spawn_fixed_intruders
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
    scenario: ScenarioConfig | None = None

    @classmethod
    def from_training_defaults(
        cls,
        policy_path: str,
        *,
        duration: float = 20.0,
        seed: int = 42,
        full_policy: bool = False,
        device: str = 'cpu',
        scenario: ScenarioConfig | None = None,
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
            scenario=scenario,
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
    failure_reason: str = 'success'
    time_to_cpa_s: float = float('inf')
    action_saturation_rate: float = 0.0
    scenario_name: str = ''

    @property
    def success(self) -> bool:
        return self.failure_reason == 'success'


@dataclass
class BatchEvalResult:
    episodes: list[EpisodeResult] = field(default_factory=list)
    scenario_name: str = ''

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

    def failure_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for ep in self.episodes:
            counts[ep.failure_reason] = counts.get(ep.failure_reason, 0) + 1
        return counts

    def mean_min_separation(self) -> float:
        vals = [ep.min_separation_m for ep in self.episodes if np.isfinite(ep.min_separation_m)]
        return float(np.mean(vals)) if vals else float('inf')

    def mean_sim_time(self) -> float:
        return float(np.mean([ep.sim_time for ep in self.episodes])) if self.episodes else 0.0

    def mean_time_to_cpa(self) -> float:
        vals = [ep.time_to_cpa_s for ep in self.episodes if np.isfinite(ep.time_to_cpa_s)]
        return float(np.mean(vals)) if vals else float('inf')


def min_intruder_distance(dynamics: FlightDynamics, intruder_manager: IntruderManager) -> float:
    if not intruder_manager.intruders:
        return float('inf')
    own = dynamics.state.position
    dists = [
        float(np.linalg.norm(intruder.dynamics.state.position - own))
        for intruder in intruder_manager.intruders
    ]
    return min(dists)


def _build_sim_config(config: EvalConfig) -> SimulationConfig:
    if config.training_fidelity:
        return TrainingFidelityConfig.defaults().simulation_config(config.dt)
    return SimulationConfig(dt=config.dt)


def _classify_failure(
    dynamics: FlightDynamics,
    *,
    min_dist: float,
    nmac_violation: bool,
    ground_collision: bool,
    timeout: bool,
    crash_message: str,
) -> str:
    if 'NaN' in crash_message:
        return 'nan'
    if dynamics.crash_state.crashed:
        crash_type = dynamics.crash_state.crash_type
        if crash_type == CrashType.OVERSPEED:
            return 'overspeed'
        if crash_type == CrashType.GROUND_COLLISION:
            return 'ground'
        if crash_type in (CrashType.STALL_SPIN, CrashType.UNDERSPEED):
            return 'stall'
        if crash_type == CrashType.STRUCTURAL_FAILURE:
            return 'structural'
        return 'crash'
    if nmac_violation:
        return 'nmac'
    if ground_collision:
        return 'ground'
    if timeout:
        return 'success'
    if min_dist < NMAC_DISTANCE_M:
        return 'nmac'
    return 'success'


def _setup_episode(
    aircraft: AircraftConfig,
    config: EvalConfig,
    seed: int,
) -> tuple[FlightDynamics, IntruderManager, TrimResult, Callable[[], ControlInputs], dict]:
    environment = Environment()
    sim_config = _build_sim_config(config)
    dynamics = FlightDynamics(aircraft, environment, sim_config)

    episode_seed = config.scenario.seed if config.scenario and config.scenario.seed is not None else seed

    cruise_speed = TrainingFidelityConfig.defaults().cruise_speed_mps if config.training_fidelity else 25.0
    cruise_altitude = 1000.0 if config.training_fidelity else 100.0

    if config.training_fidelity:
        apply_training_initial_state(dynamics, episode_seed)
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

    if config.training_fidelity:
        intruder_config = TrainingFidelityConfig.defaults().intruder_config(
            spawn_rate=config.spawn_rate,
            max_intruders=config.max_intruders,
        )
    else:
        intruder_config = IntruderConfig(
            spawn_rate=config.spawn_rate,
            max_intruders=config.max_intruders,
            spawn_distance_min=300.0,
            spawn_distance_max=1200.0,
        )
    if config.scenario is not None:
        intruder_config = config.scenario.apply_intruder_config(intruder_config)

    intruder_manager = IntruderManager(intruder_config, environment)
    intruder_manager.random.seed(episode_seed)

    throttle_floor = trim.controls.throttle if (config.cruise_throttle_floor and trim.success) else None
    metrics_state = {'saturated_steps': 0, 'total_steps': 0}

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

        def get_controls() -> ControlInputs:
            action = policy.predict_action(dynamics, intruder_manager)
            metrics_state['total_steps'] += 1
            if np.max(np.abs(action[:4])) > 0.95:
                metrics_state['saturated_steps'] += 1
            return policy.action_to_controls(action, aircraft)

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
            recovery_state=dynamics.state,
        )

        def get_controls() -> ControlInputs:
            action = assisted.model_policy.predict_action(dynamics, intruder_manager)
            metrics_state['total_steps'] += 1
            if np.max(np.abs(action[:4])) > 0.95:
                metrics_state['saturated_steps'] += 1
            return assisted.compute_controls(dynamics, intruder_manager)

    return dynamics, intruder_manager, trim, get_controls, metrics_state


def run_episode(
    aircraft: AircraftConfig,
    config: EvalConfig,
    seed: int,
) -> EpisodeResult:
    dynamics, intruder_manager, _trim, get_controls, metrics_state = _setup_episode(
        aircraft, config, seed,
    )

    spawn_count = 0
    if config.scenario and config.scenario.fixed_intruders:
        spawn_count = spawn_fixed_intruders(
            intruder_manager,
            dynamics.state,
            config.scenario.fixed_intruders,
        )
    elif config.training_fidelity:
        spawn_count += intruder_manager.spawn_initial_intruders(dynamics.state)

    steps = int(config.duration / config.dt)
    min_dist_overall = float('inf')
    time_to_cpa = float('inf')
    nmac_events = 0
    crash_message = ''

    for _ in range(steps):
        if intruder_manager.should_spawn_intruder(config.dt, dynamics.state):
            intruder_manager.spawn_intruder(dynamics.state)
            spawn_count += 1

        dynamics.step(get_controls())
        intruder_manager.update_intruders(config.dt, dynamics.state)

        d = min_intruder_distance(dynamics, intruder_manager)
        if d < min_dist_overall:
            min_dist_overall = d
            time_to_cpa = dynamics.state.time
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
    failure_reason = _classify_failure(
        dynamics,
        min_dist=min_dist_overall,
        nmac_violation=nmac_violation,
        ground_collision=ground_collision,
        timeout=timeout,
        crash_message=crash_message,
    )
    total = max(metrics_state['total_steps'], 1)
    saturation = metrics_state['saturated_steps'] / total

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
        failure_reason=failure_reason,
        time_to_cpa_s=time_to_cpa,
        action_saturation_rate=float(saturation),
        scenario_name=config.scenario.name if config.scenario else '',
    )


def run_batch_evaluation(
    aircraft: AircraftConfig,
    config: EvalConfig,
    num_episodes: int,
    base_seed: int = 42,
) -> BatchEvalResult:
    episodes = [run_episode(aircraft, config, base_seed + i) for i in range(num_episodes)]
    scenario_name = config.scenario.name if config.scenario else ''
    return BatchEvalResult(episodes=episodes, scenario_name=scenario_name)


def format_batch_summary(batch: BatchEvalResult) -> str:
    lines = [
        '',
        '=' * 70,
        'BATCH RESULTS',
        '=' * 70,
    ]
    if batch.scenario_name:
        lines.append(f'  Scenario: {batch.scenario_name}')
    lines.extend([
        f'  Episodes: {len(batch.episodes)}',
        f'  Success rate: {batch.success_rate:.1f}%',
        f'  NMAC violations: {batch.nmac_rate:.1f}%',
        f'  Ground collisions: {batch.ground_collision_rate:.1f}%',
        f'  Timeouts: {batch.timeout_rate:.1f}%',
        f'  Mean min separation: {batch.mean_min_separation():.1f} m',
        f'  Mean time to CPA: {batch.mean_time_to_cpa():.1f} s',
        f'  Mean episode length: {batch.mean_sim_time():.1f} s',
    ])

    failure_counts = batch.failure_counts()
    if failure_counts:
        lines.append('  Failure breakdown:')
        for reason, count in sorted(failure_counts.items()):
            lines.append(f'    {reason}: {count}')

    return '\n'.join(lines)


def export_batch_results(batch: BatchEvalResult, path: str | Path) -> None:
    """Write per-episode JSON records for CI or trend tracking."""
    output_path = Path(path)

    def _json_safe(value):
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, (np.bool_, bool)):
            return bool(value)
        if isinstance(value, dict):
            return {k: _json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_json_safe(v) for v in value]
        return value

    payload = _json_safe({
        'scenario': batch.scenario_name,
        'summary': {
            'episodes': len(batch.episodes),
            'success_rate_pct': batch.success_rate,
            'nmac_rate_pct': batch.nmac_rate,
            'ground_collision_rate_pct': batch.ground_collision_rate,
            'timeout_rate_pct': batch.timeout_rate,
            'mean_min_separation_m': batch.mean_min_separation(),
            'mean_time_to_cpa_s': batch.mean_time_to_cpa(),
            'mean_sim_time_s': batch.mean_sim_time(),
            'failure_counts': batch.failure_counts(),
        },
        'episodes': [asdict(ep) for ep in batch.episodes],
    })
    output_path.write_text(json.dumps(payload, indent=2))
