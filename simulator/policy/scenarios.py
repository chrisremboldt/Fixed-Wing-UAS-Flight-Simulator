"""Deterministic DAA scenario definitions for policy evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..frames import Quaternion
from ..intruders import IntruderBehavior, IntruderConfig, IntruderManager, IntruderType
from ..state import AircraftState


@dataclass
class FixedIntruderSpawn:
    """Deterministic intruder placement relative to ownship at episode start."""

    intruder_type: IntruderType = IntruderType.CRUISE
    behavior: IntruderBehavior = IntruderBehavior.COOPERATIVE
    distance_m: float = 800.0
    bearing_deg: float = 0.0
    altitude_offset_m: float = 0.0
    speed_mps: float = 40.0
    heading_offset_deg: float = 180.0
    lifetime_s: float = 120.0


@dataclass
class ScenarioConfig:
    """Reproducible evaluation scenario."""

    name: str = 'default'
    description: str = ''
    seed: int | None = None
    spawn_rate: float | None = None
    max_intruders: int | None = None
    spawn_distance_min: float | None = None
    spawn_distance_max: float | None = None
    fixed_intruders: list[FixedIntruderSpawn] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> 'ScenarioConfig':
        data = yaml.safe_load(Path(path).read_text())
        if not isinstance(data, dict):
            raise ValueError(f'Scenario YAML must be a mapping: {path}')

        intruders = data.get('intruders', {})
        fixed_raw = intruders.get('fixed', [])
        fixed = [_parse_fixed_spawn(item) for item in fixed_raw]

        random_cfg = intruders.get('random', {})
        return cls(
            name=str(data.get('name', Path(path).stem)),
            description=str(data.get('description', '')),
            seed=data.get('seed'),
            spawn_rate=random_cfg.get('spawn_rate', intruders.get('spawn_rate')),
            max_intruders=random_cfg.get('max_intruders', intruders.get('max_intruders')),
            spawn_distance_min=random_cfg.get('spawn_distance_min'),
            spawn_distance_max=random_cfg.get('spawn_distance_max'),
            fixed_intruders=fixed,
        )

    def apply_intruder_config(self, base: IntruderConfig) -> IntruderConfig:
        """Merge scenario overrides into base intruder config."""
        return IntruderConfig(
            spawn_rate=self.spawn_rate if self.spawn_rate is not None else base.spawn_rate,
            max_intruders=self.max_intruders if self.max_intruders is not None else base.max_intruders,
            spawn_distance_min=(
                self.spawn_distance_min
                if self.spawn_distance_min is not None
                else base.spawn_distance_min
            ),
            spawn_distance_max=(
                self.spawn_distance_max
                if self.spawn_distance_max is not None
                else base.spawn_distance_max
            ),
            cruise_speed=base.cruise_speed,
            speed_variation=base.speed_variation,
            altitude_change_rate=base.altitude_change_rate,
            behavior_distribution=base.behavior_distribution,
            min_lifetime=base.min_lifetime,
            max_lifetime=base.max_lifetime,
        )


def _parse_fixed_spawn(raw: dict[str, Any]) -> FixedIntruderSpawn:
    return FixedIntruderSpawn(
        intruder_type=IntruderType(str(raw.get('intruder_type', 'cruise'))),
        behavior=IntruderBehavior(str(raw.get('behavior', 'cooperative'))),
        distance_m=float(raw.get('distance_m', 800.0)),
        bearing_deg=float(raw.get('bearing_deg', 0.0)),
        altitude_offset_m=float(raw.get('altitude_offset_m', 0.0)),
        speed_mps=float(raw.get('speed_mps', 40.0)),
        heading_offset_deg=float(raw.get('heading_offset_deg', 180.0)),
        lifetime_s=float(raw.get('lifetime_s', 120.0)),
    )


def resolve_scenario_path(scenario: str) -> Path:
    """Resolve scenario name or path to a YAML file."""
    path = Path(scenario)
    if path.exists():
        return path
    candidate = Path('configs/scenarios') / scenario
    if candidate.exists():
        return candidate
    if not scenario.endswith('.yaml'):
        candidate = Path('configs/scenarios') / f'{scenario}.yaml'
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'Scenario not found: {scenario}')


def spawn_fixed_intruders(
    manager: IntruderManager,
    ownship_state: AircraftState,
    spawns: list[FixedIntruderSpawn],
) -> int:
    """Place deterministic intruders at episode start."""
    count = 0
    ownship_heading = ownship_state.psi

    for spec in spawns:
        if len(manager.intruders) >= manager.config.max_intruders:
            break

        bearing = np.radians(spec.bearing_deg)
        rel_n = spec.distance_m * np.cos(bearing)
        rel_e = spec.distance_m * np.sin(bearing)
        spawn_pos = ownship_state.position + np.array([
            rel_n,
            rel_e,
            -spec.altitude_offset_m,
        ])

        aircraft = manager._create_intruder_aircraft()
        heading = ownship_heading + np.radians(spec.heading_offset_deg)
        quaternion = Quaternion.from_euler(0.0, 0.0, heading)
        initial_state = AircraftState(
            position=spawn_pos,
            velocity_body=np.array([spec.speed_mps, 0.0, 0.0]),
            quaternion=quaternion,
            omega_body=np.zeros(3),
            time=ownship_state.time,
        )

        from ..dynamics import FlightDynamics, SimulationConfig

        dynamics = FlightDynamics(
            aircraft,
            manager.environment,
            SimulationConfig(dt=0.01),
        )
        dynamics.reset(initial_state)

        from ..intruders import IntruderState

        intruder = IntruderState(
            id=manager.next_id,
            aircraft=aircraft,
            dynamics=dynamics,
            behavior=spec.behavior,
            intruder_type=spec.intruder_type,
            spawn_time=ownship_state.time,
            lifetime=spec.lifetime_s,
            last_maneuver_time=ownship_state.time,
        )
        manager.next_id += 1
        manager.intruders.append(intruder)
        count += 1

        distance = float(np.linalg.norm(spawn_pos[:2] - ownship_state.position[:2]))
        altitude = -spawn_pos[2]
        print(
            f'🎯 Scenario intruder {intruder.id} at {distance:.0f}m, '
            f'{altitude:.0f}m alt, {spec.speed_mps:.1f} m/s '
            f'(type: {spec.intruder_type.value}, behavior: {spec.behavior.value})'
        )

    return count
