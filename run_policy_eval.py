#!/usr/bin/env python3
"""
Evaluate a trained DAA checkpoint in the UAS sim (headless).

Mirrors scratch_built_daa/evaluate_model.py but uses this simulator's physics
and CPU pixel observations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from simulator.aircraft import AircraftConfig
from simulator.dynamics import FlightDynamics, SimulationConfig
from simulator.environment import Environment
from simulator.intruders import IntruderConfig, IntruderManager
from simulator.main import create_default_aircraft
from simulator.policy import ModelPolicy
from simulator.trim import TrimCondition, compute_trim

NMAC_DISTANCE_M = 152.4  # 500 ft


def min_intruder_distance(dynamics: FlightDynamics, intruder_manager: IntruderManager) -> float:
    if not intruder_manager.intruders:
        return float('inf')
    own = dynamics.state.position
    dists = [
        float(np.linalg.norm(intruder.dynamics.state.position - own))
        for intruder in intruder_manager.intruders
    ]
    return min(dists)


def run_evaluation(args: argparse.Namespace) -> int:
    print('=' * 70)
    print('DAA Policy Evaluation (UAS Sim)')
    print('=' * 70)
    print(f'Model: {args.policy}')
    print(f'Duration: {args.duration}s | Seed: {args.seed} | Device: {args.device}')

    aircraft = create_default_aircraft()
    environment = Environment()
    sim_config = SimulationConfig(dt=args.dt)
    dynamics = FlightDynamics(aircraft, environment, sim_config)

    trim = compute_trim(TrimCondition(airspeed=25.0, altitude=100.0), aircraft, environment)
    if trim.success:
        dynamics.reset(trim.state)
    else:
        dynamics.reset()

    intruder_config = IntruderConfig(
        spawn_rate=args.spawn_rate,
        max_intruders=args.max_intruders,
        spawn_distance_min=300.0,
        spawn_distance_max=1200.0,
    )
    intruder_manager = IntruderManager(intruder_config, environment)
    intruder_manager.random.seed(args.seed)

    policy = ModelPolicy(
        args.policy,
        device=args.device,
        deterministic=not args.stochastic,
    )

    steps = int(args.duration / sim_config.dt)
    min_dist_overall = float('inf')
    nmac_events = 0
    spawn_count = 0

    for step in range(steps):
        if intruder_manager.should_spawn_intruder(sim_config.dt, dynamics.state):
            intruder_manager.spawn_intruder(dynamics.state)
            spawn_count += 1

        controls = policy.action_to_controls(
            policy.predict_action(dynamics, intruder_manager),
            aircraft,
        )
        dynamics.step(controls)
        intruder_manager.update_intruders(sim_config.dt, dynamics.state)

        d = min_intruder_distance(dynamics, intruder_manager)
        min_dist_overall = min(min_dist_overall, d)
        if d < NMAC_DISTANCE_M:
            nmac_events += 1

        if dynamics.crash_state.crashed:
            print(f'Crash at t={dynamics.state.time:.1f}s: {dynamics.crash_state.crash_message}')
            break

        if np.any(np.isnan(dynamics.state.to_array())):
            print(f'NaN at t={dynamics.state.time:.1f}s')
            break

    print('\n--- Results ---')
    print(f'  Sim time: {dynamics.state.time:.1f}s')
    print(f'  Final altitude: {dynamics.state.altitude:.1f} m')
    airspeed = float(np.linalg.norm(dynamics.state.velocity_body))
    print(f'  Final airspeed: {airspeed:.1f} m/s')
    print(f'  Intruders spawned: {spawn_count}')
    print(f'  Min separation: {min_dist_overall:.1f} m')
    print(f'  NMAC proximity steps (<{NMAC_DISTANCE_M:.0f} m): {nmac_events}')
    print(f'  Crashed: {dynamics.crash_state.crashed}')
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate DAA policy in UAS sim')
    parser.add_argument('--policy', type=str, default='final_model.pt', help='Checkpoint path')
    parser.add_argument('--duration', type=float, default=120.0, help='Episode length (s)')
    parser.add_argument('--dt', type=float, default=0.01, help='Physics timestep')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--spawn-rate', type=float, default=0.15, dest='spawn_rate')
    parser.add_argument('--max-intruders', type=int, default=5, dest='max_intruders')
    parser.add_argument('--stochastic', action='store_true', help='Sample actions (default: deterministic)')
    args = parser.parse_args()

    if not Path(args.policy).exists():
        raise SystemExit(f'Checkpoint not found: {args.policy}')

    raise SystemExit(run_evaluation(args))


if __name__ == '__main__':
    main()
