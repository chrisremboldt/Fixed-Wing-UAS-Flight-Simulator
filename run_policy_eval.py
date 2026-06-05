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
from simulator.policy import (
    ModelPolicy,
    TrainingFidelityConfig,
    TrimAssistedModelPolicy,
    TrimAssistConfig,
    apply_training_initial_state,
)
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
    print(f'Duration: {args.duration}s | dt={args.dt}s | Seed: {args.seed}')
    print(f'Policy device: {args.device} | Renderer: {args.renderer}')
    if args.training_fidelity and args.full_policy:
        print('Mode: training fidelity + full policy (Warp-like closed loop)')
    elif args.training_fidelity:
        print('Mode: training fidelity (50Hz, training renderer, trim-assisted)')
    elif args.full_policy:
        print('Mode: full policy (trim assist off)')
    else:
        print('Mode: trim-assisted (default)')

    aircraft = create_default_aircraft()
    environment = Environment()
    sim_config = SimulationConfig(dt=args.dt)
    dynamics = FlightDynamics(aircraft, environment, sim_config)

    tf = TrainingFidelityConfig.defaults() if args.training_fidelity else None
    cruise_speed = tf.cruise_speed_mps if tf else 25.0
    cruise_altitude = 1000.0 if tf else 100.0

    if args.training_fidelity:
        apply_training_initial_state(dynamics, args.seed)
        trim = compute_trim(
            TrimCondition(airspeed=cruise_speed, altitude=cruise_altitude),
            aircraft,
            environment,
        )
    else:
        trim = compute_trim(
            TrimCondition(airspeed=cruise_speed, altitude=cruise_altitude),
            aircraft,
            environment,
        )
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

    if args.full_policy:
        policy = ModelPolicy(
            args.policy,
            device=args.device,
            deterministic=not args.stochastic,
            renderer_backend=args.renderer,
            render_device=args.render_device,
            throttle_mode=args.throttle_mode,
        )
        get_controls = lambda: policy.action_to_controls(
            policy.predict_action(dynamics, intruder_manager),
            aircraft,
        )
    else:
        if not trim.success:
            raise SystemExit('Trim failed; cannot run trim-assisted policy mode')
        assisted = TrimAssistedModelPolicy.from_checkpoint(
            args.policy,
            trim.controls,
            device=args.device,
            config=TrimAssistConfig(
                engage_distance_m=args.engage_distance,
                max_authority=args.max_authority,
                hold_trim_throttle=not args.policy_throttle,
            ),
            renderer_backend=args.renderer,
            throttle_mode=args.throttle_mode,
            render_device=args.render_device,
        )
        get_controls = lambda: assisted.compute_controls(dynamics, intruder_manager)

    steps = int(args.duration / sim_config.dt)
    min_dist_overall = float('inf')
    nmac_events = 0
    spawn_count = 0

    for step in range(steps):
        if intruder_manager.should_spawn_intruder(sim_config.dt, dynamics.state):
            intruder_manager.spawn_intruder(dynamics.state)
            spawn_count += 1

        dynamics.step(get_controls())
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
    parser.add_argument('--dt', type=float, default=None, help='Physics timestep (default 0.01, 0.02 with --training-fidelity)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--spawn-rate', type=float, default=0.15, dest='spawn_rate')
    parser.add_argument('--max-intruders', type=int, default=5, dest='max_intruders')
    parser.add_argument('--stochastic', action='store_true', help='Sample actions (default: deterministic)')
    parser.add_argument(
        '--full-policy',
        action='store_true',
        help='Use raw policy controls (no trim assist; likely unstable in this sim)',
    )
    parser.add_argument(
        '--engage-distance',
        type=float,
        default=1000.0,
        dest='engage_distance',
        help='Start blending policy inside this range (m)',
    )
    parser.add_argument(
        '--max-authority',
        type=float,
        default=0.6,
        dest='max_authority',
        help='Max policy blend at closest threat',
    )
    parser.add_argument(
        '--policy-throttle',
        action='store_true',
        help='Let policy command throttle (default: hold trim throttle)',
    )
    parser.add_argument(
        '--training-fidelity',
        action='store_true',
        help='Match scratch_built_daa: dt=0.02, training init, renderer, clamp throttle (add --full-policy for raw controls)',
    )
    parser.add_argument(
        '--renderer',
        type=str,
        default=None,
        choices=['auto', 'training', 'gpu', 'legacy'],
        help='Observation renderer backend',
    )
    parser.add_argument(
        '--render-device',
        type=str,
        default='cpu',
        dest='render_device',
        help='Device for gpu renderer backend (cuda:0 when available)',
    )
    parser.add_argument(
        '--throttle-mode',
        type=str,
        default=None,
        choices=['symmetric', 'clamp'],
        dest='throttle_mode',
        help='Map policy throttle: symmetric (a+1)/2 or clamp [0,1] like Warp physics',
    )
    args = parser.parse_args()

    if args.training_fidelity:
        tf = TrainingFidelityConfig.defaults()
        if args.dt is None:
            args.dt = tf.dt
        if args.renderer is None:
            args.renderer = tf.renderer_backend
        if args.throttle_mode is None:
            args.throttle_mode = tf.throttle_mode
        if args.spawn_rate == 0.15:
            args.spawn_rate = tf.spawn_rate
        if args.render_device == 'cpu' and tf.policy_device.startswith('cuda'):
            args.render_device = tf.policy_device
    else:
        if args.dt is None:
            args.dt = 0.01
        if args.renderer is None:
            args.renderer = 'training'
        if args.throttle_mode is None:
            args.throttle_mode = 'symmetric'

    if not Path(args.policy).exists():
        raise SystemExit(f'Checkpoint not found: {args.policy}')

    raise SystemExit(run_evaluation(args))


if __name__ == '__main__':
    main()
