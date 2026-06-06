#!/usr/bin/env python3
"""
Evaluate a trained DAA checkpoint in the UAS sim (headless).

Mirrors scratch_built_daa/evaluate_model.py but uses this simulator's physics
and training-matched pixel observations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from simulator.main import create_default_aircraft
from simulator.policy import (
    EvalConfig,
    TrainingFidelityConfig,
    format_batch_summary,
    run_batch_evaluation,
    run_episode,
)

NMAC_DISTANCE_M = 152.4


def build_eval_config(args: argparse.Namespace) -> EvalConfig:
    if args.training_fidelity:
        tf = TrainingFidelityConfig.defaults()
        duration = args.duration if args.duration is not None else tf.episode_duration_s
        return EvalConfig(
            policy_path=args.policy,
            duration=duration,
            dt=args.dt if args.dt is not None else tf.dt,
            seed=args.seed,
            device=args.device,
            spawn_rate=tf.spawn_rate if args.spawn_rate == 0.15 else args.spawn_rate,
            max_intruders=args.max_intruders,
            training_fidelity=True,
            full_policy=args.full_policy,
            renderer=args.renderer if args.renderer is not None else tf.renderer_backend,
            render_device=args.render_device,
            throttle_mode=args.throttle_mode if args.throttle_mode is not None else tf.throttle_mode,
            stochastic=args.stochastic,
            engage_distance=args.engage_distance,
            max_authority=args.max_authority,
            policy_throttle=args.policy_throttle,
            surface_scale=tf.surface_scale,
            cruise_throttle_floor=tf.cruise_throttle_floor and not args.no_throttle_floor,
        )

    return EvalConfig(
        policy_path=args.policy,
        duration=args.duration if args.duration is not None else 120.0,
        dt=args.dt if args.dt is not None else 0.01,
        seed=args.seed,
        device=args.device,
        spawn_rate=args.spawn_rate,
        max_intruders=args.max_intruders,
        training_fidelity=False,
        full_policy=args.full_policy,
        renderer=args.renderer if args.renderer is not None else 'training',
        render_device=args.render_device,
        throttle_mode=args.throttle_mode if args.throttle_mode is not None else 'symmetric',
        stochastic=args.stochastic,
        engage_distance=args.engage_distance,
        max_authority=args.max_authority,
        policy_throttle=args.policy_throttle,
        cruise_throttle_floor=False,
    )


def print_header(config: EvalConfig, args: argparse.Namespace) -> None:
    print('=' * 70)
    print('DAA Policy Evaluation (UAS Sim)')
    print('=' * 70)
    print(f'Model: {config.policy_path}')
    print(f'Duration: {config.duration}s | dt={config.dt}s | Seed: {config.seed}')
    print(f'Policy device: {config.device} | Renderer: {config.renderer}')
    if args.episodes > 1:
        print(f'Episodes: {args.episodes}')
    if config.training_fidelity and config.full_policy:
        print('Mode: training fidelity + full policy')
    elif config.training_fidelity:
        print('Mode: training fidelity (trim-assisted)')
    elif config.full_policy:
        print('Mode: full policy (trim assist off)')
    else:
        print('Mode: trim-assisted (default)')


def print_episode_result(result) -> None:
    if result.crash_message:
        print(f'Crash at t={result.sim_time:.1f}s: {result.crash_message}')
    print('\n--- Results ---')
    print(f'  Sim time: {result.sim_time:.1f}s')
    print(f'  Final altitude: {result.final_altitude_m:.1f} m')
    print(f'  Final airspeed: {result.final_airspeed_mps:.1f} m/s')
    print(f'  Intruders spawned: {result.intruders_spawned}')
    print(f'  Min separation: {result.min_separation_m:.1f} m')
    print(f'  NMAC proximity steps (<{NMAC_DISTANCE_M:.0f} m): {result.nmac_proximity_steps}')
    print(f'  Crashed: {result.crashed}')
    print(f'  Success: {result.success}')


def run_evaluation(args: argparse.Namespace) -> int:
    config = build_eval_config(args)
    print_header(config, args)

    aircraft = create_default_aircraft()

    if args.episodes > 1:
        batch = run_batch_evaluation(aircraft, config, args.episodes, base_seed=config.seed)
        print(format_batch_summary(batch))
        return 0

    result = run_episode(aircraft, config, config.seed)
    print_episode_result(result)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate DAA policy in UAS sim')
    parser.add_argument('--policy', type=str, default='final_model.pt', help='Checkpoint path')
    parser.add_argument('--duration', type=float, default=None, help='Episode length (s)')
    parser.add_argument('--episodes', type=int, default=1, help='Number of test episodes')
    parser.add_argument('--dt', type=float, default=None, help='Physics timestep')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--spawn-rate', type=float, default=0.15, dest='spawn_rate')
    parser.add_argument('--max-intruders', type=int, default=5, dest='max_intruders')
    parser.add_argument('--stochastic', action='store_true', help='Sample actions (default: deterministic)')
    parser.add_argument(
        '--full-policy',
        action='store_true',
        help='Use raw policy controls (no trim assist)',
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
        '--no-throttle-floor',
        action='store_true',
        help='Disable cruise throttle floor in full-policy training-fidelity mode',
    )
    parser.add_argument(
        '--training-fidelity',
        action='store_true',
        help='Match scratch_built_daa presets (20s episodes, 50Hz, training init/renderer)',
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

    if not Path(args.policy).exists():
        raise SystemExit(f'Checkpoint not found: {args.policy}')

    raise SystemExit(run_evaluation(args))


if __name__ == '__main__':
    main()
