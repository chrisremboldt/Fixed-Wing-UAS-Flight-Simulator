"""Batch evaluation and control mapping helpers."""

import numpy as np

from simulator.aircraft import AircraftConfig
from simulator.policy.actions import training_action_to_controls
from simulator.policy.evaluation import EvalConfig, run_batch_evaluation, run_episode
from simulator.main import create_default_aircraft

CHECKPOINT = 'final_model.pt'


def test_cruise_throttle_floor():
    aircraft = AircraftConfig()
    action = np.array([-0.9, 0.1, 0.1, -0.1])
    controls = training_action_to_controls(
        action,
        aircraft,
        throttle_mode='clamp',
        cruise_throttle_floor=0.42,
    )
    assert controls.throttle == 0.42


def test_batch_evaluation_runs():
    aircraft = create_default_aircraft()
    config = EvalConfig(
        policy_path=CHECKPOINT,
        duration=2.0,
        dt=0.02,
        training_fidelity=True,
        full_policy=False,
    )
    batch = run_batch_evaluation(aircraft, config, num_episodes=2, base_seed=100)
    assert len(batch.episodes) == 2
    assert batch.success_rate >= 0.0


def test_training_fidelity_episode_completes():
    aircraft = create_default_aircraft()
    config = EvalConfig.from_training_defaults(
        CHECKPOINT,
        duration=5.0,
        seed=42,
        full_policy=False,
    )
    result = run_episode(aircraft, config, seed=42)
    assert result.sim_time >= 4.0
    assert not result.crashed
