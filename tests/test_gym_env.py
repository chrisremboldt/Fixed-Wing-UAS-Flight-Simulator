"""Tests for gym-style UAS DAA env spike (issue #14)."""

import numpy as np

from simulator.gym_env import UASDAAEnv, UASDAAEnvConfig
from simulator.main import create_default_aircraft


def test_gym_env_reset_and_step():
    env = UASDAAEnv(
        aircraft=create_default_aircraft(),
        config=UASDAAEnvConfig(duration=2.0, dt=0.02, seed=42, policy_path=None),
    )
    obs, info = env.reset(seed=42)
    assert obs.shape == (128, 128, 3)
    assert obs.dtype == np.uint8

    action = np.zeros(4, dtype=np.float32)
    obs2, reward, terminated, truncated, step_info = env.step(action)
    assert obs2.shape == (128, 128, 3)
    assert isinstance(reward, float)
    assert 'sim_time' in step_info
