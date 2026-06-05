"""Tests for DAA policy integration."""

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / 'final_model.pt'


pytest.importorskip('torch')

from simulator.policy.architecture import PufferPolicy, load_policy_checkpoint
from simulator.policy.actions import training_action_to_controls
from simulator.policy.observation import PixelObservationBuilder
from simulator.policy.rendering import CPUPixelRenderer
from simulator.aircraft import AircraftConfig
from simulator.dynamics import FlightDynamics, SimulationConfig
from simulator.environment import Environment
from simulator.state import ControlInputs
from simulator.trim import TrimCondition, compute_trim


@pytest.mark.skipif(not CHECKPOINT.exists(), reason='final_model.pt not present')
def test_checkpoint_loads_strict():
    policy = load_policy_checkpoint(str(CHECKPOINT), device='cpu')
    assert isinstance(policy, PufferPolicy)
    num_params = sum(p.numel() for p in policy.parameters())
    assert num_params > 100_000


@pytest.mark.skipif(not CHECKPOINT.exists(), reason='final_model.pt not present')
def test_deterministic_forward_shape():
    import torch

    policy = load_policy_checkpoint(str(CHECKPOINT), device='cpu')
    obs = torch.zeros(1, 128, 128, 3, dtype=torch.uint8)
    action = policy.deterministic_action(obs)
    assert action.shape == (1, 4)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)


def test_pixel_observation_shape():
    aircraft = AircraftConfig()
    env = Environment()
    dynamics = FlightDynamics(aircraft, env, SimulationConfig(dt=0.01))
    trim = compute_trim(TrimCondition(airspeed=25.0, altitude=100.0), aircraft, env)
    dynamics.reset(trim.state if trim.success else None)

    builder = PixelObservationBuilder()
    obs = builder.build(dynamics, None)
    assert obs.shape == (128, 128, 3)
    assert obs.dtype == np.uint8


def test_action_mapping_order():
    aircraft = AircraftConfig()
    action = np.array([0.5, -0.25, 0.1, -0.05], dtype=np.float32)
    controls = training_action_to_controls(action, aircraft)
    assert controls.throttle == pytest.approx(0.75, abs=0.01)
    assert controls.aileron == pytest.approx(-0.25 * aircraft.max_aileron)
    assert controls.elevator == pytest.approx(0.1 * aircraft.max_elevator)
    assert controls.rudder == pytest.approx(-0.05 * aircraft.max_rudder)


@pytest.mark.skipif(not CHECKPOINT.exists(), reason='final_model.pt not present')
def test_model_policy_end_to_end():
    from simulator.policy.model_policy import ModelPolicy

    aircraft = AircraftConfig()
    environment = Environment()
    dynamics = FlightDynamics(aircraft, environment, SimulationConfig(dt=0.01))
    trim = compute_trim(TrimCondition(airspeed=25.0, altitude=100.0), aircraft, environment)
    dynamics.reset(trim.state if trim.success else None)

    policy = ModelPolicy(CHECKPOINT, device='cpu')
    controls = policy.intervene(ControlInputs(), dynamics, None)
    assert isinstance(controls, ControlInputs)
    assert 0.0 <= controls.throttle <= 1.0
