"""Tests for training-matched renderer."""

from pathlib import Path

import numpy as np

from simulator.aircraft import AircraftConfig
from simulator.dynamics import FlightDynamics, SimulationConfig
from simulator.environment import Environment
from simulator.intruders import IntruderManager, IntruderConfig
from simulator.policy.training_render import TrainingPixelRenderer, create_policy_renderer
from simulator.trim import TrimCondition, compute_trim

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / 'final_model.pt'


def _setup_with_intruder():
    aircraft = AircraftConfig()
    env = Environment()
    dyn = FlightDynamics(aircraft, env, SimulationConfig(dt=0.01))
    trim = compute_trim(TrimCondition(airspeed=25.0, altitude=100.0), aircraft, env)
    dyn.reset(trim.state if trim.success else None)
    mgr = IntruderManager(IntruderConfig(max_intruders=1), env)
    mgr.spawn_intruder(dyn.state)
    return dyn, mgr


def test_training_renderer_output_shape():
    dyn, mgr = _setup_with_intruder()
    renderer = TrainingPixelRenderer()
    obs = renderer.render(dyn, mgr)
    assert obs.shape == (128, 128, 3)
    assert obs.dtype == np.uint8
    assert obs.mean() > 10


def test_create_policy_renderer_training_backend():
    renderer = create_policy_renderer(backend='training')
    dyn, mgr = _setup_with_intruder()
    obs = renderer.render(dyn, mgr)
    assert obs.shape == (128, 128, 3)


def test_training_renderer_changes_policy_action():
    pytest = __import__('pytest')
    if not CHECKPOINT.exists():
        pytest.skip('final_model.pt not present')

    from simulator.policy import ModelPolicy

    dyn, mgr = _setup_with_intruder()
    legacy = ModelPolicy(CHECKPOINT, renderer_backend='legacy', device='cpu')
    training = ModelPolicy(CHECKPOINT, renderer_backend='training', device='cpu')

    a_legacy = legacy.predict_action(dyn, mgr)
    a_training = training.predict_action(dyn, mgr)
    assert not np.allclose(a_legacy, a_training, atol=1e-3)
