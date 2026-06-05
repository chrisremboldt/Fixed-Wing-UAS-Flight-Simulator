"""Training-matched initial state."""

import numpy as np

from simulator.aircraft import AircraftConfig
from simulator.dynamics import FlightDynamics, SimulationConfig
from simulator.environment import Environment
from simulator.policy.training_init import (
    TRAINING_CRUISE_SPEED_MPS,
    apply_training_initial_state,
    sample_training_initial_state,
)


def test_sample_training_initial_state_ranges():
    state = sample_training_initial_state(42)
    assert 500.0 <= state.altitude <= 1500.0
    assert state.u == TRAINING_CRUISE_SPEED_MPS
    assert np.linalg.norm(state.omega_body) == 0.0


def test_apply_training_initial_state_resets_dynamics():
    dyn = FlightDynamics(AircraftConfig(), Environment(), SimulationConfig(dt=0.02))
    apply_training_initial_state(dyn, 7)
    assert 500.0 <= dyn.state.altitude <= 1500.0
    assert dyn.state.u == TRAINING_CRUISE_SPEED_MPS
