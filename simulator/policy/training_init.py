"""Initial state aligned with scratch_built_daa FixedWingPhysics.reset."""

from __future__ import annotations

import numpy as np

from ..dynamics import FlightDynamics
from ..frames import Quaternion
from ..state import AircraftState


TRAINING_CRUISE_SPEED_MPS = 40.0
TRAINING_ALTITUDE_MIN_M = 500.0
TRAINING_ALTITUDE_MAX_M = 1500.0


def sample_training_initial_state(seed: int) -> AircraftState:
    """Sample ego state matching training env reset (level cruise at 40 m/s)."""
    rng = np.random.default_rng(seed)
    altitude = float(rng.uniform(TRAINING_ALTITUDE_MIN_M, TRAINING_ALTITUDE_MAX_M))
    heading = float(rng.uniform(0.0, 2.0 * np.pi))

    return AircraftState(
        position=np.array([
            float(rng.uniform(-1000.0, 1000.0)),
            float(rng.uniform(-1000.0, 1000.0)),
            -altitude,
        ]),
        velocity_body=np.array([TRAINING_CRUISE_SPEED_MPS, 0.0, 0.0]),
        quaternion=Quaternion.from_euler(0.0, 0.0, heading),
        omega_body=np.zeros(3),
        time=0.0,
    )


def apply_training_initial_state(dynamics: FlightDynamics, seed: int) -> AircraftState:
    """Reset dynamics to training-matched initial conditions."""
    state = sample_training_initial_state(seed)
    dynamics.reset(state)
    return state
