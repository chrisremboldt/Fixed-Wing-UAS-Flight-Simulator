"""Tests for intruder spawning and visualization export."""

import numpy as np
import pytest

from simulator.environment import Environment
from simulator.frames import Quaternion
from simulator.intruders import IntruderConfig, IntruderManager
from simulator.state import AircraftState


@pytest.fixture
def intruder_manager():
    return IntruderManager(
        IntruderConfig(spawn_rate=1.0, max_intruders=1),
        Environment(),
    )


def test_intruder_spawn_velocity_in_body_frame(intruder_manager):
    """Spawned intruders should have forward body speed, not NED components."""
    heading = np.pi / 3
    ownship = AircraftState(
        position=np.array([0.0, 0.0, -100.0]),
        velocity_body=np.array([25.0, 0.0, 0.0]),
        quaternion=Quaternion.from_euler(0.0, 0.0, heading),
        time=0.0,
    )

    intruder = intruder_manager.spawn_intruder(ownship)
    state = intruder.dynamics.state

    speed = np.linalg.norm(state.velocity_body)
    assert speed > 20.0
    assert abs(state.velocity_body[1]) < 1e-6
    assert abs(state.velocity_body[2]) < 1e-6
    assert state.velocity_body[0] > 0.0

    v_ned = state.velocity_ned
    expected_north = speed * np.cos(state.psi)
    expected_east = speed * np.sin(state.psi)
    assert abs(v_ned[0] - expected_north) < 0.5
    assert abs(v_ned[1] - expected_east) < 0.5


def test_intruder_state_export_matches_ownship_schema(intruder_manager):
    """Intruder WebSocket payload should use the same position/quaternion shape."""
    ownship = AircraftState(
        position=np.array([0.0, 0.0, -100.0]),
        velocity_body=np.array([25.0, 0.0, 0.0]),
        quaternion=Quaternion.from_euler(0.0, 0.0, 0.0),
        time=0.0,
    )
    intruder_manager.spawn_intruder(ownship)
    exported = intruder_manager.get_intruder_states()

    assert len(exported) == 1
    item = exported[0]
    assert set(item['position'].keys()) == {'x', 'y', 'z'}
    assert set(item['quaternion'].keys()) == {'w', 'x', 'y', 'z'}
