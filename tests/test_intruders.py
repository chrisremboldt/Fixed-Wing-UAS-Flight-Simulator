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


def test_intruder_spawn_altitude_relative_to_ownship():
    """Training-fidelity spawns should stay near ownship altitude, not sea level."""
    config = IntruderConfig(
        spawn_rate=1.0,
        max_intruders=1,
        spawn_altitude_relative_to_ownship=True,
        spawn_altitude_offset_std_m=0.0,
    )
    manager = IntruderManager(config, Environment())
    ownship_alt = 1200.0
    ownship = AircraftState(
        position=np.array([0.0, 0.0, -ownship_alt]),
        velocity_body=np.array([40.0, 0.0, 0.0]),
        quaternion=Quaternion.from_euler(0.0, 0.0, 0.0),
        time=0.0,
    )

    intruder = manager.spawn_intruder(ownship)
    intruder_alt = -intruder.dynamics.state.position[2]
    assert abs(intruder_alt - ownship_alt) < 50.0


def test_adversarial_forward_spawn_bias():
    """Forward-biased config should place most spawns ahead of ownship."""
    config = IntruderConfig(
        spawn_rate=1.0,
        max_intruders=50,
        spawn_forward_fraction=1.0,
        spawn_forward_cone_deg=45.0,
    )
    manager = IntruderManager(config, Environment())
    manager.random.seed(7)
    heading = np.pi / 3
    ownship = AircraftState(
        position=np.array([100.0, -200.0, -1200.0]),
        velocity_body=np.array([40.0, 0.0, 0.0]),
        quaternion=Quaternion.from_euler(0.0, 0.0, heading),
        time=0.0,
    )

    in_front = 0
    for _ in range(40):
        manager.intruders.clear()
        pos = manager._generate_spawn_position(ownship)
        rel_n = pos[0] - ownship.p_north
        rel_e = pos[1] - ownship.p_east
        bearing = np.arctan2(rel_e, rel_n)
        delta = np.arctan2(np.sin(bearing - heading), np.cos(bearing - heading))
        if abs(delta) <= np.radians(45.0):
            in_front += 1

    assert in_front >= 38


def test_spawn_initial_intruders():
    config = IntruderConfig(
        initial_spawn_count=3,
        max_intruders=5,
        spawn_forward_fraction=1.0,
        spawn_forward_cone_deg=60.0,
    )
    manager = IntruderManager(config, Environment())
    ownship = AircraftState(
        position=np.array([0.0, 0.0, -1000.0]),
        velocity_body=np.array([40.0, 0.0, 0.0]),
        quaternion=Quaternion.from_euler(0.0, 0.0, 0.0),
        time=0.0,
    )
    assert manager.spawn_initial_intruders(ownship) == 3
    assert len(manager.intruders) == 3
