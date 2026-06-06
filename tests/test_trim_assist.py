"""Tests for trim-assisted policy blending."""

import numpy as np

from simulator.aircraft import AircraftConfig
from simulator.dynamics import FlightDynamics, SimulationConfig
from simulator.environment import Environment
from simulator.intruders import IntruderManager, IntruderConfig
from unittest.mock import MagicMock

from simulator.policy.trim_assist import (
    TrimAssistConfig,
    TrimAssistedModelPolicy,
    blend_trim_and_policy,
    compute_threat_authority,
)
from simulator.frames import Quaternion
from simulator.state import AircraftState, ControlInputs
from simulator.trim import TrimCondition, compute_trim


def test_authority_zero_without_intruders():
    aircraft = AircraftConfig()
    env = Environment()
    dyn = FlightDynamics(aircraft, env, SimulationConfig(dt=0.01))
    trim = compute_trim(TrimCondition(airspeed=25.0, altitude=100.0), aircraft, env)
    dyn.reset(trim.state if trim.success else None)

    auth = compute_threat_authority(dyn, None, TrimAssistConfig())
    assert auth == 0.0


def test_blend_holds_trim_throttle():
    trim = ControlInputs(elevator=-0.05, aileron=0.0, rudder=0.0, throttle=0.24)
    policy = ControlInputs(elevator=0.2, aileron=0.3, rudder=-0.1, throttle=0.05)
    blended = blend_trim_and_policy(trim, policy, authority=0.5, hold_trim_throttle=True)
    assert blended.throttle == trim.throttle
    assert blended.aileron == 0.15


def test_recovery_baseline_differs_from_static_trim_when_disturbed():
    aircraft = AircraftConfig()
    env = Environment()
    trim = compute_trim(TrimCondition(airspeed=40.0, altitude=1000.0), aircraft, env)
    assert trim.success

    assisted = TrimAssistedModelPolicy(
        model_policy=MagicMock(),
        trim_controls=trim.controls,
        recovery_state=trim.state,
    )
    pitched = AircraftState(
        position=trim.state.position.copy(),
        velocity_body=trim.state.velocity_body.copy(),
        quaternion=Quaternion.from_euler(0.08, 0.12, trim.state.psi),
        omega_body=np.array([0.0, 0.08, 0.0]),
        time=0.0,
    )

    recovery_controls = assisted._baseline_controls(
        FlightDynamics(aircraft, env, SimulationConfig(dt=0.02)),
    )
    assisted.recovery.reset()
    recovery_controls = assisted.recovery.update(pitched)

    assert recovery_controls.elevator != trim.controls.elevator
    assert recovery_controls.aileron != trim.controls.aileron


def test_trim_assisted_survives_short_run():
    pytest = __import__('pytest')
    checkpoint = __import__('pathlib').Path(__file__).resolve().parent.parent / 'final_model.pt'
    if not checkpoint.exists():
        pytest.skip('final_model.pt not present')

    from simulator.policy import TrimAssistedModelPolicy

    aircraft = AircraftConfig()
    env = Environment()
    dt = 0.01
    dyn = FlightDynamics(aircraft, env, SimulationConfig(dt=dt))
    trim = compute_trim(TrimCondition(airspeed=25.0, altitude=100.0), aircraft, env)
    dyn.reset(trim.state if trim.success else None)

    policy = TrimAssistedModelPolicy.from_checkpoint(
        str(checkpoint),
        trim.controls,
        recovery_state=trim.state,
    )
    mgr = IntruderManager(IntruderConfig(spawn_rate=0.3, max_intruders=3), env)

    for _ in range(2000):
        if mgr.should_spawn_intruder(dt, dyn.state):
            mgr.spawn_intruder(dyn.state)
        dyn.step(policy.compute_controls(dyn, mgr))
        mgr.update_intruders(dt, dyn.state)
        if dyn.crash_state.crashed:
            break

    assert dyn.state.time >= 8.0
