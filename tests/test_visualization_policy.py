"""Tests for interactive WebSocket policy loop (issue #11)."""

import json

from simulator.main import create_default_aircraft
from simulator.policy.interactive_policy import (
    InteractivePolicyConfig,
    build_interactive_policy_setup,
)
from simulator.visualization import SimulationServer, state_to_message


def test_state_message_includes_policy_telemetry():
    aircraft = create_default_aircraft()
    config = InteractivePolicyConfig(
        policy_path='final_model.pt',
        training_fidelity=True,
        full_policy=False,
    )
    dynamics, intruder_manager, _controller, controls_provider = build_interactive_policy_setup(
        aircraft,
        config,
        seed=42,
        enable_intruders=True,
    )

    controls, telemetry = controls_provider(dynamics, intruder_manager)
    dynamics.step(controls)

    msg = json.loads(state_to_message(
        dynamics.state,
        dynamics.forces_moments,
        dynamics.crash_state,
        intruder_manager,
        telemetry,
    ))

    assert 'policy' in msg
    assert msg['policy']['active'] is True
    assert msg['policy']['mode'] == 'trim_assisted'
    assert 'min_separation_m' in msg['policy']
    assert 'authority' in msg['policy']


def test_simulation_server_accepts_controls_provider():
    aircraft = create_default_aircraft()
    config = InteractivePolicyConfig(policy_path='final_model.pt', training_fidelity=True)
    dynamics, intruder_manager, _controller, controls_provider = build_interactive_policy_setup(
        aircraft,
        config,
        seed=1,
        enable_intruders=False,
    )

    server = SimulationServer(dynamics, intruder_manager, update_rate=50.0)
    server.controls_provider = controls_provider

    controls, telemetry = server.controls_provider(dynamics, intruder_manager)
    assert controls.throttle >= 0.0
    assert telemetry['active'] is True
