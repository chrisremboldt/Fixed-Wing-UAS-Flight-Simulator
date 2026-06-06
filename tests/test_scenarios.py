"""Tests for scenario YAMLs and training-fidelity physics."""

from pathlib import Path

import pytest

from simulator.main import create_default_aircraft
from simulator.policy.evaluation import EvalConfig, export_batch_results, run_episode
from simulator.policy.scenarios import ScenarioConfig, resolve_scenario_path, spawn_fixed_intruders
from simulator.policy.training_config import TrainingFidelityConfig
from simulator.dynamics import SimulationConfig

CHECKPOINT = 'final_model.pt'
SCENARIO_DIR = Path('configs/scenarios')


@pytest.mark.skipif(not Path(CHECKPOINT).exists(), reason='final_model.pt not present')
def test_full_policy_training_fidelity_completes_20s():
    """Issue #6: full-policy eval should survive 20s without Vne crash."""
    aircraft = create_default_aircraft()
    config = EvalConfig.from_training_defaults(
        CHECKPOINT,
        duration=20.0,
        seed=42,
        full_policy=True,
    )
    result = run_episode(aircraft, config, seed=42)
    assert result.sim_time >= 19.0
    assert result.failure_reason != 'overspeed'
    assert result.success or result.failure_reason in ('nmac', 'ground', 'stall', 'structural')


def test_training_fidelity_disables_overspeed_crash():
    sim = TrainingFidelityConfig.defaults().simulation_config()
    assert sim.disable_overspeed_crash is True
    assert sim.disable_structural_g_crash is True


def test_scenario_yaml_loads():
    path = resolve_scenario_path('head_on')
    scenario = ScenarioConfig.from_yaml(path)
    assert scenario.name == 'head_on'
    assert len(scenario.fixed_intruders) == 1
    assert scenario.fixed_intruders[0].heading_offset_deg == 180.0


def test_scenario_deterministic_spawn():
    aircraft = create_default_aircraft()
    config = EvalConfig.from_training_defaults(
        CHECKPOINT,
        duration=2.0,
        seed=99,
        scenario=ScenarioConfig.from_yaml(SCENARIO_DIR / 'head_on.yaml'),
    )
    r1 = run_episode(aircraft, config, seed=99)
    r2 = run_episode(aircraft, config, seed=99)
    assert r1.intruders_spawned == r2.intruders_spawned == 1
    assert r1.min_separation_m == pytest.approx(r2.min_separation_m, rel=1e-6)


def test_export_batch_results(tmp_path):
    from simulator.policy.evaluation import BatchEvalResult, EpisodeResult

    batch = BatchEvalResult(
        episodes=[
            EpisodeResult(
                seed=1,
                sim_time=20.0,
                min_separation_m=300.0,
                final_altitude_m=1000.0,
                final_airspeed_mps=40.0,
                intruders_spawned=1,
                nmac_proximity_steps=0,
                crashed=False,
                ground_collision=False,
                nmac_violation=False,
                timeout=True,
                failure_reason='success',
            )
        ],
        scenario_name='head_on',
    )
    out = tmp_path / 'results.json'
    export_batch_results(batch, out)
    assert out.exists()
    assert 'head_on' in out.read_text()
