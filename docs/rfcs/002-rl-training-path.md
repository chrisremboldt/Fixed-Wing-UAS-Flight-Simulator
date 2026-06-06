# RFC 002: RL Training / Fine-Tune Path in UAS Sim

**Status:** Spike implemented (2026-06-06)  
**Related:** RFC 001 (#5), issue #14

## Summary

Training lives in `drones/scratch_built_daa`; this repo is the eval/SITL harness.
Sim-to-sim gap closes via training-fidelity physics, pixel renderer parity, and
optional fine-tuning against UAS sim dynamics.

## Chosen path (phase 1)

| Phase | Approach | Status |
|-------|----------|--------|
| 1 | Train in scratch_built_daa, eval here | Done |
| 2 | `UASDAAEnv` gym wrapper + obs export | Spike (`simulator/gym_env.py`) |
| 3 | PufferLib / gymnasium fine-tune on UAS physics | Future |

## `UASDAAEnv` interface

```python
from simulator.gym_env import UASDAAEnv, UASDAAEnvConfig

env = UASDAAEnv(config=UASDAAEnvConfig(policy_path='final_model.pt'))
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(action)
```

- **Observation:** `(128, 128, 3)` uint8 RGB from `TrainingPixelRenderer`
- **Action:** `[throttle, aileron, elevator, rudder]` in `[-1, 1]`
- **Physics:** `TrainingFidelityConfig` when `training_fidelity=True`

## Eval gap metrics

Track between native `evaluate_model.py` and `run_policy_eval.py`:

- Success rate per scenario + seed
- Min separation / time-to-CPA
- Action saturation rate
- Failure reason breakdown (JSON export)

## Non-goals

- Full distributed RL training in this repo (stay in scratch_built_daa for now)
- Replacing Warp physics with 6-DOF aero in training loop
