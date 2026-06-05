# RFC 001: PyTorch Policy Integration Harness

**Status:** Phase 1 implemented (2026-06-05)  
**Author:** Simulator review (2026-06-05)  
**Repo:** [chrisremboldt/Fixed-Wing-UAS-Flight-Simulator](https://github.com/chrisremboldt/Fixed-Wing-UAS-Flight-Simulator)

## Summary

Add a closed-loop path to load `final_model.pt`, build observations, run inference, and inject actions into the simulator (standalone, WebSocket, or PX4 bridge) with training-time parity.

## Motivation

- `torch>=2.0.0` is in `requirements.txt` and `final_model.pt` exists in the repo, but no code loads or runs the model.
- `PX4MavlinkBridge` already exposes `ControlInterventionPolicy`; only a geometry placeholder (`SimpleBearingAvoidancePolicy`) is wired today.
- DAA evaluation needs deterministic scenarios, documented observation/action contracts, and real-time pacing aligned with physics (`dt=0.01`).

## `final_model.pt` inspection (current checkpoint)

| Property | Value |
|----------|-------|
| Architecture | 3× CNN blocks (3→16→32→32 channels) |
| Input | RGB, ~128×128 (inferred from 8192-dim flatten) |
| Output | 4 continuous actions (`log_std` shape `(4,)`) |
| Training source | `drones/scratch_built_daa/train.py` |
| Action order | `[throttle, aileron, elevator, rudder]` in `[-1, 1]` after `tanh` |
| Normalization | `obs / 255.0` (uint8 RGB) |

## Proposed design

### 1. `ObservationBuilder`

Frozen schema exported as JSON alongside the checkpoint:

```python
@dataclass
class ObservationSpec:
    image_size: tuple[int, int]  # (H, W)
    channels: int = 3
    normalize_mean: tuple[float, ...]
    normalize_std: tuple[float, ...]
    # Optional state vector append for hybrid policies
    state_features: list[str]  # e.g. ["airspeed", "alpha", "beta"]
```

**Implemented:** `PixelObservationBuilder` + `CPUPixelRenderer` (128×128×3 uint8, 90° FOV).
Legacy bearing-map builder retained for debugging only.

### 2. `ModelPolicy` (`ControlInterventionPolicy`) — implemented

- `simulator/policy/architecture.py` — `ImpalaCNN` + `PufferPolicy` (strict checkpoint load)
- `simulator/policy/model_policy.py` — inference + absolute/blend control modes
- `simulator/policy/actions.py` — training action → `ControlInputs` mapping

### 3. Camera / render path (v1)

Phased:

1. **v0 (done):** CPU pinhole renderer — sky/ground + intruder blobs.
2. **v1:** Port nvdiffrast mesh renderer from training env (GPU).
3. **v2:** Noise, latency, resolution degradation for domain randomization parity.

### 4. Evaluation harness

```bash
python run_policy_eval.py \
  --checkpoint final_model.pt \
  --scenario configs/scenarios/head_on.yaml \
  --episodes 100 \
  --seed 42
```

Logs per episode: min separation, time-to-CPA, collision, intervention count, action saturation.

### 5. Integration points

| Entry point | Integration |
|-------------|-------------|
| `run_px4_bridge.py` | `--policy final_model.pt` → `ModelPolicy` as `intervention_policy` |
| `simulator/main.py` | `--policy` headless eval mode |
| `SimulationServer` | Optional policy loop instead of keyboard controls |

## Real-time pacing

Visualization and policy loops must advance physics at `1/dt` Hz (100 Hz default). The WebSocket server should not run physics slower than the integrator assumes.

## Open questions

1. Where does the `PolicyNetwork` class definition live (this repo vs external training repo)?
2. Exact action scaling used during RL training?
3. Should policy run on every physics step or at a lower rate with hold?
4. Reward/success metrics for automated regression gates?

## Implementation phases

| Phase | Scope | Exit criteria |
|-------|-------|---------------|
| **0** | P0 axis/viz bug fixes | Intruders visually correct; velocity frame fixed |
| **1** | `ObservationBuilder` v0 + `ModelPolicy` stub | Checkpoint loads; random actions injected without crash |
| **2** | Synthetic image observations + eval CLI | 100 seeded episodes complete; CSV log |
| **3** | Camera sim + training parity audit | Qualitative match to training env screenshots |
| **4** | CI regression | Policy eval smoke test in GitHub Actions |

## Non-goals (this RFC)

- Retraining the policy
- Full photorealistic rendering
- Replacing PX4 controller (policy is intervention layer only)
