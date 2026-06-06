# Design: Sim-backed expert and label schema

**RFC:** [`docs/rfcs/003-lightweight-visual-daa.md`](../docs/rfcs/003-lightweight-visual-daa.md)

Companion to [README.md](./README.md). Defines the geometric expert, optical threat features, and training labels.

## Coordinate frames

- **NED**: sim native; forward = North component of body x.
- **Body**: x forward, y right, z down.
- **Camera**: x right in image, y down in image; optical axis ≈ body +x.
- **Image**: `(u, v)` pixels, origin top-left; `(cx, cy)` = principal point.

The parent renderer uses pinhole projection via `camera_math.py` (`projection_matrix`, `view_matrix`, `clip_to_screen`).

## Ground-truth optical flow (sim)

For intruder point `P(t)` in camera frame, projected to `(u, v)`:

```
u = fx * X/Z + cx
v = fy * Y/Z + cy
```

Differentiate (finite difference over `dt`):

```
du/dt, dv/dt
```

Log per intruder **centroid** flow and optionally per-pixel flow inside projected mask.

## Threat features (per intruder)

Given range `R = |P|`, subtended angle `θ ≈ 2 arctan(w_obj / (2R))` or blob radius in pixels `r`:

| Feature | Formula / proxy | Meaning |
|---------|-----------------|--------|
| Looming | `(dr/dt) / max(r, ε)` or `−(dR/dt)/R` | Closing |
| Bearing offset | `(u−cx, v−cy)` normalized by FOV | Off boresight |
| Convergence | `(u−cx)*(du/dt) + (v−cy)*(dv/dt) < 0` | Moving toward center |
| Visibility | in FOV, `Z > 0`, `r ≥ r_min` | Detectability |

**Composite threat** (tunable weights):

```
S = α * loom+ + β * conv+ + γ * exp(−||(u,v)−(cx,cy)|| / σ)
```

`loom+ = max(0, loom)`, `conv+ = max(0, −sign(convergence) * ||(du,dv)||)`.

## Expert escape rule (fixed-wing)

**Goal**: increase image-plane separation—push threat centroid toward nearest FOV edge.

1. Select intruder with max `S` (or merge nearby blobs).
2. Desired lateral escape sign:
   - `sign_aileron = sign(u − cx)` (turn so blob moves toward edge; refine with predicted `(du/dv)` under candidate turn).
3. Clamp to aircraft `max_bank`, rate limits from autopilot.
4. If `S < S_min`: hold wings-level / return trim.

**Coordinated turn**: when applying aileron, add proportional elevator/rudder from trim tables (reuse `simulator/trim.py` patterns).

This expert is **myopic** but generates consistent **(image → escape)** pairs.

## Label schema (per frame)

```yaml
frame_id: int
time_s: float
image: uint8[H, W, 1 or 3]
threat_mask: float32[H, W]      # soft 0..1, rasterized projected intruder + S threshold
optical_flow: float32[H, W, 2]  # optional; background subtracted in post
meta:
  intruders:
    - id: int
      u: float
      v: float
      r_px: float
      R_m: float
      du_dt: float
      dv_dt: float
      S: float
expert_action:
  aileron: float   # [-1, 1]
  elevator: float
  rudder: float
  throttle: float
escape_hint:
  turn_sign: int    # -1 | 0 | +1
  urgency: float    # 0..1
```

Storage: `.npz` shards or LMDB for training; keep PNG+JSON for debug.

## Student model heads (Tier 1)

**Option A — Heatmap only**

- Output: `H×W` sigmoid mask.
- Deploy: centroid + rules → turn (expert logic on device).

**Option B — Heatmap + escape**

- Output: mask + `urgency` + `turn_sign` (classification) or `(Δu_desired, Δv_desired)` in image space.

**Option C — Flow residual**

- Predict `(du, dv)` residual after global background flow; cluster residuals → threat.

Prefer **A or B** for first train; smallest deploy footprint.

## Losses

```
L = λ_mask * Dice(pred_mask, gt_mask)
  + λ_turn * CE(turn_sign_pred, turn_sign_gt)
  + λ_urg * MSE(urgency_pred, urgency_gt)
```

Hard-negative mining: frames with **empty mask** (clear sky) to reduce false swerves.

## Domain randomization (minimum set)

- `time_of_day`, `weather_fog` (existing)
- Gaussian noise, JPEG artifacts
- Random exposure / gamma
- Intruder color/contrast vs. sky
- Spawn rate and max count
- Small camera pitch offset (mount misalignment)

## Integration with parent sim

```python
# Pseudocode — future expert_policy.py
from simulator.dynamics import FlightDynamics
from simulator.intruders import IntruderManager
from simulator.policy.training_render import TrainingPixelRenderer

renderer = TrainingPixelRenderer()
frame = renderer.render(dynamics, intruders)
labels = expert.compute_labels(dynamics, intruders, camera_config)
action = expert.compute_action(labels)
```

Deploy path mirrors `ModelPolicy` / `ControlInterventionPolicy`: replace ImpalaCNN with student, keep action mapping in `simulator/policy/actions.py`.

## Open design choices

1. **Horizon ROI**: crop band around horizon (±15°) to ignore most ground texture for cruise DAA.
2. **Temporal stack**: single frame vs. `(t, t−1)` two-frame input for flow-free looming.
3. **Multi-threat**: max-S vs. sum of top-k masks.
4. **Autopilot coupling**: absolute surface commands vs. offset from trim (prefer offset for BVLOS).

Resolve with ablations on `head_on` + `crossing` scenarios before broad training.
