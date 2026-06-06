# PX4 / MAVLink SITL Integration

This simulator now includes a MAVLink bridge so PX4 SITL can treat it as an external
physics backend.

## What Is Implemented

- PX4 actuator ingestion via `HIL_ACTUATOR_CONTROLS`
- Sensor/state publishing via:
  - `HIL_SENSOR`
  - `HIL_GPS`
  - `HIL_STATE_QUATERNION`
- Optional intruder-based `OBSTACLE_DISTANCE` publishing

Bridge module:

- `/Users/remboldt/Code/Special Output/fixed_wing_UAS_sim/simulator/px4_bridge.py`
- `/Users/remboldt/Code/Special Output/fixed_wing_UAS_sim/run_px4_bridge.py`

## Install

```bash
cd "/Users/remboldt/Code/Special Output/fixed_wing_UAS_sim"
pip install -r requirements.txt
```

## Run

1. Start PX4 SITL configured to use simulator MAVLink on TCP `4560`.
2. In this repo, run:

```bash
python run_px4_bridge.py --connection tcp:127.0.0.1:4560
```

Optional:

```bash
python run_px4_bridge.py \
  --aircraft configs/generic_uav.yaml \
  --home-lat 36.1627 \
  --home-lon -86.7816 \
  --home-alt 180 \
  --enable-intruders \
  --intruder-rate 0.2 \
  --simple-daa
```

## Actuator Mapping Notes

Default channel mapping:

- channel 0 -> aileron
- channel 1 -> elevator
- channel 2 -> rudder
- channel 3 -> throttle

You can invert surfaces if needed:

```bash
python run_px4_bridge.py --reverse-aileron --reverse-elevator
```

Throttle is assumed bipolar (`[-1, 1]`) by default and remapped to `[0, 1]`.
If your PX4 output is already `[0, 1]`, use:

```bash
python run_px4_bridge.py --throttle-unipolar
```

## Policy inference (RFC 001)

Load a PyTorch checkpoint as a control intervention layer:

```bash
python run_px4_bridge.py \
  --connection tcp:127.0.0.1:4560 \
  --enable-intruders \
  --policy final_model.pt \
  --training-fidelity
```

Training-fidelity mode sets `dt=0.02` (50 Hz) to match the RL training loop.

### Actuator mapping vs training

Training action order: `[throttle, aileron, elevator, rudder]` in `[-1, 1]` after `tanh`.

| Training channel | PX4 HIL channel | Bridge mapping |
|------------------|-----------------|--------------|
| throttle | 3 | bipolar `[-1,1]` → `[0,1]` (use `--throttle-unipolar` if PX4 sends `[0,1]`) |
| aileron | 0 | direct (use `--reverse-aileron` if inverted) |
| elevator | 1 | direct (use `--reverse-elevator` if inverted) |
| rudder | 2 | direct (use `--reverse-rudder` if inverted) |

### SITL + policy test procedure

1. Start PX4 SITL with MAVLink TCP on port 4560.
2. Run the bridge with training fidelity and intruders:

```bash
python run_px4_bridge.py \
  --connection tcp:127.0.0.1:4560 \
  --enable-intruders \
  --intruder-rate 0.2 \
  --policy final_model.pt \
  --training-fidelity \
  --policy-device cpu
```

3. Arm and take off in QGC; verify bridge logs show 50 Hz loop.
4. Confirm aileron/elevator/rudder signs match expected DAA maneuvers (no silent inversion).

Uses CPU pixel observations (128×128 RGB) and trim-assisted control mapping. See `docs/rfcs/001-pytorch-policy-integration.md`.

## Interactive visualization with policy

```bash
python -m simulator.main --policy final_model.pt --training-fidelity
```

Opens http://localhost:8080 with policy telemetry overlay (min separation, authority, obs thumbnail).

## Current Limitations

- `--simple-daa` is geometry-based (no camera model)
- No camera/image transport over MAVLink yet
- No direct PX4 parameter auto-configuration

The bridge is designed to be the base for those additions.
