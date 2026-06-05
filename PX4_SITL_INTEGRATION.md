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

## Policy inference (RFC 001 phase 1)

Load a PyTorch checkpoint as a control intervention layer:

```bash
python run_px4_bridge.py \
  --connection tcp:127.0.0.1:4560 \
  --enable-intruders \
  --policy final_model.pt
```

Uses synthetic bearing-map observations until a camera renderer lands. Action scaling is provisional — see `docs/rfcs/001-pytorch-policy-integration.md`.

## Current Limitations

- `--simple-daa` is geometry-based (no camera model)
- No camera/image transport bridge yet (next integration step)
- Policy action scaling not yet validated against training code
- No direct PX4 parameter auto-configuration

The bridge is designed to be the base for those additions.
