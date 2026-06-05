"""Map training-policy actions to simulator control inputs."""

from __future__ import annotations

import numpy as np

from ..aircraft import AircraftConfig
from ..state import ControlInputs

# Training order: [throttle, aileron, elevator, rudder] in [-1, 1]
THROTTLE_IDX = 0
AILERON_IDX = 1
ELEVATOR_IDX = 2
RUDDER_IDX = 3


def training_action_to_controls(
    action: np.ndarray,
    aircraft: AircraftConfig,
    *,
    throttle_mode: str = 'symmetric',
) -> ControlInputs:
    """
    Convert tanh policy output to UAS sim ControlInputs.

    Training physics applies normalized [-1, 1] surface commands and clamps
    throttle to [0, 1]. We map throttle with symmetric (action+1)/2 by default.
    """
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    if action.shape[0] < 4:
        raise ValueError(f'Expected 4 action dims, got {action.shape}')

    if throttle_mode == 'clamp':
        throttle = float(np.clip(action[THROTTLE_IDX], 0.0, 1.0))
    else:
        throttle = float(np.clip((action[THROTTLE_IDX] + 1.0) / 2.0, 0.0, 1.0))

    return ControlInputs(
        elevator=float(np.clip(
            action[ELEVATOR_IDX] * aircraft.max_elevator,
            -aircraft.max_elevator,
            aircraft.max_elevator,
        )),
        aileron=float(np.clip(
            action[AILERON_IDX] * aircraft.max_aileron,
            -aircraft.max_aileron,
            aircraft.max_aileron,
        )),
        rudder=float(np.clip(
            action[RUDDER_IDX] * aircraft.max_rudder,
            -aircraft.max_rudder,
            aircraft.max_rudder,
        )),
        throttle=throttle,
    )
