"""Observation builders for policy inference (RFC 001 phase 1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..dynamics import FlightDynamics
from ..intruders import IntruderManager


@dataclass
class ObservationSpec:
    """Documented observation contract for policy checkpoints."""

    image_size: tuple[int, int] = (128, 128)
    channels: int = 3
    normalize_mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    normalize_std: tuple[float, float, float] = (1.0, 1.0, 1.0)


class BearingMapObservationBuilder:
    """
    v0 observation: synthetic forward-sector occupancy grid from intruder geometry.

    Unblocks inference before a full camera renderer exists.
    """

    def __init__(self, spec: Optional[ObservationSpec] = None):
        self.spec = spec or ObservationSpec()

    def build(
        self,
        dynamics: FlightDynamics,
        intruder_manager: Optional[IntruderManager],
    ) -> np.ndarray:
        h, w = self.spec.image_size
        grid = np.zeros((self.spec.channels, h, w), dtype=np.float32)

        if intruder_manager is None or not intruder_manager.intruders:
            return grid

        own = dynamics.state
        R_ned_to_body = own.quaternion.to_dcm().T
        fov_rad = np.radians(90.0)

        for intruder in intruder_manager.intruders:
            rel_ned = intruder.dynamics.state.position - own.position
            rel_body = R_ned_to_body @ rel_ned
            x_fwd, y_right = rel_body[0], rel_body[1]
            if x_fwd <= 0.0:
                continue

            az = np.arctan2(y_right, x_fwd)
            if abs(az) > fov_rad / 2:
                continue

            range_m = float(np.linalg.norm(rel_body))
            u = int(np.clip((az / (fov_rad / 2) * 0.5 + 0.5) * (w - 1), 0, w - 1))
            v = int(np.clip((1.0 - min(range_m, 2000.0) / 2000.0) * (h - 1), 0, h - 1))
            grid[0, v, u] = max(grid[0, v, u], 1.0 - range_m / 2000.0)

        return grid
