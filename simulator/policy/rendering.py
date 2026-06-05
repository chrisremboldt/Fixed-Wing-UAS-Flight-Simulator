"""
CPU forward-camera renderer for DAA policy observations.

Produces 128x128x3 uint8 RGB images aligned with scratch_built_daa training:
- 90° horizontal FOV nose camera
- Sky/ground horizon from aircraft pitch
- Intruders drawn as aircraft silhouettes in the image plane
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..dynamics import FlightDynamics
from ..intruders import IntruderManager


@dataclass
class CameraConfig:
    img_size: int = 128
    fov_deg: float = 90.0
    max_range_m: float = 5000.0


class CPUPixelRenderer:
    """Lightweight CPU renderer — no GPU / nvdiffrast dependency."""

    def __init__(self, config: Optional[CameraConfig] = None):
        self.config = config or CameraConfig()
        self.h = self.w = self.config.img_size
        self.fov_rad = math.radians(self.config.fov_deg)

    def render(
        self,
        dynamics: FlightDynamics,
        intruder_manager: Optional[IntruderManager],
    ) -> np.ndarray:
        """
        Returns:
            (H, W, 3) uint8 RGB observation matching training layout.
        """
        h, w = self.h, self.w
        image = self._render_background(dynamics)

        if intruder_manager is not None:
            own = dynamics.state
            R_body = own.quaternion.to_dcm().T

            for intruder in intruder_manager.intruders:
                rel_ned = intruder.dynamics.state.position - own.position
                rel_body = R_body @ rel_ned
                x_fwd, y_right, z_down = rel_body
                if x_fwd <= 1.0:
                    continue

                az = math.atan2(y_right, x_fwd)
                if abs(az) > self.fov_rad / 2:
                    continue

                el = math.atan2(-z_down, x_fwd)
                vert_fov = self.fov_rad
                if abs(el) > vert_fov / 2:
                    continue

                u = int(np.clip((az / (self.fov_rad / 2) * 0.5 + 0.5) * (w - 1), 0, w - 1))
                v = int(np.clip((0.5 - el / vert_fov) * (h - 1), 0, h - 1))

                range_m = float(np.linalg.norm(rel_body))
                scale = int(np.clip(400.0 / max(range_m, 80.0), 4, 24))

                self._draw_aircraft_blob(image, u, v, scale)

        return image

    def _render_background(self, dynamics: FlightDynamics) -> np.ndarray:
        h, w = self.h, self.w
        image = np.zeros((h, w, 3), dtype=np.uint8)

        sky_top = np.array([102, 156, 255], dtype=np.float32)
        sky_horizon = np.array([179, 204, 255], dtype=np.float32)
        ground = np.array([77, 128, 51], dtype=np.float32)

        euler = dynamics.state.euler_angles
        pitch = float(euler[1])
        horizon_v = int(np.clip((0.5 - pitch / self.fov_rad) * (h - 1), 0, h - 1))

        for row in range(h):
            if row < horizon_v:
                t = row / max(horizon_v, 1)
                color = sky_top * (1 - t) + sky_horizon * t
            else:
                color = ground
            image[row, :, :] = np.clip(color, 0, 255).astype(np.uint8)

        return image

    def _draw_aircraft_blob(self, image: np.ndarray, cx: int, cy: int, size: int) -> None:
        h, w = image.shape[:2]
        gray = np.array([153, 153, 153], dtype=np.uint8)
        wing_half = size
        fuselage_half = max(2, size // 3)

        for dy in range(-fuselage_half, fuselage_half + 1):
            row = cy + dy
            if 0 <= row < h:
                c0 = max(0, cx - wing_half)
                c1 = min(w, cx + wing_half + 1)
                image[row, c0:c1] = gray

        nose_row = cy - fuselage_half - 1
        if 0 <= nose_row < h and 0 <= cx < w:
            image[nose_row, cx] = np.array([200, 200, 200], dtype=np.uint8)


def batch_obs_from_renderer(
    renderer: CPUPixelRenderer,
    dynamics: FlightDynamics,
    intruder_manager: Optional[IntruderManager],
) -> np.ndarray:
    """Single-env helper returning HWC uint8."""
    return renderer.render(dynamics, intruder_manager)
