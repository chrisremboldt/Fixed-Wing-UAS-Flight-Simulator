"""
Training-matched forward camera renderer (scratch_built_daa parity).

Replicates GPURenderer sky/horizon logic and mesh projection on CPU.
Optional nvdiffrast backend via gpu_rendering module when CUDA is available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..dynamics import FlightDynamics
from ..frames import Quaternion
from ..intruders import IntruderManager
from .camera_math import clip_to_screen, model_matrix, projection_matrix, view_matrix
from .coords import ned_position_to_training, ned_quaternion_to_training
from .mesh_data import AIRCRAFT_FACES, AIRCRAFT_VERTICES


@dataclass
class TrainingRenderConfig:
    img_size: int = 128
    fov_deg: float = 90.0
    near: float = 0.1
    far: float = 10000.0
    time_of_day: float = 0.5
    weather_fog: float = 0.0
    intruder_color: tuple[float, float, float] = (0.6, 0.6, 0.6)


class TrainingPixelRenderer:
    """CPU renderer aligned with scratch_built_daa/env/rendering.py."""

    def __init__(self, config: Optional[TrainingRenderConfig] = None):
        self.config = config or TrainingRenderConfig()
        self.h = self.w = self.config.img_size
        self._proj = projection_matrix(
            self.config.fov_deg, 1.0, self.config.near, self.config.far,
        )
        self._vertices = AIRCRAFT_VERTICES
        self._faces = AIRCRAFT_FACES

    def render(
        self,
        dynamics: FlightDynamics,
        intruder_manager: Optional[IntruderManager],
    ) -> np.ndarray:
        ego_pos = ned_position_to_training(dynamics.state.position)
        ego_quat = ned_quaternion_to_training(dynamics.state.quaternion)

        bg = self._render_background(ego_quat)
        if intruder_manager is None or not intruder_manager.intruders:
            return (np.clip(bg * 255.0, 0, 255)).astype(np.uint8)

        view = view_matrix(ego_pos, ego_quat)
        mvp = self._proj @ view

        for intruder in intruder_manager.intruders:
            intr_pos = ned_position_to_training(intruder.dynamics.state.position)
            intr_quat = ned_quaternion_to_training(intruder.dynamics.state.quaternion)
            self._render_intruder_mesh(bg, mvp, intr_pos, intr_quat)

        return (np.clip(bg * 255.0, 0, 255)).astype(np.uint8)

    def _render_background(self, ego_quat_wxyz: np.ndarray) -> np.ndarray:
        h, w = self.h, self.w
        y_coords = np.linspace(1.0, -1.0, h, dtype=np.float64)

        wq, _, _, zq = ego_quat_wxyz
        pitch = 2.0 * math.atan2(float(zq), float(wq))
        horizon_y = math.sin(pitch) * 0.5

        sky_top = np.array([0.4, 0.6, 1.0], dtype=np.float64)
        sky_horizon = np.array([0.7, 0.8, 1.0], dtype=np.float64)
        ground = np.array([0.3, 0.5, 0.2], dtype=np.float64)
        sky_full = sky_horizon * 0.8

        image = np.zeros((h, w, 3), dtype=np.float64)
        for row in range(h):
            y = y_coords[row]
            if y > horizon_y:
                image[row, :, :] = sky_full
            else:
                image[row, :, :] = ground

        brightness = 0.3 + 0.7 * self.config.time_of_day
        image *= brightness

        if self.config.weather_fog > 0.0:
            fog_color = np.array([0.7, 0.7, 0.7], dtype=np.float64)
            fog = self.config.weather_fog
            image = image * (1.0 - fog) + fog_color * fog

        return np.clip(image, 0.0, 1.0)

    def _render_intruder_mesh(
        self,
        image: np.ndarray,
        mvp: np.ndarray,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
    ) -> None:
        model = model_matrix(position, quaternion_wxyz)
        verts_h = np.concatenate(
            [self._vertices, np.ones((self._vertices.shape[0], 1), dtype=np.float32)],
            axis=1,
        ).astype(np.float64)

        world = (model @ verts_h.T).T
        clip = (mvp @ world.T).T
        screen, valid_verts = clip_to_screen(clip, self.h)

        color = np.array(self.config.intruder_color, dtype=np.float64)
        depth = clip[:, 2] / np.maximum(clip[:, 3], 1e-6)

        for face in self._faces:
            i0, i1, i2 = int(face[0]), int(face[1]), int(face[2])
            if not (valid_verts[i0] and valid_verts[i1] and valid_verts[i2]):
                continue
            self._fill_triangle(
                image,
                screen[i0], screen[i1], screen[i2],
                depth[i0], depth[i1], depth[i2],
                color,
            )

    def _fill_triangle(
        self,
        image: np.ndarray,
        p0: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray,
        z0: float,
        z1: float,
        z2: float,
        color: np.ndarray,
    ) -> None:
        pts = np.stack([p0, p1, p2], axis=0).astype(np.float64)
        min_x = int(max(0, np.floor(pts[:, 0].min())))
        max_x = int(min(self.w - 1, np.ceil(pts[:, 0].max())))
        min_y = int(max(0, np.floor(pts[:, 1].min())))
        max_y = int(min(self.h - 1, np.ceil(pts[:, 1].max())))

        if min_x > max_x or min_y > max_y:
            return

        v0, v1, v2 = pts[0], pts[1], pts[2]
        area = (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1])
        if abs(area) < 1e-6:
            return

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                px, py = float(x), float(y)
                w0 = ((v1[0] - v0[0]) * (py - v0[1]) - (v1[1] - v0[1]) * (px - v0[0])) / area
                w1 = ((v2[0] - v1[0]) * (py - v1[1]) - (v2[1] - v1[1]) * (px - v1[0])) / area
                w2 = 1.0 - w0 - w1
                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                    z = w0 * z0 + w1 * z1 + w2 * z2
                    if z < 0:
                        continue
                    image[y, x, :] = color


def create_policy_renderer(
    backend: str = 'auto',
    img_size: int = 128,
    fov_deg: float = 90.0,
    device: str = 'cpu',
):
    """
    Factory for observation renderers.

    backend: auto | training | gpu | legacy
    """
    backend = backend.lower()
    if backend == 'auto':
        try:
            from .gpu_rendering import NvdiffrastPolicyRenderer
            if NvdiffrastPolicyRenderer.is_available(device):
                return NvdiffrastPolicyRenderer(img_size=img_size, fov_deg=fov_deg, device=device)
        except Exception:
            pass
        return TrainingPixelRenderer(TrainingRenderConfig(img_size=img_size, fov_deg=fov_deg))

    if backend == 'gpu':
        from .gpu_rendering import NvdiffrastPolicyRenderer
        return NvdiffrastPolicyRenderer(img_size=img_size, fov_deg=fov_deg, device=device)

    if backend == 'legacy':
        from .rendering import CPUPixelRenderer, CameraConfig
        return CPUPixelRenderer(CameraConfig(img_size=img_size, fov_deg=fov_deg))

    return TrainingPixelRenderer(TrainingRenderConfig(img_size=img_size, fov_deg=fov_deg))
