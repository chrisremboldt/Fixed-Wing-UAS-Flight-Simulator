"""
Vendored nvdiffrast GPU renderer (subset of scratch_built_daa GPURenderer).

Single-batch rendering for policy eval; no EdgeTAM dependency.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .mesh_data import AIRCRAFT_FACES, AIRCRAFT_VERTICES


class VendoredGPURenderer:
    """Batched GPU rasterizer using nvdiffrast and vendored mesh data."""

    def __init__(
        self,
        img_size: int = 128,
        device: str = 'cuda:0',
        fov_deg: float = 90.0,
    ):
        import nvdiffrast.torch as dr

        self.dr = dr
        self.img_size = img_size
        self.device = device
        self.fov = fov_deg
        self.near = 0.1
        self.far = 10000.0

        self.glctx = dr.RasterizeGLContext(device=device)
        self.intruder_vertices = torch.tensor(AIRCRAFT_VERTICES, dtype=torch.float32, device=device)
        self.intruder_faces = torch.tensor(AIRCRAFT_FACES, dtype=torch.int32, device=device)

        self.sky_color_top = torch.tensor([0.4, 0.6, 1.0], device=device)
        self.sky_color_horizon = torch.tensor([0.7, 0.8, 1.0], device=device)
        self.ground_color = torch.tensor([0.3, 0.5, 0.2], device=device)

    def _compute_projection_matrix(self, fov_deg: float, aspect: float, near: float, far: float):
        fov_rad = math.radians(fov_deg)
        f = 1.0 / math.tan(fov_rad / 2.0)

        proj = torch.zeros((4, 4), device=self.device)
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = (2.0 * far * near) / (near - far)
        proj[3, 2] = -1.0
        return proj

    def _compute_view_matrix(self, position: torch.Tensor, quaternion: torch.Tensor):
        w, x, y, z = quaternion[0], quaternion[1], quaternion[2], quaternion[3]

        R = torch.zeros((3, 3), device=self.device)
        R[0, 0] = 1 - 2 * y * y - 2 * z * z
        R[0, 1] = 2 * x * y - 2 * w * z
        R[0, 2] = 2 * x * z + 2 * w * y
        R[1, 0] = 2 * x * y + 2 * w * z
        R[1, 1] = 1 - 2 * x * x - 2 * z * z
        R[1, 2] = 2 * y * z - 2 * w * x
        R[2, 0] = 2 * x * z - 2 * w * y
        R[2, 1] = 2 * y * z + 2 * w * x
        R[2, 2] = 1 - 2 * x * x - 2 * y * y

        camera_rot = torch.tensor(
            [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
            dtype=torch.float32,
            device=self.device,
        )
        R_cam = camera_rot @ R

        view = torch.eye(4, device=self.device)
        view[:3, :3] = R_cam.T
        view[:3, 3] = -R_cam.T @ position
        return view

    def render_batch(
        self,
        ego_positions: torch.Tensor,
        ego_quaternions: torch.Tensor,
        intruder_positions: torch.Tensor,
        intruder_quaternions: torch.Tensor,
        time_of_day: float = 0.5,
        weather_fog: float = 0.0,
    ) -> torch.Tensor:
        batch_size = ego_positions.shape[0]
        num_intruders = intruder_positions.shape[1]
        proj = self._compute_projection_matrix(self.fov, 1.0, self.near, self.far)

        images = []
        for env_idx in range(batch_size):
            view = self._compute_view_matrix(
                ego_positions[env_idx],
                ego_quaternions[env_idx],
            )
            mvp = proj @ view
            bg_image = self._render_background(ego_quaternions[env_idx], time_of_day, weather_fog)

            for i in range(num_intruders):
                intruder_img = self._render_intruder(
                    mvp,
                    intruder_positions[env_idx, i],
                    intruder_quaternions[env_idx, i],
                )
                bg_image = torch.where(intruder_img[..., 3:4] > 0.5, intruder_img[..., :3], bg_image)

            images.append(bg_image)

        return torch.stack(images, dim=0)

    def _render_background(self, ego_quat: torch.Tensor, time_of_day: float, fog: float):
        h, w = self.img_size, self.img_size
        y_coords = torch.linspace(1, -1, h, device=self.device)

        pitch = 2.0 * torch.atan2(ego_quat[2], ego_quat[0])
        horizon_y = torch.sin(pitch) * 0.5

        y_coords_2d = y_coords.view(-1, 1).expand(h, w)
        is_sky = (y_coords_2d > horizon_y).float()

        ground_color = self.ground_color.view(1, 1, 3).expand(h, w, 3)
        sky_color_full = self.sky_color_horizon.view(1, 1, 3).expand(h, w, 3) * 0.8

        is_sky_3d = is_sky.unsqueeze(-1).expand(h, w, 3)
        image = is_sky_3d * sky_color_full + (1 - is_sky_3d) * ground_color

        brightness = 0.3 + 0.7 * time_of_day
        image = image * brightness

        fog_color = torch.tensor([0.7, 0.7, 0.7], device=self.device)
        image = image * (1 - fog) + fog_color * fog
        return torch.clamp(image, 0, 1)

    def _render_intruder(self, mvp: torch.Tensor, position: torch.Tensor, quaternion: torch.Tensor):
        w, x, y, z = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
        R = torch.zeros((3, 3), device=self.device)
        R[0, 0] = 1 - 2 * y * y - 2 * z * z
        R[0, 1] = 2 * x * y - 2 * w * z
        R[0, 2] = 2 * x * z + 2 * w * y
        R[1, 0] = 2 * x * y + 2 * w * z
        R[1, 1] = 1 - 2 * x * x - 2 * z * z
        R[1, 2] = 2 * y * z - 2 * w * x
        R[2, 0] = 2 * x * z - 2 * w * y
        R[2, 1] = 2 * y * z + 2 * w * x
        R[2, 2] = 1 - 2 * x * x - 2 * y * y

        model = torch.eye(4, device=self.device)
        model[:3, :3] = R
        model[:3, 3] = position

        vertices_homo = torch.cat([
            self.intruder_vertices,
            torch.ones((self.intruder_vertices.shape[0], 1), device=self.device),
        ], dim=1)

        vertices_world = (model @ vertices_homo.T).T
        vertices_clip = (mvp @ vertices_world.T).T
        vertices_clip = vertices_clip.unsqueeze(0).contiguous()
        faces = self.intruder_faces.contiguous()

        rast_out, _ = self.dr.rasterize(
            self.glctx,
            vertices_clip,
            faces,
            resolution=[self.img_size, self.img_size],
        )

        color = torch.tensor([0.6, 0.6, 0.6, 1.0], device=self.device)
        color = color.view(1, 1, 1, 4).expand(1, self.img_size, self.img_size, 4)
        mask = (rast_out[..., 3:4] > 0).float()
        image = color * mask
        return image.squeeze(0)
