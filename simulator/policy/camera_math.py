"""Camera matrices matching scratch_built_daa GPURenderer."""

from __future__ import annotations

import math

import numpy as np


def quat_to_rotation_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in quat_wxyz.reshape(4)]
    R = np.zeros((3, 3), dtype=np.float64)
    R[0, 0] = 1 - 2 * y * y - 2 * z * z
    R[0, 1] = 2 * x * y - 2 * w * z
    R[0, 2] = 2 * x * z + 2 * w * y
    R[1, 0] = 2 * x * y + 2 * w * z
    R[1, 1] = 1 - 2 * x * x - 2 * z * z
    R[1, 2] = 2 * y * z - 2 * w * x
    R[2, 0] = 2 * x * z - 2 * w * y
    R[2, 1] = 2 * y * z + 2 * w * x
    R[2, 2] = 1 - 2 * x * x - 2 * y * y
    return R


def projection_matrix(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    fov_rad = math.radians(fov_deg)
    f = 1.0 / math.tan(fov_rad / 2.0)
    proj = np.zeros((4, 4), dtype=np.float64)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2.0 * far * near) / (near - far)
    proj[3, 2] = -1.0
    return proj


def view_matrix(position: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray:
    R = quat_to_rotation_matrix(quaternion_wxyz)
    camera_rot = np.array([
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)
    R_cam = camera_rot @ R
    view = np.eye(4, dtype=np.float64)
    view[:3, :3] = R_cam.T
    view[:3, 3] = -R_cam.T @ position.reshape(3)
    return view


def model_matrix(position: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray:
    R = quat_to_rotation_matrix(quaternion_wxyz)
    model = np.eye(4, dtype=np.float64)
    model[:3, :3] = R
    model[:3, 3] = position.reshape(3)
    return model


def clip_to_screen(
    vertices_clip: np.ndarray,
    img_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Perspective divide; return NDC xy and valid depth mask."""
    w = vertices_clip[:, 3]
    valid = w > 1e-6
    ndc = np.zeros((vertices_clip.shape[0], 3), dtype=np.float64)
    ndc[valid] = vertices_clip[valid, :3] / w[valid, None]
    # Map NDC [-1,1] to pixel centers
    px = ((ndc[:, 0] * 0.5 + 0.5) * (img_size - 1)).astype(np.int32)
    py = ((1.0 - (ndc[:, 1] * 0.5 + 0.5)) * (img_size - 1)).astype(np.int32)
    in_bounds = valid & (ndc[:, 2] >= -1.0) & (ndc[:, 2] <= 1.0)
    in_bounds &= (px >= 0) & (px < img_size) & (py >= 0) & (py < img_size)
    screen = np.stack([px, py], axis=1)
    return screen, in_bounds
