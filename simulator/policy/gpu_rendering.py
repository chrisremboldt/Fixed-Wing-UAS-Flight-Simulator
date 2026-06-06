"""
Optional nvdiffrast renderer (CUDA) for training pixel parity.

Uses the vendored TrainingPixelRenderer (CPU) when nvdiffrast/CUDA is unavailable.
No external repo path imports — mesh data lives in mesh_data.py.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..dynamics import FlightDynamics
from ..intruders import IntruderManager
from .coords import ned_position_to_training, ned_quaternion_to_training


class NvdiffrastPolicyRenderer:
    """
    Policy observation renderer with optional CUDA nvdiffrast backend.

    Falls back to TrainingPixelRenderer on Mac/CPU hosts.
    """

    def __init__(
        self,
        img_size: int = 128,
        fov_deg: float = 90.0,
        device: str = 'cuda:0',
    ):
        from .training_render import TrainingPixelRenderer, TrainingRenderConfig

        self._fallback = TrainingPixelRenderer(
            TrainingRenderConfig(img_size=img_size, fov_deg=fov_deg),
        )
        self._device = device
        self._gpu_renderer = None
        self._init_error: str | None = None

        try:
            import importlib.util
            if importlib.util.find_spec('nvdiffrast') is None:
                raise ImportError('nvdiffrast not installed')

            import torch
            from .nvdiffrast_renderer import VendoredGPURenderer

            if device.startswith('cuda') and not torch.cuda.is_available():
                raise RuntimeError('CUDA not available')

            self._gpu_renderer = VendoredGPURenderer(
                img_size=img_size,
                device=device,
                fov_deg=fov_deg,
            )
        except Exception as exc:
            self._init_error = str(exc)
            self._gpu_renderer = None

    @classmethod
    def is_available(cls, device: str = 'cuda:0') -> bool:
        try:
            import importlib.util
            import torch
            if importlib.util.find_spec('nvdiffrast') is None:
                return False
            if device.startswith('cuda') and not torch.cuda.is_available():
                return False
            from .nvdiffrast_renderer import VendoredGPURenderer  # noqa: F401
            return True
        except Exception:
            return False

    @property
    def using_gpu(self) -> bool:
        return self._gpu_renderer is not None

    @property
    def init_error(self) -> str | None:
        return self._init_error

    def render(
        self,
        dynamics: FlightDynamics,
        intruder_manager: Optional[IntruderManager],
    ) -> np.ndarray:
        if self._gpu_renderer is None:
            return self._fallback.render(dynamics, intruder_manager)

        import torch

        ego_pos = ned_position_to_training(dynamics.state.position)
        ego_quat = ned_quaternion_to_training(dynamics.state.quaternion)

        ego_positions = torch.tensor(ego_pos, device=self._device, dtype=torch.float32).unsqueeze(0)
        ego_quaternions = torch.tensor(ego_quat, device=self._device, dtype=torch.float32).unsqueeze(0)

        max_intruders = 5
        intruder_positions = torch.zeros((1, max_intruders, 3), device=self._device)
        intruder_quaternions = torch.zeros((1, max_intruders, 4), device=self._device)
        intruder_quaternions[..., 0] = 1.0

        if intruder_manager is not None:
            for i, intruder in enumerate(intruder_manager.intruders[:max_intruders]):
                intruder_positions[0, i] = torch.tensor(
                    ned_position_to_training(intruder.dynamics.state.position),
                    device=self._device,
                    dtype=torch.float32,
                )
                intruder_quaternions[0, i] = torch.tensor(
                    ned_quaternion_to_training(intruder.dynamics.state.quaternion),
                    device=self._device,
                    dtype=torch.float32,
                )

        images = self._gpu_renderer.render_batch(
            ego_positions,
            ego_quaternions,
            intruder_positions,
            intruder_quaternions,
        )
        rgb = (images[0].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        return rgb
