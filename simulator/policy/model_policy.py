"""PyTorch policy loader and intervention hook (RFC 001 phase 1)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..dynamics import FlightDynamics
from ..intruders import IntruderManager
from ..px4_bridge import ControlInterventionPolicy
from ..state import ControlInputs
from .observation import BearingMapObservationBuilder, ObservationSpec

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class PolicyNetwork(nn.Module):
    """
    CNN backbone matching `final_model.pt` layer names.

    Output head may be absent from checkpoint; inference uses available weights only.
    """

    def __init__(self, action_dim: int = 4):
        super().__init__()
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ),
            nn.Sequential(
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ),
            nn.Sequential(
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ),
        ])
        self.fc = nn.Sequential(
            nn.Linear(8192, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.conv_blocks:
            x = block(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


class ModelPolicy:
    """
    Loads a checkpoint and intervenes on commanded controls.

    Falls back to zero action when checkpoint is incompatible until training
    metadata (action scaling, full head weights) is checked in.
    """

    def __init__(
        self,
        checkpoint: Path,
        spec: Optional[ObservationSpec] = None,
        device: str = 'cpu',
    ):
        if not HAS_TORCH:
            raise ImportError('torch is required for ModelPolicy')

        self.spec = spec or ObservationSpec()
        self.device = torch.device(device)
        self.observation_builder = BearingMapObservationBuilder(self.spec)
        self.network = PolicyNetwork().to(self.device)
        self._loaded = self._load_checkpoint(checkpoint)

    def _load_checkpoint(self, checkpoint: Path) -> bool:
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        if not isinstance(state, dict):
            return False
        try:
            self.network.load_state_dict(state, strict=False)
            self.network.eval()
            return True
        except Exception:
            return False

    def intervene(
        self,
        controls: ControlInputs,
        dynamics: FlightDynamics,
        intruder_manager: Optional[IntruderManager],
    ) -> ControlInputs:
        obs = self.observation_builder.build(dynamics, intruder_manager)
        tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self._loaded:
                action = self.network(tensor).squeeze(0).cpu().numpy()
            else:
                action = np.zeros(4, dtype=np.float32)

        # TODO(RFC-001): confirm action scaling from training repo
        delta_e, delta_a, delta_r, delta_t = [float(x) for x in action[:4]]
        aircraft = dynamics.aircraft
        return ControlInputs(
            elevator=float(np.clip(controls.elevator + delta_e, -aircraft.max_elevator, aircraft.max_elevator)),
            aileron=float(np.clip(controls.aileron + delta_a, -aircraft.max_aileron, aircraft.max_aileron)),
            rudder=float(np.clip(controls.rudder + delta_r, -aircraft.max_rudder, aircraft.max_rudder)),
            throttle=float(np.clip(controls.throttle + delta_t, 0.0, 1.0)),
        )
