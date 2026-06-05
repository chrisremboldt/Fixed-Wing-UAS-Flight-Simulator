"""
IMPALA CNN policy architecture from scratch_built_daa training.

Must match train.py exactly so final_model.pt loads with strict=True.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

import torch
import torch.nn as nn


@dataclass(frozen=True)
class PolicySpaces:
    """Minimal observation/action space metadata for checkpoint loading."""

    observation_shape: Tuple[int, int, int] = (128, 128, 3)
    action_dim: int = 4


class ImpalaCNN(nn.Module):
    """IMPALA-style CNN for pixel-based RL (matches scratch_built_daa/train.py)."""

    def __init__(self, input_shape: Tuple[int, int, int] = (128, 128, 3), output_dim: int = 4):
        super().__init__()

        input_channels = input_shape[2] if len(input_shape) == 3 else input_shape[0]

        self.conv_blocks = nn.ModuleList([
            self._make_conv_block(input_channels, 16),
            self._make_conv_block(16, 32),
            self._make_conv_block(32, 32),
        ])

        with torch.no_grad():
            h, w = input_shape[0], input_shape[1]
            dummy = torch.zeros(1, input_channels, h, w)
            dummy_out = self._forward_conv(dummy)
            conv_out_size = dummy_out.view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(256, output_dim)
        self.value_head = nn.Linear(256, 1)
        self.apply(self._init_weights)

    def _make_conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.conv_blocks:
            x = block(x)
        return x

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: (B, H, W, C) uint8 [0, 255] or float already normalized
        """
        if obs.dtype == torch.uint8:
            x = obs.float() / 255.0
        else:
            x = obs.float()
            if x.max() > 1.5:
                x = x / 255.0

        x = x.permute(0, 3, 1, 2)
        x = self._forward_conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class PufferPolicy(nn.Module):
    """Policy wrapper matching scratch_built_daa/train.py checkpoint layout."""

    def __init__(self, spaces: PolicySpaces | None = None):
        super().__init__()
        spaces = spaces or PolicySpaces()
        self.spaces = spaces
        self.network = ImpalaCNN(
            input_shape=spaces.observation_shape,
            output_dim=spaces.action_dim,
        )
        self.log_std = nn.Parameter(torch.zeros(spaces.action_dim))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.network(obs)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        _, value = self.network(obs)
        return value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, value = self.network(obs)

        if deterministic:
            action_out = torch.tanh(mean)
            log_prob = torch.zeros(mean.shape[0], device=mean.device)
            entropy = torch.zeros(mean.shape[0], device=mean.device)
            return action_out, log_prob, entropy, value

        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        action_out = torch.tanh(action)
        return action_out, log_prob, entropy, value

    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Mean policy output in [-1, 1] (eval mode)."""
        mean, _ = self.network(obs)
        return torch.tanh(mean)


def load_policy_checkpoint(
    checkpoint_path: str,
    device: torch.device | str = 'cpu',
    spaces: PolicySpaces | None = None,
) -> PufferPolicy:
    """Load final_model.pt with strict key matching."""
    policy = PufferPolicy(spaces=spaces).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(state, dict) and 'policy_state_dict' in state:
        state = state['policy_state_dict']

    missing, unexpected = policy.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f'Checkpoint mismatch: missing={missing}, unexpected={unexpected}')

    policy.eval()
    return policy
