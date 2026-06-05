"""Policy integration for DAA model evaluation (RFC 001)."""

from .actions import training_action_to_controls
from .architecture import ImpalaCNN, PufferPolicy, load_policy_checkpoint
from .model_policy import ModelPolicy
from .observation import ObservationSpec, PixelObservationBuilder, BearingMapObservationBuilder
from .rendering import CPUPixelRenderer, CameraConfig
from .trim_assist import (
    TrimAssistConfig,
    TrimAssistedModelPolicy,
    blend_trim_and_policy,
    compute_threat_authority,
)

__all__ = [
    'CameraConfig',
    'CPUPixelRenderer',
    'ImpalaCNN',
    'PufferPolicy',
    'ModelPolicy',
    'ObservationSpec',
    'PixelObservationBuilder',
    'BearingMapObservationBuilder',
    'load_policy_checkpoint',
    'training_action_to_controls',
    'TrimAssistConfig',
    'TrimAssistedModelPolicy',
    'blend_trim_and_policy',
    'compute_threat_authority',
]
