"""Policy integration for DAA model evaluation (RFC 001)."""

from .actions import training_action_to_controls
from .architecture import ImpalaCNN, PufferPolicy, load_policy_checkpoint
from .model_policy import ModelPolicy
from .observation import ObservationSpec, PixelObservationBuilder, BearingMapObservationBuilder
from .rendering import CPUPixelRenderer, CameraConfig
from .evaluation import (
    BatchEvalResult,
    EpisodeResult,
    EvalConfig,
    export_batch_results,
    format_batch_summary,
    run_batch_evaluation,
    run_episode,
)
from .scenarios import ScenarioConfig, resolve_scenario_path
from .training_config import TrainingFidelityConfig
from .training_init import (
    TRAINING_CRUISE_SPEED_MPS,
    apply_training_initial_state,
    sample_training_initial_state,
)
from .training_render import TrainingPixelRenderer, TrainingRenderConfig, create_policy_renderer
from .trim_assist import (
    TrimAssistConfig,
    TrimAssistedModelPolicy,
    blend_trim_and_policy,
    compute_threat_authority,
)

__all__ = [
    'CameraConfig',
    'CPUPixelRenderer',
    'BatchEvalResult',
    'EpisodeResult',
    'EvalConfig',
    'format_batch_summary',
    'export_batch_results',
    'run_batch_evaluation',
    'run_episode',
    'ScenarioConfig',
    'resolve_scenario_path',
    'TrainingFidelityConfig',
    'TRAINING_CRUISE_SPEED_MPS',
    'apply_training_initial_state',
    'sample_training_initial_state',
    'TrainingPixelRenderer',
    'TrainingRenderConfig',
    'create_policy_renderer',
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
