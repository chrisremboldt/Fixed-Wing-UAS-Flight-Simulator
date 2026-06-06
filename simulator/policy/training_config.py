"""Training-fidelity presets aligned with scratch_built_daa."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingFidelityConfig:
    """Parameters matching the RL training environment defaults."""

    dt: float = 0.02
    img_size: int = 128
    fov_deg: float = 90.0
    max_intruders: int = 5
    spawn_rate: float = 0.2
    renderer_backend: str = 'training'  # auto | training | gpu | legacy
    throttle_mode: str = 'clamp'  # warp clamps throttle to [0, 1]
    policy_device: str = 'cpu'
    use_trim_assist: bool = True  # hold cruise throttle unless --full-policy
    deterministic_policy: bool = True
    cruise_speed_mps: float = 40.0
    max_episode_steps: int = 1000
    surface_scale: float = 1.5
    cruise_throttle_floor: bool = True

    # Warp training physics has no Vne limit or g-load structural failure
    disable_overspeed_crash: bool = True
    disable_structural_g_crash: bool = True

    @classmethod
    def defaults(cls) -> 'TrainingFidelityConfig':
        return cls()

    def simulation_config(self, dt: float | None = None) -> 'SimulationConfig':
        """Build SimulationConfig with training-fidelity crash rules."""
        from ..dynamics import SimulationConfig

        step = dt if dt is not None else self.dt
        return SimulationConfig(
            dt=step,
            disable_overspeed_crash=self.disable_overspeed_crash,
            disable_structural_g_crash=self.disable_structural_g_crash,
        )

    @property
    def episode_duration_s(self) -> float:
        return self.max_episode_steps * self.dt
